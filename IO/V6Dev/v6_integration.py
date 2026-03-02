#!/usr/bin/env python3
"""
V6 Integration Bridge
=====================

Wires all V6 modules into the existing V5 controller **without modifying
any V5 files**.  The bridge is instantiated once during ``main()`` setup
and provides three hook methods that slot into the V5 main loop:

    v6 = V6Integration(meta, sliders, db, behavior, light)

    # … in the main loop …
    behavior_params = behavior.update(dt, active, passive, …)
    behavior_status = behavior.get_status()

    # ➊ Post-behavior hook (replaces V5 auto-tuner call)
    v6.post_behavior_update(behavior_status, behavior_params, tracked_people, now)

    # ➋ Pre-render hook (modifies falloff before light.update)
    behavior_params = v6.pre_render(behavior_params, behavior_status, tracked_people, dt)

    # ➌ Daily report hook
    v6.on_daily_report(report)

Nothing inside IO/light_behavior.py or IO/lightController_osc.py needs
to change.  The V5 ``AutoTuningManager`` can optionally be disabled; the
``SmartAutoTuner`` inside V6 replaces it.

Quick start::

    # Add these lines to main() in lightController_osc.py:
    from V6Dev.v6_integration import V6Integration
    v6 = V6Integration(meta, sliders, db, behavior, light,
                        reports_dir='reports/daily')

    # In main loop (after behavior.update, before light.update):
    behavior_params = v6.tick(behavior_status, behavior_params,
                              tracked_people, dt, now)

    # Disable V5 auto-tuner:
    auto_tuner.set_enabled(False)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .engagement_score import EngagementScorer
from .predictive_context import PredictiveContextEngine
from .strategy_bandit import StrategyBandit
from .feedback_learning_v6 import FeedbackLearningV6
from .falloff_strategies import FalloffStrategyManager
from .smart_autotuner import SmartAutoTuner
from .mode_intelligence import ModeIntelligence
from .modifier_resolver import (
    ModifierResolver, ModifierIntent, Direction, Priority,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class V6Config:
    """Top-level V6 configuration."""
    enabled: bool = True
    reports_dir: str = 'reports/daily'

    # Module toggles (can selectively disable any subsystem)
    enable_smart_autotuner: bool = True
    enable_predictive_context: bool = True
    enable_strategy_bandit: bool = True
    enable_feedback_v6: bool = True
    enable_falloff_strategies: bool = True
    enable_mode_intelligence: bool = True
    enable_modifier_resolver: bool = True

    # Logging
    log_adjustments: bool = True
    log_interval: float = 60.0  # status log every N seconds


# ---------------------------------------------------------------------------
# Integration Bridge
# ---------------------------------------------------------------------------

class V6Integration:
    """Central wiring between V6 modules and the V5 controller.

    Parameters
    ----------
    meta : MetaParameters
        V5 shared parameter object (read/write).
    sliders : dict
        V5 GUI slider dict.
    database
        V5 TrackingDatabase instance (or None for testing).
    behavior
        V5 BehaviorSystem instance (read-only).
    light
        V5 PointLight instance (for falloff overrides).
    reports_dir : str
        Directory containing daily JSON reports.
    config : V6Config | None
        Top-level config.
    """

    def __init__(
        self,
        meta,
        sliders: dict,
        database=None,
        behavior=None,
        light=None,
        reports_dir: str = 'reports/daily',
        config: V6Config = None,
    ):
        self.meta = meta
        self.sliders = sliders
        self.db = database
        self.behavior = behavior
        self.light = light
        self.config = config or V6Config(reports_dir=reports_dir)

        # Resolve reports directory relative to IO/
        if not os.path.isabs(self.config.reports_dir):
            io_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config.reports_dir = os.path.join(io_dir, self.config.reports_dir)

        # ----- Instantiate V6 modules -----

        # 1. Engagement scorer (provides fitness signal to autotuner)
        self.scorer = EngagementScorer(database)

        # 2. Predictive context engine (learned time-of-day profiles)
        self.context_engine = None
        if self.config.enable_predictive_context:
            self.context_engine = PredictiveContextEngine(
                reports_dir=self.config.reports_dir,
            )

        # 3. Strategy bandit (Thompson Sampling for almost-engaged)
        self.bandit = None
        if self.config.enable_strategy_bandit:
            self.bandit = StrategyBandit()

        # 4. Feedback learning V6 (bidirectional, decaying)
        self.feedback = None
        if self.config.enable_feedback_v6:
            self.feedback = FeedbackLearningV6()

        # 5. Falloff strategy manager (spatial expression)
        self.falloff_mgr = None
        if self.config.enable_falloff_strategies:
            self.falloff_mgr = FalloffStrategyManager()

        # 6. Smart auto-tuner (gradient-informed)
        self.autotuner = None
        if self.config.enable_smart_autotuner:
            self.autotuner = SmartAutoTuner(
                meta=meta,
                sliders=sliders,
                database=database,
                scorer=self.scorer,
                context_engine=self.context_engine,
            )

        # 7. Mode intelligence (predictive transitions)
        self.mode_intel = None
        if self.config.enable_mode_intelligence:
            self.mode_intel = ModeIntelligence()

        # 8. Modifier resolver (intent-based merging)
        self.resolver = None
        if self.config.enable_modifier_resolver:
            from .smart_autotuner import DEFAULT_SAFE_FLOORS
            self.resolver = ModifierResolver(
                safe_floors=dict(DEFAULT_SAFE_FLOORS),
            )

        # Timing
        self._last_log_time = 0.0
        self._frame_count = 0
        self._total_adjustments = 0

        logger.info(
            f"V6 Integration initialized: "
            f"autotuner={self.autotuner is not None}, "
            f"context={self.context_engine is not None}, "
            f"bandit={self.bandit is not None}, "
            f"feedback={self.feedback is not None}, "
            f"falloff={self.falloff_mgr is not None}, "
            f"mode_intel={self.mode_intel is not None}, "
            f"resolver={self.resolver is not None}"
        )

    # ------------------------------------------------------------------
    # Main tick (single-call convenience)
    # ------------------------------------------------------------------

    def tick(
        self,
        behavior_status: dict,
        behavior_params: dict,
        tracked_people: List[dict],
        dt: float,
        now: float,
    ) -> dict:
        """Single-call entry: runs all V6 subsystems and returns
        modified ``behavior_params``.

        This replaces the V5 ``auto_tuner.update()`` call and modifies
        ``behavior_params`` before they're applied to the light.

        Parameters
        ----------
        behavior_status : dict
            From ``behavior.get_status()``.
        behavior_params : dict
            From ``behavior.update(dt, …)``.
        tracked_people : list[dict]
            Current tracked persons (from TrackedPersonManager).
        dt : float
            Frame delta time.
        now : float
            ``time.time()``.

        Returns
        -------
        dict
            Modified behavior_params (falloff, brightness, etc. may change).
        """
        if not self.config.enabled:
            return behavior_params

        self._frame_count += 1

        # ➊ Mode intelligence
        mode_overlay = None
        if self.mode_intel:
            mode_overlay = self.mode_intel.update(
                behavior_status, tracked_people, now,
            )

        # ➋ Predictive context
        context = None
        if self.context_engine:
            context = self.context_engine.get_context()

        # ➌ Strategy bandit (for almost-engaged candidates)
        strategy_effect = None
        if self.bandit and behavior_status.get('almost_engaged'):
            almost = behavior_status['almost_engaged']
            candidates = almost.get('candidates', [])
            if candidates:
                from .strategy_bandit import BanditContext
                bandit_ctx = BanditContext(
                    time_period=self._time_period_from_status(behavior_status),
                    active_count=behavior_status.get('active_count', 0),
                    mode=behavior_status.get('mode', 'idle'),
                )
                strategy_result = self.bandit.select_strategy(bandit_ctx, now)
                if strategy_result:
                    strategy_effect = strategy_result

        # ➍ Feedback learning
        feedback_mods = None
        if self.feedback:
            mode = behavior_status.get('mode', 'idle')
            active = behavior_status.get('active_count', 0)
            dwell = behavior_status.get('dwell_bonus', 0)
            speed_bucket = self._speed_bucket(tracked_people)
            group_bucket = self._group_bucket(behavior_status)
            regime = context.regime if context else 'steady'

            feedback_mods = self.feedback.get_modifiers(
                mode=mode,
                active_count=active,
                dwell_phase=behavior_status.get('dwell_phase', 'notice'),
                speed_bucket=speed_bucket,
                group_bucket=group_bucket,
                regime_bucket=regime,
            )

        # ➎ Smart auto-tuner
        tuner_result = None
        if self.autotuner:
            tuner_result = self.autotuner.update(behavior_status, now)

        # ➏ Falloff strategy manager
        falloff_shape = None
        if self.falloff_mgr:
            mode = behavior_status.get('mode', 'idle')
            flow_info = behavior_status.get('flow', {})
            flow_dir = flow_info.get('direction', 0)
            flow_str = flow_info.get('strength', 0)
            nearest_z = behavior_params.get('nearest_z', 1.0)
            gesture = behavior_status.get('gesture')

            falloff_shape = self.falloff_mgr.compute_shape(
                mode=mode,
                nearest_z=nearest_z,
                flow_direction=flow_dir,
                flow_strength=flow_str,
                gesture=gesture,
                dt=dt,
            )

        # ➐ Modifier resolver (if enabled, merge all intents)
        if self.resolver:
            behavior_params = self._resolve_modifiers(
                behavior_params, tuner_result, feedback_mods,
                strategy_effect, mode_overlay, context,
                falloff_shape,
            )
        else:
            # Simpler fallback: apply V6 outputs directly
            behavior_params = self._apply_direct(
                behavior_params, tuner_result, feedback_mods,
                strategy_effect, mode_overlay, falloff_shape,
            )

        # Periodic logging
        if self.config.log_adjustments and now - self._last_log_time > self.config.log_interval:
            self._last_log_time = now
            self._log_status(behavior_status, mode_overlay, context)

        return behavior_params

    # ------------------------------------------------------------------
    # Hook: daily report
    # ------------------------------------------------------------------

    def on_daily_report(self, report: dict):
        """Called when a daily report is generated.

        Feeds report data to V6 subsystems for learning.
        """
        # Update predictive context with new data
        if self.context_engine:
            self.context_engine.load_reports()  # re-scan directory

        # Smart autotuner daily learnings
        if self.autotuner:
            learnings = self.autotuner.compute_daily_learnings(report)
            logger.info(f"V6 daily learnings: score={learnings.get('engagement_score', 0):.3f}")

        # Bandit persistence
        if self.bandit:
            self.bandit.save_priors()

        # Feedback persistence
        if self.feedback:
            self.feedback.save()

        # Mode intelligence session reset (midnight)
        if self.mode_intel:
            self.mode_intel.reset_session()

    # ------------------------------------------------------------------
    # Hook: person events (chain after V5 behavior callbacks)
    # ------------------------------------------------------------------

    def on_person_entered(self, person: dict, now: float):
        """Chain after behavior.on_person_entered()."""
        if self.feedback:
            self.feedback.on_person_entered(person, now)

    def on_person_left(self, person: dict, now: float):
        """Chain after behavior.on_person_left()."""
        if self.feedback:
            dwell_time = now - person.get('first_seen', now)
            self.feedback.on_engagement_ended(
                mode=person.get('last_mode', 'idle'),
                dwell_seconds=dwell_time,
                active_count=0,
                speed_bucket=self._speed_bucket_single(person),
                group_bucket='solo',
            )
        if self.bandit:
            # Report outcome for the almost-engaged strategy
            self.bandit.report_outcome(
                converted=person.get('reached_engaged', False),
                dwell_seconds=now - person.get('first_seen', now),
                now=now,
            )

    # ------------------------------------------------------------------
    # Modifier resolution (intent-based merge)
    # ------------------------------------------------------------------

    def _resolve_modifiers(
        self,
        params: dict,
        tuner_result: Optional[dict],
        feedback_mods: Optional[dict],
        strategy_effect: Optional[object],
        mode_overlay: Optional[object],
        context: Optional[object],
        falloff_shape: Optional[object],
    ) -> dict:
        """Use ModifierResolver to merge all V6 signals."""
        current_params = {
            'brightness_global': getattr(self.meta, 'brightness_global', 1.0),
            'speed_global': getattr(self.meta, 'speed_global', 1.0),
            'pulse_global': getattr(self.meta, 'pulse_global', 1.0),
            'responsiveness': getattr(self.meta, 'responsiveness', 0.5),
            'energy': getattr(self.meta, 'energy', 0.5),
            'sociability': getattr(self.meta, 'sociability', 0.5),
            'exploration': getattr(self.meta, 'exploration', 0.5),
            'attention_span': getattr(self.meta, 'attention_span', 0.5),
        }

        self.resolver.begin_frame(current_params)

        # Auto-tuner intents
        if tuner_result:
            deltas = tuner_result.get('deltas', {})
            intents = ModifierResolver.intents_from_autotuner_deltas(deltas)
            self.resolver.add_many(intents)

        # Feedback intents
        if feedback_mods:
            mods = feedback_mods if isinstance(feedback_mods, dict) else {}
            intents = ModifierResolver.intents_from_feedback(mods)
            self.resolver.add_many(intents)

        # Strategy intents
        if strategy_effect:
            effect_dict = {}
            if hasattr(strategy_effect, 'effect'):
                e = strategy_effect.effect
                effect_dict = {
                    'brightness_mult': getattr(e, 'brightness_mult', 1.0),
                    'speed_mult': getattr(e, 'speed_mult', 1.0),
                    'pulse_mult': getattr(e, 'pulse_mult', 1.0),
                    'scale_x': getattr(e, 'scale_x', None),
                    'scale_y': getattr(e, 'scale_y', None),
                    'scale_z': getattr(e, 'scale_z', None),
                    'rotation': getattr(e, 'rotation', None),
                }
            intents = ModifierResolver.intents_from_strategy(effect_dict)
            self.resolver.add_many(intents)

        # Mode intelligence intents
        if mode_overlay:
            intents = ModifierResolver.intents_from_mode_overlay(mode_overlay)
            self.resolver.add_many(intents)

        # Context intents
        if context:
            intents = ModifierResolver.intents_from_context(context, current_params)
            self.resolver.add_many(intents)

        # Resolve
        results = self.resolver.resolve()

        # Apply resolved modifiers to MetaParameters
        for r in results:
            if hasattr(self.meta, r.parameter):
                setattr(self.meta, r.parameter, r.new_value)
                if r.parameter in self.sliders:
                    self.sliders[r.parameter].value = r.new_value
            self._total_adjustments += 1

        # Apply falloff shape to behavior_params (renderer-facing)
        if falloff_shape:
            params['falloff_scale_x'] = falloff_shape.scale_x
            params['falloff_scale_y'] = falloff_shape.scale_y
            params['falloff_scale_z'] = falloff_shape.scale_z
            params['falloff_rotation'] = falloff_shape.rotation

        return params

    # ------------------------------------------------------------------
    # Direct application (fallback when resolver is disabled)
    # ------------------------------------------------------------------

    def _apply_direct(
        self,
        params: dict,
        tuner_result: Optional[dict],
        feedback_mods: Optional[dict],
        strategy_effect: Optional[object],
        mode_overlay: Optional[object],
        falloff_shape: Optional[object],
    ) -> dict:
        """Apply V6 outputs directly (simpler, less conflict resolution)."""
        # Feedback multipliers
        if feedback_mods and isinstance(feedback_mods, dict):
            brightness_mult = feedback_mods.get('brightness', 1.0)
            speed_mult = feedback_mods.get('speed', 1.0)
            pulse_mult = feedback_mods.get('pulse', 1.0)
            params['brightness_max'] = params.get('brightness_max', 1.0) * brightness_mult
            params['pulse_speed'] = params.get('pulse_speed', 2000) * pulse_mult
            params['move_speed'] = params.get('move_speed', 1.0) * speed_mult

        # Mode intensity
        if mode_overlay and mode_overlay.intensity_mult != 1.0:
            params['brightness_max'] = params.get('brightness_max', 1.0) * mode_overlay.intensity_mult

        # Falloff shape
        if falloff_shape:
            params['falloff_scale_x'] = falloff_shape.scale_x
            params['falloff_scale_y'] = falloff_shape.scale_y
            params['falloff_scale_z'] = falloff_shape.scale_z
            params['falloff_rotation'] = falloff_shape.rotation

        return params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_period_from_status(self, status: dict) -> str:
        """Map V5 time_of_day string to bandit time period."""
        tod = status.get('time_of_day', 'active')
        mapping = {
            'sleepy': 'night',
            'waking': 'morning',
            'active': 'afternoon',
            'rush': 'rush',
            'evening': 'evening',
        }
        return mapping.get(tod, 'afternoon')

    @staticmethod
    def _speed_bucket(tracked_people: List[dict]) -> str:
        """Classify avg speed of tracked people."""
        if not tracked_people:
            return 'still'
        speeds = []
        for p in tracked_people:
            vx = p.get('vx', 0)
            vz = p.get('vz', 0)
            speeds.append((vx**2 + vz**2) ** 0.5)
        avg = sum(speeds) / len(speeds) if speeds else 0
        if avg < 10:
            return 'still'
        elif avg < 50:
            return 'walking'
        elif avg < 120:
            return 'brisk'
        else:
            return 'running'

    @staticmethod
    def _speed_bucket_single(person: dict) -> str:
        vx = person.get('vx', 0)
        vz = person.get('vz', 0)
        speed = (vx**2 + vz**2) ** 0.5
        if speed < 10:
            return 'still'
        elif speed < 50:
            return 'walking'
        elif speed < 120:
            return 'brisk'
        return 'running'

    @staticmethod
    def _group_bucket(status: dict) -> str:
        active = status.get('active_count', 0)
        if active == 0:
            return 'empty'
        elif active == 1:
            return 'solo'
        elif active <= 3:
            return 'pair'
        elif active <= 8:
            return 'group'
        return 'crowd'

    def _log_status(self, status: dict, overlay, context):
        """Periodic status log."""
        parts = [f"V6 frame={self._frame_count}, adj={self._total_adjustments}"]
        if overlay:
            parts.append(f"mode={overlay.effective_mode.value}")
            if overlay.pre_transition_blend > 0.05:
                parts.append(f"pre-transition={overlay.pre_transition_blend:.2f}")
        if context:
            parts.append(f"regime={context.regime}")
        if self.autotuner:
            parts.append(f"budget={self.autotuner.budget:.0f}")
        logger.info(', '.join(parts))

    # ------------------------------------------------------------------
    # WebSocket state extension
    # ------------------------------------------------------------------

    def get_state_extension(self) -> dict:
        """Return V6-specific state fields for WebSocket broadcast."""
        ext = {'v6_enabled': self.config.enabled}

        if self.mode_intel:
            # Last computed overlay info (read from last tick)
            ext['v6_mode_familiarity'] = round(self.mode_intel._familiarity, 2)
            ext['v6_mode_momentum'] = round(self.mode_intel._momentum, 2)

        if self.autotuner:
            ext['v6_tuner_budget'] = round(self.autotuner.budget, 1)
            grads = self.autotuner.estimated_gradients
            top_grads = sorted(grads.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            ext['v6_top_gradients'] = {k: round(v, 4) for k, v in top_grads}

        if self.bandit:
            ext['v6_bandit_best'] = self.bandit.get_best_strategy()

        if self.context_engine:
            ctx = self.context_engine.get_context()
            ext['v6_regime'] = ctx.regime
            ext['v6_predicted_traffic'] = round(ctx.predicted_activity, 1)

        if self.scorer:
            ext['v6_engagement_score'] = round(self.scorer.smoothed_score(), 3)

        return ext
