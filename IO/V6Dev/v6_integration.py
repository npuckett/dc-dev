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
            from .smart_autotuner import DEFAULT_SAFE_FLOORS, PARAM_RANGES
            # V6.1f: Wire PARAM_RANGES upper bounds as caps into the resolver
            # so per-frame multiplicative mods can't push values past range max.
            resolver_caps = {name: hi for name, (lo, hi) in PARAM_RANGES.items()}
            self.resolver = ModifierResolver(
                safe_floors=dict(DEFAULT_SAFE_FLOORS),
                caps=resolver_caps,
            )

        # Timing
        self._last_log_time = 0.0
        self._frame_count = 0
        self._total_adjustments = 0

        # V6.1f: Rate-limit per-frame intent sources to prevent runaway
        # Context/feedback/mode intents were compounding at ~28fps, driving
        # all params to caps in seconds. Now apply every 2s instead.
        self._intent_interval = 2.0  # seconds between context/feedback/mode applications
        self._last_intent_time = 0.0
        self._cached_feedback_mods = None
        self._cached_context = None
        self._cached_mode_overlay = None

        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_interval = 300.0  # every 5 minutes
        self._passivity_spiral_detected = False

        # Bandit outcome tracking (to pass context to record_outcome)
        self._last_bandit_strategy = None
        self._last_bandit_context = None

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

        # V6.1f: Rate-limit context/feedback/mode intent computation
        # These were applying every frame (~28fps), compounding multiplicative
        # and additive pushes that drove all params to caps in seconds.
        # Now compute fresh values only every _intent_interval seconds;
        # between ticks, use cached values (or None to skip entirely).
        apply_intents = (now - self._last_intent_time >= self._intent_interval)
        if apply_intents:
            self._last_intent_time = now

        # ➁ Mode intelligence (always compute for mode tracking, cache for intents)
        mode_overlay = None
        if self.mode_intel:
            mode_overlay = self.mode_intel.update(
                behavior_status, tracked_people, now,
            )
            if apply_intents:
                self._cached_mode_overlay = mode_overlay

        # ➂ Predictive context (always compute for regime, cache for intents)
        context = None
        if self.context_engine:
            active_count = behavior_status.get('active_count', 0)
            context = self.context_engine.get_context(current_people=active_count)
            if apply_intents:
                self._cached_context = context

        # ➌ Strategy bandit — V6.5: expression strategy per mode (not conversion)
        strategy_effect = None
        if self.bandit:
            from .strategy_bandit import BanditContext
            bandit_ctx = BanditContext(
                hour=int(time.strftime('%H')),
                mode=behavior_status.get('mode', 'idle'),
                passive_rate=behavior_status.get('passive_rate', 0.0),
                regime=context.regime if context else 'steady',
            )
            # Switch strategy when timer expires
            if self.bandit.should_switch_strategy(now):
                # Record outcome of previous strategy (if any)
                if self.bandit.current_strategy and self.bandit.current_context and self.scorer:
                    # Score based on dynamic range from the scorer
                    last_snap = self.scorer.history[-1] if self.scorer.history else None
                    quality = last_snap.components.get('dynamic_range', 0.5) if last_snap else 0.5
                    self.bandit.record_outcome(
                        self.bandit.current_strategy,
                        self.bandit.current_context,
                        quality=quality,
                    )
                # Select new strategy
                selected = self.bandit.select(bandit_ctx)
                strategy_effect = self.bandit.get_effect(selected)
                self.bandit.set_active_strategy(selected, bandit_ctx, now)
                if self.scorer:
                    self.scorer.record_strategy_attempt()
            elif self.bandit.current_strategy:
                # Continue applying current strategy
                strategy_effect = self.bandit.get_effect(self.bandit.current_strategy)

        # ➄ Feedback learning (rate-limited)
        feedback_mods = None
        if self.feedback and apply_intents:
            from .feedback_learning_v6 import FeedbackContext
            from datetime import datetime as _dt
            _now_dt = _dt.now()
            ctx = FeedbackContext()
            ctx.timestamp = now
            ctx.hour = _now_dt.hour
            ctx.source_mode = behavior_status.get('mode', 'idle')
            ctx.group_size = behavior_status.get('active_count', 0)
            ctx.regime = context.regime if context else 'steady'
            # V6.5: passive traffic context for per-tier feedback learning
            ctx.passive_rate = behavior_status.get('passive_rate', 0.0)
            _mode = ctx.source_mode
            ctx.passive_tier = (
                'busy' if _mode == 'aware' else
                'flow' if _mode == 'flow' else 'quiet'
            )

            feedback_mods = self.feedback.get_modifiers(ctx)
            self._cached_feedback_mods = feedback_mods

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
            nearest_z = behavior_params.get('nearest_z', 300.0)
            nearest_x = behavior_params.get('nearest_x', -150.0)

            falloff_shape = self.falloff_mgr.compute_shape(
                mode=mode,
                dt=dt,
                nearest_person_z=nearest_z,
                nearest_person_x=nearest_x,
                flow_direction=flow_dir,
                flow_strength=flow_str,
                active_count=behavior_status.get('active_count', 0),
                passive_count=behavior_status.get('passive_count', 0),
            )

        # ➐ Modifier resolver (if enabled, merge all intents)
        # V6.1f: Use cached context/feedback/mode for rate-limited intents
        if self.resolver:
            behavior_params = self._resolve_modifiers(
                behavior_params, tuner_result,
                self._cached_feedback_mods if apply_intents else None,
                strategy_effect,
                self._cached_mode_overlay if apply_intents else None,
                self._cached_context if apply_intents else None,
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

        # Periodic health check
        if now - self._last_health_check > self._health_check_interval:
            self._last_health_check = now
            self.v6_health_check()

        return behavior_params

    # ------------------------------------------------------------------
    # Hook: daily report
    # ------------------------------------------------------------------

    def on_daily_report(self, report: dict):
        """Called when a daily report is generated.

        Feeds report data to V6 subsystems for learning.
        Detects passivity spiral pattern and auto-resets if needed.
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
            self.bandit._save()

        # Feedback persistence
        if self.feedback:
            self.feedback.save()

        # Mode intelligence session reset (midnight)
        if self.mode_intel:
            self.mode_intel.reset_session()

        # V6.5: Passivity spiral detection REMOVED.
        # IDLE is a valid long-term mode. The engagement scorer now rewards
        # dynamic range within each mode rather than conversion metrics.
        self._passivity_spiral_detected = False

    # ------------------------------------------------------------------
    # Hook: person events (chain after V5 behavior callbacks)
    # ------------------------------------------------------------------

    def on_person_entered(self, person: dict, now: float):
        """Chain after behavior.on_person_entered()."""
        # Wire proactive counter: gesture attempt when person enters active zone
        if self.scorer:
            self.scorer.record_gesture_attempt()

    def on_person_left(self, person: dict, now: float):
        """Chain after behavior.on_person_left().

        V6.5: No longer records conversion outcomes — bandit now tracks
        expression quality via dynamic range, not per-person conversions.
        """
        pass

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
            from dataclasses import asdict as _asdict
            if isinstance(feedback_mods, dict):
                mods = feedback_mods
            elif hasattr(feedback_mods, '__dataclass_fields__'):
                mods = _asdict(feedback_mods)
            else:
                mods = {}
            intents = ModifierResolver.intents_from_feedback(mods)
            self.resolver.add_many(intents)

        # Strategy intents
        if strategy_effect:
            # strategy_effect is a StrategyEffect dataclass directly
            # Map its fields to what intents_from_strategy expects
            se = strategy_effect
            # brightness_boost is additive DMX; convert to mult ~1.0+boost/255
            br_mult = 1.0 + (getattr(se, 'brightness_boost', 0.0) / 255.0)
            effect_dict = {
                'brightness_mult': br_mult,
                'speed_mult': getattr(se, 'move_speed_mult', 1.0),
                'pulse_mult': 1.0,
                'scale_x': getattr(se, 'falloff_scale_x', None),
                'scale_y': getattr(se, 'falloff_scale_y', None),
                'scale_z': getattr(se, 'falloff_scale_z', None),
                'rotation': getattr(se, 'falloff_rotation', None),
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
        # V6.5: log passive tier info
        passive_rate = status.get('passive_rate', 0.0)
        mode = status.get('mode', 'idle')
        tier = 'busy' if mode == 'aware' else ('flow' if mode == 'flow' else 'quiet')
        parts.append(f"passive={passive_rate:.1f}/min({tier})")
        if self.autotuner:
            parts.append(f"budget={self.autotuner.budget:.0f}")
        if self._passivity_spiral_detected:
            parts.append("WARN:passivity_spiral")
        logger.info(', '.join(parts))

    # ------------------------------------------------------------------
    # Health check & passivity spiral auto-reset
    # ------------------------------------------------------------------

    def v6_health_check(self) -> dict:
        """Run a health check on the V6 system.

        Returns a dict with health indicators.
        Called automatically every 5 minutes from tick().
        """
        health = {'ok': True, 'warnings': []}

        # Check autotuner budget
        if self.autotuner:
            budget = self.autotuner.budget
            if budget < 10:
                health['warnings'].append(
                    f"autotuner budget={budget:.0f} (nearly depleted)"
                )

        # Check engagement score
        if self.scorer:
            score = self.scorer.smoothed_score()
            if score < 0.05:
                health['warnings'].append(
                    f"engagement score={score:.3f} (near zero — gradient estimator blind)"
                )

        if health['warnings']:
            health['ok'] = False
            logger.warning(f"V6 health check: {'; '.join(health['warnings'])}")
        else:
            logger.debug("V6 health check: OK")

        return health

    def _reset_passivity_spiral(self):
        """V6.5: No-op — passivity spiral detection removed.

        IDLE is a valid long-term mode. The engagement scorer now rewards
        dynamic range within each mode rather than conversion metrics.
        Kept as stub for API compatibility.
        """
        pass

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
            stats = self.bandit.get_stats()
            ext['v6_bandit_best'] = max(
                stats.get('strategies', {}).items(),
                key=lambda kv: kv[1].get('mean_posterior', 0),
                default=('none', {})
            )[0] if stats.get('strategies') else 'none'

        if self.context_engine:
            ctx = self.context_engine.get_context()
            ext['v6_regime'] = ctx.regime
            ext['v6_predicted_traffic'] = round(ctx.expected_people, 1)
            # V6.5: passive prediction info
            ext['v6_expected_passive'] = round(ctx.expected_passive_count, 1)
            ext['v6_expected_tier'] = ctx.expected_passive_tier

        if self.scorer:
            ext['v6_engagement_score'] = round(self.scorer.smoothed_score(), 3)

        return ext
