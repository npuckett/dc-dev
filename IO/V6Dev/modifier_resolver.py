#!/usr/bin/env python3
"""
V6 Modifier Resolver
====================

Replaces V5's 13-step last-write-wins modifier chain with an
**intent-based** system where each modifier *declares* what it wants
rather than directly overwriting values.

Design
------
Each V6 subsystem produces a ``ModifierIntent`` for the parameters it
cares about.  An intent is:

    (parameter, direction, strength, source, priority)

The resolver collects all intents per-parameter, groups them by
direction (UP / DOWN / SET), and produces a single merged value using
weighted-priority resolution.

This prevents:
- Feedback system boosting brightness while regime damper is
  suppressing it → instead, both intents are weighted and
  the stronger one wins with contribution from the other.
- Multiple systems redundantly pushing responsiveness →
  the resolver caps the combined effect.

Priority levels:
    1. SAFETY      – safe floors, caps (cannot be overridden)
    2. CONTEXT     – time-of-day regime, crowd safety
    3. STRATEGY    – bandit / auto-tuner suggestions
    4. FEEDBACK    – learned feedback weights
    5. AESTHETIC    – mode-default shapes, gesture overlays

Lower number = higher priority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Direction(Enum):
    """What the intent wants to happen to the parameter."""
    UP = 'up'           # increase the value
    DOWN = 'down'       # decrease the value
    SET = 'set'         # set to a specific absolute value
    MULTIPLY = 'mult'   # multiply the current value


class Priority(IntEnum):
    """Priority levels (lower = stronger)."""
    SAFETY = 1
    CONTEXT = 2
    STRATEGY = 3
    FEEDBACK = 4
    AESTHETIC = 5


@dataclass
class ModifierIntent:
    """A single intent from a V6 subsystem.

    Attributes
    ----------
    parameter : str
        Name of the parameter to modify (e.g., 'brightness_global').
    direction : Direction
        What kind of change.
    strength : float
        For UP/DOWN: the desired delta (positive).
        For SET: the absolute target value.
        For MULTIPLY: the scale factor.
    source : str
        Identifier of the subsystem (for debugging / logging).
    priority : Priority
        Priority level.
    confidence : float
        0–1 how confident the source is in this intent.  Used for
        weighted blending between same-priority intents.
    """
    parameter: str
    direction: Direction
    strength: float
    source: str
    priority: Priority
    confidence: float = 1.0

    def effective_delta(self, current_value: float) -> float:
        """Convert this intent into a signed delta."""
        if self.direction == Direction.UP:
            return abs(self.strength)
        elif self.direction == Direction.DOWN:
            return -abs(self.strength)
        elif self.direction == Direction.SET:
            return self.strength - current_value
        elif self.direction == Direction.MULTIPLY:
            return current_value * (self.strength - 1.0)
        return 0.0


@dataclass
class ResolvedModifier:
    """The result of resolving all intents for one parameter."""
    parameter: str
    old_value: float
    new_value: float
    delta: float
    dominant_source: str       # which source contributed most
    intent_count: int          # how many intents competed
    conflict: bool = False     # were there opposing intents?


# ---------------------------------------------------------------------------
# Budget per source (optional caps on how much each source can move)
# ---------------------------------------------------------------------------

@dataclass
class SourceBudget:
    """Optional per-source adjustment budget per cycle."""
    max_total_delta: float = 0.5      # sum of absolute deltas
    max_per_param_delta: float = 0.1
    remaining: float = 0.5

    def clamp(self, delta: float) -> float:
        d = max(-self.max_per_param_delta,
                min(self.max_per_param_delta, delta))
        allowed = min(abs(d), self.remaining)
        self.remaining -= allowed
        return allowed if d >= 0 else -allowed

    def reset(self):
        self.remaining = self.max_total_delta


DEFAULT_SOURCE_BUDGETS = {
    'smart_autotuner':   SourceBudget(max_total_delta=0.50, max_per_param_delta=0.12),  # raised from 0.30/0.08
    'feedback_learning': SourceBudget(max_total_delta=0.40, max_per_param_delta=0.10),
    'strategy_bandit':   SourceBudget(max_total_delta=0.20, max_per_param_delta=0.06),
    'falloff_strategy':  SourceBudget(max_total_delta=0.50, max_per_param_delta=0.15),
    'mode_intelligence': SourceBudget(max_total_delta=0.25, max_per_param_delta=0.08),
    'context_engine':    SourceBudget(max_total_delta=0.60, max_per_param_delta=0.15),  # V6.1b: pulled back (was 0.80/0.20 — too permissive)
}


# ---------------------------------------------------------------------------
# Modifier Resolver
# ---------------------------------------------------------------------------

class ModifierResolver:
    """Collects intents from all V6 subsystems and merges them.

    Usage::

        resolver = ModifierResolver()
        resolver.begin_frame(current_params)
        resolver.add(ModifierIntent('brightness_global', Direction.UP, 0.05,
                                    'feedback_learning', Priority.FEEDBACK))
        resolver.add(ModifierIntent('brightness_global', Direction.DOWN, 0.08,
                                    'context_engine', Priority.CONTEXT))
        results = resolver.resolve()
        for r in results:
            meta.set(r.parameter, r.new_value)

    Parameters
    ----------
    source_budgets : dict | None
        Per-source budget overrides.  Keys are source names, values
        are ``SourceBudget`` instances.
    safe_floors : dict | None
        Minimum values per parameter that SAFETY intents enforce.
    caps : dict | None
        Maximum values per parameter.
    """

    def __init__(
        self,
        source_budgets: Dict[str, SourceBudget] = None,
        safe_floors: Dict[str, float] = None,
        caps: Dict[str, float] = None,
    ):
        self._source_budgets = source_budgets or dict(DEFAULT_SOURCE_BUDGETS)
        self._safe_floors = safe_floors or {}
        self._caps = caps or {}

        self._intents: List[ModifierIntent] = []
        self._current_params: Dict[str, float] = {}
        self._log: List[dict] = []

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self, current_params: Dict[str, float]):
        """Start a new resolution frame with current parameter values."""
        self._intents.clear()
        self._current_params = dict(current_params)
        # Reset per-source budgets
        for budget in self._source_budgets.values():
            budget.reset()

    def add(self, intent: ModifierIntent):
        """Add an intent from a subsystem."""
        self._intents.append(intent)

    def add_many(self, intents: List[ModifierIntent]):
        """Add multiple intents."""
        self._intents.extend(intents)

    def resolve(self) -> List[ResolvedModifier]:
        """Resolve all accumulated intents and return modifiers."""
        # Group intents by parameter
        by_param: Dict[str, List[ModifierIntent]] = {}
        for intent in self._intents:
            by_param.setdefault(intent.parameter, []).append(intent)

        results = []
        for param, intents in by_param.items():
            result = self._resolve_param(param, intents)
            if result is not None:
                results.append(result)

        self._log.append({
            'intent_count': len(self._intents),
            'params_modified': len(results),
            'conflicts': sum(1 for r in results if r.conflict),
        })
        if len(self._log) > 200:
            self._log.pop(0)

        return results

    @property
    def frame_log(self) -> List[dict]:
        return list(self._log)

    # ------------------------------------------------------------------
    # Resolution algorithm
    # ------------------------------------------------------------------

    def _resolve_param(
        self,
        param: str,
        intents: List[ModifierIntent],
    ) -> Optional[ResolvedModifier]:
        """Resolve intents for a single parameter.

        Algorithm:
        1. Sort by priority (ascending = strongest first).
        2. Group by direction.
        3. If conflicting directions exist, resolve by priority:
           - Highest-priority direction wins but is modulated by the
             opposing direction's strength.
        4. Apply per-source budget clamping.
        5. Apply safety floors/caps.
        """
        current = self._current_params.get(param, 0.5)
        if not intents:
            return None

        # Sort by priority
        intents.sort(key=lambda i: (i.priority.value, -i.confidence))

        # Compute weighted delta from all intents
        # Group: positive (UP/SET>current/MULT>1) vs negative
        positive_sum = 0.0
        positive_weight = 0.0
        negative_sum = 0.0
        negative_weight = 0.0
        dominant_source = intents[0].source

        for intent in intents:
            delta = intent.effective_delta(current)

            # Apply source budget
            budget = self._source_budgets.get(intent.source)
            if budget:
                delta = budget.clamp(delta)

            # Weight = confidence / priority (lower priority num = higher weight)
            weight = intent.confidence * (6 - intent.priority.value)  # 1–5 scale inversion

            if delta >= 0:
                positive_sum += delta * weight
                positive_weight += weight
            else:
                negative_sum += delta * weight
                negative_weight += weight

        # Determine conflict
        conflict = positive_weight > 0 and negative_weight > 0

        # Compute final delta: net of weighted positive and negative
        total_weight = positive_weight + negative_weight
        if total_weight < 1e-10:
            return None

        net_delta = (positive_sum + negative_sum) / total_weight

        # Apply net delta
        new_value = current + net_delta

        # Safety floor
        floor = self._safe_floors.get(param)
        if floor is not None:
            new_value = max(new_value, floor)

        # Cap
        cap = self._caps.get(param)
        if cap is not None:
            new_value = min(new_value, cap)

        actual_delta = new_value - current

        if abs(actual_delta) < 0.001:
            return None

        return ResolvedModifier(
            parameter=param,
            old_value=round(current, 5),
            new_value=round(new_value, 5),
            delta=round(actual_delta, 5),
            dominant_source=dominant_source,
            intent_count=len(intents),
            conflict=conflict,
        )

    # ------------------------------------------------------------------
    # Convenience: build intents from V6 subsystem outputs
    # ------------------------------------------------------------------

    @staticmethod
    def intents_from_autotuner_deltas(
        deltas: Dict[str, float],
    ) -> List[ModifierIntent]:
        """Convert SmartAutoTuner deltas → intents."""
        intents = []
        for param, delta in deltas.items():
            if abs(delta) < 0.001:
                continue
            direction = Direction.UP if delta > 0 else Direction.DOWN
            intents.append(ModifierIntent(
                parameter=param,
                direction=direction,
                strength=abs(delta),
                source='smart_autotuner',
                priority=Priority.STRATEGY,
                confidence=0.7,
            ))
        return intents

    @staticmethod
    def intents_from_feedback(
        feedback_modifiers: dict,
    ) -> List[ModifierIntent]:
        """Convert FeedbackLearningV6 modifiers → intents.

        Feedback modifiers are multiplicative weights centered on 1.0.
        """
        intents = []
        for param, weight in feedback_modifiers.items():
            if abs(weight - 1.0) < 0.01:
                continue

            # Special V6 feedback params
            if param in ('falloff_reach_mult', 'falloff_width_mult'):
                intents.append(ModifierIntent(
                    parameter=param,
                    direction=Direction.MULTIPLY,
                    strength=weight,
                    source='feedback_learning',
                    priority=Priority.FEEDBACK,
                    confidence=0.8,
                ))
                continue

            direction = Direction.MULTIPLY
            intents.append(ModifierIntent(
                parameter=param,
                direction=direction,
                strength=weight,
                source='feedback_learning',
                priority=Priority.FEEDBACK,
                confidence=0.8,
            ))
        return intents

    @staticmethod
    def intents_from_strategy(
        strategy_effect: dict,
    ) -> List[ModifierIntent]:
        """Convert StrategyBandit effect → intents.

        strategy_effect keys: brightness_mult, speed_mult, pulse_mult,
        drift_amount, scale_x, scale_y, scale_z, rotation, etc.
        """
        intents = []

        # Multiplicative effects
        mult_map = {
            'brightness_mult': 'brightness_global',
            'speed_mult': 'speed_global',
            'pulse_mult': 'pulse_global',
        }
        for effect_key, param in mult_map.items():
            val = strategy_effect.get(effect_key, 1.0)
            if abs(val - 1.0) > 0.01:
                intents.append(ModifierIntent(
                    parameter=param,
                    direction=Direction.MULTIPLY,
                    strength=val,
                    source='strategy_bandit',
                    priority=Priority.STRATEGY,
                    confidence=0.9,
                ))

        # Falloff shape overrides (passed through as SET intents)
        for key in ('scale_x', 'scale_y', 'scale_z', 'rotation'):
            val = strategy_effect.get(key)
            if val is not None:
                intents.append(ModifierIntent(
                    parameter=f'falloff_{key}',
                    direction=Direction.SET,
                    strength=val,
                    source='strategy_bandit',
                    priority=Priority.STRATEGY,
                    confidence=0.85,
                ))

        return intents

    @staticmethod
    def intents_from_mode_overlay(
        overlay,  # ModeOverlay
    ) -> List[ModifierIntent]:
        """Convert ModeIntelligence overlay → intents.

        Pre-transition blending and intensity multiplier become intents
        on relevant parameters.
        """
        intents = []

        if overlay.intensity_mult != 1.0:
            for param in ('brightness_global', 'energy', 'responsiveness'):
                intents.append(ModifierIntent(
                    parameter=param,
                    direction=Direction.MULTIPLY,
                    strength=overlay.intensity_mult,
                    source='mode_intelligence',
                    priority=Priority.AESTHETIC,
                    confidence=0.6,
                ))

        if overlay.pre_transition_blend > 0.05:
            # Nudge toward engaged-friendly values
            intents.append(ModifierIntent(
                parameter='responsiveness',
                direction=Direction.UP,
                strength=overlay.pre_transition_blend * 0.1,
                source='mode_intelligence',
                priority=Priority.AESTHETIC,
                confidence=overlay.pre_transition_blend,
            ))
            intents.append(ModifierIntent(
                parameter='brightness_global',
                direction=Direction.UP,
                strength=overlay.pre_transition_blend * 0.2,
                source='mode_intelligence',
                priority=Priority.AESTHETIC,
                confidence=overlay.pre_transition_blend,
            ))

        return intents

    @staticmethod
    def intents_from_context(
        context,  # PredictiveContext
        current_params: Dict[str, float],
    ) -> List[ModifierIntent]:
        """Convert PredictiveContextEngine context → safety/regime intents."""
        intents = []

        # If regime is 'dead', BOOST outputs to attract attention.
        # On a passive-heavy sidewalk, quiet periods are opportunities
        # to be more expressive, not to dim down.
        regime = getattr(context, 'regime', 'steady')
        if regime == 'dead':
            for param in ('brightness_global', 'exploration'):
                intents.append(ModifierIntent(
                    parameter=param,
                    direction=Direction.UP,
                    strength=0.05,
                    source='context_engine',
                    priority=Priority.CONTEXT,
                    confidence=0.8,
                ))
            # Gently push energy up too
            intents.append(ModifierIntent(
                parameter='energy',
                direction=Direction.UP,
                strength=0.03,
                source='context_engine',
                priority=Priority.CONTEXT,
                confidence=0.7,
            ))
        elif regime == 'trickle':
            # Moderate boost to be noticeable
            for param in ('brightness_global', 'exploration'):
                intents.append(ModifierIntent(
                    parameter=param,
                    direction=Direction.UP,
                    strength=0.03,
                    source='context_engine',
                    priority=Priority.CONTEXT,
                    confidence=0.7,
                ))
        elif regime == 'event':
            # Anomalous event — boost responsiveness as context-level
            intents.append(ModifierIntent(
                parameter='responsiveness',
                direction=Direction.UP,
                strength=0.08,
                source='context_engine',
                priority=Priority.CONTEXT,
                confidence=0.8,
            ))

        return intents
