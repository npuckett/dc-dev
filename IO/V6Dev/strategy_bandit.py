#!/usr/bin/env python3
"""
V6 Multi-Armed Bandit Strategy Selector
=========================================

Replaces the round-robin A/B test in V5's ``AlmostEngagedState`` with
Thompson Sampling.  Adds a new ``FALLOFF_RESHAPE`` strategy that uses
V5's anisotropic falloff to reach toward candidates.

Each strategy maintains a Beta distribution per *context bucket*
(time_period × flow_direction) so that learning is context-aware —
e.g. DRIFT_TOWARD may dominate during rush-hour RTL flow while
PAUSE_AND_LOOK wins during quiet evenings.

Priors persist across restarts via JSON file.

Usage::

    bandit = StrategyBandit()
    strategy = bandit.select(context)
    # … apply strategy, observe outcome …
    bandit.record_outcome(strategy, context, converted=True)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Strategies (extends V5 AttractionStrategy)
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    """Attraction strategies for almost-engaged pedestrians."""
    BRIGHTNESS_PULSE = 'brightness_pulse'
    DRIFT_TOWARD     = 'drift_toward'
    PAUSE_AND_LOOK   = 'pause_and_look'
    FALLOFF_RESHAPE  = 'falloff_reshape'   # V6: use anisotropic falloff


# ---------------------------------------------------------------------------
# Context bucketing
# ---------------------------------------------------------------------------

class TimePeriod(str, Enum):
    MORNING   = 'morning'     # 6–12
    AFTERNOON = 'afternoon'   # 12–18
    EVENING   = 'evening'     # 18–22
    NIGHT     = 'night'       # 22–6

class FlowBucket(str, Enum):
    LTR      = 'ltr'
    RTL      = 'rtl'
    BALANCED = 'balanced'
    NONE     = 'none'


@dataclass
class BanditContext:
    """Contextual features for strategy selection."""
    hour: int = 0
    flow_direction: float = 0.0   # -1 (RTL) to +1 (LTR)
    flow_strength: float = 0.0    # 0 to 1
    candidate_speed: float = 0.0  # cm/s
    candidate_distance: float = 0.0  # cm to active zone
    regime: str = 'steady'        # from PredictiveContextEngine

    @property
    def time_period(self) -> TimePeriod:
        if 6 <= self.hour < 12:
            return TimePeriod.MORNING
        elif 12 <= self.hour < 18:
            return TimePeriod.AFTERNOON
        elif 18 <= self.hour < 22:
            return TimePeriod.EVENING
        else:
            return TimePeriod.NIGHT

    @property
    def flow_bucket(self) -> FlowBucket:
        if self.flow_strength < 0.2:
            return FlowBucket.NONE
        if self.flow_direction > 0.3:
            return FlowBucket.LTR
        elif self.flow_direction < -0.3:
            return FlowBucket.RTL
        return FlowBucket.BALANCED

    @property
    def bucket_key(self) -> str:
        return f"{self.time_period.value}_{self.flow_bucket.value}"


# ---------------------------------------------------------------------------
# Strategy effects (consumed by the modifier resolver / behavior system)
# ---------------------------------------------------------------------------

@dataclass
class StrategyEffect:
    """Describes the desired change when applying a strategy."""
    strategy: Strategy
    duration: float = 2.0           # seconds

    # Brightness
    brightness_boost: float = 0.0   # additive DMX-scale boost
    brightness_min_boost: float = 0.0

    # Movement
    move_speed_mult: float = 1.0
    wander_interval_mult: float = 1.0
    wander_x_offset: float = 0.0    # cm, toward candidate

    # V6: Falloff shape overrides
    falloff_scale_x: Optional[float] = None
    falloff_scale_y: Optional[float] = None
    falloff_scale_z: Optional[float] = None
    falloff_rotation: Optional[float] = None


# Pre-defined effects per strategy
STRATEGY_EFFECTS: Dict[Strategy, StrategyEffect] = {
    Strategy.BRIGHTNESS_PULSE: StrategyEffect(
        strategy=Strategy.BRIGHTNESS_PULSE,
        duration=2.0,
        brightness_boost=10,
        brightness_min_boost=5,
    ),
    Strategy.DRIFT_TOWARD: StrategyEffect(
        strategy=Strategy.DRIFT_TOWARD,
        duration=2.0,
        wander_x_offset=50,  # cm toward candidate (sign set at application time)
    ),
    Strategy.PAUSE_AND_LOOK: StrategyEffect(
        strategy=Strategy.PAUSE_AND_LOOK,
        duration=2.0,
        move_speed_mult=0.3,
        wander_interval_mult=2.0,
        brightness_boost=5,
    ),
    Strategy.FALLOFF_RESHAPE: StrategyEffect(
        strategy=Strategy.FALLOFF_RESHAPE,
        duration=2.5,
        # Extend the falloff ellipsoid toward passive zone (Z stretch)
        # and slightly widen X to create an inviting "reaching" shape.
        # The rotation is set dynamically toward the candidate.
        falloff_scale_x=1.4,
        falloff_scale_y=1.0,
        falloff_scale_z=2.2,   # deep Z reach into passive zone
        falloff_rotation=0.0,  # dynamically overwritten toward candidate
        brightness_boost=5,
    ),
}


# ---------------------------------------------------------------------------
# Beta-distribution arms
# ---------------------------------------------------------------------------

@dataclass
class BetaArm:
    """Thompson Sampling arm with Beta(alpha, beta) prior."""
    successes: float = 1.0   # alpha (start at 1 = uniform prior)
    failures: float = 1.0    # beta
    attempts: int = 0

    def sample(self) -> float:
        """Draw from Beta(alpha, beta)."""
        return random.betavariate(self.successes, self.failures)

    @property
    def mean(self) -> float:
        return self.successes / (self.successes + self.failures)

    @property
    def conversion_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return (self.successes - 1) / max(1, self.attempts)  # subtract prior

    def to_dict(self) -> dict:
        return {'s': self.successes, 'f': self.failures, 'n': self.attempts}

    @classmethod
    def from_dict(cls, d: dict) -> 'BetaArm':
        return cls(successes=d.get('s', 1.0), failures=d.get('f', 1.0),
                   attempts=d.get('n', 0))


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------

class StrategyBandit:
    """Context-aware Thompson Sampling bandit for attraction strategies.

    Arms are maintained per (strategy, context_bucket).  The bandit
    automatically balances exploration and exploitation.
    """

    PERSIST_FILE = 'bandit_priors.json'

    def __init__(self, persist_dir: str = None, decay_rate: float = 0.998):
        """
        Parameters
        ----------
        persist_dir : str
            Directory for bandit_priors.json.  Defaults to V6Dev/.
        decay_rate : float
            Per-hour decay applied to successes and failures to allow
            the bandit to adapt to changing conditions.  0.998 means
            observations have a half-life of ~346 hours (~14 days).
        """
        if persist_dir is None:
            persist_dir = os.path.dirname(os.path.abspath(__file__))
        self._persist_path = os.path.join(persist_dir, self.PERSIST_FILE)
        self._decay_rate = decay_rate

        # arms[(strategy, bucket_key)] → BetaArm
        self._arms: Dict[Tuple[str, str], BetaArm] = {}

        # Cooldown tracking
        self._last_attempt_time: float = 0.0
        self._cooldown: float = 5.0  # seconds between attempts

        # Stats
        self._total_attempts: int = 0
        self._total_conversions: int = 0

        self._load()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, context: BanditContext) -> Strategy:
        """Choose the best strategy for the given context via Thompson Sampling."""
        bucket = context.bucket_key
        best_score = -1.0
        best_strategy = Strategy.BRIGHTNESS_PULSE

        for strategy in Strategy:
            arm = self._get_arm(strategy.value, bucket)
            score = arm.sample()
            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def get_effect(
        self,
        strategy: Strategy,
        candidate_x: float = 0.0,
        light_x: float = 0.0,
        candidate_z: float = 0.0,
        light_z: float = 0.0,
    ) -> StrategyEffect:
        """Return the effect for a strategy, with dynamic adjustments.

        For DRIFT_TOWARD and FALLOFF_RESHAPE, the direction toward the
        candidate is computed from positions.
        """
        base = STRATEGY_EFFECTS[strategy]

        effect = StrategyEffect(
            strategy=base.strategy,
            duration=base.duration,
            brightness_boost=base.brightness_boost,
            brightness_min_boost=base.brightness_min_boost,
            move_speed_mult=base.move_speed_mult,
            wander_interval_mult=base.wander_interval_mult,
            wander_x_offset=base.wander_x_offset,
            falloff_scale_x=base.falloff_scale_x,
            falloff_scale_y=base.falloff_scale_y,
            falloff_scale_z=base.falloff_scale_z,
            falloff_rotation=base.falloff_rotation,
        )

        # Dynamic directional adjustments
        dx = candidate_x - light_x
        dz = candidate_z - light_z

        if strategy == Strategy.DRIFT_TOWARD:
            # Sign the X offset toward candidate
            effect.wander_x_offset = math.copysign(base.wander_x_offset, dx)

        elif strategy == Strategy.FALLOFF_RESHAPE:
            # Rotate the falloff ellipsoid toward the candidate in XZ plane
            if abs(dx) > 1 or abs(dz) > 1:
                angle = math.atan2(dx, dz)  # rotation around Y axis
                effect.falloff_rotation = angle * 0.6  # partial rotation, not full snap

        return effect

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(self, strategy: Strategy, context: BanditContext, converted: bool):
        """Record whether the strategy led to a conversion."""
        bucket = context.bucket_key
        arm = self._get_arm(strategy.value, bucket)
        arm.attempts += 1
        self._total_attempts += 1

        if converted:
            arm.successes += 1.0
            self._total_conversions += 1
        else:
            arm.failures += 1.0

        # Persist periodically (every 10 outcomes)
        if self._total_attempts % 10 == 0:
            self._save()

    def can_attempt(self) -> bool:
        """Check cooldown."""
        return (time.time() - self._last_attempt_time) >= self._cooldown

    def mark_attempted(self):
        """Mark that an attempt was just started."""
        self._last_attempt_time = time.time()

    # ------------------------------------------------------------------
    # Decay (call hourly)
    # ------------------------------------------------------------------

    def apply_decay(self):
        """Decay all arms toward the uniform prior.

        This allows the bandit to adapt to non-stationary conditions
        (e.g. different strategies work in different seasons).
        """
        for arm in self._arms.values():
            arm.successes = max(1.0, arm.successes * self._decay_rate)
            arm.failures = max(1.0, arm.failures * self._decay_rate)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return summary stats for logging / display."""
        by_strategy: Dict[str, dict] = {}
        for strategy in Strategy:
            total_s = 0.0
            total_f = 0.0
            total_n = 0
            for (s, _), arm in self._arms.items():
                if s == strategy.value:
                    total_s += arm.successes - 1  # subtract prior
                    total_f += arm.failures - 1
                    total_n += arm.attempts
            rate = total_s / max(1, total_n)
            by_strategy[strategy.value] = {
                'attempts': total_n,
                'conversions': int(total_s),
                'rate': round(rate, 4),
                'mean_posterior': round(
                    (total_s + 1) / (total_s + total_f + 2), 4
                ),
            }

        return {
            'total_attempts': self._total_attempts,
            'total_conversions': self._total_conversions,
            'overall_rate': round(self._total_conversions / max(1, self._total_attempts), 4),
            'strategies': by_strategy,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_arm(self, strategy: str, bucket: str) -> BetaArm:
        key = (strategy, bucket)
        if key not in self._arms:
            self._arms[key] = BetaArm()
        return self._arms[key]

    def _save(self):
        data = {}
        for (s, b), arm in self._arms.items():
            key = f"{s}|{b}"
            data[key] = arm.to_dict()
        data['_meta'] = {
            'total_attempts': self._total_attempts,
            'total_conversions': self._total_conversions,
            'saved_at': time.time(),
        }
        try:
            with open(self._persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _load(self):
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        meta = data.pop('_meta', {})
        self._total_attempts = meta.get('total_attempts', 0)
        self._total_conversions = meta.get('total_conversions', 0)

        for key, arm_data in data.items():
            if '|' not in key:
                continue
            s, b = key.split('|', 1)
            self._arms[(s, b)] = BetaArm.from_dict(arm_data)
