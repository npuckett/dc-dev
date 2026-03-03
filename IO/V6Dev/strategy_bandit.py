#!/usr/bin/env python3
"""
V6.5 Multi-Armed Bandit — Mode Expression Strategies
=====================================================

V6.5 CHANGE: Repurposed from passive→active conversion to learning
which expression strategies produce the best dynamic range within
each mode.

Each strategy represents a "personality lean" — a set of parameter
biases that the system can apply within any mode. Thompson Sampling
with Beta distribution arms learns which lean produces the most
expressive, varied output in each (mode × time_period) context.

Strategies:
- EXPLORE_WIDE: Wider wander, more position variation
- PULSE_VARIED: More pulse speed variation, breathing rhythm changes
- SHAPE_SHIFT: Varied falloff shapes, anisotropic expression
- ENERGY_BURST: Higher energy/brightness variation
- SETTLE_DEEP: Slower, deeper, more contemplative expression

Outcomes are scored by the EngagementScorer's dynamic_range component
rather than boolean conversion.

Priors persist across restarts via JSON file.

Usage::

    bandit = StrategyBandit()
    strategy = bandit.select(context)
    effect = bandit.get_effect(strategy)
    # … apply effect, measure dynamic range …
    bandit.record_outcome(strategy, context, quality=0.72)
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
# Strategies — V6.5: Expression strategies (replaces conversion strategies)
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    """Expression strategies for dynamic behavior within modes."""
    EXPLORE_WIDE   = 'explore_wide'      # Wide wander, position variation
    PULSE_VARIED   = 'pulse_varied'      # Varied pulse rhythm
    SHAPE_SHIFT    = 'shape_shift'       # Varied falloff shapes
    ENERGY_BURST   = 'energy_burst'      # High energy/brightness variation
    SETTLE_DEEP    = 'settle_deep'       # Slow, contemplative, deep expression


# ---------------------------------------------------------------------------
# Context bucketing — V6.5: mode × time_period
# ---------------------------------------------------------------------------

class TimePeriod(str, Enum):
    MORNING   = 'morning'     # 6–12
    AFTERNOON = 'afternoon'   # 12–18
    EVENING   = 'evening'     # 18–22
    NIGHT     = 'night'       # 22–6

class ModeBucket(str, Enum):
    IDLE    = 'idle'
    FLOW    = 'flow'
    AWARE   = 'aware'
    ENGAGED = 'engaged'
    CROWD   = 'crowd'


@dataclass
class BanditContext:
    """V6.5: Context for expression strategy selection — mode-based."""
    hour: int = 0
    mode: str = 'idle'               # Current behavior mode
    passive_rate: float = 0.0        # People/min in passive zone
    regime: str = 'steady'           # From PredictiveContextEngine

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
    def mode_bucket(self) -> ModeBucket:
        try:
            return ModeBucket(self.mode)
        except ValueError:
            return ModeBucket.IDLE

    @property
    def bucket_key(self) -> str:
        return f"{self.mode_bucket.value}_{self.time_period.value}"


# ---------------------------------------------------------------------------
# Strategy effects — V6.5: Expression biases (not conversion attempts)
# ---------------------------------------------------------------------------

@dataclass
class StrategyEffect:
    """V6.5: Describes parameter biases for an expression strategy.

    These are multiplicative/additive nudges applied to behavior params
    to encourage varied output. Much gentler than the old conversion
    strategies — they lean the personality, not force it.
    """
    strategy: Strategy
    duration: float = 15.0          # seconds — longer windows for expression

    # Brightness biases
    brightness_boost: float = 0.0   # additive DMX-scale boost
    brightness_min_boost: float = 0.0

    # Movement biases
    move_speed_mult: float = 1.0
    wander_interval_mult: float = 1.0
    wander_x_offset: float = 0.0    # cm shift

    # Falloff shape biases
    falloff_scale_x: Optional[float] = None
    falloff_scale_y: Optional[float] = None
    falloff_scale_z: Optional[float] = None
    falloff_rotation: Optional[float] = None

    # V6.5: Pulse rhythm bias
    pulse_speed_mult: float = 1.0

    # V6.5: Exploration bias
    exploration_mult: float = 1.0


# Pre-defined effects per strategy — V6.5 expression strategies
STRATEGY_EFFECTS: Dict[Strategy, StrategyEffect] = {
    Strategy.EXPLORE_WIDE: StrategyEffect(
        strategy=Strategy.EXPLORE_WIDE,
        duration=20.0,
        wander_interval_mult=0.6,    # Faster target changes
        move_speed_mult=1.3,         # Move more
        exploration_mult=1.5,        # Wider range
    ),
    Strategy.PULSE_VARIED: StrategyEffect(
        strategy=Strategy.PULSE_VARIED,
        duration=15.0,
        pulse_speed_mult=0.7,        # Faster pulse (lower = faster)
        brightness_boost=3,          # Slight brightness lift
    ),
    Strategy.SHAPE_SHIFT: StrategyEffect(
        strategy=Strategy.SHAPE_SHIFT,
        duration=20.0,
        falloff_scale_x=1.4,
        falloff_scale_y=1.0,
        falloff_scale_z=1.8,
        falloff_rotation=0.3,        # Gentle rotation
    ),
    Strategy.ENERGY_BURST: StrategyEffect(
        strategy=Strategy.ENERGY_BURST,
        duration=12.0,
        brightness_boost=8,
        brightness_min_boost=4,
        move_speed_mult=1.4,
        pulse_speed_mult=0.6,        # Much faster pulse
    ),
    Strategy.SETTLE_DEEP: StrategyEffect(
        strategy=Strategy.SETTLE_DEEP,
        duration=25.0,
        move_speed_mult=0.5,         # Slow movement
        wander_interval_mult=2.0,    # Longer pauses
        pulse_speed_mult=1.5,        # Slower pulse (higher = slower)
        brightness_boost=-2,         # Slightly dimmer
        falloff_scale_z=1.3,         # Deeper reach
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
    def avg_quality(self) -> float:
        """Average quality score (V6.5)."""
        if self.attempts == 0:
            return 0.5
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
    """V6.5: Context-aware Thompson Sampling bandit for expression strategies.

    Learns which expression strategies produce the best dynamic range
    within each (mode × time_period) context. Arms are Beta distributions
    updated with continuous quality scores (not binary conversion).
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

        # V6.5: Strategy cycling — switch every 15–25s
        self._current_strategy: Optional[Strategy] = None
        self._current_context: Optional[BanditContext] = None
        self._strategy_start_time: float = 0.0
        self._strategy_min_duration: float = 15.0
        self._strategy_max_duration: float = 25.0

        # Stats
        self._total_attempts: int = 0
        self._total_quality_sum: float = 0.0

        self._load()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, context: BanditContext) -> Strategy:
        """Choose the best expression strategy for the given context.

        V6.5: Uses Thompson Sampling to pick the strategy most likely to
        produce high dynamic range in the current mode × time context.
        """
        bucket = context.bucket_key
        best_score = -1.0
        best_strategy = Strategy.EXPLORE_WIDE

        for strategy in Strategy:
            arm = self._get_arm(strategy.value, bucket)
            score = arm.sample()
            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def get_effect(self, strategy: Strategy, **kwargs) -> StrategyEffect:
        """Return the effect for a strategy.

        V6.5: Simplified — no candidate-based dynamic adjustments needed.
        Expression strategies have fixed biases.
        """
        base = STRATEGY_EFFECTS[strategy]
        return StrategyEffect(
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
            pulse_speed_mult=base.pulse_speed_mult,
            exploration_mult=base.exploration_mult,
        )

    # ------------------------------------------------------------------
    # Outcome recording — V6.5: quality-based, not boolean
    # ------------------------------------------------------------------

    def record_outcome(self, strategy: Strategy, context: BanditContext,
                       quality: float = 0.5, converted: bool = False):
        """Record the quality of a strategy's expression.

        V6.5: Uses continuous quality score (0–1) from the dynamic range
        metric. Quality > 0.5 increases successes, < 0.5 increases failures.
        The `converted` param is kept for API compat but ignored.
        """
        bucket = context.bucket_key
        arm = self._get_arm(strategy.value, bucket)
        arm.attempts += 1
        self._total_attempts += 1
        self._total_quality_sum += quality

        # Map quality to Beta updates:
        # quality=1.0 → +1.0 success, quality=0.0 → +1.0 failure
        # quality=0.5 → +0.5 each (neutral)
        arm.successes += quality
        arm.failures += (1.0 - quality)

        # Persist periodically (every 10 outcomes)
        if self._total_attempts % 10 == 0:
            self._save()

    def should_switch_strategy(self, now: float) -> bool:
        """Check if it's time to switch to a new strategy."""
        if self._current_strategy is None:
            return True
        elapsed = now - self._strategy_start_time
        return elapsed >= self._strategy_min_duration

    def set_active_strategy(self, strategy: Strategy, context: BanditContext, now: float):
        """Mark a strategy as currently active."""
        self._current_strategy = strategy
        self._current_context = context
        self._strategy_start_time = now

    @property
    def current_strategy(self) -> Optional[Strategy]:
        return self._current_strategy

    @property
    def current_context(self) -> Optional[BanditContext]:
        return self._current_context

    def can_attempt(self) -> bool:
        """Check if a new strategy can be selected (compat shim)."""
        return self.should_switch_strategy(time.time())

    def mark_attempted(self):
        """Mark that a strategy was just started (compat shim)."""
        pass  # handled by set_active_strategy

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
            avg_quality = (total_s + 1) / (total_s + total_f + 2)
            by_strategy[strategy.value] = {
                'attempts': total_n,
                'avg_quality': round(avg_quality, 4),
                'mean_posterior': round(
                    (total_s + 1) / (total_s + total_f + 2), 4
                ),
            }

        return {
            'total_attempts': self._total_attempts,
            'avg_quality': round(self._total_quality_sum / max(1, self._total_attempts), 4),
            'current_strategy': self._current_strategy.value if self._current_strategy else None,
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
            'total_quality_sum': self._total_quality_sum,
            'version': '6.5',
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
        self._total_quality_sum = meta.get('total_quality_sum', 0.0)

        for key, arm_data in data.items():
            if '|' not in key:
                continue
            s, b = key.split('|', 1)
            self._arms[(s, b)] = BetaArm.from_dict(arm_data)
