#!/usr/bin/env python3
"""
V6 Bidirectional Feedback Learning
====================================

Fixes the V5 FeedbackLearning system's three problems:

1. **Monotonic weights** — V5 weights only increase (no negative reinforcement),
   so they all converge toward 2.0 and lose discriminative power.
   V6 adds decay for non-conversion and global hourly drift toward neutral.

2. **Missing dwell quality** — V5's ``EngagementContext.dwell_duration`` is always
   0.0.  V6 uses dwell duration to scale the reinforcement signal.

3. **Thin context** — V5 has 18 weight buckets.  V6 adds speed, group size,
   and regime-based contexts for richer adaptation.

Usage::

    fb = FeedbackLearningV6()
    fb.record_engagement(context, dwell_seconds=25.0)
    fb.record_non_engagement(context)
    modifiers = fb.get_modifiers(current_context)
    # modifiers.brightness_mult, modifiers.pulse_mult, etc.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Context buckets (extended from V5's 18 to 30)
# ---------------------------------------------------------------------------

@dataclass
class FeedbackContext:
    """Rich context snapshot at the moment of engagement/non-engagement."""
    timestamp: float = 0.0
    hour: int = 0

    # Existing V5 features
    aggression_level: float = 0.0      # 0–1
    light_x_normalised: float = 0.5    # 0–1 across panel range
    light_z_normalised: float = 0.5    # 0–1 through depth
    flow_direction: float = 0.0        # -1 (RTL) to +1 (LTR)
    source_mode: str = 'idle'          # mode before engagement

    # V6 additions
    person_speed: float = 0.0          # cm/s
    group_size: int = 1                # number of people in active zone
    regime: str = 'steady'             # from PredictiveContextEngine
    dwell_duration: float = 0.0        # seconds in active zone (for engagements)
    strategy_used: str = ''            # attraction strategy that preceded this

    # Falloff shape at moment of engagement
    falloff_scale_x: float = 1.0
    falloff_scale_z: float = 1.0
    falloff_rotation: float = 0.0


def _time_bucket(hour: int) -> str:
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    return 'late_night'


def _aggression_bucket(level: float) -> str:
    if level < 0.3:
        return 'low_aggression'
    elif level < 0.6:
        return 'mid_aggression'
    return 'high_aggression'


def _position_bucket(x_norm: float) -> str:
    if x_norm < 0.33:
        return 'left_position'
    elif x_norm < 0.66:
        return 'center_position'
    return 'right_position'


def _flow_bucket(direction: float) -> str:
    if direction > 0.3:
        return 'flow_ltr'
    elif direction < -0.3:
        return 'flow_rtl'
    return 'flow_neutral'


def _speed_bucket(speed: float) -> str:
    if speed < 40:
        return 'speed_slow'       # lingering
    elif speed < 100:
        return 'speed_medium'     # casual walk
    return 'speed_fast'           # rushing


def _group_bucket(size: int) -> str:
    if size <= 1:
        return 'group_single'
    elif size <= 2:
        return 'group_pair'
    return 'group_crowd'


def _regime_bucket(regime: str) -> str:
    return f'regime_{regime}'


def get_all_buckets(ctx: FeedbackContext) -> List[str]:
    """Return all applicable bucket keys for a context."""
    return [
        _time_bucket(ctx.hour),
        _aggression_bucket(ctx.aggression_level),
        _position_bucket(ctx.light_x_normalised),
        _flow_bucket(ctx.flow_direction),
        f'from_{ctx.source_mode}',
        # V6 buckets
        _speed_bucket(ctx.person_speed),
        _group_bucket(ctx.group_size),
        _regime_bucket(ctx.regime),
    ]


# ---------------------------------------------------------------------------
# Output modifiers
# ---------------------------------------------------------------------------

@dataclass
class FeedbackModifiers:
    """Multiplicative modifiers derived from feedback weights."""
    brightness_mult: float = 1.0
    pulse_mult: float = 1.0
    move_speed_mult: float = 1.0
    # V6: falloff shape modifiers
    falloff_reach_mult: float = 1.0   # multiplier on falloff_scale_z (depth reach)
    falloff_width_mult: float = 1.0   # multiplier on falloff_scale_x (width)


# ---------------------------------------------------------------------------
# Feedback Learning V6
# ---------------------------------------------------------------------------

class FeedbackLearningV6:
    """Bidirectional feedback with decay, dwell-quality scaling, and
    expanded context buckets.

    Parameters
    ----------
    persist_dir : str
        Directory to save/load weights.
    learning_rate : float
        Base learning rate per engagement event.
    negative_rate : float
        Rate for non-engagement (fraction of learning_rate).
    weight_min, weight_max : float
        Bounds for weights.
    hourly_decay : float
        Decay factor applied once per hour, pulling weights toward 1.0.
    """

    PERSIST_FILE = 'feedback_weights_v6.json'

    def __init__(
        self,
        persist_dir: str = None,
        learning_rate: float = 0.05,       # raised from 0.03: learn faster from rare events
        negative_rate: float = 0.5,
        weight_min: float = 0.4,
        weight_max: float = 2.5,
        hourly_decay: float = 0.993,       # lowered from 0.997: forget stale quiet-hour weights faster
    ):
        if persist_dir is None:
            persist_dir = os.path.dirname(os.path.abspath(__file__))
        self._persist_path = os.path.join(persist_dir, self.PERSIST_FILE)

        self.learning_rate = learning_rate
        self.negative_rate = negative_rate
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.hourly_decay = hourly_decay

        # Weights: bucket_key → float
        self._weights: Dict[str, float] = {}

        # Ring buffer of recent contexts (for analysis)
        self._recent_engagements: List[FeedbackContext] = []
        self._recent_non_engagements: List[FeedbackContext] = []
        self._max_recent = 100

        # Quiet mode tracking: time since last engagement
        self._last_engagement_time: float = time.time()
        self._quiet_mode_threshold: float = 300.0  # 5 minutes

        # Hourly counters
        self._engagements_by_hour: Dict[int, int] = {h: 0 for h in range(24)}
        self._non_engagements_by_hour: Dict[int, int] = {h: 0 for h in range(24)}

        # Last decay timestamp
        self._last_decay_time: float = time.time()

        self._load()

    # ------------------------------------------------------------------
    # Engagement recording
    # ------------------------------------------------------------------

    def record_engagement(self, ctx: FeedbackContext, dwell_seconds: float = 0.0):
        """Record a successful engagement.

        The dwell duration scales the reward: 30s+ engagement teaches
        more than a 2s pass-through.
        """
        # Dwell quality multiplier: 0.3 (brief) to 2.0 (deep)
        if dwell_seconds >= 30:
            quality = 2.0
        elif dwell_seconds >= 10:
            quality = 1.0 + (dwell_seconds - 10) / 20  # 1.0 → 2.0
        elif dwell_seconds >= 3:
            quality = 0.5 + 0.5 * (dwell_seconds - 3) / 7  # 0.5 → 1.0
        else:
            quality = max(0.3, dwell_seconds / 3 * 0.5)  # 0 → 0.5

        buckets = get_all_buckets(ctx)
        delta = self.learning_rate * quality

        for bucket in buckets:
            current = self._weights.get(bucket, 1.0)
            new_val = min(self.weight_max, current + delta)
            self._weights[bucket] = new_val

        # Record
        ctx.dwell_duration = dwell_seconds
        self._recent_engagements.append(ctx)
        if len(self._recent_engagements) > self._max_recent:
            self._recent_engagements.pop(0)
        self._engagements_by_hour[ctx.hour % 24] = \
            self._engagements_by_hour.get(ctx.hour % 24, 0) + 1
        self._last_engagement_time = time.time()

    def record_non_engagement(self, ctx: FeedbackContext):
        """Record a failed engagement attempt (passive person left).

        Applies negative reinforcement to the matching buckets.
        """
        buckets = get_all_buckets(ctx)
        delta = self.learning_rate * self.negative_rate

        for bucket in buckets:
            current = self._weights.get(bucket, 1.0)
            new_val = max(self.weight_min, current - delta)
            self._weights[bucket] = new_val

        self._recent_non_engagements.append(ctx)
        if len(self._recent_non_engagements) > self._max_recent:
            self._recent_non_engagements.pop(0)
        self._non_engagements_by_hour[ctx.hour % 24] = \
            self._non_engagements_by_hour.get(ctx.hour % 24, 0) + 1

    # ------------------------------------------------------------------
    # Hourly decay
    # ------------------------------------------------------------------

    def maybe_apply_decay(self):
        """Apply hourly decay if enough time has passed.

        Call this every auto-tune cycle; it only acts once per hour.
        """
        now = time.time()
        if now - self._last_decay_time < 3600:
            return

        self._last_decay_time = now
        for bucket in list(self._weights.keys()):
            current = self._weights[bucket]
            # Decay toward 1.0 (neutral)
            self._weights[bucket] = 1.0 + (current - 1.0) * self.hourly_decay

        self._save()

    # ------------------------------------------------------------------
    # Modifier computation
    # ------------------------------------------------------------------

    def get_modifiers(self, ctx: FeedbackContext) -> FeedbackModifiers:
        """Compute multiplicative modifiers for the current context.

        Averages the weights of all matching buckets, then maps the
        average to modifier ranges.
        """
        buckets = get_all_buckets(ctx)
        weights = [self._weights.get(b, 1.0) for b in buckets]

        if not weights:
            return FeedbackModifiers()

        avg_weight = sum(weights) / len(weights)

        # Map avg_weight [weight_min, weight_max] → modifier range
        # weight < 1.0 = reduce, weight > 1.0 = boost
        # Brightness: [0.8, 1.4] — V6.1e: tightened (was [0.7, 1.6])
        brightness_mult = 0.55 + avg_weight * 0.35
        brightness_mult = max(0.8, min(1.4, brightness_mult))

        # Pulse: [0.88, 1.15] — V6.1e: tightened (was [0.85, 1.2])
        pulse_mult = 0.70 + avg_weight * 0.22
        pulse_mult = max(0.88, min(1.15, pulse_mult))

        # Move speed: [0.88, 1.12] — V6.1e: tightened (was [0.85, 1.15])
        speed_mult = 0.72 + avg_weight * 0.18
        speed_mult = max(0.88, min(1.12, speed_mult))

        # Falloff reach (Z scale): [0.8, 1.5]
        reach_mult = 0.5 + avg_weight * 0.4
        reach_mult = max(0.8, min(1.5, reach_mult))

        # Falloff width (X scale): [0.9, 1.3]
        width_mult = 0.7 + avg_weight * 0.25
        width_mult = max(0.9, min(1.3, width_mult))

        # Quiet mode boost: if no engagement for > 5 minutes,
        # increase brightness and pulse to be more visible
        quiet_boost = 1.0
        time_since_engagement = time.time() - self._last_engagement_time
        if time_since_engagement > self._quiet_mode_threshold:
            # Ramp up over the next 10 minutes: 1.0 → 1.08 — V6.1e: reduced (was 1.12)
            quiet_ramp = min(1.0, (time_since_engagement - self._quiet_mode_threshold) / 600.0)
            quiet_boost = 1.0 + quiet_ramp * 0.08
            brightness_mult *= quiet_boost
            pulse_mult *= min(1.2, pulse_mult * (1.0 + quiet_ramp * 0.05))

        return FeedbackModifiers(
            brightness_mult=brightness_mult,
            pulse_mult=pulse_mult,
            move_speed_mult=speed_mult,
            falloff_reach_mult=reach_mult,
            falloff_width_mult=width_mult,
        )

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_weight_summary(self) -> Dict[str, float]:
        """Return all weights, sorted by distance from neutral (1.0)."""
        sorted_w = sorted(
            self._weights.items(),
            key=lambda kv: abs(kv[1] - 1.0),
            reverse=True,
        )
        return dict(sorted_w)

    def get_discriminative_power(self) -> float:
        """0–1 measure of how spread the weights are.

        0 = all weights at 1.0 (no learning).
        1 = weights at extremes (strong differentiation).
        """
        if not self._weights:
            return 0.0
        deviations = [abs(w - 1.0) for w in self._weights.values()]
        max_possible = max(self.weight_max - 1.0, 1.0 - self.weight_min)
        avg_dev = sum(deviations) / len(deviations)
        return min(1.0, avg_dev / max_possible)

    @property
    def total_engagements(self) -> int:
        return sum(self._engagements_by_hour.values())

    @property
    def total_non_engagements(self) -> int:
        return sum(self._non_engagements_by_hour.values())

    def get_stats(self) -> dict:
        return {
            'total_engagements': self.total_engagements,
            'total_non_engagements': self.total_non_engagements,
            'discriminative_power': round(self.get_discriminative_power(), 3),
            'num_buckets': len(self._weights),
            'top_positive': dict(
                sorted(
                    ((k, v) for k, v in self._weights.items() if v > 1.0),
                    key=lambda kv: kv[1], reverse=True
                )[:5]
            ),
            'top_negative': dict(
                sorted(
                    ((k, v) for k, v in self._weights.items() if v < 1.0),
                    key=lambda kv: kv[1]
                )[:5]
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        data = {
            'weights': self._weights,
            'engagements_by_hour': self._engagements_by_hour,
            'non_engagements_by_hour': self._non_engagements_by_hour,
            'last_decay_time': self._last_decay_time,
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

        self._weights = data.get('weights', {})
        self._engagements_by_hour = {
            int(k): v for k, v in data.get('engagements_by_hour', {}).items()
        }
        self._non_engagements_by_hour = {
            int(k): v for k, v in data.get('non_engagements_by_hour', {}).items()
        }
        self._last_decay_time = data.get('last_decay_time', time.time())

    def save(self):
        """Public save (call on shutdown)."""
        self._save()
