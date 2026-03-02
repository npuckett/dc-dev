#!/usr/bin/env python3
"""
V6 Composite Engagement Score
==============================

Replaces raw activity level as the auto-tuner's optimisation target.
Computes a single 0–1 score from five weighted components, all derivable
from data already collected by the V5 tracking_database and daily reports.

Usage:
    scorer = EngagementScorer(database)
    score  = scorer.compute(behavior_status, tracked_manager)
    hourly = scorer.compute_from_report(report_dict)

The score is stored per auto-tune cycle in `behavior_adjustments` and
surfaced to the SmartAutoTuner as its fitness signal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports with tracking_database

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """Weights for each score component.  Must sum to 1.0."""
    conversion_rate: float = 0.30
    dwell_depth: float = 0.25
    mode_diversity: float = 0.20
    return_visits: float = 0.15
    parameter_stability: float = 0.10

    def __post_init__(self):
        total = (self.conversion_rate + self.dwell_depth + self.mode_diversity
                 + self.return_visits + self.parameter_stability)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"ScoringWeights must sum to 1.0, got {total:.3f}")


@dataclass
class EngagementSnapshot:
    """One scored sample – stored alongside each auto-tune cycle."""
    timestamp: float
    score: float
    components: Dict[str, float]  # individual 0–1 component values
    raw: Dict[str, float]         # underlying counts/ratios for debugging


# ---------------------------------------------------------------------------
# Dwell-phase mapping (mirrors BehaviorSystem phases)
# ---------------------------------------------------------------------------

DWELL_PHASE_SCORES = {
    'notice':  0.00,
    'greet':   0.33,
    'engage':  0.66,
    'bond':    1.00,
}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class EngagementScorer:
    """Computes a composite engagement quality score.

    Designed to be called every auto-tune cycle (~5 s) from the main loop.
    All heavy DB queries are optional; the scorer degrades gracefully when
    data is unavailable.
    """

    def __init__(self, database=None, weights: ScoringWeights = None):
        self.db = database
        self.w = weights or ScoringWeights()

        # Rolling history for smoothing (last 60 samples ≈ 5 min)
        self._history: List[EngagementSnapshot] = []
        self._max_history = 60

        # Cache for param-stability (computed once per cycle from the
        # auto-tuner's own adjustment log; injected externally)
        self._last_param_stability: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        behavior_status: dict,
        tracked_manager=None,
        param_journey: Optional[Dict] = None,
    ) -> EngagementSnapshot:
        """Real-time score from live system state.

        Parameters
        ----------
        behavior_status : dict
            The dict emitted by BehaviorSystem.get_status() each frame.
            Expected keys: active_count, passive_count, mode, dwell_phase,
            dwell_durations (list), re_entry_count, unique_today.
        tracked_manager :
            TrackedPersonManager instance (optional – for richer stats).
        param_journey : dict | None
            If supplied, maps param_name → {total_movement: float}.
            Used for the stability component.
        """
        now = time.time()
        raw: Dict[str, float] = {}
        comps: Dict[str, float] = {}

        # --- 1. Conversion rate ------------------------------------------
        active = behavior_status.get('active_count', 0)
        passive = behavior_status.get('passive_count', 0)
        total_obs = max(1, active + passive)
        conversion = active / total_obs
        # Normalise: 10 % engagement maps to score 1.0 (above 10 % is bonus)
        comps['conversion_rate'] = min(1.0, conversion / 0.10)
        raw['active'] = active
        raw['passive'] = passive
        raw['conversion'] = conversion

        # --- 2. Dwell depth ----------------------------------------------
        dwell_phase = behavior_status.get('dwell_phase', 'notice')
        phase_score = DWELL_PHASE_SCORES.get(dwell_phase, 0.0)

        # If we have per-person dwell durations, average them
        dwell_durations = behavior_status.get('dwell_durations', [])
        if dwell_durations:
            # Map durations to phase scores:
            # 0-3s → notice, 3-10s → greet, 10-30s → engage, 30s+ → bond
            avg_dur = sum(dwell_durations) / len(dwell_durations)
            if avg_dur >= 30:
                dur_score = 1.0
            elif avg_dur >= 10:
                dur_score = 0.66 + 0.34 * ((avg_dur - 10) / 20)
            elif avg_dur >= 3:
                dur_score = 0.33 + 0.33 * ((avg_dur - 3) / 7)
            else:
                dur_score = 0.33 * (avg_dur / 3)
            phase_score = max(phase_score, dur_score)

        comps['dwell_depth'] = phase_score
        raw['dwell_phase'] = dwell_phase
        raw['avg_dwell_s'] = (sum(dwell_durations) / len(dwell_durations)) if dwell_durations else 0.0

        # --- 3. Mode diversity -------------------------------------------
        mode_dist = behavior_status.get('mode_distribution', {})
        comps['mode_diversity'] = self._shannon_entropy(mode_dist)
        raw['mode_dist'] = mode_dist

        # --- 4. Return visits --------------------------------------------
        re_entries = behavior_status.get('re_entry_count', 0)
        unique = max(1, behavior_status.get('unique_today', 1))
        return_ratio = re_entries / unique
        # 5 % return rate maps to 1.0
        comps['return_visits'] = min(1.0, return_ratio / 0.05)
        raw['re_entries'] = re_entries
        raw['unique_today'] = unique

        # --- 5. Parameter stability --------------------------------------
        if param_journey:
            total_movement = sum(
                v.get('total_movement', 0.0) for v in param_journey.values()
            )
            # Lower movement = more stable.  Scale: 0 movement → 1.0,
            # movement of 5.0 (very noisy) → 0.0
            comps['parameter_stability'] = max(0.0, 1.0 - total_movement / 5.0)
        else:
            comps['parameter_stability'] = self._last_param_stability

        self._last_param_stability = comps['parameter_stability']

        # --- Weighted sum ------------------------------------------------
        score = (
            self.w.conversion_rate     * comps['conversion_rate']
            + self.w.dwell_depth       * comps['dwell_depth']
            + self.w.mode_diversity    * comps['mode_diversity']
            + self.w.return_visits     * comps['return_visits']
            + self.w.parameter_stability * comps['parameter_stability']
        )
        score = max(0.0, min(1.0, score))

        snap = EngagementSnapshot(
            timestamp=now,
            score=score,
            components=comps,
            raw=raw,
        )
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        return snap

    def smoothed_score(self, window: int = 12) -> float:
        """Exponentially-weighted mean of the last *window* scores.

        Useful for the auto-tuner to avoid reacting to single-cycle noise.
        """
        if not self._history:
            return 0.5
        recent = self._history[-window:]
        if len(recent) == 1:
            return recent[0].score
        total_w = 0.0
        weighted = 0.0
        for i, snap in enumerate(recent):
            w = 0.5 + 0.5 * (i / (len(recent) - 1))  # 0.5 → 1.0
            weighted += snap.score * w
            total_w += w
        return weighted / total_w

    # ------------------------------------------------------------------
    # Offline / report-based scoring
    # ------------------------------------------------------------------

    def compute_from_report(self, report: dict) -> float:
        """Score a whole day from a daily JSON report (IO/reports/daily/).

        Returns a single 0–1 composite.  Useful for the PredictiveContextEngine
        to rank historical days.
        """
        summary = report.get('summary', {})
        light = report.get('light_behavior', {})
        auto = report.get('auto_tuning', {})
        hourly = report.get('hourly_trends', [])

        # 1. Conversion
        active_total = sum(h.get('active_count', 0) for h in hourly)
        passive_total = sum(h.get('passive_count', 0) for h in hourly)
        total = max(1, active_total + passive_total)
        conv = min(1.0, (active_total / total) / 0.10)

        # 2. Dwell depth – not directly in reports, estimate from
        #    engaged mode fraction (higher engaged % ≈ deeper dwell)
        mode_dist = light.get('mode_distribution', {})
        engaged_frac = mode_dist.get('engaged', 0) + mode_dist.get('crowd', 0)
        dwell_est = min(1.0, engaged_frac / 0.15)  # 15% engaged → score 1

        # 3. Mode diversity
        diversity = self._shannon_entropy(mode_dist)

        # 4. Return visits – estimate from events / unique ratio
        total_events = summary.get('total_events', 0)
        unique = max(1, summary.get('total_unique_people', 1))
        # More events per person ≈ more repeated visits
        events_per_person = total_events / unique
        return_est = min(1.0, events_per_person / 50)

        # 5. Stability – from param journeys
        journeys = auto.get('param_journeys', {})
        if journeys:
            total_movement = sum(
                v.get('total_movement', 0) for v in journeys.values()
            )
            stability = max(0.0, 1.0 - total_movement / 5.0)
        else:
            stability = 0.5  # unknown

        score = (
            self.w.conversion_rate     * conv
            + self.w.dwell_depth       * dwell_est
            + self.w.mode_diversity    * diversity
            + self.w.return_visits     * return_est
            + self.w.parameter_stability * stability
        )
        return max(0.0, min(1.0, score))

    def compute_hourly_from_report(self, report: dict) -> Dict[int, float]:
        """Per-hour scores from a daily report, keyed 0–23."""
        hourly = report.get('hourly_trends', [])
        light = report.get('light_behavior', {})
        mode_dist = light.get('mode_distribution', {})

        scores: Dict[int, float] = {}
        for entry in hourly:
            hour = entry.get('hour', 0)
            active = entry.get('active_count', 0)
            passive = entry.get('passive_count', 0)
            total = max(1, active + passive)
            conv = min(1.0, (active / total) / 0.10)

            # Approximate dwell from dominant mode
            dom = entry.get('dominant_mode', 'idle')
            if dom in ('engaged', 'crowd'):
                dwell_est = 0.6
            elif dom == 'flow':
                dwell_est = 0.3
            else:
                dwell_est = 0.1

            scores[hour] = (
                self.w.conversion_rate * conv
                + self.w.dwell_depth   * dwell_est
                + self.w.mode_diversity * self._shannon_entropy(mode_dist)
                + self.w.return_visits * 0.5   # no per-hour data
                + self.w.parameter_stability * 0.5
            )
        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(dist: dict) -> float:
        """Normalised Shannon entropy of a probability distribution.

        Returns 0 if all mass on one mode, 1 if perfectly uniform over
        the non-zero modes.
        """
        vals = [v for v in dist.values() if v > 0]
        if len(vals) <= 1:
            return 0.0
        total = sum(vals)
        if total <= 0:
            return 0.0
        probs = [v / total for v in vals]
        H = -sum(p * math.log2(p) for p in probs if p > 0)
        H_max = math.log2(len(probs))
        return H / H_max if H_max > 0 else 0.0

    @property
    def history(self) -> List[EngagementSnapshot]:
        return list(self._history)
