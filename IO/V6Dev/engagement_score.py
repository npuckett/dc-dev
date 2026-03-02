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
    """Weights for each score component.  Must sum to 1.0.

    Calibrated for passive-heavy sidewalk installations where most
    pedestrians pass through without entering the active zone.
    """
    conversion_rate: float = 0.20      # lowered: active/total is typically 1-3%
    dwell_depth: float = 0.25          # kept: valuable when it happens
    mode_diversity: float = 0.15       # lowered slightly
    passive_awareness: float = 0.15    # NEW: rewards flow detection & mode transitions
    proactive_reach: float = 0.10      # NEW: rewards system expressiveness during quiet
    return_visits: float = 0.10        # lowered: unreliable at scale
    parameter_stability: float = 0.05  # lowered: less important than expressiveness

    def __post_init__(self):
        total = (self.conversion_rate + self.dwell_depth + self.mode_diversity
                 + self.passive_awareness + self.proactive_reach
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

        # Proactive reach tracking: count of gestures/strategy attempts
        # in the current scoring window (reset each cycle externally)
        self._gesture_attempts: int = 0
        self._strategy_attempts: int = 0
        self._mode_transitions: int = 0

        # Daytime score floor: ensures gradient estimator always has signal
        self._daytime_floor: float = 0.15  # minimum score during 7am-11pm

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
        # Normalise: 3% engagement maps to score 1.0 (calibrated for
        # passive-heavy sidewalk where typical active ratio is 1-3%)
        comps['conversion_rate'] = min(1.0, conversion / 0.03)
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

        # --- 6. Passive awareness (V6.1 NEW) -----------------------------
        # Rewards the system for detecting and responding to passive traffic
        # even when people don't enter the active zone
        flow_info = behavior_status.get('flow', {})
        flow_strength = flow_info.get('strength', 0.0) if isinstance(flow_info, dict) else 0.0
        mode = behavior_status.get('mode', 'idle')
        passive_score = 0.0
        # Flow detection: system is aware of pedestrian movement
        if passive > 0:
            passive_score += min(0.4, passive / 50.0)  # some passive traffic = awareness
        # Flow strength: system is tracking directional movement
        passive_score += min(0.3, flow_strength * 0.5)
        # Mode variety: system is responding (not stuck in idle)
        if mode != 'idle':
            passive_score += 0.3
        comps['passive_awareness'] = min(1.0, passive_score)
        raw['flow_strength'] = flow_strength

        # --- 7. Proactive reach (V6.1 NEW) --------------------------------
        # Rewards the system for trying to attract attention during quiet periods
        reach_score = 0.0
        # Gesture attempts (injected externally via record_attempt())
        reach_score += min(0.4, self._gesture_attempts * 0.1)
        # Strategy attempts from bandit
        reach_score += min(0.3, self._strategy_attempts * 0.1)
        # Mode transitions show the system is actively adapting
        reach_score += min(0.3, self._mode_transitions * 0.15)
        comps['proactive_reach'] = min(1.0, reach_score)
        raw['gesture_attempts'] = self._gesture_attempts
        raw['strategy_attempts'] = self._strategy_attempts

        # --- Weighted sum ------------------------------------------------
        score = (
            self.w.conversion_rate     * comps['conversion_rate']
            + self.w.dwell_depth       * comps['dwell_depth']
            + self.w.mode_diversity    * comps['mode_diversity']
            + self.w.passive_awareness * comps.get('passive_awareness', 0.0)
            + self.w.proactive_reach   * comps.get('proactive_reach', 0.0)
            + self.w.return_visits     * comps['return_visits']
            + self.w.parameter_stability * comps['parameter_stability']
        )
        score = max(0.0, min(1.0, score))

        # Daytime floor: ensure gradient estimator always has signal
        # during operating hours (7am-11pm)
        current_hour = time.localtime().tm_hour
        if 7 <= current_hour <= 23:
            score = max(self._daytime_floor, score)

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

    def smoothed_score(self, window: int = 8) -> float:
        """Exponentially-weighted mean of the last *window* scores.

        Useful for the auto-tuner to avoid reacting to single-cycle noise.
        Window reduced from 12 to 8 for faster responsiveness to brief
        engagement events on quiet sidewalks.
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
    # Proactive tracking (call from integration layer)
    # ------------------------------------------------------------------

    def record_gesture_attempt(self):
        """Record that a gesture was attempted (for proactive_reach score)."""
        self._gesture_attempts += 1

    def record_strategy_attempt(self):
        """Record that a bandit strategy was tried."""
        self._strategy_attempts += 1

    def record_mode_transition(self):
        """Record a mode transition."""
        self._mode_transitions += 1

    def reset_proactive_counters(self):
        """Reset attempt counters (call each scoring cycle)."""
        self._gesture_attempts = 0
        self._strategy_attempts = 0
        self._mode_transitions = 0

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

        # 1. Conversion (normalised to 3% for passive-heavy installations)
        active_total = sum(h.get('active_count', 0) for h in hourly)
        passive_total = sum(h.get('passive_count', 0) for h in hourly)
        total = max(1, active_total + passive_total)
        conv = min(1.0, (active_total / total) / 0.03)

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

        # 6. Passive awareness – from traffic and mode distribution
        passive_awareness = 0.0
        total_passive = sum(h.get('passive_count', 0) for h in hourly)
        if total_passive > 0:
            passive_awareness += 0.4  # there was passive traffic
        mode_dist = light.get('mode_distribution', {})
        idle_frac = mode_dist.get('idle', 1.0)
        if idle_frac < 0.7:
            passive_awareness += 0.3  # system wasn't stuck in idle
        flow_frac = mode_dist.get('flow', 0.0)
        if flow_frac > 0.1:
            passive_awareness += 0.3  # flow mode was used
        passive_awareness = min(1.0, passive_awareness)

        # 7. Proactive reach – from autotuner adjustments and params
        total_adjustments = auto.get('total_adjustments', 0)
        proactive = min(1.0, total_adjustments / 10000.0)  # 10k+ adjustments = active system

        score = (
            self.w.conversion_rate     * conv
            + self.w.dwell_depth       * dwell_est
            + self.w.mode_diversity    * diversity
            + self.w.passive_awareness * passive_awareness
            + self.w.proactive_reach   * proactive
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
            conv = min(1.0, (active / total) / 0.03)

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
                + self.w.passive_awareness * (0.5 if passive > 0 else 0.0)
                + self.w.proactive_reach * 0.3   # assume moderate activity
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
