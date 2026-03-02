#!/usr/bin/env python3
"""
V6 Predictive Context Engine
==============================

Replaces the hand-authored 6-anchor time-of-day profile with a learned model
built from the daily JSON reports in IO/reports/daily/.

Capabilities
------------
* Hourly traffic prediction (people, active ratio, flow balance)
* Day-of-week weighting with exponential recency decay
* Optimal parameter lookup per hour from best-scoring historical days
* Anomaly detection (current vs predicted) to modulate auto-tuner aggressiveness
* Regime classification: dead / trickle / steady / rush / event

Usage::

    engine = PredictiveContextEngine(reports_dir="IO/reports/daily")
    engine.load()
    ctx = engine.get_context(hour=16, day_of_week=3)
    # ctx.expected_people, ctx.optimal_params, ctx.confidence, ctx.regime, ...
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .engagement_score import EngagementScorer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HourlyPrediction:
    """Predicted state for a given hour+day_of_week context."""
    hour: int
    expected_people: float        # average unique people this hour
    expected_active_ratio: float  # expected active / (active+passive)
    expected_flow_balance: float  # -1 (RTL) to +1 (LTR)
    optimal_params: Dict[str, float]  # best parameter values from highest-scoring day
    confidence: float             # 0–1, higher = more historical data for this slot
    regime: str                   # dead / trickle / steady / rush / event
    stddev_people: float          # standard deviation of people count
    anomaly_factor: float = 0.0   # how anomalous the current traffic is (0 = normal, >2 = extreme)


@dataclass
class DailyProfile:
    """Summarised daily data loaded from one report JSON."""
    date: str
    day_of_week: int             # 0=Mon … 6=Sun
    total_people: int
    hourly_people: Dict[int, int]
    hourly_active: Dict[int, int]
    hourly_passive: Dict[int, int]
    hourly_flow_ltr: Dict[int, int]
    hourly_flow_rtl: Dict[int, int]
    hourly_dominant_mode: Dict[int, str]
    mode_distribution: Dict[str, float]
    optimal_values: Dict[str, float]
    engagement_score: float       # computed via EngagementScorer
    hourly_scores: Dict[int, float]


# ---------------------------------------------------------------------------
# Regime thresholds (active-zone people per hour)
# Calibrated for passive-heavy sidewalk installations where most
# pedestrians walk through the passive zone without engaging.
# Uses active-zone entries, not total pedestrian count.
# ---------------------------------------------------------------------------

REGIME_THRESHOLDS = {
    'dead':    2,      # <2 active-zone entries/hr
    'trickle': 15,     # 2-15 active-zone entries/hr
    'steady':  100,    # 15-100 active-zone entries/hr
    'rush':    500,    # 100-500 active-zone entries/hr
    # above 'rush' = 'event' if > 2σ above predicted, else still 'rush'
}

# V5 time-of-day anchor profiles for cold-start seeding.
# Used when report history < 14 days so the system has reasonable
# home values to revert toward instead of drifting to 0.5 defaults.
V5_TIME_PROFILE_ANCHORS = {
    0:  {'responsiveness': 0.35, 'energy': 0.30, 'brightness_global': 0.70,
         'speed_global': 0.50, 'pulse_global': 0.50, 'exploration': 0.40},
    6:  {'responsiveness': 0.45, 'energy': 0.45, 'brightness_global': 0.85,
         'speed_global': 0.60, 'pulse_global': 0.55, 'exploration': 0.50},
    10: {'responsiveness': 0.55, 'energy': 0.55, 'brightness_global': 1.00,
         'speed_global': 0.70, 'pulse_global': 0.65, 'exploration': 0.55},
    14: {'responsiveness': 0.60, 'energy': 0.58, 'brightness_global': 1.10,
         'speed_global': 0.75, 'pulse_global': 0.70, 'exploration': 0.55},
    18: {'responsiveness': 0.55, 'energy': 0.52, 'brightness_global': 1.00,
         'speed_global': 0.65, 'pulse_global': 0.60, 'exploration': 0.50},
    22: {'responsiveness': 0.40, 'energy': 0.35, 'brightness_global': 0.75,
         'speed_global': 0.55, 'pulse_global': 0.50, 'exploration': 0.45},
}

# Minimum confidence floor during cold-start period (< 14 days of reports)
COLD_START_MIN_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PredictiveContextEngine:
    """Learns hourly patterns from stored daily reports.

    The engine is read-only with respect to the report files — it never
    modifies them.  Call ``load()`` on start-up (or periodically) to
    ingest new reports.
    """

    def __init__(self, reports_dir: str = None, max_days: int = 60):
        if reports_dir is None:
            # Default: IO/reports/daily relative to this file
            reports_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'reports', 'daily',
            )
        self.reports_dir = reports_dir
        self.max_days = max_days
        self._profiles: List[DailyProfile] = []
        self._scorer = EngagementScorer()

        # Caches keyed by (hour, day_of_week)
        self._predictions: Dict[Tuple[int, int], HourlyPrediction] = {}

        # Current live anomaly factor (updated externally each cycle)
        self._current_anomaly: float = 0.0

        # Use active-zone counts for regime classification
        self._use_active_zone_counts: bool = True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Load and score all daily reports.  Returns count loaded."""
        self._profiles.clear()
        self._predictions.clear()

        index_path = os.path.join(self.reports_dir, '_index.json')
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            files = [e.get('filename', e.get('file', ''))
                     for e in index if isinstance(e, dict)]
        else:
            # Fallback: glob .json files
            files = sorted(f for f in os.listdir(self.reports_dir)
                           if f.endswith('.json') and f != '_index.json')

        for fname in files[-self.max_days:]:
            path = os.path.join(self.reports_dir, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as f:
                    report = json.load(f)
                profile = self._parse_report(report)
                if profile is not None:
                    self._profiles.append(profile)
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        self._build_prediction_cache()
        return len(self._profiles)

    def _parse_report(self, report: dict) -> Optional[DailyProfile]:
        """Convert a raw JSON report into a DailyProfile."""
        date_str = report.get('date', '')
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None

        summary = report.get('summary', {})
        hourly = report.get('hourly_trends', [])
        light = report.get('light_behavior', {})
        auto = report.get('auto_tuning', {})

        h_people: Dict[int, int] = {}
        h_active: Dict[int, int] = {}
        h_passive: Dict[int, int] = {}
        h_flow_ltr: Dict[int, int] = {}
        h_flow_rtl: Dict[int, int] = {}
        h_mode: Dict[int, str] = {}

        for entry in hourly:
            h = entry.get('hour', 0)
            h_people[h] = entry.get('total_people', 0)
            h_active[h] = entry.get('active_count', 0)
            h_passive[h] = entry.get('passive_count', 0)
            h_flow_ltr[h] = entry.get('flow_ltr', 0)
            h_flow_rtl[h] = entry.get('flow_rtl', 0)
            h_mode[h] = entry.get('dominant_mode', 'idle')

        # Engagement score for the whole day
        eng_score = self._scorer.compute_from_report(report)
        hourly_scores = self._scorer.compute_hourly_from_report(report)

        return DailyProfile(
            date=date_str,
            day_of_week=dt.weekday(),
            total_people=summary.get('total_unique_people', 0),
            hourly_people=h_people,
            hourly_active=h_active,
            hourly_passive=h_passive,
            hourly_flow_ltr=h_flow_ltr,
            hourly_flow_rtl=h_flow_rtl,
            hourly_dominant_mode=h_mode,
            mode_distribution=light.get('mode_distribution', {}),
            optimal_values=auto.get('optimal_values', {}),
            engagement_score=eng_score,
            hourly_scores=hourly_scores,
        )

    # ------------------------------------------------------------------
    # Prediction cache
    # ------------------------------------------------------------------

    def _build_prediction_cache(self):
        """Pre-compute weighted predictions for all (hour, dow) pairs.

        Uses active-zone counts for regime classification (not total people)
        to properly handle passive-heavy sidewalk installations.
        Seeds with V5 time profiles when history < 14 days.
        """
        self._predictions.clear()
        is_cold_start = len(self._profiles) < 14

        if not self._profiles:
            # No data at all — seed from V5 time profiles
            if is_cold_start:
                self._seed_from_v5_profiles()
            return

        # Recency weight: newest profile = 1.0, oldest = 0.3
        # Also weight same-day-of-week profiles 2× higher
        now_date = datetime.now()

        for hour in range(24):
            for dow in range(7):
                people_vals: List[Tuple[float, float]] = []  # (weight, value)
                active_ratios: List[Tuple[float, float]] = []
                flow_balances: List[Tuple[float, float]] = []
                best_score = -1.0
                best_params: Dict[str, float] = {}

                for i, p in enumerate(self._profiles):
                    if hour not in p.hourly_people:
                        continue

                    # Recency weight
                    try:
                        age_days = (now_date - datetime.strptime(p.date, '%Y-%m-%d')).days
                    except ValueError:
                        age_days = 30
                    recency_w = max(0.3, 1.0 - age_days * 0.015)

                    # Same day-of-week bonus
                    dow_w = 2.0 if p.day_of_week == dow else 1.0

                    w = recency_w * dow_w

                    # Use active-zone count for regime classification
                    active_count = p.hourly_active.get(hour, 0)
                    count = active_count if self._use_active_zone_counts else p.hourly_people.get(hour, 0)
                    people_vals.append((w, count))

                    active = p.hourly_active.get(hour, 0)
                    passive = p.hourly_passive.get(hour, 0)
                    total = max(1, active + passive)
                    active_ratios.append((w, active / total))

                    ltr = p.hourly_flow_ltr.get(hour, 0)
                    rtl = p.hourly_flow_rtl.get(hour, 0)
                    flow_total = max(1, ltr + rtl)
                    balance = (ltr - rtl) / flow_total
                    flow_balances.append((w, balance))

                    # Track best scoring day for optimal params
                    h_score = p.hourly_scores.get(hour, p.engagement_score)
                    if h_score > best_score and p.optimal_values:
                        best_score = h_score
                        best_params = dict(p.optimal_values)

                if not people_vals:
                    continue

                expected_people = self._weighted_mean(people_vals)
                stddev_people = self._weighted_stddev(people_vals)
                expected_active = self._weighted_mean(active_ratios)
                expected_flow = self._weighted_mean(flow_balances)

                # Confidence: based on sample count and recency
                n_samples = len(people_vals)
                confidence = min(1.0, n_samples / 10.0)  # 10+ samples → full confidence
                # Cold-start: ensure minimum confidence so reversion targets something useful
                if is_cold_start:
                    confidence = max(COLD_START_MIN_CONFIDENCE, confidence)

                # Regime classification (uses active-zone counts)
                regime = self._classify_regime(expected_people, stddev_people)

                self._predictions[(hour, dow)] = HourlyPrediction(
                    hour=hour,
                    expected_people=expected_people,
                    expected_active_ratio=expected_active,
                    expected_flow_balance=expected_flow,
                    optimal_params=best_params,
                    confidence=confidence,
                    regime=regime,
                    stddev_people=stddev_people,
                )

        # During cold start, fill any gaps from V5 time profiles
        if is_cold_start:
            self._seed_from_v5_profiles()

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_context(
        self,
        hour: int = None,
        day_of_week: int = None,
        current_people: float = None,
    ) -> HourlyPrediction:
        """Get the predicted context for a given hour and day.

        If ``current_people`` is provided, the anomaly_factor is computed
        and the regime may be upgraded to 'event'.
        """
        if hour is None:
            hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()

        pred = self._predictions.get((hour, day_of_week))
        if pred is None:
            # No data — return a safe default
            return HourlyPrediction(
                hour=hour,
                expected_people=0,
                expected_active_ratio=0.05,
                expected_flow_balance=0.0,
                optimal_params={},
                confidence=0.0,
                regime='trickle',
                stddev_people=0.0,
            )

        # Compute anomaly if live data available
        if current_people is not None and pred.stddev_people > 0:
            z_score = abs(current_people - pred.expected_people) / max(1.0, pred.stddev_people)
            pred.anomaly_factor = z_score
            if z_score > 2.0 and current_people > pred.expected_people:
                pred.regime = 'event'
        else:
            pred.anomaly_factor = 0.0

        return pred

    def get_optimal_home_values(self, hour: int = None, day_of_week: int = None) -> Dict[str, float]:
        """Return learned optimal parameter values for the given time slot.

        These replace the static ``_time_profiles`` anchors in V5's
        AutoTuningManager._get_current_home_values().
        """
        ctx = self.get_context(hour, day_of_week)
        return ctx.optimal_params

    def get_budget_multiplier(self, hour: int = None, day_of_week: int = None) -> float:
        """Context-dependent budget multiplier.

        INVERTED for dead/trickle: on a quiet sidewalk, the system should
        be MORE expressive to attract attention, not frozen.
        During anomalies → 3× to allow rapid adaptation.
        """
        ctx = self.get_context(hour, day_of_week)

        if ctx.anomaly_factor > 2.0:
            return 3.0

        # V6.1d: budget multipliers reduced — prevent always-full budget
        # which made tuning effectively unconstrained
        regime_mults = {
            'dead': 1.2,       # V6.1d: was 1.5
            'trickle': 1.0,    # V6.1d: was 1.25 — no bonus for trickle
            'steady': 1.0,
            'rush': 1.25,      # V6.1d: was 1.5
            'event': 2.0,      # V6.1d: was 3.0
        }
        return regime_mults.get(ctx.regime, 1.0)

    def get_tune_interval(self, hour: int = None, day_of_week: int = None) -> float:
        """Recommended auto-tune interval in seconds.

        V6.1d: intervals lengthened across the board to reduce parameter churn.
        The system was adjusting 12 params every 5-8s, causing erratic behavior.
        """
        ctx = self.get_context(hour, day_of_week)
        regime_intervals = {
            'dead': 30.0,      # V6.1d: was 15s — calmer during quiet
            'trickle': 15.0,   # V6.1d: was 8s — less frequent tuning
            'steady': 10.0,    # V6.1d: was 5s
            'rush': 6.0,       # V6.1d: was 3s
            'event': 4.0,      # V6.1d: was 2s
        }
        return regime_intervals.get(ctx.regime, 10.0)

    def get_mean_reversion_strength(self, hour: int = None, day_of_week: int = None) -> float:
        """Anomaly-aware mean reversion strength multiplier.

        During normal operation → 1.0 (standard reversion).
        During anomalies → 0.3 (let the system explore freely).
        Low confidence → 0.5 (don't revert strongly to uncertain targets).
        """
        ctx = self.get_context(hour, day_of_week)

        if ctx.anomaly_factor > 2.0:
            return 0.3

        # Scale with confidence: low confidence → weaker reversion
        return 0.5 + 0.5 * ctx.confidence

    # ------------------------------------------------------------------
    # Cold-start seeding from V5 time profiles
    # ------------------------------------------------------------------

    def _seed_from_v5_profiles(self):
        """Seed prediction cache with V5 time-of-day anchor profiles.

        Used when report history < 14 days to provide reasonable home
        values instead of defaulting everything to 0.5.
        """
        anchor_hours = sorted(V5_TIME_PROFILE_ANCHORS.keys())
        for hour in range(24):
            for dow in range(7):
                if (hour, dow) in self._predictions:
                    continue  # don't override existing data

                # Interpolate V5 anchor values for this hour
                params = self._interpolate_v5_anchors(hour, anchor_hours)

                self._predictions[(hour, dow)] = HourlyPrediction(
                    hour=hour,
                    expected_people=0,
                    expected_active_ratio=0.02,
                    expected_flow_balance=0.0,
                    optimal_params=params,
                    confidence=COLD_START_MIN_CONFIDENCE,
                    regime='trickle',
                    stddev_people=0.0,
                )

    @staticmethod
    def _interpolate_v5_anchors(hour: int, anchor_hours: List[int]) -> Dict[str, float]:
        """Linearly interpolate V5 time profile anchors for a given hour."""
        # Find surrounding anchors
        lower_h = anchor_hours[0]
        upper_h = anchor_hours[-1]
        for i, h in enumerate(anchor_hours):
            if h <= hour:
                lower_h = h
            if h >= hour:
                upper_h = h
                break

        if lower_h == upper_h:
            return dict(V5_TIME_PROFILE_ANCHORS[lower_h])

        lower_vals = V5_TIME_PROFILE_ANCHORS[lower_h]
        upper_vals = V5_TIME_PROFILE_ANCHORS[upper_h]
        t = (hour - lower_h) / max(1, upper_h - lower_h)

        result = {}
        for key in lower_vals:
            lo = lower_vals[key]
            hi = upper_vals.get(key, lo)
            result[key] = lo + (hi - lo) * t
        return result

    # ------------------------------------------------------------------
    # Retrospective analysis  (called after daily report generation)
    # ------------------------------------------------------------------

    def analyse_day(self, report: dict) -> Dict[str, any]:
        """Compare a day's actual performance against predictions.

        Returns per-hour over/under-performance for the daily report system.
        """
        hourly = report.get('hourly_trends', [])
        date_str = report.get('date', '')
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            dow = dt.weekday()
        except ValueError:
            dow = 0

        results = {}
        for entry in hourly:
            hour = entry.get('hour', 0)
            actual_people = entry.get('total_people', 0)
            pred = self.get_context(hour, dow)
            expected = pred.expected_people

            if expected > 0:
                ratio = actual_people / expected
            else:
                ratio = 1.0 if actual_people == 0 else float('inf')

            results[hour] = {
                'actual_people': actual_people,
                'expected_people': expected,
                'ratio': ratio,
                'regime_predicted': pred.regime,
                'anomaly': abs(actual_people - expected) / max(1, pred.stddev_people) if pred.stddev_people > 0 else 0,
            }

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_mean(values: List[Tuple[float, float]]) -> float:
        if not values:
            return 0.0
        total_w = sum(w for w, _ in values)
        if total_w <= 0:
            return 0.0
        return sum(w * v for w, v in values) / total_w

    @staticmethod
    def _weighted_stddev(values: List[Tuple[float, float]]) -> float:
        if len(values) < 2:
            return 0.0
        mean = PredictiveContextEngine._weighted_mean(values)
        total_w = sum(w for w, _ in values)
        if total_w <= 0:
            return 0.0
        variance = sum(w * (v - mean) ** 2 for w, v in values) / total_w
        return math.sqrt(variance)

    @staticmethod
    def _classify_regime(expected_people: float, stddev: float) -> str:
        if expected_people < REGIME_THRESHOLDS['dead']:
            return 'dead'
        elif expected_people < REGIME_THRESHOLDS['trickle']:
            return 'trickle'
        elif expected_people < REGIME_THRESHOLDS['steady']:
            return 'steady'
        else:
            return 'rush'

    @property
    def profiles(self) -> List[DailyProfile]:
        return list(self._profiles)

    @property
    def days_loaded(self) -> int:
        return len(self._profiles)
