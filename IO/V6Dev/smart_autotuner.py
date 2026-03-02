#!/usr/bin/env python3
"""
V6 Smart Auto-Tuner
=====================

Replaces V5's heuristic delta rules with a score-driven system:

1. **Fitness function**: Uses ``EngagementScorer`` instead of raw activity.
2. **Gradient estimation**: Maintains a sliding window of (param_vector, score)
   pairs and estimates which parameter changes correlate with improvements.
3. **Cross-parameter awareness**: Detects linked parameters that co-move
   during high-score periods and adjusts them together.
4. **Context-dependent budgets**: Uses ``PredictiveContextEngine`` to vary
   budget, tune interval, and mean reversion by time-of-day and anomaly.
5. **Regime-specific strategies**: Different tuning behaviour for
   dead / trickle / steady / rush / event regimes.

The SmartAutoTuner is a *drop-in replacement* for V5's ``AutoTuningManager``.
It reads the same ``MetaParameters`` and writes back via the same slider sync.

Usage::

    tuner = SmartAutoTuner(meta, sliders, database,
                           scorer=scorer, context_engine=engine)
    # In the main loop:
    result = tuner.update(behavior_status, now)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .engagement_score import EngagementScorer, EngagementSnapshot
from .predictive_context import PredictiveContextEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Parameters managed by the tuner (same set as V5)
TUNABLE_PARAMS = [
    'responsiveness', 'energy', 'attention_span', 'sociability',
    'exploration', 'memory',
    'brightness_global', 'speed_global', 'pulse_global',
    'follow_speed_global', 'dwell_influence', 'idle_trend_weight',
]

# Default ranges (overridable via overrides JSON)
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    'responsiveness':     (0.0, 1.0),
    'energy':             (0.0, 1.0),
    'attention_span':     (0.0, 1.0),
    'sociability':        (0.0, 1.0),
    'exploration':        (0.0, 1.0),
    'memory':             (0.0, 1.0),
    'brightness_global':  (0.2, 5.0),
    'speed_global':       (0.2, 2.0),
    'pulse_global':       (0.3, 3.0),
    'follow_speed_global':(0.5, 3.0),
    'dwell_influence':    (0.0, 2.0),
    'idle_trend_weight':  (0.0, 2.0),
}

# Safe floors (minimum values the tuner won't go below)
DEFAULT_SAFE_FLOORS: Dict[str, float] = {
    'responsiveness': 0.30,
    'energy': 0.25,
    'sociability': 0.20,
    'exploration': 0.15,
    'attention_span': 0.10,
    'brightness_global': 0.60,
    'speed_global': 0.35,
    'pulse_global': 0.35,
    'follow_speed_global': 0.60,
    'idle_trend_weight': 0.10,
}

# Personality params (increase with activity) vs output params (inverse adjust)
PERSONALITY_PARAMS = {'responsiveness', 'energy', 'sociability',
                      'attention_span', 'exploration', 'memory', 'follow_speed_global'}
OUTPUT_PARAMS = {'brightness_global', 'speed_global', 'pulse_global'}


@dataclass
class TunerConfig:
    """Tunable hyperparameters for the SmartAutoTuner."""
    # Base interval (overridden by PredictiveContextEngine per regime)
    base_interval: float = 5.0

    # Budget
    budget_max: float = 200.0
    budget_restore_seconds: float = 345.0
    cost_scale: float = 12.0

    # Step limits
    max_personality_step: float = 0.03
    max_output_step: float = 0.08
    min_step_threshold: float = 0.002

    # Gradient estimation
    gradient_window: int = 50        # samples for regression
    gradient_learning_rate: float = 0.01

    # Cross-parameter detection
    correlation_window: int = 30     # samples for co-movement detection
    correlation_threshold: float = 0.6  # Pearson r above this = "linked"

    # Mean reversion
    base_reversion: float = 0.01
    progressive_reversion: float = 0.03
    output_reversion_scale: float = 0.5  # weaker reversion for outputs

    # Curiosity
    curiosity_interval: float = 30.0
    curiosity_strength: float = 0.04
    curiosity_home_bias: float = 0.6

    # Engagement-based budget bonus
    engagement_bonus_rate: float = 3.33  # budget_max/60 per second engaged


# ---------------------------------------------------------------------------
# Gradient sample
# ---------------------------------------------------------------------------

@dataclass
class GradientSample:
    """One (params, score) observation for regression."""
    timestamp: float
    params: Dict[str, float]
    score: float
    regime: str = 'steady'


# ---------------------------------------------------------------------------
# Smart Auto-Tuner
# ---------------------------------------------------------------------------

class SmartAutoTuner:
    """Score-driven auto-tuner with gradient estimation and context awareness.

    Parameters
    ----------
    meta : MetaParameters
        Shared parameter object (read/write).
    sliders : dict
        GUI slider dict for syncing values.
    database : TrackingDatabase | None
        For reading/writing learnings.
    scorer : EngagementScorer | None
        For computing the fitness signal.
    context_engine : PredictiveContextEngine | None
        For context-dependent budget/interval/reversion.
    config : TunerConfig | None
        Hyperparameters.
    overrides_file : str | None
        Path to autotune_overrides.json for hot-reload.
    """

    def __init__(
        self,
        meta,
        sliders: dict,
        database=None,
        scorer: EngagementScorer = None,
        context_engine: PredictiveContextEngine = None,
        config: TunerConfig = None,
        overrides_file: str = None,
    ):
        self.meta = meta
        self.sliders = sliders
        self.db = database
        self.scorer = scorer or EngagementScorer(database)
        self.context_engine = context_engine
        self.config = config or TunerConfig()

        if overrides_file is None:
            overrides_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'autotune_overrides.json',
            )
        self.overrides_file = overrides_file

        # State
        self._enabled = True
        self._budget = self.config.budget_max
        self._last_update_time = 0.0
        self._last_curiosity_time = 0.0
        self._last_overrides_check = 0.0

        # Gradient estimation window
        self._gradient_samples: deque = deque(maxlen=self.config.gradient_window)
        self._estimated_gradients: Dict[str, float] = {p: 0.0 for p in TUNABLE_PARAMS}

        # Cross-parameter correlation cache
        self._param_links: Dict[Tuple[str, str], float] = {}
        self._last_correlation_time = 0.0

        # Home values (from context engine or overrides)
        self._home_values: Dict[str, float] = {}
        self._safe_floors: Dict[str, float] = dict(DEFAULT_SAFE_FLOORS)
        self._caps: Dict[str, float] = {}

        # Adjustment history (for reports)
        self._adjustment_history: List[dict] = []
        self._max_history = 500

        # Param journey tracking
        self._journey: Dict[str, dict] = {}
        self._init_journey()

        # Load overrides on start
        self._load_overrides()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, behavior_status: dict, now: float) -> Optional[dict]:
        """Main auto-tune cycle.  Call every frame; internally rate-limits.

        Returns an adjustment dict if tuning happened, else None.
        """
        if not self._enabled:
            return None

        # Determine interval from context engine
        interval = self.config.base_interval
        if self.context_engine:
            interval = self.context_engine.get_tune_interval()

        if now - self._last_update_time < interval:
            return None
        dt = now - self._last_update_time
        self._last_update_time = now

        # Hot-reload overrides
        if now - self._last_overrides_check > 30.0:
            self._last_overrides_check = now
            self._load_overrides()

        # Restore budget
        self._restore_budget(dt, behavior_status)

        # Compute engagement score
        snap = self.scorer.compute(behavior_status)
        score = self.scorer.smoothed_score(window=12)

        # Get current param values
        current = self._get_values()

        # Record gradient sample
        regime = 'steady'
        if self.context_engine:
            ctx = self.context_engine.get_context()
            regime = ctx.regime
        self._gradient_samples.append(GradientSample(
            timestamp=now, params=dict(current), score=score, regime=regime,
        ))

        # Estimate gradients
        if len(self._gradient_samples) >= 10:
            self._estimate_gradients()

        # Detect cross-parameter correlations (every 5 minutes)
        if now - self._last_correlation_time > 300:
            self._last_correlation_time = now
            self._detect_correlations()

        # Compute deltas
        deltas = self._compute_deltas(current, score, behavior_status, regime)

        # Apply curiosity perturbation
        if now - self._last_curiosity_time > self.config.curiosity_interval:
            self._last_curiosity_time = now
            deltas = self._apply_curiosity(deltas, current)

        # Apply mean reversion
        deltas = self._apply_mean_reversion(deltas, current)

        # Clamp steps
        deltas = self._clamp_steps(deltas)

        # Budget enforcement
        deltas = self._enforce_budget(deltas)

        # Apply
        if not any(abs(d) >= self.config.min_step_threshold for d in deltas.values()):
            return None

        new_values = {}
        for name in TUNABLE_PARAMS:
            val = current.get(name, 0.5) + deltas.get(name, 0.0)
            val = self._clamp(name, val)
            new_values[name] = val

        self._apply_values(new_values)
        self._update_journey(current, new_values)

        # Record
        adjustment = {
            'timestamp': now,
            'score': round(score, 4),
            'regime': regime,
            'old_values': {k: round(v, 4) for k, v in current.items()},
            'new_values': {k: round(v, 4) for k, v in new_values.items()},
            'deltas': {k: round(v, 5) for k, v in deltas.items() if abs(v) > 0.001},
            'gradients': {k: round(v, 5) for k, v in self._estimated_gradients.items()
                         if abs(v) > 0.001},
            'budget': round(self._budget, 1),
        }
        self._adjustment_history.append(adjustment)
        if len(self._adjustment_history) > self._max_history:
            self._adjustment_history.pop(0)

        return adjustment

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def adjustment_history(self) -> List[dict]:
        return list(self._adjustment_history)

    @property
    def param_journey(self) -> Dict[str, dict]:
        return dict(self._journey)

    @property
    def estimated_gradients(self) -> Dict[str, float]:
        return dict(self._estimated_gradients)

    @property
    def param_links(self) -> Dict[Tuple[str, str], float]:
        return dict(self._param_links)

    # ------------------------------------------------------------------
    # Delta computation (gradient-informed)
    # ------------------------------------------------------------------

    def _compute_deltas(
        self,
        current: Dict[str, float],
        score: float,
        behavior_status: dict,
        regime: str,
    ) -> Dict[str, float]:
        """Compute parameter deltas using gradient estimates + regime heuristics."""
        deltas: Dict[str, float] = {}

        # Primary signal: follow the estimated gradient (gradient ascent)
        for name in TUNABLE_PARAMS:
            grad = self._estimated_gradients.get(name, 0.0)
            # Scale gradient by learning rate
            delta = grad * self.config.gradient_learning_rate
            deltas[name] = delta

        # Regime-specific adjustments (on top of gradient)
        idle_trends = behavior_status.get('idle_trends', {})
        short_activity = idle_trends.get('short_activity', 0.0)
        aggression = behavior_status.get('aggression_level', 0.0)

        if regime == 'dead':
            # During dead hours, minimize churn — only curiosity perturbations
            for name in TUNABLE_PARAMS:
                deltas[name] *= 0.2

        elif regime == 'trickle':
            # Boost attention-seeking params when traffic is light
            for name in ('brightness_global', 'exploration', 'energy'):
                if short_activity < 0.1:
                    deltas[name] += 0.005

        elif regime == 'rush':
            # During rush, personality should rise to meet demand
            for name in PERSONALITY_PARAMS:
                if short_activity > 0.5:
                    deltas[name] += 0.003
            # But calm down outputs (don't overwhelm)
            for name in OUTPUT_PARAMS:
                if short_activity > 0.6:
                    deltas[name] -= 0.002

        elif regime == 'event':
            # Anomalous traffic — widen exploration, be responsive
            deltas['responsiveness'] = deltas.get('responsiveness', 0) + 0.01
            deltas['energy'] = deltas.get('energy', 0) + 0.01
            deltas['exploration'] = deltas.get('exploration', 0) + 0.005

        # Aggression damping (preserve from V5)
        if aggression > 0.8:
            damping = 0.4
        elif aggression > 0.6:
            damping = 0.7
        else:
            damping = 1.0

        if damping < 1.0:
            for name in deltas:
                deltas[name] *= damping

        # Cross-parameter linking: if two params are correlated and one
        # has a strong gradient, share the delta
        for (a, b), corr in self._param_links.items():
            if corr < self.config.correlation_threshold:
                continue
            grad_a = abs(self._estimated_gradients.get(a, 0))
            grad_b = abs(self._estimated_gradients.get(b, 0))
            if grad_a > grad_b * 2:
                # a has stronger signal → pull b in same direction
                deltas[b] = deltas.get(b, 0) + deltas.get(a, 0) * 0.3 * corr
            elif grad_b > grad_a * 2:
                deltas[a] = deltas.get(a, 0) + deltas.get(b, 0) * 0.3 * corr

        return deltas

    # ------------------------------------------------------------------
    # Gradient estimation (simple linear regression)
    # ------------------------------------------------------------------

    def _estimate_gradients(self):
        """Estimate ∂score/∂param for each parameter using OLS on the
        sliding window of (param, score) samples.

        Uses the correlation between each parameter's change and the
        corresponding score change.
        """
        samples = list(self._gradient_samples)
        n = len(samples)
        if n < 10:
            return

        for name in TUNABLE_PARAMS:
            # Compute Δparam and Δscore between consecutive samples
            param_changes = []
            score_changes = []
            for i in range(1, n):
                dp = samples[i].params.get(name, 0) - samples[i-1].params.get(name, 0)
                ds = samples[i].score - samples[i-1].score
                param_changes.append(dp)
                score_changes.append(ds)

            if not param_changes:
                continue

            # Pearson correlation as a gradient proxy
            n_c = len(param_changes)
            mean_dp = sum(param_changes) / n_c
            mean_ds = sum(score_changes) / n_c

            cov = sum((param_changes[i] - mean_dp) * (score_changes[i] - mean_ds)
                       for i in range(n_c)) / n_c
            var_dp = sum((param_changes[i] - mean_dp) ** 2 for i in range(n_c)) / n_c

            if var_dp > 1e-10:
                # Regression coefficient: how much score changes per unit param change
                gradient = cov / var_dp
                # Clip to prevent wild swings
                gradient = max(-1.0, min(1.0, gradient))
                # EMA blend with previous estimate
                old = self._estimated_gradients.get(name, 0.0)
                self._estimated_gradients[name] = old * 0.7 + gradient * 0.3
            # else: param didn't change → keep previous estimate

    # ------------------------------------------------------------------
    # Cross-parameter correlation detection
    # ------------------------------------------------------------------

    def _detect_correlations(self):
        """Detect pairs of parameters that co-move during high-score periods."""
        samples = [s for s in self._gradient_samples
                   if s.score > 0.3]  # only look at non-trivial periods
        if len(samples) < 15:
            return

        self._param_links.clear()
        params = list(TUNABLE_PARAMS)
        n = len(samples)

        # Extract param series
        series: Dict[str, List[float]] = {}
        for name in params:
            series[name] = [s.params.get(name, 0.5) for s in samples]

        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                a, b = params[i], params[j]
                corr = self._pearson(series[a], series[b])
                if abs(corr) > 0.4:  # store if meaningfully correlated
                    self._param_links[(a, b)] = corr

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> float:
        """Pearson correlation coefficient."""
        n = len(x)
        if n < 3:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
        if std_x < 1e-10 or std_y < 1e-10:
            return 0.0
        return cov / (std_x * std_y)

    # ------------------------------------------------------------------
    # Mean reversion
    # ------------------------------------------------------------------

    def _apply_mean_reversion(
        self,
        deltas: Dict[str, float],
        current: Dict[str, float],
    ) -> Dict[str, float]:
        """Pull parameters toward time-of-day home values."""
        home = self._get_home_values()

        # Reversion strength from context engine
        strength_mult = 1.0
        if self.context_engine:
            strength_mult = self.context_engine.get_mean_reversion_strength()

        for name in TUNABLE_PARAMS:
            home_val = home.get(name, current.get(name, 0.5))
            distance = home_val - current.get(name, 0.5)

            base = self.config.base_reversion
            progressive = self.config.progressive_reversion * abs(distance)
            reversion = (base + progressive) * strength_mult

            # Weaker reversion for output params (preserve V5 behaviour)
            if name in OUTPUT_PARAMS:
                reversion *= self.config.output_reversion_scale

            reversion_delta = distance * reversion
            deltas[name] = deltas.get(name, 0) + reversion_delta

        return deltas

    # ------------------------------------------------------------------
    # Curiosity perturbation
    # ------------------------------------------------------------------

    def _apply_curiosity(
        self,
        deltas: Dict[str, float],
        current: Dict[str, float],
    ) -> Dict[str, float]:
        """Random nudge biased toward home values."""
        import random
        home = self._get_home_values()

        for name in TUNABLE_PARAMS:
            nudge = random.uniform(-self.config.curiosity_strength,
                                    self.config.curiosity_strength)

            # Bias toward home
            home_val = home.get(name, 0.5)
            current_val = current.get(name, 0.5)
            if (home_val > current_val and nudge < 0) or \
               (home_val < current_val and nudge > 0):
                # Opposing direction — apply bias
                if random.random() < self.config.curiosity_home_bias:
                    nudge = -nudge

            deltas[name] = deltas.get(name, 0) + nudge

        return deltas

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    def _restore_budget(self, dt: float, behavior_status: dict):
        restore_rate = self.config.budget_max / self.config.budget_restore_seconds
        self._budget += restore_rate * dt

        # Engagement bonus
        active = behavior_status.get('active_count', 0)
        if active > 0:
            self._budget += self.config.engagement_bonus_rate * dt

        # Context-dependent budget scaling
        if self.context_engine:
            mult = self.context_engine.get_budget_multiplier()
            # Scale the max budget, not the current level
            effective_max = self.config.budget_max * mult
        else:
            effective_max = self.config.budget_max

        self._budget = min(self._budget, effective_max)

    def _enforce_budget(self, deltas: Dict[str, float]) -> Dict[str, float]:
        total_cost = sum(abs(d) for d in deltas.values()) * self.config.cost_scale
        if total_cost <= 0:
            return deltas
        if total_cost > self._budget:
            scale = self._budget / total_cost
            deltas = {k: v * scale for k, v in deltas.items()}
            self._budget = 0
        else:
            self._budget -= total_cost
        return deltas

    # ------------------------------------------------------------------
    # Step clamping
    # ------------------------------------------------------------------

    def _clamp_steps(self, deltas: Dict[str, float]) -> Dict[str, float]:
        for name, d in list(deltas.items()):
            if name in PERSONALITY_PARAMS:
                max_step = self.config.max_personality_step
            else:
                max_step = self.config.max_output_step
            deltas[name] = max(-max_step, min(max_step, d))
            if abs(deltas[name]) < self.config.min_step_threshold:
                deltas[name] = 0.0
        return deltas

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def _get_values(self) -> Dict[str, float]:
        return {name: getattr(self.meta, name, 0.5) for name in TUNABLE_PARAMS}

    def _apply_values(self, new_values: Dict[str, float]):
        for name, val in new_values.items():
            setattr(self.meta, name, val)
            # Sync to GUI slider
            if name in self.sliders:
                self.sliders[name].value = val

    def _clamp(self, name: str, value: float) -> float:
        lo, hi = PARAM_RANGES.get(name, (0.0, 1.0))
        floor = self._safe_floors.get(name, lo)
        cap = self._caps.get(name, hi)
        return max(max(lo, floor), min(min(hi, cap), value))

    def _get_home_values(self) -> Dict[str, float]:
        if self.context_engine:
            learned = self.context_engine.get_optimal_home_values()
            if learned:
                return learned
        return self._home_values

    # ------------------------------------------------------------------
    # Journey tracking
    # ------------------------------------------------------------------

    def _init_journey(self):
        current = self._get_values()
        for name in TUNABLE_PARAMS:
            val = current.get(name, 0.5)
            self._journey[name] = {
                'start': val,
                'current': val,
                'min': val,
                'max': val,
                'total_movement': 0.0,
            }

    def _update_journey(self, old_values: Dict, new_values: Dict):
        for name in TUNABLE_PARAMS:
            old_v = old_values.get(name, 0.5)
            new_v = new_values.get(name, 0.5)
            j = self._journey.get(name, {})
            j['current'] = new_v
            j['min'] = min(j.get('min', new_v), new_v)
            j['max'] = max(j.get('max', new_v), new_v)
            j['total_movement'] = j.get('total_movement', 0) + abs(new_v - old_v)
            self._journey[name] = j

    # ------------------------------------------------------------------
    # Overrides hot-reload
    # ------------------------------------------------------------------

    def _load_overrides(self):
        if not os.path.exists(self.overrides_file):
            return
        try:
            with open(self.overrides_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        # Home values
        if 'home_values' in data:
            for k, v in data['home_values'].items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self._home_values[k] = v

        # Safe floors
        if 'safe_floors' in data:
            for k, v in data['safe_floors'].items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self._safe_floors[k] = v

        # Caps
        if 'caps' in data:
            for k, v in data['caps'].items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self._caps[k] = v

        # Budget overrides
        if 'budget' in data:
            budget = data['budget']
            if 'max' in budget:
                self.config.budget_max = float(budget['max'])
            if 'restore_seconds' in budget:
                self.config.budget_restore_seconds = float(budget['restore_seconds'])
            if 'cost_scale' in budget:
                self.config.cost_scale = float(budget['cost_scale'])

        # Curiosity overrides
        if 'curiosity' in data:
            cur = data['curiosity']
            if 'interval' in cur:
                self.config.curiosity_interval = float(cur['interval'])
            if 'strength' in cur:
                self.config.curiosity_strength = float(cur['strength'])

        logger.info(f"🔄 Loaded overrides: {len(self._home_values)} home values, "
                    f"{len(self._safe_floors)} floors, {len(self._caps)} caps")

    # ------------------------------------------------------------------
    # Daily learning (interface for report pipeline)
    # ------------------------------------------------------------------

    def compute_daily_learnings(self, report: dict) -> Dict:
        """Compute what to learn from today's ended day.

        Called by the daily report scheduler after midnight.
        """
        score = self.scorer.compute_from_report(report)
        hourly_scores = self.scorer.compute_hourly_from_report(report)

        return {
            'date': report.get('date', ''),
            'engagement_score': round(score, 4),
            'hourly_scores': {k: round(v, 4) for k, v in hourly_scores.items()},
            'param_journey': self.param_journey,
            'estimated_gradients': {k: round(v, 5) for k, v in self._estimated_gradients.items()},
            'param_links': {f"{a}↔{b}": round(c, 3)
                           for (a, b), c in self._param_links.items()},
            'total_adjustments': len(self._adjustment_history),
        }
