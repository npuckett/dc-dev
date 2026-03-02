#!/usr/bin/env python3
"""
V6 Mode Intelligence
====================

Enhances V5's binary mode transitions with:

1. **Predictive pre-transition**: Begin interpolating toward the next
   mode *before* the trigger threshold is fully met (e.g., start blending
   ENGAGED at 30 % when a high-scoring candidate is approaching).

2. **Adaptive stickiness**: A mode's persistence window scales with
   dwell quality. Bond-phase engagement holds for 10 s; a brief greet
   only 3 s.  Prevents thrashing while still being responsive.

3. **CROWD sub-modes**: Distinguishes CROWD_SOCIAL (clustered group
   interacting together) from CROWD_SCATTERED (many unrelated movers).
   Each gets a different lighting strategy and falloff shape.

4. **Mode momentum**: Repeated visits to ENGAGED build *session
   familiarity* — each subsequent ENGAGED period ramps faster and
   reaches higher intensity.

Compatibility
~~~~~~~~~~~~~
``ModeIntelligence`` wraps V5's existing ``BehaviorSystem`` without
modifying it.  It reads behaviour state, applies overlays, and exposes
a ``ModeOverlay`` that the integration layer feeds to the renderer.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class V6Mode(Enum):
    """Extended mode set, superset of V5."""
    IDLE = 'idle'
    AWARE = 'aware'
    ENGAGED = 'engaged'
    CROWD = 'crowd'
    CROWD_SOCIAL = 'crowd_social'
    CROWD_SCATTERED = 'crowd_scattered'
    FLOW = 'flow'


@dataclass
class ModeOverlay:
    """Computed overlay for the current frame.

    The integration layer reads these fields and modulates the
    renderer/behaviour accordingly.
    """
    # The *effective* mode after intelligence processing
    effective_mode: V6Mode = V6Mode.IDLE

    # Blend factor toward next predicted mode (0 = current, 1 = next)
    pre_transition_blend: float = 0.0
    predicted_next_mode: Optional[V6Mode] = None

    # Stickiness: how many more seconds the current mode is guaranteed
    stickiness_remaining: float = 0.0

    # Session familiarity multiplier (1.0 = first visit, up to 2.0)
    familiarity_mult: float = 1.0

    # CROWD sub-mode clustering score (0 = scattered, 1 = tight cluster)
    crowd_clustering: float = 0.0

    # Intensity multiplier (from momentum / repeated engagement)
    intensity_mult: float = 1.0


# ---------------------------------------------------------------------------
# Candidate scoring (for pre-transition)
# ---------------------------------------------------------------------------

@dataclass
class CandidateScore:
    """Evaluated potential of a tracked person to trigger ENGAGED."""
    person_id: int = 0
    score: float = 0.0          # 0–1 composite
    distance: float = 999.0     # distance in normalized units
    dwell_seconds: float = 0.0  # how long they've been present
    speed: float = 0.0          # movement speed
    approaching: bool = False   # moving closer?
    facing: bool = False        # rough facing estimate


# ---------------------------------------------------------------------------
# Mode Intelligence
# ---------------------------------------------------------------------------

class ModeIntelligence:
    """Processes V5 behaviour state and outputs a :class:`ModeOverlay`.

    Parameters
    ----------
    config : dict | None
        Override defaults (stickiness durations, thresholds, etc.)
    """

    def __init__(self, config: dict = None):
        config = config or {}

        # -- Stickiness durations (seconds) by dwell phase --
        self.stickiness = {
            'notice': config.get('stickiness_notice', 2.0),
            'greet':  config.get('stickiness_greet', 3.0),
            'engage': config.get('stickiness_engage', 6.0),
            'bond':   config.get('stickiness_bond', 10.0),
        }
        # Minimum stickiness for any mode
        self.min_stickiness: float = config.get('min_stickiness', 1.5)

        # -- Pre-transition thresholds --
        self.pre_transition_score: float = config.get('pre_transition_score', 0.55)
        self.pre_transition_max_blend: float = config.get('pre_transition_max_blend', 0.40)

        # -- Crowd clustering --
        self.cluster_distance_threshold: float = config.get('cluster_distance', 0.15)
        self.crowd_min_people: int = config.get('crowd_min_people', 3)

        # -- Session familiarity --
        self.familiarity_increment: float = config.get('familiarity_increment', 0.15)
        self.familiarity_max: float = config.get('familiarity_max', 2.0)
        self.familiarity_decay: float = config.get('familiarity_decay', 0.97)  # per minute

        # -- Momentum --
        self.momentum_growth: float = config.get('momentum_growth', 0.05)      # per engagement
        self.momentum_max: float = config.get('momentum_max', 1.6)
        self.momentum_decay: float = config.get('momentum_decay', 0.995)       # per second idle

        # State
        self._current_mode = V6Mode.IDLE
        self._stickiness_until: float = 0.0
        self._familiarity: float = 1.0
        self._momentum: float = 1.0
        self._last_mode_time: float = 0.0
        self._engagement_count: int = 0          # session engagement events
        self._last_familiarity_decay: float = 0.0

        # Candidate history (for pre-transition)
        self._candidate_scores: Dict[int, List[float]] = {}  # person_id → recent scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        behavior_status: dict,
        tracked_people: List[dict],
        now: float,
    ) -> ModeOverlay:
        """Compute a :class:`ModeOverlay` for this frame.

        Parameters
        ----------
        behavior_status : dict
            The status dict from V5's BehaviorSystem (contains 'mode',
            'dwell_phase', 'active_count', 'engaged_count', etc.)
        tracked_people : list[dict]
            List of tracked people from the camera tracker.  Each has
            keys like 'id', 'pos', 'speed', 'dwell_time'.
        now : float
            Current time.time() value.
        """
        overlay = ModeOverlay()
        v5_mode_str = behavior_status.get('mode', 'idle')
        dwell_phase = behavior_status.get('dwell_phase', 'notice')
        active_count = behavior_status.get('active_count', 0)

        # Decay familiarity once per minute
        if now - self._last_familiarity_decay > 60.0:
            self._last_familiarity_decay = now
            self._familiarity = max(
                1.0,
                self._familiarity * self.familiarity_decay,
            )

        # Decay momentum when idle
        if v5_mode_str == 'idle':
            dt = max(0, now - self._last_mode_time) if self._last_mode_time else 0
            if dt > 0 and dt < 60:  # sanity bound
                self._momentum = max(1.0, self._momentum * (self.momentum_decay ** dt))

        self._last_mode_time = now

        # Map V5 mode to V6
        v6_mode = self._map_v5_mode(v5_mode_str, tracked_people, overlay)

        # Stickiness check
        if now < self._stickiness_until:
            # Locked into current mode
            overlay.effective_mode = self._current_mode
            overlay.stickiness_remaining = self._stickiness_until - now
        else:
            # Mode transition allowed
            if v6_mode != self._current_mode:
                self._on_mode_change(self._current_mode, v6_mode, dwell_phase, now)
            overlay.effective_mode = v6_mode
            self._current_mode = v6_mode

        # Pre-transition scoring
        if self._current_mode in (V6Mode.IDLE, V6Mode.AWARE, V6Mode.FLOW):
            candidates = self._score_candidates(tracked_people)
            best = max(candidates, key=lambda c: c.score, default=None)
            if best and best.score > self.pre_transition_score:
                blend = min(
                    self.pre_transition_max_blend,
                    (best.score - self.pre_transition_score) / 0.3,
                )
                overlay.pre_transition_blend = blend
                overlay.predicted_next_mode = V6Mode.ENGAGED

        # Session familiarity & momentum
        overlay.familiarity_mult = self._familiarity
        overlay.intensity_mult = self._momentum

        return overlay

    def reset_session(self):
        """Reset session state (e.g., at midnight)."""
        self._familiarity = 1.0
        self._momentum = 1.0
        self._engagement_count = 0
        self._candidate_scores.clear()

    # ------------------------------------------------------------------
    # Internal: mode mapping
    # ------------------------------------------------------------------

    def _map_v5_mode(
        self,
        v5_mode: str,
        tracked: List[dict],
        overlay: ModeOverlay,
    ) -> V6Mode:
        """Map V5 mode string + context → V6Mode (with CROWD sub-modes)."""
        mode_map = {
            'idle': V6Mode.IDLE,
            'aware': V6Mode.AWARE,
            'engaged': V6Mode.ENGAGED,
            'flow': V6Mode.FLOW,
        }

        if v5_mode == 'crowd':
            clustering = self._compute_clustering(tracked)
            overlay.crowd_clustering = clustering
            if clustering > 0.6:
                return V6Mode.CROWD_SOCIAL
            else:
                return V6Mode.CROWD_SCATTERED

        return mode_map.get(v5_mode, V6Mode.IDLE)

    # ------------------------------------------------------------------
    # Internal: crowd clustering
    # ------------------------------------------------------------------

    def _compute_clustering(self, tracked: List[dict]) -> float:
        """Compute a 0–1 clustering score for the crowd.

        Simple approach: median pairwise distance of all tracked people.
        Low distance = tight cluster = SOCIAL.
        """
        positions = []
        for p in tracked:
            pos = p.get('pos')
            if pos and len(pos) >= 2:
                positions.append(pos[:2])

        if len(positions) < self.crowd_min_people:
            return 0.0

        # Pairwise distances
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dists.append(math.sqrt(dx * dx + dy * dy))

        if not dists:
            return 0.0

        dists.sort()
        median_dist = dists[len(dists) // 2]

        # Map to 0–1 (tighter cluster = higher score)
        # threshold = 0.15 → score=1, threshold × 5 → score=0
        score = max(0.0, 1.0 - (median_dist / (self.cluster_distance_threshold * 5)))
        return score

    # ------------------------------------------------------------------
    # Internal: candidate scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        tracked: List[dict],
    ) -> List[CandidateScore]:
        """Evaluate each tracked person's engagement potential."""
        candidates = []
        for person in tracked:
            pid = person.get('id', 0)
            dist = person.get('distance', 999)
            dwell = person.get('dwell_time', 0)
            speed = person.get('speed', 0)
            approaching = person.get('approaching', False)

            # Composite score
            score = 0.0

            # Distance factor (closer = higher)
            if dist < 0.5:
                score += 0.3 * (1.0 - dist / 0.5)

            # Dwell factor (longer = higher)
            dwell_score = min(1.0, dwell / 15.0)  # saturates at 15s
            score += 0.25 * dwell_score

            # Approaching factor
            if approaching:
                score += 0.2

            # Low speed = lingering = interested
            if speed < 0.05:
                score += 0.15
            elif speed < 0.1:
                score += 0.10

            # Familiarity bonus
            score_hist = self._candidate_scores.get(pid, [])
            if len(score_hist) > 3:
                score += 0.1  # returning candidate

            # Track history
            score_hist.append(score)
            if len(score_hist) > 20:
                score_hist = score_hist[-20:]
            self._candidate_scores[pid] = score_hist

            candidates.append(CandidateScore(
                person_id=pid,
                score=min(1.0, score),
                distance=dist,
                dwell_seconds=dwell,
                speed=speed,
                approaching=approaching,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Internal: mode transitions
    # ------------------------------------------------------------------

    def _on_mode_change(
        self,
        old_mode: V6Mode,
        new_mode: V6Mode,
        dwell_phase: str,
        now: float,
    ):
        """Handle a confirmed mode transition."""
        # Set stickiness
        if dwell_phase in self.stickiness:
            lock = self.stickiness[dwell_phase]
        else:
            lock = self.min_stickiness
        self._stickiness_until = now + lock

        # Track engagement events for momentum/familiarity
        if new_mode == V6Mode.ENGAGED:
            self._engagement_count += 1
            self._familiarity = min(
                self.familiarity_max,
                self._familiarity + self.familiarity_increment,
            )
            self._momentum = min(
                self.momentum_max,
                self._momentum + self.momentum_growth,
            )

        logger.debug(
            f"Mode: {old_mode.value} → {new_mode.value} "
            f"(stickiness={lock:.1f}s, fam={self._familiarity:.2f}, "
            f"momentum={self._momentum:.2f})"
        )
