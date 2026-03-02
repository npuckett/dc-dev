#!/usr/bin/env python3
"""
V6 Falloff Strategies — Expressive Use of Anisotropic Falloff
==============================================================

V5 introduced per-axis falloff scaling (X/Y/Z) and Y-axis rotation with
spring inertia, but only uses them for 2 gestures (SWEEP, FOCUS) plus
ambient oscillation.  The Y-axis scale is never driven.  Rotation is
limited to ±0.4 rad.

V6 treats the falloff *shape* as a first-class expressive channel:

**Per-mode default shapes** (was always [1,1,1]):
- IDLE: wider/flatter ellipsoid → ambient scanning feel
- ENGAGED: taller/deeper → spotlight focus on person
- CROWD: wide + deep → umbrella covering the group
- FLOW: stretched along flow direction via rotation

**Proximity-reactive shaping**:
- As someone approaches, the Z-axis contracts (tighter beam)
  and brightness concentrates — the light "focuses attention"
- Y-axis finally used: taller falloff when person is close
  (light envelops them vertically)

**New gesture profiles** that exploit the full shape space:
- REACH: deep Z stretch + rotation toward target (for bandit's FALLOFF_RESHAPE)
- EMBRACE: wide X + tall Y (wrapping effect for group moments)
- BEACON: rhythmic Z pulsation (lighthouse effect when idle)
- TWIRL: rotation sweep with contracting scale

**Flow-aligned rotation**: In FLOW mode, the ellipsoid continuously
rotates to point *into* the flow direction, creating a directional
light shape that pedestrians walk through.

All outputs are ``FalloffIntent`` objects consumed by the ``ModifierResolver``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Falloff shape representation
# ---------------------------------------------------------------------------

@dataclass
class FalloffShape:
    """Describes a desired ellipsoidal falloff configuration."""
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    rotation: float = 0.0   # radians around Y axis
    radius_mult: float = 1.0  # multiplier on base falloff_radius

    def lerp(self, other: 'FalloffShape', t: float) -> 'FalloffShape':
        """Linear interpolation toward ``other`` by factor t ∈ [0, 1]."""
        t = max(0.0, min(1.0, t))
        return FalloffShape(
            scale_x=self.scale_x + (other.scale_x - self.scale_x) * t,
            scale_y=self.scale_y + (other.scale_y - self.scale_y) * t,
            scale_z=self.scale_z + (other.scale_z - self.scale_z) * t,
            rotation=self.rotation + (other.rotation - self.rotation) * t,
            radius_mult=self.radius_mult + (other.radius_mult - self.radius_mult) * t,
        )

    def __mul__(self, other: 'FalloffShape') -> 'FalloffShape':
        """Component-wise multiplication (for layering modifiers)."""
        return FalloffShape(
            scale_x=self.scale_x * other.scale_x,
            scale_y=self.scale_y * other.scale_y,
            scale_z=self.scale_z * other.scale_z,
            rotation=self.rotation + other.rotation,
            radius_mult=self.radius_mult * other.radius_mult,
        )


NEUTRAL_SHAPE = FalloffShape()


# ---------------------------------------------------------------------------
# Per-mode default shapes (replaces the implicit [1,1,1] in V5)
# ---------------------------------------------------------------------------

MODE_FALLOFF_DEFAULTS: Dict[str, FalloffShape] = {
    'idle': FalloffShape(
        scale_x=1.3,   # wider horizontal spread — ambient scanning
        scale_y=0.9,   # slightly squashed vertically
        scale_z=1.1,   # mild depth
        rotation=0.0,
        radius_mult=1.0,
    ),
    # idle_beacon: used when idle for extended periods (>60s)
    # Wider, taller, higher-radius to maximize visual presence on sidewalk
    'idle_beacon': FalloffShape(
        scale_x=1.6,   # extra wide — cast light into passive zone
        scale_y=1.2,   # taller — more vertical presence
        scale_z=1.4,   # deeper — reach further into the sidewalk
        rotation=0.0,
        radius_mult=1.15,  # larger reach radius
    ),
    'engaged': FalloffShape(
        scale_x=0.8,   # narrower — focusing on the person
        scale_y=1.3,   # taller — enveloping vertically
        scale_z=0.9,   # tighter depth
        rotation=0.0,
        radius_mult=0.9,
    ),
    'crowd': FalloffShape(
        scale_x=1.5,   # wide umbrella covering the group
        scale_y=1.1,   # slightly tall
        scale_z=1.3,   # deep to reach everyone
        rotation=0.0,
        radius_mult=1.1,
    ),
    'flow': FalloffShape(
        scale_x=1.2,
        scale_y=0.9,
        scale_z=1.4,   # long reach into the flow path
        rotation=0.0,  # dynamically set toward flow direction
        radius_mult=1.0,
    ),
}


# ---------------------------------------------------------------------------
# V6 gesture falloff profiles
# ---------------------------------------------------------------------------

class V6Gesture(str, Enum):
    """New V6 gestures that focus on shape expression."""
    REACH    = 'reach'     # Deep Z stretch toward a target
    EMBRACE  = 'embrace'   # Wide X + tall Y for group moments
    BEACON   = 'beacon'    # Rhythmic Z pulsation (lighthouse)
    TWIRL    = 'twirl'     # Rotation sweep with contracting scale
    # V5 gestures carried forward with enhanced profiles
    SWEEP    = 'sweep'
    FOCUS    = 'focus'


@dataclass
class GestureProfile:
    """Defines a gesture's falloff shape animation."""
    target_shape: FalloffShape
    duration: float = 2.0
    # Animation curve: 'bell' (sine peak in middle), 'attack' (fast in, slow out),
    # 'sustain' (hold near peak), 'pulse' (rapid on/off)
    curve: str = 'bell'
    # Position offsets (cm) applied during gesture
    position_offset_x: float = 0.0
    position_offset_y: float = 0.0
    position_offset_z: float = 0.0
    brightness_boost: float = 0.0


V6_GESTURE_PROFILES: Dict[V6Gesture, GestureProfile] = {
    V6Gesture.REACH: GestureProfile(
        target_shape=FalloffShape(
            scale_x=0.8,    # narrow sides
            scale_y=1.0,
            scale_z=2.5,    # deep Z reach
            rotation=0.0,   # dynamically set toward target
            radius_mult=1.2,
        ),
        duration=2.5,
        curve='attack',     # fast extension, slow retraction
        brightness_boost=8,
    ),
    V6Gesture.EMBRACE: GestureProfile(
        target_shape=FalloffShape(
            scale_x=2.0,    # very wide
            scale_y=1.6,    # tall (first real Y-axis use!)
            scale_z=1.2,
            rotation=0.0,
            radius_mult=1.1,
        ),
        duration=3.0,
        curve='sustain',
        brightness_boost=12,
    ),
    V6Gesture.BEACON: GestureProfile(
        target_shape=FalloffShape(
            scale_x=0.6,    # narrow
            scale_y=1.0,
            scale_z=2.0,    # deep pulse
            rotation=0.0,   # rotates during animation
            radius_mult=0.9,
        ),
        duration=4.0,
        curve='pulse',      # rhythmic on/off
        brightness_boost=5,
    ),
    V6Gesture.TWIRL: GestureProfile(
        target_shape=FalloffShape(
            scale_x=0.7,
            scale_y=1.2,
            scale_z=1.8,
            rotation=math.pi * 0.75,  # 135° sweep
            radius_mult=0.85,
        ),
        duration=3.0,
        curve='bell',
        brightness_boost=6,
        position_offset_y=10,  # slight lift during twirl
    ),
    # Enhanced V5 gestures
    V6Gesture.SWEEP: GestureProfile(
        target_shape=FalloffShape(
            scale_x=2.8,
            scale_y=0.9,
            scale_z=0.8,
            rotation=0.0,
            radius_mult=1.0,
        ),
        duration=2.0,
        curve='bell',
        position_offset_x=30,  # wider X scan
    ),
    V6Gesture.FOCUS: GestureProfile(
        target_shape=FalloffShape(
            scale_x=0.4,    # very tight
            scale_y=0.5,    # V6: now uses Y-axis contraction too
            scale_z=0.4,
            rotation=0.0,
            radius_mult=0.7,
        ),
        duration=1.5,
        curve='attack',
        brightness_boost=25,
    ),
}


# ---------------------------------------------------------------------------
# Animation curves
# ---------------------------------------------------------------------------

def eval_curve(curve: str, t: float) -> float:
    """Evaluate animation curve at normalised time t ∈ [0, 1].

    Returns a value in [0, 1] representing the intensity at time t.
    """
    t = max(0.0, min(1.0, t))
    if curve == 'bell':
        return math.sin(t * math.pi)
    elif curve == 'attack':
        # Fast attack (sqrt), slow release (squared)
        if t < 0.3:
            return math.sqrt(t / 0.3)
        else:
            return 1.0 - ((t - 0.3) / 0.7) ** 2
    elif curve == 'sustain':
        # Quick ramp to 80% then hold
        if t < 0.15:
            return t / 0.15 * 0.8
        elif t < 0.75:
            return 0.8 + 0.2 * math.sin((t - 0.15) / 0.6 * math.pi)
        else:
            return (1.0 - t) / 0.25
    elif curve == 'pulse':
        # Two pulses within the duration
        return abs(math.sin(t * math.pi * 2))
    return t  # linear fallback


# ---------------------------------------------------------------------------
# Falloff Strategy Manager
# ---------------------------------------------------------------------------

@dataclass
class ActiveGestureState:
    """State for a currently-playing gesture animation."""
    gesture: V6Gesture
    profile: GestureProfile
    start_time: float
    dynamic_rotation: float = 0.0   # overridden rotation (toward target)


class FalloffStrategyManager:
    """Computes the desired falloff shape each frame based on mode, proximity,
    flow, and active gestures.

    This does NOT directly set falloff values on the PointLight — it emits
    ``FalloffShape`` objects that the ``ModifierResolver`` merges with other
    system intents.
    """

    def __init__(self):
        self._active_gesture: Optional[ActiveGestureState] = None
        self._last_gesture_end: float = 0.0
        self._gesture_cooldown: float = 3.0  # seconds between gestures

        # Proximity tracking
        self._nearest_person_z: float = 300.0  # cm (far)
        self._nearest_person_x: float = -150.0

        # Flow state
        self._flow_direction: float = 0.0   # -1 to +1
        self._flow_strength: float = 0.0    # 0 to 1

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def compute_shape(
        self,
        mode: str,
        dt: float,
        nearest_person_z: float = 300.0,
        nearest_person_x: float = -150.0,
        flow_direction: float = 0.0,
        flow_strength: float = 0.0,
        active_count: int = 0,
        passive_count: int = 0,
    ) -> FalloffShape:
        """Compute the desired falloff shape for this frame.

        Returns a ``FalloffShape`` combining mode default, proximity
        response, flow alignment, and active gesture.
        """
        now = time.time()
        self._nearest_person_z = nearest_person_z
        self._nearest_person_x = nearest_person_x
        self._flow_direction = flow_direction
        self._flow_strength = flow_strength

        # 1. Mode default
        mode_shape = MODE_FALLOFF_DEFAULTS.get(mode, NEUTRAL_SHAPE)

        # 2. Proximity response (only when someone is in active zone)
        prox_shape = self._compute_proximity_shape(mode, nearest_person_z, active_count)

        # 3. Flow alignment (IDLE and FLOW modes)
        flow_shape = self._compute_flow_shape(mode, flow_direction, flow_strength)

        # 4. Active gesture overlay
        gesture_shape = self._compute_gesture_shape(now)

        # Combine: mode × proximity × flow × gesture
        combined = mode_shape * prox_shape * flow_shape * gesture_shape

        # Clamp to sane ranges
        combined.scale_x = max(0.3, min(3.0, combined.scale_x))
        combined.scale_y = max(0.3, min(2.5, combined.scale_y))
        combined.scale_z = max(0.3, min(3.5, combined.scale_z))
        combined.rotation = max(-math.pi, min(math.pi, combined.rotation))
        combined.radius_mult = max(0.5, min(1.5, combined.radius_mult))

        return combined

    # ------------------------------------------------------------------
    # Gesture control
    # ------------------------------------------------------------------

    def start_gesture(
        self,
        gesture: V6Gesture,
        target_x: float = None,
        target_z: float = None,
        light_x: float = -150.0,
        light_z: float = 0.0,
    ) -> bool:
        """Start a V6 gesture animation.  Returns False if on cooldown."""
        now = time.time()
        if self._active_gesture is not None:
            return False
        if now - self._last_gesture_end < self._gesture_cooldown:
            return False

        profile = V6_GESTURE_PROFILES.get(gesture)
        if profile is None:
            return False

        # Compute dynamic rotation toward target
        dynamic_rot = 0.0
        if target_x is not None and target_z is not None:
            dx = target_x - light_x
            dz = target_z - light_z
            if abs(dx) > 1 or abs(dz) > 1:
                dynamic_rot = math.atan2(dx, dz)

        self._active_gesture = ActiveGestureState(
            gesture=gesture,
            profile=profile,
            start_time=now,
            dynamic_rotation=dynamic_rot,
        )
        return True

    @property
    def gesture_active(self) -> bool:
        return self._active_gesture is not None

    @property
    def active_gesture_name(self) -> Optional[str]:
        if self._active_gesture:
            return self._active_gesture.gesture.value
        return None

    # ------------------------------------------------------------------
    # Internal shape computations
    # ------------------------------------------------------------------

    def _compute_proximity_shape(
        self,
        mode: str,
        z_distance: float,
        active_count: int,
    ) -> FalloffShape:
        """Proximity-reactive shaping: closer person → tighter, taller beam."""
        if mode not in ('engaged', 'crowd') or active_count == 0:
            return NEUTRAL_SHAPE

        # z_distance: ~78 (right at zone edge) to ~283 (back of active zone)
        # Normalise: 0 (closest) to 1 (farthest)
        z_norm = max(0.0, min(1.0, (z_distance - 78) / 205))
        proximity = 1.0 - z_norm  # 1.0 when closest

        # Close: narrow X, deep Y (enveloping), tight Z
        return FalloffShape(
            scale_x=1.0 - proximity * 0.25,    # 1.0 → 0.75
            scale_y=1.0 + proximity * 0.4,     # 1.0 → 1.4 (Y-axis finally used!)
            scale_z=1.0 - proximity * 0.2,     # 1.0 → 0.8
            rotation=0.0,
            radius_mult=1.0 - proximity * 0.15,  # tighter radius when close
        )

    def _compute_flow_shape(
        self,
        mode: str,
        flow_direction: float,
        flow_strength: float,
    ) -> FalloffShape:
        """Flow-aligned rotation: ellipsoid points into the flow."""
        if mode not in ('idle', 'flow') or flow_strength < 0.15:
            return NEUTRAL_SHAPE

        # Rotation: point the long Z-axis toward incoming traffic
        # flow_direction: +1 (LTR) means traffic comes from left → rotate left
        rotation = -flow_direction * flow_strength * 0.5  # max ±0.5 rad (~28°)

        # Stretch Z proportional to flow strength (more flow = longer reach)
        z_stretch = 1.0 + flow_strength * 0.4  # 1.0 → 1.4

        return FalloffShape(
            scale_x=1.0,
            scale_y=1.0,
            scale_z=z_stretch,
            rotation=rotation,
            radius_mult=1.0,
        )

    def _compute_gesture_shape(self, now: float) -> FalloffShape:
        """Evaluate the active gesture animation."""
        if self._active_gesture is None:
            return NEUTRAL_SHAPE

        gs = self._active_gesture
        elapsed = now - gs.start_time
        if elapsed >= gs.profile.duration:
            # Gesture finished
            self._active_gesture = None
            self._last_gesture_end = now
            return NEUTRAL_SHAPE

        t = elapsed / gs.profile.duration
        intensity = eval_curve(gs.profile.curve, t)

        target = gs.profile.target_shape
        shape = NEUTRAL_SHAPE.lerp(target, intensity)

        # Apply dynamic rotation for directional gestures (REACH, BEACON)
        if gs.gesture in (V6Gesture.REACH, V6Gesture.BEACON):
            shape.rotation = gs.dynamic_rotation * intensity

        # BEACON: add slow rotation sweep during pulse
        if gs.gesture == V6Gesture.BEACON:
            shape.rotation += math.sin(t * math.pi * 4) * 0.3 * intensity

        # TWIRL: full rotation is the point
        if gs.gesture == V6Gesture.TWIRL:
            shape.rotation = target.rotation * t  # linear rotation sweep

        return shape

    # ------------------------------------------------------------------
    # Gesture suggestion (for the behavior system to pick from)
    # ------------------------------------------------------------------

    def suggest_gesture(
        self,
        mode: str,
        dwell_phase: str = 'notice',
        active_count: int = 0,
        energy: float = 0.5,
        sociability: float = 0.5,
    ) -> Optional[V6Gesture]:
        """Suggest an appropriate V6 gesture based on current state.

        Returns None if no gesture is appropriate.  The behavior system
        can use this alongside V5 gesture selection.
        """
        if self.gesture_active:
            return None

        if mode == 'idle':
            # BEACON when alone and trying to attract
            # Lowered energy threshold from 0.4 to 0.3 — on this passive-heavy
            # installation, energy often sits near the floor, but we still
            # want BEACON to fire to attract attention
            if active_count == 0 and energy > 0.3:
                return V6Gesture.BEACON
            return None

        if mode == 'engaged':
            if dwell_phase == 'notice':
                return V6Gesture.REACH  # extend toward the person
            elif dwell_phase == 'bond' and sociability > 0.5:
                return V6Gesture.EMBRACE  # deep connection gesture
            elif dwell_phase == 'engage' and energy > 0.6:
                return V6Gesture.TWIRL
            return None

        if mode == 'crowd':
            if active_count >= 3 and sociability > 0.5:
                return V6Gesture.EMBRACE  # wrap the group
            elif energy > 0.6:
                return V6Gesture.SWEEP
            return None

        return None
