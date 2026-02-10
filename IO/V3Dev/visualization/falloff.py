"""
Falloff Calculations
====================
Distance-based brightness falloff and gradient functions.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple


# =============================================================================
# FALLOFF TYPES
# =============================================================================

class FalloffType(Enum):
    """Types of distance falloff curves."""
    LINEAR = "linear"           # 1 - (d / radius)
    QUADRATIC = "quadratic"     # 1 - (d / radius)^2
    SMOOTH = "smooth"           # smoothstep curve
    INVERSE = "inverse"         # 1 / (1 + d/radius)
    GAUSSIAN = "gaussian"       # exp(-(d/radius)^2)
    NONE = "none"               # No falloff (constant)


# =============================================================================
# FALLOFF FUNCTIONS
# =============================================================================

def linear_falloff(distance: float, radius: float) -> float:
    """
    Linear falloff: 1 at distance=0, 0 at distance=radius.
    
    Args:
        distance: Distance from light source
        radius: Falloff radius (where brightness reaches 0)
        
    Returns:
        Brightness factor (0.0 to 1.0)
    """
    if radius <= 0:
        return 1.0
    return max(0.0, 1.0 - distance / radius)


def quadratic_falloff(distance: float, radius: float) -> float:
    """
    Quadratic (inverse-square-ish) falloff.
    Drops off faster near the edge.
    """
    if radius <= 0:
        return 1.0
    t = min(1.0, distance / radius)
    return max(0.0, 1.0 - t * t)


def smooth_falloff(distance: float, radius: float) -> float:
    """
    Smooth (hermite) falloff using smoothstep.
    Gentle at center and edge, steeper in middle.
    """
    if radius <= 0:
        return 1.0
    t = min(1.0, distance / radius)
    # Smoothstep: 3t² - 2t³
    return max(0.0, 1.0 - (3 * t * t - 2 * t * t * t))


def inverse_falloff(distance: float, radius: float) -> float:
    """
    Inverse falloff: 1/(1 + d/r).
    Never reaches 0, but asymptotes.
    """
    if radius <= 0:
        return 1.0
    return 1.0 / (1.0 + distance / radius)


def gaussian_falloff(distance: float, radius: float) -> float:
    """
    Gaussian falloff: exp(-(d/r)²).
    Soft, natural-looking falloff.
    """
    if radius <= 0:
        return 1.0
    t = distance / radius
    return math.exp(-t * t)


def no_falloff(distance: float, radius: float) -> float:
    """No falloff - constant brightness."""
    return 1.0


# Lookup table for falloff functions
FALLOFF_FUNCTIONS: Dict[FalloffType, Callable[[float, float], float]] = {
    FalloffType.LINEAR: linear_falloff,
    FalloffType.QUADRATIC: quadratic_falloff,
    FalloffType.SMOOTH: smooth_falloff,
    FalloffType.INVERSE: inverse_falloff,
    FalloffType.GAUSSIAN: gaussian_falloff,
    FalloffType.NONE: no_falloff,
}


# =============================================================================
# FALLOFF CALCULATOR
# =============================================================================

@dataclass
class FalloffParams:
    """Parameters for falloff calculation."""
    radius: float = 80.0           # Base falloff radius in cm
    falloff_type: FalloffType = FalloffType.SMOOTH
    min_brightness: float = 0.0    # Minimum brightness (ambient)
    max_brightness: float = 1.0    # Maximum brightness
    
    # Direction-aware falloff (for panel normals)
    use_normals: bool = False
    normal_influence: float = 0.3  # How much facing direction affects brightness


class FalloffCalculator:
    """
    Calculates brightness falloff based on distance and optional direction.
    """
    
    def __init__(self, params: Optional[FalloffParams] = None):
        self.params = params or FalloffParams()
        self._falloff_fn = FALLOFF_FUNCTIONS[self.params.falloff_type]
    
    def set_radius(self, radius: float):
        """Update falloff radius."""
        self.params.radius = radius
    
    def set_type(self, falloff_type: FalloffType):
        """Change falloff type."""
        self.params.falloff_type = falloff_type
        self._falloff_fn = FALLOFF_FUNCTIONS[falloff_type]
    
    def calculate(self, distance: float) -> float:
        """
        Calculate brightness for a given distance.
        
        Args:
            distance: Distance from light source in cm
            
        Returns:
            Brightness (0.0 to 1.0)
        """
        raw = self._falloff_fn(distance, self.params.radius)
        # Scale to min/max range
        return self.params.min_brightness + raw * (self.params.max_brightness - self.params.min_brightness)
    
    def calculate_with_normal(self, distance: float,
                               light_direction: Tuple[float, float, float],
                               panel_normal: Tuple[float, float, float]) -> float:
        """
        Calculate brightness considering panel facing direction.
        
        Args:
            distance: Distance from light source
            light_direction: Unit vector from light to panel
            panel_normal: Unit vector of panel's facing direction
            
        Returns:
            Brightness (0.0 to 1.0)
        """
        base = self.calculate(distance)
        
        if not self.params.use_normals:
            return base
        
        # Dot product of light direction and panel normal
        # Positive = light is in front of panel
        # Negative = light is behind panel
        dot = (light_direction[0] * panel_normal[0] + 
               light_direction[1] * panel_normal[1] + 
               light_direction[2] * panel_normal[2])
        
        # Facing factor: 1.0 when directly facing, 0.0 when perpendicular or behind
        facing = max(0.0, dot)
        
        # Blend between base brightness and facing-modulated brightness
        influence = self.params.normal_influence
        return base * (1.0 - influence + influence * facing)
    
    def calculate_3d(self, light_pos: Tuple[float, float, float],
                      target_pos: Tuple[float, float, float],
                      target_normal: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Calculate brightness from 3D positions.
        
        Args:
            light_pos: Light position (x, y, z)
            target_pos: Target position (x, y, z)
            target_normal: Optional target normal vector
            
        Returns:
            Brightness (0.0 to 1.0)
        """
        # Calculate distance
        dx = target_pos[0] - light_pos[0]
        dy = target_pos[1] - light_pos[1]
        dz = target_pos[2] - light_pos[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        if target_normal is None or not self.params.use_normals:
            return self.calculate(distance)
        
        # Normalize direction vector
        if distance > 0.001:
            direction = (dx / distance, dy / distance, dz / distance)
        else:
            direction = (0.0, 0.0, 1.0)
        
        return self.calculate_with_normal(distance, direction, target_normal)


# =============================================================================
# GRADIENT FUNCTIONS
# =============================================================================

def gradient_horizontal(x: float, x_min: float, x_max: float,
                        brightness_left: float, brightness_right: float) -> float:
    """
    Create a horizontal gradient.
    
    Args:
        x: Current X position
        x_min, x_max: X range
        brightness_left: Brightness at left edge
        brightness_right: Brightness at right edge
        
    Returns:
        Interpolated brightness
    """
    if x_max <= x_min:
        return brightness_left
    
    t = (x - x_min) / (x_max - x_min)
    t = max(0.0, min(1.0, t))
    return brightness_left + t * (brightness_right - brightness_left)


def gradient_vertical(y: float, y_min: float, y_max: float,
                      brightness_bottom: float, brightness_top: float) -> float:
    """
    Create a vertical gradient.
    
    Args:
        y: Current Y position
        y_min, y_max: Y range
        brightness_bottom: Brightness at bottom
        brightness_top: Brightness at top
        
    Returns:
        Interpolated brightness
    """
    if y_max <= y_min:
        return brightness_bottom
    
    t = (y - y_min) / (y_max - y_min)
    t = max(0.0, min(1.0, t))
    return brightness_bottom + t * (brightness_top - brightness_bottom)


def gradient_radial(x: float, y: float,
                    center_x: float, center_y: float,
                    radius: float,
                    brightness_center: float, brightness_edge: float,
                    falloff: FalloffType = FalloffType.SMOOTH) -> float:
    """
    Create a radial gradient.
    
    Args:
        x, y: Current position
        center_x, center_y: Gradient center
        radius: Gradient radius
        brightness_center: Brightness at center
        brightness_edge: Brightness at edge
        falloff: Falloff curve type
        
    Returns:
        Interpolated brightness
    """
    dx = x - center_x
    dy = y - center_y
    distance = math.sqrt(dx * dx + dy * dy)
    
    falloff_fn = FALLOFF_FUNCTIONS[falloff]
    t = falloff_fn(distance, radius)
    
    return brightness_edge + t * (brightness_center - brightness_edge)


# =============================================================================
# PULSE FUNCTIONS
# =============================================================================

def pulse_sine(phase: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Sinusoidal pulse.
    
    Args:
        phase: Current phase (0 to 2π for one cycle)
        min_val, max_val: Output range
        
    Returns:
        Pulsing value
    """
    t = (math.sin(phase) + 1.0) / 2.0  # 0 to 1
    return min_val + t * (max_val - min_val)


def pulse_triangle(phase: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Triangle wave pulse.
    
    Args:
        phase: Current phase (0 to 2π for one cycle)
        min_val, max_val: Output range
        
    Returns:
        Pulsing value
    """
    # Normalize phase to 0-1
    t = (phase % (2 * math.pi)) / (2 * math.pi)
    # Triangle: 0→1 for first half, 1→0 for second half
    if t < 0.5:
        v = t * 2
    else:
        v = 2 - t * 2
    return min_val + v * (max_val - min_val)


def pulse_breath(phase: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Breathing pulse (slow rise, faster fall).
    
    Args:
        phase: Current phase (0 to 2π for one cycle)
        min_val, max_val: Output range
        
    Returns:
        Pulsing value
    """
    # Use cosine for smooth breathing effect
    t = (math.cos(phase) + 1.0) / 2.0
    # Apply slight easing for more organic feel
    t = t * t * (3 - 2 * t)  # Smoothstep
    return min_val + t * (max_val - min_val)


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

_default_falloff: Optional[FalloffCalculator] = None


def get_default_falloff() -> FalloffCalculator:
    """Get the default falloff calculator."""
    global _default_falloff
    if _default_falloff is None:
        _default_falloff = FalloffCalculator()
    return _default_falloff
