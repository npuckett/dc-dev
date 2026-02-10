"""
Panel Renderer
==============
Converts virtual light position to panel brightness values.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .panels import Panel, PanelGeometry, get_panel_geometry, PANEL_SIZE_CM
from .falloff import FalloffCalculator, FalloffParams, FalloffType


# =============================================================================
# POINT LIGHT
# =============================================================================

@dataclass
class PointLight:
    """
    Virtual point light source.
    
    The light has a position, target position (for smooth movement),
    and brightness parameters.
    """
    # Position
    x: float = -150.0
    y: float = 60.0
    z: float = 0.0
    
    # Target position (for smooth movement)
    target_x: float = -150.0
    target_y: float = 60.0
    target_z: float = 0.0
    
    # Brightness parameters
    brightness_min: int = 5
    brightness_max: int = 40
    
    # Pulse
    pulse_speed: float = 2000.0  # ms per cycle
    pulse_phase: float = 0.0
    pulse_enabled: bool = True
    
    # Falloff
    falloff_radius: float = 80.0
    
    # Movement
    move_speed: float = 50.0  # cm/s
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Get current position as tuple."""
        return (self.x, self.y, self.z)
    
    @position.setter
    def position(self, pos: Tuple[float, float, float]):
        """Set position from tuple."""
        self.x, self.y, self.z = pos
    
    @property
    def target_position(self) -> Tuple[float, float, float]:
        """Get target position as tuple."""
        return (self.target_x, self.target_y, self.target_z)
    
    @target_position.setter
    def target_position(self, pos: Tuple[float, float, float]):
        """Set target position from tuple."""
        self.target_x, self.target_y, self.target_z = pos
    
    def get_pulse_brightness(self) -> float:
        """
        Get current pulse brightness (0.0 to 1.0).
        """
        if not self.pulse_enabled:
            return 1.0
        return (math.sin(self.pulse_phase) + 1.0) / 2.0
    
    def update(self, dt: float):
        """
        Update light state.
        
        Args:
            dt: Delta time in seconds
        """
        # Update pulse phase
        if self.pulse_enabled and self.pulse_speed > 0:
            self.pulse_phase += (2 * math.pi * dt * 1000) / self.pulse_speed
            self.pulse_phase %= (2 * math.pi)
        
        # Move toward target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        if dist > 0.1:
            move = min(self.move_speed * dt, dist)
            self.x += (dx / dist) * move
            self.y += (dy / dist) * move
            self.z += (dz / dist) * move
    
    def set_target(self, x: float, y: float, z: float):
        """Set target position."""
        self.target_x = x
        self.target_y = y
        self.target_z = z
    
    def snap_to_target(self):
        """Instantly move to target."""
        self.x = self.target_x
        self.y = self.target_y
        self.z = self.target_z


# =============================================================================
# PANEL RENDERER
# =============================================================================

@dataclass
class RenderOutput:
    """Output of a render pass."""
    # Per-panel brightness (0.0-1.0)
    brightness: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Per-panel DMX values
    dmx_values: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    # DMX array in channel order (12 values)
    dmx_array: List[int] = field(default_factory=list)
    
    # Light state at render time
    light_x: float = 0.0
    light_y: float = 0.0
    light_z: float = 0.0
    light_pulse: float = 1.0


class PanelRenderer:
    """
    Renders a point light to panel brightness values.
    
    Usage:
        renderer = PanelRenderer()
        light = PointLight()
        
        # Each frame:
        light.update(dt)
        output = renderer.render(light)
        
        # Use output.dmx_array for DMX output
    """
    
    def __init__(self, 
                 geometry: Optional[PanelGeometry] = None,
                 falloff: Optional[FalloffCalculator] = None):
        """
        Initialize renderer.
        
        Args:
            geometry: Panel geometry (uses global if not provided)
            falloff: Falloff calculator (creates default if not provided)
        """
        self.geometry = geometry or get_panel_geometry()
        self.falloff = falloff or FalloffCalculator(FalloffParams(
            radius=80.0,
            falloff_type=FalloffType.SMOOTH,
        ))
        
        # Output state
        self._last_output = RenderOutput()
    
    def set_falloff_radius(self, radius: float):
        """Update falloff radius."""
        self.falloff.set_radius(radius)
    
    def set_falloff_type(self, falloff_type: FalloffType):
        """Change falloff curve type."""
        self.falloff.set_type(falloff_type)
    
    def render(self, light: PointLight) -> RenderOutput:
        """
        Render the light to panel brightness values.
        
        Args:
            light: Point light source
            
        Returns:
            RenderOutput with brightness and DMX values
        """
        output = RenderOutput()
        
        # Store light state
        output.light_x = light.x
        output.light_y = light.y
        output.light_z = light.z
        output.light_pulse = light.get_pulse_brightness()
        
        # Update falloff radius from light
        self.falloff.set_radius(light.falloff_radius)
        
        # Calculate brightness for each panel
        for panel in self.geometry.iter_panels():
            brightness = self._calculate_panel_brightness(light, panel)
            output.brightness[panel.key] = brightness
            
            # Convert to DMX
            dmx = self._brightness_to_dmx(brightness, light.brightness_min, light.brightness_max)
            output.dmx_values[panel.key] = dmx
        
        # Build DMX array in channel order
        output.dmx_array = [
            output.dmx_values.get((u, p), 0)
            for u in range(4) for p in range(1, 4)
        ]
        
        self._last_output = output
        return output
    
    def _calculate_panel_brightness(self, light: PointLight, panel: Panel) -> float:
        """Calculate brightness for a single panel."""
        # Distance from light to panel center
        dx = panel.center_x - light.x
        dy = panel.center_y - light.y
        dz = panel.center_z - light.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        # Base falloff
        falloff = self.falloff.calculate(distance)
        
        # Apply pulse
        intensity = light.get_pulse_brightness()
        
        return falloff * intensity
    
    def _brightness_to_dmx(self, brightness: float, 
                           dmx_min: int, dmx_max: int) -> int:
        """Convert brightness (0-1) to DMX value."""
        dmx_range = dmx_max - dmx_min
        dmx = int(dmx_min + brightness * dmx_range)
        return max(1, min(50, dmx))  # Clamp to safe DMX range
    
    def get_last_output(self) -> RenderOutput:
        """Get the most recent render output."""
        return self._last_output


# =============================================================================
# MULTI-LIGHT RENDERER
# =============================================================================

class MultiLightRenderer:
    """
    Renders multiple point lights to panels.
    Combines brightness using max blending.
    """
    
    def __init__(self, geometry: Optional[PanelGeometry] = None):
        self.geometry = geometry or get_panel_geometry()
        self.falloff = FalloffCalculator()
        self.dmx_min = 1
        self.dmx_max = 50
    
    def render(self, lights: List[PointLight]) -> RenderOutput:
        """
        Render multiple lights.
        
        Args:
            lights: List of point lights
            
        Returns:
            Combined render output
        """
        output = RenderOutput()
        
        # Initialize brightness to 0
        for panel in self.geometry.iter_panels():
            output.brightness[panel.key] = 0.0
        
        # Accumulate max brightness from each light
        for light in lights:
            self.falloff.set_radius(light.falloff_radius)
            intensity = light.get_pulse_brightness()
            
            for panel in self.geometry.iter_panels():
                dx = panel.center_x - light.x
                dy = panel.center_y - light.y
                dz = panel.center_z - light.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                
                falloff = self.falloff.calculate(distance)
                brightness = falloff * intensity
                
                # Max blending
                output.brightness[panel.key] = max(
                    output.brightness[panel.key],
                    brightness
                )
        
        # Convert to DMX
        for key, brightness in output.brightness.items():
            dmx_range = self.dmx_max - self.dmx_min
            dmx = int(self.dmx_min + brightness * dmx_range)
            output.dmx_values[key] = max(1, min(50, dmx))
        
        # Build array
        output.dmx_array = [
            output.dmx_values.get((u, p), 0)
            for u in range(4) for p in range(1, 4)
        ]
        
        return output


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_light(x: float = -150, y: float = 60, z: float = 0,
                 brightness_min: int = 5, brightness_max: int = 40,
                 falloff_radius: float = 80) -> PointLight:
    """
    Create a point light with common defaults.
    
    Args:
        x, y, z: Initial position
        brightness_min, brightness_max: DMX brightness range
        falloff_radius: Falloff distance
        
    Returns:
        Configured PointLight
    """
    light = PointLight()
    light.x = light.target_x = x
    light.y = light.target_y = y
    light.z = light.target_z = z
    light.brightness_min = brightness_min
    light.brightness_max = brightness_max
    light.falloff_radius = falloff_radius
    return light
