"""
Debug Visualization
===================
Optional debug views using pygame/matplotlib for development.
This module is entirely optional - the system works without it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math

# Check for optional dependencies
try:
    import pygame
    from pygame import gfxdraw
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# DEBUG VIEW CONFIG
# =============================================================================

@dataclass
class DebugViewConfig:
    """Configuration for debug view."""
    width: int = 800
    height: int = 600
    
    # Colors (RGB)
    background_color: Tuple[int, int, int] = (20, 20, 30)
    panel_off_color: Tuple[int, int, int] = (40, 40, 50)
    panel_on_color: Tuple[int, int, int] = (255, 255, 200)
    light_color: Tuple[int, int, int] = (255, 200, 100)
    grid_color: Tuple[int, int, int] = (50, 50, 60)
    text_color: Tuple[int, int, int] = (200, 200, 200)
    active_zone_color: Tuple[int, int, int] = (50, 100, 50)
    passive_zone_color: Tuple[int, int, int] = (100, 100, 50)
    
    # Scale and offset
    scale: float = 2.0  # pixels per cm
    offset_x: float = 450  # Center offset
    offset_y: float = 200


# =============================================================================
# 2D DEBUG VIEW (Pygame)
# =============================================================================

class DebugView2D:
    """
    Simple 2D debug view using pygame.
    Shows panels, light position, and zones from above.
    """
    
    def __init__(self, config: Optional[DebugViewConfig] = None):
        if not HAS_PYGAME:
            raise ImportError("pygame not available - install with: pip install pygame")
        
        self.config = config or DebugViewConfig()
        self.screen = None
        self.font = None
        self._running = False
    
    def start(self) -> bool:
        """Initialize pygame and create window."""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.config.width, self.config.height)
            )
            pygame.display.set_caption("Light Controller Debug")
            self.font = pygame.font.Font(None, 24)
            self._running = True
            return True
        except Exception as e:
            print(f"Debug view init failed: {e}")
            return False
    
    def stop(self):
        """Close the window."""
        if self._running:
            pygame.quit()
            self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def process_events(self) -> bool:
        """
        Process pygame events.
        
        Returns:
            False if window was closed
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return False
        return True
    
    def _world_to_screen(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        # X maps to screen X (right is positive in world, but we flip for display)
        # Z maps to screen Y (positive Z is "away" from viewer)
        sx = int(self.config.offset_x + x * self.config.scale)
        sy = int(self.config.offset_y + z * self.config.scale)
        return (sx, sy)
    
    def render(self, 
               light_x: float = -150, 
               light_z: float = 0,
               panel_brightness: Optional[Dict[Tuple[int, int], float]] = None,
               tracked_people: Optional[List[Dict]] = None,
               zones: Optional[Dict[str, Dict]] = None,
               info_text: Optional[List[str]] = None):
        """
        Render a frame.
        
        Args:
            light_x, light_z: Light position (ignoring Y for 2D view)
            panel_brightness: Dict of (unit, panel) -> brightness (0-1)
            tracked_people: List of person dicts with 'x', 'z', 'zone'
            zones: Zone definitions with 'min_x', 'max_x', 'min_z', 'max_z'
            info_text: Lines of text to display
        """
        if not self._running:
            return
        
        # Clear
        self.screen.fill(self.config.background_color)
        
        # Draw zones
        if zones:
            self._draw_zones(zones)
        
        # Draw panels
        self._draw_panels(panel_brightness or {})
        
        # Draw light
        self._draw_light(light_x, light_z)
        
        # Draw tracked people
        if tracked_people:
            self._draw_people(tracked_people)
        
        # Draw info text
        if info_text:
            self._draw_info(info_text)
        
        # Flip
        pygame.display.flip()
    
    def _draw_zones(self, zones: Dict[str, Dict]):
        """Draw zone rectangles."""
        for name, zone in zones.items():
            color = (self.config.active_zone_color if 'active' in name.lower() 
                    else self.config.passive_zone_color)
            
            min_x = zone.get('min_x', -200)
            max_x = zone.get('max_x', 0)
            min_z = zone.get('min_z', 0)
            max_z = zone.get('max_z', 300)
            
            p1 = self._world_to_screen(min_x, min_z)
            p2 = self._world_to_screen(max_x, max_z)
            
            rect = pygame.Rect(
                min(p1[0], p2[0]), min(p1[1], p2[1]),
                abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            )
            pygame.draw.rect(self.screen, color, rect, 1)
    
    def _draw_panels(self, brightness: Dict[Tuple[int, int], float]):
        """Draw panel representations."""
        # Simple panel representation - 4 units in a row
        for unit in range(4):
            unit_x = -(unit * 80 + 30)  # Same as real positions
            
            for panel in range(1, 4):
                b = brightness.get((unit, panel), 0.0)
                
                # Interpolate color
                off = self.config.panel_off_color
                on = self.config.panel_on_color
                color = tuple(int(off[i] + b * (on[i] - off[i])) for i in range(3))
                
                # Panel position (simplified - all at Z=0)
                px, py = self._world_to_screen(unit_x, 0)
                panel_size = int(20 * self.config.scale)
                
                # Offset panels vertically for visibility
                py_offset = (panel - 2) * 15
                
                rect = pygame.Rect(
                    px - panel_size // 2,
                    py + py_offset - panel_size // 2,
                    panel_size, panel_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
    
    def _draw_light(self, x: float, z: float):
        """Draw the light position."""
        sx, sy = self._world_to_screen(x, z)
        radius = 10
        
        # Glow effect
        for r in range(radius * 3, 0, -2):
            alpha = int(255 * (1 - r / (radius * 3)) * 0.3)
            color = (*self.config.light_color, alpha)
            # Use gfxdraw for anti-aliasing if available
            try:
                gfxdraw.filled_circle(self.screen, sx, sy, r, color)
            except:
                pygame.draw.circle(self.screen, self.config.light_color, (sx, sy), r)
        
        # Core
        pygame.draw.circle(self.screen, self.config.light_color, (sx, sy), radius)
    
    def _draw_people(self, people: List[Dict]):
        """Draw tracked people."""
        for person in people:
            x = person.get('x', 0)
            z = person.get('z', 0)
            zone = person.get('zone', 'unknown')
            
            sx, sy = self._world_to_screen(x, z)
            
            # Color by zone
            if zone == 'active':
                color = (100, 255, 100)
            elif zone == 'passive':
                color = (255, 255, 100)
            else:
                color = (150, 150, 150)
            
            pygame.draw.circle(self.screen, color, (sx, sy), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 8, 1)
    
    def _draw_info(self, lines: List[str]):
        """Draw info text."""
        y = 10
        for line in lines:
            text = self.font.render(line, True, self.config.text_color)
            self.screen.blit(text, (10, y))
            y += 20


# =============================================================================
# SIMPLE CONSOLE DEBUG
# =============================================================================

class ConsoleDebug:
    """
    Simple console-based debug output.
    Works without any graphical dependencies.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_mode = None
        self._last_update = 0
        self._update_interval = 1.0  # Print at most once per second
    
    def log_state(self, 
                  mode: str,
                  light_pos: Tuple[float, float, float],
                  active_count: int,
                  passive_count: int,
                  dmx_values: Optional[List[int]] = None):
        """Log current state to console."""
        if not self.enabled:
            return
        
        import time
        now = time.time()
        
        # Rate limit
        if now - self._last_update < self._update_interval:
            return
        
        self._last_update = now
        
        # Only log on mode change or periodically
        if mode != self._last_mode:
            print(f"\n[MODE] {mode}")
            self._last_mode = mode
        
        # Compact state line
        x, y, z = light_pos
        dmx_str = ""
        if dmx_values:
            dmx_str = f" DMX=[{','.join(str(v) for v in dmx_values[:4])}...]"
        
        print(f"  Light: ({x:.0f},{y:.0f},{z:.0f}) | Active: {active_count} | Passive: {passive_count}{dmx_str}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_debug_view(view_type: str = "auto",
                      config: Optional[DebugViewConfig] = None) -> Any:
    """
    Create a debug view.
    
    Args:
        view_type: "2d", "console", or "auto" (pick best available)
        config: Optional configuration
        
    Returns:
        Debug view instance, or None if unavailable
    """
    if view_type == "auto":
        if HAS_PYGAME:
            view_type = "2d"
        else:
            view_type = "console"
    
    if view_type == "2d":
        if not HAS_PYGAME:
            print("pygame not available, falling back to console")
            return ConsoleDebug()
        return DebugView2D(config)
    
    elif view_type == "console":
        return ConsoleDebug()
    
    return None


def check_dependencies() -> Dict[str, bool]:
    """Check available visualization dependencies."""
    return {
        'pygame': HAS_PYGAME,
        'matplotlib': HAS_MATPLOTLIB,
    }
