"""
GUI Sliders and Checkboxes
==========================
Interactive controls for the OpenGL interface.
Extracted from lightController_osc.py.
"""

from typing import Optional

try:
    import pygame
    from pygame.locals import MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION
    from OpenGL.GL import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

# Import for text rendering
from .hud import draw_text_2d


class Checkbox:
    """Simple checkbox for GUI"""
    
    def __init__(self, x: int, y: int, size: int, label: str, checked: bool = False):
        """
        Initialize checkbox.
        
        Args:
            x, y: Position
            size: Checkbox size in pixels
            label: Text label
            checked: Initial state
        """
        self.rect = pygame.Rect(x, y, size, size) if OPENGL_AVAILABLE else None
        self.x = x
        self.y = y
        self.label = label
        self.checked = checked
        self.size = size
    
    def handle_event(self, event, screen_height: int) -> bool:
        """
        Handle mouse events.
        
        Args:
            event: pygame event
            screen_height: Screen height for coordinate conversion
            
        Returns:
            True if value changed
        """
        if not OPENGL_AVAILABLE:
            return False
            
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse_y = screen_height - event.pos[1]
            if self.rect.collidepoint(event.pos[0], mouse_y):
                self.checked = not self.checked
                return True
        return False
    
    def draw(self, font):
        """Draw the checkbox using OpenGL"""
        if not OPENGL_AVAILABLE:
            return
            
        x, y, s = self.x, self.y, self.size
        
        # Background
        glColor4f(0.2, 0.2, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + s, y)
        glVertex2f(x + s, y + s)
        glVertex2f(x, y + s)
        glEnd()
        
        # Checkmark if checked
        if self.checked:
            glColor4f(0.3, 0.8, 0.4, 1.0)
            margin = s * 0.2
            glBegin(GL_QUADS)
            glVertex2f(x + margin, y + margin)
            glVertex2f(x + s - margin, y + margin)
            glVertex2f(x + s - margin, y + s - margin)
            glVertex2f(x + margin, y + s - margin)
            glEnd()
        
        # Border
        glColor4f(0.5, 0.5, 0.5, 1.0)
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + s, y)
        glVertex2f(x + s, y + s)
        glVertex2f(x, y + s)
        glEnd()
        
        # Label
        draw_text_2d(x + s + 8, y + 2, self.label, font)


class Slider:
    """Simple horizontal slider for GUI"""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 min_val: float, max_val: float, value: float, 
                 label: str, format_str: str = "{:.1f}"):
        """
        Initialize slider.
        
        Args:
            x, y: Position
            width, height: Size
            min_val, max_val: Value range
            value: Initial value
            label: Text label
            format_str: Format string for value display
        """
        self.rect = pygame.Rect(x, y, width, height) if OPENGL_AVAILABLE else None
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.format_str = format_str
        self.dragging = False
    
    def handle_event(self, event, screen_height: int) -> bool:
        """
        Handle mouse events.
        
        Args:
            event: pygame event
            screen_height: Screen height for coordinate conversion
            
        Returns:
            True if value changed
        """
        if not OPENGL_AVAILABLE:
            return False
            
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse_y = screen_height - event.pos[1]
            if self.rect.collidepoint(event.pos[0], mouse_y):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
            return True
        return False
    
    def _update_value(self, mouse_x: int):
        """Update value based on mouse position."""
        rel_x = max(0, min(mouse_x - self.x, self.width))
        ratio = rel_x / self.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
    
    def draw(self, font):
        """Draw the slider using OpenGL"""
        if not OPENGL_AVAILABLE:
            return
            
        x, y, w, h = self.x, self.y, self.width, self.height
        
        # Background
        glColor4f(0.2, 0.2, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Fill based on value
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w = w * ratio
        glColor4f(0.3, 0.6, 0.8, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + fill_w, y)
        glVertex2f(x + fill_w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Border
        glColor4f(0.5, 0.5, 0.5, 1.0)
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Label and value
        val_str = self.format_str.format(self.value)
        draw_text_2d(x, y + h + 5, f"{self.label}: {val_str}", font)


def create_calibration_sliders(slider_x: int, display_height: int, 
                                slider_w: int, slider_h: int) -> dict:
    """
    Create the calibration sliders.
    
    Args:
        slider_x: X position for sliders
        display_height: Screen height
        slider_w, slider_h: Slider dimensions
        
    Returns:
        Dict of slider name -> Slider object
    """
    return {
        'offset_x': Slider(slider_x, display_height - 100, slider_w, slider_h, -200, 200, 0, "Offset X"),
        'offset_z': Slider(slider_x, display_height - 140, slider_w, slider_h, 0, 500, 250, "Offset Z"),
        'scale_x': Slider(slider_x, display_height - 190, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale X", "{:.2f}"),
        'scale_z': Slider(slider_x, display_height - 230, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale Z", "{:.2f}"),
    }


def create_personality_sliders(slider_x: int, display_height: int,
                                slider_w: int, slider_h: int) -> dict:
    """
    Create the personality sliders.
    
    Args:
        slider_x: X position for sliders
        display_height: Screen height
        slider_w, slider_h: Slider dimensions
        
    Returns:
        Dict of slider name -> Slider object
    """
    return {
        'responsiveness': Slider(slider_x, display_height - 330, slider_w, slider_h, 0, 1, 0.5, "Responsiveness", "{:.2f}"),
        'energy': Slider(slider_x, display_height - 370, slider_w, slider_h, 0, 1, 0.5, "Energy", "{:.2f}"),
        'attention_span': Slider(slider_x, display_height - 410, slider_w, slider_h, 0, 1, 0.5, "Attention", "{:.2f}"),
        'sociability': Slider(slider_x, display_height - 450, slider_w, slider_h, 0, 1, 0.5, "Sociability", "{:.2f}"),
        'exploration': Slider(slider_x, display_height - 490, slider_w, slider_h, 0, 1, 0.5, "Exploration", "{:.2f}"),
        'memory': Slider(slider_x, display_height - 530, slider_w, slider_h, 0, 1, 0.5, "Memory", "{:.2f}"),
    }


def create_global_sliders(slider_x: int, display_height: int,
                          slider_w: int, slider_h: int) -> dict:
    """
    Create the global multiplier sliders.
    
    Args:
        slider_x: X position for sliders
        display_height: Screen height
        slider_w, slider_h: Slider dimensions
        
    Returns:
        Dict of slider name -> Slider object
    """
    return {
        'brightness_global': Slider(slider_x, display_height - 600, slider_w, slider_h, 0.2, 2.0, 1.0, "Brightness ×", "{:.2f}"),
        'speed_global': Slider(slider_x, display_height - 640, slider_w, slider_h, 0.2, 2.0, 1.0, "Speed ×", "{:.2f}"),
        'pulse_global': Slider(slider_x, display_height - 680, slider_w, slider_h, 0.3, 3.0, 1.0, "Pulse ×", "{:.2f}"),
        'follow_speed_global': Slider(slider_x, display_height - 720, slider_w, slider_h, 0.5, 3.0, 1.0, "Follow Speed ×", "{:.2f}"),
        'dwell_influence': Slider(slider_x, display_height - 760, slider_w, slider_h, 0.0, 2.0, 1.0, "Dwell Influence", "{:.2f}"),
        'idle_trend_weight': Slider(slider_x, display_height - 800, slider_w, slider_h, 0.0, 2.0, 1.0, "Idle Trend ×", "{:.2f}"),
    }


def create_checkboxes(slider_x: int, display_height: int) -> dict:
    """
    Create checkboxes.
    
    Args:
        slider_x: X position
        display_height: Screen height
        
    Returns:
        Dict of checkbox name -> Checkbox object
    """
    return {
        'invert_x': Checkbox(slider_x, display_height - 265, 14, "Invert X Direction", checked=False),
    }
