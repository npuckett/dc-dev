"""
OpenGL Setup
=============
Display initialization and projection setup.
Extracted from lightController_osc.py.
"""

import math
from typing import Tuple, Optional

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME, FULLSCREEN
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import numpy as np
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    np = None


# =============================================================================
# DISPLAY INITIALIZATION
# =============================================================================

def init_display(fullscreen: bool = True, 
                 windowed_size: Tuple[int, int] = (1920, 1080)) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """
    Initialize pygame display with OpenGL context.
    
    Args:
        fullscreen: If True, use fullscreen mode
        windowed_size: Window size if not fullscreen
        
    Returns:
        (screen surface, display size tuple)
    """
    if not OPENGL_AVAILABLE:
        raise RuntimeError("pygame and OpenGL are required for display")
    
    pygame.init()
    pygame.font.init()
    
    # Get display info for fullscreen
    display_info = pygame.display.Info()
    fullscreen_size = (display_info.current_w, display_info.current_h)
    
    if fullscreen:
        display = fullscreen_size
        # Use NOFRAME instead of FULLSCREEN - stays visible when focus is lost
        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | NOFRAME)
    else:
        display = windowed_size
        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    pygame.display.set_caption("3D Light Controller V3")
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.1, 0.1, 0.15, 1.0)
    
    return screen, display


def create_fonts() -> Tuple:
    """
    Create pygame fonts for text rendering.
    
    Returns:
        (font, font_small, font_label)
    """
    if not OPENGL_AVAILABLE:
        return None, None, None
        
    font = pygame.font.SysFont('monospace', 14)
    font_small = pygame.font.SysFont('monospace', 12)
    font_label = pygame.font.SysFont('monospace', 11)
    
    return font, font_small, font_label


# =============================================================================
# PROJECTION SETUP
# =============================================================================

def setup_3d_projection(display_size: Tuple[int, int], 
                        view_width: Optional[int] = None,
                        fov: float = 45.0,
                        near: float = 1.0,
                        far: float = 5000.0):
    """
    Set up 3D perspective projection.
    
    Args:
        display_size: (width, height) of display
        view_width: Width of 3D viewport (for GUI panel offset)
        fov: Field of view in degrees
        near, far: Clipping planes
    """
    if not OPENGL_AVAILABLE:
        return
    
    if view_width is None:
        view_width = display_size[0]
    
    glViewport(0, 0, view_width, display_size[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = view_width / display_size[1]
    gluPerspective(fov, aspect, near, far)
    glMatrixMode(GL_MODELVIEW)


def setup_2d_projection(display_size: Tuple[int, int]):
    """
    Set up 2D orthographic projection for HUD/GUI.
    
    Args:
        display_size: (width, height) of display
    """
    if not OPENGL_AVAILABLE:
        return
    
    glViewport(0, 0, display_size[0], display_size[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, display_size[0], 0, display_size[1], -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)


def restore_3d_projection():
    """Restore 3D rendering state after 2D drawing."""
    if not OPENGL_AVAILABLE:
        return
    glEnable(GL_DEPTH_TEST)


# =============================================================================
# CAMERA CONTROL
# =============================================================================

class CameraController:
    """
    Handles 3D camera position and movement.
    
    The camera orbits around a target point with rotation and distance controls.
    """
    
    def __init__(self):
        """Initialize camera with default position."""
        # Camera rotation (degrees)
        self.rot_x = 25          # Looking down at the scene
        self.rot_y = 0           # Looking toward panels
        
        # Camera distance from target
        self.distance = 900
        
        # Target point (where camera looks at)
        self.target = np.array([-150.0, 0.0, 150.0]) if np else [-150.0, 0.0, 150.0]
        
        # Default values for reset
        self._default_rot_x = self.rot_x
        self._default_rot_y = self.rot_y
        self._default_distance = self.distance
        self._default_target = self.target.copy() if np else list(self.target)
        
        # Mouse state
        self.middle_mouse_down = False
        self._last_mouse_pos = None
    
    def reset(self):
        """Reset camera to default position."""
        self.rot_x = self._default_rot_x
        self.rot_y = self._default_rot_y
        self.distance = self._default_distance
        if np:
            self.target = self._default_target.copy()
        else:
            self.target = list(self._default_target)
    
    def handle_event(self, event) -> bool:
        """
        Handle pygame events for camera control.
        
        Args:
            event: pygame event
            
        Returns:
            True if event was handled
        """
        if not OPENGL_AVAILABLE:
            return False
            
        from pygame.locals import (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION,
                                   KEYDOWN, K_HOME)
        
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 2:  # Middle mouse
                self.middle_mouse_down = True
                self._last_mouse_pos = event.pos
                return True
            elif event.button == 4:  # Scroll up
                self.distance = max(100, self.distance - 50)
                return True
            elif event.button == 5:  # Scroll down
                self.distance = min(2000, self.distance + 50)
                return True
        
        elif event.type == MOUSEBUTTONUP:
            if event.button == 2:
                self.middle_mouse_down = False
                return True
        
        elif event.type == MOUSEMOTION:
            if event.buttons[0]:  # Left mouse - rotate
                if self._last_mouse_pos:
                    dx = event.pos[0] - self._last_mouse_pos[0]
                    dy = event.pos[1] - self._last_mouse_pos[1]
                    self.rot_y += dx * 0.3
                    self.rot_x = max(-90, min(90, self.rot_x + dy * 0.3))
                self._last_mouse_pos = event.pos
                return True
            elif self.middle_mouse_down or (event.buttons[0] and pygame.key.get_mods() & pygame.KMOD_SHIFT):
                # Middle mouse or Shift+left mouse - pan
                if self._last_mouse_pos:
                    dx = event.pos[0] - self._last_mouse_pos[0]
                    dy = event.pos[1] - self._last_mouse_pos[1]
                    # Pan in screen space
                    pan_speed = self.distance * 0.002
                    if np:
                        self.target[0] -= dx * pan_speed
                        self.target[1] += dy * pan_speed
                    else:
                        self.target[0] -= dx * pan_speed
                        self.target[1] += dy * pan_speed
                self._last_mouse_pos = event.pos
                return True
            else:
                self._last_mouse_pos = event.pos
        
        elif event.type == KEYDOWN:
            if event.key == K_HOME:
                self.reset()
                return True
        
        return False
    
    def apply_view_transform(self):
        """Apply camera view transformation to OpenGL modelview matrix."""
        if not OPENGL_AVAILABLE:
            return
            
        glLoadIdentity()
        
        # Calculate camera position from rotation and distance
        cam_pos = calculate_camera_position(
            self.target, self.distance, self.rot_x, self.rot_y
        )
        
        # Look at target
        if np:
            gluLookAt(
                cam_pos[0], cam_pos[1], cam_pos[2],
                self.target[0], self.target[1], self.target[2],
                0, 1, 0
            )
        else:
            gluLookAt(
                cam_pos[0], cam_pos[1], cam_pos[2],
                self.target[0], self.target[1], self.target[2],
                0, 1, 0
            )


def calculate_camera_position(target, distance: float, 
                               rot_x: float, rot_y: float) -> Tuple[float, float, float]:
    """
    Calculate camera position from target, distance, and rotation.
    
    Args:
        target: Target point (x, y, z)
        distance: Distance from target
        rot_x: Rotation around X axis (pitch) in degrees
        rot_y: Rotation around Y axis (yaw) in degrees
        
    Returns:
        (x, y, z) camera position
    """
    # Convert to radians
    rx = math.radians(rot_x)
    ry = math.radians(rot_y)
    
    # Calculate offset from target
    # Camera is positioned on a sphere around the target
    x = distance * math.sin(ry) * math.cos(rx)
    y = distance * math.sin(rx)
    z = -distance * math.cos(ry) * math.cos(rx)
    
    # Add to target
    if hasattr(target, '__getitem__'):
        return (target[0] + x, target[1] + y, target[2] + z)
    else:
        return (x, y, z)


# =============================================================================
# RENDER FRAME
# =============================================================================

def clear_frame():
    """Clear the frame buffer for new frame."""
    if not OPENGL_AVAILABLE:
        return
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


def swap_buffers():
    """Swap display buffers (call at end of frame)."""
    if OPENGL_AVAILABLE:
        pygame.display.flip()
