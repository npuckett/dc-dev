#!/usr/bin/env python3
"""
3D Point Light LED Controller - Pygame/OpenGL Version

A more compatible 3D visualization using Pygame and OpenGL.
Features:
- 3D panel geometry (12 panels)
- Point light as a sphere
- Real-time brightness calculation
- Interactive camera controls
- Trackzone visualization
- Simulated person
- Wandering behavior
- Art-Net output
- GUI controls for parameters

All units in centimeters.

Controls:
- Arrow keys: Move light manually
- W/S: Move light in Z
- P: Toggle person
- Space: Toggle wandering
- Mouse drag (in 3D view): Rotate camera
- Scroll: Zoom
- Q/ESC: Quit
- Click sliders in GUI panel to adjust values
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time
import random
import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

# Try to import Art-Net library
try:
    from stupidArtnet import StupidArtnet
    ARTNET_AVAILABLE = True
except ImportError:
    ARTNET_AVAILABLE = False
    print("stupidArtnet not available - running in visualization-only mode")

# =============================================================================
# CONFIGURATION (all units in centimeters)
# =============================================================================

# Art-Net settings
TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

# DMX range
DMX_MIN = 1
DMX_MAX = 50

# Panel dimensions (cm)
PANEL_SIZE = 60

# Unit spacing (cm)
UNIT_SPACING = 80

# Panel positions relative to unit center (y, z) in cm
PANEL_LOCAL_POSITIONS = {
    1: (90, 0),
    2: (30, 12),
    3: (30, -12),
}

# Panel angles (degrees from vertical)
PANEL_ANGLES = {
    1: 0,
    2: 22.5,
    3: -22.5,
}

# Panel normals
PANEL_NORMALS = {
    1: np.array([0.0, 0.0, 1.0]),
    2: np.array([0.0, 0.38268, 0.92388]),
    3: np.array([0.0, -0.38268, 0.92388]),
}

# Trackzone (cm)
TRACKZONE = {
    'width': 475,
    'depth': 205,
    'height': 300,
    'offset_z': 78,
    'offset_y': -66,
    'center_x': 120,
}

# Wander box (cm) - extends 20cm behind panels, 50cm gap from trackzone
# Panels back is at z=-12, so -12 - 20 = -32
# Trackzone starts at z=78, 50cm gap means max_z = 28
WANDER_BOX = {
    'min_x': -50, 'max_x': 290,
    'min_y': 0, 'max_y': 150,
    'min_z': -32, 'max_z': 28,
}

# =============================================================================
# CALIBRATION MARKERS (from MARKER_PLACEMENT_DIAGRAM.md)
# Coordinate system: X=left-right, Y=height (floor=0, street=-66), Z=depth from panels (Z=0 at panels)
# =============================================================================

MARKER_SIZE = 15  # cm - ArUco marker size

# Street level is 66 cm below storefront floor
# Standard convention: positive Y = UP, so below floor = negative Y
STREET_LEVEL_Y = -66
CAMERA_LEDGE_Y = -16  # Cameras are 50cm above street (16cm below floor)

# Marker positions: (X, Y, Z) in centimeters
# Flat markers: Y = STREET_LEVEL_Y (on the street)
# Vertical marker 5: Y = CAMERA_LEDGE_Y (at camera height, on storefront)
MARKER_POSITIONS = {
    0: {'pos': (-40, STREET_LEVEL_Y, 90), 'desc': 'Left front', 'camera': 'Cam 1', 'vertical': False},
    1: {'pos': (120, STREET_LEVEL_Y, 90), 'desc': 'Center front (SHARED)', 'camera': 'Both', 'vertical': False},
    2: {'pos': (280, STREET_LEVEL_Y, 90), 'desc': 'Right front', 'camera': 'Cam 2', 'vertical': False},
    3: {'pos': (-40, STREET_LEVEL_Y, 141), 'desc': 'Left back', 'camera': 'Cam 1', 'vertical': False},
    4: {'pos': (280, STREET_LEVEL_Y, 141), 'desc': 'Right back', 'camera': 'Cam 2', 'vertical': False},
    5: {'pos': (120, CAMERA_LEDGE_Y, 550), 'desc': 'Subway wall (VERTICAL)', 'camera': 'Both', 'vertical': True},
}

# Marker image files
MARKER_IMAGE_PATH = '/Users/npmac/Documents/GitHub/dc-dev/marker_{}.png'

# Toggle for marker visibility
SHOW_MARKERS = True


@dataclass
class PointLight:
    """Virtual point light"""
    position: np.ndarray = field(default_factory=lambda: np.array([120.0, 60.0, -30.0]))
    target_position: np.ndarray = field(default_factory=lambda: np.array([120.0, 60.0, -30.0]))
    
    brightness_min: int = 5
    brightness_max: int = 40
    pulse_speed: float = 2000
    falloff_radius: float = 50
    
    target_brightness_min: int = 5
    target_brightness_max: int = 40
    target_falloff_radius: float = 50
    
    move_speed: float = 50
    property_lerp_speed: float = 2.0
    pulse_phase: float = 0.0
    
    def get_brightness(self) -> float:
        return (math.sin(self.pulse_phase) + 1) / 2
    
    def update(self, dt: float):
        self.pulse_phase += (2 * math.pi * dt * 1000) / self.pulse_speed
        
        diff = self.target_position - self.position
        dist = np.linalg.norm(diff)
        if dist > 0.1:
            self.position += (diff / dist) * min(self.move_speed * dt, dist)
        
        # Note: falloff_radius is now controlled directly by slider, no lerping


@dataclass
class SimulatedPerson:
    enabled: bool = False
    position: np.ndarray = field(default_factory=lambda: np.array([120.0, -66.0, 150.0]))
    size: Tuple[float, float, float] = (40, 170, 30)
    
    def get_center(self) -> np.ndarray:
        return self.position + np.array([0, self.size[1]/2, 0])


class PanelSystem:
    def __init__(self):
        self.panels: Dict[Tuple[int, int], dict] = {}
        self._build_panels()
    
    def _build_panels(self):
        for unit in range(4):
            unit_x = unit * UNIT_SPACING
            for panel_num in range(1, 4):
                local_y, local_z = PANEL_LOCAL_POSITIONS[panel_num]
                self.panels[(unit, panel_num)] = {
                    'center': np.array([unit_x, local_y, local_z], dtype=float),
                    'normal': PANEL_NORMALS[panel_num].copy(),
                    'angle': PANEL_ANGLES[panel_num],
                    'brightness': 0.5,
                    'dmx_value': DMX_MIN,
                }
    
    def calculate_brightness(self, light: PointLight, use_normals: bool = False):
        intensity = light.get_brightness()
        
        for key, panel in self.panels.items():
            diff = panel['center'] - light.position
            distance = np.linalg.norm(diff)
            
            # Linear falloff: 1.0 at distance=0, 0.0 at distance=falloff_radius
            if light.falloff_radius > 0:
                falloff = max(0.0, 1.0 - (distance / light.falloff_radius))
            else:
                falloff = 0.0
            
            # Optional: Factor in panel normal direction
            # This makes panels facing the light brighter
            if use_normals and distance > 0.01:
                light_dir = -diff / distance  # Direction from panel to light
                normal_factor = max(0.0, np.dot(panel['normal'], light_dir))
                # Blend: 50% distance, 50% normal-weighted
                brightness = falloff * (0.5 + 0.5 * normal_factor) * intensity
            else:
                # Simple distance-only calculation
                brightness = falloff * intensity
            
            panel['brightness'] = max(0.0, min(1.0, brightness))
            
            # DMX output
            dmx_range = light.brightness_max - light.brightness_min
            panel['dmx_value'] = int(light.brightness_min + panel['brightness'] * dmx_range)
            panel['dmx_value'] = max(DMX_MIN, min(DMX_MAX, panel['dmx_value']))
    
    def get_dmx_values(self) -> List[int]:
        return [self.panels[(u, p)]['dmx_value'] for u in range(4) for p in range(1, 4)]


class WanderBehavior:
    def __init__(self, light: PointLight, wander_box: dict):
        self.light = light
        self.wander_box = wander_box
        self.wander_target = self._random_point()
        self.wander_timer = 0
        self.wander_interval = 3.0
        self.enabled = True
    
    def _random_point(self) -> np.ndarray:
        return np.array([
            random.uniform(self.wander_box['min_x'], self.wander_box['max_x']),
            random.uniform(self.wander_box['min_y'], self.wander_box['max_y']),
            random.uniform(self.wander_box['min_z'], self.wander_box['max_z']),
        ])
    
    def update(self, dt: float, person: Optional[SimulatedPerson] = None):
        if not self.enabled:
            return
        
        if person and person.enabled:
            # Map person's X and Z to the light, but clamp to wander box
            person_center = person.get_center()
            
            # Clamp X to wander box
            target_x = max(self.wander_box['min_x'], 
                          min(self.wander_box['max_x'], person_center[0]))
            
            # Keep Y within wander box (use current target or middle)
            target_y = self.light.target_position[1]
            target_y = max(self.wander_box['min_y'], 
                          min(self.wander_box['max_y'], target_y))
            
            # Map person's Z to wander box range
            # Person is in trackzone (positive Z), light stays in wander box (negative Z typically)
            # Simple approach: mirror the person's Z relative to panels
            target_z = max(self.wander_box['min_z'], 
                          min(self.wander_box['max_z'], -person_center[2] * 0.1))
            
            self.light.target_position = np.array([target_x, target_y, target_z])
            return
        
        self.wander_timer += dt
        dist = np.linalg.norm(self.light.position - self.wander_target)
        
        if dist < 10 or self.wander_timer > self.wander_interval:
            self.wander_target = self._random_point()
            self.wander_timer = 0
            # Only change position, not falloff - let slider control falloff
            # self.light.target_brightness_min = random.randint(3, 10)
            # self.light.target_brightness_max = random.randint(30, 45)
        
        self.light.target_position = self.wander_target.copy()


def draw_box_wireframe(bounds, color):
    """Draw wireframe box from bounds (xmin, xmax, ymin, ymax, zmin, zmax)"""
    x0, x1, y0, y1, z0, z1 = bounds
    
    glColor4f(*color)
    glBegin(GL_LINES)
    
    # Bottom face
    glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
    glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
    glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1)
    glVertex3f(x0, y0, z1); glVertex3f(x0, y0, z0)
    
    # Top face
    glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
    glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
    glVertex3f(x0, y1, z1); glVertex3f(x0, y1, z0)
    
    # Vertical edges
    glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0)
    glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0)
    glVertex3f(x1, y0, z1); glVertex3f(x1, y1, z1)
    glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1)
    
    glEnd()


def draw_panel(center, angle, size, brightness):
    """Draw a panel as a quad"""
    half = size / 2
    
    glPushMatrix()
    glTranslatef(*center)
    glRotatef(-angle, 1, 0, 0)
    
    # Set color based on brightness
    gray = 0.2 + brightness * 0.8
    glColor4f(gray, gray, gray, 1.0)
    
    glBegin(GL_QUADS)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    # Draw outline
    glColor4f(0.3, 0.3, 0.3, 1.0)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    glPopMatrix()


def draw_sphere(center, radius, color, segments=12):
    """Draw a simple sphere"""
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, segments, segments)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_sphere_wireframe(center, radius, color, segments=16):
    """Draw a wireframe sphere to show falloff radius"""
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    glLineWidth(1)
    
    # Draw latitude circles
    for i in range(segments // 2 + 1):
        lat = math.pi * i / (segments // 2) - math.pi / 2
        r = radius * math.cos(lat)
        y = radius * math.sin(lat)
        
        glBegin(GL_LINE_LOOP)
        for j in range(segments):
            lon = 2 * math.pi * j / segments
            x = r * math.cos(lon)
            z = r * math.sin(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    # Draw longitude circles
    for j in range(segments // 2):
        lon = math.pi * j / (segments // 2)
        
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            lat = 2 * math.pi * i / segments
            x = radius * math.cos(lat) * math.cos(lon)
            y = radius * math.sin(lat)
            z = radius * math.cos(lat) * math.sin(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    glPopMatrix()


def draw_person(position, size, enabled):
    """Draw person as a box"""
    if not enabled:
        return
    
    x, y, z = position
    w, h, d = size
    
    glColor4f(1.0, 0.5, 0.0, 0.7)
    
    bounds = (x - w/2, x + w/2, y, y + h, z - d/2, z + d/2)
    x0, x1, y0, y1, z0, z1 = bounds
    
    glBegin(GL_QUADS)
    # Front
    glVertex3f(x0, y0, z1); glVertex3f(x1, y0, z1)
    glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
    # Back
    glVertex3f(x1, y0, z0); glVertex3f(x0, y0, z0)
    glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
    # Left
    glVertex3f(x0, y0, z0); glVertex3f(x0, y0, z1)
    glVertex3f(x0, y1, z1); glVertex3f(x0, y1, z0)
    # Right
    glVertex3f(x1, y0, z1); glVertex3f(x1, y0, z0)
    glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1)
    # Top
    glVertex3f(x0, y1, z1); glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z0); glVertex3f(x0, y1, z0)
    # Bottom
    glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
    glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1)
    glEnd()


def draw_floor(y_level, color, size=400):
    """Draw a floor plane"""
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex3f(-100, y_level, -200)
    glVertex3f(400, y_level, -200)
    glVertex3f(400, y_level, 400)
    glVertex3f(-100, y_level, 400)
    glEnd()


def load_marker_textures() -> Dict[int, int]:
    """Load ArUco marker images as OpenGL textures"""
    textures = {}
    
    for marker_id in MARKER_POSITIONS.keys():
        path = MARKER_IMAGE_PATH.format(marker_id)
        if not os.path.exists(path):
            print(f"Warning: Marker image not found: {path}")
            continue
        
        try:
            # Load image with pygame
            surface = pygame.image.load(path)
            # Convert to RGBA
            surface = surface.convert_alpha()
            # Flip vertically for OpenGL
            surface = pygame.transform.flip(surface, False, True)
            
            width = surface.get_width()
            height = surface.get_height()
            data = pygame.image.tostring(surface, "RGBA", True)
            
            # Create OpenGL texture
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            
            textures[marker_id] = tex_id
            print(f"Loaded marker {marker_id} texture")
        except Exception as e:
            print(f"Error loading marker {marker_id}: {e}")
    
    return textures


def draw_marker(marker_id: int, position: Tuple[float, float, float], size: float, 
                texture_id: Optional[int], font, vertical: bool = False):
    """
    Draw a calibration marker as a textured plane.
    If vertical=False: lies flat on floor facing upward
    If vertical=True: stands upright facing outward (toward positive Z / street)
    """
    x, y, z = position
    half = size / 2
    
    glPushMatrix()
    glTranslatef(x, y, z)
    
    if vertical:
        # Vertical marker: stands upright, facing outward toward street (positive Z)
        # No rotation needed - just draw in XY plane
        glTranslatef(0, 0, 0.5)  # Slightly forward to avoid z-fighting with wall
    else:
        # Horizontal marker: lies flat on floor, facing up
        glTranslatef(0, 0.5, 0)  # Slightly above floor to avoid z-fighting
        glRotatef(-90, 1, 0, 0)  # Rotate to lie flat
    
    if texture_id is not None:
        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1, 1, 1, 1)  # Full brightness, no tint
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-half, -half, 0)
        glTexCoord2f(1, 0); glVertex3f(half, -half, 0)
        glTexCoord2f(1, 1); glVertex3f(half, half, 0)
        glTexCoord2f(0, 1); glVertex3f(-half, half, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
    else:
        # Draw white placeholder with number
        glColor4f(1, 1, 1, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(-half, -half, 0)
        glVertex3f(half, -half, 0)
        glVertex3f(half, half, 0)
        glVertex3f(-half, half, 0)
        glEnd()
    
    # Draw border
    glColor4f(0, 0, 0, 1)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-half, -half, 0.1)
    glVertex3f(half, -half, 0.1)
    glVertex3f(half, half, 0.1)
    glVertex3f(-half, half, 0.1)
    glEnd()
    
    glPopMatrix()
    
    # Draw marker ID label floating nearby
    glPushMatrix()
    if vertical:
        glTranslatef(x, y + half + 5, z)  # Above the vertical marker
    else:
        glTranslatef(x, y + 5, z)  # Above floor marker
    
    # Draw a small sphere as position indicator
    glColor4f(1, 1, 0, 1)  # Yellow
    quadric = gluNewQuadric()
    gluSphere(quadric, 2, 8, 8)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_marker_labels(marker_positions: Dict, font, cam_target: np.ndarray):
    """
    Draw 2D labels for markers (called in HUD/2D rendering phase)
    This is a placeholder - actual 3D->2D projection would be needed for accurate positioning
    """
    pass  # Labels are drawn in HUD section instead


def draw_text(x, y, text, font, color=(255, 255, 255)):
    """Draw text on screen"""
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)


class Slider:
    """Simple horizontal slider for GUI"""
    def __init__(self, x, y, width, height, min_val, max_val, value, label, format_str="{:.0f}"):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.format_str = format_str
        self.dragging = False
    
    def handle_event(self, event, screen_height):
        """Handle mouse events. Returns True if value changed."""
        # Convert OpenGL y to pygame y
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            # Pygame y is from top, we need to check in pygame coords
            py_y = screen_height - self.rect.y - self.rect.height
            click_rect = pygame.Rect(self.rect.x, py_y, self.rect.width, self.rect.height)
            if click_rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
            return True
        return False
    
    def _update_value(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.rect.x, self.rect.width))
        ratio = rel_x / self.rect.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
    
    def draw(self, font):
        """Draw the slider using OpenGL"""
        x, y, w, h = self.rect.x, self.rect.y, self.rect.width, self.rect.height
        
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
        draw_text(x, y + h + 5, f"{self.label}: {val_str}", font)


def main():
    pygame.init()
    pygame.font.init()
    
    display = (1400, 800)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Point Light Controller")
    
    font = pygame.font.SysFont('monospace', 14)
    font_small = pygame.font.SysFont('monospace', 12)
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.1, 0.1, 0.15, 1.0)
    
    # Camera
    cam_rot_x = 20
    cam_rot_y = -30
    cam_distance = 500
    cam_target = np.array([120.0, 50.0, 0.0])
    
    # GUI panel width
    gui_width = 250
    view_width = display[0] - gui_width
    
    # Create system
    panel_system = PanelSystem()
    light = PointLight()
    person = SimulatedPerson()
    
    # Dynamic wander box (copy from defaults)
    wander_box = dict(WANDER_BOX)
    wander = WanderBehavior(light, wander_box)
    
    # Create sliders (x, y from bottom-left in OpenGL coords)
    slider_x = view_width + 20
    slider_w = gui_width - 40
    slider_h = 15
    
    sliders = {
        'brightness_min': Slider(slider_x, 700, slider_w, slider_h, 1, 50, light.brightness_min, "Brightness Min"),
        'brightness_max': Slider(slider_x, 650, slider_w, slider_h, 1, 50, light.brightness_max, "Brightness Max"),
        'pulse_speed': Slider(slider_x, 600, slider_w, slider_h, 500, 5000, light.pulse_speed, "Pulse Speed (ms)"),
        'falloff_radius': Slider(slider_x, 550, slider_w, slider_h, 1, 200, light.falloff_radius, "Falloff Radius (cm)"),
        'wander_speed': Slider(slider_x, 500, slider_w, slider_h, 10, 200, light.move_speed, "Wander Speed (cm/s)"),
        'wander_min_x': Slider(slider_x, 430, slider_w, slider_h, -100, 100, wander_box['min_x'], "Wander Min X"),
        'wander_max_x': Slider(slider_x, 380, slider_w, slider_h, 200, 400, wander_box['max_x'], "Wander Max X"),
        'wander_min_y': Slider(slider_x, 330, slider_w, slider_h, -50, 50, wander_box['min_y'], "Wander Min Y"),
        'wander_max_y': Slider(slider_x, 280, slider_w, slider_h, 50, 200, wander_box['max_y'], "Wander Max Y"),
        'wander_min_z': Slider(slider_x, 230, slider_w, slider_h, -100, 0, wander_box['min_z'], "Wander Min Z"),
        'wander_max_z': Slider(slider_x, 180, slider_w, slider_h, 0, 100, wander_box['max_z'], "Wander Max Z"),
    }
    
    # Art-Net
    artnet = None
    if ARTNET_AVAILABLE:
        try:
            artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            artnet.start()
            print(f"Art-Net connected to {TARGET_IP}")
        except Exception as e:
            print(f"Art-Net failed: {e}")
    
    # Load marker textures
    marker_textures = load_marker_textures()
    show_markers = SHOW_MARKERS
    
    clock = pygame.time.Clock()
    last_time = time.time()
    mouse_down = False
    last_mouse = (0, 0)
    slider_active = False
    
    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    running = False
                elif event.key == K_p:
                    person.enabled = not person.enabled
                    print(f"Person {'enabled' if person.enabled else 'disabled'}")
                elif event.key == K_SPACE:
                    wander.enabled = not wander.enabled
                    print(f"Wandering {'enabled' if wander.enabled else 'disabled'}")
                elif event.key == K_m:
                    show_markers = not show_markers
                    print(f"Markers {'visible' if show_markers else 'hidden'}")
            
            # Handle slider events first
            slider_active = False
            for name, slider in sliders.items():
                if slider.handle_event(event, display[1]):
                    slider_active = True
                    # Update values from sliders
                    if name == 'brightness_min':
                        light.brightness_min = int(slider.value)
                        light.target_brightness_min = int(slider.value)
                    elif name == 'brightness_max':
                        light.brightness_max = int(slider.value)
                        light.target_brightness_max = int(slider.value)
                    elif name == 'pulse_speed':
                        light.pulse_speed = slider.value
                    elif name == 'falloff_radius':
                        light.falloff_radius = slider.value
                        light.target_falloff_radius = slider.value
                    elif name == 'wander_speed':
                        light.move_speed = slider.value
                    elif name.startswith('wander_') and name != 'wander_speed':
                        key = name.replace('wander_', '')
                        wander_box[key] = slider.value
            
            # Camera controls only in 3D view area and when not dragging slider
            if not slider_active:
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1 and event.pos[0] < view_width:
                        mouse_down = True
                        last_mouse = event.pos
                    elif event.button == 4:  # Scroll up
                        cam_distance = max(100, cam_distance - 30)
                    elif event.button == 5:  # Scroll down
                        cam_distance = min(1000, cam_distance + 30)
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        mouse_down = False
                elif event.type == MOUSEMOTION:
                    if mouse_down and event.pos[0] < view_width:
                        dx = event.pos[0] - last_mouse[0]
                        dy = event.pos[1] - last_mouse[1]
                        cam_rot_y += dx * 0.5
                        cam_rot_x += dy * 0.5
                        cam_rot_x = max(-89, min(89, cam_rot_x))
                        last_mouse = event.pos
        
        # Keyboard controls for light
        keys = pygame.key.get_pressed()
        if not wander.enabled:
            speed = 100
            dt_key = clock.get_time() / 1000
            if keys[K_LEFT]:
                light.target_position[0] -= speed * dt_key
            if keys[K_RIGHT]:
                light.target_position[0] += speed * dt_key
            if keys[K_UP]:
                light.target_position[1] += speed * dt_key
            if keys[K_DOWN]:
                light.target_position[1] -= speed * dt_key
            if keys[K_w]:
                light.target_position[2] += speed * dt_key
            if keys[K_s]:
                light.target_position[2] -= speed * dt_key
        
        # Update
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now
        
        wander.update(dt, person if person.enabled else None)
        light.update(dt)
        panel_system.calculate_brightness(light)
        
        # Send Art-Net
        if artnet:
            artnet.set(panel_system.get_dmx_values())
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up perspective for 3D view (left portion only)
        glViewport(0, 0, view_width, display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, view_width/display[1], 10, 2000)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera position
        cam_x = cam_target[0] + cam_distance * math.cos(math.radians(cam_rot_x)) * math.sin(math.radians(cam_rot_y))
        cam_y = cam_target[1] + cam_distance * math.sin(math.radians(cam_rot_x))
        cam_z = cam_target[2] + cam_distance * math.cos(math.radians(cam_rot_x)) * math.cos(math.radians(cam_rot_y))
        
        gluLookAt(cam_x, cam_y, cam_z, *cam_target, 0, 1, 0)
        
        # Draw floors
        draw_floor(TRACKZONE['offset_y'], (0.2, 0.2, 0.25, 0.5))  # Street level
        draw_floor(0, (0.25, 0.25, 0.3, 0.5))  # Storefront level
        
        # Draw trackzone
        tz = TRACKZONE
        tz_bounds = (
            tz['center_x'] - tz['width']/2, tz['center_x'] + tz['width']/2,
            tz['offset_y'], tz['offset_y'] + tz['height'],
            tz['offset_z'], tz['offset_z'] + tz['depth']
        )
        draw_box_wireframe(tz_bounds, (0, 1, 1, 0.5))
        
        # Draw wander box (use dynamic values)
        wb = wander_box
        wb_bounds = (wb['min_x'], wb['max_x'], wb['min_y'], wb['max_y'], wb['min_z'], wb['max_z'])
        draw_box_wireframe(wb_bounds, (0, 1, 0, 0.3))
        
        # Draw panels
        for (unit, panel_num), panel in panel_system.panels.items():
            draw_panel(panel['center'], panel['angle'], PANEL_SIZE, panel['brightness'])
        
        # Draw calibration markers
        if show_markers:
            for marker_id, marker_data in MARKER_POSITIONS.items():
                pos = marker_data['pos']
                tex_id = marker_textures.get(marker_id)
                is_vertical = marker_data.get('vertical', False)
                draw_marker(marker_id, pos, MARKER_SIZE, tex_id, font, vertical=is_vertical)
        
        # Draw light
        brightness = light.get_brightness()
        radius = 8 + brightness * 7
        draw_sphere(light.position, radius, (1, 1, brightness, 1))
        
        # Draw falloff radius wireframe sphere
        draw_sphere_wireframe(light.position, light.falloff_radius, (1, 0.8, 0, 0.3), segments=24)
        
        # Draw person
        draw_person(person.position, person.size, person.enabled)
        
        # Draw HUD
        glViewport(0, 0, display[0], display[1])  # Full viewport for HUD
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], 0, display[1], -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Draw GUI panel background
        glColor4f(0.12, 0.12, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(view_width, 0)
        glVertex2f(display[0], 0)
        glVertex2f(display[0], display[1])
        glVertex2f(view_width, display[1])
        glEnd()
        
        # Draw separator line
        glColor4f(0.4, 0.4, 0.5, 1.0)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(view_width, 0)
        glVertex2f(view_width, display[1])
        glEnd()
        
        # GUI title
        draw_text(view_width + 20, display[1] - 30, "CONTROLS", font)
        draw_text(view_width + 20, display[1] - 50, "─" * 20, font)
        
        # Draw sliders
        for slider in sliders.values():
            slider.draw(font_small)
        
        # Status info at bottom of GUI
        draw_text(view_width + 20, 150, f"Markers: {'ON' if show_markers else 'OFF'}", font_small)
        draw_text(view_width + 20, 130, f"Wander: {'ON' if wander.enabled else 'OFF'}", font_small)
        draw_text(view_width + 20, 110, f"Person: {'ON' if person.enabled else 'OFF'}", font_small)
        draw_text(view_width + 20, 80, "KEYS:", font_small)
        draw_text(view_width + 20, 60, "SPACE=wander P=person M=markers", font_small)
        draw_text(view_width + 20, 40, "Arrows/W/S = move light", font_small)
        draw_text(view_width + 20, 20, "Q/ESC = quit", font_small)
        
        # HUD text in 3D view
        dmx_vals = panel_system.get_dmx_values()
        info_lines = [
            f"Light: ({light.position[0]:.0f}, {light.position[1]:.0f}, {light.position[2]:.0f}) cm",
            f"DMX: {dmx_vals}",
        ]
        
        for i, line in enumerate(info_lines):
            draw_text(10, display[1] - 25 - i * 20, line, font)
        
        # Draw marker legend if markers are visible
        if show_markers:
            draw_text(10, 140, "CALIBRATION MARKERS:", font_small, (255, 255, 0))
            y_offset = 120
            for marker_id, marker_data in MARKER_POSITIONS.items():
                pos = marker_data['pos']
                desc = marker_data['desc']
                cam = marker_data['camera']
                draw_text(10, y_offset, f"  [{marker_id}] ({pos[0]}, {pos[1]}, {pos[2]}) - {desc} ({cam})", font_small)
                y_offset -= 18
        
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(FPS)
    
    if artnet:
        artnet.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
