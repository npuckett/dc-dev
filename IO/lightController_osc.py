#!/usr/bin/env python3
"""
3D Point Light LED Controller with OSC Input

Receives tracked person positions via OSC and visualizes them in 3D space.
Includes calibration sliders to offset and scale the incoming tracking data.

OSC Messages Received:
  /tracker/person/<id> <x> <z>  - Position of tracked person (cm)
  /tracker/count <n>            - Number of people currently tracked

Controls:
- Arrow keys: Move light manually (when wander disabled)
- W/S: Move light in Z
- P: Toggle simulated person
- Space: Toggle wandering
- Mouse drag (in 3D view): Rotate camera
- Scroll: Zoom
- Q/ESC: Quit
- Sliders: Adjust calibration offsets and scales

All units in centimeters.
"""

import sys
import os
import math
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# OSC
from pythonosc import dispatcher, osc_server

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

# OSC settings
OSC_IP = "0.0.0.0"  # Listen on all interfaces
OSC_PORT = 7000

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

# Trackzone (cm) - defines the ACTIVE tracking area (engaging with installation)
TRACKZONE = {
    'width': 475,
    'depth': 205,
    'height': 300,
    'offset_z': 78,
    'offset_y': -66,  # Street level (below storefront)
    'center_x': 120,
}

# Passive trackzone (cm) - people passing by on sidewalk, not engaging
# Starts at back of active trackzone, extends further out
PASSIVE_TRACKZONE = {
    'width': 650,           # 650cm wide
    'depth': 330,           # 330cm deep
    'height': 300,
    'offset_z': 78 + 205,   # Starts at back of active zone (283cm)
    'offset_y': -66,        # Same street level
    'center_x': 120,        # Centered on panel midline
}

# Street level Y coordinate (where tracked people are placed)
STREET_LEVEL_Y = -66

# Wander box (cm)
WANDER_BOX = {
    'min_x': -50, 'max_x': 290,
    'min_y': 0, 'max_y': 150,
    'min_z': -32, 'max_z': 28,
}


# =============================================================================
# TRACKED PERSON FROM OSC
# =============================================================================

@dataclass
class TrackedPerson:
    """Represents a person tracked via OSC"""
    track_id: int
    x: float  # World X position (cm)
    z: float  # World Z position (cm)
    y: float = STREET_LEVEL_Y  # Fixed at street level
    last_update: float = 0.0
    
    def get_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class TrackedPersonManager:
    """Manages all tracked people received via OSC"""
    
    def __init__(self):
        self.people: Dict[int, TrackedPerson] = {}
        self.lock = threading.Lock()
        self.timeout = 1.0  # Remove person after 1 second without updates
        
        # Calibration offsets and scales
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0
    
    def update_person(self, track_id: int, raw_x: float, raw_z: float):
        """Update or add a tracked person with calibration applied"""
        # Apply calibration: scaled position + offset
        x = raw_x * self.scale_x + self.offset_x
        z = raw_z * self.scale_z + self.offset_z
        y = STREET_LEVEL_Y * self.scale_y + self.offset_y
        
        with self.lock:
            if track_id in self.people:
                self.people[track_id].x = x
                self.people[track_id].z = z
                self.people[track_id].y = y
                self.people[track_id].last_update = time.time()
            else:
                self.people[track_id] = TrackedPerson(
                    track_id=track_id,
                    x=x, z=z, y=y,
                    last_update=time.time()
                )
    
    def cleanup_stale(self):
        """Remove people who haven't been updated recently"""
        now = time.time()
        with self.lock:
            stale_ids = [pid for pid, p in self.people.items() 
                        if now - p.last_update > self.timeout]
            for pid in stale_ids:
                del self.people[pid]
    
    def get_all(self) -> List[TrackedPerson]:
        """Get list of all tracked people"""
        with self.lock:
            return list(self.people.values())
    
    def count(self) -> int:
        """Get count of tracked people"""
        with self.lock:
            return len(self.people)


# =============================================================================
# OSC HANDLER
# =============================================================================

class OSCHandler:
    """Handles incoming OSC messages"""
    
    def __init__(self, manager: TrackedPersonManager):
        self.manager = manager
        self.last_count = 0
    
    def handle_person(self, address: str, *args):
        """Handle /tracker/person/<id> messages"""
        try:
            # Extract track_id from address
            parts = address.split('/')
            track_id = int(parts[-1])
            
            if len(args) >= 2:
                x, z = float(args[0]), float(args[1])
                self.manager.update_person(track_id, x, z)
        except (ValueError, IndexError) as e:
            print(f"OSC parse error: {e}")
    
    def handle_count(self, address: str, *args):
        """Handle /tracker/count messages"""
        if args:
            self.last_count = int(args[0])


# =============================================================================
# POINT LIGHT & PANEL SYSTEM (from original)
# =============================================================================

@dataclass
class PointLight:
    """Virtual point light"""
    position: np.ndarray = field(default_factory=lambda: np.array([120.0, 60.0, -30.0]))
    target_position: np.ndarray = field(default_factory=lambda: np.array([120.0, 60.0, -30.0]))
    
    brightness_min: int = 5
    brightness_max: int = 40
    pulse_speed: float = 2000
    falloff_radius: float = 50
    
    move_speed: float = 50
    pulse_phase: float = 0.0
    
    def get_brightness(self) -> float:
        return (math.sin(self.pulse_phase) + 1) / 2
    
    def update(self, dt: float):
        self.pulse_phase += (2 * math.pi * dt * 1000) / self.pulse_speed
        
        diff = self.target_position - self.position
        dist = np.linalg.norm(diff)
        if dist > 0.1:
            move = min(self.move_speed * dt, dist)
            self.position += (diff / dist) * move


class PanelSystem:
    def __init__(self):
        self.panels: Dict[Tuple[int, int], dict] = {}
        self._build_panels()
    
    def _build_panels(self):
        for unit in range(4):
            unit_x = unit * UNIT_SPACING
            for panel_num in range(1, 4):
                local_y, local_z = PANEL_LOCAL_POSITIONS[panel_num]
                center = np.array([unit_x, local_y, local_z])
                
                self.panels[(unit, panel_num)] = {
                    'center': center,
                    'angle': PANEL_ANGLES[panel_num],
                    'normal': PANEL_NORMALS[panel_num].copy(),
                    'brightness': 0.0,
                    'dmx_value': 0,
                }
    
    def calculate_brightness(self, light: PointLight):
        intensity = light.get_brightness()
        
        for key, panel in self.panels.items():
            diff = panel['center'] - light.position
            distance = np.linalg.norm(diff)
            
            if light.falloff_radius > 0:
                falloff = max(0, 1.0 - distance / light.falloff_radius)
            else:
                falloff = 1.0
            
            final_brightness = falloff * intensity
            panel['brightness'] = final_brightness
            
            dmx_range = light.brightness_max - light.brightness_min
            panel['dmx_value'] = int(light.brightness_min + final_brightness * dmx_range)
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
    
    def update(self, dt: float):
        if not self.enabled:
            return
        
        self.wander_timer += dt
        dist = np.linalg.norm(self.light.position - self.wander_target)
        
        if dist < 10 or self.wander_timer > self.wander_interval:
            self.wander_target = self._random_point()
            self.wander_timer = 0
            self.wander_interval = random.uniform(2, 5)
        
        self.light.target_position = self.wander_target.copy()


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

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
    
    gray = 0.2 + brightness * 0.8
    glColor4f(gray, gray, gray, 1.0)
    
    glBegin(GL_QUADS)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
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
    """Draw a wireframe sphere"""
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    glLineWidth(1)
    
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
    
    for j in range(segments // 2):
        lon = math.pi * j / (segments // 2)
        
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            lat = 2 * math.pi * i / segments
            x = radius * math.cos(lat) * math.sin(lon)
            y = radius * math.sin(lat)
            z = radius * math.cos(lat) * math.cos(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    glPopMatrix()


def draw_tracked_person(person: TrackedPerson):
    """Draw a tracked person as a cylinder/capsule"""
    pos = person.get_position()
    
    # Draw as a colored cylinder (person height ~170cm)
    height = 170
    radius = 20
    
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    
    # Body cylinder
    glColor4f(0.2, 0.8, 0.2, 0.8)  # Green for tracked people
    
    quadric = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)  # Rotate to stand upright
    gluCylinder(quadric, radius, radius, height, 16, 1)
    
    # Top cap (head)
    glTranslatef(0, 0, height)
    gluSphere(quadric, radius, 12, 12)
    
    gluDeleteQuadric(quadric)
    glPopMatrix()
    
    # Draw ID label
    glColor4f(1, 1, 1, 1)
    # Note: Text rendering in 3D would need billboarding, skip for now


def draw_floor(y_level, color, size=400):
    """Draw a floor plane"""
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex3f(-100, y_level, -200)
    glVertex3f(400, y_level, -200)
    glVertex3f(400, y_level, 400)
    glVertex3f(-100, y_level, 400)
    glEnd()


def draw_text(x, y, text, font, color=(255, 255, 255)):
    """Draw text on screen"""
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)


# =============================================================================
# GUI SLIDER
# =============================================================================

class Slider:
    """Simple horizontal slider for GUI"""
    def __init__(self, x, y, width, height, min_val, max_val, value, label, format_str="{:.1f}"):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.format_str = format_str
        self.dragging = False
    
    def handle_event(self, event, screen_height):
        """Handle mouse events. Returns True if value changed."""
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    pygame.init()
    pygame.font.init()
    
    display = (1400, 800)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Light Controller - OSC Input")
    
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
    gui_width = 280
    view_width = display[0] - gui_width
    
    # Create systems
    panel_system = PanelSystem()
    light = PointLight()
    wander_box = dict(WANDER_BOX)
    wander = WanderBehavior(light, wander_box)
    
    # Tracked person manager
    tracked_manager = TrackedPersonManager()
    
    # OSC setup
    osc_handler = OSCHandler(tracked_manager)
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/tracker/person/*", osc_handler.handle_person)
    osc_dispatcher.map("/tracker/count", osc_handler.handle_count)
    
    osc_server_instance = osc_server.ThreadingOSCUDPServer(
        (OSC_IP, OSC_PORT), osc_dispatcher
    )
    osc_thread = threading.Thread(target=osc_server_instance.serve_forever, daemon=True)
    osc_thread.start()
    print(f"📡 OSC server listening on {OSC_IP}:{OSC_PORT}")
    
    # Create sliders
    slider_x = view_width + 20
    slider_w = gui_width - 40
    slider_h = 15
    
    # Calibration sliders
    sliders = {
        # Offset sliders
        'offset_x': Slider(slider_x, 700, slider_w, slider_h, -200, 200, 0, "Offset X (cm)"),
        'offset_y': Slider(slider_x, 650, slider_w, slider_h, -100, 100, 0, "Offset Y (cm)"),
        'offset_z': Slider(slider_x, 600, slider_w, slider_h, 0, 500, 250, "Offset Z (cm)"),
        # Scale sliders
        'scale_x': Slider(slider_x, 530, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale X", "{:.2f}"),
        'scale_y': Slider(slider_x, 480, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale Y", "{:.2f}"),
        'scale_z': Slider(slider_x, 430, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale Z", "{:.2f}"),
        # Light sliders (existing)
        'falloff_radius': Slider(slider_x, 360, slider_w, slider_h, 1, 200, light.falloff_radius, "Falloff Radius"),
        'brightness_max': Slider(slider_x, 310, slider_w, slider_h, 1, 50, light.brightness_max, "Brightness Max"),
    }
    
    # Art-Net
    artnet = None
    if ARTNET_AVAILABLE:
        try:
            artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            artnet.start()
            print(f"🎨 Art-Net output to {TARGET_IP}")
        except Exception as e:
            print(f"Art-Net failed: {e}")
    
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
            
            # Check sliders first
            for name, slider in sliders.items():
                if slider.handle_event(event, display[1]):
                    slider_active = True
                    # Update calibration values
                    if name.startswith('offset_') or name.startswith('scale_'):
                        tracked_manager.offset_x = sliders['offset_x'].value
                        tracked_manager.offset_y = sliders['offset_y'].value
                        tracked_manager.offset_z = sliders['offset_z'].value
                        tracked_manager.scale_x = sliders['scale_x'].value
                        tracked_manager.scale_y = sliders['scale_y'].value
                        tracked_manager.scale_z = sliders['scale_z'].value
                    elif name == 'falloff_radius':
                        light.falloff_radius = slider.value
                    elif name == 'brightness_max':
                        light.brightness_max = int(slider.value)
            
            if event.type == MOUSEBUTTONUP:
                slider_active = False
            
            if event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    running = False
                elif event.key == K_SPACE:
                    wander.enabled = not wander.enabled
            
            # Camera rotation (only in 3D view area)
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < view_width and not slider_active:
                    mouse_down = True
                    last_mouse = event.pos
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
            elif event.type == MOUSEMOTION and mouse_down:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                cam_rot_y += dx * 0.5
                cam_rot_x += dy * 0.3
                cam_rot_x = max(-89, min(89, cam_rot_x))
                last_mouse = event.pos
            elif event.type == MOUSEWHEEL:
                cam_distance -= event.y * 30
                cam_distance = max(100, min(1500, cam_distance))
        
        # Keyboard controls for light
        keys = pygame.key.get_pressed()
        if not wander.enabled:
            move_speed = 100
            now = time.time()
            dt_keys = min(now - last_time, 0.1)
            if keys[K_LEFT]:
                light.target_position[0] -= move_speed * dt_keys
            if keys[K_RIGHT]:
                light.target_position[0] += move_speed * dt_keys
            if keys[K_UP]:
                light.target_position[1] += move_speed * dt_keys
            if keys[K_DOWN]:
                light.target_position[1] -= move_speed * dt_keys
            if keys[K_w]:
                light.target_position[2] -= move_speed * dt_keys
            if keys[K_s]:
                light.target_position[2] += move_speed * dt_keys
        
        # Update
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now
        
        wander.update(dt)
        light.update(dt)
        panel_system.calculate_brightness(light)
        
        # Cleanup stale tracked people
        tracked_manager.cleanup_stale()
        
        # Send Art-Net
        if artnet:
            artnet.set(panel_system.get_dmx_values())
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up perspective for 3D view
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
        draw_floor(STREET_LEVEL_Y, (0.2, 0.2, 0.25, 0.5))  # Street level
        draw_floor(0, (0.25, 0.25, 0.3, 0.5))  # Storefront level
        
        # Draw trackzone (active - cyan)
        tz = TRACKZONE
        tz_bounds = (
            tz['center_x'] - tz['width']/2, tz['center_x'] + tz['width']/2,
            tz['offset_y'], tz['offset_y'] + tz['height'],
            tz['offset_z'], tz['offset_z'] + tz['depth']
        )
        draw_box_wireframe(tz_bounds, (0, 1, 1, 0.5))
        
        # Draw passive trackzone (orange/yellow)
        ptz = PASSIVE_TRACKZONE
        ptz_bounds = (
            ptz['center_x'] - ptz['width']/2, ptz['center_x'] + ptz['width']/2,
            ptz['offset_y'], ptz['offset_y'] + ptz['height'],
            ptz['offset_z'], ptz['offset_z'] + ptz['depth']
        )
        draw_box_wireframe(ptz_bounds, (1, 0.6, 0, 0.4))
        
        # Draw wander box
        wb = wander_box
        wb_bounds = (wb['min_x'], wb['max_x'], wb['min_y'], wb['max_y'], wb['min_z'], wb['max_z'])
        draw_box_wireframe(wb_bounds, (0, 1, 0, 0.3))
        
        # Draw panels
        for (unit, panel_num), panel in panel_system.panels.items():
            draw_panel(panel['center'], panel['angle'], PANEL_SIZE, panel['brightness'])
        
        # Draw light
        brightness = light.get_brightness()
        radius = 8 + brightness * 7
        draw_sphere(light.position, radius, (1, 1, brightness, 1))
        draw_sphere_wireframe(light.position, light.falloff_radius, (1, 0.8, 0, 0.3), segments=24)
        
        # Draw tracked people
        for person in tracked_manager.get_all():
            draw_tracked_person(person)
        
        # Draw HUD
        glViewport(0, 0, display[0], display[1])
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
        draw_text(view_width + 20, display[1] - 30, "CALIBRATION", font)
        draw_text(view_width + 20, display[1] - 50, "─" * 24, font)
        
        # Section labels
        draw_text(view_width + 20, 740, "Position Offsets:", font_small)
        draw_text(view_width + 20, 565, "Position Scales:", font_small)
        draw_text(view_width + 20, 395, "Light Settings:", font_small)
        
        # Draw sliders
        for slider in sliders.values():
            slider.draw(font_small)
        
        # Status info
        tracked_count = tracked_manager.count()
        draw_text(view_width + 20, 250, f"Tracked People: {tracked_count}", font)
        draw_text(view_width + 20, 230, f"OSC Port: {OSC_PORT}", font_small)
        draw_text(view_width + 20, 210, f"Wander: {'ON' if wander.enabled else 'OFF'}", font_small)
        
        # Show tracked person positions
        draw_text(view_width + 20, 180, "Tracked Positions:", font_small)
        y_pos = 160
        for person in tracked_manager.get_all()[:5]:  # Show max 5
            draw_text(view_width + 20, y_pos, 
                     f"  ID {person.track_id}: ({person.x:.0f}, {person.z:.0f})", font_small)
            y_pos -= 18
        
        # Controls help
        draw_text(view_width + 20, 60, "KEYS:", font_small)
        draw_text(view_width + 20, 40, "SPACE = toggle wander", font_small)
        draw_text(view_width + 20, 20, "Q/ESC = quit", font_small)
        
        # HUD text in 3D view
        dmx_vals = panel_system.get_dmx_values()
        info_lines = [
            f"Light: ({light.position[0]:.0f}, {light.position[1]:.0f}, {light.position[2]:.0f}) cm",
            f"DMX: {dmx_vals}",
        ]
        
        for i, line in enumerate(info_lines):
            draw_text(10, display[1] - 20 - i * 20, line, font_small)
        
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Cleanup
    osc_server_instance.shutdown()
    if artnet:
        artnet.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
