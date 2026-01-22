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
- M: Toggle calibration markers
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
import socket
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

# Tracking database
from tracking_database import TrackingDatabase

# Behavior system
from light_behavior import (
    BehaviorSystem, BehaviorMode, MetaParameters, GestureType,
    PRESETS, load_preset
)

# Try to import Art-Net library
try:
    from stupidArtnet import StupidArtnet
    ARTNET_AVAILABLE = True
except ImportError:
    ARTNET_AVAILABLE = False
    print("stupidArtnet not available - running in visualization-only mode")

# Try to import websockets library for public viewer
try:
    import asyncio
    import websockets
    import json
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("websockets not available - public viewer disabled (pip install websockets)")

# =============================================================================
# CONFIGURATION (all units in centimeters)
# =============================================================================

# OSC settings
OSC_IP = "0.0.0.0"  # Listen on all interfaces
OSC_PORT = 7000

# WebSocket settings (for public viewer)
WEBSOCKET_PORT = 8765
WEBSOCKET_ENABLED = True

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
CAMERA_LEDGE_Y = -16  # Cameras are 50cm above street (16cm below floor)

# Wander box (cm)
WANDER_BOX = {
    'min_x': -50, 'max_x': 290,
    'min_y': 0, 'max_y': 150,
    'min_z': -32, 'max_z': 28,
}

# =============================================================================
# CALIBRATION MARKERS
# =============================================================================

MARKER_SIZE = 15  # cm - ArUco marker size

# Marker positions: (X, Y, Z) in centimeters
MARKER_POSITIONS = {
    0: {'pos': (-40, STREET_LEVEL_Y, 90), 'desc': 'Left front', 'camera': 'Cam 1', 'vertical': False},
    1: {'pos': (120, STREET_LEVEL_Y, 90), 'desc': 'Center front (SHARED)', 'camera': 'Both', 'vertical': False},
    2: {'pos': (280, STREET_LEVEL_Y, 90), 'desc': 'Right front', 'camera': 'Cam 2', 'vertical': False},
    3: {'pos': (-40, STREET_LEVEL_Y, 141), 'desc': 'Left back', 'camera': 'Cam 1', 'vertical': False},
    4: {'pos': (280, STREET_LEVEL_Y, 141), 'desc': 'Right back', 'camera': 'Cam 2', 'vertical': False},
    5: {'pos': (120, CAMERA_LEDGE_Y, 550), 'desc': 'Subway wall (VERTICAL)', 'camera': 'Both', 'vertical': True},
}

# Marker image path (relative to workspace root)
MARKER_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'marker_{}.png')

# Toggle for marker visibility
SHOW_MARKERS = True


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
    first_seen: float = 0.0  # When first tracked
    zone: str = "unknown"  # "active", "passive", or "unknown"
    
    def get_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def is_in_active_zone(self) -> bool:
        return self.zone == "active"
    
    def is_in_passive_zone(self) -> bool:
        return self.zone == "passive"


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
        
        # Zone boundaries
        self.active_zone = {
            'x_min': TRACKZONE['center_x'] - TRACKZONE['width']/2,
            'x_max': TRACKZONE['center_x'] + TRACKZONE['width']/2,
            'z_min': TRACKZONE['offset_z'],
            'z_max': TRACKZONE['offset_z'] + TRACKZONE['depth'],
        }
        self.passive_zone = {
            'x_min': PASSIVE_TRACKZONE['center_x'] - PASSIVE_TRACKZONE['width']/2,
            'x_max': PASSIVE_TRACKZONE['center_x'] + PASSIVE_TRACKZONE['width']/2,
            'z_min': PASSIVE_TRACKZONE['offset_z'],
            'z_max': PASSIVE_TRACKZONE['offset_z'] + PASSIVE_TRACKZONE['depth'],
        }
        
        # Callbacks for behavior system
        self.on_person_entered = None
        self.on_person_left = None
        self.on_position_updated = None
    
    def _get_zone(self, x: float, z: float) -> str:
        """Determine which zone a position is in"""
        az = self.active_zone
        pz = self.passive_zone
        
        if (az['x_min'] <= x <= az['x_max'] and 
            az['z_min'] <= z <= az['z_max']):
            return "active"
        elif (pz['x_min'] <= x <= pz['x_max'] and 
              pz['z_min'] <= z <= pz['z_max']):
            return "passive"
        return "unknown"
    
    def update_person(self, track_id: int, raw_x: float, raw_z: float):
        """Update or add a tracked person with calibration applied"""
        # Apply calibration: scaled position + offset
        x = raw_x * self.scale_x + self.offset_x
        z = raw_z * self.scale_z + self.offset_z
        y = STREET_LEVEL_Y * self.scale_y + self.offset_y
        
        zone = self._get_zone(x, z)
        now = time.time()
        
        with self.lock:
            is_new = track_id not in self.people
            
            if is_new:
                self.people[track_id] = TrackedPerson(
                    track_id=track_id,
                    x=x, z=z, y=y,
                    last_update=now,
                    first_seen=now,
                    zone=zone
                )
                # Notify behavior system
                if self.on_person_entered:
                    pos = np.array([x, y, z])
                    is_active = zone == "active"
                    self.on_person_entered(track_id, pos, is_active)
            else:
                self.people[track_id].x = x
                self.people[track_id].z = z
                self.people[track_id].y = y
                self.people[track_id].zone = zone
                self.people[track_id].last_update = now
                
                # Notify position update
                if self.on_position_updated:
                    pos = np.array([x, y, z])
                    self.on_position_updated(track_id, pos)
    
    def cleanup_stale(self):
        """Remove people who haven't been updated recently"""
        now = time.time()
        with self.lock:
            stale_ids = [pid for pid, p in self.people.items() 
                        if now - p.last_update > self.timeout]
            for pid in stale_ids:
                del self.people[pid]
                if self.on_person_left:
                    self.on_person_left(pid)
    
    def get_all(self) -> List[TrackedPerson]:
        """Get list of all tracked people"""
        with self.lock:
            return list(self.people.values())
    
    def count(self) -> int:
        """Get count of tracked people"""
        with self.lock:
            return len(self.people)
    
    def count_active(self) -> int:
        """Count people in active zone"""
        with self.lock:
            return sum(1 for p in self.people.values() if p.is_in_active_zone())
    
    def count_passive(self) -> int:
        """Count people in passive zone"""
        with self.lock:
            return sum(1 for p in self.people.values() if p.is_in_passive_zone())
    
    def get_active_positions(self) -> List[np.ndarray]:
        """Get positions of people in active zone"""
        with self.lock:
            return [p.get_position() for p in self.people.values() if p.is_in_active_zone()]


# =============================================================================
# WEBSOCKET BROADCASTER (for public viewer)
# =============================================================================

class WebSocketBroadcaster:
    """Broadcasts installation state to web clients"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients = set()
        self.loop = None
        self.server = None
        self.thread = None
        self.current_state = {}
        self.running = False
    
    async def handler(self, websocket, path):
        """Handle a WebSocket connection"""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0]
        print(f"🌐 WebSocket client connected: {client_ip}")
        
        try:
            # Send current state immediately
            if self.current_state:
                await websocket.send(json.dumps(self.current_state))
            
            # Keep connection alive
            async for message in websocket:
                pass  # We don't expect messages from clients
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"🌐 WebSocket client disconnected: {client_ip}")
    
    async def broadcast(self, state: dict):
        """Broadcast state to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(state)
        # Send to all clients, removing dead connections
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except:
                dead_clients.add(client)
        
        self.clients -= dead_clients
    
    def update_state(self, state: dict):
        """Update the current state (called from main thread)"""
        self.current_state = state
        
        if self.loop and self.running:
            # Schedule broadcast on the event loop
            asyncio.run_coroutine_threadsafe(
                self.broadcast(state),
                self.loop
            )
    
    async def _run_server(self):
        """Run the WebSocket server"""
        self.server = await websockets.serve(
            self.handler,
            "0.0.0.0",
            self.port
        )
        print(f"🌐 WebSocket server started on port {self.port}")
        
        # Get local IP for display
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"   Public viewer URL: http://{local_ip}:8080")
        except:
            print(f"   Public viewer: connect to port {self.port}")
        
        await self.server.wait_closed()
    
    def _thread_main(self):
        """Main function for the WebSocket thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.running = True
        
        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as e:
            print(f"WebSocket server error: {e}")
        finally:
            self.running = False
            self.loop.close()
    
    def start(self):
        """Start the WebSocket server in a background thread"""
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()


# =============================================================================
# OSC HANDLER
# =============================================================================

class OSCHandler:
    """Handles incoming OSC messages"""
    
    def __init__(self, manager: TrackedPersonManager, database: TrackingDatabase = None):
        self.manager = manager
        self.database = database
        self.last_count = 0
        self.message_count = 0
        self.last_debug_time = time.time()
    
    def handle_person(self, address: str, *args):
        """Handle /tracker/person/<id> messages"""
        try:
            # Extract track_id from address
            parts = address.split('/')
            track_id = int(parts[-1])
            
            if len(args) >= 2:
                x, z = float(args[0]), float(args[1])
                self.manager.update_person(track_id, x, z)
                
                # Record to database
                if self.database:
                    self.database.record_position(track_id, x, z)
                
                # Debug output every 2 seconds
                self.message_count += 1
                now = time.time()
                if now - self.last_debug_time > 2.0:
                    print(f"📥 OSC: {self.message_count} messages, latest: person {track_id} at ({x:.0f}, {z:.0f})")
                    self.last_debug_time = now
                    self.message_count = 0
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
        
        # For behavior system integration
        self.follow_target = None
        self.follow_smoothing = 0.05
        self.gesture_target = None
    
    def _random_point(self) -> np.ndarray:
        return np.array([
            random.uniform(self.wander_box['min_x'], self.wander_box['max_x']),
            random.uniform(self.wander_box['min_y'], self.wander_box['max_y']),
            random.uniform(self.wander_box['min_z'], self.wander_box['max_z']),
        ])
    
    def update_wander_box(self, new_box: dict):
        """Update wander box (called by behavior system)"""
        self.wander_box = new_box
    
    def set_follow_target(self, target: np.ndarray, smoothing: float = 0.05):
        """Set a target to follow (from behavior system)"""
        self.follow_target = target
        self.follow_smoothing = smoothing
    
    def clear_follow_target(self):
        """Clear follow target, return to wandering"""
        self.follow_target = None
    
    def set_gesture_target(self, target: np.ndarray):
        """Set a gesture target (overrides other movement)"""
        self.gesture_target = target
    
    def clear_gesture_target(self):
        """Clear gesture target"""
        self.gesture_target = None
    
    def update(self, dt: float):
        if not self.enabled:
            return
        
        # Gesture target takes priority
        if self.gesture_target is not None:
            self.light.target_position = self.gesture_target.copy()
            return
        
        # Following takes priority over wandering
        if self.follow_target is not None:
            # Smooth follow - interpolate toward target
            current = self.light.target_position
            diff = self.follow_target - current
            
            # Apply smoothing (lower = smoother)
            smooth_factor = 1.0 - math.pow(1.0 - self.follow_smoothing, dt * 60)
            self.light.target_position = current + diff * smooth_factor
            
            # Clamp to wander box (keep within bounds)
            self.light.target_position[0] = np.clip(
                self.light.target_position[0],
                self.wander_box['min_x'], self.wander_box['max_x']
            )
            self.light.target_position[1] = np.clip(
                self.light.target_position[1],
                self.wander_box['min_y'], self.wander_box['max_y']
            )
            self.light.target_position[2] = np.clip(
                self.light.target_position[2],
                self.wander_box['min_z'], self.wander_box['max_z']
            )
            return
        
        # Default: wander randomly
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


def draw_floor(y_level, color, z_max=None):
    """Draw a floor plane. z_max limits depth (defaults to full size)"""
    glColor4f(*color)
    # Floor extends from X=-100 to X=400, Z=-200 to z_max
    z_back = z_max if z_max is not None else 400
    glBegin(GL_QUADS)
    glVertex3f(-100, y_level, -200)
    glVertex3f(400, y_level, -200)
    glVertex3f(400, y_level, z_back)
    glVertex3f(-100, y_level, z_back)
    glEnd()


def draw_text(x, y, text, font, color=(255, 255, 255)):
    """Draw text on screen"""
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)


# =============================================================================
# CALIBRATION MARKER RENDERING
# =============================================================================

def load_marker_textures() -> Dict[int, int]:
    """Load marker PNG files as OpenGL textures"""
    textures = {}
    
    for marker_id in MARKER_POSITIONS.keys():
        image_path = MARKER_IMAGE_PATH.format(marker_id)
        if os.path.exists(image_path):
            try:
                surface = pygame.image.load(image_path)
                texture_data = pygame.image.tostring(surface, "RGBA", True)
                width, height = surface.get_size()
                
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                            GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                
                textures[marker_id] = texture_id
                print(f"Loaded marker {marker_id} texture")
            except Exception as e:
                print(f"Failed to load marker {marker_id}: {e}")
        else:
            print(f"Marker image not found: {image_path}")
    
    return textures


def draw_marker(marker_id: int, position: Tuple[float, float, float], size: float,
                texture_id: Optional[int], vertical: bool = False):
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
        # Vertical marker: stands upright, facing outward toward street
        glTranslatef(0, 0, 0.5)
    else:
        # Horizontal marker: lies flat on floor, facing up
        glTranslatef(0, 0.5, 0)
        glRotatef(-90, 1, 0, 0)
    
    if texture_id is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1, 1, 1, 1)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-half, -half, 0)
        glTexCoord2f(1, 0); glVertex3f(half, -half, 0)
        glTexCoord2f(1, 1); glVertex3f(half, half, 0)
        glTexCoord2f(0, 1); glVertex3f(-half, half, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
    else:
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
    
    # Draw marker ID indicator
    glPushMatrix()
    if vertical:
        glTranslatef(x, y + half + 5, z)
    else:
        glTranslatef(x, y + 5, z)
    
    glColor4f(1, 1, 0, 1)  # Yellow
    quadric = gluNewQuadric()
    gluSphere(quadric, 2, 8, 8)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


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
    
    # Load marker textures
    marker_textures = load_marker_textures()
    show_markers = SHOW_MARKERS
    
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
    
    # Tracking database
    tracking_db = TrackingDatabase("tracking_history.db")
    print(f"💾 Tracking database: tracking_history.db")
    
    # Behavior system with default personality
    meta_params = MetaParameters()
    behavior = BehaviorSystem(meta=meta_params, database=tracking_db)
    print(f"🧠 Behavior system initialized")
    
    # Connect tracked manager callbacks to behavior system
    tracked_manager.on_person_entered = behavior.on_person_entered
    tracked_manager.on_person_left = behavior.on_person_left
    tracked_manager.on_position_updated = behavior.update_person_position
    
    # For periodic stats refresh
    last_stats_update = time.time()
    db_stats = {'people_last_minute': 0, 'avg_speed': 0, 'flow_left_to_right': 0, 
                'flow_right_to_left': 0, 'active_events': 0, 'passive_events': 0}
    
    # OSC setup
    osc_handler = OSCHandler(tracked_manager, tracking_db)
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/tracker/person/*", osc_handler.handle_person)
    osc_dispatcher.map("/tracker/count", osc_handler.handle_count)
    
    # Create OSC server with SO_REUSEADDR for reliability
    osc_server_instance = osc_server.ThreadingOSCUDPServer(
        (OSC_IP, OSC_PORT), osc_dispatcher
    )
    # Allow socket reuse (helps when restarting quickly)
    osc_server_instance.socket.setsockopt(
        socket.SOL_SOCKET, 
        socket.SO_REUSEADDR, 
        1
    )
    osc_thread = threading.Thread(target=osc_server_instance.serve_forever, daemon=True)
    osc_thread.start()
    print(f"📡 OSC server listening on {OSC_IP}:{OSC_PORT}")
    
    # WebSocket broadcaster for public viewer
    ws_broadcaster = None
    if WEBSOCKET_AVAILABLE and WEBSOCKET_ENABLED:
        ws_broadcaster = WebSocketBroadcaster(port=WEBSOCKET_PORT)
        ws_broadcaster.start()
    
    # Create sliders
    slider_x = view_width + 20
    slider_w = gui_width - 40
    slider_h = 12
    
    # Calibration sliders (top section)
    sliders = {
        # Offset sliders
        'offset_x': Slider(slider_x, 700, slider_w, slider_h, -200, 200, 0, "Offset X"),
        'offset_z': Slider(slider_x, 660, slider_w, slider_h, 0, 500, 250, "Offset Z"),
        # Scale sliders
        'scale_x': Slider(slider_x, 610, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale X", "{:.2f}"),
        'scale_z': Slider(slider_x, 570, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale Z", "{:.2f}"),
    }
    
    # Personality sliders (middle section)
    personality_sliders = {
        'responsiveness': Slider(slider_x, 500, slider_w, slider_h, 0, 1, 0.5, "Responsiveness", "{:.2f}"),
        'energy': Slider(slider_x, 460, slider_w, slider_h, 0, 1, 0.5, "Energy", "{:.2f}"),
        'attention_span': Slider(slider_x, 420, slider_w, slider_h, 0, 1, 0.5, "Attention", "{:.2f}"),
        'sociability': Slider(slider_x, 380, slider_w, slider_h, 0, 1, 0.5, "Sociability", "{:.2f}"),
        'exploration': Slider(slider_x, 340, slider_w, slider_h, 0, 1, 0.5, "Exploration", "{:.2f}"),
        'memory': Slider(slider_x, 300, slider_w, slider_h, 0, 1, 0.5, "Memory", "{:.2f}"),
    }
    
    # Global multiplier sliders (lower section)
    global_sliders = {
        'brightness_global': Slider(slider_x, 240, slider_w, slider_h, 0.2, 2.0, 1.0, "Brightness ×", "{:.2f}"),
        'speed_global': Slider(slider_x, 200, slider_w, slider_h, 0.2, 2.0, 1.0, "Speed ×", "{:.2f}"),
        'pulse_global': Slider(slider_x, 160, slider_w, slider_h, 0.3, 3.0, 1.0, "Pulse ×", "{:.2f}"),
    }
    
    # Combine all sliders
    all_sliders = {**sliders, **personality_sliders, **global_sliders}
    
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
    
    # Current preset name
    current_preset = "default"
    preset_names = list(PRESETS.keys())

    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            # Check all sliders
            for name, slider in all_sliders.items():
                if slider.handle_event(event, display[1]):
                    slider_active = True
                    # Update calibration values
                    if name in ('offset_x', 'offset_z', 'scale_x', 'scale_z'):
                        tracked_manager.offset_x = sliders['offset_x'].value
                        tracked_manager.offset_z = sliders['offset_z'].value
                        tracked_manager.scale_x = sliders['scale_x'].value
                        tracked_manager.scale_z = sliders['scale_z'].value
                    # Update personality values
                    elif name in personality_sliders:
                        setattr(meta_params, name, slider.value)
                    # Update global multipliers
                    elif name in global_sliders:
                        setattr(meta_params, name, slider.value)
            
            if event.type == MOUSEBUTTONUP:
                slider_active = False
            
            if event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    running = False
                elif event.key == K_SPACE:
                    wander.enabled = not wander.enabled
                elif event.key == K_m:
                    show_markers = not show_markers
                    print(f"Markers {'visible' if show_markers else 'hidden'}")
                elif event.key == K_p:
                    # Cycle through presets
                    idx = preset_names.index(current_preset)
                    idx = (idx + 1) % len(preset_names)
                    current_preset = preset_names[idx]
                    meta_params = load_preset(current_preset)
                    behavior.meta = meta_params
                    # Update sliders to match preset
                    for name, slider in personality_sliders.items():
                        slider.value = getattr(meta_params, name)
                    for name, slider in global_sliders.items():
                        slider.value = getattr(meta_params, name)
                    print(f"🎭 Preset: {current_preset}")
            
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
        
        # Cleanup stale tracked people
        tracked_manager.cleanup_stale()
        
        # Get zone counts
        active_count = tracked_manager.count_active()
        passive_count = tracked_manager.count_passive()
        
        # Get flow balance from database stats
        ltr = db_stats.get('flow_left_to_right', 0)
        rtl = db_stats.get('flow_right_to_left', 0)
        total_flow = ltr + rtl
        flow_balance = (ltr - rtl) / total_flow if total_flow > 0 else 0.0
        
        # Calculate passive rate (people per minute)
        passive_rate = db_stats.get('passive_events', 0) / 60.0  # Rough estimate
        
        # Update behavior system
        current_pos = tuple(light.position)
        behavior_params = behavior.update(
            dt=dt,
            active_count=active_count,
            passive_count=passive_count,
            current_pos=current_pos,
            passive_rate=passive_rate,
            flow_balance=flow_balance
        )
        
        # Apply behavior parameters to light
        light.brightness_min = int(behavior_params.get('brightness_min', 5))
        light.brightness_max = int(behavior_params.get('brightness_max', 30))
        light.pulse_speed = behavior_params.get('pulse_speed', 2000)
        light.move_speed = behavior_params.get('move_speed', 50)
        light.falloff_radius = behavior_params.get('falloff_radius', 50)
        
        # Update wander behavior based on behavior system
        wander.update_wander_box(behavior.get_wander_box())
        wander.wander_interval = behavior_params.get('wander_interval', 3.0)
        
        # Handle follow target from behavior system
        if behavior.should_wander():
            wander.clear_follow_target()
        else:
            follow_target = behavior.get_follow_target(active_count)
            if follow_target is not None:
                # Map person position to light position (constrain to wander box)
                target_y = np.clip(follow_target[1] + 120, 0, 150)  # Offset upward
                target_z = np.clip(0, -32, 28)  # Keep in wander box Z
                light_target = np.array([follow_target[0], target_y, target_z])
                smoothing = behavior_params.get('follow_smoothing', 0.05)
                wander.set_follow_target(light_target, smoothing)
            else:
                wander.clear_follow_target()
        
        # Handle gesture target
        gesture_target = behavior.get_gesture_target()
        if gesture_target is not None:
            wander.set_gesture_target(gesture_target)
        else:
            wander.clear_gesture_target()
        
        # Update wander and light
        wander.update(dt)
        light.update(dt)
        panel_system.calculate_brightness(light)
        
        # Broadcast state to WebSocket clients
        if ws_broadcaster:
            state = {
                'light': {
                    'x': light.x,
                    'y': light.y,
                    'z': light.z,
                    'brightness': light.brightness,
                    'falloff_radius': light.falloff_radius
                },
                'panels': panel_system.get_dmx_values()[:12],  # First 12 values (panel brightnesses)
                'people': [
                    {'id': pid, 'x': p['x'], 'y': p['y'], 'z': p['z']}
                    for pid, p in tracked_people.items()
                ],
                'mode': behavior_system.mode.name if behavior_system else 'UNKNOWN',
                'gesture': behavior_system.current_gesture.name if behavior_system and behavior_system.current_gesture else None,
                'status': behavior_text
            }
            ws_broadcaster.update_state(state)
        
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
        
        # Draw floor - only storefront level, stopping at near edge of active zone
        active_zone_near = TRACKZONE['offset_z']  # Near edge at Z=78
        draw_floor(0, (0.25, 0.25, 0.3, 0.5), z_max=active_zone_near)
        
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
        
        # Draw wander box (from behavior system)
        wb = behavior.get_wander_box()
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
                draw_marker(marker_id, pos, MARKER_SIZE, tex_id, vertical=is_vertical)
        
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
        draw_text(view_width + 20, display[1] - 30, "LIGHT BEHAVIOR CONTROL", font)
        draw_text(view_width + 20, display[1] - 50, "─" * 24, font)
        
        # Section labels
        draw_text(view_width + 20, 740, "Calibration:", font_small, (150, 150, 200))
        draw_text(view_width + 20, 540, "Personality:", font_small, (150, 200, 150))
        draw_text(view_width + 20, 275, "Global Multipliers:", font_small, (200, 150, 150))
        
        # Draw all sliders
        for slider in all_sliders.values():
            slider.draw(font_small)
        
        # Update database stats periodically (every 2 seconds)
        if time.time() - last_stats_update > 2.0:
            db_stats = tracking_db.get_current_stats()
            last_stats_update = time.time()
        
        # Behavior status section
        behavior_status = behavior.get_status()
        draw_text(view_width + 20, 125, "─" * 20, font_small)
        draw_text(view_width + 20, 110, "BEHAVIOR STATUS:", font_small, (255, 200, 100))
        
        # Mode and preset
        mode_colors = {
            'idle': (100, 100, 200),
            'engaged': (100, 200, 100),
            'crowd': (200, 200, 100),
            'flow': (200, 150, 100),
        }
        mode_color = mode_colors.get(behavior_status['mode'], (200, 200, 200))
        draw_text(view_width + 20, 92, f"  Mode: {behavior_status['mode'].upper()}", font_small, mode_color)
        draw_text(view_width + 20, 76, f"  Preset: {current_preset}", font_small)
        draw_text(view_width + 20, 60, f"  Time: {behavior_status['time_of_day']}", font_small)
        
        # Status text (for public display)
        if behavior_status['status_text']:
            draw_text(view_width + 20, 42, f"  \"{behavior_status['status_text']}\"", font_small, (200, 200, 255))
        
        # Controls help at bottom
        draw_text(view_width + 20, 20, "SPACE=wander  M=markers  P=preset  Q=quit", font_small, (120, 120, 120))
        
        # Marker legend in 3D view area
        if show_markers:
            draw_text(10, 140, "CALIBRATION MARKERS:", font_small, (255, 255, 0))
            y_offset = 120
            for marker_id, marker_data in MARKER_POSITIONS.items():
                pos = marker_data['pos']
                desc = marker_data['desc']
                draw_text(10, y_offset, f"  [{marker_id}] ({pos[0]}, {pos[1]}, {pos[2]}) - {desc}", font_small)
                y_offset -= 16
        
        # HUD text in 3D view
        dmx_vals = panel_system.get_dmx_values()
        behavior_status = behavior.get_status()
        
        # Main HUD (top left)
        info_lines = [
            f"Light: ({light.position[0]:.0f}, {light.position[1]:.0f}, {light.position[2]:.0f}) cm",
            f"DMX: {dmx_vals}",
            f"Mode: {behavior_status['mode'].upper()}  Active: {active_count}  Passive: {passive_count}",
        ]
        
        for i, line in enumerate(info_lines):
            draw_text(10, display[1] - 20 - i * 20, line, font_small)
        
        # Status text overlay (bottom center of 3D view)
        if behavior_status['status_text'] and meta_params.status_text_enabled:
            status = behavior_status['status_text']
            # Draw with a background
            glColor4f(0.0, 0.0, 0.0, 0.6)
            status_x = view_width // 2 - 100
            status_y = 30
            glBegin(GL_QUADS)
            glVertex2f(status_x - 10, status_y - 5)
            glVertex2f(status_x + 220, status_y - 5)
            glVertex2f(status_x + 220, status_y + 25)
            glVertex2f(status_x - 10, status_y + 25)
            glEnd()
            draw_text(status_x, status_y, f'"{status}"', font, (255, 255, 200))
        
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Cleanup
    print("Shutting down...")
    osc_server_instance.shutdown()
    if artnet:
        artnet.stop()
    if ws_broadcaster:
        ws_broadcaster.stop()
        print("🌐 WebSocket server stopped.")
    tracking_db.close()
    print("📊 Tracking database saved.")
    pygame.quit()


if __name__ == "__main__":
    main()
