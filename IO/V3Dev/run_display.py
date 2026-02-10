#!/usr/bin/env python3
"""
Light Controller V3 - Full 3D Display
=====================================
Production-ready light controller with 3D OpenGL display matching the original.

This version uses the display/ module for proper 3D visualization.

Usage:
    cd IO && python V3Dev/run_display.py              # Normal operation
    cd IO && python V3Dev/run_display.py --no-lock    # Skip single-instance lock
    cd IO && python V3Dev/run_display.py --windowed   # Start in windowed mode
"""

import sys
import os
import time
import math
import argparse
import signal
import atexit
import socket
import select
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Add IO directory for V3Dev imports (run from IO directory)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_io_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _io_dir)

# Also add V3Dev for direct display imports
sys.path.insert(0, _script_dir)

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    import pygame
    from pygame.locals import (
        QUIT, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL,
        K_q, K_ESCAPE, K_SPACE, K_m, K_l, K_c, K_p, K_t, K_f, K_r, K_d, K_HOME,
        K_LEFT, K_RIGHT, K_UP, K_DOWN, K_w, K_s,
        DOUBLEBUF, OPENGL, NOFRAME, RESIZABLE, KMOD_SHIFT
    )
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("❌ pygame not installed. Run: pip3 install pygame")
    sys.exit(1)

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import numpy as np
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("❌ PyOpenGL not installed. Run: pip3 install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("⚠️  python-osc not installed. OSC disabled.")

try:
    from stupidArtnet import StupidArtnet
    ARTNET_AVAILABLE = True
except ImportError:
    ARTNET_AVAILABLE = False
    print("⚠️  stupidArtnet not installed. Art-Net disabled.")

# =============================================================================
# V3 MODULE IMPORTS
# =============================================================================

from V3Dev.config.zones import TRACKZONE as V3_TRACKZONE, PASSIVE_TRACKZONE as V3_PASSIVE_TRACKZONE
from V3Dev.tracking import TrackedPersonManager, TrackedPerson
try:
    from V3Dev.tracking import OSCHandler
except ImportError:
    OSCHandler = None
from V3Dev.behavior import BehaviorSystem, MetaParameters

# Display module (matching original lightController_osc.py)
from display import (
    init_display, create_fonts, setup_3d_projection, setup_2d_projection,
    restore_3d_projection, CameraController, clear_frame, swap_buffers,
    draw_box_wireframe, draw_panel, draw_sphere, draw_sphere_wireframe,
    draw_floor, draw_tracked_person, draw_text_2d, draw_text_3d_billboard,
    draw_realtime_trends, draw_trends_visualization,
    Slider, Checkbox, SceneRenderer
)
from display.sliders import create_calibration_sliders, create_personality_sliders, create_global_sliders, create_checkboxes
from display.scene import TRACKZONE, PASSIVE_TRACKZONE, WANDER_BOX, CAMERA_POSITIONS, MARKER_POSITIONS

# =============================================================================
# LOGGING SETUP  
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('v3_light_controller.log', mode='a')
    ]
)
logger = logging.getLogger('LightControllerV3')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Network
OSC_IP = "0.0.0.0"
OSC_PORT = 7000  # Match pedestrian_simulator.py
TARGET_IP = "192.168.1.20"
UNIVERSE = 0
FPS = 30

# Panel dimensions (cm)
PANEL_SIZE = 60

# Health monitoring
HEALTH_LOG_INTERVAL = 300  # 5 minutes

# Display flags
SHOW_MARKERS = True
SHOW_LABELS = True

# =============================================================================
# PANEL SYSTEM
# =============================================================================

class PanelSystem:
    """
    Manages the physical light panels (4 units × 3 panels each).
    
    Coordinate system (matches original lightController_osc.py):
    - X = 0 at back right corner of Unit 0, negative X goes left
    - Unit centers: Unit 0 at X=-30, Unit 1 at X=-110, Unit 2 at X=-190, Unit 3 at X=-270
    - Y = 0 at floor level, positive Y is up
    - Z = 0 at panel surface, positive Z is toward viewer (into room)
    """
    
    # Panel local positions relative to unit center (Y, Z) in cm
    PANEL_LOCAL_POSITIONS = {
        1: (90, 0),    # Top panel - vertical
        2: (30, 12),   # Bottom front - angled forward
        3: (30, -12),  # Bottom back - angled backward
    }
    
    # Panel angles (degrees from vertical)
    PANEL_ANGLES = {
        1: 0,       # Top: vertical
        2: 22.5,    # Bottom front: tilted forward
        3: -22.5,   # Bottom back: tilted backward
    }
    
    UNIT_SPACING = 80  # cm between unit centers
    PANEL_SIZE = 60    # cm
    
    def __init__(self):
        self.panels = {}
        self._build_panels()
    
    def _build_panels(self):
        """Build panel positions matching original exactly."""
        # Unit 0 is rightmost, with back right corner at X=0
        # Unit centers: X=-30, -110, -190, -270
        for unit in range(4):
            unit_x = -(unit * self.UNIT_SPACING + 30)
            
            for panel_num in range(1, 4):
                local_y, local_z = self.PANEL_LOCAL_POSITIONS[panel_num]
                center = np.array([float(unit_x), float(local_y), float(local_z)])
                
                self.panels[(unit, panel_num)] = {
                    'center': center,
                    'angle': self.PANEL_ANGLES[panel_num],
                    'brightness': 0.0,  # 0.0 - 1.0 range
                    'dmx_value': 0,
                }
    
    def calculate_brightness(self, light):
        """Calculate panel brightnesses based on light position."""
        intensity = light.get_brightness()  # 0.0 - 1.0
        
        for key, panel in self.panels.items():
            diff = panel['center'] - np.array(light.position)
            distance = np.linalg.norm(diff)
            
            if light.falloff_radius > 0:
                falloff = max(0, 1.0 - distance / light.falloff_radius)
            else:
                falloff = 1.0
            
            # Final brightness as 0.0-1.0
            panel['brightness'] = falloff * intensity
            
            # DMX value (1-50 range from original)
            dmx_range = light.brightness_max - light.brightness_min
            panel['dmx_value'] = int(light.brightness_min + panel['brightness'] * dmx_range)
            panel['dmx_value'] = max(1, min(50, panel['dmx_value']))
    
    def get_dmx_values(self) -> list:
        """Get DMX values for all panels (12 channels)."""
        # Unit 0 = DMX CH1-3, Unit 1 = CH4-6, Unit 2 = CH7-9, Unit 3 = CH10-12
        return [self.panels[(u, p)]['dmx_value'] for u in range(4) for p in range(1, 4)]
    
    def get_unit_centers(self) -> dict:
        """Get center position of each unit (for labeling)."""
        centers = {}
        for unit in range(4):
            unit_x = -(unit * self.UNIT_SPACING + 30)
            # Unit center is at Y=60 (midpoint of panels), Z=0
            centers[unit] = np.array([float(unit_x), 60.0, 0.0])
        return centers


# =============================================================================
# POINT LIGHT
# =============================================================================

class PointLight:
    """
    Virtual light source.
    Matches original lightController_osc.py behavior exactly.
    
    The light moves in front of the panels within the WANDER_BOX bounds.
    WANDER_BOX coordinates:
    - X: -280 to -20 (spanning all 4 units)
    - Y: 0 to 150 (floor to above eye level)  
    - Z: -28 to 32 (in front of panels, Z=0 is panel surface)
    """
    
    def __init__(self):
        # Start in center of wander box (matches original default)
        self.position = np.array([-160.0, 60.0, -10.0])
        self.target_position = np.array([-160.0, 60.0, -10.0])
        
        self.brightness_min = 5
        self.brightness_max = 40
        self.pulse_speed = 2000.0
        self.falloff_radius = 50.0
        self.move_speed = 50.0
        self.pulse_phase = 0.0
    
    def get_brightness(self) -> float:
        """Get current brightness as 0.0-1.0 value."""
        return (math.sin(self.pulse_phase) + 1) / 2
    
    def update(self, dt: float):
        """Update light position and pulse phase."""
        # Update pulse phase
        self.pulse_phase += (2 * math.pi * dt * 1000) / self.pulse_speed
        
        # Move toward target position
        diff = self.target_position - self.position
        dist = np.linalg.norm(diff)
        if dist > 0.1:
            move = min(self.move_speed * dt, dist)
            self.position = self.position + (diff / dist) * move
# WANDER BEHAVIOR
# =============================================================================

class WanderBehavior:
    """
    Automatic light movement within bounds.
    Matches original lightController_osc.py behavior exactly.
    """
    
    def __init__(self, light: PointLight, wander_box: dict):
        self.light = light
        self.wander_box = dict(wander_box)
        self.wander_target = self._random_point()
        self.wander_timer = 0.0
        self.wander_interval = 3.0
        self.enabled = True
        
        # For behavior system integration
        self.follow_target = None
        self.follow_smoothing = 0.05
        self.follow_x_only = False
        self.gesture_target = None
    
    def _random_point(self) -> np.ndarray:
        """Generate a random point within the wander box."""
        import random
        return np.array([
            random.uniform(self.wander_box['min_x'], self.wander_box['max_x']),
            random.uniform(self.wander_box['min_y'], self.wander_box['max_y']),
            random.uniform(self.wander_box['min_z'], self.wander_box['max_z']),
        ])
    
    def update_wander_box(self, new_box: dict):
        """Update wander box (called by behavior system)."""
        self.wander_box = dict(new_box)
    
    def set_follow_target(self, target, smoothing: float = 0.05, x_only: bool = False):
        """Set a target to follow (from behavior system)."""
        self.follow_target = np.array(target) if target is not None else None
        self.follow_smoothing = smoothing
        self.follow_x_only = x_only
    
    def clear_follow_target(self):
        """Clear follow target, return to wandering."""
        self.follow_target = None
        self.follow_x_only = False
    
    def set_gesture_target(self, target):
        """Set a gesture target (overrides other movement)."""
        if target is not None:
            self.gesture_target = np.array([target[0], self.light.position[1], target[1]])
        else:
            self.gesture_target = None
    
    def clear_gesture_target(self):
        """Clear gesture target."""
        self.gesture_target = None
    
    def update(self, dt: float):
        if not self.enabled:
            return
        
        # Gesture target takes priority
        if self.gesture_target is not None:
            self.light.target_position = list(self.gesture_target)
            return
        
        # Always clamp wander target to current box bounds (box may have moved)
        self.wander_target[0] = np.clip(self.wander_target[0], self.wander_box['min_x'], self.wander_box['max_x'])
        self.wander_target[1] = np.clip(self.wander_target[1], self.wander_box['min_y'], self.wander_box['max_y'])
        self.wander_target[2] = np.clip(self.wander_target[2], self.wander_box['min_z'], self.wander_box['max_z'])
        
        # Update wander timer and check if we need a new target
        self.wander_timer += dt
        light_pos = np.array(self.light.position)
        dist = np.linalg.norm(light_pos - self.wander_target)
        
        # Use longer interval in engaged mode (small box = frequent clamping)
        min_interval = max(3.0, self.wander_interval)
        
        # Pick new target when we reach current one or timer expires
        if dist < 10 or self.wander_timer > min_interval:
            self.wander_target = self._random_point()
            self.wander_timer = 0
            # Randomize around the base interval
            import random
            self.wander_interval = random.uniform(min_interval, min_interval + 3)
        
        # Smoothly move toward wander target (already clamped to box)
        current = np.array(self.light.target_position)
        target = self.wander_target
        
        # Smooth movement toward target - lower = slower, smoother
        diff = target - current
        smooth = 0.03  # Gentle, slow movement (matches original)
        new_target = current + diff * smooth
        self.light.target_position = list(new_target)


# =============================================================================
# ANIMATED WANDER BOX
# =============================================================================

class AnimatedWanderBox:
    """
    Animated wander box that smoothly transitions based on tracking state.
    Matches the original light_behavior.py behavior exactly.
    
    The box contracts around tracked people when engaged,
    and expands back to base when idle.
    """
    
    def __init__(self, base_box: dict):
        self.base_box = dict(base_box)
        self.current_box = dict(base_box)
        self.target_box = dict(base_box)
        self.animated_box = dict(base_box)
        
        # Animation settings
        self.lerp_speed = 3.0  # Higher = faster, more responsive
        
        # Engaged box padding (how tight around people)
        self.engaged_padding_x = 15   # ±15cm in X - very tight
        self.engaged_padding_y = 35   # ±Y padding
        self.engaged_padding_z = 15   # ±Z padding
        self.engaged_y_offset = 100   # Offset upward from person height
        
        # Active zone people for engaged box calculation
        self.active_positions: list = []
        
    def update(self, dt: float, mode: str, active_positions: list):
        """
        Update the animated wander box.
        
        Args:
            dt: Delta time in seconds
            mode: Current behavior mode ('idle', 'engaged', 'crowd', etc.)
            active_positions: List of (x, y, z) positions of active zone people
        """
        self.active_positions = active_positions
        
        # Determine target box based on mode
        if mode in ('engaged', 'crowd') and active_positions:
            self.target_box = self._calculate_engaged_box()
        else:
            # Return to base box
            self.target_box = dict(self.current_box if hasattr(self, 'current_box') else self.base_box)
        
        # Lerp each dimension toward target
        lerp_factor = 1.0 - math.exp(-self.lerp_speed * dt)
        
        for key in self.animated_box:
            current = self.animated_box[key]
            target = self.target_box[key]
            self.animated_box[key] = current + (target - current) * lerp_factor
    
    def _calculate_engaged_box(self) -> dict:
        """
        Calculate wander box based on active zone people.
        
        Multi-person strategy:
        - 1 person: Tight box centered on them
        - 2 people: Box covers both, weighted toward longest-present
        - 3+ people: Wider box to roam between them
        """
        if not self.active_positions:
            return dict(self.base_box)
        
        count = len(self.active_positions)
        
        # Extract X coordinates
        all_x = [pos[0] for pos in self.active_positions]
        
        if count == 1:
            # Single person - very tight focus
            person_x = all_x[0]
            padding_x = self.engaged_padding_x
        elif count == 2:
            # Two people - cover both with some weight
            spread_x = max(all_x) - min(all_x)
            person_x = sum(all_x) / len(all_x)  # Centroid
            padding_x = self.engaged_padding_x + spread_x * 0.4 + 10
        else:
            # 3+ people (crowd) - wider roaming box
            person_x = sum(all_x) / len(all_x)  # Centroid
            spread_x = max(all_x) - min(all_x)
            padding_x = self.engaged_padding_x + spread_x * 0.5 + count * 8
        
        # Create engaged box - X follows people, Y/Z stay at base ranges
        engaged_box = {
            'min_x': max(self.base_box['min_x'], person_x - padding_x),
            'max_x': min(self.base_box['max_x'], person_x + padding_x),
            'min_y': self.base_box['min_y'],
            'max_y': self.base_box['max_y'],
            'min_z': self.base_box['min_z'],
            'max_z': self.base_box['max_z'],
        }
        
        return engaged_box
    
    def get_box(self) -> dict:
        """Get the current animated wander box."""
        return dict(self.animated_box)
    
    def update_base_box(self, new_base: dict):
        """Update the base box (e.g., from sliders)."""
        self.base_box = dict(new_base)
        self.current_box = dict(new_base)


# =============================================================================
# SINGLE INSTANCE LOCK
# =============================================================================

_lock_fd = None
_lock_file = "/tmp/light_controller_v3.lock"

def acquire_single_instance_lock() -> bool:
    global _lock_fd
    import fcntl
    try:
        _lock_fd = open(_lock_file, 'w')
        fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        logger.info(f"Single instance lock acquired (PID {os.getpid()})")
        return True
    except (IOError, OSError) as e:
        logger.error(f"Another instance is already running: {e}")
        return False

def release_single_instance_lock():
    global _lock_fd
    if _lock_fd:
        import fcntl
        try:
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
            _lock_fd.close()
            os.remove(_lock_file)
            logger.info("Single instance lock released")
        except Exception:
            pass
        _lock_fd = None


# =============================================================================
# SLIDER SETTINGS
# =============================================================================

import json

SLIDER_SETTINGS_FILE = "slider_settings_v3.json"

def load_slider_settings() -> Optional[dict]:
    try:
        if os.path.exists(SLIDER_SETTINGS_FILE):
            with open(SLIDER_SETTINGS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load slider settings: {e}")
    return None

def save_slider_settings(sliders: dict, checkboxes: dict):
    try:
        settings_data = {
            'sliders': {name: s.value for name, s in sliders.items()},
            'checkboxes': {name: c.checked for name, c in checkboxes.items()},
            'saved_at': datetime.now().isoformat()
        }
        with open(SLIDER_SETTINGS_FILE, 'w') as f:
            json.dump(settings_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save slider settings: {e}")

def apply_slider_settings(sliders: dict, settings_data: dict, checkboxes: dict):
    if 'sliders' in settings_data:
        for name, value in settings_data['sliders'].items():
            if name in sliders:
                sliders[name].value = value
    if 'checkboxes' in settings_data:
        for name, checked in settings_data['checkboxes'].items():
            if name in checkboxes:
                checkboxes[name].checked = checked


# =============================================================================
# PRESETS
# =============================================================================

PRESETS = {
    'default': MetaParameters(),
    'calm': MetaParameters(responsiveness=0.3, energy=0.4, exploration=0.3),
    'energetic': MetaParameters(responsiveness=0.9, energy=0.8, exploration=0.6),
    'mysterious': MetaParameters(responsiveness=0.4, energy=0.6, exploration=0.8),
}

def load_preset(name: str) -> MetaParameters:
    return PRESETS.get(name, MetaParameters())


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Light Controller V3 - Full 3D Display")
    parser.add_argument('--no-lock', action='store_true', help="Skip single-instance lock")
    parser.add_argument('--windowed', action='store_true', help="Start in windowed mode")
    args = parser.parse_args()
    
    # Single instance check
    if not args.no_lock:
        if not acquire_single_instance_lock():
            sys.exit(1)
        atexit.register(release_single_instance_lock)
    
    # Graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize display
    screen, display = init_display(fullscreen=not args.windowed)
    font, font_small, font_label = create_fonts()
    
    fullscreen_size = display
    windowed_size = (1920, 1080)
    is_fullscreen = not args.windowed
    
    # GUI layout
    gui_width = 280
    view_width = display[0] - gui_width
    
    # Camera controller
    camera = CameraController()
    
    # Scene renderer
    scene = SceneRenderer()
    
    # Create systems
    panel_system = PanelSystem()
    light = PointLight()
    wander = WanderBehavior(light, dict(WANDER_BOX))
    animated_box = AnimatedWanderBox(dict(WANDER_BOX))  # Animated wander box
    tracked_manager = TrackedPersonManager()
    
    # Behavior system
    meta_params = MetaParameters()
    behavior = BehaviorSystem(meta_params=meta_params)
    behavior.start()  # IMPORTANT: Start the behavior system!
    
    # Note: V3 BehaviorSystem uses different callback pattern
    # Tracked people updates will be handled in the main loop
    
    # OSC
    osc_server_instance = None
    if OSC_AVAILABLE and OSCHandler is not None:
        osc_handler = OSCHandler(tracked_manager)
        osc_disp = dispatcher.Dispatcher()
        osc_disp.map("/tracker/person/*", osc_handler.handle_person)
        osc_disp.map("/tracker/zone/*", osc_handler.handle_zone)
        osc_disp.map("/tracker/count", osc_handler.handle_count)
        
        osc_server_instance = osc_server.BlockingOSCUDPServer((OSC_IP, OSC_PORT), osc_disp)
        osc_server_instance.timeout = 0.001
        osc_server_instance.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info(f"OSC server listening on {OSC_IP}:{OSC_PORT}")
    
    # Art-Net
    artnet = None
    if ARTNET_AVAILABLE:
        try:
            artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            artnet.start()
            logger.info(f"Art-Net output to {TARGET_IP}")
        except Exception as e:
            logger.warning(f"Art-Net failed: {e}")
    
    # Create sliders
    slider_x = view_width + 30
    slider_w = 200
    slider_h = 20
    sliders = create_calibration_sliders(slider_x, display[1], slider_w, slider_h)
    personality_sliders = create_personality_sliders(slider_x, display[1], slider_w, slider_h)
    global_sliders = create_global_sliders(slider_x, display[1], slider_w, slider_h)
    all_sliders = {**sliders, **personality_sliders, **global_sliders}
    checkboxes = create_checkboxes(slider_x, display[1])
    
    # Load saved settings
    saved_settings = load_slider_settings()
    if saved_settings:
        apply_slider_settings(all_sliders, saved_settings, checkboxes)
        tracked_manager.offset_x = sliders['offset_x'].value
        tracked_manager.offset_z = sliders['offset_z'].value
        tracked_manager.scale_x = sliders['scale_x'].value
        tracked_manager.scale_z = sliders['scale_z'].value
        tracked_manager.invert_x = checkboxes['invert_x'].checked
        for name, slider in personality_sliders.items():
            setattr(meta_params, name, slider.value)
        for name, slider in global_sliders.items():
            setattr(meta_params, name, slider.value)
        logger.info(f"Restored slider settings")
    
    # State
    last_slider_save = time.time()
    sliders_dirty = False
    show_markers = SHOW_MARKERS
    show_labels = SHOW_LABELS
    show_trends = True
    current_preset = "default"
    preset_names = list(PRESETS.keys())
    
    clock = pygame.time.Clock()
    last_time = time.time()
    slider_active = False
    start_time = time.time()
    frame_count = 0
    last_health_log = time.time()
    last_logged_mode = None  # Track mode changes for logging
    
    # Print controls
    print("\n" + "="*60)
    print("V3 LIGHT CONTROLLER - Full 3D Display")
    print("="*60)
    print("Controls:")
    print("  L = Toggle labels    M = Toggle markers")
    print("  T = Toggle trends    P = Cycle presets")
    print("  F = Toggle fullscreen")
    print("  SPACE = Toggle wandering")
    print("  HOME = Reset camera")
    print("  Arrow keys = Move light (when wander disabled)")
    print("  Q/ESC = Quit")
    print("="*60 + "\n")
    
    running = True
    while running and not shutdown_requested:
        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            # Slider events
            for name, slider in all_sliders.items():
                if slider.handle_event(event, display[1]):
                    slider_active = True
                    sliders_dirty = True
                    if name in ('offset_x', 'offset_z', 'scale_x', 'scale_z'):
                        tracked_manager.offset_x = sliders['offset_x'].value
                        tracked_manager.offset_z = sliders['offset_z'].value
                        tracked_manager.scale_x = sliders['scale_x'].value
                        tracked_manager.scale_z = sliders['scale_z'].value
                    elif name in personality_sliders:
                        setattr(meta_params, name, slider.value)
                    elif name in global_sliders:
                        setattr(meta_params, name, slider.value)
            
            for name, checkbox in checkboxes.items():
                if checkbox.handle_event(event, display[1]):
                    sliders_dirty = True
                    if name == 'invert_x':
                        tracked_manager.invert_x = checkbox.checked
            
            if event.type == MOUSEBUTTONUP:
                slider_active = False
            
            # Camera events
            if event.type in (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL):
                if hasattr(event, 'pos') and event.pos[0] < view_width and not slider_active:
                    camera.handle_event(event)
            
            # Keyboard
            if event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    running = False
                elif event.key == K_SPACE:
                    wander.enabled = not wander.enabled
                    print(f"Wandering {'enabled' if wander.enabled else 'disabled'}")
                elif event.key == K_m:
                    show_markers = not show_markers
                elif event.key == K_l:
                    show_labels = not show_labels
                elif event.key == K_p:
                    idx = (preset_names.index(current_preset) + 1) % len(preset_names)
                    current_preset = preset_names[idx]
                    meta_params = load_preset(current_preset)
                    behavior.meta_params = meta_params
                    for name, slider in personality_sliders.items():
                        slider.value = getattr(meta_params, name, slider.value)
                    for name, slider in global_sliders.items():
                        slider.value = getattr(meta_params, name, slider.value)
                    print(f"Preset: {current_preset}")
                elif event.key == K_t:
                    show_trends = not show_trends
                elif event.key == K_f:
                    is_fullscreen = not is_fullscreen
                    pygame.display.quit()
                    pygame.display.init()
                    if is_fullscreen:
                        display = fullscreen_size
                        pygame.display.set_mode(display, DOUBLEBUF | OPENGL | NOFRAME)
                    else:
                        display = windowed_size
                        pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
                    glEnable(GL_DEPTH_TEST)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glClearColor(0.1, 0.1, 0.15, 1.0)
                    view_width = display[0] - gui_width
                elif event.key == K_HOME:
                    camera.reset()
        
        # Manual light control
        keys = pygame.key.get_pressed()
        if not wander.enabled:
            now = time.time()
            dt_keys = min(now - last_time, 0.1)
            move_speed = 100
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
        
        # Process OSC
        if osc_server_instance:
            for _ in range(100):
                ready, _, _ = select.select([osc_server_instance.socket], [], [], 0)
                if not ready:
                    break
                try:
                    osc_server_instance.handle_request()
                except Exception:
                    break
        
        # Update
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now
        
        tracked_manager.cleanup_stale()
        active_count = tracked_manager.count_active()
        passive_count = tracked_manager.count_passive()
        
        # V3 BehaviorSystem returns BehaviorOutput object
        behavior_output = behavior.update(
            dt=dt, active_count=active_count, passive_count=passive_count
        )
        
        # Debug: Log mode changes
        current_mode = behavior_output.mode.value if hasattr(behavior_output.mode, 'value') else str(behavior_output.mode)
        if current_mode != last_logged_mode:
            logger.info(f"🎯 MODE CHANGE: {last_logged_mode} → {current_mode} (active={active_count}, passive={passive_count})")
            last_logged_mode = current_mode
        
        # Apply behavior output multipliers to light
        light.brightness_min = 5
        light.brightness_max = int(30 * behavior_output.brightness_mult)
        light.pulse_speed = 2000
        light.move_speed = int(50 * behavior_output.move_speed_mult)
        light.falloff_radius = 80
        
        # Get active zone positions for animated wander box
        active_positions = []
        for person in tracked_manager.get_all():
            if hasattr(person, 'zone') and person.zone == 'active':
                pos = person.get_position() if hasattr(person, 'get_position') else person.position
                active_positions.append(tuple(pos))
        
        # Determine mode string for animated box
        mode_str = behavior_output.mode.value if hasattr(behavior_output.mode, 'value') else str(behavior_output.mode)
        
        # Update animated wander box (smooth transition to/from engaged box)
        animated_box.update(dt, mode_str, active_positions)
        
        # Get the animated box and apply X offset from behavior if needed
        wander_box = animated_box.get_box()
        if behavior_output.wander_x_offset != 0:
            # Shift X range based on flow alignment
            wander_box['min_x'] += behavior_output.wander_x_offset
            wander_box['max_x'] += behavior_output.wander_x_offset
        wander.update_wander_box(wander_box)
        
        # Handle gesture target
        if behavior_output.gesture_type is not None:
            wander.set_gesture_target((behavior_output.gesture_target_x, behavior_output.gesture_target_z))
        else:
            wander.clear_gesture_target()
        
        wander.update(dt)
        light.update(dt)
        panel_system.calculate_brightness(light)
        
        if artnet:
            try:
                artnet.set(panel_system.get_dmx_values())
            except Exception:
                pass
        
        if sliders_dirty and now - last_slider_save > 2.0:
            save_slider_settings(all_sliders, checkboxes)
            last_slider_save = now
            sliders_dirty = False
        
        # =====================================================================
        # RENDER
        # =====================================================================
        
        clear_frame()
        
        # 3D View
        setup_3d_projection(display, view_width)
        camera.apply_view_transform()
        
        # Floor
        draw_floor(0, (0.25, 0.25, 0.3, 0.5), z_max=TRACKZONE['offset_z'])
        
        # Origin and cameras
        scene.draw_origin_marker(font_label)
        if show_labels:
            draw_text_3d_billboard([0, 0, 0], "ORIGIN (0,0,0)", font_label, (255, 255, 0), offset_y=20)
        scene.draw_camera_markers(font_label)
        
        # Zones (includes wander box drawn in blue)
        scene.draw_zones(wander_box)
        if show_labels:
            scene.draw_zone_corner_labels(font_label)
        
        # Panels
        for (unit, panel_num), panel in panel_system.panels.items():
            draw_panel(panel['center'], panel['angle'], PANEL_SIZE, panel['brightness'])
        
        # Calibration Markers
        if show_markers:
            scene.draw_calibration_markers(font_label)
        
        # Light
        brightness = light.get_brightness()
        draw_sphere(light.position, 8 + brightness * 7, (1, 1, brightness, 1))
        draw_sphere_wireframe(light.position, light.falloff_radius, (1, 0.8, 0, 0.3), segments=24)
        
        # Tracked people
        for person in tracked_manager.get_all():
            pos = person.get_position()
            zone = tracked_manager.get_zone(pos[0], pos[2])
            draw_tracked_person((pos[0], pos[1], pos[2]), zone)
        
        # =====================================================================
        # HUD
        # =====================================================================
        
        setup_2d_projection(display)
        
        # GUI panel
        glColor4f(0.12, 0.12, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(view_width, 0)
        glVertex2f(display[0], 0)
        glVertex2f(display[0], display[1])
        glVertex2f(view_width, display[1])
        glEnd()
        
        glColor4f(0.4, 0.4, 0.5, 1.0)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(view_width, 0)
        glVertex2f(view_width, display[1])
        glEnd()
        
        # Title
        draw_text_2d(view_width + 20, display[1] - 30, "LIGHT CONTROLLER V3", font)
        draw_text_2d(view_width + 20, display[1] - 50, "─" * 24, font)
        
        # Section labels
        draw_text_2d(view_width + 20, display[1] - 70, "Calibration:", font_small, (150, 150, 200))
        draw_text_2d(view_width + 20, display[1] - 300, "Personality:", font_small, (150, 200, 150))
        draw_text_2d(view_width + 20, display[1] - 570, "Global Multipliers:", font_small, (200, 150, 150))
        
        # Sliders and checkboxes
        for slider in all_sliders.values():
            slider.draw(font_small)
        for checkbox in checkboxes.values():
            checkbox.draw(font_small)
        
        # Mode status - use V3 get_stats() API
        behavior_stats = behavior.get_stats()
        mode_name = behavior_stats.get('mode', 'idle')
        mode_colors = {
            'idle': (100, 100, 200), 'engaged': (100, 200, 100),
            'crowd': (200, 200, 100), 'flow': (200, 150, 100),
        }
        mode_color = mode_colors.get(mode_name, (200, 200, 200))
        
        status_y = 200
        draw_text_2d(view_width + 20, status_y + 50, "─" * 20, font_small)
        draw_text_2d(view_width + 20, status_y + 35, "MODE DECISION:", font_small, (255, 200, 100))
        draw_text_2d(view_width + 20, status_y + 17, f"  Active: {active_count}", font_small, mode_color)
        draw_text_2d(view_width + 20, status_y + 1, f"  Passive: {passive_count}", font_small)
        draw_text_2d(view_width + 20, status_y - 17, f"  Mode: {mode_name.upper()}", font_small, mode_color)
        draw_text_2d(view_width + 20, status_y - 33, f"  Preset: {current_preset}", font_small)
        
        # Help text
        draw_text_2d(view_width + 20, 50, "SPC=wander M=markers L=labels P=preset Q=quit", font_small, (120, 120, 120))
        draw_text_2d(view_width + 20, 35, "T=trends F=fullscreen HOME=reset camera", font_small, (120, 120, 120))
        
        # Legend
        draw_text_2d(10, display[1] - 20, "V3 VISUAL DEBUG:", font_small, (255, 200, 100))
        draw_text_2d(10, display[1] - 40, "  Yellow = ORIGIN  Red = Cam1  Blue = Cam2", font_small, (200, 200, 200))
        
        # Info
        dmx = panel_system.get_dmx_values()
        draw_text_2d(10, 100, f"Light: ({light.position[0]:.0f}, {light.position[1]:.0f}, {light.position[2]:.0f})", font_small)
        draw_text_2d(10, 118, f"DMX: {dmx}", font_small)
        draw_text_2d(10, 136, f"Mode: {mode_name.upper()}  A:{active_count} P:{passive_count}", font_small)
        
        # Trends
        if show_trends:
            # V3 stats structure
            aggression = behavior_stats.get('aggression', 0.0)
            flow_stats = behavior_stats.get('flow', {})
            almost_engaged = behavior_stats.get('almost_engaged', {})
            feedback = behavior_stats.get('feedback', {})
            draw_realtime_trends(None, 10, display[1] - 100, font, font_small,
                               aggression, flow_stats, almost_engaged, feedback)
        
        restore_3d_projection()
        swap_buffers()
        clock.tick(FPS)
        frame_count += 1
        
        # Health log
        if now - last_health_log >= HEALTH_LOG_INTERVAL:
            elapsed = now - start_time
            uptime = timedelta(seconds=int(elapsed))
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"HEALTH: uptime={uptime}, fps={avg_fps:.1f}, mode={mode_name}")
            last_health_log = now
    
    # Cleanup
    logger.info("Shutting down...")
    save_slider_settings(all_sliders, checkboxes)
    
    if osc_server_instance:
        try:
            osc_server_instance.server_close()
        except Exception:
            pass
    if artnet:
        artnet.stop()
    pygame.quit()
    
    elapsed = time.time() - start_time
    print(f"\n🛑 Stopped after {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()
