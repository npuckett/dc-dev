"""
Scene Renderer
==============
Draws the complete 3D scene including zones, panels, cameras, markers.
Extracted from lightController_osc.py.
"""

import os
from typing import Dict, List, Tuple, Optional, Any

try:
    import pygame
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from .primitives import (
    draw_box_wireframe, draw_panel, draw_sphere, draw_sphere_wireframe,
    draw_floor, draw_tracked_person, draw_axis_lines, draw_camera_cone
)
from .hud import draw_text_3d_billboard


# =============================================================================
# ZONE CONFIGURATION (from original lightController_osc.py)
# =============================================================================

# Trackzone (cm) - defines the ACTIVE tracking area
TRACKZONE = {
    'width': 260,
    'depth': 205,
    'height': 300,
    'offset_z': 78,
    'offset_y': -66,
    'center_x': -150,
}

# Passive trackzone (cm)
PASSIVE_TRACKZONE = {
    'width': 400,
    'depth': 270,
    'height': 300,
    'offset_z': 78 + 205,  # Starts at back of active zone (283cm)
    'offset_y': -66,
    'center_x': -150,
}

# Wander box (cm)
WANDER_BOX = {
    'min_x': -280, 'max_x': -20,
    'min_y': 0, 'max_y': 150,
    'min_z': -28, 'max_z': 32,
}

# Street level Y coordinate
STREET_LEVEL_Y = -66
CAMERA_LEDGE_Y = -16

# Camera positions
CAMERA_Y = -15
CAMERA_Z = TRACKZONE['offset_z']

CAMERA_POSITIONS = {
    'Camera 1': {
        'pos': (-30, CAMERA_Y, CAMERA_Z),
        'desc': 'Right camera - angled toward center',
        'color': (1.0, 0.3, 0.3, 1.0),
        'target': (-150, STREET_LEVEL_Y, 180),
        'rotation': {'pitch': 22, 'yaw': -25, 'roll': 0},
        'fov': {'horizontal': 80, 'vertical': 48},
    },
    'Camera 2': {
        'pos': (-270, CAMERA_Y, CAMERA_Z),
        'desc': 'Left camera - angled toward center',
        'color': (0.3, 0.3, 1.0, 1.0),
        'target': (-150, STREET_LEVEL_Y, 180),
        'rotation': {'pitch': 22, 'yaw': 25, 'roll': 0},
        'fov': {'horizontal': 80, 'vertical': 48},
    },
}

# Marker positions
MARKER_SIZE = 15

MARKER_POSITIONS = {
    0: {'pos': (-30, STREET_LEVEL_Y, 168), 'desc': 'Right front', 'camera': 'Cam 1', 'vertical': False},
    1: {'pos': (-150, STREET_LEVEL_Y, 168), 'desc': 'Center front (SHARED)', 'camera': 'Both', 'vertical': False},
    2: {'pos': (-270, STREET_LEVEL_Y, 168), 'desc': 'Left front', 'camera': 'Cam 2', 'vertical': False},
    3: {'pos': (-30, STREET_LEVEL_Y, 219), 'desc': 'Right back', 'camera': 'Cam 1', 'vertical': False},
    4: {'pos': (-270, STREET_LEVEL_Y, 219), 'desc': 'Left back', 'camera': 'Cam 2', 'vertical': False},
    5: {'pos': (-150, CAMERA_Y, 578), 'desc': 'Subway wall (VERTICAL, ~5m from cams)', 'camera': 'Both', 'vertical': True},
    6: {'pos': (-150, STREET_LEVEL_Y, 219), 'desc': 'Center back (SHARED)', 'camera': 'Both', 'vertical': False},
}


def get_zone_bounds(zone_config: dict) -> Tuple[float, ...]:
    """
    Convert zone config dict to bounds tuple.
    
    Returns:
        (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    half_w = zone_config['width'] / 2
    center_x = zone_config['center_x']
    offset_y = zone_config['offset_y']
    offset_z = zone_config['offset_z']
    
    return (
        center_x - half_w,  # x_min
        center_x + half_w,  # x_max
        offset_y,           # y_min
        offset_y + zone_config['height'],  # y_max
        offset_z,           # z_min
        offset_z + zone_config['depth'],   # z_max
    )


def get_wander_bounds(wander_box: dict) -> Tuple[float, ...]:
    """Convert wander box dict to bounds tuple."""
    return (
        wander_box['min_x'], wander_box['max_x'],
        wander_box['min_y'], wander_box['max_y'],
        wander_box['min_z'], wander_box['max_z'],
    )


class SceneRenderer:
    """
    Renders the complete 3D scene.
    
    Usage:
        renderer = SceneRenderer()
        renderer.draw_scene(panel_system, light, tracked_people)
    """
    
    def __init__(self):
        """Initialize the scene renderer."""
        self.show_markers = False
        self.show_labels = True
        self.show_camera_views = False
        
        # Marker textures (loaded on first draw if needed)
        self.marker_textures: Dict[int, int] = {}
        self._textures_loaded = False
    
    def load_marker_textures(self, calibration_path: str):
        """
        Load marker PNG files as OpenGL textures.
        
        Args:
            calibration_path: Path to calibration folder containing marker_N.png files
        """
        if not OPENGL_AVAILABLE:
            return
            
        for marker_id in MARKER_POSITIONS.keys():
            image_path = os.path.join(calibration_path, f'marker_{marker_id}.png')
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
                    
                    self.marker_textures[marker_id] = texture_id
                except Exception as e:
                    print(f"Failed to load marker {marker_id}: {e}")
        
        self._textures_loaded = True
    
    def draw_zones(self, wander_box: Optional[dict] = None):
        """Draw the tracking zones as wireframes."""
        if not OPENGL_AVAILABLE:
            return
        
        # Active zone (cyan - matches original)
        active_bounds = get_zone_bounds(TRACKZONE)
        draw_box_wireframe(active_bounds, (0.0, 1.0, 1.0, 0.5))
        
        # Passive zone (orange - matches original)
        passive_bounds = get_zone_bounds(PASSIVE_TRACKZONE)
        draw_box_wireframe(passive_bounds, (1.0, 0.6, 0.0, 0.4))
        
        # Wander box (green - matches original)
        if wander_box:
            wander_bounds = get_wander_bounds(wander_box)
            draw_box_wireframe(wander_bounds, (0.0, 1.0, 0.0, 0.3))
    
    def draw_floor_plane(self, z_max: Optional[float] = None):
        """Draw the floor plane."""
        if not OPENGL_AVAILABLE:
            return
        draw_floor(0, (0.15, 0.15, 0.18, 1.0), z_max)
    
    def draw_street_plane(self, z_max: Optional[float] = None):
        """Draw the street level plane."""
        if not OPENGL_AVAILABLE:
            return
        draw_floor(STREET_LEVEL_Y, (0.12, 0.12, 0.14, 1.0), z_max)
    
    def draw_origin_marker(self, font):
        """Draw a sphere at the origin (0,0,0) with axes."""
        if not OPENGL_AVAILABLE:
            return
            
        origin_pos = (0, 0, 0)
        
        # Draw sphere at origin
        draw_sphere(origin_pos, 10, (1.0, 1.0, 0.0, 1.0), segments=16)
        
        # Draw axis lines
        draw_axis_lines(origin_pos, length=50)
        
        # Draw label
        if self.show_labels:
            draw_text_3d_billboard(origin_pos, "Origin (0,0,0)", font, (255, 255, 0), offset_y=20)
    
    def draw_camera_markers(self, font):
        """Draw spheres at camera positions with labels and viewing cones."""
        if not OPENGL_AVAILABLE:
            return
            
        for cam_name, cam_data in CAMERA_POSITIONS.items():
            pos = cam_data['pos']
            color = cam_data['color']
            rotation = cam_data.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
            
            # Draw camera as a sphere
            draw_sphere(pos, 15, color, segments=16)
            
            # Draw viewing cone
            draw_camera_cone(pos, rotation, color)
            
            # Draw labels
            if self.show_labels:
                draw_text_3d_billboard(pos, cam_name, font, (255, 255, 255), offset_y=25)
                coord_text = f"({pos[0]}, {pos[1]}, {pos[2]})"
                draw_text_3d_billboard(pos, coord_text, font, (200, 200, 200), offset_y=10)
    
    def draw_calibration_markers(self, font):
        """Draw the calibration markers."""
        if not OPENGL_AVAILABLE or not self.show_markers:
            return
            
        for marker_id, marker_data in MARKER_POSITIONS.items():
            pos = marker_data['pos']
            
            # Draw marker as a small sphere
            draw_sphere(pos, MARKER_SIZE / 2, (1.0, 0.5, 0.0, 0.8), segments=12)
            
            # Draw label
            if self.show_labels:
                label = f"M{marker_id}"
                draw_text_3d_billboard(pos, label, font, (255, 200, 100), offset_y=15)
    
    def draw_panels(self, panel_system, brightness_values: Optional[Dict] = None):
        """
        Draw all panels from the panel system.
        
        Args:
            panel_system: PanelSystem object with panels dict
            brightness_values: Optional dict of (unit, panel) -> brightness
        """
        if not OPENGL_AVAILABLE:
            return
            
        for (unit, panel_num), panel_data in panel_system.panels.items():
            center = panel_data['center']
            angle = panel_data['angle']
            
            # Get brightness from values dict or panel data
            if brightness_values and (unit, panel_num) in brightness_values:
                brightness = brightness_values[(unit, panel_num)]
            else:
                brightness = panel_data.get('brightness', 0.5)
            
            draw_panel(center, angle, 60, brightness)  # PANEL_SIZE = 60
    
    def draw_unit_labels(self, panel_system, font):
        """Draw labels for each panel unit."""
        if not OPENGL_AVAILABLE or not self.show_labels:
            return
            
        unit_centers = panel_system.get_unit_centers()
        
        for unit_num, center in unit_centers.items():
            # Draw unit label
            unit_label = f"Unit {unit_num}"
            draw_text_3d_billboard(center, unit_label, font, (255, 200, 100), offset_y=80)
            
            # Draw coordinate
            coord_text = f"X={center[0]}"
            draw_text_3d_billboard(center, coord_text, font, (180, 180, 180), offset_y=65)
    
    def draw_panel_centers(self, panel_system, font):
        """Draw wireframe spheres at each panel center with labels."""
        if not OPENGL_AVAILABLE:
            return
            
        # Colors for each panel position within a unit
        panel_colors = {
            1: (1.0, 0.5, 0.5, 0.8),  # Panel 1 (top) - light red
            2: (0.5, 1.0, 0.5, 0.8),  # Panel 2 (bottom left) - light green
            3: (0.5, 0.5, 1.0, 0.8),  # Panel 3 (bottom right) - light blue
        }
        
        for (unit, panel_num), panel in panel_system.panels.items():
            center = panel['center']
            color = panel_colors.get(panel_num, (1.0, 1.0, 1.0, 0.8))
            
            # Draw small wireframe sphere at panel center
            draw_sphere_wireframe(center, 2, color, segments=12)
            
            # Draw label
            if self.show_labels:
                label = f"U{unit}P{panel_num}"
                draw_text_3d_billboard(center, label, font, (255, 255, 255), offset_y=15)
    
    def draw_light_position(self, position: Tuple[float, float, float], 
                            radius: float = 15, color: Tuple[float, ...] = None):
        """
        Draw the light position as a glowing sphere.
        
        Args:
            position: (x, y, z) light position
            radius: Sphere radius
            color: RGBA color (default warm yellow)
        """
        if not OPENGL_AVAILABLE:
            return
            
        if color is None:
            color = (1.0, 0.9, 0.4, 1.0)  # Warm yellow
        
        draw_sphere(position, radius, color, segments=20)
    
    def draw_tracked_people(self, people: List[Any], zone_checker=None):
        """
        Draw all tracked people.
        
        Args:
            people: List of tracked person objects with get_position() method
            zone_checker: Optional function(x, z) -> str for zone determination
        """
        if not OPENGL_AVAILABLE:
            return
            
        for person in people:
            pos = person.get_position() if hasattr(person, 'get_position') else person.position
            
            # Determine zone
            if zone_checker:
                zone = zone_checker(pos[0], pos[2])
            elif hasattr(person, 'zone'):
                zone = person.zone
            else:
                zone = 'unknown'
            
            draw_tracked_person(tuple(pos), zone)
    
    def draw_zone_corner_labels(self, font):
        """Draw coordinate labels at zone corners."""
        if not OPENGL_AVAILABLE or not self.show_labels:
            return
            
        # Active zone
        active_bounds = get_zone_bounds(TRACKZONE)
        self._draw_corner_labels(active_bounds, "ACTIVE", font, (100, 200, 100))
        
        # Passive zone
        passive_bounds = get_zone_bounds(PASSIVE_TRACKZONE)
        self._draw_corner_labels(passive_bounds, "PASSIVE", font, (200, 200, 100))
    
    def _draw_corner_labels(self, bounds: Tuple[float, ...], name: str, 
                            font, color: Tuple[int, int, int]):
        """Draw labels at corners of a zone."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Draw labels at bottom corners
        corners = [
            (x_min, y_min, z_min),
            (x_max, y_min, z_min),
            (x_min, y_min, z_max),
            (x_max, y_min, z_max),
        ]
        
        for x, y, z in corners:
            coord_text = f"({int(x)},{int(y)},{int(z)})"
            draw_text_3d_billboard((x, y, z), coord_text, font, color, offset_y=5)
        
        # Draw zone name at center top
        center_x = (x_min + x_max) / 2
        center_z = (z_min + z_max) / 2
        draw_text_3d_billboard((center_x, y_max, center_z), name, font, color, offset_y=10)
