#!/usr/bin/env python3
"""
3D Point Light LED Controller with PyVista Visualization

Features:
- 3D panel geometry (12 panels as rectangles)
- Point light as a sphere with real-time brightness calculation
- Interactive controls for light position
- Trackzone visualization (wireframe box)
- Simulated person in trackzone
- Animation of wandering behavior
- Art-Net output to real panels
- Linear falloff based on distance

All units in centimeters.
"""

import numpy as np
import pyvista as pv
from pyvista import themes
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import random

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
PANEL_SIZE = 60  # 60cm x 60cm

# Unit spacing (cm)
UNIT_SPACING = 80  # 80cm center-to-center

# Panel positions relative to unit center (y, z) in cm
# Panel 1: top vertical, Panel 2: lower front tilted up, Panel 3: lower back tilted down
PANEL_LOCAL_POSITIONS = {
    1: (90, 0),    # Panel 1: top, z=0
    2: (30, 12),   # Panel 2: lower front, z=+12
    3: (30, -12),  # Panel 3: lower back, z=-12
}

# Panel angles (degrees from vertical, positive = tilted toward +Z)
PANEL_ANGLES = {
    1: 0,      # Vertical
    2: 22.5,   # Tilted forward (up toward viewers)
    3: -22.5,  # Tilted forward but angled down
}

# Panel normals (unit vectors)
PANEL_NORMALS = {
    1: np.array([0, 0, 1]),                    # Faces forward
    2: np.array([0, 0.38268, 0.92388]),        # Tilted UP, faces outward
    3: np.array([0, -0.38268, 0.92388]),       # Tilted DOWN, faces forward
}

# Trackzone dimensions (cm)
TRACKZONE = {
    'width': 475,      # X dimension
    'depth': 205,      # Z dimension  
    'height': 300,     # Y dimension
    'offset_z': 78,    # Distance from panel front
    'offset_y': -66,   # Street is 66cm below panel base
    'center_x': 120,   # Center on panel array (between unit 1 and 2)
}

# Wandering bounding box (cm) - interior volume behind panels
WANDER_BOX = {
    'min_x': -50,
    'max_x': 290,
    'min_y': 0,
    'max_y': 150,
    'min_z': -100,
    'max_z': 0,
}


@dataclass
class PointLight:
    """Virtual point light properties"""
    # Current position (cm)
    position: np.ndarray = field(default_factory=lambda: np.array([120.0, 60.0, -30.0]))
    
    # Target position for animation
    target_position: np.ndarray = field(default_factory=lambda: np.array([120.0, 60.0, -30.0]))
    
    # Brightness properties
    brightness_min: int = 5
    brightness_max: int = 40
    pulse_speed: float = 2000  # ms for full cycle
    falloff_radius: float = 200  # cm where brightness hits 0
    
    # Target values for animation
    target_brightness_min: int = 5
    target_brightness_max: int = 40
    target_falloff_radius: float = 200
    
    # Animation speed
    move_speed: float = 50  # cm per second
    property_lerp_speed: float = 2.0  # properties per second
    
    # Current pulse phase
    pulse_phase: float = 0.0
    
    def get_current_brightness(self) -> float:
        """Get current brightness value (0-1) based on pulse"""
        # Sine wave oscillation
        pulse = (math.sin(self.pulse_phase) + 1) / 2
        return pulse
    
    def update(self, dt: float):
        """Update light animation"""
        # Update pulse phase
        self.pulse_phase += (2 * math.pi * dt * 1000) / self.pulse_speed
        
        # Animate position toward target
        diff = self.target_position - self.position
        dist = np.linalg.norm(diff)
        if dist > 0.1:
            move_amount = min(self.move_speed * dt, dist)
            self.position += (diff / dist) * move_amount
        
        # Animate brightness properties
        lerp = min(1.0, self.property_lerp_speed * dt)
        self.brightness_min += int((self.target_brightness_min - self.brightness_min) * lerp)
        self.brightness_max += int((self.target_brightness_max - self.brightness_max) * lerp)
        self.falloff_radius += (self.target_falloff_radius - self.falloff_radius) * lerp


@dataclass
class SimulatedPerson:
    """Simulated person in trackzone"""
    enabled: bool = False
    position: np.ndarray = field(default_factory=lambda: np.array([120.0, 80.0, 150.0]))
    size: Tuple[float, float, float] = (40, 170, 30)  # width, height, depth in cm
    
    def get_center(self) -> np.ndarray:
        """Get center of person bounding box"""
        return self.position + np.array([0, self.size[1]/2, 0])


class PanelSystem:
    """Manages all 12 panels"""
    
    def __init__(self):
        self.panels: Dict[Tuple[int, int], dict] = {}
        self._build_panels()
    
    def _build_panels(self):
        """Create all panel data"""
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
    
    def calculate_brightness(self, light: PointLight) -> None:
        """Calculate brightness for all panels based on point light"""
        current_intensity = light.get_current_brightness()
        
        for key, panel in self.panels.items():
            # Distance from light to panel center
            diff = panel['center'] - light.position
            distance = np.linalg.norm(diff)
            
            # Linear falloff
            if distance >= light.falloff_radius:
                falloff = 0.0
            else:
                falloff = 1.0 - (distance / light.falloff_radius)
            
            # Optional: Factor in panel normal (dot product with light direction)
            # This makes panels facing the light brighter
            if distance > 0.01:
                light_dir = -diff / distance  # Direction from panel to light
                dot = np.dot(panel['normal'], light_dir)
                # Only use positive dot products (light in front of panel)
                normal_factor = max(0, dot)
            else:
                normal_factor = 1.0
            
            # Combine factors
            brightness = falloff * normal_factor * current_intensity
            panel['brightness'] = max(0.0, min(1.0, brightness))
            
            # Calculate DMX value
            dmx_range = light.brightness_max - light.brightness_min
            panel['dmx_value'] = int(light.brightness_min + brightness * dmx_range)
            panel['dmx_value'] = max(DMX_MIN, min(DMX_MAX, panel['dmx_value']))
    
    def get_dmx_values(self) -> List[int]:
        """Get DMX values for all 12 channels"""
        values = []
        for unit in range(4):
            for panel_num in range(1, 4):
                values.append(self.panels[(unit, panel_num)]['dmx_value'])
        return values


class WanderBehavior:
    """Wandering behavior for the point light"""
    
    def __init__(self, light: PointLight):
        self.light = light
        self.wander_target = self._random_point()
        self.wander_timer = 0
        self.wander_interval = 3.0  # seconds between new targets
        self.enabled = True
    
    def _random_point(self) -> np.ndarray:
        """Generate random point in wander box"""
        return np.array([
            random.uniform(WANDER_BOX['min_x'], WANDER_BOX['max_x']),
            random.uniform(WANDER_BOX['min_y'], WANDER_BOX['max_y']),
            random.uniform(WANDER_BOX['min_z'], WANDER_BOX['max_z']),
        ])
    
    def update(self, dt: float, person: Optional[SimulatedPerson] = None):
        """Update wandering behavior"""
        if not self.enabled:
            return
        
        # If person is in trackzone, follow them
        if person and person.enabled:
            self.light.target_position = person.get_center().copy()
            # Adjust falloff to reach panels from person position
            person_z = person.position[2]
            self.light.target_falloff_radius = max(200, person_z + 50)
            return
        
        # Otherwise, wander
        self.wander_timer += dt
        
        # Check if we reached target or timer expired
        dist_to_target = np.linalg.norm(self.light.position - self.wander_target)
        if dist_to_target < 10 or self.wander_timer > self.wander_interval:
            self.wander_target = self._random_point()
            self.wander_timer = 0
            # Randomize some properties
            self.light.target_brightness_min = random.randint(3, 10)
            self.light.target_brightness_max = random.randint(30, 45)
            self.light.target_falloff_radius = random.uniform(150, 300)
        
        self.light.target_position = self.wander_target.copy()


class Visualizer:
    """PyVista 3D visualization"""
    
    def __init__(self, panel_system: PanelSystem, light: PointLight, 
                 person: SimulatedPerson, wander: WanderBehavior):
        self.panel_system = panel_system
        self.light = light
        self.person = person
        self.wander = wander
        
        # Art-Net
        self.artnet = None
        self._init_artnet()
        
        # PyVista setup
        self.plotter = pv.Plotter(title="3D Point Light Controller")
        self.plotter.set_background('#1a1a2a')
        
        # Mesh actors (stored for updates)
        self.panel_actors = {}
        self.light_actor = None
        self.person_actor = None
        self.trackzone_actor = None
        self.wander_box_actor = None
        
        # Build scene
        self._build_scene()
        
        # Animation state
        self.last_time = time.time()
        self.running = True
    
    def _init_artnet(self):
        """Initialize Art-Net connection"""
        if not ARTNET_AVAILABLE:
            return
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            self.artnet.start()
            print(f"Art-Net connected to {TARGET_IP}")
        except Exception as e:
            print(f"Art-Net init failed: {e}")
            self.artnet = None
    
    def _create_panel_mesh(self, center: np.ndarray, angle: float) -> pv.PolyData:
        """Create a panel mesh at given position and angle"""
        # Create a plane
        plane = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=PANEL_SIZE,
            j_size=PANEL_SIZE,
        )
        
        # Rotate around X axis based on panel angle
        plane.rotate_x(-angle, inplace=True)
        
        # Translate to position
        plane.translate(center, inplace=True)
        
        return plane
    
    def _build_scene(self):
        """Build the 3D scene"""
        # Create floor plane (at street level, y = -66)
        floor = pv.Plane(
            center=(120, TRACKZONE['offset_y'], 100),
            direction=(0, 1, 0),
            i_size=600,
            j_size=400,
        )
        self.plotter.add_mesh(floor, color='#333344', opacity=0.5)
        
        # Create storefront floor (at y = 0)
        storefront_floor = pv.Plane(
            center=(120, 0, -50),
            direction=(0, 1, 0),
            i_size=400,
            j_size=150,
        )
        self.plotter.add_mesh(storefront_floor, color='#444455', opacity=0.5)
        
        # Create panels
        for (unit, panel_num), panel_data in self.panel_system.panels.items():
            mesh = self._create_panel_mesh(
                panel_data['center'],
                panel_data['angle']
            )
            actor = self.plotter.add_mesh(
                mesh, 
                color='white',
                show_edges=True,
                edge_color='#666666',
                name=f'panel_{unit}_{panel_num}'
            )
            self.panel_actors[(unit, panel_num)] = {
                'mesh': mesh,
                'center': panel_data['center'],
                'angle': panel_data['angle'],
            }
        
        # Create point light sphere
        light_sphere = pv.Sphere(radius=10, center=self.light.position)
        self.light_actor = self.plotter.add_mesh(
            light_sphere, 
            color='yellow',
            name='point_light'
        )
        
        # Create trackzone wireframe
        trackzone_box = pv.Box(bounds=(
            TRACKZONE['center_x'] - TRACKZONE['width']/2,
            TRACKZONE['center_x'] + TRACKZONE['width']/2,
            TRACKZONE['offset_y'],
            TRACKZONE['offset_y'] + TRACKZONE['height'],
            TRACKZONE['offset_z'],
            TRACKZONE['offset_z'] + TRACKZONE['depth'],
        ))
        self.trackzone_actor = self.plotter.add_mesh(
            trackzone_box,
            style='wireframe',
            color='cyan',
            line_width=2,
            opacity=0.5,
            name='trackzone'
        )
        
        # Create wander box wireframe
        wander_box = pv.Box(bounds=(
            WANDER_BOX['min_x'], WANDER_BOX['max_x'],
            WANDER_BOX['min_y'], WANDER_BOX['max_y'],
            WANDER_BOX['min_z'], WANDER_BOX['max_z'],
        ))
        self.wander_box_actor = self.plotter.add_mesh(
            wander_box,
            style='wireframe',
            color='green',
            line_width=1,
            opacity=0.3,
            name='wander_box'
        )
        
        # Create simulated person (initially hidden)
        person_box = pv.Box(bounds=(
            -self.person.size[0]/2, self.person.size[0]/2,
            0, self.person.size[1],
            -self.person.size[2]/2, self.person.size[2]/2,
        ))
        person_box.translate(self.person.position, inplace=True)
        self.person_mesh = person_box
        self.person_actor = self.plotter.add_mesh(
            person_box,
            color='orange',
            opacity=0.7 if self.person.enabled else 0.0,
            name='person'
        )
        
        # Add coordinate axes
        self.plotter.add_axes()
        
        # Add labels
        self.plotter.add_text(
            "Trackzone (cyan) | Wander Box (green) | Press 'p' to toggle person",
            position='lower_left',
            font_size=10,
            color='white'
        )
        
        # Set camera
        self.plotter.camera_position = [
            (120, 200, 500),  # Camera position
            (120, 50, 0),     # Focal point
            (0, 1, 0),        # Up vector
        ]
        
        # Key bindings
        self.plotter.add_key_event('p', self._toggle_person)
        self.plotter.add_key_event('w', self._toggle_wander)
        self.plotter.add_key_event('q', self._quit)
    
    def _toggle_person(self):
        """Toggle simulated person"""
        self.person.enabled = not self.person.enabled
        print(f"Person {'enabled' if self.person.enabled else 'disabled'}")
    
    def _toggle_wander(self):
        """Toggle wandering behavior"""
        self.wander.enabled = not self.wander.enabled
        print(f"Wandering {'enabled' if self.wander.enabled else 'disabled'}")
    
    def _quit(self):
        """Quit the application"""
        self.running = False
        self.plotter.close()
    
    def _update_panel_colors(self):
        """Update panel colors based on brightness"""
        for (unit, panel_num), panel_data in self.panel_system.panels.items():
            brightness = panel_data['brightness']
            # Map brightness to color (dark gray to white)
            gray = int(50 + brightness * 205)
            color = f'#{gray:02x}{gray:02x}{gray:02x}'
            
            # Recreate mesh with new color
            mesh = self._create_panel_mesh(
                self.panel_actors[(unit, panel_num)]['center'],
                self.panel_actors[(unit, panel_num)]['angle']
            )
            self.plotter.remove_actor(f'panel_{unit}_{panel_num}')
            self.plotter.add_mesh(
                mesh,
                color=color,
                show_edges=True,
                edge_color='#666666',
                name=f'panel_{unit}_{panel_num}'
            )
    
    def _update_light_position(self):
        """Update light sphere position"""
        self.plotter.remove_actor('point_light')
        
        # Size based on current brightness
        brightness = self.light.get_current_brightness()
        radius = 8 + brightness * 7
        
        light_sphere = pv.Sphere(radius=radius, center=self.light.position)
        
        # Color based on brightness (yellow to white)
        intensity = int(200 + brightness * 55)
        color = f'#{intensity:02x}{intensity:02x}00'
        
        self.plotter.add_mesh(
            light_sphere,
            color=color,
            name='point_light'
        )
    
    def _update_person(self):
        """Update person mesh"""
        self.plotter.remove_actor('person')
        
        if self.person.enabled:
            person_box = pv.Box(bounds=(
                self.person.position[0] - self.person.size[0]/2,
                self.person.position[0] + self.person.size[0]/2,
                self.person.position[1],
                self.person.position[1] + self.person.size[1],
                self.person.position[2] - self.person.size[2]/2,
                self.person.position[2] + self.person.size[2]/2,
            ))
            self.plotter.add_mesh(
                person_box,
                color='orange',
                opacity=0.7,
                name='person'
            )
    
    def _send_artnet(self):
        """Send DMX values via Art-Net"""
        if not self.artnet:
            return
        
        dmx_values = self.panel_system.get_dmx_values()
        self.artnet.set(dmx_values)
    
    def update(self):
        """Main update loop"""
        now = time.time()
        dt = min(now - self.last_time, 0.1)
        self.last_time = now
        
        # Update wandering behavior
        self.wander.update(dt, self.person if self.person.enabled else None)
        
        # Update light animation
        self.light.update(dt)
        
        # Calculate panel brightness
        self.panel_system.calculate_brightness(self.light)
        
        # Update visuals
        self._update_panel_colors()
        self._update_light_position()
        self._update_person()
        
        # Send Art-Net
        self._send_artnet()
    
    def run(self):
        """Run the visualization"""
        # Use a timer thread for animation since add_callback may not be available
        def animation_loop():
            while self.running:
                try:
                    self.update()
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Animation error: {e}")
                    break
        
        # Start animation thread
        anim_thread = threading.Thread(target=animation_loop, daemon=True)
        anim_thread.start()
        
        # Show the plotter (blocking)
        self.plotter.show()
        
        # Cleanup
        self.running = False
        if self.artnet:
            self.artnet.stop()


def main():
    """Main entry point"""
    print("3D Point Light Controller")
    print("=" * 40)
    print("Controls:")
    print("  p - Toggle simulated person")
    print("  w - Toggle wandering behavior")
    print("  q - Quit")
    print("  Mouse - Rotate/zoom camera")
    print("=" * 40)
    
    # Create system components
    panel_system = PanelSystem()
    light = PointLight()
    person = SimulatedPerson()
    wander = WanderBehavior(light)
    
    # Create and run visualizer
    viz = Visualizer(panel_system, light, person, wander)
    viz.run()


if __name__ == "__main__":
    main()
