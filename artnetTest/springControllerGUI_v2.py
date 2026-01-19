#!/usr/bin/env python3
"""
Spring-Based LED Panel Controller with GUI v2
Enhanced with: spring visualization, tuning controls, more panel contrast
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from stupidArtnet import StupidArtnet

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

NUM_UNITS = 4
PANELS_PER_UNIT = 3
TOTAL_CHANNELS = NUM_UNITS * PANELS_PER_UNIT

# DMX output range - we only use 1-50 (not the full 1-212 hardware range)
# This matches the behavior spec and artnetTest.py usage
MIN_DMX_VALUE = 1
MAX_DMX_VALUE = 50  # Practical max we use on the panels

# Time of day parameters
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 16
EVENING_RUSH_END = 19
EVENING_START = 18
NIGHT_START = 22

# =============================================================================
# SPRING PHYSICS - Enhanced with row differentiation
# =============================================================================

@dataclass
class SpringConfig:
    # Core spring properties
    stiffness: float = 0.20           # How quickly springs respond (higher = snappier)
    damping: float = 0.75             # Energy loss (lower = more movement/oscillation)
    
    # Coupling - VERY LOW for independence between panels
    horizontal_coupling: float = 0.05  # Left-right coupling (same row)
    vertical_coupling: float = 0.02    # Up-down coupling (between rows)
    
    # Row differentiation - KEY FOR CONTRAST
    row_delay: float = 0.4            # Row 1 responds slower than row 0
    row_amplitude_diff: float = 0.6   # Row 1 has different amplitude (higher = more diff)
    row_phase_offset: float = 0.7     # Row 1 phase offset in seconds
    
    # Force magnitudes - MUCH HIGHER for visible response
    active_force_base: float = 40.0   # Strong response to presence
    active_force_dwell_mult: float = 2.0  # Builds up over time
    passive_force_base: float = 5.0
    
    # Position falloff - SHARPER for left/right contrast
    position_falloff: float = 0.6     # Higher = sharper focus on position
    
    # Per-column variation (for organic feel)
    column_variation: float = 0.3     # Random variation per column
    
    # Velocity and response
    max_velocity: float = 8.0         # Much higher for dramatic movement
    
    # Ambient behavior
    base_brightness: float = 3.0      # Lower base for more headroom
    passive_brightness_mult: float = 0.3
    ambient_wave_speed: float = 0.5   # Faster ambient oscillation
    ambient_wave_amplitude: float = 6.0  # Bigger waves
    
    # Engagement
    engagement_decay_rate: float = 0.02
    acknowledgment_boost_max: float = 3.0
    acknowledgment_decay_time: float = 300.0


@dataclass
class InputState:
    active_population: int = 0
    active_position_x: float = 0.5
    active_dwell_time: float = 0.0
    passive_population: int = 0
    passive_flow_speed: float = 0.0
    passive_flow_direction: float = 0.0
    cyclist_count: int = 0
    vehicle_count: int = 0
    time_since_active: float = 999.0


@dataclass
class NarrativeStatus:
    state: str = "Ambient"
    action: str = "Breathing"
    reason: str = "No visitors"
    detail: str = ""
    mood: str = "Calm"


@dataclass
class SpringNode:
    row: int
    col: int
    position: float = 5.0
    velocity: float = 0.0
    rest_position: float = 5.0
    force: float = 0.0
    neighbors: List['SpringNode'] = field(default_factory=list)
    neighbor_couplings: List[float] = field(default_factory=list)  # Per-neighbor coupling strength
    
    # Per-node variation
    phase_offset: float = 0.0
    amplitude_mult: float = 1.0
    response_delay: float = 0.0
    
    # Force history for delayed response
    force_history: List[float] = field(default_factory=list)
    
    def apply_force(self, f: float, delay_frames: int = 0):
        """Apply force with optional delay"""
        if delay_frames > 0:
            # Store for delayed application
            while len(self.force_history) <= delay_frames:
                self.force_history.append(0.0)
            self.force_history[delay_frames] += f
        else:
            self.force += f
    
    def update(self, config: SpringConfig, dt: float):
        # Process delayed forces
        if self.force_history:
            self.force += self.force_history.pop(0)
            
        # Spring force toward rest position
        spring_force = -config.stiffness * (self.position - self.rest_position)
        
        # Coupling force from neighbors (with per-neighbor strength)
        coupling_force = 0.0
        for i, neighbor in enumerate(self.neighbors):
            coupling = self.neighbor_couplings[i] if i < len(self.neighbor_couplings) else 0.1
            coupling_force += coupling * (neighbor.position - self.position)
        
        # Apply amplitude multiplier to external forces
        external_force = self.force * self.amplitude_mult
        
        # Total acceleration
        acceleration = spring_force + coupling_force + external_force
        
        # Update velocity with damping
        self.velocity = (self.velocity + acceleration * dt) * config.damping
        self.velocity = max(-config.max_velocity, min(config.max_velocity, self.velocity))
        
        # Update position
        self.position += self.velocity * dt
        self.position = max(MIN_DMX_VALUE, min(MAX_DMX_VALUE, self.position))
        
        # Clear force
        self.force = 0.0
    
    @property
    def brightness(self) -> float:
        return self.position


@dataclass
class BackPanel:
    col: int
    brightness: float = 3.0
    target: float = 3.0
    phase: float = 0.0
    speed: float = 0.3
    
    def update(self, dt: float, time_factor: float, passive_energy: float):
        self.phase += self.speed * dt * (1 + time_factor)
        wave = math.sin(self.phase + self.col * 0.7) * 0.5 + 0.5
        amplitude = 3 + time_factor * 8
        base = 2 + time_factor * 5
        passive_boost = passive_energy * 0.1
        
        self.target = base + wave * amplitude + passive_boost
        self.target = max(MIN_DMX_VALUE, min(MAX_DMX_VALUE, self.target))
        
        diff = self.target - self.brightness
        self.brightness += diff * 0.05
        self.brightness = max(MIN_DMX_VALUE, min(MAX_DMX_VALUE, self.brightness))


# =============================================================================
# SPRING CONTROLLER - Enhanced
# =============================================================================

class SpringController:
    def __init__(self, config: Optional[SpringConfig] = None):
        self.config = config or SpringConfig()
        
        # Create front grid with per-node variation
        self.grid: List[List[SpringNode]] = []
        for row in range(2):
            row_nodes = []
            for col in range(4):
                node = SpringNode(row=row, col=col, position=self.config.base_brightness)
                node.rest_position = self.config.base_brightness
                
                # Row 1 (panel 2) has different characteristics
                if row == 1:
                    node.amplitude_mult = 1.0 - self.config.row_amplitude_diff  # Reduced response
                    node.phase_offset = self.config.row_phase_offset
                    node.response_delay = self.config.row_delay
                else:
                    node.amplitude_mult = 1.0 + self.config.row_amplitude_diff * 0.5  # Enhanced response
                    node.phase_offset = 0.0
                    node.response_delay = 0.0
                
                # Per-column variation for organic feel
                node.amplitude_mult *= (1.0 + (random.random() - 0.5) * self.config.column_variation)
                node.phase_offset += random.random() * 0.3
                
                row_nodes.append(node)
            self.grid.append(row_nodes)
        
        self._setup_neighbors()
        
        # Create back panels
        self.back_panels = [BackPanel(col=i, phase=random.random() * math.pi * 2) for i in range(4)]
        
        # State
        self.input = InputState()
        self.status = NarrativeStatus()
        self.last_engagement_time = time.time() - 300
        self.engagement_level = 0.0
        
        # Ambient animation
        self.ambient_phase = 0.0
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX_VALUE] * TOTAL_CHANNELS
        
        # Timing
        self.last_update = time.time()
        self.frame_count = 0
    
    def _setup_neighbors(self):
        """Setup neighbors with differentiated coupling strengths"""
        for row in range(2):
            for col in range(4):
                node = self.grid[row][col]
                neighbors = []
                couplings = []
                
                # Horizontal neighbors (same row) - stronger coupling
                if col > 0:
                    neighbors.append(self.grid[row][col - 1])
                    couplings.append(self.config.horizontal_coupling)
                if col < 3:
                    neighbors.append(self.grid[row][col + 1])
                    couplings.append(self.config.horizontal_coupling)
                
                # Vertical neighbors (different row) - weaker coupling
                if row > 0:
                    neighbors.append(self.grid[row - 1][col])
                    couplings.append(self.config.vertical_coupling)
                if row < 1:
                    neighbors.append(self.grid[row + 1][col])
                    couplings.append(self.config.vertical_coupling)
                
                node.neighbors = neighbors
                node.neighbor_couplings = couplings
    
    def init_artnet(self) -> bool:
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, TOTAL_CHANNELS, FPS)
            self.artnet.start()
            return True
        except Exception as e:
            print(f"Art-Net initialization failed: {e}")
            return False
    
    def get_time_factors(self) -> dict:
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        
        rush_factor = 0.0
        if MORNING_RUSH_START <= hour <= MORNING_RUSH_END:
            mid = (MORNING_RUSH_START + MORNING_RUSH_END) / 2
            rush_factor = 1.0 - abs(hour - mid) / ((MORNING_RUSH_END - MORNING_RUSH_START) / 2)
        elif EVENING_RUSH_START <= hour <= EVENING_RUSH_END:
            mid = (EVENING_RUSH_START + EVENING_RUSH_END) / 2
            rush_factor = 1.0 - abs(hour - mid) / ((EVENING_RUSH_END - EVENING_RUSH_START) / 2)
        
        evening_factor = 0.0
        if hour >= EVENING_START:
            if hour >= NIGHT_START:
                evening_factor = 1.0
            else:
                evening_factor = (hour - EVENING_START) / (NIGHT_START - EVENING_START)
        
        if hour < 6:
            daytime_brightness = 0.3
        elif hour < 8:
            daytime_brightness = 0.3 + (hour - 6) * 0.35
        elif hour < 18:
            daytime_brightness = 1.0
        elif hour < 22:
            daytime_brightness = 1.0 - (hour - 18) * 0.15
        else:
            daytime_brightness = 0.4
        
        return {
            'rush_factor': rush_factor,
            'evening_factor': evening_factor,
            'daytime_brightness': daytime_brightness,
            'hour': hour
        }
    
    def calculate_acknowledgment_boost(self) -> float:
        time_since = time.time() - self.last_engagement_time
        time_factors = self.get_time_factors()
        
        time_boost = min(time_since / self.config.acknowledgment_decay_time, 1.0)
        time_boost *= self.config.acknowledgment_boost_max
        
        if time_factors['rush_factor'] < 0.3:
            time_boost *= 1.5
        if time_factors['evening_factor'] > 0.5:
            time_boost *= 1.3
        
        return 1.0 + time_boost
    
    def apply_active_input(self, dt: float):
        """Apply forces with strong position-based and row differentiation"""
        if self.input.active_population <= 0:
            return
        
        boost = self.calculate_acknowledgment_boost()
        self.last_engagement_time = time.time()
        
        base_force = self.config.active_force_base * self.input.active_population
        dwell_force = self.config.active_force_dwell_mult * self.input.active_dwell_time
        total_force = (base_force + dwell_force) * boost
        
        center_col = self.input.active_position_x * 3  # 0-3 range
        
        for col in range(4):
            distance = abs(col - center_col)
            # SHARP falloff - panels far from position get much less
            falloff = max(0, 1.0 - distance * self.config.position_falloff)
            falloff = falloff ** 1.5  # Make it even sharper (exponential)
            col_force = total_force * falloff
            
            # Row 0 (Panel 1) - immediate, STRONG response
            self.grid[0][col].apply_force(col_force * 1.5)
            
            # Row 1 (Panel 2) - delayed, INVERTED or weaker response for contrast
            delay_frames = int(self.config.row_delay * FPS)
            # Sometimes inverted, sometimes weaker - creates visual interest
            phase_mod = math.sin(self.ambient_phase * 2 + col * 0.8)
            if phase_mod > 0.3:
                row1_force = col_force * 0.3  # Much weaker
            elif phase_mod < -0.3:
                row1_force = -col_force * 0.4  # Inverted!
            else:
                row1_force = col_force * 0.5  # Moderate
            self.grid[1][col].apply_force(row1_force, delay_frames)
    
    def apply_passive_input(self, dt: float):
        time_factors = self.get_time_factors()
        
        passive_energy = (
            self.input.passive_population * 0.5 +
            self.input.cyclist_count * 0.3 +
            self.input.vehicle_count * 0.2
        )
        passive_energy *= (1.0 + self.input.passive_flow_speed * 0.2)
        passive_energy *= (1.0 - time_factors['rush_factor'] * 0.5)
        
        force = self.config.passive_force_base * passive_energy * 0.1
        direction = self.input.passive_flow_direction
        
        for row in range(2):
            row_mult = 1.0 if row == 0 else 0.7  # Row differentiation
            for col in range(4):
                col_bias = (col - 1.5) / 1.5 * direction * 0.3
                self.grid[row][col].apply_force(force * (1 + col_bias) * row_mult)
        
        # Update rest positions
        base = self.config.base_brightness * time_factors['daytime_brightness']
        passive_offset = passive_energy * self.config.passive_brightness_mult
        
        for row in range(2):
            for col in range(4):
                # Different rest positions per row
                row_offset = 0 if row == 0 else -1.5
                self.grid[row][col].rest_position = base + passive_offset + row_offset
    
    def apply_ambient_animation(self, dt: float):
        """Add subtle ambient waves for visual interest"""
        self.ambient_phase += self.config.ambient_wave_speed * dt
        
        for row in range(2):
            for col in range(4):
                node = self.grid[row][col]
                # Different wave patterns per row
                if row == 0:
                    wave = math.sin(self.ambient_phase + col * 0.8 + node.phase_offset)
                else:
                    wave = math.sin(self.ambient_phase * 0.7 + col * 0.6 + node.phase_offset + math.pi * 0.3)
                
                ambient_force = wave * self.config.ambient_wave_amplitude * 0.1
                node.apply_force(ambient_force)
    
    def update_engagement_level(self, dt: float):
        if self.input.active_population > 0:
            target = min(1.0, (
                self.input.active_population * 0.4 +
                min(self.input.active_dwell_time / 10.0, 1.0) * 0.4 +
                0.2
            ))
            self.engagement_level += (target - self.engagement_level) * 0.1
        else:
            decay = self.config.engagement_decay_rate * dt
            self.engagement_level = max(0, self.engagement_level - decay)
    
    def update_narrative_status(self, time_factors: dict):
        s = self.status
        boost = self.calculate_acknowledgment_boost()
        
        if self.engagement_level >= 0.8:
            s.state = "Peak"
        elif self.engagement_level >= 0.5:
            s.state = "Engaged"
        elif self.engagement_level >= 0.2:
            s.state = "Acknowledging"
        else:
            s.state = "Ambient"
        
        if self.input.active_population > 0:
            people = self.input.active_population
            dwell = self.input.active_dwell_time
            
            if dwell < 2:
                s.action = f"Noticing {'someone' if people == 1 else f'{people} people'}"
                s.reason = "New presence detected"
                s.mood = "Attentive"
            elif dwell < 5:
                s.action = f"Responding to {'visitor' if people == 1 else 'visitors'}"
                s.reason = f"Dwelling for {dwell:.1f}s"
                s.mood = "Engaged"
            elif dwell < 10:
                s.action = "Building engagement"
                s.reason = f"Sustained presence ({dwell:.1f}s)"
                s.mood = "Connected"
            else:
                s.action = "Deep engagement"
                s.reason = f"Long dwell time ({dwell:.1f}s)"
                s.mood = "Resonant"
            
            pos = self.input.active_position_x
            s.detail = "Left" if pos < 0.25 else "Right" if pos > 0.75 else "Center"
                
        elif self.input.time_since_active < 5:
            s.action = "Settling after visitor departed"
            s.reason = "Visitor just left"
            s.mood = "Contemplative"
            s.detail = "Springs returning to rest"
        else:
            s.action = "Ambient breathing"
            s.reason = "Waiting for visitors"
            s.mood = "Calm"
            s.detail = ""
    
    def update(self):
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        self.frame_count += 1
        
        time_factors = self.get_time_factors()
        
        if self.input.active_population > 0:
            self.input.time_since_active = 0
            self.input.active_dwell_time += dt
        else:
            self.input.time_since_active += dt
            self.input.active_dwell_time = 0
        
        # Apply all inputs
        self.apply_active_input(dt)
        self.apply_passive_input(dt)
        self.apply_ambient_animation(dt)
        self.update_engagement_level(dt)
        
        # Update spring physics
        for row in range(2):
            for col in range(4):
                self.grid[row][col].update(self.config, dt)
        
        # Update back panels
        passive_energy = self.input.passive_population + self.input.cyclist_count * 0.5
        for panel in self.back_panels:
            panel.update(dt, time_factors['evening_factor'], passive_energy)
        
        # Send to Art-Net
        self._send_artnet()
        self.update_narrative_status(time_factors)
        
        return time_factors
    
    def _send_artnet(self):
        if not self.artnet:
            return
        
        # DMX values are used directly (1-50 range)
        # No mapping needed - spring positions ARE the DMX values
        def clamp_dmx(val: float) -> int:
            return max(MIN_DMX_VALUE, min(MAX_DMX_VALUE, int(val)))
        
        for unit in range(4):
            col = 3 - unit
            
            ch_panel1 = unit * 3 + 0
            self.channel_values[ch_panel1] = clamp_dmx(self.grid[0][col].brightness)
            
            ch_panel2 = unit * 3 + 1
            self.channel_values[ch_panel2] = clamp_dmx(self.grid[1][col].brightness)
            
            ch_panel3 = unit * 3 + 2
            self.channel_values[ch_panel3] = clamp_dmx(self.back_panels[col].brightness)
        
        self.artnet.set(self.channel_values)
    
    def shutdown(self):
        if self.artnet:
            for i in range(len(self.channel_values)):
                self.channel_values[i] = MIN_DMX_VALUE
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


# =============================================================================
# GUI APPLICATION - Enhanced with spring visualization and tuning
# =============================================================================

class SpringControllerGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("Spring LED Controller v2 - Enhanced Contrast")
        self.root.geometry("1400x950")
        
        self.controller = SpringController()
        
        if not no_artnet:
            if not self.controller.init_artnet():
                print("Warning: Art-Net failed to initialize")
        
        self.setup_gui()
        
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Main view
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main View")
        self.setup_main_tab(main_tab)
        
        # Tab 2: Spring Tuning
        tuning_tab = ttk.Frame(notebook)
        notebook.add(tuning_tab, text="Spring Tuning")
        self.setup_tuning_tab(tuning_tab)
        
        # Tab 3: Spring Visualization
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="Spring Diagram")
        self.setup_spring_viz_tab(viz_tab)
    
    def setup_main_tab(self, parent):
        # Top: Panel visualization
        viz_frame = ttk.LabelFrame(parent, text="Panel Output (Brightness 1-50, DMX 1-212)", padding=10)
        viz_frame.pack(fill="both", expand=True, pady=(0, 10), padx=5)
        self.create_panel_visualization(viz_frame)
        
        # Middle row
        middle_frame = ttk.Frame(parent)
        middle_frame.pack(fill="both", expand=True, padx=5)
        
        # Input controls
        input_frame = ttk.LabelFrame(middle_frame, text="Input Controls", padding=10)
        input_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.create_input_controls(input_frame)
        
        # Status
        status_frame = ttk.LabelFrame(middle_frame, text="System Status", padding=10)
        status_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.create_status_display(status_frame)
        
        # Bottom: Narrative
        narrative_frame = ttk.LabelFrame(parent, text="Narrative Status", padding=10)
        narrative_frame.pack(fill="x", pady=(10, 5), padx=5)
        self.create_narrative_display(narrative_frame)
    
    def setup_tuning_tab(self, parent):
        """Create spring parameter tuning controls"""
        # Left: Spring parameters
        left_frame = ttk.Frame(parent)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Core spring params
        core_frame = ttk.LabelFrame(left_frame, text="Core Spring Parameters", padding=10)
        core_frame.pack(fill="x", pady=(0, 10))
        
        self.tuning_vars = {}
        
        params = [
            ("stiffness", "Stiffness", 0.01, 0.5, self.controller.config.stiffness),
            ("damping", "Damping", 0.5, 0.99, self.controller.config.damping),
            ("max_velocity", "Max Velocity", 1.0, 15.0, self.controller.config.max_velocity),
        ]
        
        for i, (key, label, min_v, max_v, default) in enumerate(params):
            self.create_tuning_slider(core_frame, key, label, min_v, max_v, default, i)
        
        # Coupling params
        coupling_frame = ttk.LabelFrame(left_frame, text="Coupling (Panel Independence)", padding=10)
        coupling_frame.pack(fill="x", pady=(0, 10))
        
        coupling_params = [
            ("horizontal_coupling", "Horizontal (same row)", 0.0, 0.5, self.controller.config.horizontal_coupling),
            ("vertical_coupling", "Vertical (between rows)", 0.0, 0.5, self.controller.config.vertical_coupling),
        ]
        
        for i, (key, label, min_v, max_v, default) in enumerate(coupling_params):
            self.create_tuning_slider(coupling_frame, key, label, min_v, max_v, default, i)
        
        # Row differentiation
        row_frame = ttk.LabelFrame(left_frame, text="Row Differentiation (Panel 1 vs 2)", padding=10)
        row_frame.pack(fill="x", pady=(0, 10))
        
        row_params = [
            ("row_delay", "Row 1 Delay (sec)", 0.0, 1.0, self.controller.config.row_delay),
            ("row_amplitude_diff", "Amplitude Difference", 0.0, 0.8, self.controller.config.row_amplitude_diff),
            ("row_phase_offset", "Phase Offset (sec)", 0.0, 2.0, self.controller.config.row_phase_offset),
        ]
        
        for i, (key, label, min_v, max_v, default) in enumerate(row_params):
            self.create_tuning_slider(row_frame, key, label, min_v, max_v, default, i)
        
        # Right side: Force params
        right_frame = ttk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        force_frame = ttk.LabelFrame(right_frame, text="Force Parameters", padding=10)
        force_frame.pack(fill="x", pady=(0, 10))
        
        force_params = [
            ("active_force_base", "Active Force Base", 10.0, 80.0, self.controller.config.active_force_base),
            ("active_force_dwell_mult", "Dwell Multiplier", 0.0, 5.0, self.controller.config.active_force_dwell_mult),
            ("passive_force_base", "Passive Force", 0.5, 15.0, self.controller.config.passive_force_base),
            ("position_falloff", "Position Sharpness", 0.2, 1.0, self.controller.config.position_falloff),
        ]
        
        for i, (key, label, min_v, max_v, default) in enumerate(force_params):
            self.create_tuning_slider(force_frame, key, label, min_v, max_v, default, i)
        
        # Ambient animation
        ambient_frame = ttk.LabelFrame(right_frame, text="Ambient Animation", padding=10)
        ambient_frame.pack(fill="x", pady=(0, 10))
        
        ambient_params = [
            ("ambient_wave_speed", "Wave Speed", 0.1, 2.0, self.controller.config.ambient_wave_speed),
            ("ambient_wave_amplitude", "Wave Amplitude", 0.0, 15.0, self.controller.config.ambient_wave_amplitude),
            ("column_variation", "Column Variation", 0.0, 0.6, self.controller.config.column_variation),
        ]
        
        for i, (key, label, min_v, max_v, default) in enumerate(ambient_params):
            self.create_tuning_slider(ambient_frame, key, label, min_v, max_v, default, i)
        
        # Presets
        preset_frame = ttk.LabelFrame(right_frame, text="Presets", padding=10)
        preset_frame.pack(fill="x")
        
        ttk.Button(preset_frame, text="High Contrast", command=self.preset_high_contrast).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Smooth/Uniform", command=self.preset_smooth).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Organic", command=self.preset_organic).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Reset Default", command=self.preset_default).pack(side="left", padx=5)
    
    def create_tuning_slider(self, parent, key, label, min_v, max_v, default, row):
        """Create a labeled slider for parameter tuning"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=20, anchor="w").pack(side="left")
        
        var = tk.DoubleVar(value=default)
        self.tuning_vars[key] = var
        
        value_label = ttk.Label(frame, text=f"{default:.3f}", width=8)
        value_label.pack(side="right")
        
        def on_change(val, k=key, v=var, lbl=value_label):
            value = float(val)
            lbl.config(text=f"{value:.3f}")
            setattr(self.controller.config, k, value)
            # Re-setup neighbors if coupling changed
            if 'coupling' in k:
                self.controller._setup_neighbors()
        
        slider = ttk.Scale(frame, from_=min_v, to=max_v, variable=var,
                          orient="horizontal", command=on_change)
        slider.pack(side="left", fill="x", expand=True, padx=10)
    
    def preset_high_contrast(self):
        """Maximum differentiation between panels - DRAMATIC"""
        presets = {
            'stiffness': 0.25,
            'damping': 0.65,  # Low damping = more oscillation
            'horizontal_coupling': 0.02,  # Almost independent
            'vertical_coupling': 0.01,
            'row_delay': 0.6,
            'row_amplitude_diff': 0.7,
            'row_phase_offset': 1.0,
            'active_force_base': 50.0,  # Very strong
            'max_velocity': 12.0,  # Fast movement
            'ambient_wave_amplitude': 8.0,
            'position_falloff': 0.8,  # Sharp focus
        }
        self.apply_presets(presets)
    
    def preset_smooth(self):
        """More uniform, synchronized behavior"""
        presets = {
            'stiffness': 0.12,
            'damping': 0.92,
            'horizontal_coupling': 0.3,
            'vertical_coupling': 0.25,
            'row_delay': 0.1,
            'row_amplitude_diff': 0.15,
            'row_phase_offset': 0.2,
            'active_force_base': 15.0,
            'ambient_wave_amplitude': 2.0,
        }
        self.apply_presets(presets)
    
    def preset_organic(self):
        """Natural, chaotic, alive feel"""
        presets = {
            'stiffness': 0.15,
            'damping': 0.72,  # Bouncy
            'horizontal_coupling': 0.08,
            'vertical_coupling': 0.04,
            'row_delay': 0.45,
            'row_amplitude_diff': 0.5,
            'row_phase_offset': 0.8,
            'active_force_base': 35.0,
            'max_velocity': 10.0,
            'column_variation': 0.4,
            'ambient_wave_amplitude': 7.0,
            'ambient_wave_speed': 0.6,
            'position_falloff': 0.5,
        }
        self.apply_presets(presets)
    
    def preset_default(self):
        """Reset to defaults"""
        default_config = SpringConfig()
        for key, var in self.tuning_vars.items():
            if hasattr(default_config, key):
                var.set(getattr(default_config, key))
                setattr(self.controller.config, key, getattr(default_config, key))
        self.controller._setup_neighbors()
    
    def apply_presets(self, presets):
        for key, value in presets.items():
            if key in self.tuning_vars:
                self.tuning_vars[key].set(value)
                setattr(self.controller.config, key, value)
        self.controller._setup_neighbors()
    
    def setup_spring_viz_tab(self, parent):
        """Create spring network visualization"""
        self.spring_canvas = tk.Canvas(parent, bg="#1a1a2e", height=600)
        self.spring_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Legend
        legend_frame = ttk.Frame(parent)
        legend_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(legend_frame, text="● Node size = brightness | ─ Line thickness = coupling strength | Color = velocity").pack()
    
    def create_panel_visualization(self, parent):
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)
        
        self.panel_canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", height=280)
        self.panel_canvas.pack(fill="both", expand=True)
        
        self.panel_rects = []
        self.panel_labels = []
        self.panel_bars = []  # Brightness bars
        
        panel_width = 70
        panel_height = 70
        spacing = 15
        start_x = 40
        start_y = 40
        
        for unit in range(4):
            unit_x = start_x + unit * (panel_width * 3 + spacing * 3 + 40)
            
            self.panel_canvas.create_text(
                unit_x + panel_width * 1.5 + spacing,
                start_y - 20,
                text=f"Unit {unit + 1}",
                fill="#888888",
                font=("Helvetica", 10, "bold")
            )
            
            for panel in range(3):
                x = unit_x + panel * (panel_width + spacing)
                y = start_y
                
                # Panel background
                rect = self.panel_canvas.create_rectangle(
                    x, y, x + panel_width, y + panel_height,
                    fill="#333333", outline="#555555", width=2
                )
                self.panel_rects.append(rect)
                
                # Brightness bar (vertical)
                bar = self.panel_canvas.create_rectangle(
                    x + 5, y + panel_height - 5,
                    x + 15, y + panel_height - 5,
                    fill="#00ff00", outline=""
                )
                self.panel_bars.append(bar)
                
                # Label
                label_text = f"P{panel + 1}\n--\n--"
                label = self.panel_canvas.create_text(
                    x + panel_width / 2 + 5, y + panel_height / 2,
                    text=label_text, fill="#ffffff",
                    font=("Courier", 9, "bold")
                )
                self.panel_labels.append(label)
                
                # Channel number
                ch = unit * 3 + panel
                self.panel_canvas.create_text(
                    x + panel_width / 2, y + panel_height + 12,
                    text=f"CH{ch + 1}",
                    fill="#666666",
                    font=("Helvetica", 8)
                )
        
        # Row labels
        self.panel_canvas.create_text(
            start_x - 25, start_y + 20,
            text="P1\nP2\nP3",
            fill="#666666",
            font=("Helvetica", 8)
        )
    
    def create_input_controls(self, parent):
        # Active Zone
        active_frame = ttk.LabelFrame(parent, text="Active Zone", padding=10)
        active_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(active_frame, text="Population:").grid(row=0, column=0, sticky="w", pady=2)
        self.active_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(active_frame, from_=0, to=10, textvariable=self.active_pop_var,
                   width=10, command=self.on_active_change).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        ttk.Label(active_frame, text="Position X:").grid(row=1, column=0, sticky="w", pady=2)
        self.active_pos_var = tk.DoubleVar(value=0.5)
        pos_frame = ttk.Frame(active_frame)
        pos_frame.grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Label(pos_frame, text="L").pack(side="left")
        ttk.Scale(pos_frame, from_=0, to=1, variable=self.active_pos_var,
                 orient="horizontal", command=lambda v: self.on_active_change()).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(pos_frame, text="R").pack(side="left")
        
        btn_frame = ttk.Frame(active_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame, text="1 Center", command=lambda: self.set_active(1, 0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="2 Left", command=lambda: self.set_active(2, 0.2)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="3 Right", command=lambda: self.set_active(3, 0.8)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Clear", command=self.clear_active).pack(side="left", padx=2)
        
        active_frame.columnconfigure(1, weight=1)
        
        # Passive Zone
        passive_frame = ttk.LabelFrame(parent, text="Passive Zone", padding=10)
        passive_frame.pack(fill="x")
        
        ttk.Label(passive_frame, text="Population:").grid(row=0, column=0, sticky="w", pady=2)
        self.passive_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(passive_frame, from_=0, to=50, textvariable=self.passive_pop_var,
                   width=10, command=self.on_passive_change).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        ttk.Label(passive_frame, text="Cyclists:").grid(row=1, column=0, sticky="w", pady=2)
        self.cyclist_var = tk.IntVar(value=0)
        ttk.Spinbox(passive_frame, from_=0, to=20, textvariable=self.cyclist_var,
                   width=10, command=self.on_passive_change).grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        preset_frame = ttk.Frame(passive_frame)
        preset_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(preset_frame, text="Quiet", command=lambda: self.set_passive(2, 0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Busy", command=lambda: self.set_passive(15, 3)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Clear", command=self.clear_passive).pack(side="left", padx=2)
        
        passive_frame.columnconfigure(1, weight=1)
    
    def create_status_display(self, parent):
        ttk.Label(parent, text="Engagement Level:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.engagement_bar = ttk.Progressbar(parent, mode='determinate', length=200)
        self.engagement_bar.pack(fill="x", pady=(0, 5))
        self.engagement_label = ttk.Label(parent, text="0%")
        self.engagement_label.pack(anchor="w", pady=(0, 10))
        
        ttk.Label(parent, text="Panel Contrast:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.contrast_label = ttk.Label(parent, text="Row diff: 0.0 | Col spread: 0.0")
        self.contrast_label.pack(anchor="w", pady=(0, 10))
        
        ttk.Label(parent, text="Time Context:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.time_info = tk.Text(parent, height=6, width=30, wrap="word", bg="#f0f0f0")
        self.time_info.pack(fill="both", expand=True)
        self.time_info.config(state="disabled")
    
    def create_narrative_display(self, parent):
        self.narrative_text = tk.Text(parent, height=4, wrap="word", bg="#2d2d2d", fg="#ffffff", font=("Courier", 10))
        self.narrative_text.pack(fill="both", expand=True)
        self.narrative_text.config(state="disabled")
    
    def set_active(self, pop, pos):
        self.active_pop_var.set(pop)
        self.active_pos_var.set(pos)
        self.on_active_change()
    
    def clear_active(self):
        self.active_pop_var.set(0)
        self.on_active_change()
    
    def set_passive(self, pop, cyclists):
        self.passive_pop_var.set(pop)
        self.cyclist_var.set(cyclists)
        self.on_passive_change()
    
    def clear_passive(self):
        self.passive_pop_var.set(0)
        self.cyclist_var.set(0)
        self.on_passive_change()
    
    def on_active_change(self):
        self.controller.input.active_population = self.active_pop_var.get()
        self.controller.input.active_position_x = self.active_pos_var.get()
    
    def on_passive_change(self):
        self.controller.input.passive_population = self.passive_pop_var.get()
        self.controller.input.cyclist_count = self.cyclist_var.get()
    
    def update_panel_display(self):
        for unit in range(4):
            col = 3 - unit
            
            for panel in range(3):
                idx = unit * 3 + panel
                
                if panel < 2:
                    brightness = self.controller.grid[panel][col].brightness
                else:
                    brightness = self.controller.back_panels[col].brightness
                
                dmx_value = self.controller.channel_values[idx]
                
                # Color based on brightness
                intensity = int((brightness - MIN_DMX_VALUE) / (MAX_DMX_VALUE - MIN_DMX_VALUE) * 200 + 55)
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                
                self.panel_canvas.itemconfig(self.panel_rects[idx], fill=color)
                
                # Update label with brightness and DMX
                label_text = f"P{panel + 1}\nB:{brightness:.1f}\nD:{dmx_value}"
                self.panel_canvas.itemconfig(self.panel_labels[idx], text=label_text)
                
                # Update brightness bar
                bar_height = (brightness - MIN_DMX_VALUE) / (MAX_DMX_VALUE - MIN_DMX_VALUE) * 60
                x = 40 + unit * (70 * 3 + 15 * 3 + 40) + panel * (70 + 15)
                y_bottom = 40 + 70 - 5
                self.panel_canvas.coords(
                    self.panel_bars[idx],
                    x + 5, y_bottom - bar_height,
                    x + 15, y_bottom
                )
                
                # Bar color based on velocity
                if panel < 2:
                    vel = abs(self.controller.grid[panel][col].velocity)
                else:
                    vel = 0
                if vel > 1.5:
                    bar_color = "#ff4444"
                elif vel > 0.5:
                    bar_color = "#ffaa00"
                else:
                    bar_color = "#44ff44"
                self.panel_canvas.itemconfig(self.panel_bars[idx], fill=bar_color)
    
    def update_spring_visualization(self):
        """Draw the spring network"""
        self.spring_canvas.delete("all")
        
        canvas_w = self.spring_canvas.winfo_width()
        canvas_h = self.spring_canvas.winfo_height()
        
        if canvas_w < 100:
            return
        
        # Calculate node positions
        node_spacing_x = canvas_w / 5
        node_spacing_y = canvas_h / 4
        start_x = node_spacing_x
        start_y = node_spacing_y
        
        node_positions = {}
        
        # Draw connections first (so they're behind nodes)
        for row in range(2):
            for col in range(4):
                x = start_x + col * node_spacing_x
                y = start_y + row * node_spacing_y
                node_positions[(row, col)] = (x, y)
                
                node = self.controller.grid[row][col]
                
                # Draw springs to neighbors
                for i, neighbor in enumerate(node.neighbors):
                    nx, ny = node_positions.get((neighbor.row, neighbor.col), (0, 0))
                    if nx > 0:
                        coupling = node.neighbor_couplings[i] if i < len(node.neighbor_couplings) else 0.1
                        line_width = max(1, int(coupling * 20))
                        
                        # Color based on tension
                        tension = abs(node.position - neighbor.position)
                        if tension > 10:
                            line_color = "#ff4444"
                        elif tension > 5:
                            line_color = "#ffaa00"
                        else:
                            line_color = "#4466aa"
                        
                        self.spring_canvas.create_line(
                            x, y, nx, ny,
                            fill=line_color, width=line_width,
                            dash=(4, 2)
                        )
        
        # Draw nodes
        for row in range(2):
            for col in range(4):
                x, y = node_positions[(row, col)]
                node = self.controller.grid[row][col]
                
                # Node size based on brightness
                radius = 10 + (node.brightness - MIN_DMX_VALUE) / (MAX_DMX_VALUE - MIN_DMX_VALUE) * 30
                
                # Color based on velocity
                vel = node.velocity
                if vel > 0:
                    r = min(255, int(100 + vel * 50))
                    g = 100
                    b = 100
                else:
                    r = 100
                    g = 100
                    b = min(255, int(100 - vel * 50))
                color = f"#{r:02x}{g:02x}{b:02x}"
                
                self.spring_canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color, outline="#ffffff", width=2
                )
                
                # Label
                self.spring_canvas.create_text(
                    x, y,
                    text=f"{node.brightness:.1f}",
                    fill="#ffffff",
                    font=("Helvetica", 9, "bold")
                )
                
                # Row/Col label
                self.spring_canvas.create_text(
                    x, y - radius - 10,
                    text=f"R{row+1}C{col+1}",
                    fill="#888888",
                    font=("Helvetica", 8)
                )
        
        # Draw back panels
        back_y = start_y + 2.5 * node_spacing_y
        for col in range(4):
            x = start_x + col * node_spacing_x
            panel = self.controller.back_panels[col]
            
            radius = 10 + (panel.brightness - MIN_DMX_VALUE) / (MAX_DMX_VALUE - MIN_DMX_VALUE) * 25
            
            self.spring_canvas.create_oval(
                x - radius, back_y - radius,
                x + radius, back_y + radius,
                fill="#666688", outline="#aaaacc", width=2
            )
            
            self.spring_canvas.create_text(
                x, back_y,
                text=f"{panel.brightness:.1f}",
                fill="#ffffff",
                font=("Helvetica", 9)
            )
            
            self.spring_canvas.create_text(
                x, back_y - radius - 10,
                text=f"Back {col+1}",
                fill="#888888",
                font=("Helvetica", 8)
            )
    
    def update_status_display(self, time_factors):
        engagement_pct = self.controller.engagement_level * 100
        self.engagement_bar['value'] = engagement_pct
        self.engagement_label.config(text=f"{engagement_pct:.1f}%")
        
        # Calculate contrast metrics
        row0_avg = sum(n.brightness for n in self.controller.grid[0]) / 4
        row1_avg = sum(n.brightness for n in self.controller.grid[1]) / 4
        row_diff = abs(row0_avg - row1_avg)
        
        all_vals = [n.brightness for row in self.controller.grid for n in row]
        col_spread = max(all_vals) - min(all_vals)
        
        self.contrast_label.config(text=f"Row diff: {row_diff:.1f} | Range: {col_spread:.1f}")
        
        # Time info
        self.time_info.config(state="normal")
        self.time_info.delete(1.0, tk.END)
        
        now = datetime.now()
        info = f"Time: {now.strftime('%H:%M:%S')}\n"
        info += f"Rush: {time_factors['rush_factor']:.2f}\n"
        info += f"Evening: {time_factors['evening_factor']:.2f}\n"
        info += f"Since Active: {self.controller.input.time_since_active:.1f}s\n"
        info += f"Dwell: {self.controller.input.active_dwell_time:.1f}s"
        
        self.time_info.insert(1.0, info)
        self.time_info.config(state="disabled")
    
    def update_narrative_display(self):
        s = self.controller.status
        
        self.narrative_text.config(state="normal")
        self.narrative_text.delete(1.0, tk.END)
        
        text = f"STATE: {s.state} | MOOD: {s.mood}\n"
        text += f"ACTION: {s.action}\n"
        text += f"REASON: {s.reason}"
        if s.detail:
            text += f" | {s.detail}"
        
        self.narrative_text.insert(1.0, text)
        self.narrative_text.config(state="disabled")
    
    def update_loop(self):
        if not self.running:
            return
        
        time_factors = self.controller.update()
        
        self.update_panel_display()
        self.update_spring_visualization()
        self.update_status_display(time_factors)
        self.update_narrative_display()
        
        self.root.after(int(1000 / FPS), self.update_loop)
    
    def on_closing(self):
        self.running = False
        self.controller.shutdown()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spring LED Controller GUI v2')
    parser.add_argument('--no-artnet', action='store_true', help='Run without Art-Net')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = SpringControllerGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
