#!/usr/bin/env python3
"""
Spring-Based LED Panel Controller with GUI
Interactive interface showing real-time panel values and input controls.
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
# CONFIGURATION (same as springController.py)
# =============================================================================

TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

NUM_UNITS = 4
PANELS_PER_UNIT = 3
TOTAL_CHANNELS = NUM_UNITS * PANELS_PER_UNIT

MAX_DMX_VALUE = 212
MIN_DMX_VALUE = 1

MIN_BRIGHTNESS = 1
MAX_BRIGHTNESS = 50

# Time of day parameters
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 16
EVENING_RUSH_END = 19
EVENING_START = 18
NIGHT_START = 22

# =============================================================================
# SPRING PHYSICS
# =============================================================================

@dataclass
class SpringConfig:
    stiffness: float = 0.15
    damping: float = 0.85
    coupling_strength: float = 0.3
    active_force_base: float = 15.0
    active_force_dwell_mult: float = 0.5
    passive_force_base: float = 2.0
    engagement_decay_rate: float = 0.02
    acknowledgment_boost_max: float = 3.0
    acknowledgment_decay_time: float = 300.0
    max_velocity: float = 2.0
    base_brightness: float = 5.0
    passive_brightness_mult: float = 0.5


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
    position: float = 0.0
    velocity: float = 0.0
    rest_position: float = 5.0
    force: float = 0.0
    neighbors: List['SpringNode'] = field(default_factory=list)
    
    def apply_force(self, f: float):
        self.force += f
    
    def update(self, config: SpringConfig, dt: float):
        spring_force = -config.stiffness * (self.position - self.rest_position)
        
        coupling_force = 0.0
        for neighbor in self.neighbors:
            coupling_force += config.coupling_strength * (neighbor.position - self.position)
        
        acceleration = spring_force + coupling_force + self.force
        self.velocity = (self.velocity + acceleration * dt) * config.damping
        self.velocity = max(-config.max_velocity, min(config.max_velocity, self.velocity))
        self.position += self.velocity * dt
        self.position = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, self.position))
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
        self.target = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, self.target))
        
        diff = self.target - self.brightness
        self.brightness += diff * 0.05
        self.brightness = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, self.brightness))


# =============================================================================
# SPRING CONTROLLER (headless)
# =============================================================================

class SpringController:
    def __init__(self, config: Optional[SpringConfig] = None):
        self.config = config or SpringConfig()
        
        # Create front grid
        self.grid: List[List[SpringNode]] = []
        for row in range(2):
            row_nodes = []
            for col in range(4):
                node = SpringNode(row=row, col=col, position=self.config.base_brightness)
                node.rest_position = self.config.base_brightness
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
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX_VALUE] * TOTAL_CHANNELS
        
        # Timing
        self.last_update = time.time()
    
    def _setup_neighbors(self):
        for row in range(2):
            for col in range(4):
                node = self.grid[row][col]
                neighbors = []
                if col > 0:
                    neighbors.append(self.grid[row][col - 1])
                if col < 3:
                    neighbors.append(self.grid[row][col + 1])
                if row > 0:
                    neighbors.append(self.grid[row - 1][col])
                if row < 1:
                    neighbors.append(self.grid[row + 1][col])
                node.neighbors = neighbors
    
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
    
    def apply_active_input(self):
        if self.input.active_population <= 0:
            return
        
        boost = self.calculate_acknowledgment_boost()
        self.last_engagement_time = time.time()
        
        base_force = self.config.active_force_base * self.input.active_population
        dwell_force = self.config.active_force_dwell_mult * self.input.active_dwell_time
        total_force = (base_force + dwell_force) * boost
        
        center_col = self.input.active_position_x * 3
        
        for col in range(4):
            distance = abs(col - center_col)
            falloff = max(0, 1.0 - distance * 0.4)
            col_force = total_force * falloff
            
            for row in range(2):
                self.grid[row][col].apply_force(col_force)
    
    def apply_passive_input(self):
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
            for col in range(4):
                col_bias = (col - 1.5) / 1.5 * direction * 0.3
                self.grid[row][col].apply_force(force * (1 + col_bias))
        
        base = self.config.base_brightness * time_factors['daytime_brightness']
        passive_offset = passive_energy * self.config.passive_brightness_mult
        
        for row in range(2):
            for col in range(4):
                self.grid[row][col].rest_position = base + passive_offset
    
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
        
        self.config.coupling_strength = 0.2 + self.engagement_level * 0.3
    
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
                s.action = f"Noticing {'someone' if people == 1 else f'{people} people'} approaching"
                if boost > 2.0:
                    s.reason = f"First visitor in {int(time.time() - self.last_engagement_time)}s!"
                else:
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
            if pos < 0.25:
                s.detail = "Left side"
            elif pos > 0.75:
                s.detail = "Right side"
            else:
                s.detail = "Center"
                
        elif self.input.time_since_active < 5:
            s.action = "Settling after visitor departed"
            s.reason = "Visitor just left the zone"
            s.mood = "Contemplative"
            s.detail = "Springs returning to rest"
            
        elif self.input.time_since_active < 30:
            s.action = "Fading back to ambient"
            s.reason = f"No one in active zone for {self.input.time_since_active:.0f}s"
            s.mood = "Relaxing"
            s.detail = ""
            
        else:
            passive_total = self.input.passive_population + self.input.cyclist_count
            
            if time_factors['evening_factor'] > 0.7:
                s.action = "Evening ambient mode"
                s.reason = "Night time - back panels more active"
                s.mood = "Nocturnal"
            elif time_factors['rush_factor'] > 0.5:
                s.action = "Rush hour ambient"
                s.reason = f"{passive_total} people in passive zones"
                s.mood = "Busy"
            elif passive_total > 10:
                s.action = "High ambient activity"
                s.reason = f"{passive_total} people nearby"
                s.mood = "Lively"
            elif passive_total > 3:
                s.action = "Moderate ambient activity"
                s.reason = f"{passive_total} people nearby"
                s.mood = "Aware"
            else:
                s.action = "Low ambient breathing"
                s.reason = "Quiet period"
                s.mood = "Calm"
            s.detail = ""
    
    def update(self):
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        
        time_factors = self.get_time_factors()
        
        if self.input.active_population > 0:
            self.input.time_since_active = 0
            self.input.active_dwell_time += dt
        else:
            self.input.time_since_active += dt
            self.input.active_dwell_time = 0
        
        self.apply_active_input()
        self.apply_passive_input()
        self.update_engagement_level(dt)
        
        for row in range(2):
            for col in range(4):
                self.grid[row][col].update(self.config, dt)
        
        passive_energy = self.input.passive_population + self.input.cyclist_count * 0.5
        for panel in self.back_panels:
            panel.update(dt, time_factors['evening_factor'], passive_energy)
        
        self._send_artnet()
        self.update_narrative_status(time_factors)
        
        return time_factors
    
    def _send_artnet(self):
        if not self.artnet:
            return
        
        def brightness_to_dmx(b: float) -> int:
            normalized = (b - MIN_BRIGHTNESS) / (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            dmx = int(MIN_DMX_VALUE + normalized * (MAX_DMX_VALUE - MIN_DMX_VALUE))
            return max(MIN_DMX_VALUE, min(MAX_DMX_VALUE, dmx))
        
        for unit in range(4):
            col = 3 - unit
            
            ch_panel1 = unit * 3 + 0
            self.channel_values[ch_panel1] = brightness_to_dmx(self.grid[0][col].brightness)
            
            ch_panel2 = unit * 3 + 1
            self.channel_values[ch_panel2] = brightness_to_dmx(self.grid[1][col].brightness)
            
            ch_panel3 = unit * 3 + 2
            self.channel_values[ch_panel3] = brightness_to_dmx(self.back_panels[col].brightness)
        
        self.artnet.set(self.channel_values)
    
    def shutdown(self):
        if self.artnet:
            for i in range(len(self.channel_values)):
                self.channel_values[i] = MIN_DMX_VALUE
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


# =============================================================================
# GUI APPLICATION
# =============================================================================

class SpringControllerGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("Spring LED Controller - Live View")
        self.root.geometry("1200x900")
        
        # Initialize controller
        self.controller = SpringController()
        
        if not no_artnet:
            if not self.controller.init_artnet():
                print("Warning: Art-Net failed to initialize")
        else:
            print("Running without Art-Net (test mode)")
        
        self.setup_gui()
        
        # Start update loop
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill="both", expand=True)
        
        # Top section: Panel visualization
        viz_frame = ttk.LabelFrame(main_container, text="Panel Output (Live)", padding=10)
        viz_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.create_panel_visualization(viz_frame)
        
        # Middle left: Input controls
        middle_frame = ttk.Frame(main_container)
        middle_frame.pack(fill="both", expand=True)
        
        input_frame = ttk.LabelFrame(middle_frame, text="Input Controls", padding=10)
        input_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.create_input_controls(input_frame)
        
        # Middle right: Status
        status_frame = ttk.LabelFrame(middle_frame, text="System Status", padding=10)
        status_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.create_status_display(status_frame)
        
        # Bottom: Narrative status
        narrative_frame = ttk.LabelFrame(main_container, text="Narrative Status", padding=10)
        narrative_frame.pack(fill="x", pady=(10, 0))
        
        self.create_narrative_display(narrative_frame)
    
    def create_panel_visualization(self, parent):
        # Create canvas for visual representation
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)
        
        self.panel_canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", height=300)
        self.panel_canvas.pack(fill="both", expand=True)
        
        # Store references to canvas items
        self.panel_rects = []
        self.panel_labels = []
        
        # Draw 12 panels (4 units × 3 panels)
        panel_width = 80
        panel_height = 80
        spacing = 20
        start_x = 50
        start_y = 50
        
        for unit in range(4):
            unit_x = start_x + unit * (panel_width * 3 + spacing * 3 + 30)
            
            # Unit label
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
                
                # Panel rectangle
                rect = self.panel_canvas.create_rectangle(
                    x, y, x + panel_width, y + panel_height,
                    fill="#333333", outline="#555555", width=2
                )
                self.panel_rects.append(rect)
                
                # Panel label
                label_text = f"P{panel + 1}\n--"
                label = self.panel_canvas.create_text(
                    x + panel_width / 2, y + panel_height / 2,
                    text=label_text, fill="#ffffff",
                    font=("Courier", 12, "bold")
                )
                self.panel_labels.append(label)
                
                # Channel number
                ch = unit * 3 + panel
                self.panel_canvas.create_text(
                    x + panel_width / 2, y + panel_height + 10,
                    text=f"CH{ch + 1}",
                    fill="#666666",
                    font=("Helvetica", 8)
                )
        
        # Add legend
        legend_y = start_y + panel_height + 40
        self.panel_canvas.create_text(
            start_x, legend_y,
            text="Panel 1-2: Spring Grid (front) | Panel 3: Back (independent)",
            fill="#888888",
            font=("Helvetica", 9),
            anchor="w"
        )
    
    def create_input_controls(self, parent):
        # Active Zone controls
        active_frame = ttk.LabelFrame(parent, text="Active Zone (TrackZone 1-2)", padding=10)
        active_frame.pack(fill="x", pady=(0, 10))
        
        # Population
        ttk.Label(active_frame, text="Population:").grid(row=0, column=0, sticky="w", pady=2)
        self.active_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(
            active_frame, from_=0, to=10, textvariable=self.active_pop_var,
            width=10, command=self.on_active_change
        ).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        # Position X
        ttk.Label(active_frame, text="Position X:").grid(row=1, column=0, sticky="w", pady=2)
        self.active_pos_var = tk.DoubleVar(value=0.5)
        pos_frame = ttk.Frame(active_frame)
        pos_frame.grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Label(pos_frame, text="L").pack(side="left")
        pos_slider = ttk.Scale(
            pos_frame, from_=0, to=1, variable=self.active_pos_var,
            orient="horizontal", command=lambda v: self.on_active_change()
        )
        pos_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(pos_frame, text="R").pack(side="left")
        
        # Quick buttons
        btn_frame = ttk.Frame(active_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame, text="1 Person Center", command=lambda: self.set_active(1, 0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="2 People Left", command=lambda: self.set_active(2, 0.25)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Clear", command=self.clear_active).pack(side="left", padx=2)
        
        active_frame.columnconfigure(1, weight=1)
        
        # Passive Zone controls
        passive_frame = ttk.LabelFrame(parent, text="Passive Zone (TrackZone 3-10)", padding=10)
        passive_frame.pack(fill="x", pady=(0, 10))
        
        # Population
        ttk.Label(passive_frame, text="Population:").grid(row=0, column=0, sticky="w", pady=2)
        self.passive_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(
            passive_frame, from_=0, to=50, textvariable=self.passive_pop_var,
            width=10, command=self.on_passive_change
        ).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        # Cyclists
        ttk.Label(passive_frame, text="Cyclists:").grid(row=1, column=0, sticky="w", pady=2)
        self.cyclist_var = tk.IntVar(value=0)
        ttk.Spinbox(
            passive_frame, from_=0, to=20, textvariable=self.cyclist_var,
            width=10, command=self.on_passive_change
        ).grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        # Vehicles
        ttk.Label(passive_frame, text="Vehicles:").grid(row=2, column=0, sticky="w", pady=2)
        self.vehicle_var = tk.IntVar(value=0)
        ttk.Spinbox(
            passive_frame, from_=0, to=10, textvariable=self.vehicle_var,
            width=10, command=self.on_passive_change
        ).grid(row=2, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        # Flow speed
        ttk.Label(passive_frame, text="Flow Speed:").grid(row=3, column=0, sticky="w", pady=2)
        self.flow_speed_var = tk.DoubleVar(value=0)
        ttk.Scale(
            passive_frame, from_=0, to=3, variable=self.flow_speed_var,
            orient="horizontal", command=lambda v: self.on_passive_change()
        ).grid(row=3, column=1, sticky="ew", pady=2, padx=(5, 0))
        
        # Flow direction
        ttk.Label(passive_frame, text="Flow Direction:").grid(row=4, column=0, sticky="w", pady=2)
        self.flow_dir_var = tk.DoubleVar(value=0)
        dir_frame = ttk.Frame(passive_frame)
        dir_frame.grid(row=4, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Label(dir_frame, text="←").pack(side="left")
        ttk.Scale(
            dir_frame, from_=-1, to=1, variable=self.flow_dir_var,
            orient="horizontal", command=lambda v: self.on_passive_change()
        ).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(dir_frame, text="→").pack(side="left")
        
        # Quick presets
        preset_frame = ttk.Frame(passive_frame)
        preset_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(preset_frame, text="Quiet", command=lambda: self.set_passive(2, 0, 0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Moderate", command=lambda: self.set_passive(8, 2, 1)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Busy", command=lambda: self.set_passive(20, 5, 3)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Clear", command=self.clear_passive).pack(side="left", padx=2)
        
        passive_frame.columnconfigure(1, weight=1)
    
    def create_status_display(self, parent):
        # Engagement level
        ttk.Label(parent, text="Engagement Level:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.engagement_bar = ttk.Progressbar(parent, mode='determinate', length=200)
        self.engagement_bar.pack(fill="x", pady=(0, 5))
        
        self.engagement_label = ttk.Label(parent, text="0%")
        self.engagement_label.pack(anchor="w", pady=(0, 10))
        
        # Time factors
        ttk.Label(parent, text="Time Context:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.time_info = tk.Text(parent, height=8, width=30, wrap="word", bg="#f0f0f0")
        self.time_info.pack(fill="both", expand=True)
        self.time_info.config(state="disabled")
        
        # Spring physics
        ttk.Label(parent, text="Spring Coupling:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.coupling_label = ttk.Label(parent, text="0.30")
        self.coupling_label.pack(anchor="w")
    
    def create_narrative_display(self, parent):
        self.narrative_text = tk.Text(parent, height=6, wrap="word", bg="#2d2d2d", fg="#ffffff", font=("Courier", 10))
        self.narrative_text.pack(fill="both", expand=True)
        self.narrative_text.config(state="disabled")
    
    def set_active(self, population, position):
        self.active_pop_var.set(population)
        self.active_pos_var.set(position)
        self.on_active_change()
    
    def clear_active(self):
        self.active_pop_var.set(0)
        self.on_active_change()
    
    def set_passive(self, population, cyclists, vehicles):
        self.passive_pop_var.set(population)
        self.cyclist_var.set(cyclists)
        self.vehicle_var.set(vehicles)
        self.flow_speed_var.set(1.5)
        self.flow_dir_var.set(random.choice([-0.5, 0, 0.5]))
        self.on_passive_change()
    
    def clear_passive(self):
        self.passive_pop_var.set(0)
        self.cyclist_var.set(0)
        self.vehicle_var.set(0)
        self.flow_speed_var.set(0)
        self.flow_dir_var.set(0)
        self.on_passive_change()
    
    def on_active_change(self):
        self.controller.input.active_population = self.active_pop_var.get()
        self.controller.input.active_position_x = self.active_pos_var.get()
    
    def on_passive_change(self):
        self.controller.input.passive_population = self.passive_pop_var.get()
        self.controller.input.cyclist_count = self.cyclist_var.get()
        self.controller.input.vehicle_count = self.vehicle_var.get()
        self.controller.input.passive_flow_speed = self.flow_speed_var.get()
        self.controller.input.passive_flow_direction = self.flow_dir_var.get()
    
    def update_panel_display(self):
        # Update each panel's color and label based on brightness
        for unit in range(4):
            col = 3 - unit  # Reverse mapping
            
            for panel in range(3):
                idx = unit * 3 + panel
                
                if panel < 2:
                    # Front grid panels
                    brightness = self.controller.grid[panel][col].brightness
                else:
                    # Back panel
                    brightness = self.controller.back_panels[col].brightness
                
                # Map brightness (1-50) to DMX (1-212)
                dmx_value = self.controller.channel_values[idx]
                
                # Color mapping: darker at low values, brighter at high
                intensity = int((brightness - MIN_BRIGHTNESS) / (MAX_BRIGHTNESS - MIN_BRIGHTNESS) * 200 + 55)
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                
                # Update rectangle color
                self.panel_canvas.itemconfig(self.panel_rects[idx], fill=color)
                
                # Update label
                label_text = f"P{panel + 1}\n{brightness:.1f}\nDMX:{dmx_value}"
                self.panel_canvas.itemconfig(self.panel_labels[idx], text=label_text)
    
    def update_status_display(self, time_factors):
        # Engagement level
        engagement_pct = self.controller.engagement_level * 100
        self.engagement_bar['value'] = engagement_pct
        self.engagement_label.config(text=f"{engagement_pct:.1f}%")
        
        # Time info
        self.time_info.config(state="normal")
        self.time_info.delete(1.0, tk.END)
        
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        
        info_text = f"Current Time: {time_str}\n"
        info_text += f"Hour: {time_factors['hour']:.2f}\n\n"
        info_text += f"Rush Factor: {time_factors['rush_factor']:.2f}\n"
        info_text += f"Evening Factor: {time_factors['evening_factor']:.2f}\n"
        info_text += f"Daytime Brightness: {time_factors['daytime_brightness']:.2f}\n\n"
        info_text += f"Time Since Active: {self.controller.input.time_since_active:.1f}s\n"
        
        boost = self.controller.calculate_acknowledgment_boost()
        info_text += f"Ack Boost: {boost:.2f}x"
        
        self.time_info.insert(1.0, info_text)
        self.time_info.config(state="disabled")
        
        # Coupling strength
        self.coupling_label.config(text=f"{self.controller.config.coupling_strength:.3f}")
    
    def update_narrative_display(self):
        s = self.controller.status
        
        self.narrative_text.config(state="normal")
        self.narrative_text.delete(1.0, tk.END)
        
        narrative = f"STATE: {s.state}\n"
        narrative += f"ACTION: {s.action}\n"
        narrative += f"REASON: {s.reason}\n"
        narrative += f"MOOD: {s.mood}\n"
        if s.detail:
            narrative += f"DETAIL: {s.detail}\n"
        
        self.narrative_text.insert(1.0, narrative)
        self.narrative_text.config(state="disabled")
    
    def update_loop(self):
        if not self.running:
            return
        
        # Update controller
        time_factors = self.controller.update()
        
        # Update GUI
        self.update_panel_display()
        self.update_status_display(time_factors)
        self.update_narrative_display()
        
        # Schedule next update
        self.root.after(int(1000 / FPS), self.update_loop)
    
    def on_closing(self):
        self.running = False
        self.controller.shutdown()
        self.root.destroy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spring LED Controller GUI')
    parser.add_argument('--no-artnet', action='store_true', help='Run without Art-Net hardware')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = SpringControllerGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
