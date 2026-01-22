#!/usr/bin/env python3
"""
Wave Field LED Panel Controller
Creates constantly flowing, liquid-like patterns using layered waves and noise.

Unlike the spring system which settles to equilibrium, this is ALWAYS moving.
People inject ripples that spread and interact with the base wave field.
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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

# DMX output range for front panels (1-50)
MIN_DMX = 1
MAX_DMX = 50

# DMX output range for back panels (Panel 3) - full range
MIN_DMX_BACK = 1
MAX_DMX_BACK = 212

# Time of day
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 16
EVENING_RUSH_END = 19
EVENING_START = 18
NIGHT_START = 22


# =============================================================================
# PERLIN NOISE (simplified 1D/2D)
# =============================================================================

class PerlinNoise:
    """Simple Perlin-like noise generator for organic movement"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.permutation = list(range(256))
        random.shuffle(self.permutation)
        self.permutation = self.permutation * 2
        
    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)
    
    def _grad(self, hash_val: int, x: float) -> float:
        h = hash_val & 15
        grad = 1 + (h & 7)
        if h & 8:
            grad = -grad
        return grad * x
    
    def noise1d(self, x: float) -> float:
        """1D Perlin noise, returns value in [-1, 1]"""
        xi = int(x) & 255
        xf = x - int(x)
        
        u = self._fade(xf)
        
        a = self.permutation[xi]
        b = self.permutation[xi + 1]
        
        return self._lerp(self._grad(a, xf), self._grad(b, xf - 1), u)
    
    def noise2d(self, x: float, y: float) -> float:
        """2D Perlin-like noise using 1D noise combination"""
        return (self.noise1d(x + y * 0.7) + self.noise1d(y - x * 0.3)) * 0.5


# =============================================================================
# WAVE CONFIGURATION
# =============================================================================

@dataclass
class WaveConfig:
    """Configuration for the wave field system"""
    
    # MASTER SPEED - scales ALL time-based movement
    master_speed: float = 1.0       # 0.1 = very slow, 1.0 = normal, 2.0 = fast
    
    # Base wave layers (always running)
    wave1_speed: float = 0.8        # Slow primary wave
    wave1_frequency: float = 0.5    # Wavelength
    wave1_amplitude: float = 8.0    # Strength
    
    wave2_speed: float = 1.3        # Medium secondary wave
    wave2_frequency: float = 0.8
    wave2_amplitude: float = 5.0
    
    wave3_speed: float = 2.1        # Faster tertiary wave
    wave3_frequency: float = 1.2
    wave3_amplitude: float = 3.0
    
    # Noise modulation
    noise_speed: float = 0.15       # How fast noise evolves
    noise_amplitude: float = 4.0    # How much noise affects output
    noise_phase_mod: float = 0.5    # How much noise shifts wave phases
    
    # Row differentiation (Panel 1 vs Panel 2)
    row_phase_offset: float = 1.2   # Phase difference between rows
    row_speed_mult: float = 0.85    # Row 1 speed multiplier
    row_amplitude_mult: float = 0.7 # Row 1 amplitude multiplier
    
    # Column spread (left-right variation)
    column_phase_spread: float = 0.6  # Phase offset per column
    
    # Ripple system (from people)
    ripple_speed: float = 3.0       # How fast ripples spread
    ripple_decay: float = 0.92      # How quickly ripples fade (per frame)
    ripple_amplitude: float = 15.0  # Initial ripple strength
    ripple_frequency: float = 2.0   # Ripple wave frequency
    
    # Overall
    base_brightness: float = 10.0   # Center point of oscillation
    max_speed_clamp: float = 5.0    # Max change per frame
    
    # Back panel (panel 3) - separate wave system
    back_wave_speed: float = 0.4    # Slower, independent waves
    back_wave_amplitude: float = 6.0
    back_noise_influence: float = 0.8


@dataclass
class InputState:
    """Current input from tracking system"""
    active_population: int = 0
    active_position_x: float = 0.5   # 0=left, 1=right
    active_dwell_time: float = 0.0
    passive_population: int = 0
    cyclist_count: int = 0
    vehicle_count: int = 0
    time_since_active: float = 999.0


@dataclass
class Ripple:
    """A ripple wave emanating from a point"""
    origin_col: float          # Where it started (0-3)
    origin_row: float          # Which row (0 or 1)
    birth_time: float          # When it was created
    amplitude: float           # Current strength
    speed: float               # Spread speed
    frequency: float           # Wave frequency
    
    def get_value(self, col: int, row: int, time_elapsed: float) -> float:
        """Get ripple contribution at a given panel position"""
        # Distance from origin
        dist = math.sqrt((col - self.origin_col) ** 2 + (row - self.origin_row) ** 2)
        
        # Wave front position
        front = time_elapsed * self.speed
        
        # Ring wave pattern
        wave_pos = dist - front
        
        # Sine wave with decay over distance
        if dist < front + 2:  # Only compute near the wave front
            wave = math.sin(wave_pos * self.frequency * math.pi)
            # Amplitude decays with distance and time
            decay = self.amplitude * math.exp(-abs(wave_pos) * 0.5)
            return wave * decay
        return 0.0


# =============================================================================
# WAVE FIELD CONTROLLER
# =============================================================================

class WaveFieldController:
    """Main controller using layered waves instead of spring physics"""
    
    def __init__(self, config: Optional[WaveConfig] = None):
        self.config = config or WaveConfig()
        
        # Time tracking
        self.time = 0.0
        self.last_update = time.time()
        
        # Perlin noise generators
        self.noise = PerlinNoise(seed=42)
        self.noise2 = PerlinNoise(seed=123)
        
        # Panel values (front grid: 2 rows × 4 cols)
        self.front_values = [[self.config.base_brightness] * 4 for _ in range(2)]
        self.back_values = [self.config.base_brightness] * 4
        
        # Previous values for smooth transitions
        self.prev_front = [[self.config.base_brightness] * 4 for _ in range(2)]
        self.prev_back = [self.config.base_brightness] * 4
        
        # Active ripples
        self.ripples: List[Ripple] = []
        
        # Input state
        self.input = InputState()
        
        # Engagement tracking
        self.engagement_level = 0.0
        self.last_engagement_time = time.time() - 300
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX] * TOTAL_CHANNELS
    
    def init_artnet(self) -> bool:
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, TOTAL_CHANNELS, FPS)
            self.artnet.start()
            return True
        except Exception as e:
            print(f"Art-Net initialization failed: {e}")
            return False
    
    def get_time_factors(self) -> dict:
        """Get time-of-day influence factors"""
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
        
        return {
            'rush_factor': rush_factor,
            'evening_factor': evening_factor,
            'hour': hour
        }
    
    def spawn_ripple(self, col: float, row: float, amplitude_mult: float = 1.0):
        """Create a new ripple at the given position"""
        ripple = Ripple(
            origin_col=col,
            origin_row=row,
            birth_time=self.time,
            amplitude=self.config.ripple_amplitude * amplitude_mult,
            speed=self.config.ripple_speed,
            frequency=self.config.ripple_frequency
        )
        self.ripples.append(ripple)
    
    def process_input(self, dt: float):
        """Process input and spawn ripples as needed"""
        if self.input.active_population > 0:
            self.input.time_since_active = 0
            self.input.active_dwell_time += dt
            self.last_engagement_time = time.time()
            
            # Spawn ripples based on presence
            # More people = more frequent ripples
            ripple_chance = 0.05 * self.input.active_population
            if random.random() < ripple_chance:
                # Ripple at person's position
                col = self.input.active_position_x * 3  # 0-3
                row = random.choice([0, 1])
                self.spawn_ripple(col, row, 0.5 + random.random() * 0.5)
        else:
            self.input.time_since_active += dt
            self.input.active_dwell_time = 0
        
        # Update engagement level
        if self.input.active_population > 0:
            target = min(1.0, self.input.active_population * 0.3 + self.input.active_dwell_time * 0.05)
            self.engagement_level += (target - self.engagement_level) * 0.1
        else:
            self.engagement_level *= 0.995
    
    def update_ripples(self, dt: float):
        """Update and decay ripples"""
        # Decay existing ripples
        for ripple in self.ripples:
            ripple.amplitude *= self.config.ripple_decay
        
        # Remove dead ripples
        self.ripples = [r for r in self.ripples if r.amplitude > 0.1]
    
    def calculate_base_waves(self, col: int, row: int) -> float:
        """Calculate the base wave value at a position"""
        cfg = self.config
        t = self.time
        
        # Per-column phase offset for left-right movement
        col_phase = col * cfg.column_phase_spread
        
        # Per-row modifications
        if row == 1:
            row_phase = cfg.row_phase_offset
            speed_mult = cfg.row_speed_mult
            amp_mult = cfg.row_amplitude_mult
        else:
            row_phase = 0
            speed_mult = 1.0
            amp_mult = 1.0
        
        # Noise modulation for organic feel
        noise_val = self.noise.noise2d(col * 0.5 + t * cfg.noise_speed, row * 0.3)
        noise_phase = noise_val * cfg.noise_phase_mod
        
        # Layer 1: Slow primary wave
        wave1 = math.sin(
            t * cfg.wave1_speed * speed_mult + 
            col_phase * cfg.wave1_frequency + 
            row_phase + noise_phase
        ) * cfg.wave1_amplitude * amp_mult
        
        # Layer 2: Medium secondary wave (different direction)
        wave2 = math.sin(
            t * cfg.wave2_speed * speed_mult * 0.9 - 
            col_phase * cfg.wave2_frequency * 1.3 + 
            row_phase * 0.7 + noise_phase * 0.5
        ) * cfg.wave2_amplitude * amp_mult
        
        # Layer 3: Fast tertiary wave (adds shimmer)
        wave3 = math.sin(
            t * cfg.wave3_speed * speed_mult + 
            col_phase * cfg.wave3_frequency * 0.8 - 
            row_phase * 1.2
        ) * cfg.wave3_amplitude * amp_mult * 0.7
        
        # Noise layer
        noise_contribution = self.noise2.noise1d(
            t * cfg.noise_speed * 2 + col * 0.8 + row * 0.5
        ) * cfg.noise_amplitude
        
        # Combine all layers
        total = cfg.base_brightness + wave1 + wave2 + wave3 + noise_contribution
        
        return total
    
    def calculate_ripple_contribution(self, col: int, row: int) -> float:
        """Calculate total ripple contribution at a position"""
        total = 0.0
        for ripple in self.ripples:
            time_elapsed = self.time - ripple.birth_time
            total += ripple.get_value(col, row, time_elapsed)
        return total
    
    def calculate_front_panel(self, col: int, row: int) -> float:
        """Calculate final value for a front panel"""
        base = self.calculate_base_waves(col, row)
        ripples = self.calculate_ripple_contribution(col, row)
        
        # Engagement boosts wave amplitude
        engagement_boost = 1.0 + self.engagement_level * 0.5
        
        # Final value
        value = base + ripples * engagement_boost
        
        # Clamp to valid range
        return max(MIN_DMX, min(MAX_DMX, value))
    
    def calculate_back_panel(self, col: int) -> float:
        """Calculate value for back panel (panel 3) - separate wave system, uses 1-212 range"""
        cfg = self.config
        t = self.time
        
        # Calculate normalized 0-1 value first
        # Slower, more independent waves
        wave = math.sin(
            t * cfg.back_wave_speed + col * 0.9
        ) * 0.3  # Normalized amplitude
        
        # Different noise pattern
        noise_val = self.noise.noise1d(t * cfg.noise_speed * 0.5 + col * 1.2)
        noise = noise_val * cfg.back_noise_influence * 0.15
        
        # Secondary slow wave
        wave2 = math.sin(
            t * cfg.back_wave_speed * 0.6 - col * 0.5 + 1.5
        ) * 0.15
        
        # Normalized value 0-1 centered around 0.5
        normalized = 0.5 + wave + wave2 + noise
        normalized = max(0.0, min(1.0, normalized))
        
        # Scale to full 1-212 range
        value = MIN_DMX_BACK + normalized * (MAX_DMX_BACK - MIN_DMX_BACK)
        
        return max(MIN_DMX_BACK, min(MAX_DMX_BACK, value))
    
    def apply_smoothing(self, current: float, previous: float, dt: float) -> float:
        """Apply rate limiting for smooth transitions (rule #1)"""
        max_change = self.config.max_speed_clamp * dt * 30  # Scaled to frame rate
        diff = current - previous
        
        if abs(diff) > max_change:
            return previous + math.copysign(max_change, diff)
        return current
    
    def update(self) -> dict:
        """Main update loop"""
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        # Apply master speed to time progression
        self.time += dt * self.config.master_speed
        
        time_factors = self.get_time_factors()
        
        # Process input and ripples
        self.process_input(dt)
        self.update_ripples(dt)
        
        # Calculate all front panel values
        for row in range(2):
            for col in range(4):
                raw_value = self.calculate_front_panel(col, row)
                # Apply smoothing
                smoothed = self.apply_smoothing(raw_value, self.prev_front[row][col], dt)
                self.front_values[row][col] = smoothed
                self.prev_front[row][col] = smoothed
        
        # Calculate back panel values
        for col in range(4):
            raw_value = self.calculate_back_panel(col)
            smoothed = self.apply_smoothing(raw_value, self.prev_back[col], dt)
            self.back_values[col] = smoothed
            self.prev_back[col] = smoothed
        
        # Send to Art-Net
        self._send_artnet()
        
        return time_factors
    
    def _send_artnet(self):
        """Convert values to DMX and send"""
        if not self.artnet:
            return
        
        for unit in range(4):
            col = 3 - unit  # Reverse mapping
            
            # Panel 1 (row 0)
            ch1 = unit * 3 + 0
            self.channel_values[ch1] = int(max(MIN_DMX, min(MAX_DMX, self.front_values[0][col])))
            
            # Panel 2 (row 1)
            ch2 = unit * 3 + 1
            self.channel_values[ch2] = int(max(MIN_DMX, min(MAX_DMX, self.front_values[1][col])))
            
            # Panel 3 (back) - uses full 1-212 range
            ch3 = unit * 3 + 2
            self.channel_values[ch3] = int(max(MIN_DMX_BACK, min(MAX_DMX_BACK, self.back_values[col])))
        
        self.artnet.set(self.channel_values)
    
    def shutdown(self):
        if self.artnet:
            for i in range(len(self.channel_values)):
                self.channel_values[i] = MIN_DMX
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


# =============================================================================
# GUI APPLICATION
# =============================================================================

class WaveFieldGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("Wave Field LED Controller")
        self.root.geometry("1400x900")
        
        self.controller = WaveFieldController()
        
        if not no_artnet:
            if not self.controller.init_artnet():
                print("Warning: Art-Net failed to initialize")
        
        self.setup_gui()
        
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Main view
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main View")
        self.setup_main_tab(main_tab)
        
        # Tab 2: Wave Tuning
        wave_tab = ttk.Frame(notebook)
        notebook.add(wave_tab, text="Wave Tuning")
        self.setup_wave_tab(wave_tab)
        
        # Tab 3: Wave Visualization
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="Wave Visualization")
        self.setup_viz_tab(viz_tab)
    
    def setup_main_tab(self, parent):
        # Panel visualization
        viz_frame = ttk.LabelFrame(parent, text="Panel Output (DMX 1-50)", padding=10)
        viz_frame.pack(fill="both", expand=True, pady=(0, 10), padx=5)
        self.create_panel_visualization(viz_frame)
        
        # Middle row
        middle_frame = ttk.Frame(parent)
        middle_frame.pack(fill="both", expand=True, padx=5)
        
        # Input controls
        input_frame = ttk.LabelFrame(middle_frame, text="Input Simulation", padding=10)
        input_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.create_input_controls(input_frame)
        
        # Status
        status_frame = ttk.LabelFrame(middle_frame, text="System Status", padding=10)
        status_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.create_status_display(status_frame)
    
    def setup_wave_tab(self, parent):
        """Wave parameter tuning"""
        # Left column: Base waves
        left_frame = ttk.Frame(parent)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.tuning_vars = {}
        
        # Wave 1
        w1_frame = ttk.LabelFrame(left_frame, text="Wave Layer 1 (Slow)", padding=10)
        w1_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(w1_frame, "wave1_speed", "Speed", 0.1, 3.0)
        self.create_slider(w1_frame, "wave1_frequency", "Frequency", 0.1, 2.0)
        self.create_slider(w1_frame, "wave1_amplitude", "Amplitude", 0.0, 15.0)
        
        # Wave 2
        w2_frame = ttk.LabelFrame(left_frame, text="Wave Layer 2 (Medium)", padding=10)
        w2_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(w2_frame, "wave2_speed", "Speed", 0.1, 4.0)
        self.create_slider(w2_frame, "wave2_frequency", "Frequency", 0.1, 2.5)
        self.create_slider(w2_frame, "wave2_amplitude", "Amplitude", 0.0, 12.0)
        
        # Wave 3
        w3_frame = ttk.LabelFrame(left_frame, text="Wave Layer 3 (Fast)", padding=10)
        w3_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(w3_frame, "wave3_speed", "Speed", 0.5, 5.0)
        self.create_slider(w3_frame, "wave3_frequency", "Frequency", 0.2, 3.0)
        self.create_slider(w3_frame, "wave3_amplitude", "Amplitude", 0.0, 10.0)
        
        # Right column
        right_frame = ttk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Noise
        noise_frame = ttk.LabelFrame(right_frame, text="Noise Modulation", padding=10)
        noise_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(noise_frame, "noise_speed", "Evolution Speed", 0.01, 0.5)
        self.create_slider(noise_frame, "noise_amplitude", "Amplitude", 0.0, 10.0)
        self.create_slider(noise_frame, "noise_phase_mod", "Phase Modulation", 0.0, 2.0)
        
        # Row differentiation
        row_frame = ttk.LabelFrame(right_frame, text="Row Differentiation (Panel 1 vs 2)", padding=10)
        row_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(row_frame, "row_phase_offset", "Phase Offset", 0.0, 3.0)
        self.create_slider(row_frame, "row_speed_mult", "Speed Multiplier", 0.3, 1.5)
        self.create_slider(row_frame, "row_amplitude_mult", "Amplitude Mult", 0.2, 1.5)
        self.create_slider(row_frame, "column_phase_spread", "Column Spread", 0.0, 2.0)
        
        # Ripples
        ripple_frame = ttk.LabelFrame(right_frame, text="Ripple System (from people)", padding=10)
        ripple_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(ripple_frame, "ripple_speed", "Spread Speed", 0.5, 8.0)
        self.create_slider(ripple_frame, "ripple_decay", "Decay Rate", 0.8, 0.99)
        self.create_slider(ripple_frame, "ripple_amplitude", "Amplitude", 5.0, 30.0)
        
        # General
        general_frame = ttk.LabelFrame(right_frame, text="General", padding=10)
        general_frame.pack(fill="x", pady=(0, 10))
        
        # MASTER SPEED - most important slider, make it prominent
        master_frame = ttk.Frame(general_frame)
        master_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(master_frame, text="★ MASTER SPEED:", font=("Helvetica", 10, "bold")).pack(side="left")
        self.master_speed_var = tk.DoubleVar(value=1.0)
        master_label = ttk.Label(master_frame, text="1.00", width=6)
        master_label.pack(side="right")
        def on_master_change(val):
            v = float(val)
            master_label.config(text=f"{v:.2f}")
            self.controller.config.master_speed = v
        master_slider = ttk.Scale(master_frame, from_=0.1, to=2.0, variable=self.master_speed_var,
                                  orient="horizontal", command=on_master_change)
        master_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.tuning_vars["master_speed"] = self.master_speed_var
        
        self.create_slider(general_frame, "base_brightness", "Base Brightness", 1.0, 25.0)
        self.create_slider(general_frame, "max_speed_clamp", "Smoothing", 1.0, 15.0)
        
        # Presets
        preset_frame = ttk.Frame(right_frame)
        preset_frame.pack(fill="x")
        ttk.Button(preset_frame, text="Calm", command=self.preset_calm).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Active", command=self.preset_active).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Dramatic", command=self.preset_dramatic).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Subtle", command=self.preset_subtle).pack(side="left", padx=5)
    
    def setup_viz_tab(self, parent):
        """Wave visualization"""
        self.wave_canvas = tk.Canvas(parent, bg="#0a0a1a", height=500)
        self.wave_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        legend = ttk.Frame(parent)
        legend.pack(fill="x", padx=10, pady=5)
        ttk.Label(legend, text="• Bar height = brightness | Color = rate of change | ○ = active ripples").pack()
    
    def create_slider(self, parent, key: str, label: str, min_v: float, max_v: float):
        """Create a tuning slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=16, anchor="w").pack(side="left")
        
        current_val = getattr(self.controller.config, key)
        var = tk.DoubleVar(value=current_val)
        self.tuning_vars[key] = var
        
        value_label = ttk.Label(frame, text=f"{current_val:.2f}", width=6)
        value_label.pack(side="right")
        
        def on_change(val, k=key, lbl=value_label):
            v = float(val)
            lbl.config(text=f"{v:.2f}")
            setattr(self.controller.config, k, v)
        
        slider = ttk.Scale(frame, from_=min_v, to=max_v, variable=var,
                          orient="horizontal", command=on_change)
        slider.pack(side="left", fill="x", expand=True, padx=10)
    
    def create_panel_visualization(self, parent):
        self.panel_canvas = tk.Canvas(parent, bg="#1a1a2a", height=250)
        self.panel_canvas.pack(fill="both", expand=True)
        
        self.panel_bars = []
        self.panel_labels = []
        
        bar_width = 50
        bar_max_height = 150
        spacing = 25
        start_x = 60
        start_y = 200
        
        for unit in range(4):
            unit_x = start_x + unit * (bar_width * 3 + spacing * 3 + 50)
            
            # Unit label
            self.panel_canvas.create_text(
                unit_x + bar_width * 1.5 + spacing,
                30,
                text=f"Unit {unit + 1}",
                fill="#888888",
                font=("Helvetica", 11, "bold")
            )
            
            for panel in range(3):
                x = unit_x + panel * (bar_width + spacing)
                
                # Bar
                bar = self.panel_canvas.create_rectangle(
                    x, start_y,
                    x + bar_width, start_y,
                    fill="#4488ff", outline="#6699ff", width=1
                )
                self.panel_bars.append(bar)
                
                # Value label
                label = self.panel_canvas.create_text(
                    x + bar_width / 2, start_y + 15,
                    text="--",
                    fill="#aaaaaa",
                    font=("Courier", 9)
                )
                self.panel_labels.append(label)
                
                # Panel label
                self.panel_canvas.create_text(
                    x + bar_width / 2, start_y + 30,
                    text=f"P{panel + 1}",
                    fill="#666666",
                    font=("Helvetica", 8)
                )
    
    def create_input_controls(self, parent):
        # Active zone
        active_frame = ttk.LabelFrame(parent, text="Active Zone (TrackZone 1-2)", padding=10)
        active_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(active_frame, text="Population:").grid(row=0, column=0, sticky="w", pady=2)
        self.active_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(active_frame, from_=0, to=10, textvariable=self.active_pop_var,
                   width=10, command=self.on_input_change).grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(active_frame, text="Position:").grid(row=1, column=0, sticky="w", pady=2)
        self.active_pos_var = tk.DoubleVar(value=0.5)
        pos_frame = ttk.Frame(active_frame)
        pos_frame.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(pos_frame, text="L").pack(side="left")
        ttk.Scale(pos_frame, from_=0, to=1, variable=self.active_pos_var,
                 orient="horizontal", command=lambda v: self.on_input_change()).pack(side="left", fill="x", expand=True)
        ttk.Label(pos_frame, text="R").pack(side="left")
        
        # Quick buttons
        btn_frame = ttk.Frame(active_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame, text="1 Left", command=lambda: self.set_input(1, 0.15)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="1 Center", command=lambda: self.set_input(1, 0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="1 Right", command=lambda: self.set_input(1, 0.85)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Clear", command=lambda: self.set_input(0, 0.5)).pack(side="left", padx=2)
        
        # Manual ripple button
        ripple_frame = ttk.Frame(active_frame)
        ripple_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(ripple_frame, text="Spawn Ripple", command=self.spawn_manual_ripple).pack(side="left", padx=2)
        ttk.Button(ripple_frame, text="Ripple Storm", command=self.ripple_storm).pack(side="left", padx=2)
        
        active_frame.columnconfigure(1, weight=1)
        
        # Passive zone
        passive_frame = ttk.LabelFrame(parent, text="Passive Zone (TrackZone 3-10)", padding=10)
        passive_frame.pack(fill="x")
        
        ttk.Label(passive_frame, text="Population:").grid(row=0, column=0, sticky="w", pady=2)
        self.passive_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(passive_frame, from_=0, to=50, textvariable=self.passive_pop_var,
                   width=10).grid(row=0, column=1, sticky="ew", padx=5)
        
        passive_frame.columnconfigure(1, weight=1)
    
    def create_status_display(self, parent):
        ttk.Label(parent, text="Engagement Level:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.engagement_bar = ttk.Progressbar(parent, mode='determinate', length=200)
        self.engagement_bar.pack(fill="x", pady=(0, 5))
        self.engagement_label = ttk.Label(parent, text="0%")
        self.engagement_label.pack(anchor="w", pady=(0, 10))
        
        ttk.Label(parent, text="Active Ripples:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.ripple_label = ttk.Label(parent, text="0")
        self.ripple_label.pack(anchor="w", pady=(0, 10))
        
        ttk.Label(parent, text="Wave Stats:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.wave_stats = tk.Text(parent, height=8, width=30, bg="#f0f0f0")
        self.wave_stats.pack(fill="both", expand=True)
        self.wave_stats.config(state="disabled")
    
    def set_input(self, pop: int, pos: float):
        self.active_pop_var.set(pop)
        self.active_pos_var.set(pos)
        self.on_input_change()
    
    def on_input_change(self):
        self.controller.input.active_population = self.active_pop_var.get()
        self.controller.input.active_position_x = self.active_pos_var.get()
    
    def spawn_manual_ripple(self):
        """Spawn a ripple at current position"""
        col = self.active_pos_var.get() * 3
        row = random.choice([0, 1])
        self.controller.spawn_ripple(col, row, 1.0)
    
    def ripple_storm(self):
        """Spawn multiple ripples"""
        for _ in range(5):
            col = random.random() * 3
            row = random.choice([0, 1])
            self.controller.spawn_ripple(col, row, 0.5 + random.random() * 0.5)
    
    def preset_calm(self):
        presets = {
            'wave1_speed': 0.5, 'wave1_amplitude': 5.0,
            'wave2_speed': 0.8, 'wave2_amplitude': 3.0,
            'wave3_speed': 1.2, 'wave3_amplitude': 2.0,
            'noise_amplitude': 2.0,
            'ripple_amplitude': 10.0,
            'base_brightness': 12.0,
        }
        self.apply_presets(presets)
    
    def preset_active(self):
        presets = {
            'wave1_speed': 1.0, 'wave1_amplitude': 10.0,
            'wave2_speed': 1.8, 'wave2_amplitude': 7.0,
            'wave3_speed': 2.5, 'wave3_amplitude': 4.0,
            'noise_amplitude': 5.0,
            'ripple_amplitude': 18.0,
            'base_brightness': 10.0,
        }
        self.apply_presets(presets)
    
    def preset_dramatic(self):
        presets = {
            'wave1_speed': 1.5, 'wave1_amplitude': 12.0,
            'wave2_speed': 2.5, 'wave2_amplitude': 9.0,
            'wave3_speed': 3.5, 'wave3_amplitude': 6.0,
            'noise_amplitude': 7.0,
            'noise_phase_mod': 1.5,
            'row_phase_offset': 2.0,
            'ripple_amplitude': 25.0,
            'base_brightness': 8.0,
        }
        self.apply_presets(presets)
    
    def preset_subtle(self):
        presets = {
            'wave1_speed': 0.3, 'wave1_amplitude': 3.0,
            'wave2_speed': 0.5, 'wave2_amplitude': 2.0,
            'wave3_speed': 0.8, 'wave3_amplitude': 1.5,
            'noise_amplitude': 1.5,
            'ripple_amplitude': 8.0,
            'base_brightness': 15.0,
        }
        self.apply_presets(presets)
    
    def apply_presets(self, presets):
        for key, value in presets.items():
            if key in self.tuning_vars:
                self.tuning_vars[key].set(value)
            setattr(self.controller.config, key, value)
    
    def update_panel_display(self):
        bar_max_height = 150
        start_y = 200
        bar_width = 50
        spacing = 25
        start_x = 60
        
        all_values = []
        
        for unit in range(4):
            col = 3 - unit
            unit_x = start_x + unit * (bar_width * 3 + spacing * 3 + 50)
            
            for panel in range(3):
                idx = unit * 3 + panel
                x = unit_x + panel * (bar_width + spacing)
                
                if panel < 2:
                    value = self.controller.front_values[panel][col]
                    prev = self.controller.prev_front[panel][col]
                    min_v, max_v = MIN_DMX, MAX_DMX
                else:
                    value = self.controller.back_values[col]
                    prev = self.controller.prev_back[col]
                    min_v, max_v = MIN_DMX_BACK, MAX_DMX_BACK
                
                all_values.append(value)
                
                # Bar height based on value (normalized to each panel's range)
                height = (value - min_v) / (max_v - min_v) * bar_max_height
                
                # Color based on rate of change
                change_rate = abs(value - prev) * 10
                if change_rate > 3:
                    color = "#ff6644"  # Fast change = orange/red
                elif change_rate > 1:
                    color = "#ffaa44"  # Medium = yellow
                else:
                    color = "#4488ff"  # Slow = blue
                
                # Update bar
                self.panel_canvas.coords(
                    self.panel_bars[idx],
                    x, start_y - height,
                    x + bar_width, start_y
                )
                self.panel_canvas.itemconfig(self.panel_bars[idx], fill=color)
                
                # Update label
                self.panel_canvas.itemconfig(
                    self.panel_labels[idx],
                    text=f"{int(value)}"
                )
    
    def update_wave_visualization(self):
        """Draw wave patterns"""
        self.wave_canvas.delete("all")
        
        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()
        
        if w < 100:
            return
        
        # Draw wave history for each column
        center_y = h // 2
        col_width = w // 5
        
        for col in range(4):
            x = col_width + col * col_width
            
            # Row 0 (Panel 1) - upper wave
            val0 = self.controller.front_values[0][col]
            y0 = center_y - 80 - (val0 - 25) * 3
            
            # Row 1 (Panel 2) - lower wave  
            val1 = self.controller.front_values[1][col]
            y1 = center_y + 80 - (val1 - 25) * 3
            
            # Draw nodes
            self.wave_canvas.create_oval(
                x - 20, y0 - 20, x + 20, y0 + 20,
                fill=f"#{int(val0*4):02x}{int(val0*3):02x}ff",
                outline="#8888ff", width=2
            )
            self.wave_canvas.create_text(x, y0, text=f"{int(val0)}", fill="white", font=("Helvetica", 10, "bold"))
            
            self.wave_canvas.create_oval(
                x - 20, y1 - 20, x + 20, y1 + 20,
                fill=f"#{int(val1*4):02x}ff{int(val1*3):02x}",
                outline="#88ff88", width=2
            )
            self.wave_canvas.create_text(x, y1, text=f"{int(val1)}", fill="white", font=("Helvetica", 10, "bold"))
            
            # Labels
            self.wave_canvas.create_text(x, 30, text=f"Col {col+1}", fill="#888888")
        
        # Draw ripples
        for ripple in self.controller.ripples:
            rx = col_width + ripple.origin_col * col_width
            ry = center_y - 80 if ripple.origin_row == 0 else center_y + 80
            radius = (self.controller.time - ripple.birth_time) * ripple.speed * 30
            alpha = int(ripple.amplitude * 10)
            
            self.wave_canvas.create_oval(
                rx - radius, ry - radius,
                rx + radius, ry + radius,
                outline=f"#ff{alpha:02x}44", width=2
            )
        
        # Labels
        self.wave_canvas.create_text(40, center_y - 80, text="Panel 1", fill="#8888ff", anchor="w")
        self.wave_canvas.create_text(40, center_y + 80, text="Panel 2", fill="#88ff88", anchor="w")
    
    def update_status_display(self):
        # Engagement
        eng_pct = self.controller.engagement_level * 100
        self.engagement_bar['value'] = eng_pct
        self.engagement_label.config(text=f"{eng_pct:.1f}%")
        
        # Ripples
        self.ripple_label.config(text=f"{len(self.controller.ripples)}")
        
        # Wave stats
        self.wave_stats.config(state="normal")
        self.wave_stats.delete(1.0, tk.END)
        
        front = self.controller.front_values
        back = self.controller.back_values
        stats = f"Time: {self.controller.time:.1f}s\n"
        stats += f"Master Speed: {self.controller.config.master_speed:.2f}x\n\n"
        stats += "Front Panel Values:\n"
        stats += f"  P1: {[f'{v:.1f}' for v in front[0]]}\n"
        stats += f"  P2: {[f'{v:.1f}' for v in front[1]]}\n"
        stats += f"  P3: {[f'{v:.1f}' for v in back]}\n\n"
        
        # Contrast metrics
        row0_avg = sum(front[0]) / 4
        row1_avg = sum(front[1]) / 4
        back_avg = sum(back) / 4
        row_diff = abs(row0_avg - row1_avg)
        col_range = max(max(front[0]) - min(front[0]), max(front[1]) - min(front[1]))
        
        stats += f"Row Contrast: {row_diff:.1f}\n"
        stats += f"Column Spread: {col_range:.1f}\n"
        
        self.wave_stats.insert(1.0, stats)
        self.wave_stats.config(state="disabled")
    
    def update_loop(self):
        if not self.running:
            return
        
        self.controller.update()
        
        self.update_panel_display()
        self.update_wave_visualization()
        self.update_status_display()
        
        self.root.after(int(1000 / FPS), self.update_loop)
    
    def on_closing(self):
        self.running = False
        self.controller.shutdown()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Wave Field LED Controller')
    parser.add_argument('--no-artnet', action='store_true', help='Run without Art-Net')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = WaveFieldGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
