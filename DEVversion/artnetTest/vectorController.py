#!/usr/bin/env python3
"""
Vector-Based LED Panel Controller
Movement vectors sweep across the panel grid based on tracking input.

Coordinate System:
  - Origin (0,0,0) at bottom-right of Unit 1
  - +X toward Unit 4 (left across row)
  - +Y going up
  - +Z going out toward street

Panel Positions:
  Unit 1: x=0, Unit 2: x=1, Unit 3: x=2, Unit 4: x=3
  Panel 1 (top front): y=1, z=0
  Panel 2 (bottom front): y=0, z=0  
  Panel 3 (back): y=0, z=1
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple
from stupidArtnet import StupidArtnet

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

NUM_UNITS = 4

# DMX ranges
MIN_DMX_FRONT = 1
MAX_DMX_FRONT = 50
MIN_DMX_BACK = 1
MAX_DMX_BACK = 200

# Panel 3D positions (x, y, z)
# x: 0=Unit1, 1=Unit2, 2=Unit3, 3=Unit4
# y: 0=bottom, 1=top
# z: 0=front, 1=back
PANEL_POSITIONS = {
    # (unit, panel): (x, y, z)
    (0, 0): (0, 1, 0),  # Unit1, Panel1 (top front)
    (0, 1): (0, 0, 0),  # Unit1, Panel2 (bottom front)
    (0, 2): (0, 0, 1),  # Unit1, Panel3 (back)
    (1, 0): (1, 1, 0),  # Unit2, Panel1
    (1, 1): (1, 0, 0),  # Unit2, Panel2
    (1, 2): (1, 0, 1),  # Unit2, Panel3
    (2, 0): (2, 1, 0),  # Unit3, Panel1
    (2, 1): (2, 0, 0),  # Unit3, Panel2
    (2, 2): (2, 0, 1),  # Unit3, Panel3
    (3, 0): (3, 1, 0),  # Unit4, Panel1
    (3, 1): (3, 0, 0),  # Unit4, Panel2
    (3, 2): (3, 0, 1),  # Unit4, Panel3
}


@dataclass
class VectorConfig:
    """Configuration for vector-based movement"""
    
    # Master speed control
    master_speed: float = 0.5
    
    # Movement vector
    vector_dx: float = 1.0      # X component of movement direction
    vector_dy: float = 0.3      # Y component
    wave_speed: float = 1.0     # How fast the wave moves
    wave_width: float = 2.0     # Width of the brightness wave
    
    # Z offset (Panel 3 delay relative to Panel 2)
    z_delay: float = 0.5        # Time offset for back panels
    
    # Noise/organic feel
    noise_amount: float = 0.15  # Base noise level
    noise_speed: float = 0.3    # How fast noise evolves
    
    # Sync factor (how strongly panels try to follow the pattern)
    sync_strength: float = 0.8  # 0=full noise, 1=perfect sync
    
    # Brightness
    base_brightness: float = 0.5  # 0-1, center point
    wave_amplitude: float = 0.4   # How much the wave affects brightness
    
    # Ripple from tracking
    ripple_strength: float = 0.3
    ripple_decay: float = 0.95


class VectorController:
    """Controls panels based on movement vectors"""
    
    def __init__(self, config: VectorConfig = None):
        self.config = config or VectorConfig()
        
        self.time = 0.0
        self.last_update = time.time()
        
        # Current vector state
        self.wave_position = 0.0  # Position along the wave direction
        
        # Wandering target (for passive mode)
        self.wander_angle = random.random() * math.pi * 2
        self.wander_speed = 0.5
        
        # Panel values (normalized 0-1)
        self.panel_values = {}
        for key in PANEL_POSITIONS:
            self.panel_values[key] = 0.5
        
        # Noise offsets per panel
        self.noise_offsets = {}
        for key in PANEL_POSITIONS:
            self.noise_offsets[key] = random.random() * 100
        
        # Input state
        self.active_population = 0
        self.active_position_x = 0.5  # 0=left, 1=right
        self.passive_population = 0
        
        # Ripple state
        self.ripple_intensity = 0.0
        self.ripple_center_x = 0.5
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX_FRONT] * 12
    
    def init_artnet(self) -> bool:
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            self.artnet.start()
            return True
        except Exception as e:
            print(f"Art-Net init failed: {e}")
            return False
    
    def update_input(self, active_pop: int, active_pos: float, passive_pop: int):
        """Update from tracking input"""
        self.active_population = active_pop
        self.active_position_x = active_pos
        self.passive_population = passive_pop
        
        # Active tracking creates ripples
        if active_pop > 0:
            self.ripple_intensity = min(1.0, self.ripple_intensity + 0.1 * active_pop)
            self.ripple_center_x = active_pos
    
    def _noise(self, seed: float, t: float) -> float:
        """Simple noise function returning -1 to 1"""
        return math.sin(seed * 12.9898 + t * 3.14159) * math.sin(seed * 78.233 + t * 2.71828)
    
    def _calculate_wave_value(self, x: float, y: float, z: float, t: float) -> float:
        """Calculate the wave brightness at a 3D position"""
        cfg = self.config
        
        # Normalize direction vector
        mag = math.sqrt(cfg.vector_dx**2 + cfg.vector_dy**2)
        if mag < 0.001:
            mag = 1.0
        dx = cfg.vector_dx / mag
        dy = cfg.vector_dy / mag
        
        # Project position onto movement direction
        proj = x * dx + y * dy
        
        # Apply z delay for back panels
        proj -= z * cfg.z_delay
        
        # Wave position (moves over time)
        wave_phase = proj - t * cfg.wave_speed
        
        # Sine wave with configurable width
        wave = math.sin(wave_phase * math.pi / cfg.wave_width)
        
        # Convert to 0-1 range
        wave_normalized = (wave + 1) / 2
        
        return wave_normalized
    
    def update(self) -> dict:
        """Main update loop"""
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        
        # Apply master speed
        self.time += dt * self.config.master_speed
        
        cfg = self.config
        t = self.time
        
        # Update wandering (passive movement)
        self.wander_angle += (random.random() - 0.5) * 0.1 * dt
        
        # If no active tracking, slowly wander the vector direction
        if self.active_population == 0:
            # Gentle wandering
            target_dx = math.cos(self.wander_angle) * 0.5 + 0.5
            target_dy = math.sin(self.wander_angle) * 0.3
            cfg.vector_dx += (target_dx - cfg.vector_dx) * 0.02
            cfg.vector_dy += (target_dy - cfg.vector_dy) * 0.02
        else:
            # Active tracking influences direction
            # Person position affects dx (left/right movement)
            target_dx = (self.active_position_x - 0.5) * 2  # -1 to 1
            cfg.vector_dx += (target_dx - cfg.vector_dx) * 0.1
        
        # Update ripple decay
        self.ripple_intensity *= cfg.ripple_decay
        
        # Calculate each panel's value
        for (unit, panel), (x, y, z) in PANEL_POSITIONS.items():
            # Base wave value
            wave_val = self._calculate_wave_value(x, y, z, t)
            
            # Apply wave amplitude
            target = cfg.base_brightness + (wave_val - 0.5) * cfg.wave_amplitude * 2
            
            # Add noise for organic feel
            noise_seed = self.noise_offsets[(unit, panel)]
            noise = self._noise(noise_seed, t * cfg.noise_speed) * cfg.noise_amount
            
            # Ripple contribution (based on distance from ripple center)
            if self.ripple_intensity > 0.01:
                ripple_dist = abs(x / 3.0 - self.ripple_center_x)
                ripple_contrib = self.ripple_intensity * cfg.ripple_strength * (1 - ripple_dist)
                ripple_wave = math.sin(t * 5 - ripple_dist * 4) * ripple_contrib
                noise += ripple_wave
            
            # Blend between noisy and synced
            noisy_value = target + noise
            synced_value = target
            final_value = synced_value * cfg.sync_strength + noisy_value * (1 - cfg.sync_strength)
            
            # Clamp to 0-1
            self.panel_values[(unit, panel)] = max(0.0, min(1.0, final_value))
        
        # Send to Art-Net
        self._send_artnet()
        
        return {
            'time': t,
            'vector_dx': cfg.vector_dx,
            'vector_dy': cfg.vector_dy,
            'ripple': self.ripple_intensity
        }
    
    def _send_artnet(self):
        """Convert values to DMX and send"""
        if not self.artnet:
            return
        
        for unit in range(4):
            col = 3 - unit  # Reverse mapping for physical layout
            
            for panel in range(3):
                ch = unit * 3 + panel
                val = self.panel_values[(col, panel)]
                
                if panel < 2:  # Front panels
                    dmx = int(MIN_DMX_FRONT + val * (MAX_DMX_FRONT - MIN_DMX_FRONT))
                    dmx = max(MIN_DMX_FRONT, min(MAX_DMX_FRONT, dmx))
                else:  # Back panel
                    dmx = int(MIN_DMX_BACK + val * (MAX_DMX_BACK - MIN_DMX_BACK))
                    dmx = max(MIN_DMX_BACK, min(MAX_DMX_BACK, dmx))
                
                self.channel_values[ch] = dmx
        
        self.artnet.set(self.channel_values)
    
    def shutdown(self):
        if self.artnet:
            self.channel_values = [MIN_DMX_FRONT] * 12
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


# =============================================================================
# GUI
# =============================================================================

class VectorControllerGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("Vector LED Controller")
        self.root.geometry("1200x800")
        
        self.controller = VectorController()
        
        if not no_artnet:
            if not self.controller.init_artnet():
                print("Warning: Art-Net failed")
        
        self.setup_gui()
        
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        # Main layout: left = visualization, right = controls
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Panel visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Panel Visualization", padding=10)
        viz_frame.pack(side="left", fill="both", expand=True)
        
        self.viz_canvas = tk.Canvas(viz_frame, bg="#2a2a3a", width=700, height=500)
        self.viz_canvas.pack(fill="both", expand=True)
        
        # Right: Controls
        ctrl_frame = ttk.Frame(main_frame, width=400)
        ctrl_frame.pack(side="right", fill="y", padx=(10, 0))
        ctrl_frame.pack_propagate(False)
        
        self.setup_controls(ctrl_frame)
    
    def setup_controls(self, parent):
        self.sliders = {}
        
        # Master speed
        master_frame = ttk.LabelFrame(parent, text="★ Master Speed", padding=10)
        master_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(master_frame, "master_speed", "Speed", 0.1, 2.0, 0.5)
        
        # Vector controls
        vec_frame = ttk.LabelFrame(parent, text="Movement Vector", padding=10)
        vec_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(vec_frame, "vector_dx", "Direction X", -2.0, 2.0, 1.0)
        self.create_slider(vec_frame, "vector_dy", "Direction Y", -1.0, 1.0, 0.3)
        self.create_slider(vec_frame, "wave_speed", "Wave Speed", 0.1, 3.0, 1.0)
        self.create_slider(vec_frame, "wave_width", "Wave Width", 0.5, 5.0, 2.0)
        
        # Z offset
        z_frame = ttk.LabelFrame(parent, text="Z Offset (Panel 3 Delay)", padding=10)
        z_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(z_frame, "z_delay", "Back Panel Delay", 0.0, 2.0, 0.5)
        
        # Noise/organic
        noise_frame = ttk.LabelFrame(parent, text="Organic Feel", padding=10)
        noise_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(noise_frame, "noise_amount", "Noise Amount", 0.0, 0.5, 0.15)
        self.create_slider(noise_frame, "noise_speed", "Noise Speed", 0.1, 2.0, 0.3)
        self.create_slider(noise_frame, "sync_strength", "Sync Strength", 0.0, 1.0, 0.8)
        
        # Brightness
        bright_frame = ttk.LabelFrame(parent, text="Brightness", padding=10)
        bright_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(bright_frame, "base_brightness", "Base Level", 0.1, 0.9, 0.5)
        self.create_slider(bright_frame, "wave_amplitude", "Wave Amplitude", 0.1, 0.5, 0.4)
        
        # Ripple
        ripple_frame = ttk.LabelFrame(parent, text="Tracking Ripple", padding=10)
        ripple_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(ripple_frame, "ripple_strength", "Ripple Strength", 0.0, 1.0, 0.3)
        
        # Input simulation
        input_frame = ttk.LabelFrame(parent, text="Input Simulation", padding=10)
        input_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(input_frame, text="Active People:").pack(anchor="w")
        self.active_pop_var = tk.IntVar(value=0)
        ttk.Spinbox(input_frame, from_=0, to=10, textvariable=self.active_pop_var,
                   command=self.on_input_change).pack(fill="x", pady=(0, 5))
        
        ttk.Label(input_frame, text="Position (L-R):").pack(anchor="w")
        self.active_pos_var = tk.DoubleVar(value=0.5)
        pos_frame = ttk.Frame(input_frame)
        pos_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(pos_frame, text="L").pack(side="left")
        ttk.Scale(pos_frame, from_=0, to=1, variable=self.active_pos_var,
                 orient="horizontal", command=lambda v: self.on_input_change()).pack(side="left", fill="x", expand=True)
        ttk.Label(pos_frame, text="R").pack(side="left")
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="1 Left", command=lambda: self.set_input(1, 0.2)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="1 Center", command=lambda: self.set_input(1, 0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="1 Right", command=lambda: self.set_input(1, 0.8)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Clear", command=lambda: self.set_input(0, 0.5)).pack(side="left", padx=2)
    
    def create_slider(self, parent, key: str, label: str, min_v: float, max_v: float, default: float):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=14, anchor="w").pack(side="left")
        
        var = tk.DoubleVar(value=default)
        self.sliders[key] = var
        
        value_label = ttk.Label(frame, text=f"{default:.2f}", width=5)
        value_label.pack(side="right")
        
        def on_change(val, k=key, lbl=value_label):
            v = float(val)
            lbl.config(text=f"{v:.2f}")
            setattr(self.controller.config, k, v)
        
        slider = ttk.Scale(frame, from_=min_v, to=max_v, variable=var,
                          orient="horizontal", command=on_change)
        slider.pack(side="left", fill="x", expand=True, padx=5)
    
    def set_input(self, pop: int, pos: float):
        self.active_pop_var.set(pop)
        self.active_pos_var.set(pos)
        self.on_input_change()
    
    def on_input_change(self):
        self.controller.update_input(
            self.active_pop_var.get(),
            self.active_pos_var.get(),
            0
        )
    
    def draw_visualization(self):
        """Draw panel shapes similar to the physical layout"""
        self.viz_canvas.delete("all")
        
        w = self.viz_canvas.winfo_width()
        h = self.viz_canvas.winfo_height()
        
        if w < 100:
            return
        
        # Layout constants
        unit_width = 140
        unit_spacing = 30
        total_width = unit_width * 4 + unit_spacing * 3
        start_x = (w - total_width) // 2
        
        panel1_height = 120  # Top panel (tall)
        panel2_height = 80   # Bottom front panel
        panel3_height = 70   # Back panel
        
        base_y = 100  # Top of panel 1
        hinge_gap = 8
        
        for unit in range(4):
            # Reverse order to match physical layout (Unit 4 on left)
            display_unit = 3 - unit
            x = start_x + display_unit * (unit_width + unit_spacing)
            
            # Get values
            val1 = self.panel_values_display.get((unit, 0), 0.5)
            val2 = self.panel_values_display.get((unit, 1), 0.5)
            val3 = self.panel_values_display.get((unit, 2), 0.5)
            
            # Colors based on brightness (blue to white gradient)
            def val_to_color(v, is_back=False):
                # v is 0-1
                if is_back:
                    # Back panels: purple tint
                    r = int(100 + v * 155)
                    g = int(80 + v * 120)
                    b = int(150 + v * 105)
                else:
                    # Front panels: blue-white
                    r = int(80 + v * 175)
                    g = int(100 + v * 155)
                    b = int(180 + v * 75)
                return f"#{r:02x}{g:02x}{b:02x}"
            
            color1 = val_to_color(val1)
            color2 = val_to_color(val2)
            color3 = val_to_color(val3, is_back=True)
            
            # Draw Panel 1 (top, front - vertical rectangle)
            p1_top = base_y
            p1_bottom = base_y + panel1_height
            self.viz_canvas.create_rectangle(
                x, p1_top, x + unit_width, p1_bottom,
                fill=color1, outline="#555555", width=2
            )
            # Label
            self.viz_canvas.create_text(
                x + unit_width // 2, p1_top + 20,
                text="1", fill="#333333", font=("Helvetica", 16, "bold")
            )
            # DMX value
            dmx1 = int(MIN_DMX_FRONT + val1 * (MAX_DMX_FRONT - MIN_DMX_FRONT))
            self.viz_canvas.create_text(
                x + unit_width // 2, p1_bottom - 20,
                text=f"{dmx1}", fill="#444444", font=("Courier", 12)
            )
            
            # Hinge point (orange triangle)
            hinge_y = p1_bottom + hinge_gap // 2
            self.viz_canvas.create_polygon(
                x + unit_width // 2 - 10, hinge_y - 5,
                x + unit_width // 2 + 10, hinge_y - 5,
                x + unit_width // 2, hinge_y + 8,
                fill="#ff8844", outline="#aa5522"
            )
            
            # Draw Panel 2 (bottom front - angled, smaller)
            p2_top = p1_bottom + hinge_gap
            p2_bottom = p2_top + panel2_height
            # Slight angle to show it's tilted
            offset = 15
            self.viz_canvas.create_polygon(
                x + offset, p2_top,
                x + unit_width - offset, p2_top,
                x + unit_width, p2_bottom,
                x, p2_bottom,
                fill=color2, outline="#555555", width=2
            )
            self.viz_canvas.create_text(
                x + unit_width // 2, p2_top + 25,
                text="2", fill="#333333", font=("Helvetica", 14, "bold")
            )
            dmx2 = int(MIN_DMX_FRONT + val2 * (MAX_DMX_FRONT - MIN_DMX_FRONT))
            self.viz_canvas.create_text(
                x + unit_width // 2, p2_bottom - 15,
                text=f"{dmx2}", fill="#444444", font=("Courier", 11)
            )
            
            # Draw Panel 3 (back - behind panel 2, shown offset)
            p3_top = p2_top + 20
            p3_bottom = p3_top + panel3_height
            p3_offset = 25  # Offset to show it's behind
            self.viz_canvas.create_polygon(
                x - p3_offset, p3_top,
                x + unit_width // 2 - 20, p3_top,
                x + unit_width // 2 - 10, p3_bottom,
                x - p3_offset + 10, p3_bottom,
                fill=color3, outline="#666666", width=2
            )
            self.viz_canvas.create_text(
                x - p3_offset + 30, p3_top + panel3_height // 2,
                text="3", fill="#333333", font=("Helvetica", 12, "bold")
            )
            dmx3 = int(MIN_DMX_BACK + val3 * (MAX_DMX_BACK - MIN_DMX_BACK))
            self.viz_canvas.create_text(
                x - p3_offset + 30, p3_bottom - 12,
                text=f"{dmx3}", fill="#444444", font=("Courier", 10)
            )
            
            # Unit label
            self.viz_canvas.create_text(
                x + unit_width // 2, p2_bottom + 25,
                text=f"Unit {unit + 1}", fill="#888888", font=("Helvetica", 10)
            )
        
        # Draw vector indicator
        vec_x = self.controller.config.vector_dx
        vec_y = self.controller.config.vector_dy
        arrow_start_x = w // 2
        arrow_start_y = h - 60
        arrow_len = 50
        mag = math.sqrt(vec_x**2 + vec_y**2)
        if mag > 0.01:
            arrow_end_x = arrow_start_x + (vec_x / mag) * arrow_len
            arrow_end_y = arrow_start_y - (vec_y / mag) * arrow_len * 0.5
            self.viz_canvas.create_line(
                arrow_start_x, arrow_start_y,
                arrow_end_x, arrow_end_y,
                fill="#ffaa44", width=3, arrow=tk.LAST
            )
        self.viz_canvas.create_text(
            w // 2, h - 30,
            text=f"Vector: ({vec_x:.2f}, {vec_y:.2f})",
            fill="#888888", font=("Helvetica", 10)
        )
        
        # Ripple indicator
        if self.controller.ripple_intensity > 0.01:
            ripple_x = start_x + self.controller.ripple_center_x * total_width
            ripple_r = self.controller.ripple_intensity * 50
            self.viz_canvas.create_oval(
                ripple_x - ripple_r, base_y - ripple_r,
                ripple_x + ripple_r, base_y + ripple_r,
                outline="#ff6644", width=2
            )
    
    def update_loop(self):
        if not self.running:
            return
        
        state = self.controller.update()
        
        # Cache values for drawing
        self.panel_values_display = dict(self.controller.panel_values)
        
        self.draw_visualization()
        
        self.root.after(int(1000 / FPS), self.update_loop)
    
    def on_closing(self):
        self.running = False
        self.controller.shutdown()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vector LED Controller')
    parser.add_argument('--no-artnet', action='store_true', help='Run without Art-Net')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = VectorControllerGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
