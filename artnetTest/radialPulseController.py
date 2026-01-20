#!/usr/bin/env python3
"""
3D Point Light LED Controller

An animated pulsing point light in 3D space. Each panel has a collector point
at its center. Brightness is calculated from 3D distance with falloff.

Physical Layout (60cm x 60cm panels):
  - Units spaced 60cm apart (center to center)
  - Panel 1: top front, center at y=0.9m (top of 60cm panel)
  - Panel 2: bottom front, angled, center at y=0.3m 
  - Panel 3: back, center at z=0.3m behind front
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import random
from dataclasses import dataclass
from stupidArtnet import StupidArtnet

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

# DMX range (same for all panels)
MIN_DMX = 1
MAX_DMX = 50

# Panel size in meters
PANEL_SIZE = 0.6  # 60cm

# Panel 3D center positions in meters (x, y, z)
# x: 0, 0.6, 1.2, 1.8 for units 1-4
# y: height from ground
# z: depth (0 = front plane, positive = toward street)
PANEL_POSITIONS_3D = {
    # Unit 1
    (0, 0): (0.0, 0.9, 0.0),   # Panel 1 (top front)
    (0, 1): (0.0, 0.3, 0.0),   # Panel 2 (bottom front)
    (0, 2): (0.0, 0.3, 0.3),   # Panel 3 (back)
    # Unit 2
    (1, 0): (0.6, 0.9, 0.0),
    (1, 1): (0.6, 0.3, 0.0),
    (1, 2): (0.6, 0.3, 0.3),
    # Unit 3
    (2, 0): (1.2, 0.9, 0.0),
    (2, 1): (1.2, 0.3, 0.0),
    (2, 2): (1.2, 0.3, 0.3),
    # Unit 4
    (3, 0): (1.8, 0.9, 0.0),
    (3, 1): (1.8, 0.3, 0.0),
    (3, 2): (1.8, 0.3, 0.3),
}


@dataclass
class PulseConfig:
    """3D Point Light configuration"""
    
    # Master speed
    master_speed: float = 0.5
    
    # Light origin in meters (can extend beyond panel grid)
    origin_x: float = 0.9    # Center of grid
    origin_y: float = 0.6    # Middle height
    origin_z: float = -0.5   # In front of panels
    
    # Pulse properties
    pulse_speed: float = 1.0   # Pulse animation speed
    pulse_amplitude: float = 0.5  # How much light pulses (0-1)
    
    # Falloff (inverse square law modified)
    falloff: float = 2.0       # Falloff exponent (2 = inverse square)
    max_distance: float = 3.0  # Distance at which brightness = 0
    
    # Noise
    noise_amount: float = 0.05
    noise_speed: float = 0.5
    
    # DMX output range
    dmx_min: int = 1
    dmx_max: int = 50


class RadialPulseController:
    """3D Point Light controller"""
    
    def __init__(self, config: PulseConfig = None):
        self.config = config or PulseConfig()
        
        self.time = 0.0
        self.last_update = time.time()
        
        # Panel values (normalized 0-1)
        self.panel_values = {key: 0.5 for key in PANEL_POSITIONS_3D}
        
        # Noise seeds per panel
        self.noise_seeds = {key: random.random() * 100 for key in PANEL_POSITIONS_3D}
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX] * 12
    
    def init_artnet(self) -> bool:
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            self.artnet.start()
            return True
        except Exception as e:
            print(f"Art-Net init failed: {e}")
            return False
    
    def _noise(self, seed: float, t: float) -> float:
        """Simple noise -1 to 1"""
        return math.sin(seed * 12.9898 + t) * math.sin(seed * 78.233 + t * 0.7)
    
    def update(self):
        """Main update"""
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        self.time += dt * self.config.master_speed
        
        cfg = self.config
        t = self.time
        
        # Pulsing light intensity (oscillates between 1-amplitude and 1)
        pulse = (math.sin(t * cfg.pulse_speed * math.pi) + 1) / 2
        light_intensity = (1 - cfg.pulse_amplitude) + pulse * cfg.pulse_amplitude
        
        for (unit, panel), (px, py, pz) in PANEL_POSITIONS_3D.items():
            # 3D distance from light origin to panel center
            dx = px - cfg.origin_x
            dy = py - cfg.origin_y
            dz = pz - cfg.origin_z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            
            # Falloff calculation (inverse power law)
            if dist < 0.01:
                falloff_factor = 1.0
            else:
                # Normalize distance and apply falloff
                normalized_dist = dist / cfg.max_distance
                falloff_factor = max(0.0, 1.0 - pow(normalized_dist, 1.0 / cfg.falloff))
            
            # Apply light intensity and falloff
            brightness = light_intensity * falloff_factor
            
            # Add noise
            noise = self._noise(self.noise_seeds[(unit, panel)], t * cfg.noise_speed)
            brightness += noise * cfg.noise_amount
            
            # Clamp
            self.panel_values[(unit, panel)] = max(0.0, min(1.0, brightness))
        
        self._send_artnet()
    
    def _send_artnet(self):
        if not self.artnet:
            return
        
        cfg = self.config
        
        for unit in range(4):
            col = 3 - unit  # Reverse for physical layout
            
            for panel in range(3):
                ch = unit * 3 + panel
                val = self.panel_values[(col, panel)]
                
                # Same DMX range for all panels
                dmx = int(cfg.dmx_min + val * (cfg.dmx_max - cfg.dmx_min))
                dmx = max(MIN_DMX, min(MAX_DMX, dmx))
                
                self.channel_values[ch] = dmx
        
        self.artnet.set(self.channel_values)
    
    def shutdown(self):
        if self.artnet:
            self.channel_values = [MIN_DMX] * 12
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


# =============================================================================
# GUI
# =============================================================================

class RadialPulseGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("3D Point Light Controller")
        self.root.geometry("1100x750")
        
        self.controller = RadialPulseController()
        
        if not no_artnet:
            self.controller.init_artnet()
        
        self.setup_gui()
        
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Visualization
        viz_frame = ttk.LabelFrame(main, text="Panel Visualization", padding=10)
        viz_frame.pack(side="left", fill="both", expand=True)
        
        self.canvas = tk.Canvas(viz_frame, bg="#1a1a2a", width=650, height=500)
        self.canvas.pack(fill="both", expand=True)
        
        # Right: Controls (scrollable)
        ctrl_container = ttk.Frame(main, width=380)
        ctrl_container.pack(side="right", fill="y", padx=(10, 0))
        ctrl_container.pack_propagate(False)
        
        canvas_ctrl = tk.Canvas(ctrl_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(ctrl_container, orient="vertical", command=canvas_ctrl.yview)
        ctrl_frame = ttk.Frame(canvas_ctrl)
        
        ctrl_frame.bind("<Configure>", lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))
        canvas_ctrl.create_window((0, 0), window=ctrl_frame, anchor="nw")
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)
        
        canvas_ctrl.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.sliders = {}
        
        # Master Speed
        speed_frame = ttk.LabelFrame(ctrl_frame, text="★ Master Speed", padding=10)
        speed_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(speed_frame, "master_speed", "Speed", 0.1, 2.0)
        
        # Light Origin Position (3D in meters)
        origin_frame = ttk.LabelFrame(ctrl_frame, text="Light Origin (meters)", padding=10)
        origin_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(origin_frame, "origin_x", "X (left-right)", -0.5, 2.5)
        self.create_slider(origin_frame, "origin_y", "Y (up-down)", -0.5, 1.5)
        self.create_slider(origin_frame, "origin_z", "Z (front-back)", -1.5, 1.0)
        
        # Quick origin buttons
        btn_frame = ttk.Frame(origin_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Left", command=lambda: self.set_origin(0.0, 0.6, -0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Center", command=lambda: self.set_origin(0.9, 0.6, -0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Right", command=lambda: self.set_origin(1.8, 0.6, -0.5)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Close", command=lambda: self.set_origin(0.9, 0.6, -0.2)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Far", command=lambda: self.set_origin(0.9, 0.6, -1.0)).pack(side="left", padx=2)
        
        # Pulse Properties
        pulse_frame = ttk.LabelFrame(ctrl_frame, text="Light Pulse", padding=10)
        pulse_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(pulse_frame, "pulse_speed", "Pulse Speed", 0.2, 3.0)
        self.create_slider(pulse_frame, "pulse_amplitude", "Amplitude", 0.0, 1.0)
        
        # Falloff
        falloff_frame = ttk.LabelFrame(ctrl_frame, text="Distance Falloff", padding=10)
        falloff_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(falloff_frame, "falloff", "Falloff Exp", 0.5, 4.0)
        self.create_slider(falloff_frame, "max_distance", "Max Distance", 1.0, 5.0)
        
        # Noise
        noise_frame = ttk.LabelFrame(ctrl_frame, text="Noise", padding=10)
        noise_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(noise_frame, "noise_amount", "Amount", 0.0, 0.2)
        self.create_slider(noise_frame, "noise_speed", "Speed", 0.1, 2.0)
        
        # DMX Range
        dmx_frame = ttk.LabelFrame(ctrl_frame, text="DMX Output Range", padding=10)
        dmx_frame.pack(fill="x", pady=(0, 10))
        self.create_slider(dmx_frame, "dmx_min", "Min DMX", 0, 50)
        self.create_slider(dmx_frame, "dmx_max", "Max DMX", 10, 255)
    
    def create_slider(self, parent, key: str, label: str, min_v: float, max_v: float):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=12, anchor="w").pack(side="left")
        
        current = getattr(self.controller.config, key)
        var = tk.DoubleVar(value=current)
        self.sliders[key] = var
        
        val_label = ttk.Label(frame, text=f"{current:.2f}", width=5)
        val_label.pack(side="right")
        
        def on_change(val, k=key, lbl=val_label):
            v = float(val)
            lbl.config(text=f"{v:.2f}")
            setattr(self.controller.config, k, v)
        
        ttk.Scale(frame, from_=min_v, to=max_v, variable=var,
                 orient="horizontal", command=on_change).pack(side="left", fill="x", expand=True, padx=5)
    
    def set_origin(self, x: float, y: float, z: float):
        self.sliders["origin_x"].set(x)
        self.sliders["origin_y"].set(y)
        self.sliders["origin_z"].set(z)
        self.controller.config.origin_x = x
        self.controller.config.origin_y = y
        self.controller.config.origin_z = z
    
    def draw_visualization(self):
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 100:
            return
        
        # Layout
        unit_width = 120
        unit_spacing = 25
        total_width = unit_width * 4 + unit_spacing * 3
        start_x = (w - total_width) // 2
        
        panel1_h = 100
        panel2_h = 70
        panel3_h = 60
        base_y = 80
        hinge_gap = 6
        
        cfg = self.controller.config
        
        for unit in range(4):
            display_unit = 3 - unit
            x = start_x + display_unit * (unit_width + unit_spacing)
            
            val1 = self.controller.panel_values.get((unit, 0), 0.5)
            val2 = self.controller.panel_values.get((unit, 1), 0.5)
            val3 = self.controller.panel_values.get((unit, 2), 0.5)
            
            def val_to_color(v, is_back=False):
                if is_back:
                    r = int(100 + v * 155)
                    g = int(80 + v * 120)
                    b = int(150 + v * 105)
                else:
                    r = int(80 + v * 175)
                    g = int(100 + v * 155)
                    b = int(180 + v * 75)
                return f"#{r:02x}{g:02x}{b:02x}"
            
            # Panel 1 (top front)
            p1_top = base_y
            p1_bot = base_y + panel1_h
            self.canvas.create_rectangle(
                x, p1_top, x + unit_width, p1_bot,
                fill=val_to_color(val1), outline="#555", width=2
            )
            dmx1 = int(cfg.dmx_min + val1 * (cfg.dmx_max - cfg.dmx_min))
            self.canvas.create_text(x + unit_width//2, p1_top + 15, text="1", fill="#333", font=("Helvetica", 14, "bold"))
            self.canvas.create_text(x + unit_width//2, p1_bot - 15, text=f"{dmx1}", fill="#444", font=("Courier", 10))
            self.canvas.create_text(x + unit_width//2, p1_bot - 15, text=f"{dmx1}", fill="#444", font=("Courier", 10))
            
            # Hinge
            hy = p1_bot + hinge_gap // 2
            self.canvas.create_polygon(
                x + unit_width//2 - 8, hy - 4,
                x + unit_width//2 + 8, hy - 4,
                x + unit_width//2, hy + 6,
                fill="#ff8844"
            )
            
            # Panel 2 (bottom front, angled)
            p2_top = p1_bot + hinge_gap
            p2_bot = p2_top + panel2_h
            off = 12
            self.canvas.create_polygon(
                x + off, p2_top,
                x + unit_width - off, p2_top,
                x + unit_width, p2_bot,
                x, p2_bot,
                fill=val_to_color(val2), outline="#555", width=2
            )
            dmx2 = int(cfg.dmx_min + val2 * (cfg.dmx_max - cfg.dmx_min))
            self.canvas.create_text(x + unit_width//2, p2_top + 20, text="2", fill="#333", font=("Helvetica", 12, "bold"))
            self.canvas.create_text(x + unit_width//2, p2_bot - 12, text=f"{dmx2}", fill="#444", font=("Courier", 9))
            
            # Panel 3 (back)
            p3_top = p2_top + 15
            p3_bot = p3_top + panel3_h
            p3_off = 20
            self.canvas.create_polygon(
                x - p3_off, p3_top,
                x + unit_width//2 - 15, p3_top,
                x + unit_width//2 - 8, p3_bot,
                x - p3_off + 8, p3_bot,
                fill=val_to_color(val3, True), outline="#666", width=2
            )
            dmx3 = int(cfg.dmx_min + val3 * (cfg.dmx_max - cfg.dmx_min))
            self.canvas.create_text(x - p3_off + 25, p3_top + panel3_h//2, text="3", fill="#333", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(x - p3_off + 25, p3_bot - 10, text=f"{dmx3}", fill="#444", font=("Courier", 8))
            
            # Unit label
            self.canvas.create_text(x + unit_width//2, p2_bot + 20, text=f"Unit {unit+1}", fill="#888", font=("Helvetica", 9))
        
        # Draw origin point (map 3D coords to canvas)
        # X: 0-1.8m maps across units, Z affects size (perspective)
        z_scale = max(0.3, 1.0 + cfg.origin_z)  # Further back = smaller
        origin_canvas_x = start_x + (1.8 - cfg.origin_x) / 1.8 * total_width
        origin_canvas_y = base_y + panel1_h - cfg.origin_y / 1.2 * (panel1_h + hinge_gap + panel2_h)
        
        # Pulsing light effect
        pulse = (math.sin(self.controller.time * cfg.pulse_speed * math.pi) + 1) / 2
        light_radius = 12 + pulse * 8
        
        # Draw glow rings based on max_distance
        for i in range(3):
            ring_r = (i + 1) * 40 * z_scale
            alpha = max(0, int(100 - i * 30))
            self.canvas.create_oval(
                origin_canvas_x - ring_r, origin_canvas_y - ring_r * 0.6,
                origin_canvas_x + ring_r, origin_canvas_y + ring_r * 0.6,
                outline=f"#ff{alpha:02x}44", width=1, dash=(3, 3)
            )
        
        # Origin light (pulsing)
        brightness = int(170 + pulse * 85)
        self.canvas.create_oval(
            origin_canvas_x - light_radius, origin_canvas_y - light_radius,
            origin_canvas_x + light_radius, origin_canvas_y + light_radius,
            fill=f"#ff{brightness:02x}44", outline="#ff6622", width=2
        )
        
        # Origin label with Z
        self.canvas.create_text(
            origin_canvas_x, origin_canvas_y + light_radius + 15,
            text=f"({cfg.origin_x:.1f}, {cfg.origin_y:.1f}, {cfg.origin_z:.1f})m",
            fill="#ffaa44", font=("Helvetica", 9)
        )
    
    def update_loop(self):
        if not self.running:
            return
        
        self.controller.update()
        self.draw_visualization()
        
        self.root.after(int(1000 / FPS), self.update_loop)
    
    def on_closing(self):
        self.running = False
        self.controller.shutdown()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-artnet', action='store_true')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = RadialPulseGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
