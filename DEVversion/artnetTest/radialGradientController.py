#!/usr/bin/env python3
"""
Radial Gradient Pulse Controller
An animated radial gradient expands from origin.
Simple greyscale mapping with min/max control.
"""

import tkinter as tk
from tkinter import ttk
import math
import time
import random
from stupidArtnet import StupidArtnet

# =============================================================================
# CONFIG
# =============================================================================

TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

# DMX range for all panels (unified)
MIN_DMX = 1
MAX_DMX = 50


class RadialGradientController:
    def __init__(self):
        self.time = 0.0
        self.last_update = time.time()
        
        # Target values (set by sliders)
        self.target_origin_x = 1.5
        self.target_speed = 1.0
        self.target_phase_scale = 1.0
        self.target_min_dmx = 1
        self.target_max_dmx = 50
        self.target_noise = 2
        
        # Current values (lerp toward targets)
        self.origin_x = 1.5
        self.speed = 1.0
        self.phase_scale = 1.0
        self.min_dmx = 1.0
        self.max_dmx = 50.0
        self.noise_amount = 2.0
        
        # Lerp speed (lower = slower/smoother transition)
        self.lerp_speed = 0.5  # Per second - takes ~2 seconds to reach target
        
        # Unit DMX values
        self.unit_values = [25, 25, 25, 25]
        self.noise_seeds = [random.random() * 100 for _ in range(4)]
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX] * 12
    
    def init_artnet(self) -> bool:
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            self.artnet.start()
            return True
        except Exception as e:
            print(f"Art-Net failed: {e}")
            return False
    
    def _lerp(self, current: float, target: float, t: float) -> float:
        """Linear interpolation"""
        return current + (target - current) * min(1.0, t)
    
    def update(self):
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        self.time += dt
        
        # Smoothly lerp all parameters toward their targets
        lerp_t = self.lerp_speed * dt
        self.origin_x = self._lerp(self.origin_x, self.target_origin_x, lerp_t)
        self.speed = self._lerp(self.speed, self.target_speed, lerp_t)
        self.phase_scale = self._lerp(self.phase_scale, self.target_phase_scale, lerp_t)
        self.min_dmx = self._lerp(self.min_dmx, self.target_min_dmx, lerp_t)
        self.max_dmx = self._lerp(self.max_dmx, self.target_max_dmx, lerp_t)
        self.noise_amount = self._lerp(self.noise_amount, self.target_noise, lerp_t)
        
        for unit in range(4):
            x = unit
            
            # Distance from origin determines phase offset
            dist = abs(x - self.origin_x)
            phase_offset = dist * self.phase_scale
            
            # All panels pulse continuously - phase offset creates radial appearance
            pulse = math.sin((self.time * self.speed - phase_offset) * math.pi)
            
            # Normalize pulse from -1,1 to 0,1
            normalized = (pulse + 1) / 2
            
            # Map to min/max DMX
            dmx_value = self.min_dmx + normalized * (self.max_dmx - self.min_dmx)
            
            # Add noise (in DMX units)
            noise = math.sin(self.noise_seeds[unit] + self.time * 2.5) * self.noise_amount
            dmx_value += noise
            
            self.unit_values[unit] = max(MIN_DMX, min(MAX_DMX, dmx_value))
        
        self._send_artnet()
    
    def _send_artnet(self):
        if not self.artnet:
            return
        
        for unit in range(4):
            col = 3 - unit
            base_dmx = self.unit_values[col]
            
            for panel in range(3):
                ch = unit * 3 + panel
                
                # Small per-panel noise variation
                panel_noise = math.sin(self.noise_seeds[col] * (panel + 1) + self.time * 2) * self.noise_amount * 0.3
                dmx = int(base_dmx + panel_noise)
                dmx = max(MIN_DMX, min(MAX_DMX, dmx))
                
                self.channel_values[ch] = dmx
        
        self.artnet.set(self.channel_values)
    
    def shutdown(self):
        if self.artnet:
            self.channel_values = [MIN_DMX] * 12
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


class RadialGradientGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("Radial Gradient Pulse")
        self.root.geometry("900x600")
        
        self.controller = RadialGradientController()
        
        if not no_artnet:
            self.controller.init_artnet()
        
        self.setup_gui()
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Visualization
        self.canvas = tk.Canvas(main, bg="#1a1a2a", height=350)
        self.canvas.pack(fill="both", expand=True, pady=(0, 10))
        
        # Controls
        ctrl = ttk.LabelFrame(main, text="Controls", padding=10)
        ctrl.pack(fill="x")
        
        # Speed
        self.create_slider(ctrl, "Speed", 0.1, 3.0, 1.0, self.on_speed)
        
        # Origin (phase offset center)
        self.create_slider(ctrl, "Origin X", -1.0, 4.0, 1.5, self.on_origin)
        
        # Phase Scale (how much distance offsets phase)
        self.create_slider(ctrl, "Phase Scale", 0.0, 3.0, 1.0, self.on_scale)
        
        # Min/Max DMX (actual DMX values)
        self.create_slider(ctrl, "Min DMX", 1, 50, 1, self.on_min, is_int=True)
        self.create_slider(ctrl, "Max DMX", 1, 50, 50, self.on_max, is_int=True)
        
        # Noise (in DMX units)
        self.create_slider(ctrl, "Noise", 0, 10, 2, self.on_noise, is_int=True)
    
    def create_slider(self, parent, label, min_v, max_v, default, callback, is_int=False):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        
        ttk.Label(frame, text=f"{label}:", width=14).pack(side="left")
        
        var = tk.DoubleVar(value=default)
        fmt = "{:.0f}" if is_int else "{:.2f}"
        val_label = ttk.Label(frame, text=fmt.format(default), width=6)
        val_label.pack(side="right")
        
        def on_change(val, lbl=val_label, cb=callback, f=fmt, integer=is_int):
            v = float(val)
            if integer:
                v = int(v)
            lbl.config(text=f.format(v))
            cb(v)
        
        ttk.Scale(frame, from_=min_v, to=max_v, variable=var,
                 orient="horizontal", command=on_change).pack(side="left", fill="x", expand=True, padx=5)
    
    def on_speed(self, v): self.controller.target_speed = v
    def on_origin(self, v): self.controller.target_origin_x = v
    def on_scale(self, v): self.controller.target_phase_scale = v
    def on_min(self, v): self.controller.target_min_dmx = int(v)
    def on_max(self, v): self.controller.target_max_dmx = int(v)
    def on_noise(self, v): self.controller.target_noise = int(v)
    
    def draw(self):
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50:
            return
        
        unit_w = 150
        spacing = 30
        total = unit_w * 4 + spacing * 3
        start_x = (w - total) // 2
        bar_h = 200
        base_y = 60
        
        # Draw units
        for unit in range(4):
            display = 3 - unit
            x = start_x + display * (unit_w + spacing)
            dmx_val = self.controller.unit_values[unit]
            
            # Greyscale color based on DMX value (1-200 mapped to grey)
            normalized = (dmx_val - MIN_DMX) / (MAX_DMX - MIN_DMX)
            grey = int(50 + normalized * 205)
            color = f"#{grey:02x}{grey:02x}{grey:02x}"
            
            self.canvas.create_rectangle(
                x, base_y, x + unit_w, base_y + bar_h,
                fill=color, outline="#555", width=3
            )
            
            # Unit label
            self.canvas.create_text(
                x + unit_w // 2, base_y + bar_h + 20,
                text=f"Unit {unit + 1}", fill="#888", font=("Helvetica", 11)
            )
            
            # DMX value (same for all panels)
            text_color = "#000" if normalized > 0.5 else "#fff"
            self.canvas.create_text(
                x + unit_w // 2, base_y + bar_h // 2,
                text=f"DMX: {int(dmx_val)}",
                fill=text_color, font=("Courier", 14, "bold")
            )
            
            # Show phase offset indicator
            dist = abs(unit - self.controller.origin_x)
            phase_offset = dist * self.controller.phase_scale
            self.canvas.create_text(
                x + unit_w // 2, base_y - 15,
                text=f"φ {phase_offset:.1f}", fill="#aaa", font=("Helvetica", 9)
            )
        
        # Origin marker
        origin_canvas_x = start_x + (3 - self.controller.origin_x) * (unit_w + spacing) + unit_w // 2
        self.canvas.create_line(
            origin_canvas_x, base_y - 30, origin_canvas_x, base_y + bar_h + 10,
            fill="#ff8844", width=2, dash=(4, 2)
        )
        self.canvas.create_text(
            origin_canvas_x, base_y - 40,
            text="Origin", fill="#ff8844", font=("Helvetica", 9)
        )
    
    def update_loop(self):
        if not self.running:
            return
        
        self.controller.update()
        self.draw()
        
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
    app = RadialGradientGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
