#!/usr/bin/env python3
"""
Simple Linear Wave Controller
Just a wave sweeping across X. All 3 panels per unit respond together.
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

MIN_DMX_FRONT = 1
MAX_DMX_FRONT = 50
MIN_DMX_BACK = 1
MAX_DMX_BACK = 200


class SimpleWaveController:
    def __init__(self):
        self.time = 0.0
        self.last_update = time.time()
        
        # Controls
        self.origin_x = -1.0      # -2 to 5
        self.wave_speed = 1.0     # 0.1 to 3
        self.wave_width = 2.0     # 0.5 to 4
        self.noise_amount = 0.05  # 0 to 0.2
        
        # Unit values (0-1), all 3 panels per unit share same base value
        self.unit_values = [0.5, 0.5, 0.5, 0.5]
        
        # Noise per panel
        self.noise_seeds = [[random.random() * 100 for _ in range(3)] for _ in range(4)]
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX_FRONT] * 12
    
    def init_artnet(self) -> bool:
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            self.artnet.start()
            return True
        except Exception as e:
            print(f"Art-Net failed: {e}")
            return False
    
    def update(self):
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now
        self.time += dt
        
        t = self.time
        
        # Calculate value for each unit (x = 0, 1, 2, 3)
        for unit in range(4):
            x = unit
            
            # Distance from origin
            dist = x - self.origin_x
            
            # Wave: moves outward from origin over time
            phase = dist - t * self.wave_speed
            wave = math.sin(phase * math.pi / self.wave_width)
            
            # Normalize to 0-1
            self.unit_values[unit] = (wave + 1) / 2
        
        self._send_artnet()
    
    def _send_artnet(self):
        if not self.artnet:
            return
        
        for unit in range(4):
            col = 3 - unit  # Reverse mapping
            base_val = self.unit_values[col]
            
            for panel in range(3):
                ch = unit * 3 + panel
                
                # Add small noise per panel
                noise = math.sin(self.noise_seeds[col][panel] + self.time * 2) * self.noise_amount
                val = max(0.0, min(1.0, base_val + noise))
                
                if panel < 2:
                    dmx = int(MIN_DMX_FRONT + val * (MAX_DMX_FRONT - MIN_DMX_FRONT))
                    dmx = max(MIN_DMX_FRONT, min(MAX_DMX_FRONT, dmx))
                else:
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


class SimpleWaveGUI:
    def __init__(self, root, no_artnet=False):
        self.root = root
        self.root.title("Simple Linear Wave")
        self.root.geometry("900x550")
        
        self.controller = SimpleWaveController()
        
        if not no_artnet:
            self.controller.init_artnet()
        
        self.setup_gui()
        self.running = True
        self.update_loop()
    
    def setup_gui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Visualization
        self.canvas = tk.Canvas(main, bg="#1a1a2a", height=300)
        self.canvas.pack(fill="both", expand=True, pady=(0, 10))
        
        # Controls
        ctrl = ttk.LabelFrame(main, text="Controls", padding=10)
        ctrl.pack(fill="x")
        
        # Origin X
        f1 = ttk.Frame(ctrl)
        f1.pack(fill="x", pady=5)
        ttk.Label(f1, text="Origin X:", width=12).pack(side="left")
        self.origin_var = tk.DoubleVar(value=-1.0)
        self.origin_label = ttk.Label(f1, text="-1.00", width=6)
        self.origin_label.pack(side="right")
        ttk.Scale(f1, from_=-2, to=5, variable=self.origin_var,
                 orient="horizontal", command=self.on_origin).pack(side="left", fill="x", expand=True, padx=5)
        
        # Wave Speed
        f2 = ttk.Frame(ctrl)
        f2.pack(fill="x", pady=5)
        ttk.Label(f2, text="Wave Speed:", width=12).pack(side="left")
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_label = ttk.Label(f2, text="1.00", width=6)
        self.speed_label.pack(side="right")
        ttk.Scale(f2, from_=0.1, to=3.0, variable=self.speed_var,
                 orient="horizontal", command=self.on_speed).pack(side="left", fill="x", expand=True, padx=5)
        
        # Wave Width
        f3 = ttk.Frame(ctrl)
        f3.pack(fill="x", pady=5)
        ttk.Label(f3, text="Wave Width:", width=12).pack(side="left")
        self.width_var = tk.DoubleVar(value=2.0)
        self.width_label = ttk.Label(f3, text="2.00", width=6)
        self.width_label.pack(side="right")
        ttk.Scale(f3, from_=0.5, to=4.0, variable=self.width_var,
                 orient="horizontal", command=self.on_width).pack(side="left", fill="x", expand=True, padx=5)
        
        # Noise
        f4 = ttk.Frame(ctrl)
        f4.pack(fill="x", pady=5)
        ttk.Label(f4, text="Noise:", width=12).pack(side="left")
        self.noise_var = tk.DoubleVar(value=0.05)
        self.noise_label = ttk.Label(f4, text="0.05", width=6)
        self.noise_label.pack(side="right")
        ttk.Scale(f4, from_=0, to=0.2, variable=self.noise_var,
                 orient="horizontal", command=self.on_noise).pack(side="left", fill="x", expand=True, padx=5)
    
    def on_origin(self, val):
        v = float(val)
        self.origin_label.config(text=f"{v:.2f}")
        self.controller.origin_x = v
    
    def on_speed(self, val):
        v = float(val)
        self.speed_label.config(text=f"{v:.2f}")
        self.controller.wave_speed = v
    
    def on_width(self, val):
        v = float(val)
        self.width_label.config(text=f"{v:.2f}")
        self.controller.wave_width = v
    
    def on_noise(self, val):
        v = float(val)
        self.noise_label.config(text=f"{v:.2f}")
        self.controller.noise_amount = v
    
    def draw(self):
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50:
            return
        
        # Draw 4 units as simple rectangles
        unit_w = 150
        spacing = 30
        total = unit_w * 4 + spacing * 3
        start_x = (w - total) // 2
        bar_h = 180
        base_y = 50
        
        for unit in range(4):
            display = 3 - unit
            x = start_x + display * (unit_w + spacing)
            val = self.controller.unit_values[unit]
            
            # Color based on value
            brightness = int(80 + val * 175)
            color = f"#{brightness:02x}{brightness:02x}{int(180 + val * 75):02x}"
            
            # Main rectangle
            self.canvas.create_rectangle(
                x, base_y, x + unit_w, base_y + bar_h,
                fill=color, outline="#555", width=3
            )
            
            # Unit label
            self.canvas.create_text(
                x + unit_w // 2, base_y + bar_h + 20,
                text=f"Unit {unit + 1}", fill="#888", font=("Helvetica", 11)
            )
            
            # DMX value
            dmx_front = int(MIN_DMX_FRONT + val * (MAX_DMX_FRONT - MIN_DMX_FRONT))
            dmx_back = int(MIN_DMX_BACK + val * (MAX_DMX_BACK - MIN_DMX_BACK))
            self.canvas.create_text(
                x + unit_w // 2, base_y + bar_h // 2,
                text=f"P1,P2: {dmx_front}\nP3: {dmx_back}",
                fill="#333", font=("Courier", 10)
            )
        
        # Draw origin marker
        origin_x = start_x + (3 - self.controller.origin_x) * (unit_w + spacing) + unit_w // 2
        self.canvas.create_line(origin_x, 20, origin_x, base_y + bar_h + 10, fill="#ff8844", width=2, dash=(4, 2))
        self.canvas.create_oval(origin_x - 6, 14, origin_x + 6, 26, fill="#ff8844", outline="#aa5522")
    
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
    app = SimpleWaveGUI(root, no_artnet=args.no_artnet)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
