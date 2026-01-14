#!/usr/bin/env python3
"""
LED Panel Controller via Art-Net
Controls 12 LED panels (4 units × 3 panels each) through a DMX decoder.

Art-Net → DMX Decoder → 0-10V signal → LED Panels

Network Setup:
- Computer: 10.42.0.1 (macOS Internet Sharing host)
- Art-Net Device: 10.42.0.200 (set this IP on your DMX decoder)
- Cameras: 10.42.0.76, 10.42.0.173, etc.
"""

import tkinter as tk
from tkinter import ttk
from stupidArtnet import StupidArtnet
import random
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Art-Net device IP - Set your DMX decoder to this address
# Using 10.42.0.200 to match your network (10.42.0.x subnet)
TARGET_IP = "10.42.0.200"

# Art-Net universe (0-15 typically, check your decoder settings)
UNIVERSE = 0

# Number of LED panels
NUM_UNITS = 4
PANELS_PER_UNIT = 3
TOTAL_CHANNELS = NUM_UNITS * PANELS_PER_UNIT  # 12 channels

# DMX frame rate (Art-Net spec allows up to 44fps, 30 is safe)
FPS = 30

# Maximum DMX value (212 = 10V output from 12V decoder)
# This prevents over-driving the LED panels
MAX_DMX_VALUE = 212

# =============================================================================
# ART-NET SETUP
# =============================================================================

# Create Art-Net sender (not server - we're sending DMX data)
try:
    artnet = StupidArtnet(TARGET_IP, UNIVERSE, TOTAL_CHANNELS, FPS)
    artnet.start()
    print(f"✓ Art-Net initialized: {TARGET_IP} Universe {UNIVERSE}")
except Exception as e:
    print(f"✗ Art-Net initialization failed: {e}")
    print("  Check that the target IP is reachable on your network")
    sys.exit(1)

# Store current channel values (0-255 DMX range)
channel_values = [0] * TOTAL_CHANNELS
sliders = []  # Store slider references for master controls


def send_artnet():
    """Send current channel values via Art-Net"""
    artnet.set(channel_values)


def update_channel(channel_index, value):
    """Update a channel value and send Art-Net packet"""
    channel_values[channel_index] = int(float(value))
    send_artnet()


def set_all(value):
    """Set all channels to a specific value"""
    for i, slider in enumerate(sliders):
        slider.set(value)
    # Values are sent via the slider callbacks


def fade_all(target, duration_ms=1000, steps=30):
    """Fade all channels to target value over duration"""
    if not sliders:
        return
    
    current = [s.get() for s in sliders]
    step_delay = duration_ms // steps
    
    def fade_step(step):
        if step > steps:
            return
        progress = step / steps
        for i, slider in enumerate(sliders):
            new_val = int(current[i] + (target - current[i]) * progress)
            slider.set(new_val)
        root.after(step_delay, lambda: fade_step(step + 1))
    
    fade_step(1)


def chase_effect():
    """Simple chase effect through all panels"""
    def chase_step(channel):
        set_all(0)
        if channel < TOTAL_CHANNELS:
            sliders[channel].set(MAX_DMX_VALUE)
            root.after(100, lambda: chase_step(channel + 1))
    chase_step(0)


# Random fade mode state
random_fade_active = False
random_fade_id = None

def random_fade_mode():
    """Toggle random fade mode - panels slowly fade between random values 0-50"""
    global random_fade_active, random_fade_id
    
    if random_fade_active:
        # Stop the mode
        random_fade_active = False
        if random_fade_id:
            root.after_cancel(random_fade_id)
            random_fade_id = None
        random_btn.config(text="Random Fade")
        return
    
    # Start the mode
    random_fade_active = True
    random_btn.config(text="Stop Random")
    
    # Store current and target values for smooth fading
    targets = [random.randint(0, 50) for _ in range(TOTAL_CHANNELS)]
    
    def fade_step():
        global random_fade_id
        if not random_fade_active:
            return
        
        all_reached = True
        for i, slider in enumerate(sliders):
            current = int(slider.get())
            target = targets[i]
            
            if current != target:
                all_reached = False
                # Move slowly toward target (1-2 units per step)
                if current < target:
                    slider.set(min(current + 2, target))
                else:
                    slider.set(max(current - 2, target))
        
        # If all reached their targets, pick new random targets
        if all_reached:
            for i in range(TOTAL_CHANNELS):
                targets[i] = random.randint(0, 50)
        
        # Schedule next step (50ms = slow smooth fade)
        random_fade_id = root.after(50, fade_step)
    
    fade_step()


def on_closing():
    """Clean shutdown"""
    print("Shutting down Art-Net...")
    set_all(0)
    artnet.stop()
    root.destroy()


# =============================================================================
# GUI SETUP
# =============================================================================

root = tk.Tk()
root.title("LED Panel Controller - Art-Net")
root.geometry("500x700")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Style
style = ttk.Style()
style.configure("TLabelframe", padding=10)
style.configure("TLabelframe.Label", font=("Helvetica", 11, "bold"))

# Header with connection info
header_frame = ttk.Frame(root, padding=10)
header_frame.pack(fill="x")

ttk.Label(
    header_frame, 
    text=f"Art-Net Target: {TARGET_IP}  |  Universe: {UNIVERSE}  |  Channels: {TOTAL_CHANNELS}",
    font=("Helvetica", 10)
).pack()

# Create sliders organized by unit
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill="both", expand=True)

for unit in range(NUM_UNITS):
    unit_frame = ttk.LabelFrame(main_frame, text=f"Unit {unit + 1}")
    unit_frame.pack(padx=5, pady=5, fill="x")
    
    for panel in range(PANELS_PER_UNIT):
        channel = unit * PANELS_PER_UNIT + panel
        
        panel_frame = ttk.Frame(unit_frame)
        panel_frame.pack(fill="x", pady=2, padx=5)
        
        # Channel number and label
        ttk.Label(
            panel_frame, 
            text=f"CH{channel + 1:02d} - Panel {panel + 1}:", 
            width=18
        ).pack(side="left")
        
        # Value display
        value_var = tk.StringVar(value="0")
        value_label = ttk.Label(panel_frame, textvariable=value_var, width=4)
        value_label.pack(side="right", padx=(5, 0))
        
        # Slider
        def make_callback(ch, var):
            def callback(val):
                var.set(str(int(float(val))))
                update_channel(ch, val)
            return callback
        
        slider = ttk.Scale(
            panel_frame,
            from_=0,
            to=MAX_DMX_VALUE,
            orient="horizontal",
            command=make_callback(channel, value_var),
            length=250
        )
        slider.pack(side="left", fill="x", expand=True, padx=(10, 5))
        sliders.append(slider)

# Master controls
control_frame = ttk.LabelFrame(root, text="Master Controls", padding=10)
control_frame.pack(fill="x", padx=10, pady=10)

button_frame = ttk.Frame(control_frame)
button_frame.pack(fill="x")

ttk.Button(button_frame, text="All ON (100%)", command=lambda: set_all(MAX_DMX_VALUE)).pack(side="left", padx=5)
ttk.Button(button_frame, text="All 50%", command=lambda: set_all(MAX_DMX_VALUE // 2)).pack(side="left", padx=5)
ttk.Button(button_frame, text="All OFF", command=lambda: set_all(0)).pack(side="left", padx=5)
ttk.Button(button_frame, text="Fade In", command=lambda: fade_all(MAX_DMX_VALUE)).pack(side="left", padx=5)
ttk.Button(button_frame, text="Fade Out", command=lambda: fade_all(0)).pack(side="left", padx=5)
ttk.Button(button_frame, text="Chase", command=chase_effect).pack(side="left", padx=5)
random_btn = ttk.Button(button_frame, text="Random Fade", command=random_fade_mode)
random_btn.pack(side="left", padx=5)

# Master slider
master_frame = ttk.Frame(control_frame)
master_frame.pack(fill="x", pady=(10, 0))

ttk.Label(master_frame, text="Master:").pack(side="left")
master_value = tk.StringVar(value="0")
ttk.Label(master_frame, textvariable=master_value, width=4).pack(side="right")

def master_callback(val):
    master_value.set(str(int(float(val))))
    set_all(int(float(val)))

master_slider = ttk.Scale(
    master_frame,
    from_=0,
    to=MAX_DMX_VALUE,
    orient="horizontal",
    command=master_callback,
    length=350
)
master_slider.pack(side="left", fill="x", expand=True, padx=10)

# Status bar
status_frame = ttk.Frame(root, padding=5)
status_frame.pack(fill="x", side="bottom")
ttk.Label(
    status_frame, 
    text="💡 Set your Art-Net/DMX decoder to IP: 10.42.0.200 (subnet 255.255.255.0)",
    font=("Helvetica", 9)
).pack()

print("LED Panel Controller started. Close window to exit.")
root.mainloop()