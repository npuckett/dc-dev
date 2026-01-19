#!/usr/bin/env python3
"""
Spring-Based LED Panel Controller via Art-Net
Controls 12 LED panels (4 units × 3 panels each) using a 2D spring physics simulation.

Layout:
- Front Grid (4×2): Panels 1 & 2 from each unit, coupled via springs
- Back Strip (4×1): Panel 3 from each unit, independent behavior

Spring System:
- External inputs apply forces to springs
- Springs propagate motion to neighbors
- Damping ensures smooth transitions
- Rest position influenced by time of day and passive activity

Usage:
    python springController.py [--simulate] [--debug]

Press keys during operation:
    1-4: Simulate active zone entry at column 1-4
    SPACE: Simulate active zone exit
    Q: Quit
"""

import time
import math
import random
import threading
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from stupidArtnet import StupidArtnet
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Art-Net settings
TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

# Panel layout
NUM_UNITS = 4
PANELS_PER_UNIT = 3
TOTAL_CHANNELS = NUM_UNITS * PANELS_PER_UNIT  # 12 channels

# DMX limits (from artnetTest.py)
MAX_DMX_VALUE = 212  # 10V output max
MIN_DMX_VALUE = 1    # Never fully off

# Brightness limits (mapped to DMX)
MIN_BRIGHTNESS = 1
MAX_BRIGHTNESS = 50  # As specified in behavior doc

# =============================================================================
# TIME OF DAY PARAMETERS
# =============================================================================

# Rush hours (24h format)
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 16  # 4pm
EVENING_RUSH_END = 19    # 7pm

# Evening mode (quieter, back panels more active)
EVENING_START = 18  # 6pm
NIGHT_START = 22    # 10pm

# =============================================================================
# SPRING PHYSICS PARAMETERS
# =============================================================================

@dataclass
class SpringConfig:
    """Tunable spring system parameters"""
    # Base spring properties
    stiffness: float = 0.15          # How quickly springs respond (0.05-0.3)
    damping: float = 0.85            # Energy loss per frame (0.7-0.95, higher = slower)
    coupling_strength: float = 0.3   # How much neighbors influence each other
    
    # Force magnitudes
    active_force_base: float = 15.0  # Force from active zone entry
    active_force_dwell_mult: float = 0.5  # Additional force per second of dwell
    passive_force_base: float = 2.0  # Force from passive zone activity
    
    # Decay rates
    engagement_decay_rate: float = 0.02  # How fast engagement fades (per second)
    acknowledgment_boost_max: float = 3.0  # Max multiplier for long absence
    acknowledgment_decay_time: float = 300.0  # Seconds to reach max boost (5 min)
    
    # Speed limits (max change per frame, ensures smooth transitions)
    max_velocity: float = 2.0  # Max brightness change per frame
    
    # Rest position
    base_brightness: float = 5.0     # Minimum ambient brightness
    passive_brightness_mult: float = 0.5  # How much passive activity raises rest position


# =============================================================================
# INPUT STATE (to be driven by camera later)
# =============================================================================

@dataclass
class InputState:
    """Current input from tracking system (simulated for now)"""
    # Active zones (trackzone 1-2)
    active_population: int = 0
    active_position_x: float = 0.5  # 0=left, 1=right
    active_dwell_time: float = 0.0  # Seconds since entry
    
    # Passive zones (trackzone 3-10)
    passive_population: int = 0
    passive_flow_speed: float = 0.0  # Average speed (m/s estimate)
    passive_flow_direction: float = 0.0  # -1=left, 0=none, 1=right
    
    # Separate tracking for bikes/vehicles
    cyclist_count: int = 0
    vehicle_count: int = 0
    
    # Derived timing
    time_since_active: float = 999.0  # Seconds since last active engagement


# =============================================================================
# NARRATIVE STATUS SYSTEM
# =============================================================================

@dataclass
class NarrativeStatus:
    """Human-readable description of what the installation is doing and why"""
    state: str = "Ambient"          # Current behavioral state
    action: str = "Breathing"       # What it's doing
    reason: str = "No visitors"     # Why it's doing that
    detail: str = ""                # Additional context
    mood: str = "Calm"              # Emotional quality
    
    def format_display(self) -> str:
        """Format for on-screen display"""
        lines = [
            f"┌{'─' * 58}┐",
            f"│ {'STATE:':<8} {self.state:<47} │",
            f"│ {'DOING:':<8} {self.action:<47} │",
            f"│ {'WHY:':<8} {self.reason:<47} │",
        ]
        if self.detail:
            # Word wrap detail if needed
            detail_text = self.detail[:47]
            lines.append(f"│ {'DETAIL:':<8} {detail_text:<47} │")
        lines.append(f"│ {'MOOD:':<8} {self.mood:<47} │")
        lines.append(f"└{'─' * 58}┘")
        return "\n".join(lines)


# =============================================================================
# SPRING NODE
# =============================================================================

@dataclass
class SpringNode:
    """A single spring node representing one panel in the front grid"""
    # Position in grid
    row: int  # 0=panel1, 1=panel2
    col: int  # 0-3 (units 4,3,2,1 from left to right)
    
    # Spring state
    position: float = 0.0      # Current "displacement" from rest
    velocity: float = 0.0      # Rate of change
    rest_position: float = 5.0  # Where spring settles
    
    # External force accumulator
    force: float = 0.0
    
    # Neighbors (set after construction)
    neighbors: List['SpringNode'] = field(default_factory=list)
    
    def apply_force(self, f: float):
        """Add force to this node"""
        self.force += f
    
    def update(self, config: SpringConfig, dt: float):
        """Update spring physics"""
        # Spring force toward rest position
        spring_force = -config.stiffness * (self.position - self.rest_position)
        
        # Coupling force from neighbors
        coupling_force = 0.0
        for neighbor in self.neighbors:
            coupling_force += config.coupling_strength * (neighbor.position - self.position)
        
        # Total acceleration
        acceleration = spring_force + coupling_force + self.force
        
        # Update velocity with damping
        self.velocity = (self.velocity + acceleration * dt) * config.damping
        
        # Clamp velocity for smooth motion
        self.velocity = max(-config.max_velocity, min(config.max_velocity, self.velocity))
        
        # Update position
        self.position += self.velocity * dt
        
        # Clamp to valid brightness range
        self.position = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, self.position))
        
        # Clear force for next frame
        self.force = 0.0
    
    @property
    def brightness(self) -> float:
        """Get current brightness value"""
        return self.position


# =============================================================================
# BACK PANEL (Independent behavior)
# =============================================================================

@dataclass
class BackPanel:
    """Panel 3 - independent ambient behavior, more active in evening"""
    col: int  # 0-3
    
    # State
    brightness: float = 3.0
    target: float = 3.0
    phase: float = 0.0  # For sine wave animation
    
    # Animation speed
    speed: float = 0.3
    
    def update(self, dt: float, time_factor: float, passive_energy: float):
        """
        Update back panel with ambient behavior.
        time_factor: 0-1, higher in evening
        passive_energy: influence from passive zones
        """
        # Update phase
        self.phase += self.speed * dt * (1 + time_factor)
        
        # Base sine wave with per-panel offset
        wave = math.sin(self.phase + self.col * 0.7) * 0.5 + 0.5
        
        # Evening makes panels more dynamic
        amplitude = 3 + time_factor * 8  # 3-11 range
        base = 2 + time_factor * 5  # 2-7 base
        
        # Add passive energy influence
        passive_boost = passive_energy * 0.1
        
        self.target = base + wave * amplitude + passive_boost
        self.target = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, self.target))
        
        # Smooth toward target
        diff = self.target - self.brightness
        self.brightness += diff * 0.05  # Slow smoothing
        self.brightness = max(MIN_BRIGHTNESS, min(MAX_BRIGHTNESS, self.brightness))


# =============================================================================
# SPRING CONTROLLER
# =============================================================================

class SpringController:
    """Main controller managing spring grid and back panels"""
    
    def __init__(self, config: Optional[SpringConfig] = None, debug: bool = False):
        self.config = config or SpringConfig()
        self.debug = debug
        
        # Create front grid (4 cols × 2 rows)
        self.grid: List[List[SpringNode]] = []
        for row in range(2):
            row_nodes = []
            for col in range(4):
                node = SpringNode(row=row, col=col, position=self.config.base_brightness)
                node.rest_position = self.config.base_brightness
                row_nodes.append(node)
            self.grid.append(row_nodes)
        
        # Set up neighbor connections
        self._setup_neighbors()
        
        # Create back panels
        self.back_panels = [BackPanel(col=i, phase=random.random() * math.pi * 2) for i in range(4)]
        
        # Input state
        self.input = InputState()
        
        # Engagement tracking
        self.last_engagement_time = time.time() - 300  # Start as if no recent engagement
        self.engagement_level = 0.0
        
        # Narrative status
        self.status = NarrativeStatus()
        self.show_status = True  # Toggle with 'S' key
        
        # Art-Net
        self.artnet = None
        self.channel_values = [MIN_DMX_VALUE] * TOTAL_CHANNELS
        
        # Timing
        self.last_update = time.time()
        self.frame_count = 0
    
    def _setup_neighbors(self):
        """Connect grid nodes to their neighbors"""
        for row in range(2):
            for col in range(4):
                node = self.grid[row][col]
                neighbors = []
                
                # Left neighbor
                if col > 0:
                    neighbors.append(self.grid[row][col - 1])
                # Right neighbor
                if col < 3:
                    neighbors.append(self.grid[row][col + 1])
                # Up neighbor
                if row > 0:
                    neighbors.append(self.grid[row - 1][col])
                # Down neighbor
                if row < 1:
                    neighbors.append(self.grid[row + 1][col])
                
                node.neighbors = neighbors
    
    def init_artnet(self) -> bool:
        """Initialize Art-Net connection"""
        try:
            self.artnet = StupidArtnet(TARGET_IP, UNIVERSE, TOTAL_CHANNELS, FPS)
            self.artnet.start()
            print(f"✓ Art-Net initialized: {TARGET_IP} Universe {UNIVERSE}")
            return True
        except Exception as e:
            print(f"✗ Art-Net initialization failed: {e}")
            return False
    
    def get_time_factors(self) -> dict:
        """Calculate time-of-day influence factors"""
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        
        # Rush hour factor (0-1)
        rush_factor = 0.0
        if MORNING_RUSH_START <= hour <= MORNING_RUSH_END:
            # Peak at middle of rush
            mid = (MORNING_RUSH_START + MORNING_RUSH_END) / 2
            rush_factor = 1.0 - abs(hour - mid) / ((MORNING_RUSH_END - MORNING_RUSH_START) / 2)
        elif EVENING_RUSH_START <= hour <= EVENING_RUSH_END:
            mid = (EVENING_RUSH_START + EVENING_RUSH_END) / 2
            rush_factor = 1.0 - abs(hour - mid) / ((EVENING_RUSH_END - EVENING_RUSH_START) / 2)
        
        # Evening factor (0-1, peaks at night)
        evening_factor = 0.0
        if hour >= EVENING_START:
            if hour >= NIGHT_START:
                evening_factor = 1.0
            else:
                evening_factor = (hour - EVENING_START) / (NIGHT_START - EVENING_START)
        
        # Daytime base brightness (lower at night)
        if hour < 6:
            daytime_brightness = 0.3
        elif hour < 8:
            daytime_brightness = 0.3 + (hour - 6) * 0.35  # Ramp up
        elif hour < 18:
            daytime_brightness = 1.0  # Full day
        elif hour < 22:
            daytime_brightness = 1.0 - (hour - 18) * 0.15  # Ramp down
        else:
            daytime_brightness = 0.4  # Night
        
        return {
            'rush_factor': rush_factor,
            'evening_factor': evening_factor,
            'daytime_brightness': daytime_brightness,
            'hour': hour
        }
    
    def calculate_acknowledgment_boost(self) -> float:
        """
        Calculate boost multiplier for acknowledging new engagement.
        Higher boost if it's been a long time since last engagement,
        especially outside rush hours.
        """
        time_since = time.time() - self.last_engagement_time
        time_factors = self.get_time_factors()
        
        # Base boost from time since engagement
        time_boost = min(time_since / self.config.acknowledgment_decay_time, 1.0)
        time_boost *= self.config.acknowledgment_boost_max
        
        # Extra boost outside rush hours (installation is "lonely")
        if time_factors['rush_factor'] < 0.3:
            time_boost *= 1.5
        
        # Extra boost in evening (special attention to visitors)
        if time_factors['evening_factor'] > 0.5:
            time_boost *= 1.3
        
        return 1.0 + time_boost
    
    def apply_active_input(self):
        """Apply forces from active zone engagement"""
        if self.input.active_population <= 0:
            return
        
        # Calculate acknowledgment boost
        boost = self.calculate_acknowledgment_boost()
        
        # Update engagement tracking
        self.last_engagement_time = time.time()
        
        # Calculate force magnitude
        base_force = self.config.active_force_base * self.input.active_population
        dwell_force = self.config.active_force_dwell_mult * self.input.active_dwell_time
        total_force = (base_force + dwell_force) * boost
        
        # Determine which columns to apply force to (based on position)
        # position_x: 0=left (col 0), 1=right (col 3)
        center_col = self.input.active_position_x * 3  # 0-3
        
        # Apply force with distance falloff
        for col in range(4):
            distance = abs(col - center_col)
            falloff = max(0, 1.0 - distance * 0.4)  # Gradual falloff
            col_force = total_force * falloff
            
            # Apply to both rows in this column
            for row in range(2):
                self.grid[row][col].apply_force(col_force)
        
        if self.debug and self.frame_count % 30 == 0:
            print(f"Active: pop={self.input.active_population}, dwell={self.input.active_dwell_time:.1f}s, "
                  f"boost={boost:.2f}, force={total_force:.1f}")
    
    def apply_passive_input(self):
        """Apply ambient influence from passive zones"""
        time_factors = self.get_time_factors()
        
        # Calculate passive energy
        passive_energy = (
            self.input.passive_population * 0.5 +
            self.input.cyclist_count * 0.3 +
            self.input.vehicle_count * 0.2
        )
        
        # Flow speed adds to energy
        passive_energy *= (1.0 + self.input.passive_flow_speed * 0.2)
        
        # During rush hour, dampen passive influence (it's expected)
        passive_energy *= (1.0 - time_factors['rush_factor'] * 0.5)
        
        # Apply gentle force to all nodes
        force = self.config.passive_force_base * passive_energy * 0.1
        
        # Add directional bias
        direction = self.input.passive_flow_direction
        
        for row in range(2):
            for col in range(4):
                # Slight directional variation
                col_bias = (col - 1.5) / 1.5 * direction * 0.3
                self.grid[row][col].apply_force(force * (1 + col_bias))
        
        # Update rest positions based on passive activity and time
        base = self.config.base_brightness * time_factors['daytime_brightness']
        passive_offset = passive_energy * self.config.passive_brightness_mult
        
        for row in range(2):
            for col in range(4):
                self.grid[row][col].rest_position = base + passive_offset
    
    def update_engagement_level(self, dt: float):
        """Update the overall engagement level (0-1)"""
        if self.input.active_population > 0:
            # Increase engagement
            target = min(1.0, (
                self.input.active_population * 0.4 +
                min(self.input.active_dwell_time / 10.0, 1.0) * 0.4 +
                0.2
            ))
            self.engagement_level += (target - self.engagement_level) * 0.1
        else:
            # Decay engagement
            decay = self.config.engagement_decay_rate * dt
            self.engagement_level = max(0, self.engagement_level - decay)
        
        # Engagement affects spring coupling (more engaged = more connected)
        self.config.coupling_strength = 0.2 + self.engagement_level * 0.3
    
    def update(self):
        """Main update loop"""
        now = time.time()
        dt = min(now - self.last_update, 0.1)  # Cap dt to prevent jumps
        self.last_update = now
        self.frame_count += 1
        
        time_factors = self.get_time_factors()
        
        # Update time since active
        if self.input.active_population > 0:
            self.input.time_since_active = 0
        else:
            self.input.time_since_active += dt
        
        # Apply inputs
        self.apply_active_input()
        self.apply_passive_input()
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
        
        # Update narrative status
        self.update_narrative_status(time_factors)
        
        # Print status (throttled to reduce flicker)
        if self.frame_count % 10 == 0:  # ~3 times per second
            self.print_status()
        
        # Debug output
        if self.debug and self.frame_count % 60 == 0:
            self._print_debug(time_factors)
    
    def _send_artnet(self):
        """Convert brightness to DMX and send via Art-Net"""
        if not self.artnet:
            return
        
        # Map brightness (1-50) to DMX (1-212)
        def brightness_to_dmx(b: float) -> int:
            # Linear mapping from brightness range to DMX range
            normalized = (b - MIN_BRIGHTNESS) / (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            dmx = int(MIN_DMX_VALUE + normalized * (MAX_DMX_VALUE - MIN_DMX_VALUE))
            return max(MIN_DMX_VALUE, min(MAX_DMX_VALUE, dmx))
        
        # Channel mapping:
        # Unit 1: CH1=Panel1, CH2=Panel2, CH3=Panel3
        # Unit 2: CH4=Panel1, CH5=Panel2, CH6=Panel3
        # etc.
        
        # Grid is arranged: col 0=Unit4, col 1=Unit3, col 2=Unit2, col 3=Unit1
        # So we need to reverse the column order for channel mapping
        
        for unit in range(4):
            col = 3 - unit  # Reverse: unit 0 (CH1-3) = col 3, unit 3 (CH10-12) = col 0
            
            # Panel 1 (row 0)
            ch_panel1 = unit * 3 + 0
            self.channel_values[ch_panel1] = brightness_to_dmx(self.grid[0][col].brightness)
            
            # Panel 2 (row 1)
            ch_panel2 = unit * 3 + 1
            self.channel_values[ch_panel2] = brightness_to_dmx(self.grid[1][col].brightness)
            
            # Panel 3 (back)
            ch_panel3 = unit * 3 + 2
            self.channel_values[ch_panel3] = brightness_to_dmx(self.back_panels[col].brightness)
        
        self.artnet.set(self.channel_values)
    
    def _print_debug(self, time_factors: dict):
        """Print debug information"""
        print(f"\n--- Frame {self.frame_count} ---")
        print(f"Time: {time_factors['hour']:.1f}h | Rush: {time_factors['rush_factor']:.2f} | "
              f"Evening: {time_factors['evening_factor']:.2f}")
        print(f"Engagement: {self.engagement_level:.2f} | Since active: {self.input.time_since_active:.1f}s")
        
        print("Front Grid (Panel 1 | Panel 2):")
        for col in range(4):
            p1 = self.grid[0][col].brightness
            p2 = self.grid[1][col].brightness
            print(f"  Col {col}: {p1:5.1f} | {p2:5.1f}")
        
        print("Back Panels:")
        for col in range(4):
            print(f"  Col {col}: {self.back_panels[col].brightness:5.1f}")
    
    def update_narrative_status(self, time_factors: dict):
        """Generate human-readable status explaining current behavior"""
        s = self.status
        boost = self.calculate_acknowledgment_boost()
        
        # Determine state based on engagement level
        if self.engagement_level >= 0.8:
            s.state = "Peak"
        elif self.engagement_level >= 0.5:
            s.state = "Engaged"
        elif self.engagement_level >= 0.2:
            s.state = "Acknowledging"
        else:
            s.state = "Ambient"
        
        # Determine what it's doing and why
        if self.input.active_population > 0:
            # Active engagement
            people = self.input.active_population
            dwell = self.input.active_dwell_time
            
            if dwell < 2:
                s.action = f"Noticing {'someone' if people == 1 else f'{people} people'} approaching"
                if boost > 2.0:
                    s.reason = "It's been quiet - excited to see a visitor!"
                    s.mood = "Excited"
                elif boost > 1.5:
                    s.reason = "Haven't seen anyone in a while"
                    s.mood = "Curious"
                else:
                    s.reason = "Someone stepped into the active zone"
                    s.mood = "Attentive"
            elif dwell < 5:
                s.action = f"Responding to {'visitor' if people == 1 else 'visitors'}"
                s.reason = f"They've been here {dwell:.0f} seconds"
                s.mood = "Engaged"
            elif dwell < 10:
                s.action = f"Building connection with {'visitor' if people == 1 else 'visitors'}"
                s.reason = f"Sustained attention for {dwell:.0f} seconds"
                s.mood = "Connected"
            else:
                s.action = f"Deep engagement with {'visitor' if people == 1 else 'visitors'}"
                s.reason = f"Extended interaction ({dwell:.0f}s) - showing appreciation"
                s.mood = "Joyful"
            
            # Position detail
            pos = self.input.active_position_x
            if pos < 0.25:
                s.detail = "Focus on left side"
            elif pos > 0.75:
                s.detail = "Focus on right side"
            else:
                s.detail = "Focus on center"
                
        elif self.input.time_since_active < 5:
            # Just left
            s.action = "Settling after visitor departed"
            s.reason = "Visitor just left the zone"
            s.mood = "Contemplative"
            s.detail = "Springs returning to rest"
            
        elif self.input.time_since_active < 30:
            # Recently active
            s.action = "Fading back to ambient"
            s.reason = f"No one in active zone for {self.input.time_since_active:.0f}s"
            s.mood = "Relaxing"
            s.detail = ""
            
        else:
            # Ambient mode
            passive_total = self.input.passive_population + self.input.cyclist_count
            
            if time_factors['evening_factor'] > 0.7:
                s.action = "Evening contemplation"
                s.reason = "Night time - quieter, more introspective"
                s.mood = "Serene"
                s.detail = "Back panels more active"
            elif time_factors['rush_factor'] > 0.5:
                s.action = "Observing rush hour flow"
                s.reason = f"Peak traffic time with {passive_total} people passing"
                s.mood = "Watchful"
                s.detail = "Dampened response to expected activity"
            elif passive_total > 10:
                s.action = "Sensing busy sidewalk"
                s.reason = f"{passive_total} people and cyclists passing by"
                s.mood = "Alert"
                s.detail = ""
            elif passive_total > 3:
                s.action = "Breathing with street rhythm"
                s.reason = f"Light foot traffic ({passive_total} passing)"
                s.mood = "Calm"
                s.detail = ""
            else:
                s.action = "Quiet breathing"
                s.reason = "Little activity outside"
                s.mood = "Peaceful"
                if self.input.time_since_active > 120:
                    s.detail = f"Waiting... ({self.input.time_since_active/60:.0f}m since last visitor)"
                else:
                    s.detail = ""
        
        # Add time context
        hour = time_factors['hour']
        if hour < 6:
            time_note = "Late night"
        elif hour < 9:
            time_note = "Morning"
        elif hour < 12:
            time_note = "Late morning"
        elif hour < 14:
            time_note = "Midday"
        elif hour < 17:
            time_note = "Afternoon"
        elif hour < 20:
            time_note = "Evening"
        else:
            time_note = "Night"
        
        if s.detail:
            s.detail = f"{time_note} · {s.detail}"
        else:
            s.detail = time_note
    
    def print_status(self):
        """Print narrative status to terminal"""
        if not self.show_status:
            return
        # Move cursor up and overwrite (creates updating display)
        print("\033[8A", end="")  # Move up 8 lines
        print(self.status.format_display())
        print()  # Extra line for spacing)
    
    def simulate_active_entry(self, position_x: float = 0.5, population: int = 1):
        """Simulate someone entering the active zone"""
        self.input.active_population = population
        self.input.active_position_x = position_x
        self.input.active_dwell_time = 0.0
        print(f"→ Simulated active entry at x={position_x:.2f}")
    
    def simulate_active_exit(self):
        """Simulate everyone leaving the active zone"""
        self.input.active_population = 0
        self.input.active_dwell_time = 0.0
        print("← Simulated active exit")
    
    def simulate_passive_activity(self, population: int = 5, cyclists: int = 1, vehicles: int = 2):
        """Set simulated passive zone activity"""
        self.input.passive_population = population
        self.input.cyclist_count = cyclists
        self.input.vehicle_count = vehicles
        self.input.passive_flow_speed = 1.0 + random.random()
        self.input.passive_flow_direction = random.choice([-1, 0, 1])
    
    def run_simulation(self):
        """Run with simulated random inputs"""
        print("Starting simulation mode...")
        print("Press Ctrl+C to stop")
        print("\n" * 8)  # Space for status display
        
        # Set initial passive activity
        self.simulate_passive_activity()
        
        next_passive_update = time.time() + 10
        next_active_event = time.time() + random.uniform(5, 15)
        active_exit_time = None
        
        try:
            while True:
                now = time.time()
                
                # Periodically update passive activity
                if now >= next_passive_update:
                    self.simulate_passive_activity(
                        population=random.randint(2, 15),
                        cyclists=random.randint(0, 5),
                        vehicles=random.randint(0, 8)
                    )
                    next_passive_update = now + random.uniform(8, 20)
                
                # Random active zone events
                if now >= next_active_event:
                    if self.input.active_population == 0:
                        # Someone enters
                        self.simulate_active_entry(
                            position_x=random.random(),
                            population=random.randint(1, 3)
                        )
                        # Schedule exit
                        active_exit_time = now + random.uniform(3, 15)
                        next_active_event = now + 0.5  # Check for exit
                    else:
                        next_active_event = now + random.uniform(10, 60)
                
                # Check for active exit
                if active_exit_time and now >= active_exit_time:
                    self.simulate_active_exit()
                    active_exit_time = None
                    next_active_event = now + random.uniform(10, 60)
                
                # Increment dwell time if active
                if self.input.active_population > 0:
                    self.input.active_dwell_time += 1.0 / FPS
                
                # Update controller
                self.update()
                
                # Sleep to maintain frame rate
                time.sleep(1.0 / FPS)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped.")
    
    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down...")
        if self.artnet:
            # Fade to minimum
            for i in range(len(self.channel_values)):
                self.channel_values[i] = MIN_DMX_VALUE
            self.artnet.set(self.channel_values)
            time.sleep(0.1)
            self.artnet.stop()


# =============================================================================
# INTERACTIVE MODE WITH KEYBOARD INPUT
# =============================================================================

def run_interactive(controller: SpringController):
    """Run with keyboard controls for testing"""
    import sys
    import select
    import termios
    import tty
    
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Controls:")
    print("  1-4: Trigger active entry at column 1-4 (left to right)")
    print("  SPACE: Exit active zone")
    print("  P: Toggle passive activity simulation")
    print("  S: Toggle status display")
    print("  D: Toggle debug output")
    print("  Q: Quit")
    print("="*50)
    
    # Print initial empty lines for status display area
    print("\n" * 8)
    
    # Set up non-blocking keyboard input
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    passive_sim = True
    controller.simulate_passive_activity()
    
    try:
        while True:
            # Check for keyboard input
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == 'q':
                    break
                elif key in '1234':
                    col = int(key) - 1
                    position_x = col / 3.0  # 0, 0.33, 0.67, 1.0
                    controller.simulate_active_entry(position_x=position_x)
                elif key == ' ':
                    controller.simulate_active_exit()
                elif key == 'p':
                    passive_sim = not passive_sim
                    if passive_sim:
                        controller.simulate_passive_activity()
                        print("\033[10B")  # Move down to avoid overwriting status
                        print("Passive simulation ON")
                        print("\033[10A")  # Move back up
                    else:
                        controller.input.passive_population = 0
                        controller.input.cyclist_count = 0
                        controller.input.vehicle_count = 0
                        print("\033[10B")
                        print("Passive simulation OFF")
                        print("\033[10A")
                elif key == 's':
                    controller.show_status = not controller.show_status
                    if not controller.show_status:
                        print("\033[10B")
                        print("Status display OFF")
                        print("\033[10A")
                elif key == 'd':
                    controller.debug = not controller.debug
                    print("\033[10B")
                    print(f"Debug: {'ON' if controller.debug else 'OFF'}")
                    print("\033[10A")
            
            # Increment dwell time if active
            if controller.input.active_population > 0:
                controller.input.active_dwell_time += 1.0 / FPS
            
            # Occasionally vary passive input
            if passive_sim and random.random() < 0.01:
                controller.simulate_passive_activity(
                    population=random.randint(2, 15),
                    cyclists=random.randint(0, 5),
                    vehicles=random.randint(0, 8)
                )
            
            controller.update()
            time.sleep(1.0 / FPS)
            
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Spring-based LED Panel Controller')
    parser.add_argument('--simulate', action='store_true', help='Run autonomous simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--no-artnet', action='store_true', help='Run without Art-Net (for testing)')
    args = parser.parse_args()
    
    print("="*50)
    print("SPRING LED PANEL CONTROLLER")
    print("="*50)
    
    config = SpringConfig()
    controller = SpringController(config=config, debug=args.debug)
    
    if not args.no_artnet:
        if not controller.init_artnet():
            print("Failed to initialize Art-Net. Use --no-artnet to run without hardware.")
            sys.exit(1)
    else:
        print("Running without Art-Net (test mode)")
    
    try:
        if args.simulate:
            controller.run_simulation()
        else:
            run_interactive(controller)
    finally:
        controller.shutdown()
    
    print("Done.")


if __name__ == "__main__":
    main()
