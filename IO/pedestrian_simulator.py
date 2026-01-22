#!/usr/bin/env python3
"""
Pedestrian Simulator for Light Installation Testing (Headless)

Simulates pedestrian traffic on a busy Toronto sidewalk.
Runs in terminal without a window - view results in lightController.

Pedestrian Types:
1. PASSIVE - Walk straight through the passive zone (sidewalk traffic)
2. ACTIVE - Spawn in active zone, wander, then leave
3. CURIOUS - Start in passive zone, notice installation, enter active zone,
              explore for a while, then exit back through passive zone

Sends OSC messages in the same format as the camera tracker:
  /tracker/person/<id> <x> <z>  - Position of tracked person (cm)
  /tracker/count <n>            - Number of people currently tracked

Controls (keyboard in terminal):
  +/-   : Adjust passive pedestrian spawn rate
  a     : Spawn a person in active zone
  c     : Spawn a curious person (passive -> active -> exit)
  p     : Pause/resume simulation
  r     : Reset simulation
  q     : Quit

All units in centimeters.
"""

import sys
import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto

# OSC client
from pythonosc import udp_client

# =============================================================================
# CONFIGURATION (all units in centimeters)
# =============================================================================

# OSC settings - send to lightController
OSC_TARGET_IP = "127.0.0.1"
OSC_TARGET_PORT = 7000

# Simulation settings
FPS = 30

# Zone definitions (matching lightController)
# Active zone - between columns, where engaged people are
ACTIVE_ZONE = {
    'width': 475,
    'depth': 205,
    'offset_z': 78,
    'center_x': 120,
}

# Passive zone - sidewalk traffic passing by
PASSIVE_ZONE = {
    'width': 650,
    'depth': 330,
    'offset_z': 78 + 205,  # Starts at back of active zone (283cm)
    'center_x': 120,
}

# Pedestrian settings
PEDESTRIAN_SPEED_MIN = 80   # cm/s (slow walker)
PEDESTRIAN_SPEED_MAX = 150  # cm/s (fast walker)
ACTIVE_SPEED_MIN = 30       # cm/s (wandering slowly)
ACTIVE_SPEED_MAX = 60       # cm/s

# Spawn rates (people per minute)
PASSIVE_SPAWN_RATE = 30     # Busy sidewalk
ACTIVE_SPAWN_RATE = 0.5     # Rare - someone actually enters
CURIOUS_SPAWN_RATE = 2.0    # Occasionally someone gets curious

# Walking directions
DIRECTION_LEFT_TO_RIGHT = 1
DIRECTION_RIGHT_TO_LEFT = -1


# =============================================================================
# SIMULATION MODES
# =============================================================================

class SimulationMode(Enum):
    """Different simulation modes"""
    NORMAL = auto()      # Original mode - constant traffic
    LONG_RUN = auto()    # Hours-long realistic simulation with gaps


# Long run traffic patterns (time in simulated hours since start)
# Each entry: (hour_start, hour_end, passive_rate, curious_rate, active_rate, description)
LONG_RUN_PATTERNS = [
    # Early morning - very quiet
    (0, 2, 5, 0.2, 0.0, "quiet night"),
    # Pre-dawn lull
    (2, 5, 2, 0.1, 0.0, "dead of night"),
    # Early commuters
    (5, 7, 15, 0.5, 0.1, "early morning"),
    # Morning rush
    (7, 9, 45, 1.5, 0.3, "morning rush"),
    # Late morning
    (9, 11, 25, 2.0, 0.5, "mid-morning"),
    # Lunch time surge
    (11, 13, 40, 3.0, 0.8, "lunch rush"),
    # Afternoon
    (13, 16, 30, 2.5, 0.6, "afternoon"),
    # Evening rush
    (16, 19, 50, 2.0, 0.4, "evening rush"),
    # Evening casual
    (19, 21, 35, 3.5, 1.0, "evening leisure"),
    # Late night
    (21, 24, 15, 1.0, 0.3, "late night"),
]

# Gap patterns for long run (makes traffic more realistic)
# Probability and duration of random quiet periods
GAP_SETTINGS = {
    'gap_probability': 0.02,     # Chance per second of starting a gap
    'min_gap_duration': 5,       # Minimum gap in seconds
    'max_gap_duration': 45,      # Maximum gap in seconds
    'lull_probability': 0.005,   # Chance per second of extended lull
    'min_lull_duration': 60,     # 1 minute minimum lull
    'max_lull_duration': 180,    # 3 minute maximum lull
}


# =============================================================================
# PEDESTRIAN STATES
# =============================================================================

class PedestrianState(Enum):
    """State machine for pedestrian behavior"""
    PASSIVE_WALKING = auto()      # Walking through passive zone
    ENTERING_ACTIVE = auto()      # Noticed installation, moving toward active zone
    ACTIVE_WANDERING = auto()     # Exploring in active zone
    EXITING_ACTIVE = auto()       # Leaving active zone
    EXITING_PASSIVE = auto()      # Walking out through passive zone
    DONE = auto()                 # Ready to be removed


# =============================================================================
# SIMULATED PEDESTRIAN
# =============================================================================

@dataclass
class SimulatedPerson:
    """A simulated pedestrian with state machine behavior"""
    id: int
    x: float
    z: float
    speed: float
    direction: int  # 1 = left-to-right, -1 = right-to-left
    state: PedestrianState = PedestrianState.PASSIVE_WALKING
    
    # Target for wandering/moving
    target_x: Optional[float] = None
    target_z: Optional[float] = None
    
    # Timing
    dwell_time: float = 0.0
    max_dwell: float = 0.0
    state_time: float = 0.0       # Time in current state
    wander_targets_hit: int = 0   # Number of wander targets reached
    max_wander_targets: int = 3   # How many spots to visit before leaving
    
    def update(self, dt: float) -> bool:
        """Update position based on state. Returns False if person should be removed."""
        self.state_time += dt
        
        if self.state == PedestrianState.PASSIVE_WALKING:
            return self._update_passive_walking(dt)
        elif self.state == PedestrianState.ENTERING_ACTIVE:
            return self._update_entering_active(dt)
        elif self.state == PedestrianState.ACTIVE_WANDERING:
            return self._update_active_wandering(dt)
        elif self.state == PedestrianState.EXITING_ACTIVE:
            return self._update_exiting_active(dt)
        elif self.state == PedestrianState.EXITING_PASSIVE:
            return self._update_exiting_passive(dt)
        elif self.state == PedestrianState.DONE:
            return False
        
        return True
    
    def _update_passive_walking(self, dt: float) -> bool:
        """Walk straight through passive zone"""
        self.x += self.direction * self.speed * dt
        
        # Add slight z wandering
        self.z += random.uniform(-5, 5) * dt
        
        # Clamp z to passive zone
        pz = PASSIVE_ZONE
        min_z = pz['offset_z']
        max_z = pz['offset_z'] + pz['depth']
        self.z = max(min_z + 20, min(max_z - 20, self.z))
        
        # Check if exited
        pz_min_x = pz['center_x'] - pz['width'] / 2 - 50
        pz_max_x = pz['center_x'] + pz['width'] / 2 + 50
        
        if self.x < pz_min_x or self.x > pz_max_x:
            return False  # Remove
        
        return True
    
    def _update_entering_active(self, dt: float) -> bool:
        """Moving from passive zone into active zone"""
        if self.target_x is None:
            # Pick entry point in active zone
            az = ACTIVE_ZONE
            self.target_x = random.uniform(
                az['center_x'] - az['width']/3,
                az['center_x'] + az['width']/3
            )
            self.target_z = az['offset_z'] + az['depth'] - 30  # Near back of active zone
        
        # Move toward target
        if self._move_toward_target(dt, speed_mult=0.7):
            # Reached active zone, start wandering
            self.state = PedestrianState.ACTIVE_WANDERING
            self.state_time = 0
            self.target_x = None
            self.target_z = None
            self.max_wander_targets = random.randint(2, 5)
            self.wander_targets_hit = 0
        
        return True
    
    def _update_active_wandering(self, dt: float) -> bool:
        """Wander around in active zone"""
        if self.target_x is None:
            self._pick_active_target()
        
        # Move toward target
        if self._move_toward_target(dt, speed_mult=0.5):
            # Reached target, dwell for a bit
            self.dwell_time += dt
            if self.dwell_time > self.max_dwell:
                self.wander_targets_hit += 1
                self.dwell_time = 0
                
                # Check if done exploring
                if self.wander_targets_hit >= self.max_wander_targets:
                    self.state = PedestrianState.EXITING_ACTIVE
                    self.state_time = 0
                    self.target_x = None
                    self.target_z = None
                else:
                    self._pick_active_target()
        
        return True
    
    def _update_exiting_active(self, dt: float) -> bool:
        """Leaving active zone, heading back to passive"""
        if self.target_x is None:
            # Pick exit point at edge of active/passive boundary
            az = ACTIVE_ZONE
            self.target_x = self.x + self.direction * 50  # Move in original direction
            self.target_z = az['offset_z'] + az['depth'] + 30  # Just into passive zone
        
        # Move toward exit
        if self._move_toward_target(dt, speed_mult=0.8):
            self.state = PedestrianState.EXITING_PASSIVE
            self.state_time = 0
            self.target_x = None
            self.target_z = None
            # Speed up to normal walking speed
            self.speed = random.uniform(PEDESTRIAN_SPEED_MIN, PEDESTRIAN_SPEED_MAX)
        
        return True
    
    def _update_exiting_passive(self, dt: float) -> bool:
        """Walking out through passive zone"""
        self.x += self.direction * self.speed * dt
        
        # Check if exited
        pz = PASSIVE_ZONE
        pz_min_x = pz['center_x'] - pz['width'] / 2 - 50
        pz_max_x = pz['center_x'] + pz['width'] / 2 + 50
        
        if self.x < pz_min_x or self.x > pz_max_x:
            return False  # Remove
        
        return True
    
    def _move_toward_target(self, dt: float, speed_mult: float = 1.0) -> bool:
        """Move toward target. Returns True if reached."""
        if self.target_x is None:
            return True
        
        dx = self.target_x - self.x
        dz = self.target_z - self.z
        dist = math.sqrt(dx*dx + dz*dz)
        
        if dist < 15:
            return True  # Reached
        
        # Move toward target
        move_dist = self.speed * speed_mult * dt
        self.x += (dx / dist) * move_dist
        self.z += (dz / dist) * move_dist
        
        return False
    
    def _pick_active_target(self):
        """Pick a new wander target in active zone"""
        az = ACTIVE_ZONE
        self.target_x = random.uniform(
            az['center_x'] - az['width']/2 + 50,
            az['center_x'] + az['width']/2 - 50
        )
        self.target_z = random.uniform(
            az['offset_z'] + 30,
            az['offset_z'] + az['depth'] - 30
        )
        self.max_dwell = random.uniform(1, 5)


# =============================================================================
# SIMULATOR
# =============================================================================

class PedestrianSimulator:
    """Manages simulated pedestrians"""
    
    def __init__(self, mode: SimulationMode = SimulationMode.NORMAL):
        self.people: List[SimulatedPerson] = []
        self.next_id = 1
        self.mode = mode
        
        # Spawn timing
        self.passive_spawn_timer = 0.0
        self.active_spawn_timer = 0.0
        self.curious_spawn_timer = 0.0
        
        self.passive_spawn_rate = PASSIVE_SPAWN_RATE
        self.active_spawn_rate = ACTIVE_SPAWN_RATE
        self.curious_spawn_rate = CURIOUS_SPAWN_RATE
        
        self.paused = False
        
        # Long run mode state
        self.simulation_start_time = time.time()
        self.simulated_hours = 0.0  # For time acceleration
        self.time_scale = 1.0       # 1.0 = real-time, higher = faster
        self.current_pattern_desc = "starting"
        
        # Gap/lull tracking
        self.in_gap = False
        self.gap_end_time = 0.0
        self.in_lull = False
        self.lull_end_time = 0.0
        
        # Statistics
        self.total_spawned = 0
        self.total_curious = 0
        self.total_active = 0
    
    def spawn_passive_person(self):
        """Spawn a person walking through passive zone"""
        pz = PASSIVE_ZONE
        
        # Random direction
        direction = random.choice([DIRECTION_LEFT_TO_RIGHT, DIRECTION_RIGHT_TO_LEFT])
        
        # Start position
        if direction == DIRECTION_LEFT_TO_RIGHT:
            x = pz['center_x'] - pz['width']/2 - 30
        else:
            x = pz['center_x'] + pz['width']/2 + 30
        
        # Random z within passive zone
        z = random.uniform(pz['offset_z'] + 30, pz['offset_z'] + pz['depth'] - 30)
        
        # Random speed
        speed = random.uniform(PEDESTRIAN_SPEED_MIN, PEDESTRIAN_SPEED_MAX)
        
        person = SimulatedPerson(
            id=self.next_id,
            x=x,
            z=z,
            speed=speed,
            direction=direction,
            state=PedestrianState.PASSIVE_WALKING
        )
        self.next_id += 1
        self.people.append(person)
        self.total_spawned += 1
    
    def spawn_active_person(self):
        """Spawn a person directly in active zone"""
        az = ACTIVE_ZONE
        
        x = random.uniform(az['center_x'] - az['width']/3, az['center_x'] + az['width']/3)
        z = random.uniform(az['offset_z'] + 50, az['offset_z'] + az['depth'] - 50)
        
        person = SimulatedPerson(
            id=self.next_id,
            x=x,
            z=z,
            speed=random.uniform(ACTIVE_SPEED_MIN, ACTIVE_SPEED_MAX),
            direction=random.choice([1, -1]),
            state=PedestrianState.ACTIVE_WANDERING,
            max_wander_targets=random.randint(2, 5)
        )
        self.next_id += 1
        self.people.append(person)
        self.total_spawned += 1
        self.total_active += 1
        return person.id
    
    def spawn_curious_person(self):
        """Spawn a person who starts in passive zone, enters active, explores, then leaves"""
        pz = PASSIVE_ZONE
        az = ACTIVE_ZONE
        
        # Random direction
        direction = random.choice([DIRECTION_LEFT_TO_RIGHT, DIRECTION_RIGHT_TO_LEFT])
        
        # Start at edge of passive zone
        if direction == DIRECTION_LEFT_TO_RIGHT:
            x = pz['center_x'] - pz['width']/2 - 30
        else:
            x = pz['center_x'] + pz['width']/2 + 30
        
        # Start in passive zone but close to active
        z = random.uniform(pz['offset_z'] + 20, pz['offset_z'] + 80)
        
        person = SimulatedPerson(
            id=self.next_id,
            x=x,
            z=z,
            speed=random.uniform(PEDESTRIAN_SPEED_MIN, PEDESTRIAN_SPEED_MAX),
            direction=direction,
            state=PedestrianState.ENTERING_ACTIVE,  # Will head to active zone
            max_wander_targets=random.randint(3, 6)
        )
        self.next_id += 1
        self.people.append(person)
        self.total_spawned += 1
        self.total_curious += 1
        return person.id
    
    def update(self, dt: float):
        """Update all pedestrians and handle spawning"""
        if self.paused:
            return
        
        # Update simulated time for long run mode
        if self.mode == SimulationMode.LONG_RUN:
            self.simulated_hours += (dt * self.time_scale) / 3600.0  # Convert to hours
            self._update_long_run_rates()
            self._update_gaps(dt)
        
        # Check if we're in a gap/lull (no spawning)
        if self.in_gap or self.in_lull:
            # Update existing people but don't spawn new ones
            self.people = [p for p in self.people if p.update(dt)]
            return
        
        # Spawn passive zone people
        self.passive_spawn_timer += dt
        spawn_interval = 60.0 / max(0.1, self.passive_spawn_rate)
        while self.passive_spawn_timer >= spawn_interval:
            self.spawn_passive_person()
            self.passive_spawn_timer -= spawn_interval
        
        # Spawn active zone people (rare)
        self.active_spawn_timer += dt
        active_interval = 60.0 / max(0.1, self.active_spawn_rate)
        while self.active_spawn_timer >= active_interval:
            self.spawn_active_person()
            self.active_spawn_timer -= active_interval
        
        # Spawn curious people
        self.curious_spawn_timer += dt
        curious_interval = 60.0 / max(0.1, self.curious_spawn_rate)
        while self.curious_spawn_timer >= curious_interval:
            self.spawn_curious_person()
            self.curious_spawn_timer -= curious_interval
        
        # Update all people
        self.people = [p for p in self.people if p.update(dt)]
    
    def get_stats(self):
        """Get counts by state"""
        passive = sum(1 for p in self.people if p.state == PedestrianState.PASSIVE_WALKING)
        entering = sum(1 for p in self.people if p.state == PedestrianState.ENTERING_ACTIVE)
        active = sum(1 for p in self.people if p.state == PedestrianState.ACTIVE_WANDERING)
        exiting = sum(1 for p in self.people if p.state in (
            PedestrianState.EXITING_ACTIVE, PedestrianState.EXITING_PASSIVE))
        return {
            'passive': passive,
            'entering': entering,
            'active': active,
            'exiting': exiting,
            'total': len(self.people),
            'in_gap': self.in_gap,
            'in_lull': self.in_lull,
            'pattern': self.current_pattern_desc,
            'simulated_hours': self.simulated_hours,
            'total_spawned': self.total_spawned,
        }
    
    def _update_long_run_rates(self):
        """Update spawn rates based on simulated time of day"""
        # Get hour of simulated day (wraps every 24 hours)
        hour = self.simulated_hours % 24
        
        # Find matching pattern
        for start, end, passive, curious, active, desc in LONG_RUN_PATTERNS:
            if start <= hour < end:
                # Add some random variation (+/- 20%)
                variation = random.uniform(0.8, 1.2)
                self.passive_spawn_rate = passive * variation
                self.curious_spawn_rate = curious * variation
                self.active_spawn_rate = active * variation
                self.current_pattern_desc = desc
                return
        
        # Default to quiet
        self.passive_spawn_rate = 10
        self.curious_spawn_rate = 0.5
        self.active_spawn_rate = 0.1
        self.current_pattern_desc = "default"
    
    def _update_gaps(self, dt: float):
        """Update gap/lull state for realistic traffic patterns"""
        now = time.time()
        
        # Check if current gap/lull has ended
        if self.in_gap and now >= self.gap_end_time:
            self.in_gap = False
        if self.in_lull and now >= self.lull_end_time:
            self.in_lull = False
        
        # Don't start new gaps during existing gaps/lulls
        if self.in_gap or self.in_lull:
            return
        
        # Random chance of starting a gap (short pause in traffic)
        if random.random() < GAP_SETTINGS['gap_probability'] * dt:
            self.in_gap = True
            gap_duration = random.uniform(
                GAP_SETTINGS['min_gap_duration'],
                GAP_SETTINGS['max_gap_duration']
            )
            self.gap_end_time = now + gap_duration
            return
        
        # Random chance of starting a lull (longer quiet period)
        if random.random() < GAP_SETTINGS['lull_probability'] * dt:
            self.in_lull = True
            lull_duration = random.uniform(
                GAP_SETTINGS['min_lull_duration'],
                GAP_SETTINGS['max_lull_duration']
            )
            self.lull_end_time = now + lull_duration
    
    def set_time_scale(self, scale: float):
        """Set time acceleration for long run mode (1.0 = real-time)"""
        self.time_scale = max(0.1, min(100.0, scale))


# =============================================================================
# OSC SENDER
# =============================================================================

class OSCSender:
    """Sends tracking data via OSC"""
    
    def __init__(self, ip: str, port: int):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.last_count = 0
        self.message_count = 0
        self.last_debug_time = time.time()
        print(f"📤 OSC sender initialized: sending to {ip}:{port}")
    
    def send_people(self, people: List[SimulatedPerson]):
        """Send all person positions"""
        for person in people:
            self.client.send_message(
                f"/tracker/person/{person.id}",
                [float(person.x), float(person.z)]
            )
            self.message_count += 1
        
        # Send count
        count = len(people)
        if count != self.last_count:
            self.client.send_message("/tracker/count", [count])
            self.last_count = count
        
        # Debug output every 5 seconds
        now = time.time()
        if now - self.last_debug_time > 5.0 and self.message_count > 0:
            print(f"  📤 Sent {self.message_count} OSC messages ({count} people)")
            self.last_debug_time = now
            self.message_count = 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pedestrian Simulator for Light Installation')
    parser.add_argument('--mode', choices=['normal', 'longrun'], default='normal',
                        help='Simulation mode: normal (constant traffic) or longrun (realistic patterns)')
    parser.add_argument('--timescale', type=float, default=1.0,
                        help='Time acceleration for longrun mode (1.0=realtime, 10=10x faster)')
    parser.add_argument('--hours', type=float, default=0,
                        help='For longrun: starting hour of day (0-24)')
    args = parser.parse_args()
    
    # Determine mode
    mode = SimulationMode.LONG_RUN if args.mode == 'longrun' else SimulationMode.NORMAL
    
    print("=" * 60)
    print("  PEDESTRIAN SIMULATOR (Headless)")
    print("  View results in lightController_osc.py")
    print("=" * 60)
    print()
    
    if mode == SimulationMode.LONG_RUN:
        print(f"  MODE: Long Run (realistic traffic patterns)")
        print(f"  Time Scale: {args.timescale}x")
        print(f"  Starting Hour: {args.hours:.1f}")
        print()
        print("  Traffic patterns vary throughout simulated day:")
        print("    - Dead of night (2-5am): Very quiet")
        print("    - Morning rush (7-9am): Heavy traffic")
        print("    - Lunch (11am-1pm): Busy")
        print("    - Evening rush (4-7pm): Heaviest")
        print("    - Evening leisure (7-9pm): Many curious visitors")
        print("    - Random gaps and lulls throughout")
    else:
        print(f"  MODE: Normal (constant traffic)")
    
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Create simulator and OSC sender
    simulator = PedestrianSimulator(mode=mode)
    if mode == SimulationMode.LONG_RUN:
        simulator.set_time_scale(args.timescale)
        simulator.simulated_hours = args.hours
    
    osc_sender = OSCSender(OSC_TARGET_IP, OSC_TARGET_PORT)
    
    print(f"Starting simulation at {FPS} FPS...")
    print()
    
    last_time = time.time()
    last_status_time = time.time()
    start_time = time.time()
    
    try:
        running = True
        while running:
            # Update simulation
            now = time.time()
            dt = min(now - last_time, 0.1)
            last_time = now
            
            simulator.update(dt)
            
            # Send OSC
            osc_sender.send_people(simulator.people)
            
            # Print status every 3 seconds
            if now - last_status_time > 3.0:
                stats = simulator.get_stats()
                elapsed = now - start_time
                
                if mode == SimulationMode.LONG_RUN:
                    sim_hour = stats['simulated_hours'] % 24
                    sim_day = int(stats['simulated_hours'] // 24) + 1
                    gap_status = ""
                    if stats['in_lull']:
                        gap_status = " [LULL]"
                    elif stats['in_gap']:
                        gap_status = " [gap]"
                    
                    print(f"  Day {sim_day} {sim_hour:05.2f}h ({stats['pattern']}){gap_status} | "
                          f"Total: {stats['total']} | "
                          f"Active: {stats['active']} | "
                          f"Spawned: {stats['total_spawned']}")
                else:
                    print(f"  Total: {stats['total']} | "
                          f"Passive: {stats['passive']} | "
                          f"Entering: {stats['entering']} | "
                          f"Active: {stats['active']} | "
                          f"Exiting: {stats['exiting']}")
                
                last_status_time = now
            
            # Sleep to maintain FPS
            time.sleep(1.0 / FPS)
    
    except KeyboardInterrupt:
        pass
    
    print()
    stats = simulator.get_stats()
    elapsed = time.time() - start_time
    print(f"Simulator stopped after {elapsed/60:.1f} minutes real time")
    if mode == SimulationMode.LONG_RUN:
        print(f"  Simulated {stats['simulated_hours']:.2f} hours")
        print(f"  Total pedestrians spawned: {stats['total_spawned']}")


if __name__ == "__main__":
    main()
