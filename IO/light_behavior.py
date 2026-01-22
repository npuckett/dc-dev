#!/usr/bin/env python3
"""
Light Behavior System

Manages the simulated point light's behavior modes, personality, and evolution.
Uses tracking database for self-analysis to avoid repetition and evolve over time.

Modes:
- IDLE: No one in active zone, gentle wander
- ENGAGED: 1-2 people in active zone, following behavior
- CROWD: 3+ people in active zone, energetic
- FLOW: Heavy passive traffic, drift with crowd

Meta Parameters (personality sliders 0-1):
- responsiveness: How quickly the light reacts
- energy: Overall liveliness and pulse speed
- attention_span: How long it stays focused
- sociability: Eagerness to engage with people
- exploration: How much it wanders
- memory: How much anti-repetition affects behavior
"""

import math
import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np


class BehaviorMode(Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    CROWD = "crowd"
    FLOW = "flow"


class GestureType(Enum):
    NONE = None
    ACKNOWLEDGE = "acknowledge"  # Brief move toward passerby
    CURIOUS = "curious"          # Slow approach toward person
    WELCOME = "welcome"          # Entrance flash for new person
    BORED = "bored"              # Attention-seeking movement
    FAREWELL = "farewell"        # Reluctant goodbye when leaving
    SURPRISED = "surprised"      # Quick pulse when someone appears suddenly
    THINKING = "thinking"        # Slow drift pause, as if contemplating
    HESITANT = "hesitant"        # Partial approach then retreat
    PLAYFUL = "playful"          # Quick zig-zag movement
    BLOOM = "bloom"              # Expand radius to illuminate all panels


@dataclass
class MetaParameters:
    """
    High-level personality controls (0.0 - 1.0).
    These affect how base parameters are calculated.
    """
    responsiveness: float = 0.5   # Low=slow/contemplative, High=quick/reactive
    energy: float = 0.5           # Low=calm/gentle, High=lively/dynamic
    attention_span: float = 0.5   # Low=easily distracted, High=focused/loyal
    sociability: float = 0.5      # Low=reserved, High=eager to engage
    exploration: float = 0.5      # Low=stays put, High=wanders widely
    memory: float = 0.5           # Low=forgets quickly, High=avoids repetition
    
    # Global multipliers
    brightness_global: float = 1.0
    speed_global: float = 1.0
    pulse_global: float = 1.0
    trend_weight: float = 1.0
    time_of_day_weight: float = 1.0
    anti_repetition_weight: float = 1.0
    
    # Toggles
    gestures_enabled: bool = True
    follow_enabled: bool = True
    flow_mode_enabled: bool = True
    dwell_rewards_enabled: bool = True
    entrance_flash_enabled: bool = True
    self_analysis_enabled: bool = True
    status_text_enabled: bool = True
    
    def lerp(self, low: float, high: float, param: float) -> float:
        """Linear interpolation based on a parameter (0-1)"""
        return low + (high - low) * param


@dataclass
class TimeOfDayModifier:
    """Modifiers based on time of day"""
    brightness_mult: float = 1.0
    pulse_mult: float = 1.0
    wander_y_min: float = 0
    wander_y_max: float = 150
    mood: str = "active"


@dataclass
class BehaviorState:
    """Current behavior state for the light"""
    mode: BehaviorMode = BehaviorMode.IDLE
    mode_start_time: float = 0.0
    mode_duration: float = 0.0
    
    # Current gesture
    gesture: GestureType = GestureType.NONE
    gesture_start_time: float = 0.0
    gesture_duration: float = 0.0
    gesture_target: Optional[np.ndarray] = None
    
    # Transition state
    transitioning: bool = False
    transition_start_time: float = 0.0
    transition_duration: float = 1.0
    transition_from_mode: BehaviorMode = BehaviorMode.IDLE
    
    # Dwell tracking (for engagement rewards)
    dwell_start_time: float = 0.0
    current_dwell_bonus: float = 0.0
    
    # Boredom tracking
    last_interaction_time: float = 0.0
    is_bored: bool = False
    
    # Bloom tracking (full-panel illumination moments)
    bloom_active: bool = False
    bloom_progress: float = 0.0  # 0-1 for smooth transition
    last_bloom_time: float = 0.0
    
    # Status text
    status_text: str = "..."
    
    # Last recorded time (for database)
    last_record_time: float = 0.0


class BehaviorSystem:
    """
    Main behavior controller for the light installation.
    
    Updates light parameters based on:
    - Current mode (IDLE/ENGAGED/CROWD/FLOW)
    - Meta parameters (personality)
    - Time of day
    - Trend data from database
    - Self-analysis to avoid repetition
    """
    
    # Time of day configurations (24-hour)
    TIME_CONFIGS = {
        (0, 6): TimeOfDayModifier(0.4, 1.5, 0, 60, "sleepy"),
        (6, 9): TimeOfDayModifier(0.7, 1.2, 0, 100, "waking"),
        (9, 17): TimeOfDayModifier(1.0, 1.0, 0, 150, "active"),
        (17, 20): TimeOfDayModifier(1.1, 0.9, 0, 150, "rush"),
        (20, 24): TimeOfDayModifier(0.6, 1.3, 0, 80, "evening"),
    }
    
    # Base mode parameters
    MODE_PARAMS = {
        BehaviorMode.IDLE: {
            'move_speed': 20,
            'wander_interval': 5.0,
            'brightness_min': 3,
            'brightness_max': 15,
            'pulse_speed': 4000,
            'falloff_radius': 80,
            'follow_smoothing': 0.0,  # Not following
        },
        BehaviorMode.ENGAGED: {
            'move_speed': 40,
            'wander_interval': 0.0,  # Not wandering
            'brightness_min': 8,
            'brightness_max': 30,
            'pulse_speed': 2500,
            'falloff_radius': 50,
            'follow_smoothing': 0.05,
        },
        BehaviorMode.CROWD: {
            'move_speed': 60,
            'wander_interval': 0.0,
            'brightness_min': 12,
            'brightness_max': 45,
            'pulse_speed': 1500,
            'falloff_radius': 40,
            'follow_smoothing': 0.03,
        },
        BehaviorMode.FLOW: {
            'move_speed': 25,
            'wander_interval': 3.0,
            'brightness_min': 5,
            'brightness_max': 20,
            'pulse_speed': 3000,
            'falloff_radius': 70,
            'follow_smoothing': 0.0,
        },
    }
    
    # Transition durations
    TRANSITIONS = {
        (BehaviorMode.IDLE, BehaviorMode.ENGAGED): 1.0,
        (BehaviorMode.ENGAGED, BehaviorMode.IDLE): 3.0,
        (BehaviorMode.ENGAGED, BehaviorMode.CROWD): 0.5,
        (BehaviorMode.CROWD, BehaviorMode.ENGAGED): 1.5,
    }
    
    # Status text templates - plain language describing what's happening
    STATUS_TEXTS = {
        ('idle', 'quiet'): [
            "No one nearby. Wandering slowly.",
            "Idle mode. Scanning for movement.",
            "Low activity. Brightness reduced.",
        ],
        ('idle', 'bored'): [
            "No interaction for 60+ seconds.",
            "Idle too long. Seeking attention.",
            "Boredom threshold reached.",
        ],
        ('idle', 'gesture'): [
            "Detected passive movement. Acknowledging.",
            "Person in passive zone. Brief gesture.",
            "Movement on sidewalk. Responding.",
        ],
        ('engaged', 1): [
            "1 person in active zone. Following.",
            "Tracking single visitor. Engaged mode.",
            "One person detected. Brightness increased.",
        ],
        ('engaged', 2): [
            "2 people in active zone. Following nearest.",
            "Multiple visitors. Tracking closest.",
            "Two people detected. Higher energy.",
        ],
        ('engaged', 'dwell'): [
            "Visitor staying. Dwell bonus active.",
            "Extended engagement. Rewarding with brightness.",
            "Long visit detected. Parameters boosted.",
        ],
        ('engaged', 'approach'): [
            "Moving closer to engaged visitor.",
            "Approaching interested person.",
            "Drawing nearer. Building connection.",
        ],
        ('crowd', 'default'): [
            "2+ people. Crowd mode. Following centroid.",
            "High activity. Maximum engagement.",
            "Multiple visitors. Pulse speed increased.",
        ],
        ('flow', 'default'): [
            "Heavy sidewalk traffic. Drifting with flow.",
            "Flow mode. Matching crowd direction.",
            "Passive zone busy. No active engagement.",
        ],
        ('bloom', 'default'): [
            "Bloom moment. Full panel illumination.",
            "Expanding to embrace entire space.",
            "Radiance spreading across all panels.",
        ],
    }
    
    def __init__(self, meta: MetaParameters = None, database = None):
        self.meta = meta or MetaParameters()
        self.database = database
        self.state = BehaviorState()
        self.state.mode_start_time = time.time()
        self.state.last_interaction_time = time.time()
        
        # Wander box (can be modified by behavior)
        # Extended Z range to allow light to approach visitors
        self.base_wander_box = {
            'min_x': -50, 'max_x': 290,
            'min_y': 0, 'max_y': 150,
            'min_z': -32, 'max_z': 200,  # Extended from 28 to approach visitors
        }
        self.current_wander_box = dict(self.base_wander_box)
        
        # Approach settings (how far forward the light moves toward people)
        self.approach_z_min = -32    # Closest to panels
        self.approach_z_max = 250    # Maximum approach toward visitors
        self.approach_speed = 0.1    # How quickly to approach (0-1 per second)
        
        # People tracking
        self.known_people: Dict[int, float] = {}  # id -> first_seen_time
        self.people_positions: Dict[int, np.ndarray] = {}  # id -> position
        
        # Current calculated parameters
        self.current_params = dict(self.MODE_PARAMS[BehaviorMode.IDLE])
        
        # Cooldowns
        self.last_gesture_time = 0.0
        self.gesture_cooldown = 5.0  # Minimum seconds between gestures
        
        # Flow threshold
        self.flow_threshold = 3  # people per minute in passive zone
        
        # Bloom settings (full-panel illumination)
        self.bloom_radius = 300  # Radius large enough to cover all panels
        self.bloom_cooldown = 45.0  # Minimum seconds between blooms
        self.bloom_duration = 3.0  # How long bloom lasts
        self.bloom_chance_per_minute = 0.15  # ~15% chance per minute in eligible states
    
    def get_time_of_day_modifier(self) -> TimeOfDayModifier:
        """Get modifier based on current hour"""
        hour = datetime.now().hour
        for (start, end), modifier in self.TIME_CONFIGS.items():
            if start <= hour < end:
                return modifier
        return TimeOfDayModifier()  # Default
    
    def determine_mode(self, active_count: int, passive_count: int,
                       passive_rate: float = 0.0) -> BehaviorMode:
        """Determine which mode we should be in based on inputs"""
        # CROWD threshold lowered from 3 to 2 for more crowd mode activation
        if active_count >= 2:
            return BehaviorMode.CROWD
        elif active_count >= 1:
            return BehaviorMode.ENGAGED
        elif self.meta.flow_mode_enabled and passive_rate >= self.flow_threshold:
            return BehaviorMode.FLOW
        else:
            return BehaviorMode.IDLE
    
    def start_transition(self, new_mode: BehaviorMode):
        """Begin transitioning to a new mode"""
        if new_mode == self.state.mode:
            return
        
        duration = self.TRANSITIONS.get(
            (self.state.mode, new_mode), 
            2.0  # Default transition
        )
        
        self.state.transitioning = True
        self.state.transition_start_time = time.time()
        self.state.transition_duration = duration
        self.state.transition_from_mode = self.state.mode
        self.state.mode = new_mode
        self.state.mode_start_time = time.time()
        self.state.mode_duration = 0.0
        
        # Reset dwell if leaving engaged
        if self.state.transition_from_mode in (BehaviorMode.ENGAGED, BehaviorMode.CROWD):
            self.state.dwell_start_time = 0.0
            self.state.current_dwell_bonus = 0.0
    
    def update_transition(self, dt: float):
        """Update transition progress"""
        if not self.state.transitioning:
            return 1.0  # Fully in current mode
        
        elapsed = time.time() - self.state.transition_start_time
        progress = min(1.0, elapsed / self.state.transition_duration)
        
        if progress >= 1.0:
            self.state.transitioning = False
        
        return progress
    
    def trigger_gesture(self, gesture_type: GestureType, target: np.ndarray = None,
                        duration: float = 1.5):
        """Start a gesture if allowed"""
        if not self.meta.gestures_enabled:
            return False
        
        # Check cooldown
        if time.time() - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        self.state.gesture = gesture_type
        self.state.gesture_start_time = time.time()
        self.state.gesture_duration = duration
        self.state.gesture_target = target
        self.last_gesture_time = time.time()
        
        return True
    
    def update_gesture(self, dt: float) -> bool:
        """Update gesture state. Returns True if gesture is active."""
        if self.state.gesture == GestureType.NONE:
            return False
        
        elapsed = time.time() - self.state.gesture_start_time
        if elapsed >= self.state.gesture_duration:
            self.state.gesture = GestureType.NONE
            self.state.gesture_target = None
            return False
        
        return True
    
    def check_boredom(self) -> bool:
        """Check if light should be bored (no interaction for a while)"""
        boredom_threshold = 60.0  # seconds
        time_since_interaction = time.time() - self.state.last_interaction_time
        return time_since_interaction > boredom_threshold
    
    def _update_bloom(self, dt: float, now: float, active_count: int):
        """Update bloom state - occasional full-panel illumination moments"""
        
        # Update bloom progress (smooth transition in/out)
        if self.state.bloom_active:
            # Bloom is on - check if it should end
            bloom_elapsed = now - self.state.last_bloom_time
            if bloom_elapsed < self.bloom_duration * 0.3:
                # Ramp up (first 30% of duration)
                self.state.bloom_progress = min(1.0, self.state.bloom_progress + dt * 2.0)
            elif bloom_elapsed > self.bloom_duration * 0.7:
                # Ramp down (last 30% of duration)
                self.state.bloom_progress = max(0.0, self.state.bloom_progress - dt * 2.0)
            
            if bloom_elapsed >= self.bloom_duration:
                self.state.bloom_active = False
                self.state.bloom_progress = 0.0
        else:
            # Bloom is off - maybe trigger one
            time_since_bloom = now - self.state.last_bloom_time
            if time_since_bloom < self.bloom_cooldown:
                return  # Still in cooldown
            
            # Bloom chance varies by context
            base_chance = self.bloom_chance_per_minute / 60.0  # per second
            
            # Higher chance during dwell (rewarding engagement)
            if self.state.current_dwell_bonus > 0:
                base_chance *= 2.0
            
            # Higher chance in crowd mode
            if self.state.mode == BehaviorMode.CROWD:
                base_chance *= 1.5
            
            # Lower chance if bored (save the bloom for when people arrive)
            if self.state.is_bored:
                base_chance *= 0.3
            
            # Roll for bloom
            if random.random() < base_chance * dt:
                self.state.bloom_active = True
                self.state.last_bloom_time = now
                self.state.bloom_progress = 0.0
                self.trigger_gesture(GestureType.BLOOM, duration=self.bloom_duration)

    def on_person_entered(self, person_id: int, position: np.ndarray, is_active_zone: bool):
        """Called when a new person is detected"""
        self.known_people[person_id] = time.time()
        self.people_positions[person_id] = position
        self.state.last_interaction_time = time.time()
        self.state.is_bored = False
        
        # Entrance gesture for active zone (with variety)
        if is_active_zone and self.meta.entrance_flash_enabled:
            # Choose entrance gesture based on context
            if len(self.known_people) == 1:
                # First person - welcome
                self.trigger_gesture(GestureType.WELCOME, position, duration=0.5)
            elif self.state.mode_duration < 2.0 and len(self.known_people) > 2:
                # Multiple people arriving quickly - surprised
                self.trigger_gesture(GestureType.SURPRISED, position, duration=0.8)
            elif self.meta.sociability > 0.7:
                # High sociability - curious approach
                self.trigger_gesture(GestureType.CURIOUS, position, duration=1.0)
            else:
                # Default welcome
                self.trigger_gesture(GestureType.WELCOME, position, duration=0.5)
        
        # Start dwell timer if first person in active zone
        if is_active_zone and self.state.dwell_start_time == 0:
            self.state.dwell_start_time = time.time()
    
    def on_person_left(self, person_id: int):
        """Called when a person leaves tracking"""
        if person_id in self.known_people:
            del self.known_people[person_id]
        if person_id in self.people_positions:
            del self.people_positions[person_id]
    
    def update_person_position(self, person_id: int, position: np.ndarray):
        """Update tracked person position"""
        self.people_positions[person_id] = position
    
    def get_follow_target(self, active_count: int, current_z: float = 0.0) -> Optional[np.ndarray]:
        """Calculate target position for following behavior with approach"""
        if not self.meta.follow_enabled or not self.people_positions:
            return None
        
        positions = list(self.people_positions.values())
        
        if active_count >= 2:
            # CROWD: Follow centroid
            centroid = np.mean(positions, axis=0)
            target = centroid.copy()
        else:
            # ENGAGED: Follow nearest to panels (smallest Z)
            nearest = min(positions, key=lambda p: p[2])
            target = nearest.copy()
        
        # APPROACH BEHAVIOR: Light moves forward toward person's Z position
        # but stays some distance back (30% of the way there)
        person_z = target[2]
        approach_target_z = min(person_z * 0.3, self.approach_z_max)  # Approach 30% of way to person
        approach_target_z = max(approach_target_z, self.approach_z_min)  # Don't go behind panels
        
        # Smooth Z approach based on engagement duration
        engagement_time = self.state.mode_duration
        approach_factor = min(1.0, engagement_time / 10.0)  # Full approach after 10s
        
        # Blend current Z toward approach target
        target[2] = self.approach_z_min + (approach_target_z - self.approach_z_min) * approach_factor
        
        return target
    
    def apply_meta_modifiers(self, params: Dict) -> Dict:
        """Apply meta parameter modifiers to base parameters"""
        m = self.meta
        result = dict(params)
        
        # Responsiveness affects smoothing and speeds
        result['follow_smoothing'] = m.lerp(0.02, 0.10, m.responsiveness) * result.get('follow_smoothing', 0.05)
        result['move_speed'] = result['move_speed'] * m.lerp(0.6, 1.4, m.responsiveness) * m.speed_global
        
        # Energy affects pulse and brightness
        result['pulse_speed'] = result['pulse_speed'] * m.lerp(1.3, 0.7, m.energy) * m.pulse_global
        result['brightness_max'] = result['brightness_max'] * m.lerp(0.7, 1.3, m.energy) * m.brightness_global
        result['brightness_min'] = result['brightness_min'] * m.lerp(0.8, 1.2, m.energy) * m.brightness_global
        
        # Exploration affects wander box and interval
        result['wander_interval'] = result.get('wander_interval', 3.0) * m.lerp(1.5, 0.5, m.exploration)
        
        return result
    
    def apply_time_of_day(self, params: Dict) -> Dict:
        """Apply time of day modifiers"""
        tod = self.get_time_of_day_modifier()
        weight = self.meta.time_of_day_weight
        
        result = dict(params)
        result['brightness_max'] *= (1.0 + (tod.brightness_mult - 1.0) * weight)
        result['brightness_min'] *= (1.0 + (tod.brightness_mult - 1.0) * weight)
        result['pulse_speed'] *= (1.0 + (tod.pulse_mult - 1.0) * weight)
        
        # Update wander box based on time
        self.current_wander_box['min_y'] = self.base_wander_box['min_y']
        self.current_wander_box['max_y'] = int(
            self.base_wander_box['min_y'] + 
            (tod.wander_y_max - self.base_wander_box['min_y']) * weight +
            (self.base_wander_box['max_y'] - tod.wander_y_max) * (1 - weight)
        )
        
        return result
    
    def apply_dwell_rewards(self, params: Dict) -> Dict:
        """Apply bonuses for people who dwell in active zone"""
        if not self.meta.dwell_rewards_enabled:
            return params
        
        if self.state.dwell_start_time == 0:
            return params
        
        dwell_time = time.time() - self.state.dwell_start_time
        
        # Bonus starts after 10s, max at 40s
        if dwell_time < 10:
            return params
        
        # +5 brightness per 10s, max +15
        bonus = min(15, int((dwell_time - 10) / 10) * 5)
        self.state.current_dwell_bonus = bonus
        
        result = dict(params)
        result['brightness_max'] = result['brightness_max'] + bonus
        result['brightness_min'] = result['brightness_min'] + bonus * 0.5
        
        return result
    
    def apply_anti_repetition(self, params: Dict, current_pos: Tuple[float, float, float]) -> Dict:
        """Apply modifications to avoid repetitive behavior"""
        if not self.meta.self_analysis_enabled or not self.database:
            return params
        
        weight = self.meta.anti_repetition_weight * self.meta.memory
        if weight < 0.1:
            return params
        
        result = dict(params)
        
        # Check position entropy
        entropy = self.database.get_position_entropy(60)
        if entropy < 0.3:
            # Low entropy = too repetitive, encourage exploration
            result['wander_interval'] = result.get('wander_interval', 3.0) * (1 - 0.3 * weight)
            result['move_speed'] = result['move_speed'] * (1 + 0.2 * weight)
        
        # Check response similarity
        people_count = len(self.people_positions)
        similarity = self.database.get_response_similarity(people_count)
        if similarity > 0.7:
            # Too similar to past responses, add variety
            result['pulse_speed'] = result['pulse_speed'] * random.uniform(0.85, 1.15)
            result['brightness_max'] = result['brightness_max'] * random.uniform(0.9, 1.1)
        
        return result
    
    def calculate_parameters(self, active_count: int, passive_count: int,
                            current_pos: Tuple[float, float, float],
                            flow_balance: float = 0.0) -> Dict:
        """Calculate final light parameters based on all factors"""
        
        # Get base params for current mode
        base_params = dict(self.MODE_PARAMS[self.state.mode])
        
        # Handle transitions (interpolate between modes)
        if self.state.transitioning:
            progress = self.update_transition(0)
            from_params = self.MODE_PARAMS[self.state.transition_from_mode]
            for key in base_params:
                if isinstance(base_params[key], (int, float)):
                    base_params[key] = (
                        from_params[key] * (1 - progress) +
                        base_params[key] * progress
                    )
        
        # Scale based on people count in ENGAGED/CROWD
        if self.state.mode in (BehaviorMode.ENGAGED, BehaviorMode.CROWD):
            people_scale = min(1.0 + active_count * 0.1, 1.5)
            base_params['brightness_max'] *= people_scale
            base_params['pulse_speed'] *= (1.0 / people_scale)
        
        # Apply modifiers in order
        params = self.apply_meta_modifiers(base_params)
        params = self.apply_time_of_day(params)
        params = self.apply_dwell_rewards(params)
        params = self.apply_anti_repetition(params, current_pos)
        
        # Apply flow bias to wander box
        if self.state.mode in (BehaviorMode.IDLE, BehaviorMode.FLOW):
            flow_bias = flow_balance * 50 * self.meta.trend_weight
            self.current_wander_box['min_x'] = self.base_wander_box['min_x'] + flow_bias
            self.current_wander_box['max_x'] = self.base_wander_box['max_x'] + flow_bias
        
        # Apply bloom effect (smooth transition to full-panel radius)
        if self.state.bloom_active or self.state.bloom_progress > 0:
            base_radius = params.get('falloff_radius', 50)
            # Lerp between normal radius and bloom radius
            params['falloff_radius'] = base_radius + (self.bloom_radius - base_radius) * self.state.bloom_progress
            # Also boost brightness during bloom
            params['brightness_max'] = params.get('brightness_max', 30) * (1 + 0.5 * self.state.bloom_progress)
        
        self.current_params = params
        return params
    
    def update_status_text(self, active_count: int):
        """Update the public-facing status text"""
        if not self.meta.status_text_enabled:
            self.state.status_text = ""
            return
        
        key = None
        
        if self.state.bloom_active:
            key = ('bloom', 'default')
        elif self.state.gesture != GestureType.NONE:
            key = ('idle', 'gesture')
        elif self.state.mode == BehaviorMode.IDLE:
            if self.state.is_bored:
                key = ('idle', 'bored')
            else:
                key = ('idle', 'quiet')
        elif self.state.mode == BehaviorMode.ENGAGED:
            if self.state.current_dwell_bonus > 0:
                key = ('engaged', 'dwell')
            elif active_count == 1:
                key = ('engaged', 1)
            else:
                key = ('engaged', 2)
        elif self.state.mode == BehaviorMode.CROWD:
            key = ('crowd', 'default')
        elif self.state.mode == BehaviorMode.FLOW:
            key = ('flow', 'default')
        
        if key and key in self.STATUS_TEXTS:
            self.state.status_text = random.choice(self.STATUS_TEXTS[key])
        else:
            self.state.status_text = "..."
    
    def update(self, dt: float, active_count: int, passive_count: int,
               current_pos: Tuple[float, float, float],
               passive_rate: float = 0.0, flow_balance: float = 0.0) -> Dict:
        """
        Main update method. Call every frame.
        
        Args:
            dt: Delta time in seconds
            active_count: People in active zone
            passive_count: People in passive zone
            current_pos: Current light position
            passive_rate: People per minute in passive zone
            flow_balance: -1 (R→L) to +1 (L→R)
        
        Returns:
            Dict of calculated parameters to apply to light
        """
        now = time.time()
        
        # Update mode duration
        self.state.mode_duration = now - self.state.mode_start_time
        
        # Check for mode change
        new_mode = self.determine_mode(active_count, passive_count, passive_rate)
        if new_mode != self.state.mode and not self.state.transitioning:
            self.start_transition(new_mode)
        
        # Update gesture
        self.update_gesture(dt)
        
        # Check boredom (IDLE mode only)
        if self.state.mode == BehaviorMode.IDLE:
            self.state.is_bored = self.check_boredom()
            
            # Maybe trigger bored gesture (with variety)
            if self.state.is_bored and self.state.gesture == GestureType.NONE:
                if random.random() < 0.005:  # Low chance per frame
                    # Pick from variety of idle gestures based on personality and time
                    gesture_options = [
                        (GestureType.BORED, 2.0),
                        (GestureType.THINKING, 3.0),
                        (GestureType.CURIOUS, 2.5),
                    ]
                    # Add playful if high energy
                    if self.meta.energy > 0.6:
                        gesture_options.append((GestureType.PLAYFUL, 1.5))
                    # Add hesitant if low sociability
                    if self.meta.sociability < 0.4:
                        gesture_options.append((GestureType.HESITANT, 2.0))
                    
                    gesture_type, duration = random.choice(gesture_options)
                    self.trigger_gesture(gesture_type, duration=duration)
        
        # Maybe trigger passive acknowledgment
        if (self.state.mode == BehaviorMode.IDLE and 
            passive_count > 0 and 
            self.state.gesture == GestureType.NONE):
            gesture_chance = 0.02 * self.meta.sociability
            if random.random() < gesture_chance * dt:
                # Pick a random edge position
                edge_x = random.choice([
                    self.current_wander_box['min_x'],
                    self.current_wander_box['max_x']
                ])
                target = np.array([edge_x, 30, 0])
                self.trigger_gesture(GestureType.ACKNOWLEDGE, target, duration=1.5)
        
        # Update bloom state
        self._update_bloom(dt, now, active_count)
        
        # Calculate parameters
        params = self.calculate_parameters(
            active_count, passive_count, current_pos, flow_balance
        )
        
        # Update status text periodically
        if int(now) % 5 == 0:  # Every 5 seconds
            self.update_status_text(active_count)
        
        # Record to database (if enabled)
        record_interval = 0.5 if active_count > 0 else 2.0
        if self.database and now - self.state.last_record_time >= record_interval:
            self.database.record_light_state(
                mode=self.state.mode.value,
                position=current_pos,
                target=current_pos,  # Would need actual target
                brightness=params.get('brightness_max', 30),
                pulse_speed=params.get('pulse_speed', 2000),
                move_speed=params.get('move_speed', 50),
                people_count=active_count + passive_count,
                active_count=active_count,
                passive_count=passive_count,
                gesture_type=self.state.gesture.value if self.state.gesture != GestureType.NONE else None,
                status_text=self.state.status_text
            )
            self.state.last_record_time = now
        
        return params
    
    def get_wander_box(self) -> Dict:
        """Get current (modified) wander box"""
        return self.current_wander_box
    
    def get_gesture_target(self) -> Optional[np.ndarray]:
        """Get target position for current gesture, if any"""
        if self.state.gesture == GestureType.NONE:
            return None
        return self.state.gesture_target
    
    def should_wander(self) -> bool:
        """Check if light should be wandering (vs following)"""
        return self.state.mode in (BehaviorMode.IDLE, BehaviorMode.FLOW)
    
    def get_status(self) -> Dict:
        """Get current behavior status for display"""
        return {
            'mode': self.state.mode.value,
            'mode_duration': self.state.mode_duration,
            'transitioning': self.state.transitioning,
            'gesture': self.state.gesture.value if self.state.gesture != GestureType.NONE else None,
            'is_bored': self.state.is_bored,
            'dwell_bonus': self.state.current_dwell_bonus,
            'status_text': self.state.status_text,
            'time_of_day': self.get_time_of_day_modifier().mood,
        }


# =============================================================================
# PRESET PERSONALITIES
# =============================================================================

PRESETS = {
    'default': MetaParameters(),
    'shy': MetaParameters(
        responsiveness=0.3, energy=0.3, attention_span=0.7,
        sociability=0.2, exploration=0.3, memory=0.6
    ),
    'eager': MetaParameters(
        responsiveness=0.8, energy=0.7, attention_span=0.4,
        sociability=0.9, exploration=0.6, memory=0.4
    ),
    'zen': MetaParameters(
        responsiveness=0.2, energy=0.2, attention_span=0.9,
        sociability=0.4, exploration=0.4, memory=0.8
    ),
    'playful': MetaParameters(
        responsiveness=0.7, energy=0.8, attention_span=0.3,
        sociability=0.7, exploration=0.9, memory=0.3
    ),
    'night_owl': MetaParameters(
        responsiveness=0.4, energy=0.3, attention_span=0.6,
        sociability=0.5, exploration=0.2, memory=0.7
    ),
}


def load_preset(name: str) -> MetaParameters:
    """Load a preset personality"""
    return PRESETS.get(name, PRESETS['default'])


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing BehaviorSystem...")
    
    # Create with default personality
    behavior = BehaviorSystem()
    
    # Simulate some updates
    pos = (120.0, 60.0, 0.0)
    
    print("\n--- IDLE mode ---")
    for i in range(5):
        params = behavior.update(0.1, active_count=0, passive_count=2,
                                 current_pos=pos, passive_rate=1.0)
        print(f"  Frame {i}: mode={behavior.state.mode.value}, "
              f"brightness_max={params['brightness_max']:.1f}, "
              f"status='{behavior.state.status_text}'")
    
    print("\n--- Person enters (ENGAGED) ---")
    behavior.on_person_entered(1, np.array([100, -66, 150]), is_active_zone=True)
    for i in range(5):
        params = behavior.update(0.1, active_count=1, passive_count=0,
                                 current_pos=pos)
        print(f"  Frame {i}: mode={behavior.state.mode.value}, "
              f"brightness_max={params['brightness_max']:.1f}, "
              f"gesture={behavior.state.gesture.value}")
    
    print("\n--- Testing presets ---")
    for name, meta in PRESETS.items():
        behavior = BehaviorSystem(meta=meta)
        params = behavior.update(0.1, active_count=1, passive_count=0, current_pos=pos)
        print(f"  {name}: move_speed={params['move_speed']:.1f}, "
              f"pulse_speed={params['pulse_speed']:.0f}")
    
    print("\nDone!")
