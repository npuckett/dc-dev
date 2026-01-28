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
import threading
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


class TimePeriod(Enum):
    """Time of day periods for trend analysis"""
    LATE_NIGHT = "late_night"  # 0-6
    MORNING = "morning"        # 6-12
    AFTERNOON = "afternoon"    # 12-17
    EVENING = "evening"        # 17-24
    
    @staticmethod
    def current() -> 'TimePeriod':
        hour = datetime.now().hour
        if 0 <= hour < 6:
            return TimePeriod.LATE_NIGHT
        elif 6 <= hour < 12:
            return TimePeriod.MORNING
        elif 12 <= hour < 17:
            return TimePeriod.AFTERNOON
        else:
            return TimePeriod.EVENING


@dataclass
class IdleTrends:
    """Processed trend data for IDLE mode behavior"""
    # Recent immediate data (last 1 minute) - events in that window
    recent_passive_count: int = 0
    recent_active_count: int = 0
    recent_flow_direction: float = 0.0  # -1 = all right-to-left, +1 = all left-to-right
    recent_avg_speed: float = 0.0
    
    # Short term (5 minutes) - cumulative events in window
    short_passive_count: int = 0
    short_active_count: int = 0
    short_flow_direction: float = 0.0
    short_activity_level: float = 0.0  # 0 = dead, 1 = very busy
    
    # Medium term (30 minutes) - cumulative events in window
    medium_passive_count: int = 0
    medium_active_count: int = 0
    medium_flow_direction: float = 0.0
    medium_activity_level: float = 0.0
    
    # Long term (1 hour) - cumulative events in window
    long_passive_count: int = 0
    long_active_count: int = 0
    long_flow_direction: float = 0.0
    long_activity_level: float = 0.0
    
    # Time period pattern (historical average for this time period)
    period_typical_count: int = 0
    period_typical_flow: float = 0.0
    period_name: str = "unknown"
    
    # Computed influence values (0-1 normalized)
    activity_anticipation: float = 0.5    # Should we be ready for action?
    flow_momentum: float = 0.0            # Sustained directional momentum
    energy_level: float = 0.5             # Overall energy to match
    
    # Data availability flags
    has_recent_data: bool = False         # Has 1-minute data
    has_short_data: bool = False          # Has 5-minute data
    has_medium_data: bool = False         # Has 30-minute data
    has_long_data: bool = False           # Has 1-hour data
    has_historical_data: bool = False     # Has multi-day historical pattern
    database_error: str = ""              # Last error if any
    
    # Timestamps
    last_update: float = 0.0


@dataclass
class AggressionState:
    """
    Tracks the "attention-seeking" aggression level of the system.
    
    Aggression rises when:
    - Time passes without engagement (people walk by without stopping)
    - High passive traffic with low conversion
    
    Aggression falls when:
    - Someone engages (enters active zone)
    - Recent engagement success
    
    Aggression is capped by time of day (e.g., low at night in financial district).
    """
    level: float = 0.3              # Current aggression (0.0 = passive, 1.0 = maximum)
    raw_level: float = 0.3          # Uncapped level (before time-of-day cap)
    
    # Factors that increase aggression
    seconds_since_engagement: float = 0.0     # Time since last active zone entry
    passive_without_conversion: int = 0       # Passive zone visits without any conversion
    
    # Factors that decrease aggression
    recent_engagements: int = 0               # Engagements in the last 5 minutes
    current_engagement: bool = False          # Someone currently in active zone
    
    # Time-of-day cap (set by current hour)
    time_of_day_cap: float = 0.8              # Maximum aggression allowed right now
    
    # EMA smoothing (prevents jarring changes)
    ema_alpha: float = 0.1                    # Smoothing factor for level updates
    
    # Tracking for conversion rate
    last_engagement_time: float = 0.0         # When someone last entered active zone
    last_passive_count: int = 0               # For tracking new passive zone visits
    
    # Timestamps
    last_update: float = 0.0


# Time-of-day aggression caps for financial district location
# Key: hour (0-23), Value: max aggression level
# Morning/evening rush: Low aggression (commuters won't stop)
# Mid-day: Higher aggression (people might explore)
# Night: Very low (area is dead)
AGGRESSION_TIME_CAPS = {
    0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2,   # Late night: very low
    6: 0.3, 7: 0.3,                                     # Early morning: low
    8: 0.4, 9: 0.5,                                     # Morning rush easing: medium-low
    10: 0.7, 11: 0.8,                                   # Late morning: higher
    12: 0.8, 13: 0.8, 14: 0.7,                          # Lunch: high
    15: 0.6, 16: 0.5,                                   # Afternoon: medium
    17: 0.4, 18: 0.4,                                   # Evening rush: low
    19: 0.5, 20: 0.4,                                   # Evening: medium-low
    21: 0.3, 22: 0.3, 23: 0.2,                          # Night: low
}


@dataclass
class FlowState:
    """
    Tracks real-time pedestrian flow direction for anticipatory positioning.
    
    Updated more frequently than IdleTrends (every 1-2 seconds vs 5-10 seconds).
    Uses a 30-second window for responsive flow tracking.
    
    Flow direction affects wander box X positioning:
    - Positive flow (left-to-right) = people coming from left (negative X)
    - Negative flow (right-to-left) = people coming from right (positive X)
    - Light should position toward where people are coming FROM
    """
    # Current smoothed flow direction (-1 to +1)
    direction: float = 0.0              # -1 = right-to-left, +1 = left-to-right
    raw_direction: float = 0.0          # Unsmoothed direction for display
    
    # Flow strength (how confident we are in the direction)
    strength: float = 0.0               # 0 = no flow/mixed, 1 = strong consistent flow
    
    # Raw counts from last 30-second window
    left_to_right_count: int = 0        # People moving left-to-right
    right_to_left_count: int = 0        # People moving right-to-left
    total_events: int = 0               # Total movement events
    
    # Derived positioning offset for wander box
    # Negative = shift box left (toward left edge), Positive = shift right
    x_offset: float = 0.0               # In centimeters, applied to wander box
    
    # EMA smoothing (faster than aggression for responsiveness)
    ema_alpha: float = 0.25             # Higher alpha = more responsive
    
    # Timing
    last_update: float = 0.0
    update_interval: float = 1.5        # Update every 1.5 seconds (faster than trends)


class AttractionStrategy(Enum):
    """Strategies for attracting almost-engaged people"""
    NONE = "none"
    BRIGHTNESS_PULSE = "brightness_pulse"   # Subtle brightness increase
    DRIFT_TOWARD = "drift_toward"           # Move light toward person
    PAUSE_AND_LOOK = "pause_and_look"       # Stop and focus on them
    COMBINED = "combined"                   # All strategies together


@dataclass
class AlmostEngagedCandidate:
    """Tracks a single person who might be about to engage"""
    person_id: int
    first_detected: float = 0.0         # When we first noticed them slowing
    last_seen: float = 0.0              # Last position update
    
    # Position tracking
    position_x: float = 0.0
    position_z: float = 0.0
    
    # Speed tracking (cm/s)
    current_speed: float = 0.0
    initial_speed: float = 0.0          # Speed when first detected
    min_speed_seen: float = 999.0       # Lowest speed observed
    
    # Distance to active zone boundary
    distance_to_active: float = 0.0     # cm to nearest active zone edge
    
    # Attraction attempt tracking
    strategy_used: AttractionStrategy = AttractionStrategy.NONE
    strategy_start_time: float = 0.0
    
    # Outcome tracking
    converted: bool = False             # Did they enter active zone?
    left_area: bool = False             # Did they leave without converting?
    outcome_logged: bool = False        # Has outcome been recorded?


@dataclass 
class AlmostEngagedState:
    """
    Tracks "almost engaged" detection - people who slow down in passive zone
    near the active zone boundary.
    
    These are prime targets for attraction strategies:
    - Subtle brightness pulse
    - Gentle drift toward them  
    - Pause and "look" at them
    
    We track outcomes to learn which strategies work best.
    """
    # Currently tracked candidates
    candidates: Dict[int, AlmostEngagedCandidate] = field(default_factory=dict)
    
    # Detection thresholds
    slow_speed_threshold: float = 50.0      # Below 50 cm/s = slowing down
    near_active_threshold: float = 100.0    # Within 100cm of active zone
    min_detection_time: float = 1.0         # Must be slow for 1+ seconds
    
    # Current attraction state
    active_attraction: bool = False         # Are we actively attracting someone?
    attraction_target_id: int = -1          # Who are we attracting?
    current_strategy: AttractionStrategy = AttractionStrategy.NONE
    
    # Strategy rotation (for A/B testing)
    strategy_index: int = 0                 # Which strategy to try next
    strategies_to_test: List[AttractionStrategy] = field(default_factory=lambda: [
        AttractionStrategy.BRIGHTNESS_PULSE,
        AttractionStrategy.DRIFT_TOWARD,
        AttractionStrategy.PAUSE_AND_LOOK,
    ])
    
    # Conversion tracking (for learning)
    total_detected: int = 0                 # Total almost-engaged detected
    total_converted: int = 0                # How many entered active zone
    strategy_stats: Dict[str, Dict] = field(default_factory=lambda: {
        'brightness_pulse': {'attempts': 0, 'conversions': 0},
        'drift_toward': {'attempts': 0, 'conversions': 0},
        'pause_and_look': {'attempts': 0, 'conversions': 0},
        'none': {'attempts': 0, 'conversions': 0},
    })
    
    # Cooldown to prevent spamming
    last_attraction_time: float = 0.0
    attraction_cooldown: float = 5.0        # Seconds between attraction attempts
    
    # Timing
    last_update: float = 0.0


@dataclass
class EngagementContext:
    """
    Captures the behavior state at the moment someone engages.
    
    This snapshot lets us learn what behavior parameters were active
    when engagement occurred, so we can weight successful patterns higher.
    """
    # Timestamp
    timestamp: float = 0.0
    time_of_day: str = ""                   # Period name
    hour: int = 0
    
    # Mode at engagement
    mode_before: str = "idle"               # What mode were we in?
    mode_duration: float = 0.0              # How long in that mode?
    
    # Aggression level
    aggression_level: float = 0.0
    
    # Flow state
    flow_direction: float = 0.0
    flow_x_offset: float = 0.0
    
    # Light position at engagement (normalized 0-1)
    light_x_normalized: float = 0.5
    light_z_normalized: float = 0.5
    
    # Movement characteristics
    move_speed: float = 0.0
    brightness: float = 0.0
    intensity: float = 0.0
    
    # Wander box position
    wander_x_offset: float = 0.0
    
    # Person info
    person_zone: str = "active"             # Where they came from
    dwell_duration: float = 0.0             # How long they stayed after engaging
    
    # Almost-engaged tracking
    was_almost_engaged: bool = False        # Were they an almost-engaged candidate?
    attraction_strategy_used: str = "none"
    

@dataclass
class FeedbackLearning:
    """
    Tracks behavior-to-engagement correlations and learns weights.
    
    The system logs what it was doing when people engage, then gradually
    weights successful behaviors higher. This creates emergent learning:
    "What was I doing when people engaged? Do more of that."
    
    Learning is very slow and conservative to avoid overcorrection.
    """
    # Recent engagement contexts (ring buffer, last 50)
    recent_contexts: List[EngagementContext] = field(default_factory=list)
    max_contexts: int = 50
    
    # Learned weights (start at 1.0 = neutral)
    # These multiply the corresponding behavior parameters
    weights: Dict[str, float] = field(default_factory=lambda: {
        # Aggression effectiveness
        'low_aggression': 1.0,          # Aggression < 0.3
        'mid_aggression': 1.0,          # Aggression 0.3-0.6
        'high_aggression': 1.0,         # Aggression > 0.6
        
        # Position effectiveness
        'left_position': 1.0,           # Light on left side
        'center_position': 1.0,         # Light in center
        'right_position': 1.0,          # Light on right side
        
        # Flow alignment
        'flow_aligned': 1.0,            # Light was toward incoming flow
        'flow_neutral': 1.0,            # No significant flow
        'flow_opposed': 1.0,            # Light was away from flow
        
        # Time of day
        'morning': 1.0,
        'afternoon': 1.0,
        'evening': 1.0,
        'late_night': 1.0,
        
        # Mode effectiveness
        'from_idle': 1.0,               # Engaged from idle mode
        'from_flow': 1.0,               # Engaged from flow mode
    })
    
    # Learning rate (very slow for stability)
    learning_rate: float = 0.02         # Weight change per engagement
    weight_min: float = 0.5             # Minimum weight
    weight_max: float = 2.0             # Maximum weight
    
    # Statistics
    total_engagements: int = 0          # Total engagements tracked
    engagements_by_hour: Dict[int, int] = field(default_factory=lambda: {h: 0 for h in range(24)})
    
    # Log file
    log_file: str = "engagement_feedback.log"
    
    # Timing
    last_update: float = 0.0
    session_start: float = 0.0


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
    follow_speed_global: float = 1.0  # Multiplier for follow tracking speed
    dwell_influence: float = 1.0      # How much dwell time affects behavior (0=none, 2=double)
    trend_weight: float = 1.0
    time_of_day_weight: float = 1.0
    anti_repetition_weight: float = 1.0
    idle_trend_weight: float = 1.0     # How much passive zone trends affect IDLE behavior (0=none, 2=double)
    
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
    
    # Mode stickiness - pending mode change
    pending_mode: Optional[BehaviorMode] = None
    pending_mode_start: float = 0.0  # When conditions for pending mode started
    
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
    
    # Entry pulse tracking (acknowledgment when someone enters active zone)
    entry_pulse_active: bool = False
    entry_pulse_start: float = 0.0
    entry_pulse_duration: float = 0.8  # Duration of entry pulse effect
    
    # Proximity tracking (for Z-based responses)
    nearest_person_z: float = 500.0  # Z of nearest person (lower = closer to panels)
    proximity_factor: float = 0.0     # 0-1, higher = closer to panels
    
    # Status text
    status_text: str = "..."
    
    # Last recorded time (for database)
    last_record_time: float = 0.0
    
    # Decision inputs (for display/debugging)
    last_active_count: int = 0
    last_passive_count: int = 0
    last_passive_rate: float = 0.0
    
    # Idle trend tracking
    idle_trends: Optional['IdleTrends'] = None
    last_trend_update: float = 0.0
    
    # Aggression tracking
    aggression: Optional['AggressionState'] = None
    
    # Flow tracking (for anticipatory positioning)
    flow: Optional['FlowState'] = None
    
    # Almost-engaged tracking (for attraction strategies)
    almost_engaged: Optional['AlmostEngagedState'] = None
    
    # Feedback learning (behavior-to-engagement correlation)
    feedback_learning: Optional['FeedbackLearning'] = None


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
            'move_speed': 25,           # Slower, more contemplative
            'wander_interval': 4.0,     # Longer pauses between movements
            'brightness_min': 8,
            'brightness_max': 30,
            'pulse_speed': 2500,
            'falloff_radius': 50,
            'follow_smoothing': 0.03,   # Gentler following
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
    
    # Transition durations - how long parameter interpolation takes
    TRANSITIONS = {
        (BehaviorMode.IDLE, BehaviorMode.ENGAGED): 0.8,    # Quick engage
        (BehaviorMode.ENGAGED, BehaviorMode.IDLE): 3.0,    # Slow fade out - reluctant goodbye
        (BehaviorMode.ENGAGED, BehaviorMode.CROWD): 0.5,   # Quick escalate
        (BehaviorMode.CROWD, BehaviorMode.ENGAGED): 2.0,   # Gradual de-escalate
        (BehaviorMode.CROWD, BehaviorMode.IDLE): 4.0,      # Slow fade when everyone leaves
        (BehaviorMode.IDLE, BehaviorMode.FLOW): 2.0,       # Gradual flow transition
        (BehaviorMode.FLOW, BehaviorMode.IDLE): 3.0,       # Slow exit from flow
        (BehaviorMode.FLOW, BehaviorMode.ENGAGED): 0.8,    # Quick engage from flow
    }
    
    # Mode stickiness - minimum time conditions must persist before switching
    # Exception: IDLE -> ENGAGED (active zone entry) is always immediate
    MODE_STICKINESS = {
        # (from_mode, to_mode): seconds conditions must persist
        (BehaviorMode.IDLE, BehaviorMode.ENGAGED): 0.0,      # Immediate when someone enters active zone
        (BehaviorMode.IDLE, BehaviorMode.FLOW): 15.0,        # Wait 15s of passive traffic before flow mode
        (BehaviorMode.ENGAGED, BehaviorMode.IDLE): 5.0,      # Wait 5s after last person leaves
        (BehaviorMode.ENGAGED, BehaviorMode.CROWD): 3.0,     # Wait 3s with 2+ people before crowd
        (BehaviorMode.CROWD, BehaviorMode.ENGAGED): 5.0,     # Wait 5s after crowd thins
        (BehaviorMode.CROWD, BehaviorMode.IDLE): 5.0,        # Wait 5s after everyone leaves
        (BehaviorMode.FLOW, BehaviorMode.IDLE): 10.0,        # Wait 10s of low traffic before idle
        (BehaviorMode.FLOW, BehaviorMode.ENGAGED): 0.0,      # Immediate when someone enters active zone
    }
    
    # Minimum time to stay in a mode before any switch (except emergency immediate switches)
    MIN_MODE_DURATION = 8.0  # Stay in mode at least 8 seconds
    
    # Z Proximity settings - how the light responds to distance from panels
    # Active zone Z range: 78 (camera ledge) to 283 (back of active zone)
    PROXIMITY_Z_NEAR = 100    # Z at which proximity_factor = 1.0 (very close)
    PROXIMITY_Z_FAR = 280     # Z at which proximity_factor = 0.0 (far edge of active zone)
    
    # Proximity multipliers - how much proximity affects parameters
    PROXIMITY_SPEED_MULT = {'near': 0.6, 'far': 1.4}   # Slower when close (more deliberate)
    PROXIMITY_BRIGHTNESS_MULT = {'near': 1.4, 'far': 0.8}  # Brighter when close
    PROXIMITY_SMOOTHING_MULT = {'near': 0.7, 'far': 1.3}  # More precise when close
    
    # Entry pulse settings
    ENTRY_PULSE_BRIGHTNESS_BOOST = 25  # Additional brightness during entry pulse
    ENTRY_PULSE_DURATION = 0.8         # Seconds
    
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
        
        # Initialize aggression state
        self.state.aggression = AggressionState()
        self.state.aggression.last_engagement_time = time.time()
        self.state.aggression.last_update = time.time()
        
        # Initialize flow state (for anticipatory positioning)
        self.state.flow = FlowState()
        self.state.flow.last_update = time.time()
        
        # Initialize almost-engaged state (for attraction strategies)
        self.state.almost_engaged = AlmostEngagedState()
        self.state.almost_engaged.last_update = time.time()
        
        # Initialize feedback learning (behavior-to-engagement correlation)
        self.state.feedback_learning = FeedbackLearning()
        self.state.feedback_learning.session_start = time.time()
        self.state.feedback_learning.last_update = time.time()
        
        # Current light position (for feedback context capture)
        self.current_light_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        
        # Background trends updater (to avoid blocking main loop)
        self._trends_thread: Optional[threading.Thread] = None
        self._trends_lock = threading.Lock()
        self._pending_trends: Optional[IdleTrends] = None
        
        # Velocity tracking for passive zone people (for almost-engaged detection)
        # person_id -> list of (timestamp, x, z) recent positions
        self.passive_velocity_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self.velocity_history_max_age: float = 2.0  # Keep 2 seconds of history
        
        # Zone boundaries (should match lightController)
        self.active_zone_bounds = {
            'x_min': -117.5, 'x_max': 357.5,  # From TRACKZONE
            'z_min': 78, 'z_max': 283,
        }
        self.passive_zone_bounds = {
            'x_min': -205, 'x_max': 445,
            'z_min': 283, 'z_max': 613,
        }
        
        # Wander box (can be modified by behavior)
        # X range covers panels (Unit 0 at X=-40 to Unit 3 at X=-280)
        # Z range kept close to panels for good illumination
        self.base_wander_box = {
            'min_x': -290, 'max_x': -30,
            'min_y': 0, 'max_y': 150,
            'min_z': -32, 'max_z': 28,  # Stay close to panels
        }
        self.current_wander_box = dict(self.base_wander_box)
        
        # Animated wander box - smoothly transitions between base and engaged boxes
        self.animated_wander_box = dict(self.base_wander_box)
        self.target_wander_box = dict(self.base_wander_box)
        self.wander_box_lerp_speed = 3.0  # How fast box animates (higher = faster, more responsive)
        
        # Engaged box settings (how tight the box contracts around people)
        # X is very tight so light stays nearly centered on person
        self.engaged_box_padding_x = 15   # +/- 15cm in X - very tight!
        self.engaged_box_padding_y = 35   # +/- Y padding for some vertical movement
        self.engaged_box_padding_z = 15   # +/- Z padding
        self.engaged_box_y_offset = 100   # Offset box upward from person height
        
        # Approach settings (how far forward the light moves toward people)
        # Keep conservative - too far = dim panels
        self.approach_z_min = -32    # Closest to panels  
        self.approach_z_max = 60     # Max approach (still illuminates panels well)
        self.approach_speed = 0.1    # How quickly to approach (0-1 per second)
        
        # People tracking
        self.known_people: Dict[int, float] = {}  # id -> first_seen_time
        self.people_positions: Dict[int, np.ndarray] = {}  # id -> position (all people)
        self.active_zone_people: Dict[int, np.ndarray] = {}  # id -> position (active zone only)
        self.active_zone_dwell: Dict[int, float] = {}  # id -> time entered active zone
        
        # Dwell phases and their effects
        # Phase 0: Notice (0-3s) - Light acknowledges, moves toward
        # Phase 1: Greet (3-10s) - Light settles, brightness increases
        # Phase 2: Engage (10-30s) - Deeper connection, tighter tracking
        # Phase 3: Bond (30s+) - Special behaviors, maximum intimacy
        self.DWELL_PHASES = {
            'notice': (0, 3),      # Just noticed
            'greet': (3, 10),      # Greeting phase
            'engage': (10, 30),    # Active engagement
            'bond': (30, float('inf')),  # Deep connection
        }
        
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
    
    def calculate_proximity_factor(self, z_position: float) -> float:
        """
        Calculate proximity factor based on Z position.
        Returns 0-1 where 1 = very close to panels, 0 = far from panels.
        
        Z coordinates in V2 system:
        - Lower Z = closer to camera/panels
        - Higher Z = farther from panels (toward sidewalk)
        """
        if z_position <= self.PROXIMITY_Z_NEAR:
            return 1.0
        elif z_position >= self.PROXIMITY_Z_FAR:
            return 0.0
        else:
            # Linear interpolation
            return 1.0 - (z_position - self.PROXIMITY_Z_NEAR) / (self.PROXIMITY_Z_FAR - self.PROXIMITY_Z_NEAR)
    
    def update_proximity(self):
        """
        Update proximity tracking based on nearest person position.
        Called during update() to set state.proximity_factor.
        """
        if not self.people_positions:
            self.state.nearest_person_z = 500.0
            self.state.proximity_factor = 0.0
            return
        
        # Find nearest person (smallest Z = closest to panels)
        nearest_z = min(pos[2] for pos in self.people_positions.values())
        self.state.nearest_person_z = nearest_z
        self.state.proximity_factor = self.calculate_proximity_factor(nearest_z)
    
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

    def trigger_entry_pulse(self):
        """
        Trigger the entry acknowledgment pulse.
        This creates a visible brightness surge when someone enters the active zone.
        """
        self.state.entry_pulse_active = True
        self.state.entry_pulse_start = time.time()
        self.state.entry_pulse_duration = self.ENTRY_PULSE_DURATION
    
    def _update_entry_pulse(self, now: float) -> float:
        """
        Update entry pulse state and return current pulse intensity (0-1).
        Uses a quick ramp-up and slower decay for a "flash" effect.
        """
        if not self.state.entry_pulse_active:
            return 0.0
        
        elapsed = now - self.state.entry_pulse_start
        if elapsed >= self.state.entry_pulse_duration:
            self.state.entry_pulse_active = False
            return 0.0
        
        # Quick ramp up (first 20%), slow decay (last 80%)
        ramp_point = 0.2
        if elapsed < self.state.entry_pulse_duration * ramp_point:
            # Ramp up phase
            return elapsed / (self.state.entry_pulse_duration * ramp_point)
        else:
            # Decay phase
            decay_elapsed = elapsed - self.state.entry_pulse_duration * ramp_point
            decay_duration = self.state.entry_pulse_duration * (1.0 - ramp_point)
            return 1.0 - (decay_elapsed / decay_duration)
    
    def on_person_entered(self, person_id: int, position: np.ndarray, is_active_zone: bool):
        """Called when a new person is detected"""
        self.known_people[person_id] = time.time()
        self.people_positions[person_id] = position
        self.state.last_interaction_time = time.time()
        self.state.is_bored = False
        
        # Check if this was an almost-engaged person who converted!
        if is_active_zone:
            self.check_almost_engaged_conversion(person_id)
            # Log feedback for engagement learning
            self.on_engagement(person_id, "active")
        
        # Entrance gesture and pulse for active zone
        if is_active_zone and self.meta.entrance_flash_enabled:
            # ALWAYS trigger the entry pulse for visibility
            self.trigger_entry_pulse()
            
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
        if person_id in self.active_zone_people:
            del self.active_zone_people[person_id]
        if person_id in self.active_zone_dwell:
            del self.active_zone_dwell[person_id]
    
    def update_person_position(self, person_id: int, position: np.ndarray):
        """Update tracked person position"""
        self.people_positions[person_id] = position
        
        # Determine zone for velocity tracking
        x, z = position[0], position[2]
        az = self.active_zone_bounds
        pz = self.passive_zone_bounds
        
        if pz['x_min'] <= x <= pz['x_max'] and pz['z_min'] <= z <= pz['z_max']:
            zone = 'passive'
        elif az['x_min'] <= x <= az['x_max'] and az['z_min'] <= z <= az['z_max']:
            zone = 'active'
        else:
            zone = 'unknown'
        
        # Track velocity for almost-engaged detection
        self.update_passive_velocity(person_id, x, z, zone)
    
    def set_person_active(self, person_id: int, is_active: bool, position: np.ndarray):
        """Update whether a person is in the active zone"""
        now = time.time()
        if is_active:
            was_already_active = person_id in self.active_zone_people
            self.active_zone_people[person_id] = position
            # Start dwell timer if not already tracking
            if person_id not in self.active_zone_dwell:
                self.active_zone_dwell[person_id] = now
            # Check if this was an almost-engaged person who converted!
            self.check_almost_engaged_conversion(person_id)
            # Log feedback for engagement learning (only if newly active)
            if not was_already_active:
                self.on_engagement(person_id, "active")
            self.check_almost_engaged_conversion(person_id)
        else:
            if person_id in self.active_zone_people:
                del self.active_zone_people[person_id]
            if person_id in self.active_zone_dwell:
                del self.active_zone_dwell[person_id]
    
    def get_dwell_phase(self, person_id: int) -> str:
        """Get the dwell phase for a specific person"""
        if person_id not in self.active_zone_dwell:
            return 'none'
        
        dwell_time = time.time() - self.active_zone_dwell[person_id]
        
        for phase, (min_t, max_t) in self.DWELL_PHASES.items():
            if min_t <= dwell_time < max_t:
                return phase
        return 'bond'  # Default to deepest phase
    
    def get_primary_person(self) -> Optional[Tuple[int, np.ndarray, float]]:
        """Get the primary person to focus on.
        
        Priority:
        1. Longest dwell time (most engaged)
        2. Tie-breaker: closest to panels (smallest Z)
        
        Returns: (person_id, position, dwell_time) or None
        """
        if not self.active_zone_people:
            return None
        
        now = time.time()
        candidates = []
        
        for pid, pos in self.active_zone_people.items():
            dwell_time = now - self.active_zone_dwell.get(pid, now)
            candidates.append((pid, pos, dwell_time))
        
        # Sort by dwell time (descending), then by Z (ascending = closer to panels)
        candidates.sort(key=lambda x: (-x[2], x[1][2]))
        
        return candidates[0] if candidates else None
    
    def get_active_zone_stats(self) -> Dict:
        """Get statistics about active zone occupancy"""
        now = time.time()
        
        if not self.active_zone_people:
            return {
                'count': 0,
                'primary_id': None,
                'primary_dwell': 0,
                'primary_phase': 'none',
                'total_dwell': 0,
                'avg_dwell': 0,
            }
        
        # Get all dwell times
        dwell_times = [
            now - self.active_zone_dwell.get(pid, now)
            for pid in self.active_zone_people
        ]
        
        primary = self.get_primary_person()
        primary_phase = self.get_dwell_phase(primary[0]) if primary else 'none'
        
        return {
            'count': len(self.active_zone_people),
            'primary_id': primary[0] if primary else None,
            'primary_dwell': primary[2] if primary else 0,
            'primary_phase': primary_phase,
            'total_dwell': sum(dwell_times),
            'avg_dwell': sum(dwell_times) / len(dwell_times) if dwell_times else 0,
        }
    
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
        # Higher responsiveness = higher follow_smoothing = faster following
        # Base range 0.03-0.15, scaled by responsiveness
        base_smoothing = result.get('follow_smoothing', 0.05)
        if base_smoothing > 0:  # Only modify if following is enabled
            result['follow_smoothing'] = m.lerp(0.03, 0.20, m.responsiveness) * m.follow_speed_global
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
        """Apply dwell-based parameter adjustments.
        
        Dwell phases affect:
        - Box tightness (how closely light tracks)
        - Brightness (increases with engagement)
        - Movement speed (slower = more intimate)
        - Wander interval (longer pauses = more focused)
        
        All effects are scaled by meta.dwell_influence (0=none, 1=normal, 2=double)
        """
        if not self.meta.dwell_rewards_enabled:
            return params
        
        influence = self.meta.dwell_influence
        if influence <= 0:
            self.state.current_dwell_bonus = 0
            return params
        
        stats = self.get_active_zone_stats()
        if stats['count'] == 0:
            self.state.current_dwell_bonus = 0
            return params
        
        phase = stats['primary_phase']
        dwell_time = stats['primary_dwell']
        count = stats['count']
        
        result = dict(params)
        
        # Phase-based adjustments (all scaled by influence)
        if phase == 'notice':  # 0-3s
            # Just noticed - quick acknowledgment
            speed_boost = 1.0 + 0.2 * influence  # 1.2 at influence=1
            result['move_speed'] = result.get('move_speed', 25) * speed_boost
            self.state.current_dwell_bonus = 0
            
        elif phase == 'greet':  # 3-10s
            # Greeting - settling in
            brightness_add = 5 * influence
            interval_add = 1 * influence
            result['brightness_max'] = result.get('brightness_max', 30) + brightness_add
            result['wander_interval'] = result.get('wander_interval', 4.0) + interval_add
            self.state.current_dwell_bonus = 5 * influence
            
        elif phase == 'engage':  # 10-30s
            # Active engagement - deeper connection
            base_bonus = min(10, int((dwell_time - 10) / 5) * 2)  # +2 per 5s, max +10
            bonus = base_bonus * influence
            result['brightness_max'] = result.get('brightness_max', 30) + bonus
            result['brightness_min'] = result.get('brightness_min', 8) + bonus * 0.5
            speed_mult = 1.0 - 0.2 * influence  # 0.8 at influence=1
            result['move_speed'] = result.get('move_speed', 25) * max(0.4, speed_mult)
            result['wander_interval'] = result.get('wander_interval', 4.0) + 2 * influence
            self.state.current_dwell_bonus = bonus
            
        elif phase == 'bond':  # 30s+
            # Deep bond - maximum intimacy
            bonus = 15 * influence
            result['brightness_max'] = result.get('brightness_max', 30) + bonus
            result['brightness_min'] = result.get('brightness_min', 8) + 8 * influence
            speed_mult = 1.0 - 0.4 * influence  # 0.6 at influence=1
            result['move_speed'] = result.get('move_speed', 25) * max(0.3, speed_mult)
            result['wander_interval'] = result.get('wander_interval', 4.0) + 4 * influence
            pulse_mult = 1.0 + 0.3 * influence  # 1.3 at influence=1
            result['pulse_speed'] = result.get('pulse_speed', 2500) * pulse_mult
            self.state.current_dwell_bonus = bonus
            self.state.current_dwell_bonus = 15
        
        # Multi-person adjustments
        if count >= 2:
            # With 2+ people, be slightly more energetic
            result['move_speed'] = result.get('move_speed', 25) * (1 + 0.1 * min(count, 4))
            result['brightness_max'] = result.get('brightness_max', 30) * (1 + 0.05 * count)
            result['wander_interval'] = result.get('wander_interval', 4.0) * 0.8  # Move between people
        
        return result
    
    def apply_anti_repetition(self, params: Dict, current_pos: Tuple[float, float, float]) -> Dict:
        """Apply modifications to avoid repetitive behavior.
        Uses cached values to avoid querying database every frame.
        """
        if not self.meta.self_analysis_enabled or not self.database:
            return params
        
        weight = self.meta.anti_repetition_weight * self.meta.memory
        if weight < 0.1:
            return params
        
        result = dict(params)
        now = time.time()
        
        # Cache anti-repetition values (update every 10 seconds, not every frame)
        if not hasattr(self, '_anti_rep_cache') or now - self._anti_rep_cache.get('time', 0) > 10.0:
            # Query in background would be ideal, but for now just throttle
            try:
                entropy = self.database.get_position_entropy(60)
                people_count = len(self.people_positions)
                similarity = self.database.get_response_similarity(people_count)
                self._anti_rep_cache = {
                    'time': now,
                    'entropy': entropy,
                    'similarity': similarity,
                }
            except Exception:
                # On error, use neutral values
                self._anti_rep_cache = {
                    'time': now,
                    'entropy': 0.5,
                    'similarity': 0.5,
                }
        
        entropy = self._anti_rep_cache.get('entropy', 0.5)
        similarity = self._anti_rep_cache.get('similarity', 0.5)
        
        # Check position entropy
        if entropy < 0.3:
            # Low entropy = too repetitive, encourage exploration
            result['wander_interval'] = result.get('wander_interval', 3.0) * (1 - 0.3 * weight)
            result['move_speed'] = result['move_speed'] * (1 + 0.2 * weight)
        
        # Check response similarity
        if similarity > 0.7:
            # Too similar to past responses, add variety
            result['pulse_speed'] = result['pulse_speed'] * random.uniform(0.85, 1.15)
            result['brightness_max'] = result['brightness_max'] * random.uniform(0.9, 1.1)
        
        return result
    
    # =========================================================================
    # IDLE TREND ANALYSIS
    # =========================================================================
    
    # Activity level thresholds (passive people per time window)
    ACTIVITY_THRESHOLDS = {
        'dead': 0,      # No one around
        'quiet': 3,     # A few people
        'moderate': 10, # Regular foot traffic
        'busy': 25,     # Heavy traffic
        'rush': 50,     # Peak times
    }
    
    def update_idle_trends(self) -> IdleTrends:
        """
        Non-blocking trends update. Checks for completed background work,
        then starts a new background query if needed.
        Returns the current (possibly stale) trends immediately.
        """
        now = time.time()
        
        # Check if background thread completed with new data
        with self._trends_lock:
            if self._pending_trends is not None:
                self.state.idle_trends = self._pending_trends
                self._pending_trends = None
                self.state.last_trend_update = now
        
        # If no background thread running and it's time to update, start one
        if self._trends_thread is None or not self._trends_thread.is_alive():
            if now - self.state.last_trend_update > 5.0:  # Update every 5 seconds
                self._trends_thread = threading.Thread(
                    target=self._background_trends_query,
                    daemon=True
                )
                self._trends_thread.start()
        
        # Return current trends (may be from previous update)
        return self.state.idle_trends if self.state.idle_trends else IdleTrends(last_update=now)
    
    def _background_trends_query(self):
        """
        Run expensive database queries in background thread.
        Uses a separate read-only database connection to avoid blocking main thread.
        Stores result in _pending_trends for main thread to pick up.
        """
        now = time.time()
        trends = IdleTrends(last_update=now)
        
        if not self.database:
            trends.database_error = "No database connected"
            with self._trends_lock:
                self._pending_trends = trends
            return
        
        # Get current time period
        period = TimePeriod.current()
        trends.period_name = period.value
        
        try:
            # Create a separate read-only connection for this thread
            # This avoids blocking the main thread's database operations
            import sqlite3
            bg_conn = sqlite3.connect(str(self.database.db_path), check_same_thread=False)
            bg_conn.row_factory = sqlite3.Row
            cursor = bg_conn.cursor()
            
            # Recent (1 minute) - for immediate response
            one_minute_ago = now - 60
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) as people,
                       AVG(speed) as avg_speed,
                       SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active_events,
                       SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive_events,
                       SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                       SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                FROM tracking_events
                WHERE timestamp > ?
            ''', (one_minute_ago,))
            row = cursor.fetchone()
            if row:
                trends.recent_passive_count = row['passive_events'] or 0
                trends.recent_active_count = row['active_events'] or 0
                trends.recent_avg_speed = row['avg_speed'] or 0.0
                ltr = row['ltr'] or 0
                rtl = row['rtl'] or 0
                if ltr + rtl > 0:
                    trends.recent_flow_direction = (ltr - rtl) / (ltr + rtl)
                trends.has_recent_data = (trends.recent_passive_count > 0 or trends.recent_active_count > 0)
            
            # Short term (5 minutes) - simplified query
            five_min_ago = now - 300
            cursor.execute('''
                SELECT SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive,
                       SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active,
                       SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                       SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                FROM tracking_events WHERE timestamp > ?
            ''', (five_min_ago,))
            row = cursor.fetchone()
            if row:
                trends.short_passive_count = row['passive'] or 0
                trends.short_active_count = row['active'] or 0
                ltr, rtl = row['ltr'] or 0, row['rtl'] or 0
                if ltr + rtl > 0:
                    trends.short_flow_direction = (ltr - rtl) / (ltr + rtl)
                total_short = trends.short_passive_count + trends.short_active_count
                trends.short_activity_level = min(1.0, total_short / 50.0)
                trends.has_short_data = (total_short > 0)
            
            # Medium term (30 minutes) - simplified query
            thirty_min_ago = now - 1800
            cursor.execute('''
                SELECT SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive,
                       SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active,
                       SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                       SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                FROM tracking_events WHERE timestamp > ?
            ''', (thirty_min_ago,))
            row = cursor.fetchone()
            if row:
                trends.medium_passive_count = row['passive'] or 0
                trends.medium_active_count = row['active'] or 0
                ltr, rtl = row['ltr'] or 0, row['rtl'] or 0
                if ltr + rtl > 0:
                    trends.medium_flow_direction = (ltr - rtl) / (ltr + rtl)
                total_medium = trends.medium_passive_count + trends.medium_active_count
                trends.medium_activity_level = min(1.0, total_medium / 200.0)
                trends.has_medium_data = (total_medium > 0)
            
            # Skip expensive historical queries for performance
            # They take 2+ seconds and aren't critical for real-time behavior
            trends.has_historical_data = False
            trends.period_typical_count = 0
            trends.period_typical_flow = 0.0
            
            # Compute derived values (only with available data)
            trends.activity_anticipation = self._compute_activity_anticipation(trends)
            trends.flow_momentum = self._compute_flow_momentum(trends)
            trends.energy_level = self._compute_energy_level(trends)
            
            # Close the background connection
            bg_conn.close()
            
        except Exception as e:
            # Database query failed, store error
            trends.database_error = str(e)
        
        # Store result for main thread to pick up
        with self._trends_lock:
            self._pending_trends = trends
    
    def _compute_activity_anticipation(self, trends: IdleTrends) -> float:
        """
        Compute how ready we should be for action based on trend patterns.
        Returns 0.0 (expect quiet) to 1.0 (expect busy).
        """
        # Weight recent data more heavily, but consider longer patterns
        recent_weight = 0.4
        short_weight = 0.3
        medium_weight = 0.2
        historical_weight = 0.1
        
        recent = min(1.0, trends.recent_passive_count / 5.0)  # 5+ in a minute = very active
        short = trends.short_activity_level
        medium = trends.medium_activity_level
        
        # Compare current to historical - if we're below typical, might pick up
        historical = 0.5
        if trends.period_typical_count > 0:
            # If current is below typical, anticipate increase
            current_vs_typical = trends.medium_passive_count / max(1, trends.period_typical_count * 30)
            if current_vs_typical < 0.5:
                historical = 0.7  # Below typical, likely to increase
            elif current_vs_typical > 1.5:
                historical = 0.3  # Above typical, might slow down
            else:
                historical = 0.5  # Around typical
        
        return (
            recent * recent_weight +
            short * short_weight +
            medium * medium_weight +
            historical * historical_weight
        )
    
    def _compute_flow_momentum(self, trends: IdleTrends) -> float:
        """
        Compute sustained directional flow momentum.
        Returns -1.0 (strong right-to-left) to +1.0 (strong left-to-right).
        Only significant if flow has been consistent across time windows.
        """
        # If all time windows agree on direction, that's strong momentum
        directions = [
            trends.recent_flow_direction,
            trends.short_flow_direction,
            trends.medium_flow_direction,
        ]
        
        # Check if all have same sign (consistent direction)
        if all(d >= 0 for d in directions) or all(d <= 0 for d in directions):
            # Consistent direction - average with recent weighted higher
            return (
                trends.recent_flow_direction * 0.5 +
                trends.short_flow_direction * 0.3 +
                trends.medium_flow_direction * 0.2
            )
        else:
            # Mixed directions - dampen momentum
            return trends.short_flow_direction * 0.5
    
    def _compute_energy_level(self, trends: IdleTrends) -> float:
        """
        Compute overall energy level to match.
        Based on activity and speed of passersby.
        Returns 0.0 (very low energy) to 1.0 (high energy).
        """
        # Base energy on activity
        base_energy = (trends.short_activity_level + trends.medium_activity_level) / 2
        
        # Speed influences energy - fast walkers = higher energy
        # Average walking speed is about 120 cm/s
        speed_factor = min(1.0, trends.recent_avg_speed / 150.0) if trends.recent_avg_speed > 0 else 0.5
        
        return base_energy * 0.7 + speed_factor * 0.3
    
    def apply_idle_trends(self, params: Dict) -> Dict:
        """
        Modify IDLE mode parameters based on passive zone trends.
        This makes the light responsive to foot traffic patterns even when
        no one is in the active zone.
        """
        if self.state.mode != BehaviorMode.IDLE:
            return params
        
        trends = self.state.idle_trends
        if not trends:
            return params
        
        weight = self.meta.idle_trend_weight
        if weight < 0.1:
            return params
        
        result = dict(params)
        
        # ======================
        # ACTIVITY ANTICIPATION
        # ======================
        # High anticipation = be more alert (faster, brighter, ready)
        anticipation = trends.activity_anticipation * weight
        
        if anticipation > 0.6:
            # Busy or getting busy - be more alert
            result['wander_interval'] = result.get('wander_interval', 5.0) * (1.0 - 0.3 * anticipation)
            result['brightness_max'] = result.get('brightness_max', 15) * (1.0 + 0.3 * anticipation)
            result['brightness_min'] = result.get('brightness_min', 3) * (1.0 + 0.2 * anticipation)
        elif anticipation < 0.2:
            # Very quiet - be more subdued
            result['wander_interval'] = result.get('wander_interval', 5.0) * (1.0 + 0.5 * (0.2 - anticipation))
            result['brightness_max'] = result.get('brightness_max', 15) * 0.8
            result['pulse_speed'] = result.get('pulse_speed', 4000) * 1.3  # Slower pulse
        
        # ======================
        # FLOW MOMENTUM
        # ======================
        # Sustained directional flow - bias wander box toward that direction
        momentum = trends.flow_momentum * weight
        
        if abs(momentum) > 0.3:
            # Significant flow momentum - shift wander box to anticipate
            # Positive momentum = left-to-right flow = shift box rightward (more negative X)
            flow_shift = momentum * 40  # Up to 40cm shift
            self.current_wander_box['min_x'] = self.base_wander_box['min_x'] - flow_shift
            self.current_wander_box['max_x'] = self.base_wander_box['max_x'] - flow_shift
        
        # ======================
        # ENERGY MATCHING
        # ======================
        # Match overall energy to the environment
        energy = trends.energy_level * weight
        
        # Speed scales with energy
        energy_speed_mult = 0.7 + (energy * 0.6)  # 0.7x to 1.3x
        result['move_speed'] = result.get('move_speed', 20) * energy_speed_mult
        
        # Pulse speed inversely - high energy = faster pulse (lower value)
        energy_pulse_mult = 1.3 - (energy * 0.6)  # 1.3x to 0.7x
        result['pulse_speed'] = result.get('pulse_speed', 4000) * energy_pulse_mult
        
        # ======================
        # IMMEDIATE PASSIVE RESPONSE
        # ======================
        # React to immediate passive zone activity
        if trends.recent_passive_count > 0:
            # Someone walking by right now - be more alert
            immediate_boost = min(1.0, trends.recent_passive_count / 3.0) * weight
            result['brightness_max'] = result.get('brightness_max', 15) * (1.0 + 0.2 * immediate_boost)
            
            # Bias toward the flow direction
            if abs(trends.recent_flow_direction) > 0.3:
                immediate_shift = trends.recent_flow_direction * 25 * weight
                self.current_wander_box['min_x'] = self.current_wander_box.get('min_x', -290) - immediate_shift
                self.current_wander_box['max_x'] = self.current_wander_box.get('max_x', -30) - immediate_shift
        
        return result

    def update_aggression(self, dt: float, passive_count: int, active_count: int):
        """
        Update aggression level based on engagement patterns.
        Called every frame (or periodically) to smoothly adjust aggression.
        
        Aggression rises when:
        - Time passes without anyone in active zone
        - People pass through passive zone without engaging
        
        Aggression falls when:
        - Someone is currently in active zone
        - Recent engagements have occurred
        """
        agg = self.state.aggression
        if not agg:
            return
        
        now = time.time()
        agg.last_update = now
        
        # Track if someone is currently engaging
        agg.current_engagement = active_count > 0
        
        # Update time since last engagement
        if agg.current_engagement:
            agg.seconds_since_engagement = 0.0
            agg.last_engagement_time = now
            # Count this as a recent engagement
            agg.recent_engagements = min(10, agg.recent_engagements + 1)
            # Reset the passive without conversion counter
            agg.passive_without_conversion = 0
        else:
            agg.seconds_since_engagement = now - agg.last_engagement_time
            # Decay recent engagements over time (halve every 5 minutes)
            decay_rate = 0.693 / 300.0  # Half-life of 5 minutes
            agg.recent_engagements = max(0, agg.recent_engagements - decay_rate * dt)
        
        # Track passive zone visits without conversion
        if passive_count > agg.last_passive_count and not agg.current_engagement:
            # New passive zone activity without current engagement
            agg.passive_without_conversion += (passive_count - agg.last_passive_count)
        agg.last_passive_count = passive_count
        
        # Update time-of-day cap
        current_hour = datetime.now().hour
        agg.time_of_day_cap = AGGRESSION_TIME_CAPS.get(current_hour, 0.5)
        
        # Calculate raw aggression level
        # Component 1: Time factor - rises with time since engagement
        # Reaches 0.4 after 10 minutes, 0.7 after 30 minutes
        time_factor = min(0.8, agg.seconds_since_engagement / 2400.0)  # Max 0.8 after 40 min
        
        # Component 2: Conversion failure - rises when people pass without engaging
        # Each unconverted passive visit adds a small amount
        conversion_factor = min(0.5, agg.passive_without_conversion / 100.0)  # Max 0.5 after 100 passes
        
        # Component 3: Success factor - recent engagements reduce aggression
        # More recent engagements = lower aggression
        success_factor = max(0.3, 1.0 - agg.recent_engagements * 0.15)  # Min 0.3 at 4+ engagements
        
        # Combine factors:
        # Base aggression from time and conversion, scaled by success
        raw_aggression = (time_factor * 0.5 + conversion_factor * 0.5) * success_factor
        
        # If currently engaged, drop aggression immediately
        if agg.current_engagement:
            raw_aggression *= 0.3  # Very low aggression when engaged
        
        agg.raw_level = raw_aggression
        
        # Apply time-of-day cap
        capped_aggression = min(raw_aggression, agg.time_of_day_cap)
        
        # EMA smoothing for smooth transitions
        if agg.level is None:
            agg.level = capped_aggression
        else:
            agg.level += agg.ema_alpha * (capped_aggression - agg.level)
    
    def update_flow(self, dt: float):
        """
        Update real-time flow tracking for anticipatory positioning.
        
        Uses a 30-second window (more responsive than 1-minute trends).
        Called periodically to update flow direction with EMA smoothing.
        Uses background thread to avoid blocking main loop.
        """
        flow = self.state.flow
        if not flow or not self.database:
            return
        
        now = time.time()
        
        # Check for pending flow results from background thread
        if hasattr(self, '_pending_flow') and self._pending_flow is not None:
            with self._flow_lock:
                pending = self._pending_flow
                self._pending_flow = None
            
            if pending:
                flow.left_to_right_count = pending.get('ltr', 0)
                flow.right_to_left_count = pending.get('rtl', 0)
                flow.total_events = pending.get('total', 0)
                
                # Calculate raw flow direction (-1 to +1)
                total_directional = flow.left_to_right_count + flow.right_to_left_count
                if total_directional > 0:
                    raw_direction = (flow.left_to_right_count - flow.right_to_left_count) / total_directional
                else:
                    raw_direction = 0.0
                
                flow.raw_direction = raw_direction
                
                # Calculate flow strength (confidence in direction)
                if flow.total_events >= 3:
                    flow.strength = min(1.0, abs(raw_direction) * (flow.total_events / 10.0))
                else:
                    flow.strength = 0.0
                
                # EMA smoothing for direction
                flow.direction += flow.ema_alpha * (raw_direction - flow.direction)
                
                # Calculate X offset for wander box positioning
                max_shift = 60.0
                if flow.strength > 0.2:
                    flow.x_offset = -flow.direction * flow.strength * max_shift
                else:
                    flow.x_offset *= 0.9
        
        # Only start new query at the specified interval
        if now - flow.last_update < flow.update_interval:
            return
        
        # Check if a query is already running
        if hasattr(self, '_flow_thread') and self._flow_thread is not None and self._flow_thread.is_alive():
            return
        
        flow.last_update = now
        
        # Initialize flow lock if needed
        if not hasattr(self, '_flow_lock'):
            self._flow_lock = threading.Lock()
            self._pending_flow = None
        
        # Start background query
        self._flow_thread = threading.Thread(target=self._background_flow_query, daemon=True)
        self._flow_thread.start()
    
    def _background_flow_query(self):
        """Run flow query in background thread with separate connection."""
        try:
            import sqlite3
            bg_conn = sqlite3.connect(str(self.database.db_path), check_same_thread=False)
            bg_conn.row_factory = sqlite3.Row
            cursor = bg_conn.cursor()
            
            thirty_seconds_ago = time.time() - 30
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                    SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl,
                    COUNT(*) as total
                FROM tracking_events
                WHERE timestamp > ? AND zone = 'passive'
            ''', (thirty_seconds_ago,))
            
            row = cursor.fetchone()
            bg_conn.close()
            
            if row:
                with self._flow_lock:
                    self._pending_flow = {
                        'ltr': row['ltr'] or 0,
                        'rtl': row['rtl'] or 0,
                        'total': row['total'] or 0
                    }
        except Exception:
            pass  # Keep previous values on error

    def apply_flow_positioning(self, params: Dict) -> Dict:
        """
        Apply flow-responsive wander box positioning.
        
        When there's clear directional flow:
        - Shift wander box toward incoming traffic direction
        - Creates "anticipatory" positioning that catches people's attention
        
        Only applies in IDLE mode (when fishing for attention).
        """
        flow = self.state.flow
        if not flow:
            return params
        
        result = dict(params)
        
        # Only apply in IDLE mode
        if self.state.mode != BehaviorMode.IDLE:
            return result
        
        # Apply X offset to wander box
        if abs(flow.x_offset) > 1.0:  # More than 1cm offset
            # Shift the entire wander box
            self.current_wander_box['min_x'] = self.base_wander_box['min_x'] + flow.x_offset
            self.current_wander_box['max_x'] = self.base_wander_box['max_x'] + flow.x_offset
            
            # Clamp to reasonable bounds (don't go off the panels)
            self.current_wander_box['min_x'] = max(-350, self.current_wander_box['min_x'])
            self.current_wander_box['max_x'] = min(30, self.current_wander_box['max_x'])
        
        return result

    # =========================================================================
    # ALMOST-ENGAGED DETECTION (Phase 2C)
    # =========================================================================
    
    def update_passive_velocity(self, person_id: int, x: float, z: float, zone: str):
        """
        Track velocity for people in passive zone.
        Called when position updates come in from the tracker.
        
        Args:
            person_id: Track ID
            x, z: Current position in cm
            zone: 'active', 'passive', or 'unknown'
        """
        now = time.time()
        
        # Only track passive zone people
        if zone != 'passive':
            # Clean up if they left passive zone
            if person_id in self.passive_velocity_history:
                del self.passive_velocity_history[person_id]
            return
        
        # Initialize history for new person
        if person_id not in self.passive_velocity_history:
            self.passive_velocity_history[person_id] = []
        
        # Add current position
        history = self.passive_velocity_history[person_id]
        history.append((now, x, z))
        
        # Clean up old entries
        cutoff = now - self.velocity_history_max_age
        self.passive_velocity_history[person_id] = [
            (t, px, pz) for t, px, pz in history if t > cutoff
        ]
    
    def calculate_person_speed(self, person_id: int) -> float:
        """
        Calculate current speed for a person based on recent position history.
        Returns speed in cm/s, or -1 if not enough data.
        """
        if person_id not in self.passive_velocity_history:
            return -1.0
        
        history = self.passive_velocity_history[person_id]
        if len(history) < 2:
            return -1.0
        
        # Use first and last points for average speed
        t1, x1, z1 = history[0]
        t2, x2, z2 = history[-1]
        
        dt = t2 - t1
        if dt < 0.1:  # Need at least 100ms of data
            return -1.0
        
        dx = x2 - x1
        dz = z2 - z1
        distance = math.sqrt(dx*dx + dz*dz)
        
        return distance / dt
    
    def distance_to_active_zone(self, x: float, z: float) -> float:
        """
        Calculate distance from a point to the nearest active zone edge.
        Returns distance in cm. Negative = inside active zone.
        """
        az = self.active_zone_bounds
        
        # Check if inside active zone
        inside_x = az['x_min'] <= x <= az['x_max']
        inside_z = az['z_min'] <= z <= az['z_max']
        
        if inside_x and inside_z:
            return -1.0  # Inside active zone
        
        # Distance to each edge
        dx_min = az['x_min'] - x if x < az['x_min'] else 0
        dx_max = x - az['x_max'] if x > az['x_max'] else 0
        dz_min = az['z_min'] - z if z < az['z_min'] else 0
        dz_max = z - az['z_max'] if z > az['z_max'] else 0
        
        # Closest distance (Euclidean to nearest edge/corner)
        dx = max(dx_min, dx_max)
        dz = max(dz_min, dz_max)
        
        if dx > 0 and dz > 0:
            return math.sqrt(dx*dx + dz*dz)  # Corner distance
        else:
            return max(dx, dz)  # Edge distance
    
    def update_almost_engaged(self, dt: float):
        """
        Update almost-engaged detection and attraction.
        
        Detects people in passive zone who:
        1. Are near the active zone boundary
        2. Have slowed down significantly
        
        Then tries attraction strategies and tracks outcomes.
        """
        state = self.state.almost_engaged
        if not state:
            return
        
        now = time.time()
        state.last_update = now
        
        # Clean up stale candidates (not seen in 2 seconds)
        stale_ids = []
        for pid, candidate in state.candidates.items():
            if now - candidate.last_seen > 2.0:
                stale_ids.append(pid)
                # Log outcome if not already done
                if not candidate.outcome_logged:
                    self._log_almost_engaged_outcome(candidate, converted=False)
        
        for pid in stale_ids:
            del state.candidates[pid]
            if pid in self.passive_velocity_history:
                del self.passive_velocity_history[pid]
        
        # Check each person in passive zone for almost-engaged behavior
        for person_id, history in list(self.passive_velocity_history.items()):
            if len(history) < 2:
                continue
            
            # Get current position and speed
            _, x, z = history[-1]
            speed = self.calculate_person_speed(person_id)
            if speed < 0:
                continue
            
            # Check distance to active zone
            dist_to_active = self.distance_to_active_zone(x, z)
            
            # Check if this person qualifies as "almost engaged"
            is_slow = speed < state.slow_speed_threshold
            is_near = 0 < dist_to_active < state.near_active_threshold
            
            if is_slow and is_near:
                # Track this candidate
                if person_id not in state.candidates:
                    # New almost-engaged candidate
                    state.candidates[person_id] = AlmostEngagedCandidate(
                        person_id=person_id,
                        first_detected=now,
                        last_seen=now,
                        position_x=x,
                        position_z=z,
                        current_speed=speed,
                        initial_speed=speed,
                        min_speed_seen=speed,
                        distance_to_active=dist_to_active,
                    )
                    state.total_detected += 1
                else:
                    # Update existing candidate
                    c = state.candidates[person_id]
                    c.last_seen = now
                    c.position_x = x
                    c.position_z = z
                    c.current_speed = speed
                    c.min_speed_seen = min(c.min_speed_seen, speed)
                    c.distance_to_active = dist_to_active
            elif person_id in state.candidates:
                # Person sped up or moved away - update last_seen anyway
                c = state.candidates[person_id]
                c.last_seen = now
                c.position_x = x
                c.position_z = z
                c.current_speed = speed
        
        # Try attraction on best candidate
        self._try_attraction(state, now)
    
    def _try_attraction(self, state: 'AlmostEngagedState', now: float):
        """
        Select and apply an attraction strategy to the best candidate.
        """
        # Check cooldown
        if now - state.last_attraction_time < state.attraction_cooldown:
            return
        
        # Find best candidate (slowest, closest to active zone, been there longest)
        best_candidate = None
        best_score = 0
        
        for pid, c in state.candidates.items():
            # Must have been detected for minimum time
            if now - c.first_detected < state.min_detection_time:
                continue
            
            # Already being attracted or already converted
            if c.converted or c.strategy_used != AttractionStrategy.NONE:
                continue
            
            # Score: lower speed + closer to active + longer detection = higher score
            speed_score = max(0, state.slow_speed_threshold - c.current_speed) / state.slow_speed_threshold
            dist_score = max(0, state.near_active_threshold - c.distance_to_active) / state.near_active_threshold
            time_score = min(1.0, (now - c.first_detected) / 3.0)  # Max at 3 seconds
            
            score = speed_score * 0.4 + dist_score * 0.4 + time_score * 0.2
            
            if score > best_score:
                best_score = score
                best_candidate = c
        
        if best_candidate is None:
            return
        
        # Select strategy (rotate through for A/B testing)
        strategy = state.strategies_to_test[state.strategy_index % len(state.strategies_to_test)]
        state.strategy_index += 1
        
        # Apply attraction
        best_candidate.strategy_used = strategy
        best_candidate.strategy_start_time = now
        state.active_attraction = True
        state.attraction_target_id = best_candidate.person_id
        state.current_strategy = strategy
        state.last_attraction_time = now
        
        # Record attempt
        state.strategy_stats[strategy.value]['attempts'] += 1
    
    def _log_almost_engaged_outcome(self, candidate: 'AlmostEngagedCandidate', converted: bool):
        """Log the outcome of an almost-engaged detection."""
        state = self.state.almost_engaged
        if not state or candidate.outcome_logged:
            return
        
        candidate.outcome_logged = True
        candidate.converted = converted
        
        if converted:
            state.total_converted += 1
            if candidate.strategy_used != AttractionStrategy.NONE:
                state.strategy_stats[candidate.strategy_used.value]['conversions'] += 1
    
    def check_almost_engaged_conversion(self, person_id: int):
        """
        Check if an almost-engaged person has converted (entered active zone).
        Called when someone enters the active zone.
        """
        state = self.state.almost_engaged
        if not state:
            return
        
        if person_id in state.candidates:
            candidate = state.candidates[person_id]
            self._log_almost_engaged_outcome(candidate, converted=True)
            
            # Clear attraction state if this was our target
            if state.attraction_target_id == person_id:
                state.active_attraction = False
                state.attraction_target_id = -1
                state.current_strategy = AttractionStrategy.NONE
            
            # Remove from candidates
            del state.candidates[person_id]
            if person_id in self.passive_velocity_history:
                del self.passive_velocity_history[person_id]
    
    def apply_almost_engaged_attraction(self, params: Dict, light_position: Tuple[float, float, float]) -> Dict:
        """
        Apply attraction modifiers when we have an almost-engaged target.
        
        Strategies:
        - BRIGHTNESS_PULSE: Increase brightness briefly
        - DRIFT_TOWARD: Move light toward the candidate
        - PAUSE_AND_LOOK: Stop wandering and focus on them
        """
        state = self.state.almost_engaged
        if not state or not state.active_attraction:
            return params
        
        # Only apply in IDLE mode
        if self.state.mode != BehaviorMode.IDLE:
            return params
        
        result = dict(params)
        target_id = state.attraction_target_id
        
        if target_id not in state.candidates:
            state.active_attraction = False
            return result
        
        candidate = state.candidates[target_id]
        strategy = state.current_strategy
        
        now = time.time()
        elapsed = now - candidate.strategy_start_time
        
        # Attraction effect duration
        effect_duration = 2.0  # 2 second effect
        if elapsed > effect_duration:
            # Effect expired
            state.active_attraction = False
            state.attraction_target_id = -1
            state.current_strategy = AttractionStrategy.NONE
            return result
        
        # Apply strategy effects
        effect_strength = 1.0 - (elapsed / effect_duration)  # Fade out
        
        if strategy == AttractionStrategy.BRIGHTNESS_PULSE:
            # Subtle brightness increase (not too aggressive)
            pulse_boost = 10 * effect_strength  # Up to +10 brightness
            result['brightness_max'] = result.get('brightness_max', 15) + pulse_boost
            result['brightness_min'] = result.get('brightness_min', 5) + pulse_boost * 0.5
            
        elif strategy == AttractionStrategy.DRIFT_TOWARD:
            # Move wander box center toward the candidate
            # The candidate is at (position_x, position_z) in world coords
            # We want to shift the wander box to be closer to them
            target_x = candidate.position_x
            
            # Calculate shift toward target (limited to 50cm)
            light_x = light_position[0]
            shift = (target_x - light_x) * 0.3 * effect_strength
            shift = max(-50, min(50, shift))
            
            # Apply shift to wander box
            self.current_wander_box['min_x'] = self.base_wander_box['min_x'] + shift
            self.current_wander_box['max_x'] = self.base_wander_box['max_x'] + shift
            
        elif strategy == AttractionStrategy.PAUSE_AND_LOOK:
            # Slow down movement and increase wander interval (pause more)
            result['move_speed'] = result.get('move_speed', 20) * 0.3  # Much slower
            result['wander_interval'] = result.get('wander_interval', 5.0) * 2.0  # Longer pauses
            # Also slight brightness increase
            result['brightness_max'] = result.get('brightness_max', 15) + 5 * effect_strength
        
        return result

    # ==========================================================================
    # PHASE 3: FEEDBACK LEARNING
    # ==========================================================================
    
    def capture_engagement_context(self, person_id: int, person_zone: str = "active") -> EngagementContext:
        """
        Capture a snapshot of current behavior state when someone engages.
        
        This creates a record of "what was I doing when this person engaged?"
        Used for learning which behavior patterns are most effective.
        """
        now = time.time()
        period = TimePeriod.current()
        
        # Normalize light position to 0-1 range
        light_x, light_y, light_z = self.current_light_position
        
        # X normalization: wander box is roughly -300 to 0, normalize to 0-1
        x_norm = (light_x + 300) / 300.0
        x_norm = max(0.0, min(1.0, x_norm))
        
        # Z normalization: -32 to 60 roughly, normalize to 0-1
        z_norm = (light_z + 32) / 92.0
        z_norm = max(0.0, min(1.0, z_norm))
        
        # Get current flow offset
        flow_x_offset = 0.0
        flow_direction = 0.0
        if self.state.flow:
            flow_x_offset = self.state.flow.x_offset
            flow_direction = self.state.flow.direction
        
        # Check if this person was an almost-engaged candidate
        was_almost = False
        strategy_used = "none"
        if self.state.almost_engaged and person_id in self.state.almost_engaged.candidates:
            was_almost = True
            candidate = self.state.almost_engaged.candidates[person_id]
            strategy_used = candidate.strategy_used.value
        
        # Current aggression level
        aggression = 0.0
        if self.state.aggression:
            aggression = self.state.aggression.level
        
        context = EngagementContext(
            timestamp=now,
            time_of_day=period.value,
            hour=datetime.now().hour,
            mode_before=self.state.mode.value,
            mode_duration=self.state.mode_duration,
            aggression_level=aggression,
            flow_direction=flow_direction,
            flow_x_offset=flow_x_offset,
            light_x_normalized=x_norm,
            light_z_normalized=z_norm,
            move_speed=self.current_wander_box.get('max_x', 0) - self.current_wander_box.get('min_x', 0),
            brightness=self.meta.brightness_global,
            intensity=self.meta.pulse_global,
            wander_x_offset=flow_x_offset,  # Same as flow offset for now
            person_zone=person_zone,
            dwell_duration=0.0,  # Will be updated when they leave
            was_almost_engaged=was_almost,
            attraction_strategy_used=strategy_used,
        )
        
        return context
    
    def log_engagement_feedback(self, context: EngagementContext):
        """
        Log the engagement context to both the internal buffer and log file.
        """
        fl = self.state.feedback_learning
        if not fl:
            return
        
        # Add to ring buffer
        fl.recent_contexts.append(context)
        if len(fl.recent_contexts) > fl.max_contexts:
            fl.recent_contexts.pop(0)
        
        # Update statistics
        fl.total_engagements += 1
        fl.engagements_by_hour[context.hour] = fl.engagements_by_hour.get(context.hour, 0) + 1
        
        # Write to log file
        try:
            log_path = f"/Users/npmac/Documents/GitHub/dc-dev/IO/V2Dev/{fl.log_file}"
            with open(log_path, 'a') as f:
                f.write(f"\n=== Engagement at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Hour: {context.hour}, Period: {context.time_of_day}\n")
                f.write(f"Mode: {context.mode_before} (duration: {context.mode_duration:.1f}s)\n")
                f.write(f"Aggression: {context.aggression_level:.2f}\n")
                f.write(f"Flow: direction={context.flow_direction:.2f}, x_offset={context.flow_x_offset:.1f}\n")
                f.write(f"Light position (normalized): x={context.light_x_normalized:.2f}, z={context.light_z_normalized:.2f}\n")
                f.write(f"Almost-engaged: {context.was_almost_engaged}, strategy: {context.attraction_strategy_used}\n")
                f.write(f"Person zone: {context.person_zone}\n")
        except Exception as e:
            print(f"[FEEDBACK] Error writing log: {e}")
    
    def update_feedback_weights(self, context: EngagementContext):
        """
        Update behavior weights based on this engagement.
        
        The learning is conservative and slow - we slightly increase the weight
        of whatever conditions were present during the engagement.
        
        Over time, this means successful patterns are weighted higher.
        """
        fl = self.state.feedback_learning
        if not fl:
            return
        
        lr = fl.learning_rate
        
        # Update aggression weights
        if context.aggression_level < 0.3:
            fl.weights['low_aggression'] = min(fl.weight_max, fl.weights['low_aggression'] + lr)
        elif context.aggression_level < 0.6:
            fl.weights['mid_aggression'] = min(fl.weight_max, fl.weights['mid_aggression'] + lr)
        else:
            fl.weights['high_aggression'] = min(fl.weight_max, fl.weights['high_aggression'] + lr)
        
        # Update position weights
        if context.light_x_normalized < 0.33:
            fl.weights['left_position'] = min(fl.weight_max, fl.weights['left_position'] + lr)
        elif context.light_x_normalized < 0.66:
            fl.weights['center_position'] = min(fl.weight_max, fl.weights['center_position'] + lr)
        else:
            fl.weights['right_position'] = min(fl.weight_max, fl.weights['right_position'] + lr)
        
        # Update flow alignment weights
        if abs(context.flow_direction) < 0.2:
            fl.weights['flow_neutral'] = min(fl.weight_max, fl.weights['flow_neutral'] + lr)
        elif context.flow_x_offset * context.flow_direction > 0:  # Aligned with flow
            fl.weights['flow_aligned'] = min(fl.weight_max, fl.weights['flow_aligned'] + lr)
        else:
            fl.weights['flow_opposed'] = min(fl.weight_max, fl.weights['flow_opposed'] + lr)
        
        # Update time-of-day weights
        period_key = context.time_of_day
        if period_key in fl.weights:
            fl.weights[period_key] = min(fl.weight_max, fl.weights[period_key] + lr)
        
        # Update mode weights
        if context.mode_before == 'idle':
            fl.weights['from_idle'] = min(fl.weight_max, fl.weights['from_idle'] + lr)
        elif context.mode_before == 'flow':
            fl.weights['from_flow'] = min(fl.weight_max, fl.weights['from_flow'] + lr)
        
        fl.last_update = time.time()
    
    def on_engagement(self, person_id: int, person_zone: str = "active"):
        """
        Called when someone engages (enters active zone or becomes active).
        Captures context, logs it, and updates weights.
        """
        context = self.capture_engagement_context(person_id, person_zone)
        self.log_engagement_feedback(context)
        self.update_feedback_weights(context)
    
    def apply_feedback_learning(self, params: Dict) -> Dict:
        """
        Apply learned weights to behavior parameters.
        
        Uses the current context to determine which weights apply,
        then modulates behavior accordingly.
        """
        fl = self.state.feedback_learning
        if not fl:
            return params
        
        result = dict(params)
        
        # Get current conditions
        aggression = 0.0
        if self.state.aggression:
            aggression = self.state.aggression.level
        
        flow_direction = 0.0
        if self.state.flow:
            flow_direction = self.state.flow.direction
        
        period = TimePeriod.current().value
        
        # Calculate composite weight based on current conditions
        weight_sum = 0.0
        count = 0
        
        # Aggression weight contribution
        if aggression < 0.3:
            weight_sum += fl.weights['low_aggression']
        elif aggression < 0.6:
            weight_sum += fl.weights['mid_aggression']
        else:
            weight_sum += fl.weights['high_aggression']
        count += 1
        
        # Time-of-day weight contribution
        if period in fl.weights:
            weight_sum += fl.weights[period]
            count += 1
        
        # Mode weight contribution
        if self.state.mode == BehaviorMode.IDLE:
            weight_sum += fl.weights['from_idle']
            count += 1
        elif self.state.mode == BehaviorMode.FLOW:
            weight_sum += fl.weights['from_flow']
            count += 1
        
        # Calculate average weight (will be around 1.0 initially)
        if count > 0:
            avg_weight = weight_sum / count
        else:
            avg_weight = 1.0
        
        # Apply weight to key parameters (subtle effect)
        # Higher weight = behavior is working, so lean into it more
        # But we keep the effect subtle to avoid wild swings
        weight_effect = 0.5 + (avg_weight * 0.5)  # Normalize to 0.5-1.5 range
        
        # Apply to sociability-influenced parameters
        if 'brightness_max' in result:
            result['brightness_max'] = result['brightness_max'] * weight_effect
        
        if 'pulse_chance' in result:
            result['pulse_chance'] = result['pulse_chance'] * weight_effect
        
        return result
    
    def get_feedback_learning_status(self) -> Dict:
        """Get current feedback learning state for visualization."""
        fl = self.state.feedback_learning
        if not fl:
            return {}
        
        # Find highest weighted behaviors
        sorted_weights = sorted(fl.weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_weights[:3]
        bottom_3 = sorted_weights[-3:]
        
        return {
            'total_engagements': fl.total_engagements,
            'session_engagements': len(fl.recent_contexts),
            'learning_rate': fl.learning_rate,
            'top_weights': {k: round(v, 2) for k, v in top_3},
            'bottom_weights': {k: round(v, 2) for k, v in bottom_3},
            'engagements_by_hour': dict(fl.engagements_by_hour),
            'all_weights': {k: round(v, 2) for k, v in fl.weights.items()},
        }
    
    def set_light_position(self, x: float, y: float, z: float):
        """Update current light position (called from controller for feedback context)."""
        self.current_light_position = (x, y, z)

    def apply_aggression_modifiers(self, params: Dict) -> Dict:
        """
        Apply aggression-based modifiers to behavior parameters.
        
        Higher aggression means:
        - Larger wander box (more movement, more visible)
        - More frequent pulses (attention-getting)
        - Faster movement (more dynamic, eye-catching)
        - Stronger response to passive zone flow
        - Wander box shifts toward passive zone edge
        """
        agg = self.state.aggression
        if not agg or agg.level <= 0:
            return params
        
        result = dict(params)
        level = agg.level  # 0.0 to 1.0
        
        # Only apply aggression effects in IDLE mode (when fishing for attention)
        if self.state.mode != BehaviorMode.IDLE:
            return result
        
        # ======================
        # WANDER BOX EXPANSION
        # ======================
        # Higher aggression = larger wander box, shifted toward passive zone (higher Z)
        # Base Z range: -32 to 28 (close to panels)
        # High aggression: -32 to 60 (extends toward passive zone)
        z_expansion = level * 40  # Up to 40cm expansion at max aggression
        self.current_wander_box['max_z'] = self.base_wander_box['max_z'] + z_expansion
        
        # Also expand Y range slightly for more vertical movement
        y_expansion = level * 30  # Up to 30cm additional height
        self.current_wander_box['max_y'] = min(
            180,  # Cap at reasonable height
            self.base_wander_box['max_y'] + y_expansion
        )
        
        # ======================
        # PULSE FREQUENCY
        # ======================
        # Higher aggression = faster pulse (lower pulse_speed value)
        # This creates more eye-catching movement
        pulse_mult = 1.0 - (level * 0.4)  # 1.0x to 0.6x (faster pulse)
        result['pulse_speed'] = result.get('pulse_speed', 4000) * pulse_mult
        
        # ======================
        # MOVEMENT SPEED
        # ======================
        # Higher aggression = faster, more dynamic movement
        speed_mult = 1.0 + (level * 0.5)  # 1.0x to 1.5x speed
        result['move_speed'] = result.get('move_speed', 20) * speed_mult
        
        # ======================
        # WANDER INTERVAL
        # ======================
        # Higher aggression = shorter pauses between movements
        interval_mult = 1.0 - (level * 0.4)  # 1.0x to 0.6x interval
        result['wander_interval'] = result.get('wander_interval', 5.0) * interval_mult
        
        # ======================
        # BRIGHTNESS BOOST
        # ======================
        # Slightly brighter when aggressive (more visible)
        bright_mult = 1.0 + (level * 0.2)  # 1.0x to 1.2x brightness
        result['brightness_max'] = result.get('brightness_max', 15) * bright_mult
        
        return result

    def apply_proximity_modifiers(self, params: Dict) -> Dict:
        """
        Apply Z proximity-based modifiers to parameters.
        When person is closer to panels (lower Z), the light:
        - Moves slower (more deliberate, controlled)
        - Gets brighter (more intense engagement)
        - Has tighter tracking (more precise follow)
        """
        if self.state.mode not in (BehaviorMode.ENGAGED, BehaviorMode.CROWD):
            return params
        
        prox = self.state.proximity_factor  # 0 = far, 1 = close
        if prox <= 0:
            return params
        
        result = dict(params)
        
        # Speed: slower when close (more deliberate movement)
        # prox=1 -> multiply by 0.6, prox=0 -> multiply by 1.4
        speed_mult = self.PROXIMITY_SPEED_MULT['far'] + (self.PROXIMITY_SPEED_MULT['near'] - self.PROXIMITY_SPEED_MULT['far']) * prox
        result['move_speed'] = result.get('move_speed', 40) * speed_mult
        
        # Brightness: brighter when close (more intense)
        # prox=1 -> multiply by 1.4, prox=0 -> multiply by 0.8
        bright_mult = self.PROXIMITY_BRIGHTNESS_MULT['far'] + (self.PROXIMITY_BRIGHTNESS_MULT['near'] - self.PROXIMITY_BRIGHTNESS_MULT['far']) * prox
        result['brightness_max'] = result.get('brightness_max', 30) * bright_mult
        result['brightness_min'] = result.get('brightness_min', 8) * bright_mult
        
        # Smoothing: tighter tracking when close (more precise)
        # prox=1 -> multiply by 0.7, prox=0 -> multiply by 1.3
        smooth_mult = self.PROXIMITY_SMOOTHING_MULT['far'] + (self.PROXIMITY_SMOOTHING_MULT['near'] - self.PROXIMITY_SMOOTHING_MULT['far']) * prox
        result['follow_smoothing'] = result.get('follow_smoothing', 0.05) * smooth_mult
        
        return result
    
    def calculate_parameters(self, active_count: int, passive_count: int,
                            current_pos: Tuple[float, float, float],
                            flow_balance: float = 0.0) -> Dict:
        """Calculate final light parameters based on all factors"""
        now = time.time()
        
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
        # More people = brighter and faster pulse
        if self.state.mode in (BehaviorMode.ENGAGED, BehaviorMode.CROWD):
            # Brightness scales more aggressively: +20% per person, max 2x
            brightness_scale = min(1.0 + active_count * 0.20, 2.0)
            base_params['brightness_max'] *= brightness_scale
            base_params['brightness_min'] *= (1.0 + active_count * 0.15)  # Min brightness also increases
            
            # Pulse speed increases with people: faster pulse = more energy
            # Lower pulse_speed value = faster pulse
            pulse_scale = max(0.5, 1.0 - active_count * 0.15)  # 15% faster per person, min 0.5x
            base_params['pulse_speed'] *= pulse_scale
        
        # Apply modifiers in order
        params = self.apply_meta_modifiers(base_params)
        params = self.apply_time_of_day(params)
        params = self.apply_dwell_rewards(params)
        params = self.apply_anti_repetition(params, current_pos)
        params = self.apply_idle_trends(params)           # Passive zone trends (IDLE mode only)
        params = self.apply_aggression_modifiers(params)  # Aggression-based attention-seeking (IDLE mode only)
        params = self.apply_flow_positioning(params)      # Flow-responsive positioning (IDLE mode only)
        params = self.apply_almost_engaged_attraction(params, current_pos)  # Almost-engaged attraction (IDLE mode only)
        params = self.apply_feedback_learning(params)     # Learned behavior weights
        params = self.apply_proximity_modifiers(params)   # Z proximity response
        
        # Apply flow bias to wander box
        if self.state.mode in (BehaviorMode.IDLE, BehaviorMode.FLOW):
            flow_bias = flow_balance * 50 * self.meta.trend_weight
            self.current_wander_box['min_x'] = self.base_wander_box['min_x'] + flow_bias
            self.current_wander_box['max_x'] = self.base_wander_box['max_x'] + flow_bias
        
        # Apply entry pulse effect (brightness boost when someone enters)
        entry_pulse_intensity = self._update_entry_pulse(now)
        if entry_pulse_intensity > 0:
            params['brightness_max'] = params.get('brightness_max', 30) + self.ENTRY_PULSE_BRIGHTNESS_BOOST * entry_pulse_intensity
            params['brightness_min'] = params.get('brightness_min', 8) + self.ENTRY_PULSE_BRIGHTNESS_BOOST * 0.5 * entry_pulse_intensity
            # Also tighten falloff for more focused effect
            params['falloff_radius'] = params.get('falloff_radius', 50) * (1.0 - 0.3 * entry_pulse_intensity)
        
        # Apply bloom effect (smooth transition to full-panel radius)
        if self.state.bloom_active or self.state.bloom_progress > 0:
            base_radius = params.get('falloff_radius', 50)
            # Lerp between normal radius and bloom radius
            params['falloff_radius'] = base_radius + (self.bloom_radius - base_radius) * self.state.bloom_progress
            # Also boost brightness during bloom
            params['brightness_max'] = params.get('brightness_max', 30) * (1 + 0.5 * self.state.bloom_progress)
        
        # Store proximity info in params for display/debugging
        params['proximity_factor'] = self.state.proximity_factor
        params['nearest_z'] = self.state.nearest_person_z
        params['entry_pulse'] = entry_pulse_intensity
        
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
        
        # Store decision inputs for display
        self.state.last_active_count = active_count
        self.state.last_passive_count = passive_count
        self.state.last_passive_rate = passive_rate
        
        # Update proximity tracking (Z distance response)
        self.update_proximity()
        
        # Update mode duration
        self.state.mode_duration = now - self.state.mode_start_time
        
        # Check for mode change with stickiness
        desired_mode = self.determine_mode(active_count, passive_count, passive_rate)
        
        if desired_mode != self.state.mode and not self.state.transitioning:
            # Get stickiness for this transition
            stickiness_key = (self.state.mode, desired_mode)
            required_time = self.MODE_STICKINESS.get(stickiness_key, 5.0)  # Default 5s
            
            # Check minimum mode duration (unless it's an immediate transition)
            if required_time > 0 and self.state.mode_duration < self.MIN_MODE_DURATION:
                # Haven't been in current mode long enough, don't start pending
                pass
            elif required_time == 0:
                # Immediate transition (e.g., someone enters active zone)
                self.state.pending_mode = None
                self.state.pending_mode_start = 0.0
                self.start_transition(desired_mode)
            elif self.state.pending_mode == desired_mode:
                # Already tracking this pending mode, check if enough time has passed
                time_pending = now - self.state.pending_mode_start
                if time_pending >= required_time:
                    # Conditions met long enough, do the transition
                    self.state.pending_mode = None
                    self.state.pending_mode_start = 0.0
                    self.start_transition(desired_mode)
            else:
                # Start tracking a new pending mode
                self.state.pending_mode = desired_mode
                self.state.pending_mode_start = now
        elif desired_mode == self.state.mode:
            # Conditions no longer support pending mode, clear it
            self.state.pending_mode = None
            self.state.pending_mode_start = 0.0
        
        # Update animated wander box (smooth transition to/from engaged box)
        self.update_animated_wander_box(dt)
        
        # Update gesture
        self.update_gesture(dt)
        
        # Update idle trends periodically (always, regardless of mode)
        # This keeps the trends data fresh for display even when engaged
        if now - self.state.last_trend_update > 5.0:
            self.update_idle_trends()
        
        # Update aggression level (always, uses EMA smoothing)
        self.update_aggression(dt, passive_count, active_count)
        
        # Update flow tracking (always, for anticipatory positioning)
        self.update_flow(dt)
        
        # Update almost-engaged detection (IDLE mode mainly, but always track)
        self.update_almost_engaged(dt)
        
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
        
        # Update status text periodically (every 8 seconds, using a proper interval check)
        if not hasattr(self, '_last_status_update'):
            self._last_status_update = 0.0
        if now - self._last_status_update >= 8.0:
            self.update_status_text(active_count)
            self._last_status_update = now
        
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
        """Get current (animated) wander box"""
        return self.animated_wander_box
    
    def calculate_engaged_wander_box(self) -> Dict:
        """Calculate wander box based on active zone people.
        
        Multi-person strategy:
        - 1 person: Tight box centered on them
        - 2 people: Box covers both, but weighted toward primary (longest dwell)
        - 3+ people: Wider box to roam between them, centered on centroid
        """
        if not self.active_zone_people:
            return dict(self.base_wander_box)
        
        count = len(self.active_zone_people)
        primary = self.get_primary_person()
        
        if count == 1 and primary:
            # Single person - very tight focus
            person_x = primary[1][0]
            padding_x = self.engaged_box_padding_x  # ±15cm
            
        elif count == 2:
            # Two people - cover both but weight toward primary
            positions = list(self.active_zone_people.values())
            all_x = [p[0] for p in positions]
            
            # Weight 70% toward primary, 30% toward other
            if primary:
                primary_x = primary[1][0]
                centroid_x = sum(all_x) / len(all_x)
                person_x = primary_x * 0.7 + centroid_x * 0.3
            else:
                person_x = sum(all_x) / len(all_x)
            
            spread_x = max(all_x) - min(all_x)
            padding_x = self.engaged_box_padding_x + spread_x * 0.4 + 10
            
        else:
            # 3+ people (crowd) - wider roaming box
            positions = list(self.active_zone_people.values())
            all_x = [p[0] for p in positions]
            person_x = sum(all_x) / len(all_x)  # Centroid
            spread_x = max(all_x) - min(all_x)
            padding_x = self.engaged_box_padding_x + spread_x * 0.5 + count * 8
        
        # Create engaged box - X follows people, Y/Z stay at base ranges
        engaged_box = {
            'min_x': max(self.base_wander_box['min_x'], person_x - padding_x),
            'max_x': min(self.base_wander_box['max_x'], person_x + padding_x),
            'min_y': self.base_wander_box['min_y'],
            'max_y': self.base_wander_box['max_y'],
            'min_z': self.base_wander_box['min_z'],
            'max_z': self.base_wander_box['max_z'],
        }
        
        return engaged_box
    
    def update_animated_wander_box(self, dt: float):
        """Smoothly animate wander box toward target"""
        # Determine target based on mode
        if self.state.mode in (BehaviorMode.ENGAGED, BehaviorMode.CROWD):
            self.target_wander_box = self.calculate_engaged_wander_box()
        else:
            # Return to base box (possibly with flow bias applied)
            self.target_wander_box = dict(self.current_wander_box)
        
        # Lerp each dimension toward target
        lerp_factor = 1.0 - math.exp(-self.wander_box_lerp_speed * dt)
        
        for key in self.animated_wander_box:
            current = self.animated_wander_box[key]
            target = self.target_wander_box[key]
            self.animated_wander_box[key] = current + (target - current) * lerp_factor
    
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
        now = time.time()
        
        # Calculate pending mode info
        pending_info = None
        if self.state.pending_mode:
            time_pending = now - self.state.pending_mode_start
            stickiness_key = (self.state.mode, self.state.pending_mode)
            required_time = self.MODE_STICKINESS.get(stickiness_key, 5.0)
            pending_info = {
                'mode': self.state.pending_mode.value,
                'time_pending': time_pending,
                'time_required': required_time,
                'progress': min(1.0, time_pending / required_time) if required_time > 0 else 1.0
            }
        
        # Get the current driving parameters with their sources
        driving_factors = {}
        
        # Decision inputs - what's driving mode selection
        driving_factors['active_count'] = self.state.last_active_count
        driving_factors['passive_count'] = self.state.last_passive_count
        driving_factors['passive_rate'] = self.state.last_passive_rate
        driving_factors['flow_threshold'] = self.flow_threshold
        driving_factors['flow_enabled'] = self.meta.flow_mode_enabled
        
        # Mode thresholds for reference
        driving_factors['thresholds'] = {
            'crowd': 2,      # active_count >= 2
            'engaged': 1,    # active_count >= 1
            'flow': self.flow_threshold,  # passive_rate >= this
        }
        
        # Mode-based factors
        driving_factors['base_mode'] = self.state.mode.value
        driving_factors['mode_duration'] = self.state.mode_duration
        driving_factors['min_duration'] = self.MIN_MODE_DURATION
        driving_factors['mode_stable'] = self.state.mode_duration >= self.MIN_MODE_DURATION
        
        # Time of day influence
        tod = self.get_time_of_day_modifier()
        driving_factors['time_mood'] = tod.mood
        driving_factors['time_brightness'] = tod.brightness_mult
        driving_factors['time_pulse'] = tod.pulse_mult
        
        # Dwell/engagement rewards
        if self.state.dwell_start_time > 0:
            dwell_time = now - self.state.dwell_start_time
            driving_factors['dwell_time'] = dwell_time
            driving_factors['dwell_bonus'] = self.state.current_dwell_bonus
        
        # Current calculated parameters (the actual values being used)
        driving_factors['current_params'] = {
            'brightness_max': self.current_params.get('brightness_max', 30),
            'brightness_min': self.current_params.get('brightness_min', 8),
            'pulse_speed': self.current_params.get('pulse_speed', 2500),
            'move_speed': self.current_params.get('move_speed', 40),
            'falloff_radius': self.current_params.get('falloff_radius', 50),
        }
        
        # Proximity tracking (Z distance response)
        driving_factors['proximity'] = {
            'factor': self.state.proximity_factor,
            'nearest_z': self.state.nearest_person_z,
            'z_near': self.PROXIMITY_Z_NEAR,
            'z_far': self.PROXIMITY_Z_FAR,
        }
        
        # Entry pulse state
        if self.state.entry_pulse_active:
            elapsed = now - self.state.entry_pulse_start
            driving_factors['entry_pulse'] = {
                'active': True,
                'elapsed': elapsed,
                'duration': self.state.entry_pulse_duration,
            }
        
        # Bloom state
        if self.state.bloom_active or self.state.bloom_progress > 0:
            driving_factors['bloom_progress'] = self.state.bloom_progress
        
        # Idle trends (for IDLE mode display)
        idle_trends_info = None
        if self.state.idle_trends:
            t = self.state.idle_trends
            # Calculate seconds since last trend update
            time_since_update = now - self.state.last_trend_update
            idle_trends_info = {
                'period': t.period_name,
                'activity_anticipation': t.activity_anticipation,
                'flow_momentum': t.flow_momentum,
                'energy_level': t.energy_level,
                # Recent (1 min)
                'recent_passive': t.recent_passive_count,
                'recent_active': t.recent_active_count,
                'recent_speed': t.recent_avg_speed,
                # Short (5 min)
                'short_activity': t.short_activity_level,
                'short_passive': t.short_passive_count,
                'short_active': t.short_active_count,
                # Medium (30 min)
                'medium_activity': t.medium_activity_level,
                'medium_passive': t.medium_passive_count,
                'medium_active': t.medium_active_count,
                # Long (1 hr)
                'long_activity': t.long_activity_level,
                'long_passive': t.long_passive_count,
                'long_active': t.long_active_count,
                # Data availability
                'has_recent': t.has_recent_data,
                'has_short': t.has_short_data,
                'has_medium': t.has_medium_data,
                'has_long': t.has_long_data,
                'has_historical': t.has_historical_data,
                'database_error': t.database_error,
                # Update timing
                'seconds_since_update': time_since_update,
            }
        
        # Aggression state info
        aggression_info = None
        if self.state.aggression:
            a = self.state.aggression
            aggression_info = {
                'level': a.level,
                'raw_level': a.raw_level,
                'time_of_day_cap': a.time_of_day_cap,
                'seconds_since_engagement': a.seconds_since_engagement,
                'passive_without_conversion': a.passive_without_conversion,
                'recent_engagements': a.recent_engagements,
                'current_engagement': a.current_engagement,
            }
        
        # Flow state info (for anticipatory positioning)
        flow_info = None
        if self.state.flow:
            f = self.state.flow
            flow_info = {
                'direction': f.direction,
                'raw_direction': f.raw_direction,
                'strength': f.strength,
                'x_offset': f.x_offset,
                'left_to_right': f.left_to_right_count,
                'right_to_left': f.right_to_left_count,
                'total_events': f.total_events,
            }
        
        # Almost-engaged state info
        almost_engaged_info = None
        if self.state.almost_engaged:
            ae = self.state.almost_engaged
            # Get info about current candidates
            candidates_info = []
            for pid, c in ae.candidates.items():
                candidates_info.append({
                    'id': pid,
                    'speed': c.current_speed,
                    'distance': c.distance_to_active,
                    'duration': time.time() - c.first_detected,
                    'strategy': c.strategy_used.value,
                })
            
            almost_engaged_info = {
                'total_detected': ae.total_detected,
                'total_converted': ae.total_converted,
                'conversion_rate': ae.total_converted / ae.total_detected if ae.total_detected > 0 else 0,
                'active_attraction': ae.active_attraction,
                'current_strategy': ae.current_strategy.value,
                'target_id': ae.attraction_target_id,
                'candidate_count': len(ae.candidates),
                'candidates': candidates_info[:3],  # Top 3 only for display
                'strategy_stats': {
                    k: {'attempts': v['attempts'], 'conversions': v['conversions'],
                        'rate': v['conversions'] / v['attempts'] if v['attempts'] > 0 else 0}
                    for k, v in ae.strategy_stats.items()
                },
            }
        
        # Feedback learning state info
        feedback_info = self.get_feedback_learning_status()
        
        return {
            'mode': self.state.mode.value,
            'mode_duration': self.state.mode_duration,
            'transitioning': self.state.transitioning,
            'gesture': self.state.gesture.value if self.state.gesture != GestureType.NONE else None,
            'is_bored': self.state.is_bored,
            'dwell_bonus': self.state.current_dwell_bonus,
            'status_text': self.state.status_text,
            'time_of_day': self.get_time_of_day_modifier().mood,
            'pending_mode': pending_info,
            'driving_factors': driving_factors,
            'proximity_factor': self.state.proximity_factor,
            'entry_pulse_active': self.state.entry_pulse_active,
            'idle_trends': idle_trends_info,
            'aggression': aggression_info,
            'flow': flow_info,
            'almost_engaged': almost_engaged_info,
            'feedback_learning': feedback_info,
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
