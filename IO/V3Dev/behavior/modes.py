"""
Behavior Modes and State Machine
================================
Defines behavior modes and transition logic.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from datetime import datetime


class BehaviorMode(Enum):
    """
    Operating modes for the light behavior system.
    
    IDLE: No one in active zone, gentle wander
    ENGAGED: 1-2 people in active zone, following behavior
    CROWD: 3+ people in active zone, energetic
    FLOW: Heavy passive traffic, drift with crowd
    """
    IDLE = "idle"
    ENGAGED = "engaged"
    CROWD = "crowd"
    FLOW = "flow"


class TimePeriod(Enum):
    """Time of day periods for trend analysis and behavior modulation."""
    LATE_NIGHT = "late_night"  # 0-6
    MORNING = "morning"        # 6-12
    AFTERNOON = "afternoon"    # 12-17
    EVENING = "evening"        # 17-24
    
    @staticmethod
    def current() -> 'TimePeriod':
        """Get the current time period."""
        hour = datetime.now().hour
        if 0 <= hour < 6:
            return TimePeriod.LATE_NIGHT
        elif 6 <= hour < 12:
            return TimePeriod.MORNING
        elif 12 <= hour < 17:
            return TimePeriod.AFTERNOON
        else:
            return TimePeriod.EVENING
    
    @staticmethod
    def from_hour(hour: int) -> 'TimePeriod':
        """Get time period for a specific hour."""
        if 0 <= hour < 6:
            return TimePeriod.LATE_NIGHT
        elif 6 <= hour < 12:
            return TimePeriod.MORNING
        elif 12 <= hour < 17:
            return TimePeriod.AFTERNOON
        else:
            return TimePeriod.EVENING


class GestureType(Enum):
    """
    Types of expressive gestures the light can perform.
    
    Gestures are temporary behaviors that overlay the current mode.
    """
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


# =============================================================================
# TIME OF DAY CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TimeOfDayModifier:
    """Modifiers applied based on time of day."""
    brightness_mult: float = 1.0
    pulse_mult: float = 1.0
    wander_y_min: float = 0
    wander_y_max: float = 150
    mood: str = "active"


# Time configurations by hour range
TIME_CONFIGS: Dict[Tuple[int, int], TimeOfDayModifier] = {
    (0, 6): TimeOfDayModifier(0.4, 1.5, 0, 60, "sleepy"),
    (6, 9): TimeOfDayModifier(0.7, 1.2, 0, 100, "waking"),
    (9, 17): TimeOfDayModifier(1.0, 1.0, 0, 150, "active"),
    (17, 20): TimeOfDayModifier(1.1, 0.9, 0, 150, "rush"),
    (20, 24): TimeOfDayModifier(0.6, 1.3, 0, 80, "evening"),
}


def get_time_of_day_modifier(hour: Optional[int] = None) -> TimeOfDayModifier:
    """Get the modifier for current or specified hour."""
    if hour is None:
        hour = datetime.now().hour
    
    for (start, end), modifier in TIME_CONFIGS.items():
        if start <= hour < end:
            return modifier
    
    return TimeOfDayModifier()  # Default


# =============================================================================
# MODE PARAMETERS
# =============================================================================

# Base parameters for each mode
MODE_PARAMS: Dict[BehaviorMode, Dict[str, Any]] = {
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


# Transition durations between modes (seconds)
TRANSITION_DURATIONS: Dict[Tuple[BehaviorMode, BehaviorMode], float] = {
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
MODE_STICKINESS: Dict[Tuple[BehaviorMode, BehaviorMode], float] = {
    (BehaviorMode.IDLE, BehaviorMode.ENGAGED): 0.0,      # Immediate when someone enters
    (BehaviorMode.IDLE, BehaviorMode.FLOW): 15.0,        # Wait 15s of passive traffic
    (BehaviorMode.ENGAGED, BehaviorMode.IDLE): 5.0,      # Wait 5s after last person leaves
    (BehaviorMode.ENGAGED, BehaviorMode.CROWD): 3.0,     # Wait 3s with 2+ people
    (BehaviorMode.CROWD, BehaviorMode.ENGAGED): 5.0,     # Wait 5s after crowd thins
    (BehaviorMode.CROWD, BehaviorMode.IDLE): 5.0,        # Wait 5s after everyone leaves
    (BehaviorMode.FLOW, BehaviorMode.IDLE): 10.0,        # Wait 10s of low traffic
    (BehaviorMode.FLOW, BehaviorMode.ENGAGED): 0.0,      # Immediate when someone enters
}

# Minimum time to stay in a mode before any switch
MIN_MODE_DURATION = 8.0


# =============================================================================
# MODE STATE MACHINE
# =============================================================================

@dataclass
class ModeState:
    """Tracks the current mode and transition state."""
    mode: BehaviorMode = BehaviorMode.IDLE
    mode_start_time: float = 0.0
    
    # Pending mode change (for stickiness)
    pending_mode: Optional[BehaviorMode] = None
    pending_mode_start: float = 0.0
    
    # Transition state
    transitioning: bool = False
    transition_start_time: float = 0.0
    transition_duration: float = 1.0
    transition_from_mode: BehaviorMode = BehaviorMode.IDLE
    
    def __post_init__(self):
        if self.mode_start_time == 0.0:
            self.mode_start_time = time.time()
    
    @property
    def mode_duration(self) -> float:
        """Time spent in current mode."""
        return time.time() - self.mode_start_time
    
    @property
    def transition_progress(self) -> float:
        """Progress through current transition (0-1)."""
        if not self.transitioning:
            return 1.0
        elapsed = time.time() - self.transition_start_time
        return min(1.0, elapsed / self.transition_duration)
    
    def get_transition_duration(self, from_mode: BehaviorMode, to_mode: BehaviorMode) -> float:
        """Get duration for a mode transition."""
        return TRANSITION_DURATIONS.get((from_mode, to_mode), 1.0)
    
    def get_stickiness(self, from_mode: BehaviorMode, to_mode: BehaviorMode) -> float:
        """Get required wait time for a mode transition."""
        return MODE_STICKINESS.get((from_mode, to_mode), 5.0)


class ModeStateMachine:
    """
    Manages mode transitions with stickiness and smooth interpolation.
    """
    
    def __init__(self):
        self.state = ModeState()
    
    @property
    def current_mode(self) -> BehaviorMode:
        """Get current mode."""
        return self.state.mode
    
    @property
    def mode_duration(self) -> float:
        """Get time in current mode."""
        return self.state.mode_duration
    
    def determine_mode(self, active_count: int, passive_count: int,
                       passive_rate: float = 0.0,
                       flow_threshold: float = 10.0,
                       crowd_threshold: int = 3) -> BehaviorMode:
        """
        Determine desired mode based on current conditions.
        
        Args:
            active_count: People in active zone
            passive_count: People in passive zone
            passive_rate: People per minute in passive zone
            flow_threshold: Passive rate threshold for FLOW mode
            crowd_threshold: Active count threshold for CROWD mode
            
        Returns:
            Desired BehaviorMode
        """
        if active_count >= crowd_threshold:
            return BehaviorMode.CROWD
        elif active_count >= 1:
            return BehaviorMode.ENGAGED
        elif passive_rate > flow_threshold and passive_count > 2:
            return BehaviorMode.FLOW
        else:
            return BehaviorMode.IDLE
    
    def update(self, dt_or_mode, active_count: int = 0, passive_count: int = 0,
               passive_rate: float = 0.0) -> bool:
        """
        Update mode state machine.
        
        Can be called two ways:
        1. update(dt, active_count, passive_count) - auto-determines mode
        2. update(desired_mode) - uses provided mode
        
        Args:
            dt_or_mode: Either delta time (float) or desired BehaviorMode
            active_count: People in active zone
            passive_count: People in passive zone
            passive_rate: People per minute in passive zone
            
        Returns:
            True if mode changed this frame
        """
        # Determine if first arg is a BehaviorMode or float (dt)
        if isinstance(dt_or_mode, BehaviorMode):
            desired_mode = dt_or_mode
        else:
            # dt_or_mode is dt, determine mode from counts
            desired_mode = self.determine_mode(active_count, passive_count, passive_rate)
        now = time.time()
        changed = False
        
        if desired_mode != self.state.mode and not self.state.transitioning:
            # Get stickiness for this transition
            required_time = self.state.get_stickiness(self.state.mode, desired_mode)
            
            # Check minimum mode duration
            if required_time > 0 and self.state.mode_duration < MIN_MODE_DURATION:
                # Haven't been in current mode long enough
                pass
            elif required_time == 0:
                # Immediate transition
                self._start_transition(desired_mode)
                changed = True
            elif self.state.pending_mode == desired_mode:
                # Check if pending long enough
                time_pending = now - self.state.pending_mode_start
                if time_pending >= required_time:
                    self._start_transition(desired_mode)
                    changed = True
            else:
                # Start tracking new pending mode
                self.state.pending_mode = desired_mode
                self.state.pending_mode_start = now
        elif desired_mode == self.state.mode:
            # Clear pending mode
            self.state.pending_mode = None
            self.state.pending_mode_start = 0.0
        
        # Update transition progress
        if self.state.transitioning:
            if self.state.transition_progress >= 1.0:
                self.state.transitioning = False
        
        return changed
    
    def _start_transition(self, new_mode: BehaviorMode):
        """Start a mode transition."""
        now = time.time()
        duration = self.state.get_transition_duration(self.state.mode, new_mode)
        
        self.state.transition_from_mode = self.state.mode
        self.state.mode = new_mode
        self.state.mode_start_time = now
        self.state.transitioning = True
        self.state.transition_start_time = now
        self.state.transition_duration = duration
        
        # Clear pending
        self.state.pending_mode = None
        self.state.pending_mode_start = 0.0
    
    def force_mode(self, mode: BehaviorMode):
        """Force an immediate mode change without transition."""
        self.state.mode = mode
        self.state.mode_start_time = time.time()
        self.state.transitioning = False
        self.state.pending_mode = None
        self.state.pending_mode_start = 0.0
    
    def interpolate_params(self, params_from: Dict, params_to: Dict) -> Dict:
        """
        Interpolate parameters during transition.
        
        Args:
            params_from: Parameters for previous mode
            params_to: Parameters for new mode
            
        Returns:
            Interpolated parameters
        """
        if not self.state.transitioning:
            return params_to
        
        t = self.state.transition_progress
        result = {}
        
        for key in params_to:
            if key in params_from:
                from_val = params_from[key]
                to_val = params_to[key]
                if isinstance(to_val, (int, float)) and isinstance(from_val, (int, float)):
                    result[key] = from_val + (to_val - from_val) * t
                else:
                    result[key] = to_val
            else:
                result[key] = params_to[key]
        
        return result
