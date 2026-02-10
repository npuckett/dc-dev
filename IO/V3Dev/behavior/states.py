"""
Behavior States
===============
Stateful behavior components: Aggression, Flow, Almost-Engaged detection.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# =============================================================================
# AGGRESSION STATE
# =============================================================================

# Time-of-day aggression caps for financial district location
AGGRESSION_TIME_CAPS: Dict[int, float] = {
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
class AggressionState:
    """
    Tracks the "attention-seeking" aggression level of the system.
    
    Aggression rises when:
    - Time passes without engagement
    - High passive traffic with low conversion
    
    Aggression falls when:
    - Someone engages (enters active zone)
    - Recent engagement success
    
    Aggression is capped by time of day.
    """
    level: float = 0.3              # Current aggression (0.0-1.0)
    raw_level: float = 0.3          # Uncapped level
    
    # Factors that increase aggression
    seconds_since_engagement: float = 0.0
    passive_without_conversion: int = 0
    
    # Factors that decrease aggression
    recent_engagements: int = 0
    current_engagement: bool = False
    
    # Time-of-day cap
    time_of_day_cap: float = 0.8
    
    # EMA smoothing
    ema_alpha: float = 0.1
    
    # Tracking
    last_engagement_time: float = 0.0
    last_passive_count: int = 0
    last_update: float = 0.0
    
    def update(self, dt: float, passive_count: int, active_count: int):
        """
        Update aggression level based on current conditions.
        
        Args:
            dt: Delta time in seconds
            passive_count: Current passive zone count
            active_count: Current active zone count
        """
        now = time.time()
        
        # Update time-of-day cap
        hour = datetime.now().hour
        self.time_of_day_cap = AGGRESSION_TIME_CAPS.get(hour, 0.5)
        
        # Track engagement
        if active_count > 0:
            if not self.current_engagement:
                # New engagement!
                self.current_engagement = True
                self.recent_engagements += 1
                self.last_engagement_time = now
                self.passive_without_conversion = 0
            self.seconds_since_engagement = 0
        else:
            self.current_engagement = False
            self.seconds_since_engagement = now - self.last_engagement_time
        
        # Track passive traffic without conversion
        if passive_count > self.last_passive_count and active_count == 0:
            self.passive_without_conversion += 1
        self.last_passive_count = passive_count
        
        # Decay recent engagements over time (5 minute window)
        if now - self.last_update > 60:
            self.recent_engagements = max(0, self.recent_engagements - 1)
        
        # Calculate target aggression
        target = self._calculate_target_aggression()
        
        # Apply EMA smoothing
        self.raw_level = self.raw_level + self.ema_alpha * (target - self.raw_level)
        
        # Apply time-of-day cap
        self.level = min(self.raw_level, self.time_of_day_cap)
        
        self.last_update = now
    
    def _calculate_target_aggression(self) -> float:
        """Calculate target aggression based on factors."""
        target = 0.3  # Base level
        
        # Increase based on time since engagement
        if self.seconds_since_engagement > 60:
            # Ramp up after 1 minute
            minutes_waiting = (self.seconds_since_engagement - 60) / 60
            target += min(0.4, minutes_waiting * 0.1)
        
        # Increase based on passive traffic without conversion
        target += min(0.2, self.passive_without_conversion * 0.02)
        
        # Decrease based on recent engagements
        target -= min(0.3, self.recent_engagements * 0.1)
        
        # Decrease if currently engaged
        if self.current_engagement:
            target -= 0.3
        
        return max(0.0, min(1.0, target))
    
    def on_person_engaged(self):
        """Call when someone enters active zone."""
        self.current_engagement = True
        self.recent_engagements += 1
        self.last_engagement_time = time.time()
        self.passive_without_conversion = 0
        self.seconds_since_engagement = 0
    
    def get_influence(self) -> Dict[str, float]:
        """
        Get behavior modifiers based on aggression level.
        
        Returns:
            Dict of modifier names to values
        """
        return {
            'brightness_boost': self.level * 0.3,      # Up to 30% brighter
            'move_speed_mult': 1.0 + self.level * 0.5, # Up to 50% faster
            'gesture_chance_mult': 1.0 + self.level,   # Up to 2x gesture chance
        }


# =============================================================================
# FLOW STATE
# =============================================================================

@dataclass
class FlowState:
    """
    Tracks real-time pedestrian flow direction for anticipatory positioning.
    
    Updated more frequently than IdleTrends (every 1-2 seconds).
    Uses a 30-second window for responsive flow tracking.
    """
    # Current smoothed flow direction (-1 to +1)
    direction: float = 0.0          # -1 = R→L, +1 = L→R
    raw_direction: float = 0.0      # Unsmoothed
    
    # Flow strength (confidence in direction)
    strength: float = 0.0           # 0 = no flow, 1 = strong flow
    
    # Raw counts from last window
    left_to_right_count: int = 0
    right_to_left_count: int = 0
    total_events: int = 0
    
    # Derived positioning offset (cm)
    x_offset: float = 0.0
    
    # EMA smoothing
    ema_alpha: float = 0.25
    
    # Timing
    last_update: float = 0.0
    update_interval: float = 1.5
    
    def update(self, dt: float, ltr_count: int = 0, rtl_count: int = 0):
        """
        Update flow state with new counts.
        
        Args:
            dt: Delta time
            ltr_count: Left-to-right movements since last update
            rtl_count: Right-to-left movements since last update
        """
        now = time.time()
        
        # Accumulate counts
        self.left_to_right_count = ltr_count
        self.right_to_left_count = rtl_count
        self.total_events = ltr_count + rtl_count
        
        if self.total_events > 0:
            # Calculate raw direction
            self.raw_direction = (ltr_count - rtl_count) / self.total_events
            
            # Calculate strength (how one-sided is the flow)
            self.strength = abs(self.raw_direction)
        else:
            self.raw_direction = 0.0
            self.strength = 0.0
        
        # Apply EMA smoothing
        self.direction = self.direction + self.ema_alpha * (self.raw_direction - self.direction)
        
        # Calculate X offset for positioning
        # Light should position toward where people are coming FROM
        # Positive direction = L→R = people from left, so bias RIGHT
        max_offset = 50.0  # Maximum offset in cm
        self.x_offset = -self.direction * self.strength * max_offset
        
        self.last_update = now
    
    def get_wander_offset(self) -> float:
        """Get X offset to apply to wander box."""
        return self.x_offset
    
    def to_dict(self) -> dict:
        """Serialize for debugging."""
        return {
            'direction': round(self.direction, 2),
            'strength': round(self.strength, 2),
            'x_offset': round(self.x_offset, 1),
            'ltr': self.left_to_right_count,
            'rtl': self.right_to_left_count,
        }


# =============================================================================
# ALMOST ENGAGED STATE
# =============================================================================

class AttractionStrategy(Enum):
    """Strategies for attracting almost-engaged people."""
    NONE = "none"
    BRIGHTNESS_PULSE = "brightness_pulse"
    DRIFT_TOWARD = "drift_toward"
    PAUSE_AND_LOOK = "pause_and_look"
    COMBINED = "combined"


@dataclass
class AlmostEngagedCandidate:
    """Tracks a single person who might be about to engage."""
    person_id: int
    first_detected: float = 0.0
    last_seen: float = 0.0
    
    # Position
    position_x: float = 0.0
    position_z: float = 0.0
    
    # Speed tracking (cm/s)
    current_speed: float = 0.0
    initial_speed: float = 0.0
    min_speed_seen: float = 999.0
    
    # Distance to active zone
    distance_to_active: float = 0.0
    
    # Attraction tracking
    strategy_used: AttractionStrategy = AttractionStrategy.NONE
    strategy_start_time: float = 0.0
    
    # Outcome
    converted: bool = False
    left_area: bool = False
    outcome_logged: bool = False


@dataclass
class AlmostEngagedState:
    """
    Tracks "almost engaged" detection - people who slow down near active zone.
    
    These are candidates for attraction strategies.
    """
    # Currently tracked candidates
    candidates: Dict[int, AlmostEngagedCandidate] = field(default_factory=dict)
    
    # Detection thresholds
    slow_speed_threshold: float = 50.0      # Below this = slowing
    near_active_threshold: float = 100.0    # Within this distance of active zone
    min_detection_time: float = 1.0         # Must be slow for this long
    
    # Current attraction state
    active_attraction: bool = False
    attraction_target_id: int = -1
    current_strategy: AttractionStrategy = AttractionStrategy.NONE
    
    # Strategy rotation (A/B testing)
    strategy_index: int = 0
    strategies_to_test: List[AttractionStrategy] = field(default_factory=lambda: [
        AttractionStrategy.BRIGHTNESS_PULSE,
        AttractionStrategy.DRIFT_TOWARD,
        AttractionStrategy.PAUSE_AND_LOOK,
    ])
    
    # Conversion tracking
    total_detected: int = 0
    total_converted: int = 0
    strategy_stats: Dict[str, Dict] = field(default_factory=lambda: {
        'brightness_pulse': {'attempts': 0, 'conversions': 0},
        'drift_toward': {'attempts': 0, 'conversions': 0},
        'pause_and_look': {'attempts': 0, 'conversions': 0},
        'none': {'attempts': 0, 'conversions': 0},
    })
    
    # Cooldown
    last_attraction_time: float = 0.0
    attraction_cooldown: float = 5.0
    
    last_update: float = 0.0
    
    def update_candidate(self, person_id: int, x: float, z: float, 
                         speed: float, distance_to_active: float):
        """Update or add a candidate."""
        now = time.time()
        
        if person_id in self.candidates:
            c = self.candidates[person_id]
            c.position_x = x
            c.position_z = z
            c.current_speed = speed
            c.min_speed_seen = min(c.min_speed_seen, speed)
            c.distance_to_active = distance_to_active
            c.last_seen = now
        else:
            self.candidates[person_id] = AlmostEngagedCandidate(
                person_id=person_id,
                first_detected=now,
                last_seen=now,
                position_x=x,
                position_z=z,
                current_speed=speed,
                initial_speed=speed,
                min_speed_seen=speed,
                distance_to_active=distance_to_active,
            )
            self.total_detected += 1
    
    def cleanup_stale(self, timeout: float = 2.0):
        """Remove candidates not seen recently."""
        now = time.time()
        stale = [pid for pid, c in self.candidates.items() 
                if now - c.last_seen > timeout]
        for pid in stale:
            del self.candidates[pid]
    
    def get_best_candidate(self) -> Optional[AlmostEngagedCandidate]:
        """Get the best candidate for attraction (closest, slowest)."""
        valid = [c for c in self.candidates.values() 
                if c.current_speed < self.slow_speed_threshold
                and c.distance_to_active < self.near_active_threshold
                and time.time() - c.first_detected >= self.min_detection_time]
        
        if not valid:
            return None
        
        # Sort by distance, then speed
        valid.sort(key=lambda c: (c.distance_to_active, c.current_speed))
        return valid[0]
    
    def should_attract(self) -> bool:
        """Check if we should attempt attraction."""
        now = time.time()
        if now - self.last_attraction_time < self.attraction_cooldown:
            return False
        return self.get_best_candidate() is not None
    
    def start_attraction(self, target_id: int) -> AttractionStrategy:
        """Start attracting a specific target."""
        strategy = self.strategies_to_test[self.strategy_index]
        self.strategy_index = (self.strategy_index + 1) % len(self.strategies_to_test)
        
        self.active_attraction = True
        self.attraction_target_id = target_id
        self.current_strategy = strategy
        self.last_attraction_time = time.time()
        
        if target_id in self.candidates:
            self.candidates[target_id].strategy_used = strategy
            self.candidates[target_id].strategy_start_time = time.time()
        
        # Track attempt
        self.strategy_stats[strategy.value]['attempts'] += 1
        
        return strategy
    
    def end_attraction(self, converted: bool = False):
        """End current attraction attempt."""
        if self.attraction_target_id in self.candidates and converted:
            self.strategy_stats[self.current_strategy.value]['conversions'] += 1
            self.total_converted += 1
        
        self.active_attraction = False
        self.attraction_target_id = -1
        self.current_strategy = AttractionStrategy.NONE
    
    def get_conversion_rate(self) -> float:
        """Get overall conversion rate."""
        if self.total_detected == 0:
            return 0.0
        return self.total_converted / self.total_detected
    
    def get_strategy_stats(self) -> Dict[str, float]:
        """Get conversion rates per strategy."""
        rates = {}
        for name, stats in self.strategy_stats.items():
            if stats['attempts'] > 0:
                rates[name] = stats['conversions'] / stats['attempts']
            else:
                rates[name] = 0.0
        return rates


# =============================================================================
# ENGAGEMENT CONTEXT (for feedback learning)
# =============================================================================

@dataclass
class EngagementContext:
    """
    Captures behavior state at the moment someone engages.
    Used for learning what behavior parameters lead to engagement.
    """
    timestamp: float = 0.0
    time_of_day: str = ""
    hour: int = 0
    
    # Mode at engagement
    mode_before: str = "idle"
    mode_duration: float = 0.0
    
    # State at engagement
    aggression_level: float = 0.0
    flow_direction: float = 0.0
    flow_x_offset: float = 0.0
    
    # Light position (normalized 0-1)
    light_x_normalized: float = 0.5
    light_z_normalized: float = 0.5
    
    # Movement characteristics
    move_speed: float = 0.0
    brightness: float = 0.0
    intensity: float = 0.0
    
    # Wander box position
    wander_x_offset: float = 0.0
    
    # Person info
    person_zone: str = "active"
    dwell_duration: float = 0.0
    
    # Almost-engaged tracking
    was_almost_engaged: bool = False
    attraction_strategy_used: str = "none"
    
    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            'timestamp': self.timestamp,
            'time_of_day': self.time_of_day,
            'hour': self.hour,
            'mode': self.mode_before,
            'aggression': round(self.aggression_level, 2),
            'flow_direction': round(self.flow_direction, 2),
            'light_x': round(self.light_x_normalized, 2),
            'light_z': round(self.light_z_normalized, 2),
            'was_almost_engaged': self.was_almost_engaged,
            'attraction_strategy': self.attraction_strategy_used,
        }


@dataclass
class FeedbackLearning:
    """
    Tracks behavior-to-engagement correlations and learns weights.
    
    The system logs what it was doing when people engage, then
    weights successful behaviors higher.
    """
    # Recent engagement contexts (ring buffer)
    recent_contexts: List[EngagementContext] = field(default_factory=list)
    max_contexts: int = 50
    
    # Learned weights (start at 1.0 = neutral)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'low_aggression': 1.0,
        'mid_aggression': 1.0,
        'high_aggression': 1.0,
        'left_position': 1.0,
        'center_position': 1.0,
        'right_position': 1.0,
        'flow_aligned': 1.0,
        'flow_neutral': 1.0,
        'flow_opposed': 1.0,
        'morning': 1.0,
        'afternoon': 1.0,
        'evening': 1.0,
        'late_night': 1.0,
        'from_idle': 1.0,
        'from_flow': 1.0,
    })
    
    # Learning parameters
    learning_rate: float = 0.02
    weight_min: float = 0.5
    weight_max: float = 2.0
    
    # Statistics
    total_engagements: int = 0
    engagements_by_hour: Dict[int, int] = field(default_factory=lambda: {h: 0 for h in range(24)})
    
    last_update: float = 0.0
    session_start: float = 0.0
    
    def record_engagement(self, context: EngagementContext):
        """Record an engagement context for learning."""
        # Add to ring buffer
        self.recent_contexts.append(context)
        if len(self.recent_contexts) > self.max_contexts:
            self.recent_contexts.pop(0)
        
        # Update statistics
        self.total_engagements += 1
        self.engagements_by_hour[context.hour] = self.engagements_by_hour.get(context.hour, 0) + 1
        
        # Update weights based on what worked
        self._update_weights(context)
        self.last_update = time.time()
    
    def _update_weights(self, context: EngagementContext):
        """Update weights based on successful engagement context."""
        # Aggression level at engagement
        if context.aggression_level < 0.3:
            self._boost_weight('low_aggression')
        elif context.aggression_level < 0.6:
            self._boost_weight('mid_aggression')
        else:
            self._boost_weight('high_aggression')
        
        # Position at engagement
        if context.light_x_normalized < 0.33:
            self._boost_weight('left_position')
        elif context.light_x_normalized < 0.66:
            self._boost_weight('center_position')
        else:
            self._boost_weight('right_position')
        
        # Time of day
        if context.time_of_day:
            self._boost_weight(context.time_of_day)
        
        # Mode before engagement
        if context.mode_before == 'idle':
            self._boost_weight('from_idle')
        elif context.mode_before == 'flow':
            self._boost_weight('from_flow')
    
    def _boost_weight(self, key: str):
        """Increase a weight slightly."""
        if key in self.weights:
            new_weight = self.weights[key] + self.learning_rate
            self.weights[key] = min(self.weight_max, new_weight)
    
    def get_weight(self, key: str) -> float:
        """Get current weight for a factor."""
        return self.weights.get(key, 1.0)
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            'total_engagements': self.total_engagements,
            'by_hour': dict(self.engagements_by_hour),
            'weights': {k: round(v, 2) for k, v in self.weights.items() if v != 1.0},
        }
