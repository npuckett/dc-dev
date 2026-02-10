"""
Behavior System
================
Main coordinator for all behavior components.
Composes ModeStateMachine, AggressionState, FlowState, AlmostEngaged, and Trends.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

# Local imports
from .modes import BehaviorMode, GestureType, ModeStateMachine, TimePeriod
from .parameters import MetaParameters, load_preset
from .trends import IdleTrends, TrendAnalyzer
from .states import (
    AggressionState, FlowState, AlmostEngagedState, 
    AttractionStrategy, FeedbackLearning, EngagementContext
)


# =============================================================================
# BEHAVIOR OUTPUT
# =============================================================================

@dataclass
class BehaviorOutput:
    """
    Output of the behavior system each frame.
    Drives light animation parameters.
    """
    # Current mode
    mode: BehaviorMode = BehaviorMode.IDLE
    mode_blend: float = 1.0  # 0-1, used during transitions
    
    # Position modifiers
    wander_x_offset: float = 0.0        # cm, from flow alignment
    wander_center_x: float = -150.0     # cm, base wander center
    wander_center_z: float = 180.0      # cm, base wander center
    
    # Movement modifiers (multiply with base values)
    move_speed_mult: float = 1.0
    responsiveness_mult: float = 1.0
    
    # Brightness/intensity modifiers
    brightness_mult: float = 1.0
    intensity_mult: float = 1.0
    brightness_boost: float = 0.0       # Additive boost (0.0-0.5)
    
    # Gesture suggestions
    gesture_type: Optional[GestureType] = None
    gesture_priority: float = 0.0       # 0-1, urgency
    gesture_target_x: float = 0.0       # If gesture should orient
    gesture_target_z: float = 0.0
    
    # Attraction state
    attracting: bool = False
    attraction_strategy: AttractionStrategy = AttractionStrategy.NONE
    attraction_target_x: float = 0.0
    attraction_target_z: float = 0.0
    
    # Debug info
    aggression_level: float = 0.0
    flow_direction: float = 0.0
    time_period: str = "daytime"


# =============================================================================
# BEHAVIOR SYSTEM
# =============================================================================

class BehaviorSystem:
    """
    Main behavior coordinator.
    
    Call update() each frame with current tracking state.
    Returns BehaviorOutput for light animation system.
    
    Usage:
        behavior = BehaviorSystem()
        behavior.start()
        
        # Each frame:
        output = behavior.update(dt, tracked_people)
        apply_to_light(output)
    """
    
    def __init__(self, meta_params: Optional[MetaParameters] = None):
        """
        Initialize behavior system.
        
        Args:
            meta_params: Override meta parameters, or use defaults
        """
        # Meta-parameters (tune overall character)
        self.params = meta_params or load_preset('balanced')
        
        # Sub-systems
        self.mode_machine = ModeStateMachine()
        self.aggression = AggressionState()
        self.flow = FlowState()
        self.almost_engaged = AlmostEngagedState()
        self.feedback = FeedbackLearning()
        
        # Trends (for idle mode)
        self.trends = TrendAnalyzer()
        
        # State
        self._running = False
        self._frame_count = 0
        self._last_update = 0.0
        
        # Callbacks
        self._on_mode_change: List[Callable[[BehaviorMode, BehaviorMode], None]] = []
        self._on_engagement: List[Callable[[EngagementContext], None]] = []
        
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def start(self):
        """Start the behavior system."""
        self._running = True
        self._last_update = time.time()
        self.feedback.session_start = time.time()
    
    def stop(self):
        """Stop the behavior system."""
        self._running = False
    
    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def on_mode_change(self, callback: Callable[[BehaviorMode, BehaviorMode], None]):
        """Register callback for mode changes. Called with (old_mode, new_mode)."""
        self._on_mode_change.append(callback)
    
    def on_engagement(self, callback: Callable[[EngagementContext], None]):
        """Register callback for engagements. Called with context."""
        self._on_engagement.append(callback)
    
    # -------------------------------------------------------------------------
    # Main Update
    # -------------------------------------------------------------------------
    
    def update(self, dt: float, 
               active_count: int = 0,
               passive_count: int = 0,
               active_people: Optional[List[dict]] = None,
               passive_people: Optional[List[dict]] = None) -> BehaviorOutput:
        """
        Update behavior system with current tracking state.
        
        Args:
            dt: Delta time in seconds
            active_count: Number of people in active zone
            passive_count: Number of people in passive zone
            active_people: List of dicts with person info (x, z, speed)
            passive_people: List of dicts with person info
            
        Returns:
            BehaviorOutput with current behavior parameters
        """
        if not self._running:
            return BehaviorOutput()
        
        now = time.time()
        active_people = active_people or []
        passive_people = passive_people or []
        
        self._frame_count += 1
        
        # Update sub-systems
        self._update_aggression(dt, passive_count, active_count)
        self._update_flow(dt)
        self._update_almost_engaged(dt, passive_people, active_people)
        
        # Update mode
        old_mode = self.mode_machine.current_mode
        self.mode_machine.update(dt, active_count, passive_count)
        new_mode = self.mode_machine.current_mode
        
        # Fire mode change callbacks
        if old_mode != new_mode:
            for cb in self._on_mode_change:
                try:
                    cb(old_mode, new_mode)
                except Exception:
                    pass
        
        # Detect new engagements
        self._check_for_engagements(active_people)
        
        # Update trends (less frequently)
        if self._frame_count % 30 == 0:
            self.trends.record_event(active_count, passive_count)
        
        # Build output
        output = self._build_output(dt, new_mode, active_count, active_people)
        
        self._last_update = now
        return output
    
    # -------------------------------------------------------------------------
    # Sub-system Updates
    # -------------------------------------------------------------------------
    
    def _update_aggression(self, dt: float, passive_count: int, active_count: int):
        """Update aggression state."""
        self.aggression.update(dt, passive_count, active_count)
    
    def _update_flow(self, dt: float):
        """Update flow tracking (placeholder - needs velocity data)."""
        # In full implementation, this gets L→R and R→L counts from tracking
        self.flow.update(dt)
    
    def _update_almost_engaged(self, dt: float, 
                                passive_people: List[dict],
                                active_people: List[dict]):
        """Update almost-engaged tracking."""
        # Cleanup stale candidates
        self.almost_engaged.cleanup_stale()
        
        # Update candidates from passive people
        for p in passive_people:
            pid = p.get('id', 0)
            x = p.get('x', 0)
            z = p.get('z', 0)
            speed = p.get('speed', 100)
            
            # Estimate distance to active zone
            # (simplified - use zones module for proper calculation)
            distance_to_active = max(0, z - 283)  # Active zone ends at z=283
            
            self.almost_engaged.update_candidate(pid, x, z, speed, distance_to_active)
        
        # Check if we should try attraction
        if self.almost_engaged.should_attract():
            candidate = self.almost_engaged.get_best_candidate()
            if candidate:
                self.almost_engaged.start_attraction(candidate.person_id)
        
        # Check if attraction target entered active zone
        if self.almost_engaged.active_attraction:
            target_id = self.almost_engaged.attraction_target_id
            for p in active_people:
                if p.get('id') == target_id:
                    # Conversion!
                    self.almost_engaged.end_attraction(converted=True)
                    break
    
    def _check_for_engagements(self, active_people: List[dict]):
        """Check for new engagements and record context."""
        # This would track which people are newly in active zone
        # and record engagement context for learning
        pass  # Simplified for now
    
    # -------------------------------------------------------------------------
    # Output Building
    # -------------------------------------------------------------------------
    
    def _build_output(self, dt: float, 
                      mode: BehaviorMode,
                      active_count: int,
                      active_people: List[dict]) -> BehaviorOutput:
        """Build behavior output from current state."""
        output = BehaviorOutput()
        output.mode = mode
        
        # Time period
        from datetime import datetime
        hour = datetime.now().hour
        if 6 <= hour < 9:
            output.time_period = "morning"
        elif 9 <= hour < 17:
            output.time_period = "daytime"
        elif 17 <= hour < 20:
            output.time_period = "evening"
        else:
            output.time_period = "night"
        
        # Apply aggression influence
        agg = self.aggression.get_influence()
        output.brightness_boost = agg['brightness_boost'] * self.params.brightness_global
        output.move_speed_mult = agg['move_speed_mult']
        
        # Apply flow offset
        output.wander_x_offset = self.flow.get_wander_offset() * self.params.energy
        
        # Apply meta-parameters
        output.responsiveness_mult = self.params.responsiveness
        output.brightness_mult = self.params.brightness_global
        output.intensity_mult = self.params.pulse_global
        
        # Mode-specific adjustments
        if mode == BehaviorMode.IDLE:
            # Apply idle trend influences
            if self.trends.has_data():
                trends = self.trends.get_current_trends()
                if trends:
                    trends.compute_influences()  # Updates internal state
                    # Get wander bias from computed influences
                    wander_bias = trends.get_wander_bias() if hasattr(trends, 'get_wander_bias') else 0
                    output.wander_x_offset += wander_bias * 20  # Scale to cm
        
        elif mode == BehaviorMode.ENGAGED:
            # More responsive when engaged
            output.responsiveness_mult *= 1.5
            
            # Orient toward engaged person
            if active_people:
                closest = min(active_people, key=lambda p: p.get('z', 999))
                output.gesture_target_x = closest.get('x', 0)
                output.gesture_target_z = closest.get('z', 0)
        
        elif mode == BehaviorMode.CROWD:
            # Slower, calmer in crowds
            output.move_speed_mult *= 0.7
            output.brightness_mult *= 0.9
        
        elif mode == BehaviorMode.FLOW:
            # Anticipatory positioning
            output.wander_x_offset = self.flow.x_offset * 1.5
        
        # Attraction state
        if self.almost_engaged.active_attraction:
            output.attracting = True
            output.attraction_strategy = self.almost_engaged.current_strategy
            candidate = self.almost_engaged.candidates.get(
                self.almost_engaged.attraction_target_id
            )
            if candidate:
                output.attraction_target_x = candidate.position_x
                output.attraction_target_z = candidate.position_z
        
        # Debug info
        output.aggression_level = self.aggression.level
        output.flow_direction = self.flow.direction
        
        return output
    
    # -------------------------------------------------------------------------
    # Gestures
    # -------------------------------------------------------------------------
    
    def suggest_gesture(self) -> Optional[GestureType]:
        """
        Get a gesture suggestion based on current state.
        
        Returns:
            GestureType if a gesture is warranted, None otherwise
        """
        if not self.params.gestures_enabled:
            return None
        
        mode = self.mode_machine.current_mode
        
        if mode == BehaviorMode.IDLE:
            # Occasional attention-seeking gestures
            chance = 0.01 * self.aggression.level * self.params.gesture_freq_mult
            import random
            if random.random() < chance:
                return random.choice([
                    GestureType.WAVE,
                    GestureType.PULSE,
                    GestureType.SWAY,
                ])
        
        elif mode == BehaviorMode.ENGAGED:
            # Interactive gestures
            chance = 0.02 * self.params.gesture_freq_mult
            import random
            if random.random() < chance:
                return random.choice([
                    GestureType.PULSE,
                    GestureType.RIPPLE,
                    GestureType.ORBIT,
                ])
        
        return None
    
    # -------------------------------------------------------------------------
    # State Access
    # -------------------------------------------------------------------------
    
    def get_mode(self) -> BehaviorMode:
        """Get current behavior mode."""
        return self.mode_machine.current_mode
    
    def get_mode_duration(self) -> float:
        """Get time in current mode."""
        return self.mode_machine.mode_duration
    
    def get_aggression(self) -> float:
        """Get current aggression level (0-1)."""
        return self.aggression.level
    
    def get_flow_direction(self) -> float:
        """Get flow direction (-1 to +1)."""
        return self.flow.direction
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            'mode': self.mode_machine.current_mode.value,
            'mode_duration': round(self.mode_machine.mode_duration, 1),
            'aggression': round(self.aggression.level, 2),
            'flow': self.flow.to_dict(),
            'almost_engaged': {
                'candidates': len(self.almost_engaged.candidates),
                'conversion_rate': round(self.almost_engaged.get_conversion_rate(), 2),
                'strategy_stats': self.almost_engaged.get_strategy_stats(),
            },
            'feedback': self.feedback.get_stats(),
            'frame_count': self._frame_count,
        }
    
    # -------------------------------------------------------------------------
    # Presets
    # -------------------------------------------------------------------------
    
    def apply_preset(self, name: str):
        """Apply a named preset."""
        self.params = load_preset(name)
    
    def set_params(self, params: MetaParameters):
        """Set meta-parameters directly."""
        self.params = params
