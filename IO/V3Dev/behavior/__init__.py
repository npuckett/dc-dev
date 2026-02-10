"""
Behavior Module
================
High-level behavior system for the light installation.

This module extracts the behavior logic from light_behavior.py into
a cleaner, more maintainable structure.

Components:
-----------
- modes: BehaviorMode enum, GestureType, ModeStateMachine
- parameters: MetaParameters for tuning character, PRESETS
- trends: IdleTrends for multi-tier historical analysis
- states: AggressionState, FlowState, AlmostEngagedState, FeedbackLearning
- system: Main BehaviorSystem coordinator

Usage:
------
    from V3Dev.behavior import BehaviorSystem, load_preset
    
    behavior = BehaviorSystem()
    behavior.apply_preset('responsive')
    behavior.start()
    
    # Each frame:
    output = behavior.update(dt, active_count, passive_count)
    
    # Use output to drive animation
    light.brightness_mult = output.brightness_mult
    light.wander_offset_x = output.wander_x_offset
"""

# Modes
from .modes import (
    BehaviorMode,
    GestureType,
    TimePeriod,
    ModeStateMachine,
    MODE_PARAMS,
    TRANSITION_DURATIONS,
)

# Parameters
from .parameters import (
    MetaParameters,
    PRESETS,
    load_preset,
)

# Trends
from .trends import (
    IdleTrends,
    TrendAnalyzer,
)

# States
from .states import (
    AggressionState,
    FlowState,
    AlmostEngagedState,
    AttractionStrategy,
    AlmostEngagedCandidate,
    EngagementContext,
    FeedbackLearning,
    AGGRESSION_TIME_CAPS,
)

# System
from .system import (
    BehaviorSystem,
    BehaviorOutput,
)

__all__ = [
    # Modes
    'BehaviorMode',
    'GestureType',
    'TimePeriod',
    'ModeStateMachine',
    'MODE_PARAMS',
    'TRANSITION_DURATIONS',
    
    # Parameters
    'MetaParameters',
    'PRESETS',
    'load_preset',
    
    # Trends
    'IdleTrends',
    'TrendAnalyzer',
    
    # States
    'AggressionState',
    'FlowState',
    'AlmostEngagedState',
    'AttractionStrategy',
    'AlmostEngagedCandidate',
    'EngagementContext',
    'FeedbackLearning',
    'AGGRESSION_TIME_CAPS',
    
    # System
    'BehaviorSystem',
    'BehaviorOutput',
]
