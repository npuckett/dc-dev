"""Test the behavior module."""
import sys
sys.path.insert(0, '/Users/npmac/Documents/GitHub/dc-dev/IO')

# Test behavior module imports
from V3Dev.behavior import (
    BehaviorMode, GestureType, ModeStateMachine,
    MetaParameters, PRESETS, load_preset,
    AggressionState, FlowState, AlmostEngagedState,
    BehaviorSystem, BehaviorOutput
)

print('✓ All imports successful')

# Test mode machine
mm = ModeStateMachine()
print(f'✓ ModeStateMachine created, mode={mm.current_mode.value}')

# Test preset loading
params = load_preset('responsive')
print(f'✓ Loaded preset: responsiveness={params.responsiveness}')

# Test aggression state
agg = AggressionState()
agg.update(0.016, passive_count=5, active_count=0)
print(f'✓ AggressionState: level={agg.level:.2f}')

# Test flow state
flow = FlowState()
flow.update(0.016, ltr_count=3, rtl_count=1)
print(f'✓ FlowState: direction={flow.direction:.2f}, offset={flow.x_offset:.1f}cm')

# Test almost-engaged
ae = AlmostEngagedState()
ae.update_candidate(1, x=-100, z=300, speed=40, distance_to_active=50)
print(f'✓ AlmostEngaged: candidates={len(ae.candidates)}')

# Test full system
behavior = BehaviorSystem()
behavior.start()
output = behavior.update(0.016, active_count=0, passive_count=3)
print(f'✓ BehaviorSystem: mode={output.mode.value}, aggression={output.aggression_level:.2f}')

# Test with engagement
output2 = behavior.update(0.016, active_count=1, passive_count=2,
    active_people=[{'id': 1, 'x': -150, 'z': 200, 'speed': 30}])
print(f'✓ With engagement: mode={output2.mode.value}')

# Update mode machine to test transition
for _ in range(5):
    mm.update(0.5, active_count=1, passive_count=0)
print(f'✓ After updates: mode={mm.current_mode.value}, duration={mm.mode_duration:.1f}s')

print()
print('All behavior module tests passed!')
