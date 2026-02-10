"""Test the visualization module."""
import sys
sys.path.insert(0, '/Users/npmac/Documents/GitHub/dc-dev/IO')

# Test visualization module imports
from V3Dev.visualization import (
    Panel, PanelGeometry, get_panel_geometry,
    FalloffType, FalloffCalculator, FalloffParams,
    PointLight, PanelRenderer, RenderOutput, create_light,
    DMXOutput, MockDMXOutput,
    PANEL_SIZE_CM, NUM_UNITS, TOTAL_PANELS,
)

print('✓ All imports successful')

# Test panel geometry
geometry = get_panel_geometry()
print(f'✓ PanelGeometry: {len(geometry)} panels')

# Test unit centers
centers = geometry.get_all_unit_centers()
print(f'✓ Unit centers: Unit 0 at x={centers[0][0]:.0f}, Unit 3 at x={centers[3][0]:.0f}')

# Test falloff calculator
falloff = FalloffCalculator(FalloffParams(radius=80, falloff_type=FalloffType.SMOOTH))
b1 = falloff.calculate(0)
b2 = falloff.calculate(40)
b3 = falloff.calculate(80)
print(f'✓ Falloff: d=0→{b1:.2f}, d=40→{b2:.2f}, d=80→{b3:.2f}')

# Test point light
light = create_light(x=-150, y=60, z=0)
print(f'✓ PointLight created at ({light.x}, {light.y}, {light.z})')

# Test light update
light.set_target(-100, 60, 10)
for _ in range(10):
    light.update(0.1)
print(f'✓ After 1s of updates: x={light.x:.1f} (target=-100)')

# Test panel renderer
renderer = PanelRenderer()
output = renderer.render(light)
print(f'✓ Rendered: {len(output.brightness)} panels, DMX array length={len(output.dmx_array)}')

# Check DMX values
print(f'✓ DMX values: {output.dmx_array}')

# Test mock DMX output
received_values = []
def on_dmx(values):
    received_values.append(values)

mock_dmx = MockDMXOutput(callback=on_dmx)
mock_dmx.start()
mock_dmx.send(output.dmx_array)
print(f'✓ MockDMX sent, received={len(received_values)} frames')

# Test real DMX (should fail gracefully without hardware)
real_dmx = DMXOutput()
print(f'✓ Real DMX available: {real_dmx.is_available}')

# Test panel positions
panel_0_1 = geometry.get_panel(0, 1)
panel_3_3 = geometry.get_panel(3, 3)
print(f'✓ Panel (0,1) at x={panel_0_1.center_x:.0f}, Panel (3,3) at x={panel_3_3.center_x:.0f}')

# Test coordinate helpers
from V3Dev.visualization import unit_x_center, panel_to_dmx_channel, dmx_channel_to_panel
print(f'✓ Unit 0 center X: {unit_x_center(0):.0f}')
print(f'✓ Panel (2,3) → DMX channel: {panel_to_dmx_channel(2, 3)}')
print(f'✓ DMX channel 7 → Panel: {dmx_channel_to_panel(7)}')

print()
print('All visualization module tests passed!')
