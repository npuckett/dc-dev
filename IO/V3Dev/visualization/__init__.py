"""
Visualization Module
====================
Panel rendering, DMX output, and debug visualization.

This module handles converting light positions to panel brightness
and outputting to Art-Net DMX.

Components:
-----------
- panels: Panel geometry, positions, normals
- falloff: Distance-based brightness falloff calculations
- renderer: PanelRenderer converts light position to brightness
- dmx: Art-Net DMX output handling
- debug: Optional pygame-based debug visualization

Usage:
------
    from V3Dev.visualization import (
        PanelRenderer, PointLight, DMXOutput, create_light
    )
    
    # Create components
    renderer = PanelRenderer()
    light = create_light(x=-150, y=60, z=0)
    dmx = DMXOutput()
    
    # Start DMX output
    dmx.start()
    
    # Each frame:
    light.update(dt)
    output = renderer.render(light)
    dmx.send(output.dmx_array)
"""

# Panels
from .panels import (
    Panel,
    PanelGeometry,
    get_panel_geometry,
    unit_x_center,
    panel_center,
    dmx_channel_to_panel,
    panel_to_dmx_channel,
    PANEL_SIZE_CM,
    UNIT_SPACING_CM,
    NUM_UNITS,
    PANELS_PER_UNIT,
    TOTAL_PANELS,
    PANEL_LOCAL_POSITIONS,
    PANEL_ANGLES_DEG,
    PANEL_NORMALS,
)

# Falloff
from .falloff import (
    FalloffType,
    FalloffParams,
    FalloffCalculator,
    get_default_falloff,
    linear_falloff,
    quadratic_falloff,
    smooth_falloff,
    inverse_falloff,
    gaussian_falloff,
    gradient_horizontal,
    gradient_vertical,
    gradient_radial,
    pulse_sine,
    pulse_triangle,
    pulse_breath,
)

# Renderer
from .renderer import (
    PointLight,
    RenderOutput,
    PanelRenderer,
    MultiLightRenderer,
    create_light,
)

# DMX
from .dmx import (
    ArtNetConfig,
    DMXOutput,
    MockDMXOutput,
    create_dmx_output,
)

# Debug (optional - may not have pygame)
try:
    from .debug import (
        DebugViewConfig,
        DebugView2D,
        ConsoleDebug,
        create_debug_view,
        check_dependencies,
    )
    HAS_DEBUG = True
except ImportError:
    HAS_DEBUG = False


__all__ = [
    # Panels
    'Panel',
    'PanelGeometry',
    'get_panel_geometry',
    'unit_x_center',
    'panel_center',
    'dmx_channel_to_panel',
    'panel_to_dmx_channel',
    'PANEL_SIZE_CM',
    'UNIT_SPACING_CM',
    'NUM_UNITS',
    'PANELS_PER_UNIT',
    'TOTAL_PANELS',
    'PANEL_LOCAL_POSITIONS',
    'PANEL_ANGLES_DEG',
    'PANEL_NORMALS',
    
    # Falloff
    'FalloffType',
    'FalloffParams',
    'FalloffCalculator',
    'get_default_falloff',
    'linear_falloff',
    'quadratic_falloff',
    'smooth_falloff',
    'gradient_horizontal',
    'gradient_vertical',
    'gradient_radial',
    'pulse_sine',
    'pulse_triangle',
    'pulse_breath',
    
    # Renderer
    'PointLight',
    'RenderOutput',
    'PanelRenderer',
    'MultiLightRenderer',
    'create_light',
    
    # DMX
    'ArtNetConfig',
    'DMXOutput',
    'MockDMXOutput',
    'create_dmx_output',
    
    # Debug
    'HAS_DEBUG',
]

# Add debug exports if available
if HAS_DEBUG:
    __all__.extend([
        'DebugViewConfig',
        'DebugView2D',
        'ConsoleDebug',
        'create_debug_view',
        'check_dependencies',
    ])
