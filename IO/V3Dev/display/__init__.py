"""
Display Module
==============
OpenGL rendering for the light controller visualization.
Extracts display code from lightController_osc.py for use with V3 modular architecture.
"""

from .opengl_setup import (
    init_display,
    create_fonts,
    setup_3d_projection,
    setup_2d_projection,
    restore_3d_projection,
    CameraController,
    calculate_camera_position,
    clear_frame,
    swap_buffers,
)

from .primitives import (
    draw_box_wireframe,
    draw_panel,
    draw_sphere,
    draw_sphere_wireframe,
    draw_floor,
    draw_tracked_person,
)

from .hud import (
    draw_text_2d,
    draw_text_3d_billboard,
    draw_realtime_trends,
    draw_trends_visualization,
)

from .sliders import Slider, Checkbox

from .scene import SceneRenderer

__all__ = [
    # Setup
    'init_display',
    'create_fonts',
    'setup_3d_projection',
    'setup_2d_projection',
    'restore_3d_projection',
    'CameraController',
    'calculate_camera_position',
    'clear_frame',
    'swap_buffers',
    # Primitives
    'draw_box_wireframe',
    'draw_panel',
    'draw_sphere',
    'draw_sphere_wireframe',
    'draw_floor',
    'draw_tracked_person',
    # HUD
    'draw_text_2d',
    'draw_text_3d_billboard',
    'draw_realtime_trends',
    'draw_trends_visualization',
    # GUI
    'Slider',
    'Checkbox',
    # Scene
    'SceneRenderer',
]
