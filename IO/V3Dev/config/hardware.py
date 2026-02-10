"""
Hardware Configuration - Single Source of Truth
================================================
All hardware-related constants: panels, DMX, Art-Net, cameras.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List

# numpy is optional - only used for panel normals
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# =============================================================================
# ART-NET / DMX CONFIGURATION
# =============================================================================

# Art-Net target
ARTNET_TARGET_IP = "10.42.0.200"
ARTNET_UNIVERSE = 0
ARTNET_FPS = 30

# DMX value range (clamping)
DMX_MIN = 1
DMX_MAX = 50

# DMX channel mapping
# Unit 0 = CH 1-3, Unit 1 = CH 4-6, Unit 2 = CH 7-9, Unit 3 = CH 10-12
def get_dmx_channel(unit: int, panel: int) -> int:
    """
    Get DMX channel for a specific unit and panel.
    
    Args:
        unit: Unit number (0-3)
        panel: Panel number within unit (1-3)
    
    Returns:
        DMX channel number (1-12)
    """
    return unit * 3 + panel


# =============================================================================
# PANEL PHYSICAL CONFIGURATION
# =============================================================================

# Panel dimensions (cm)
PANEL_SIZE_CM = 60

# Unit spacing (cm) - distance between unit centers
UNIT_SPACING_CM = 80

# Number of units
NUM_UNITS = 4

# Panels per unit
PANELS_PER_UNIT = 3


@dataclass(frozen=True)
class PanelLocalPosition:
    """Position of panel relative to unit center."""
    y: float  # Height offset (cm)
    z: float  # Depth offset (cm)


# Panel positions relative to unit center (y, z) in cm
PANEL_LOCAL_POSITIONS: Dict[int, PanelLocalPosition] = {
    1: PanelLocalPosition(y=90, z=0),
    2: PanelLocalPosition(y=30, z=12),
    3: PanelLocalPosition(y=30, z=-12),
}

# Panel angles (degrees from vertical)
PANEL_ANGLES: Dict[int, float] = {
    1: 0.0,
    2: 22.5,
    3: -22.5,
}

# Panel normal vectors (pre-computed for lighting calculations)
# Only available if numpy is installed
if NUMPY_AVAILABLE:
    PANEL_NORMALS: Dict[int, 'np.ndarray'] = {
        1: np.array([0.0, 0.0, 1.0]),
        2: np.array([0.0, 0.38268, 0.92388]),
        3: np.array([0.0, -0.38268, 0.92388]),
    }
else:
    # Fallback as tuples
    PANEL_NORMALS: Dict[int, Tuple[float, float, float]] = {
        1: (0.0, 0.0, 1.0),
        2: (0.0, 0.38268, 0.92388),
        3: (0.0, -0.38268, 0.92388),
    }


def get_panel_world_position(unit: int, panel: int) -> Tuple[float, float, float]:
    """
    Calculate world position of a panel center.
    
    Args:
        unit: Unit number (0-3)
        panel: Panel number (1-3)
    
    Returns:
        (x, y, z) world coordinates in cm
    """
    # Unit X positions: Unit 0 at X=-30, each unit 80cm further left
    unit_x = -30 - (unit * UNIT_SPACING_CM)
    
    local = PANEL_LOCAL_POSITIONS[panel]
    return (unit_x, local.y, local.z)


# =============================================================================
# WANDER BOX - Light Movement Bounds
# =============================================================================

@dataclass(frozen=True)
class WanderBounds:
    """Defines the 3D box where the light point can move."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    
    def clamp(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Clamp a position to within bounds."""
        return (
            max(self.min_x, min(self.max_x, x)),
            max(self.min_y, min(self.max_y, y)),
            max(self.min_z, min(self.max_z, z)),
        )
    
    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if position is within bounds."""
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y and
                self.min_z <= z <= self.max_z)
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )


# Light movement bounds (cm)
WANDER_BOX = WanderBounds(
    min_x=-280, max_x=-20,
    min_y=0, max_y=150,
    min_z=-28, max_z=32,
)


# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

# Street level Y coordinate (where tracked people are placed)
STREET_LEVEL_Y = -66

# Camera ledge Y coordinate (cameras are 50cm above street)
CAMERA_LEDGE_Y = -16


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single camera."""
    name: str
    position: Tuple[float, float, float]  # (x, y, z) in cm
    target: Tuple[float, float, float]    # Look-at target
    pitch: float    # X-axis rotation (degrees)
    yaw: float      # Y-axis rotation (degrees)
    roll: float     # Z-axis rotation (degrees)
    fov_h: float    # Horizontal FOV (degrees)
    fov_v: float    # Vertical FOV (degrees)
    color: Tuple[float, float, float, float]  # RGBA for visualization


# Camera positions - aligned with tracking zone front edge
CAMERA_Z = 78  # Front edge of active tracking zone

CAMERAS: Dict[str, CameraConfig] = {
    'Camera 1': CameraConfig(
        name='Camera 1',
        position=(-30, CAMERA_LEDGE_Y, CAMERA_Z),
        target=(-150, STREET_LEVEL_Y, 180),
        pitch=22, yaw=-25, roll=0,
        fov_h=80, fov_v=48,
        color=(1.0, 0.3, 0.3, 1.0),  # Red
    ),
    'Camera 2': CameraConfig(
        name='Camera 2',
        position=(-270, CAMERA_LEDGE_Y, CAMERA_Z),
        target=(-150, STREET_LEVEL_Y, 180),
        pitch=22, yaw=25, roll=0,
        fov_h=80, fov_v=48,
        color=(0.3, 0.3, 1.0, 1.0),  # Blue
    ),
}


# =============================================================================
# CALIBRATION MARKERS
# =============================================================================

MARKER_SIZE_CM = 15  # ArUco marker size

# Marker positions are defined in calibration files, not hardcoded here
# See IO/camera_calibration.json for marker world positions


# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

# OSC settings (receiving tracking data)
OSC_LISTEN_IP = "0.0.0.0"
OSC_LISTEN_PORT = 7000

# WebSocket settings (broadcasting to public viewer)
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
