"""
Panel Geometry
==============
Panel positions, normals, and coordinate helpers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

# Optional numpy for vector math
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Panel dimensions (cm)
PANEL_SIZE_CM = 60

# Unit spacing (cm) - distance between unit centers
UNIT_SPACING_CM = 80

# Number of units
NUM_UNITS = 4

# Panels per unit
PANELS_PER_UNIT = 3

# Total panels
TOTAL_PANELS = NUM_UNITS * PANELS_PER_UNIT  # 12

# Panel local positions relative to unit center (Y, Z) in cm
# Panel 1 = top, Panel 2 = bottom-front, Panel 3 = bottom-back
PANEL_LOCAL_POSITIONS: Dict[int, Tuple[float, float]] = {
    1: (90, 0),    # Top panel
    2: (30, 12),   # Bottom front (angled forward)
    3: (30, -12),  # Bottom back (angled backward)
}

# Panel angles (degrees from vertical, positive = tilted forward)
PANEL_ANGLES_DEG: Dict[int, float] = {
    1: 0,      # Top: vertical
    2: 22.5,   # Bottom front: tilted forward 22.5°
    3: -22.5,  # Bottom back: tilted backward 22.5°
}


# =============================================================================
# PANEL NORMALS (pre-computed)
# =============================================================================

def _compute_panel_normals() -> Dict[int, Tuple[float, float, float]]:
    """Compute panel normal vectors from angles."""
    normals = {}
    for panel_num, angle_deg in PANEL_ANGLES_DEG.items():
        angle_rad = math.radians(angle_deg)
        # Normal points outward from panel face
        # For panel tilted by angle around X axis:
        # ny = sin(angle), nz = cos(angle)
        normals[panel_num] = (0.0, math.sin(angle_rad), math.cos(angle_rad))
    return normals


PANEL_NORMALS: Dict[int, Tuple[float, float, float]] = _compute_panel_normals()

# Numpy versions if available
if HAS_NUMPY:
    PANEL_NORMALS_NP: Dict[int, 'np.ndarray'] = {
        k: np.array(v) for k, v in PANEL_NORMALS.items()
    }


# =============================================================================
# PANEL DATACLASS
# =============================================================================

@dataclass
class Panel:
    """
    Represents a single light panel.
    
    Attributes:
        unit: Unit index (0-3, right to left)
        panel_num: Panel number within unit (1-3)
        center: 3D center position (x, y, z) in cm
        normal: Normal vector (pointing outward)
        angle_deg: Angle from vertical in degrees
        brightness: Current brightness (0.0-1.0)
        dmx_value: Current DMX value
        dmx_channel: DMX channel number (1-12)
    """
    unit: int
    panel_num: int
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    angle_deg: float = 0.0
    brightness: float = 0.0
    dmx_value: int = 0
    dmx_channel: int = 0
    
    @property
    def key(self) -> Tuple[int, int]:
        """Get (unit, panel_num) key."""
        return (self.unit, self.panel_num)
    
    @property
    def center_x(self) -> float:
        return self.center[0]
    
    @property
    def center_y(self) -> float:
        return self.center[1]
    
    @property
    def center_z(self) -> float:
        return self.center[2]


# =============================================================================
# PANEL SYSTEM
# =============================================================================

class PanelGeometry:
    """
    Manages panel geometry and positions.
    
    Coordinate system:
    - X = 0 at back right corner of Unit 0, negative X goes left
    - Unit centers: Unit 0 at X=-30, Unit 1 at X=-110, Unit 2 at X=-190, Unit 3 at X=-270
    - Y = 0 at floor level, positive Y is up
    - Z = 0 at panel surface, positive Z is toward viewer (into room)
    """
    
    def __init__(self):
        self.panels: Dict[Tuple[int, int], Panel] = {}
        self._build_panels()
    
    def _build_panels(self):
        """Build all panel geometry."""
        for unit in range(NUM_UNITS):
            # Unit 0 center at X=-30, then -110, -190, -270
            unit_x = -(unit * UNIT_SPACING_CM + 30)
            
            for panel_num in range(1, PANELS_PER_UNIT + 1):
                local_y, local_z = PANEL_LOCAL_POSITIONS[panel_num]
                center = (unit_x, local_y, local_z)
                normal = PANEL_NORMALS[panel_num]
                angle = PANEL_ANGLES_DEG[panel_num]
                
                # DMX channels: Unit 0 = 1-3, Unit 1 = 4-6, etc.
                dmx_channel = unit * PANELS_PER_UNIT + panel_num
                
                self.panels[(unit, panel_num)] = Panel(
                    unit=unit,
                    panel_num=panel_num,
                    center=center,
                    normal=normal,
                    angle_deg=angle,
                    dmx_channel=dmx_channel,
                )
    
    def get_panel(self, unit: int, panel_num: int) -> Optional[Panel]:
        """Get a panel by unit and panel number."""
        return self.panels.get((unit, panel_num))
    
    def get_unit_center(self, unit: int) -> Tuple[float, float, float]:
        """Get the center position of a unit."""
        unit_x = -(unit * UNIT_SPACING_CM + 30)
        # Center at average panel Y, Z=0
        return (unit_x, 60.0, 0.0)
    
    def get_all_unit_centers(self) -> Dict[int, Tuple[float, float, float]]:
        """Get center positions of all units."""
        return {unit: self.get_unit_center(unit) for unit in range(NUM_UNITS)}
    
    def get_panel_centers(self) -> List[Tuple[float, float, float]]:
        """Get list of all panel center positions."""
        return [p.center for p in self.panels.values()]
    
    def iter_panels(self):
        """Iterate over all panels in DMX order."""
        for unit in range(NUM_UNITS):
            for panel_num in range(1, PANELS_PER_UNIT + 1):
                yield self.panels[(unit, panel_num)]
    
    def get_dmx_order_panels(self) -> List[Panel]:
        """Get panels in DMX channel order."""
        return list(self.iter_panels())
    
    def get_x_range(self) -> Tuple[float, float]:
        """Get the X range spanned by panels."""
        xs = [p.center_x for p in self.panels.values()]
        half = PANEL_SIZE_CM / 2
        return (min(xs) - half, max(xs) + half)
    
    def get_y_range(self) -> Tuple[float, float]:
        """Get the Y range spanned by panels."""
        ys = [p.center_y for p in self.panels.values()]
        half = PANEL_SIZE_CM / 2
        return (min(ys) - half, max(ys) + half)
    
    def __len__(self) -> int:
        return len(self.panels)
    
    def __iter__(self):
        return self.iter_panels()


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def unit_x_center(unit: int) -> float:
    """Get the X coordinate of a unit's center."""
    return -(unit * UNIT_SPACING_CM + 30)


def panel_center(unit: int, panel_num: int) -> Tuple[float, float, float]:
    """Get the center position of a panel."""
    unit_x = unit_x_center(unit)
    local_y, local_z = PANEL_LOCAL_POSITIONS[panel_num]
    return (unit_x, local_y, local_z)


def dmx_channel_to_panel(channel: int) -> Tuple[int, int]:
    """
    Convert DMX channel (1-12) to (unit, panel_num).
    
    Channel 1-3 = Unit 0, Panel 1-3
    Channel 4-6 = Unit 1, Panel 1-3
    etc.
    """
    channel_0 = channel - 1  # 0-indexed
    unit = channel_0 // PANELS_PER_UNIT
    panel_num = (channel_0 % PANELS_PER_UNIT) + 1
    return (unit, panel_num)


def panel_to_dmx_channel(unit: int, panel_num: int) -> int:
    """Convert (unit, panel_num) to DMX channel (1-12)."""
    return unit * PANELS_PER_UNIT + panel_num


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Singleton panel geometry (immutable after creation)
_panel_geometry: Optional[PanelGeometry] = None


def get_panel_geometry() -> PanelGeometry:
    """Get the global panel geometry instance."""
    global _panel_geometry
    if _panel_geometry is None:
        _panel_geometry = PanelGeometry()
    return _panel_geometry
