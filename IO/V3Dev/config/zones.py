"""
Zone Configuration - Single Source of Truth
============================================
All zone boundaries and zone-related logic lives here.
Import from this module instead of defining zones inline.

Coordinate System (all units in cm):
- X=0 is at back right corner of Unit 0 panel
- Negative X goes left (toward Unit 3)
- Z increases away from panels (toward the street)
- Y is vertical (street level is -66cm)
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class ZoneBounds:
    """Immutable zone boundary definition."""
    min_x: float
    min_z: float
    max_x: float
    max_z: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def depth(self) -> float:
        return self.max_z - self.min_z
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_z + self.max_z) / 2
        )
    
    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_z, max_x, max_z) for compatibility."""
        return (self.min_x, self.min_z, self.max_x, self.max_z)
    
    def contains(self, x: float, z: float) -> bool:
        """Check if a point is inside this zone."""
        return (self.min_x <= x <= self.max_x and 
                self.min_z <= z <= self.max_z)
    
    def distance_to_edge(self, x: float, z: float) -> float:
        """
        Distance from point to nearest edge.
        Positive = inside zone, Negative = outside zone.
        """
        dx = min(x - self.min_x, self.max_x - x)
        dz = min(z - self.min_z, self.max_z - z)
        return min(dx, dz)
    
    def normalized_position(self, x: float, z: float) -> Tuple[float, float]:
        """
        Convert world coordinates to normalized [0,1] within zone.
        Useful for mapping to panel positions.
        """
        nx = (x - self.min_x) / self.width if self.width > 0 else 0.5
        nz = (z - self.min_z) / self.depth if self.depth > 0 else 0.5
        return (max(0, min(1, nx)), max(0, min(1, nz)))


# =============================================================================
# RAW ZONE DEFINITIONS (legacy dict format for compatibility)
# =============================================================================
# These match the original lightController_osc.py format

TRACKZONE_RAW = {
    'width': 260,           # Narrowed from 475 to 260 for better coverage
    'depth': 205,
    'height': 300,
    'offset_z': 78,
    'offset_y': -66,        # Street level (below storefront)
    'center_x': -150,       # Center of 4 panels
}

PASSIVE_TRACKZONE_RAW = {
    'width': 400,           # Narrowed from 650 to 400 for better coverage
    'depth': 270,           # Reduced from 330 to 270 (ends at ~Z=553)
    'height': 300,
    'offset_z': 78 + 205,   # Starts at back of active zone (283cm)
    'offset_y': -66,        # Same street level
    'center_x': -150,       # Centered on panel midline
}


# =============================================================================
# CANONICAL ZONE DEFINITIONS
# =============================================================================
# These are the authoritative computed bounds - all other code imports from here

# Active tracking zone - where people trigger ENGAGED mode
# Computed from raw zone definition
ACTIVE_ZONE = ZoneBounds(
    min_x=TRACKZONE_RAW['center_x'] - TRACKZONE_RAW['width'] / 2,  # -280
    min_z=TRACKZONE_RAW['offset_z'],                                # 78
    max_x=TRACKZONE_RAW['center_x'] + TRACKZONE_RAW['width'] / 2,  # -20
    max_z=TRACKZONE_RAW['offset_z'] + TRACKZONE_RAW['depth'],      # 283
)

# Passive tracking zone - where people are detected but don't trigger engagement
# Note: This zone is BEHIND the active zone (larger Z values)
PASSIVE_ZONE = ZoneBounds(
    min_x=PASSIVE_TRACKZONE_RAW['center_x'] - PASSIVE_TRACKZONE_RAW['width'] / 2,  # -350
    min_z=PASSIVE_TRACKZONE_RAW['offset_z'],                                        # 283
    max_x=PASSIVE_TRACKZONE_RAW['center_x'] + PASSIVE_TRACKZONE_RAW['width'] / 2,  # 50
    max_z=PASSIVE_TRACKZONE_RAW['offset_z'] + PASSIVE_TRACKZONE_RAW['depth'],      # 553
)

# Legacy dict format for backward compatibility
TRACKZONE = TRACKZONE_RAW
PASSIVE_TRACKZONE = PASSIVE_TRACKZONE_RAW


# =============================================================================
# ZONE CLASSIFICATION
# =============================================================================

class ZoneClassification:
    """Enumeration of zone types."""
    ACTIVE = "active"      # Inside active zone
    PASSIVE = "passive"    # In passive zone but not active
    OUTSIDE = "outside"    # Outside all zones


def classify_position(x: float, z: float) -> str:
    """
    Classify a position into zone type.
    
    Args:
        x: X position (cm)
        z: Z position (cm, distance from panels)
    
    Returns:
        ZoneClassification.ACTIVE, PASSIVE, or OUTSIDE
    """
    if ACTIVE_ZONE.contains(x, z):
        return ZoneClassification.ACTIVE
    elif PASSIVE_ZONE.contains(x, z):
        return ZoneClassification.PASSIVE
    else:
        return ZoneClassification.OUTSIDE


def is_in_active_zone(x: float, z: float) -> bool:
    """Quick check if position is in active zone."""
    return ACTIVE_ZONE.contains(x, z)


def is_in_passive_zone(x: float, z: float) -> bool:
    """Quick check if position is in passive zone (NOT including active)."""
    return PASSIVE_ZONE.contains(x, z) and not ACTIVE_ZONE.contains(x, z)


def is_in_any_zone(x: float, z: float) -> bool:
    """Check if position is within any tracking zone."""
    return ACTIVE_ZONE.contains(x, z) or PASSIVE_ZONE.contains(x, z)


# =============================================================================
# ZONE GEOMETRY HELPERS
# =============================================================================

def get_zone_depth(x: float, z: float) -> float:
    """
    Get how deep a person is into the active zone.
    Returns 0 at edge, 1 at center, negative if outside.
    Useful for intensity/engagement calculations.
    """
    if not ACTIVE_ZONE.contains(x, z):
        return -ACTIVE_ZONE.distance_to_edge(x, z) / max(ACTIVE_ZONE.width, ACTIVE_ZONE.depth)
    
    # Distance to nearest edge, normalized
    depth = ACTIVE_ZONE.distance_to_edge(x, z)
    max_depth = min(ACTIVE_ZONE.width, ACTIVE_ZONE.depth) / 2
    return depth / max_depth if max_depth > 0 else 0


def get_zone_edge_proximity(x: float, z: float) -> float:
    """
    Get proximity to zone edge (for edge-aware behaviors).
    Returns 0 at center, 1 at edge.
    """
    return 1.0 - max(0, get_zone_depth(x, z))


# =============================================================================
# FLOOR PLAN CONSTANTS
# =============================================================================
# Physical space dimensions (in cm)

FLOOR_WIDTH = 300   # Panel array width (4 units @ 80cm spacing)
FLOOR_DEPTH = 600   # Tracking depth from panels to street

# Street level Y coordinate (where tracked people are placed)
STREET_LEVEL_Y = -66  # cm below floor level

# Panel mounting area center
PANEL_CENTER_X = -150  # Center of 4 panels
