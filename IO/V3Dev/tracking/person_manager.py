"""
Person Tracking Manager
=======================
Manages tracked people received via OSC.
Uses zone configuration from config module (single source of truth).
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any

# numpy is optional - used for position arrays
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Import zone configuration from single source of truth
from ..config import zones
from ..config.zones import ZoneClassification, STREET_LEVEL_Y
from ..config.timing import PERSON_TIMEOUT


# =============================================================================
# TRACKED PERSON
# =============================================================================

@dataclass
class TrackedPerson:
    """
    Represents a person tracked via OSC.
    
    Positions are in world coordinates (cm).
    Zone classification is computed from position using config.zones.
    """
    track_id: int
    x: float  # World X position (cm)
    z: float  # World Z position (cm)
    y: float = STREET_LEVEL_Y  # Fixed at street level by default
    
    # Timing
    last_update: float = 0.0
    first_seen: float = 0.0
    
    # Computed zone (cached, updated on position change)
    _zone: str = field(default="unknown", repr=False)
    
    def __post_init__(self):
        """Initialize timing and compute zone."""
        now = time.time()
        if self.last_update == 0.0:
            self.last_update = now
        if self.first_seen == 0.0:
            self.first_seen = now
        # Compute initial zone
        self._update_zone()
    
    def _update_zone(self):
        """Update zone classification based on current position."""
        self._zone = zones.classify_position(self.x, self.z)
    
    @property
    def zone(self) -> str:
        """Get current zone classification."""
        return self._zone
    
    @property
    def is_active(self) -> bool:
        """True if person is in active engagement zone."""
        return self._zone == ZoneClassification.ACTIVE
    
    @property
    def is_passive(self) -> bool:
        """True if person is in passive zone (including active)."""
        return self._zone in (ZoneClassification.ACTIVE, ZoneClassification.PASSIVE)
    
    @property
    def is_outside(self) -> bool:
        """True if person is outside all tracking zones."""
        return self._zone == ZoneClassification.OUTSIDE
    
    @property
    def position(self) -> Any:
        """Get position as numpy array (or list if numpy unavailable)."""
        if NUMPY_AVAILABLE:
            return np.array([self.x, self.y, self.z])
        return [self.x, self.y, self.z]
    
    @property
    def position_2d(self) -> Tuple[float, float]:
        """Get 2D position (x, z) for zone calculations."""
        return (self.x, self.z)
    
    @property
    def zone_depth(self) -> float:
        """
        Get depth into active zone (0 at edge, 1 at center, negative if outside).
        Useful for intensity calculations.
        """
        return zones.get_zone_depth(self.x, self.z)
    
    @property
    def dwell_time(self) -> float:
        """Time since first tracked (seconds)."""
        return time.time() - self.first_seen
    
    @property
    def time_since_update(self) -> float:
        """Time since last position update (seconds)."""
        return time.time() - self.last_update
    
    def update_position(self, x: float, z: float, y: Optional[float] = None):
        """
        Update position and recompute zone.
        
        Args:
            x: New X position (cm)
            z: New Z position (cm)
            y: New Y position (cm), defaults to current
        """
        self.x = x
        self.z = z
        if y is not None:
            self.y = y
        self.last_update = time.time()
        self._update_zone()
    
    # Backward compatibility aliases
    def get_position(self) -> Any:
        """Legacy method - use .position property instead."""
        return self.position
    
    def is_in_active_zone(self) -> bool:
        """Legacy method - use .is_active property instead."""
        return self.is_active
    
    def is_in_passive_zone(self) -> bool:
        """Legacy method - use .is_passive property instead."""
        return self.is_passive


# =============================================================================
# TRACKED PERSON MANAGER
# =============================================================================

# Callback type hints (position can be numpy array or list)
PersonCallback = Callable[[int, Any, bool], None]
PositionCallback = Callable[[int, Any], None]
ZoneCallback = Callable[[int, bool, Any], None]
LeaveCallback = Callable[[int], None]


class TrackedPersonManager:
    """
    Manages all tracked people received via OSC.
    
    Features:
    - Thread-safe access to person dictionary
    - Automatic zone classification using config.zones
    - Calibration offsets and scaling
    - Callbacks for behavior system integration
    - Automatic cleanup of stale tracks
    """
    
    def __init__(self, timeout: float = PERSON_TIMEOUT):
        """
        Initialize the person manager.
        
        Args:
            timeout: Seconds before removing stale tracks (default from config)
        """
        self.people: Dict[int, TrackedPerson] = {}
        self.lock = threading.Lock()
        self.timeout = timeout
        
        # Calibration parameters
        self._calibration = CalibrationParams()
        
        # Callbacks for behavior system
        self.on_person_entered: Optional[PersonCallback] = None
        self.on_person_left: Optional[LeaveCallback] = None
        self.on_position_updated: Optional[PositionCallback] = None
        self.on_zone_updated: Optional[ZoneCallback] = None
    
    # =========================================================================
    # Calibration Properties
    # =========================================================================
    
    @property
    def calibration(self) -> 'CalibrationParams':
        """Get calibration parameters."""
        return self._calibration
    
    def set_calibration(self, 
                        offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0,
                        scale_x: float = 1.0, scale_y: float = 1.0, scale_z: float = 1.0,
                        invert_x: bool = False):
        """Set calibration parameters."""
        self._calibration = CalibrationParams(
            offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
            scale_x=scale_x, scale_y=scale_y, scale_z=scale_z,
            invert_x=invert_x
        )
    
    # Convenience property accessors for calibration
    @property
    def offset_x(self) -> float:
        return self._calibration.offset_x
    
    @offset_x.setter
    def offset_x(self, value: float):
        self._calibration.offset_x = value
    
    @property
    def offset_z(self) -> float:
        return self._calibration.offset_z
    
    @offset_z.setter
    def offset_z(self, value: float):
        self._calibration.offset_z = value
    
    @property
    def scale_x(self) -> float:
        return self._calibration.scale_x
    
    @scale_x.setter
    def scale_x(self, value: float):
        self._calibration.scale_x = value
    
    @property
    def scale_z(self) -> float:
        return self._calibration.scale_z
    
    @scale_z.setter
    def scale_z(self, value: float):
        self._calibration.scale_z = value
    
    @property
    def invert_x(self) -> bool:
        return self._calibration.invert_x
    
    @invert_x.setter
    def invert_x(self, value: bool):
        self._calibration.invert_x = value

    # =========================================================================
    # Person Management
    # =========================================================================
    
    def update_person(self, track_id: int, raw_x: float, raw_z: float, 
                      raw_y: Optional[float] = None):
        """
        Update or add a tracked person with calibration applied.
        
        Zone is automatically computed from calibrated position.
        
        Args:
            track_id: Unique ID for this person
            raw_x: Raw X position from tracker
            raw_z: Raw Z position from tracker
            raw_y: Raw Y position (optional, defaults to street level)
        """
        # Apply calibration
        x, y, z = self._calibration.apply(raw_x, raw_z, raw_y)
        
        now = time.time()
        
        with self.lock:
            is_new = track_id not in self.people
            
            if is_new:
                person = TrackedPerson(
                    track_id=track_id,
                    x=x, z=z, y=y,
                    last_update=now,
                    first_seen=now,
                )
                self.people[track_id] = person
                
                # Notify behavior system
                if self.on_person_entered:
                    self.on_person_entered(track_id, person.position, person.is_active)
            else:
                person = self.people[track_id]
                old_zone = person.zone
                person.update_position(x, z, y)
                
                # Notify position update
                if self.on_position_updated:
                    self.on_position_updated(track_id, person.position)
                
                # Notify zone change
                if self.on_zone_updated:
                    self.on_zone_updated(track_id, person.is_active, person.position)
    
    def cleanup_stale(self) -> List[int]:
        """
        Remove people who haven't been updated recently.
        
        Returns:
            List of removed track IDs
        """
        now = time.time()
        removed = []
        
        with self.lock:
            stale_ids = [pid for pid, p in self.people.items() 
                        if now - p.last_update > self.timeout]
            for pid in stale_ids:
                del self.people[pid]
                removed.append(pid)
                if self.on_person_left:
                    self.on_person_left(pid)
        
        return removed
    
    def clear(self):
        """Remove all tracked people."""
        with self.lock:
            for pid in list(self.people.keys()):
                if self.on_person_left:
                    self.on_person_left(pid)
            self.people.clear()
    
    # =========================================================================
    # Queries
    # =========================================================================
    
    def get_all(self) -> List[TrackedPerson]:
        """Get list of all tracked people."""
        with self.lock:
            return list(self.people.values())
    
    def get_person(self, track_id: int) -> Optional[TrackedPerson]:
        """Get a specific tracked person by ID."""
        with self.lock:
            return self.people.get(track_id)
    
    def count(self) -> int:
        """Get total count of tracked people."""
        with self.lock:
            return len(self.people)
    
    def count_active(self) -> int:
        """Count people in active zone."""
        with self.lock:
            return sum(1 for p in self.people.values() if p.is_active)
    
    def count_passive(self) -> int:
        """Count people in passive zone (excludes active)."""
        with self.lock:
            return sum(1 for p in self.people.values() 
                      if p.zone == ZoneClassification.PASSIVE)
    
    def count_in_any_zone(self) -> int:
        """Count people in any tracking zone (active or passive)."""
        with self.lock:
            return sum(1 for p in self.people.values() if p.is_passive)
    
    def get_zone(self, x: float, z: float) -> str:
        """
        Get zone classification for a position.
        
        This method is meant to be used as a zone_checker callback:
            scene.draw_tracked_people(people, zone_checker=manager.get_zone)
        
        Args:
            x: X position (cm)
            z: Z position (cm)
            
        Returns:
            'active', 'passive', or 'outside'
        """
        return zones.classify_position(x, z)
    
    def get_active_positions(self) -> List[Any]:
        """Get positions of people in active zone."""
        with self.lock:
            return [p.position for p in self.people.values() if p.is_active]
    
    def get_passive_positions(self) -> List[Any]:
        """Get positions of people in passive-only zone (not active)."""
        with self.lock:
            return [p.position for p in self.people.values() 
                   if p.zone == ZoneClassification.PASSIVE]
    
    def get_active_people(self) -> List[TrackedPerson]:
        """Get list of people in active zone."""
        with self.lock:
            return [p for p in self.people.values() if p.is_active]
    
    def get_center_of_mass(self, active_only: bool = True) -> Optional[Any]:
        """
        Get center of mass of tracked people.
        
        Args:
            active_only: If True, only consider people in active zone
            
        Returns:
            Position as numpy array (or list if numpy unavailable), or None if no people
        """
        with self.lock:
            if active_only:
                positions = [p.position for p in self.people.values() if p.is_active]
            else:
                positions = [p.position for p in self.people.values()]
            
            if not positions:
                return None
            
            if NUMPY_AVAILABLE:
                return np.mean(positions, axis=0)
            else:
                # Fallback: compute mean manually
                n = len(positions)
                return [sum(p[i] for p in positions) / n for i in range(3)]
    
    def get_closest_to_panels(self) -> Optional[TrackedPerson]:
        """
        Get the person closest to the panel array (lowest Z value in active zone).
        
        Returns:
            TrackedPerson or None if no active people
        """
        with self.lock:
            active = [p for p in self.people.values() if p.is_active]
            if not active:
                return None
            return min(active, key=lambda p: p.z)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """
        Get current tracking statistics.
        
        Returns:
            Dict with counts, positions, etc.
        """
        with self.lock:
            active = [p for p in self.people.values() if p.is_active]
            passive = [p for p in self.people.values() 
                      if p.zone == ZoneClassification.PASSIVE]
            
            return {
                'total_count': len(self.people),
                'active_count': len(active),
                'passive_count': len(passive),
                'active_positions': [p.position.tolist() for p in active],
                'passive_positions': [p.position.tolist() for p in passive],
            }


# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

@dataclass
class CalibrationParams:
    """
    Calibration parameters for transforming raw tracker positions.
    
    Transformation: calibrated = raw * scale + offset
    """
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    invert_x: bool = False
    
    def apply(self, raw_x: float, raw_z: float, 
              raw_y: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Apply calibration to raw position.
        
        Args:
            raw_x: Raw X position
            raw_z: Raw Z position
            raw_y: Raw Y position (optional, defaults to street level)
            
        Returns:
            (x, y, z) calibrated position
        """
        # Optionally invert X
        if self.invert_x:
            raw_x = -raw_x
        
        # Apply scale and offset
        x = raw_x * self.scale_x + self.offset_x
        z = raw_z * self.scale_z + self.offset_z
        
        if raw_y is not None:
            y = raw_y * self.scale_y + self.offset_y
        else:
            y = STREET_LEVEL_Y * self.scale_y + self.offset_y
        
        return (x, y, z)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'offset_z': self.offset_z,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'scale_z': self.scale_z,
            'invert_x': self.invert_x,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationParams':
        """Deserialize from dictionary."""
        return cls(
            offset_x=data.get('offset_x', 0.0),
            offset_y=data.get('offset_y', 0.0),
            offset_z=data.get('offset_z', 0.0),
            scale_x=data.get('scale_x', 1.0),
            scale_y=data.get('scale_y', 1.0),
            scale_z=data.get('scale_z', 1.0),
            invert_x=data.get('invert_x', False),
        )
