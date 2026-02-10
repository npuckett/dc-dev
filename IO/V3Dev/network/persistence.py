"""
Settings Persistence
====================
Save/load slider values and configuration to JSON files.
"""

import json
import os
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PersistenceConfig:
    """Configuration for settings persistence."""
    settings_file: str = "slider_settings.json"
    auto_save_interval: float = 2.0  # Minimum time between saves
    backup_on_load: bool = False     # Create backup when loading


# =============================================================================
# SLIDER VALUE
# =============================================================================

@dataclass
class SliderValue:
    """Represents a slider's persistable state."""
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 1.0
    
    def clamp(self) -> float:
        """Get value clamped to valid range."""
        return max(self.min_val, min(self.max_val, self.value))


# =============================================================================
# SETTINGS STORE
# =============================================================================

class SettingsStore:
    """
    Manages saving and loading of slider/checkbox settings.
    
    Features:
    - Rate-limited auto-save
    - Dirty tracking
    - Safe file writing
    - Value validation
    
    Usage:
        store = SettingsStore("slider_settings.json")
        
        # Load on startup
        settings = store.load()
        apply_to_sliders(settings)
        
        # Mark dirty when values change
        store.set("brightness", 0.5)
        store.set("wander_enabled", True)
        
        # Save periodically (rate-limited)
        store.save_if_dirty()
        
        # Or force save
        store.save()
    """
    
    def __init__(self, 
                 file_path: Optional[str] = None,
                 config: Optional[PersistenceConfig] = None):
        """
        Initialize settings store.
        
        Args:
            file_path: Path to settings JSON file
            config: Persistence configuration
        """
        self.config = config or PersistenceConfig()
        
        # Determine file path
        if file_path:
            self.file_path = file_path
        else:
            # Default to same directory as this module
            self.file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",  # Go up from network/
                self.config.settings_file
            )
        
        # State
        self._values: Dict[str, Any] = {}
        self._dirty = False
        self._last_save_time = 0.0
        
        # Callbacks
        self._on_change: List[Callable[[str, Any], None]] = []
    
    def on_change(self, callback: Callable[[str, Any], None]):
        """Register a callback for value changes."""
        self._on_change.append(callback)
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._values.get(name, default)
    
    def set(self, name: str, value: Any):
        """Set a setting value and mark dirty."""
        if self._values.get(name) != value:
            self._values[name] = value
            self._dirty = True
            
            # Notify callbacks
            for cb in self._on_change:
                try:
                    cb(name, value)
                except Exception:
                    pass
    
    def set_many(self, values: Dict[str, Any]):
        """Set multiple values at once."""
        for name, value in values.items():
            if self._values.get(name) != value:
                self._values[name] = value
                self._dirty = True
    
    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty
    
    def load(self) -> Dict[str, Any]:
        """
        Load settings from file.
        
        Returns:
            Dict of setting names to values
        """
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    self._values = json.load(f)
                logger.info(f"📁 Loaded settings from {self.file_path}")
                return self._values.copy()
        except json.JSONDecodeError as e:
            logger.warning(f"Settings file corrupted: {e}")
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")
        
        return {}
    
    def save(self) -> bool:
        """
        Save settings to file.
        
        Returns:
            True if saved successfully
        """
        try:
            # Write to temp file first, then rename (atomic)
            temp_path = self.file_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(self._values, f, indent=2)
            
            # Atomic rename
            os.replace(temp_path, self.file_path)
            
            self._dirty = False
            self._last_save_time = time.time()
            logger.info(f"💾 Saved settings to {self.file_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not save settings: {e}")
            return False
    
    def save_if_dirty(self) -> bool:
        """
        Save if dirty and rate limit has passed.
        
        Returns:
            True if saved
        """
        if not self._dirty:
            return False
        
        now = time.time()
        if now - self._last_save_time < self.config.auto_save_interval:
            return False
        
        return self.save()
    
    def update_from_sliders(self, sliders: Dict[str, Any], 
                            checkboxes: Optional[Dict[str, Any]] = None):
        """
        Update values from slider and checkbox objects.
        
        Args:
            sliders: Dict of name -> slider object (with .value)
            checkboxes: Dict of name -> checkbox object (with .checked)
        """
        for name, slider in sliders.items():
            value = getattr(slider, 'value', None)
            if value is not None:
                self.set(name, value)
        
        if checkboxes:
            for name, checkbox in checkboxes.items():
                checked = getattr(checkbox, 'checked', None)
                if checked is not None:
                    self.set(name, checked)
    
    def apply_to_sliders(self, sliders: Dict[str, Any],
                         checkboxes: Optional[Dict[str, Any]] = None):
        """
        Apply loaded values to slider and checkbox objects.
        
        Args:
            sliders: Dict of name -> slider object
            checkboxes: Dict of name -> checkbox object
        """
        for name, value in self._values.items():
            if name in sliders:
                slider = sliders[name]
                # Clamp to valid range
                min_val = getattr(slider, 'min_val', float('-inf'))
                max_val = getattr(slider, 'max_val', float('inf'))
                clamped = max(min_val, min(max_val, value))
                slider.value = clamped
            elif checkboxes and name in checkboxes:
                checkboxes[name].checked = bool(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all values as dict."""
        return self._values.copy()


# =============================================================================
# TRACKER SETTINGS
# =============================================================================

@dataclass
class TrackerSettings:
    """Settings specific to the tracking system."""
    # Calibration
    offset_x: float = 0.0
    offset_z: float = 0.0
    scale_x: float = 1.0
    scale_z: float = 1.0
    
    # Tracking
    person_timeout: float = 1.0
    min_move_threshold: float = 5.0
    
    # Zone boundaries
    active_zone_min_x: float = -280.0
    active_zone_max_x: float = -20.0
    active_zone_min_z: float = 78.0
    active_zone_max_z: float = 283.0
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrackerSettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# BEHAVIOR SETTINGS
# =============================================================================

@dataclass
class BehaviorSettings:
    """Settings specific to the behavior system."""
    # Brightness
    brightness_min: int = 5
    brightness_max: int = 40
    
    # Movement
    move_speed: float = 50.0
    follow_smoothing: float = 0.05
    wander_interval: float = 3.0
    
    # Falloff
    falloff_radius: float = 80.0
    
    # Pulse
    pulse_speed: float = 2000.0
    pulse_enabled: bool = True
    
    # Wander box
    wander_min_x: float = -280.0
    wander_max_x: float = -20.0
    wander_min_y: float = 0.0
    wander_max_y: float = 150.0
    wander_min_z: float = -28.0
    wander_max_z: float = 32.0
    
    @classmethod
    def from_dict(cls, d: dict) -> 'BehaviorSettings':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# UNIFIED SETTINGS MANAGER
# =============================================================================

class SettingsManager:
    """
    Manages all settings across the application.
    
    Combines tracker, behavior, and UI settings into a unified interface.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize settings manager.
        
        Args:
            base_path: Base directory for settings files
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(base_path, "..")  # Go up from network/
        
        self.base_path = base_path
        
        # Individual stores
        self.sliders = SettingsStore(
            os.path.join(base_path, "slider_settings.json")
        )
        self.tracker = SettingsStore(
            os.path.join(base_path, "tracker_settings.json")
        )
    
    def load_all(self):
        """Load all settings files."""
        self.sliders.load()
        self.tracker.load()
    
    def save_all(self):
        """Save all settings files."""
        self.sliders.save()
        self.tracker.save()
    
    def save_if_dirty(self):
        """Save any dirty settings (rate-limited)."""
        self.sliders.save_if_dirty()
        self.tracker.save_if_dirty()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_settings_path(filename: str) -> str:
    """
    Get the full path for a settings file.
    
    Args:
        filename: Settings filename
        
    Returns:
        Absolute path to settings file
    """
    # Default to IO directory
    io_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(io_dir, filename)


def load_json_settings(filename: str) -> dict:
    """
    Load settings from a JSON file.
    
    Args:
        filename: Settings filename (in IO directory)
        
    Returns:
        Settings dictionary
    """
    path = get_settings_path(filename)
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
    return {}


def save_json_settings(filename: str, settings: dict) -> bool:
    """
    Save settings to a JSON file.
    
    Args:
        filename: Settings filename
        settings: Settings dictionary
        
    Returns:
        True if saved successfully
    """
    path = get_settings_path(filename)
    try:
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.warning(f"Could not save {filename}: {e}")
        return False
