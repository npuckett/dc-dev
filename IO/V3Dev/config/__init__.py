"""
Configuration Package
=====================
Single source of truth for all configuration values.

Usage:
    from config import zones, hardware, timing
    
    # Zone checks
    if zones.is_in_active_zone(x, y):
        ...
    
    # Hardware constants
    channel = hardware.get_dmx_channel(unit, panel)
    
    # Timing constants
    if elapsed > timing.PERSON_TIMEOUT:
        ...
"""

from . import zones
from . import hardware
from . import timing

# Convenience re-exports for common items
from .zones import (
    ACTIVE_ZONE,
    PASSIVE_ZONE,
    TRACKZONE,
    PASSIVE_TRACKZONE,
    TRACKZONE_RAW,
    PASSIVE_TRACKZONE_RAW,
    ZoneBounds,
    ZoneClassification,
    classify_position,
    is_in_active_zone,
    is_in_passive_zone,
    is_in_any_zone,
    get_zone_depth,
    STREET_LEVEL_Y,
)

from .hardware import (
    ARTNET_TARGET_IP,
    ARTNET_UNIVERSE,
    ARTNET_FPS,
    DMX_MIN,
    DMX_MAX,
    PANEL_SIZE_CM,
    NUM_UNITS,
    PANELS_PER_UNIT,
    WANDER_BOX,
    CAMERAS,
    OSC_LISTEN_IP,
    OSC_LISTEN_PORT,
    WEBSOCKET_PORT,
)

from .timing import (
    TARGET_FPS,
    PERSON_TIMEOUT,
    WEBSOCKET_BROADCAST_INTERVAL,
    HEALTH_LOG_INTERVAL,
    DB_PRUNE_INTERVAL,
)

__all__ = [
    'zones',
    'hardware', 
    'timing',
    # Zone exports
    'ACTIVE_ZONE',
    'PASSIVE_ZONE',
    'TRACKZONE',
    'PASSIVE_TRACKZONE',
    'TRACKZONE_RAW',
    'PASSIVE_TRACKZONE_RAW',
    'ZoneBounds',
    'ZoneClassification',
    'classify_position',
    'is_in_active_zone',
    'is_in_passive_zone',
    'is_in_any_zone',
    'get_zone_depth',
    'STREET_LEVEL_Y',
    # Hardware exports
    'ARTNET_TARGET_IP',
    'ARTNET_UNIVERSE',
    'ARTNET_FPS',
    'DMX_MIN',
    'DMX_MAX',
    'PANEL_SIZE_CM',
    'NUM_UNITS',
    'PANELS_PER_UNIT',
    'WANDER_BOX',
    'CAMERAS',
    'OSC_LISTEN_IP',
    'OSC_LISTEN_PORT',
    'WEBSOCKET_PORT',
    # Timing exports
    'TARGET_FPS',
    'PERSON_TIMEOUT',
    'WEBSOCKET_BROADCAST_INTERVAL',
    'HEALTH_LOG_INTERVAL',
    'DB_PRUNE_INTERVAL',
]
