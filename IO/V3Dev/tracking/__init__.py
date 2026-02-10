"""
Tracking Package
================
Person tracking from OSC messages with zone classification.

Usage:
    from tracking import TrackedPersonManager, TrackedPerson, OSCHandler
    
    # Create manager
    manager = TrackedPersonManager()
    
    # Set calibration
    manager.set_calibration(offset_x=10, scale_x=1.0)
    
    # Create OSC server
    handler, server = create_osc_server(manager)
    
    # In main loop
    manager.cleanup_stale()
    active_people = manager.get_active_people()
"""

from .person_manager import (
    TrackedPerson,
    TrackedPersonManager,
    CalibrationParams,
)

# OSC handler is optional (requires pythonosc)
try:
    from .osc_handler import (
        OSCHandler,
        OSCServerManager,
        create_osc_server,
    )
    OSC_AVAILABLE = True
except ImportError:
    OSCHandler = None
    OSCServerManager = None
    create_osc_server = None
    OSC_AVAILABLE = False

__all__ = [
    # Person management
    'TrackedPerson',
    'TrackedPersonManager',
    'CalibrationParams',
    # OSC handling (may be None if pythonosc not installed)
    'OSCHandler',
    'OSCServerManager',
    'create_osc_server',
    'OSC_AVAILABLE',
]
