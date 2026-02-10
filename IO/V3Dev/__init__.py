"""
V3Dev - Light Controller Refactoring
====================================
Modular, cleaner architecture for the light controller system.

Modules:
- config: Configuration constants (zones, hardware, timing)
- tracking: Person tracking from OSC messages
- behavior: Behavior system (modes, states, parameters)
- visualization: Panel rendering and DMX output
- network: WebSocket broadcasting, health monitoring, settings persistence
- application: Main application orchestrator

Usage:
    from V3Dev.config import zones, hardware, timing
    from V3Dev.tracking import TrackedPersonManager, OSCHandler
    from V3Dev.behavior import BehaviorSystem, BehaviorMode
    from V3Dev.visualization import PanelRenderer, PointLight, DMXOutput
    from V3Dev.network import WebSocketBroadcaster, HealthMonitor, SettingsStore
    from V3Dev.application import Application, create_application
"""

from . import config
from . import tracking
from . import behavior
from . import visualization
from . import network
from . import application

__version__ = "3.0.0-dev"
