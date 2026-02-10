"""
OSC Handler
===========
Handles incoming OSC messages for person tracking.
Decoupled from main loop for cleaner architecture.
"""

import time
import threading
import logging
from typing import Optional, Callable

from pythonosc import dispatcher, osc_server

from .person_manager import TrackedPersonManager
from ..config.hardware import OSC_LISTEN_IP, OSC_LISTEN_PORT

logger = logging.getLogger(__name__)


# =============================================================================
# OSC MESSAGE HANDLER
# =============================================================================

class OSCHandler:
    """
    Handles incoming OSC messages for person tracking.
    
    Supported messages:
    - /tracker/person/<id> <x> <z>  - Position of tracked person (cm)
    - /tracker/count <n>            - Number of people currently tracked
    - /tracker/zone/<id> ...        - Zone info (ignored, computed locally)
    
    Zone determination is always done locally based on calibrated position,
    not from incoming OSC data. This allows sliders/offsets to control zones.
    """
    
    def __init__(self, 
                 manager: TrackedPersonManager,
                 database = None,
                 debug_interval: float = 2.0):
        """
        Initialize OSC handler.
        
        Args:
            manager: TrackedPersonManager to update
            database: Optional TrackingDatabase for recording positions
            debug_interval: Seconds between debug log messages
        """
        self.manager = manager
        self.database = database
        self.debug_interval = debug_interval
        
        # Statistics
        self.last_count = 0
        self.message_count = 0
        self.last_debug_time = time.time()
        self.total_messages = 0
        self.errors = 0
        
        # Optional callbacks for additional processing
        self.on_person_message: Optional[Callable] = None
        self.on_count_message: Optional[Callable] = None
    
    def handle_person(self, address: str, *args):
        """
        Handle /tracker/person/<id> messages.
        
        Expected format: /tracker/person/<track_id> <x> <z>
        
        Args:
            address: OSC address (e.g., "/tracker/person/42")
            args: Message arguments (x, z positions)
        """
        try:
            # Extract track_id from address
            parts = address.split('/')
            track_id = int(parts[-1])
            
            if len(args) >= 2:
                raw_x = float(args[0])
                raw_z = float(args[1])
                
                # Update person (calibration applied inside manager)
                self.manager.update_person(track_id, raw_x, raw_z)
                
                # Record to database using CALIBRATED position
                # This ensures database zone classifications match real-time display
                if self.database:
                    person = self.manager.get_person(track_id)
                    if person:
                        self.database.record_position(track_id, person.x, person.z)
                
                # Optional callback
                if self.on_person_message:
                    self.on_person_message(track_id, raw_x, raw_z)
                
                # Statistics
                self.message_count += 1
                self.total_messages += 1
                
                # Debug output at interval
                self._debug_log(track_id, raw_x, raw_z)
                
        except (ValueError, IndexError) as e:
            self.errors += 1
            logger.warning(f"OSC parse error: {e}")
    
    def handle_count(self, address: str, *args):
        """
        Handle /tracker/count messages.
        
        Args:
            address: OSC address
            args: Message arguments (count)
        """
        if args:
            try:
                self.last_count = int(args[0])
                
                if self.on_count_message:
                    self.on_count_message(self.last_count)
            except (ValueError, TypeError):
                pass
    
    def handle_zone(self, address: str, *args):
        """
        Handle /tracker/zone/<id> messages from tracker.
        
        NOTE: We intentionally ignore the tracker's zone determination.
        Zone is always calculated locally based on calibrated position
        and the controller's zone boundaries. This allows the offset/scale
        sliders to properly control zone assignment.
        """
        # Intentionally do nothing - zone is computed locally in update_person()
        pass
    
    def _debug_log(self, track_id: int, x: float, z: float):
        """Print debug info at configured interval."""
        now = time.time()
        if now - self.last_debug_time > self.debug_interval:
            active = self.manager.count_active()
            passive = self.manager.count_passive()
            logger.info(
                f"📥 OSC: {self.message_count} msgs, "
                f"person {track_id} at ({x:.0f}, {z:.0f}), "
                f"active={active}, passive={passive}"
            )
            self.last_debug_time = now
            self.message_count = 0
    
    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            'total_messages': self.total_messages,
            'errors': self.errors,
            'last_count': self.last_count,
            'current_tracked': self.manager.count(),
            'active_count': self.manager.count_active(),
            'passive_count': self.manager.count_passive(),
        }


# =============================================================================
# OSC SERVER MANAGER
# =============================================================================

class OSCServerManager:
    """
    Manages the OSC server lifecycle.
    
    Runs the server in a background thread for non-blocking operation.
    """
    
    def __init__(self, 
                 handler: OSCHandler,
                 ip: str = OSC_LISTEN_IP,
                 port: int = OSC_LISTEN_PORT):
        """
        Initialize OSC server manager.
        
        Args:
            handler: OSCHandler to process messages
            ip: IP address to listen on (default from config)
            port: Port to listen on (default from config)
        """
        self.handler = handler
        self.ip = ip
        self.port = port
        self.server: Optional[osc_server.ThreadingOSCUDPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
    
    def start(self) -> bool:
        """
        Start the OSC server in a background thread.
        
        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("OSC server already running")
            return False
        
        try:
            # Create dispatcher and map handlers
            osc_dispatcher = dispatcher.Dispatcher()
            osc_dispatcher.map("/tracker/person/*", self.handler.handle_person)
            osc_dispatcher.map("/tracker/zone/*", self.handler.handle_zone)
            osc_dispatcher.map("/tracker/count", self.handler.handle_count)
            
            # Create server
            self.server = osc_server.ThreadingOSCUDPServer(
                (self.ip, self.port),
                osc_dispatcher
            )
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True,
                name="OSCServer"
            )
            self.thread.start()
            self.running = True
            
            logger.info(f"🎧 OSC server listening on {self.ip}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start OSC server: {e}")
            return False
    
    def stop(self):
        """Stop the OSC server."""
        if self.server:
            try:
                self.server.shutdown()
            except:
                pass
        self.running = False
        logger.info("OSC server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running and self.thread is not None and self.thread.is_alive()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_osc_server(manager: TrackedPersonManager,
                      database = None,
                      ip: str = OSC_LISTEN_IP,
                      port: int = OSC_LISTEN_PORT) -> tuple:
    """
    Create and start an OSC server for tracking.
    
    Args:
        manager: TrackedPersonManager to update
        database: Optional TrackingDatabase
        ip: IP to listen on
        port: Port to listen on
        
    Returns:
        (OSCHandler, OSCServerManager) tuple
    """
    handler = OSCHandler(manager, database)
    server_manager = OSCServerManager(handler, ip, port)
    server_manager.start()
    return handler, server_manager
