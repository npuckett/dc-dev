"""
WebSocket Server
================
Real-time state broadcast to web viewers.
"""

import json
import socket
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Check for websockets library
try:
    import asyncio
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    asyncio = None
    websockets = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WebSocketConfig:
    """WebSocket server configuration."""
    port: int = 8765
    host: str = "0.0.0.0"
    
    # Connection settings
    ping_interval: int = 20      # Send ping every N seconds
    ping_timeout: int = 10       # Wait N seconds for pong
    close_timeout: int = 5       # Allow N seconds for graceful close
    max_message_size: int = 2**20  # 1MB max message size
    
    # Broadcast settings
    broadcast_interval: float = 0.066  # ~15 FPS
    send_timeout: float = 5.0    # Timeout for sending to client
    
    # Restart settings
    max_restarts: int = 10
    restart_delay: float = 5.0   # Initial restart delay
    max_restart_delay: float = 60.0


# =============================================================================
# STATE SERIALIZER
# =============================================================================

class StateSerializer:
    """
    Efficient state serialization with caching.
    Only re-serializes when state actually changes.
    """
    
    def __init__(self):
        self._last_json: str = ""
        self._last_hash: int = 0
    
    def serialize(self, state: dict) -> str:
        """
        Serialize state to JSON, using cache if unchanged.
        
        Args:
            state: State dictionary to serialize
            
        Returns:
            JSON string
        """
        # Compute hash of key values for change detection
        state_hash = self._compute_hash(state)
        
        if state_hash != self._last_hash:
            self._last_hash = state_hash
            self._last_json = json.dumps(state, separators=(',', ':'))
        
        return self._last_json
    
    def _compute_hash(self, state: dict) -> int:
        """Compute a quick hash for change detection."""
        return hash((
            state.get('mode'),
            len(state.get('people', [])),
            state.get('report_version', 0),
            int(state.get('light', {}).get('x', 0) * 10),
            int(state.get('light', {}).get('y', 0) * 10),
        ))
    
    @property
    def cached_json(self) -> str:
        """Get the last serialized JSON."""
        return self._last_json


# =============================================================================
# WEBSOCKET BROADCASTER
# =============================================================================

class WebSocketBroadcaster:
    """
    Broadcasts installation state to web clients.
    
    Features:
    - Efficient state caching (only re-serialize on change)
    - Auto-restart on failure
    - Thread-safe client management
    - Heartbeat with ping/pong
    
    Usage:
        broadcaster = WebSocketBroadcaster()
        broadcaster.start()
        
        # Each frame:
        broadcaster.update_state({
            'mode': 'idle',
            'light': {'x': -150, 'y': 60, 'z': 0},
            'people': [...],
        })
        
        # On shutdown:
        broadcaster.stop()
    """
    
    def __init__(self, config: Optional[WebSocketConfig] = None):
        if not HAS_WEBSOCKETS:
            logger.warning("websockets not available - WebSocket disabled")
        
        self.config = config or WebSocketConfig()
        self.serializer = StateSerializer()
        
        # Client management
        self.clients: Set = set()
        self._clients_lock = None  # Created in async context
        
        # State
        self.current_state: dict = {}
        self.running = False
        
        # Threading
        self.loop = None
        self.server = None
        self.thread = None
        
        # Broadcast control
        self._pending_broadcast = False
        
        # Stats
        self.error_count = 0
        self._messages_sent = 0
        self._start_time = 0.0
    
    @property
    def is_available(self) -> bool:
        """Check if WebSocket library is available."""
        return HAS_WEBSOCKETS
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running and self.thread is not None and self.thread.is_alive()
    
    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self.clients)
    
    def start(self) -> bool:
        """
        Start the WebSocket server in a background thread.
        
        Returns:
            True if started (or already running)
        """
        if not HAS_WEBSOCKETS:
            logger.warning("Cannot start WebSocket - library not available")
            return False
        
        if self.is_running:
            return True
        
        self.running = True
        self._start_time = time.time()
        self.thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name="WebSocketServer"
        )
        self.thread.start()
        return True
    
    def stop(self):
        """Stop the WebSocket server gracefully."""
        self.running = False
        
        if self.server:
            self.server.close()
        
        if self.loop and self.loop.is_running():
            async def cleanup():
                for client in list(self.clients):
                    try:
                        await asyncio.wait_for(client.close(), timeout=2.0)
                    except Exception:
                        pass
                self.clients.clear()
            
            try:
                asyncio.run_coroutine_threadsafe(cleanup(), self.loop)
            except Exception:
                pass
        
        logger.info("WebSocket server stopped")
    
    def update_state(self, state: dict):
        """
        Update the current state and trigger broadcast.
        
        Args:
            state: State dictionary to broadcast
        """
        # Serialize (uses caching)
        self.serializer.serialize(state)
        self.current_state = state
        
        # Schedule broadcast
        if self.loop and self.running and not self._pending_broadcast:
            self._pending_broadcast = True
            
            async def do_broadcast():
                self._pending_broadcast = False
                await self._broadcast()
            
            try:
                asyncio.run_coroutine_threadsafe(do_broadcast(), self.loop)
            except Exception:
                pass
    
    async def _handler(self, websocket):
        """Handle a WebSocket connection."""
        # Thread-safe client add
        async with self._clients_lock:
            self.clients.add(websocket)
        
        client_ip = getattr(websocket, 'remote_address', ('unknown',))[0]
        logger.info(f"WebSocket client connected: {client_ip} (total: {len(self.clients)})")
        
        try:
            # Send current state immediately
            if self.serializer.cached_json:
                await websocket.send(self.serializer.cached_json)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'request_state':
                        await websocket.send(self.serializer.cached_json)
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            logger.debug(f"WebSocket connection closed: {client_ip} ({e})")
        finally:
            async with self._clients_lock:
                self.clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {client_ip} (remaining: {len(self.clients)})")
    
    async def _broadcast(self):
        """Broadcast cached state to all clients."""
        if not self.clients or not self.serializer.cached_json:
            return
        
        # Snapshot clients under lock
        async with self._clients_lock:
            clients = list(self.clients)
        
        if not clients:
            return
        
        dead_clients = []
        json_data = self.serializer.cached_json
        
        async def send_to_client(client):
            try:
                await asyncio.wait_for(
                    client.send(json_data),
                    timeout=self.config.send_timeout
                )
                self._messages_sent += 1
            except asyncio.TimeoutError:
                dead_clients.append(client)
            except Exception:
                dead_clients.append(client)
        
        await asyncio.gather(
            *[send_to_client(c) for c in clients],
            return_exceptions=True
        )
        
        # Remove dead clients
        if dead_clients:
            async with self._clients_lock:
                for client in dead_clients:
                    self.clients.discard(client)
    
    async def _run_server(self):
        """Run the WebSocket server."""
        self._clients_lock = asyncio.Lock()
        
        self.server = await websockets.serve(
            self._handler,
            self.config.host,
            self.config.port,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
            close_timeout=self.config.close_timeout,
            max_size=self.config.max_message_size,
        )
        
        logger.info(f"WebSocket server started on port {self.config.port}")
        self._log_viewer_url()
        
        await self.server.wait_closed()
    
    def _log_viewer_url(self):
        """Log the public viewer URL."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"   Public viewer URL: http://{local_ip}:8080")
        except Exception:
            print(f"   Public viewer: connect to port {self.config.port}")
    
    def _thread_main(self):
        """Main function for WebSocket thread with auto-restart."""
        restart_count = 0
        restart_delay = self.config.restart_delay
        
        while self.running and restart_count < self.config.max_restarts:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                self.loop.run_until_complete(self._run_server())
            except OSError as e:
                restart_count += 1
                logger.error(f"WebSocket server error ({restart_count}): {e}")
                if restart_count < self.config.max_restarts and self.running:
                    logger.info(f"Restarting in {restart_delay}s...")
                    time.sleep(restart_delay)
                    restart_delay = min(restart_delay * 2, self.config.max_restart_delay)
            except Exception as e:
                restart_count += 1
                logger.error(f"WebSocket server error ({restart_count}): {e}")
                if restart_count < self.config.max_restarts and self.running:
                    time.sleep(restart_delay)
                    restart_delay = min(restart_delay * 2, self.config.max_restart_delay)
            finally:
                try:
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    self.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                    self.loop.close()
                except Exception:
                    pass
        
        if restart_count >= self.config.max_restarts:
            logger.error("WebSocket server exceeded max restarts")
        
        self.running = False
    
    def get_stats(self) -> dict:
        """Get server statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            'running': self.is_running,
            'clients': self.client_count,
            'messages_sent': self._messages_sent,
            'error_count': self.error_count,
            'uptime': uptime,
        }


# =============================================================================
# MOCK BROADCASTER (for testing)
# =============================================================================

class MockWebSocketBroadcaster:
    """
    Mock WebSocket broadcaster for testing without network.
    """
    
    def __init__(self, on_state: Optional[Callable[[dict], None]] = None):
        self._on_state = on_state
        self._states: List[dict] = []
        self.running = False
        self.error_count = 0
    
    @property
    def is_available(self) -> bool:
        return True
    
    @property
    def is_running(self) -> bool:
        return self.running
    
    @property
    def client_count(self) -> int:
        return 0
    
    @property
    def clients(self) -> set:
        return set()
    
    def start(self) -> bool:
        self.running = True
        return True
    
    def stop(self):
        self.running = False
    
    def update_state(self, state: dict):
        self._states.append(state)
        if self._on_state:
            self._on_state(state)
    
    def get_stats(self) -> dict:
        return {
            'running': self.running,
            'clients': 0,
            'messages_sent': len(self._states),
            'error_count': 0,
            'uptime': 0,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_broadcaster(mock: bool = False,
                       port: int = 8765,
                       on_state: Optional[Callable] = None) -> WebSocketBroadcaster:
    """
    Create a WebSocket broadcaster.
    
    Args:
        mock: If True, create a mock broadcaster
        port: WebSocket port
        on_state: Callback for mock broadcaster
        
    Returns:
        Broadcaster instance
    """
    if mock:
        return MockWebSocketBroadcaster(on_state)
    
    if not HAS_WEBSOCKETS:
        logger.warning("websockets not available, using mock")
        return MockWebSocketBroadcaster(on_state)
    
    config = WebSocketConfig(port=port)
    return WebSocketBroadcaster(config)
