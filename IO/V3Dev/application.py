"""
Application
===========
Main application class that orchestrates all subsystems.
This is the central coordinator for the V3 light controller.
"""

import os
import sys
import time
import logging
import signal
import atexit
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

# Import V3Dev modules
from .config import zones, hardware, timing
from .tracking import TrackedPersonManager, TrackedPerson
from .behavior import BehaviorSystem, BehaviorMode, MetaParameters, load_preset
from .visualization import (
    PanelRenderer, PointLight, PanelGeometry,
    DMXOutput, MockDMXOutput, ArtNetConfig
)
from .network import (
    WebSocketBroadcaster, MockWebSocketBroadcaster,
    HealthMonitor, SettingsStore
)

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE INSTANCE LOCK (for production - prevents duplicate processes)
# =============================================================================

LOCK_FILE = "/tmp/lightController_v3.lock"
_lock_fd = None

def acquire_single_instance_lock() -> bool:
    """
    Ensure only one instance of the controller is running.
    Uses file locking (fcntl) which works with systemd restarts.
    
    Returns:
        True if lock acquired, False if another instance is running
    """
    global _lock_fd
    
    # Skip on non-Unix systems
    if sys.platform == 'win32':
        return True
    
    try:
        import fcntl
        _lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        logger.info(f"Acquired single instance lock (PID: {os.getpid()})")
        return True
    except (IOError, OSError):
        try:
            with open(LOCK_FILE, 'r') as f:
                existing_pid = f.read().strip()
            logger.error(f"Another instance already running (PID: {existing_pid})")
        except:
            logger.error("Another instance already running")
        return False
    except ImportError:
        # fcntl not available (Windows)
        return True


def release_single_instance_lock():
    """Release the single instance lock."""
    global _lock_fd
    if _lock_fd:
        try:
            import fcntl
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            logger.info("Released single instance lock")
        except:
            pass
        _lock_fd = None


# =============================================================================
# APPLICATION STATE
# =============================================================================

@dataclass
class AppState:
    """Current application state."""
    running: bool = True
    shutdown_requested: bool = False
    frame_count: int = 0
    
    # Display toggles
    show_markers: bool = True
    show_labels: bool = True
    show_trends: bool = True
    show_camera_views: bool = False
    is_fullscreen: bool = True
    
    # Current preset
    current_preset: str = "default"


@dataclass  
class FrameMetrics:
    """Metrics for the current frame."""
    dt: float = 0.0
    active_count: int = 0
    passive_count: int = 0
    osc_messages: int = 0
    fps: float = 0.0


# =============================================================================
# LIGHT CONTROLLER
# =============================================================================

class LightController:
    """
    Manages the point light's position and brightness.
    Simplified from the original PointLight + WanderBehavior combination.
    """
    
    def __init__(self):
        self.light = PointLight()
        
        # Wander state
        self.wander_enabled = True
        self.wander_target: Optional[List[float]] = None
        self.wander_timer = 0.0
        self.wander_interval = 3.0
        
        # Wander bounds (updated by behavior system)
        self.wander_box = {
            'min_x': hardware.WANDER_BOX.min_x,
            'max_x': hardware.WANDER_BOX.max_x,
            'min_y': hardware.WANDER_BOX.min_y,
            'max_y': hardware.WANDER_BOX.max_y,
            'min_z': hardware.WANDER_BOX.min_z,
            'max_z': hardware.WANDER_BOX.max_z,
        }
        
        # Follow target (for tracking people)
        self.follow_target: Optional[List[float]] = None
        self.follow_smoothing = 0.05
        
        # Gesture target (for attention-seeking movements)
        self.gesture_target: Optional[List[float]] = None
    
    def update(self, dt: float):
        """Update light position and state."""
        if self.gesture_target:
            # Gesture takes priority
            self._move_toward(self.gesture_target, dt, speed_mult=2.0)
        elif self.follow_target:
            # Follow person
            self._move_toward(self.follow_target, dt, speed_mult=1.5)
        elif self.wander_enabled:
            # Wander randomly
            self._update_wander(dt)
        
        # Update underlying light
        self.light.update(dt)
    
    def _move_toward(self, target: List[float], dt: float, speed_mult: float = 1.0):
        """Move light toward a target position."""
        pos = list(self.light.position)  # Convert tuple to list
        
        for i in range(3):
            diff = target[i] - pos[i]
            if abs(diff) > 0.1:
                pos[i] += diff * self.follow_smoothing
        
        self.light.position = tuple(pos)  # Set back as tuple
    
    def _update_wander(self, dt: float):
        """Update wandering behavior."""
        import random
        
        self.wander_timer -= dt
        
        if self.wander_timer <= 0 or self.wander_target is None:
            # Pick new target within wander box
            self.wander_target = [
                random.uniform(self.wander_box['min_x'], self.wander_box['max_x']),
                random.uniform(self.wander_box['min_y'], self.wander_box['max_y']),
                random.uniform(self.wander_box['min_z'], self.wander_box['max_z']),
            ]
            self.wander_timer = self.wander_interval
        
        # Move toward wander target
        self._move_toward(self.wander_target, dt)
    
    def update_wander_box(self, box: dict):
        """Update wander boundaries."""
        self.wander_box.update(box)
    
    def set_follow_target(self, x: float, y: float, z: float):
        """Set a target to follow."""
        self.follow_target = [x, y, z]
    
    def clear_follow_target(self):
        """Clear follow target."""
        self.follow_target = None
    
    def set_gesture_target(self, target: List[float]):
        """Set gesture target for attention-seeking."""
        self.gesture_target = list(target)
    
    def clear_gesture_target(self):
        """Clear gesture target."""
        self.gesture_target = None
    
    @property
    def position(self) -> List[float]:
        return list(self.light.position)
    
    @property
    def brightness(self) -> float:
        """Get current brightness (interpolated based on pulse)."""
        pulse = self.light.get_pulse_brightness()  # 0-1
        min_b = self.light.brightness_min
        max_b = self.light.brightness_max
        return min_b + (max_b - min_b) * pulse


# =============================================================================
# APPLICATION
# =============================================================================

class Application:
    """
    Main application class orchestrating all subsystems.
    
    Responsibilities:
    - Initialize all systems
    - Run the main loop
    - Coordinate updates between systems
    - Handle shutdown
    
    Usage:
        app = Application()
        app.run()
    """
    
    def __init__(self, 
                 headless: bool = False,
                 config_path: Optional[str] = None,
                 skip_lock: bool = False):
        """
        Initialize the application.
        
        Args:
            headless: If True, run without GUI/display
            config_path: Path to settings file
            skip_lock: If True, skip single instance lock (for testing)
        """
        self.headless = headless
        self.state = AppState()
        self._has_lock = False
        
        # Acquire single instance lock (unless skipped for testing)
        if not skip_lock:
            if not acquire_single_instance_lock():
                raise RuntimeError("Another instance is already running")
            self._has_lock = True
            atexit.register(release_single_instance_lock)
        
        # Timing
        self._last_time = time.time()
        self._start_time = time.time()
        
        # Initialize subsystems
        self._init_tracking()
        self._init_behavior()
        self._init_rendering()
        self._init_network()
        self._init_settings(config_path)
        
        # Signal handlers
        self._setup_signals()
        
        logger.info("Application initialized")
    
    def _init_tracking(self):
        """Initialize tracking system."""
        self.tracked_manager = TrackedPersonManager()
        
        # Set up callbacks
        self.tracked_manager.on_person_entered = self._on_person_entered
        self.tracked_manager.on_person_left = self._on_person_left
        
        logger.info("Tracking system initialized")
    
    def _init_behavior(self):
        """Initialize behavior system."""
        self.meta_params = MetaParameters()
        self.behavior = BehaviorSystem(meta_params=self.meta_params)
        
        logger.info("Behavior system initialized")
    
    def _init_rendering(self):
        """Initialize rendering and DMX output."""
        # Geometry
        self.panel_geometry = PanelGeometry()
        
        # Light controller
        self.light_controller = LightController()
        
        # Panel renderer
        self.renderer = PanelRenderer(self.panel_geometry)
        
        # DMX output
        if not self.headless:
            try:
                config = ArtNetConfig(
                    target_ip=hardware.ARTNET_TARGET_IP,
                    universe=hardware.ARTNET_UNIVERSE,
                    fps=timing.TARGET_FPS
                )
                self.dmx = DMXOutput(config)
                self.dmx.start()
                logger.info(f"DMX output to {hardware.ARTNET_TARGET_IP}")
            except Exception as e:
                logger.warning(f"DMX init failed: {e}, using mock")
                self.dmx = MockDMXOutput()
        else:
            self.dmx = MockDMXOutput()
        
        logger.info("Rendering system initialized")
    
    def _init_network(self):
        """Initialize network systems."""
        # WebSocket broadcaster
        if not self.headless:
            try:
                self.ws_broadcaster = WebSocketBroadcaster()
                self.ws_broadcaster.start()
                logger.info("WebSocket broadcaster started")
            except Exception as e:
                logger.warning(f"WebSocket init failed: {e}")
                self.ws_broadcaster = MockWebSocketBroadcaster()
        else:
            self.ws_broadcaster = MockWebSocketBroadcaster()
        
        # Health monitor
        self.health = HealthMonitor()
        self.health.start()
        
        logger.info("Network systems initialized")
    
    def _init_settings(self, config_path: Optional[str]):
        """Initialize settings persistence."""
        self.settings = SettingsStore(config_path)
        
        # Load saved settings
        saved = self.settings.load()
        if saved:
            self._apply_settings(saved)
            logger.info(f"Loaded {len(saved)} settings")
    
    def _setup_signals(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating shutdown...")
            self.state.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Cleanup on exit
        atexit.register(self.shutdown)
    
    def _on_person_entered(self, track_id: int, position: tuple, is_active: bool):
        """Called when a person enters tracking."""
        # Could update behavior context here in the future
        logger.debug(f"Person {track_id} entered at {position}")
    
    def _on_person_left(self, track_id: int):
        """Called when a person leaves tracking."""
        # Could update behavior context here in the future
        logger.debug(f"Person {track_id} left")
    
    def _apply_settings(self, settings: dict):
        """Apply loaded settings to systems."""
        # Calibration
        if 'offset_x' in settings:
            self.tracked_manager.offset_x = settings['offset_x']
        if 'offset_z' in settings:
            self.tracked_manager.offset_z = settings['offset_z']
        if 'scale_x' in settings:
            self.tracked_manager.scale_x = settings['scale_x']
        if 'scale_z' in settings:
            self.tracked_manager.scale_z = settings['scale_z']
        
        # Meta parameters
        for param in ['responsiveness', 'energy', 'attention_span', 
                      'sociability', 'exploration', 'memory',
                      'brightness_global', 'speed_global', 'pulse_global']:
            if param in settings:
                setattr(self.meta_params, param, settings[param])
    
    def update(self, dt: float) -> FrameMetrics:
        """
        Run one update cycle.
        
        Args:
            dt: Delta time since last update
            
        Returns:
            Frame metrics
        """
        metrics = FrameMetrics(dt=dt)
        
        # 1. Cleanup stale tracking
        self.tracked_manager.cleanup_stale()
        
        # 2. Get counts
        metrics.active_count = self.tracked_manager.count_active()
        metrics.passive_count = self.tracked_manager.count_passive()
        
        # 3. Update behavior system
        behavior_output = self.behavior.update(
            dt=dt,
            active_count=metrics.active_count,
            passive_count=metrics.passive_count,
        )
        
        # 4. Apply behavior to light controller
        self._apply_behavior_output(behavior_output)
        
        # 5. Update light controller
        self.light_controller.update(dt)
        
        # 6. Render to panels
        render_output = self.renderer.render(self.light_controller.light)
        
        # 7. Send DMX
        if render_output.dmx_values:
            self.dmx.send(render_output.dmx_values)
        
        # 8. Update health
        self.health.tick()
        self.health.update_state(
            mode=self.behavior.get_mode().value,
            active_count=metrics.active_count,
            passive_count=metrics.passive_count,
            ws_clients=self.ws_broadcaster.client_count if hasattr(self.ws_broadcaster, 'client_count') else 0
        )
        
        # 9. Broadcast WebSocket state
        self._broadcast_state(metrics)
        
        # 10. Periodic health logging
        self.health.log_health()
        
        # 11. Save settings if dirty
        self.settings.save_if_dirty()
        
        self.state.frame_count += 1
        return metrics
    
    def _apply_behavior_output(self, output):
        """Apply behavior system output to light controller."""
        light = self.light_controller.light
        
        # Apply brightness multiplier
        base_brightness_min = hardware.DMX_MIN
        base_brightness_max = hardware.DMX_MAX
        light.brightness_min = int(base_brightness_min * output.brightness_mult)
        light.brightness_max = int(base_brightness_max * output.brightness_mult)
        
        # Apply movement multiplier
        light.move_speed = 50.0 * output.move_speed_mult
        
        # Wander center from behavior
        if hasattr(output, 'wander_center_x'):
            self.light_controller.wander_box['min_x'] = output.wander_center_x - 130
            self.light_controller.wander_box['max_x'] = output.wander_center_x + 130
        
        # Gesture target
        if output.gesture_type is not None:
            self.light_controller.set_gesture_target(
                [output.gesture_target_x, 0.0, output.gesture_target_z]
            )
        else:
            self.light_controller.clear_gesture_target()
    
    def _broadcast_state(self, metrics: FrameMetrics):
        """Broadcast current state to WebSocket clients."""
        if not hasattr(self, '_last_broadcast'):
            self._last_broadcast = 0.0
        
        now = time.time()
        if now - self._last_broadcast < timing.WEBSOCKET_BROADCAST_INTERVAL:
            return
        
        self._last_broadcast = now
        
        pos = self.light_controller.position
        state = {
            'type': 'state_update',
            'light': {
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'brightness': self.light_controller.brightness,
            },
            'tracking': {
                'active_count': metrics.active_count,
                'passive_count': metrics.passive_count,
            },
            'behavior': {
                'mode': self.behavior.get_mode().value,
            },
            'health': {
                'uptime': str(self.health.uptime),
                'fps': self.health.avg_fps,
                'frame_count': self.state.frame_count,
            }
        }
        
        self.ws_broadcaster.update_state(state)
    
    def run_frame(self) -> bool:
        """
        Run a single frame.
        
        Returns:
            True if should continue, False if should exit
        """
        if self.state.shutdown_requested:
            return False
        
        # Calculate delta time
        now = time.time()
        dt = min(now - self._last_time, 0.1)  # Cap at 100ms
        self._last_time = now
        
        # Update
        self.update(dt)
        
        return self.state.running
    
    def run(self, max_frames: Optional[int] = None):
        """
        Run the main loop.
        
        Args:
            max_frames: Maximum frames to run (None = infinite)
        """
        logger.info("Starting main loop")
        
        frame = 0
        while self.run_frame():
            frame += 1
            if max_frames and frame >= max_frames:
                break
            
            # Frame rate limiting
            time.sleep(1.0 / timing.TARGET_FPS)
        
        logger.info(f"Main loop ended after {frame} frames")
    
    def shutdown(self):
        """Clean shutdown of all systems."""
        logger.info("Shutting down...")
        
        self.state.running = False
        
        # Stop network
        if hasattr(self, 'ws_broadcaster'):
            self.ws_broadcaster.stop()
        
        if hasattr(self, 'health'):
            self.health.stop()
        
        # Stop DMX
        if hasattr(self, 'dmx'):
            self.dmx.stop()
        
        # Save settings
        if hasattr(self, 'settings'):
            self.settings.save()
        
        # Release instance lock (also done via atexit, but be explicit)
        if self._has_lock:
            release_single_instance_lock()
            self._has_lock = False
        
        logger.info("Shutdown complete")


# =============================================================================
# HEADLESS APPLICATION (FOR TESTING)
# =============================================================================

class HeadlessApplication(Application):
    """
    Headless version for testing without display.
    """
    
    def __init__(self, **kwargs):
        kwargs['headless'] = True
        kwargs['skip_lock'] = kwargs.get('skip_lock', True)  # Skip lock by default for testing
        super().__init__(**kwargs)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_application(headless: bool = False, 
                       config_path: Optional[str] = None,
                       skip_lock: bool = False) -> Application:
    """
    Create and configure an application instance.
    
    Args:
        headless: Run without display
        config_path: Path to settings file
        skip_lock: Skip single instance lock (for testing)
        
    Returns:
        Configured Application instance
    """
    if headless:
        return HeadlessApplication(config_path=config_path, skip_lock=skip_lock)
    return Application(config_path=config_path, skip_lock=skip_lock)
