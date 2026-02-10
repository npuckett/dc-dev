"""
Health Monitoring
=================
System health tracking, uptime, and periodic logging for 24/7 operation.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HealthConfig:
    """Health monitoring configuration."""
    log_interval: float = 300.0       # Log health every 5 minutes
    error_threshold: int = 10         # Errors before warning
    memory_check_interval: float = 60.0  # Check memory every minute
    
    # Auto-restart thresholds
    max_error_rate: float = 0.1       # Max errors per second
    max_memory_mb: int = 500          # Max memory before warning


# =============================================================================
# ERROR TRACKER
# =============================================================================

@dataclass
class ErrorTracker:
    """Tracks errors with rate limiting and categorization."""
    
    # Error counts by category
    counts: Dict[str, int] = field(default_factory=dict)
    
    # Recent errors (for rate calculation)
    recent_errors: List[float] = field(default_factory=list)
    
    # Window for rate calculation
    rate_window: float = 60.0  # 1 minute
    
    def record(self, category: str = "general"):
        """Record an error."""
        now = time.time()
        
        # Increment count
        self.counts[category] = self.counts.get(category, 0) + 1
        
        # Add to recent
        self.recent_errors.append(now)
        
        # Prune old errors
        cutoff = now - self.rate_window
        self.recent_errors = [t for t in self.recent_errors if t > cutoff]
    
    def get_rate(self) -> float:
        """Get current error rate (per second)."""
        if not self.recent_errors:
            return 0.0
        
        now = time.time()
        cutoff = now - self.rate_window
        recent = [t for t in self.recent_errors if t > cutoff]
        
        if not recent:
            return 0.0
        
        return len(recent) / self.rate_window
    
    def get_total(self, category: Optional[str] = None) -> int:
        """Get total error count."""
        if category:
            return self.counts.get(category, 0)
        return sum(self.counts.values())
    
    def reset(self):
        """Reset all counters."""
        self.counts.clear()
        self.recent_errors.clear()


# =============================================================================
# HEALTH MONITOR
# =============================================================================

@dataclass
class HealthStats:
    """Current health statistics."""
    uptime: timedelta = field(default_factory=timedelta)
    frame_count: int = 0
    avg_fps: float = 0.0
    
    # Current state
    mode: str = "unknown"
    active_count: int = 0
    passive_count: int = 0
    
    # Network
    ws_clients: int = 0
    osc_messages: int = 0
    
    # Errors
    error_count: int = 0
    error_rate: float = 0.0
    
    # Memory (optional)
    memory_mb: float = 0.0


class HealthMonitor:
    """
    Monitors system health and logs periodic status.
    
    Features:
    - Uptime tracking
    - FPS monitoring
    - Error rate tracking
    - Periodic health logging
    - Memory monitoring (optional)
    
    Usage:
        health = HealthMonitor()
        health.start()
        
        # Each frame:
        health.tick()
        
        # Periodically:
        health.update_state(mode='idle', active_count=2)
        
        # Check if logging needed:
        if health.should_log():
            logger.info(health.get_log_message())
            health.mark_logged()
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.errors = ErrorTracker()
        
        # Timing
        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_memory_check: float = 0.0
        
        # Counters
        self._frame_count: int = 0
        self._osc_message_count: int = 0
        
        # Current state
        self._mode: str = "unknown"
        self._active_count: int = 0
        self._passive_count: int = 0
        self._ws_clients: int = 0
        
        # Memory
        self._memory_mb: float = 0.0
        
        # Running
        self._running = False
    
    def start(self):
        """Start health monitoring."""
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._running = True
        logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
    
    def tick(self):
        """Record a frame tick."""
        self._frame_count += 1
    
    def record_osc_message(self):
        """Record an OSC message received."""
        self._osc_message_count += 1
    
    def record_error(self, category: str = "general"):
        """Record an error."""
        self.errors.record(category)
    
    def update_state(self, 
                     mode: Optional[str] = None,
                     active_count: Optional[int] = None,
                     passive_count: Optional[int] = None,
                     ws_clients: Optional[int] = None):
        """Update current state values."""
        if mode is not None:
            self._mode = mode
        if active_count is not None:
            self._active_count = active_count
        if passive_count is not None:
            self._passive_count = passive_count
        if ws_clients is not None:
            self._ws_clients = ws_clients
    
    @property
    def uptime(self) -> timedelta:
        """Get current uptime."""
        if not self._start_time:
            return timedelta()
        return timedelta(seconds=int(time.time() - self._start_time))
    
    @property
    def avg_fps(self) -> float:
        """Get average FPS since start."""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed
    
    def should_log(self) -> bool:
        """Check if it's time for periodic health log."""
        return (time.time() - self._last_log_time) >= self.config.log_interval
    
    def mark_logged(self):
        """Mark that health was just logged."""
        self._last_log_time = time.time()
    
    def get_stats(self) -> HealthStats:
        """Get current health statistics."""
        self._check_memory()
        
        return HealthStats(
            uptime=self.uptime,
            frame_count=self._frame_count,
            avg_fps=self.avg_fps,
            mode=self._mode,
            active_count=self._active_count,
            passive_count=self._passive_count,
            ws_clients=self._ws_clients,
            osc_messages=self._osc_message_count,
            error_count=self.errors.get_total(),
            error_rate=self.errors.get_rate(),
            memory_mb=self._memory_mb,
        )
    
    def get_log_message(self) -> str:
        """Get formatted health log message."""
        stats = self.get_stats()
        return (
            f"HEALTH: uptime={stats.uptime}, frames={stats.frame_count}, "
            f"avg_fps={stats.avg_fps:.1f}, mode={stats.mode}, "
            f"active={stats.active_count}, passive={stats.passive_count}, "
            f"ws_clients={stats.ws_clients}, errors={stats.error_count}"
        )
    
    def log_health(self):
        """Log health if interval has passed."""
        if self.should_log():
            logger.info(self.get_log_message())
            self.mark_logged()
    
    def _check_memory(self):
        """Check memory usage (if psutil available)."""
        now = time.time()
        if now - self._last_memory_check < self.config.memory_check_interval:
            return
        
        self._last_memory_check = now
        
        try:
            import psutil
            process = psutil.Process()
            self._memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if self._memory_mb > self.config.max_memory_mb:
                logger.warning(f"High memory usage: {self._memory_mb:.1f}MB")
        except ImportError:
            pass
        except Exception:
            pass
    
    def check_health(self) -> Dict[str, bool]:
        """
        Check overall system health.
        
        Returns:
            Dict of health check names to pass/fail status
        """
        stats = self.get_stats()
        
        return {
            'fps_ok': stats.avg_fps >= 10.0,
            'error_rate_ok': stats.error_rate < self.config.max_error_rate,
            'memory_ok': stats.memory_mb < self.config.max_memory_mb or stats.memory_mb == 0,
        }
    
    def is_healthy(self) -> bool:
        """Check if all health checks pass."""
        return all(self.check_health().values())


# =============================================================================
# UPTIME TRACKER (Simple standalone version)
# =============================================================================

class UptimeTracker:
    """Simple uptime tracking without full health monitoring."""
    
    def __init__(self):
        self._start_time: float = 0.0
        self._frame_count: int = 0
    
    def start(self):
        """Start tracking."""
        self._start_time = time.time()
        self._frame_count = 0
    
    def tick(self):
        """Record a frame."""
        self._frame_count += 1
    
    @property
    def uptime(self) -> timedelta:
        """Get uptime."""
        if not self._start_time:
            return timedelta()
        return timedelta(seconds=int(time.time() - self._start_time))
    
    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time
    
    @property
    def frame_count(self) -> int:
        """Get total frames."""
        return self._frame_count
    
    @property
    def avg_fps(self) -> float:
        """Get average FPS."""
        elapsed = self.uptime_seconds
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed
    
    def format_uptime(self) -> str:
        """Get formatted uptime string."""
        return str(self.uptime)
