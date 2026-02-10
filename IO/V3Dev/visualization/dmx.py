"""
DMX Output
==========
Art-Net DMX output handling with error recovery.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ART-NET CONFIGURATION
# =============================================================================

@dataclass
class ArtNetConfig:
    """Art-Net configuration."""
    target_ip: str = "10.42.0.200"
    universe: int = 0
    num_channels: int = 12  # 4 units × 3 panels
    fps: int = 30
    
    # Error handling
    reconnect_interval: float = 30.0  # Seconds between reconnection attempts
    max_consecutive_errors: int = 100  # Log warning every N errors


# =============================================================================
# DMX OUTPUT
# =============================================================================

class DMXOutput:
    """
    Manages Art-Net DMX output with error handling and reconnection.
    
    Usage:
        dmx = DMXOutput()
        if dmx.start():
            # Each frame:
            dmx.send([10, 20, 30, ...])  # 12 DMX values
        dmx.stop()
    """
    
    def __init__(self, config: Optional[ArtNetConfig] = None):
        """
        Initialize DMX output.
        
        Args:
            config: Art-Net configuration
        """
        self.config = config or ArtNetConfig()
        self._artnet = None
        self._available = False
        self._running = False
        
        # Error tracking
        self._error_count = 0
        self._last_reconnect_attempt = 0.0
        self._consecutive_errors = 0
        
        # Statistics
        self._frames_sent = 0
        self._last_values: List[int] = []
        
        # Check for library
        self._check_library()
    
    def _check_library(self):
        """Check if stupidArtnet is available."""
        try:
            from stupidArtnet import StupidArtnet
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("stupidArtnet not available - DMX output disabled")
    
    @property
    def is_available(self) -> bool:
        """Check if Art-Net library is available."""
        return self._available
    
    @property
    def is_running(self) -> bool:
        """Check if Art-Net is currently running."""
        return self._running and self._artnet is not None
    
    def start(self) -> bool:
        """
        Start Art-Net output.
        
        Returns:
            True if started successfully
        """
        if not self._available:
            logger.warning("Cannot start Art-Net - library not available")
            return False
        
        try:
            from stupidArtnet import StupidArtnet
            
            self._artnet = StupidArtnet(
                self.config.target_ip,
                self.config.universe,
                self.config.num_channels,
                self.config.fps
            )
            self._artnet.start()
            self._running = True
            self._error_count = 0
            self._consecutive_errors = 0
            
            logger.info(f"Art-Net started: {self.config.target_ip} "
                       f"(universe {self.config.universe}, {self.config.num_channels} channels)")
            return True
            
        except Exception as e:
            logger.error(f"Art-Net start failed: {e}")
            self._artnet = None
            self._running = False
            return False
    
    def stop(self):
        """Stop Art-Net output."""
        if self._artnet:
            try:
                self._artnet.stop()
            except Exception:
                pass
            self._artnet = None
        self._running = False
        logger.info("Art-Net stopped")
    
    def send(self, values: List[int]) -> bool:
        """
        Send DMX values.
        
        Args:
            values: List of DMX values (0-255), should be 12 values
            
        Returns:
            True if sent successfully
        """
        if not self._running or not self._artnet:
            return False
        
        # Validate and pad/truncate values
        values = list(values)[:self.config.num_channels]
        while len(values) < self.config.num_channels:
            values.append(0)
        
        try:
            self._artnet.set(values)
            self._frames_sent += 1
            self._last_values = values.copy()
            
            # Reset error count on success
            if self._consecutive_errors > 0:
                logger.info("Art-Net connection restored")
                self._consecutive_errors = 0
            
            return True
            
        except Exception as e:
            self._error_count += 1
            self._consecutive_errors += 1
            
            # Log warning periodically
            if (self._consecutive_errors == 1 or 
                self._consecutive_errors % self.config.max_consecutive_errors == 0):
                logger.warning(f"Art-Net send error ({self._consecutive_errors}x): {e}")
            
            # Attempt reconnection
            self._try_reconnect()
            
            return False
    
    def _try_reconnect(self):
        """Attempt to reconnect if enough time has passed."""
        now = time.time()
        if now - self._last_reconnect_attempt < self.config.reconnect_interval:
            return
        
        self._last_reconnect_attempt = now
        logger.info("Attempting Art-Net reconnection...")
        
        try:
            from stupidArtnet import StupidArtnet
            
            # Stop old connection
            if self._artnet:
                try:
                    self._artnet.stop()
                except Exception:
                    pass
            
            # Create new connection
            self._artnet = StupidArtnet(
                self.config.target_ip,
                self.config.universe,
                self.config.num_channels,
                self.config.fps
            )
            self._artnet.start()
            
            logger.info("Art-Net reconnected successfully")
            self._consecutive_errors = 0
            
        except Exception as e:
            logger.warning(f"Art-Net reconnection failed: {e}")
    
    def get_stats(self) -> dict:
        """Get output statistics."""
        return {
            'available': self._available,
            'running': self._running,
            'frames_sent': self._frames_sent,
            'error_count': self._error_count,
            'last_values': self._last_values,
        }


# =============================================================================
# MOCK DMX OUTPUT (for testing)
# =============================================================================

class MockDMXOutput:
    """
    Mock DMX output for testing without hardware.
    Implements same interface as DMXOutput.
    """
    
    def __init__(self, callback: Optional[Callable[[List[int]], None]] = None):
        """
        Initialize mock output.
        
        Args:
            callback: Optional function called with values on each send
        """
        self._running = False
        self._callback = callback
        self._last_values: List[int] = []
        self._frames_sent = 0
    
    @property
    def is_available(self) -> bool:
        return True
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def start(self) -> bool:
        self._running = True
        return True
    
    def stop(self):
        self._running = False
    
    def send(self, values: List[int]) -> bool:
        if not self._running:
            return False
        
        self._last_values = list(values)
        self._frames_sent += 1
        
        if self._callback:
            self._callback(values)
        
        return True
    
    def get_stats(self) -> dict:
        return {
            'available': True,
            'running': self._running,
            'frames_sent': self._frames_sent,
            'error_count': 0,
            'last_values': self._last_values,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_dmx_output(mock: bool = False, 
                      target_ip: Optional[str] = None,
                      callback: Optional[Callable] = None) -> 'DMXOutput':
    """
    Create a DMX output instance.
    
    Args:
        mock: If True, create a mock output
        target_ip: Override target IP
        callback: For mock output, callback for sent values
        
    Returns:
        DMX output instance
    """
    if mock:
        return MockDMXOutput(callback)
    
    config = ArtNetConfig()
    if target_ip:
        config.target_ip = target_ip
    
    return DMXOutput(config)
