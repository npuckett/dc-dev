"""
Trend Analysis System
=====================
Multi-tier trend analysis for behavior modulation.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .modes import TimePeriod


@dataclass
class IdleTrends:
    """
    Processed trend data for IDLE mode behavior.
    
    Tracks activity across multiple time windows:
    - Recent (1 minute): Immediate responsiveness
    - Short (5 minutes): Quick patterns
    - Medium (30 minutes): Session-level patterns
    - Long (1 hour): Sustained patterns
    
    Also tracks historical patterns for the current time period.
    """
    # Recent immediate data (last 1 minute)
    recent_passive_count: int = 0
    recent_active_count: int = 0
    recent_flow_direction: float = 0.0  # -1 = all R→L, +1 = all L→R
    recent_avg_speed: float = 0.0
    
    # Short term (5 minutes)
    short_passive_count: int = 0
    short_active_count: int = 0
    short_flow_direction: float = 0.0
    short_activity_level: float = 0.0  # 0 = dead, 1 = very busy
    
    # Medium term (30 minutes)
    medium_passive_count: int = 0
    medium_active_count: int = 0
    medium_flow_direction: float = 0.0
    medium_activity_level: float = 0.0
    
    # Long term (1 hour)
    long_passive_count: int = 0
    long_active_count: int = 0
    long_flow_direction: float = 0.0
    long_activity_level: float = 0.0
    
    # Time period pattern (historical average for this time period)
    period_typical_count: int = 0
    period_typical_flow: float = 0.0
    period_name: str = "unknown"
    
    # Computed influence values (0-1 normalized)
    activity_anticipation: float = 0.5    # Should we be ready for action?
    flow_momentum: float = 0.0            # Sustained directional momentum
    energy_level: float = 0.5             # Overall energy to match
    
    # Data availability flags
    has_recent_data: bool = False         # Has 1-minute data
    has_short_data: bool = False          # Has 5-minute data
    has_medium_data: bool = False         # Has 30-minute data
    has_long_data: bool = False           # Has 1-hour data
    has_historical_data: bool = False     # Has multi-day historical pattern
    database_error: str = ""              # Last error if any
    
    # Timestamps
    last_update: float = 0.0
    
    def compute_influences(self):
        """
        Compute derived influence values from raw trend data.
        Call after updating raw values from database.
        """
        # Activity anticipation: weighted average of recent activity
        # Higher weight on short-term, lower on long-term
        activity_sum = 0.0
        weight_sum = 0.0
        
        if self.has_recent_data:
            activity_sum += self.recent_passive_count * 4.0
            weight_sum += 4.0
        if self.has_short_data:
            activity_sum += self.short_activity_level * 3.0 * 10  # Normalize
            weight_sum += 3.0
        if self.has_medium_data:
            activity_sum += self.medium_activity_level * 2.0 * 10
            weight_sum += 2.0
        if self.has_long_data:
            activity_sum += self.long_activity_level * 1.0 * 10
            weight_sum += 1.0
        
        if weight_sum > 0:
            # Normalize to 0-1 range (assuming ~20 events/min is "busy")
            raw_activity = activity_sum / weight_sum
            self.activity_anticipation = min(1.0, raw_activity / 20.0)
        
        # Flow momentum: sustained directional flow
        # Only counts if direction is consistent across time windows
        if self.has_short_data and self.has_medium_data:
            if abs(self.short_flow_direction) > 0.3 and abs(self.medium_flow_direction) > 0.3:
                # Check if same direction
                if self.short_flow_direction * self.medium_flow_direction > 0:
                    self.flow_momentum = (abs(self.short_flow_direction) + 
                                         abs(self.medium_flow_direction)) / 2
                    # Preserve sign
                    if self.short_flow_direction < 0:
                        self.flow_momentum = -self.flow_momentum
        
        # Energy level: matches activity but with some lag
        self.energy_level = self.activity_anticipation * 0.7 + 0.3
        
        self.last_update = time.time()
    
    def get_wander_bias(self) -> float:
        """
        Get X-axis bias for wander behavior based on flow.
        
        Returns:
            -1 to +1, negative = bias left, positive = bias right
        """
        # Flow momentum affects wander position
        # If people coming from left, bias right to face them
        return self.flow_momentum * 0.5
    
    def get_energy_modifier(self) -> float:
        """
        Get energy modifier based on activity level.
        
        Returns:
            0.5 to 1.5 multiplier
        """
        return 0.5 + self.energy_level
    
    def to_dict(self) -> dict:
        """Serialize for display/debugging."""
        return {
            'recent': {
                'passive': self.recent_passive_count,
                'active': self.recent_active_count,
                'flow': round(self.recent_flow_direction, 2),
            },
            'short': {
                'passive': self.short_passive_count,
                'active': self.short_active_count,
                'flow': round(self.short_flow_direction, 2),
                'activity': round(self.short_activity_level, 2),
            },
            'medium': {
                'passive': self.medium_passive_count,
                'active': self.medium_active_count,
                'flow': round(self.medium_flow_direction, 2),
                'activity': round(self.medium_activity_level, 2),
            },
            'long': {
                'passive': self.long_passive_count,
                'active': self.long_active_count,
                'flow': round(self.long_flow_direction, 2),
                'activity': round(self.long_activity_level, 2),
            },
            'computed': {
                'anticipation': round(self.activity_anticipation, 2),
                'flow_momentum': round(self.flow_momentum, 2),
                'energy': round(self.energy_level, 2),
            },
            'data_available': {
                'recent': self.has_recent_data,
                'short': self.has_short_data,
                'medium': self.has_medium_data,
                'long': self.has_long_data,
                'historical': self.has_historical_data,
            },
        }


class TrendAnalyzer:
    """
    Analyzes tracking database for trend data.
    
    Runs queries in background to avoid blocking main loop.
    """
    
    # Time windows in seconds
    WINDOW_RECENT = 60       # 1 minute
    WINDOW_SHORT = 300       # 5 minutes
    WINDOW_MEDIUM = 1800     # 30 minutes
    WINDOW_LONG = 3600       # 1 hour
    
    # Update interval (how often to query database)
    UPDATE_INTERVAL = 5.0    # Every 5 seconds
    
    def __init__(self, database=None):
        """
        Initialize trend analyzer.
        
        Args:
            database: TrackingDatabase instance (optional)
        """
        self.database = database
        self.trends = IdleTrends()
        self.last_query_time = 0.0
        self._pending_update = False
    
    def update(self, force: bool = False) -> IdleTrends:
        """
        Update trend data if interval has passed.
        
        Args:
            force: Force update even if interval hasn't passed
            
        Returns:
            Current IdleTrends
        """
        now = time.time()
        
        if force or (now - self.last_query_time > self.UPDATE_INTERVAL):
            self._query_trends()
            self.last_query_time = now
        
        return self.trends
    
    def _query_trends(self):
        """Query database for trend data."""
        if not self.database:
            self.trends.database_error = "No database configured"
            return
        
        try:
            now = time.time()
            
            # Query each time window
            self._query_window('recent', now - self.WINDOW_RECENT, now)
            self._query_window('short', now - self.WINDOW_SHORT, now)
            self._query_window('medium', now - self.WINDOW_MEDIUM, now)
            self._query_window('long', now - self.WINDOW_LONG, now)
            
            # Update time period info
            period = TimePeriod.current()
            self.trends.period_name = period.value
            
            # Compute derived values
            self.trends.compute_influences()
            self.trends.database_error = ""
            
        except Exception as e:
            self.trends.database_error = str(e)
    
    def _query_window(self, window_name: str, start_time: float, end_time: float):
        """Query a single time window from database."""
        try:
            # This would query the actual database
            # For now, just mark as having no data
            if window_name == 'recent':
                self.trends.has_recent_data = False
            elif window_name == 'short':
                self.trends.has_short_data = False
            elif window_name == 'medium':
                self.trends.has_medium_data = False
            elif window_name == 'long':
                self.trends.has_long_data = False
                
        except Exception:
            pass
    
    def get_trends(self) -> IdleTrends:
        """Get current trend data without updating."""
        return self.trends
    
    def get_current_trends(self) -> IdleTrends:
        """Alias for get_trends (for API compatibility)."""
        return self.trends
    
    def has_data(self) -> bool:
        """Check if we have any trend data."""
        return (self.trends.has_recent_data or 
                self.trends.has_short_data or 
                self.trends.has_medium_data or 
                self.trends.has_long_data)
    
    def record_event(self, active_count: int, passive_count: int):
        """
        Record a tracking event (for simple in-memory trending).
        
        Args:
            active_count: Current active zone count
            passive_count: Current passive zone count
        """
        # For simple operation without database, update trends directly
        now = time.time()
        
        # Update recent window data
        self.trends.has_recent_data = True
        self.trends.recent_active_count = active_count
        self.trends.recent_passive_count = passive_count
        
        # Update computed values
        if passive_count > 0 and active_count == 0:
            self.trends.recent_conversion_rate = 0.0
        elif passive_count > 0 and active_count > 0:
            self.trends.recent_conversion_rate = min(1.0, active_count / max(1, passive_count))
