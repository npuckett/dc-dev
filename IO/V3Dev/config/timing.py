"""
Timing Configuration - Single Source of Truth
==============================================
All timing-related constants: intervals, cooldowns, FPS targets.
"""


# =============================================================================
# MAIN LOOP TIMING
# =============================================================================

# Target frame rate for main visualization loop
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS


# =============================================================================
# TRACKING TIMING
# =============================================================================

# How long before a person is considered "gone" (seconds)
PERSON_TIMEOUT = 1.0

# Minimum time between position updates for the same person (seconds)
# Helps prevent processing duplicate OSC messages
POSITION_UPDATE_COOLDOWN = 0.033  # ~30 Hz max

# How long a person must be in active zone to count as "engaged" (seconds)
ENGAGEMENT_THRESHOLD = 0.5


# =============================================================================
# BEHAVIOR TIMING
# =============================================================================

# Mode transition cooldowns (seconds)
MODE_CHANGE_COOLDOWN = 2.0  # Minimum time between mode changes

# Idle mode timing
IDLE_WANDER_MIN_INTERVAL = 2.0  # Minimum seconds between wander target changes
IDLE_WANDER_MAX_INTERVAL = 5.0  # Maximum seconds between wander target changes

# Engaged mode timing
ENGAGED_RESPONSE_SMOOTHING = 0.15  # EMA alpha for position tracking (higher = faster)

# Crowd mode timing
CROWD_DETECTION_THRESHOLD = 3  # People needed to trigger crowd mode
CROWD_EXIT_DELAY = 5.0  # Seconds before exiting crowd mode after count drops


# =============================================================================
# WEBSOCKET TIMING
# =============================================================================

# Broadcast interval for WebSocket updates
WEBSOCKET_BROADCAST_INTERVAL = 0.066  # ~15 FPS (lower than main loop)

# WebSocket connection timeouts
WEBSOCKET_PING_INTERVAL = 20  # Seconds between pings
WEBSOCKET_PING_TIMEOUT = 10   # Seconds to wait for pong
WEBSOCKET_CLOSE_TIMEOUT = 5   # Seconds for graceful close
WEBSOCKET_SEND_TIMEOUT = 5.0  # Seconds for send operation


# =============================================================================
# DATABASE TIMING
# =============================================================================

# Health logging interval (seconds)
HEALTH_LOG_INTERVAL = 300  # Log stats every 5 minutes

# Database pruning interval (seconds)
DB_PRUNE_INTERVAL = 3600  # Prune old records every hour

# Data retention (hours)
DB_RAW_RETENTION_HOURS = 48  # Keep raw events for 48 hours


# =============================================================================
# ANIMATION TIMING
# =============================================================================

# Light movement smoothing
LIGHT_POSITION_SMOOTHING = 0.1  # EMA alpha for light position
LIGHT_BRIGHTNESS_SMOOTHING = 0.15  # EMA alpha for brightness changes

# Wander behavior
WANDER_SPEED_BASE = 50.0  # Base speed in cm/s
WANDER_SPEED_VARIANCE = 20.0  # Random variance in speed


# =============================================================================
# TREND ANALYSIS TIMING
# =============================================================================

# Idle trend windows (seconds)
TREND_WINDOW_1MIN = 60
TREND_WINDOW_5MIN = 300
TREND_WINDOW_30MIN = 1800
TREND_WINDOW_1HR = 3600

# How often to update trend calculations (seconds)
TREND_UPDATE_INTERVAL = 10.0


# =============================================================================
# DAILY REPORT TIMING
# =============================================================================

# When to generate daily reports (hour of day, 24h format)
DAILY_REPORT_HOUR = 0  # Midnight
DAILY_REPORT_MINUTE = 1  # 12:01 AM
