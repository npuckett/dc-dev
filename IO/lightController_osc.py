#!/usr/bin/env python3
"""
3D Light Controller V3 - Development Version

This is a development version with enhanced visual debugging:
- Origin sphere at (0,0,0) with label
- Camera position spheres with labels
- Light panel unit labels with coordinates
- AR marker labels with coordinates
- Camera view overlays showing what each camera sees

Based on lightController_osc.py

Controls:
- Arrow keys: Move light manually (when wander disabled)
- W/S: Move light in Z
- P: Cycle personality presets
- Space: Toggle wandering
- M: Toggle calibration markers
- L: Toggle coordinate labels
- C: Toggle camera view overlays
- Mouse drag (in 3D view): Rotate camera
- Middle mouse drag / Shift+drag: Pan camera
- Scroll: Zoom
- Home: Reset camera view
- Q/ESC: Quit

All units in centimeters.
"""

import sys
import os
import math
import time
import random
import socket
import signal
import threading
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# OSC
from pythonosc import dispatcher, osc_server

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tracking database
from tracking_database import TrackingDatabase

# Behavior system (V2 version with updated coordinate system)
from light_behavior import (
    BehaviorSystem, BehaviorMode, MetaParameters, GestureType,
    PRESETS, load_preset
)

# Try to import Art-Net library
try:
    from stupidArtnet import StupidArtnet
    ARTNET_AVAILABLE = True
except ImportError:
    ARTNET_AVAILABLE = False
    print("stupidArtnet not available - running in visualization-only mode")

# Try to import websockets library for public viewer
try:
    import asyncio
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("websockets not available - public viewer disabled (pip install websockets)")

# JSON is always needed for slider persistence and data serialization
import json

# =============================================================================
# CONFIGURATION (all units in centimeters)
# =============================================================================

# Resolve all relative paths to this script's directory (IO/)
# This ensures the DB is always in the same place regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# OSC settings
OSC_IP = "0.0.0.0"  # Listen on all interfaces
OSC_PORT = 7000

# WebSocket settings (for public viewer)
WEBSOCKET_PORT = 8765
WEBSOCKET_ENABLED = True
WEBSOCKET_BROADCAST_INTERVAL = 0.066  # ~15 FPS for WebSocket (instead of 30)

# Health monitoring (for 24/7 operation)
HEALTH_LOG_INTERVAL = 300  # Log health stats every 5 minutes
DB_PRUNE_INTERVAL = 3600  # Prune old database records every hour
DB_RAW_RETENTION_HOURS = 48  # Keep raw events for 48 hours (aggregated to hourly_stats before deletion)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SINGLE INSTANCE LOCK (for production - prevents duplicate processes)
# =============================================================================

import fcntl
import atexit

LOCK_FILE = "/tmp/lightController.lock"
_lock_fd = None

def acquire_single_instance_lock():
    """Ensure only one instance of the controller is running"""
    global _lock_fd
    try:
        _lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        return True
    except (IOError, OSError) as e:
        try:
            with open(LOCK_FILE, 'r') as f:
                existing_pid = f.read().strip()
            print(f"❌ Another lightController is already running (PID: {existing_pid})")
        except:
            print("❌ Another lightController is already running")
        return False

def release_single_instance_lock():
    """Release the single instance lock (don't remove file - let next instance overwrite)"""
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            # Don't remove lock file - the flock is what matters, not the file
            # Removing the file causes race conditions on rapid restarts
        except:
            pass

# =============================================================================
# SLIDER SETTINGS PERSISTENCE
# =============================================================================

SLIDER_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slider_settings.json')
AUTOTUNE_OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_overrides.json')

def load_slider_settings() -> dict:
    """Load slider settings from JSON file"""
    try:
        if os.path.exists(SLIDER_SETTINGS_FILE):
            with open(SLIDER_SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                logger.info(f"📁 Loaded slider settings from {SLIDER_SETTINGS_FILE}")
                return settings
    except Exception as e:
        logger.warning(f"Could not load slider settings: {e}")
    return {}

def save_slider_settings(all_sliders: dict, checkboxes: dict = None):
    """Save slider and checkbox settings to JSON file"""
    try:
        settings = {name: slider.value for name, slider in all_sliders.items()}
        # Also save checkbox states
        if checkboxes:
            for name, checkbox in checkboxes.items():
                settings[name] = checkbox.checked
        with open(SLIDER_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"💾 Saved slider settings to {SLIDER_SETTINGS_FILE}")
    except Exception as e:
        logger.warning(f"Could not save slider settings: {e}")

def apply_slider_settings(all_sliders: dict, settings: dict, checkboxes: dict = None):
    """Apply loaded settings to sliders and checkboxes"""
    for name, value in settings.items():
        if name in all_sliders:
            slider = all_sliders[name]
            # Clamp to valid range
            clamped_value = max(slider.min_val, min(slider.max_val, value))
            slider.value = clamped_value
        elif checkboxes and name in checkboxes:
            checkboxes[name].checked = bool(value)


# =============================================================================
# DAILY REPORT SYSTEM
# =============================================================================

@dataclass
class HourlyTrend:
    """Trend data for a single hour"""
    hour: int  # 0-23
    total_people: int
    active_count: int
    passive_count: int
    avg_speed: float
    flow_left_to_right: int
    flow_right_to_left: int


@dataclass
class DailyReport:
    """Daily analysis report"""
    date: str  # YYYY-MM-DD
    generated_at: str  # ISO timestamp
    
    # Summary metrics
    total_unique_people: int
    total_active_zone_visits: int
    total_passive_zone_count: int
    overall_avg_speed: float
    
    # Peak times
    peak_hour: int  # 0-23
    peak_hour_count: int
    quietest_hour: int
    quietest_hour_count: int
    
    # Flow analysis
    dominant_flow: str  # 'left_to_right', 'right_to_left', or 'balanced'
    flow_balance: float  # -1.0 to +1.0
    
    # Hourly breakdown
    hourly_trends: List[HourlyTrend] = field(default_factory=list)
    
    # Light behavior summary
    mode_distribution: Dict[str, float] = field(default_factory=dict)
    position_entropy: float = 0.0
    
    # Auto-tuning strategy analysis
    auto_tuning_analysis: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            'date': self.date,
            'generated_at': self.generated_at,
            'summary': {
                'total_unique_people': self.total_unique_people,
                'total_active_zone_visits': self.total_active_zone_visits,
                'total_passive_zone_count': self.total_passive_zone_count,
                'overall_avg_speed': round(self.overall_avg_speed, 1),
            },
            'peak_times': {
                'peak_hour': self.peak_hour,
                'peak_hour_count': self.peak_hour_count,
                'quietest_hour': self.quietest_hour,
                'quietest_hour_count': self.quietest_hour_count,
            },
            'flow': {
                'dominant_flow': self.dominant_flow,
                'flow_balance': round(self.flow_balance, 2),
            },
            'hourly_trends': [
                {
                    'hour': h.hour,
                    'total_people': h.total_people,
                    'active_count': h.active_count,
                    'passive_count': h.passive_count,
                    'avg_speed': round(h.avg_speed, 1),
                    'flow_ltr': h.flow_left_to_right,
                    'flow_rtl': h.flow_right_to_left,
                }
                for h in self.hourly_trends
            ],
            'light_behavior': {
                'mode_distribution': {k: round(v, 3) for k, v in self.mode_distribution.items()},
                'position_entropy': round(self.position_entropy, 3),
            },
            'auto_tuning': self.auto_tuning_analysis,
        }


class DailyReportGenerator:
    """Generates daily analysis reports from tracking data"""
    
    def __init__(self, database: TrackingDatabase):
        self.database = database
        self.last_report: Optional[DailyReport] = None
        self.report_history: List[DailyReport] = []
    
    def generate_report(self, date: Optional[datetime] = None) -> DailyReport:
        """
        Generate a report for the specified date (defaults to yesterday).
        
        Args:
            date: The date to analyze (defaults to yesterday)
        
        Returns:
            DailyReport with analysis results
        """
        if date is None:
            # Report for yesterday (12:01 AM trigger means yesterday's data)
            date = datetime.now() - timedelta(days=1)
        
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"📊 Generating daily report for {date_str}...")
        
        # Query tracking events for the day
        start_ts = datetime(date.year, date.month, date.day, 0, 0, 0).timestamp()
        end_ts = datetime(date.year, date.month, date.day, 23, 59, 59).timestamp()
        
        hourly_trends = []
        total_people = 0
        total_active = 0
        total_passive = 0
        total_speed_sum = 0.0
        speed_count = 0
        total_ltr = 0
        total_rtl = 0
        peak_hour = 0
        peak_count = 0
        quietest_hour = 0
        quietest_count = float('inf')
        
        with self.database.lock:
            cursor = self.database.conn.cursor()
            
            # Get hourly breakdown
            for hour in range(24):
                hour_start = datetime(date.year, date.month, date.day, hour, 0, 0).timestamp()
                hour_end = datetime(date.year, date.month, date.day, hour, 59, 59).timestamp()
                
                cursor.execute('''
                    SELECT 
                        COUNT(DISTINCT person_id) as unique_people,
                        AVG(speed) as avg_speed,
                        SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active_events,
                        SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive_events,
                        SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                        SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                    FROM tracking_events
                    WHERE timestamp >= ? AND timestamp <= ?
                ''', (hour_start, hour_end))
                
                row = cursor.fetchone()
                people = row['unique_people'] or 0
                active = row['active_events'] or 0
                passive = row['passive_events'] or 0
                avg_speed = row['avg_speed'] or 0.0
                ltr = row['ltr'] or 0
                rtl = row['rtl'] or 0
                
                hourly_trends.append(HourlyTrend(
                    hour=hour,
                    total_people=people,
                    active_count=active,
                    passive_count=passive,
                    avg_speed=avg_speed,
                    flow_left_to_right=ltr,
                    flow_right_to_left=rtl,
                ))
                
                # Track peak/quietest
                if people > peak_count:
                    peak_count = people
                    peak_hour = hour
                if people < quietest_count:
                    quietest_count = people
                    quietest_hour = hour
                
                # Accumulate totals
                total_people += people
                total_active += active
                total_passive += passive
                total_ltr += ltr
                total_rtl += rtl
                if avg_speed > 0:
                    total_speed_sum += avg_speed
                    speed_count += 1
            
            # Get unique people for the entire day
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) as unique_people
                FROM tracking_events
                WHERE timestamp >= ? AND timestamp <= ?
            ''', (start_ts, end_ts))
            row = cursor.fetchone()
            unique_people = row['unique_people'] or 0
        
        # Calculate flow balance
        total_flow = total_ltr + total_rtl
        flow_balance = 0.0
        if total_flow > 0:
            flow_balance = (total_ltr - total_rtl) / total_flow
        
        dominant_flow = 'balanced'
        if flow_balance > 0.3:
            dominant_flow = 'left_to_right'
        elif flow_balance < -0.3:
            dominant_flow = 'right_to_left'
        
        # Get light behavior stats
        mode_dist = self.database.get_mode_distribution(24)
        pos_entropy = self.database.get_position_entropy(60 * 24)  # Full day
        
        # Get auto-tuning strategy analysis
        tuning_analysis = {}
        try:
            tuning_analysis = self.database.get_daily_adjustments(date_str)
        except Exception as e:
            logger.warning(f"Auto-tuning analysis failed: {e}")
            tuning_analysis = {'total_adjustments': 0, 'strategy_summary': 'Analysis unavailable.'}
        
        # Create report
        report = DailyReport(
            date=date_str,
            generated_at=datetime.now().isoformat(),
            total_unique_people=unique_people,
            total_active_zone_visits=total_active,
            total_passive_zone_count=total_passive,
            overall_avg_speed=total_speed_sum / speed_count if speed_count > 0 else 0.0,
            peak_hour=peak_hour,
            peak_hour_count=peak_count,
            quietest_hour=quietest_hour,
            quietest_hour_count=int(quietest_count) if quietest_count != float('inf') else 0,
            dominant_flow=dominant_flow,
            flow_balance=flow_balance,
            hourly_trends=hourly_trends,
            mode_distribution=mode_dist,
            position_entropy=pos_entropy,
            auto_tuning_analysis=tuning_analysis,
        )
        
        self.last_report = report
        self.report_history.append(report)
        
        # Keep only last 30 days of reports in memory
        if len(self.report_history) > 30:
            self.report_history = self.report_history[-30:]
        
        logger.info(f"📊 Report generated: {unique_people} unique people, peak at {peak_hour}:00 ({peak_count})")
        return report


class DailyReportScheduler:
    """Schedules daily report generation at 12:01 AM"""
    
    def __init__(self, report_generator: DailyReportGenerator, 
                 ws_broadcaster: 'WebSocketBroadcaster' = None,
                 on_report_ready: callable = None):
        self.report_generator = report_generator
        self.ws_broadcaster = ws_broadcaster
        self.on_report_ready = on_report_ready
        self.thread = None
        self.running = False
        self.paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
    
    def start(self):
        """Start the scheduler thread"""
        self.running = True
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.thread.start()
        logger.info("📅 Daily report scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        self._pause_event.set()  # Unblock if paused
    
    def pause_tracking(self):
        """Pause tracking during report generation"""
        self.paused = True
        self._pause_event.clear()
        logger.info("⏸️ Tracking paused for daily report generation")
    
    def resume_tracking(self):
        """Resume tracking after report generation"""
        self.paused = False
        self._pause_event.set()
        logger.info("▶️ Tracking resumed")
    
    def is_paused(self) -> bool:
        """Check if tracking is paused"""
        return self.paused
    
    def _scheduler_loop(self):
        """Main scheduler loop - checks time and triggers report at 12:01 AM"""
        last_report_date = None
        
        while self.running:
            now = datetime.now()
            
            # Check if it's 12:01 AM and we haven't generated today's report
            if now.hour == 0 and now.minute >= 1 and now.minute < 5:
                today_str = now.strftime('%Y-%m-%d')
                
                if last_report_date != today_str:
                    try:
                        # Pause tracking
                        self.pause_tracking()
                        
                        # Wait a moment for pending data to settle
                        time.sleep(2)
                        
                        # Generate report for yesterday
                        report = self.report_generator.generate_report()
                        
                        # Broadcast over WebSocket
                        if self.ws_broadcaster:
                            self._broadcast_report(report)
                        
                        # Callback
                        if self.on_report_ready:
                            self.on_report_ready(report)
                        
                        last_report_date = today_str
                        
                    except Exception as e:
                        logger.error(f"Error generating daily report: {e}")
                    finally:
                        # Resume tracking
                        self.resume_tracking()
            
            # Sleep for 30 seconds before next check
            time.sleep(30)
    
    def _broadcast_report(self, report: DailyReport):
        """Broadcast report over WebSocket"""
        if not self.ws_broadcaster:
            return
        
        try:
            state = {
                'type': 'daily_report',
                'report': report.to_dict()
            }
            self.ws_broadcaster.update_state(state)
            logger.info("📡 Daily report broadcast over WebSocket")
        except Exception as e:
            logger.error(f"Error broadcasting report: {e}")
    
    def generate_now(self) -> Optional[DailyReport]:
        """Manually trigger report generation (for testing)"""
        try:
            self.pause_tracking()
            time.sleep(1)
            report = self.report_generator.generate_report()
            if self.ws_broadcaster:
                self._broadcast_report(report)
            if self.on_report_ready:
                self.on_report_ready(report)
            return report
        except Exception as e:
            logger.error(f"Error generating manual report: {e}")
            return None
        finally:
            self.resume_tracking()


# Art-Net settings
TARGET_IP = "10.42.0.200"
UNIVERSE = 0
FPS = 30

# DMX range
DMX_MIN = 1
DMX_MAX = 255

# Panel dimensions (cm)
PANEL_SIZE = 60

# Unit spacing (cm)
UNIT_SPACING = 80

# Panel positions relative to unit center (y, z) in cm
PANEL_LOCAL_POSITIONS = {
    1: (90, 0),
    2: (30, 12),
    3: (30, -12),
}

# Panel angles (degrees from vertical)
PANEL_ANGLES = {
    1: 0,
    2: 22.5,
    3: -22.5,
}

# Panel normals
PANEL_NORMALS = {
    1: np.array([0.0, 0.0, 1.0]),
    2: np.array([0.0, 0.38268, 0.92388]),
    3: np.array([0.0, -0.38268, 0.92388]),
}

# Trackzone (cm) - defines the ACTIVE tracking area (engaging with installation)
# Coordinate system: X=0 is back right corner of Unit 0 panel, negative X goes left
# Panels span from X=0 to X=-300 (right edge at 0, 4 units with 80cm spacing, panel width 60cm)
# OPTIMIZED: Narrowed X width to match camera FOV coverage for better accuracy
TRACKZONE = {
    'width': 400,           # Matched to passive zone width
    'depth': 205,
    'height': 300,
    'offset_z': 78,
    'offset_y': -66,        # Street level (below storefront)
    'center_x': -150,       # Center of 4 panels
}

# Passive trackzone (cm) - people passing by on sidewalk, not engaging
# Starts at back of active trackzone, extends further out
# OPTIMIZED: Narrowed width and reduced depth for reliable detection
PASSIVE_TRACKZONE = {
    'width': 400,           # Narrowed from 650 to 400 for better coverage
    'depth': 270,           # Reduced from 330 to 270 (ends at ~Z=553)
    'height': 300,
    'offset_z': 78 + 205,   # Starts at back of active zone (283cm)
    'offset_y': -66,        # Same street level
    'center_x': -150,       # Centered on panel midline
}

# Street level Y coordinate (where tracked people are placed)
STREET_LEVEL_Y = -66
CAMERA_LEDGE_Y = -15  # Cameras are 51cm above street (15cm below floor)

# Wander box (cm) - where the light can move
# X range covers panels (Unit 0 at X=-30 to Unit 3 at X=-270) plus margin
WANDER_BOX = {
    'min_x': -280, 'max_x': -20,
    'min_y': 0, 'max_y': 150,
    'min_z': -28, 'max_z': 32,
}

# =============================================================================
# CAMERA POSITIONS (for visualization)
# =============================================================================

# Camera positions in world coordinates (cm)
# Cameras are at front edge of active tracking zone (Z=78), 15cm below floor (Y=-15)
# Camera 1 is on the RIGHT (near X=0), Camera 2 is on the LEFT (more negative X)
# 
# CAMERA ANGLE RECOMMENDATIONS:
# Both cameras should be angled inward toward the center of the tracking zone (X=-150)
# With 80° horizontal FOV (Reolink RLC-520A):
#   - Camera 1 at X=-30: Angle ~50° LEFT (toward -X) to aim at zone center
#   - Camera 2 at X=-270: Angle ~50° RIGHT (toward +X) to aim at zone center
# This creates an overlap zone in the center (X=-200 to X=-100) for stereo matching
#
CAMERA_Y = -15  # 15cm below floor level (Y=0)
CAMERA_Z = TRACKZONE['offset_z']  # Front edge of active zone = 78

CAMERA_POSITIONS = {
    'Camera 1': {
        'pos': (-30, CAMERA_Y, CAMERA_Z),  # Aligned with Unit 0 center
        'desc': 'Right camera - angled toward center',
        'color': (1.0, 0.3, 0.3, 1.0),  # Red
        'target': (-150, STREET_LEVEL_Y, 180),  # Aim at center of active zone
        # Rotation angles (Euler XYZ order, degrees)
        'rotation': {
            'pitch': 22,   # X-axis: tilted down 22° (increased for better ground coverage)
            'yaw': -25,    # Y-axis: rotated 25° left (reduced to see marker 0)
            'roll': 0,     # Z-axis: level (no tilt)
        },
        'fov': {'horizontal': 80, 'vertical': 48},  # Reolink RLC-520A specs
    },
    'Camera 2': {
        'pos': (-270, CAMERA_Y, CAMERA_Z),  # Aligned with Unit 3 center
        'desc': 'Left camera - angled toward center',
        'color': (0.3, 0.3, 1.0, 1.0),  # Blue
        'target': (-150, STREET_LEVEL_Y, 180),  # Aim at center of active zone
        # Rotation angles (Euler XYZ order, degrees)
        'rotation': {
            'pitch': 22,   # X-axis: tilted down 22° (increased for better ground coverage)
            'yaw': 25,     # Y-axis: rotated 25° right (reduced to see marker 2)
            'roll': 0,     # Z-axis: level (no tilt)
        },
        'fov': {'horizontal': 80, 'vertical': 48},  # Reolink RLC-520A specs
    },
}

# =============================================================================
# CALIBRATION MARKERS
# =============================================================================

MARKER_SIZE = 20  # cm - ArUco marker size

# Marker positions: (X, Y, Z) in centimeters
# Coordinate system: X=0 at back right corner of Unit 0 panel, negative X goes left
# Marker 0 is on the RIGHT, Marker 2 is on the LEFT
# Front row (0,1,2): 90cm from front edge of tracking zone (Z=78), so Z=168
# Back row (3,6,4): 51cm behind front row, so Z=219
# Marker 5: ~550cm from cameras (Z=78+550=628) on subway wall
MARKER_POSITIONS = {
    0: {'pos': (-30, STREET_LEVEL_Y, 168), 'desc': 'Right front', 'camera': 'Cam 1', 'vertical': False},
    1: {'pos': (-150, STREET_LEVEL_Y, 168), 'desc': 'Center front (SHARED)', 'camera': 'Both', 'vertical': False},
    2: {'pos': (-270, STREET_LEVEL_Y, 168), 'desc': 'Left front', 'camera': 'Cam 2', 'vertical': False},
    3: {'pos': (-30, STREET_LEVEL_Y, 219), 'desc': 'Right back', 'camera': 'Cam 1', 'vertical': False},
    4: {'pos': (-270, STREET_LEVEL_Y, 219), 'desc': 'Left back', 'camera': 'Cam 2', 'vertical': False},
    5: {'pos': (-150, CAMERA_Y, 628), 'desc': 'Subway wall (VERTICAL, ~5.5m from cams)', 'camera': 'Both', 'vertical': True},
    6: {'pos': (-150, STREET_LEVEL_Y, 219), 'desc': 'Center back (SHARED)', 'camera': 'Both', 'vertical': False},
}

# Marker image path (in calibration folder)
MARKER_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'calibration', 'marker_{}.png')

# Toggle for marker visibility
SHOW_MARKERS = False

# Toggle for coordinate labels
SHOW_LABELS = True

# Toggle for camera preview windows
SHOW_CAMERA_VIEWS = False
CAMERA_VIEW_SIZE = (320, 240)  # Size of each camera preview window


# =============================================================================
# TRACKED PERSON FROM OSC
# =============================================================================

@dataclass
class TrackedPerson:
    """Represents a person tracked via OSC"""
    track_id: int
    daily_id: int
    x: float  # World X position (cm)
    z: float  # World Z position (cm)
    y: float = STREET_LEVEL_Y  # Fixed at street level
    last_update: float = 0.0
    first_seen: float = 0.0  # When first tracked
    zone: str = "unknown"  # "active", "passive", or "unknown"
    vx: float = 0.0  # Velocity X (cm/s)
    vz: float = 0.0  # Velocity Z (cm/s)
    
    def get_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def is_in_active_zone(self) -> bool:
        return self.zone == "active"
    
    def is_in_passive_zone(self) -> bool:
        return self.zone == "passive"


class TrackedPersonManager:
    """Manages all tracked people received via OSC"""
    
    def __init__(self):
        self.people: Dict[int, TrackedPerson] = {}
        self.lock = threading.Lock()
        self.timeout = 1.0  # Remove person after 1 second without updates
        self.daily_count = 0
        
        # Calibration offsets and scales
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0
        self.invert_x = False  # Flip X direction of incoming data
        
        # Zone boundaries
        self.active_zone = {
            'x_min': TRACKZONE['center_x'] - TRACKZONE['width']/2,
            'x_max': TRACKZONE['center_x'] + TRACKZONE['width']/2,
            'z_min': TRACKZONE['offset_z'],
            'z_max': TRACKZONE['offset_z'] + TRACKZONE['depth'],
        }
        self.passive_zone = {
            'x_min': PASSIVE_TRACKZONE['center_x'] - PASSIVE_TRACKZONE['width']/2,
            'x_max': PASSIVE_TRACKZONE['center_x'] + PASSIVE_TRACKZONE['width']/2,
            'z_min': PASSIVE_TRACKZONE['offset_z'],
            'z_max': PASSIVE_TRACKZONE['offset_z'] + PASSIVE_TRACKZONE['depth'],
        }
        
        # Callbacks for behavior system
        self.on_person_entered = None
        self.on_person_left = None
        self.on_position_updated = None
        self.on_zone_updated = None  # Called with (person_id, is_active, position)
    
    def _get_zone(self, x: float, z: float) -> str:
        """Determine which zone a position is in"""
        az = self.active_zone
        pz = self.passive_zone
        
        if (az['x_min'] <= x <= az['x_max'] and 
            az['z_min'] <= z <= az['z_max']):
            return "active"
        elif (pz['x_min'] <= x <= pz['x_max'] and 
              pz['z_min'] <= z <= pz['z_max']):
            return "passive"
        return "unknown"
    
    def update_person(self, track_id: int, raw_x: float, raw_z: float, zone: str = None):
        """Update or add a tracked person with calibration applied"""
        # Apply calibration: scaled position + offset
        # Optionally invert X direction (for mirrored camera views)
        if self.invert_x:
            raw_x = -raw_x
        x = raw_x * self.scale_x + self.offset_x
        z = raw_z * self.scale_z + self.offset_z
        y = STREET_LEVEL_Y * self.scale_y + self.offset_y
        
        # Always compute zone locally based on calibrated position
        # This allows the controller's sliders and offsets to control zone determination
        zone = self._get_zone(x, z)
        
        now = time.time()
        now_dt = datetime.now()
        
        with self.lock:
            is_new = track_id not in self.people
            
            if is_new:
                self.daily_count += 1
                self.people[track_id] = TrackedPerson(
                    track_id=track_id,
                    daily_id=self.daily_count,
                    x=x, z=z, y=y,
                    last_update=now,
                    first_seen=now,
                    zone=zone,
                    vx=0.0,
                    vz=0.0,
                )
                # Notify behavior system
                if self.on_person_entered:
                    pos = np.array([x, y, z])
                    is_active = zone == "active"
                    self.on_person_entered(track_id, pos, is_active)
            else:
                person = self.people[track_id]
                dt = max(1e-6, now - person.last_update)
                person.vx = (x - person.x) / dt
                person.vz = (z - person.z) / dt
                person.x = x
                person.z = z
                person.y = y
                person.zone = zone
                person.last_update = now
                
                # Notify position update
                pos = np.array([x, y, z])
                if self.on_position_updated:
                    self.on_position_updated(track_id, pos)
                
                # Notify zone status (for active tracking)
                if self.on_zone_updated:
                    is_active = zone == "active"
                    self.on_zone_updated(track_id, is_active, pos)
    
    def cleanup_stale(self):
        """Remove people who haven't been updated recently"""
        now = time.time()
        with self.lock:
            stale_ids = [pid for pid, p in self.people.items() 
                        if now - p.last_update > self.timeout]
            for pid in stale_ids:
                del self.people[pid]
                if self.on_person_left:
                    self.on_person_left(pid)

    def reset_daily_population(self):
        """Reset daily population count and reassign IDs to active people."""
        with self.lock:
            self.daily_count = 0
            for person in self.people.values():
                self.daily_count += 1
                person.daily_id = self.daily_count
    
    def get_all(self) -> List[TrackedPerson]:
        """Get list of all tracked people"""
        with self.lock:
            return list(self.people.values())
    
    def get_person(self, track_id: int) -> Optional[TrackedPerson]:
        """Get a specific tracked person by ID"""
        with self.lock:
            return self.people.get(track_id)
    
    def count(self) -> int:
        """Get count of tracked people"""
        with self.lock:
            return len(self.people)
    
    def count_active(self) -> int:
        """Count people in active zone based on their calibrated position"""
        with self.lock:
            return sum(1 for p in self.people.values() if self._get_zone(p.x, p.z) == "active")
    
    def count_passive(self) -> int:
        """Count people in passive zone based on their calibrated position"""
        with self.lock:
            return sum(1 for p in self.people.values() if self._get_zone(p.x, p.z) == "passive")
    
    def get_active_positions(self) -> List[np.ndarray]:
        """Get positions of people in active zone based on calibrated position"""
        with self.lock:
            return [p.get_position() for p in self.people.values() if self._get_zone(p.x, p.z) == "active"]


# =============================================================================
# WEBSOCKET BROADCASTER (for public viewer)
# =============================================================================

class WebSocketBroadcaster:
    """Broadcasts installation state to web clients with efficiency optimizations"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: set = set()
        self.clients_lock = asyncio.Lock()  # Thread-safe client management
        self.loop = None
        self.server = None
        self.thread = None
        self.current_state = {}
        self.running = False
        self._last_json: str = ""  # Cache serialized JSON
        self._last_state_hash: int = 0  # Track state changes
        self._pending_broadcast: bool = False  # Coalesce rapid updates
    
    async def handler(self, websocket):
        """Handle a WebSocket connection with ping/pong heartbeat"""
        async with self.clients_lock:
            self.clients.add(websocket)
        
        client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else 'unknown'
        logger.info(f"WebSocket client connected: {client_ip} (total: {len(self.clients)})")
        
        try:
            # Send current state immediately
            if self._last_json:
                await websocket.send(self._last_json)
            
            # Keep connection alive with ping/pong (handled by websockets library)
            async for message in websocket:
                # Handle any client messages (e.g., request full report refresh)
                try:
                    data = json.loads(message)
                    if data.get('type') == 'request_report' and self._last_json:
                        await websocket.send(self._last_json)
                except json.JSONDecodeError:
                    pass  # Ignore malformed messages
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.debug(f"WebSocket connection closed: {client_ip} (code: {e.code})")
        except Exception as e:
            logger.warning(f"WebSocket handler error for {client_ip}: {e}")
        finally:
            async with self.clients_lock:
                self.clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {client_ip} (remaining: {len(self.clients)})")
    
    async def broadcast(self):
        """Broadcast cached state to all connected clients"""
        if not self.clients or not self._last_json:
            return
        
        # Get snapshot of clients under lock
        async with self.clients_lock:
            clients_snapshot = list(self.clients)
        
        if not clients_snapshot:
            return
        
        # Broadcast to all clients concurrently
        dead_clients = []
        
        async def send_to_client(client):
            try:
                await asyncio.wait_for(client.send(self._last_json), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket send timeout, marking client dead")
                dead_clients.append(client)
            except websockets.exceptions.ConnectionClosed:
                dead_clients.append(client)
            except Exception as e:
                logger.debug(f"WebSocket send error: {e}")
                dead_clients.append(client)
        
        # Send concurrently to all clients
        await asyncio.gather(*[send_to_client(c) for c in clients_snapshot], return_exceptions=True)
        
        # Remove dead clients
        if dead_clients:
            async with self.clients_lock:
                for client in dead_clients:
                    self.clients.discard(client)
    
    def update_state(self, state: dict):
        """Update the current state (called from main thread) - optimized"""
        # Compute simple hash to detect meaningful changes
        population = state.get('population', {})
        state_hash = hash((
            state.get('mode'),
            len(state.get('people', [])),
            state.get('report_version', 0),
            state.get('auto_tuning', {}).get('revision', 0),
            int(state.get('light', {}).get('x', 0) * 10),
            int(state.get('light', {}).get('y', 0) * 10),
            population.get('daily_total', 0),
            population.get('current', 0),
        ))
        
        # Only re-serialize if state actually changed
        if state_hash != self._last_state_hash:
            self._last_state_hash = state_hash
            self._last_json = json.dumps(state, separators=(',', ':'))  # Compact JSON
        
        self.current_state = state
        
        if self.loop and self.running and not self._pending_broadcast:
            self._pending_broadcast = True
            
            async def do_broadcast():
                self._pending_broadcast = False
                await self.broadcast()
            
            asyncio.run_coroutine_threadsafe(do_broadcast(), self.loop)
    
    async def _run_server(self):
        """Run the WebSocket server with optimized settings"""
        self.server = await websockets.serve(
            self.handler,
            "0.0.0.0",
            self.port,
            ping_interval=20,  # Send ping every 20s
            ping_timeout=10,   # Wait 10s for pong
            close_timeout=5,   # Allow 5s for graceful close
            max_size=2**20,    # 1MB max message size
        )
        logger.info(f"WebSocket server started on port {self.port}")
        
        # Get local IP for display
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"   Public viewer URL: http://{local_ip}:8080")
        except Exception:
            print(f"   Public viewer: connect to port {self.port}")
        
        await self.server.wait_closed()
    
    def _thread_main(self):
        """Main function for the WebSocket thread with auto-restart"""
        restart_count = 0
        max_restarts = 10
        restart_delay = 5  # seconds
        
        while self.running and restart_count < max_restarts:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                self.loop.run_until_complete(self._run_server())
            except OSError as e:
                # Port already in use, etc.
                restart_count += 1
                logger.error(f"WebSocket server OS error ({restart_count}/{max_restarts}): {e}")
                if restart_count < max_restarts and self.running:
                    logger.info(f"WebSocket server restarting in {restart_delay}s...")
                    time.sleep(restart_delay)
                    restart_delay = min(restart_delay * 2, 60)
            except Exception as e:
                restart_count += 1
                logger.error(f"WebSocket server error ({restart_count}/{max_restarts}): {e}")
                if restart_count < max_restarts and self.running:
                    logger.info(f"WebSocket server restarting in {restart_delay}s...")
                    time.sleep(restart_delay)
                    restart_delay = min(restart_delay * 2, 60)
            finally:
                try:
                    # Clean up pending tasks
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self.loop.close()
                except Exception:
                    pass
        
        if restart_count >= max_restarts:
            logger.error("WebSocket server exceeded max restart attempts, giving up")
        self.running = False
    
    def start(self):
        """Start the WebSocket server in a background thread"""
        self.running = True  # Set BEFORE starting thread
        self.thread = threading.Thread(target=self._thread_main, daemon=True, name="WebSocketServer")
        self.thread.start()
    
    def stop(self):
        """Stop the WebSocket server gracefully"""
        self.running = False
        if self.server:
            self.server.close()
        if self.loop and self.loop.is_running():
            # Schedule cleanup on the event loop
            async def cleanup():
                async with self.clients_lock:
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


# =============================================================================
# AUTO TUNING (trend-responsive behavior adjustments)
# =============================================================================

@dataclass
class AutoTuningConfig:
    update_interval: float = 5.0
    min_step: float = 0.002
    max_step_personality: float = 0.03
    max_step_global: float = 0.08
    target_activity: float = 0.5
    damping_strong: float = 0.4
    damping_moderate: float = 0.7
    budget_cost_scale: float = 60.0


class AutoTuningManager:
    def __init__(self, meta: MetaParameters, sliders: dict, database: TrackingDatabase = None):
        self.meta = meta
        self.sliders = sliders
        self.database = database
        self.config = AutoTuningConfig()
        self.enabled = True
        self.last_update = 0.0
        self.last_adjustment: Optional[dict] = None
        self.revision = 0
        self.history: List[dict] = []
        self.budget_current = 30.0
        self.budget_last_time = time.time()

        self.param_order = [
            'responsiveness', 'energy', 'attention_span', 'sociability',
            'exploration', 'memory', 'brightness_global', 'speed_global',
            'pulse_global', 'follow_speed_global', 'dwell_influence',
            'idle_trend_weight'
        ]

        self.min_vals = {
            'responsiveness': 0.0,
            'energy': 0.0,
            'attention_span': 0.0,
            'sociability': 0.0,
            'exploration': 0.0,
            'memory': 0.0,
            'brightness_global': 0.2,
            'speed_global': 0.2,
            'pulse_global': 0.3,
            'follow_speed_global': 0.5,
            'dwell_influence': 0.0,
            'idle_trend_weight': 0.0,
        }

        self.max_vals = {
            'responsiveness': 1.0,
            'energy': 1.0,
            'attention_span': 1.0,
            'sociability': 1.0,
            'exploration': 1.0,
            'memory': 1.0,
            'brightness_global': 5.0,
            'speed_global': 2.0,
            'pulse_global': 3.0,
            'follow_speed_global': 3.0,
            'dwell_influence': 2.0,
            'idle_trend_weight': 2.0,
        }

        # Soft caps to avoid obnoxious behavior
        self.caps = {
            'brightness_global': 3.0,
            'speed_global': 1.6,
            'pulse_global': 2.0,
            'energy': 0.85,
            'responsiveness': 0.9,
        }
        
        # Safe minimums — auto-tuner cannot push below these
        # Raised significantly to prevent the "zombie light" problem
        # (bright but unresponsive when activity is high)
        self.safe_floors = {
            'responsiveness': 0.30,
            'energy': 0.25,
            'brightness_global': 0.6,
            'speed_global': 0.35,
            'pulse_global': 0.35,
            'follow_speed_global': 0.6,
            'exploration': 0.15,
            'sociability': 0.20,
            'attention_span': 0.10,
            'idle_trend_weight': 0.10,
        }
        
        # Mean-reversion targets — params drift back toward these when
        # not being actively pushed. Prevents extremes from being sticky.
        self.home_values = {
            'responsiveness': 0.50,
            'energy': 0.45,
            'attention_span': 0.50,
            'sociability': 0.45,
            'exploration': 0.40,
            'memory': 0.30,
            'brightness_global': 1.2,
            'speed_global': 0.70,
            'pulse_global': 0.80,
            'follow_speed_global': 1.0,
            'dwell_influence': 0.50,
            'idle_trend_weight': 0.40,
        }
        
        # Curiosity: periodic random perturbation to explore parameter space
        # Increased strength (0.015→0.04) and frequency (60s→30s) so curiosity
        # can meaningfully counteract steady delta pressure toward floors
        self._curiosity_interval = 30.0  # Every 30 seconds (was 60)
        self._last_curiosity_time = time.time()
        self._curiosity_strength = 0.04  # Stronger nudge (was 0.015)
        
        # Fix 3: Adaptive target — tracks rolling median of short_activity
        # so the target matches actual traffic instead of a static 0.5
        self._activity_history: List[float] = []  # Rolling window of short_activity values
        self._activity_history_max = 500  # ~42 minutes at 5s intervals
        self._adaptive_target: Optional[float] = None  # None = use config default until enough data
        self._adaptive_target_min = 0.03  # Don't let target go below this
        self._adaptive_target_max = 0.40  # Don't let target go above this
        
        # Fix 6: Periodic resets toward home values at time-of-day transitions
        self._last_reset_hour: Optional[int] = None  # Track which reset window we last applied
        self._reset_hours = {0, 6, 12, 18}  # Reset at midnight, 6am, noon, 6pm
        self._reset_blend = 0.40  # Blend 40% toward home values on reset
        
        # Mean reversion parameters (overridable by meta-tuner)
        self._reversion_base = 0.02
        self._reversion_progressive = 0.06
        
        # Budget restore time (overridable by meta-tuner)
        self._budget_restore_seconds = 300.0
        
        # Hot-reload: check autotune_overrides.json for meta-tuner config changes
        self._override_file = AUTOTUNE_OVERRIDES_FILE
        self._override_mtime: float = 0.0  # Last known mtime of override file
        self._override_check_interval: float = 30.0  # Check every 30 seconds
        self._last_override_check: float = 0.0
        self._load_overrides()  # Load on startup if file exists
        
        # Learned biases from previous days' reports (loaded from DB)
        self.learned_starting_values: Dict[str, float] = {}
        self.learned_caps_adjustments: Dict[str, float] = {}
        self.days_of_learning: int = 0

    def _load_overrides(self):
        """
        Load config overrides from autotune_overrides.json (written by meta-tuner).
        Updates home_values, safe_floors, caps, curiosity, reversion, and budget settings.
        
        Safeguards:
        - All values are clamped to sane ranges (no NaN/inf, no extreme values)
        - Malformed JSON is caught and the file is renamed with .bad suffix
        - If the file can't be read, the controller continues with current values
        - Unknown keys are silently ignored (forward-compatible)
        """
        try:
            if not os.path.exists(self._override_file):
                return False
            
            mtime = os.path.getmtime(self._override_file)
            if mtime == self._override_mtime:
                return False  # File hasn't changed
            
            with open(self._override_file, 'r') as f:
                raw = f.read()
            
            if not raw.strip():
                logger.warning("Meta-tuner override file is empty — ignoring")
                self._override_mtime = mtime
                return False
            
            overrides = json.loads(raw)
            
            if not isinstance(overrides, dict):
                logger.warning(f"Meta-tuner override file root is not a dict ({type(overrides).__name__}) — ignoring")
                self._override_mtime = mtime
                return False
            
            self._override_mtime = mtime
            changes = []
            
            def _safe_float(v, lo, hi):
                """Convert to float, reject NaN/inf, clamp to [lo, hi]."""
                f = float(v)
                if not math.isfinite(f):
                    return None  # Reject NaN and inf
                return max(lo, min(hi, f))
            
            # Apply home_values overrides (clamped to [0, 5])
            if 'home_values' in overrides and isinstance(overrides['home_values'], dict):
                for name, val in overrides['home_values'].items():
                    if name in self.home_values:
                        clamped = _safe_float(val, 0.0, 5.0)
                        if clamped is None:
                            logger.warning(f"Override home[{name}]={val} is NaN/inf — skipping")
                            continue
                        old = self.home_values[name]
                        if abs(old - clamped) > 0.001:
                            self.home_values[name] = clamped
                            changes.append(f"home[{name}]:{old:.3f}→{clamped:.3f}")
            
            # Apply safe_floors overrides (clamped to [0, 2])
            if 'safe_floors' in overrides and isinstance(overrides['safe_floors'], dict):
                for name, val in overrides['safe_floors'].items():
                    if name in self.safe_floors:
                        clamped = _safe_float(val, 0.0, 2.0)
                        if clamped is None:
                            continue
                        old = self.safe_floors[name]
                        if abs(old - clamped) > 0.001:
                            self.safe_floors[name] = clamped
                            changes.append(f"floor[{name}]:{old:.3f}→{clamped:.3f}")
            
            # Apply caps overrides (clamped to [0.1, 10])
            if 'caps' in overrides and isinstance(overrides['caps'], dict):
                for name, val in overrides['caps'].items():
                    if name in self.caps:
                        clamped = _safe_float(val, 0.1, 10.0)
                        if clamped is None:
                            continue
                        old = self.caps[name]
                        if abs(old - clamped) > 0.001:
                            self.caps[name] = clamped
                            changes.append(f"cap[{name}]:{old:.3f}→{clamped:.3f}")
            
            # Apply curiosity overrides (interval [5, 600], strength [0, 0.2])
            if 'curiosity' in overrides and isinstance(overrides.get('curiosity'), dict):
                c = overrides['curiosity']
                if 'interval' in c:
                    v = _safe_float(c['interval'], 5.0, 600.0)
                    if v is not None:
                        self._curiosity_interval = v
                if 'strength' in c:
                    v = _safe_float(c['strength'], 0.0, 0.2)
                    if v is not None:
                        self._curiosity_strength = v
            
            # Apply reversion overrides (base [0, 0.1], progressive [0, 0.3])
            if 'reversion' in overrides and isinstance(overrides.get('reversion'), dict):
                r = overrides['reversion']
                if 'base' in r:
                    v = _safe_float(r['base'], 0.0, 0.1)
                    if v is not None:
                        self._reversion_base = v
                if 'progressive' in r:
                    v = _safe_float(r['progressive'], 0.0, 0.3)
                    if v is not None:
                        self._reversion_progressive = v
            
            # Apply budget overrides (max [5, 100], cost_scale [1, 200], restore_seconds [30, 1200])
            if 'budget' in overrides and isinstance(overrides.get('budget'), dict):
                b = overrides['budget']
                if 'max' in b:
                    v = _safe_float(b['max'], 5.0, 100.0)
                    if v is not None:
                        slider = self.sliders.get('interaction_budget')
                        if slider is not None:
                            slider.value = v
                if 'cost_scale' in b:
                    v = _safe_float(b['cost_scale'], 1.0, 200.0)
                    if v is not None:
                        self.config.budget_cost_scale = v
                if 'restore_seconds' in b:
                    v = _safe_float(b['restore_seconds'], 30.0, 1200.0)
                    if v is not None:
                        self._budget_restore_seconds = v
            
            if changes:
                logger.info(f"🔄 Loaded {len(changes)} meta-tuner overrides: {', '.join(changes[:5])}{'...' if len(changes) > 5 else ''}")
            return True
            
        except json.JSONDecodeError as e:
            # Corrupted JSON — rename the bad file so we don't retry every 30s
            logger.error(f"Corrupted meta-tuner override file: {e}")
            try:
                bad_path = self._override_file + '.bad'
                os.rename(self._override_file, bad_path)
                logger.warning(f"Renamed corrupted override file to {bad_path}")
            except OSError:
                pass
            self._override_mtime = 0.0  # Reset so we re-check if a new file appears
            return False
        except (IOError, OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to load autotune overrides: {e}")
            return False
        except Exception as e:
            # Catch-all: never let the override loader crash the controller
            logger.error(f"Unexpected error loading autotune overrides: {e}")
            return False
    
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
    
    def load_learnings_from_db(self):
        """
        Load historical auto-tune learnings from the database.
        Uses the last 7 days of data to compute weighted optimal starting values
        and any learned cap adjustments.
        """
        if not self.database:
            return
        
        try:
            learnings = self.database.get_recent_autotune_learnings(days=7)
            if not learnings:
                logger.info("🧠 No previous auto-tune learnings found")
                return
            
            self.days_of_learning = len(learnings)
            
            # Compute weighted average of optimal values from recent days
            # More recent days get higher weight
            weighted_values: Dict[str, float] = {}
            weight_counts: Dict[str, float] = {}
            
            for i, day in enumerate(learnings):
                # Weight: most recent = 1.0, oldest = 0.3
                weight = 1.0 - (i * 0.1)
                weight = max(0.3, weight)
                
                optimal = day.get('optimal_values', {})
                for name, val in optimal.items():
                    if name in self.param_order:
                        weighted_values[name] = weighted_values.get(name, 0.0) + val * weight
                        weight_counts[name] = weight_counts.get(name, 0.0) + weight
            
            # Compute averages
            for name in weighted_values:
                if weight_counts.get(name, 0) > 0:
                    self.learned_starting_values[name] = weighted_values[name] / weight_counts[name]
            
            # Merge any learned cap adjustments from the most recent day
            if learnings:
                most_recent_caps = learnings[0].get('learned_caps', {})
                self.learned_caps_adjustments = most_recent_caps
            
            if self.learned_starting_values:
                logger.info(f"🧠 Loaded auto-tune learnings from {self.days_of_learning} days")
                top3 = sorted(self.learned_starting_values.items(), 
                             key=lambda kv: abs(kv[1] - self._get_values().get(kv[0], 0.5)))[:3]
                for name, val in top3:
                    current = self._get_values().get(name, 0.5)
                    logger.info(f"   {name}: learned={val:.3f} current={current:.3f}")
        except Exception as e:
            logger.warning(f"Failed to load auto-tune learnings: {e}")
    
    def apply_learnings_to_values(self):
        """
        Blend learned optimal values with current slider values.
        Only applies if learnings exist and values differ meaningfully.
        Uses a gentle blend (30% learned, 70% current) to avoid jarring changes.
        """
        if not self.learned_starting_values:
            return
        
        current = self._get_values()
        blended = {}
        blend_factor = min(0.3, 0.1 * self.days_of_learning)  # More data = more confidence, max 30%
        
        for name, learned_val in self.learned_starting_values.items():
            current_val = current.get(name)
            if current_val is None:
                continue
            
            # Only blend if there's a meaningful difference
            diff = abs(learned_val - current_val)
            if diff < 0.01:
                continue
            
            new_val = current_val * (1 - blend_factor) + learned_val * blend_factor
            new_val = self._clamp(name, new_val)
            blended[name] = new_val
        
        if blended:
            self._apply_values(blended)
            logger.info(f"🧠 Applied learned values to {len(blended)} params (blend={blend_factor:.0%})")
    
    def compute_daily_learnings(self, report) -> Dict:
        """
        Analyze a daily report and extract learnings for future auto-tuning.
        Computes optimal starting values based on end-of-day parameter positions
        and parameter journey analysis.
        
        Args:
            report: DailyReport instance
            
        Returns:
            Dict with optimal_values and learned_caps
        """
        tuning = report.auto_tuning_analysis
        journeys = tuning.get('param_journeys', {})
        
        if not journeys:
            return {'optimal_values': {}, 'learned_caps': {}}
        
        # The "optimal" values are where each parameter settled by end of day
        # weighted by how much the tuner moved them (more movement = more confidence)
        optimal_values = {}
        for name, journey in journeys.items():
            if name not in self.param_order:
                continue
            
            end_val = journey.get('end', None)
            if end_val is None:
                continue
            
            # Use the end-of-day value as the optimal starting point
            # But bias toward the middle of the range it explored
            min_val = journey.get('min', end_val)
            max_val = journey.get('max', end_val)
            midpoint = (min_val + max_val) / 2.0
            
            # Blend: 60% end value (where it settled) + 40% midpoint (where it explored)
            optimal = end_val * 0.6 + midpoint * 0.4
            optimal_values[name] = round(optimal, 4)
        
        # Learn cap adjustments: if a param consistently hit its cap,
        # and the tuner kept pushing, consider loosening the cap
        learned_caps = dict(self.caps)  # Start from current caps
        for name, journey in journeys.items():
            if name not in self.caps:
                continue
            current_cap = self.caps[name]
            end_val = journey.get('end', 0)
            max_val = journey.get('max', 0)
            
            # If the param spent time at or near the cap, nudge cap up slightly
            if max_val >= current_cap * 0.95 and journey.get('direction') == 'up':
                learned_caps[name] = round(min(current_cap * 1.1, self.max_vals.get(name, current_cap)), 3)
        
        return {
            'optimal_values': optimal_values,
            'learned_caps': learned_caps,
        }

    def _get_values(self) -> dict:
        return {name: float(getattr(self.meta, name)) for name in self.param_order}

    def _budget_max(self) -> float:
        slider = self.sliders.get('interaction_budget')
        if slider is None:
            return 30.0
        return max(0.0, float(slider.value))

    def _apply_values(self, new_values: dict):
        for name, value in new_values.items():
            setattr(self.meta, name, value)
            if name in self.sliders:
                self.sliders[name].value = value

    def _clamp(self, name: str, value: float) -> float:
        max_val = min(self.max_vals.get(name, 1.0), self.caps.get(name, self.max_vals.get(name, 1.0)))
        min_val = max(self.min_vals.get(name, 0.0), self.safe_floors.get(name, 0.0))
        return max(min_val, min(max_val, value))

    def update(self, behavior_status: dict, now: float) -> Optional[dict]:
        # --- Periodic hot-reload of meta-tuner overrides ---
        if now - self._last_override_check > self._override_check_interval:
            self._last_override_check = now
            self._load_overrides()
        
        budget_max = self._budget_max()
        dt_budget = max(0.0, now - self.budget_last_time)
        if dt_budget > 0:
            # Budget restore rate from instance var (overridable by meta-tuner)
            restore_rate = budget_max / self._budget_restore_seconds if budget_max > 0 else 0.0
            aggression = behavior_status.get('aggression', {})
            engagement_bonus = budget_max / 60.0 if aggression.get('current_engagement') else 0.0
            self.budget_current = min(budget_max, self.budget_current + dt_budget * (restore_rate + engagement_bonus))
            self.budget_last_time = now

        if not self.enabled or budget_max <= 0:
            return None

        if now - self.last_update < self.config.update_interval:
            return None

        idle_trends = behavior_status.get('idle_trends', {})
        if not idle_trends or not idle_trends.get('has_short', False):
            return None

        short_activity = float(idle_trends.get('short_activity', 0.0))
        medium_activity = float(idle_trends.get('medium_activity', short_activity))
        long_activity = float(idle_trends.get('long_activity', medium_activity))
        energy_level = float(idle_trends.get('energy_level', 0.5))

        aggression = behavior_status.get('aggression', {})
        aggression_level = float(aggression.get('level', 0.0))
        seconds_since_eng = float(aggression.get('seconds_since_engagement', 9999))

        damping = 1.0
        if aggression_level > 0.8 or seconds_since_eng < 10:
            damping = self.config.damping_strong
        elif aggression_level > 0.6:
            damping = self.config.damping_moderate

        # --- FIX 6: PERIODIC TIME-OF-DAY RESETS ---
        # Every 6 hours (midnight, 6am, noon, 6pm), blend params 40% toward
        # home values. This breaks floor-clamping cycles even if deltas have issues.
        current_hour = datetime.now().hour
        if current_hour in self._reset_hours:
            if self._last_reset_hour != current_hour:
                self._last_reset_hour = current_hour
                current_values = self._get_values()
                reset_values = {}
                for name, home_val in self.home_values.items():
                    cur = current_values.get(name)
                    if cur is not None:
                        blended = cur * (1.0 - self._reset_blend) + home_val * self._reset_blend
                        reset_values[name] = self._clamp(name, blended)
                self._apply_values(reset_values)
                logger.info(f"🔄 Time-of-day reset (hour={current_hour}): blended {len(reset_values)} params {self._reset_blend:.0%} toward home")
        elif self._last_reset_hour is not None and current_hour not in self._reset_hours:
            # Clear the flag so the next reset hour triggers again
            if self._last_reset_hour in self._reset_hours and current_hour != self._last_reset_hour:
                self._last_reset_hour = None

        # --- FIX 3: ADAPTIVE TARGET ---
        # Track rolling history of short_activity and use its median as the target,
        # so deltas are relative to actual traffic rather than a static 0.5
        self._activity_history.append(short_activity)
        if len(self._activity_history) > self._activity_history_max:
            self._activity_history = self._activity_history[-self._activity_history_max:]
        
        if len(self._activity_history) >= 20:  # Need ~100s of data before adapting
            sorted_history = sorted(self._activity_history)
            median_activity = sorted_history[len(sorted_history) // 2]
            self._adaptive_target = max(self._adaptive_target_min, 
                                        min(self._adaptive_target_max, median_activity))
            target = self._adaptive_target
        else:
            target = self.config.target_activity  # Fallback to static 0.5 until enough data

        activity_excess = max(-0.5, min(0.5, short_activity - target))  # positive when busy
        medium_excess = max(-0.5, min(0.5, medium_activity - target))

        # --- FIX 4: DECOUPLE BRIGHTNESS FROM EMPTY LONG-TERM DATA ---
        # Only use long_activity deficit if we have meaningful long-term data.
        # When long_activity is 0 (fresh DB / no history), fall back to medium_activity
        # to avoid a constant upward push on brightness.
        effective_long = long_activity
        if long_activity <= 0.001 and medium_activity > 0.001:
            effective_long = medium_activity  # Use medium as proxy until long-term fills in
        long_deficit = max(-0.3, min(0.3, target - effective_long))  # positive when quiet

        deltas = {}

        # --- FIX 1: ASYMMETRIC PERSONALITY DELTAS ---
        # Only push personality UP when activity is above target (busy).
        # When quiet (activity_excess < 0), do NOT actively suppress personality —
        # let mean reversion handle the gentle drift back toward home.
        # This prevents personality from being perpetually driven to the floor
        # on a normally-quiet sidewalk.
        if activity_excess > 0:
            # Busy: increase responsiveness, sociability, energy to match the crowd
            deltas['responsiveness'] = activity_excess * 0.04 * damping
            deltas['sociability'] = activity_excess * 0.04 * damping
            deltas['energy'] = activity_excess * 0.03 * damping
            deltas['follow_speed_global'] = activity_excess * 0.05 * damping
        # else: personality deltas are 0; mean reversion handles quiet-period drift

        # DISPLAY PARAMS: inversely adjust to activity
        # When quiet: brighter, more pulsing, more speed (attract attention)
        # When busy: moderate down (don't be overwhelming)
        deltas['brightness_global'] = -activity_excess * 0.04 * damping + long_deficit * 0.06 * damping
        deltas['speed_global'] = -activity_excess * 0.03 * damping + long_deficit * 0.04 * damping
        deltas['pulse_global'] = -activity_excess * 0.03 * damping + long_deficit * 0.04 * damping

        # EXPLORATION: increase when things are quiet (search for people)
        # decrease when busy (focus on the crowd)
        if medium_activity < 0.3:
            deltas['exploration'] = (0.3 - medium_activity) * 0.06 * damping
        elif medium_activity > 0.6:
            deltas['exploration'] = -(medium_activity - 0.6) * 0.04 * damping

        # ATTENTION SPAN: longer when quiet (contemplate), shorter when busy (reactive)
        deltas['attention_span'] = -activity_excess * 0.03 * damping

        deltas['memory'] = (effective_long - 0.5) * 0.03 * damping
        deltas['dwell_influence'] = (energy_level - 0.5) * 0.03 * damping
        deltas['idle_trend_weight'] = (0.5 - short_activity) * 0.03 * damping

        # --- FIX 2: STRONGER PROGRESSIVE MEAN REVERSION ---
        # Base strength and progressive component are overridable by meta-tuner
        current_values = self._get_values()
        reversion_base = self._reversion_base
        reversion_progressive = self._reversion_progressive
        for name, home_val in self.home_values.items():
            current_val = current_values.get(name)
            if current_val is not None:
                distance_from_home = home_val - current_val
                # Progressive: stronger pull when further from home
                strength = reversion_base + reversion_progressive * abs(distance_from_home)
                pull = distance_from_home * strength
                deltas[name] = deltas.get(name, 0.0) + pull

        # --- FIX 5: STRONGER CURIOSITY BIASED TOWARD HOME ---
        # Increased strength (0.015→0.04) and frequency (60s→30s).
        # Also biased: 60% of the nudge is toward home, 40% is random.
        # This helps exploration while gently countering floor-clamping.
        if now - self._last_curiosity_time > self._curiosity_interval:
            self._last_curiosity_time = now
            for name in self.param_order:
                # Random component
                random_nudge = (random.random() - 0.5) * 2.0 * self._curiosity_strength
                # Bias toward home: if below home, bias nudge positive (and vice versa)
                home_val = self.home_values.get(name)
                current_val = current_values.get(name)
                if home_val is not None and current_val is not None:
                    home_direction = 1.0 if home_val > current_val else -1.0
                    biased_nudge = 0.4 * random_nudge + 0.6 * abs(random_nudge) * home_direction
                else:
                    biased_nudge = random_nudge
                deltas[name] = deltas.get(name, 0.0) + biased_nudge

        old_values = self._get_values()
        new_values = dict(old_values)
        applied = {}

        for name, delta in deltas.items():
            max_step = self.config.max_step_personality if name in (
                'responsiveness', 'energy', 'attention_span', 'sociability',
                'exploration', 'memory'
            ) else self.config.max_step_global
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step

            if abs(delta) < self.config.min_step:
                continue

            new_val = self._clamp(name, old_values[name] + delta)
            applied_delta = new_val - old_values[name]
            if abs(applied_delta) < self.config.min_step:
                continue

            new_values[name] = new_val
            applied[name] = applied_delta

        if not applied:
            return None

        raw_cost = sum(abs(val) for val in applied.values()) * self.config.budget_cost_scale
        scale = 1.0
        if raw_cost > self.budget_current and raw_cost > 0:
            scale = max(0.0, self.budget_current / raw_cost)

        if scale < 1.0:
            new_values = dict(old_values)
            applied = {}
            for name, delta in deltas.items():
                max_step = self.config.max_step_personality if name in (
                    'responsiveness', 'energy', 'attention_span', 'sociability',
                    'exploration', 'memory'
                ) else self.config.max_step_global
                delta = max(-max_step, min(max_step, delta * scale))
                if abs(delta) < self.config.min_step:
                    continue
                new_val = self._clamp(name, old_values[name] + delta)
                applied_delta = new_val - old_values[name]
                if abs(applied_delta) < self.config.min_step:
                    continue
                new_values[name] = new_val
                applied[name] = applied_delta

            if not applied:
                return None

        self._apply_values(new_values)
        self.last_update = now
        self.revision += 1

        cost_used = sum(abs(val) for val in applied.values()) * self.config.budget_cost_scale
        budget_before = self.budget_current
        self.budget_current = max(0.0, self.budget_current - cost_used)

        adjustment = {
            'timestamp': now,
            'revision': self.revision,
            'enabled': self.enabled,
            'short_activity': short_activity,
            'medium_activity': medium_activity,
            'long_activity': long_activity,
            'energy_level': energy_level,
            'aggression_level': aggression_level,
            'seconds_since_engagement': seconds_since_eng,
            'damping': damping,
            'budget_before': budget_before,
            'budget_after': self.budget_current,
            'budget_max': budget_max,
            'budget_cost': cost_used,
            'old_values': old_values,
            'new_values': new_values,
            'applied_deltas': applied,
            'caps': self.caps,
        }

        self.last_adjustment = adjustment
        top_deltas = sorted(applied.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        
        # Debug print for monitoring
        delta_str = ', '.join(f"{n}:{d:+.3f}" for n, d in top_deltas)
        print(f"🎛️  Auto-tune #{self.revision}: activity={short_activity:.2f}/{medium_activity:.2f} → {delta_str}")
        
        self.history.append({
            'timestamp': now,
            'short_activity': short_activity,
            'medium_activity': medium_activity,
            'long_activity': long_activity,
            'deltas': top_deltas,
        })
        if len(self.history) > 8:
            self.history = self.history[-8:]

        if self.database:
            try:
                self.database.record_behavior_adjustment(
                    enabled=self.enabled,
                    reason='auto_tune',
                    short_activity=short_activity,
                    medium_activity=medium_activity,
                    long_activity=long_activity,
                    energy_level=energy_level,
                    aggression_level=aggression_level,
                    adjustments=adjustment,
                    timestamp=now
                )
            except Exception as e:
                logger.warning(f"Auto-tune adjustment log failed: {e}")

        return adjustment


# =============================================================================
# OSC HANDLER
# =============================================================================

class OSCHandler:
    """Handles incoming OSC messages"""
    
    def __init__(self, manager: TrackedPersonManager, database: TrackingDatabase = None):
        self.manager = manager
        self.database = database
        self.last_count = 0
        self.message_count = 0
        self.last_debug_time = time.time()
    
    def handle_person(self, address: str, *args):
        """Handle /tracker/person/<id> messages"""
        try:
            # Extract track_id from address
            parts = address.split('/')
            track_id = int(parts[-1])
            
            if len(args) >= 2:
                x, z = float(args[0]), float(args[1])
                self.manager.update_person(track_id, x, z)
                
                # Record to database using CALIBRATED position (not raw)
                # This ensures database zone classifications match real-time display
                if self.database:
                    person = self.manager.get_person(track_id)
                    if person:
                        self.database.record_position(track_id, person.x, person.z)
                
                # Debug output every 2 seconds
                self.message_count += 1
                now = time.time()
                if now - self.last_debug_time > 2.0:
                    print(f"📥 OSC: {self.message_count} messages, latest: person {track_id} at ({x:.0f}, {z:.0f})")
                    self.last_debug_time = now
                    self.message_count = 0
        except (ValueError, IndexError) as e:
            print(f"OSC parse error: {e}")
    
    def handle_count(self, address: str, *args):
        """Handle /tracker/count messages"""
        if args:
            self.last_count = int(args[0])
    
    def handle_zone(self, address: str, *args):
        """Handle /tracker/zone/<id> messages from V2 tracker
        
        NOTE: We ignore the tracker's zone determination.
        Zone is always calculated locally based on calibrated position
        and the controller's zone boundaries. This allows the offset/scale
        sliders to properly control zone assignment.
        """
        # Intentionally do nothing - zone is computed locally in update_person()
        pass


# =============================================================================
# POINT LIGHT & PANEL SYSTEM (from original)
# =============================================================================

@dataclass
class PointLight:
    """Virtual point light"""
    position: np.ndarray = field(default_factory=lambda: np.array([-160.0, 60.0, -10.0]))
    target_position: np.ndarray = field(default_factory=lambda: np.array([-160.0, 60.0, -10.0]))
    
    brightness_min: int = 5
    brightness_max: int = 40
    pulse_speed: float = 2000
    falloff_radius: float = 50
    
    move_speed: float = 50
    pulse_phase: float = 0.0
    
    def get_brightness(self) -> float:
        return (math.sin(self.pulse_phase) + 1) / 2
    
    def update(self, dt: float):
        self.pulse_phase += (2 * math.pi * dt * 1000) / self.pulse_speed
        
        diff = self.target_position - self.position
        dist = np.linalg.norm(diff)
        if dist > 0.1:
            move = min(self.move_speed * dt, dist)
            self.position += (diff / dist) * move


class PanelSystem:
    def __init__(self):
        self.panels: Dict[Tuple[int, int], dict] = {}
        self._build_panels()
    
    def _build_panels(self):
        # Unit 0 is rightmost, with back right corner at X=0
        # Unit 0 center at X=-30, Unit 1 at X=-110, Unit 2 at X=-190, Unit 3 at X=-270
        for unit in range(4):
            # Back right corner of unit 0 is at X=0, so center is at -(unit * UNIT_SPACING + 30)
            unit_x = -(unit * UNIT_SPACING + 30)
            for panel_num in range(1, 4):
                local_y, local_z = PANEL_LOCAL_POSITIONS[panel_num]
                center = np.array([unit_x, local_y, local_z])
                
                self.panels[(unit, panel_num)] = {
                    'center': center,
                    'angle': PANEL_ANGLES[panel_num],
                    'normal': PANEL_NORMALS[panel_num].copy(),
                    'brightness': 0.0,
                    'dmx_value': 0,
                }
    
    def calculate_brightness(self, light: PointLight):
        intensity = light.get_brightness()
        
        for key, panel in self.panels.items():
            diff = panel['center'] - light.position
            distance = np.linalg.norm(diff)
            
            if light.falloff_radius > 0 and distance > light.falloff_radius:
                # Outside the radius — panel is off
                panel['brightness'] = 0.0
                panel['dmx_value'] = DMX_MIN
                continue
            
            if light.falloff_radius > 0:
                falloff = 1.0 - distance / light.falloff_radius
            else:
                falloff = 1.0
            
            final_brightness = falloff * intensity
            panel['brightness'] = final_brightness
            
            dmx_range = light.brightness_max - light.brightness_min
            panel['dmx_value'] = int(light.brightness_min + final_brightness * dmx_range)
            panel['dmx_value'] = max(DMX_MIN, min(DMX_MAX, panel['dmx_value']))
    
    def get_dmx_values(self) -> List[int]:
        # Unit 0 = DMX CH1-3, Unit 1 = CH4-6, Unit 2 = CH7-9, Unit 3 = CH10-12
        return [self.panels[(u, p)]['dmx_value'] for u in range(4) for p in range(1, 4)]
    
    def get_unit_centers(self) -> Dict[int, np.ndarray]:
        """Get center position of each unit (for labeling)"""
        centers = {}
        for unit in range(4):
            # Right edge of unit 0 is at X=0, so center is at -(unit + 0.5) * UNIT_SPACING
            unit_x = -(unit + 0.5) * UNIT_SPACING
            # Unit center is at Y=60 (midpoint of panels), Z=0
            centers[unit] = np.array([unit_x, 60, 0])
        return centers


class WanderBehavior:
    def __init__(self, light: PointLight, wander_box: dict):
        self.light = light
        self.wander_box = wander_box
        self.wander_target = self._random_point()
        self.wander_timer = 0
        self.wander_interval = 3.0
        self.enabled = True
        
        # For behavior system integration
        self.follow_target = None
        self.follow_smoothing = 0.05
        self.follow_x_only = False  # If True, only X follows target, Y/Z wander
        self.gesture_target = None
    
    def _random_point(self) -> np.ndarray:
        return np.array([
            random.uniform(self.wander_box['min_x'], self.wander_box['max_x']),
            random.uniform(self.wander_box['min_y'], self.wander_box['max_y']),
            random.uniform(self.wander_box['min_z'], self.wander_box['max_z']),
        ])
    
    def update_wander_box(self, new_box: dict):
        """Update wander box (called by behavior system)"""
        self.wander_box = new_box
    
    def set_follow_target(self, target: np.ndarray, smoothing: float = 0.05, x_only: bool = False):
        """Set a target to follow (from behavior system)
        
        Args:
            target: Target position to follow
            smoothing: How quickly to follow (0-1, higher = faster)
            x_only: If True, only follow X axis, let Y/Z wander within box
        """
        self.follow_target = target
        self.follow_smoothing = smoothing
        self.follow_x_only = x_only
    
    def clear_follow_target(self):
        """Clear follow target, return to wandering"""
        self.follow_target = None
        self.follow_x_only = False
    
    def set_gesture_target(self, target: np.ndarray):
        """Set a gesture target (overrides other movement)"""
        self.gesture_target = target
    
    def clear_gesture_target(self):
        """Clear gesture target"""
        self.gesture_target = None
    
    def update(self, dt: float):
        if not self.enabled:
            return
        
        # Gesture target takes priority
        if self.gesture_target is not None:
            self.light.target_position = self.gesture_target.copy()
            return
        
        # Always clamp wander target to current box bounds (box may have moved)
        self.wander_target[0] = np.clip(self.wander_target[0], self.wander_box['min_x'], self.wander_box['max_x'])
        self.wander_target[1] = np.clip(self.wander_target[1], self.wander_box['min_y'], self.wander_box['max_y'])
        self.wander_target[2] = np.clip(self.wander_target[2], self.wander_box['min_z'], self.wander_box['max_z'])
        
        # Update wander timer and check if we need a new target
        # Only pick new target when we reach current one or timer expires
        self.wander_timer += dt
        dist = np.linalg.norm(self.light.position - self.wander_target)
        
        # Use longer interval in engaged mode (small box = frequent clamping)
        min_interval = max(3.0, self.wander_interval)  # At least 3 seconds
        
        if dist < 10 or self.wander_timer > min_interval:
            self.wander_target = self._random_point()
            self.wander_timer = 0
            # Randomize around the base interval
            self.wander_interval = random.uniform(min_interval, min_interval + 3)
        
        # Smoothly move toward wander target (already clamped to box)
        current = self.light.target_position
        target = self.wander_target
        
        # Smooth movement toward target - lower = slower, smoother
        diff = target - current
        smooth = 0.03  # Gentle, slow movement
        self.light.target_position = current + diff * smooth


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def draw_box_wireframe(bounds, color):
    """Draw wireframe box from bounds (xmin, xmax, ymin, ymax, zmin, zmax)"""
    x0, x1, y0, y1, z0, z1 = bounds
    
    glColor4f(*color)
    glBegin(GL_LINES)
    
    # Bottom face
    glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
    glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
    glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1)
    glVertex3f(x0, y0, z1); glVertex3f(x0, y0, z0)
    
    # Top face
    glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
    glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
    glVertex3f(x0, y1, z1); glVertex3f(x0, y1, z0)
    
    # Vertical edges
    glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0)
    glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0)
    glVertex3f(x1, y0, z1); glVertex3f(x1, y1, z1)
    glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1)
    
    glEnd()


def draw_panel(center, angle, size, brightness):
    """Draw a panel as a quad. Brightness is 0.0-1.0 float from panel system."""
    half = size / 2
    
    glPushMatrix()
    glTranslatef(*center)
    glRotatef(-angle, 1, 0, 0)
    
    # Panel face: dark base + warm white proportional to brightness
    # V2-style: simple 0.2 + brightness * 0.8 mapping
    b = max(0.0, min(1.0, brightness))
    base = 0.08
    r = base + b * 0.92  # Warm white: slightly more red
    g = base + b * 0.88
    bl = base + b * 0.78  # Less blue for warm LED look
    glColor4f(r, g, bl, 1.0)
    
    glBegin(GL_QUADS)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    # Additive glow quad (slightly larger, semi-transparent)
    if b > 0.05:
        glow_half = half * (1.0 + b * 0.15)  # Grow slightly with brightness
        glow_alpha = b * 0.25
        glColor4f(1.0, 0.95, 0.8, glow_alpha)
        glBegin(GL_QUADS)
        glVertex3f(-glow_half, -glow_half, 0.5)
        glVertex3f(glow_half, -glow_half, 0.5)
        glVertex3f(glow_half, glow_half, 0.5)
        glVertex3f(-glow_half, glow_half, 0.5)
        glEnd()
    
    # Border frame
    frame_brightness = 0.2 + b * 0.3
    glColor4f(frame_brightness, frame_brightness, frame_brightness, 1.0)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    glPopMatrix()


def draw_sphere(center, radius, color, segments=12):
    """Draw a simple sphere"""
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, segments, segments)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_sphere_wireframe(center, radius, color, segments=16):
    """Draw a wireframe sphere"""
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    glLineWidth(1)
    
    for i in range(segments // 2 + 1):
        lat = math.pi * i / (segments // 2) - math.pi / 2
        r = radius * math.cos(lat)
        y = radius * math.sin(lat)
        
        glBegin(GL_LINE_LOOP)
        for j in range(segments):
            lon = 2 * math.pi * j / segments
            x = r * math.cos(lon)
            z = r * math.sin(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    for j in range(segments // 2):
        lon = math.pi * j / (segments // 2)
        
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            lat = 2 * math.pi * i / segments
            x = radius * math.cos(lat) * math.sin(lon)
            y = radius * math.sin(lat)
            z = radius * math.cos(lat) * math.cos(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    glPopMatrix()


def draw_tracked_person(person: TrackedPerson, zone_checker=None):
    """Draw a tracked person as a cylinder/capsule
    
    Args:
        person: The tracked person to draw
        zone_checker: Optional function(x, z) -> str that returns 'active', 'passive', or 'unknown'
    """
    pos = person.get_position()
    
    # Determine zone based on current position if checker provided
    if zone_checker:
        zone = zone_checker(pos[0], pos[2])
    else:
        zone = person.zone
    
    # Color based on zone
    if zone == "active":
        color = (0.2, 0.8, 0.2, 0.8)  # Green for active
    elif zone == "passive":
        color = (0.8, 0.8, 0.2, 0.8)  # Yellow for passive
    else:
        color = (0.5, 0.5, 0.5, 0.6)  # Gray for unknown
    
    # Draw as a colored cylinder (person height ~170cm)
    height = 170
    radius = 20
    
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    
    # Body cylinder
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)  # Rotate to stand upright
    gluCylinder(quadric, radius, radius, height, 16, 1)
    
    # Top cap (head)
    glTranslatef(0, 0, height)
    gluSphere(quadric, radius, 12, 12)
    
    gluDeleteQuadric(quadric)
    glPopMatrix()
    
    # Draw population ID label above head
    label_pos = np.array([pos[0], pos[1] + height + radius + 10, pos[2]])
    # Project 3D position to screen using current matrices
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    try:
        sx, sy, sz = gluProject(label_pos[0], label_pos[1], label_pos[2],
                                modelview, projection, viewport)
        if sz > 0 and sz < 1:  # Visible (in front of camera)
            label = f"#{person.daily_id}"
            _id_font = pygame.font.SysFont('monospace', 16, bold=True)
            text_surface = _id_font.render(label, True, (255, 255, 255))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glDisable(GL_DEPTH_TEST)
            glWindowPos2d(int(sx) - text_surface.get_width() // 2, int(sy))
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                         GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glEnable(GL_DEPTH_TEST)
    except Exception:
        pass  # Skip if projection fails

    # Draw velocity vector (scaled)
    speed = math.sqrt(person.vx ** 2 + person.vz ** 2)
    if speed > 5:
        vec_scale = 0.2
        vx = person.vx * vec_scale
        vz = person.vz * vec_scale
        glLineWidth(2)
        glColor4f(color[0], color[1], color[2], 0.9)
        glBegin(GL_LINES)
        glVertex3f(pos[0], pos[1] + 10, pos[2])
        glVertex3f(pos[0] + vx, pos[1] + 10, pos[2] + vz)
        glEnd()
        glLineWidth(1)


def draw_floor(y_level, color, z_max=None):
    """Draw a floor plane. z_max limits depth (defaults to full size)"""
    glColor4f(*color)
    # Floor extends from X=110 to X=-390 (toward Unit 3), Z=-200 to z_max
    z_back = z_max if z_max is not None else 400
    glBegin(GL_QUADS)
    glVertex3f(110, y_level, -200)
    glVertex3f(-390, y_level, -200)
    glVertex3f(-390, y_level, z_back)
    glVertex3f(110, y_level, z_back)
    glEnd()


def draw_text_2d(x, y, text, font, color=(255, 255, 255)):
    """Draw text on screen (2D HUD)"""
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)


def draw_trends_visualization(report: 'DailyReport', x: int, y: int, width: int, height: int, 
                               font, font_small):
    """
    Draw a visualization of daily trends as a bar chart overlay.
    
    Args:
        report: The DailyReport to visualize
        x, y: Bottom-left corner position
        width, height: Size of the visualization area
        font, font_small: Fonts for labels
    """
    if not report or not report.hourly_trends:
        return
    
    # Background panel with transparency
    glColor4f(0.1, 0.1, 0.15, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    # Border
    glColor4f(0.3, 0.5, 0.7, 1.0)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    # Title
    title = f"Daily Report: {report.date}"
    draw_text_2d(x + 10, y + height - 25, title, font, (255, 255, 200))
    
    # Summary line
    summary = f"Total: {report.total_unique_people} people | Peak: {report.peak_hour}:00 ({report.peak_hour_count}) | Flow: {report.dominant_flow}"
    draw_text_2d(x + 10, y + height - 45, summary, font_small, (200, 200, 200))
    
    # Chart area
    chart_x = x + 50
    chart_y = y + 30
    chart_width = width - 70
    chart_height = height - 100
    
    # Find max values for scaling - use separate scales for active and passive
    # Active counts are typically much lower than passive
    max_active = max((h.active_count for h in report.hourly_trends), default=1) or 1
    max_passive = max((h.passive_count for h in report.hourly_trends), default=1) or 1
    # Use combined max for stacked bars, with minimum thresholds for visibility
    max_combined = max(max_active + max_passive // 3, 10)  # Minimum scale of 10
    
    # Draw hour bars
    bar_width = chart_width / 24
    bar_gap = 2
    
    for trend in report.hourly_trends:
        hour = trend.hour
        bx = chart_x + hour * bar_width
        
        # Active zone bar (green) - scale to fill ~40% of chart max
        active_height = (trend.active_count / max_combined) * chart_height * 0.8
        glColor4f(0.2, 0.7, 0.3, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bx + bar_gap, chart_y)
        glVertex2f(bx + bar_width - bar_gap, chart_y)
        glVertex2f(bx + bar_width - bar_gap, chart_y + active_height)
        glVertex2f(bx + bar_gap, chart_y + active_height)
        glEnd()
        
        # Passive zone bar (stacked, blue) - scale down since passive >> active
        passive_height = (trend.passive_count / 3 / max_combined) * chart_height * 0.8
        glColor4f(0.3, 0.3, 0.7, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bx + bar_gap, chart_y + active_height)
        glVertex2f(bx + bar_width - bar_gap, chart_y + active_height)
        glVertex2f(bx + bar_width - bar_gap, chart_y + active_height + passive_height)
        glVertex2f(bx + bar_gap, chart_y + active_height + passive_height)
        glEnd()
        
        # Highlight peak hour
        if hour == report.peak_hour:
            glColor4f(1.0, 1.0, 0.3, 0.3)
            glBegin(GL_QUADS)
            glVertex2f(bx, chart_y)
            glVertex2f(bx + bar_width, chart_y)
            glVertex2f(bx + bar_width, chart_y + chart_height)
            glVertex2f(bx, chart_y + chart_height)
            glEnd()
    
    # X-axis labels (hours)
    for hour in range(0, 24, 3):
        label_x = chart_x + hour * bar_width + bar_width / 2 - 5
        draw_text_2d(int(label_x), chart_y - 15, f"{hour:02d}", font_small, (150, 150, 150))
    
    # Y-axis label
    draw_text_2d(x + 5, chart_y + chart_height // 2, "Pop", font_small, (150, 150, 150))
    
    # Legend
    legend_y = y + height - 65
    glColor4f(0.2, 0.7, 0.3, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(x + 10, legend_y)
    glVertex2f(x + 25, legend_y)
    glVertex2f(x + 25, legend_y + 10)
    glVertex2f(x + 10, legend_y + 10)
    glEnd()
    draw_text_2d(x + 30, legend_y - 2, "Active", font_small, (100, 200, 100))
    
    glColor4f(0.3, 0.3, 0.7, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(x + 90, legend_y)
    glVertex2f(x + 105, legend_y)
    glVertex2f(x + 105, legend_y + 10)
    glVertex2f(x + 90, legend_y + 10)
    glEnd()
    draw_text_2d(x + 110, legend_y - 2, "Passive", font_small, (100, 100, 200))
    
    # Flow balance indicator
    flow_x = x + 200
    flow_width = 100
    flow_center = flow_x + flow_width // 2
    
    draw_text_2d(flow_x, legend_y - 2, "Flow:", font_small, (200, 200, 200))
    
    # Flow bar background
    glColor4f(0.3, 0.3, 0.3, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(flow_x + 40, legend_y)
    glVertex2f(flow_x + 40 + flow_width, legend_y)
    glVertex2f(flow_x + 40 + flow_width, legend_y + 10)
    glVertex2f(flow_x + 40, legend_y + 10)
    glEnd()
    
    # Flow indicator
    indicator_x = flow_x + 40 + flow_width // 2 + (report.flow_balance * flow_width // 2)
    if report.flow_balance > 0:
        glColor4f(0.2, 0.7, 0.2, 1.0)  # Green for L->R
    else:
        glColor4f(0.7, 0.2, 0.2, 1.0)  # Red for R->L
    
    glBegin(GL_TRIANGLES)
    glVertex2f(indicator_x, legend_y - 2)
    glVertex2f(indicator_x - 5, legend_y + 12)
    glVertex2f(indicator_x + 5, legend_y + 12)
    glEnd()
    
    # Close hint
    draw_text_2d(x + width - 80, y + 10, "T to close", font_small, (120, 120, 120))


def draw_auto_tuning_panel(x: int, y: int, width: int, height: int, font, font_small,
                           enabled: bool, last_adjustment: Optional[dict], history: List[dict],
                           budget_current: float, budget_max: float):
    """Draw a compact auto-tuning status panel with latest adjustment and history."""
    glColor4f(0.08, 0.08, 0.12, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()

    glColor4f(0.3, 0.4, 0.6, 0.8)
    glLineWidth(1)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()

    status_color = (100, 255, 100) if enabled else (180, 180, 180)
    status_text = "ON" if enabled else "OFF"
    draw_text_2d(x + 10, y + height - 18, "AUTO TUNE", font, (120, 200, 255))
    draw_text_2d(x + width - 50, y + height - 18, status_text, font_small, status_color)

    budget_ratio = budget_current / budget_max if budget_max > 0 else 0.0
    budget_bar = "#" * int(budget_ratio * 12) + "." * (12 - int(budget_ratio * 12))
    draw_text_2d(
        x + 10,
        y + height - 36,
        f"Budget [{budget_bar}] {budget_current:.0f}/{budget_max:.0f}",
        font_small,
        (180, 180, 180)
    )

    if not last_adjustment:
        draw_text_2d(x + 10, y + height - 54, "No adjustments yet", font_small, (150, 150, 150))
        return

    now = time.time()
    age = max(0.0, now - last_adjustment.get('timestamp', now))
    short_act = last_adjustment.get('short_activity', 0.0)
    med_act = last_adjustment.get('medium_activity', 0.0)
    long_act = last_adjustment.get('long_activity', 0.0)
    damping = last_adjustment.get('damping', 1.0)

    # Split into two shorter lines to fit column
    draw_text_2d(x + 10, y + height - 54, f"Last: {age:.0f}s ago  Damp:{damping:.2f}", font_small, (180, 180, 180))
    draw_text_2d(x + 10, y + height - 68, f"S:{short_act:.2f} M:{med_act:.2f} L:{long_act:.2f}", font_small, (160, 160, 160))

    label_map = {
        'responsiveness': 'resp',
        'energy': 'enrg',
        'attention_span': 'attn',
        'sociability': 'socl',
        'exploration': 'expl',
        'memory': 'mem',
        'brightness_global': 'brit',
        'speed_global': 'spd',
        'pulse_global': 'puls',
        'follow_speed_global': 'fllw',
        'dwell_influence': 'dwel',
        'idle_trend_weight': 'idle',
    }

    deltas = last_adjustment.get('applied_deltas', {})
    # Show all deltas, sorted by magnitude
    all_deltas = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    
    # Current values section
    draw_text_2d(x + 10, y + height - 86, "Adjustments:", font_small, (120, 200, 120))
    delta_y = y + height - 102
    for k, v in all_deltas:
        color = (120, 220, 120) if v > 0 else (220, 120, 120)
        draw_text_2d(x + 10, delta_y, f" {label_map.get(k, k):>4s} {v:+.3f}", font_small, color)
        delta_y -= 14
        if delta_y < y + 40:  # Don't overflow past bottom
            break

    # History section - use remaining space
    if delta_y > y + 30:
        delta_y -= 8
        draw_text_2d(x + 10, delta_y, "History:", font_small, (140, 140, 180))
        delta_y -= 16
        max_history = min(8, len(history))
        for item in reversed(history[-max_history:]):
            h_age = max(0.0, now - item.get('timestamp', now))
            h_deltas = item.get('deltas', [])
            if not h_deltas:
                continue
            # One delta per line to stay within column
            first = True
            for k, v in h_deltas:
                prefix = f"{h_age:3.0f}s " if first else "     "
                first = False
                draw_text_2d(x + 10, delta_y, f"{prefix}{label_map.get(k, k):>4s}{v:+.2f}", font_small, (140, 140, 160))
                delta_y -= 14
                if delta_y < y + 10:
                    break
            if delta_y < y + 10:
                break


def _build_behavior_description(behavior, active_count: int, passive_count: int) -> str:
    """
    Build a concise human-readable description of the light's current behavior
    for the public viewer subheading.
    
    Returns a short phrase like:
    - "Wandering · Scanning"
    - "Engaged · Following"  
    - "Engaged · Breathing Together"
    - "Engaged · Nodding"
    - "Idle · Acknowledging Passerby"
    - "Crowd · Orbiting"
    """
    if not behavior:
        return ""
    
    mode = behavior.state.mode
    gesture = behavior.state.gesture
    is_bored = behavior.state.is_bored
    dwell_bonus = behavior.state.current_dwell_bonus
    breathing = behavior.state.engaged_breathe_active
    breathe_depth = behavior.state.engaged_breathe_depth
    mode_duration = behavior.state.mode_duration
    
    # Gesture descriptions (takes priority when active)
    gesture_descriptions = {
        GestureType.WELCOME: "Welcoming",
        GestureType.SURPRISED: "Surprised",
        GestureType.CURIOUS: "Approaching",
        GestureType.ACKNOWLEDGE: "Acknowledging Passerby",
        GestureType.FAREWELL: "Saying Goodbye",
        GestureType.NOD: "Nodding",
        GestureType.LEAN: "Leaning In",
        GestureType.SWAY: "Swaying",
        GestureType.ORBIT: "Orbiting",
        GestureType.SETTLE: "Settling In",
        GestureType.BREATHE: "Breathing",
        GestureType.BLOOM: "Blooming",
        GestureType.BORED: "Restless",
        GestureType.THINKING: "Thinking",
        GestureType.HESITANT: "Hesitant",
        GestureType.PLAYFUL: "Playing",
    }
    
    # Mode labels
    mode_labels = {
        BehaviorMode.IDLE: "Idle",
        BehaviorMode.ENGAGED: "Engaged",
        BehaviorMode.CROWD: "Crowd",
        BehaviorMode.FLOW: "Flow",
    }
    
    mode_label = mode_labels.get(mode, "")
    
    # Active gesture takes priority for the action description
    if gesture != GestureType.NONE and gesture in gesture_descriptions:
        action = gesture_descriptions[gesture]
        return f"{mode_label} · {action}"
    
    # No gesture — describe based on mode and state
    if mode == BehaviorMode.IDLE:
        if is_bored:
            return "Idle · Waiting"
        elif passive_count > 0:
            return "Idle · Watching"
        else:
            return "Idle · Wandering"
    
    elif mode in (BehaviorMode.ENGAGED, BehaviorMode.CROWD):
        # Describe engagement depth
        if breathing and breathe_depth > 0.5:
            return f"{mode_label} · Breathing Together"
        elif dwell_bonus > 10:
            return f"{mode_label} · Deep Connection"
        elif dwell_bonus > 5:
            return f"{mode_label} · Bonding"
        elif mode_duration > 10:
            return f"{mode_label} · Following"
        elif mode_duration > 3:
            return f"{mode_label} · Greeting"
        else:
            return f"{mode_label} · Noticing"
    
    elif mode == BehaviorMode.FLOW:
        return "Flow · Drifting with Traffic"
    
    return mode_label


def draw_realtime_trends(idle_trends: dict, x: int, y: int, font, font_small, aggression: dict = None, flow: dict = None, almost_engaged: dict = None, feedback_learning: dict = None):
    """
    Draw real-time trends panel on the left side of the screen.
    Shows current activity levels, flow direction, aggression, and data availability.
    
    Args:
        idle_trends: Dict from behavior_status.get('idle_trends')
        x, y: Top-left position
        font, font_small: Fonts for rendering
        aggression: Dict from behavior_status.get('aggression')
        flow: Dict from behavior_status.get('flow')
        almost_engaged: Dict from behavior_status.get('almost_engaged')
        feedback_learning: Dict from behavior_status.get('feedback_learning')
    """
    if not idle_trends:
        return
    
    panel_width = 260
    panel_height = 520  # Reduced height - content is more compact now
    
    # Background panel
    glColor4f(0.08, 0.08, 0.12, 0.85)
    glBegin(GL_QUADS)
    glVertex2f(x, y - panel_height)
    glVertex2f(x + panel_width, y - panel_height)
    glVertex2f(x + panel_width, y)
    glVertex2f(x, y)
    glEnd()
    
    # Border
    glColor4f(0.3, 0.4, 0.6, 0.8)
    glLineWidth(1)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y - panel_height)
    glVertex2f(x + panel_width, y - panel_height)
    glVertex2f(x + panel_width, y)
    glVertex2f(x, y)
    glEnd()
    
    # Title
    draw_text_2d(x + 10, y - 18, "REALTIME TRENDS", font, (100, 180, 255))
    
    # Update timing
    seconds_since = idle_trends.get('seconds_since_update', 0)
    update_color = (100, 255, 100) if seconds_since < 6 else (255, 200, 100) if seconds_since < 15 else (255, 100, 100)
    draw_text_2d(x + 130, y - 18, f"({seconds_since:.1f}s ago)", font_small, update_color)
    
    curr_y = y - 35
    line_height = 14  # Reduced from 16 for more compact display
    min_y = y - panel_height + 15  # Stop drawing before going off panel
    
    # Period indicator
    period = idle_trends.get('period', 'unknown')
    period_colors = {
        'late_night': (100, 100, 180),
        'morning': (255, 200, 100),
        'afternoon': (255, 255, 150),
        'evening': (180, 130, 200),
    }
    period_color = period_colors.get(period, (150, 150, 150))
    draw_text_2d(x + 10, curr_y, f"Period: {period.upper()}", font_small, period_color)
    curr_y -= line_height + 3
    
    # Database error if any
    db_error = idle_trends.get('database_error', '')
    if db_error:
        draw_text_2d(x + 10, curr_y, f"⚠ {db_error[:25]}", font_small, (255, 100, 100))
        curr_y -= line_height
    
    # Section: REALTIME (1 min)
    if curr_y < min_y: return
    has_recent = idle_trends.get('has_recent', False)
    status_char = "●" if has_recent else "○"
    status_color = (100, 255, 100) if has_recent else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Now (1m)", font_small, status_color)
    recent_passive = idle_trends.get('recent_passive', 0)
    recent_active = idle_trends.get('recent_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{recent_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{recent_active}", font_small, (255, 180, 100))
    curr_y -= line_height
    
    # Section: SHORT TERM (5 min)
    has_short = idle_trends.get('has_short', False)
    status_char = "●" if has_short else "○"
    status_color = (100, 255, 100) if has_short else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Short (5m)", font_small, status_color)
    short_passive = idle_trends.get('short_passive', 0)
    short_active = idle_trends.get('short_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{short_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{short_active}", font_small, (255, 180, 100))
    short_act = idle_trends.get('short_activity', 0)
    bar = "█" * int(short_act * 6) + "░" * (6 - int(short_act * 6))
    draw_text_2d(x + 180, curr_y, f"[{bar}]", font_small, (100, 200, 100))
    curr_y -= line_height
    
    # Section: MEDIUM TERM (30 min)
    has_medium = idle_trends.get('has_medium', False)
    status_char = "●" if has_medium else "○"
    status_color = (100, 255, 100) if has_medium else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Med (30m)", font_small, status_color)
    med_passive = idle_trends.get('medium_passive', 0)
    med_active = idle_trends.get('medium_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{med_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{med_active}", font_small, (255, 180, 100))
    med_act = idle_trends.get('medium_activity', 0)
    bar = "█" * int(med_act * 6) + "░" * (6 - int(med_act * 6))
    draw_text_2d(x + 180, curr_y, f"[{bar}]", font_small, (100, 150, 200))
    curr_y -= line_height
    
    # Section: LONG TERM (1 hr)
    has_long = idle_trends.get('has_long', False)
    status_char = "●" if has_long else "○"
    status_color = (100, 255, 100) if has_long else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Long (1h)", font_small, status_color)
    long_passive = idle_trends.get('long_passive', 0)
    long_active = idle_trends.get('long_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{long_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{long_active}", font_small, (255, 180, 100))
    long_act = idle_trends.get('long_activity', 0)
    bar = "█" * int(long_act * 6) + "░" * (6 - int(long_act * 6))
    draw_text_2d(x + 180, curr_y, f"[{bar}]", font_small, (150, 150, 255))
    curr_y -= line_height
    
    # Section: HISTORICAL
    has_hist = idle_trends.get('has_historical', False)
    status_char = "●" if has_hist else "○"
    status_color = (100, 255, 100) if has_hist else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Historical (7d)", font_small, status_color)
    curr_y -= line_height + 6
    
    # Divider line
    if curr_y < min_y: return
    glColor4f(0.3, 0.4, 0.6, 0.5)
    glBegin(GL_LINES)
    glVertex2f(x + 10, curr_y + 3)
    glVertex2f(x + panel_width - 10, curr_y + 3)
    glEnd()
    
    # COMPUTED VALUES section
    draw_text_2d(x + 10, curr_y, "COMPUTED", font_small, (180, 180, 200))
    curr_y -= line_height
    
    # Anticipation
    anticipation = idle_trends.get('activity_anticipation', 0.5)
    ant_bar = "█" * int(anticipation * 10) + "░" * (10 - int(anticipation * 10))
    ant_color = (100, 255, 100) if anticipation > 0.6 else (255, 200, 100) if anticipation > 0.3 else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, "Anticipation:", font_small, (180, 180, 180))
    draw_text_2d(x + 95, curr_y, f"[{ant_bar}]", font_small, ant_color)
    curr_y -= line_height
    
    # Flow momentum
    momentum = idle_trends.get('flow_momentum', 0)
    if abs(momentum) > 0.1:
        arrow_count = int(abs(momentum) * 5)
        arrows = "→" * arrow_count if momentum > 0 else "←" * arrow_count
        mom_color = (100, 200, 255) if momentum > 0 else (255, 200, 100)
        draw_text_2d(x + 10, curr_y, "Flow:", font_small, (180, 180, 180))
        draw_text_2d(x + 55, curr_y, f"{arrows} ({momentum:+.2f})", font_small, mom_color)
    else:
        draw_text_2d(x + 10, curr_y, "Flow: balanced", font_small, (100, 100, 100))
    curr_y -= line_height
    
    # Energy level
    energy = idle_trends.get('energy_level', 0.5)
    energy_bar = "█" * int(energy * 10) + "░" * (10 - int(energy * 10))
    energy_color = (255, 200, 100) if energy > 0.6 else (150, 200, 150) if energy > 0.3 else (100, 100, 150)
    draw_text_2d(x + 10, curr_y, "Energy:", font_small, (180, 180, 180))
    draw_text_2d(x + 65, curr_y, f"[{energy_bar}]", font_small, energy_color)
    curr_y -= line_height + 6
    
    # ======================
    # AGGRESSION SECTION
    # ======================
    if aggression and curr_y > min_y:
        # Divider line
        glColor4f(0.3, 0.4, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 3)
        glVertex2f(x + panel_width - 10, curr_y + 3)
        glEnd()
        
        draw_text_2d(x + 10, curr_y, "AGGRESSION", font_small, (255, 150, 100))
        curr_y -= line_height
        
        # Aggression level bar
        level = aggression.get('level', 0)
        cap = aggression.get('time_of_day_cap', 1.0)
        bar_filled = int(level * 10)
        bar_cap = int(cap * 10)
        
        # Build a bar showing level and cap
        # Filled = current level, dim = available up to cap, dark = capped out
        bar = ""
        for i in range(10):
            if i < bar_filled:
                bar += "█"
            elif i < bar_cap:
                bar += "▒"
            else:
                bar += "░"
        
        # Color based on level: green=low, yellow=medium, red=high
        if level < 0.3:
            agg_color = (100, 200, 100)  # Green - calm
        elif level < 0.6:
            agg_color = (255, 200, 100)  # Yellow - moderate
        else:
            agg_color = (255, 100, 100)  # Red - high aggression
        
        draw_text_2d(x + 10, curr_y, "Level:", font_small, (180, 180, 180))
        draw_text_2d(x + 55, curr_y, f"[{bar}]", font_small, agg_color)
        draw_text_2d(x + 175, curr_y, f"{level:.2f}", font_small, agg_color)
        curr_y -= line_height
        
        # Time of day cap
        hour = datetime.now().hour
        draw_text_2d(x + 10, curr_y, f"ToD Cap ({hour:02d}:00):", font_small, (150, 150, 150))
        draw_text_2d(x + 115, curr_y, f"{cap:.1f}", font_small, (180, 180, 200))
        curr_y -= line_height
        
        # Time since engagement
        since_eng = aggression.get('seconds_since_engagement', 0)
        if since_eng < 60:
            time_str = f"{since_eng:.0f}s"
        else:
            time_str = f"{since_eng/60:.1f}m"
        eng_color = (100, 255, 100) if since_eng < 30 else (255, 200, 100) if since_eng < 300 else (255, 100, 100)
        draw_text_2d(x + 10, curr_y, "Since engage:", font_small, (150, 150, 150))
        draw_text_2d(x + 100, curr_y, time_str, font_small, eng_color)
        
        # Current engagement indicator
        if aggression.get('current_engagement'):
            draw_text_2d(x + 160, curr_y, "ENGAGED", font_small, (100, 255, 100))
        curr_y -= line_height + 6
    
    # ======================
    # FLOW POSITIONING SECTION (Phase 2B)
    # ======================
    if flow and curr_y > min_y:
        # Divider line
        glColor4f(0.3, 0.4, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 3)
        glVertex2f(x + panel_width - 10, curr_y + 3)
        glEnd()
        
        draw_text_2d(x + 10, curr_y, "FLOW", font_small, (100, 200, 255))
        curr_y -= line_height
        
        # Flow direction visualization with arrows
        direction = flow.get('direction', 0)
        strength = flow.get('strength', 0)
        x_offset = flow.get('x_offset', 0)
        
        # Visual flow indicator
        if strength > 0.2 and abs(direction) > 0.1:
            arrow_count = min(5, max(1, int(strength * 5)))
            if direction > 0:
                arrows = "→" * arrow_count
                flow_label = "L→R"
                flow_color = (100, 200, 255)  # Blue for left-to-right
            else:
                arrows = "←" * arrow_count
                flow_label = "R→L"
                flow_color = (255, 180, 100)  # Orange for right-to-left
            draw_text_2d(x + 10, curr_y, f"Flow: {flow_label}", font_small, (180, 180, 180))
            draw_text_2d(x + 80, curr_y, arrows, font_small, flow_color)
            draw_text_2d(x + 150, curr_y, f"({direction:+.2f})", font_small, flow_color)
        else:
            draw_text_2d(x + 10, curr_y, "Flow: none/mixed", font_small, (100, 100, 100))
        curr_y -= line_height
        
        # Strength indicator
        strength_bar = "█" * int(strength * 6) + "░" * (6 - int(strength * 6))
        strength_color = (100, 255, 100) if strength > 0.5 else (200, 200, 100) if strength > 0.2 else (100, 100, 100)
        draw_text_2d(x + 10, curr_y, "Strength:", font_small, (150, 150, 150))
        draw_text_2d(x + 75, curr_y, f"[{strength_bar}]", font_small, strength_color)
        curr_y -= line_height
        
        # X offset (anticipatory positioning)
        if abs(x_offset) > 1:
            offset_dir = "←" if x_offset < 0 else "→"
            offset_color = (100, 255, 200)
            draw_text_2d(x + 10, curr_y, "Box offset:", font_small, (150, 150, 150))
            draw_text_2d(x + 85, curr_y, f"{offset_dir} {abs(x_offset):.0f}cm", font_small, offset_color)
        else:
            draw_text_2d(x + 10, curr_y, "Box offset: centered", font_small, (100, 100, 100))
        curr_y -= line_height
        
        # Event counts
        ltr = flow.get('left_to_right', 0)
        rtl = flow.get('right_to_left', 0)
        total = flow.get('total_events', 0)
        draw_text_2d(x + 10, curr_y, f"30s: L→R:{ltr} R→L:{rtl} ({total})", font_small, (120, 120, 150))
        curr_y -= line_height + 6
    
    # ======================
    # ALMOST-ENGAGED SECTION (Phase 2C)
    # ======================
    if almost_engaged and curr_y > min_y:
        # Divider line
        glColor4f(0.3, 0.4, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 3)
        glVertex2f(x + panel_width - 10, curr_y + 3)
        glEnd()
        
        draw_text_2d(x + 10, curr_y, "ALMOST-ENGAGED", font_small, (255, 200, 100))
        curr_y -= line_height
        
        # Conversion stats
        total_det = almost_engaged.get('total_detected', 0)
        total_conv = almost_engaged.get('total_converted', 0)
        conv_rate = almost_engaged.get('conversion_rate', 0) * 100
        
        rate_color = (100, 255, 100) if conv_rate > 30 else (255, 200, 100) if conv_rate > 10 else (150, 150, 150)
        draw_text_2d(x + 10, curr_y, f"Detected: {total_det}", font_small, (180, 180, 180))
        draw_text_2d(x + 100, curr_y, f"Conv: {total_conv}", font_small, (180, 180, 180))
        draw_text_2d(x + 170, curr_y, f"({conv_rate:.0f}%)", font_small, rate_color)
        curr_y -= line_height
        
        # Current attraction state
        if almost_engaged.get('active_attraction'):
            strategy = almost_engaged.get('current_strategy', 'none')
            target_id = almost_engaged.get('target_id', -1)
            draw_text_2d(x + 10, curr_y, f"→ Attracting #{target_id}", font_small, (100, 255, 200))
            draw_text_2d(x + 130, curr_y, f"[{strategy}]", font_small, (255, 200, 100))
        else:
            cand_count = almost_engaged.get('candidate_count', 0)
            if cand_count > 0:
                draw_text_2d(x + 10, curr_y, f"Watching {cand_count} candidate(s)", font_small, (200, 200, 150))
            else:
                draw_text_2d(x + 10, curr_y, "No candidates", font_small, (100, 100, 100))
        curr_y -= line_height
        
        # Show candidates (up to 2)
        candidates = almost_engaged.get('candidates', [])
        for i, c in enumerate(candidates[:2]):
            speed = c.get('speed', 0)
            dist = c.get('distance', 0)
            dur = c.get('duration', 0)
            pid = c.get('id', 0)
            strat = c.get('strategy', 'none')
            
            # Color based on whether being attracted
            if strat != 'none':
                c_color = (100, 255, 200)  # Attracting
            elif dist < 50:
                c_color = (255, 200, 100)  # Very close!
            else:
                c_color = (150, 150, 180)  # Watching
            
            draw_text_2d(x + 10, curr_y, f"#{pid}: {speed:.0f}cm/s d={dist:.0f}cm t={dur:.1f}s", font_small, c_color)
            curr_y -= line_height
            if curr_y < min_y: return

    # ======================
    # FEEDBACK LEARNING SECTION (Phase 3)
    # ======================
    if feedback_learning and curr_y > min_y:
        # Divider line
        glColor4f(0.4, 0.3, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 3)
        glVertex2f(x + panel_width - 10, curr_y + 3)
        glEnd()
        
        draw_text_2d(x + 10, curr_y, "LEARNING", font_small, (200, 150, 255))
        curr_y -= line_height
        
        # Total engagements
        total_eng = feedback_learning.get('total_engagements', 0)
        session_eng = feedback_learning.get('session_engagements', 0)
        
        if curr_y > min_y:
            draw_text_2d(x + 10, curr_y, f"Eng: {total_eng} (sess: {session_eng})", font_small, (180, 180, 180))
            curr_y -= line_height
        
        # Top weighted behaviors (compact - only show 2)
        top_weights = feedback_learning.get('top_weights', {})
        if top_weights and curr_y > min_y:
            for name, weight in list(top_weights.items())[:2]:
                if curr_y < min_y: break
                # Color based on weight (> 1.0 = good, green tint)
                if weight > 1.1:
                    w_color = (100, 255, 150)
                elif weight > 1.0:
                    w_color = (180, 255, 180)
                else:
                    w_color = (180, 180, 180)
                draw_text_2d(x + 10, curr_y, f"{name}: {weight:.2f}", font_small, w_color)
                curr_y -= line_height - 2


def draw_text_3d_billboard(position, text, font, color=(255, 255, 255), offset_y=0):
    """
    Draw text in 3D space as a billboard (always faces camera).
    This uses screen-space rendering at the projected 3D position.
    """
    # Get current matrices
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    # Project 3D position to screen coordinates
    try:
        screen_x, screen_y, screen_z = gluProject(
            position[0], position[1] + offset_y, position[2],
            modelview, projection, viewport
        )
        
        # Only draw if in front of camera
        if screen_z < 1.0:
            # Render text
            text_surface = font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            
            # Center text horizontally
            text_x = int(screen_x - text_surface.get_width() / 2)
            text_y = int(screen_y)
            
            glWindowPos2d(text_x, text_y)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    except:
        pass  # Projection failed, skip


def draw_origin_marker(font):
    """Draw a sphere at the origin (0,0,0) with label - at back right corner of panel 0"""
    # Origin is now at back right corner of panel 0: X=0, Y=0, Z=0
    origin_pos = (0, 0, 0)
    
    # Draw sphere at origin
    draw_sphere(origin_pos, 10, (1.0, 1.0, 0.0, 1.0), segments=16)  # Yellow sphere
    
    # Draw axis lines from origin
    glLineWidth(3)
    glBegin(GL_LINES)
    # X axis - Red (pointing right/positive)
    glColor4f(1, 0, 0, 1)
    glVertex3f(origin_pos[0], origin_pos[1], origin_pos[2])
    glVertex3f(origin_pos[0] + 50, origin_pos[1], origin_pos[2])
    # Y axis - Green (pointing up)
    glColor4f(0, 1, 0, 1)
    glVertex3f(origin_pos[0], origin_pos[1], origin_pos[2])
    glVertex3f(origin_pos[0], origin_pos[1] + 50, origin_pos[2])
    # Z axis - Blue (pointing forward into tracking zone)
    glColor4f(0, 0, 1, 1)
    glVertex3f(origin_pos[0], origin_pos[1], origin_pos[2])
    glVertex3f(origin_pos[0], origin_pos[1], origin_pos[2] + 50)
    glEnd()
    glLineWidth(1)


def draw_camera_markers(font, show_labels):
    """Draw spheres at camera positions with labels and rotated viewing cones"""
    for cam_name, cam_data in CAMERA_POSITIONS.items():
        pos = cam_data['pos']
        color = cam_data['color']
        rotation = cam_data.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
        
        # Draw camera as a sphere
        draw_sphere(pos, 15, color, segments=16)
        
        # Draw viewing direction cone with proper rotation
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        
        # Apply rotations: Yaw (Y), then Pitch (X), then Roll (Z)
        # Yaw rotates around Y axis (left/right)
        glRotatef(rotation['yaw'], 0, 1, 0)
        # Pitch rotates around X axis (up/down)
        glRotatef(rotation['pitch'], 1, 0, 0)
        # Roll rotates around Z axis (tilt)
        glRotatef(rotation['roll'], 0, 0, 1)
        
        glColor4f(*color)
        
        # Draw a simple pyramid/cone shape pointing toward +Z (forward)
        # The rotation transforms will orient it correctly
        cone_length = 80  # Length of viewing cone
        cone_half_width = 30  # Half-width at end (based on FOV)
        
        glBegin(GL_LINES)
        # Lines from camera to viewing direction corners
        glVertex3f(0, 0, 0)
        glVertex3f(-cone_half_width, -cone_half_width * 0.6, cone_length)
        glVertex3f(0, 0, 0)
        glVertex3f(cone_half_width, -cone_half_width * 0.6, cone_length)
        glVertex3f(0, 0, 0)
        glVertex3f(cone_half_width, cone_half_width * 0.6, cone_length)
        glVertex3f(0, 0, 0)
        glVertex3f(-cone_half_width, cone_half_width * 0.6, cone_length)
        # Connect the corners to form rectangle at end
        glVertex3f(-cone_half_width, -cone_half_width * 0.6, cone_length)
        glVertex3f(cone_half_width, -cone_half_width * 0.6, cone_length)
        glVertex3f(cone_half_width, -cone_half_width * 0.6, cone_length)
        glVertex3f(cone_half_width, cone_half_width * 0.6, cone_length)
        glVertex3f(cone_half_width, cone_half_width * 0.6, cone_length)
        glVertex3f(-cone_half_width, cone_half_width * 0.6, cone_length)
        glVertex3f(-cone_half_width, cone_half_width * 0.6, cone_length)
        glVertex3f(-cone_half_width, -cone_half_width * 0.6, cone_length)
        # Center line (optical axis)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, cone_length)
        glEnd()
        
        glPopMatrix()
        
        # Draw label if enabled
        if show_labels:
            label = f"{cam_name}\n({pos[0]}, {pos[1]}, {pos[2]})"
            draw_text_3d_billboard(pos, cam_name, font, (255, 255, 255), offset_y=25)
            coord_text = f"({pos[0]}, {pos[1]}, {pos[2]})"
            draw_text_3d_billboard(pos, coord_text, font, (200, 200, 200), offset_y=10)


def draw_unit_labels(panel_system, font, show_labels):
    """Draw labels for each panel unit"""
    if not show_labels:
        return
    
    unit_centers = panel_system.get_unit_centers()
    
    for unit_num, center in unit_centers.items():
        # Draw unit label
        unit_label = f"Unit {unit_num}"
        draw_text_3d_billboard(center, unit_label, font, (255, 200, 100), offset_y=80)
        
        # Draw coordinate
        coord_text = f"X={center[0]}"
        draw_text_3d_billboard(center, coord_text, font, (180, 180, 180), offset_y=65)


def draw_panel_centers(panel_system, font, show_labels):
    """Draw wireframe spheres at each panel center with panel number labels"""
    # Colors for each panel position within a unit
    panel_colors = {
        1: (1.0, 0.5, 0.5, 0.8),  # Panel 1 (top) - light red
        2: (0.5, 1.0, 0.5, 0.8),  # Panel 2 (bottom left) - light green
        3: (0.5, 0.5, 1.0, 0.8),  # Panel 3 (bottom right) - light blue
    }
    
    for (unit, panel_num), panel in panel_system.panels.items():
        center = panel['center']
        color = panel_colors.get(panel_num, (1.0, 1.0, 1.0, 0.8))
        
        # Draw small wireframe sphere at panel center
        draw_sphere_wireframe(center, 2, color, segments=12)
        
        # Draw label with panel number
        if show_labels:
            label = f"U{unit}P{panel_num}"
            draw_text_3d_billboard(center, label, font, (255, 255, 255), offset_y=15)


def draw_zone_corner_labels(bounds, name, font, color, show_labels):
    """
    Draw coordinate labels at the corners of a zone.
    bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    if not show_labels:
        return
    
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Draw labels at bottom corners (y_min level)
    corners = [
        (x_min, y_min, z_min, "near-left"),
        (x_max, y_min, z_min, "near-right"),
        (x_min, y_min, z_max, "far-left"),
        (x_max, y_min, z_max, "far-right"),
    ]
    
    for x, y, z, corner_name in corners:
        pos = [x, y, z]
        coord_text = f"({int(x)},{int(y)},{int(z)})"
        draw_text_3d_billboard(pos, coord_text, font, color, offset_y=5)
    
    # Draw zone name at center top
    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2
    draw_text_3d_billboard([center_x, y_max, center_z], name, font, color, offset_y=10)


# =============================================================================
# CALIBRATION MARKER RENDERING
# =============================================================================

def load_marker_textures() -> Dict[int, int]:
    """Load marker PNG files as OpenGL textures"""
    textures = {}
    
    for marker_id in MARKER_POSITIONS.keys():
        image_path = MARKER_IMAGE_PATH.format(marker_id)
        if os.path.exists(image_path):
            try:
                surface = pygame.image.load(image_path)
                texture_data = pygame.image.tostring(surface, "RGBA", True)
                width, height = surface.get_size()
                
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                            GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                
                textures[marker_id] = texture_id
                print(f"Loaded marker {marker_id} texture")
            except Exception as e:
                print(f"Failed to load marker {marker_id}: {e}")
        else:
            print(f"Marker image not found: {image_path}")
    
    return textures


def draw_marker(marker_id: int, position: Tuple[float, float, float], size: float,
                texture_id: Optional[int], vertical: bool = False):
    """
    Draw a calibration marker as a textured plane.
    If vertical=False: lies flat on floor facing upward
    If vertical=True: stands upright facing outward (toward positive Z / street)
    """
    x, y, z = position
    half = size / 2
    
    glPushMatrix()
    glTranslatef(x, y, z)
    
    if vertical:
        # Vertical marker: stands upright, facing outward toward street
        glTranslatef(0, 0, 0.5)
    else:
        # Horizontal marker: lies flat on floor, facing up
        glTranslatef(0, 0.5, 0)
        glRotatef(-90, 1, 0, 0)
    
    if texture_id is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1, 1, 1, 1)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-half, -half, 0)
        glTexCoord2f(1, 0); glVertex3f(half, -half, 0)
        glTexCoord2f(1, 1); glVertex3f(half, half, 0)
        glTexCoord2f(0, 1); glVertex3f(-half, half, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
    else:
        glColor4f(1, 1, 1, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(-half, -half, 0)
        glVertex3f(half, -half, 0)
        glVertex3f(half, half, 0)
        glVertex3f(-half, half, 0)
        glEnd()
    
    # Draw border
    glColor4f(0, 0, 0, 1)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-half, -half, 0.1)
    glVertex3f(half, -half, 0.1)
    glVertex3f(half, half, 0.1)
    glVertex3f(-half, half, 0.1)
    glEnd()
    
    glPopMatrix()
    
    # Draw marker ID indicator sphere
    glPushMatrix()
    if vertical:
        glTranslatef(x, y + half + 5, z)
    else:
        glTranslatef(x, y + 5, z)
    
    glColor4f(1, 1, 0, 1)  # Yellow
    quadric = gluNewQuadric()
    gluSphere(quadric, 2, 8, 8)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_marker_labels(font, show_labels):
    """Draw labels for all AR markers with ID and coordinates"""
    if not show_labels:
        return
    
    for marker_id, marker_data in MARKER_POSITIONS.items():
        pos = marker_data['pos']
        desc = marker_data['desc']
        
        # Label position (above the marker)
        label_y_offset = 30 if not marker_data.get('vertical', False) else 40
        
        # Draw marker ID
        id_label = f"Marker {marker_id}"
        draw_text_3d_billboard(pos, id_label, font, (255, 255, 0), offset_y=label_y_offset)
        
        # Draw coordinates
        coord_text = f"({pos[0]}, {pos[1]}, {pos[2]})"
        draw_text_3d_billboard(pos, coord_text, font, (200, 200, 200), offset_y=label_y_offset - 15)


# =============================================================================
# CAMERA VIEW RENDERING
# =============================================================================

def create_camera_fbo(width, height):
    """Create a framebuffer object for rendering camera view to texture"""
    # Create framebuffer
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    
    # Create texture to render to
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    
    # Create depth buffer
    depth_rb = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb)
    
    # Check if framebuffer is complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Warning: Camera FBO not complete")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    return {'fbo': fbo, 'texture': texture, 'depth': depth_rb, 'width': width, 'height': height}


def render_camera_view(camera_data, fbo_data, panel_system, light, tracked_manager, marker_textures, show_markers):
    """Render the scene from a camera's perspective to a framebuffer"""
    pos = camera_data['pos']
    rotation = camera_data.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
    fov = camera_data.get('fov', {'horizontal': 80, 'vertical': 48})
    
    width = fbo_data['width']
    height = fbo_data['height']
    
    # Bind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_data['fbo'])
    glViewport(0, 0, width, height)
    
    # Clear
    glClearColor(0.05, 0.05, 0.1, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Set up projection using camera FOV
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Use vertical FOV for gluPerspective, extend far plane to see marker 5
    gluPerspective(fov['vertical'], width / height, 10, 2000)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Calculate look-at direction from rotation angles
    pitch = math.radians(rotation['pitch'])
    yaw = math.radians(rotation['yaw'])
    
    # Forward direction based on yaw and pitch
    # Start with forward vector (0, 0, 1) and rotate
    forward_x = math.sin(yaw) * math.cos(pitch)
    forward_y = -math.sin(pitch)  # Negative because positive pitch looks down
    forward_z = math.cos(yaw) * math.cos(pitch)
    
    # Look-at point is camera position + forward direction * some distance
    look_distance = 200
    look_at = (
        pos[0] + forward_x * look_distance,
        pos[1] + forward_y * look_distance,
        pos[2] + forward_z * look_distance
    )
    
    gluLookAt(pos[0], pos[1], pos[2], look_at[0], look_at[1], look_at[2], 0, 1, 0)
    
    # Enable depth test for rendering
    glEnable(GL_DEPTH_TEST)
    
    # Draw floor
    active_zone_near = TRACKZONE['offset_z']
    draw_floor(0, (0.25, 0.25, 0.3, 0.5), z_max=active_zone_near)
    
    # Draw trackzones (wireframe)
    tz = TRACKZONE
    tz_bounds = (
        tz['center_x'] - tz['width']/2, tz['center_x'] + tz['width']/2,
        tz['offset_y'], tz['offset_y'] + tz['height'],
        tz['offset_z'], tz['offset_z'] + tz['depth']
    )
    draw_box_wireframe(tz_bounds, (0, 1, 1, 0.3))
    
    ptz = PASSIVE_TRACKZONE
    ptz_bounds = (
        ptz['center_x'] - ptz['width']/2, ptz['center_x'] + ptz['width']/2,
        ptz['offset_y'], ptz['offset_y'] + ptz['height'],
        ptz['offset_z'], ptz['offset_z'] + ptz['depth']
    )
    draw_box_wireframe(ptz_bounds, (1, 0.6, 0, 0.2))
    
    # Draw panels
    for (unit, panel_num), panel in panel_system.panels.items():
        draw_panel(panel['center'], panel['angle'], PANEL_SIZE, panel['brightness'])
    
    # Draw calibration markers with ID indicators
    if show_markers:
        # Color coding for marker IDs
        marker_colors = {
            0: (1, 0, 0, 1),      # Red
            1: (0, 1, 0, 1),      # Green
            2: (0, 0, 1, 1),      # Blue
            3: (1, 1, 0, 1),      # Yellow
            4: (1, 0, 1, 1),      # Magenta
            5: (0, 1, 1, 1),      # Cyan
            6: (1, 0.5, 0, 1),    # Orange
        }
        for marker_id, marker_data in MARKER_POSITIONS.items():
            pos_m = marker_data['pos']
            tex_id = marker_textures.get(marker_id)
            is_vertical = marker_data.get('vertical', False)
            draw_marker(marker_id, pos_m, MARKER_SIZE, tex_id, vertical=is_vertical)
            
            # Draw colored sphere above marker as ID indicator
            label_offset = 25 if not is_vertical else 35
            sphere_pos = (pos_m[0], pos_m[1] + label_offset, pos_m[2])
            color = marker_colors.get(marker_id, (1, 1, 1, 1))
            draw_sphere(sphere_pos, 8, color, segments=8)
    
    # Draw light
    brightness = light.get_brightness()
    radius = 8 + brightness * 7
    draw_sphere(light.position, radius, (1, 1, brightness, 1))
    
    # Draw tracked people
    for person in tracked_manager.get_all():
        draw_tracked_person(person, zone_checker=tracked_manager._get_zone)
    
    # Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def draw_camera_view_overlay(fbo_data, x, y, width, height, label, font, border_color):
    """Draw a camera view texture as a 2D overlay"""
    # Draw border
    glColor4f(*border_color)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    # Draw camera view texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, fbo_data['texture'])
    glColor4f(1, 1, 1, 1)
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x, y)
    glTexCoord2f(1, 0); glVertex2f(x + width, y)
    glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
    glTexCoord2f(0, 1); glVertex2f(x, y + height)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    
    # Draw label
    draw_text_2d(x + 5, y + height - 20, label, font, (255, 255, 255))


# =============================================================================
# GUI SLIDER
# =============================================================================

class Checkbox:
    """Simple checkbox for GUI"""
    def __init__(self, x, y, size, label, checked=False):
        self.rect = pygame.Rect(x, y, size, size)
        self.label = label
        self.checked = checked
        self.size = size
    
    def handle_event(self, event, screen_height):
        """Handle mouse events. Returns True if value changed."""
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse_y = screen_height - event.pos[1]
            if self.rect.collidepoint(event.pos[0], mouse_y):
                self.checked = not self.checked
                return True
        return False
    
    def draw(self, font):
        """Draw the checkbox using OpenGL"""
        x, y, s = self.rect.x, self.rect.y, self.size
        
        # Background
        glColor4f(0.2, 0.2, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + s, y)
        glVertex2f(x + s, y + s)
        glVertex2f(x, y + s)
        glEnd()
        
        # Checkmark if checked
        if self.checked:
            glColor4f(0.3, 0.8, 0.4, 1.0)
            margin = s * 0.2
            glBegin(GL_QUADS)
            glVertex2f(x + margin, y + margin)
            glVertex2f(x + s - margin, y + margin)
            glVertex2f(x + s - margin, y + s - margin)
            glVertex2f(x + margin, y + s - margin)
            glEnd()
        
        # Border
        glColor4f(0.5, 0.5, 0.5, 1.0)
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + s, y)
        glVertex2f(x + s, y + s)
        glVertex2f(x, y + s)
        glEnd()
        
        # Label
        draw_text_2d(x + s + 8, y + 2, self.label, font)


class Slider:
    """Simple horizontal slider for GUI"""
    def __init__(self, x, y, width, height, min_val, max_val, value, label, format_str="{:.1f}", autotuned=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.format_str = format_str
        self.dragging = False
        self.autotuned = autotuned  # If True, slider shows auto-tuned visual style
        # Store original Y offset from display height for repositioning
        self._y_offset = None
    
    def handle_event(self, event, screen_height):
        """Handle mouse events. Returns True if value changed."""
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse_y = screen_height - event.pos[1]
            # Expand click area: include label above (+22px) and padding below (-5px)
            # Total clickable height = height + 27px (much easier to hit)
            expanded = pygame.Rect(self.rect.x, self.rect.y - 5, self.rect.width, self.rect.height + 27)
            if expanded.collidepoint(event.pos[0], mouse_y):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
            return True
        return False
    
    def _update_value(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.rect.x, self.rect.width))
        ratio = rel_x / self.rect.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
    
    def draw(self, font):
        """Draw the slider using OpenGL"""
        x, y, w, h = self.rect.x, self.rect.y, self.rect.width, self.rect.height
        
        # Background - slightly different for autotuned
        if self.autotuned:
            glColor4f(0.15, 0.15, 0.22, 1.0)
        else:
            glColor4f(0.2, 0.2, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Fill based on value - auto-tuned uses a distinct color
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w = w * ratio
        if self.autotuned:
            glColor4f(0.25, 0.45, 0.55, 0.8)  # Muted teal for auto-tuned
        else:
            glColor4f(0.3, 0.6, 0.8, 1.0)     # Bright blue for manual
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + fill_w, y)
        glVertex2f(x + fill_w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Thumb indicator (only for manual sliders)
        if not self.autotuned:
            thumb_x = x + fill_w
            glColor4f(0.9, 0.9, 0.9, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(thumb_x - 2, y - 2)
            glVertex2f(thumb_x + 2, y - 2)
            glVertex2f(thumb_x + 2, y + h + 2)
            glVertex2f(thumb_x - 2, y + h + 2)
            glEnd()
        
        # Highlight when dragging
        if self.dragging:
            glColor4f(0.4, 0.8, 1.0, 0.3)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + h)
            glVertex2f(x, y + h)
            glEnd()
        
        # Border - auto-tuned gets dashed-look subtle border
        if self.dragging:
            border_color = (0.6, 0.8, 1.0, 1.0)
        elif self.autotuned:
            border_color = (0.35, 0.45, 0.55, 0.6)
        else:
            border_color = (0.5, 0.5, 0.5, 1.0)
        glColor4f(*border_color)
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Label and value
        val_str = self.format_str.format(self.value)
        if self.autotuned:
            draw_text_2d(x, y + h + 5, f"{self.label}: {val_str}", font, (160, 180, 190))
        else:
            draw_text_2d(x, y + h + 5, f"{self.label}: {val_str}", font)


# =============================================================================
# GUI SECTION HEADERS
# =============================================================================

def draw_section_header(x, y, width, title, font, color=(150, 150, 200), icon=""):
    """Draw a section header with a horizontal line and title"""
    label = f"{icon} {title}" if icon else title
    # Draw header text
    draw_text_2d(x, y, label, font, color)
    # Draw subtle line under title
    line_y = y - 4
    glColor4f(color[0]/255*0.5, color[1]/255*0.5, color[2]/255*0.5, 0.4)
    glLineWidth(1)
    glBegin(GL_LINES)
    glVertex2f(x, line_y)
    glVertex2f(x + width, line_y)
    glEnd()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Single instance check
    if not acquire_single_instance_lock():
        sys.exit(1)
    atexit.register(release_single_instance_lock)
    
    # Graceful shutdown flag
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    pygame.init()
    pygame.font.init()
    
    # Get display info for fullscreen
    display_info = pygame.display.Info()
    fullscreen_size = (display_info.current_w, display_info.current_h)
    windowed_size = (1920, 1080)  # Fallback windowed size
    is_fullscreen = True
    display = fullscreen_size
    # Use NOFRAME instead of FULLSCREEN - stays visible when focus is lost
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | NOFRAME)
    pygame.display.set_caption("3D Light Controller V3 - Production")
    
    font = pygame.font.SysFont('monospace', 14)
    font_small = pygame.font.SysFont('monospace', 12)
    font_label = pygame.font.SysFont('monospace', 11)
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.1, 0.1, 0.15, 1.0)
    
    # Load marker textures
    marker_textures = load_marker_textures()
    show_markers = SHOW_MARKERS
    show_labels = SHOW_LABELS
    show_camera_views = SHOW_CAMERA_VIEWS
    
    # Camera view framebuffers (initialized later when needed)
    camera_fbos = {}
    
    # Camera - positioned beyond passive zone (high Z), looking back at panels (low Z)
    # Panels are near Z=0, passive zone ends at Z=553
    cam_rot_x = 25          # Looking down at the scene
    cam_rot_y = 0           # Looking toward panels (opposite of 180)
    cam_distance = 900      # Even farther back
    cam_target = np.array([-150.0, 0.0, 150.0])  # Target near panels
    cam_target_default = cam_target.copy()  # For reset
    cam_rot_x_default = cam_rot_x
    cam_rot_y_default = cam_rot_y
    cam_distance_default = cam_distance
    middle_mouse_down = False  # For panning
    
    # GUI panel width (two-column layout)
    gui_width = 560
    view_width = display[0] - gui_width
    
    # Create systems
    panel_system = PanelSystem()
    light = PointLight()
    wander_box = dict(WANDER_BOX)
    wander = WanderBehavior(light, wander_box)
    
    # Tracked person manager
    tracked_manager = TrackedPersonManager()
    
    # Find available database files (always in script's directory)
    import glob
    db_pattern_dir = SCRIPT_DIR
    db_files = sorted(
        os.path.basename(f) for f in 
        glob.glob(os.path.join(db_pattern_dir, "*.db")) + glob.glob(os.path.join(db_pattern_dir, "tracking_*.db"))
    )
    db_files = list(dict.fromkeys(db_files))  # deduplicate preserving order
    if "tracking_history.db" not in db_files:
        db_files.insert(0, "tracking_history.db")
    else:
        # Move tracking_history.db to front
        db_files.remove("tracking_history.db")
        db_files.insert(0, "tracking_history.db")
    current_db_file = os.path.join(SCRIPT_DIR, "tracking_history.db")
    current_db_index = 0
    
    # Tracking database (absolute path so it works from any cwd)
    tracking_db = TrackingDatabase(current_db_file)
    print(f"💾 Tracking database: {current_db_file}")
    
    # Restore daily_count from database so it persists across restarts
    try:
        startup_stats = tracking_db.get_current_stats()
        restored_count = startup_stats.get('daily_unique_people', 0)
        tracked_manager.daily_count = restored_count
        print(f"📊 Restored daily count from DB: {restored_count} unique people today")
    except Exception as e:
        print(f"⚠️ Could not restore daily count from DB: {e}")
    
    # Behavior system with default personality
    meta_params = MetaParameters()
    behavior = BehaviorSystem(meta=meta_params, database=tracking_db)
    print(f"🧠 Behavior system initialized")
    
    # Connect tracked manager callbacks to behavior system
    tracked_manager.on_person_entered = behavior.on_person_entered
    tracked_manager.on_person_left = behavior.on_person_left
    tracked_manager.on_position_updated = behavior.update_person_position
    tracked_manager.on_zone_updated = behavior.set_person_active
    
    # For periodic stats refresh
    last_stats_update = time.time()
    db_stats = {'people_last_minute': 0, 'avg_speed': 0, 'flow_left_to_right': 0, 
                'flow_right_to_left': 0, 'active_events': 0, 'passive_events': 0}
    
    # OSC setup
    osc_handler = OSCHandler(tracked_manager, tracking_db)
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/tracker/person/*", osc_handler.handle_person)
    osc_dispatcher.map("/tracker/zone/*", osc_handler.handle_zone)
    osc_dispatcher.map("/tracker/count", osc_handler.handle_count)
    
    # Create OSC server - use BlockingOSCUDPServer to avoid thread exhaustion
    # ThreadingOSCUDPServer creates a new thread per message which exhausts
    # resources at high message rates (150+ msgs/sec)
    osc_server_instance = osc_server.BlockingOSCUDPServer(
        (OSC_IP, OSC_PORT), osc_dispatcher
    )
    # Set a short timeout so handle_request doesn't block forever
    osc_server_instance.timeout = 0.001
    # Allow socket reuse (helps when restarting quickly)
    osc_server_instance.socket.setsockopt(
        socket.SOL_SOCKET, 
        socket.SO_REUSEADDR, 
        1
    )
    print(f"📡 OSC server listening on {OSC_IP}:{OSC_PORT}")
    
    # WebSocket broadcaster for public viewer
    ws_broadcaster = None
    if WEBSOCKET_AVAILABLE and WEBSOCKET_ENABLED:
        ws_broadcaster = WebSocketBroadcaster(port=WEBSOCKET_PORT)
        ws_broadcaster.start()
    
    # Daily report system
    report_generator = DailyReportGenerator(tracking_db)
    daily_report_scheduler = DailyReportScheduler(
        report_generator=report_generator,
        ws_broadcaster=ws_broadcaster,
        on_report_ready=lambda r: logger.info(f"📊 Daily report ready: {r.total_unique_people} people tracked")
    )
    daily_report_scheduler.start()
    
    # Track current report for visualization
    current_daily_report: Optional[DailyReport] = None
    cached_report_dict: Optional[dict] = None  # Cached serialized report
    report_version = 0  # Increment when report changes
    last_sent_report_version = -1  # Track what version client has
    show_trends = True  # Toggle with 'T' key - ON by default
    population_day = datetime.now().date()
    
    def on_report_ready(report: DailyReport):
        nonlocal current_daily_report, cached_report_dict, report_version
        current_daily_report = report
        cached_report_dict = report.to_dict() if report else None
        report_version += 1
        
        # Save auto-tune learnings to database and apply them
        try:
            if report.auto_tuning_analysis:
                learnings = auto_tuner.compute_daily_learnings(report)
                
                # Save to database
                report_data = {
                    'total_unique_people': report.total_unique_people,
                    'peak_hour': report.peak_hour,
                }
                tracking_db.save_autotune_learnings(
                    date_str=report.date,
                    report_data=report_data,
                    tuning_analysis=report.auto_tuning_analysis,
                    optimal_values=learnings.get('optimal_values'),
                    learned_caps=learnings.get('learned_caps'),
                )
                logger.info(f"🧠 Auto-tune learnings saved for {report.date}")
                
                # Apply learned caps immediately for tomorrow
                learned_caps = learnings.get('learned_caps', {})
                if learned_caps:
                    auto_tuner.learned_caps_adjustments = learned_caps
                    for name, cap_val in learned_caps.items():
                        if name in auto_tuner.caps:
                            auto_tuner.caps[name] = cap_val
                    logger.info(f"🧠 Applied {len(learned_caps)} learned cap adjustments")
        except Exception as e:
            logger.warning(f"Failed to save/apply auto-tune learnings: {e}")
    
    daily_report_scheduler.on_report_ready = on_report_ready
    
    # Create sliders - Two column layout
    # Left column: Manual controls (calibration + budget + mode status)
    # Right column: Auto-tuned parameters (personality + output multipliers + auto-tune status)
    col_padding = 20
    col_gap = 20
    col_width = (gui_width - col_padding * 2 - col_gap) // 2  # ~250px each
    left_col_x = view_width + col_padding
    right_col_x = view_width + col_padding + col_width + col_gap
    slider_x = left_col_x  # Keep for backward compat
    slider_w = col_width
    slider_h = 16
    
    # Y-offset definitions (from top of screen)
    # Left column: Calibration (Manual) + Manual Controls
    # Right column: Personality (Auto-tuned) + Output Multipliers (Auto-tuned)
    # Note: section header ~70, sub-label ~90, first slider needs to clear label text
    slider_y_offsets = {
        # ── LEFT COLUMN: Calibration (Manual) ──
        'offset_x': 120, 'offset_z': 160,
        'scale_x': 205, 'scale_z': 245,
        # Checkbox
        'invert_x_cb': 280,
        # ── LEFT COLUMN: Manual Controls ──
        'interaction_budget': 420,
        # ── RIGHT COLUMN: Personality (Auto-tuned) ──
        'responsiveness': 120, 'energy': 160, 'attention_span': 200,
        'sociability': 240, 'exploration': 280, 'memory': 320,
        # ── RIGHT COLUMN: Output Multipliers (Auto-tuned) ──
        'brightness_global': 385, 'speed_global': 425, 'pulse_global': 465,
        'follow_speed_global': 505, 'dwell_influence': 545, 'idle_trend_weight': 585,
    }
    
    # Track which sliders go in which column
    left_col_sliders = {'offset_x', 'offset_z', 'scale_x', 'scale_z', 'interaction_budget'}
    right_col_sliders = {'responsiveness', 'energy', 'attention_span', 'sociability',
                         'exploration', 'memory', 'brightness_global', 'speed_global',
                         'pulse_global', 'follow_speed_global', 'dwell_influence', 'idle_trend_weight'}
    
    def reposition_sliders():
        """Reposition all sliders/checkboxes based on current display size"""
        nonlocal left_col_x, right_col_x, col_width, slider_x, slider_w
        left_col_x = view_width + col_padding
        right_col_x = view_width + col_padding + col_width + col_gap
        slider_x = left_col_x
        slider_w = col_width
        for name, slider in all_sliders.items():
            y_off = slider_y_offsets[name]
            if name in right_col_sliders:
                slider.rect.x = right_col_x
            else:
                slider.rect.x = left_col_x
            slider.rect.y = display[1] - y_off
            slider.rect.width = col_width
        for name, checkbox in checkboxes.items():
            cb_off = slider_y_offsets.get(f'{name}_cb', 260)
            checkbox.rect.x = left_col_x
            checkbox.rect.y = display[1] - cb_off
    
    # Calibration sliders (LEFT column - manual, user adjustable)
    sliders = {
        'offset_x': Slider(left_col_x, display[1] - 120, col_width, slider_h, -200, 200, 0, "Offset X"),
        'offset_z': Slider(left_col_x, display[1] - 160, col_width, slider_h, 0, 500, 250, "Offset Z"),
        'scale_x': Slider(left_col_x, display[1] - 205, col_width, slider_h, 0.5, 2.0, 1.0, "Scale X", "{:.2f}"),
        'scale_z': Slider(left_col_x, display[1] - 245, col_width, slider_h, 0.5, 2.0, 1.0, "Scale Z", "{:.2f}"),
    }
    
    # Calibration checkboxes (LEFT column)
    checkboxes = {
        'invert_x': Checkbox(left_col_x, display[1] - 280, 14, "Invert X Direction", checked=False),
    }
    
    # Personality sliders (RIGHT column - auto-tuned by behavior system)
    personality_sliders = {
        'responsiveness': Slider(right_col_x, display[1] - 120, col_width, slider_h, 0, 1, 0.5, "Responsiveness", "{:.2f}", autotuned=True),
        'energy': Slider(right_col_x, display[1] - 160, col_width, slider_h, 0, 1, 0.5, "Energy", "{:.2f}", autotuned=True),
        'attention_span': Slider(right_col_x, display[1] - 200, col_width, slider_h, 0, 1, 0.5, "Attention", "{:.2f}", autotuned=True),
        'sociability': Slider(right_col_x, display[1] - 240, col_width, slider_h, 0, 1, 0.5, "Sociability", "{:.2f}", autotuned=True),
        'exploration': Slider(right_col_x, display[1] - 280, col_width, slider_h, 0, 1, 0.5, "Exploration", "{:.2f}", autotuned=True),
        'memory': Slider(right_col_x, display[1] - 320, col_width, slider_h, 0, 1, 0.5, "Memory", "{:.2f}", autotuned=True),
    }
    
    # Global multiplier sliders (RIGHT column - auto-tuned) + interaction_budget (LEFT column - manual)
    global_sliders = {
        'brightness_global': Slider(right_col_x, display[1] - 385, col_width, slider_h, 0.2, 5.0, 1.0, "Brightness ×", "{:.2f}", autotuned=True),
        'speed_global': Slider(right_col_x, display[1] - 425, col_width, slider_h, 0.2, 2.0, 1.0, "Speed ×", "{:.2f}", autotuned=True),
        'pulse_global': Slider(right_col_x, display[1] - 465, col_width, slider_h, 0.3, 3.0, 1.0, "Pulse ×", "{:.2f}", autotuned=True),
        'follow_speed_global': Slider(right_col_x, display[1] - 505, col_width, slider_h, 0.5, 3.0, 1.0, "Follow Spd ×", "{:.2f}", autotuned=True),
        'dwell_influence': Slider(right_col_x, display[1] - 545, col_width, slider_h, 0.0, 2.0, 1.0, "Dwell Influence", "{:.2f}", autotuned=True),
        'idle_trend_weight': Slider(right_col_x, display[1] - 585, col_width, slider_h, 0.0, 2.0, 1.0, "Idle Trend ×", "{:.2f}", autotuned=True),
        'interaction_budget': Slider(left_col_x, display[1] - 420, col_width, slider_h, 0.0, 120.0, 60.0, "Interaction Budget", "{:.0f}", autotuned=False),
    }
    
    # Combine all sliders
    all_sliders = {**sliders, **personality_sliders, **global_sliders}
    
    # Load saved slider settings
    saved_settings = load_slider_settings()
    if saved_settings:
        apply_slider_settings(all_sliders, saved_settings, checkboxes)
        # Apply calibration settings to tracked manager
        tracked_manager.offset_x = sliders['offset_x'].value
        tracked_manager.offset_z = sliders['offset_z'].value
        tracked_manager.scale_x = sliders['scale_x'].value
        tracked_manager.scale_z = sliders['scale_z'].value
        tracked_manager.invert_x = checkboxes['invert_x'].checked
        # Apply personality settings to meta params
        for name, slider in personality_sliders.items():
            setattr(meta_params, name, slider.value)
        for name, slider in global_sliders.items():
            if name != 'interaction_budget':
                setattr(meta_params, name, slider.value)
        print(f"📁 Restored {len(saved_settings)} slider settings")

    # Auto-tuning manager (trend responsive adjustments)
    auto_tuner = AutoTuningManager(meta=meta_params, sliders=all_sliders, database=tracking_db)
    
    # Load and apply historical learnings from previous days' reports
    auto_tuner.load_learnings_from_db()
    auto_tuner.apply_learnings_to_values()
    
    # Track when to save sliders (debounce saves)
    last_slider_save = time.time()
    slider_save_interval = 2.0  # Save at most every 2 seconds
    sliders_dirty = False  # Track if sliders have changed
    
    # Art-Net
    artnet = None
    if ARTNET_AVAILABLE:
        try:
            artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
            artnet.start()
            print(f"🎨 Art-Net output to {TARGET_IP}")
        except Exception as e:
            print(f"Art-Net failed: {e}")
    
    clock = pygame.time.Clock()
    last_time = time.time()
    mouse_down = False
    last_mouse = (0, 0)
    slider_active = False
    
    # Current preset name
    current_preset = "default"
    preset_names = list(PRESETS.keys())
    
    # Health monitoring
    start_time = time.time()
    last_health_log = time.time()
    last_db_prune = time.time()
    last_hour_aggregated = datetime.now().hour  # Track hourly aggregation
    frame_count = 0
    total_osc_messages = 0
    
    logger.info("Light controller V3 started - entering main loop")
    print("\n" + "="*60)
    print("V3 DEVELOPMENT VERSION - Visual Debugging Enabled")
    print("="*60)
    print("Controls:")
    print("  L = Toggle coordinate labels")
    print("  M = Toggle AR markers")
    print("  SPACE = Toggle wandering")
    print("  A = Toggle auto-tuning")
    print("  P = Cycle presets")
    print("  T = Toggle trends visualization")
    print("  F = Toggle fullscreen/windowed")
    print("  R = Generate daily report (manual)")
    print("  F2 = Take screenshot")
    print("  Q/ESC = Quit")
    print("="*60)
    print("📅 Daily report auto-generates at 12:01 AM")
    print("="*60 + "\n")

    running = True
    while running and not shutdown_requested:
        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            # Check all sliders
            for name, slider in all_sliders.items():
                if slider.handle_event(event, display[1]):
                    slider_active = True
                    sliders_dirty = True  # Mark for saving
                    # Update calibration values
                    if name in ('offset_x', 'offset_z', 'scale_x', 'scale_z'):
                        tracked_manager.offset_x = sliders['offset_x'].value
                        tracked_manager.offset_z = sliders['offset_z'].value
                        tracked_manager.scale_x = sliders['scale_x'].value
                        tracked_manager.scale_z = sliders['scale_z'].value
                    # Update personality values
                    elif name in personality_sliders:
                        setattr(meta_params, name, slider.value)
                    # Update global multipliers
                    elif name in global_sliders:
                        if name != 'interaction_budget':
                            setattr(meta_params, name, slider.value)
            
            # Debug: log click position when clicking in GUI area
            if event.type == MOUSEBUTTONDOWN and event.button == 1 and event.pos[0] >= view_width:
                mouse_y_gl = display[1] - event.pos[1]
                hit_any = False
                for sname, s in all_sliders.items():
                    expanded = pygame.Rect(s.rect.x, s.rect.y - 5, s.rect.width, s.rect.height + 25)
                    if expanded.collidepoint(event.pos[0], mouse_y_gl):
                        hit_any = True
                        break
                if not hit_any:
                    print(f"🎯 Click in GUI at pygame=({event.pos[0]}, {event.pos[1]}) gl_y={mouse_y_gl} - no slider hit")
                    # Find nearest slider
                    nearest = min(all_sliders.items(), key=lambda kv: abs(kv[1].rect.y - mouse_y_gl))
                    print(f"   Nearest slider: {nearest[0]} at gl_y={nearest[1].rect.y} (dist={abs(nearest[1].rect.y - mouse_y_gl)})")
            
            # Check checkboxes
            for name, checkbox in checkboxes.items():
                if checkbox.handle_event(event, display[1]):
                    sliders_dirty = True  # Mark for saving
                    if name == 'invert_x':
                        tracked_manager.invert_x = checkbox.checked
                        print(f"🔄 Invert X: {'ON' if checkbox.checked else 'OFF'}")
            
            if event.type == MOUSEBUTTONUP:
                slider_active = False
            
            if event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    running = False
                elif event.key == K_SPACE:
                    wander.enabled = not wander.enabled
                elif event.key == K_a:
                    auto_tuner.set_enabled(not auto_tuner.enabled)
                    print(f"Auto-tuning {'enabled' if auto_tuner.enabled else 'disabled'}")
                elif event.key == K_m:
                    show_markers = not show_markers
                    print(f"Markers {'visible' if show_markers else 'hidden'}")
                elif event.key == K_l:
                    show_labels = not show_labels
                    print(f"Labels {'visible' if show_labels else 'hidden'}")
                elif event.key == K_c:
                    show_camera_views = not show_camera_views
                    print(f"Camera views {'visible' if show_camera_views else 'hidden'}")
                elif event.key == K_p:
                    # Cycle through presets
                    idx = preset_names.index(current_preset)
                    idx = (idx + 1) % len(preset_names)
                    current_preset = preset_names[idx]
                    meta_params = load_preset(current_preset)
                    behavior.meta = meta_params
                    # Update sliders to match preset
                    for name, slider in personality_sliders.items():
                        slider.value = getattr(meta_params, name)
                    for name, slider in global_sliders.items():
                        slider.value = getattr(meta_params, name)
                    print(f"🎭 Preset: {current_preset}")
                elif event.key == K_t:
                    show_trends = not show_trends
                    print(f"Trends visualization {'visible' if show_trends else 'hidden'}")
                elif event.key == K_f:
                    # Toggle fullscreen mode
                    is_fullscreen = not is_fullscreen
                    pygame.display.quit()
                    pygame.display.init()
                    if is_fullscreen:
                        display = fullscreen_size
                        # Use NOFRAME instead of FULLSCREEN - stays visible when focus is lost
                        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | NOFRAME)
                    else:
                        display = windowed_size
                        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
                    pygame.display.set_caption("3D Light Controller V3 - Production")
                    # Reinitialize OpenGL state after display change
                    glEnable(GL_DEPTH_TEST)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glClearColor(0.1, 0.1, 0.15, 1.0)
                    glViewport(0, 0, display[0], display[1])
                    # Update layout for new display size
                    view_width = display[0] - gui_width
                    reposition_sliders()
                    print(f"{'Fullscreen' if is_fullscreen else 'Windowed'} mode ({display[0]}x{display[1]})")
                elif event.key == K_HOME:
                    # Reset camera to default view
                    cam_rot_x = cam_rot_x_default
                    cam_rot_y = cam_rot_y_default
                    cam_distance = cam_distance_default
                    cam_target = cam_target_default.copy()
                    print("📷 Camera reset to default view")
                elif event.key == K_r:
                    # Manual report generation (for testing)
                    print("📊 Generating manual daily report...")
                    report = daily_report_scheduler.generate_now()
                    if report:
                        current_daily_report = report
                        print(f"📊 Report ready: {report.total_unique_people} people, peak at {report.peak_hour}:00")
                elif event.key == K_d:
                    # Cycle through available database files
                    if len(db_files) > 1:
                        current_db_index = (current_db_index + 1) % len(db_files)
                        new_db_file = os.path.join(SCRIPT_DIR, db_files[current_db_index])
                        # Close old database and open new one
                        tracking_db.close()
                        tracking_db = TrackingDatabase(new_db_file)
                        behavior.database = tracking_db
                        # Update OSC handler and report generator
                        osc_handler.database = tracking_db
                        report_generator.database = tracking_db
                        current_db_file = new_db_file
                        # Restore daily count from new database
                        try:
                            switch_stats = tracking_db.get_current_stats()
                            tracked_manager.daily_count = switch_stats.get('daily_unique_people', 0)
                        except Exception:
                            pass
                        # Clear cached report
                        current_daily_report = None
                        print(f"💾 Switched to database: {new_db_file}")
                    else:
                        print(f"💾 Only one database available: {current_db_file}")
                elif event.key == K_F2:
                    # Screenshot - save window capture to repo
                    try:
                        screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'screenshots')
                        os.makedirs(screenshot_dir, exist_ok=True)
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        screenshot_path = os.path.join(screenshot_dir, f'screenshot_{timestamp_str}.png')
                        screenshot_surface = pygame.Surface(display)
                        pixel_data = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
                        screenshot_surface = pygame.image.fromstring(pixel_data, display, 'RGB', True)
                        pygame.image.save(screenshot_surface, screenshot_path)
                        print(f"📸 Screenshot saved: {screenshot_path}")
                    except Exception as e:
                        print(f"⚠️ Screenshot failed: {e}")
            
            # Camera rotation (only in 3D view area)
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < view_width and not slider_active:
                    mouse_down = True
                    last_mouse = event.pos
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
            # Middle mouse button for panning
            elif event.type == MOUSEBUTTONDOWN and event.button == 2:
                if event.pos[0] < view_width and not slider_active:
                    middle_mouse_down = True
                    last_mouse = event.pos
            elif event.type == MOUSEBUTTONUP and event.button == 2:
                middle_mouse_down = False
            elif event.type == MOUSEMOTION:
                mods = pygame.key.get_mods()
                if middle_mouse_down or (mouse_down and mods & KMOD_SHIFT):
                    # Pan camera (Shift+left drag or middle mouse drag)
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    # Pan speed scales with distance
                    pan_speed = cam_distance * 0.002
                    # Pan in camera-relative directions
                    angle_rad = math.radians(cam_rot_y)
                    cam_target[0] -= dx * pan_speed * math.cos(angle_rad)
                    cam_target[2] -= dx * pan_speed * math.sin(angle_rad)
                    cam_target[1] += dy * pan_speed
                    last_mouse = event.pos
                elif mouse_down:
                    # Rotate camera
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    cam_rot_y += dx * 0.5
                    cam_rot_x += dy * 0.3
                    cam_rot_x = max(-89, min(89, cam_rot_x))
                    last_mouse = event.pos
            elif event.type == MOUSEWHEEL:
                cam_distance -= event.y * 30
                cam_distance = max(100, min(1500, cam_distance))
        
        # Keyboard controls for light
        keys = pygame.key.get_pressed()
        if not wander.enabled:
            move_speed = 100
            now = time.time()
            dt_keys = min(now - last_time, 0.1)
            if keys[K_LEFT]:
                light.target_position[0] -= move_speed * dt_keys
            if keys[K_RIGHT]:
                light.target_position[0] += move_speed * dt_keys
            if keys[K_UP]:
                light.target_position[1] += move_speed * dt_keys
            if keys[K_DOWN]:
                light.target_position[1] -= move_speed * dt_keys
            if keys[K_w]:
                light.target_position[2] -= move_speed * dt_keys
            if keys[K_s]:
                light.target_position[2] += move_speed * dt_keys
        
        # Process OSC messages (non-blocking with select())
        # Use select() to check if data is available before calling handle_request()
        # This avoids the timeout wait when no messages are pending
        import select
        osc_messages_this_frame = 0
        max_osc_per_frame = 100  # Can handle more since we're not waiting on timeouts
        while osc_messages_this_frame < max_osc_per_frame:
            # Check if socket has data ready (0 timeout = immediate return)
            ready, _, _ = select.select([osc_server_instance.socket], [], [], 0)
            if not ready:
                break  # No data available, exit loop immediately
            try:
                osc_server_instance.handle_request()
                osc_messages_this_frame += 1
            except Exception:
                break  # Error handling request
        
        # Update
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now
        now_dt = datetime.now()

        if now_dt.date() != population_day and (now_dt.hour > 0 or now_dt.minute >= 1):
            tracked_manager.reset_daily_population()
            population_day = now_dt.date()
        
        # Cleanup stale tracked people
        tracked_manager.cleanup_stale()
        
        # Get zone counts
        active_count = tracked_manager.count_active()
        passive_count = tracked_manager.count_passive()
        
        # Get flow balance from database stats
        ltr = db_stats.get('flow_left_to_right', 0)
        rtl = db_stats.get('flow_right_to_left', 0)
        total_flow = ltr + rtl
        flow_balance = (ltr - rtl) / total_flow if total_flow > 0 else 0.0
        
        # Calculate passive rate (people per minute)
        passive_rate = db_stats.get('passive_events', 0) / 60.0  # Rough estimate
        
        # Update behavior system
        current_pos = tuple(light.position)
        behavior_params = behavior.update(
            dt=dt,
            active_count=active_count,
            passive_count=passive_count,
            current_pos=current_pos,
            passive_rate=passive_rate,
            flow_balance=flow_balance
        )

        behavior_status = behavior.get_status()
        auto_tuner.update(behavior_status, now)
        
        # Update light position for feedback learning context
        behavior.set_light_position(*current_pos)
        
        # Apply behavior parameters to light
        light.brightness_min = int(behavior_params.get('brightness_min', 5))
        light.brightness_max = int(behavior_params.get('brightness_max', 30))
        light.pulse_speed = behavior_params.get('pulse_speed', 2000)
        light.move_speed = behavior_params.get('move_speed', 50)
        light.falloff_radius = behavior_params.get('falloff_radius', 50)
        
        # Update wander behavior based on behavior system
        wander.update_wander_box(behavior.get_wander_box())
        wander.wander_interval = behavior_params.get('wander_interval', 3.0)
        
        # Animated wander box handles tracking - no follow target needed
        # The box contracts tightly around people, so normal wandering
        # within the box naturally tracks them
        wander.clear_follow_target()
        
        # Handle gesture target
        gesture_target = behavior.get_gesture_target()
        if gesture_target is not None:
            wander.set_gesture_target(gesture_target)
        else:
            wander.clear_gesture_target()
        
        # Update wander and light
        wander.update(dt)
        light.update(dt)
        panel_system.calculate_brightness(light)
        
        # Broadcast state to WebSocket clients (throttled)
        if ws_broadcaster and (not hasattr(ws_broadcaster, 'last_broadcast') or 
                                time.time() - ws_broadcaster.last_broadcast >= WEBSOCKET_BROADCAST_INTERVAL):
            try:
                # Build behavior status text
                status_text = behavior_status.get('status_text', '')
                
                # Extract realtime trends for public viewer
                idle_trends = behavior_status.get('idle_trends', {})
                realtime_trends = None
                if idle_trends:
                    realtime_trends = {
                        'period': idle_trends.get('period', 'unknown'),
                        'seconds_since_update': idle_trends.get('seconds_since_update', 0),
                        # 1 minute window
                        'recent': {
                            'available': idle_trends.get('has_recent', False),
                            'passive': idle_trends.get('recent_passive', 0),
                            'active': idle_trends.get('recent_active', 0),
                        },
                        # 5 minute window
                        'short': {
                            'available': idle_trends.get('has_short', False),
                            'passive': idle_trends.get('short_passive', 0),
                            'active': idle_trends.get('short_active', 0),
                            'activity': idle_trends.get('short_activity', 0),
                        },
                        # 15 minute window  
                        'medium': {
                            'available': idle_trends.get('has_medium', False),
                            'passive': idle_trends.get('medium_passive', 0),
                            'active': idle_trends.get('medium_active', 0),
                            'activity': idle_trends.get('medium_activity', 0),
                        },
                        # 60 minute window
                        'long': {
                            'available': idle_trends.get('has_long', False),
                            'passive': idle_trends.get('long_passive', 0),
                            'active': idle_trends.get('long_active', 0),
                            'activity': idle_trends.get('long_activity', 0),
                        },
                    }
                
                state = {
                    'type': 'state_update',
                    'light': {
                        'x': float(light.position[0]),
                        'y': float(light.position[1]),
                        'z': float(light.position[2]),
                        'brightness': float(light.get_brightness()),
                        'falloff_radius': float(light.falloff_radius)
                    },
                    'wander_box': {
                        'min_x': float(wander.wander_box['min_x']),
                        'max_x': float(wander.wander_box['max_x']),
                        'min_y': float(wander.wander_box['min_y']),
                        'max_y': float(wander.wander_box['max_y']),
                        'min_z': float(wander.wander_box['min_z']),
                        'max_z': float(wander.wander_box['max_z']),
                        'enabled': wander.enabled
                    },
                    'panels': panel_system.get_dmx_values()[:12],
                    'people': [
                        {
                            'id': p.track_id,
                            'daily_id': p.daily_id,
                            'x': p.x,
                            'y': p.y,
                            'z': p.z,
                            'vx': p.vx,
                            'vz': p.vz,
                            'zone': p.zone
                        }
                        for p in tracked_manager.get_all()
                    ],
                    'counts': {
                        'active': active_count,
                        'passive': passive_count,
                        'total': len(tracked_manager.get_all())
                    },
                    'population_count': tracked_manager.daily_count,
                    'wander_box': wander.wander_box,  # Current wander box (can change dynamically)
                    'mode': behavior.state.mode.name if behavior else 'UNKNOWN',
                    'gesture': behavior.state.gesture.name if behavior and behavior.state.gesture else None,
                    'status': status_text,
                    'behavior_description': _build_behavior_description(behavior, active_count, passive_count) if behavior else '',
                    'dwell_phase': behavior_status.get('driving_factors', {}).get('dwell_time', 0) if behavior_status else 0,
                    'engaged_breathing': behavior_status.get('engaged_breathing', {}) if behavior_status else {},
                    'realtime_trends': realtime_trends,
                    'auto_tuning': {
                        'enabled': auto_tuner.enabled,
                        'revision': auto_tuner.revision,
                        'params': {name: round(float(getattr(meta_params, name)), 3) for name in auto_tuner.param_order},
                        # Compact summary instead of full last_adjustment blob
                        'activity': {
                            'short': round(auto_tuner.last_adjustment.get('short_activity', 0), 3) if auto_tuner.last_adjustment else 0,
                            'medium': round(auto_tuner.last_adjustment.get('medium_activity', 0), 3) if auto_tuner.last_adjustment else 0,
                            'long': round(auto_tuner.last_adjustment.get('long_activity', 0), 3) if auto_tuner.last_adjustment else 0,
                            'energy': round(auto_tuner.last_adjustment.get('energy_level', 0), 3) if auto_tuner.last_adjustment else 0,
                        } if auto_tuner.last_adjustment else None,
                        'budget': {
                            'current': round(auto_tuner.budget_current, 1),
                            'max': round(auto_tuner._budget_max(), 1),
                        },
                        'top_deltas': [
                            {'name': n, 'delta': round(d, 4)}
                            for n, d in sorted(
                                (auto_tuner.last_adjustment.get('applied_deltas', {}) if auto_tuner.last_adjustment else {}).items(),
                                key=lambda kv: abs(kv[1]), reverse=True
                            )[:4]
                        ] if auto_tuner.last_adjustment else [],
                    },
                    'population': {
                        'current': len(tracked_manager.get_all()),
                        'active': active_count,
                        'passive': passive_count,
                        'daily_total': tracked_manager.daily_count,
                        'daily_unique_db': db_stats.get('daily_unique_people', 0),
                    },
                    'daily_report_available': current_daily_report is not None,
                    'daily_report_date': current_daily_report.date if current_daily_report else None,
                    'report_version': report_version,
                    # Include cached report data (pre-serialized for efficiency)
                    'daily_report': cached_report_dict,
                }
                ws_broadcaster.update_state(state)
                ws_broadcaster.last_broadcast = time.time()
            except Exception as e:
                if not hasattr(ws_broadcaster, 'error_count'):
                    ws_broadcaster.error_count = 0
                ws_broadcaster.error_count += 1
                if ws_broadcaster.error_count <= 5 or ws_broadcaster.error_count % 100 == 0:
                    logger.warning(f"WebSocket broadcast error ({ws_broadcaster.error_count}x): {e}")
        
        # Send Art-Net with error handling and reconnection
        if artnet:
            try:
                artnet.set(panel_system.get_dmx_values())
                if hasattr(artnet, '_error_count') and artnet._error_count > 0:
                    logger.info("Art-Net connection restored")
                    artnet._error_count = 0
            except Exception as e:
                if not hasattr(artnet, '_error_count'):
                    artnet._error_count = 0
                artnet._error_count += 1
                if artnet._error_count == 1 or artnet._error_count % 100 == 0:
                    logger.warning(f"Art-Net send error ({artnet._error_count}x): {e}")
                # Attempt reconnection every 30 seconds after failures
                if artnet._error_count % (30 * FPS) == 0:
                    logger.info("Attempting Art-Net reconnection...")
                    try:
                        artnet.stop()
                        artnet = StupidArtnet(TARGET_IP, UNIVERSE, 12, FPS)
                        artnet.start()
                        logger.info("Art-Net reconnected successfully")
                    except Exception as re:
                        logger.warning(f"Art-Net reconnection failed: {re}")
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up perspective for 3D view
        glViewport(0, 0, view_width, display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, view_width/display[1], 10, 2000)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera position
        cam_x = cam_target[0] + cam_distance * math.cos(math.radians(cam_rot_x)) * math.sin(math.radians(cam_rot_y))
        cam_y = cam_target[1] + cam_distance * math.sin(math.radians(cam_rot_x))
        cam_z = cam_target[2] + cam_distance * math.cos(math.radians(cam_rot_x)) * math.cos(math.radians(cam_rot_y))
        
        gluLookAt(cam_x, cam_y, cam_z, *cam_target, 0, 1, 0)
        
        # Draw floor - only storefront level, stopping at near edge of active zone
        active_zone_near = TRACKZONE['offset_z']  # Near edge at Z=78
        draw_floor(0, (0.25, 0.25, 0.3, 0.5), z_max=active_zone_near)
        
        # =====================================================================
        # V2 ADDITIONS: Origin marker, camera markers, and labels
        # =====================================================================
        
        # Draw origin marker (0,0,0) with axis lines
        draw_origin_marker(font_label)
        if show_labels:
            draw_text_3d_billboard([0, 0, 0], "ORIGIN (0,0,0)", font_label, (255, 255, 0), offset_y=20)
        
        # Draw camera position markers
        draw_camera_markers(font_label, show_labels)
        
        # Draw unit labels for light panels
        draw_unit_labels(panel_system, font_label, show_labels)
        
        # Draw marker coordinate labels (only if markers are visible)
        if show_markers:
            draw_marker_labels(font_label, show_labels)
        
        # =====================================================================
        # END V2 ADDITIONS
        # =====================================================================
        
        # Draw trackzone (active - cyan)
        tz = TRACKZONE
        tz_bounds = (
            tz['center_x'] - tz['width']/2, tz['center_x'] + tz['width']/2,
            tz['offset_y'], tz['offset_y'] + tz['height'],
            tz['offset_z'], tz['offset_z'] + tz['depth']
        )
        draw_box_wireframe(tz_bounds, (0, 1, 1, 0.5))
        draw_zone_corner_labels(tz_bounds, "ACTIVE ZONE", font_label, (0, 255, 255), show_labels)
        
        # Draw passive trackzone (orange/yellow)
        ptz = PASSIVE_TRACKZONE
        ptz_bounds = (
            ptz['center_x'] - ptz['width']/2, ptz['center_x'] + ptz['width']/2,
            ptz['offset_y'], ptz['offset_y'] + ptz['height'],
            ptz['offset_z'], ptz['offset_z'] + ptz['depth']
        )
        draw_box_wireframe(ptz_bounds, (1, 0.6, 0, 0.4))
        draw_zone_corner_labels(ptz_bounds, "PASSIVE ZONE", font_label, (255, 150, 0), show_labels)
        
        # Draw wander box (from behavior system)
        wb = behavior.get_wander_box()
        wb_bounds = (wb['min_x'], wb['max_x'], wb['min_y'], wb['max_y'], wb['min_z'], wb['max_z'])
        draw_box_wireframe(wb_bounds, (0, 1, 0, 0.3))
        draw_zone_corner_labels(wb_bounds, "WANDER BOX", font_label, (0, 255, 0), show_labels)
        
        # Draw panels
        for (unit, panel_num), panel in panel_system.panels.items():
            draw_panel(panel['center'], panel['angle'], PANEL_SIZE, panel['brightness'])
            draw_text_3d_billboard(
                panel['center'],
                str(panel['dmx_value']),
                font_small,
                (255, 0, 255),
                offset_y=10
            )
        
        # Draw panel center indicators (wireframe spheres with labels)
        draw_panel_centers(panel_system, font_label, show_labels)
        
        # Draw calibration markers
        if show_markers:
            for marker_id, marker_data in MARKER_POSITIONS.items():
                pos = marker_data['pos']
                tex_id = marker_textures.get(marker_id)
                is_vertical = marker_data.get('vertical', False)
                draw_marker(marker_id, pos, MARKER_SIZE, tex_id, vertical=is_vertical)
        
        # Draw light
        brightness = light.get_brightness()
        radius = 8 + brightness * 7
        draw_sphere(light.position, radius, (1, 1, brightness, 1))
        draw_sphere_wireframe(light.position, light.falloff_radius, (1, 0.8, 0, 0.3), segments=24)
        
        # Draw tracked people
        for person in tracked_manager.get_all():
            draw_tracked_person(person, zone_checker=tracked_manager._get_zone)
        
        # Render camera views to framebuffers (if enabled)
        if show_camera_views:
            # Create FBOs on first use
            if not camera_fbos:
                for cam_name in CAMERA_POSITIONS.keys():
                    camera_fbos[cam_name] = create_camera_fbo(CAMERA_VIEW_SIZE[0], CAMERA_VIEW_SIZE[1])
            
            # Render each camera view
            for cam_name, cam_data in CAMERA_POSITIONS.items():
                if cam_name in camera_fbos:
                    render_camera_view(cam_data, camera_fbos[cam_name], panel_system, light, 
                                       tracked_manager, marker_textures, show_markers)
            
            # Restore main viewport
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, view_width, display[1])
        
        # Draw HUD
        glViewport(0, 0, display[0], display[1])
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], 0, display[1], -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Draw GUI panel background
        glColor4f(0.12, 0.12, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(view_width, 0)
        glVertex2f(display[0], 0)
        glVertex2f(display[0], display[1])
        glVertex2f(view_width, display[1])
        glEnd()
        
        # Draw separator line
        glColor4f(0.4, 0.4, 0.5, 1.0)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(view_width, 0)
        glVertex2f(view_width, display[1])
        glEnd()
        
        # Draw camera view overlays (if enabled)
        if show_camera_views and camera_fbos:
            cam_view_w, cam_view_h = CAMERA_VIEW_SIZE
            padding = 10
            y_pos = display[1] - cam_view_h - padding
            
            for i, (cam_name, cam_data) in enumerate(CAMERA_POSITIONS.items()):
                if cam_name in camera_fbos:
                    x_pos = padding + i * (cam_view_w + padding)
                    border_color = cam_data['color']
                    draw_camera_view_overlay(camera_fbos[cam_name], x_pos, y_pos, 
                                           cam_view_w, cam_view_h, cam_name, font_small, border_color)
        
        # GUI title (centered across full panel)
        title_center_x = view_width + gui_width // 2 - 100
        draw_text_2d(title_center_x, display[1] - 30, "LIGHT CONTROLLER V3", font)
        draw_text_2d(view_width + col_padding, display[1] - 50, "─" * 62, font)
        
        # Column headers
        # Left column: Manual controls
        draw_section_header(left_col_x, display[1] - 70, col_width, "MANUAL CONTROLS", font_small, (150, 180, 220), "🎛️")
        # Right column: Auto-tuned
        draw_section_header(right_col_x, display[1] - 70, col_width, "AUTO-TUNED", font_small, (120, 180, 170), "🤖")
        
        # Sub-section headers
        # Left: Calibration label
        draw_text_2d(left_col_x, display[1] - 100, "Calibration", font_small, (130, 140, 170))
        # Left: Budget label  
        draw_text_2d(left_col_x, display[1] - 350, "Tuning Budget", font_small, (130, 140, 170))
        # Right: Personality label
        draw_text_2d(right_col_x, display[1] - 100, "Personality", font_small, (100, 150, 140))
        # Right: Output multipliers label
        draw_text_2d(right_col_x, display[1] - 345, "Output Multipliers", font_small, (100, 150, 140))
        
        # Draw all sliders
        for slider in all_sliders.values():
            slider.draw(font_small)
        
        # Draw checkboxes
        for checkbox in checkboxes.values():
            checkbox.draw(font_small)
        
        # Update database stats periodically (every 2 seconds)
        if time.time() - last_stats_update > 2.0:
            db_stats = tracking_db.get_current_stats()
            last_stats_update = time.time()
        
        # Save slider settings periodically if changed
        if sliders_dirty and time.time() - last_slider_save > slider_save_interval:
            save_slider_settings(all_sliders, checkboxes)
            last_slider_save = time.time()
            sliders_dirty = False
        
        # ─── BOTTOM SECTION: Mode Decision (left) + Auto-Tune (right) ───
        bottom_section_top = display[1] - 640
        draw_text_2d(view_width + col_padding, bottom_section_top + 14, "─" * 62, font_small, (80, 80, 100))
        
        # Behavior status section (LEFT side of bottom)
        behavior_status = behavior.get_status()
        status_y = bottom_section_top - 10
        draw_text_2d(left_col_x, status_y + 14, "MODE DECISION", font_small, (255, 200, 100))
        
        # Mode and preset
        mode_colors = {
            'idle': (100, 100, 200),
            'engaged': (100, 200, 100),
            'crowd': (200, 200, 100),
            'flow': (200, 150, 100),
        }
        mode_color = mode_colors.get(behavior_status['mode'], (200, 200, 200))
        
        # Get driving factors
        factors = behavior_status.get('driving_factors', {})
        params = factors.get('current_params', {})
        
        # DECISION INPUTS - what's actually driving the mode
        active = factors.get('active_count', 0)
        passive = factors.get('passive_count', 0)
        passive_rate = factors.get('passive_rate', 0.0)
        flow_thresh = factors.get('flow_threshold', 3)
        flow_enabled = factors.get('flow_enabled', True)
        
        # Show inputs with threshold comparisons
        # Active count - drives ENGAGED (>=1) or CROWD (>=2)
        active_color = (100, 255, 100) if active >= 1 else (150, 150, 150)
        active_indicator = ""
        if active >= 2:
            active_indicator = " → CROWD"
        elif active >= 1:
            active_indicator = " → ENGAGED"
        draw_text_2d(left_col_x, status_y - 4, f"  Active: {active}{active_indicator}", font_small, active_color)
        
        # Passive count and rate - drives FLOW mode
        passive_color = (200, 200, 100) if passive_rate >= flow_thresh else (150, 150, 150)
        flow_indicator = " → FLOW" if (passive_rate >= flow_thresh and active == 0 and flow_enabled) else ""
        draw_text_2d(left_col_x, status_y - 20, f"  Passive: {passive} ({passive_rate:.1f}/min){flow_indicator}", font_small, passive_color)
        
        # Mode with duration and stability
        mode_duration = factors.get('mode_duration', 0)
        min_duration = factors.get('min_duration', 8.0)
        mode_stable = factors.get('mode_stable', False)
        stability_pct = min(100, int(mode_duration / min_duration * 100))
        
        # Current mode line
        stability_char = "●" if mode_stable else f"◐{stability_pct}%"
        mode_text = f"  {behavior_status['mode'].upper()} [{mode_duration:.1f}s] {stability_char}"
        draw_text_2d(left_col_x, status_y - 50, mode_text, font_small, mode_color)
        
        # Pending mode if any
        pending = behavior_status.get('pending_mode')
        if pending:
            pending_color = mode_colors.get(pending['mode'], (200, 200, 200))
            pending_pct = int(pending['progress'] * 100)
            pending_text = f"  → {pending['mode'].upper()} ({pending_pct}%)"
            draw_text_2d(left_col_x, status_y - 66, pending_text, font_small, pending_color)
            y_next = status_y - 82
        else:
            y_next = status_y - 66
        
        # Current output parameters (condensed)
        draw_text_2d(left_col_x, y_next, f"  B{params.get('brightness_min', 0):.0f}-{params.get('brightness_max', 0):.0f} P{params.get('pulse_speed', 0):.0f} R{params.get('falloff_radius', 0):.0f}", font_small, (150, 150, 150))
        y_next -= 14
        
        # Preset and status text
        draw_text_2d(left_col_x, y_next, f"  Preset: {current_preset}", font_small)
        y_next -= 14
        
        # Status text (for public display)
        if behavior_status['status_text']:
            draw_text_2d(left_col_x, y_next, f"  \"{behavior_status['status_text']}\"", font_small, (200, 200, 255))

        # Auto-tuning panel (RIGHT side of bottom section - expanded to fill space)
        auto_panel_width = col_width
        auto_panel_x = right_col_x
        auto_panel_top = bottom_section_top - 10
        auto_panel_bottom = 80  # Leave room for help text at bottom
        auto_panel_height = auto_panel_top - auto_panel_bottom
        auto_panel_y = auto_panel_bottom
        draw_auto_tuning_panel(
            auto_panel_x,
            auto_panel_y,
            auto_panel_width,
            auto_panel_height,
            font,
            font_small,
            auto_tuner.enabled,
            auto_tuner.last_adjustment,
            auto_tuner.history,
            auto_tuner.budget_current,
            auto_tuner._budget_max(),
        )

        # Controls help at bottom
        draw_text_2d(left_col_x, 50, "SPC=wander A=tune T=trends P=preset M=markers L=labels R=report D=db Q=quit", font_small, (120, 120, 120))
        # Current database indicator
        db_color = (100, 180, 255) if len(db_files) > 1 else (120, 120, 120)
        draw_text_2d(left_col_x, 20, f"DB: {current_db_file} ({current_db_index+1}/{len(db_files)})", font_small, db_color)
        
        # Legend in 3D view area (top left)
        if show_labels or show_markers or show_camera_views:
            draw_text_2d(10, display[1] - 20, "DEBUG:", font_small, (255, 200, 100))
            draw_text_2d(10, display[1] - 40, "  Origin = yellow sphere", font_small, (255, 255, 0))
            draw_text_2d(10, display[1] - 55, "  Axis: R=X, G=Y, B=Z", font_small, (200, 200, 200))
        
        # Marker legend
        if show_markers:
            draw_text_2d(10, 350, "AR MARKERS:", font_small, (255, 255, 0))
            y_offset = 330
            for marker_id, marker_data in MARKER_POSITIONS.items():
                desc = marker_data['desc']
                draw_text_2d(10, y_offset, f"  [{marker_id}] {desc}", font_small)
                y_offset -= 16
        
        # HUD text in 3D view (bottom left)
        dmx_vals = panel_system.get_dmx_values()
        
        # Build mode status with pending info
        mode_text = f"Mode: {behavior_status['mode'].upper()}"
        if factors.get('mode_stable'):
            mode_text += " ●"
        else:
            mode_text += f" ({stability_pct}%)"
        if pending:
            mode_text += f" → {pending['mode'].upper()}({int(pending['progress']*100)}%)"
        mode_text += f"  Active: {active_count}  Passive: {passive_count}"
        
        # Build proximity status
        prox_factor = behavior_status.get('proximity_factor', 0)
        prox_info = factors.get('proximity', {})
        nearest_z = prox_info.get('nearest_z', 500)
        prox_bar = "█" * int(prox_factor * 10) + "░" * (10 - int(prox_factor * 10))
        prox_text = f"Proximity: [{prox_bar}] {prox_factor:.0%} (Z={nearest_z:.0f})"
        
        # Entry pulse indicator
        pulse_text = ""
        if behavior_status.get('entry_pulse_active'):
            pulse_text = " ⚡PULSE"
        
        info_lines = [
            f"Light: ({light.position[0]:.0f}, {light.position[1]:.0f}, {light.position[2]:.0f}) cm",
            mode_text + pulse_text,
            prox_text,
        ]
        
        for i, line in enumerate(info_lines):
            # Highlight proximity line when someone is close
            if i == 2 and prox_factor > 0.5:
                draw_text_2d(10, 100 + i * 18, line, font_small, (100, 255, 100))
            elif "⚡PULSE" in line:
                draw_text_2d(10, 100 + i * 18, line, font_small, (255, 255, 100))
            else:
                draw_text_2d(10, 100 + i * 18, line, font_small)
        
        # Realtime trends panel (left side, always visible when trends enabled)
        if show_trends:
            idle_trends = behavior_status.get('idle_trends')
            aggression = behavior_status.get('aggression')
            flow = behavior_status.get('flow')
            almost_engaged = behavior_status.get('almost_engaged')
            feedback_learning = behavior_status.get('feedback_learning')
            draw_realtime_trends(idle_trends, 10, display[1] - 100, font, font_small, aggression, flow, almost_engaged, feedback_learning)
        
        # Daily trends visualization (below realtime trends panel on left)
        if show_trends and current_daily_report:
            trends_width = 260
            trends_height = 200
            trends_x = 10  # Same x as realtime trends
            trends_y = display[1] - 100 - 520 - 10  # Below realtime trends panel (height=520)
            draw_trends_visualization(current_daily_report, trends_x, trends_y, 
                                     trends_width, trends_height, font, font_small)
        elif show_trends and not current_daily_report:
            # No report yet - show message below realtime trends
            msg_x = 10
            msg_y = display[1] - 100 - 640 - 30  # Below realtime trends panel (height=640)
            glColor4f(0.1, 0.1, 0.15, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(msg_x, msg_y - 50)
            glVertex2f(msg_x + 260, msg_y - 50)
            glVertex2f(msg_x + 260, msg_y + 20)
            glVertex2f(msg_x, msg_y + 20)
            glEnd()
            draw_text_2d(msg_x + 10, msg_y, "No daily report", font_small, (255, 255, 200))
            draw_text_2d(msg_x + 10, msg_y - 18, "Press R to generate", font_small, (150, 150, 150))
            draw_text_2d(msg_x + 10, msg_y - 36, "(Auto at 12:01 AM)", font_small, (120, 120, 120))
        
        # Status text overlay (bottom center of 3D view)
        if behavior_status['status_text'] and meta_params.status_text_enabled and not show_trends:
            status = behavior_status['status_text']
            # Draw with a background
            glColor4f(0.0, 0.0, 0.0, 0.6)
            status_x = view_width // 2 - 100
            status_y = 30
            glBegin(GL_QUADS)
            glVertex2f(status_x - 10, status_y - 5)
            glVertex2f(status_x + 220, status_y - 5)
            glVertex2f(status_x + 220, status_y + 25)
            glVertex2f(status_x - 10, status_y + 25)
            glEnd()
            draw_text_2d(status_x, status_y, f'"{status}"', font, (255, 255, 200))
        
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1
        
        # Periodic health logging
        current_time = time.time()
        if current_time - last_health_log >= HEALTH_LOG_INTERVAL:
            elapsed_total = current_time - start_time
            uptime = timedelta(seconds=int(elapsed_total))
            avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
            
            # Get current state
            behavior_status = behavior.get_status()
            
            logger.info(
                f"HEALTH: uptime={uptime}, frames={frame_count}, avg_fps={avg_fps:.1f}, "
                f"mode={behavior_status['mode']}, active={active_count}, passive={passive_count}, "
                f"ws_clients={len(ws_broadcaster.clients) if ws_broadcaster else 0}"
            )
            
            last_health_log = current_time
        
        # Periodic database pruning (aggregate then prune - keeps hourly stats forever)
        if current_time - last_db_prune >= DB_PRUNE_INTERVAL:
            try:
                # Smart prune: aggregate old hours, then delete raw events
                results = tracking_db.prune_with_aggregation(
                    raw_retention_hours=DB_RAW_RETENTION_HOURS
                )
                if results['events_pruned'] > 0 or results['hours_aggregated'] > 0:
                    logger.info(
                        f"📊 DB maintenance: aggregated {results['hours_aggregated']} hours, "
                        f"pruned {results['events_pruned']} events, "
                        f"{results['behavior_pruned']} behavior records"
                    )
            except Exception as e:
                logger.warning(f"Database maintenance failed: {e}")
            
            last_db_prune = current_time
        
        # Hourly aggregation trigger (aggregate completed hour for fresh stats)
        current_hour = datetime.now().hour
        if current_hour != last_hour_aggregated:
            try:
                # Aggregate the hour that just ended
                prev_hour = (current_hour - 1) % 24
                if prev_hour > current_hour:  # Crossed midnight
                    date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                stats = tracking_db.aggregate_hour(date_str, prev_hour)
                if stats['total_events'] > 0:
                    logger.info(
                        f"📊 Hourly aggregate [{date_str} {prev_hour}:00]: "
                        f"{stats['unique_people']} people, "
                        f"{stats['active_count']} active, {stats['passive_count']} passive"
                    )
            except Exception as e:
                logger.warning(f"Hourly aggregation failed: {e}")
            
            last_hour_aggregated = current_hour
    
    # Cleanup
    logger.info("Shutting down...")
    
    # Save slider settings before exit
    save_slider_settings(all_sliders, checkboxes)
    
    # Stop background threads first
    daily_report_scheduler.stop()
    
    # Close OSC server socket (BlockingOSCUDPServer doesn't use shutdown())
    try:
        osc_server_instance.server_close()
    except:
        pass
    
    if artnet:
        artnet.stop()
    if ws_broadcaster:
        ws_broadcaster.stop()
        logger.info("WebSocket server stopped")
    tracking_db.close()
    logger.info("Tracking database closed")
    pygame.quit()
    
    # Final stats
    elapsed = time.time() - start_time
    uptime = timedelta(seconds=int(elapsed))
    logger.info(f"Shutdown complete - uptime: {uptime}, frames: {frame_count}")
    print(f"\n🛑 Stopped after {uptime}")


if __name__ == "__main__":
    main()
