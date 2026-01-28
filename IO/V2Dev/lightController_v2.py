#!/usr/bin/env python3
"""
3D Light Controller V2 - Development Version

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
- Scroll: Zoom
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
from light_behavior_v2 import (
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
DB_RETENTION_DAYS = 7  # Keep 7 days of tracking history

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

LOCK_FILE = "/tmp/lightController_v2.lock"
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
            print(f"❌ Another lightController_v2 is already running (PID: {existing_pid})")
        except:
            print("❌ Another lightController_v2 is already running")
        return False

def release_single_instance_lock():
    """Release the single instance lock"""
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            os.remove(LOCK_FILE)
        except:
            pass

# =============================================================================
# SLIDER SETTINGS PERSISTENCE
# =============================================================================

SLIDER_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slider_settings_v2.json')

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
            }
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
DMX_MAX = 50

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
    'width': 260,           # Narrowed from 475 to 260 for better coverage
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
CAMERA_LEDGE_Y = -16  # Cameras are 50cm above street (16cm below floor)

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

MARKER_SIZE = 15  # cm - ArUco marker size

# Marker positions: (X, Y, Z) in centimeters
# Coordinate system: X=0 at back right corner of Unit 0 panel, negative X goes left
# Marker 0 is on the RIGHT, Marker 2 is on the LEFT
# Front row (0,1,2): 90cm from front edge of tracking zone (Z=78), so Z=168
# Back row (3,6,4): 51cm behind front row, so Z=219
# Marker 5: ~500cm from cameras (Z=78+500=578) on subway wall
MARKER_POSITIONS = {
    0: {'pos': (-30, STREET_LEVEL_Y, 168), 'desc': 'Right front', 'camera': 'Cam 1', 'vertical': False},
    1: {'pos': (-150, STREET_LEVEL_Y, 168), 'desc': 'Center front (SHARED)', 'camera': 'Both', 'vertical': False},
    2: {'pos': (-270, STREET_LEVEL_Y, 168), 'desc': 'Left front', 'camera': 'Cam 2', 'vertical': False},
    3: {'pos': (-30, STREET_LEVEL_Y, 219), 'desc': 'Right back', 'camera': 'Cam 1', 'vertical': False},
    4: {'pos': (-270, STREET_LEVEL_Y, 219), 'desc': 'Left back', 'camera': 'Cam 2', 'vertical': False},
    5: {'pos': (-150, CAMERA_Y, 578), 'desc': 'Subway wall (VERTICAL, ~5m from cams)', 'camera': 'Both', 'vertical': True},
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
    x: float  # World X position (cm)
    z: float  # World Z position (cm)
    y: float = STREET_LEVEL_Y  # Fixed at street level
    last_update: float = 0.0
    first_seen: float = 0.0  # When first tracked
    zone: str = "unknown"  # "active", "passive", or "unknown"
    
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
        
        # Use zone from OSC if provided, otherwise compute locally
        if zone is None:
            zone = self._get_zone(x, z)
        
        now = time.time()
        
        with self.lock:
            is_new = track_id not in self.people
            
            if is_new:
                self.people[track_id] = TrackedPerson(
                    track_id=track_id,
                    x=x, z=z, y=y,
                    last_update=now,
                    first_seen=now,
                    zone=zone
                )
                # Notify behavior system
                if self.on_person_entered:
                    pos = np.array([x, y, z])
                    is_active = zone == "active"
                    self.on_person_entered(track_id, pos, is_active)
            else:
                self.people[track_id].x = x
                self.people[track_id].z = z
                self.people[track_id].y = y
                self.people[track_id].zone = zone
                self.people[track_id].last_update = now
                
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
    
    def get_all(self) -> List[TrackedPerson]:
        """Get list of all tracked people"""
        with self.lock:
            return list(self.people.values())
    
    def count(self) -> int:
        """Get count of tracked people"""
        with self.lock:
            return len(self.people)
    
    def count_active(self) -> int:
        """Count people in active zone"""
        with self.lock:
            return sum(1 for p in self.people.values() if p.is_in_active_zone())
    
    def count_passive(self) -> int:
        """Count people in passive zone"""
        with self.lock:
            return sum(1 for p in self.people.values() if p.is_in_passive_zone())
    
    def get_active_positions(self) -> List[np.ndarray]:
        """Get positions of people in active zone"""
        with self.lock:
            return [p.get_position() for p in self.people.values() if p.is_in_active_zone()]


# =============================================================================
# WEBSOCKET BROADCASTER (for public viewer)
# =============================================================================

class WebSocketBroadcaster:
    """Broadcasts installation state to web clients"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients = set()
        self.loop = None
        self.server = None
        self.thread = None
        self.current_state = {}
        self.running = False
    
    async def handler(self, websocket):
        """Handle a WebSocket connection"""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else 'unknown'
        print(f"🌐 WebSocket client connected: {client_ip}")
        
        try:
            # Send current state immediately
            if self.current_state:
                await websocket.send(json.dumps(self.current_state))
            
            # Keep connection alive
            async for message in websocket:
                pass  # We don't expect messages from clients
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"🌐 WebSocket client disconnected: {client_ip}")
    
    async def broadcast(self, state: dict):
        """Broadcast state to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(state)
        # Send to all clients, removing dead connections
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except:
                dead_clients.add(client)
        
        self.clients -= dead_clients
    
    def update_state(self, state: dict):
        """Update the current state (called from main thread)"""
        self.current_state = state
        
        if self.loop and self.running:
            # Schedule broadcast on the event loop
            asyncio.run_coroutine_threadsafe(
                self.broadcast(state),
                self.loop
            )
    
    async def _run_server(self):
        """Run the WebSocket server"""
        self.server = await websockets.serve(
            self.handler,
            "0.0.0.0",
            self.port
        )
        print(f"🌐 WebSocket server started on port {self.port}")
        
        # Get local IP for display
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"   Public viewer URL: http://{local_ip}:8080")
        except:
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
            except Exception as e:
                restart_count += 1
                logger.error(f"WebSocket server error ({restart_count}/{max_restarts}): {e}")
                if restart_count < max_restarts:
                    logger.info(f"WebSocket server restarting in {restart_delay}s...")
                    time.sleep(restart_delay)
                    restart_delay = min(restart_delay * 2, 60)  # Exponential backoff, max 60s
            finally:
                try:
                    self.loop.close()
                except:
                    pass
        
        if restart_count >= max_restarts:
            logger.error("WebSocket server exceeded max restart attempts, giving up")
        self.running = False
    
    def start(self):
        """Start the WebSocket server in a background thread"""
        self.running = True  # Set BEFORE starting thread
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()


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
                
                # Record to database
                if self.database:
                    self.database.record_position(track_id, x, z)
                
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
        
        Args format: zone (str)
        """
        try:
            # Extract track_id from address: /tracker/zone/123
            parts = address.split('/')
            track_id = int(parts[-1])
            
            if args:
                zone = str(args[0]).lower()  # 'active' or 'passive'
                # Update the zone for this person
                with self.manager.lock:
                    if track_id in self.manager.people:
                        self.manager.people[track_id].zone = zone
        except (ValueError, IndexError) as e:
            pass  # Silently ignore parse errors for zone updates


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
            
            if light.falloff_radius > 0:
                falloff = max(0, 1.0 - distance / light.falloff_radius)
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
    """Draw a panel as a quad"""
    half = size / 2
    
    glPushMatrix()
    glTranslatef(*center)
    glRotatef(-angle, 1, 0, 0)
    
    gray = 0.2 + brightness * 0.8
    glColor4f(gray, gray, gray, 1.0)
    
    glBegin(GL_QUADS)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    glColor4f(0.3, 0.3, 0.3, 1.0)
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


def draw_tracked_person(person: TrackedPerson):
    """Draw a tracked person as a cylinder/capsule"""
    pos = person.get_position()
    
    # Draw as a colored cylinder (person height ~170cm)
    height = 170
    radius = 20
    
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    
    # Body cylinder
    glColor4f(0.2, 0.8, 0.2, 0.8)  # Green for tracked people
    
    quadric = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)  # Rotate to stand upright
    gluCylinder(quadric, radius, radius, height, 16, 1)
    
    # Top cap (head)
    glTranslatef(0, 0, height)
    gluSphere(quadric, radius, 12, 12)
    
    gluDeleteQuadric(quadric)
    glPopMatrix()


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
    
    # Find max value for scaling
    max_people = max(h.total_people for h in report.hourly_trends) if report.hourly_trends else 1
    max_people = max(max_people, 1)  # Avoid division by zero
    
    # Draw hour bars
    bar_width = chart_width / 24
    bar_gap = 2
    
    for trend in report.hourly_trends:
        hour = trend.hour
        bx = chart_x + hour * bar_width
        
        # Active zone bar (green)
        active_height = (trend.active_count / max(max_people, 1)) * chart_height * 0.8
        glColor4f(0.2, 0.7, 0.3, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bx + bar_gap, chart_y)
        glVertex2f(bx + bar_width - bar_gap, chart_y)
        glVertex2f(bx + bar_width - bar_gap, chart_y + active_height)
        glVertex2f(bx + bar_gap, chart_y + active_height)
        glEnd()
        
        # Passive zone bar (stacked, blue)
        passive_height = (trend.passive_count / max(max_people * 3, 1)) * chart_height * 0.8
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
    panel_height = 640  # Increased height for feedback learning display
    
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
    
    curr_y = y - 40
    line_height = 16
    
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
    curr_y -= line_height + 5
    
    # Database error if any
    db_error = idle_trends.get('database_error', '')
    if db_error:
        draw_text_2d(x + 10, curr_y, f"⚠ {db_error[:25]}", font_small, (255, 100, 100))
        curr_y -= line_height
    
    # Section: REALTIME (1 min)
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
    curr_y -= line_height + 10
    
    # Divider line
    glColor4f(0.3, 0.4, 0.6, 0.5)
    glBegin(GL_LINES)
    glVertex2f(x + 10, curr_y + 5)
    glVertex2f(x + panel_width - 10, curr_y + 5)
    glEnd()
    curr_y -= 5
    
    # COMPUTED VALUES section
    draw_text_2d(x + 10, curr_y, "COMPUTED VALUES", font_small, (180, 180, 200))
    curr_y -= line_height + 2
    
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
    curr_y -= line_height + 10
    
    # ======================
    # AGGRESSION SECTION
    # ======================
    if aggression:
        # Divider line
        glColor4f(0.3, 0.4, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 5)
        glVertex2f(x + panel_width - 10, curr_y + 5)
        glEnd()
        curr_y -= 5
        
        draw_text_2d(x + 10, curr_y, "AGGRESSION", font_small, (255, 150, 100))
        curr_y -= line_height + 2
        
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
        curr_y -= line_height + 10
    
    # ======================
    # FLOW POSITIONING SECTION (Phase 2B)
    # ======================
    if flow:
        # Divider line
        glColor4f(0.3, 0.4, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 5)
        glVertex2f(x + panel_width - 10, curr_y + 5)
        glEnd()
        curr_y -= 5
        
        draw_text_2d(x + 10, curr_y, "FLOW POSITIONING", font_small, (100, 200, 255))
        curr_y -= line_height + 2
        
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
        curr_y -= line_height + 10
    
    # ======================
    # ALMOST-ENGAGED SECTION (Phase 2C)
    # ======================
    if almost_engaged:
        # Divider line
        glColor4f(0.3, 0.4, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 5)
        glVertex2f(x + panel_width - 10, curr_y + 5)
        glEnd()
        curr_y -= 5
        
        draw_text_2d(x + 10, curr_y, "ALMOST-ENGAGED", font_small, (255, 200, 100))
        curr_y -= line_height + 2
        
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

    # ======================
    # FEEDBACK LEARNING SECTION (Phase 3)
    # ======================
    if feedback_learning:
        # Divider line
        glColor4f(0.4, 0.3, 0.6, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 10, curr_y + 5)
        glVertex2f(x + panel_width - 10, curr_y + 5)
        glEnd()
        curr_y -= 5
        
        draw_text_2d(x + 10, curr_y, "FEEDBACK LEARNING", font_small, (200, 150, 255))
        curr_y -= line_height + 2
        
        # Total engagements
        total_eng = feedback_learning.get('total_engagements', 0)
        session_eng = feedback_learning.get('session_engagements', 0)
        lr = feedback_learning.get('learning_rate', 0.02)
        
        draw_text_2d(x + 10, curr_y, f"Engagements: {total_eng}", font_small, (180, 180, 180))
        draw_text_2d(x + 130, curr_y, f"(session: {session_eng})", font_small, (150, 150, 150))
        curr_y -= line_height
        
        # Top weighted behaviors
        top_weights = feedback_learning.get('top_weights', {})
        if top_weights:
            draw_text_2d(x + 10, curr_y, "Top weights:", font_small, (150, 200, 150))
            curr_y -= line_height
            for name, weight in list(top_weights.items())[:3]:
                # Color based on weight (> 1.0 = good, green tint)
                if weight > 1.1:
                    w_color = (100, 255, 150)
                elif weight > 1.0:
                    w_color = (180, 255, 180)
                else:
                    w_color = (180, 180, 180)
                draw_text_2d(x + 20, curr_y, f"{name}: {weight:.2f}", font_small, w_color)
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
        draw_tracked_person(person)
    
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
    def __init__(self, x, y, width, height, min_val, max_val, value, label, format_str="{:.1f}"):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.format_str = format_str
        self.dragging = False
    
    def handle_event(self, event, screen_height):
        """Handle mouse events. Returns True if value changed."""
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse_y = screen_height - event.pos[1]
            if self.rect.collidepoint(event.pos[0], mouse_y):
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
        
        # Background
        glColor4f(0.2, 0.2, 0.25, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Fill based on value
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w = w * ratio
        glColor4f(0.3, 0.6, 0.8, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + fill_w, y)
        glVertex2f(x + fill_w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Border
        glColor4f(0.5, 0.5, 0.5, 1.0)
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Label and value
        val_str = self.format_str.format(self.value)
        draw_text_2d(x, y + h + 5, f"{self.label}: {val_str}", font)


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
    
    display = (1920, 1080)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Light Controller V2 - Production")
    
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
    
    # Camera
    cam_rot_x = 20
    cam_rot_y = -30
    cam_distance = 500
    cam_target = np.array([-160.0, 50.0, 50.0])  # Center on panels, look at tracking area
    
    # GUI panel width
    gui_width = 280
    view_width = display[0] - gui_width
    
    # Create systems
    panel_system = PanelSystem()
    light = PointLight()
    wander_box = dict(WANDER_BOX)
    wander = WanderBehavior(light, wander_box)
    
    # Tracked person manager
    tracked_manager = TrackedPersonManager()
    
    # Find available database files
    import glob
    db_files = sorted(glob.glob("*.db") + glob.glob("tracking_*.db"))
    if "tracking_history.db" not in db_files:
        db_files.insert(0, "tracking_history.db")
    else:
        # Move tracking_history.db to front
        db_files.remove("tracking_history.db")
        db_files.insert(0, "tracking_history.db")
    current_db_file = "tracking_history.db"
    current_db_index = 0
    
    # Tracking database
    tracking_db = TrackingDatabase(current_db_file)
    print(f"💾 Tracking database: {current_db_file}")
    
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
    show_trends = True  # Toggle with 'T' key - ON by default
    
    def on_report_ready(report: DailyReport):
        nonlocal current_daily_report
        current_daily_report = report
    
    daily_report_scheduler.on_report_ready = on_report_ready
    
    # Create sliders
    slider_x = view_width + 20
    slider_w = gui_width - 40
    slider_h = 12
    
    # Calibration sliders (top section - below title)
    sliders = {
        # Offset sliders
        'offset_x': Slider(slider_x, display[1] - 100, slider_w, slider_h, -200, 200, 0, "Offset X"),
        'offset_z': Slider(slider_x, display[1] - 140, slider_w, slider_h, 0, 500, 250, "Offset Z"),
        # Scale sliders
        'scale_x': Slider(slider_x, display[1] - 190, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale X", "{:.2f}"),
        'scale_z': Slider(slider_x, display[1] - 230, slider_w, slider_h, 0.5, 2.0, 1.0, "Scale Z", "{:.2f}"),
    }
    
    # Calibration checkboxes
    checkboxes = {
        'invert_x': Checkbox(slider_x, display[1] - 265, 14, "Invert X Direction", checked=False),
    }
    
    # Personality sliders (middle section - starts after checkbox)
    personality_sliders = {
        'responsiveness': Slider(slider_x, display[1] - 330, slider_w, slider_h, 0, 1, 0.5, "Responsiveness", "{:.2f}"),
        'energy': Slider(slider_x, display[1] - 370, slider_w, slider_h, 0, 1, 0.5, "Energy", "{:.2f}"),
        'attention_span': Slider(slider_x, display[1] - 410, slider_w, slider_h, 0, 1, 0.5, "Attention", "{:.2f}"),
        'sociability': Slider(slider_x, display[1] - 450, slider_w, slider_h, 0, 1, 0.5, "Sociability", "{:.2f}"),
        'exploration': Slider(slider_x, display[1] - 490, slider_w, slider_h, 0, 1, 0.5, "Exploration", "{:.2f}"),
        'memory': Slider(slider_x, display[1] - 530, slider_w, slider_h, 0, 1, 0.5, "Memory", "{:.2f}"),
    }
    
    # Global multiplier sliders (lower section)
    global_sliders = {
        'brightness_global': Slider(slider_x, display[1] - 600, slider_w, slider_h, 0.2, 2.0, 1.0, "Brightness ×", "{:.2f}"),
        'speed_global': Slider(slider_x, display[1] - 640, slider_w, slider_h, 0.2, 2.0, 1.0, "Speed ×", "{:.2f}"),
        'pulse_global': Slider(slider_x, display[1] - 680, slider_w, slider_h, 0.3, 3.0, 1.0, "Pulse ×", "{:.2f}"),
        'follow_speed_global': Slider(slider_x, display[1] - 720, slider_w, slider_h, 0.5, 3.0, 1.0, "Follow Speed ×", "{:.2f}"),
        'dwell_influence': Slider(slider_x, display[1] - 760, slider_w, slider_h, 0.0, 2.0, 1.0, "Dwell Influence", "{:.2f}"),
        'idle_trend_weight': Slider(slider_x, display[1] - 800, slider_w, slider_h, 0.0, 2.0, 1.0, "Idle Trend ×", "{:.2f}"),
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
            setattr(meta_params, name, slider.value)
        print(f"📁 Restored {len(saved_settings)} slider settings")
    
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
    frame_count = 0
    total_osc_messages = 0
    
    logger.info("Light controller V2 started - entering main loop")
    print("\n" + "="*60)
    print("V2 DEVELOPMENT VERSION - Visual Debugging Enabled")
    print("="*60)
    print("Controls:")
    print("  L = Toggle coordinate labels")
    print("  M = Toggle AR markers")
    print("  SPACE = Toggle wandering")
    print("  P = Cycle presets")
    print("  T = Toggle trends visualization")
    print("  R = Generate daily report (manual)")
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
                        setattr(meta_params, name, slider.value)
            
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
                        new_db_file = db_files[current_db_index]
                        # Close old database and open new one
                        tracking_db.close()
                        tracking_db = TrackingDatabase(new_db_file)
                        behavior.database = tracking_db
                        # Update OSC handler and report generator
                        osc_handler.database = tracking_db
                        report_generator.database = tracking_db
                        current_db_file = new_db_file
                        # Clear cached report
                        current_daily_report = None
                        print(f"💾 Switched to database: {new_db_file}")
                    else:
                        print(f"💾 Only one database available: {current_db_file}")
            
            # Camera rotation (only in 3D view area)
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < view_width and not slider_active:
                    mouse_down = True
                    last_mouse = event.pos
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
            elif event.type == MOUSEMOTION and mouse_down:
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
        
        # Process OSC messages (non-blocking, handles multiple per frame)
        # This replaces the threaded approach to avoid thread exhaustion
        osc_messages_this_frame = 0
        max_osc_per_frame = 50  # Limit to prevent blocking too long
        while osc_messages_this_frame < max_osc_per_frame:
            try:
                osc_server_instance.handle_request()
                osc_messages_this_frame += 1
            except:
                break  # No more messages or timeout
        
        # Update
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now
        
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
                behavior_status = behavior.get_status()
                status_text = behavior_status.get('status_text', '')
                
                state = {
                    'type': 'state_update',
                    'light': {
                        'x': float(light.position[0]),
                        'y': float(light.position[1]),
                        'z': float(light.position[2]),
                        'brightness': float(light.get_brightness()),
                        'falloff_radius': float(light.falloff_radius)
                    },
                    'panels': panel_system.get_dmx_values()[:12],
                    'people': [
                        {'id': p.track_id, 'x': p.x, 'y': p.y, 'z': p.z, 'zone': p.zone}
                        for p in tracked_manager.get_all()
                    ],
                    'counts': {
                        'active': active_count,
                        'passive': passive_count,
                        'total': len(tracked_manager.get_all())
                    },
                    'mode': behavior.state.mode.name if behavior else 'UNKNOWN',
                    'gesture': behavior.state.gesture.name if behavior and behavior.state.gesture else None,
                    'status': status_text,
                    'daily_report_available': current_daily_report is not None,
                    'daily_report_date': current_daily_report.date if current_daily_report else None,
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
            draw_tracked_person(person)
        
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
        
        # GUI title
        draw_text_2d(view_width + 20, display[1] - 30, "LIGHT CONTROLLER V2 - DEV", font)
        draw_text_2d(view_width + 20, display[1] - 50, "─" * 24, font)
        
        # Section labels (adjusted for checkbox)
        draw_text_2d(view_width + 20, display[1] - 70, "Calibration:", font_small, (150, 150, 200))
        draw_text_2d(view_width + 20, display[1] - 300, "Personality:", font_small, (150, 200, 150))
        draw_text_2d(view_width + 20, display[1] - 570, "Global Multipliers:", font_small, (200, 150, 150))
        
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
        
        # Behavior status section (at bottom of GUI)
        behavior_status = behavior.get_status()
        status_y = 200  # Start position from bottom
        draw_text_2d(view_width + 20, status_y + 50, "─" * 20, font_small)
        draw_text_2d(view_width + 20, status_y + 35, "MODE DECISION:", font_small, (255, 200, 100))
        
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
        thresholds = factors.get('thresholds', {})
        
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
        draw_text_2d(view_width + 20, status_y + 17, f"  Active: {active}{active_indicator}", font_small, active_color)
        
        # Passive count and rate - drives FLOW mode
        passive_color = (200, 200, 100) if passive_rate >= flow_thresh else (150, 150, 150)
        flow_indicator = " → FLOW" if (passive_rate >= flow_thresh and active == 0 and flow_enabled) else ""
        draw_text_2d(view_width + 20, status_y + 1, f"  Passive: {passive} ({passive_rate:.1f}/min){flow_indicator}", font_small, passive_color)
        
        # Thresholds reference
        draw_text_2d(view_width + 20, status_y - 15, f"  Thresholds: CROWD≥2, ENG≥1, FLOW≥{flow_thresh}/m", font_small, (120, 120, 120))
        
        # Mode with duration and stability
        mode_duration = factors.get('mode_duration', 0)
        min_duration = factors.get('min_duration', 8.0)
        mode_stable = factors.get('mode_stable', False)
        stability_pct = min(100, int(mode_duration / min_duration * 100))
        
        # Current mode line
        stability_char = "●" if mode_stable else f"◐{stability_pct}%"
        mode_text = f"  Current: {behavior_status['mode'].upper()} [{mode_duration:.1f}s] {stability_char}"
        draw_text_2d(view_width + 20, status_y - 33, mode_text, font_small, mode_color)
        
        # Pending mode if any
        pending = behavior_status.get('pending_mode')
        if pending:
            pending_color = mode_colors.get(pending['mode'], (200, 200, 200))
            pending_pct = int(pending['progress'] * 100)
            pending_text = f"  Pending: {pending['mode'].upper()} ({pending_pct}% of {pending['time_required']:.0f}s)"
            draw_text_2d(view_width + 20, status_y - 49, pending_text, font_small, pending_color)
            y_next = status_y - 65
        else:
            y_next = status_y - 49
        
        # Current output parameters (condensed)
        draw_text_2d(view_width + 20, y_next, f"  Output: B{params.get('brightness_min', 0):.0f}-{params.get('brightness_max', 0):.0f} P{params.get('pulse_speed', 0):.0f} R{params.get('falloff_radius', 0):.0f}", font_small, (150, 150, 150))
        y_next -= 14
        
        # Time of day influence
        time_mood = factors.get('time_mood', 'active')
        time_bright = factors.get('time_brightness', 1.0)
        draw_text_2d(view_width + 20, y_next, f"  Time: {time_mood} (×{time_bright:.1f})", font_small, (150, 150, 180))
        y_next -= 14
        
        # Dwell bonus if active
        dwell_bonus = factors.get('dwell_bonus', 0)
        if dwell_bonus > 0:
            dwell_time = factors.get('dwell_time', 0)
            draw_text_2d(view_width + 20, y_next, f"  Dwell: +{dwell_bonus:.0f} ({dwell_time:.0f}s)", font_small, (100, 255, 100))
            y_next -= 14
        
        # Bloom if active
        bloom_progress = factors.get('bloom_progress', 0)
        if bloom_progress > 0:
            draw_text_2d(view_width + 20, y_next, f"  BLOOM: {bloom_progress*100:.0f}%", font_small, (255, 200, 100))
            y_next -= 14
        
        # Idle trends (when in IDLE mode)
        idle_trends = behavior_status.get('idle_trends')
        if idle_trends and behavior_status['mode'] == 'idle':
            anticipation = idle_trends.get('activity_anticipation', 0.5)
            momentum = idle_trends.get('flow_momentum', 0)
            energy = idle_trends.get('energy_level', 0.5)
            period = idle_trends.get('period', 'unknown')
            
            # Anticipation bar (how ready for action)
            ant_bar = "█" * int(anticipation * 10) + "░" * (10 - int(anticipation * 10))
            ant_color = (100, 255, 100) if anticipation > 0.6 else (150, 150, 150)
            draw_text_2d(view_width + 20, y_next, f"  Anticipation: [{ant_bar}]", font_small, ant_color)
            y_next -= 14
            
            # Flow momentum indicator (-1 to +1)
            if abs(momentum) > 0.1:
                flow_dir = "→" if momentum > 0 else "←"
                flow_str = f"{flow_dir * int(abs(momentum) * 5)}"
                flow_color = (100, 200, 255) if momentum > 0 else (255, 200, 100)
                draw_text_2d(view_width + 20, y_next, f"  Flow: {flow_str} ({momentum:+.2f})", font_small, flow_color)
                y_next -= 14
            
            # Energy level
            energy_bar = "█" * int(energy * 10) + "░" * (10 - int(energy * 10))
            draw_text_2d(view_width + 20, y_next, f"  Energy: [{energy_bar}] {period}", font_small, (180, 180, 255))
            y_next -= 14
        
        # Preset and status text
        draw_text_2d(view_width + 20, y_next, f"  Preset: {current_preset}", font_small)
        y_next -= 14
        
        # Status text (for public display)
        if behavior_status['status_text']:
            draw_text_2d(view_width + 20, y_next, f"  \"{behavior_status['status_text']}\"", font_small, (200, 200, 255))

        # Controls help at bottom (three lines now with database info)
        draw_text_2d(view_width + 20, 50, "SPC=wander M=markers L=labels P=preset Q=quit", font_small, (120, 120, 120))
        draw_text_2d(view_width + 20, 35, "T=trends R=report D=database", font_small, (120, 120, 120))
        # Current database indicator
        db_color = (100, 180, 255) if len(db_files) > 1 else (120, 120, 120)
        draw_text_2d(view_width + 20, 20, f"DB: {current_db_file} ({current_db_index+1}/{len(db_files)})", font_small, db_color)
        
        # Legend in 3D view area (top left)
        draw_text_2d(10, display[1] - 20, "V2 VISUAL DEBUG:", font_small, (255, 200, 100))
        draw_text_2d(10, display[1] - 40, "  Yellow sphere = ORIGIN (0,0,0)", font_small, (255, 255, 0))
        draw_text_2d(10, display[1] - 55, "  Red sphere = Camera 1", font_small, (255, 100, 100))
        draw_text_2d(10, display[1] - 70, "  Blue sphere = Camera 2", font_small, (100, 100, 255))
        draw_text_2d(10, display[1] - 85, "  Axis: R=X, G=Y, B=Z", font_small, (200, 200, 200))
        
        # Marker legend
        if show_markers:
            draw_text_2d(10, 350, "AR MARKERS:", font_small, (255, 255, 0))
            y_offset = 330
            for marker_id, marker_data in MARKER_POSITIONS.items():
                pos = marker_data['pos']
                desc = marker_data['desc']
                draw_text_2d(10, y_offset, f"  [{marker_id}] ({pos[0]}, {pos[1]}, {pos[2]}) - {desc}", font_small)
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
            f"DMX: {dmx_vals}",
            mode_text + pulse_text,
            prox_text,
            f"Labels: {'ON' if show_labels else 'OFF'}  Markers: {'ON' if show_markers else 'OFF'}  Trends: {'ON' if show_trends else 'OFF'}",
        ]
        
        for i, line in enumerate(info_lines):
            # Highlight proximity line when someone is close
            if i == 3 and prox_factor > 0.5:
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
            trends_y = display[1] - 100 - 640 - 10  # Below realtime trends panel (height=640)
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
        if behavior_status['status_text'] and meta_params.status_text_enabled:
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
        
        # Periodic database pruning (keep DB from growing forever)
        if current_time - last_db_prune >= DB_PRUNE_INTERVAL:
            try:
                # Prune records older than retention period
                cutoff = current_time - (DB_RETENTION_DAYS * 86400)
                pruned = tracking_db.prune_old_records(cutoff)
                if pruned > 0:
                    logger.info(f"Pruned {pruned} old tracking records from database")
            except Exception as e:
                logger.warning(f"Database prune failed: {e}")
            
            last_db_prune = current_time
    
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
