#!/usr/bin/env python3
"""
Tracking Database for Light Installation

Stores tracking data with velocity vectors for trend analysis.
Uses SQLite for cross-platform compatibility (macOS/Linux).

Data stored:
- Raw tracking events with position and velocity
- Session summaries (per hour)
- Daily patterns
- Flow direction analysis for passive zone

All units in centimeters, velocity in cm/s.
"""

import sqlite3
import time
import math
import threading
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from enum import Enum


class Zone(Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    UNKNOWN = "unknown"


class FlowDirection(Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    STATIONARY = "stationary"


@dataclass
class TrackingEvent:
    """A single tracking observation"""
    timestamp: float
    person_id: int
    x: float
    z: float
    vx: float = 0.0  # Velocity X (cm/s)
    vz: float = 0.0  # Velocity Z (cm/s)
    zone: Zone = Zone.UNKNOWN
    
    @property
    def speed(self) -> float:
        """Speed in cm/s"""
        return math.sqrt(self.vx**2 + self.vz**2)
    
    @property
    def flow_direction(self) -> FlowDirection:
        """Primary flow direction based on X velocity"""
        if abs(self.vx) < 10:  # Less than 10 cm/s = stationary
            return FlowDirection.STATIONARY
        return FlowDirection.LEFT_TO_RIGHT if self.vx > 0 else FlowDirection.RIGHT_TO_LEFT


@dataclass
class TrendSummary:
    """Summary of tracking trends over a time period"""
    period_start: datetime
    period_end: datetime
    total_people: int
    unique_people: int
    active_zone_visits: int
    passive_zone_count: int
    avg_speed: float
    left_to_right_count: int
    right_to_left_count: int
    avg_dwell_time_active: float  # seconds
    peak_count: int
    peak_time: Optional[datetime] = None


class TrackingDatabase:
    """
    SQLite-based tracking database with velocity tracking.
    
    Automatically calculates velocity from consecutive position updates.
    Provides trend analysis over different time scales.
    Uses batched commits for better performance (commits every 1 second or 50 writes).
    """
    
    def __init__(self, db_path: str = "tracking_history.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        
        # Check for corruption and recover if needed
        self.conn = self._safe_connect()
        self.conn.row_factory = sqlite3.Row
        
        # Enable WAL mode for better crash resilience
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        
        # Track previous positions for velocity calculation
        self.prev_positions: Dict[int, Tuple[float, float, float, float]] = {}  # id -> (x, z, timestamp)
        
        # Batched commit settings (for performance)
        self._pending_writes = 0
        self._last_commit_time = time.time()
        self._commit_interval = 1.0  # Commit at least every 1 second
        self._commit_batch_size = 50  # Or after 50 writes
        
        # Zone boundaries (MUST match lightController_osc.py zones)
        # Active zone: close to installation (engaging with it)
        # center_x=-150, width=260, offset_z=78, depth=205
        self.active_zone = {
            'x_min': -150 - 260/2,  # -280
            'x_max': -150 + 260/2,  # -20
            'z_min': 78,
            'z_max': 78 + 205,      # 283
        }
        # Passive zone: further out on sidewalk (passing by)
        # center_x=-150, width=400, offset_z=283, depth=270
        self.passive_zone = {
            'x_min': -150 - 400/2,  # -350
            'x_max': -150 + 400/2,  # 50
            'z_min': 283,
            'z_max': 283 + 270,     # 553
        }
        
        self._create_tables()
    
    def _safe_connect(self) -> sqlite3.Connection:
        """Connect to database with integrity check. Auto-recovers if corrupted."""
        if not self.db_path.exists():
            print(f"\U0001f4be Creating new tracking database: {self.db_path}")
            return sqlite3.connect(str(self.db_path), check_same_thread=False)
        
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if result == "ok":
                return conn
            else:
                print(f"\u26a0\ufe0f Database integrity check failed: {result}")
                conn.close()
                return self._recover_database()
        except sqlite3.DatabaseError as e:
            print(f"\u26a0\ufe0f Database error on open: {e}")
            return self._recover_database()
    
    def _recover_database(self) -> sqlite3.Connection:
        """Attempt to recover a corrupted database. Falls back to fresh DB."""
        import subprocess
        backup_path = self.db_path.with_suffix(
            f".db.bak.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"\U0001f4e6 Backing up corrupted DB to: {backup_path}")
        
        try:
            import shutil
            shutil.copy2(str(self.db_path), str(backup_path))
        except Exception as e:
            print(f"  Backup failed: {e}")
        
        # Try to recover using sqlite3 .recover command
        recovered_path = self.db_path.with_suffix(".db.recovered")
        try:
            result = subprocess.run(
                ['sqlite3', str(self.db_path), '.recover'],
                capture_output=True, text=True, timeout=60
            )
            if result.stdout.strip():
                subprocess.run(
                    ['sqlite3', str(recovered_path)],
                    input=result.stdout, capture_output=True, text=True, timeout=60
                )
                # Check if recovery has any data
                test_conn = sqlite3.connect(str(recovered_path))
                tables = test_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                test_conn.close()
                
                if tables:
                    print(f"\u2705 Recovered database with {len(tables)} tables")
                    self.db_path.unlink()
                    recovered_path.rename(self.db_path)
                    # Remove leftover journal
                    journal = self.db_path.with_suffix(".db-journal")
                    if journal.exists():
                        journal.unlink()
                    return sqlite3.connect(str(self.db_path), check_same_thread=False)
                else:
                    print("\u26a0\ufe0f Recovery produced empty database")
                    recovered_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"\u26a0\ufe0f Recovery failed: {e}")
            recovered_path.unlink(missing_ok=True)
        
        # Last resort: start fresh
        print("\U0001f195 Creating fresh database (old data preserved in backup)")
        self.db_path.unlink(missing_ok=True)
        # Remove leftover journal
        journal = self.db_path.with_suffix(".db-journal")
        if journal.exists():
            journal.unlink()
        return sqlite3.connect(str(self.db_path), check_same_thread=False)
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Raw tracking events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracking_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    person_id INTEGER NOT NULL,
                    x REAL NOT NULL,
                    z REAL NOT NULL,
                    vx REAL DEFAULT 0,
                    vz REAL DEFAULT 0,
                    speed REAL DEFAULT 0,
                    zone TEXT NOT NULL,
                    flow_direction TEXT
                )
            ''')
            
            # Index for time-based queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON tracking_events(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_datetime 
                ON tracking_events(datetime)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_zone 
                ON tracking_events(zone)
            ''')
            
            # Hourly summaries (aggregated periodically)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hourly_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour_start TEXT NOT NULL UNIQUE,
                    total_events INTEGER DEFAULT 0,
                    unique_people INTEGER DEFAULT 0,
                    active_zone_events INTEGER DEFAULT 0,
                    passive_zone_events INTEGER DEFAULT 0,
                    avg_speed REAL DEFAULT 0,
                    left_to_right_count INTEGER DEFAULT 0,
                    right_to_left_count INTEGER DEFAULT 0,
                    peak_count INTEGER DEFAULT 0,
                    peak_minute TEXT
                )
            ''')
            
            # Daily summaries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_people INTEGER DEFAULT 0,
                    active_zone_visits INTEGER DEFAULT 0,
                    passive_zone_count INTEGER DEFAULT 0,
                    avg_speed REAL DEFAULT 0,
                    dominant_flow TEXT,
                    busiest_hour INTEGER,
                    peak_count INTEGER DEFAULT 0
                )
            ''')
            
            # Light behavior history for self-analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS light_behavior (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    position_x REAL NOT NULL,
                    position_y REAL NOT NULL,
                    position_z REAL NOT NULL,
                    target_x REAL,
                    target_y REAL,
                    target_z REAL,
                    brightness REAL NOT NULL,
                    pulse_speed REAL NOT NULL,
                    move_speed REAL NOT NULL,
                    people_count INTEGER DEFAULT 0,
                    active_count INTEGER DEFAULT 0,
                    passive_count INTEGER DEFAULT 0,
                    gesture_type TEXT,
                    status_text TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_light_timestamp 
                ON light_behavior(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_light_mode 
                ON light_behavior(mode)
            ''')

            # Auto-adjustment history (behavior tuning events)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    reason TEXT,
                    short_activity REAL,
                    medium_activity REAL,
                    long_activity REAL,
                    energy_level REAL,
                    aggression_level REAL,
                    adjustments_json TEXT
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_adjustments_timestamp
                ON behavior_adjustments(timestamp)
            ''')
            
            # Person sessions (tracks individual visit durations)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration REAL,
                    entered_active_zone INTEGER DEFAULT 0,
                    primary_flow_direction TEXT,
                    avg_speed REAL
                )
            ''')
            
            # Hourly statistics (aggregated from raw events - kept forever)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hourly_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    total_events INTEGER DEFAULT 0,
                    unique_people INTEGER DEFAULT 0,
                    active_count INTEGER DEFAULT 0,
                    passive_count INTEGER DEFAULT 0,
                    avg_speed REAL DEFAULT 0,
                    left_to_right INTEGER DEFAULT 0,
                    right_to_left INTEGER DEFAULT 0,
                    bloom_count INTEGER DEFAULT 0,
                    dominant_mode TEXT,
                    avg_brightness REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, hour)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hourly_date ON hourly_stats(date)')
            
            # Daily statistics (aggregated from hourly - kept forever)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_events INTEGER DEFAULT 0,
                    unique_people INTEGER DEFAULT 0,
                    total_active INTEGER DEFAULT 0,
                    total_passive INTEGER DEFAULT 0,
                    total_blooms INTEGER DEFAULT 0,
                    peak_hour INTEGER,
                    peak_count INTEGER,
                    quietest_hour INTEGER,
                    dominant_flow TEXT,
                    flow_balance REAL DEFAULT 0,
                    avg_speed REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Auto-tune daily learnings (what the system learned each day)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS autotune_daily_learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_adjustments INTEGER DEFAULT 0,
                    total_unique_people INTEGER DEFAULT 0,
                    peak_hour INTEGER,
                    optimal_values_json TEXT,
                    param_journeys_json TEXT,
                    hourly_activity_json TEXT,
                    strategy_summary TEXT,
                    learned_caps_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_learnings_date ON autotune_daily_learnings(date)')
            
            # Meta-tuning reviews (periodic automated analysis of auto-tuner performance)
            # Written by the meta-tuner program (autotune_meta_review.py) 3x/day
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS meta_tuning_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    review_window_hours REAL NOT NULL,
                    
                    -- Analysis results
                    total_adjustments INTEGER DEFAULT 0,
                    total_tracking_events INTEGER DEFAULT 0,
                    unique_people INTEGER DEFAULT 0,
                    
                    -- Activity level stats
                    avg_short_activity REAL,
                    median_short_activity REAL,
                    avg_medium_activity REAL,
                    avg_energy_level REAL,
                    
                    -- Floor/ceiling clamping analysis
                    pct_at_floor_json TEXT,
                    pct_at_ceiling_json TEXT,
                    
                    -- Light mode distribution
                    mode_distribution_json TEXT,
                    
                    -- Parameter stats (avg/min/max for each param)
                    param_stats_json TEXT,
                    
                    -- Changes applied
                    old_config_json TEXT,
                    new_config_json TEXT,
                    changes_summary TEXT,
                    
                    -- Diagnostics
                    diagnosis TEXT,
                    recommendations_json TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_meta_timestamp ON meta_tuning_reviews(timestamp)')
            
            self.conn.commit()
    
    def _get_zone(self, x: float, z: float) -> Zone:
        """Determine which zone a position is in"""
        az = self.active_zone
        pz = self.passive_zone
        
        if (az['x_min'] <= x <= az['x_max'] and 
            az['z_min'] <= z <= az['z_max']):
            return Zone.ACTIVE
        elif (pz['x_min'] <= x <= pz['x_max'] and 
              pz['z_min'] <= z <= pz['z_max']):
            return Zone.PASSIVE
        return Zone.UNKNOWN
    
    def record_position(self, person_id: int, x: float, z: float, 
                        timestamp: Optional[float] = None):
        """
        Record a position update with automatic velocity calculation.
        Uses batched commits for better performance.
        
        Args:
            person_id: Unique ID of tracked person
            x, z: Position in cm
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate velocity from previous position
        vx, vz = 0.0, 0.0
        if person_id in self.prev_positions:
            prev_x, prev_z, prev_time = self.prev_positions[person_id]
            dt = timestamp - prev_time
            if dt > 0 and dt < 1.0:  # Only if reasonable time gap
                vx = (x - prev_x) / dt
                vz = (z - prev_z) / dt
        
        # Update previous position
        self.prev_positions[person_id] = (x, z, timestamp)
        
        # Determine zone and flow direction
        zone = self._get_zone(x, z)
        speed = math.sqrt(vx**2 + vz**2)
        
        flow_dir = None
        if zone == Zone.PASSIVE:
            if abs(vx) >= 10:
                flow_dir = FlowDirection.LEFT_TO_RIGHT.value if vx > 0 else FlowDirection.RIGHT_TO_LEFT.value
            else:
                flow_dir = FlowDirection.STATIONARY.value
        
        # Store in database
        dt_str = datetime.fromtimestamp(timestamp).isoformat()
        
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO tracking_events 
                    (timestamp, datetime, person_id, x, z, vx, vz, speed, zone, flow_direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, dt_str, person_id, x, z, vx, vz, speed, zone.value, flow_dir))
                
                # Batched commit: only commit when batch is full or interval elapsed
                self._pending_writes += 1
                now = time.time()
                if self._pending_writes >= self._commit_batch_size or \
                   now - self._last_commit_time >= self._commit_interval:
                    self.conn.commit()
                    self._pending_writes = 0
                    self._last_commit_time = now
        except sqlite3.DatabaseError as e:
            print(f"\u26a0\ufe0f DB write error in record_position: {e}")
    
    def remove_person(self, person_id: int):
        """Called when a person leaves tracking - cleans up velocity state"""
        if person_id in self.prev_positions:
            del self.prev_positions[person_id]
    
    # =========================================================================
    # LIGHT BEHAVIOR RECORDING (Self-Analysis)
    # =========================================================================
    
    def record_light_state(self, mode: str, position: Tuple[float, float, float],
                           target: Tuple[float, float, float], brightness: float,
                           pulse_speed: float, move_speed: float,
                           people_count: int = 0, active_count: int = 0,
                           passive_count: int = 0, gesture_type: str = None,
                           status_text: str = None, timestamp: float = None):
        """
        Record the light's current state for self-analysis.
        Call this periodically (0.5s when active, 2s when idle).
        Uses batched commits for better performance.
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt_str = datetime.fromtimestamp(timestamp).isoformat()
        
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO light_behavior 
                    (timestamp, datetime, mode, position_x, position_y, position_z,
                     target_x, target_y, target_z, brightness, pulse_speed, move_speed,
                     people_count, active_count, passive_count, gesture_type, status_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, dt_str, mode, 
                      position[0], position[1], position[2],
                      target[0], target[1], target[2],
                      brightness, pulse_speed, move_speed,
                      people_count, active_count, passive_count,
                      gesture_type, status_text))
                
                # Batched commit: only commit when batch is full or interval elapsed
                self._pending_writes += 1
                now = time.time()
                if self._pending_writes >= self._commit_batch_size or \
                   now - self._last_commit_time >= self._commit_interval:
                    self.conn.commit()
                    self._pending_writes = 0
                    self._last_commit_time = now
        except sqlite3.DatabaseError as e:
            print(f"\u26a0\ufe0f DB write error in record_light_state: {e}")

    def record_behavior_adjustment(self, enabled: bool, reason: str,
                                   short_activity: float, medium_activity: float,
                                   long_activity: float, energy_level: float,
                                   aggression_level: float, adjustments: Dict,
                                   timestamp: float = None):
        """
        Record an auto-tuning adjustment event.

        adjustments is a dict that should include old/new values and deltas.
        """
        if timestamp is None:
            timestamp = time.time()

        dt_str = datetime.fromtimestamp(timestamp).isoformat()
        adjustments_json = json.dumps(adjustments, separators=(',', ':'))

        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO behavior_adjustments
                    (timestamp, datetime, enabled, reason, short_activity, medium_activity,
                     long_activity, energy_level, aggression_level, adjustments_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, dt_str, int(enabled), reason, short_activity,
                      medium_activity, long_activity, energy_level, aggression_level,
                      adjustments_json))

                # Batched commit: only commit when batch is full or interval elapsed
                self._pending_writes += 1
                now = time.time()
                if self._pending_writes >= self._commit_batch_size or \
                   now - self._last_commit_time >= self._commit_interval:
                    self.conn.commit()
                    self._pending_writes = 0
                    self._last_commit_time = now
        except sqlite3.DatabaseError as e:
            print(f"\u26a0\ufe0f DB write error in record_behavior_adjustment: {e}")

    def get_daily_adjustments(self, date_str: str) -> Dict:
        """
        Get auto-tuning adjustment analysis for a specific day.
        Returns summary of parameter changes, strategies used, and trends.
        
        Args:
            date_str: Date in YYYY-MM-DD format
        """
        start_dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=1)
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT timestamp, short_activity, medium_activity, long_activity,
                       energy_level, aggression_level, adjustments_json
                FROM behavior_adjustments
                WHERE timestamp >= ? AND timestamp < ? AND enabled = 1
                ORDER BY timestamp
            ''', (start_ts, end_ts))
            
            rows = cursor.fetchall()
            
        if not rows:
            return {
                'total_adjustments': 0,
                'param_journeys': {},
                'hourly_activity': {},
                'strategy_summary': 'No auto-tuning adjustments recorded.',
            }
        
        # Track parameter evolution over the day
        param_start = {}  # First values seen
        param_end = {}    # Last values seen
        param_min = {}    # Minimum values
        param_max = {}    # Maximum values
        param_total_delta = {}  # Sum of absolute deltas
        hourly_adjustment_counts = {}  # Adjustments per hour
        hourly_activity_levels = {}    # Activity levels per hour
        
        for row in rows:
            ts = row['timestamp']
            hour = datetime.fromtimestamp(ts).hour
            hourly_adjustment_counts[hour] = hourly_adjustment_counts.get(hour, 0) + 1
            
            # Track activity levels by hour
            if hour not in hourly_activity_levels:
                hourly_activity_levels[hour] = []
            hourly_activity_levels[hour].append({
                'short': row['short_activity'],
                'medium': row['medium_activity'],
                'long': row['long_activity'],
                'energy': row['energy_level'],
                'aggression': row['aggression_level'],
            })
            
            try:
                adj = json.loads(row['adjustments_json'])
            except (json.JSONDecodeError, TypeError):
                continue
            
            new_values = adj.get('new_values', {})
            applied_deltas = adj.get('applied_deltas', {})
            
            for name, val in new_values.items():
                if name not in param_start:
                    # First occurrence - set from old_values
                    old_vals = adj.get('old_values', {})
                    param_start[name] = old_vals.get(name, val)
                    param_min[name] = param_start[name]
                    param_max[name] = param_start[name]
                    param_total_delta[name] = 0.0
                
                param_end[name] = val
                param_min[name] = min(param_min[name], val)
                param_max[name] = max(param_max[name], val)
            
            for name, delta in applied_deltas.items():
                param_total_delta[name] = param_total_delta.get(name, 0.0) + abs(delta)
        
        # Build parameter journeys
        param_journeys = {}
        for name in param_start:
            net_change = param_end.get(name, 0) - param_start.get(name, 0)
            param_journeys[name] = {
                'start': round(param_start[name], 3),
                'end': round(param_end[name], 3),
                'min': round(param_min[name], 3),
                'max': round(param_max[name], 3),
                'net_change': round(net_change, 3),
                'total_movement': round(param_total_delta.get(name, 0), 3),
                'direction': 'up' if net_change > 0.01 else ('down' if net_change < -0.01 else 'stable'),
            }
        
        # Identify strategies: which params moved the most and in which direction
        sorted_by_movement = sorted(param_journeys.items(), 
                                     key=lambda kv: kv[1]['total_movement'], reverse=True)
        most_active_params = sorted_by_movement[:5]
        
        # Build strategy narrative
        strategies = []
        for name, journey in most_active_params:
            if journey['total_movement'] < 0.01:
                continue
            direction = journey['direction']
            label = name.replace('_', ' ').title()
            if direction == 'up':
                strategies.append(f"{label} increased {journey['start']:.2f} → {journey['end']:.2f} (range {journey['min']:.2f}–{journey['max']:.2f})")
            elif direction == 'down':
                strategies.append(f"{label} decreased {journey['start']:.2f} → {journey['end']:.2f} (range {journey['min']:.2f}–{journey['max']:.2f})")
            else:
                strategies.append(f"{label} explored around {journey['end']:.2f} (range {journey['min']:.2f}–{journey['max']:.2f})")
        
        # Compute avg activity per hour
        hourly_activity = {}
        for hour, levels in hourly_activity_levels.items():
            hourly_activity[hour] = {
                'adjustments': hourly_adjustment_counts[hour],
                'avg_short_activity': round(sum(l['short'] for l in levels) / len(levels), 3),
                'avg_energy': round(sum(l['energy'] for l in levels) / len(levels), 3),
                'avg_aggression': round(sum(l['aggression'] for l in levels) / len(levels), 3),
            }
        
        return {
            'total_adjustments': len(rows),
            'param_journeys': param_journeys,
            'hourly_activity': hourly_activity,
            'most_tuned_params': [name for name, _ in most_active_params if _['total_movement'] >= 0.01],
            'strategy_summary': '; '.join(strategies) if strategies else 'Minimal tuning activity.',
        }
    
    def save_autotune_learnings(self, date_str: str, report_data: Dict, 
                                 tuning_analysis: Dict, optimal_values: Dict = None,
                                 learned_caps: Dict = None):
        """
        Save auto-tune daily learnings derived from the daily report analysis.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            report_data: Dict with total_unique_people, peak_hour etc.
            tuning_analysis: The auto_tuning_analysis from the daily report
            optimal_values: Computed optimal starting values for next day
            learned_caps: Any adjusted caps based on the day's experience
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO autotune_daily_learnings 
                (date, total_adjustments, total_unique_people, peak_hour,
                 optimal_values_json, param_journeys_json, hourly_activity_json,
                 strategy_summary, learned_caps_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                tuning_analysis.get('total_adjustments', 0),
                report_data.get('total_unique_people', 0),
                report_data.get('peak_hour', 0),
                json.dumps(optimal_values or {}),
                json.dumps(tuning_analysis.get('param_journeys', {})),
                json.dumps(tuning_analysis.get('hourly_activity', {})),
                tuning_analysis.get('strategy_summary', ''),
                json.dumps(learned_caps or {}),
            ))
            self.conn.commit()
    
    def get_recent_autotune_learnings(self, days: int = 7) -> List[Dict]:
        """
        Get the most recent auto-tune learnings from the database.
        Returns a list of daily learning records, newest first.
        
        Args:
            days: Number of days of history to retrieve
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT date, total_adjustments, total_unique_people, peak_hour,
                       optimal_values_json, param_journeys_json, hourly_activity_json,
                       strategy_summary, learned_caps_json
                FROM autotune_daily_learnings
                ORDER BY date DESC
                LIMIT ?
            ''', (days,))
            
            rows = cursor.fetchall()
            
        results = []
        for row in rows:
            try:
                optimal = json.loads(row['optimal_values_json']) if row['optimal_values_json'] else {}
                journeys = json.loads(row['param_journeys_json']) if row['param_journeys_json'] else {}
                hourly = json.loads(row['hourly_activity_json']) if row['hourly_activity_json'] else {}
                caps = json.loads(row['learned_caps_json']) if row['learned_caps_json'] else {}
            except (json.JSONDecodeError, TypeError):
                optimal, journeys, hourly, caps = {}, {}, {}, {}
            
            results.append({
                'date': row['date'],
                'total_adjustments': row['total_adjustments'],
                'total_unique_people': row['total_unique_people'],
                'peak_hour': row['peak_hour'],
                'optimal_values': optimal,
                'param_journeys': journeys,
                'hourly_activity': hourly,
                'strategy_summary': row['strategy_summary'],
                'learned_caps': caps,
            })
        
        return results
    
    def get_light_position_history(self, minutes: int = 5) -> List[Tuple[float, float, float]]:
        """Get recent light positions for pattern analysis"""
        cutoff = time.time() - (minutes * 60)
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT position_x, position_y, position_z
                FROM light_behavior
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff,))
            
            return [(row['position_x'], row['position_y'], row['position_z']) 
                    for row in cursor.fetchall()]
    
    def get_position_entropy(self, minutes: int = 60) -> float:
        """
        Calculate position entropy - measures how well the light uses the space.
        Higher = better coverage, lower = stuck in one area.
        Returns 0.0 to 1.0
        """
        positions = self.get_light_position_history(minutes)
        if len(positions) < 10:
            return 0.5  # Not enough data
        
        # Divide space into grid cells and count occupancy
        grid_size = 30  # cm per cell
        x_min, x_max = -50, 290
        y_min, y_max = 0, 150
        z_min, z_max = -32, 28
        
        # Create occupancy grid
        x_cells = int((x_max - x_min) / grid_size) + 1
        y_cells = int((y_max - y_min) / grid_size) + 1
        z_cells = int((z_max - z_min) / grid_size) + 1
        
        occupied = set()
        for x, y, z in positions:
            cx = int((x - x_min) / grid_size)
            cy = int((y - y_min) / grid_size)
            cz = int((z - z_min) / grid_size)
            cx = max(0, min(x_cells - 1, cx))
            cy = max(0, min(y_cells - 1, cy))
            cz = max(0, min(z_cells - 1, cz))
            occupied.add((cx, cy, cz))
        
        total_cells = x_cells * y_cells * z_cells
        return len(occupied) / total_cells
    
    def get_gesture_counts(self, minutes: int = 5) -> Dict[str, int]:
        """Count gestures by type in recent history"""
        cutoff = time.time() - (minutes * 60)
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT gesture_type, COUNT(*) as count
                FROM light_behavior
                WHERE timestamp > ? AND gesture_type IS NOT NULL
                GROUP BY gesture_type
            ''', (cutoff,))
            
            return {row['gesture_type']: row['count'] for row in cursor.fetchall()}
    
    def get_mode_distribution(self, hours: int = 24) -> Dict[str, float]:
        """Get percentage of time spent in each mode"""
        cutoff = time.time() - (hours * 3600)
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT mode, COUNT(*) as count
                FROM light_behavior
                WHERE timestamp > ?
                GROUP BY mode
            ''', (cutoff,))
            
            counts = {row['mode']: row['count'] for row in cursor.fetchall()}
            total = sum(counts.values())
            if total == 0:
                return {}
            return {mode: count / total for mode, count in counts.items()}
    
    def is_position_recently_visited(self, x: float, y: float, z: float, 
                                     seconds: int = 30, radius: float = 40) -> bool:
        """Check if position was recently visited (for cooldown)"""
        cutoff = time.time() - seconds
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT position_x, position_y, position_z
                FROM light_behavior
                WHERE timestamp > ?
            ''', (cutoff,))
            
            for row in cursor.fetchall():
                dx = row['position_x'] - x
                dy = row['position_y'] - y
                dz = row['position_z'] - z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < radius:
                    return True
            return False
    
    def get_response_similarity(self, current_people: int, hours: int = 24) -> float:
        """
        Check how similar recent responses are to past responses for similar inputs.
        Returns 0.0 (very different) to 1.0 (very similar).
        High similarity suggests we need more variety.
        """
        cutoff = time.time() - (hours * 3600)
        
        with self.lock:
            cursor = self.conn.cursor()
            # Get recent state entries with similar people count (±1)
            cursor.execute('''
                SELECT position_x, position_y, position_z, brightness, pulse_speed
                FROM light_behavior
                WHERE timestamp > ? AND people_count BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', (cutoff, max(0, current_people - 1), current_people + 1))
            
            rows = cursor.fetchall()
            if len(rows) < 10:
                return 0.0  # Not enough data to determine similarity
            
            # Calculate variance in responses
            positions = [(r['position_x'], r['position_y'], r['position_z']) for r in rows]
            brightnesses = [r['brightness'] for r in rows]
            
            # Position variance
            mean_x = sum(p[0] for p in positions) / len(positions)
            mean_y = sum(p[1] for p in positions) / len(positions)
            mean_z = sum(p[2] for p in positions) / len(positions)
            pos_var = sum((p[0]-mean_x)**2 + (p[1]-mean_y)**2 + (p[2]-mean_z)**2 
                         for p in positions) / len(positions)
            
            # Brightness variance
            mean_b = sum(brightnesses) / len(brightnesses)
            bright_var = sum((b - mean_b)**2 for b in brightnesses) / len(brightnesses)
            
            # Low variance = high similarity
            # Normalize: assume variance > 1000 is "good" variety for position
            # and variance > 100 is good for brightness
            pos_similarity = max(0, 1 - (pos_var / 2000))
            bright_similarity = max(0, 1 - (bright_var / 200))
            
            return (pos_similarity + bright_similarity) / 2
    
    def get_behavior_analysis(self) -> Dict:
        """Get comprehensive self-analysis of light behavior"""
        return {
            'position_entropy_1h': self.get_position_entropy(60),
            'mode_distribution_24h': self.get_mode_distribution(24),
            'recent_gestures_5m': self.get_gesture_counts(5),
            'response_similarity': self.get_response_similarity(0),  # idle baseline
        }
    
    def get_current_stats(self) -> Dict:
        """Get real-time statistics"""
        now = time.time()
        one_minute_ago = now - 60
        
        with self.lock:
            cursor = self.conn.cursor()
            
            # Count in last minute
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) as people,
                       AVG(speed) as avg_speed,
                       SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active_events,
                       SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive_events,
                       SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                       SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                FROM tracking_events
                WHERE timestamp > ?
            ''', (one_minute_ago,))
            
            row = cursor.fetchone()
            
            # Also get today's unique people count
            today_str = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) as daily_unique
                FROM tracking_events
                WHERE datetime >= ?
            ''', (today_str,))
            daily_row = cursor.fetchone()
            
            return {
                'people_last_minute': row['people'] or 0,
                'avg_speed': row['avg_speed'] or 0,
                'active_events': row['active_events'] or 0,
                'passive_events': row['passive_events'] or 0,
                'flow_left_to_right': row['ltr'] or 0,
                'flow_right_to_left': row['rtl'] or 0,
                'daily_unique_people': daily_row['daily_unique'] or 0,
            }
    
    def get_trends(self, minutes: int = 60) -> TrendSummary:
        """Get trend summary for the last N minutes"""
        now = time.time()
        period_start = now - (minutes * 60)
        
        with self.lock:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT person_id) as unique_people,
                    AVG(speed) as avg_speed,
                    SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active_events,
                    SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive_events,
                    SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                    SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl,
                    MAX(person_id) as peak_id
                FROM tracking_events
                WHERE timestamp > ?
            ''', (period_start,))
            
            row = cursor.fetchone()
            
            return TrendSummary(
                period_start=datetime.fromtimestamp(period_start),
                period_end=datetime.now(),
                total_people=row['total_events'] or 0,
                unique_people=row['unique_people'] or 0,
                active_zone_visits=row['active_events'] or 0,
                passive_zone_count=row['passive_events'] or 0,
                avg_speed=row['avg_speed'] or 0,
                left_to_right_count=row['ltr'] or 0,
                right_to_left_count=row['rtl'] or 0,
                avg_dwell_time_active=0,  # TODO: Calculate from sessions
                peak_count=0,  # TODO: Calculate peak
            )
    
    def get_flow_balance(self, minutes: int = 60) -> float:
        """
        Get flow balance for passive zone.
        Returns: -1.0 (all right-to-left) to +1.0 (all left-to-right)
        """
        trends = self.get_trends(minutes)
        total = trends.left_to_right_count + trends.right_to_left_count
        if total == 0:
            return 0.0
        return (trends.left_to_right_count - trends.right_to_left_count) / total
    
    def get_hourly_pattern(self, days: int = 7) -> Dict[int, Dict]:
        """
        Get average activity by hour of day over the last N days.
        Returns dict: hour (0-23) -> {avg_people, avg_speed, flow_balance}
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    CAST(strftime('%H', datetime) AS INTEGER) as hour,
                    COUNT(DISTINCT person_id) as people,
                    AVG(speed) as avg_speed,
                    SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                    SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                FROM tracking_events
                WHERE timestamp > ?
                GROUP BY hour
                ORDER BY hour
            ''', (cutoff_ts,))
            
            result = {}
            for row in cursor.fetchall():
                hour = row['hour']
                total_flow = (row['ltr'] or 0) + (row['rtl'] or 0)
                flow_balance = 0.0
                if total_flow > 0:
                    flow_balance = ((row['ltr'] or 0) - (row['rtl'] or 0)) / total_flow
                
                result[hour] = {
                    'avg_people': row['people'] or 0,
                    'avg_speed': row['avg_speed'] or 0,
                    'flow_balance': flow_balance,
                }
            
            return result
    
    def aggregate_hourly(self):
        """
        Aggregate recent data into hourly summaries.
        Call this periodically (e.g., every hour or on startup).
        """
        with self.lock:
            cursor = self.conn.cursor()
            
            # Find hours that need aggregation
            cursor.execute('''
                SELECT DISTINCT strftime('%Y-%m-%dT%H:00:00', datetime) as hour_start
                FROM tracking_events
                WHERE hour_start NOT IN (SELECT hour_start FROM hourly_summary)
                ORDER BY hour_start
            ''')
            
            hours = [row['hour_start'] for row in cursor.fetchall()]
            
            for hour_start in hours:
                hour_end = (datetime.fromisoformat(hour_start) + timedelta(hours=1)).isoformat()
                
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(DISTINCT person_id) as unique_people,
                        AVG(speed) as avg_speed,
                        SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active_events,
                        SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive_events,
                        SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                        SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                    FROM tracking_events
                    WHERE datetime >= ? AND datetime < ?
                ''', (hour_start, hour_end))
                
                row = cursor.fetchone()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO hourly_summary 
                    (hour_start, total_events, unique_people, active_zone_events, 
                     passive_zone_events, avg_speed, left_to_right_count, right_to_left_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (hour_start, row['total_events'], row['unique_people'],
                      row['active_events'], row['passive_events'], row['avg_speed'],
                      row['ltr'], row['rtl']))
            
            self.conn.commit()
    
    # =========================================================================
    # AGGREGATION METHODS (for long-term data retention)
    # =========================================================================
    
    def aggregate_hour(self, date_str: str, hour: int) -> dict:
        """
        Aggregate raw events for a specific hour into hourly_stats.
        Call this at the END of each hour or during maintenance.
        
        Returns stats dict for logging/verification.
        """
        # Calculate time bounds
        start_dt = datetime.strptime(f"{date_str} {hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + timedelta(hours=1)
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()
        
        with self.lock:
            cursor = self.conn.cursor()
            
            # Aggregate tracking events
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT person_id) as unique_people,
                    SUM(CASE WHEN zone = 'active' THEN 1 ELSE 0 END) as active_count,
                    SUM(CASE WHEN zone = 'passive' THEN 1 ELSE 0 END) as passive_count,
                    AVG(speed) as avg_speed,
                    SUM(CASE WHEN flow_direction = 'left_to_right' THEN 1 ELSE 0 END) as ltr,
                    SUM(CASE WHEN flow_direction = 'right_to_left' THEN 1 ELSE 0 END) as rtl
                FROM tracking_events
                WHERE timestamp >= ? AND timestamp < ?
            ''', (start_ts, end_ts))
            tracking = cursor.fetchone()
            
            # Aggregate light behavior
            cursor.execute('''
                SELECT 
                    COUNT(*) as behavior_count,
                    AVG(brightness) as avg_brightness
                FROM light_behavior
                WHERE timestamp >= ? AND timestamp < ?
            ''', (start_ts, end_ts))
            behavior = cursor.fetchone()
            
            # Get dominant mode
            cursor.execute('''
                SELECT mode, COUNT(*) as cnt
                FROM light_behavior
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY mode ORDER BY cnt DESC LIMIT 1
            ''', (start_ts, end_ts))
            mode_row = cursor.fetchone()
            dominant_mode = mode_row[0] if mode_row else 'unknown'
            
            # Build stats dict
            stats = {
                'date': date_str,
                'hour': hour,
                'total_events': tracking[0] or 0,
                'unique_people': tracking[1] or 0,
                'active_count': tracking[2] or 0,
                'passive_count': tracking[3] or 0,
                'avg_speed': tracking[4] or 0.0,
                'left_to_right': tracking[5] or 0,
                'right_to_left': tracking[6] or 0,
                'dominant_mode': dominant_mode,
                'avg_brightness': behavior[1] or 0.0 if behavior else 0.0,
            }
            
            # Insert or update hourly stats
            cursor.execute('''
                INSERT OR REPLACE INTO hourly_stats 
                (date, hour, total_events, unique_people, active_count, passive_count,
                 avg_speed, left_to_right, right_to_left, dominant_mode, avg_brightness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (stats['date'], stats['hour'], stats['total_events'], 
                  stats['unique_people'], stats['active_count'], stats['passive_count'],
                  stats['avg_speed'], stats['left_to_right'], stats['right_to_left'],
                  stats['dominant_mode'], stats['avg_brightness']))
            
            self.conn.commit()
            return stats
    
    def aggregate_day(self, date_str: str) -> dict:
        """
        Aggregate hourly stats into daily summary.
        Call this at midnight for the previous day.
        """
        with self.lock:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT 
                    SUM(total_events) as total_events,
                    SUM(unique_people) as unique_people,
                    SUM(active_count) as total_active,
                    SUM(passive_count) as total_passive,
                    AVG(avg_speed) as avg_speed,
                    SUM(left_to_right) as ltr,
                    SUM(right_to_left) as rtl
                FROM hourly_stats
                WHERE date = ?
            ''', (date_str,))
            row = cursor.fetchone()
            
            if not row or row[0] is None:
                return {'date': date_str, 'total_events': 0}
            
            # Find peak and quietest hours
            cursor.execute('''
                SELECT hour, total_events FROM hourly_stats
                WHERE date = ? ORDER BY total_events DESC LIMIT 1
            ''', (date_str,))
            peak = cursor.fetchone()
            
            cursor.execute('''
                SELECT hour, total_events FROM hourly_stats
                WHERE date = ? AND total_events > 0 ORDER BY total_events ASC LIMIT 1
            ''', (date_str,))
            quietest = cursor.fetchone()
            
            ltr = row[5] or 0
            rtl = row[6] or 0
            flow_balance = (ltr - rtl) / (ltr + rtl) if (ltr + rtl) > 0 else 0
            dominant_flow = 'left_to_right' if ltr > rtl else 'right_to_left' if rtl > ltr else 'balanced'
            
            stats = {
                'date': date_str,
                'total_events': row[0] or 0,
                'unique_people': row[1] or 0,
                'total_active': row[2] or 0,
                'total_passive': row[3] or 0,
                'avg_speed': row[4] or 0.0,
                'peak_hour': peak[0] if peak else 0,
                'peak_count': peak[1] if peak else 0,
                'quietest_hour': quietest[0] if quietest else 0,
                'dominant_flow': dominant_flow,
                'flow_balance': flow_balance,
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_stats_v2
                (date, total_events, unique_people, total_active, total_passive,
                 avg_speed, peak_hour, peak_count, quietest_hour, dominant_flow, flow_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (stats['date'], stats['total_events'], stats['unique_people'],
                  stats['total_active'], stats['total_passive'], stats['avg_speed'],
                  stats['peak_hour'], stats['peak_count'], stats['quietest_hour'],
                  stats['dominant_flow'], stats['flow_balance']))
            
            self.conn.commit()
            return stats
    
    def prune_with_aggregation(self, raw_retention_hours: int = 48) -> dict:
        """
        Smart pruning: aggregate before deleting.
        
        1. Aggregate any un-aggregated hours from raw data
        2. Delete raw events older than retention_hours
        3. Hourly stats are kept FOREVER
        
        Returns dict with counts of aggregated/pruned records.
        """
        now = datetime.now()
        results = {'hours_aggregated': 0, 'events_pruned': 0, 'behavior_pruned': 0}
        
        cutoff_raw = now - timedelta(hours=raw_retention_hours)
        cutoff_ts = cutoff_raw.timestamp()
        
        # Find hours that need aggregation (have raw data but no hourly_stats)
        with self.lock:
            cursor = self.conn.cursor()
            
            # Get distinct date/hour combinations from old events
            cursor.execute('''
                SELECT DISTINCT 
                    date(datetime) as date,
                    CAST(strftime('%H', datetime) AS INTEGER) as hour
                FROM tracking_events
                WHERE timestamp < ?
            ''', (cutoff_ts,))
            old_hours = cursor.fetchall()
        
        # Aggregate each hour that's about to be pruned
        for date_str, hour in old_hours:
            # Check if already aggregated
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    'SELECT 1 FROM hourly_stats WHERE date = ? AND hour = ?',
                    (date_str, hour)
                )
                if cursor.fetchone():
                    continue  # Already aggregated
            
            try:
                self.aggregate_hour(date_str, hour)
                results['hours_aggregated'] += 1
            except Exception as e:
                print(f"Warning: Failed to aggregate {date_str} hour {hour}: {e}")
        
        # Now safe to delete old raw events
        with self.lock:
            cursor = self.conn.cursor()
            
            cursor.execute('DELETE FROM tracking_events WHERE timestamp < ?', (cutoff_ts,))
            results['events_pruned'] = cursor.rowcount
            
            cursor.execute('DELETE FROM light_behavior WHERE timestamp < ?', (cutoff_ts,))
            results['behavior_pruned'] = cursor.rowcount
            
            self.conn.commit()
        
        return results
    
    def cleanup_old_events(self, keep_days: int = 30):
        """Remove raw events older than N days (keeps summaries)"""
        cutoff = datetime.now() - timedelta(days=keep_days)
        cutoff_ts = cutoff.timestamp()
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM tracking_events WHERE timestamp < ?', (cutoff_ts,))
            deleted = cursor.rowcount
            self.conn.commit()
            return deleted
    
    def prune_old_records(self, cutoff_timestamp: float) -> int:
        """Remove raw events older than the given timestamp (for 24/7 operation)"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM tracking_events WHERE timestamp < ?', (cutoff_timestamp,))
            deleted = cursor.rowcount
            self.conn.commit()
            return deleted
    
    def save_meta_tuning_review(self, review: Dict):
        """
        Save a meta-tuning review (periodic automated analysis of auto-tuner performance).
        Written by autotune_meta_review.py 3x/day.
        """
        with self.lock:
            cursor = self.conn.cursor()
            now = time.time()
            cursor.execute('''
                INSERT INTO meta_tuning_reviews (
                    timestamp, datetime, review_window_hours,
                    total_adjustments, total_tracking_events, unique_people,
                    avg_short_activity, median_short_activity, avg_medium_activity, avg_energy_level,
                    pct_at_floor_json, pct_at_ceiling_json,
                    mode_distribution_json, param_stats_json,
                    old_config_json, new_config_json, changes_summary,
                    diagnosis, recommendations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                now,
                datetime.now().isoformat(),
                review.get('review_window_hours', 8),
                review.get('total_adjustments', 0),
                review.get('total_tracking_events', 0),
                review.get('unique_people', 0),
                review.get('avg_short_activity'),
                review.get('median_short_activity'),
                review.get('avg_medium_activity'),
                review.get('avg_energy_level'),
                json.dumps(review.get('pct_at_floor', {})),
                json.dumps(review.get('pct_at_ceiling', {})),
                json.dumps(review.get('mode_distribution', {})),
                json.dumps(review.get('param_stats', {})),
                json.dumps(review.get('old_config', {})),
                json.dumps(review.get('new_config', {})),
                review.get('changes_summary', ''),
                review.get('diagnosis', ''),
                json.dumps(review.get('recommendations', [])),
            ))
            self.conn.commit()
    
    def get_recent_meta_reviews(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent meta-tuning reviews.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM meta_tuning_reviews
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            try:
                results.append({
                    'id': row['id'],
                    'datetime': row['datetime'],
                    'review_window_hours': row['review_window_hours'],
                    'total_adjustments': row['total_adjustments'],
                    'total_tracking_events': row['total_tracking_events'],
                    'unique_people': row['unique_people'],
                    'avg_short_activity': row['avg_short_activity'],
                    'median_short_activity': row['median_short_activity'],
                    'avg_medium_activity': row['avg_medium_activity'],
                    'avg_energy_level': row['avg_energy_level'],
                    'pct_at_floor': json.loads(row['pct_at_floor_json'] or '{}'),
                    'pct_at_ceiling': json.loads(row['pct_at_ceiling_json'] or '{}'),
                    'mode_distribution': json.loads(row['mode_distribution_json'] or '{}'),
                    'param_stats': json.loads(row['param_stats_json'] or '{}'),
                    'old_config': json.loads(row['old_config_json'] or '{}'),
                    'new_config': json.loads(row['new_config_json'] or '{}'),
                    'changes_summary': row['changes_summary'],
                    'diagnosis': row['diagnosis'],
                    'recommendations': json.loads(row['recommendations_json'] or '[]'),
                })
            except (json.JSONDecodeError, TypeError):
                pass
        return results

    def flush(self):
        """Commit any pending writes immediately"""
        with self.lock:
            if self._pending_writes > 0:
                self.conn.commit()
                self._pending_writes = 0
                self._last_commit_time = time.time()
    
    def close(self):
        """Close database connection, committing any pending writes"""
        self.flush()
        self.conn.close()


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("Testing TrackingDatabase...")
    db = TrackingDatabase("test_tracking.db")
    
    # Simulate some tracking data
    print("\nSimulating 100 tracking events...")
    for i in range(100):
        person_id = random.randint(1, 10)
        x = random.uniform(-200, 440)
        z = random.uniform(78, 600)
        db.record_position(person_id, x, z)
        time.sleep(0.01)
    
    # Get stats
    print("\nCurrent stats:")
    stats = db.get_current_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\nTrends (last 60 min):")
    trends = db.get_trends(60)
    print(f"  Unique people: {trends.unique_people}")
    print(f"  Active zone visits: {trends.active_zone_visits}")
    print(f"  Passive zone count: {trends.passive_zone_count}")
    print(f"  Avg speed: {trends.avg_speed:.1f} cm/s")
    print(f"  Flow L->R: {trends.left_to_right_count}")
    print(f"  Flow R->L: {trends.right_to_left_count}")
    
    print(f"\nFlow balance: {db.get_flow_balance(60):.2f}")
    
    db.close()
    print("\nDone!")
