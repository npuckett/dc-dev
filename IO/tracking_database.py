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
    """
    
    def __init__(self, db_path: str = "tracking_history.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        
        # Track previous positions for velocity calculation
        self.prev_positions: Dict[int, Tuple[float, float, float, float]] = {}  # id -> (x, z, timestamp)
        
        # Zone boundaries (should match lightController)
        self.active_zone = {
            'x_min': 120 - 475/2,  # -117.5
            'x_max': 120 + 475/2,  # 357.5
            'z_min': 78,
            'z_max': 78 + 205,     # 283
        }
        self.passive_zone = {
            'x_min': 120 - 650/2,  # -205
            'x_max': 120 + 650/2,  # 445
            'z_min': 283,
            'z_max': 283 + 330,    # 613
        }
        
        self._create_tables()
    
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
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO tracking_events 
                (timestamp, datetime, person_id, x, z, vx, vz, speed, zone, flow_direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, dt_str, person_id, x, z, vx, vz, speed, zone.value, flow_dir))
            self.conn.commit()
    
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
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt_str = datetime.fromtimestamp(timestamp).isoformat()
        
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
            self.conn.commit()
    
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
            return {
                'people_last_minute': row['people'] or 0,
                'avg_speed': row['avg_speed'] or 0,
                'active_events': row['active_events'] or 0,
                'passive_events': row['passive_events'] or 0,
                'flow_left_to_right': row['ltr'] or 0,
                'flow_right_to_left': row['rtl'] or 0,
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
    
    def close(self):
        """Close database connection"""
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
