# Database Scaling Plan (Updated January 29, 2026)

## Context

The Drop Ceiling installation will run for **30+ days** continuously. Yesterday's performance improvements addressed immediate issues with smooth movement, but database growth remains a concern for long-term operation.

### Current State (Post-Performance Fixes)

**What We Have:**
- `tracking_database.py` with SQLite storage
- **Batched commits** (every 50 writes or 1 second) - reduces I/O overhead
- **Background trends queries** in `light_behavior.py` - uses separate read-only connection to avoid blocking main thread
- **7-day retention** with hourly pruning (`DB_RETENTION_DAYS = 7`, `DB_PRUNE_INTERVAL = 3600`)
- Existing tables: `tracking_events`, `light_behavior`, `hourly_summary`, `daily_summary`, `person_sessions`

**Performance Improvements Made:**
1. Batched writes instead of per-event commits
2. Background thread for expensive trend queries
3. Separate read-only DB connection for analytics
4. Skipped expensive historical queries in real-time loop

**What's Missing:**
- Aggregation tables are defined but **never populated**
- No nightly rollup job
- Current prune deletes raw data but **doesn't preserve summaries first**
- 7-day retention is good but might want longer aggregate history

---

## Growth Projections

### Event Rate Analysis
| Period | Events/sec | Events/hour | Events/day |
|--------|------------|-------------|------------|
| Busy (pedestrian traffic) | 30-60 | ~150,000 | ~1.5M |
| Quiet (night/rain) | 1-5 | ~10,000 | ~100K |
| Average | ~15 | ~54,000 | ~650K |

### 30-Day Raw Data Projection
- ~20 million tracking_events rows
- ~100K light_behavior rows (recorded every 0.5-2s)
- At ~100 bytes/row: **~2GB raw data**

### Current 7-Day Retention
- ~4.5 million rows max
- ~450MB typical size
- **Acceptable** but loses historical patterns

---

## Recommended Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     RAW EVENTS (Hot)                           │
│  Table: tracking_events, light_behavior                        │
│  Retention: 48 hours                                           │
│  Purpose: Real-time trends, immediate analysis                 │
│  Size: ~7M rows max = ~700MB                                   │
└────────────────────────────────────────────────────────────────┘
                              │
                   (Hourly aggregation job)
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   HOURLY STATS (Warm)                          │
│  Table: hourly_stats                                           │
│  Retention: FOREVER (for post-exhibition analysis)             │
│  Purpose: Time-of-day patterns, weekly comparisons             │
│  Size: ~720 rows/month = ~100KB/month                          │
└────────────────────────────────────────────────────────────────┘
                              │
                    (Daily aggregation)
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    DAILY STATS (Cold)                          │
│  Table: daily_stats_v2                                         │
│  Retention: FOREVER (exhibition lifetime)                      │
│  Purpose: Long-term trends, total counts                       │
│  Size: ~30-60 rows = ~10KB                                     │
└────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Create Proper Aggregation Tables (Now)

Add new optimized table schema to `tracking_database.py`:

```python
# In _create_tables():

# Hourly statistics (aggregated from raw events)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS hourly_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,                -- YYYY-MM-DD
        hour INTEGER NOT NULL,             -- 0-23
        -- Tracking stats
        total_events INTEGER DEFAULT 0,
        unique_people INTEGER DEFAULT 0,
        active_count INTEGER DEFAULT 0,    -- Zone entries
        passive_count INTEGER DEFAULT 0,
        avg_speed REAL DEFAULT 0,
        -- Flow stats
        left_to_right INTEGER DEFAULT 0,
        right_to_left INTEGER DEFAULT 0,
        -- Engagement metrics
        bloom_count INTEGER DEFAULT 0,     -- Full engagements
        almost_engaged INTEGER DEFAULT 0,  -- Slowdowns in passive
        -- Light behavior stats
        dominant_mode TEXT,
        avg_aggression REAL DEFAULT 0,
        avg_brightness REAL DEFAULT 0,
        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(date, hour)
    )
''')

# Daily summary (aggregated from hourly)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_stats_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL UNIQUE,         -- YYYY-MM-DD
        -- Totals
        total_events INTEGER DEFAULT 0,
        unique_people INTEGER DEFAULT 0,
        total_active INTEGER DEFAULT 0,
        total_passive INTEGER DEFAULT 0,
        total_blooms INTEGER DEFAULT 0,
        -- Patterns
        peak_hour INTEGER,
        peak_count INTEGER,
        quietest_hour INTEGER,
        dominant_flow TEXT,
        flow_balance REAL DEFAULT 0,
        -- Averages
        avg_speed REAL DEFAULT 0,
        avg_aggression REAL DEFAULT 0,
        avg_conversion_rate REAL DEFAULT 0,
        -- Metadata
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Index for fast date lookups
cursor.execute('CREATE INDEX IF NOT EXISTS idx_hourly_date ON hourly_stats(date)')
```

### Phase 2: Add Aggregation Methods (Now)

Add to `tracking_database.py`:

```python
def aggregate_hour(self, date_str: str, hour: int) -> dict:
    """
    Aggregate raw events for a specific hour into hourly_stats.
    Call this at the END of each hour or during nightly job.
    
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
        
        # Insert or update hourly stats
        stats = {
            'date': date_str,
            'hour': hour,
            'total_events': tracking['total_events'] or 0,
            'unique_people': tracking['unique_people'] or 0,
            'active_count': tracking['active_count'] or 0,
            'passive_count': tracking['passive_count'] or 0,
            'avg_speed': tracking['avg_speed'] or 0.0,
            'left_to_right': tracking['ltr'] or 0,
            'right_to_left': tracking['rtl'] or 0,
            'dominant_mode': dominant_mode,
            'avg_brightness': behavior['avg_brightness'] or 0.0,
        }
        
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
                SUM(right_to_left) as rtl,
                AVG(avg_brightness) as avg_brightness
            FROM hourly_stats
            WHERE date = ?
        ''', (date_str,))
        row = cursor.fetchone()
        
        # Find peak and quietest hours
        cursor.execute('''
            SELECT hour, total_events FROM hourly_stats
            WHERE date = ? ORDER BY total_events DESC LIMIT 1
        ''', (date_str,))
        peak = cursor.fetchone()
        
        cursor.execute('''
            SELECT hour, total_events FROM hourly_stats
            WHERE date = ? ORDER BY total_events ASC LIMIT 1
        ''', (date_str,))
        quietest = cursor.fetchone()
        
        ltr = row['ltr'] or 0
        rtl = row['rtl'] or 0
        flow_balance = (ltr - rtl) / (ltr + rtl) if (ltr + rtl) > 0 else 0
        dominant_flow = 'left_to_right' if ltr > rtl else 'right_to_left' if rtl > ltr else 'balanced'
        
        stats = {
            'date': date_str,
            'total_events': row['total_events'] or 0,
            'unique_people': row['unique_people'] or 0,
            'total_active': row['total_active'] or 0,
            'total_passive': row['total_passive'] or 0,
            'avg_speed': row['avg_speed'] or 0.0,
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


def prune_with_aggregation(self, raw_retention_hours: int = 48, 
                           hourly_retention_days: int = 30) -> dict:
    """
    Smart pruning: aggregate before deleting.
    
    1. Aggregate any un-aggregated hours from raw data
    2. Delete raw events older than retention_hours
    3. Delete hourly stats older than retention_days
    
    Returns dict with counts of aggregated/pruned records.
    """
    now = datetime.now()
    results = {'hours_aggregated': 0, 'events_pruned': 0, 'hourly_pruned': 0}
    
    with self.lock:
        cursor = self.conn.cursor()
        
        # Find hours that have raw data but no aggregation
        cutoff_raw = now - timedelta(hours=raw_retention_hours)
        
        # Get distinct hours from raw events that don't have hourly_stats
        cursor.execute('''
            SELECT DISTINCT 
                date(datetime(timestamp, 'unixepoch')) as date,
                strftime('%H', datetime(timestamp, 'unixepoch')) as hour
            FROM tracking_events
            WHERE timestamp < ?
            AND NOT EXISTS (
                SELECT 1 FROM hourly_stats 
                WHERE hourly_stats.date = date(datetime(tracking_events.timestamp, 'unixepoch'))
                AND hourly_stats.hour = CAST(strftime('%H', datetime(tracking_events.timestamp, 'unixepoch')) AS INTEGER)
            )
        ''', (cutoff_raw.timestamp(),))
        
        missing_hours = cursor.fetchall()
    
    # Aggregate missing hours (outside lock to avoid long hold)
    for date_str, hour_str in missing_hours:
        try:
            self.aggregate_hour(date_str, int(hour_str))
            results['hours_aggregated'] += 1
        except Exception as e:
            print(f"Warning: Failed to aggregate {date_str} hour {hour_str}: {e}")
    
    with self.lock:
        cursor = self.conn.cursor()
        
        # Now safe to delete old raw events
        cursor.execute('DELETE FROM tracking_events WHERE timestamp < ?', 
                       (cutoff_raw.timestamp(),))
        results['events_pruned'] = cursor.rowcount
        
        # Also prune old light_behavior
        cursor.execute('DELETE FROM light_behavior WHERE timestamp < ?',
                       (cutoff_raw.timestamp(),))
        
        # Prune old hourly stats
        cutoff_hourly = (now - timedelta(days=hourly_retention_days)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM hourly_stats WHERE date < ?', (cutoff_hourly,))
        results['hourly_pruned'] = cursor.rowcount
        
        self.conn.commit()
    
    return results
```

### Phase 3: Integrate into Controller (Now)

Update `lightController_osc.py`:

```python
# Change constants
DB_RETENTION_HOURS = 48    # Keep raw events for 48 hours (was 7 days)
HOURLY_RETENTION_DAYS = 30  # Keep hourly aggregates for 30 days
DB_PRUNE_INTERVAL = 3600   # Still prune hourly

# In main loop, replace pruning section:
if current_time - last_db_prune >= DB_PRUNE_INTERVAL:
    try:
        results = tracking_db.prune_with_aggregation(
            raw_retention_hours=DB_RETENTION_HOURS,
            hourly_retention_days=HOURLY_RETENTION_DAYS
        )
        if results['events_pruned'] > 0 or results['hours_aggregated'] > 0:
            logger.info(
                f"📊 DB maintenance: aggregated {results['hours_aggregated']} hours, "
                f"pruned {results['events_pruned']} events, "
                f"pruned {results['hourly_pruned']} old hourly stats"
            )
    except Exception as e:
        logger.warning(f"Database maintenance failed: {e}")
    
    last_db_prune = current_time
```

### Phase 4: Add Hourly Aggregation Trigger (Optional Enhancement)

For fresher data, trigger aggregation at the end of each hour:

```python
# In main loop, add:
last_hour_aggregated = datetime.now().hour

# Later in loop:
current_hour = datetime.now().hour
if current_hour != last_hour_aggregated:
    # Hour just changed - aggregate the previous hour
    try:
        prev_hour = (current_hour - 1) % 24
        if prev_hour > current_hour:  # Crossed midnight
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        stats = tracking_db.aggregate_hour(date_str, prev_hour)
        logger.info(f"📊 Aggregated hour {prev_hour}: {stats['unique_people']} people, "
                    f"{stats['active_count']} active, {stats['passive_count']} passive")
    except Exception as e:
        logger.warning(f"Hourly aggregation failed: {e}")
    
    last_hour_aggregated = current_hour
```

---

## Migration for Existing Data

Since you have data already collected, run this one-time migration:

```python
def migrate_existing_data(db: TrackingDatabase):
    """One-time migration to aggregate existing raw data into hourly_stats."""
    from datetime import datetime, timedelta
    
    with db.lock:
        cursor = db.conn.cursor()
        
        # Find date range of existing data
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM tracking_events')
        row = cursor.fetchone()
        if not row or not row[0]:
            print("No existing data to migrate")
            return
        
        start_dt = datetime.fromtimestamp(row[0])
        end_dt = datetime.fromtimestamp(row[1])
    
    print(f"Migrating data from {start_dt} to {end_dt}")
    
    # Iterate through each hour
    current = start_dt.replace(minute=0, second=0, microsecond=0)
    hours_processed = 0
    
    while current < end_dt:
        date_str = current.strftime('%Y-%m-%d')
        hour = current.hour
        
        try:
            stats = db.aggregate_hour(date_str, hour)
            if stats['total_events'] > 0:
                hours_processed += 1
                print(f"  {date_str} hour {hour}: {stats['unique_people']} people")
        except Exception as e:
            print(f"  Error aggregating {date_str} hour {hour}: {e}")
        
        current += timedelta(hours=1)
    
    print(f"\nMigration complete: {hours_processed} hours aggregated")

# Run with:
# from tracking_database import TrackingDatabase
# db = TrackingDatabase("tracking_history.db")
# migrate_existing_data(db)
```

---

## Storage Projections After Implementation

| Layer | Retention | Max Rows (30 days) | Max Size |
|-------|-----------|----------|----------|
| Raw events | 48 hours | ~7M | ~700 MB |
| Light behavior | 48 hours | ~100K | ~20 MB |
| Hourly stats | Forever | ~720 | ~100 KB |
| Daily stats | Forever | ~30 | ~5 KB |
| **Total** | - | ~7.1M | **~720 MB** |

After 1 year:
- Hourly stats: ~8,760 rows = ~1 MB
- Daily stats: ~365 rows = ~50 KB

vs. unbounded 30-day raw data: **~2.5 GB** → **70% reduction**

---

## Implementation Priority

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Add `hourly_stats` table | **HIGH** | 30 min | Preserves historical patterns |
| Add `aggregate_hour()` method | **HIGH** | 1 hour | Enables smart pruning |
| Add `prune_with_aggregation()` | **HIGH** | 30 min | Safe data lifecycle |
| Update controller pruning | **HIGH** | 15 min | Activates new system |
| Run migration on existing data | **MEDIUM** | 5 min | Backfills history |
| Add hourly trigger | **LOW** | 15 min | Fresher aggregates |
| Daily aggregation | **LOW** | 30 min | Nice to have |

**Recommended: Complete HIGH priority items today before more data accumulates.**

---

## Testing Checklist

- [ ] `hourly_stats` table created successfully
- [ ] `aggregate_hour()` produces correct counts
- [ ] `prune_with_aggregation()` aggregates before deleting
- [ ] Migration script processes existing data
- [ ] Controller runs smoothly with new pruning
- [ ] No performance regression in main loop
- [ ] Verify data preserved after 48-hour prune cycle
