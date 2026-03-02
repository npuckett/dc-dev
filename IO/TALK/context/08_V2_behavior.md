# V2 Behavior

## Overview
This installation is looking out onto a busy sidewalk, but the active tracking zone is a bit inset from the street because of the large pillars. From the first few days of operation, it is clear that very few people actually take the time to stop and engage. Pretty much everyone just walks past, so the system needs to really acknowledge and reward getting people to stop and interact. It should be trying to get as many people as possible to stop and look at the panels, but within the overall constraints of its general movement guidelines. Outside of that it should function to develop language and personality over time that reflects the movement in the passive tracking zone.

## Basics
- Always prioritize response to people in the active tracking zone.
- The goal is to try and get people to stop and interact, but only within the constraints of the overall movement guidelines.
- When not responding to people in the active zone, it should be moving in response to input from the passive zone.
  - Current Data
  - Short Term Trends
  - Time of Day

---

## V2 Behavior Plan

### Primary Goal
**Convert passersby into engagers** - Use passive zone awareness to attract attention and maximize the chance someone will step into the active zone, then reward that engagement richly.

### Behavior Mode Hierarchy

The system operates in a clear priority hierarchy:

```
1. FOLLOW     (highest priority - someone is in active zone)
2. DWELL      (someone is staying in active zone)
3. BLOOM      (reward for deep engagement)
4. IDLE       (lowest priority - respond to passive trends)
```

---

## Mode Breakdown

### 1. IDLE Mode - "Fishing for Attention"

**Purpose**: When no one is in the active zone, use passive zone data to create movement that might catch the eye of passersby.

**Current Data Available**:
- `recent_passive_count` - People passing in last 1 minute
- `recent_active_count` - Active zone visits in last 1 minute  
- `short/medium/long_activity_level` - Activity over 5min/30min/1hr
- `flow_momentum` - Directional bias (-1 to +1)
- `activity_anticipation` - Expectation of upcoming activity (0-1)
- `energy_level` - Overall energy to match (0-1)

**Proposed IDLE Behaviors**:

| Passive Zone State | Light Behavior | Goal |
|-------------------|----------------|------|
| **High traffic, low engagement** | Move toward passive zone edge, pulse gently with flow | "Hey, look at me" |
| **Low traffic** | Slower, more contemplative movement in center | Conserve energy, be available |
| **Flow left-to-right** | Drift with flow, then pause at edge | Anticipate where eyes might land |
| **Flow right-to-left** | Mirror the flow, pause opposite edge | Same as above |
| **Activity spike detected** | Quick attention-getting movement toward passive zone | Capitalize on foot traffic |
| **Activity dropping** | Retreat toward center, slow down | Prepare for quiet period |

**Key Mechanism**: The wander box should shift toward the passive zone edge during high traffic, and retreat to center during low traffic. Speed should increase slightly when anticipation is high.

---

### 2. FOLLOW Mode - "I See You"

**Purpose**: Someone has entered the active zone. Track them, but don't be creepy.

**Current Implementation**:
- Light moves toward nearest active person
- Speed scales with distance
- Entry pulse when first detected

**Proposed Refinements**:

| Situation | Behavior | Rationale |
|-----------|----------|-----------|
| **Person enters from passive zone** | Strong entry pulse + quick initial movement | They transitioned from passing to engaging |
| **Person enters from deep** | Gentler response | They came specifically, no need to startle |
| **Person moving fast** | Match their energy, move with them | They're passing through, acknowledge |
| **Person slowing down** | Slow down too, start dwell transition | They might stay |

---

### 3. DWELL Mode - "You're Staying? Let Me Reward You"

**Purpose**: Someone has stopped in the active zone. This is the rare and valuable engagement we want to encourage.

**Current Dwell Phase System**:
- Phase 0 (0-3s): CURIOUS - Initial acknowledgment
- Phase 1 (3-8s): ENGAGED - Active following
- Phase 2 (8-15s): REWARDED - Special behaviors unlock
- Phase 3 (15s+): DEEP - Bloom gesture available

**Proposed Refinements**:

| Dwell Phase | Light Behavior | Audio/Visual Cue |
|-------------|----------------|------------------|
| **CURIOUS** | Slow approach, brightness increases | Subtle acknowledgment |
| **ENGAGED** | Tighter follow, gentle pulse rhythm | "I'm with you" |
| **REWARDED** | Playful movements, gestures | "You're special for staying" |
| **DEEP** | Bloom gesture triggered, expansive | Full reward for engagement |

**Key Insight**: Since most people just walk past, anyone who stays for even 8 seconds deserves significant reward. The dwell phases should feel like escalating gifts.

---

### 4. BLOOM Mode - "Thank You for Engaging"

**Purpose**: The ultimate reward for deep engagement (15+ seconds dwell).

**Current Implementation**:
- Light expands to illuminate all panels
- Brightness increases dramatically
- Slow return to normal after bloom

**No changes proposed** - this is working well as the reward mechanism.

---

## Using Trends to Influence Behavior

### Time Window Purpose

| Window | Purpose | How It's Used |
|--------|---------|---------------|
| **1 minute** | Immediate reactivity | Wander box position, speed |
| **5 minutes** | Short-term rhythm | Energy level baseline |
| **30 minutes** | Session energy | Personality mood |
| **1 hour** | Daily pattern | Anticipation of busy/quiet periods |
| **Historical** | Typical for this time of day | Compare current to expected |

### Trend-Influenced Parameters

1. **Wander Box Position**
   - High activity anticipation → Box shifts toward passive zone edge
   - Low activity → Box centers, smaller size

2. **Movement Speed**
   - High energy level → Faster movement
   - Low energy → Slower, more contemplative

3. **Pulse Intensity**  
   - High traffic with low engagement → More frequent subtle pulses
   - Low traffic → Fewer, gentler pulses

4. **Brightness**
   - Evening/night → Slightly brighter to be visible
   - Morning → Softer, not too aggressive

---

## Attention-Getting Strategies

### The "Look At Me" Problem
People walking past don't know the panels are interactive. The system needs to occasionally do something that catches peripheral vision without being annoying.

**Proposed Strategies** (to implement):

1. **Anticipatory Positioning**
   - When flow_momentum indicates traffic coming from one direction
   - Move toward that side of the panels in advance
   - Person sees light already "looking" toward them

2. **Peripheral Pulse**
   - When passive count is high but active count is 0
   - Subtle pulse that's visible in peripheral vision
   - Not constant - triggered by flow spikes

3. **Flow Following in IDLE**
   - When someone passes in passive zone
   - Light drifts with their direction
   - Creates sense of "awareness" even when not engaging

4. **Invitation Gesture**
   - When traffic is high but engagement is low
   - Occasional movement toward the active zone edge
   - Like beckoning without being desperate

---

## Success Metrics to Track

1. **Conversion Rate**: `active_zone_visits / passive_zone_passes`
   - Goal: Increase this over time
   - Track by time of day, day of week

2. **Dwell Depth**: Average dwell time when someone does engage
   - Goal: Increase average dwell time
   - Track how often we reach DEEP phase

3. **Engagement per Hour**: Total active zone time / hour
   - Combines conversion rate and dwell depth
   - Primary optimization target

---

## Implementation Phases

### Phase 1: Trend-Responsive IDLE (Current)
✅ Wander box influenced by trends
✅ Speed/brightness modulated by energy level
✅ Flow momentum affects movement
✅ Visualization of trends

### Phase 2: Anticipatory Positioning (Next)
- [ ] Shift wander box toward passive zone when anticipation is high
- [ ] Move toward traffic flow direction
- [ ] "Peripheral pulse" when high passive, low active

### Phase 3: Enhanced Dwell Rewards
- [ ] More dramatic phase transitions
- [ ] Unique behaviors at each dwell phase
- [ ] Track and display dwell statistics

### Phase 4: Pattern Learning (Future)
- [ ] Learn which behaviors correlate with engagement
- [ ] Adjust personality parameters based on success
- [ ] Time-of-day specific strategies

---

## Questions to Resolve

1. **How aggressive should "attention-getting" be?**
   - Must stay within the movement/behavior bounds - no random flashing
   - Aggression level is a parameter that changes based on inputs:
     - High traffic + low engagement = more aggressive
     - Low traffic = less aggressive (no one to attract)
     - Recent engagement success = less aggressive (working fine)
     - Long drought = gradually increase
   - Think of aggression as a "desperation meter" that rises/falls

2. **Should behavior differ by time of day?**
   - **Yes, significantly**. The location is near a subway in a financial district:
     - **Morning rush (7-9am)**: Fast commuters, unlikely to stop. Low aggression, acknowledge flow.
     - **Mid-morning (9-11am)**: Slower foot traffic, higher chance of curiosity. Medium aggression.
     - **Lunch (11am-2pm)**: Mixed - some rushing, some strolling. Responsive aggression.
     - **Afternoon (2-5pm)**: Office workers, declining engagement. Medium-high aggression.
     - **Evening (5-7pm)**: Commuters leaving, fast-moving. Low aggression, acknowledge flow.
     - **Night (7pm+)**: Financial district gets dead. Very low aggression, contemplative.
     - **Weekend**: Different patterns entirely - more exploratory visitors.

3. **How to handle the "almost engaged" person?**
   - This is a key opportunity for testing different attractors
   - Someone who slows in passive zone but doesn't enter active
   - **Proposed experiments**:
     - Subtle brightness pulse when someone slows down
     - Gentle drift toward them
     - Pause and "look" at them
   - Track which approach leads to conversion

---

## Technical Framework Considerations

### Current Architecture Limitations
The current system updates trends every 5 seconds with database queries. This works but has limitations:
- No smoothing between updates
- No pattern detection beyond simple averages
- No learning/adaptation mechanism

### Framework Comparison

---

#### 1. **Pandas (Time-Series Analysis)**

**What it does**: DataFrame-based data manipulation with powerful time-series operations.

**Pros**:
- ✅ Already installed in most Python environments
- ✅ Excellent for rolling windows, resampling, aggregation
- ✅ Built-in handling of time indices
- ✅ Easy to prototype and test queries
- ✅ Great for batch analysis (daily reports, historical patterns)
- ✅ Integrates well with matplotlib for visualization

**Cons**:
- ❌ Memory overhead for real-time use (creates new DataFrames)
- ❌ Not designed for streaming/real-time data
- ❌ Overkill for simple running averages
- ❌ Can be slow if used per-frame

**Best for**: Periodic trend analysis (every 5-30 seconds), daily report generation, historical pattern detection.

**Not ideal for**: Per-frame calculations, real-time smoothing.

```python
# Good use case: Session analysis every 30 seconds
df = pd.DataFrame(self.event_buffer)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
activity_5min = df.last('5min').groupby('zone').size()
```

---

#### 2. **Exponential Moving Average (EMA) - Custom Implementation**

**What it does**: Lightweight smoothing that weights recent values more heavily.

**Pros**:
- ✅ Extremely fast (single multiply-add per update)
- ✅ No memory overhead (stores single value)
- ✅ Perfect for real-time/per-frame smoothing
- ✅ No dependencies
- ✅ Tunable responsiveness via alpha parameter
- ✅ Natural decay of old data

**Cons**:
- ❌ Only tracks one value (no windowed aggregations)
- ❌ Can't easily answer "how many in last 5 minutes?"
- ❌ Need to implement yourself
- ❌ Alpha tuning requires experimentation

**Best for**: Smoothing position, speed, brightness, aggression level between frames.

**Not ideal for**: Counting events in time windows, complex aggregations.

```python
class EMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += self.alpha * (x - self.value)
        return self.value
```

---

#### 3. **Ring Buffer / Circular Queue**

**What it does**: Fixed-size buffer that overwrites oldest data, perfect for sliding windows.

**Pros**:
- ✅ Fixed memory footprint
- ✅ O(1) append, O(n) aggregation
- ✅ Easy to implement
- ✅ Natural sliding window behavior
- ✅ Can store full event data for analysis

**Cons**:
- ❌ Size must be pre-determined
- ❌ Aggregation still requires iterating
- ❌ Different time windows need different buffers
- ❌ Time-based windows tricky (events aren't evenly spaced)

**Best for**: Keeping last N events for analysis, recent event lookup.

**Not ideal for**: Time-based windows with variable event rates.

```python
from collections import deque

class EventBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, event):
        self.buffer.append((time.time(), event))
    
    def get_last_n_seconds(self, seconds):
        cutoff = time.time() - seconds
        return [e for t, e in self.buffer if t > cutoff]
```

---

#### 4. **SciPy (Signal Processing)**

**What it does**: Scientific computing library with signal processing, statistics, optimization.

**Pros**:
- ✅ Powerful pattern detection (peaks, periodicity)
- ✅ Statistical functions (correlation, regression)
- ✅ Filter design (low-pass, high-pass for smoothing)
- ✅ Well-tested, optimized implementations
- ✅ Good for detecting activity spikes

**Cons**:
- ❌ Heavyweight dependency
- ❌ Designed for batch processing, not streaming
- ❌ Overkill for simple trend tracking
- ❌ Learning curve for signal processing concepts
- ❌ Requires numpy arrays

**Best for**: Detecting patterns in historical data, spike detection, periodicity analysis.

**Not ideal for**: Real-time per-frame processing, simple averaging.

```python
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# Detect activity spikes in last hour
activity = np.array(hourly_counts)
smoothed = uniform_filter1d(activity, size=5)
peaks, _ = find_peaks(smoothed, height=threshold, distance=10)
```

---

#### 5. **SQLite with Time-Based Views**

**What it does**: Use the existing database more intelligently with views and triggers.

**Pros**:
- ✅ Already have SQLite database
- ✅ Efficient for time-window queries (with proper indexing)
- ✅ Persistent storage
- ✅ Can create views for common aggregations
- ✅ No additional dependencies
- ✅ Handles time-based windows naturally

**Cons**:
- ❌ Query overhead (disk I/O, parsing)
- ❌ Not suitable for per-frame use
- ❌ Schema changes needed for new metrics
- ❌ Harder to do complex calculations (vs Python)

**Best for**: Trend queries every 5+ seconds, historical analysis, persistence.

**Not ideal for**: Real-time smoothing, complex pattern matching.

```sql
-- Create a view for quick trend access
CREATE VIEW IF NOT EXISTS activity_trends AS
SELECT 
    strftime('%H', datetime) as hour,
    COUNT(*) as total_events,
    SUM(CASE WHEN zone='active' THEN 1 ELSE 0 END) as active_events,
    AVG(speed) as avg_speed
FROM tracking_events
WHERE timestamp > unixepoch() - 3600
GROUP BY hour;
```

---

#### 6. **py_trees (Behavior Trees)**

**What it does**: Framework for hierarchical behavior selection, popular in robotics/games.

**Pros**:
- ✅ Clear visual representation of behavior logic
- ✅ Built-in priority handling (selector nodes)
- ✅ Easy to add/remove behaviors
- ✅ Good debugging tools
- ✅ Well-suited for complex decision making

**Cons**:
- ❌ Additional dependency to install
- ❌ Learning curve for behavior tree concepts
- ❌ Might be overkill (current mode system is simple)
- ❌ Refactoring existing code to fit the paradigm
- ❌ Less common in art installations

**Best for**: Complex behavior selection with many conditions, if current mode system becomes unwieldy.

**Not ideal for**: Simple state machines (like current IDLE/FOLLOW/DWELL/BLOOM).

---

#### 7. **Simple State Machine (Current Approach)**

**What it does**: Enum-based modes with explicit transitions.

**Pros**:
- ✅ Already implemented
- ✅ Easy to understand
- ✅ Fast (just enum comparison)
- ✅ No dependencies
- ✅ Adequate for current complexity

**Cons**:
- ❌ Can get messy with many states
- ❌ Transitions scattered through code
- ❌ Hard to visualize complex logic
- ❌ No built-in parallel behaviors

**Best for**: Current system with 4 main modes.

**Consider upgrading if**: Mode count grows beyond 6-8, or transitions become complex.

---

### Recommended Architecture

Based on this analysis, a **hybrid approach** using what's already available:

```
┌─────────────────────────────────────────────────────────────┐
│                     REAL-TIME LAYER                         │
│  (Per-frame, ~60Hz)                                         │
│                                                             │
│  • EMA smoothing for position, speed, brightness            │
│  • Simple state machine for mode selection                  │
│  • Ring buffer for last 60 seconds of events               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SHORT-TERM LAYER                         │
│  (Every 5-10 seconds)                                       │
│                                                             │
│  • SQLite queries for 1min/5min/30min/1hr trends           │
│  • Update aggression level                                  │
│  • Adjust wander box based on trends                       │
│  • Ring buffer aggregation for quick stats                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SESSION LAYER                            │
│  (Every 5-30 minutes)                                       │
│                                                             │
│  • Pandas for session analysis (optional)                   │
│  • Compare current vs historical patterns                   │
│  • Log behavior-engagement correlations                    │
│  • Adjust time-of-day parameters                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DAILY LAYER                             │
│  (At midnight)                                              │
│                                                             │
│  • Full daily report with pandas                            │
│  • Historical baseline updates                              │
│  • Pattern learning (future)                               │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Priority

| Component | Framework | Priority | Effort |
|-----------|-----------|----------|--------|
| Aggression smoothing | EMA | High | Low |
| Trend queries | SQLite (existing) | High | Low |
| Event buffer | Ring buffer | Medium | Low |
| Session analysis | Pandas (optional) | Low | Medium |
| Pattern detection | SciPy (future) | Low | High |
| Behavior trees | py_trees | None | High |

### Recommended Approach

Given the real-time nature of the system, I recommend a **lightweight layered approach**:

```
Layer 1: Real-time (per-frame)
├── Smooth position/speed using EMA
├── React to immediate events (person enters/exits)
└── Apply current aggression level

Layer 2: Short-term (every 5-10 seconds)
├── Update trend calculations
├── Adjust aggression level based on engagement ratio
└── Shift wander box based on flow

Layer 3: Session (every 5-30 minutes)
├── Compare current period to historical
├── Log behavior-to-engagement correlations
└── Adjust personality parameters if needed

Layer 4: Daily (midnight)
├── Generate daily report
├── Update historical baselines
└── Review behavior effectiveness
```

### Aggression Level System

A new parameter that modulates attention-getting behaviors:

```python
@dataclass
class AggressionState:
    level: float = 0.3  # 0.0 = passive, 1.0 = maximum attention-seeking
    
    # Factors that increase aggression
    minutes_since_engagement: float = 0.0
    passive_count_without_conversion: int = 0
    
    # Factors that decrease aggression
    recent_engagement_count: int = 0
    current_engagement: bool = False
    
    # Time-of-day modifier
    time_of_day_aggression_cap: float = 0.8  # e.g., 0.3 for night

def update_aggression(self, dt: float):
    # Base aggression rises with time since engagement
    time_factor = min(1.0, self.minutes_since_engagement / 30.0)  # Max after 30 min
    
    # Conversion failure increases aggression
    conversion_factor = min(1.0, self.passive_count_without_conversion / 50.0)
    
    # Recent success decreases aggression
    success_factor = max(0.0, 1.0 - self.recent_engagement_count * 0.2)
    
    # Combine factors
    raw_aggression = (time_factor * 0.4 + conversion_factor * 0.4) * success_factor
    
    # Cap by time of day
    self.level = min(raw_aggression, self.time_of_day_aggression_cap)
```

---

## Revised Implementation Plan

### Phase 2A: Aggression System ✅ COMPLETE
- [x] Add `AggressionState` to behavior system
- [x] Aggression influences:
  - Wander box size (higher = larger, toward passive zone)
  - Pulse frequency (higher = more frequent subtle pulses)
  - Movement speed (higher = more dynamic)
  - Response to passive zone flow (higher = more tracking)
- [x] Time-of-day caps on aggression
- [x] Visualize aggression level in trends panel

### Phase 2B: Flow-Responsive Positioning
- [x] Track real-time flow direction (last 30 seconds)
- [x] Wander box shifts toward incoming traffic direction
- [x] "Anticipatory" positioning when flow detected

### Phase 2C: Almost-Engaged Detection ✅
- [x] Track people who slow down in passive zone
- [x] When someone slows (speed < threshold) near active zone boundary:
  - Log their position and behavior state
  - Try subtle attraction (brightness pulse, drift toward)
  - Track if they convert or leave
- [x] Compare conversion rates of different attraction strategies

### Phase 3: Feedback Learning ✅
- [x] Log behavior-context when engagement occurs
- [x] Build simple model: "what was I doing when people engaged?"
- [x] Gradually weight successful behaviors higher

---

## Project Timeline & Complexity

### Revised Timeline: ~4 Weeks (Not 10 Days)

Given the scope of behavior development, testing, and iteration, this work will likely span approximately **1 month** of development. This extended timeline allows for:

1. **Iterative testing in real-world conditions** - Each phase needs observation in the actual environment
2. **Tuning parameters** - Aggression curves, time-of-day patterns, and threshold values require real data
3. **Unexpected discoveries** - Pedestrian behavior may surprise us and require adjustments
4. **Stability testing** - Long-running systems need robustness testing

### Week-by-Week Breakdown

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Aggression System (Phase 2A) | AggressionState, time-of-day caps, visualization |
| **Week 2** | Flow Positioning + Testing | Flow-responsive wander box, parameter tuning |
| **Week 3** | Almost-Engaged Detection | Passive zone tracking, attraction experiments |
| **Week 4** | Feedback Learning + Polish | Correlation logging, analysis tools, stability |

### Future Complex Implementations (Post-V2)

These are documented for future work but **not part of the current 4-week scope**:

#### Pattern Learning System (Future - Weeks 5-8)
- Machine learning model to predict engagement likelihood
- Automatic adjustment of behavior parameters based on success rates
- A/B testing framework for comparing attraction strategies
- Seasonal/weather pattern awareness (if data available)

#### Advanced Analytics Dashboard (Future - Weeks 6-10)
- Web-based dashboard for real-time monitoring
- Historical engagement visualization
- Pattern discovery tools
- Remote parameter adjustment

#### Multi-Installation Coordination (Future)
- If multiple installations exist, share learned patterns
- Aggregate data for better baseline understanding
- Coordinated behavior for nearby panels

---

## Database Growth & Retention Strategy

### The Problem: Unbounded Growth

The tracking database stores every event. With ~30 days of operation:

**Estimated Event Rate**:
- Busy periods: ~50-100 tracking events/second
- Quiet periods: ~1-10 events/second
- Average: ~20 events/second

**30-Day Projection**:
- 20 events/sec × 60 sec × 60 min × 12 hrs/day × 30 days = **~26 million events**
- At ~100 bytes/event = **~2.5 GB of raw event data**

This will:
- Slow down trend queries
- Fill disk space
- Make historical analysis unwieldy

### Layered Retention Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW EVENTS (Hot)                         │
│  Retention: 24-48 hours                                     │
│                                                             │
│  • Full tracking_events table                               │
│  • All columns preserved                                    │
│  • Used for: Real-time trends, dwell analysis              │
│  • Auto-purge: Events older than 48 hours                  │
└─────────────────────────────────────────────────────────────┘
                              │
                    (Aggregate nightly)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  HOURLY SUMMARIES (Warm)                    │
│  Retention: 30 days                                         │
│                                                             │
│  • hourly_stats table                                       │
│  • Columns: hour, date, passive_count, active_count,       │
│             avg_dwell, conversion_rate, peak_flow          │
│  • Used for: Time-of-day patterns, weekly trends           │
│  • Auto-purge: Summaries older than 30 days                │
└─────────────────────────────────────────────────────────────┘
                              │
                    (Aggregate weekly)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  DAILY SUMMARIES (Cold)                     │
│  Retention: 1 year                                          │
│                                                             │
│  • daily_stats table                                        │
│  • Columns: date, total_passive, total_active,             │
│             bloom_count, avg_conversion_rate               │
│  • Used for: Long-term trends, seasonal patterns           │
│  • Auto-purge: After 1 year (or never for analysis)        │
└─────────────────────────────────────────────────────────────┘
                              │
                    (Aggregate monthly)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 MONTHLY BASELINES (Archive)                 │
│  Retention: Forever                                         │
│                                                             │
│  • monthly_baselines table                                  │
│  • Per-hour-of-day averages for each month                 │
│  • Used for: "Typical" comparisons, anomaly detection      │
│  • Size: ~365 rows/year (tiny)                             │
└─────────────────────────────────────────────────────────────┘
```

### Estimated Storage After 1 Year

| Layer | Rows | Size |
|-------|------|------|
| Raw Events (48 hrs) | ~3.5 million | ~350 MB |
| Hourly Summaries (30 days) | ~720 | ~100 KB |
| Daily Summaries (1 year) | ~365 | ~50 KB |
| Monthly Baselines | ~12 | ~5 KB |
| **Total** | ~3.5 million | **~350 MB** |

This is a **~85% reduction** from unbounded growth.

### Implementation Plan for Database Retention

**Phase 1 (Week 1-2): Add Aggregation Tables**
```sql
CREATE TABLE hourly_stats (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    passive_count INTEGER DEFAULT 0,
    active_count INTEGER DEFAULT 0,
    avg_dwell_seconds REAL DEFAULT 0,
    bloom_count INTEGER DEFAULT 0,
    conversion_rate REAL DEFAULT 0,
    avg_aggression REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, hour)
);

CREATE TABLE daily_stats (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL UNIQUE,
    total_passive INTEGER DEFAULT 0,
    total_active INTEGER DEFAULT 0,
    total_dwell_seconds REAL DEFAULT 0,
    bloom_count INTEGER DEFAULT 0,
    avg_conversion_rate REAL DEFAULT 0,
    peak_hour INTEGER,
    notes TEXT
);
```

**Phase 2 (Week 2): Add Nightly Aggregation Job**
```python
def nightly_aggregation():
    """Run at midnight to aggregate yesterday's data."""
    yesterday = date.today() - timedelta(days=1)
    
    # Aggregate to hourly_stats
    for hour in range(24):
        stats = db.query_hour_stats(yesterday, hour)
        db.insert_hourly_stats(yesterday, hour, stats)
    
    # Aggregate to daily_stats
    daily = db.query_daily_stats(yesterday)
    db.insert_daily_stats(yesterday, daily)
```

**Phase 3 (Week 3): Add Purge Job**
```python
def purge_old_events():
    """Run daily to remove old raw events."""
    cutoff = datetime.now() - timedelta(hours=48)
    db.execute("DELETE FROM tracking_events WHERE timestamp < ?", cutoff)
    
    # Also purge old hourly stats
    cutoff_hourly = date.today() - timedelta(days=30)
    db.execute("DELETE FROM hourly_stats WHERE date < ?", cutoff_hourly)
```

**Phase 4 (Optional): Monthly Baseline Updates**
```python
def update_monthly_baseline():
    """Run on 1st of each month."""
    last_month = date.today().replace(day=1) - timedelta(days=1)
    
    # Calculate per-hour-of-day averages for the month
    for hour in range(24):
        avg_stats = db.query_monthly_hour_average(last_month.year, last_month.month, hour)
        db.insert_monthly_baseline(last_month.year, last_month.month, hour, avg_stats)
```

### When to Implement

| Task | Priority | When |
|------|----------|------|
| Create aggregation tables | High | Week 1-2 |
| Nightly aggregation job | High | Week 2 |
| Purge job (48-hour retention) | High | Week 2-3 |
| Monthly baselines | Medium | Week 4 |
| Migrate trend queries to use summaries | Low | After V2 complete |

This ensures the system can run indefinitely without database growth becoming a problem.

---

## Next Steps

1. ✅ Review and refine this plan
2. ✅ Document database retention strategy
3. Implement AggressionState system
4. Add aggregation tables to database
5. Add time-of-day aggression caps
6. Implement nightly aggregation + purge job
7. Implement flow-responsive wander box positioning
8. Add "almost-engaged" detection and attraction
9. Set up feedback logging for behavior-engagement correlation





