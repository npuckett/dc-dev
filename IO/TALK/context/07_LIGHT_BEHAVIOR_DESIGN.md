# Light Behavior Design

This document outlines how tracking input affects the simulated point light behavior.

---

## Current Light Parameters

### PointLight Class
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | [x, y, z] | [120, 60, -30] | Current 3D position in cm |
| `target_position` | [x, y, z] | [120, 60, -30] | Where the light is moving toward |
| `brightness_min` | int | 5 | Minimum DMX brightness (pulsing low) |
| `brightness_max` | int | 40 | Maximum DMX brightness (pulsing high) |
| `pulse_speed` | float | 2000 | Milliseconds per pulse cycle |
| `falloff_radius` | float | 50 | Distance (cm) at which brightness = 50% |
| `move_speed` | float | 50 | How fast light moves toward target (cm/s) |
| `pulse_phase` | float | 0.0 | Current phase in pulse cycle (radians) |

### WanderBehavior Class
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wander_box` | dict | see below | 3D bounds for random wandering |
| `wander_target` | [x, y, z] | random | Current wander destination |
| `wander_timer` | float | 0 | Time since last target change |
| `wander_interval` | float | 3.0 | Seconds between new random targets |
| `enabled` | bool | True | Whether wandering is active |

### Wander Box (cm)
```
min_x: -50    max_x: 290   (X = along panels)
min_y: 0      max_y: 150   (Y = vertical)
min_z: -32    max_z: 28    (Z = toward/away from viewer)
```

---

## Input Parameters (from Tracking)

### Real-time (per frame)
| Parameter | Source | Description |
|-----------|--------|-------------|
| Person count | OSC `/tracker/count` | Number of people currently tracked |
| Person positions | OSC `/tracker/person/<id>` | X, Z position per person (cm) |
| Zone | Calculated | ACTIVE (close) or PASSIVE (sidewalk) |

### Calculated from positions
| Parameter | Description |
|-----------|-------------|
| Velocity (vx, vz) | Speed and direction of each person (cm/s) |
| Speed | Magnitude of velocity |
| Flow direction | L→R or R→L for passive zone |
| Nearest person | Closest person to panels in active zone |

### Database Trends (from `tracking_database.py`)
| Parameter | Time Scale | Description |
|-----------|------------|-------------|
| `people_last_minute` | 1 min | Unique people in last 60 seconds |
| `avg_speed` | 1 min | Average walking speed (cm/s) |
| `active_events` | 1 min | Tracking events in active zone |
| `passive_events` | 1 min | Tracking events in passive zone |
| `flow_balance` | 1 min | -1.0 (all R→L) to +1.0 (all L→R) |
| `get_trends(minutes)` | configurable | Same stats for any time window |
| `get_hourly_pattern(days)` | 7 days | Average by hour of day |

---

## Zone Definitions (cm)

### Active Zone (engaging with installation)
- **Width**: 475 cm (X: -117.5 to 357.5)
- **Depth**: 205 cm (Z: 78 to 283)
- **Y Level**: -66 (street level)

### Passive Zone (sidewalk traffic)
- **Width**: 650 cm (X: -205 to 445)  
- **Depth**: 330 cm (Z: 283 to 613)
- **Y Level**: -66 (street level)

---

## Current Behavior (Default)

1. Light **wanders randomly** within wander box
2. Light **pulses** sinusoidally between `brightness_min` and `brightness_max`
3. Brightness falls off with distance from light to each panel

---

## Proposed Behavior Modes

### 1. **IDLE** (No one in active zone)
**Trigger**: `active_zone_count == 0`

| Parameter | Behavior |
|-----------|----------|
| Position | Slow, gentle wander |
| Pulse speed | Slow (e.g., 3000-5000ms) |
| Brightness | Low (dim ambient) |
| Movement | Smooth, dreamy |

**Trend influence**: 
- If passive zone busy → occasional "acknowledgment" moves toward busy side
- Time of day patterns → different idle moods (morning calm vs evening active)
- Weather conditions → adjust brightness and movement (e.g., more dynamic on sunny days)
- overall vector of passive traffic flow could bias wander direction slightly
- It gets bored after a while and occassionally makes a gesture when somone walks by in the passive zone

---

### 2. **ENGAGED** (People in active zone)
**Trigger**: `active_zone_count >= 1`

| Parameter | Behavior |
|-----------|----------|
| Position | Follow nearest person in active zone |
| Pulse speed | Medium (e.g., 1500-2500ms) - matches walking rhythm |
| Brightness | Brighter (engaged) |
| Movement | Responsive but not frantic |

**Options**:
- A) Follow single nearest person, but don't jump too quickly. Use smoothing. other parameters should change in response to the number of people in the active zone. and how long they have been there. it is toronto in january and cold outside. dwell time should be rewarded with more engagement. Be sure that even that there is evolution in the behavior so that it does not feel static. Acknowledge briefly when someone new enters the active zone.


---

### 3. **CROWD** (Multiple people in active zone)
**Trigger**: `active_zone_count >= 3`

| Parameter | Behavior |
|-----------|----------|
| Position | Move between people, or expand/contract |
| Pulse speed | Faster, more energetic |
| Brightness | Higher max |
| Movement | More dynamic |

See ### 2 for details on how to handle multiple people.
- be sure not to just max out all parameters. there should be a sense of dynamics and evolution in the behavior. it should feel alive and responsive rather than just hitting max values. it can get excited, but still needs to maintain a sense of smoothness and continuity.


---

### 4. **FLOW** (Heavy passive traffic, no one engaging)
**Trigger**: `active_zone_count == 0 AND passive_zone_count > threshold`

| Parameter | Behavior |
|-----------|----------|
| Position | Drift with traffic flow direction |
| Pulse speed | Match average walking speed |
| Brightness | Medium - present but not demanding |
| Movement | Smooth lateral drift |

-throw in an occasional gesture toward the passive traffic when it is busy to acknowledge their presence without being too reactive.

---

### Other considerations
- use the trends to see if the current situation is typical or not
- ensure smooth transitions between modes
- Time of day matters beyond rush hour. This works 24 hours. consider different moods for late night vs early morning vs midday.
- use trends to evolve responses over longer time periods
- Add simple text that explains the current mode and why it is behaving that way. This should be in plain language as this will be visible to the public on a screen nearby.
- Don't forget to use the size of the wander box to create interesting movement patterns. for instance at night it could mostly stay in the lower part and primarily engage panels 2 and 3 for passive mode. 

---

## Parameter Mapping by Mode

This section maps the design intentions above to specific parameter values. Edit these values to tune behavior.

### IDLE Mode Parameters

| Parameter | Base Value | Trend Influence | Notes |
|-----------|------------|-----------------|-------|
| **Position** | | | |
| `wander_box.min_y` | 0 | Late night: stay low (0-60) | Lower = panels 2&3 only |
| `wander_box.max_y` | 100 | Daytime: full range (0-150) | |
| `wander_box` X range | Full | Bias toward flow direction ±30cm | |
| `move_speed` | 20 | Quieter hour → slower (10-15) | Dreamy, gentle |
| `wander_interval` | 5.0 | Busy passive → shorter (3.0) | More restless when people nearby |
| **Brightness** | | | |
| `brightness_min` | 3 | Night: dimmer (1-2) | Ambient glow |
| `brightness_max` | 15 | Day: brighter (20-25) | |
| `pulse_speed` | 4000 | Night: slower (5000-6000) | Breathing rhythm |
| `falloff_radius` | 80 | Wider, softer glow | |
| **Gestures** | | | |
| `gesture_chance` | 0.02 | Per passive person per second | Occasional acknowledgment |
| `gesture_duration` | 1.5s | Quick move toward edge | |
| `boredom_threshold` | 60s | Time before seeking attention | |

---

### ENGAGED Mode Parameters

| Parameter | 1 Person | 2 People | 3+ People | Notes |
|-----------|----------|----------|-----------|-------|
| **Position** | | | | |
| `follow_smoothing` | 0.05 | 0.04 | 0.03 | Lower = smoother (0-1) |
| `follow_target` | nearest | nearest | centroid | What to follow |
| `move_speed` | 40 | 50 | 60 | Faster with more people |
| **Brightness** | | | | |
| `brightness_min` | 8 | 10 | 12 | |
| `brightness_max` | 30 | 38 | 45 | |
| `pulse_speed` | 2500 | 2000 | 1500 | Faster with energy |
| `falloff_radius` | 50 | 45 | 40 | Tighter focus |
| **Dwell Rewards** | | | | |
| `dwell_bonus_start` | 10s | When to start rewarding | |
| `brightness_dwell_bonus` | +5 per 10s | Max +15 | Warmer over time |
| `pulse_variation` | ±200ms | Add subtle rhythm changes | Keeps it alive |
| **New Person** | | | | |
| `entrance_flash_duration` | 0.5s | Brief brightness bump | |
| `entrance_flash_amount` | +10 | Added to current brightness | |

---

### FLOW Mode Parameters (Passive traffic, no active)

| Parameter | Light Traffic | Heavy Traffic | Notes |
|-----------|---------------|---------------|-------|
| `flow_threshold` | 3 people/min | When to enter FLOW | |
| `drift_speed` | 15 | 25 | Lateral movement |
| `drift_direction` | flow_balance | -1 to +1 mapped to X | Follow the crowd |
| `brightness_min` | 5 | 8 | |
| `brightness_max` | 20 | 28 | |
| `pulse_speed` | 3000 | 2500 | Match walking energy |
| `gesture_chance` | 0.01 | 0.03 | More gestures when busy |

---

### Time-of-Day Modifiers

| Time | Mood | brightness_max | pulse_speed | wander_box Y | Notes |
|------|------|----------------|-------------|--------------|-------|
| 00:00-06:00 | Sleepy | ×0.4 | ×1.5 | 0-60 | Very dim, slow, low |
| 06:00-09:00 | Waking | ×0.7 | ×1.2 | 0-100 | Gradually brightening |
| 09:00-17:00 | Active | ×1.0 | ×1.0 | 0-150 | Full range |
| 17:00-20:00 | Rush | ×1.1 | ×0.9 | 0-150 | Slightly more energetic |
| 20:00-24:00 | Evening | ×0.6 | ×1.3 | 0-80 | Calming down |

---

### Trend Modifiers

| Trend Condition | Effect | Notes |
|-----------------|--------|-------|
| Busier than usual (>150% of hourly avg) | +10% brightness, -10% pulse_speed | More alive |
| Quieter than usual (<50% of hourly avg) | -10% brightness, +20% pulse_speed | More contemplative |
| Strong L→R flow (>0.7) | wander_box bias +50cm X | Drift with traffic |
| Strong R→L flow (<-0.7) | wander_box bias -50cm X | Drift with traffic |
| High avg_speed (>120 cm/s) | -10% pulse_speed | Match hurried energy |
| Low avg_speed (<60 cm/s) | +15% pulse_speed | Match leisurely pace |

---

### Mode Transition Smoothing

| Transition | Duration | Method |
|------------|----------|--------|
| IDLE → ENGAGED | 1.0s | Lerp all parameters |
| ENGAGED → IDLE | 3.0s | Slow fade, reluctant goodbye |
| ENGAGED → CROWD | 0.5s | Quick energy boost |
| CROWD → ENGAGED | 1.5s | Gradual calm |
| Any → FLOW | 2.0s | Smooth drift start |

---

### Status Text (for public display)

| Mode | Example Text |
|------|--------------|
| IDLE (quiet) | "Waiting... watching the night" |
| IDLE (bored) | "Anyone out there?" |
| IDLE (gesture) | "Oh, hello!" |
| ENGAGED (1) | "I see you" |
| ENGAGED (dwell) | "Thanks for staying" |
| CROWD | "So many friends!" |
| FLOW | "Busy evening..." |

---

## Light Behavior Database (Self-Analysis)

The database should track the light's own behavior so it can analyze patterns and avoid repetition. The goal is **evolution** - the light should learn and grow, not just react.

### Light State Recording

Record light state at regular intervals (e.g., every 0.5s when active, every 2s when idle):

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | When this state was recorded |
| `mode` | enum | IDLE, ENGAGED, CROWD, FLOW |
| `position` | x, y, z | Current light position |
| `brightness` | int | Current brightness level |
| `pulse_speed` | float | Current pulse speed |
| `target_position` | x, y, z | Where light is heading |
| `people_count` | int | People in active zone |
| `gesture_type` | string | If gesture in progress (null otherwise) |

### Behavior Pattern Analysis

Query the light's own history to detect:

| Pattern | Detection Method | Response |
|---------|------------------|----------|
| **Position repetition** | Cluster analysis of last 5 min positions | Bias toward unexplored areas |
| **Gesture spam** | Count gestures in last 2 min | Reduce `gesture_chance` temporarily |
| **Stuck in mode** | Same mode for >10 min | Introduce micro-variations |
| **Movement monotony** | Low variance in `move_speed` | Add occasional speed changes |
| **Brightness flatline** | Low variance in brightness | Widen pulse range briefly |
| **Same entry response** | Similar responses to new people | Vary entrance behavior |

### Evolution Metrics

Track over longer periods to ensure the light is "growing":

| Metric | Time Window | Goal |
|--------|-------------|------|
| `position_entropy` | 1 hour | High = good coverage of space |
| `mode_diversity` | 1 day | All modes used proportionally |
| `gesture_variety` | 1 day | Different gesture types used |
| `brightness_range_used` | 1 hour | Using full dynamic range |
| `speed_range_used` | 1 hour | Using full movement range |
| `response_similarity` | 1 day | Low = varied responses to similar inputs |

### Anti-Repetition Rules

| Rule | Description |
|------|-------------|
| **Position cooldown** | Don't return to same area within 30s |
| **Gesture cooldown** | Same gesture type can't repeat within 60s |
| **Mode memory** | After long IDLE, first engagement should be "eager" |
| **Time diversity** | Compare current hour to same hour yesterday, try something different |
| **Novelty bonus** | When doing something "new" (rare in history), hold it slightly longer |

---

## Meta Parameters (Manual Tuning)

These high-level knobs allow adjusting the overall personality without editing individual values.

### Personality Sliders (0.0 - 1.0)

| Parameter | Low (0.0) | High (1.0) | Affects |
|-----------|-----------|------------|---------|
| `responsiveness` | Slow, contemplative | Quick, reactive | `follow_smoothing`, `move_speed`, transition durations |
| `energy` | Calm, gentle | Lively, dynamic | `pulse_speed`, `brightness_max`, `gesture_chance` |
| `attention_span` | Easily distracted | Focused, loyal | `follow_smoothing`, dwell rewards, mode stickiness |
| `sociability` | Reserved | Eager to engage | `gesture_chance`, entrance flash, ENGAGED brightness |
| `exploration` | Stays in comfort zone | Wanders widely | `wander_box` size, `wander_interval`, position variety |
| `memory` | Forgets quickly | Remembers patterns | Anti-repetition strength, trend influence weight |

### Global Multipliers

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `brightness_global` | 1.0 | 0.1 - 2.0 | Scale all brightness values |
| `speed_global` | 1.0 | 0.1 - 3.0 | Scale all movement speeds |
| `pulse_global` | 1.0 | 0.3 - 3.0 | Scale all pulse speeds |
| `trend_weight` | 1.0 | 0.0 - 2.0 | How much trends influence behavior |
| `time_of_day_weight` | 1.0 | 0.0 - 2.0 | How much time affects behavior |
| `anti_repetition_weight` | 1.0 | 0.0 - 2.0 | How aggressively to avoid patterns |

### Behavior Toggles

| Toggle | Default | Description |
|--------|---------|-------------|
| `gestures_enabled` | true | Allow gesture behaviors |
| `follow_enabled` | true | Follow people in active zone |
| `flow_mode_enabled` | true | React to passive traffic patterns |
| `dwell_rewards_enabled` | true | Reward people who stay |
| `entrance_flash_enabled` | true | Acknowledge new arrivals |
| `self_analysis_enabled` | true | Use behavior database for evolution |
| `status_text_enabled` | true | Generate public status messages |

### Preset Personalities

| Preset | responsiveness | energy | attention_span | sociability | exploration | memory |
|--------|----------------|--------|----------------|-------------|-------------|--------|
| **Default** | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| **Shy** | 0.3 | 0.3 | 0.7 | 0.2 | 0.3 | 0.6 |
| **Eager** | 0.8 | 0.7 | 0.4 | 0.9 | 0.6 | 0.4 |
| **Zen** | 0.2 | 0.2 | 0.9 | 0.4 | 0.4 | 0.8 |
| **Playful** | 0.7 | 0.8 | 0.3 | 0.7 | 0.9 | 0.3 |
| **Night Owl** | 0.4 | 0.3 | 0.6 | 0.5 | 0.2 | 0.7 |

### Parameter Calculation Example

When `energy = 0.7`:
```
base_pulse_speed = 2500  (from mode table)
energy_modifier = lerp(1.3, 0.7, energy)  → 0.88
final_pulse_speed = 2500 × 0.88 = 2200ms  (faster pulse)
```

When `responsiveness = 0.3` and `attention_span = 0.8`:
```
base_follow_smoothing = 0.05
responsiveness_mod = lerp(0.02, 0.10, responsiveness) → 0.044
attention_mod = lerp(0.8, 1.2, attention_span) → 1.12
final_smoothing = 0.044 × 1.12 = 0.049  (smooth, focused)
```

---

## Notes

_Add design decisions here as we discuss..._

