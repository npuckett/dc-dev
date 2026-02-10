# Light Controller V3 - Architecture Documentation

## Overview

V3 is a complete refactor of the ~8,000 line `lightController_osc.py` into a modular, maintainable architecture. The system tracks people via OSC messages from a camera tracker and controls DMX lights to respond to their presence and movement.

---

## Directory Structure

```
V3Dev/
├── __init__.py              # Package exports all modules
├── application.py           # Main Application class (orchestrator)
├── run.py                   # Entry point with CLI flags
│
├── config/                  # Configuration & Constants
│   ├── __init__.py
│   ├── zones.py             # Zone boundaries (ACTIVE_ZONE, PASSIVE_ZONE)
│   ├── hardware.py          # Panel layout, DMX addresses, network settings
│   └── timing.py            # Frame timing, transition speeds, animation rates
│
├── tracking/                # Person Tracking
│   ├── __init__.py
│   ├── person_manager.py    # TrackedPerson, PersonManager classes
│   └── osc_handler.py       # OSCHandler - receives /blob messages
│
├── behavior/                # Light Behavior Logic
│   ├── __init__.py
│   ├── modes.py             # BehaviorMode enum (IDLE, ACTIVE, AMBIENT, etc.)
│   ├── parameters.py        # BehaviorParameters - spring constants, speeds
│   ├── trends.py            # TrendAnalyzer - movement pattern detection
│   ├── states.py            # BehaviorState - per-person behavioral state
│   └── system.py            # BehaviorSystem - main behavior orchestrator
│
├── visualization/           # Light Output & Display
│   ├── __init__.py
│   ├── panels.py            # LightPanel, PanelArray - 4 units × 3 panels
│   ├── falloff.py           # FalloffCalculator - distance-based brightness
│   ├── renderer.py          # Renderer - pygame visualization
│   ├── dmx.py               # DMXOutput - Art-Net packet transmission
│   └── debug.py             # DebugOverlay - on-screen debug info
│
├── network/                 # Network & Persistence
│   ├── __init__.py
│   ├── websocket.py         # WebSocketBroadcaster - real-time state streaming
│   ├── health.py            # HealthMonitor - uptime, FPS, system stats
│   └── persistence.py       # SettingsStore - save/load slider settings
│
└── tests/
    ├── test_behavior.py     # Behavior module tests
    ├── test_visualization.py # Visualization module tests
    └── test_network.py      # Network module tests
```

---

## How It Works

### Data Flow

```
Camera Tracker (external)
        │
        ▼ OSC /blob messages (UDP port 7777)
┌───────────────────┐
│   OSCHandler      │  ← Receives person coordinates
└───────────────────┘
        │
        ▼ Updates person positions
┌───────────────────┐
│   PersonManager   │  ← Tracks active/stale persons, calculates velocities
└───────────────────┘
        │
        ▼ Person data
┌───────────────────┐
│  BehaviorSystem   │  ← Determines mode, applies spring physics, trends
└───────────────────┘
        │
        ▼ Light position target
┌───────────────────┐
│   FalloffCalculator│  ← Computes per-panel brightness from distance
└───────────────────┘
        │
        ▼ Panel brightness values
┌───────────────────┐
│    PanelArray     │  ← Manages 12 panels (4 units × 3 panels)
└───────────────────┘
        │
        ├──────────────────────┐
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│   DMXOutput   │      │   Renderer    │
│  (Art-Net)    │      │  (pygame)     │
└───────────────┘      └───────────────┘
        │                      │
        ▼                      ▼
   Physical Lights       Debug Window
```

### Main Loop (60 FPS target)

Each frame:
1. **OSC Receive** - Process incoming person tracking messages
2. **Update Tracking** - Remove stale persons, calculate velocities
3. **Update Behavior** - Determine mode, animate light position
4. **Calculate Falloff** - Distance-based brightness per panel
5. **Output DMX** - Send Art-Net packets to lights
6. **Render** - Update pygame visualization
7. **Broadcast** - Send state to WebSocket clients
8. **Handle Events** - Keyboard input, window events

---

## Module Details

### config/

**zones.py**
- `ACTIVE_ZONE`: Rectangle where person presence triggers active tracking
  - Bounds: x=[-280, -20], z=[78, 283] cm
- `PASSIVE_ZONE`: Extended area for ambient awareness
  - Bounds: x=[-350, 50], z=[283, 553] cm

**hardware.py**
- `PANEL_CONFIG`: 4 units, 3 panels each, 80cm unit spacing, 60cm panel size
- `DMX_CONFIG`: Art-Net target `10.42.0.200`, universe 0, 12 channels
- `NETWORK_CONFIG`: OSC port 7777, WebSocket port 8765

**timing.py**
- `FRAME_TIMING`: Target 60 FPS (16.67ms frame time)
- `TRANSITIONS`: Mode change durations, fade speeds
- `ANIMATION`: Spring rates, easing curves

---

### tracking/

**person_manager.py**
- `TrackedPerson`: Stores position, velocity, last_seen time
- `PersonManager`: 
  - Maintains dict of active persons by ID
  - Removes stale persons (not seen for 2+ seconds)
  - Selects "dominant" person (closest to lights)
  - Calculates smoothed velocities

**osc_handler.py**
- `OSCHandler`: Binds to UDP port 7777
- Parses `/blob` messages: `[person_id, x, y, z]`
- Updates PersonManager with new positions

---

### behavior/

**modes.py**
- `BehaviorMode` enum: IDLE, ACTIVE, FOLLOWING, AMBIENT, PULSE, WAVE

**parameters.py**
- `BehaviorParameters`: Runtime-adjustable settings
  - Spring stiffness, damping
  - Brightness multipliers
  - Animation speeds
  - Accessed via sliders in UI

**trends.py**
- `TrendAnalyzer`: Detects movement patterns
  - Stationary vs moving
  - Approach vs retreat
  - Pacing/oscillating
  - Used to trigger mode transitions

**states.py**
- `BehaviorState`: Per-person state machine
  - Current mode, transition progress
  - Animation timers
  - Engagement level

**system.py**
- `BehaviorSystem`: Main orchestrator
  - Maintains light position (spring-based following)
  - Selects active person
  - Coordinates mode transitions
  - Exposes `get_light_position()` for rendering

---

### visualization/

**panels.py**
- `LightPanel`: Single panel with position, brightness, color
- `PanelArray`: Collection of 12 panels in 4×3 grid
  - Methods: `set_all()`, `set_panel()`, `get_dmx_values()`

**falloff.py**
- `FalloffCalculator`: Distance → brightness mapping
  - Configurable curve (linear, quadratic, exponential)
  - Near/far thresholds
  - Returns 0.0-1.0 brightness per panel

**renderer.py**
- `Renderer`: pygame-based visualization
  - Draws zones, panels, persons, light position
  - Slider controls for behavior parameters
  - Keyboard shortcuts

**dmx.py**
- `DMXOutput`: Art-Net transmission
  - Constructs Art-Net packets
  - Sends to configured IP/universe
  - Rate limiting to prevent flooding

**debug.py**
- `DebugOverlay`: On-screen debug info
  - FPS, person count, current mode
  - Light position, dominant person
  - Network status

---

### network/

**websocket.py**
- `WebSocketBroadcaster`: Async WebSocket server on port 8765
- `StateSerializer`: Converts system state to JSON
  - Caches to avoid redundant serialization
- Broadcasts: persons, light position, mode, panel states

**health.py**
- `HealthMonitor`: System health metrics
  - Uptime tracking
  - FPS averaging
  - Memory usage
  - Last error tracking

**persistence.py**
- `SettingsStore`: Save/load slider settings
  - JSON file: `slider_settings_v2.json`
  - Rate-limited auto-save (every 5 seconds max)
  - Graceful fallback to defaults

---

### application.py

`Application` class - main orchestrator:
- Initializes all modules
- Runs main loop
- Handles graceful shutdown
- **Single instance lock**: Uses `fcntl.flock()` on `/tmp/lightController_v3.lock`
  - Prevents duplicate processes (important for systemd)

---

### run.py

Entry point with CLI flags:

```bash
python3 V3Dev/run.py [options]
```

---

## Important Commands

### Running the Controller

```bash
# Normal operation (with pygame window)
cd /path/to/dc-dev/IO
python3 V3Dev/run.py

# Headless mode (no pygame window, for servers)
python3 V3Dev/run.py --headless

# Run integration tests
python3 V3Dev/run.py --test

# Skip single-instance lock (for debugging)
python3 V3Dev/run.py --no-lock

# Timed run (useful for testing)
python3 V3Dev/run.py --headless --duration 30
```

### Running Module Tests

```bash
# All tests
python3 -m pytest V3Dev/tests/

# Individual test files
python3 V3Dev/tests/test_behavior.py
python3 V3Dev/tests/test_visualization.py
python3 V3Dev/tests/test_network.py
```

### Systemd Service (Production)

```bash
# Install service
sudo cp IO/systemd/light-controller.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable light-controller
sudo systemctl start light-controller

# Check status
sudo systemctl status light-controller
journalctl -u light-controller -f  # Live logs

# Restart after code changes
sudo systemctl restart light-controller
```

### WebSocket Testing

```bash
# Test WebSocket connection (requires websocat)
websocat ws://localhost:8765

# Or use the public viewer
open IO/public-viewer/index.html
```

### OSC Testing

```bash
# Send test blob (requires oscsend)
oscsend localhost 7777 /blob iifff 1 100 0 -150 200
#                              │  │  │   │    │
#                              │  │  │   │    └─ z position (cm)
#                              │  │  │   └────── x position (cm)  
#                              │  │  └────────── y (unused)
#                              │  └───────────── frame number
#                              └──────────────── person ID
```

---

## Keyboard Shortcuts (pygame window)

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `D` | Toggle debug overlay |
| `R` | Reset behavior parameters to defaults |
| `S` | Save current slider settings |
| `SPACE` | Toggle pause |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `slider_settings_v2.json` | Saved behavior parameters |
| `tracker_settings.json` | Camera tracker configuration |
| `camera_calibration.json` | Camera calibration data |
| `world_coordinates.json` | World coordinate system |

---

## Network Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 7777 | UDP | OSC input (from camera tracker) |
| 8765 | TCP | WebSocket output (to viewers) |
| 6454 | UDP | Art-Net output (to DMX lights) |

---

## Zone Coordinate System

```
         Z (depth from lights)
         ▲
         │
    553 ─┼─────────────────────────┐
         │                         │
         │     PASSIVE_ZONE        │
         │                         │
    283 ─┼─────────────────────────┤
         │                         │
         │     ACTIVE_ZONE         │
         │   (triggers tracking)   │
         │                         │
     78 ─┼─────────────────────────┘
         │
         └────────────────────────────▶ X (left-right)
          -350  -280  -150  -20   50

Units: centimeters
Origin: Light panel center
```

---

## Comparison: V2 vs V3

| Aspect | V2 (monolithic) | V3 (modular) |
|--------|-----------------|--------------|
| Lines of code | ~8,000 in one file | ~2,500 across 20 files |
| Testing | Difficult | Unit tests per module |
| Modification | High risk | Isolated changes |
| Understanding | Read 8K lines | Read relevant module |
| Dependencies | All loaded always | Import what you need |
| Single instance | Manual | Built-in lock file |

---

## Future Improvements

- [ ] Add config file for zone boundaries (currently hardcoded)
- [ ] Implement mode presets (save/load behavior profiles)
- [ ] Add OSC output for external integrations
- [ ] Create web-based control panel (beyond viewer)
- [ ] Add multi-camera support
- [ ] Implement behavior recording/playback
