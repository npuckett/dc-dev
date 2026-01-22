# Running Long Simulations

Quick guide for running overnight/multi-day simulations on the production machine.

## Required Packages

```bash
cd ~/dc-dev  # or wherever the project is
source .venv/bin/activate

# Core dependencies
pip install pygame PyOpenGL PyOpenGL_accelerate
pip install python-osc
pip install numpy

# Optional (for full features)
pip install websockets        # Public viewer
pip install stupidArtnet      # Art-Net LED output
```

## Running the Simulation

Open two terminal windows (or use tmux/screen).

### Terminal 1: Light Controller
```bash
cd ~/dc-dev
source .venv/bin/activate
python IO/lightController_osc.py
```

### Terminal 2: Pedestrian Simulator
```bash
cd ~/dc-dev
source .venv/bin/activate

# Run for 12 hours (overnight)
python IO/pedestrian_simulator.py --mode longrun --hours 12

# Or run indefinitely until Ctrl+C
python IO/pedestrian_simulator.py --mode longrun --hours 999
```

## One-Liner (Background)

Run both in background with logs:

```bash
cd ~/dc-dev && source .venv/bin/activate

# Start light controller (logs to file)
nohup python IO/lightController_osc.py > logs/light_controller.log 2>&1 &

# Start simulator (logs to file)  
nohup python IO/pedestrian_simulator.py --mode longrun --hours 12 > logs/simulator.log 2>&1 &

# Create logs folder first if needed
mkdir -p logs
```

## Stopping

```bash
# Stop both
pkill -f "lightController_osc.py"
pkill -f "pedestrian_simulator.py"

# Or gracefully with Ctrl+C in each terminal
```

## Monitoring

```bash
# Check if running
ps aux | grep -E "(lightController|pedestrian_simulator)"

# Watch logs
tail -f logs/light_controller.log
tail -f logs/simulator.log

# Health logs appear every 5 minutes in light controller output
```

## Simulation Modes

| Mode | Description |
|------|-------------|
| `--mode scripted` | Pre-defined test scenarios (default) |
| `--mode random` | Random pedestrian generation |
| `--mode longrun` | Realistic 24-hour traffic patterns |

### Long Run Traffic Patterns

The `longrun` mode simulates realistic daily traffic:
- **Dead of night (2-5am)**: Very quiet, occasional passersby
- **Morning rush (7-9am)**: Heavy commuter traffic
- **Lunch (11am-1pm)**: Busy with curious visitors
- **Evening rush (4-7pm)**: Heaviest traffic
- **Evening leisure (7-9pm)**: Many curious visitors who stop to engage
- Random gaps (5-45s) and lulls (1-3min) throughout
