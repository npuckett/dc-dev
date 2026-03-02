#!/usr/bin/env bash
# =============================================================================
# Remote Launcher - Light Controller + Pedestrian Simulator
# =============================================================================
#
# Runs both lightController_osc.py and pedestrian_simulator.py together.
# Works on a fresh machine without a pre-existing database (SQLite creates
# the tracking_history.db automatically on first run).
#
# Usage:
#   ./run_remote.sh                  # Normal pedestrian traffic
#   ./run_remote.sh --longrun        # Realistic time-of-day traffic patterns
#   ./run_remote.sh --timescale 10   # Accelerated longrun (10x speed)
#   ./run_remote.sh --no-sim         # Light controller only (no simulator)
#   ./run_remote.sh --install        # Install Python dependencies and exit
#
# Requirements:
#   Python 3.8+, pip
#   macOS or Linux (uses fcntl for single-instance lock)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$IO_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# PIDs for cleanup
SIM_PID=""
CONTROLLER_PID=""

# -----------------------------------------------------------------------------
# Cleanup on exit
# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    
    if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
        echo "  Stopping pedestrian simulator (PID $SIM_PID)..."
        kill "$SIM_PID" 2>/dev/null || true
        wait "$SIM_PID" 2>/dev/null || true
    fi
    
    if [[ -n "$CONTROLLER_PID" ]] && kill -0 "$CONTROLLER_PID" 2>/dev/null; then
        echo "  Stopping light controller (PID $CONTROLLER_PID)..."
        kill "$CONTROLLER_PID" 2>/dev/null || true
        wait "$CONTROLLER_PID" 2>/dev/null || true
    fi
    
    # Clean up lock file
    rm -f /tmp/lightController.lock 2>/dev/null || true
    
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
SIM_MODE="normal"
SIM_TIMESCALE="1.0"
SIM_HOURS="0"
SIM_DURATION="0"
NO_SIM=false
INSTALL_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --longrun)
            SIM_MODE="longrun"
            shift
            ;;
        --timescale)
            SIM_TIMESCALE="$2"
            shift 2
            ;;
        --hours)
            SIM_HOURS="$2"
            shift 2
            ;;
        --duration)
            SIM_DURATION="$2"
            shift 2
            ;;
        --no-sim)
            NO_SIM=true
            shift
            ;;
        --install)
            INSTALL_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --longrun           Realistic time-of-day traffic patterns"
            echo "  --timescale N       Time acceleration for longrun (default: 1.0)"
            echo "  --hours N           Starting hour of day for longrun (0-24)"
            echo "  --duration N        Run for N real-time hours (0=indefinite)"
            echo "  --no-sim            Run light controller only (no pedestrian simulator)"
            echo "  --install           Install Python dependencies and exit"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Find Python
# -----------------------------------------------------------------------------
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${RED}Error: Python 3 not found. Install Python 3.8+ first.${NC}"
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${CYAN}Using Python $PY_VERSION ($PYTHON)${NC}"

# Activate venv if present (check IO/../.venv, IO/.venv)
if [[ -f "$IO_DIR/../.venv/bin/activate" ]]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source "$IO_DIR/../.venv/bin/activate"
    PYTHON="python3"
elif [[ -f "$IO_DIR/.venv/bin/activate" ]]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source "$IO_DIR/.venv/bin/activate"
    PYTHON="python3"
fi

# -----------------------------------------------------------------------------
# Check / install dependencies
# -----------------------------------------------------------------------------
install_deps() {
    echo -e "${CYAN}Checking Python dependencies...${NC}"
    
    MISSING=()
    
    # Check each required package
    "$PYTHON" -c "import numpy" 2>/dev/null          || MISSING+=("numpy")
    "$PYTHON" -c "import pygame" 2>/dev/null          || MISSING+=("pygame")
    "$PYTHON" -c "import OpenGL" 2>/dev/null          || MISSING+=("PyOpenGL PyOpenGL-accelerate")
    "$PYTHON" -c "import pythonosc" 2>/dev/null       || MISSING+=("python-osc")
    
    # Optional packages (not critical)
    "$PYTHON" -c "import websockets" 2>/dev/null      || echo -e "  ${YELLOW}websockets not installed (public viewer disabled)${NC}"
    "$PYTHON" -c "import stupidArtnet" 2>/dev/null    || echo -e "  ${YELLOW}stupidArtnet not installed (visualization-only mode)${NC}"
    
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Installing missing packages: ${MISSING[*]}${NC}"
        "$PYTHON" -m pip install ${MISSING[*]}
    else
        echo -e "${GREEN}All required dependencies installed.${NC}"
    fi
}

install_deps

if [[ "$INSTALL_ONLY" == true ]]; then
    echo -e "${GREEN}Dependencies installed. Exiting.${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Verify required files exist
# -----------------------------------------------------------------------------
echo ""
echo -e "${CYAN}Checking required files...${NC}"

REQUIRED_FILES=(
    "lightController_osc.py"
    "light_behavior.py"
    "tracking_database.py"
    "pedestrian_simulator.py"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$IO_DIR/$f" ]]; then
        echo -e "  ${GREEN}✓${NC} $f"
    else
        echo -e "  ${RED}✗${NC} $f - MISSING"
        ALL_OK=false
    fi
done

# Optional files
OPTIONAL_FILES=(
    "world_coordinates.json"
    "slider_settings.json"
    "autotune_overrides.json"
)
for f in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "$IO_DIR/$f" ]]; then
        echo -e "  ${GREEN}✓${NC} $f"
    else
        echo -e "  ${YELLOW}○${NC} $f (optional, using defaults)"
    fi
done

if [[ "$ALL_OK" != true ]]; then
    echo -e "${RED}Missing required files. Cannot start.${NC}"
    exit 1
fi

# Check for database file
if [[ -f "$IO_DIR/tracking_history.db" ]]; then
    echo -e "  ${GREEN}✓${NC} tracking_history.db (existing)"
else
    echo -e "  ${YELLOW}○${NC} tracking_history.db (will be created automatically)"
fi

# Remove stale lock file from previous runs
rm -f /tmp/lightController.lock 2>/dev/null || true

# -----------------------------------------------------------------------------
# Start Pedestrian Simulator (background)
# -----------------------------------------------------------------------------
echo ""

if [[ "$NO_SIM" != true ]]; then
    SIM_ARGS=("--mode" "$SIM_MODE")
    
    if [[ "$SIM_MODE" == "longrun" ]]; then
        SIM_ARGS+=("--timescale" "$SIM_TIMESCALE")
        [[ "$SIM_HOURS" != "0" ]] && SIM_ARGS+=("--hours" "$SIM_HOURS")
        [[ "$SIM_DURATION" != "0" ]] && SIM_ARGS+=("--duration" "$SIM_DURATION")
    fi
    
    echo -e "${GREEN}Starting Pedestrian Simulator (mode: $SIM_MODE)...${NC}"
    "$PYTHON" "$IO_DIR/pedestrian_simulator.py" "${SIM_ARGS[@]}" &
    SIM_PID=$!
    echo -e "  PID: $SIM_PID"
    
    # Give simulator a moment to start and bind its OSC client
    sleep 1
    
    # Make sure it didn't crash immediately
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo -e "${RED}Pedestrian simulator failed to start!${NC}"
        SIM_PID=""
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Start Light Controller (foreground)
# -----------------------------------------------------------------------------
echo -e "${GREEN}Starting Light Controller...${NC}"
echo -e "${CYAN}Press Q or ESC in the controller window to quit both processes.${NC}"
echo ""

"$PYTHON" "$IO_DIR/lightController_osc.py" &
CONTROLLER_PID=$!

# Wait for the controller to exit (it's the main process)
wait "$CONTROLLER_PID" 2>/dev/null || true
CONTROLLER_PID=""

# cleanup runs via trap
