#!/bin/bash
# =============================================================================
# Screenshot Capture Script for Art-Net Controller Prototypes (Ubuntu/X11)
# =============================================================================
# Run this on the production Ubuntu machine where tkinter/pygame work.
# Art-Net connection errors are expected (no DMX decoder needed for screenshots).
#
# Requirements (install if missing):
#   sudo apt install xdotool imagemagick xvfb
#
# Usage:
#   cd /path/to/dc-dev
#   chmod +x UIs/capture_screenshots.sh
#   ./UIs/capture_screenshots.sh
#
# Each app launches, waits for the window to appear, screenshots it, then kills it.
# Screenshots are saved as PNGs in the UIs/ folder.
#
# If no display is available (headless server), the script auto-starts Xvfb.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UI_DIR="$SCRIPT_DIR"
XVFB_PID=""

cd "$PROJECT_DIR"

# --- Dependency check ---
check_deps() {
    local missing=()
    command -v xdotool  >/dev/null 2>&1 || missing+=("xdotool")
    command -v import   >/dev/null 2>&1 || missing+=("imagemagick")

    if [ ${#missing[@]} -gt 0 ]; then
        echo "Missing required packages: ${missing[*]}"
        echo "Install them with:"
        echo "  sudo apt install ${missing[*]}"
        exit 1
    fi
}
check_deps

# --- Virtual framebuffer for headless machines ---
setup_display() {
    if [ -z "$DISPLAY" ]; then
        echo "No DISPLAY detected — starting Xvfb virtual framebuffer..."
        if ! command -v Xvfb >/dev/null 2>&1; then
            echo "Xvfb not found. Install with: sudo apt install xvfb"
            exit 1
        fi
        Xvfb :99 -screen 0 1920x1080x24 &
        XVFB_PID=$!
        export DISPLAY=:99
        sleep 1
        echo "Xvfb running on :99 (PID $XVFB_PID)"
    fi
}
setup_display

cleanup() {
    if [ -n "$XVFB_PID" ]; then
        kill "$XVFB_PID" 2>/dev/null || true
        echo "Stopped Xvfb."
    fi
}
trap cleanup EXIT

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

PYTHON=$(which python3 || which python)
echo "Using Python: $PYTHON"
echo "DISPLAY: $DISPLAY"
echo "Saving screenshots to: $UI_DIR"
echo ""

# --- Screenshot helper: full-screen capture (most reliable on X11) ---
screenshot_fullscreen() {
    local output_path="$1"
    import -window root "$output_path"
    return $?
}

# --- Art-Net stub Python snippet (reused by helpers) ---
ARTNET_STUB='
import sys, types

class FakeArtnet:
    def __init__(self, *a, **k): pass
    def start(self): pass
    def stop(self): pass
    def set(self, *a): pass
    def blackout(self): pass

mod = types.ModuleType("stupidArtnet")
mod.StupidArtnet = FakeArtnet
sys.modules["stupidArtnet"] = mod
'

# Helper: launch app, wait, take full-screen screenshot, kill
capture_app() {
    local script_path="$1"
    local output_name="$2"
    local wait_time="${3:-4}"  # seconds to wait for window to fully render

    echo "━━━ Capturing: $output_name ━━━"
    echo "    Script: $script_path"

    if [ ! -f "$script_path" ]; then
        echo "    ✗ File not found. Skipping."
        echo ""
        return 1
    fi

    # Launch the app in background
    $PYTHON "$script_path" >/dev/null 2>&1 &
    local pid=$!

    sleep "$wait_time"

    # Check if process is still alive
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "    ⚠ App exited early. Trying with Art-Net stub..."
        $PYTHON -c "
${ARTNET_STUB}
exec(open('$script_path').read())
" >/dev/null 2>&1 &
        pid=$!
        sleep "$wait_time"

        if ! kill -0 "$pid" 2>/dev/null; then
            echo "    ✗ Could not launch. Skipping."
            echo ""
            return 1
        fi
    fi

    # Full-screen capture
    if screenshot_fullscreen "$UI_DIR/${output_name}.png"; then
        echo "    ✓ Saved: UIs/${output_name}.png"
    else
        echo "    ✗ Screenshot failed."
    fi

    # Kill the app
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 0.5
    echo ""
}

echo "╔══════════════════════════════════════════════════╗"
echo "║  Art-Net Controller UI Screenshot Capture       ║"
echo "║  Ubuntu/X11 — full-screen capture mode          ║"
echo "║  Each app will open briefly for a screenshot.   ║"
echo "║  Don't click or move windows during capture.    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# 1. artnetTest.py — Tkinter
capture_app "DMXtest/artnetTest.py" \
    "01_artnetTest" 4

# 2. simpleWaveController — Tkinter
capture_app "DEVversion/artnetTest/simpleWaveController.py" \
    "02_simpleWaveController" 4

# 3. radialGradientController — Tkinter
capture_app "DEVversion/artnetTest/radialGradientController.py" \
    "03_radialGradientController" 4

# 4. radialPulseController — Tkinter
capture_app "DEVversion/artnetTest/radialPulseController.py" \
    "04_radialPulseController" 4

# 5. vectorController — Tkinter
capture_app "DEVversion/artnetTest/vectorController.py" \
    "05_vectorController" 4

# 6. springController — CLI only, skip screenshot
echo "━━━ Skipping: springController.py (CLI only, no GUI) ━━━"
echo ""

# 7. springControllerGUI — Tkinter
capture_app "DEVversion/artnetTest/springControllerGUI.py" \
    "07_springControllerGUI" 4

# 8. springControllerGUI_v2 — Tkinter
capture_app "DEVversion/artnetTest/springControllerGUI_v2.py" \
    "08_springControllerGUI_v2" 4

# 9. pointLightController3D — PyVista (may not be installed)
echo "━━━ Capturing: pointLightController3D (PyVista) ━━━"
if $PYTHON -c "import pyvista" 2>/dev/null; then
    capture_app "DEVversion/artnetTest/pointLightController3D.py" \
        "09_pointLightController3D" 5
else
    echo "    ⚠ PyVista not installed. Skipping."
    echo ""
fi

# 10. pointLightController3D_pygame — Pygame/OpenGL
capture_app "DEVversion/artnetTest/pointLightController3D_pygame.py" \
    "10_pointLightController3D_pygame" 5

# 11. waveFieldController — Tkinter
capture_app "DEVversion/artnetTest/waveFieldController.py" \
    "11_waveFieldController" 4

# 12. V4 lightController — Pygame/OpenGL
capture_app "V4Dev/lightController_osc.py" \
    "12_lightController_V4" 6

# 13. V5 lightController — Pygame/OpenGL (current dev version)
capture_app "IO/v5Dev/lightController_osc_v5.py" \
    "13_lightController_V5" 6

echo "╔══════════════════════════════════════════════════╗"
echo "║  Done! Check UIs/ folder for screenshots.       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
ls -lh "$UI_DIR"/*.png 2>/dev/null || echo "No screenshots captured."
