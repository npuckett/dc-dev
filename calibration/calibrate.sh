#!/bin/bash
# =============================================================================
# Calibration Mode Script
# =============================================================================
# Stops production tracker, runs calibration, then restarts production.
#
# Usage:
#   ./calibrate.sh
#
# In calibration mode:
#   - Press C to enter calibration view
#   - Press A to auto-calibrate (recommended)
#   - Press Q to quit and restart production
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  DC-DEV Calibration Mode"
echo "=========================================="

# Check if running as root (needed for systemctl)
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo for systemctl commands"
    echo ""
fi

# Stop production tracker
echo ""
echo "🛑 Stopping production camera tracker..."
sudo systemctl stop camera-tracker 2>/dev/null || echo "   (camera-tracker not running)"

echo ""
echo "🎯 Starting calibration mode..."
echo "   Press C for calibration view"
echo "   Press A for auto-calibrate"
echo "   Press Q when done"
echo ""

# Activate venv and run calibration tracker
cd "$SCRIPT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
python camera_tracker_cuda.py

# Restart production tracker
echo ""
echo "🚀 Restarting production camera tracker..."
sudo systemctl start camera-tracker

echo ""
echo "✅ Done! Production tracker is running again."
echo "   Check status: sudo systemctl status camera-tracker"
