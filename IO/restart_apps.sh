#!/bin/bash
# Restart tracking and light controller in the current session
# Run this from an XRDP session to see the windows

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping any running instances..."
pkill -f "camera_tracker_osc.py" 2>/dev/null
pkill -f "lightController_osc.py" 2>/dev/null
sleep 1

echo "Starting camera tracker..."
cd "$SCRIPT_DIR"
python3 camera_tracker_osc.py &

echo "Starting light controller..."
python3 lightController_osc.py &

echo ""
echo "Apps started! Windows should appear in this session."
echo "Press Ctrl+C to stop watching, or close this terminal to keep them running."
wait
