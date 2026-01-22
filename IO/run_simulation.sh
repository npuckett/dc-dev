#!/bin/bash
# Launch Light Controller and Pedestrian Simulator together

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Starting Light Controller..."
python IO/lightController_osc.py &
LIGHT_PID=$!

# Wait for OSC server to start
sleep 2

echo "Starting Pedestrian Simulator..."
python IO/pedestrian_simulator.py

# When simulator exits, kill light controller
kill $LIGHT_PID 2>/dev/null
