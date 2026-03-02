#!/bin/bash
# =============================================================================
# V2 Deployment Script
# Moves V2 files to production locations and fixes import paths
# 
# Usage: cd /path/to/dc-dev/IO && ./deploy_v2.sh
# =============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IO_DIR="$SCRIPT_DIR"
V2_DIR="$IO_DIR/V2Dev"

echo "=============================================="
echo "  V2 Deployment Script"
echo "  $(date)"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "🔍 Running pre-flight checks..."

# Check we're in the right place
if [[ ! -d "$V2_DIR" ]]; then
    echo "❌ ERROR: V2Dev directory not found at $V2_DIR"
    exit 1
fi

# Check required V2 files exist
REQUIRED_FILES=(
    "$V2_DIR/camera_tracker_osc_v2.py"
    "$V2_DIR/lightController_v2.py"
    "$V2_DIR/light_behavior_v2.py"
    "$V2_DIR/pedestrian_simulator_v2.py"
    "$V2_DIR/camera_calibration_v2.py"
    "$V2_DIR/world_coordinates.json"
    "$V2_DIR/slider_settings_v2.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ ERROR: Required file not found: $f"
        exit 1
    fi
done
echo "   ✓ All required V2 files present"

# -----------------------------------------------------------------------------
# Stop services (if running on Linux with systemd)
# -----------------------------------------------------------------------------
if command -v systemctl &> /dev/null; then
    echo ""
    echo "🛑 Stopping services..."
    sudo systemctl stop camera-tracker.service 2>/dev/null || echo "   (camera-tracker not running)"
    sudo systemctl stop light-controller.service 2>/dev/null || echo "   (light-controller not running)"
    sleep 2
else
    echo ""
    echo "⚠️  No systemctl found - make sure processes are stopped manually"
    echo "   Press Enter to continue or Ctrl+C to abort..."
    read
fi

# -----------------------------------------------------------------------------
# Backup existing V1 files (if they exist)
# -----------------------------------------------------------------------------
echo ""
echo "📦 Backing up V1 files..."

BACKUP_DIR="$IO_DIR/v1_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

V1_FILES=(
    "camera_tracker_osc.py"
    "lightController_osc.py"
    "light_behavior.py"
    "pedestrian_simulator.py"
)

for f in "${V1_FILES[@]}"; do
    if [[ -f "$IO_DIR/$f" ]]; then
        cp "$IO_DIR/$f" "$BACKUP_DIR/"
        echo "   ✓ Backed up $f"
    fi
done
echo "   Backup location: $BACKUP_DIR"

# -----------------------------------------------------------------------------
# Move and rename V2 files
# -----------------------------------------------------------------------------
echo ""
echo "📁 Moving V2 files to production..."

# Core Python files (overwrite V1)
cp "$V2_DIR/camera_tracker_osc_v2.py" "$IO_DIR/camera_tracker_osc.py"
echo "   ✓ camera_tracker_osc_v2.py → camera_tracker_osc.py"

cp "$V2_DIR/lightController_v2.py" "$IO_DIR/lightController_osc.py"
echo "   ✓ lightController_v2.py → lightController_osc.py"

cp "$V2_DIR/light_behavior_v2.py" "$IO_DIR/light_behavior.py"
echo "   ✓ light_behavior_v2.py → light_behavior.py"

cp "$V2_DIR/pedestrian_simulator_v2.py" "$IO_DIR/pedestrian_simulator.py"
echo "   ✓ pedestrian_simulator_v2.py → pedestrian_simulator.py"

cp "$V2_DIR/camera_calibration_v2.py" "$IO_DIR/camera_calibration.py"
echo "   ✓ camera_calibration_v2.py → camera_calibration.py"

# JSON config files
cp "$V2_DIR/world_coordinates.json" "$IO_DIR/world_coordinates.json"
echo "   ✓ world_coordinates.json"

cp "$V2_DIR/slider_settings_v2.json" "$IO_DIR/slider_settings.json"
echo "   ✓ slider_settings_v2.json → slider_settings.json"

# -----------------------------------------------------------------------------
# Fix import paths in moved files
# -----------------------------------------------------------------------------
echo ""
echo "🔧 Fixing import paths..."

# lightController_osc.py: change "from light_behavior_v2" to "from light_behavior"
sed -i.bak 's/from light_behavior_v2 import/from light_behavior import/g' "$IO_DIR/lightController_osc.py"
rm -f "$IO_DIR/lightController_osc.py.bak"
echo "   ✓ lightController_osc.py: light_behavior_v2 → light_behavior"

# Check for any remaining _v2 imports that need fixing
if grep -q "_v2" "$IO_DIR/lightController_osc.py" 2>/dev/null; then
    echo "   ⚠️  Warning: lightController_osc.py still contains '_v2' references"
fi

if grep -q "_v2" "$IO_DIR/camera_tracker_osc.py" 2>/dev/null; then
    echo "   ⚠️  Warning: camera_tracker_osc.py still contains '_v2' references"
fi

# -----------------------------------------------------------------------------
# Verify syntax of moved files
# -----------------------------------------------------------------------------
echo ""
echo "🧪 Verifying Python syntax..."

PYTHON_CMD="${PYTHON_CMD:-python3}"

for f in camera_tracker_osc.py lightController_osc.py light_behavior.py pedestrian_simulator.py; do
    if $PYTHON_CMD -m py_compile "$IO_DIR/$f" 2>/dev/null; then
        echo "   ✓ $f: syntax OK"
    else
        echo "   ❌ $f: SYNTAX ERROR!"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Restart services (if on Linux with systemd)
# -----------------------------------------------------------------------------
if command -v systemctl &> /dev/null; then
    echo ""
    echo "🚀 Restarting services..."
    sudo systemctl start camera-tracker.service
    sudo systemctl start light-controller.service
    sleep 2
    
    echo ""
    echo "📊 Service status:"
    systemctl is-active camera-tracker.service && echo "   ✓ camera-tracker: running" || echo "   ❌ camera-tracker: not running"
    systemctl is-active light-controller.service && echo "   ✓ light-controller: running" || echo "   ❌ light-controller: not running"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  ✅ V2 Deployment Complete!"
echo "=============================================="
echo ""
echo "Files deployed to: $IO_DIR"
echo "V1 backup at: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify tracking: tail -f /var/log/syslog | grep -E 'camera-tracker|light-controller'"
echo "  2. Check Art-Net output on 10.42.0.200"
echo "  3. Test with pedestrian_simulator.py if needed"
echo ""
