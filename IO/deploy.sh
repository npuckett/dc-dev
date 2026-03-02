#!/bin/bash
# =============================================================================
# Production Deployment Script (V6+)
# 
# The service runs directly from files in IO/ (committed to git).
# This script just validates syntax and restarts services.
# All code changes should be committed to git BEFORE deploying.
#
# Usage: cd /path/to/dc-dev && bash IO/deploy.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IO_DIR="$SCRIPT_DIR"
REPO_DIR="$(dirname "$IO_DIR")"

echo "=============================================="
echo "  Production Deploy"
echo "  $(date)"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Safety: warn if working tree is dirty
# -----------------------------------------------------------------------------
cd "$REPO_DIR"
if ! git diff --quiet HEAD -- IO/; then
    echo "⚠️  WARNING: IO/ has uncommitted changes vs HEAD"
    echo "   These changes WILL be deployed but are NOT in git."
    echo ""
    git diff --stat HEAD -- IO/
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Verify V6 imports are present (prevent accidental V2 regression)
# -----------------------------------------------------------------------------
echo "🔍 Verifying V6 codebase..."
if ! grep -q "V6Dev.v6_integration" "$IO_DIR/lightController_osc.py"; then
    echo "❌ ERROR: lightController_osc.py missing V6 imports!"
    echo "   This looks like a V2 file. Refusing to deploy."
    echo "   Restore from git: git checkout HEAD -- IO/lightController_osc.py"
    exit 1
fi
echo "   ✓ V6 imports present"

# -----------------------------------------------------------------------------
# Verify Python syntax
# -----------------------------------------------------------------------------
echo ""
echo "🧪 Verifying Python syntax..."
PYTHON_CMD="${PYTHON_CMD:-python3}"

for f in lightController_osc.py light_behavior.py camera_tracker_osc.py; do
    if [[ -f "$IO_DIR/$f" ]]; then
        if $PYTHON_CMD -m py_compile "$IO_DIR/$f" 2>/dev/null; then
            echo "   ✓ $f: syntax OK"
        else
            echo "   ❌ $f: SYNTAX ERROR!"
            exit 1
        fi
    fi
done

# V6Dev modules
for f in "$IO_DIR"/V6Dev/*.py; do
    if [[ -f "$f" ]]; then
        fname=$(basename "$f")
        if $PYTHON_CMD -m py_compile "$f" 2>/dev/null; then
            echo "   ✓ V6Dev/$fname: syntax OK"
        else
            echo "   ❌ V6Dev/$fname: SYNTAX ERROR!"
            exit 1
        fi
    fi
done

# -----------------------------------------------------------------------------
# Restart services
# -----------------------------------------------------------------------------
if command -v systemctl &> /dev/null; then
    echo ""
    echo "🚀 Restarting services..."
    sudo systemctl restart camera-tracker.service 2>/dev/null || echo "   (camera-tracker restart failed)"
    sudo systemctl restart light-controller.service 2>/dev/null || echo "   (light-controller restart failed)"
    sleep 3

    echo ""
    echo "📊 Service status:"
    systemctl is-active camera-tracker.service && echo "   ✓ camera-tracker: running" || echo "   ❌ camera-tracker: not running"
    systemctl is-active light-controller.service && echo "   ✓ light-controller: running" || echo "   ❌ light-controller: not running"
else
    echo ""
    echo "⚠️  No systemctl — restart services manually"
fi

echo ""
echo "=============================================="
echo "  ✅ Deploy Complete"
echo "=============================================="
echo ""
echo "Verify: journalctl -u light-controller -n 30 --no-pager"
echo ""
