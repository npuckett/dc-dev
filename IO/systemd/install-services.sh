#!/bin/bash
# Drop Ceiling - Install systemd services
# Run with: sudo ./install-services.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="/etc/systemd/system"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Drop Ceiling - Installing systemd Services            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check for root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run with sudo: sudo ./install-services.sh"
    exit 1
fi

# Copy service files
echo "📦 Installing service files..."
cp "$SCRIPT_DIR/camera-tracker.service" "$SERVICE_DIR/"
cp "$SCRIPT_DIR/light-controller.service" "$SERVICE_DIR/"
cp "$SCRIPT_DIR/tailscale-funnel.service" "$SERVICE_DIR/"

echo "   ✓ camera-tracker.service"
echo "   ✓ light-controller.service"
echo "   ✓ tailscale-funnel.service"

# Reload systemd
echo ""
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable services
echo ""
echo "🔧 Enabling services for auto-start..."
systemctl enable camera-tracker.service
systemctl enable light-controller.service
systemctl enable tailscale-funnel.service

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit service files if paths differ from /home/nick/Documents/Github/dc-dev/"
echo "   2. Start services:"
echo "      sudo systemctl start camera-tracker"
echo "      sudo systemctl start light-controller"
echo "      sudo systemctl start tailscale-funnel"
echo ""
echo "   3. Check status:"
echo "      sudo systemctl status camera-tracker light-controller"
echo ""
echo "   4. View logs:"
echo "      journalctl -u camera-tracker -f"
echo "      journalctl -u light-controller -f"
echo "════════════════════════════════════════════════════════════"
