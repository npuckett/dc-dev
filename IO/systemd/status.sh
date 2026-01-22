#!/bin/bash
# Drop Ceiling - Service status dashboard
# Run with: ./status.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Drop Ceiling - Service Status                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_service() {
    local service=$1
    local status=$(systemctl is-active $service 2>/dev/null)
    local enabled=$(systemctl is-enabled $service 2>/dev/null)
    
    if [ "$status" = "active" ]; then
        echo -e "  ${GREEN}●${NC} $service: ${GREEN}running${NC} (enabled: $enabled)"
    elif [ "$status" = "inactive" ]; then
        echo -e "  ${YELLOW}○${NC} $service: ${YELLOW}stopped${NC} (enabled: $enabled)"
    else
        echo -e "  ${RED}✗${NC} $service: ${RED}$status${NC}"
    fi
}

echo "📊 Services:"
check_service "camera-tracker"
check_service "light-controller"
check_service "tailscale-funnel"
check_service "tailscaled"

echo ""
echo "📈 Resource Usage:"
echo "  Memory:"
systemctl show camera-tracker --property=MemoryCurrent 2>/dev/null | sed 's/MemoryCurrent=/    camera-tracker: /' | head -1 || echo "    camera-tracker: N/A"
systemctl show light-controller --property=MemoryCurrent 2>/dev/null | sed 's/MemoryCurrent=/    light-controller: /' | head -1 || echo "    light-controller: N/A"

echo ""
echo "🕐 Uptime:"
systemctl show camera-tracker --property=ActiveEnterTimestamp 2>/dev/null | sed 's/ActiveEnterTimestamp=/    camera-tracker started: /' | head -1 || echo "    N/A"
systemctl show light-controller --property=ActiveEnterTimestamp 2>/dev/null | sed 's/ActiveEnterTimestamp=/    light-controller started: /' | head -1 || echo "    N/A"

echo ""
echo "🌐 Network:"
echo "  Tailscale IP: $(tailscale ip -4 2>/dev/null || echo 'N/A')"
echo "  Funnel URL: https://$(tailscale status --json 2>/dev/null | grep -o '"DNSName":"[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/\.$//'  || echo 'N/A'):8765"

echo ""
echo "📋 Commands:"
echo "  View logs:    journalctl -u camera-tracker -f"
echo "  Restart all:  sudo systemctl restart camera-tracker light-controller"
echo "  Stop all:     sudo systemctl stop camera-tracker light-controller"
