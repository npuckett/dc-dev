#!/usr/bin/env bash
# =============================================================================
# Launch Spatial Editor (Vite + React dev server)
# =============================================================================
#
# Usage:
#   ./run_spatial_editor.sh            # Start dev server (default port 5173)
#   ./run_spatial_editor.sh --build    # Build for production and preview
#   ./run_spatial_editor.sh --install  # Install npm dependencies and exit
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EDITOR_DIR="$(cd "$SCRIPT_DIR/../../spatial-editor" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
BUILD_MODE=false
INSTALL_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD_MODE=true
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
            echo "  --build     Build for production and serve with 'vite preview'"
            echo "  --install   Install npm dependencies and exit"
            echo "  -h, --help  Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Check for Node.js / npm
# -----------------------------------------------------------------------------
if ! command -v node &>/dev/null; then
    echo -e "${RED}Error: Node.js not found. Install Node.js 18+ first.${NC}"
    exit 1
fi

if ! command -v npm &>/dev/null; then
    echo -e "${RED}Error: npm not found.${NC}"
    exit 1
fi

NODE_VERSION=$(node -v)
echo -e "${CYAN}Using Node.js $NODE_VERSION${NC}"

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------
cd "$EDITOR_DIR"

if [[ ! -d "node_modules" ]]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
elif [[ "$INSTALL_ONLY" == true ]]; then
    echo -e "${CYAN}Reinstalling npm dependencies...${NC}"
    npm install
else
    echo -e "${GREEN}node_modules found.${NC}"
fi

if [[ "$INSTALL_ONLY" == true ]]; then
    echo -e "${GREEN}Dependencies installed. Exiting.${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
echo ""
if [[ "$BUILD_MODE" == true ]]; then
    echo -e "${GREEN}Building spatial editor for production...${NC}"
    npm run build
    echo ""
    echo -e "${GREEN}Serving production build...${NC}"
    npm run preview -- --host
else
    echo -e "${GREEN}Starting spatial editor dev server...${NC}"
    echo -e "${CYAN}Opening at http://localhost:5173${NC}"
    echo ""
    npm run dev -- --host
fi
