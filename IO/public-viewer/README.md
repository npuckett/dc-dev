# Drop Ceiling - Public Viewer

A mobile-first Three.js visualization that displays the real-time state of the Drop Ceiling installation.

## How It Works

The Python `lightController_osc.py` runs a WebSocket server on port 8765 that broadcasts the current state (light position, panel brightness, tracked people, behavior mode) to connected clients.

This viewer connects to that WebSocket server and renders the scene in real-time using Three.js.

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION MACHINE                            │
│  (Linux, corporate WiFi, Tailscale connected)                       │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ camera_tracker  │───▶│ lightController │───▶│   Tailscale     │  │
│  │    _osc.py      │OSC │    _osc.py      │WS  │    Funnel       │  │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘  │
│                              Port 8765                   │           │
└──────────────────────────────────────────────────────────│───────────┘
                                                           │
                                                           ▼ HTTPS
                                          https://your-machine.ts.net:8765
                                                           │
                                                           ▼
                                          ┌──────────────────────────────┐
                                          │     GitHub Pages (Public)    │
                                          │   Three.js Viewer (WSS)     │
                                          └──────────────────────────────┘
                                                           │
                                                           ▼
                                          ┌──────────────────────────────┐
                                          │        Any Browser           │
                                          │   (Phone, Desktop, etc.)     │
                                          └──────────────────────────────┘
```

---

## Tailscale Funnel Setup (Required for GitHub Pages)

Tailscale Funnel exposes your WebSocket server to the public internet over HTTPS, allowing GitHub Pages (which requires HTTPS) to connect.

### Prerequisites

1. **Tailscale account** - [tailscale.com](https://tailscale.com)
2. **Tailscale installed** on the production machine
3. **Funnel enabled** in your Tailscale admin console

### macOS Setup (Testing)

```bash
# 1. Install Tailscale
brew install tailscale

# 2. Start Tailscale and login
sudo tailscaled
tailscale up

# 3. Check your Tailscale hostname
tailscale status
# Shows something like: your-mac.tail12345.ts.net

# 4. Enable Funnel in admin console
#    Go to: https://login.tailscale.com/admin/acls
#    Add to your ACL policy (or use the UI):
#    "nodeAttrs": [
#      {"target": ["*"], "attr": ["funnel"]}
#    ]

# 5. Expose the WebSocket port via Funnel
tailscale funnel 8765

# 6. Your WebSocket is now publicly accessible at:
#    wss://your-mac.tail12345.ts.net:8765
```

### Linux Setup (Production)

```bash
# 1. Install Tailscale (Debian/Ubuntu)
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Start Tailscale service
sudo systemctl enable --now tailscaled
sudo tailscale up

# 3. Check your Tailscale hostname
tailscale status
# Shows: production-machine.tail12345.ts.net

# 4. Enable Funnel (ensure it's enabled in admin console first)
sudo tailscale funnel 8765

# 5. Make Funnel persist across reboots
#    Add to /etc/systemd/system/tailscale-funnel.service:
```

Create systemd service for persistent Funnel:

```ini
# /etc/systemd/system/tailscale-funnel.service
[Unit]
Description=Tailscale Funnel for WebSocket
After=tailscaled.service
Requires=tailscaled.service

[Service]
Type=simple
ExecStart=/usr/bin/tailscale funnel 8765
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tailscale-funnel
```

### Verify Funnel is Working

```bash
# Check Funnel status
tailscale funnel status

# Should show:
# https://your-machine.tail12345.ts.net:8765 -> localhost:8765

# Test from any browser:
# Visit https://your-machine.tail12345.ts.net:8765
# (Will show "Upgrade Required" - that's correct, it expects WebSocket)
```

---

## Viewer Configuration

Update the WebSocket URL in the viewer to use your Funnel URL:

```javascript
// In viewer.js, update the connection URL:
const WS_URL = 'wss://your-machine.tail12345.ts.net:8765';
```

Or make it configurable via URL parameter:

```
https://yourusername.github.io/dc-dev/public-viewer/?ws=wss://your-machine.tail12345.ts.net:8765
```

---

## Quick Start (Local Testing)

1. Start Tailscale Funnel:
   ```bash
   tailscale funnel 8765
   ```

2. Run the light controller:
   ```bash
   python IO/lightController_osc.py
   ```

3. Open the viewer in a browser and enter your Funnel URL:
   ```
   wss://your-machine.tail12345.ts.net:8765
   ```

---

## GitHub Pages Hosting

1. Push the `public-viewer` folder to your GitHub repository

2. Enable GitHub Pages:
   - Settings → Pages → Deploy from branch → `main` (or your branch)
   - Set folder to `/public-viewer` or root depending on structure

3. Access at: `https://yourusername.github.io/dc-dev/public-viewer/`

4. The viewer will connect via secure WebSocket (WSS) to your Funnel URL

**Important:** GitHub Pages uses HTTPS, so you MUST use Tailscale Funnel (which provides WSS) - plain `ws://` connections will be blocked by browsers.

---

## Troubleshooting

### "WebSocket connection failed"
- Verify Funnel is running: `tailscale funnel status`
- Verify lightController is running
- Check browser console for specific error

### "Funnel not available"
- Ensure Funnel is enabled in Tailscale admin ACLs
- You may need to re-authenticate: `tailscale up`

### "Mixed content blocked"
- You're trying to use `ws://` from an HTTPS page
- Use the Funnel URL (`wss://...`) instead

### Connection drops frequently
- Check if Funnel service is stable: `journalctl -u tailscale-funnel -f`
- Verify network connectivity on production machine

---Then navigate to `http://localhost:8080` or `http://<your-ip>:8080` from your phone.

## Features

- **Real-time sync** - 60 FPS state updates via WebSocket
- **Mobile-first** - Optimized for portrait viewing on phones
- **Minimal aesthetic** - Clean dark design matching the installation
- **Status display** - Shows connection state, behavior mode, tracked people

## Architecture

```
┌─────────────────────┐     WebSocket (8765)     ┌──────────────────┐
│ lightController_osc │ ──────────────────────▶  │   Three.js       │
│     (Python)        │      JSON state          │   Viewer         │
└─────────────────────┘                          └──────────────────┘
         ▲                                                │
         │ OSC (9000)                                     │
         │                                                ▼
┌─────────────────────┐                          ┌──────────────────┐
│  Camera Tracking    │                          │   Mobile Phone   │
└─────────────────────┘                          └──────────────────┘
```

## State Format

The WebSocket broadcasts JSON messages with this structure:

```json
{
  "light": {
    "x": 0,
    "y": 45,
    "z": 200,
    "brightness": 1.0,
    "falloff_radius": 120
  },
  "panels": [255, 200, 150, ...],  // 12 panel brightness values
  "people": [
    {"id": "person_1", "x": 0, "y": 85, "z": 150}
  ],
  "mode": "ENGAGED",
  "gesture": "BLOOM",
  "status": "ENGAGED | 1 people | radius: 120 | personality: cautious"
}
```
