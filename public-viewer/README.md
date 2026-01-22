# Drop Ceiling - Public Viewer

A mobile-first Three.js visualization that displays the real-time state of the Drop Ceiling installation.

## How It Works

The Python `lightController_osc.py` runs a WebSocket server on port 8765 that broadcasts the current state (light position, panel brightness, tracked people, behavior mode) to connected clients.

This viewer connects to that WebSocket server and renders the scene in real-time using Three.js.

## Usage

### Local Network (Recommended)

1. Run the light controller on your machine:
   ```bash
   python lightController_osc.py
   ```

2. Note the WebSocket server address shown in the console (e.g., `ws://192.168.1.100:8765`)

3. Open `index.html` on your phone's browser (same network)

4. Enter the WebSocket URL when prompted

### GitHub Pages Hosting

The viewer can be hosted on GitHub Pages for easy access:

1. Push the `public-viewer` folder to a GitHub repository

2. Enable GitHub Pages in Settings → Pages → Deploy from branch

3. Access at `https://yourusername.github.io/your-repo/public-viewer/`

4. Enter the WebSocket URL of your local installation

**Note:** GitHub Pages serves over HTTPS, but the local WebSocket server uses `ws://` (unencrypted). Modern browsers may block mixed content. If this happens:
- Use Chrome and allow mixed content for the site
- Or serve the viewer locally from Python (see below)

### Serve Viewer Locally from Python

For the most reliable setup, you can serve the viewer files directly:

```bash
cd public-viewer
python -m http.server 8080
```

Then navigate to `http://localhost:8080` or `http://<your-ip>:8080` from your phone.

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
