# macOS Development & Testing Instructions

**Context for AI assistants (GitHub Copilot, Cursor, etc.)**

This file provides development context for working on the storefront person tracker on macOS before Linux deployment.

---

## Project Overview

**Purpose:** Real-time person detection and tracking for storefront monitoring  
**Current Phase:** macOS testing and validation  
**Next Phase:** Linux production deployment  
**Environment:** macOS development → Ubuntu 22.04 LTS production

---

## Development Environment

### macOS Setup
- **OS:** macOS 11.0+ (Big Sur or newer)
- **Python:** 3.8-3.11 (system Python or pyenv)
- **Input:** Built-in webcam (device 0) for testing
- **Output:** Live video window + OSC messages
- **Goal:** Validate algorithms before RTSP camera deployment

### Virtual Environment

```bash
# Location
~/Downloads/storefront-tracker/venv/

# Activate (always required)
source venv/bin/activate

# Deactivate
deactivate
```

### Dependencies

**Core (requirements.txt):**
- `ultralytics>=8.0.0` - YOLOv8 detection/tracking
- `opencv-python>=4.8.0` - Video capture/processing
- `numpy>=1.24.0` - Array operations
- `python-osc>=1.8.0` - OSC messaging

**Platform-specific:**
```bash
# Apple Silicon (M1/M2/M3)
pip3 install torch torchvision  # Auto-selects MPS backend

# Intel Mac
pip3 install torch torchvision  # CPU only
```

---

## Code Architecture

### Main Entry Point: `tracker.py`

**Key Classes:**
```python
class PersonTracker:
    """Main tracking engine"""
    
    def __init__(self, config_file="config.json"):
        # Loads config, initializes YOLO model
        # Sets up camera, OSC client
        
    def connect_camera(self):
        # Platform-specific: macOS uses CAP_FFMPEG
        # Returns cv2.VideoCapture object
        
    def process_frame(self, frame):
        # YOLO detection → tracking → visualization
        # Returns annotated frame + detection data
        
    def send_osc(self, detections):
        # Streams tracking data via OSC
        
    def run(self):
        # Main loop: capture → process → display → repeat
```

**Platform Detection:**
```python
self.system = platform.system()  # "Darwin" for macOS
if self.system == "Darwin":
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # macOS-specific
```

### Configuration System: `config.json`

**Active config (runtime):**
- Loaded at startup by `tracker.py`
- JSON format for easy editing
- All parameters documented with comments

**Templates:**
- `config.macos-test.json` - Webcam testing (copy to config.json)
- `config.linux-production.json` - RTSP production

**Key parameters for macOS:**
```json
{
  "camera": {
    "url": 0,                    // Webcam device index
    "device": "mps"              // Apple GPU or "cpu"
  },
  "model": {
    "confidence": 0.5            // Detection threshold
  }
}
```

---

## macOS-Specific Behaviors

### Camera Access

**First run triggers permission dialog:**
```
"Terminal" would like to access the camera.
[Don't Allow] [OK]
```

**If denied:**
```bash
# System Preferences → Security & Privacy → Camera
# Check Terminal.app
```

**Camera indices:**
- `0` - Built-in FaceTime camera (default)
- `1` - External USB camera (if connected)

### GPU Acceleration

**Apple Silicon (M1/M2/M3):**
```python
device = "mps"  # Metal Performance Shaders
# ~2x faster than CPU
# 20-30 fps typical
```

**Intel Mac:**
```python
device = "cpu"
# No GPU acceleration available
# 10-20 fps typical
```

**Auto-detection not implemented** - must set manually in config

### OpenCV GUI

**Works differently than Linux:**
- Window management is macOS native
- May require opencv-contrib-python for full GUI
- Some keyboard shortcuts behave differently

**If window doesn't appear:**
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

---

## Testing Workflow

### Standard Dev Cycle

**Terminal 1: Main tracker**
```bash
cd ~/Downloads/storefront-tracker
source venv/bin/activate
python tracker.py
# Edit code in VSCode
# Stop with Ctrl+C or 'q' key
# Restart to test changes
```

**Terminal 2: OSC monitoring (optional)**
```bash
cd ~/Downloads/storefront-tracker
source venv/bin/activate
python test_osc_receiver.py
# Shows live OSC messages
# Validates data output
```

**VSCode:**
- Edit tracker.py, config.json
- Use GitHub Copilot for suggestions
- Reference this file for context

### Quick Test Loop

```bash
# Make code change
nano tracker.py

# Test immediately
python tracker.py
# Quick visual check
# Press 'q' to quit

# Iterate
```

---

## Common Development Tasks

### Change Detection Confidence

```python
# In config.json
"confidence": 0.45  // Lower = more detections, more false positives
"confidence": 0.65  // Higher = fewer detections, fewer false positives

# Optimal range for webcam: 0.45-0.55
```

### Add New Configuration Parameter

**Step 1: Add to config.json**
```json
{
  "new_feature": {
    "enabled": true,
    "parameter": 10
  }
}
```

**Step 2: Load in tracker.py __init__**
```python
self.feature_enabled = self.config["new_feature"]["enabled"]
self.feature_param = self.config["new_feature"]["parameter"]
```

**Step 3: Use in code**
```python
if self.feature_enabled:
    # Do something with self.feature_param
    pass
```

**Convention:** Always add comments in config.json explaining parameters

### Debug Detection Issues

```python
# Add to process_frame() after YOLO detection:
print(f"Detected {len(boxes)} people")
for box in boxes:
    confidence = results[0].boxes.conf[i]
    print(f"  Confidence: {confidence:.2f}")
```

### Test Different YOLO Models

```bash
# Edit config.json
"model": { "name": "yolov8n.pt" }  // Fastest (default)
"model": { "name": "yolov8s.pt" }  // Slower, more accurate
"model": { "name": "yolov8m.pt" }  // Slowest, best accuracy

# First run downloads model (~6-50MB)
# Compare FPS and detection quality
```

---

## Validation Criteria

### Minimum Acceptable Performance (macOS)

**Apple Silicon:**
- FPS: >15 with 1 person, >10 with 3 people
- Detection rate: >80% in good lighting
- Track ID persistence: IDs stable with slow movement

**Intel Mac:**
- FPS: >10 with 1 person, >7 with 3 people  
- Detection rate: >75% in good lighting
- Track ID persistence: IDs mostly stable

**Both:**
- No crashes during 30 min test
- OSC messages transmit correctly
- Clean quit with 'q' key

### What Validates on macOS

✅ **Algorithm correctness:**
- YOLO detection logic
- Track ID assignment
- Centroid/foot calculation
- Path trail rendering
- OSC message format

✅ **Configuration system:**
- JSON loading/parsing
- Parameter application
- Cross-platform device selection

✅ **Performance baseline:**
- FPS expectations
- Multi-person tracking
- CPU/GPU utilization

❌ **Does NOT validate (Linux-specific):**
- RTSP camera connection
- systemd service operation
- Headless operation
- CUDA GPU performance
- Remote access

---

## Code Patterns & Conventions

### Error Handling

```python
# Camera operations - always handle failures
cap = self.connect_camera()
if cap is None:
    print("ERROR: Could not open camera")
    return  # Fail gracefully

# Frame reading - reconnect on failure
ret, frame = cap.read()
if not ret:
    print("Failed to read frame, reconnecting...")
    cap.release()
    cap = self.connect_camera()
```

### Cross-Platform Code

```python
# Use platform detection for OS-specific behavior
import platform

if platform.system() == "Darwin":  # macOS
    # macOS-specific code
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
else:  # Linux
    # Linux-specific code
    cap = cv2.VideoCapture(url)
```

### Configuration Comments

```json
// Always explain non-obvious parameters
{
  "min_detection_y": 50,
  "comment_detection": "Filters detections above Y=50px (removes distant/small)"
}
```

### Print Statements

```python
# Informative startup messages
print(f"Running on: {self.system}")
print(f"Camera mounting: {self.mounting_height_cm}cm height")

# Progress indicators
if frame_count % 100 == 0:
    print(f"Frames: {frame_count}, FPS: {self.fps:.2f}")
```

---

## OSC Message Format

### Messages Sent (Every Frame)

```python
# Count
/tracking/count → [int]  # Number of active tracks

# Per-person (per track ID)
/person/{id}/x → [int]           # Pixel X (0 to frame_width)
/person/{id}/y → [int]           # Pixel Y (0 to frame_height)
/person/{id}/norm_x → [float]    # Normalized X (0.0-1.0)
/person/{id}/norm_y → [float]    # Normalized Y (0.0-1.0)
/person/{id}/time → [float]      # Seconds in frame
/person/{id}/ended → [1.0]       # Sent once when track ends

# Bundled data
/person/data → [id, x, y, norm_x, norm_y, time]  # Array
```

### Testing OSC Output

```bash
# Run receiver in separate terminal
python test_osc_receiver.py

# Expected output when person in view:
📊 People tracked: 1
👤 Person 1: position updated
   Normalized position: 0.523
   Time in frame: 2.3s
📦 Bundled data - ID:1 Pos:(320,240) Norm:(0.50,0.50) Time:2.3s
```

---

## Performance Optimization

### For Faster FPS on macOS

**1. Lower confidence (less processing):**
```json
"confidence": 0.6  // Faster but may miss some people
```

**2. Process fewer frames:**
```python
# Add to run() method
if frame_count % 2 != 0:
    continue  // Process every 2nd frame
```

**3. Try CPU instead of MPS:**
```json
"device": "cpu"  // Sometimes faster on older M1
```

**4. Use nano model:**
```json
"model": { "name": "yolov8n.pt" }  // Fastest model
```

**5. Disable display features:**
```python
self.show_paths = False  # No trail rendering
self.show_info = False   # No text overlays
```

---

## Debugging Techniques

### Enable Verbose YOLO Output

```python
# In tracker.py, process_frame()
results = self.model.track(
    frame,
    persist=True,
    classes=[0],
    conf=self.confidence,
    device=self.device,
    verbose=True  # Change from False to True
)
# Shows detection details in terminal
```

### Log Detection Data

```python
# Add to process_frame() after detections loop
with open('detection_log.txt', 'a') as f:
    f.write(f"{time.time()}: {len(detections)} people\n")
```

### Visualize Detection Zones

```python
# Add to draw_info() method
if self.min_detection_y:
    cv2.line(frame, (0, self.min_detection_y), 
             (frame.shape[1], self.min_detection_y), 
             (0, 255, 255), 2)  # Yellow line
```

### Monitor Performance

```python
# Add detailed timing
import time

loop_start = time.time()
# ... processing ...
loop_time = time.time() - loop_start
print(f"Loop: {loop_time*1000:.1f}ms, "
      f"FPS: {1/loop_time:.1f}")
```

---

## Known macOS Limitations

### Webcam vs RTSP Differences

| Feature | Webcam (macOS) | RTSP (Linux) |
|---------|---------------|--------------|
| Resolution | 720p-1080p | 480p-2160p (configurable) |
| Frame rate | 30 fps max | 15-30 fps (network dependent) |
| Latency | <50ms | 100-500ms |
| Connection | USB, stable | Network, can drop |
| Angle | Fixed | Adjustable mount |

**Implications:**
- Webcam is more stable for testing
- RTSP will need reconnect logic (already implemented)
- Performance on webcam ≠ performance on RTSP

### What Won't Work on macOS

**CUDA acceleration:**
```python
"device": "cuda"  # Will fail on macOS
# Solution: Use "mps" or "cpu"
```

**Headless operation:**
```python
self.show_video = False  # Window still required on macOS
# Linux can run truly headless
```

**systemd service:**
- No systemd on macOS
- Use LaunchAgent for persistent (not needed for testing)

---

## Transition to Linux

### Settings That Transfer

**Directly transferable:**
```json
{
  "model": { "name": "yolov8n.pt", "confidence": 0.45 },
  "tracking": { "history_length": 30 },
  "osc": { "port": 8000 }
}
```

**Needs adjustment:**
```json
// macOS
"camera": { "url": 0 }
"model": { "device": "mps" }
"display": { "show_video": true }

// Linux
"camera": { "url": "rtsp://..." }
"model": { "device": "cuda" }
"display": { "show_video": false }
```

### Additional Linux Requirements

**Camera mounting:**
- Physical installation at 60cm height
- 10° upward tilt measurement
- Detection zone calibration

**Network:**
- RTSP camera IP configuration
- Firewall rules for OSC if needed
- SSH/RDP access setup

**Service:**
- systemd unit file
- Auto-restart on crash
- Log management

---

## File Organization

```
storefront-tracker/
├── tracker.py                      # Main code (cross-platform)
├── config.json                     # Active config (edit this)
├── config.macos-test.json         # macOS template
├── config.linux-production.json   # Linux template
├── test_osc_receiver.py           # OSC validation
├── test_setup.py                  # Installation check
├── requirements.txt               # Dependencies
│
├── MACOS_QUICKSTART.md           # 5-min setup (start here)
├── MACOS_TESTING.md              # Comprehensive testing
├── LINUX_INSTALL.md              # Ubuntu setup
├── CAMERA_MOUNTING.md            # Physical installation
├── README.md                     # Full documentation
└── instructions.md               # This file (Copilot context)
```

---

## Critical Reminders for AI Assistants

**When suggesting code changes:**
1. Maintain config.json driven design (no hardcoded values)
2. Keep cross-platform compatibility (check platform.system())
3. Preserve existing error handling patterns
4. Add informative print statements
5. Update config templates if adding parameters

**When debugging:**
1. Check virtual environment is activated
2. Verify config.json is valid JSON
3. Confirm webcam permissions granted
4. Test with different confidence values first
5. Try CPU device if MPS fails

**Code conventions:**
- Configuration loaded in `__init__`
- Camera operations have reconnect logic
- Display operations check `self.show_video`
- OSC sends check `self.enable_osc`
- All errors print informative messages

---

## Success Metrics

**macOS testing complete when:**
- ✅ 30+ minutes continuous operation
- ✅ Detection rate >80% in good lighting
- ✅ Track IDs stable during slow movement
- ✅ FPS >10 consistently
- ✅ OSC messages validated
- ✅ Configuration changes work as expected
- ✅ Can explain detection behavior to user

**Ready for Linux deployment when:**
- ✅ All macOS tests pass
- ✅ Optimal confidence value determined
- ✅ OSC integration validated
- ✅ Performance characteristics understood
- ✅ User comfortable with system behavior

---

## Quick Reference Commands

```bash
# Start fresh
cd ~/Downloads/storefront-tracker
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip3 install torch torchvision

# Reset config
cp config.macos-test.json config.json

# Run tracker
python tracker.py

# Run OSC receiver (separate terminal)
python test_osc_receiver.py

# Verify installation
python test_setup.py

# Edit config
nano config.json  # or use VSCode
```

---

This file serves as comprehensive context for AI coding assistants when working on this project in VSCode or similar IDEs on macOS.
