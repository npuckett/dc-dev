# Storefront Person Tracker - Proof of Concept Instructions

**Context for AI assistants and developers working on this project**

## Project Overview

### Vision
Build a real-time person detection and tracking system for storefront monitoring that runs 24/7 unattended on Linux, using RTSP security cameras and YOLOv8 computer vision.

### Current Phase: Proof of Concept Testing
We are validating the complete system with a single camera before scaling to multi-camera production deployment.

**Goals for PoC:**
1. ✅ Validate YOLOv8 detection accuracy in real storefront conditions
2. ✅ Confirm RTSP camera integration works reliably
3. ✅ Test low-angle mounting (21-24" windowsill height)
4. ✅ Verify 24/7 stability over 10-day unattended operation
5. ✅ Optimize configuration parameters for production
6. ✅ Validate OSC data output for downstream applications

**What success looks like:**
- Camera tracking people reliably for 10 days straight
- >85% detection rate in target zone (0-8 meters from window)
- Stable track IDs during pedestrian movement
- System auto-recovers from camera/network interruptions
- Performance acceptable (15+ fps with GTX 1660 Ti)

---

## System Architecture

### Hardware Configuration

**Production Computer: ABS Mage E Gaming Desktop**
- CPU: Intel Core i5-9400F (6-core, 2.9 GHz, 9th gen)
- GPU: NVIDIA GeForce GTX 1660 Ti (1536 CUDA cores, 6GB GDDR6)
- RAM: 16GB DDR4
- Storage: 512GB SSD
- OS: Ubuntu 22.04 LTS
- Expected performance: 25-30 fps single camera, 20-25 fps with 3 cameras

**Camera: Reolink RLC-520A (PoE Dome)**
- Resolution: 5MP (2560x1920) main, 640x480 sub-stream
- Frame rate: 30 fps
- Protocol: RTSP (H.264/H.265)
- Power: PoE 802.3af (no separate power needed)
- Mounting: Low windowsill (21-24" / 53-61cm height)
- Angle: 10° upward tilt from horizontal
- View: Indoor camera looking OUT through window at sidewalk/street

**Network: Simplified PoE Router Topology**
```
Internet
   ↓
PoE Router (user's existing equipment)
   ├─→ Linux Computer (ethernet)
   └─→ Camera (PoE - power + data)
```

**Critical advantage:** Computer and camera on same router = guaranteed same subnet, no network complexity.

### Camera Configuration Details

**Physical Setup:**
- Location: Lower windowsill inside storefront
- Height from ground: 60cm (±5cm variance acceptable)
- Tilt angle: 10° upward from horizontal
- Direction: Looking out through window glass
- Coverage: Sidewalk/street pedestrian traffic (0-8 meters effective range)

**Camera Settings (Critical):**
```
Sub-stream (used by tracker):
- Resolution: 640x480 VGA
- Frame rate: 30 fps
- Bitrate: 1024 Kbps
- Encoding: H.264

Main stream (for recording/viewing):
- Resolution: 1920x1080 or 2560x1920
- Frame rate: 30 fps
- Bitrate: 4096 Kbps

IR LEDs: "Stay Off" ⚠️ CRITICAL
- Indoor-to-outdoor through glass
- IR causes reflections - unusable image
- Rely on ambient/street lighting

RTSP: Enabled on port 554
Authentication: Required (admin user)
```

**RTSP URL Format:**
```
rtsp://admin:PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

---

## Development Workflow

### Phase 1: macOS Testing (Complete)
**Purpose:** Validate algorithms with webcam before hardware complexity

**Achievements:**
- YOLOv8 detection validated
- Track persistence verified
- OSC integration tested
- Configuration system proven
- Optimal parameters determined

**Key learnings:**
- Confidence threshold: 0.45 (catches partial detections)
- Path history: 30 frames (good visualization)
- CPU vs GPU performance characterized
- Detection behavior at different distances/angles

### Phase 2: Linux + RTSP Camera PoC (Current Phase)
**Purpose:** Validate complete production system with single camera

**Objectives:**
1. **Day 1: Linux Setup**
   - Install Ubuntu 22.04 LTS on ABS Mage E
   - Install NVIDIA drivers (GTX 1660 Ti)
   - Verify CUDA available
   - Install tracker dependencies
   - See: LINUX_INSTALL.md

2. **Day 1-2: Camera Setup**
   - Connect camera to PoE router
   - Find camera IP address
   - Configure sub-stream: 640x480
   - Disable IR LEDs (critical!)
   - Enable RTSP
   - Test stream in VLC
   - See: SIMPLIFIED_SETUP_POE_ROUTER.md

3. **Day 2: Integration Testing**
   - Update config.json with RTSP URL
   - Run tracker, verify video feed
   - Test person detection accuracy
   - Optimize confidence threshold
   - Configure detection zones (min_detection_y)

4. **Day 2-3: Physical Mounting**
   - Mount camera at windowsill (60cm height)
   - Set 10° upward tilt angle
   - Test field of view (1m, 3m, 5m, 8m distances)
   - Adjust angle if needed
   - See: CAMERA_MOUNTING.md

5. **Day 3-4: 24/7 Service Setup**
   - Configure systemd service
   - Enable auto-start on boot
   - Test auto-restart on crash
   - Configure remote access (SSH/RDP)
   - Set up log monitoring
   - See: README.md service configuration

6. **Day 4-14: Proof of Concept Run**
   - 10 days continuous unattended operation
   - Daily remote monitoring via SSH
   - Log review for errors/crashes
   - Performance metrics collection
   - Detection accuracy validation

**Success Criteria for PoC:**
- [ ] System runs 10 days without manual intervention
- [ ] Auto-recovers from camera disconnections
- [ ] Detection rate >85% in good lighting conditions
- [ ] FPS consistently >15 (target: 20-25)
- [ ] Track IDs remain stable during slow pedestrian movement
- [ ] No memory leaks or performance degradation over time
- [ ] Remote monitoring via SSH works reliably
- [ ] Logs capture useful debugging information

### Phase 3: Multi-Camera Expansion (Future)
**After successful PoC:**
1. Add cameras 2 and 3
2. Implement ArUco marker calibration
3. Homography ground plane mapping
4. Multi-camera deduplication
5. Unified tracking coordinate system

---

## Technical Stack

### Core Dependencies
```
ultralytics>=8.0.0      # YOLOv8 detection and tracking
opencv-python>=4.8.0    # Video capture and processing
numpy>=1.24.0           # Array operations
python-osc>=1.8.0       # OSC messaging
torch>=2.0.0            # PyTorch with CUDA support
torchvision>=0.15.0     # Vision utilities
```

### Platform-Specific
```bash
# Linux (Ubuntu 22.04) - Production
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# macOS - Development/Testing
pip3 install torch torchvision  # Auto-selects MPS or CPU
```

---

## Code Architecture

### Main Components

**tracker.py** - Core tracking engine
```python
class PersonTracker:
    """
    Single-camera person detection and tracking
    
    Key features:
    - YOLOv8n detection with built-in BoT-SORT tracking
    - Persistent track IDs across frames
    - Configurable detection zones (Y-coordinate filtering)
    - Path history visualization
    - OSC data streaming
    - Cross-platform (macOS dev, Linux prod)
    - Auto-reconnect on stream failure
    - Threaded camera capture (prevents buffering lag)
    """
```

### Threaded Camera Capture (Critical for Multi-Camera)

**Problem:** RTSP streams buffer frames faster than they can be processed. Without threading, the processing loop reads from an ever-growing buffer, causing increasing lag. This is especially problematic with high-resolution main streams and becomes critical when running multiple cameras.

**Solution:** Use a dedicated background thread per camera that continuously grabs the latest frame. The main processing loop only ever gets the most recent frame, discarding any that buffered up.

```python
class ThreadedCamera:
    """
    Threaded camera capture to prevent frame buffering lag.
    Each camera gets its own capture thread.
    
    Critical for:
    - High-resolution streams (1080p, 4K)
    - Multi-camera setups (1-3 cameras)
    - Long-running 24/7 operation
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.grabbed = False
        self.running = True
        self.lock = threading.Lock()
        
        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        """Continuously grab frames in background"""
        while self.running:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
    
    def read(self):
        """Get the most recent frame (never stale)"""
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
```

**Multi-camera usage pattern:**
```python
# Each camera gets its own threaded capture
cameras = [
    ThreadedCamera("rtsp://admin:pass@192.168.1.101:554/h264Preview_01_sub"),
    ThreadedCamera("rtsp://admin:pass@192.168.1.102:554/h264Preview_01_sub"),
    ThreadedCamera("rtsp://admin:pass@192.168.1.103:554/h264Preview_01_sub"),
]

# Main loop processes latest frame from each camera
while True:
    for i, cam in enumerate(cameras):
        ret, frame = cam.read()  # Always gets latest frame, no lag
        if ret:
            process_frame(i, frame)
```

**Performance impact:**
- Without threading: Lag increases over time (frames buffer up)
- With threading: Consistent low latency, ~29 FPS sustained
- Memory overhead: Minimal (~1 frame buffer per camera)
- CPU overhead: One lightweight thread per camera

**config.json** - Runtime configuration
```json
{
  "camera": {
    "url": "rtsp://admin:password@192.168.1.100:554/h264Preview_01_sub",
    "mounting_height_cm": 60,
    "tilt_angle_degrees": 10,
    "min_detection_y": 50,
    "max_detection_y": null
  },
  "model": {
    "name": "yolov8n.pt",
    "confidence": 0.45,
    "device": "cuda"
  },
  "display": {
    "show_video": false  // Headless for production
  },
  "osc": {
    "enabled": true,
    "port": 8000
  }
}
```

### Cross-Platform Design Patterns

**Camera Backend Selection:**
```python
if platform.system() == "Darwin":  # macOS
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
else:  # Linux
    cap = cv2.VideoCapture(url)
```

**Device Selection:**
```python
# macOS: "mps" (Apple Silicon) or "cpu"
# Linux: "cuda" (NVIDIA GPU) or "cpu"
# Configured via config.json, not hardcoded
```

**Display Handling:**
```python
if self.show_video:
    cv2.imshow('Tracker', frame)  # GUI on macOS/Linux desktop
else:
    pass  # Headless mode for production
```

---

## Configuration System

### Camera Parameters

**mounting_height_cm** (int)
- Physical camera height from ground in centimeters
- Used for documentation and future calibration
- PoC value: 60cm (24 inches - windowsill height)
- Measure with tape measure for accuracy

**tilt_angle_degrees** (int/float)
- Upward tilt angle from horizontal in degrees
- 0° = level/horizontal, 10° = slight upward
- PoC value: 10° (optimal for 0-8m coverage from low mount)
- Measure with phone protractor app

**min_detection_y** (int or null)
- Y-pixel threshold (from top of frame)
- Detections with foot position above this line are ignored
- Filters very distant/small people and background objects
- PoC value: 50 (ignore top 50 pixels)
- Adjust based on field testing (30-150 typical range)

**max_detection_y** (int or null)
- Y-pixel threshold (from top of frame)
- Detections with foot position below this line are ignored
- Filters extremely close people if causing issues
- PoC value: null (disabled - allow close people)
- Set to frame_height - 100 if close people problematic

### Model Parameters

**name** (string)
- YOLO model file: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"
- PoC value: "yolov8n.pt" (fastest, good accuracy)
- First run downloads model (~6MB)
- Can switch models to test accuracy vs speed tradeoff

**confidence** (float 0.0-1.0)
- Detection confidence threshold
- Lower = more detections, more false positives
- Higher = fewer detections, fewer false positives
- PoC value: 0.45 (tuned for low-angle partial detections)
- Typical range: 0.4-0.6 depending on use case

**device** (string)
- Compute device: "cuda", "cpu", "mps"
- PoC value: "cuda" (GTX 1660 Ti GPU acceleration)
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`
- Falls back to CPU if CUDA unavailable

---

## Proof of Concept Testing Methodology

### Initial Validation Tests (Days 1-3)

**Test 1: Camera Connectivity**
```bash
# Verify RTSP stream accessible
vlc rtsp://admin:PASSWORD@CAMERA_IP:554/h264Preview_01_sub

# Expected: Video appears within 2-5 seconds
# Check: Image quality, frame rate, no authentication errors
```

**Test 2: Tracker Integration**
```bash
cd ~/projects/storefront-tracker
source venv/bin/activate
python tracker.py

# Expected: "Camera connected: 640x480 @ 30fps"
# Check: Video window (if enabled), detection boxes, track IDs
```

**Test 3: Detection Accuracy**
```bash
# Have person walk at different distances from window
# Distances to test: 1m, 3m, 5m, 8m, 10m

# Expected detection rates:
# 0-8m: >90% detection rate
# 8-10m: >70% detection rate
# >10m: Graceful degradation acceptable

# Document in test log:
# - Detection rate at each distance
# - False positive rate
# - Track ID stability
```

**Test 4: Field of View Validation**
```bash
# With camera at 10° tilt, verify:
# - Ground plane visible in bottom 1/3 of frame
# - Feet visible for people at all target distances
# - Close people (1m) may have cropped heads (acceptable!)
# - 8m people visible in full or mostly

# Adjust tilt angle if needed:
# - Too many missed distant people → increase to 12-15°
# - Poor ground plane visibility → decrease to 5-8°
```

**Test 5: Parameter Tuning**
```bash
# Test different confidence thresholds:
for conf in 0.3 0.4 0.45 0.5 0.6:
    # Edit config.json
    # Run for 5-10 minutes
    # Document detection rate vs false positives
    
# Test detection zone filtering:
for min_y in 30 50 80 100:
    # Edit config.json: "min_detection_y": min_y
    # Run and observe filtered detections
    # Document effectiveness
```

### Long-Term Stability Tests (Days 4-14)

**Test 6: 24/7 Unattended Operation**
```bash
# Setup as systemd service
sudo systemctl enable tracker
sudo systemctl start tracker

# Monitor daily via SSH:
ssh user@LINUX_IP
sudo journalctl -u tracker -f

# Check for:
# - Memory leaks (monitor RAM usage)
# - CPU/GPU utilization stability
# - Frame rate consistency
# - Camera reconnection events
# - Any crash/restart events
```

**Test 7: Network Interruption Recovery**
```bash
# Simulate network interruption:
# 1. Unplug camera ethernet for 30 seconds
# 2. Re-plug
# 3. Verify tracker auto-reconnects

# Check logs for:
# "Failed to read frame, reconnecting..."
# "Camera connected: 640x480 @ 30fps"

# Expected: Auto-recovery within 60 seconds
```

**Test 8: Performance Monitoring**
```bash
# Daily checks via SSH:
watch -n 1 nvidia-smi  # GPU usage, temperature, memory

# Weekly log review:
sudo journalctl -u tracker --since "1 week ago" | grep -i error
sudo journalctl -u tracker --since "1 week ago" | grep -i "failed"

# Monthly performance baseline:
# - Average FPS over 24 hours
# - Detection count statistics
# - Camera disconnection frequency
# - System uptime
```

### Data Collection for PoC

**Metrics to track:**
```python
# Add logging to tracker.py for PoC evaluation:
import logging

logging.basicConfig(
    filename='tracker_metrics.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log every 100 frames:
if frame_count % 100 == 0:
    logging.info(f"FPS: {fps:.2f}, People: {len(detections)}, "
                 f"Active tracks: {len(active_tracks)}")

# Log significant events:
logging.info(f"Track {track_id} started")
logging.info(f"Track {track_id} ended, duration: {duration:.1f}s")
logging.error(f"Camera connection failed, attempting reconnect")
```

**Success metrics:**
- Uptime: >99% (max 2.4 hours downtime over 10 days)
- FPS: Average >15, target 20-25
- Detection rate: >85% in 0-8m zone during daylight
- False positives: <5% of total detections
- Track ID stability: Same person keeps same ID for >10 seconds
- Auto-recovery: Successful within 60 seconds of camera reconnect

---

## Configuration Tuning Guide

### For Low-Angle Mounting (21-24" windowsill)

**Optimal starting configuration:**
```json
{
  "camera": {
    "mounting_height_cm": 60,
    "tilt_angle_degrees": 10,
    "min_detection_y": 50,
    "max_detection_y": null
  },
  "model": {
    "confidence": 0.45
  }
}
```

**Tuning decision tree:**

**If too many false positives (background objects):**
- Increase min_detection_y to 80-100
- Or increase confidence to 0.5-0.55

**If missing close people (<2m):**
- Lower confidence to 0.40
- Or ensure max_detection_y is null

**If missing distant people (>6m):**
- Lower min_detection_y to 30
- Or increase tilt_angle_degrees to 12-15°

**If FPS too low (<15):**
- Keep yolov8n.pt (don't upgrade to larger model)
- Ensure device: "cuda" not "cpu"
- Verify GPU usage with nvidia-smi
- Consider processing every 2nd frame if desperate

### For Different Lighting Conditions

**Bright daylight (ideal):**
```json
{
  "model": {
    "confidence": 0.5  // Can be higher, fewer false positives
  }
}
```

**Low light (dawn/dusk):**
```json
{
  "model": {
    "confidence": 0.40  // Lower to catch darker silhouettes
  }
}
```

**Night (street lighting only):**
```json
{
  "model": {
    "confidence": 0.35  // Very permissive
  },
  "camera": {
    "min_detection_y": 80  // Filter more background noise
  }
}
```

**If night quality insufficient:**
- Check camera exposure settings (increase max shutter time)
- Consider upgrading to Reolink RLC-820A ColorX (better low-light)
- Or add external IR illuminator outside (not inside!)

---

## Development Practices for PoC

### Testing Workflow

**Local changes:**
```bash
# SSH into Linux box
ssh user@LINUX_IP

# Navigate to project
cd ~/projects/storefront-tracker
source venv/bin/activate

# Stop service
sudo systemctl stop tracker

# Test changes
python tracker.py
# Press 'q' to quit

# If working, restart service
sudo systemctl start tracker

# Monitor
sudo journalctl -u tracker -f
```

**Configuration changes:**
```bash
# Edit config
nano config.json

# Restart to apply
sudo systemctl restart tracker

# Verify change took effect
sudo journalctl -u tracker -n 20
# Should show new mounting_height_cm or confidence value
```

**Code changes:**
```bash
# Edit tracker.py locally in VSCode
# Then copy to Linux:
scp tracker.py user@LINUX_IP:~/projects/storefront-tracker/

# SSH in and restart
ssh user@LINUX_IP
cd ~/projects/storefront-tracker
sudo systemctl restart tracker
```

### Logging Best Practices

**Add informative log messages:**
```python
# Startup
print(f"Running on: {self.system}")
print(f"Camera mounting: {self.mounting_height_cm}cm height, "
      f"{self.tilt_angle_degrees}° tilt")
print(f"CUDA available: {torch.cuda.is_available()}")

# During operation
if frame_count % 100 == 0:
    print(f"Frames: {frame_count}, FPS: {fps:.2f}, "
          f"People: {len(detections)}")

# Errors
print(f"ERROR: Failed to connect to camera: {self.camera_url}")
print(f"WARNING: Low FPS detected: {fps:.1f}")
```

**View logs remotely:**
```bash
# Live tail
sudo journalctl -u tracker -f

# Last 100 lines
sudo journalctl -u tracker -n 100

# Filter errors
sudo journalctl -u tracker | grep ERROR

# Time range
sudo journalctl -u tracker --since "2024-12-24 09:00"
```

---

## Critical Reminders for AI Assistants

**When suggesting code changes during PoC:**

1. **Maintain stability** - This is running 24/7 unattended
   - All changes must handle errors gracefully
   - Camera disconnections must auto-recover
   - No changes that could cause crashes

2. **Configuration-driven** - No hardcoded values
   - All parameters in config.json
   - Easy to tune without code changes
   - Document all new parameters

3. **Cross-platform compatibility** - Even in PoC
   - Code may be tested on macOS then deployed to Linux
   - Use platform.system() for OS-specific code
   - Camera backend, device selection, display handling

4. **Logging is critical** - PoC evaluation depends on logs
   - Log startup parameters
   - Log significant events
   - Log errors with context
   - Don't spam logs (every 100 frames, not every frame)

5. **Performance awareness** - FPS matters
   - Profile before optimizing
   - Prefer GPU operations
   - Minimize unnecessary processing
   - Track FPS impact of changes

**When debugging PoC issues:**

1. **Check logs first** - `sudo journalctl -u tracker -f`
2. **Verify camera connectivity** - Test in VLC before blaming code
3. **Check GPU usage** - `nvidia-smi` should show tracker using GPU
4. **Review configuration** - Typos in config.json common issue
5. **Test locally** - Stop service, run manually to see all output

---

## PoC Success Criteria Summary

**Technical validation:**
- [ ] RTSP camera integration stable over 10 days
- [ ] Detection accuracy >85% in target zone
- [ ] FPS consistently 15-30 (average >20)
- [ ] Auto-recovery from network interruptions
- [ ] No memory leaks or performance degradation
- [ ] GPU acceleration confirmed working

**Operational validation:**
- [ ] systemd service reliable (auto-start, auto-restart)
- [ ] Remote monitoring via SSH effective
- [ ] Logging provides useful debugging information
- [ ] Configuration changes apply without code changes
- [ ] Physical mounting stable (angle doesn't drift)

**Business validation:**
- [ ] Detection quality sufficient for intended use case
- [ ] System reliability acceptable for 24/7 operation
- [ ] Performance meets expectations
- [ ] Configuration is well-documented
- [ ] Ready to scale to multi-camera deployment

**Upon PoC success → Proceed to Phase 3 (Multi-Camera)**

---

## Quick Reference

### Essential Files

```
tracker.py                          # Main code
config.json                         # Active configuration
config.linux-production.json        # Production template
tracker.service                     # systemd service file

SIMPLIFIED_SETUP_POE_ROUTER.md     # Camera setup (PoE router)
CAMERA_MOUNTING.md                  # Physical installation
LINUX_INSTALL.md                    # Ubuntu setup
QUICK_REFERENCE.md                  # Command cheat sheet
```

### Essential Commands

```bash
# Service control
sudo systemctl status tracker
sudo systemctl start tracker
sudo systemctl stop tracker
sudo systemctl restart tracker

# Monitoring
sudo journalctl -u tracker -f
watch -n 1 nvidia-smi
htop

# Testing
cd ~/projects/storefront-tracker
source venv/bin/activate
python tracker.py

# Camera test
ping CAMERA_IP
vlc rtsp://admin:PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

### Critical Configuration

**RTSP URL:**
```
rtsp://admin:PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

**Camera settings that MUST be configured:**
- Sub-stream: 640x480 @ 30fps
- IR LEDs: "Stay Off" (prevents glass reflections)
- RTSP: Enabled on port 554

**Tracker settings for low-angle mount:**
- mounting_height_cm: 60
- tilt_angle_degrees: 10
- min_detection_y: 50
- confidence: 0.45

---

**This PoC phase validates the complete system before multi-camera scaling. Success here = confidence to deploy production.**
