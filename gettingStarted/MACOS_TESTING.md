# macOS Testing Guide - Storefront Person Tracker

Complete guide for testing and validating the tracker on macOS before Linux deployment.

---

## 🎯 Purpose of macOS Testing

**Why test on macOS first?**
- ✅ Validate tracker logic and algorithms
- ✅ Test YOLO detection performance
- ✅ Verify OSC data output
- ✅ Tune confidence and detection parameters
- ✅ Debug code in familiar environment
- ✅ Test with webcam before RTSP cameras arrive

**What you'll validate:**
- Person detection accuracy
- Track ID persistence
- Path trail visualization
- OSC message format and timing
- Frame rate performance
- Configuration system

---

## 📋 Prerequisites

### Hardware
- Mac computer (any Intel or Apple Silicon)
- Built-in webcam or USB camera
- Internet connection for downloading models

### Software
- macOS 11.0+ (Big Sur or newer)
- Python 3.8+ (usually pre-installed)
- 2GB free disk space (for models and dependencies)

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Setup Environment

```bash
# Open Terminal (Cmd+Space, type "Terminal")

# Navigate to project folder
cd ~/Downloads/storefront-tracker

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### Step 2: Install Dependencies

```bash
# Install core packages
pip install -r requirements.txt

# Install PyTorch for Apple Silicon (M1/M2/M3)
pip3 install torch torchvision

# If Intel Mac, same command works
```

### Step 3: Configure for Webcam

```bash
# Use macOS test configuration
cp config.macos-test.json config.json

# Verify configuration
cat config.json
```

### Step 4: Run Tracker

```bash
# Start tracker with webcam
python tracker.py

# First run will download YOLOv8 model (~6MB)
# Webcam permission popup will appear - click "Allow"
# Video window should appear showing webcam feed
```

**You should see:**
- Video window with webcam feed
- Green bounding boxes around detected people
- Red dots (centroids) and blue dots (feet)
- Track IDs labeled on people
- FPS counter in top-left

**Controls:**
- `v` - Toggle video display
- `p` - Toggle path trails
- `i` - Toggle info overlay
- `o` - Toggle OSC output
- `q` - Quit

---

## ✅ Verification Tests

### Test 1: Basic Person Detection

**Procedure:**
1. Run tracker with webcam
2. Sit in front of camera
3. Move slowly left and right

**Expected:**
- ✅ Green box appears around you
- ✅ Track ID number appears (e.g. "ID: 1")
- ✅ Red dot on your torso (centroid)
- ✅ Blue dot at bottom of box (foot position)
- ✅ FPS > 10 (should be 15-25 fps)

**If not working:**
- Check Terminal for error messages
- Ensure webcam permission granted
- Try different lighting (brighter is better)
- Check `python test_setup.py` output

### Test 2: Track Persistence

**Procedure:**
1. Stand in front of camera
2. Note your Track ID (e.g. "ID: 1")
3. Move around slowly
4. Watch ID number

**Expected:**
- ✅ Same ID follows you as you move
- ✅ ID doesn't change unless you leave and return
- ✅ New person entering gets different ID (e.g. "ID: 2")

**If IDs keep changing:**
- Lower movement speed
- Improve lighting
- Check confidence level in config.json
- This is normal if leaving/re-entering frame

### Test 3: Path Trails

**Procedure:**
1. Press `p` to enable paths
2. Walk slowly across camera view
3. Watch for yellow trail line

**Expected:**
- ✅ Yellow line follows your movement
- ✅ Trail fades/disappears after ~30 positions
- ✅ Multiple people have separate trails

**If trails not showing:**
- Press `p` again to toggle on
- Check Terminal for "Path trails: ON"
- Ensure show_paths = true in config.json

### Test 4: Multiple People

**Procedure:**
1. Have 2-3 people in webcam view
2. Watch detection and tracking

**Expected:**
- ✅ Each person gets unique Track ID
- ✅ IDs remain stable as people move
- ✅ Partial occlusion handled (person behind another)
- ✅ FPS remains >10 with multiple people

**Performance notes:**
- Apple Silicon (M1/M2/M3): 20-30 fps with 3 people
- Intel Mac: 10-15 fps with 3 people
- Both are acceptable for validation testing

### Test 5: OSC Output

**Procedure:**
1. Keep tracker running
2. In new Terminal window:

```bash
cd ~/Downloads/storefront-tracker
source venv/bin/activate
python test_osc_receiver.py
```

3. Move in front of webcam

**Expected:**
- ✅ OSC receiver shows messages scrolling
- ✅ `/tracking/count` updates with person count
- ✅ `/person/1/x`, `/person/1/y` show pixel coordinates
- ✅ `/person/1/norm_x`, `/person/1/norm_y` show 0.0-1.0 values
- ✅ Messages update every frame (20-30x per second)

**If no OSC messages:**
- Press `o` in tracker window to enable OSC
- Check Terminal for "OSC output: ON"
- Ensure enable_osc = true in config.json
- Check test_osc_receiver.py is running on port 8000

---

## 🧪 Advanced Testing

### Test Configuration Changes

**Test 1: Confidence Threshold**

```bash
# Edit config
nano config.json

# Change confidence value
"confidence": 0.3  # Very permissive
# or
"confidence": 0.7  # Very strict

# Save and restart tracker
python tracker.py
```

**Observe:**
- Lower confidence (0.3): More detections, more false positives
- Higher confidence (0.7): Fewer false positives, might miss some people
- Sweet spot: 0.45-0.5 for most scenarios

**Test 2: History Length**

```json
"tracking": {
  "history_length": 10   // Short trails
}
// vs
"tracking": {
  "history_length": 50   // Long trails
}
```

**Observe:**
- Shorter = less memory, cleaner display
- Longer = better visualization of movement patterns

**Test 3: Model Comparison**

```bash
# Download and test different models
# YOLOv8n (default - fastest)
"model": { "name": "yolov8n.pt" }

# YOLOv8s (slower, more accurate)
"model": { "name": "yolov8s.pt" }

# First run downloads new model (~20MB)
# Compare FPS and detection accuracy
```

---

## 🎥 Video File Testing

If you don't want to use webcam, test with video files:

### Option 1: Download Test Video

```bash
# Download sample pedestrian video
cd ~/Downloads/storefront-tracker
curl -O https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4

# Or use your own video file
# Copy any .mp4 file to project folder
```

### Option 2: Update Config

```json
{
  "camera": {
    "url": "big_buck_bunny_720p_1mb.mp4",  // Video file path
    "use_substream": false,
    "rotate_180": false
  }
}
```

### Option 3: Run Tracker

```bash
python tracker.py

# Video will loop when it ends
# Press 'q' to quit
```

**Benefits of video testing:**
- Repeatable testing (same footage each time)
- No webcam permissions needed
- Can test with realistic storefront footage
- Great for debugging specific scenarios

---

## 📊 Performance Benchmarks

### Expected FPS on macOS

| Mac Type | CPU/GPU | 1 Person | 3 People | Notes |
|----------|---------|----------|----------|-------|
| M1/M2/M3 | MPS | 25-30 fps | 20-25 fps | Excellent |
| Intel i5/i7 | CPU | 15-20 fps | 10-15 fps | Good |
| Intel i9 | CPU | 20-25 fps | 15-20 fps | Very good |

**If FPS is low (<10):**
```json
// Lower resolution (if using video file)
"camera": { "url": "video_640x480.mp4" }

// Or process every 2nd frame
// Add to tracker.py temporarily:
if frame_count % 2 != 0:
    continue
```

---

## 🔧 macOS-Specific Configuration

### Optimal Settings for Apple Silicon (M1/M2/M3)

```json
{
  "camera": {
    "url": 0,
    "mounting_height_cm": 100,
    "tilt_angle_degrees": 0
  },
  "model": {
    "name": "yolov8n.pt",
    "confidence": 0.5,
    "device": "mps"  // Apple GPU acceleration
  },
  "display": {
    "show_video": true,
    "show_paths": true,
    "show_info": true
  },
  "osc": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8000
  }
}
```

### Optimal Settings for Intel Mac

```json
{
  "camera": {
    "url": 0,
    "mounting_height_cm": 100,
    "tilt_angle_degrees": 0
  },
  "model": {
    "name": "yolov8n.pt",
    "confidence": 0.5,
    "device": "cpu"  // CPU only on Intel Macs
  },
  "display": {
    "show_video": true,
    "show_paths": true,
    "show_info": true
  }
}
```

---

## 🐛 Troubleshooting macOS

### Issue: Webcam Not Working

**Symptoms:** 
- "Failed to open camera" error
- No video window appears
- Permission denied

**Solutions:**

```bash
# 1. Check webcam permissions
# System Preferences → Security & Privacy → Camera
# Ensure Terminal has camera access

# 2. Try different camera index
nano config.json
# Change "url": 0 to "url": 1

# 3. Check if camera in use
# Close all apps that might use camera:
# - Zoom, Teams, Skype, Photo Booth, etc.

# 4. Test camera with Photo Booth first
open -a "Photo Booth"
```

### Issue: "No module named 'ultralytics'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solution:**
```bash
# Ensure virtual environment is activated
# Look for (venv) in terminal prompt

# If not activated:
cd ~/Downloads/storefront-tracker
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "MPS backend not available"

**Symptoms:**
```
RuntimeError: MPS backend not available
```

**Solution:**
```bash
# Change device from MPS to CPU
nano config.json

# Update:
"device": "cpu"  // Change from "mps"

# Save and restart tracker
```

### Issue: Very Low FPS (<5)

**Symptoms:**
- Video window is choppy
- FPS counter shows <5

**Solutions:**

```bash
# 1. Close other applications
# Especially: Chrome, Slack, etc.

# 2. Use CPU instead of MPS (if Apple Silicon)
nano config.json
"device": "cpu"  // Sometimes faster than MPS for webcam

# 3. Lower confidence threshold
"confidence": 0.6  // Faster processing

# 4. Reduce frame processing
# Edit tracker.py, add at start of while loop:
if frame_count % 2 != 0:  # Process every 2nd frame
    continue
```

### Issue: Video Window Doesn't Appear

**Symptoms:**
- Tracker runs but no window
- Can see terminal output but no video

**Solution:**

```bash
# OpenCV GUI issue on some Macs
# Reinstall with GUI support
pip uninstall opencv-python
pip install opencv-contrib-python

# Restart tracker
python tracker.py
```

### Issue: "Address already in use" (OSC)

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**

```bash
# Another process using port 8000
# Find and kill it:
lsof -i :8000
kill -9 <PID>

# Or change OSC port:
nano config.json
"port": 8001  // Use different port
```

---

## 📝 Testing Checklist

**Before considering macOS testing complete, verify:**

- [ ] Tracker starts without errors
- [ ] Webcam feed appears in window
- [ ] People detected with bounding boxes
- [ ] Track IDs assigned and persistent
- [ ] Centroid (red) and foot (blue) dots visible
- [ ] FPS > 10 (preferably 15+)
- [ ] Path trails display when enabled (press `p`)
- [ ] Info overlay shows count and FPS (press `i`)
- [ ] OSC messages transmit (test with receiver)
- [ ] Multiple people tracked simultaneously
- [ ] Configuration changes take effect
- [ ] Can quit cleanly (press `q`)

**Optional advanced tests:**
- [ ] Test with video file instead of webcam
- [ ] Compare YOLOv8n vs YOLOv8s performance
- [ ] Test different confidence thresholds
- [ ] Validate OSC message format matches expectations
- [ ] Test on both CPU and MPS/GPU (if Apple Silicon)

---

## 🎓 What You're Learning

**Understanding before Linux deployment:**

1. **Detection behavior:** How YOLO detects people at different:
   - Distances (close vs far)
   - Angles (front, side, back)
   - Lighting conditions
   - Partial occlusions

2. **Tracking stability:** How track IDs:
   - Persist across frames
   - Handle brief occlusions
   - Behave when people leave/re-enter

3. **Performance characteristics:**
   - FPS with different numbers of people
   - Model speed (YOLOv8n vs s vs m)
   - Device performance (CPU vs GPU)

4. **Configuration impact:**
   - Confidence threshold effects
   - Detection zone filtering
   - Display options

5. **OSC data format:**
   - Message structure
   - Update frequency
   - Coordinate systems (pixel vs normalized)

**This knowledge transfers directly to Linux RTSP camera setup!**

---

## 📦 Export Configuration for Linux

Once you have working settings on macOS:

### Document Your Settings

```bash
# Create notes file
nano WORKING_CONFIG_NOTES.txt
```

```
macOS Testing Results
=====================

Working Configuration:
- Model: yolov8n.pt
- Confidence: 0.45
- Device: mps (or cpu)
- FPS achieved: 22 fps with 1 person, 18 fps with 3 people

Detection Notes:
- Works well in bright lighting
- Struggles with back-lighting
- Track IDs stable with slow movement
- Occasional ID switch with fast movement

For Linux Production:
- Use same confidence: 0.45
- Same model: yolov8n.pt
- Expected FPS (CUDA): 25-30 fps (better than macOS)
- Consider min_detection_y: 50 for low mounting

Next Steps:
- Deploy to Linux
- Test with RTSP camera
- Adjust for camera angle (10° tilt)
```

### Transfer Configuration

```bash
# Your working macOS config
cat config.json

# Copy relevant settings to Linux config
# Especially:
# - model.name
# - model.confidence
# - tracking.history_length
```

---

## 🚀 Next Steps: Linux Deployment

**After successful macOS testing:**

1. ✅ Confidence in tracker functionality
2. ✅ Understanding of detection behavior
3. ✅ Validated OSC output format
4. ✅ Known performance characteristics

**Ready for Linux:**
1. Install Ubuntu on ABS Mage E (see LINUX_INSTALL.md)
2. Transfer project files
3. Update config for RTSP camera
4. Add camera mounting parameters
5. Deploy as systemd service

**macOS testing gives you a solid foundation before dealing with:**
- RTSP camera setup
- Network configuration
- systemd service management
- Remote access
- 24/7 deployment

---

## 💡 Pro Tips

**Efficient macOS Testing Workflow:**

```bash
# Terminal 1: Tracker
cd ~/Downloads/storefront-tracker
source venv/bin/activate
python tracker.py

# Terminal 2: OSC Receiver (optional)
cd ~/Downloads/storefront-tracker
source venv/bin/activate
python test_osc_receiver.py

# Terminal 3: Config editing
cd ~/Downloads/storefront-tracker
nano config.json
# Make changes, save, restart tracker in Terminal 1
```

**Quick Configuration Testing:**

```bash
# Test different confidence levels quickly
for conf in 0.3 0.4 0.5 0.6 0.7; do
    echo "Testing confidence: $conf"
    # Manually update config.json with each value
    # Run tracker briefly
    # Note performance
done
```

**Save Your Experiments:**

```bash
# Save working configurations
cp config.json config.macos-working-v1.json
cp config.json config.macos-working-v2.json
# etc.
```

---

## 📚 Additional Resources

**Files for macOS Testing:**
- `README.md` - Full project documentation
- `QUICKSTART.md` - Quick setup guide
- `instructions.md` - GitHub Copilot context
- `test_setup.py` - Verify installation
- `test_osc_receiver.py` - OSC message monitor

**Configuration Files:**
- `config.macos-test.json` - Template for webcam
- `config.json` - Active configuration (edit this)

**When Ready for Linux:**
- `LINUX_INSTALL.md` - Linux setup guide
- `setup.sh` - Automated Linux installation
- `CAMERA_MOUNTING.md` - Physical camera setup
- `config.linux-production.json` - Template for deployment

---

## ✅ Success Criteria

**You're ready to move to Linux when:**

- ✅ Tracker runs reliably on macOS for 30+ minutes
- ✅ Detection accuracy >85% with good lighting
- ✅ FPS consistently >10 (preferably 15+)
- ✅ OSC messages received and formatted correctly
- ✅ Track IDs remain stable during movement
- ✅ You understand how confidence affects detection
- ✅ You know your optimal configuration values
- ✅ All troubleshooting steps documented

**Expected timeline:**
- First run: 15 minutes
- Basic validation: 1-2 hours
- Advanced testing: 2-4 hours
- Total macOS testing: ~4-6 hours before Linux deployment

This testing phase is valuable - it ensures you understand the system before dealing with Linux complexity!
