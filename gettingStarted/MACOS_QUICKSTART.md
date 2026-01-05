# macOS Quick Start - 5 Minute Test

Get the tracker running on your Mac in 5 minutes with webcam testing.

---

## ⚡ Super Fast Setup

```bash
# 1. Open Terminal (Cmd+Space, type "Terminal")
cd ~/Downloads/storefront-tracker

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (takes ~2 minutes)
pip install -r requirements.txt
pip3 install torch torchvision

# 4. Setup for webcam
cp config.macos-test.json config.json

# 5. Run!
python tracker.py

# First run downloads YOLO model (~6MB)
# Allow webcam access when prompted
# Video window should appear
```

**That's it! You should see:**
- ✅ Video window with webcam feed
- ✅ Green boxes around people
- ✅ Track IDs and FPS counter

---

## 🎮 Controls

Once running:
- **`p`** - Toggle path trails on/off
- **`i`** - Toggle info overlay on/off  
- **`o`** - Toggle OSC output on/off
- **`v`** - Toggle video display on/off
- **`q`** - Quit

---

## ✅ Quick Validation Tests

### Test 1: Am I Detected?
- Sit in front of webcam
- Look for green box around you
- See Track ID number (e.g., "ID: 1")
- **Pass:** Green box follows you ✅

### Test 2: Do Trails Work?
- Press `p` to enable trails
- Move slowly side to side
- Look for yellow line following you
- **Pass:** Yellow trail appears ✅

### Test 3: Multiple People
- Have 2 people in view
- Each should get different ID
- Both tracked simultaneously
- **Pass:** Both people tracked ✅

---

## 🐛 Common First-Run Issues

### "Failed to open camera"
**Fix:**
```bash
# Close any apps using camera (Zoom, Teams, etc.)
# System Preferences → Security & Privacy → Camera
# Ensure Terminal has camera access
```

### "No module named 'ultralytics'"
**Fix:**
```bash
# Make sure virtual environment is active
# You should see (venv) in terminal
source venv/bin/activate
pip install -r requirements.txt
```

### No video window appears
**Fix:**
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
python tracker.py
```

### Very low FPS (<5)
**Fix:**
```bash
# Edit config
nano config.json

# Change confidence
"confidence": 0.6  # Higher = faster

# Or change device
"device": "cpu"  # Sometimes faster than mps on older Macs
```

---

## 🧪 Test OSC Output (Optional)

Open **second Terminal window**:

```bash
cd ~/Downloads/storefront-tracker
source venv/bin/activate
python test_osc_receiver.py

# Should show messages as you move:
# 📊 People tracked: 1
# 👤 Person 1: position updated
# 📦 Bundled data - ID:1 Pos:(320,240)...
```

Press Ctrl+C to stop receiver.

---

## 📊 Performance Expectations

| Mac Type | Expected FPS |
|----------|--------------|
| M1/M2/M3 Mac | 20-30 fps |
| Intel i5/i7 Mac | 12-20 fps |
| Older Intel Mac | 8-15 fps |

**All are fine for testing!** Linux with GPU will be much faster (25-30 fps).

---

## ⚙️ Quick Config Changes

### Change confidence threshold:
```bash
nano config.json
# Edit: "confidence": 0.45
# Save: Ctrl+O, Enter, Ctrl+X
python tracker.py
```

### Change trail length:
```bash
nano config.json
# Edit: "history_length": 50
python tracker.py
```

### Disable OSC (faster):
```bash
nano config.json
# Edit: "enabled": false
python tracker.py
```

---

## 📚 What to Read Next

**Once basic test works:**
- 📖 `MACOS_TESTING.md` - Comprehensive testing guide
- 📖 `README.md` - Full documentation
- 📖 `instructions.md` - Developer context

**Ready for production:**
- 📖 `LINUX_INSTALL.md` - Ubuntu installation
- 📖 `CAMERA_MOUNTING.md` - Physical camera setup
- 📖 `QUICK_REFERENCE.md` - Command cheat sheet

---

## ✅ Success Checklist

**Before moving to Linux, verify:**
- [ ] Tracker runs without errors
- [ ] Webcam feed appears
- [ ] People detected with boxes and IDs
- [ ] FPS > 10
- [ ] Trails visible (press `p`)
- [ ] OSC messages received (if tested)
- [ ] Can quit cleanly (press `q`)

**Time to complete:** 5-30 minutes depending on testing depth

---

## 🎯 Next Steps

1. ✅ **Basic test works** → Read MACOS_TESTING.md for advanced tests
2. ✅ **Advanced tests pass** → Prepare for Linux deployment
3. ✅ **Ready for production** → Follow LINUX_INSTALL.md

**Questions?** Check troubleshooting in MACOS_TESTING.md

---

**🎉 Working? Congratulations! You've validated the tracker on macOS.**

Now you can confidently deploy to Linux knowing the code works!
