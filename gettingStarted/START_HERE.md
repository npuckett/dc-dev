# 📚 Documentation Guide - macOS Testing Phase

**Start Here:** This guide organizes all documentation for macOS testing before Linux deployment.

---

## 🎯 Your Current Phase: macOS Testing

You're about to test the person tracker on your Mac with a webcam before deploying to Linux with RTSP cameras.

**Why test on macOS first?**
- ✅ Familiar development environment  
- ✅ Easy debugging with VSCode
- ✅ Validate algorithms before hardware complexity
- ✅ No network/camera setup required yet
- ✅ Fast iteration on code changes

**Timeline:**
- macOS testing: 30 minutes - 4 hours
- Linux deployment: 1-2 hours (after macOS works)
- Total: Same day to full production

---

## 📖 Documentation Roadmap

### Phase 1: Initial Testing (Start Here) ⭐

**File:** `MACOS_QUICKSTART.md`  
**Time:** 5 minutes  
**Purpose:** Get tracker running with webcam immediately

**What you'll do:**
1. Setup virtual environment
2. Install dependencies  
3. Run tracker with webcam
4. Verify person detection works

**Read this if:**
- You want to test tracker RIGHT NOW
- You haven't run it yet
- You just want to see if it works

---

### Phase 2: Comprehensive Testing

**File:** `MACOS_TESTING.md`  
**Time:** 30 minutes - 2 hours  
**Purpose:** Thorough validation and parameter tuning

**What you'll learn:**
- How to validate detection accuracy
- How to test OSC output
- How to tune confidence threshold
- How to test with video files
- Performance benchmarks and optimization

**Read this if:**
- Basic quickstart worked
- You want to understand detection behavior
- You need to tune parameters
- You're preparing for Linux deployment

---

### Phase 3: Development & Customization

**File:** `instructions-macos.md`  
**Time:** Reference material  
**Purpose:** Context for AI assistants and code modifications

**What it contains:**
- Complete code architecture
- Configuration system details
- macOS-specific behaviors
- Development patterns
- Debugging techniques

**Use this when:**
- Working in VSCode with GitHub Copilot
- Making code changes
- Adding new features
- Debugging issues
- Understanding the codebase

---

### Phase 4: Linux Deployment

**After macOS testing passes, read these:**

**File:** `LINUX_INSTALL.md`  
**Time:** 1-2 hours  
**Purpose:** Ubuntu installation and configuration

**File:** `CAMERA_MOUNTING.md`  
**Time:** 30 minutes  
**Purpose:** Physical camera setup and angle configuration

**File:** `QUICK_REFERENCE.md`  
**Time:** Reference  
**Purpose:** Command cheat sheet for production

---

## 🗺️ Complete File List

### Essential for macOS Testing

| File | Purpose | When to Read |
|------|---------|--------------|
| **MACOS_QUICKSTART.md** | 5-min setup | **START HERE** |
| **MACOS_TESTING.md** | Full testing guide | After quickstart works |
| **instructions-macos.md** | Dev context | When coding in VSCode |
| **test_osc_receiver.py** | OSC validation | Testing data output |
| **test_setup.py** | Install verification | Troubleshooting |
| **config.macos-test.json** | Webcam config | Copy to config.json |

### Essential for Linux Deployment

| File | Purpose | When to Read |
|------|---------|--------------|
| **LINUX_INSTALL.md** | Ubuntu setup | Before installing Linux |
| **setup.sh** | Auto-installer | Run on fresh Ubuntu |
| **CAMERA_MOUNTING.md** | Physical setup | Mounting cameras |
| **config.linux-production.json** | Production config | Deploying to Linux |
| **tracker.service** | systemd service | 24/7 operation |
| **QUICK_REFERENCE.md** | Command reference | Daily operations |

### General Reference

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** | Full documentation | Comprehensive reference |
| **instructions.md** | Original dev context | General understanding |
| **tracker.py** | Main code | Modifying behavior |
| **config.json** | Active configuration | Runtime settings |

---

## 🚦 Step-by-Step Workflow

### Step 1: Get It Running (5 minutes)

```bash
# Follow MACOS_QUICKSTART.md
cd storefront-tracker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip3 install torch torchvision
cp config.macos-test.json config.json
python tracker.py
```

**Success criteria:** Video window appears with person detection

---

### Step 2: Validate Functionality (30-60 minutes)

**Follow MACOS_TESTING.md checklist:**
- [ ] Person detection works
- [ ] Track IDs persist
- [ ] Path trails display
- [ ] Multiple people tracked
- [ ] OSC messages transmit
- [ ] FPS acceptable (>10)

**Success criteria:** All tests pass

---

### Step 3: Tune Parameters (Optional, 30 minutes)

**Experiment with settings:**
```json
// Edit config.json
"confidence": 0.45  // Try 0.4, 0.5, 0.6
"history_length": 30  // Try 10, 50
"device": "mps"  // Try "cpu" for comparison
```

**Success criteria:** Found optimal settings for your use case

---

### Step 4: Setup RTSP Camera (When Camera Arrives)

**Follow camera setup guide:**
1. **Read:** REOLINK_SETUP.md (comprehensive guide)
2. **Or:** CAMERA_SETUP_FLOWCHART.md (quick diagnostic)
3. **Connect:** Camera to PoE switch
4. **Find:** Camera IP address on network
5. **Configure:** Sub-stream 640x480, disable IR
6. **Enable:** RTSP on port 554
7. **Test:** Stream in VLC
8. **Integrate:** Update config.json with RTSP URL

**Success criteria:** Camera stream works in tracker

---

### Step 5: Deploy to Linux (1-2 hours)

1. **Install Ubuntu:** Follow LINUX_INSTALL.md
2. **Transfer project:** Copy files to Linux
3. **Update config:** Use config.linux-production.json
4. **Test with camera:** Run tracker with RTSP
5. **Deploy service:** Setup systemd auto-start

**Success criteria:** Tracker running 24/7 on Linux

---

## 💡 Quick Navigation

**"I just want to test it NOW!"**  
→ Read: MACOS_QUICKSTART.md

**"It's working, what should I test?"**  
→ Read: MACOS_TESTING.md  

**"I'm coding in VSCode and need context"**  
→ Read: instructions-macos.md

**"I need to check my installation"**  
→ Run: `python test_setup.py`

**"I want to see OSC messages"**  
→ Run: `python test_osc_receiver.py`

**"Time to deploy to Linux!"**  
→ Read: LINUX_INSTALL.md

**"How do I setup my Reolink camera?"**  
→ **If you have PoE router:** SIMPLIFIED_SETUP_POE_ROUTER.md (easiest!)  
→ **Comprehensive guide:** REOLINK_SETUP.md  
→ **Quick diagnostic:** CAMERA_SETUP_FLOWCHART.md

**"How do I mount the cameras?"**  
→ Read: CAMERA_MOUNTING.md

**"I need a command reference"**  
→ Read: QUICK_REFERENCE.md

**"I want the complete manual"**  
→ Read: README.md

---

## 🎓 What Each Phase Teaches

### macOS Testing Phase (You Are Here)

**Technical skills:**
- Understanding YOLO detection behavior
- Configuring confidence thresholds
- Interpreting track persistence
- Debugging OpenCV issues
- Testing OSC integration

**Confidence gained:**
- Tracker algorithms work correctly
- Detection accuracy is acceptable  
- Performance is adequate
- Configuration system works
- Ready for production deployment

### Linux Deployment Phase (Next)

**Technical skills:**
- Ubuntu installation & setup
- NVIDIA driver configuration
- RTSP camera connection
- systemd service creation
- Remote access setup

**Production readiness:**
- 24/7 unattended operation
- Auto-restart on failure
- Remote monitoring
- Performance optimization
- Multi-camera coordination

---

## 📊 Expected Timeline

| Phase | Duration | What You're Doing |
|-------|----------|-------------------|
| **Setup** | 5 min | Install dependencies |
| **First Run** | 5 min | Get webcam working |
| **Basic Tests** | 15 min | Verify detection |
| **OSC Testing** | 15 min | Validate data output |
| **Parameter Tuning** | 30 min | Optimize settings |
| **Extended Testing** | 1 hour | Validate stability |
| **Linux Install** | 30 min | Ubuntu setup |
| **Camera Setup** | 30 min | Physical mounting |
| **Production Deploy** | 30 min | Service configuration |
| **Total** | ~4 hours | Complete deployment |

**You can skip phases based on confidence level**

---

## ✅ Success Metrics

### macOS Testing Complete When:
- ✅ Tracker runs for 30+ minutes without crash
- ✅ Detection rate >80% in good lighting
- ✅ FPS >10 consistently
- ✅ OSC messages validate correctly
- ✅ You understand confidence parameter
- ✅ Track IDs remain stable

### Ready for Linux When:
- ✅ All macOS tests pass
- ✅ You're comfortable with tracker behavior
- ✅ Optimal configuration determined
- ✅ You can debug issues independently
- ✅ You understand the code structure

---

## 🆘 Getting Help

**Issue During macOS Testing:**
1. Check MACOS_TESTING.md troubleshooting section
2. Run `python test_setup.py` to verify install
3. Review instructions-macos.md for code details

**Issue During Linux Deploy:**
1. Check LINUX_INSTALL.md troubleshooting
2. Review QUICK_REFERENCE.md for commands
3. Check systemd logs: `sudo journalctl -u tracker -f`

**General Questions:**
1. Search README.md (comprehensive)
2. Check relevant instructions.md file
3. Review code comments in tracker.py

---

## 🎯 TL;DR - Start Here

```bash
# 1. Quick test (5 minutes)
cd storefront-tracker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip3 install torch torchvision
cp config.macos-test.json config.json
python tracker.py

# 2. If it works, read: MACOS_TESTING.md

# 3. When ready, read: LINUX_INSTALL.md

# 4. Deploy and enjoy! 🎉
```

---

**Current Phase: macOS Testing**  
**Next Step: Read MACOS_QUICKSTART.md**  
**Time to Working System: 5 minutes**

Let's get started! 🚀
