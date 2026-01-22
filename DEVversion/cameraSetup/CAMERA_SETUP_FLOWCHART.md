# Camera Setup Quick Flowchart

**Use this to quickly diagnose camera setup issues.**

---

## 🚦 Step-by-Step Decision Tree

### START: Camera Unboxed

```
📦 Camera in box
    ↓
    What powers your camera?
    ├─ PoE Router (computer + camera plug into same router)
    │  → ✅ EASIEST PATH! → See SIMPLIFIED_SETUP_POE_ROUTER.md
    │  → Skip all PoE equipment! Go to "Connect Camera (PoE Router)"
    │
    ├─ PoE Switch (separate from router)
    │  → Go to "Connect Camera (PoE Switch)"
    │
    └─ No PoE equipment
       → Purchase PoE switch ($40-70) or injector ($15)
       → Then go to "Connect Camera"
```

---

### Connect Camera (PoE Router) - SIMPLIFIED PATH

```
🔌 You have PoE router (computer + camera on same router)
    ↓
    Plug camera into PoE port on router
    ↓
    Wait 60 seconds
    ↓
    Is LED green?
    ├─ YES → Camera ready! → Go to "Find on Network (Easy Mode)"
    └─ NO → Check router's PoE port LED, try different port
```

---

### Find on Network (Easy Mode) - For PoE Router Users

```
🔍 Camera on same router as computer
    ↓
    Open router admin page (192.168.1.1 or similar)
    ↓
    Look at "Connected Devices" or "DHCP Clients"
    ↓
    Find "Reolink" or "RLC-520A"
    ↓
    Write down IP: 192.168.1.___ → Go to "Camera Found"
    
    Alternative: Use Reolink app "Scan LAN"
```

---

### Connect Camera (PoE Switch) - Standard Path

```
🔌 Connect camera to PoE
    ↓
    Wait 60 seconds
    ↓
    Is LED lit up?
    ├─ YES → LED is green/blue → Go to "Find on Network"
    ├─ LED is red/blinking → Camera booting, wait 30 more seconds
    └─ NO LED → Troubleshoot power
```

**Troubleshoot Power:**
```
No LED
├─ Check PoE switch LED (port should be lit)
├─ Try different PoE port
├─ Test ethernet cable (swap with known-good)
└─ Verify PoE voltage (48V on switch/injector)
```

---

### Find on Network

```
🔍 Camera powered, now find it
    ↓
    Do you have Reolink app?
    ├─ YES → Open app → "Scan LAN" → Go to "Camera Found"
    └─ NO → Download Reolink app OR check router DHCP page
    
    Can't find in app?
    ├─ Check router's "Connected Devices" page
    ├─ Look for "Reolink" or MAC starting with 00:0F
    └─ Write down IP: 192.168.1.___ → Go to "Camera Found"
```

---

### Camera Found

```
✅ Found camera at IP: 192.168.1.XXX
    ↓
    Open web browser
    ↓
    Go to: http://192.168.1.XXX
    ↓
    Login page appears?
    ├─ YES → Username: admin, Password: (blank) → Go to "Configure Camera"
    └─ NO → Troubleshoot access
```

**Troubleshoot Access:**
```
Can't open web page
├─ Ping camera: ping 192.168.1.XXX (should respond)
├─ Try different browser (Chrome, Safari)
├─ Check firewall settings
└─ Verify computer on same subnet (192.168.1.X)
```

---

### Configure Camera

```
⚙️ In camera web interface
    ↓
    Set admin password → Save
    ↓
    Settings → Display
    ├─ Sub-stream: 640x480 @ 30fps → Save
    └─ Main stream: 1920x1080 @ 30fps → Save
    ↓
    Settings → Light
    └─ IR LEDs: "Stay Off" → Save (CRITICAL!)
    ↓
    Settings → Network → Port Settings
    └─ Enable RTSP (port 554) → Save
    ↓
    Go to "Test Connection"
```

---

### Test Connection

```
🧪 Test RTSP stream
    ↓
    Open VLC Media Player
    ↓
    Media → Open Network Stream
    ↓
    Enter: rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub
    ↓
    Click Play
    ↓
    Video appears?
    ├─ YES → ✅ Camera ready! → Go to "Integrate with Tracker"
    └─ NO → Troubleshoot RTSP
```

**Troubleshoot RTSP:**
```
No video in VLC
├─ Check password in URL is correct
├─ Verify RTSP enabled: Camera → Settings → Network → Port Settings
├─ Try main stream: ...h264Preview_01_main
├─ Check firewall: allow port 554
└─ Try direct connection: laptop → PoE injector → camera (bypass network)
```

---

### Integrate with Tracker

```
🔗 Add to tracker
    ↓
    Edit config.json:
    "url": "rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub"
    ↓
    Save file
    ↓
    Run: python tracker.py
    ↓
    Video window appears with camera feed?
    ├─ YES → ✅ SUCCESS! → Go to "Physical Mounting"
    └─ NO → Troubleshoot tracker connection
```

**Troubleshoot Tracker Connection:**
```
Tracker can't connect
├─ Test in VLC first (must work in VLC before tracker)
├─ Check URL in config.json (copy/paste from VLC)
├─ Verify no typos in password
├─ Check Terminal for error messages
└─ Try: ping 192.168.1.XXX (camera should respond)
```

---

### Physical Mounting

```
🏠 Ready to mount permanently
    ↓
    See: CAMERA_MOUNTING.md
    ↓
    Mount at windowsill (21-24" height)
    ↓
    Angle at 10° upward tilt
    ↓
    Run tracker and test field of view
    ↓
    Adjust angle if needed
    ↓
    ✅ DONE! Operating 24/7
```

---

## 🎯 Quick Diagnostic Questions

**Camera won't power on:**
- [ ] PoE switch/injector plugged in?
- [ ] Ethernet cable fully seated both ends?
- [ ] PoE switch LED lit for that port?
- [ ] Tried different PoE port?

**Can't find camera on network:**
- [ ] Waited full 60 seconds after power-on?
- [ ] Camera LED is green/blue (not red)?
- [ ] Checked router's DHCP/device list?
- [ ] Tried Reolink app "Scan LAN"?
- [ ] Computer on same network as camera?

**Can't access web interface:**
- [ ] Using correct IP address?
- [ ] Typed http:// before IP?
- [ ] Tried ping 192.168.1.XXX?
- [ ] Tried different browser?
- [ ] Camera responding to ping?

**RTSP not working in VLC:**
- [ ] RTSP enabled in camera settings?
- [ ] Using correct RTSP URL format?
- [ ] Password in URL is correct?
- [ ] Port 554 allowed in firewall?
- [ ] Tested with main stream URL?

**Tracker can't connect to camera:**
- [ ] RTSP works in VLC first?
- [ ] URL in config.json matches VLC?
- [ ] No typos in config.json?
- [ ] Virtual environment activated?
- [ ] Checked Terminal for errors?

---

## 🔢 Quick Setup Time Estimates

| Task | First Camera | Additional Cameras |
|------|--------------|-------------------|
| Physical connection | 5 min | 3 min |
| Find on network | 5 min | 2 min |
| Web interface setup | 10 min | 5 min |
| Configure settings | 10 min | 5 min |
| Enable RTSP | 3 min | 2 min |
| Test in VLC | 2 min | 1 min |
| Integrate with tracker | 5 min | 3 min |
| **Total** | **40 min** | **21 min** |

**For 3 cameras:** ~80 minutes total

---

## 📋 Pre-Flight Checklist

**Before starting camera setup, have ready:**

- [ ] Camera(s) unboxed
- [ ] PoE switch or injector ready
- [ ] Ethernet cables (to reach from camera to switch)
- [ ] Computer on same network
- [ ] Web browser (Chrome/Safari)
- [ ] VLC Media Player installed
- [ ] Notepad for documenting IPs and passwords
- [ ] Reolink app installed (optional but helpful)

**Tools you'll need later:**
- [ ] Hex wrench (included with camera)
- [ ] Mounting screws
- [ ] Drill (for mounting bracket)
- [ ] Tape measure (for height/angle)
- [ ] Phone with protractor app (for angle)

---

## 🎓 Learning Path

```
1. Setup Camera 1 on desk (40 min)
   ↓ Learn the process
   
2. Test with tracker on desk (10 min)
   ↓ Validate integration
   
3. Mount Camera 1 physically (20 min)
   ↓ Adjust angle, test detection
   
4. Setup Camera 2 (faster now - 20 min)
   ↓ Apply what you learned
   
5. Setup Camera 3 (even faster - 20 min)
   ↓ You're now an expert
   
6. Configure multi-camera in tracker (future)
   ↓ Calibration & homography
```

**Total time for 3-camera system:** ~2-3 hours

---

## ✅ Success Indicators

**Camera is working correctly when:**

- ✅ LED is solid green (not red/blinking)
- ✅ Appears in network scan with IP
- ✅ Web interface accessible
- ✅ Sub-stream: 640x480 configured
- ✅ IR LEDs disabled
- ✅ RTSP enabled
- ✅ Video plays smoothly in VLC
- ✅ Tracker connects and shows feed
- ✅ Person detection works
- ✅ Image quality good (not washed out)

**Ready for production when:**

- ✅ Static IP assigned
- ✅ Physically mounted at correct angle
- ✅ Detection zone tested and tuned
- ✅ All 3 cameras (if multi-camera) working
- ✅ Tracker runs for 1+ hour without issues
- ✅ Remote access configured (SSH/RDP)

---

## 🆘 When to Ask for Help

**Seek assistance if:**

- Camera never powers on (even after trying all ports)
- Can't find camera after 3+ attempts
- Camera found but can't access web interface
- RTSP enabled but VLC shows errors
- Everything works in VLC but tracker fails
- Image severely washed out even with IR off

**First try:**
1. Full power cycle (unplug 30 seconds)
2. Factory reset (hold button 10 seconds)
3. Try different PoE port/cable
4. Test camera + laptop only (bypass network)

**Resources:**
- REOLINK_SETUP.md - Detailed guide
- Reddit: r/reolinkcam
- Reolink support: support.reolink.com

---

**This flowchart gets you from unboxing to tracking in ~40 minutes per camera! 🎉**
