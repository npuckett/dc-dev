# Reolink RLC-520A Camera Setup Guide

Complete guide for setting up, configuring, and connecting your Reolink RLC-520A cameras for the person tracker.

---

## ⚡ Have a PoE Router? Start Here!

**If your router has built-in PoE ports (computer and camera plug into same router):**

→ **Read: SIMPLIFIED_SETUP_POE_ROUTER.md** for your streamlined setup!

**Your setup is simpler:**
- ✅ No PoE switch needed
- ✅ No PoE injectors needed  
- ✅ Computer and camera automatically on same network
- ✅ Setup time: ~30 minutes

**This guide below is comprehensive and covers all setup scenarios.**

---

## 📦 What's In The Box

Your Reolink RLC-520A includes:
- 📷 RLC-520A dome camera
- 🔌 PoE ethernet cable (~3-6 feet, depending on package)
- 🔩 Mounting bracket and screws
- 📄 Quick start guide
- 🔧 Hex wrench (for angle adjustment)

**What You Still Need:**
- PoE switch or PoE injector (if your network switch doesn't have PoE)
- Longer ethernet cable (if needed to reach your router/switch)
- Computer for initial configuration

---

## 🎯 Setup Overview

```
1. Physical Setup (10 min)
   ↓
2. Power & Network Connection (5 min)
   ↓
3. Find Camera on Network (5 min)
   ↓
4. Initial Configuration (10 min)
   ↓
5. Enable RTSP (5 min)
   ↓
6. Test Connection (5 min)
   ↓
7. Integrate with Tracker (2 min)
```

**Total Time:** 30-45 minutes per camera

---

## 🔧 Part 1: Physical Setup (Desktop Testing First)

### Step 1: Don't Mount It Yet!

**Do initial setup on your desk first:**
- Easier to access camera
- Can see LED indicators
- Easy to reset if needed
- Verify everything works before mounting

### Step 2: Understand the Camera

**Physical features:**
```
     _______________
    /               \
   |  [lens]  [LED]  |  ← Front
   |                 |
   \_______________/
        |      |
      Dome   Cable
```

**Cable has:**
- RJ45 ethernet connector (for data + power)
- No separate power adapter needed (PoE!)

**LED Indicator (during setup):**
- 🔴 Red solid = Booting up
- 🟢 Green blinking = Normal operation
- 🔵 Blue blinking = Waiting for network
- 🔴 Red blinking = Error/no network

---

## 🔌 Part 2: Power & Network Connection

### Option A: Direct to PoE Switch (Recommended)

**If you have a PoE switch:**

```
Camera ──(ethernet)──> PoE Switch ──(ethernet)──> Router ──> Internet
         (gets power)     (port 1-8)
```

**Setup:**
1. Plug camera ethernet into PoE switch port
2. Camera powers on immediately (LED lights up)
3. Switch provides both power and network
4. Wait 30-60 seconds for camera to boot

**PoE switches to consider:**
- TP-Link TL-SG1005P (5 port, ~$40) - Great for 1-2 cameras
- TP-Link TL-SG1008P (8 port, ~$70) - Good for 3+ cameras
- Netgear GS305P (5 port, ~$60) - Alternative

### Option B: PoE Injector (If No PoE Switch)

**If your regular switch doesn't have PoE:**

```
Camera ──(ethernet)──> PoE Injector ──(ethernet)──> Router
         (gets power)    (wall power)    (data only)
```

**Setup:**
1. Plug PoE injector into wall outlet
2. Connect camera to "PoE/Power+Data" port on injector
3. Connect injector's "LAN/Data" port to your router/switch
4. Camera powers on via injector

**PoE injectors to consider:**
- TP-Link TL-POE150S (~$15) - 1 camera
- BV-Tech 4-Port PoE Injector (~$35) - Up to 4 cameras

### Verification

**Camera is powered if:**
- ✅ LED indicator lights up (red initially)
- ✅ Slight motor sound (camera adjusting lens)
- ✅ LED changes to green/blue after 30-60 seconds

**If no LED:**
- Check PoE switch port is active (switch LED on)
- Verify ethernet cable is fully seated
- Try different PoE port on switch
- Test with known-good ethernet cable

---

## 🔍 Part 3: Find Camera on Network

### Method A: Reolink App (Easiest)

**Download Reolink app:**
- iOS: Search "Reolink" in App Store
- Android: Search "Reolink" in Play Store
- macOS/Windows: Download from reolink.com/software

**Steps:**
1. Open Reolink app
2. Tap "+" or "Add Device"
3. Select "Scan LAN" or "Find on Network"
4. Wait 10-20 seconds
5. Camera appears with:
   - Model: RLC-520A
   - IP address: 192.168.1.XXX (note this!)
   - Status: Online

**Write down the IP:** __________________

### Method B: Router's DHCP Page

**Access your router:**
1. Open web browser
2. Go to router admin page (usually):
   - 192.168.1.1
   - 192.168.0.1
   - 10.0.0.1
3. Login (check router label for password)
4. Find "DHCP Clients" or "Connected Devices"
5. Look for device named:
   - "Reolink"
   - "RLC-520A"
   - Or MAC address starting with: 00:0F:XX

**IP address:** __________________

### Method C: Network Scanner (Advanced)

**macOS:**
```bash
# Install nmap
brew install nmap

# Scan network
sudo nmap -sn 192.168.1.0/24

# Look for "Reolink" or "Hangzhou" in results
```

**Windows:**
- Download "Advanced IP Scanner" (free)
- Scan network
- Look for "Reolink" device

### Can't Find Camera?

**Troubleshooting:**
1. Check camera LED is green (fully booted)
2. Verify camera on same subnet as your computer
3. Try unplugging/replugging camera (wait 60 sec)
4. Check router firewall settings
5. Try reset button on camera (5 sec hold)

---

## ⚙️ Part 4: Initial Configuration

### Step 1: Access Camera Web Interface

**Using web browser:**
1. Open Chrome or Safari
2. Go to: `http://192.168.1.XXX` (your camera IP)
3. You'll see Reolink login page

**Default credentials:**
- Username: `admin`
- Password: *(blank - no password by default)*

**First login prompts:**
- Set new admin password (REQUIRED)
- Choose strong password: mix letters/numbers/symbols
- Write it down: __________________

### Step 2: Initial Setup Wizard

**The wizard will ask:**

**1. Device Name**
- Enter descriptive name: "Storefront Left" or "Camera 1"
- Helps identify in multi-camera setup

**2. Time Zone**
- Select your timezone
- Enable NTP (auto time sync)
- Server: time.google.com or pool.ntp.org

**3. Network Settings**
- Keep "DHCP" enabled (recommended for now)
- Or set static IP if you prefer:
  - IP: 192.168.1.100 (example)
  - Subnet: 255.255.255.0
  - Gateway: 192.168.1.1
  - DNS: 8.8.8.8

**4. Skip Cloud Setup (Not Needed)**
- We're using local RTSP, not cloud
- Click "Skip" or "Later"

### Step 3: Update Firmware (Recommended)

**Check for updates:**
1. Go to: Settings → Device Info
2. Current version shows (e.g., v3.1.0.956)
3. Click "Check for Update"
4. If update available, click "Upgrade"
5. Wait 5-10 minutes (don't unplug!)
6. Camera reboots automatically

---

## 🎥 Part 5: Configure Camera Settings

### Display Settings

**Navigate to:** Settings → Display

**Video Settings:**
- **Resolution (Main Stream):**
  - Set to: 2560x1920 (5MP) or 1920x1080 (HD)
  - Frame Rate: 30 FPS
  - Bitrate: 4096 Kbps (high quality)

- **Resolution (Sub Stream):** ⭐ **Important for Tracker**
  - Set to: 640x480 (VGA) or 896x512
  - Frame Rate: 30 FPS
  - Bitrate: 1024 Kbps
  - **This is what tracker will use!**

**Why sub-stream?**
- Lower bandwidth (3 cameras = less network load)
- Faster processing (better FPS on tracker)
- Main stream available for recording/viewing

### Image Settings

**Navigate to:** Settings → Image

**For Indoor-to-Outdoor (Through Window):**
- **Exposure:**
  - Mode: Auto
  - Anti-flicker: 60Hz (US) or 50Hz (Europe)
  - Shutter: Auto (or 1/100 max for motion)

- **Day/Night:**
  - Mode: Auto
  - Switch: Schedule or By Brightness
  - Sensitivity: Medium

- **White Balance:** Auto
- **Brightness:** 50 (adjust after testing)
- **Contrast:** 50 (adjust after testing)
- **Saturation:** 50
- **Sharpness:** 50

### Critical: Disable IR LEDs

**Navigate to:** Settings → Light

**Infrared Lights:**
- Mode: **Stay Off** ⚠️ **IMPORTANT!**
- Never "Auto" or "On" (causes glass reflections!)

**Why disable IR?**
- Indoor camera looking out through glass
- IR reflects off glass back into camera
- Creates white/washed out image at night
- You rely on street/ambient lighting instead

---

## 📡 Part 6: Enable RTSP Stream

### Step 1: Enable RTSP

**Navigate to:** Settings → Network → Advanced → Port Settings

**RTSP Settings:**
- ✅ Enable RTSP
- RTSP Port: 554 (default - don't change)
- Authentication: Enabled (required)

**Click "Save"**

### Step 2: Get RTSP URL

**For Reolink RLC-520A, the RTSP URLs are:**

**Main Stream (high resolution):**
```
rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_main
```

**Sub Stream (what tracker uses):** ⭐
```
rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub
```

**Replace:**
- `PASSWORD` - Your admin password
- `192.168.1.XXX` - Your camera's IP address

**Examples:**
```
# If password is "MyPass123" and IP is 192.168.1.100
rtsp://admin:MyPass123@192.168.1.100:554/h264Preview_01_sub

# If password has special characters, URL encode them:
# ! = %21, @ = %40, # = %23, $ = %24, etc.
```

**Write down your RTSP URL:**
```
Sub-stream: rtsp://admin:________________@192.168.1.___:554/h264Preview_01_sub
```

---

## 🧪 Part 7: Test RTSP Connection

### Test with VLC Media Player

**Download VLC (if not installed):**
- macOS: `brew install --cask vlc` or from videolan.org
- Windows: Download from videolan.org
- Linux: `sudo apt install vlc`

**Test the stream:**
1. Open VLC
2. Media → Open Network Stream (Cmd+N on Mac)
3. Paste your RTSP URL:
   ```
   rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub
   ```
4. Click "Play"

**Expected result:**
- ✅ Video appears within 2-5 seconds
- ✅ Image is clear and properly exposed
- ✅ Frame rate is smooth
- ✅ No "authentication failed" errors

**If video doesn't appear:**
- Check URL is correct (no typos in password)
- Verify RTSP is enabled in camera settings
- Check camera IP hasn't changed
- Try main stream URL to isolate issue
- Check firewall isn't blocking port 554

### Test Sub-Stream vs Main Stream

**Compare both streams in VLC:**

**Main stream (higher quality):**
```
rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_main
```
- Should be 1920x1080 or 2560x1920
- Higher bitrate, more detail
- More bandwidth

**Sub stream (tracker uses this):**
```
rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub
```
- Should be 640x480 or 896x512
- Lower bitrate, less detail but faster
- Better for computer vision

**Both should work!** Use sub-stream for tracker.

---

## 🔗 Part 8: Integrate with Tracker

### Update Tracker Configuration

**Edit config.json:**
```bash
cd ~/projects/storefront-tracker
nano config.json
```

**Update camera URL:**
```json
{
  "camera": {
    "url": "rtsp://admin:YOUR_PASSWORD@192.168.1.100:554/h264Preview_01_sub",
    "use_substream": true,
    "rotate_180": false,
    "mounting_height_cm": 60,
    "tilt_angle_degrees": 10
  },
  "model": {
    "device": "cuda"
  }
}
```

### Test with Tracker

**Run tracker:**
```bash
cd ~/projects/storefront-tracker
source venv/bin/activate
python tracker.py
```

**Expected result:**
- ✅ "Connecting to camera: rtsp://..." message
- ✅ "Camera connected: 640x480 @ 30fps" message
- ✅ Video window shows camera feed
- ✅ Person detection works when someone visible

**If connection fails:**
- Check URL in config.json is correct
- Verify camera accessible in VLC first
- Check firewall rules
- Try pinging camera: `ping 192.168.1.XXX`

---

## 🏠 Part 9: Assign Static IP (Recommended)

Once everything works, give camera a static IP so it doesn't change.

### Method A: Router DHCP Reservation (Best)

**On your router:**
1. Access router admin page
2. Find "DHCP Reservation" or "Static IP Assignment"
3. Add entry:
   - MAC Address: (camera's MAC - visible in router)
   - IP Address: 192.168.1.100 (or your choice)
   - Device Name: Storefront Camera 1
4. Save settings
5. Reboot camera

**Benefits:**
- IP won't change
- Easy to manage multiple cameras
- Can still use DHCP for other devices

### Method B: Camera Static IP Setting

**In camera web interface:**
1. Settings → Network → Network General
2. Change from "DHCP" to "Static"
3. Enter:
   - IP Address: 192.168.1.100
   - Subnet Mask: 255.255.255.0
   - Gateway: 192.168.1.1 (your router)
   - DNS: 8.8.8.8
4. Save and reboot camera

**Update tracker config with new IP if changed!**

---

## 📋 Multi-Camera Setup Checklist

**For 3-camera setup, repeat for each camera:**

### Camera 1 (Left)
- [ ] Connected to PoE port
- [ ] IP assigned: 192.168.1.100
- [ ] Password set: __________
- [ ] RTSP enabled and tested
- [ ] Sub-stream: 640x480 @ 30fps
- [ ] IR disabled
- [ ] VLC test passed
- [ ] RTSP URL documented

### Camera 2 (Center)
- [ ] Connected to PoE port
- [ ] IP assigned: 192.168.1.101
- [ ] Password set: __________
- [ ] RTSP enabled and tested
- [ ] Sub-stream: 640x480 @ 30fps
- [ ] IR disabled
- [ ] VLC test passed
- [ ] RTSP URL documented

### Camera 3 (Right)
- [ ] Connected to PoE port
- [ ] IP assigned: 192.168.1.102
- [ ] Password set: __________
- [ ] RTSP enabled and tested
- [ ] Sub-stream: 640x480 @ 30fps
- [ ] IR disabled
- [ ] VLC test passed
- [ ] RTSP URL documented

---

## 🔧 Advanced Configuration

### Adjust for Low-Light (If Needed)

**If sub-stream too dark at night:**

1. **Increase exposure:**
   - Settings → Image → Exposure
   - Max Shutter: 1/30 or 1/25 (slower = brighter)
   - Gain: Increase to 80-90

2. **Adjust brightness:**
   - Settings → Image → Brightness: 55-60

3. **Consider upgrading to RLC-820A ColorX:**
   - Much better low-light performance
   - Full color night vision without IR
   - Same mounting/configuration

### Optimize for Movement Detection

**For better person tracking:**

1. **Settings → Image:**
   - Shutter: 1/100 (reduces motion blur)
   - Sharpness: 60-70 (clearer edges)
   - 3D NR (Noise Reduction): Low (preserves detail)

2. **Settings → Display:**
   - H.265 encoding (more efficient)
   - Or H.264 if compatibility issues

### Network Optimization

**For 3+ cameras:**

1. **Use wired connection** (not WiFi)
2. **Gigabit switch** (not 100Mbps)
3. **Bandwidth calculation:**
   - Sub-stream: ~1 Mbps per camera
   - 3 cameras = 3 Mbps
   - Plenty of headroom on gigabit

---

## 🐛 Common Issues & Solutions

### Issue: Can't Find Camera on Network

**Solutions:**
1. Wait full 60 seconds after power on
2. Check PoE LED on switch (should be lit)
3. Connect directly to laptop + PoE injector (bypass switch)
4. Reset camera: Hold reset button 10 seconds
5. Try Reolink Device Client software (Windows/Mac)

### Issue: Authentication Failed in VLC

**Solutions:**
1. Verify password doesn't have special characters
2. Or URL-encode special characters:
   ```
   ! = %21
   @ = %40
   # = %23
   $ = %24
   & = %26
   ```
3. Try: `rtsp://admin:@IP:554/h264Preview_01_sub` (blank password)
4. Reset password in camera web interface

### Issue: Video Appears But Is Washed Out/White

**Cause:** IR LEDs reflecting off window glass

**Solution:**
1. Settings → Light → Infrared Lights → Stay Off
2. Test at night with IR disabled
3. If too dark, increase exposure/gain
4. Or add external IR illuminator outside

### Issue: Choppy/Laggy Video in Tracker

**Solutions:**
1. Verify using sub-stream (not main stream)
2. Lower sub-stream resolution to 640x480
3. Check network cable quality
4. Reduce sub-stream bitrate to 768 Kbps
5. Check GPU is being used: `nvidia-smi`

### Issue: Camera Reboots Randomly

**Possible causes:**
1. Insufficient PoE power (check switch wattage)
2. Bad ethernet cable
3. Firmware bug (update firmware)
4. Overheating (ensure ventilation)

### Issue: Can't Access Web Interface

**Solutions:**
1. Verify IP address with Reolink app
2. Try different browser (Chrome vs Safari)
3. Disable VPN if connected
4. Check computer on same subnet
5. Try reset camera

---

## 📊 Recommended Camera IP Scheme

**For organized multi-camera setup:**

```
Network: 192.168.1.0/24

Router:           192.168.1.1
Linux PC:         192.168.1.50  (static)

Camera 1 (Left):  192.168.1.100 (static)
Camera 2 (Center):192.168.1.101 (static)
Camera 3 (Right): 192.168.1.102 (static)

PoE Switch:       192.168.1.10  (if managed switch)
```

**Document your setup:**
```
Camera 1: 192.168.1.100 - Left windowsill
Camera 2: 192.168.1.101 - Center windowsill  
Camera 3: 192.168.1.102 - Right windowsill

All passwords: _______________
RTSP URLs:
- Cam 1: rtsp://admin:PASS@192.168.1.100:554/h264Preview_01_sub
- Cam 2: rtsp://admin:PASS@192.168.1.101:554/h264Preview_01_sub
- Cam 3: rtsp://admin:PASS@192.168.1.102:554/h264Preview_01_sub
```

---

## ✅ Setup Complete Checklist

**Before considering camera setup done:**

- [ ] Camera powered via PoE (LED is green)
- [ ] IP address known and documented
- [ ] Web interface accessible
- [ ] Admin password set and recorded
- [ ] Firmware updated to latest version
- [ ] Sub-stream configured: 640x480 @ 30fps
- [ ] IR LEDs disabled ("Stay Off")
- [ ] RTSP enabled on port 554
- [ ] RTSP URL tested in VLC successfully
- [ ] Static IP assigned (router or camera)
- [ ] Tracker connects and displays video
- [ ] Person detection working
- [ ] Image quality acceptable for tracking

---

## 🎓 Quick Reference

### Essential URLs

**Camera Web Interface:**
```
http://192.168.1.XXX
```

**RTSP Sub-Stream (for tracker):**
```
rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub
```

**RTSP Main Stream (high quality):**
```
rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_main
```

### Essential Commands

**Test camera connectivity:**
```bash
ping 192.168.1.XXX
```

**Test RTSP in VLC:**
```bash
vlc rtsp://admin:PASSWORD@192.168.1.XXX:554/h264Preview_01_sub
```

**Test with tracker:**
```bash
cd ~/projects/storefront-tracker
source venv/bin/activate
python tracker.py
```

### Common Settings

**Sub-stream for tracker:**
- Resolution: 640x480 VGA
- Frame rate: 30 FPS
- Bitrate: 1024 Kbps
- H.264 encoding

**Critical settings:**
- IR LEDs: Stay Off
- RTSP: Enabled (port 554)
- Authentication: Enabled

---

## 📞 Support Resources

**Reolink Resources:**
- Official support: support.reolink.com
- Downloads: reolink.com/download-center
- Manual: support.reolink.com → Search "RLC-520A"
- Forums: reddit.com/r/reolinkcam

**Tracker Integration:**
- Check QUICK_REFERENCE.md for commands
- Check CAMERA_MOUNTING.md for physical setup
- Check README.md for configuration

---

**Camera setup complete!** Now you're ready to mount it physically and start tracking. 🎉
