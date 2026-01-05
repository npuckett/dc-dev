# Your Simplified Setup - PoE Router Configuration

**You have: Computer + Camera(s) → Same PoE Router**

This is the **simplest possible setup** - no extra equipment needed!

---

## 🎯 Your Network Topology

```
                    ┌─────────────┐
                    │  Internet   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  PoE Router │ ← You have this!
                    │  (built-in  │
                    │   PoE ports)│
                    └─┬─────┬─────┘
                      │     │
          ┌───────────┘     └───────────┐
          │                             │
    ┌─────┴─────┐                 ┌────┴────┐
    │  Computer │                 │ Camera  │
    │ (Ethernet)│                 │  (PoE)  │
    └───────────┘                 └─────────┘
```

**Benefits:**
- ✅ No separate PoE switch needed - **Save $70!**
- ✅ No PoE injectors needed - **Save $45!**
- ✅ One device powers everything
- ✅ Computer and camera automatically on same subnet
- ✅ Simpler troubleshooting

---

## 📋 Simplified Camera Setup Steps

### Step 1: Connect Camera (2 minutes)

```
1. Plug camera ethernet cable into PoE port on router
   ↓
2. Camera powers on immediately (LED lights up)
   ↓
3. Wait 60 seconds for camera to boot
   ↓
4. LED turns green → Camera ready!
```

**Your PoE router provides both:**
- Power (48V via ethernet)
- Network connection

**No separate power needed!**

---

### Step 2: Find Camera IP (5 minutes)

**Method A: Router Admin Page (Easiest for you)**

Since computer and camera are on same router:

```bash
1. Open browser on your computer
2. Go to your router's IP (probably):
   - 192.168.1.1
   - 10.0.1.1
   - Or check router label
3. Login to router admin
4. Find "Connected Devices" or "DHCP Clients"
5. Look for "Reolink" or "RLC-520A"
6. Note the IP address: 192.168.1.___
```

**Method B: Reolink App**
1. Download Reolink app on phone
2. Connect phone to same WiFi
3. Tap "Scan LAN"
4. Camera appears with IP

**Write down camera IP:** ___________________

---

### Step 3: Configure Camera (15 minutes)

**Access camera:**
```bash
# On your computer (already on same network!)
http://192.168.1.___ (camera's IP)

Login:
- Username: admin
- Password: (blank - first time)
```

**Critical settings:**

**1. Set password**
- Choose strong password
- Write it down: _______________

**2. Configure sub-stream**
- Settings → Display → Sub Stream
- Resolution: **640x480**
- Frame rate: **30 FPS**
- Bitrate: 1024 Kbps

**3. Disable IR LEDs** ⚠️ **CRITICAL!**
- Settings → Light → Infrared Lights
- Mode: **"Stay Off"**
- (Prevents glass reflections)

**4. Enable RTSP**
- Settings → Network → Port Settings
- ✅ Enable RTSP
- Port: 554 (default)

**5. Save all settings**

---

### Step 4: Test RTSP (5 minutes)

**Your RTSP URL format:**
```
rtsp://admin:YOUR_PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

**Example:**
```
Password: MyPass123
Camera IP: 192.168.1.100

Full URL:
rtsp://admin:MyPass123@192.168.1.100:554/h264Preview_01_sub
```

**Test in VLC:**
1. Open VLC Media Player
2. File → Open Network Stream
3. Paste your RTSP URL
4. Click Play
5. Video should appear in 2-5 seconds ✅

---

### Step 5: Integrate with Tracker (5 minutes)

**Edit tracker config:**
```bash
cd ~/projects/storefront-tracker
nano config.json
```

**Update URL:**
```json
{
  "camera": {
    "url": "rtsp://admin:YOUR_PASSWORD@192.168.1.___:554/h264Preview_01_sub",
    "use_substream": true,
    "mounting_height_cm": 60,
    "tilt_angle_degrees": 10
  },
  "model": {
    "device": "cuda",
    "confidence": 0.45
  }
}
```

**Run tracker:**
```bash
source venv/bin/activate
python tracker.py
```

**Success = Camera feed appears with person detection!** 🎉

---

## 🎯 Your Advantages

### Advantage 1: No Network Complexity

**You don't need to worry about:**
- ❌ Separate PoE switch
- ❌ Multiple network hops
- ❌ Switch configuration
- ❌ Network loops
- ❌ Extra cables

**Everything is direct:**
```
Computer ──┐
           ├─ Router ─ Internet
Camera  ───┘
```

### Advantage 2: Guaranteed Same Subnet

**Computer and camera automatically:**
- ✅ Same network (192.168.1.X)
- ✅ Can see each other
- ✅ No routing needed
- ✅ No VLAN issues

### Advantage 3: Easier Troubleshooting

**If camera not working:**
1. Check PoE port LED on router (should be lit)
2. Check router's device list
3. Try different PoE port on router
4. Everything visible in one place!

---

## 🔧 PoE Router Port Management

### Identify PoE Ports on Your Router

**Typical PoE router has:**
- 4-8 ethernet ports
- Some ports labeled "PoE" or have ⚡ symbol
- Or all ports support PoE

**Check your router's label or manual**

### Recommended Port Assignment

```
Router Port Layout:
┌──────────────────────────────────────┐
│ [1] [2] [3] [4] [5] [6] [7] [8] [WAN]│
│ PoE PoE PoE PoE                      │
└──────────────────────────────────────┘

Port 1: Camera 1 (Left)
Port 2: Camera 2 (Center)
Port 3: Camera 3 (Right)
Port 4: Future camera / spare
Port 5: Computer (no PoE needed)
Port 6-7: Other devices
Port 8: Available
WAN: Internet connection
```

**Label ports with tape for future reference!**

---

## 📊 Bandwidth Considerations

**Your setup uses:**

**Per camera (sub-stream):**
- Resolution: 640x480
- Bitrate: ~1 Mbps
- Frame rate: 30 fps

**Total for 3 cameras:**
- 3 cameras × 1 Mbps = **3 Mbps**
- Plus computer traffic: ~10-50 Mbps typical
- **Total: ~15-55 Mbps**

**Your router (likely gigabit = 1000 Mbps):**
- Available: 1000 Mbps
- Used: 15-55 Mbps
- **Headroom: 945+ Mbps** ✅

**No bandwidth issues! Plenty of capacity.**

---

## ⚡ Power Budget Check

**Each Reolink RLC-520A uses:**
- Power: ~6-8 Watts
- PoE standard: 802.3af (15.4W available per port)

**3 cameras total:**
- 3 × 8W = **24 Watts**

**Your PoE router likely provides:**
- Total PoE budget: 60-120W (check router specs)
- Per port: 15.4W (802.3af) or 30W (802.3at)

**Plenty of power for 3 cameras!** ✅

**To verify:** Check your router's documentation for total PoE wattage.

---

## 🎯 Quick Setup Checklist

**Since you already have PoE router:**

- [ ] Identify PoE ports on router (check label)
- [ ] Plug camera into PoE port
- [ ] Wait 60 seconds (camera boots)
- [ ] Check router's device list for camera IP
- [ ] Access camera web interface: http://CAMERA_IP
- [ ] Set password and configure settings
- [ ] Disable IR LEDs ("Stay Off")
- [ ] Enable RTSP (port 554)
- [ ] Test in VLC with RTSP URL
- [ ] Update tracker config.json
- [ ] Run tracker - detect people!

**Total time: ~30 minutes** (no extra hardware needed!)

---

## 📝 Network Information Template

**Fill this out for easy reference:**

```
Network Setup
=============
Router IP: _______________
Router Admin: username: _______ password: _______

Computer:
- IP: _______________ (likely 192.168.1.X)
- Connected to router port: ___

Camera 1:
- IP: 192.168.1.___ (assign static: 192.168.1.100)
- Password: _______________
- Router port: #1 (PoE)
- RTSP URL: rtsp://admin:PASS@192.168.1.100:554/h264Preview_01_sub

Camera 2: (future)
- IP: 192.168.1.___ (assign static: 192.168.1.101)
- Password: _______________ (use same as Camera 1)
- Router port: #2 (PoE)
- RTSP URL: rtsp://admin:PASS@192.168.1.101:554/h264Preview_01_sub

Camera 3: (future)
- IP: 192.168.1.___ (assign static: 192.168.1.102)
- Password: _______________ (use same as Camera 1)
- Router port: #3 (PoE)
- RTSP URL: rtsp://admin:PASS@192.168.1.102:554/h264Preview_01_sub
```

---

## 🚀 Your Setup is Simpler!

**What you DON'T need to do:**
- ❌ Buy PoE switch ($70 saved!)
- ❌ Buy PoE injectors ($45 saved!)
- ❌ Configure switch settings
- ❌ Worry about network topology
- ❌ Deal with extra cables/power adapters

**What you DO:**
1. Plug camera into router PoE port
2. Find IP on router's device page
3. Configure camera settings
4. Update tracker config
5. Done! ✅

**From camera arrival to tracking: ~45 minutes**

---

## 💡 Pro Tip: Static IP via Router

**Assign static IPs through router DHCP reservation:**

1. Access router admin page
2. Find "DHCP Reservation" or "Address Reservation"
3. For camera's MAC address, assign:
   - Camera 1: 192.168.1.100
   - Camera 2: 192.168.1.101
   - Camera 3: 192.168.1.102

**Benefits:**
- IP never changes
- Tracker config stays valid
- Easy to remember and document

---

## ✅ Success Criteria

**Your setup is working when:**

- ✅ Camera LED is green (powered by router)
- ✅ Camera appears in router's device list
- ✅ Can access camera web interface from computer
- ✅ RTSP stream plays in VLC
- ✅ Tracker shows camera feed
- ✅ Person detection works

**All on one simple network!**

---

## 🆘 Simplified Troubleshooting

**Camera won't power on:**
- Check router's PoE port LED (should light up)
- Try different PoE port on router
- Verify ethernet cable fully seated

**Can't find camera:**
- Check router's admin page → Connected Devices
- Camera may take 60-90 seconds to appear
- Try unplugging/replugging camera

**Can't access web interface:**
- Verify computer and camera on same network (both plugged into same router = guaranteed!)
- Try pinging camera: `ping 192.168.1.XXX`
- Check firewall settings on computer

**RTSP not working:**
- Verify RTSP enabled in camera (Settings → Network → Port Settings)
- Test URL in VLC before trying tracker
- Check password in URL is correct

---

**Your setup is actually IDEAL - everything you need, nothing you don't!** 🎯

The simplified path: Plug in camera → Configure → Track people. No extra hardware required!
