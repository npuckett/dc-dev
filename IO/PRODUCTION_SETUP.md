# Drop Ceiling - Production Setup Guide

Complete guide for deploying the installation on a Linux production machine.

**Deployment method**: SSH into production machine via Tailscale, then install from local repo.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Remote Access](#remote-access)
3. [Initial Setup](#initial-setup)
4. [Python Environment](#python-environment)
5. [Camera Calibration](#camera-calibration)
6. [Network Configuration](#network-configuration)
7. [Service Installation](#service-installation)
8. [Tailscale Funnel Setup](#tailscale-funnel-setup)
9. [Testing & Verification](#testing--verification)
10. [Monitoring & Logs](#monitoring--logs)
11. [Troubleshooting](#troubleshooting)
12. [Maintenance](#maintenance)

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (for YOLO tracking)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB for OS + application + logs
- **Network**: Ethernet to PoE switch for cameras

### Software
- Ubuntu 22.04 LTS or similar Linux distribution
- Python 3.10+
- NVIDIA drivers + CUDA toolkit
- Tailscale (for remote access)

### Network
- Production machine: Static IP (e.g., 10.42.0.1)
- Cameras: RTSP streams accessible
- Art-Net DMX decoder: 10.42.0.200

---

## Remote Access

The production machine is on restricted corporate WiFi. Access via Tailscale.

### 1. SSH via Tailscale

```bash
# From your local machine, connect via Tailscale IP
ssh dc@100.x.x.x

# Or use Tailscale hostname
ssh dc@production-machine
```

### 2. Verify Connection

```bash
# Check you're on the production machine
hostname
nvidia-smi
```

---

## Initial Setup

The repo is already cloned on the production machine. Pull latest changes:

### 1. Update Repository

```bash
cd /home/nick/Documents/Github/dc-dev
git pull origin main
```

### 2. Verify NVIDIA GPU

```bash
# Check GPU is recognized
nvidia-smi

# Should show your GPU model and driver version
```

### 3. BIOS Configuration (already done)

Ensure these are set:
- **Power Recovery**: Always On (auto-start after power loss)
- **Wake on LAN**: Disabled (unless needed)

---

## Python Environment

### 1. Create Virtual Environment (if not exists)

```bash
cd /home/nick/Documents/Github/dc-dev
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install opencv-python-headless
pip install python-osc
pip install numpy

# Light controller dependencies
pip install pygame PyOpenGL PyOpenGL_accelerate
pip install stupidArtnet
pip install websockets

# Verify CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 3. Download YOLO Model

```bash
# The tracker will auto-download, but you can pre-download:
cd IO
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

---

## Camera Calibration

Calibration must be done on-site with the actual cameras.

### 1. Print Calibration Markers

```bash
# Markers are in the calibration/ folder
# Print markers 0-6 at 15cm size
ls calibration/marker_*.png
```

### 2. Place Markers

See `calibration/CALIBRATION_GUIDE.md` for marker placement:

```
Front Row (Z=90):   [0]----[1]----[2]
Back Row (Z=141):   [3]----[6]----[4]
Subway Wall:              [5]
```

### 3. Run Calibration

```bash
source .venv/bin/activate
cd calibration
python camera_tracker_cuda.py
```

Press `C` for calibration mode, then `A` for auto-calibration.

### 4. Verify Calibration Saved

```bash
ls -la calibration/camera_calibration.json
```

---

## Network Configuration

### 1. Static IP for Production Machine

Edit `/etc/netplan/01-network.yaml`:

```yaml
network:
  version: 2
  ethernets:
    eth0:  # or your interface name
      addresses:
        - 10.42.0.1/24
      routes:
        - to: default
          via: 10.42.0.254  # your router
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Apply:
```bash
sudo netplan apply
```

### 2. Verify Camera Access

```bash
# Test RTSP streams
ffprobe rtsp://admin:dc31l1ng@10.42.0.75:555/h264Preview_01_main
ffprobe rtsp://admin:dc31l1ng@10.42.0.172:555/h264Preview_01_main
```

### 3. Verify Art-Net Decoder

```bash
ping 10.42.0.200
```

---

## Service Installation

### 1. Review Service Files

Check paths match your installation:

```bash
cat IO/systemd/camera-tracker.service
cat IO/systemd/light-controller.service
```

Key settings to verify:
- `User=dc` - your username
- `WorkingDirectory=/home/nick/Documents/Github/dc-dev/IO`
- `ExecStart` path to Python

### 2. Install Services

```bash
cd /home/nick/Documents/Github/dc-dev/IO/systemd
sudo chmod +x install-services.sh
sudo ./install-services.sh
```

### 3. Start Services

```bash
# Start in order (camera tracker first)
sudo systemctl start camera-tracker
sleep 5
sudo systemctl start light-controller
sudo systemctl start tailscale-funnel
```

### 4. Verify Services Running

```bash
# Quick status
./status.sh

# Detailed status
sudo systemctl status camera-tracker light-controller
```

---

## Tailscale Funnel Setup

Tailscale Funnel exposes the WebSocket server for the public viewer.

### 1. Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled
```

### 2. Authenticate

```bash
sudo tailscale up
# Follow the URL to authenticate
```

### 3. Enable Funnel in Admin Console

1. Go to: https://login.tailscale.com/admin/acls
2. Add to your ACL policy:
```json
{
  "nodeAttrs": [
    {"target": ["*"], "attr": ["funnel"]}
  ]
}
```

### 4. Start Funnel Service

```bash
sudo systemctl start tailscale-funnel
```

### 5. Get Your Funnel URL

```bash
tailscale funnel status
# Shows: https://your-machine.tail12345.ts.net:8765
```

---

## Testing & Verification

### 1. Test Camera Tracker

```bash
# View live logs
journalctl -u camera-tracker -f

# Should see:
# ✅ Model loaded!
# 📹 Connecting to cameras...
# ✓ Camera 1 connected
# 📡 OSC output: 127.0.0.1:7000
```

### 2. Test Light Controller

```bash
journalctl -u light-controller -f

# Should see:
# 📡 OSC server listening on 0.0.0.0:7000
# 🌐 WebSocket server started on port 8765
# 📥 OSC: messages received
```

### 3. Test WebSocket (from another machine)

```bash
# Using websocat (install: cargo install websocat)
websocat wss://your-machine.tail12345.ts.net:8765

# Should receive JSON state updates
```

### 4. Test Public Viewer

Open in browser:
```
https://yourusername.github.io/dc-dev/public-viewer/?ws=wss://your-machine.tail12345.ts.net:8765
```

---

## Monitoring & Logs

### View Live Logs

```bash
# All services combined
journalctl -u camera-tracker -u light-controller -f

# Camera tracker only
journalctl -u camera-tracker -f

# Light controller only
journalctl -u light-controller -f

# Last hour
journalctl -u camera-tracker --since "1 hour ago"
```

### Health Check Dashboard

```bash
./IO/systemd/status.sh
```

### System Resource Usage

```bash
# GPU usage
nvidia-smi -l 1

# CPU/Memory
htop
```

### Log Rotation

Logs are managed by journald. Configure retention in `/etc/systemd/journald.conf`:

```ini
[Journal]
SystemMaxUse=1G
MaxRetentionSec=7day
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check detailed logs
journalctl -u camera-tracker -n 100 --no-pager

# Common issues:
# - Python path wrong: verify .venv location
# - Missing dependencies: run pip install again
# - GPU not available: check nvidia-smi
```

### Camera Connection Failed

```bash
# Test RTSP directly
ffplay rtsp://admin:dc31l1ng@10.42.0.75:555/h264Preview_01_main

# Check network
ping 10.42.0.75
```

### Art-Net Not Working

```bash
# Check decoder is reachable
ping 10.42.0.200

# Verify port 6454 (Art-Net) not blocked
sudo ufw status
```

### WebSocket Connection Refused

```bash
# Check Funnel status
tailscale funnel status

# Check light controller is running
sudo systemctl status light-controller

# Check port 8765 is listening
ss -tlnp | grep 8765
```

### Service Keeps Restarting

```bash
# Check for crash loop
systemctl show camera-tracker --property=NRestarts

# If > 5 restarts in 60 seconds, systemd stops trying
# Reset counter:
sudo systemctl reset-failed camera-tracker
sudo systemctl start camera-tracker
```

### Display/OpenGL Issues

For headless servers without a display:

```bash
# Option 1: Use virtual framebuffer
sudo apt install xvfb
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Option 2: Disable pygame display (edit light-controller.service)
Environment="SDL_VIDEODRIVER=dummy"
```

---

## Maintenance

### Daily (Automatic)

- Services auto-restart on crash
- Logs auto-rotate via journald
- Database auto-prunes records older than 7 days

### Weekly

```bash
# Check disk space
df -h

# Check log size
journalctl --disk-usage

# Review any service restarts
journalctl -u camera-tracker -u light-controller | grep -i "start\|restart\|failed"
```

### Monthly

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Check for Python package updates (be careful!)
pip list --outdated
```

### After Power Outage

Services will auto-start. Verify via SSH:

```bash
ssh dc@production-machine
./dc-dev/IO/systemd/status.sh
```

### Updating the Code

From your local machine, push changes then SSH to pull:

```bash
# LOCAL: Push your changes
cd ~/Documents/GitHub/dc-dev
git add -A && git commit -m "Update" && git push

# SSH into production machine
ssh nick@production-machine

# REMOTE: Stop services, pull, restart
cd /home/nick/Documents/Github/dc-dev
sudo systemctl stop camera-tracker light-controller
git pull
sudo systemctl start camera-tracker light-controller
```

Or as a one-liner from local:

```bash
ssh nick@production-machine "cd ~/Documents/Github/dc-dev && sudo systemctl stop camera-tracker light-controller && git pull && sudo systemctl start camera-tracker light-controller"
```

---

## Quick Reference

### Service Commands

| Action | Command |
|--------|---------|
| Start all | `sudo systemctl start camera-tracker light-controller tailscale-funnel` |
| Stop all | `sudo systemctl stop camera-tracker light-controller` |
| Restart all | `sudo systemctl restart camera-tracker light-controller` |
| Status | `./IO/systemd/status.sh` |
| Logs | `journalctl -u camera-tracker -u light-controller -f` |
| Disable auto-start | `sudo systemctl disable camera-tracker` |

### Key Ports

| Port | Service | Protocol |
|------|---------|----------|
| 7000 | OSC (tracker → controller) | UDP |
| 8765 | WebSocket (controller → viewer) | TCP/WSS |
| 6454 | Art-Net (controller → DMX) | UDP |
| 555 | RTSP (cameras) | TCP |

### Key Files

| File | Purpose |
|------|---------|
| `IO/camera_tracker_osc.py` | Production camera tracker |
| `IO/lightController_osc.py` | Light controller + Art-Net + WebSocket |
| `calibration/camera_calibration.json` | Camera calibration data |
| `IO/tracking_history.db` | SQLite tracking database |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION MACHINE                               │
│                                                                          │
│  ┌─────────────────┐     OSC      ┌──────────────────────┐              │
│  │ camera_tracker  │─────7000────▶│  lightController     │              │
│  │    _osc.py      │              │     _osc.py          │              │
│  │                 │              │                      │              │
│  │  • YOLO track   │              │  • Behavior system   │              │
│  │  • 2x RTSP      │              │  • Art-Net output    │              │
│  │  • Calibration  │              │  • WebSocket server  │              │
│  └────────┬────────┘              └──────────┬───────────┘              │
│           │                                  │                          │
│           │ RTSP                    Art-Net  │  WebSocket               │
│           ▼                           6454   │    8765                  │
│   ┌───────────────┐                   │      │                          │
│   │   Cameras     │                   ▼      ▼                          │
│   │  10.42.0.75   │           ┌─────────┐  ┌─────────────┐             │
│   │  10.42.0.172  │           │   DMX   │  │  Tailscale  │             │
│   └───────────────┘           │ Decoder │  │   Funnel    │             │
│                               └────┬────┘  └──────┬──────┘             │
└────────────────────────────────────│──────────────│─────────────────────┘
                                     │              │
                                     ▼              ▼ HTTPS
                              ┌──────────┐    ┌───────────────┐
                              │   LED    │    │ GitHub Pages  │
                              │  Panels  │    │ Public Viewer │
                              └──────────┘    └───────────────┘
```
