#!/usr/bin/env python3
"""Quick test: verify RTSP camera connections from camera_calibration.py config"""

import cv2
import time
import os
import sys

# Import camera config from the calibration script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CAMERAS = [
    {
        'name': 'Camera 1',
        'url': 'rtsp://admin:dc31l1ng@10.42.0.75:555/h264Preview_01_main',
        'enabled': True,
    },
    {
        'name': 'Camera 2',
        'url': 'rtsp://admin:dc31l1ng@10.42.0.172:555/h264Preview_01_main',
        'enabled': True,
    },
    {
        'name': 'Camera 3',
        'url': 'rtsp://admin:dc31l1ng@10.42.0.111:555/h264Preview_01_main',
        'enabled': False,
    },
]

TIMEOUT = 10  # seconds

def test_camera(name, url):
    print(f"\n{'─'*50}")
    print(f"Testing {name}: {url}")
    print(f"{'─'*50}")
    
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|'
        'fflags;nobuffer+discardcorrupt|'
        'flags;low_delay|'
        'max_delay;500000|'
        'stimeout;5000000'
    )
    
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    start = time.time()
    connected = False
    
    while time.time() - start < TIMEOUT:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ CONNECTED — Resolution: {w}x{h}")
                print(f"  ⏱  Connect time: {time.time()-start:.1f}s")
                connected = True
                break
        time.sleep(0.2)
    
    if not connected:
        print(f"  ❌ FAILED — Could not connect within {TIMEOUT}s")
        if cap.isOpened():
            print(f"     Stream opened but no frames received")
        else:
            print(f"     Stream did not open (check IP/port/credentials)")
    
    cap.release()
    return connected


if __name__ == "__main__":
    print("=" * 50)
    print("Camera Connection Test")
    print("=" * 50)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"FFMPEG backend available: {cv2.CAP_FFMPEG}")
    
    results = {}
    
    for cam in CAMERAS:
        if not cam.get('enabled', True):
            print(f"\n⏭  {cam['name']} — DISABLED (skipping)")
            results[cam['name']] = 'disabled'
            continue
        results[cam['name']] = test_camera(cam['name'], cam['url'])
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, result in results.items():
        if result == 'disabled':
            status = "⏭  DISABLED"
        elif result:
            status = "✅ OK"
        else:
            status = "❌ FAILED"
        print(f"  {name}: {status}")
    print()
