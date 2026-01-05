#!/usr/bin/env python3
"""
Basic Camera Feed Display
Tests the Reolink camera RTSP stream

Usage:
    python camera_test.py

Press 'q' to quit
"""

import cv2
import sys

# Camera settings
RTSP_URL = "rtsp://admin:dc31l1ng@192.168.2.2:555/h264Preview_01_main"

def main():
    print("=" * 50)
    print("Camera Feed Test")
    print("=" * 50)
    print(f"\nConnecting to: {RTSP_URL.replace('dc31l1ng', '****')}")
    print("\nPress 'q' to quit\n")
    
    # Open the RTSP stream
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera stream")
        print("\nTroubleshooting:")
        print("  1. Check camera is powered on")
        print("  2. Verify IP address: ping 192.168.2.2")
        print("  3. Test in VLC first")
        sys.exit(1)
    
    print("✅ Connected to camera!")
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print("\n🎬 Displaying feed...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️  Frame grab failed, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue
        
        frame_count += 1
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Reolink Camera Feed", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n🛑 Stopped")
    print(f"   Total frames displayed: {frame_count}")

if __name__ == "__main__":
    main()
