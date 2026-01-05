#!/usr/bin/env python3
"""
Camera Feed with YOLO Person Tracking
Detects and tracks people in the Reolink camera feed

Usage:
    python camera_tracker.py

Press 'q' to quit
"""

import cv2
import sys
import time
import threading
from ultralytics import YOLO

# Camera settings - using main stream for higher FPS
RTSP_URL = "rtsp://admin:dc31l1ng@192.168.2.2:555/h264Preview_01_main"

# YOLO settings
MODEL_NAME = "yolov8n.pt"  # nano model - fast
CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS_ID = 0  # COCO class 0 = person

# Performance settings
PROCESS_WIDTH = 416  # Resize to this width for YOLO (smaller = faster)
DISPLAY_WIDTH = 960  # Resize display to this width (smaller = faster rendering)
SKIP_FRAMES = 1  # Process every Nth frame (1 = every frame)
USE_GPU = True  # Use MPS (Apple Silicon) or CUDA if available


class ThreadedCamera:
    """Threaded camera capture to prevent frame buffering lag"""
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
        """Get the most recent frame"""
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def get(self, prop):
        return self.cap.get(prop)
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

def main():
    print("=" * 50)
    print("Camera Feed with YOLO Person Tracking")
    print("=" * 50)
    
    # Detect device
    import torch
    if USE_GPU and torch.backends.mps.is_available():
        device = "mps"
        print(f"\n🚀 Using Apple Silicon GPU (MPS)")
    elif USE_GPU and torch.cuda.is_available():
        device = "cuda"
        print(f"\n🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print(f"\n💻 Using CPU")
    
    # Load YOLO model
    print(f"📦 Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    model.to(device)
    print("✅ Model loaded!")
    
    # Open camera with threaded capture to prevent lag
    print(f"\n📹 Connecting to camera...")
    cap = ThreadedCamera(RTSP_URL)
    time.sleep(1)  # Give thread time to grab first frame
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera stream")
        sys.exit(1)
    
    print("✅ Connected to camera!")
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Stream: {width}x{height} @ {stream_fps:.0f} FPS")
    print(f"   Processing at: {PROCESS_WIDTH}px width")
    print(f"   Display at: {DISPLAY_WIDTH}px width")
    print("\n🎬 Starting tracking... Press 'q' to quit\n")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    last_boxes = []  # Store last detection results
    
    # Track IDs and their colors
    track_colors = {}
    
    # Calculate resize ratios
    scale = PROCESS_WIDTH / width
    process_height = int(height * scale)
    
    display_scale = DISPLAY_WIDTH / width
    display_height = int(height * display_scale)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️  Frame grab failed, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        frame_count += 1
        
        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
        
        # Only run YOLO on every Nth frame
        if frame_count % SKIP_FRAMES == 0:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, process_height))
            
            # Run YOLO tracking with ByteTrack (faster than BoTSORT)
            results = model.track(
                small_frame, 
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                imgsz=PROCESS_WIDTH,
                tracker="bytetrack.yaml"  # Faster tracker
            )
            
            # Store results scaled back to original resolution
            last_boxes = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Scale coordinates back to original size
                    x1, x2 = int(x1 / scale), int(x2 / scale)
                    y1, y2 = int(y1 / scale), int(y2 / scale)
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    last_boxes.append((x1, y1, x2, y2, conf, track_id))
        
        # Draw last known detections
        person_count = len(last_boxes)
        
        # Downsample for display first
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_height))
        
        for (x1, y1, x2, y2, conf, track_id) in last_boxes:
            # Scale box coordinates to display size
            dx1 = int(x1 * display_scale)
            dy1 = int(y1 * display_scale)
            dx2 = int(x2 * display_scale)
            dy2 = int(y2 * display_scale)
            
            # Assign color to track ID
            if track_id not in track_colors:
                import random
                random.seed(track_id)
                track_colors[track_id] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
            
            color = track_colors.get(track_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
            
            # Draw label
            label = f"Person {track_id} ({conf:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (dx1, dy1 - 20), (dx1 + label_size[0], dy1), color, -1)
            cv2.putText(display_frame, label, (dx1, dy1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw info overlay on display frame
        info_bg = display_frame.copy()
        cv2.rectangle(info_bg, (5, 5), (250, 90), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 1, info_bg, 0.5, 0)
        
        cv2.putText(display_frame, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow("Person Tracker", display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n🛑 Stopped")
    print(f"   Frames processed: {frame_count}")
    print(f"   Average FPS: {frame_count/elapsed:.1f}")

if __name__ == "__main__":
    main()
