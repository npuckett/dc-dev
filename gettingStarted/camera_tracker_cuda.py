#!/usr/bin/env python3
"""
Camera Feed with YOLO Person Tracking - CUDA Optimized
Detects and tracks people in the Reolink camera feed using NVIDIA GPU acceleration

Optimized for 25fps camera stream with low-latency processing.

Requirements:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA

Usage:
    python camera_tracker_cuda.py

Press 'q' to quit
"""

import cv2
import sys
import time
import threading
import os
import numpy as np
from ultralytics import YOLO

# Camera settings
RTSP_URL = "rtsp://admin:dc31l1ng@10.42.0.75:555/h264Preview_01_main"
CAMERA_FPS = 30  # Camera main stream at 30fps

# YOLO settings
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS_ID = 0

# Performance settings
PROCESS_WIDTH = 416
DISPLAY_WIDTH = 960

# CUDA settings
CUDA_DEVICE = 0


class LowLatencyCamera:
    """
    Low-latency camera capture optimized for 25fps.
    Uses aggressive buffer flushing to minimize latency.
    """
    def __init__(self, src, target_fps=25):
        self.src = src
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.frame = None
        self.frame_time = time.time()
        self.grabbed = False
        self.running = True
        self.lock = threading.Lock()
        self.frames_dropped = 0
        
        # Set FFmpeg environment for low latency
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay'
        
        # Use OpenCV with FFmpeg backend and minimal buffering
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Capture frames and always keep the latest one"""
        while self.running:
            grabbed, frame = self.cap.read()
            
            if grabbed:
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
                    self.frame_time = time.time()
                
                # Flush any buffered frames - always use the latest
                flush_count = 0
                while flush_count < 5:
                    grabbed2, frame2 = self.cap.read()
                    if grabbed2:
                        self.frames_dropped += 1
                        flush_count += 1
                        with self.lock:
                            self.frame = frame2
                            self.frame_time = time.time()
                    else:
                        break
            else:
                time.sleep(0.001)
    
    def read(self):
        """Get the most recent frame with latency measurement"""
        with self.lock:
            if self.frame is None:
                return False, None, 0
            
            latency_ms = (time.time() - self.frame_time) * 1000
            return self.grabbed, self.frame.copy(), latency_ms
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


class CUDAFrameProcessor:
    """GPU-accelerated frame preprocessing"""
    def __init__(self):
        self.use_cuda_cv = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_cuda_cv = True
                self.gpu_frame = cv2.cuda_GpuMat()
                self.gpu_resized = cv2.cuda_GpuMat()
                print("🎮 Using OpenCV CUDA for preprocessing")
        except Exception:
            pass
        
        if not self.use_cuda_cv:
            print("💻 Using CPU for preprocessing")
    
    def resize(self, frame, target_size):
        if self.use_cuda_cv:
            self.gpu_frame.upload(frame)
            cv2.cuda.resize(self.gpu_frame, target_size, self.gpu_resized)
            return self.gpu_resized.download()
        return cv2.resize(frame, target_size)


def main():
    print("=" * 60)
    print("Camera Feed with YOLO Person Tracking - CUDA Optimized")
    print("=" * 60)
    
    # Check CUDA
    import torch
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, using CPU")
        device = "cpu"
    else:
        device = f"cuda:{CUDA_DEVICE}"
        gpu_name = torch.cuda.get_device_name(CUDA_DEVICE)
        print(f"\n🚀 Using NVIDIA GPU: {gpu_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Load YOLO model
    print(f"\n📦 Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    model.to(device)
    
    # Warm up GPU
    if device.startswith("cuda"):
        print("   Warming up GPU...")
        dummy = torch.zeros(1, 3, PROCESS_WIDTH, PROCESS_WIDTH).to(device)
        with torch.no_grad():
            for _ in range(3):
                model.predict(dummy, verbose=False)
        del dummy
        torch.cuda.empty_cache()
    print("✅ Model loaded!")
    
    # Initialize processor
    processor = CUDAFrameProcessor()
    
    # Connect to camera
    print(f"\n📹 Connecting to camera (target: {CAMERA_FPS} FPS)...")
    cap = LowLatencyCamera(RTSP_URL, target_fps=CAMERA_FPS)
    time.sleep(1)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera")
        sys.exit(1)
    
    print(f"✅ Connected! Stream: {cap.width}x{cap.height}")
    print(f"   Processing at: {PROCESS_WIDTH}px width")
    print(f"   Display at: {DISPLAY_WIDTH}px width")
    print(f"\n🎬 Starting tracking... Press 'q' to quit\n")
    
    # Calculate scales
    scale = PROCESS_WIDTH / cap.width
    process_height = int(cap.height * scale)
    display_scale = DISPLAY_WIDTH / cap.width
    display_height = int(cap.height * display_scale)
    
    # Tracking state
    frame_count = 0
    start_time = time.time()
    last_boxes = []
    track_colors = {}
    
    # FPS tracking
    fps_history = []
    
    # Frame pacing - process at camera rate
    frame_interval = 1.0 / CAMERA_FPS
    last_process_time = 0
    
    while True:
        ret, frame, latency_ms = cap.read()
        
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        # Frame pacing - match camera FPS
        current_time = time.time()
        elapsed = current_time - last_process_time
        if elapsed < frame_interval * 0.9:
            continue
        
        last_process_time = time.time()
        frame_count += 1
        frame_start = time.time()
        
        # Run YOLO
        small_frame = processor.resize(frame, (PROCESS_WIDTH, process_height))
        
        results = model.track(
            small_frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            imgsz=PROCESS_WIDTH,
            tracker="bytetrack.yaml",
            device=device
        )
        
        last_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1 / scale), int(x2 / scale)
                y1, y2 = int(y1 / scale), int(y2 / scale)
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                last_boxes.append((x1, y1, x2, y2, conf, track_id))
        
        # Calculate FPS
        process_time = time.time() - frame_start
        instant_fps = 1.0 / max(process_time, 0.001)
        fps_history.append(instant_fps)
        if len(fps_history) > CAMERA_FPS:
            fps_history.pop(0)
        fps = sum(fps_history) / len(fps_history)
        
        # Prepare display
        display_frame = processor.resize(frame, (DISPLAY_WIDTH, display_height))
        
        # Draw detections
        for (x1, y1, x2, y2, conf, track_id) in last_boxes:
            dx1 = int(x1 * display_scale)
            dy1 = int(y1 * display_scale)
            dx2 = int(x2 * display_scale)
            dy2 = int(y2 * display_scale)
            
            if track_id not in track_colors:
                import random
                random.seed(track_id)
                track_colors[track_id] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
            
            color = track_colors.get(track_id, (0, 255, 0))
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
            
            label = f"Person {track_id} ({conf:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (dx1, dy1 - 20), (dx1 + label_size[0], dy1), color, -1)
            cv2.putText(display_frame, label, (dx1, dy1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw info overlay
        person_count = len(last_boxes)
        info_bg = display_frame.copy()
        cv2.rectangle(info_bg, (5, 5), (300, 110), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 1, info_bg, 0.5, 0)
        
        # Latency color coding
        latency_color = (0, 255, 0) if latency_ms < 50 else (0, 255, 255) if latency_ms < 100 else (0, 100, 255)
        
        cv2.putText(display_frame, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f} / {CAMERA_FPS}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Latency: {latency_ms:.0f}ms", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, latency_color, 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Person Tracker (CUDA)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    print(f"\n🛑 Stopped")
    print(f"   Frames processed: {frame_count}")
    print(f"   Average FPS: {frame_count/elapsed:.1f}")
    print(f"   Frames dropped (buffer flush): {cap.frames_dropped}")


if __name__ == "__main__":
    main()
