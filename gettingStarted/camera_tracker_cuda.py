#!/usr/bin/env python3
"""
Camera Feed with YOLO Person Tracking - CUDA Optimized
Detects and tracks people in Reolink camera feeds using NVIDIA GPU acceleration

Supports dual cameras with side-by-side display and independent tracking.
Falls back to single camera if second camera is unavailable.

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

# Camera settings - Primary camera (required)
RTSP_URL_1 = "rtsp://admin:dc31l1ng@10.42.0.76:555/h264Preview_01_main"
CAMERA_1_FPS = 30  # Camera 1: 2048x1536 @ 30fps

# Secondary camera (optional - set to None to disable)
RTSP_URL_2 = "rtsp://admin:dc31l1ng@10.42.0.173:555/h264Preview_01_main"
CAMERA_2_FPS = 30  # Camera 2: 2560x1920 @ 50fps declared, use 30 for processing

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
    print("Dual Camera YOLO Person Tracking - CUDA Optimized")
    print("=" * 60)
    
    # Check CUDA
    import torch
    import random
    
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
    
    # Connect to cameras
    cameras = []
    camera_configs = []
    
    # Camera 1 (required)
    print(f"\n📹 Connecting to Camera 1 (target: {CAMERA_1_FPS} FPS)...")
    cap1 = LowLatencyCamera(RTSP_URL_1, target_fps=CAMERA_1_FPS)
    time.sleep(1)
    
    if not cap1.isOpened():
        print("❌ ERROR: Could not open Camera 1 (required)")
        sys.exit(1)
    
    cameras.append(cap1)
    camera_configs.append({
        'name': 'Camera 1',
        'fps': CAMERA_1_FPS,
        'width': cap1.width,
        'height': cap1.height,
        'scale': PROCESS_WIDTH / cap1.width,
        'process_height': int(cap1.height * (PROCESS_WIDTH / cap1.width)),
        'track_colors': {},
        'fps_history': [],
        'last_boxes': [],
    })
    print(f"✅ Camera 1 connected! Stream: {cap1.width}x{cap1.height}")
    
    # Camera 2 (optional)
    cap2 = None
    if RTSP_URL_2:
        print(f"\n📹 Connecting to Camera 2 (target: {CAMERA_2_FPS} FPS)...")
        cap2 = LowLatencyCamera(RTSP_URL_2, target_fps=CAMERA_2_FPS)
        time.sleep(1)
        
        if cap2.isOpened():
            cameras.append(cap2)
            camera_configs.append({
                'name': 'Camera 2',
                'fps': CAMERA_2_FPS,
                'width': cap2.width,
                'height': cap2.height,
                'scale': PROCESS_WIDTH / cap2.width,
                'process_height': int(cap2.height * (PROCESS_WIDTH / cap2.width)),
                'track_colors': {},
                'fps_history': [],
                'last_boxes': [],
            })
            print(f"✅ Camera 2 connected! Stream: {cap2.width}x{cap2.height}")
        else:
            print("⚠️  Camera 2 not available, running single camera mode")
            cap2.release()
            cap2 = None
    
    num_cameras = len(cameras)
    print(f"\n🎬 Running with {num_cameras} camera(s)")
    
    # Calculate display sizes - fit both cameras side by side
    # Each camera gets half the display width in dual mode
    per_camera_width = DISPLAY_WIDTH if num_cameras == 1 else DISPLAY_WIDTH // 2
    
    display_configs = []
    for cfg in camera_configs:
        display_scale = per_camera_width / cfg['width']
        display_height = int(cfg['height'] * display_scale)
        display_configs.append({
            'width': per_camera_width,
            'height': display_height,
            'scale': display_scale,
        })
    
    # Normalize heights for side-by-side display
    if num_cameras > 1:
        max_height = max(dc['height'] for dc in display_configs)
        for dc in display_configs:
            dc['target_height'] = max_height
    else:
        display_configs[0]['target_height'] = display_configs[0]['height']
    
    print(f"   Processing at: {PROCESS_WIDTH}px width")
    print(f"   Display: {per_camera_width}px per camera")
    print(f"\n🎬 Starting tracking... Press 'q' to quit\n")
    
    # Tracking state
    frame_count = 0
    start_time = time.time()
    
    # Frame pacing
    frame_interval = 1.0 / max(CAMERA_1_FPS, CAMERA_2_FPS if RTSP_URL_2 else CAMERA_1_FPS)
    last_process_time = 0
    
    while True:
        # Frame pacing
        current_time = time.time()
        elapsed = current_time - last_process_time
        if elapsed < frame_interval * 0.9:
            time.sleep(0.001)
            continue
        
        last_process_time = time.time()
        frame_count += 1
        frame_start = time.time()
        
        display_frames = []
        total_people = 0
        
        # Process each camera
        for cam_idx, (cap, cfg, dcfg) in enumerate(zip(cameras, camera_configs, display_configs)):
            ret, frame, latency_ms = cap.read()
            
            if not ret or frame is None:
                # Create placeholder frame
                placeholder = np.zeros((dcfg['target_height'], dcfg['width'], 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{cfg['name']} - No Signal", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                display_frames.append(placeholder)
                continue
            
            # Run YOLO tracking
            small_frame = processor.resize(frame, (PROCESS_WIDTH, cfg['process_height']))
            
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
            
            # Extract detections
            cfg['last_boxes'] = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1 / cfg['scale']), int(x2 / cfg['scale'])
                    y1, y2 = int(y1 / cfg['scale']), int(y2 / cfg['scale'])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    cfg['last_boxes'].append((x1, y1, x2, y2, conf, track_id))
            
            # Calculate FPS
            process_time = time.time() - frame_start
            instant_fps = 1.0 / max(process_time, 0.001)
            cfg['fps_history'].append(instant_fps)
            if len(cfg['fps_history']) > cfg['fps']:
                cfg['fps_history'].pop(0)
            fps = sum(cfg['fps_history']) / len(cfg['fps_history'])
            
            # Prepare display frame
            display_frame = processor.resize(frame, (dcfg['width'], dcfg['height']))
            
            # Pad to target height if needed
            if display_frame.shape[0] < dcfg['target_height']:
                padding = dcfg['target_height'] - display_frame.shape[0]
                display_frame = cv2.copyMakeBorder(
                    display_frame, 0, padding, 0, 0, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            
            # Draw detections
            for (x1, y1, x2, y2, conf, track_id) in cfg['last_boxes']:
                dx1 = int(x1 * dcfg['scale'])
                dy1 = int(y1 * dcfg['scale'])
                dx2 = int(x2 * dcfg['scale'])
                dy2 = int(y2 * dcfg['scale'])
                
                # Unique color per track ID (offset by camera to avoid collisions)
                color_id = track_id + (cam_idx * 1000)
                if color_id not in cfg['track_colors']:
                    random.seed(color_id)
                    cfg['track_colors'][color_id] = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)
                    )
                
                color = cfg['track_colors'].get(color_id, (0, 255, 0))
                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
                
                label = f"P{track_id} ({conf:.0%})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(display_frame, (dx1, dy1 - 16), (dx1 + label_size[0], dy1), color, -1)
                cv2.putText(display_frame, label, (dx1, dy1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw info overlay
            person_count = len(cfg['last_boxes'])
            total_people += person_count
            
            info_bg = display_frame.copy()
            cv2.rectangle(info_bg, (5, 5), (180, 90), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(display_frame, 1, info_bg, 0.5, 0)
            
            # Latency color coding
            latency_color = (0, 255, 0) if latency_ms < 50 else (0, 255, 255) if latency_ms < 100 else (0, 100, 255)
            
            cv2.putText(display_frame, cfg['name'], (10, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(display_frame, f"People: {person_count}", (10, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Latency: {latency_ms:.0f}ms", (10, 78),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, latency_color, 1)
            
            display_frames.append(display_frame)
        
        # Combine frames side by side
        if len(display_frames) > 1:
            combined = cv2.hconcat(display_frames)
        else:
            combined = display_frames[0]
        
        # Add global status bar at bottom
        status_bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(status_bar, f"Total People: {total_people} | Cameras: {num_cameras} | Press 'q' to quit",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        combined = cv2.vconcat([combined, status_bar])
        
        window_title = f"Person Tracker - {num_cameras} Camera(s)"
        cv2.imshow(window_title, combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    print(f"\n🛑 Stopped")
    print(f"   Frames processed: {frame_count}")
    print(f"   Average FPS: {frame_count/elapsed:.1f}")
    for i, cap in enumerate(cameras):
        print(f"   Camera {i+1} frames dropped: {cap.frames_dropped}")


if __name__ == "__main__":
    main()
