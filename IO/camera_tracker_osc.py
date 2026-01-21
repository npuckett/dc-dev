#!/usr/bin/env python3
"""
Camera Tracker with OSC Output

Tracks people using YOLO and sends their synthesized floor positions via OSC.
Based on camera_tracker_cuda.py but focused on OSC output for the light controller.

OSC Messages Sent:
  /tracker/person/<id> <x> <z>  - Position of each tracked person (cm)
  /tracker/count <n>            - Number of people currently tracked

Usage:
    python camera_tracker_osc.py

Press 'q' to quit
"""

import cv2
import numpy as np
import time
import json
import threading
import os as _os
from collections import deque

import torch
from ultralytics import YOLO

# OSC
from pythonosc import udp_client

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# OSC settings
OSC_IP = "127.0.0.1"  # Send to localhost (same machine)
OSC_PORT = 7000

# Camera configuration
CAMERAS = [
    {
        'name': 'Camera 1',
        'url': 'rtsp://admin:dc31l1ng@10.42.0.75:555/h264Preview_01_main',
        'fps': 25,
        'enabled': True,
    },
    {
        'name': 'Camera 2', 
        'url': 'rtsp://admin:dc31l1ng@10.42.0.172:555/h264Preview_01_main',
        'fps': 25,
        'enabled': True,
    },
]

# Calibration file path
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
# Look for calibration in gettingStarted folder (where original tracker saves it)
CALIBRATION_FILE = _os.path.join(_SCRIPT_DIR, '..', 'gettingStarted', 'camera_calibration.json')

# Synthesized view settings (all units in CENTIMETERS)
SYNTH_VIEW_WIDTH = 800
SYNTH_VIEW_HEIGHT = 600
SYNTH_CM_PER_PIXEL = 1.0

# YOLO settings
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS_ID = 0
BICYCLE_CLASS_ID = 1
CYCLIST_CLASS_ID = -1
TRACKED_CLASSES = [PERSON_CLASS_ID, BICYCLE_CLASS_ID]

# Cyclist detection
CYCLIST_IOU_THRESHOLD = 0.3

# Performance settings
PROCESS_WIDTH = 416
DISPLAY_WIDTH = 960
HEADLESS_MODE = False  # Set to True to disable display for maximum FPS

# Camera sync
SYNC_CAMERAS = True
TARGET_FPS = 25

# Reliability settings
CONNECTION_TIMEOUT = 10.0
RECONNECT_DELAY = 2.0
MAX_FRAME_AGE = 0.5
FRAME_CACHE_ENABLED = True

# CUDA
CUDA_DEVICE = 0

# GUI
SHOW_GUI_OVERLAY = True

# Timing diagnostics
SHOW_TIMING = True  # Print per-frame timing breakdown


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def merge_cyclists(detections):
    """Merge overlapping person + bicycle detections into cyclist detections"""
    persons = [d for d in detections if d[6] == PERSON_CLASS_ID]
    bicycles = [d for d in detections if d[6] == BICYCLE_CLASS_ID]
    others = [d for d in detections if d[6] not in (PERSON_CLASS_ID, BICYCLE_CLASS_ID)]
    
    matched_persons = set()
    matched_bicycles = set()
    cyclists = []
    
    for pi, person in enumerate(persons):
        best_iou = 0
        best_bi = -1
        
        for bi, bicycle in enumerate(bicycles):
            if bi in matched_bicycles:
                continue
            iou = compute_iou(person, bicycle)
            if iou > best_iou:
                best_iou = iou
                best_bi = bi
        
        if best_iou >= CYCLIST_IOU_THRESHOLD and best_bi >= 0:
            x1, y1, x2, y2, conf, track_id, _ = person
            bicycle = bicycles[best_bi]
            cyclist_conf = max(conf, bicycle[4])
            cyclists.append((x1, y1, x2, y2, cyclist_conf, track_id, CYCLIST_CLASS_ID))
            matched_persons.add(pi)
            matched_bicycles.add(best_bi)
    
    pedestrians = [p for pi, p in enumerate(persons) if pi not in matched_persons]
    standalone_bicycles = [b for bi, b in enumerate(bicycles) if bi not in matched_bicycles]
    
    return pedestrians + cyclists + standalone_bicycles + others


# ==============================================================================
# CALIBRATION MANAGER
# ==============================================================================

class CalibrationManager:
    """Manages 3D camera calibration for floor projection"""
    
    def __init__(self, calibration_file):
        self.calibration_file = calibration_file
        self.calibrations = {}
        self.is_calibrated = False
        self.load()
    
    def load(self):
        """Load calibration from file"""
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                for name, calib_data in data.get('cameras', {}).items():
                    self.calibrations[name] = {
                        'rvec': np.array(calib_data['rvec']),
                        'tvec': np.array(calib_data['tvec']),
                        'camera_matrix': np.array(calib_data['camera_matrix']),
                        'dist_coeffs': np.array(calib_data['dist_coeffs']),
                        'image_size': tuple(calib_data['image_size']),
                    }
                self.is_calibrated = len(self.calibrations) > 0
                if self.is_calibrated:
                    print(f"📐 Loaded 3D calibration for {len(self.calibrations)} cameras")
        except FileNotFoundError:
            print(f"⚠️ Calibration file not found: {self.calibration_file}")
        except Exception as e:
            print(f"⚠️ Failed to load calibration: {e}")
    
    def image_to_floor(self, camera_name, img_x, img_y, floor_y=0.0):
        """Project image point to floor plane. Returns (world_x, world_z) in cm."""
        if camera_name not in self.calibrations:
            return None
        
        calib = self.calibrations[camera_name]
        K = calib['camera_matrix']
        dist = calib['dist_coeffs']
        rvec = calib['rvec']
        tvec = calib['tvec']
        
        # Undistort the image point
        img_pt = np.array([[[img_x, img_y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(img_pt, K, dist, P=K)
        ux, uy = undistorted[0, 0]
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Camera position in world coordinates
        camera_pos = -R.T @ tvec.flatten()
        
        # Ray direction
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([ux, uy, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform ray to world coordinates
        ray_world = R.T @ ray_cam
        
        # Intersect with floor plane y = floor_y
        if abs(ray_world[1]) < 1e-6:
            return None
        
        t = (floor_y - camera_pos[1]) / ray_world[1]
        
        if t < 0:
            return None
        
        world_pt = camera_pos + t * ray_world
        return (world_pt[0], world_pt[2])  # Return X and Z
    
    def transform_bbox_center(self, camera_name, x1, y1, x2, y2):
        """Transform bounding box to floor position using bottom center (feet)"""
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return self.image_to_floor(camera_name, foot_x, foot_y, floor_y=0.0)


# ==============================================================================
# ROBUST CAMERA (optimized from camera_tracker_cuda.py)
# ==============================================================================

class RobustCamera:
    """
    Reliable camera capture with automatic reconnection and frame caching.
    Optimized for multi-camera synchronized tracking.
    """
    
    def __init__(self, name, src, target_fps=25):
        self.name = name
        self.src = src
        self.target_fps = target_fps
        
        self.cap = None
        self.frame = None
        self.cached_frame = None
        self.frame_time = 0
        self.frame_number = 0
        self.last_returned_frame_number = -1
        
        self.connected = False
        self.running = False
        self.lock = threading.Lock()
        
        self.width = 0
        self.height = 0
        
        self.stats = {
            'frames_received': 0,
            'frames_dropped': 0,
            'reconnect_count': 0,
        }
    
    def _connect(self):
        """Establish connection to camera"""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start = time.time()
        while time.time() - start < CONNECTION_TIMEOUT:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.width = frame.shape[1]
                self.height = frame.shape[0]
                self.connected = True
                with self.lock:
                    self.frame = frame
                    self.cached_frame = frame.copy()
                    self.frame_time = time.time()
                    self.frame_number += 1
                return True
            time.sleep(0.1)
        
        return False
    
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self._connect()
    
    def _capture_loop(self):
        """Background capture loop with buffer flushing"""
        consecutive_failures = 0
        
        while self.running:
            if not self.connected:
                time.sleep(RECONNECT_DELAY)
                self.stats['reconnect_count'] += 1
                if self._connect():
                    consecutive_failures = 0
                continue
            
            try:
                grabbed, frame = self.cap.read()
                
                if grabbed and frame is not None:
                    consecutive_failures = 0
                    
                    with self.lock:
                        self.frame = frame
                        self.cached_frame = frame.copy()
                        self.frame_time = time.time()
                        self.frame_number += 1
                        self.stats['frames_received'] += 1
                    
                    # Flush buffer - read any queued frames to get latest
                    flush_count = 0
                    while flush_count < 3:
                        grabbed2, frame2 = self.cap.read()
                        if grabbed2 and frame2 is not None:
                            self.stats['frames_dropped'] += 1
                            flush_count += 1
                            with self.lock:
                                self.frame = frame2
                                self.cached_frame = frame2.copy()
                                self.frame_time = time.time()
                        else:
                            break
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        self.connected = False
                        consecutive_failures = 0
                    time.sleep(0.01)
                    
            except Exception:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    self.connected = False
                time.sleep(0.01)
    
    def read(self):
        """Get the most recent frame"""
        with self.lock:
            if self.frame is not None:
                age = time.time() - self.frame_time
                if age < MAX_FRAME_AGE:
                    return True, self.frame.copy(), age * 1000, True
            
            if FRAME_CACHE_ENABLED and self.cached_frame is not None:
                return True, self.cached_frame.copy(), -1, False
            
            return False, None, 0, False
    
    def read_new(self):
        """
        Get frame only if it's new since last read_new() call.
        Essential for synchronized multi-camera processing.
        Returns: (success, frame, latency_ms, is_new)
        """
        with self.lock:
            is_new = self.frame_number > self.last_returned_frame_number
            
            if self.frame is not None:
                age = time.time() - self.frame_time
                
                if age < MAX_FRAME_AGE:
                    if is_new:
                        self.last_returned_frame_number = self.frame_number
                    return True, self.frame.copy(), age * 1000, is_new
            
            if FRAME_CACHE_ENABLED and self.cached_frame is not None:
                return True, self.cached_frame.copy(), -1, False
            
            return False, None, 0, False
    
    def isOpened(self):
        return self.connected
    
    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==============================================================================
# FRAME PROCESSOR
# ==============================================================================

class FrameProcessor:
    """Frame preprocessing"""
    
    def __init__(self):
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            print("💻 Using CUDA for preprocessing")
        else:
            print("💻 Using CPU for preprocessing")
    
    def resize(self, frame, target_size):
        """Resize frame"""
        if frame is None:
            return None
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Camera Tracker with OSC Output")
    print("=" * 60)
    
    # Initialize OSC client
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"\n📡 OSC output: {OSC_IP}:{OSC_PORT}")
    
    # Check CUDA
    device = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(CUDA_DEVICE)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Load YOLO
    print(f"\n📦 Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    model.to(device)
    
    if device.startswith("cuda"):
        dummy = np.zeros((PROCESS_WIDTH, PROCESS_WIDTH, 3), dtype=np.uint8)
        model.track(dummy, persist=True, verbose=False, classes=TRACKED_CLASSES)
    print("✅ Model loaded!")
    
    # Initialize
    processor = FrameProcessor()
    calibration = CalibrationManager(CALIBRATION_FILE)
    
    if not calibration.is_calibrated:
        print("\n⚠️ WARNING: No calibration loaded!")
        print("   Run camera_tracker_cuda.py and calibrate first.")
        print("   Tracking will work but floor positions will not be accurate.\n")
    
    # Connect cameras
    cameras = []
    camera_configs = []
    
    enabled_cameras = [c for c in CAMERAS if c.get('enabled', True)]
    print(f"\n📹 Connecting to {len(enabled_cameras)} cameras...")
    
    for cam_cfg in enabled_cameras:
        cam = RobustCamera(cam_cfg['name'], cam_cfg['url'], cam_cfg.get('fps', 25))
        if cam.start():
            print(f"   ✓ {cam_cfg['name']} connected: {cam.width}x{cam.height}")
            cameras.append(cam)
            # Pre-compute scale factor for this camera
            scale = PROCESS_WIDTH / cam.width
            camera_configs.append({
                'name': cam_cfg['name'],
                'camera': cam,
                'width': cam.width,
                'height': cam.height,
                'scale': scale,
                'process_height': int(cam.height * scale),
                'last_boxes': [],
                'fps_history': deque(maxlen=30),
                'current_fps': 0,
            })
        else:
            print(f"   ✗ {cam_cfg['name']} failed to connect")
    
    if len(cameras) == 0:
        print("❌ No cameras connected!")
        return
    
    print(f"\n🎬 Running with {len(cameras)} camera(s)")
    print(f"   Processing at: {PROCESS_WIDTH}px width")
    print(f"   Headless mode: {HEADLESS_MODE}")
    print(f"\n🎬 Starting tracking with OSC output...")
    print("   Press 'q' to quit\n")
    
    # Main loop
    frame_count = 0
    start_time = time.time()
    frame_interval = 1.0 / TARGET_FPS
    last_process_time = 0
    yolo_times = []  # Track YOLO inference times
    
    # Create display window (unless headless)
    if not HEADLESS_MODE:
        cv2.namedWindow("Tracker OSC", cv2.WINDOW_NORMAL)
    
    while True:
        current_time = time.time()
        
        if current_time - last_process_time < frame_interval:
            time.sleep(0.001)
            continue
        
        last_process_time = current_time
        frame_count += 1
        
        # Collect frames from all cameras
        all_world_positions = []  # (track_id, x, z)
        display_frames = []
        
        for cfg in camera_configs:
            cam = cfg['camera']
            # Use read_new() for synchronized multi-camera processing
            ret, frame, latency_ms, is_new = cam.read_new()
            
            if not ret or frame is None:
                continue
            
            # Only run YOLO on new frames (saves GPU cycles)
            if is_new:
                yolo_start = time.time()
                
                # Resize frame for YOLO (using pre-computed scale)
                small_frame = processor.resize(frame, (PROCESS_WIDTH, cfg['process_height']))
                
                # Run YOLO tracking
                results = model.track(
                    small_frame,
                    persist=True,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    classes=TRACKED_CLASSES,
                    tracker="bytetrack.yaml",
                    imgsz=PROCESS_WIDTH,
                    device=device
                )
                
                # Extract detections
                raw_detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        # Scale back to original resolution using pre-computed scale
                        x1 = x1 / cfg['scale']
                        y1 = y1 / cfg['scale']
                        x2 = x2 / cfg['scale']
                        y2 = y2 / cfg['scale']
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                        track_id = int(boxes.id[i]) if boxes.id is not None else i
                        raw_detections.append((x1, y1, x2, y2, conf, track_id, cls))
                
                cfg['last_boxes'] = merge_cyclists(raw_detections)
                
                # Track YOLO timing
                yolo_time = time.time() - yolo_start
                yolo_times.append(yolo_time)
                
                # Calculate FPS (based on YOLO time)
                instant_fps = 1.0 / max(yolo_time, 0.001)
                cfg['fps_history'].append(instant_fps)
                cfg['current_fps'] = sum(cfg['fps_history']) / len(cfg['fps_history'])
            
            # Transform detections to world coordinates
            for box in cfg['last_boxes']:
                x1, y1, x2, y2, conf, track_id = box[:6]
                class_id = box[6] if len(box) > 6 else PERSON_CLASS_ID
                
                # Only track persons and cyclists
                if class_id not in (PERSON_CLASS_ID, CYCLIST_CLASS_ID):
                    continue
                
                world_pos = calibration.transform_bbox_center(cfg['name'], x1, y1, x2, y2)
                if world_pos is not None:
                    world_x, world_z = world_pos
                    all_world_positions.append((track_id, world_x, world_z, cfg['name']))
            
            # Create display frame (skip in headless mode)
            if not HEADLESS_MODE:
                display_scale = 480 / frame.shape[1]
                display_frame = processor.resize(frame, (480, int(frame.shape[0] * display_scale)))
                
                # Draw detections
                for box in cfg['last_boxes']:
                    x1, y1, x2, y2, conf, track_id = box[:6]
                    # Scale to display size
                    dx1 = int(x1 * display_scale)
                    dy1 = int(y1 * display_scale)
                    dx2 = int(x2 * display_scale)
                    dy2 = int(y2 * display_scale)
                    
                    cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID:{track_id}", (dx1, dy1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add camera info
                cv2.putText(display_frame, f"{cfg['name']} FPS:{cfg['current_fps']:.1f}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                display_frames.append(display_frame)
        
        # Send OSC messages
        osc_start = time.time()
        osc_client.send_message("/tracker/count", len(all_world_positions))
        
        # Then send each person's position
        for track_id, world_x, world_z, cam_name in all_world_positions:
            osc_client.send_message(f"/tracker/person/{track_id}", [float(world_x), float(world_z)])
        osc_time = time.time() - osc_start
        
        # Timing diagnostics
        loop_time = time.time() - current_time
        if SHOW_TIMING and frame_count % 30 == 0:
            avg_yolo_time = sum(yolo_times) / len(yolo_times) if yolo_times else 0
            print(f"Frame {frame_count}: YOLO avg={avg_yolo_time*1000:.1f}ms, OSC={osc_time*1000:.1f}ms, Total={loop_time*1000:.1f}ms, People={len(all_world_positions)}")
            yolo_times.clear()
        
        # Create combined display (skip in headless mode)
        if not HEADLESS_MODE and display_frames:
            # Stack frames horizontally
            if len(display_frames) == 1:
                combined = display_frames[0]
            else:
                # Ensure same height
                max_h = max(f.shape[0] for f in display_frames)
                padded = []
                for f in display_frames:
                    if f.shape[0] < max_h:
                        pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                        f = cv2.vconcat([f, pad])
                    padded.append(f)
                combined = cv2.hconcat(padded)
            
            # Add OSC status bar
            status = f"OSC -> {OSC_IP}:{OSC_PORT} | People: {len(all_world_positions)} | Frame: {frame_count}"
            cv2.putText(combined, status, (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Tracker OSC", combined)
        
        # Handle keys (with or without display)
        if HEADLESS_MODE:
            # In headless mode, check for Ctrl+C via keyboard interrupt
            pass  # Loop continues until killed
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    # Cleanup
    for cam in cameras:
        cam.release()
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n🛑 Stopped")
    print(f"   Frames processed: {frame_count}")
    print(f"   Average FPS: {frame_count/elapsed:.1f}")


if __name__ == "__main__":
    main()
