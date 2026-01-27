#!/usr/bin/env python3
"""
Camera Tracker with OSC Output - V2

Tracks people using YOLO and sends their synthesized floor positions via OSC.
Based on camera_tracker_osc.py with V2 improvements:
  - Loads tracking zones from world_coordinates.json
  - Zone-based detection filtering with confidence weighting
  - Dynamic fusion threshold based on distance from cameras
  - OpenCV sliders for real-time parameter tuning
  - Persistent settings saved to JSON

OSC Messages Sent:
  /tracker/person/<id> <x> <z>  - Position of each tracked person (cm)
  /tracker/count <n>            - Number of people currently tracked
  /tracker/zone/<id> <zone>     - Zone of each person ('active', 'passive', or 'outside')

Usage:
    python camera_tracker_osc_v2.py

Press 'q' to quit, 's' to save settings
"""

import cv2
import numpy as np
import time
import json
import threading
import os as _os
import signal
import sys
import logging
import fcntl
import atexit
from collections import deque
from datetime import datetime, timedelta

import torch
from ultralytics import YOLO

# OSC
from pythonosc import udp_client

# ==============================================================================
# FILE PATHS
# ==============================================================================

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))

# V2 uses world_coordinates.json as single source of truth
WORLD_COORDS_FILE = _os.path.join(_SCRIPT_DIR, 'world_coordinates.json')

# V2 calibration file (produced by camera_calibration_v2.py)
CALIBRATION_FILE = _os.path.join(_SCRIPT_DIR, 'camera_calibration_v2.json')

# Persistent settings file for slider values
SETTINGS_FILE = _os.path.join(_SCRIPT_DIR, 'tracker_settings_v2.json')

# ==============================================================================
# SINGLE INSTANCE LOCK
# ==============================================================================

LOCK_FILE = "/tmp/camera_tracker_osc_v2.lock"
_lock_fd = None

def acquire_single_instance_lock():
    """Ensure only one instance of the tracker is running"""
    global _lock_fd
    try:
        _lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(_os.getpid()))
        _lock_fd.flush()
        return True
    except (IOError, OSError) as e:
        try:
            with open(LOCK_FILE, 'r') as f:
                existing_pid = f.read().strip()
            print(f"❌ Another camera tracker V2 is already running (PID: {existing_pid})")
        except:
            print("❌ Another camera tracker V2 is already running")
        return False

def release_single_instance_lock():
    """Release the single instance lock"""
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            _os.remove(LOCK_FILE)
        except:
            pass


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

# OSC settings
OSC_IP = "127.0.0.1"
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

# YOLO settings
MODEL_NAME = "yolo11n.pt"
PERSON_CLASS_ID = 0
BICYCLE_CLASS_ID = 1
CYCLIST_CLASS_ID = -1
TRACKED_CLASSES = [PERSON_CLASS_ID, BICYCLE_CLASS_ID]

# Cyclist detection
CYCLIST_IOU_THRESHOLD = 0.3

# Performance settings
PROCESS_WIDTH = 416
DISPLAY_WIDTH = 960
HEADLESS_MODE = False

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
SHOW_TIMING = True

# Health monitoring
HEALTH_LOG_INTERVAL = 300
TRACKER_RESET_INTERVAL = 3600
MAX_YOLO_TIMING_SAMPLES = 100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PERSISTENT SETTINGS - Saved to JSON, adjustable via sliders
# ==============================================================================

class TrackerSettings:
    """
    Manages persistent tracker settings.
    Values are saved to JSON whenever updated via sliders.
    """
    
    # Default values
    DEFAULTS = {
        'confidence_threshold': 40,      # 0-100, divide by 100 for actual value
        'fusion_threshold_cm': 150,      # 50-300 cm
        'fusion_threshold_far_cm': 200,  # Additional threshold for far objects
        'track_match_threshold_cm': 80,  # 30-150 cm
        'position_smoothing': 3,         # 1-20, divide by 100 for actual value
        'velocity_smoothing': 8,         # 1-30, divide by 100 for actual value
        'max_track_age_frames': 60,      # 15-150 frames
        'zone_filter_enabled': 1,        # 0 or 1 (boolean)
        'passive_zone_confidence': 70,   # 0-100, confidence multiplier for passive zone
    }
    
    # Slider ranges (min, max)
    RANGES = {
        'confidence_threshold': (10, 80),
        'fusion_threshold_cm': (50, 300),
        'fusion_threshold_far_cm': (100, 400),
        'track_match_threshold_cm': (30, 150),
        'position_smoothing': (1, 20),
        'velocity_smoothing': (1, 30),
        'max_track_age_frames': (15, 150),
        'zone_filter_enabled': (0, 1),
        'passive_zone_confidence': (30, 100),
    }
    
    def __init__(self, settings_file):
        self.settings_file = settings_file
        self.values = dict(self.DEFAULTS)
        self.load()
        self._dirty = False
    
    def load(self):
        """Load settings from file"""
        try:
            with open(self.settings_file, 'r') as f:
                saved = json.load(f)
                for key in self.DEFAULTS:
                    if key in saved:
                        self.values[key] = saved[key]
                print(f"📋 Loaded tracker settings from {self.settings_file}")
        except FileNotFoundError:
            print(f"📋 No settings file found, using defaults")
        except Exception as e:
            print(f"⚠️ Failed to load settings: {e}")
    
    def save(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.values, f, indent=2)
            print(f"💾 Saved tracker settings to {self.settings_file}")
            self._dirty = False
        except Exception as e:
            print(f"⚠️ Failed to save settings: {e}")
    
    def get(self, key):
        """Get a setting value"""
        return self.values.get(key, self.DEFAULTS.get(key))
    
    def set(self, key, value):
        """Set a setting value and mark dirty"""
        if key in self.values:
            self.values[key] = value
            self._dirty = True
    
    # Convenience properties that return properly scaled values
    @property
    def confidence_threshold(self):
        return self.values['confidence_threshold'] / 100.0
    
    @property
    def fusion_threshold_cm(self):
        return float(self.values['fusion_threshold_cm'])
    
    @property
    def fusion_threshold_far_cm(self):
        return float(self.values['fusion_threshold_far_cm'])
    
    @property
    def track_match_threshold_cm(self):
        return float(self.values['track_match_threshold_cm'])
    
    @property
    def position_smoothing(self):
        return self.values['position_smoothing'] / 100.0
    
    @property
    def velocity_smoothing(self):
        return self.values['velocity_smoothing'] / 100.0
    
    @property
    def max_track_age_frames(self):
        return int(self.values['max_track_age_frames'])
    
    @property
    def zone_filter_enabled(self):
        return bool(self.values['zone_filter_enabled'])
    
    @property
    def passive_zone_confidence(self):
        return self.values['passive_zone_confidence'] / 100.0


# ==============================================================================
# WORLD COORDINATES LOADER
# ==============================================================================

def load_world_coordinates():
    """
    Load tracking zones and other data from world_coordinates.json.
    Returns dict with tracking_zones, reference_levels, etc.
    """
    try:
        with open(WORLD_COORDS_FILE, 'r') as f:
            data = json.load(f)
        
        # Extract tracking zones
        tracking_zones = {}
        zones_data = data.get('tracking_zones', {})
        
        for zone_name, zone_info in zones_data.items():
            bounds = zone_info.get('bounds', {})
            tracking_zones[zone_name] = {
                'x': bounds.get('x', [-300, 0]),
                'y': bounds.get('y', [-66, 234]),
                'z': bounds.get('z', [78, 283]),
                'description': zone_info.get('description', ''),
            }
        
        # Extract reference levels
        ref_levels = data.get('reference_levels', {})
        reference_levels = {
            'floor': float(ref_levels.get('floor', {}).get('y', 0)),
            'street': float(ref_levels.get('street', {}).get('y', -66)),
            'camera_ledge': float(ref_levels.get('camera_ledge', {}).get('y', -15)),
        }
        
        # Extract camera positions
        camera_positions = {}
        cameras_data = data.get('cameras', {})
        for cam_name, cam_info in cameras_data.items():
            pos = cam_info.get('position', [0, 0, 0])
            camera_positions[cam_name] = tuple(float(x) for x in pos)
        
        print(f"📐 Loaded world coordinates from {WORLD_COORDS_FILE}")
        print(f"   Tracking zones: {list(tracking_zones.keys())}")
        
        return {
            'tracking_zones': tracking_zones,
            'reference_levels': reference_levels,
            'camera_positions': camera_positions,
        }
        
    except FileNotFoundError:
        print(f"⚠️ world_coordinates.json not found at {WORLD_COORDS_FILE}")
        # Return defaults
        return {
            'tracking_zones': {
                'active': {'x': [-280, -20], 'y': [-66, 234], 'z': [78, 283]},
                'passive': {'x': [-350, 50], 'y': [-66, 234], 'z': [283, 553]},
            },
            'reference_levels': {'floor': 0, 'street': -66, 'camera_ledge': -15},
            'camera_positions': {'camera_1': (-30, -15, 78), 'camera_2': (-270, -15, 78)},
        }
    except Exception as e:
        print(f"⚠️ Error loading world_coordinates.json: {e}")
        return None


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
# ZONE CHECKER - Determines which tracking zone a position is in
# ==============================================================================

class ZoneChecker:
    """
    Checks which tracking zone a world position is in.
    Returns zone name and confidence multiplier.
    """
    
    def __init__(self, tracking_zones):
        self.zones = tracking_zones
        # Order matters: check active first, then passive
        self.zone_order = ['active', 'passive']
    
    def check(self, world_x, world_z):
        """
        Check which zone the position is in.
        
        Returns:
            (zone_name, confidence_multiplier)
            zone_name: 'active', 'passive', or 'outside'
            confidence_multiplier: 1.0 for active, 0.7 for passive, 0.0 for outside
        """
        for zone_name in self.zone_order:
            if zone_name not in self.zones:
                continue
            zone = self.zones[zone_name]
            x_bounds = zone['x']
            z_bounds = zone['z']
            
            if (x_bounds[0] <= world_x <= x_bounds[1] and
                z_bounds[0] <= world_z <= z_bounds[1]):
                
                if zone_name == 'active':
                    return 'active', 1.0
                else:
                    return 'passive', 0.7  # Default, can be overridden by settings
        
        return 'outside', 0.0
    
    def get_fusion_threshold(self, world_z, base_threshold, far_threshold):
        """
        Get dynamic fusion threshold based on Z distance.
        Objects farther from cameras have more position uncertainty.
        
        Args:
            world_z: Z coordinate (distance from panels)
            base_threshold: Threshold for active zone
            far_threshold: Threshold for far objects
            
        Returns:
            Fusion threshold in cm
        """
        # Active zone ends at Z=283, passive zone starts there
        active_z_back = 283
        passive_z_back = 553
        
        if world_z < active_z_back:
            return base_threshold
        elif world_z > passive_z_back:
            return far_threshold
        else:
            # Linear interpolation
            t = (world_z - active_z_back) / (passive_z_back - active_z_back)
            return base_threshold + t * (far_threshold - base_threshold)


# ==============================================================================
# CALIBRATION MANAGER
# ==============================================================================

class CalibrationManager:
    """Manages 3D camera calibration for floor projection"""
    
    def __init__(self, calibration_file, floor_y=-66.0):
        self.calibration_file = calibration_file
        self.calibrations = {}
        self.is_calibrated = False
        self.floor_y = floor_y  # Street level from world_coordinates.json
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
    
    def image_to_floor(self, camera_name, img_x, img_y):
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
        
        t = (self.floor_y - camera_pos[1]) / ray_world[1]
        
        if t < 0:
            return None
        
        world_pt = camera_pos + t * ray_world
        return (float(world_pt[0]), float(world_pt[2]))
    
    def transform_bbox_center(self, camera_name, x1, y1, x2, y2):
        """Transform bounding box to floor position using bottom center (feet)"""
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return self.image_to_floor(camera_name, foot_x, foot_y)


# ==============================================================================
# TRACKING FUSION - V2 with zone awareness and dynamic thresholds
# ==============================================================================

class TrackingFusionV2:
    """
    V2 fusion with zone-based filtering and dynamic thresholds.
    """
    
    def __init__(self, settings, zone_checker):
        self.settings = settings
        self.zone_checker = zone_checker
        self.smoothed_positions = {}
        self.next_stable_id = 1
        self.frame_count = 0
    
    def fuse_detections(self, world_detections):
        """
        Fuse detections from multiple cameras with dynamic thresholds.
        """
        if not world_detections:
            return []
        
        fused = []
        used = [False] * len(world_detections)
        
        for i, det in enumerate(world_detections):
            if used[i]:
                continue
            
            cluster = [det]
            used[i] = True
            
            # Get dynamic fusion threshold based on Z position
            fusion_threshold = self.zone_checker.get_fusion_threshold(
                det['z'],
                self.settings.fusion_threshold_cm,
                self.settings.fusion_threshold_far_cm
            )
            
            # Find nearby detections from OTHER cameras only
            for j, other in enumerate(world_detections):
                if used[j]:
                    continue
                
                if other['camera'] == det['camera']:
                    continue
                
                dx = det['x'] - other['x']
                dz = det['z'] - other['z']
                dist = (dx*dx + dz*dz) ** 0.5
                
                if dist < fusion_threshold:
                    cluster.append(other)
                    used[j] = True
            
            # Average position across cluster
            avg_x = sum(d['x'] for d in cluster) / len(cluster)
            avg_z = sum(d['z'] for d in cluster) / len(cluster)
            cameras = list(set(d['camera'] for d in cluster))
            best = max(cluster, key=lambda d: d['conf'])
            
            # Check zone and get confidence multiplier
            zone_name, zone_conf = self.zone_checker.check(avg_x, avg_z)
            
            # Apply zone-based confidence
            if self.settings.zone_filter_enabled:
                if zone_name == 'passive':
                    zone_conf = self.settings.passive_zone_confidence
                elif zone_name == 'outside':
                    continue  # Skip detections outside all zones
            
            fused.append({
                'x': avg_x,
                'z': avg_z,
                'cameras': cameras,
                'conf': best['conf'] * zone_conf,
                'zone': zone_name,
            })
        
        return fused
    
    def smooth_and_track(self, fused_detections):
        """Apply smoothing with configurable parameters."""
        self.frame_count += 1
        matched_tracks = set()
        output = []
        
        for det in fused_detections:
            raw_x, raw_z = det['x'], det['z']
            
            # Find closest existing track
            best_track_id = None
            best_dist = self.settings.track_match_threshold_cm
            
            for track_id, track in self.smoothed_positions.items():
                if track_id in matched_tracks:
                    continue
                pred_x = track['x'] + track.get('vx', 0)
                pred_z = track['z'] + track.get('vz', 0)
                dist = ((raw_x - pred_x)**2 + (raw_z - pred_z)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                matched_tracks.add(best_track_id)
                old = self.smoothed_positions[best_track_id]
                
                # Smooth position
                pos_smooth = self.settings.position_smoothing
                predicted_x = old['x'] + old.get('vx', 0)
                predicted_z = old['z'] + old.get('vz', 0)
                new_x = predicted_x * (1 - pos_smooth) + raw_x * pos_smooth
                new_z = predicted_z * (1 - pos_smooth) + raw_z * pos_smooth
                
                # Smooth velocity
                vel_smooth = self.settings.velocity_smoothing
                raw_vx = new_x - old['x']
                raw_vz = new_z - old['z']
                new_vx = old.get('vx', 0) * (1 - vel_smooth) + raw_vx * vel_smooth
                new_vz = old.get('vz', 0) * (1 - vel_smooth) + raw_vz * vel_smooth
                
                self.smoothed_positions[best_track_id] = {
                    'x': new_x, 'z': new_z,
                    'vx': new_vx, 'vz': new_vz,
                    'last_seen': self.frame_count,
                    'cameras': det['cameras'],
                    'zone': det['zone'],
                }
                
                output.append((best_track_id, new_x, new_z, det['cameras'], det['zone']))
            else:
                new_id = self.next_stable_id
                self.next_stable_id += 1
                
                self.smoothed_positions[new_id] = {
                    'x': raw_x, 'z': raw_z,
                    'vx': 0, 'vz': 0,
                    'last_seen': self.frame_count,
                    'cameras': det['cameras'],
                    'zone': det['zone'],
                }
                
                output.append((new_id, raw_x, raw_z, det['cameras'], det['zone']))
        
        # Remove old tracks
        max_age = self.settings.max_track_age_frames
        stale = [k for k, v in self.smoothed_positions.items()
                 if self.frame_count - v['last_seen'] > max_age]
        for k in stale:
            del self.smoothed_positions[k]
        
        return output
    
    def process(self, world_detections):
        """Full pipeline: fuse then smooth."""
        fused = self.fuse_detections(world_detections)
        return self.smooth_and_track(fused)


# ==============================================================================
# ROBUST CAMERA
# ==============================================================================

class RobustCamera:
    """Reliable camera capture with automatic reconnection."""
    
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
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self._connect()
    
    def _capture_loop(self):
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
                    
                    # Flush buffer
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
                    
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    logger.warning(f"{self.name}: Capture error ({consecutive_failures}x): {e}")
                if consecutive_failures > 10:
                    self.connected = False
                    logger.error(f"{self.name}: Too many failures, marking disconnected")
                time.sleep(0.01)
    
    def read_new(self):
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
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


# ==============================================================================
# FRAME PROCESSOR
# ==============================================================================

class FrameProcessor:
    def __init__(self):
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            print("💻 Using CUDA for preprocessing")
        else:
            print("💻 Using CPU for preprocessing")
    
    def resize(self, frame, target_size):
        if frame is None:
            return None
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


# ==============================================================================
# SLIDER CALLBACKS
# ==============================================================================

def create_slider_callback(settings, key):
    """Create a callback function for a slider that updates settings."""
    def callback(value):
        settings.set(key, value)
    return callback


def setup_sliders(window_name, settings):
    """Create trackbars for adjustable parameters."""
    for key, (min_val, max_val) in settings.RANGES.items():
        # Create a nicer label
        label = key.replace('_', ' ').title()
        if len(label) > 20:
            label = key[:20]
        
        cv2.createTrackbar(
            label,
            window_name,
            settings.get(key),
            max_val,
            create_slider_callback(settings, key)
        )


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not acquire_single_instance_lock():
        sys.exit(1)
    atexit.register(release_single_instance_lock)
    
    print("=" * 60)
    print("Camera Tracker with OSC Output - V2")
    print("=" * 60)
    
    # Load world coordinates
    world_coords = load_world_coordinates()
    if world_coords is None:
        print("❌ Failed to load world coordinates!")
        return
    
    # Initialize settings (persistent)
    settings = TrackerSettings(SETTINGS_FILE)
    
    # Initialize zone checker
    zone_checker = ZoneChecker(world_coords['tracking_zones'])
    
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
    
    # Use street level from world_coordinates
    floor_y = world_coords['reference_levels'].get('street', -66)
    calibration = CalibrationManager(CALIBRATION_FILE, floor_y=floor_y)
    
    # V2 fusion with zone awareness
    fusion = TrackingFusionV2(settings, zone_checker)
    
    if not calibration.is_calibrated:
        print("\n⚠️ WARNING: No calibration loaded!")
        print("   Run camera_calibration_v2.py and calibrate first.")
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
    print("   Press 'q' to quit, 's' to save settings\n")
    
    # Graceful shutdown handling
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Main loop variables
    frame_count = 0
    start_time = time.time()
    frame_interval = 1.0 / TARGET_FPS
    last_process_time = 0
    yolo_times = deque(maxlen=MAX_YOLO_TIMING_SAMPLES)
    
    # Health monitoring
    last_health_log = time.time()
    last_tracker_reset = time.time()
    last_settings_save = time.time()
    total_osc_errors = 0
    total_people_tracked = 0
    
    # Create windows
    if not HEADLESS_MODE:
        cv2.namedWindow("Tracker OSC V2", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Settings", 400, 400)
        setup_sliders("Settings", settings)
    
    logger.info("Tracker started - entering main loop")
    
    while not shutdown_requested:
        current_time = time.time()
        
        if current_time - last_process_time < frame_interval:
            time.sleep(0.001)
            continue
        
        last_process_time = current_time
        frame_count += 1
        
        # Collect frames from all cameras
        raw_world_detections = []
        display_frames = []
        
        for cfg in camera_configs:
            cam = cfg['camera']
            ret, frame, latency_ms, is_new = cam.read_new()
            
            if not ret or frame is None:
                continue
            
            if is_new:
                yolo_start = time.time()
                
                small_frame = processor.resize(frame, (PROCESS_WIDTH, cfg['process_height']))
                
                results = model.track(
                    small_frame,
                    persist=True,
                    verbose=False,
                    conf=settings.confidence_threshold,
                    classes=TRACKED_CLASSES,
                    tracker="bytetrack.yaml",
                    imgsz=PROCESS_WIDTH,
                    device=device
                )
                
                raw_detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        x1 = x1 / cfg['scale']
                        y1 = y1 / cfg['scale']
                        x2 = x2 / cfg['scale']
                        y2 = y2 / cfg['scale']
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                        track_id = int(boxes.id[i]) if boxes.id is not None else i
                        raw_detections.append((x1, y1, x2, y2, conf, track_id, cls))
                
                cfg['last_boxes'] = merge_cyclists(raw_detections)
                
                yolo_time = time.time() - yolo_start
                yolo_times.append(yolo_time)
                
                instant_fps = 1.0 / max(yolo_time, 0.001)
                cfg['fps_history'].append(instant_fps)
                cfg['current_fps'] = sum(cfg['fps_history']) / len(cfg['fps_history'])
            
            # Transform detections to world coordinates
            for box in cfg['last_boxes']:
                x1, y1, x2, y2, conf, track_id = box[:6]
                class_id = box[6] if len(box) > 6 else PERSON_CLASS_ID
                
                if class_id not in (PERSON_CLASS_ID, CYCLIST_CLASS_ID):
                    continue
                
                world_pos = calibration.transform_bbox_center(cfg['name'], x1, y1, x2, y2)
                if world_pos is not None:
                    world_x, world_z = world_pos
                    raw_world_detections.append({
                        'x': world_x,
                        'z': world_z,
                        'camera': cfg['name'],
                        'track_id': track_id,
                        'conf': conf,
                    })
            
            # Create display frame
            if not HEADLESS_MODE:
                display_scale = 480 / frame.shape[1]
                display_frame = processor.resize(frame, (480, int(frame.shape[0] * display_scale)))
                
                for box in cfg['last_boxes']:
                    x1, y1, x2, y2, conf, track_id = box[:6]
                    dx1 = int(x1 * display_scale)
                    dy1 = int(y1 * display_scale)
                    dx2 = int(x2 * display_scale)
                    dy2 = int(y2 * display_scale)
                    
                    cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID:{track_id}", (dx1, dy1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(display_frame, f"{cfg['name']} FPS:{cfg['current_fps']:.1f}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                display_frames.append(display_frame)
        
        # Fuse detections
        tracked_people = fusion.process(raw_world_detections)
        
        # Send OSC messages
        osc_start = time.time()
        try:
            osc_client.send_message("/tracker/count", len(tracked_people))
            
            for stable_id, world_x, world_z, cam_names, zone in tracked_people:
                osc_client.send_message(f"/tracker/person/{stable_id}", [float(world_x), float(world_z)])
                osc_client.send_message(f"/tracker/zone/{stable_id}", zone)
            
            total_people_tracked += len(tracked_people)
        except Exception as e:
            total_osc_errors += 1
            if total_osc_errors == 1 or total_osc_errors % 100 == 0:
                logger.warning(f"OSC send error ({total_osc_errors}x total): {e}")
        osc_time = time.time() - osc_start
        
        # Timing diagnostics
        loop_time = time.time() - current_time
        if SHOW_TIMING and frame_count % 30 == 0:
            avg_yolo_time = sum(yolo_times) / len(yolo_times) if yolo_times else 0
            zone_counts = {'active': 0, 'passive': 0}
            for _, _, _, _, zone in tracked_people:
                if zone in zone_counts:
                    zone_counts[zone] += 1
            print(f"Frame {frame_count}: YOLO={avg_yolo_time*1000:.1f}ms, People={len(tracked_people)} (active:{zone_counts['active']}, passive:{zone_counts['passive']})")
        
        # Auto-save settings periodically if changed
        if settings._dirty and time.time() - last_settings_save > 5.0:
            settings.save()
            last_settings_save = time.time()
        
        # Periodic health logging
        if time.time() - last_health_log >= HEALTH_LOG_INTERVAL:
            elapsed_total = time.time() - start_time
            uptime = timedelta(seconds=int(elapsed_total))
            connected_cams = sum(1 for cam in cameras if cam.connected)
            avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
            
            logger.info(
                f"HEALTH: uptime={uptime}, frames={frame_count}, avg_fps={avg_fps:.1f}, "
                f"cameras={connected_cams}/{len(cameras)}, osc_errors={total_osc_errors}, "
                f"people_tracked={total_people_tracked}"
            )
            
            for cfg in camera_configs:
                cam = cfg['camera']
                logger.info(
                    f"  {cfg['name']}: connected={cam.connected}, "
                    f"fps={cfg['current_fps']:.1f}, "
                    f"frames={cam.stats['frames_received']}, "
                    f"reconnects={cam.stats['reconnect_count']}"
                )
            
            last_health_log = time.time()
        
        # Periodic YOLO tracker reset
        if time.time() - last_tracker_reset >= TRACKER_RESET_INTERVAL:
            logger.info("Resetting YOLO tracker to prevent memory buildup...")
            try:
                dummy = np.zeros((PROCESS_WIDTH, PROCESS_WIDTH, 3), dtype=np.uint8)
                model.track(dummy, persist=False, verbose=False, classes=TRACKED_CLASSES)
                model.track(dummy, persist=True, verbose=False, classes=TRACKED_CLASSES)
                logger.info("YOLO tracker reset complete")
            except Exception as e:
                logger.warning(f"Failed to reset YOLO tracker: {e}")
            last_tracker_reset = time.time()
        
        # Create combined display
        if not HEADLESS_MODE and display_frames:
            if len(display_frames) == 1:
                combined = display_frames[0]
            else:
                max_h = max(f.shape[0] for f in display_frames)
                padded = []
                for f in display_frames:
                    if f.shape[0] < max_h:
                        pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                        f = cv2.vconcat([f, pad])
                    padded.append(f)
                combined = cv2.hconcat(padded)
            
            # Status bar with zone info
            zone_info = f"Active: {sum(1 for p in tracked_people if p[4] == 'active')} | Passive: {sum(1 for p in tracked_people if p[4] == 'passive')}"
            status = f"OSC -> {OSC_IP}:{OSC_PORT} | {zone_info} | Frame: {frame_count}"
            cv2.putText(combined, status, (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Tracker OSC V2", combined)
            
            # Keep settings window alive
            settings_img = np.zeros((50, 400, 3), dtype=np.uint8)
            cv2.putText(settings_img, "Adjust sliders - auto-saves", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Settings", settings_img)
        
        # Handle keys
        if not HEADLESS_MODE:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed")
                shutdown_requested = True
            elif key == ord('s'):
                settings.save()
                print("✅ Settings saved manually")
    
    # Save settings on exit
    if settings._dirty:
        settings.save()
    
    # Cleanup
    logger.info("Shutting down...")
    
    for cam in cameras:
        logger.info(f"Releasing {cam.name}...")
        cam.release()
    
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    uptime = timedelta(seconds=int(elapsed))
    
    logger.info(f"Shutdown complete")
    logger.info(f"  Uptime: {uptime}")
    logger.info(f"  Frames processed: {frame_count}")
    if elapsed > 0:
        logger.info(f"  Average FPS: {frame_count/elapsed:.1f}")
    logger.info(f"  Total OSC errors: {total_osc_errors}")
    logger.info(f"  Total people tracked: {total_people_tracked}")
    
    print(f"\n🛑 Stopped after {uptime}")
    if elapsed > 0:
        print(f"   Frames: {frame_count} | Avg FPS: {frame_count/elapsed:.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")
        print(f"\n❌ FATAL ERROR: {e}")
    finally:
        # Ensure lock is released even on crash
        release_single_instance_lock()
