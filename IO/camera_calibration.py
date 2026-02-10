#!/usr/bin/env python3
"""
Multi-Camera YOLO Person Tracking - CUDA Optimized
Detects and tracks people across multiple Reolink camera feeds using NVIDIA GPU acceleration

View Modes (press keys to switch):
  1-9: Show individual camera fullscreen
  S: Side-by-side view (all cameras)
  G: Grid view (2x2, 3x3, etc.)
  T: Synthesized top-down tracking view (requires calibration)
  C: Enter calibration mode

Requirements:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- OpenCV with ArUco support (for calibration)

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

# ==============================================================================
# CAMERA CONFIGURATION - Add/remove cameras here
# ==============================================================================
CAMERAS = [
    {
        'name': 'Camera 1',
        'url': 'rtsp://admin:dc31l1ng@10.42.0.75:555/h264Preview_01_main',
        'fps': 25,  # Target output FPS (matches camera setting)
        'enabled': True,
    },
    {
        'name': 'Camera 2', 
        'url': 'rtsp://admin:dc31l1ng@10.42.0.172:555/h264Preview_01_main',
        'fps': 25,
        'enabled': True,
    },
    {
        'name': 'Camera 3',
        'url': 'rtsp://admin:dc31l1ng@10.42.0.111:555/h264Preview_01_main',
        'fps': 25,
        'enabled': False,  # Disabled - using only cameras 1 and 2
    },
]

# Camera network info (for reference):
# - RTSP: port 555
# - HTTPS: port 443
# - ONVIF: port 8000

# Calibration file path (created by calibration mode)
# Use absolute path relative to script location so it works from any directory
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
CALIBRATION_FILE = _os.path.join(_SCRIPT_DIR, 'camera_calibration.json')

# Synthesized view settings (all units in CENTIMETERS)
SYNTH_VIEW_WIDTH = 800   # Width of bird's eye view in pixels
SYNTH_VIEW_HEIGHT = 600  # Height of bird's eye view
SYNTH_CM_PER_PIXEL = 1.0  # 1cm per pixel = 8m x 6m coverage (good for storefront)

# Height/Y coordinate system (positive Y = UP, negative Y = DOWN)
# Reference: Storefront floor = Y=0
#
# Physical setup:
#   - Storefront floor: Y = 0
#   - Camera ledge: 15 cm below floor → Y = -15 (51 cm above street)
#   - Street level: 66 cm below floor → Y = -66 (where people walk)
#
STREET_LEVEL_Y = -66.0     # Where people walk (66 cm below floor)
CAMERA_LEDGE_Y = -15.0     # Where cameras are mounted (15cm below floor)
CAMERA_HEIGHT_ABOVE_STREET = 51.0  # Physical camera mount height

# YOLO settings
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS_ID = 0
BICYCLE_CLASS_ID = 1  # Bicycles in COCO dataset
CAR_CLASS_ID = 2  # Cars in COCO dataset
CYCLIST_CLASS_ID = -1  # Synthetic class: person + bicycle overlap
TRACKED_CLASSES = [PERSON_CLASS_ID, BICYCLE_CLASS_ID, CAR_CLASS_ID]  # Track people, bicycles, and cars

# Cyclist detection settings
CYCLIST_IOU_THRESHOLD = 0.3  # Minimum overlap to consider person+bicycle as cyclist

# Performance settings
PROCESS_WIDTH = 416
DISPLAY_WIDTH = 960

# Camera sync settings
# When True, processes all cameras together at a synchronized rate
# Essential for multi-camera tracking fusion
SYNC_CAMERAS = True
TARGET_FPS = 25  # Target synchronized frame rate (matches camera max)

# Reliability settings
CONNECTION_TIMEOUT = 10.0  # Seconds to wait for initial connection
RECONNECT_DELAY = 2.0      # Seconds between reconnection attempts
MAX_FRAME_AGE = 0.5        # Max age (seconds) before frame considered stale
FRAME_CACHE_ENABLED = True # Cache last good frame for display continuity

# CUDA settings
CUDA_DEVICE = 0

# GUI overlay settings
SHOW_GUI_OVERLAY = True  # Set to False to hide all overlays for clean screenshots

# ==============================================================================
# VIEW MODES
# ==============================================================================
from enum import Enum
import json
import math

class ViewMode(Enum):
    SIDE_BY_SIDE = 'side_by_side'
    GRID = 'grid'
    INDIVIDUAL = 'individual'
    SYNTHESIZED = 'synthesized'
    CALIBRATION = 'calibration'


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in format (x1, y1, x2, y2, ...)
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def merge_cyclists(detections):
    """
    Merge overlapping person + bicycle detections into cyclist detections.
    Returns a new list of detections with cyclists properly labeled.
    
    Logic:
    - Find all person-bicycle pairs with IoU > threshold
    - Create a cyclist detection using the person's bounding box and track ID
    - Remove the matched bicycle detection
    - Keep unmatched persons as pedestrians, unmatched bicycles as standalone
    """
    persons = [d for d in detections if d[6] == PERSON_CLASS_ID]
    bicycles = [d for d in detections if d[6] == BICYCLE_CLASS_ID]
    others = [d for d in detections if d[6] not in (PERSON_CLASS_ID, BICYCLE_CLASS_ID)]
    
    matched_persons = set()
    matched_bicycles = set()
    cyclists = []
    
    # Find person-bicycle pairs
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
            # Create cyclist detection using person's box and track ID
            # but with the synthetic CYCLIST_CLASS_ID
            x1, y1, x2, y2, conf, track_id, _ = person
            bicycle = bicycles[best_bi]
            # Use higher confidence of the two
            cyclist_conf = max(conf, bicycle[4])
            cyclists.append((x1, y1, x2, y2, cyclist_conf, track_id, CYCLIST_CLASS_ID))
            matched_persons.add(pi)
            matched_bicycles.add(best_bi)
    
    # Collect unmatched persons (pedestrians)
    pedestrians = [p for pi, p in enumerate(persons) if pi not in matched_persons]
    
    # Collect unmatched bicycles (parked bikes, etc.)
    standalone_bicycles = [b for bi, b in enumerate(bicycles) if bi not in matched_bicycles]
    
    # Combine all detections
    return pedestrians + cyclists + standalone_bicycles + others


class CalibrationManager:
    """
    Manages 3D camera calibration for synthesized view.
    Uses ArUco markers with 3D world coordinates and solvePnP for camera pose estimation.
    Projects detected people onto floor plane (z=0) using ray intersection.
    """
    def __init__(self, calibration_file):
        self.calibration_file = calibration_file
        self.calibrations = {}  # camera_name -> {rvec, tvec, camera_matrix, dist_coeffs}
        self.is_calibrated = False
        self.load()
    
    def load(self):
        """Load calibration from file if exists"""
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
            pass
        except Exception as e:
            print(f"⚠️ Failed to load calibration: {e}")
    
    def save(self):
        """Save calibration to file"""
        data = {
            'cameras': {},
            'synth_width': SYNTH_VIEW_WIDTH,
            'synth_height': SYNTH_VIEW_HEIGHT,
            'cm_per_pixel': SYNTH_CM_PER_PIXEL,
            'calibration_type': '3d_pose',
            'units': 'centimeters',
        }
        for name, calib in self.calibrations.items():
            data['cameras'][name] = {
                'rvec': calib['rvec'].tolist(),
                'tvec': calib['tvec'].tolist(),
                'camera_matrix': calib['camera_matrix'].tolist(),
                'dist_coeffs': calib['dist_coeffs'].tolist(),
                'image_size': list(calib['image_size']),
            }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved 3D calibration to {self.calibration_file}")
    
    def set_calibration(self, camera_name, rvec, tvec, camera_matrix, dist_coeffs, image_size):
        """Set 3D calibration for a camera"""
        self.calibrations[camera_name] = {
            'rvec': rvec,
            'tvec': tvec,
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'image_size': image_size,
        }
        self.is_calibrated = True
    
    def image_to_floor(self, camera_name, img_x, img_y, floor_y=None):
        """
        Project image point to floor plane (y=floor_y) using ray intersection.
        Returns (world_x, world_z) in cm, or None if invalid.
        
        Coordinate system:
        - X: left-right along panels
        - Y: height (storefront floor = 0, street level = -66)
        - Z: depth from panels
        
        If floor_y is None, uses STREET_LEVEL_Y (default for people walking on street)
        """
        if floor_y is None:
            floor_y = STREET_LEVEL_Y
            
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
        
        # Ray direction in camera coordinates (normalized image plane)
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([ux, uy, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform ray to world coordinates
        ray_world = R.T @ ray_cam
        
        # Intersect ray with floor plane y = floor_y
        # Ray: P = camera_pos + t * ray_world
        # Plane: y = floor_y
        # Solve: camera_pos[1] + t * ray_world[1] = floor_y
        
        if abs(ray_world[1]) < 1e-6:
            return None  # Ray parallel to floor
        
        t = (floor_y - camera_pos[1]) / ray_world[1]
        
        if t < 0:
            return None  # Intersection behind camera
        
        # World intersection point
        world_pt = camera_pos + t * ray_world
        
        # Return X and Z (the floor plane coordinates)
        return float(world_pt[0]), float(world_pt[2])
    
    def transform_bbox_center(self, camera_name, x1, y1, x2, y2, floor_y=None):
        """
        Transform bounding box to floor position.
        Uses bottom center of bbox (feet) projected to floor plane.
        Returns (world_x, world_z) in cm.
        
        If floor_y is None, uses STREET_LEVEL_Y (people walking on street).
        """
        foot_x = (x1 + x2) / 2
        foot_y = y2  # Bottom of bounding box
        return self.image_to_floor(camera_name, foot_x, foot_y, floor_y=floor_y)
    
    def world_to_synth_pixels(self, world_x, world_z):
        """
        Convert world coordinates (cm) to synthesized view pixels.
        world_x -> pixel x, world_z -> pixel y (depth becomes vertical in bird's eye)
        """
        # V2 COORDINATE SYSTEM:
        # - X=0 is at right edge (Panel 0), X goes negative toward left (Unit 3)
        # - Z=78 is front of active zone, Z increases toward street
        # Put X=0 on the RIGHT side of the view
        offset_x = SYNTH_VIEW_WIDTH - 50  # X=0 near right edge of view
        offset_y = 30   # Z=0 near top (panels), tracking zone below
        
        # X is negative going left, so adding it shifts left in pixel space
        px = int(offset_x + world_x / SYNTH_CM_PER_PIXEL)
        # Z is positive going down (away from panels = down in bird's eye)
        py = int(offset_y + world_z / SYNTH_CM_PER_PIXEL)
        return px, py
    
    def get_camera_position(self, camera_name):
        """Get camera position in world coordinates (meters)"""
        if camera_name not in self.calibrations:
            return None
        calib = self.calibrations[camera_name]
        R, _ = cv2.Rodrigues(calib['rvec'])
        camera_pos = -R.T @ calib['tvec'].flatten()
        return camera_pos


class SynthesizedView:
    """
    Creates a bird's eye view combining all camera feeds.
    Shows tracked people as dots on a floor plan using 3D calibration.
    """
    def __init__(self, width, height, calibration_manager):
        self.width = width
        self.height = height
        self.calibration = calibration_manager
        self.background = None
        self.load_or_create_background()
        
        # Temporal smoothing with velocity prediction for smooth + responsive tracking
        # Format: track_key -> {'x', 'z', 'vx', 'vz', 'last_seen'}
        self.smoothed_positions = {}
        self.position_smoothing = 0.03   # Very heavy position smoothing (3% new, 97% old)
        self.velocity_smoothing = 0.08   # Velocity adapts slowly too
        self.max_age_frames = 60         # Keep tracks for 60 frames (~2 seconds at 25fps)
        self.frame_count = 0
        
        # Position-based tracking (more stable than ID-based)
        self.persistent_tracks = {}   # position-based stable tracks
        self.next_stable_id = 1
    
    def load_or_create_background(self):
        """Load floor plan image or create grid for centimeter-based coordinate system"""
        try:
            self.background = cv2.imread('floor_plan.png')
            if self.background is not None:
                self.background = cv2.resize(self.background, (self.width, self.height))
                return
        except:
            pass
        
        # Create default grid background
        self.background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.background[:] = (40, 40, 40)  # Dark gray
        
        # V2 COORDINATE SYSTEM:
        # - X=0 at right (Panel 0), X goes negative toward left (to -300)
        # - Z=0 at panels, Z=78 is front of active zone, Z increases toward street
        offset_x = self.width - 50  # X=0 near right edge
        offset_y = 30   # Z=0 near top
        
        # Draw grid lines every 100cm (1 meter)
        grid_spacing_cm = 100
        
        # Vertical lines (X axis) - from X=0 to X=-400 (covers panels and beyond)
        for x_cm in range(-400, 100, grid_spacing_cm):
            x_px = int(offset_x + x_cm / SYNTH_CM_PER_PIXEL)
            if 0 <= x_px < self.width:
                cv2.line(self.background, (x_px, 0), (x_px, self.height), (60, 60, 60), 1)
                cv2.putText(self.background, f"X={x_cm}", (x_px + 2, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Horizontal lines (Z axis - depth from panels into tracking zone)
        for z_cm in range(0, 700, grid_spacing_cm):
            y_px = int(offset_y + z_cm / SYNTH_CM_PER_PIXEL)
            if 0 <= y_px < self.height:
                cv2.line(self.background, (0, y_px), (self.width, y_px), (60, 60, 60), 1)
                cv2.putText(self.background, f"Z={z_cm}", (5, y_px - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Draw panel positions (X=0 to -300, Z=0) - V2 coordinates
        panel_y = int(offset_y)
        for unit in range(4):
            # V2: Unit 0 at X=-30, Unit 1 at X=-110, etc.
            unit_x = -(unit * 80 + 30)  # -30, -110, -190, -270
            px = int(offset_x + unit_x / SYNTH_CM_PER_PIXEL)
            cv2.rectangle(self.background, (px - 25, panel_y - 5), (px + 25, panel_y + 5), 
                         (100, 100, 150), -1)
            cv2.putText(self.background, f"U{unit}", (px - 8, panel_y + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 255), 1)
        cv2.putText(self.background, "PANELS", (int(offset_x - 180), panel_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # Mark origin at (0,0) = back right corner of Panel 0
        origin_px = (offset_x, offset_y)
        cv2.circle(self.background, origin_px, 5, (0, 255, 255), -1)  # Yellow
        cv2.putText(self.background, "ORIGIN(0,0)", (offset_x - 70, offset_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Mark camera positions
        cam1_px = int(offset_x + (-30) / SYNTH_CM_PER_PIXEL)
        cam2_px = int(offset_x + (-270) / SYNTH_CM_PER_PIXEL)
        cam_y = int(offset_y + 78 / SYNTH_CM_PER_PIXEL)  # Z=78
        cv2.drawMarker(self.background, (cam1_px, cam_y), (0, 200, 255), cv2.MARKER_DIAMOND, 10, 2)
        cv2.putText(self.background, "C1", (cam1_px + 5, cam_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
        cv2.drawMarker(self.background, (cam2_px, cam_y), (255, 100, 100), cv2.MARKER_DIAMOND, 10, 2)
        cv2.putText(self.background, "C2", (cam2_px + 5, cam_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
        
        # Draw tracking zone boundaries
        active_z_front = int(offset_y + 78 / SYNTH_CM_PER_PIXEL)
        active_z_back = int(offset_y + 283 / SYNTH_CM_PER_PIXEL)
        cv2.line(self.background, (0, active_z_front), (self.width, active_z_front), (0, 100, 0), 1)
        cv2.line(self.background, (0, active_z_back), (self.width, active_z_back), (0, 100, 0), 1)
        cv2.putText(self.background, "ACTIVE ZONE", (10, active_z_front + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 0), 1)
    
    def fuse_detections(self, all_detections, fusion_threshold_cm=150.0):
        """
        Fuse detections from multiple cameras to avoid duplicates.
        ONLY merges detections from DIFFERENT cameras (same person seen by both).
        Never merges two detections from the same camera.
        
        Args:
            all_detections: list of (camera_name, boxes) tuples
            fusion_threshold_cm: max distance (cm) to consider same person (150cm for calibration error)
            
        Returns:
            List of fused detections: (world_x, world_z, track_id, color_key, confidence)
        """
        # First, project all detections to world coordinates
        world_detections = []
        
        for camera_name, boxes in all_detections:
            for box in boxes:
                x1, y1, x2, y2, conf, track_id = box[:6]
                class_id = box[6] if len(box) > 6 else PERSON_CLASS_ID
                
                # Only process persons/cyclists
                if class_id not in (PERSON_CLASS_ID, CYCLIST_CLASS_ID):
                    continue
                
                world_pos = self.calibration.transform_bbox_center(camera_name, x1, y1, x2, y2)
                if world_pos is None:
                    continue
                
                world_x, world_z = world_pos
                color_key = f"{camera_name}_{track_id}"
                world_detections.append({
                    'x': world_x,
                    'z': world_z,
                    'track_id': track_id,
                    'camera': camera_name,
                    'color_key': color_key,
                    'conf': conf,
                    'class_id': class_id,
                })
        
        # Debug: print world positions to see how far apart they are
        if len(world_detections) >= 2:
            cams = {}
            for d in world_detections:
                cam = d['camera']
                if cam not in cams:
                    cams[cam] = []
                cams[cam].append((d['x'], d['z']))
            if len(cams) == 2:
                # Have detections from both cameras - print distance between closest pair
                cam_names = list(cams.keys())
                c1_pts, c2_pts = cams[cam_names[0]], cams[cam_names[1]]
                min_dist = float('inf')
                for p1 in c1_pts:
                    for p2 in c2_pts:
                        d = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        if d < min_dist:
                            min_dist = d
                            pair = (p1, p2)
                if min_dist < float('inf'):
                    print(f"  🔍 C1: ({pair[0][0]:.0f},{pair[0][1]:.0f}) C2: ({pair[1][0]:.0f},{pair[1][1]:.0f}) dist={min_dist:.0f}cm")
        
        if not world_detections:
            return []
        
        # Cluster nearby detections - ONLY merge across different cameras
        # Two people from the same camera are always separate
        fused = []
        used = [False] * len(world_detections)
        
        for i, det in enumerate(world_detections):
            if used[i]:
                continue
            
            # Start a new cluster with this detection
            cluster = [det]
            used[i] = True
            
            # Find nearby detections from OTHER cameras only
            for j, other in enumerate(world_detections):
                if used[j]:
                    continue
                
                # Only consider merging if from a DIFFERENT camera
                if other['camera'] == det['camera']:
                    continue
                
                # Calculate distance
                dx = det['x'] - other['x']
                dz = det['z'] - other['z']
                dist = (dx*dx + dz*dz) ** 0.5
                
                if dist < fusion_threshold_cm:
                    cluster.append(other)
                    used[j] = True
            
            # Merge cluster: average position, keep highest confidence detection's info
            avg_x = sum(d['x'] for d in cluster) / len(cluster)
            avg_z = sum(d['z'] for d in cluster) / len(cluster)
            
            # Use the detection with highest confidence for display info
            best = max(cluster, key=lambda d: d['conf'])
            
            # Get unique cameras in this cluster
            cameras = list(set(d['camera'] for d in cluster))
            
            # Debug: show when we actually fuse
            if len(cameras) > 1:
                print(f"  ✅ FUSED: {len(cluster)} detections from {cameras}")
            
            fused.append({
                'x': avg_x,
                'z': avg_z,
                'track_id': best['track_id'],
                'color_key': best['color_key'],
                'cameras': cameras,
            })
        
        return fused
    
    def render(self, all_detections, track_colors):
        """
        Render synthesized view with all detections.
        all_detections: list of (camera_name, boxes) tuples
        """
        frame = self.background.copy()
        
        if not self.calibration.is_calibrated:
            cv2.putText(frame, "NOT CALIBRATED", (self.width//2 - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'C' to enter calibration mode", (self.width//2 - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            return frame
        
        self.frame_count += 1
        
        # Draw camera positions (X -> px, Z -> py in bird's eye view)
        for cam_name in self.calibration.calibrations:
            cam_pos = self.calibration.get_camera_position(cam_name)
            if cam_pos is not None:
                # cam_pos is (X, Y, Z) in cm; bird's eye uses X and Z
                px, py = self.calibration.world_to_synth_pixels(cam_pos[0], cam_pos[2])
                if 0 <= px < self.width and 0 <= py < self.height:
                    cv2.drawMarker(frame, (px, py), (0, 200, 255), cv2.MARKER_DIAMOND, 15, 2)
                    cv2.putText(frame, cam_name, (px + 10, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        
        # Fuse detections from multiple cameras to avoid duplicates
        fused_detections = self.fuse_detections(all_detections)
        
        # Apply position-based temporal smoothing
        # Match new detections to existing smoothed tracks by proximity
        matched_tracks = set()
        smoothed_output = []
        
        for det in fused_detections:
            raw_x, raw_z = det['x'], det['z']
            
            # Find closest existing smoothed track
            best_track_id = None
            best_dist = 80.0  # Max distance to match (80cm - less than arm span)
            
            for track_id, track in self.smoothed_positions.items():
                if track_id in matched_tracks:
                    continue
                # Predict where this track should be
                pred_x = track['x'] + track.get('vx', 0)
                pred_z = track['z'] + track.get('vz', 0)
                dist = ((raw_x - pred_x)**2 + (raw_z - pred_z)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track with smoothing
                matched_tracks.add(best_track_id)
                old = self.smoothed_positions[best_track_id]
                
                # Predict then smooth
                predicted_x = old['x'] + old.get('vx', 0)
                predicted_z = old['z'] + old.get('vz', 0)
                new_x = predicted_x * (1 - self.position_smoothing) + raw_x * self.position_smoothing
                new_z = predicted_z * (1 - self.position_smoothing) + raw_z * self.position_smoothing
                
                # Update velocity
                raw_vx = new_x - old['x']
                raw_vz = new_z - old['z']
                new_vx = old.get('vx', 0) * (1 - self.velocity_smoothing) + raw_vx * self.velocity_smoothing
                new_vz = old.get('vz', 0) * (1 - self.velocity_smoothing) + raw_vz * self.velocity_smoothing
                
                self.smoothed_positions[best_track_id] = {
                    'x': new_x, 'z': new_z,
                    'vx': new_vx, 'vz': new_vz,
                    'last_seen': self.frame_count,
                    'cameras': det['cameras'],
                    'display_id': old.get('display_id', best_track_id),
                }
                
                smoothed_output.append({
                    'x': new_x, 'z': new_z,
                    'track_id': old.get('display_id', best_track_id),
                    'cameras': det['cameras'],
                })
            else:
                # Create new track
                new_id = self.next_stable_id
                self.next_stable_id += 1
                
                self.smoothed_positions[new_id] = {
                    'x': raw_x, 'z': raw_z,
                    'vx': 0, 'vz': 0,
                    'last_seen': self.frame_count,
                    'cameras': det['cameras'],
                    'display_id': new_id,
                }
                
                smoothed_output.append({
                    'x': raw_x, 'z': raw_z,
                    'track_id': new_id,
                    'cameras': det['cameras'],
                })
        
        # Remove old tracks
        stale_keys = [k for k, v in self.smoothed_positions.items() 
                      if self.frame_count - v['last_seen'] > self.max_age_frames]
        for k in stale_keys:
            del self.smoothed_positions[k]
        
        # Draw smoothed detections
        for det in smoothed_output:
            world_x, world_z = det['x'], det['z']
            px, py = self.calibration.world_to_synth_pixels(world_x, world_z)
            
            # Clamp to view bounds
            px = max(0, min(self.width - 1, px))
            py = max(0, min(self.height - 1, py))
            
            # Get color for this track (use stable track_id)
            track_id = det['track_id']
            if track_id not in track_colors:
                import random
                random.seed(track_id * 12345)  # Stable color per ID
                track_colors[track_id] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
            color = track_colors[track_id]
            
            # Draw person as circle with ID
            # Use double ring if seen by multiple cameras
            if len(det['cameras']) > 1:
                cv2.circle(frame, (px, py), 15, (255, 255, 0), 2)  # Yellow outer ring = multi-cam
            cv2.circle(frame, (px, py), 12, color, -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)
            
            # Show camera source next to track ID
            cam_label = ','.join([c.replace('Camera ', 'C') for c in det['cameras']])
            cv2.putText(frame, f"{track_id}", (px - 6, py + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(frame, cam_label, (px - 10, py + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add info overlay
        cv2.putText(frame, "SYNTHESIZED VIEW (3D Calibrated)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show count
        count_text = f"Tracking: {len(smoothed_output)} people"
        cv2.putText(frame, count_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show scale
        scale_text = f"Scale: {SYNTH_CM_PER_PIXEL:.0f} cm/px"
        cv2.putText(frame, scale_text, (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame


class CalibrationMode:
    """
    Interactive 3D calibration using ArUco markers.
    Place markers at known 3D positions, detect them, compute camera pose via solvePnP.
    
    V2 COORDINATE SYSTEM:
    - Origin (0,0,0): Back right corner of Panel Unit 0, at floor level
    - X-axis: Negative toward Unit 3 (left when facing panels)
    - Y-axis: Positive upward
    - Z-axis: Positive forward into tracking zone
    
    LIMITED VISIBILITY CALIBRATION:
    Designed for setups where each camera can only see some markers.
    Markers 1, 5, 6 are SHARED markers visible from all camera views.
    This provides a common reference frame for multi-camera fusion.
    
    Setup (6 floor markers + 1 vertical):
    - Front row (Z=168cm): markers 0, 1, 2
    - Back row (Z=219cm): markers 3, 6, 4
    - Subway wall (Z=628cm): marker 5 (vertical)
    - Camera 1 (RIGHT, X=-30): sees markers 0, 1, 3, 5, 6
    - Camera 2 (LEFT, X=-270): sees markers 1, 2, 4, 5, 6
    - Each camera needs 3 markers minimum for solvePnP
    """
    # Minimum markers required per camera (3 markers × 4 corners = 12 points)
    MIN_MARKERS_REQUIRED = 3
    
    # Path to world coordinates JSON (single source of truth)
    WORLD_COORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'world_coordinates.json')
    
    @classmethod
    def load_world_coordinates(cls):
        """
        Load marker positions and other coordinates from world_coordinates.json.
        This establishes a single source of truth for all V2 coordinates.
        
        Returns:
            dict with keys: marker_positions, marker_size, vertical_markers,
                           camera_positions, reference_levels, camera_visibility
        """
        try:
            with open(cls.WORLD_COORDS_FILE, 'r') as f:
                data = json.load(f)
            
            # Extract marker positions
            marker_positions = {}
            vertical_markers = set()
            markers_data = data.get('calibration_markers', {}).get('markers', {})
            
            for marker_id_str, marker_info in markers_data.items():
                marker_id = int(marker_id_str)
                pos = marker_info['position']
                marker_positions[marker_id] = tuple(float(x) for x in pos)
                
                # Check if vertical
                if marker_info.get('orientation') == 'vertical':
                    vertical_markers.add(marker_id)
            
            # Extract marker size
            marker_size = float(data.get('calibration_markers', {}).get('marker_size', 15))
            
            # Extract camera positions (skip non-dict entries like 'model', 'sensor', etc.)
            camera_positions = {}
            cameras_data = data.get('cameras', {})
            for cam_name, cam_info in cameras_data.items():
                if not isinstance(cam_info, dict):
                    continue
                if 'position' not in cam_info:
                    continue
                pos = cam_info['position']
                camera_positions[cam_name] = tuple(float(x) for x in pos)
            
            # Extract reference levels
            ref_levels = data.get('reference_levels', {})
            reference_levels = {
                'floor': float(ref_levels.get('floor', {}).get('y', 0)),
                'street': float(ref_levels.get('street', {}).get('y', -66)),
                'camera_ledge': float(ref_levels.get('camera_ledge', {}).get('y', -15)),
            }
            
            # Build camera visibility from marker data
            camera_visibility = {'Camera 1': [], 'Camera 2': []}
            for marker_id_str, marker_info in markers_data.items():
                marker_id = int(marker_id_str)
                visible_to = marker_info.get('visible_to', [])
                for cam in visible_to:
                    if 'camera_1' in cam.lower() or cam == 'Camera 1':
                        camera_visibility['Camera 1'].append(marker_id)
                    if 'camera_2' in cam.lower() or cam == 'Camera 2':
                        camera_visibility['Camera 2'].append(marker_id)
            
            # Sort visibility lists
            for cam in camera_visibility:
                camera_visibility[cam] = sorted(camera_visibility[cam])
            
            # Extract camera intrinsics and rotation from JSON
            camera_intrinsics = {}
            camera_rotations = {}
            for cam_name, cam_info in cameras_data.items():
                if not isinstance(cam_info, dict):
                    continue
                # Intrinsics
                if 'intrinsics' in cam_info:
                    intr = cam_info['intrinsics']
                    img_size = intr.get('image_size', [2048, 1536])
                    fl = intr.get('focal_length_px', [1220.3, 1220.3])
                    
                    # If FOV is available, compute focal length from it (more reliable)
                    fov_data = cam_info.get('fov', {})
                    if 'horizontal' in fov_data:
                        hfov_rad = math.radians(fov_data['horizontal'])
                        fl_from_fov = (img_size[0] / 2) / math.tan(hfov_rad / 2)
                        fl = [fl_from_fov, fl_from_fov]
                    
                    camera_intrinsics[cam_name] = {
                        'focal_length_px': fl,
                        'principal_point': intr.get('principal_point', [img_size[0]/2, img_size[1]/2]),
                        'image_size': img_size,
                        'dist_coeffs': intr.get('dist_coeffs', [0, 0, 0, 0, 0]),
                    }
                # Rotation (Euler angles)
                if 'rotation' in cam_info:
                    rot = cam_info['rotation']
                    euler = rot.get('euler_deg', {})
                    camera_rotations[cam_name] = {
                        'pitch': euler.get('pitch', 10),
                        'yaw': euler.get('yaw', 0),
                        'roll': euler.get('roll', 0),
                    }
            
            # Extract corner offsets for markers
            corner_offsets_data = data.get('calibration_markers', {}).get('corner_offsets', {})
            corner_offsets = {
                'horizontal': corner_offsets_data.get('horizontal', {}).get('corners', [
                    [-7.5, 0, -7.5], [7.5, 0, -7.5], [7.5, 0, 7.5], [-7.5, 0, 7.5]
                ]),
                'vertical': corner_offsets_data.get('vertical', {}).get('corners', [
                    [-7.5, 7.5, 0], [7.5, 7.5, 0], [7.5, -7.5, 0], [-7.5, -7.5, 0]
                ]),
            }
            
            print(f"📐 Loaded V2 coordinates from {cls.WORLD_COORDS_FILE}")
            print(f"   Markers: {sorted(marker_positions.keys())}")
            print(f"   Vertical markers: {vertical_markers}")
            print(f"   Camera 1 sees: {camera_visibility['Camera 1']}")
            print(f"   Camera 2 sees: {camera_visibility['Camera 2']}")
            if camera_intrinsics:
                print(f"   Camera intrinsics loaded: {list(camera_intrinsics.keys())}")
            if camera_rotations:
                print(f"   Camera rotations loaded: {list(camera_rotations.keys())}")
            
            return {
                'marker_positions': marker_positions,
                'marker_size': marker_size,
                'vertical_markers': vertical_markers,
                'camera_positions': camera_positions,
                'reference_levels': reference_levels,
                'camera_visibility': camera_visibility,
                'camera_intrinsics': camera_intrinsics,
                'camera_rotations': camera_rotations,
                'corner_offsets': corner_offsets,
            }
            
        except FileNotFoundError:
            print(f"⚠️ world_coordinates.json not found at {cls.WORLD_COORDS_FILE}")
            print(f"   Using hardcoded fallback coordinates")
            return None
        except Exception as e:
            print(f"⚠️ Error loading world_coordinates.json: {e}")
            print(f"   Using hardcoded fallback coordinates")
            return None
    
    def __init__(self, calibration_manager, synth_width, synth_height):
        self.calibration = calibration_manager
        self.synth_width = synth_width
        self.synth_height = synth_height
        
        # ArUco setup — tuned for surveillance camera conditions
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Enable built-in sub-pixel corner refinement
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 50
        self.aruco_params.cornerRefinementMinAccuracy = 0.05
        # Adaptive threshold — moderate step count to balance detection vs false positives
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 4
        # Minimum marker size — 0.02 rejects small reflection artifacts
        self.aruco_params.minMarkerPerimeterRate = 0.02
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        # Strict border check to reject reflections/noise that look like markers
        self.aruco_params.maxErroneousBitsInBorderRate = 0.25
        # Default polygon approximation (tighter = fewer false detections)
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        # Minimum distances between corners/markers (prevent duplicates)
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minMarkerDistanceRate = 0.05
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Try to load coordinates from world_coordinates.json (single source of truth)
        world_coords = self.load_world_coordinates()
        
        if world_coords is not None:
            # Use loaded coordinates
            self.marker_world_positions_3d = world_coords['marker_positions']
            self.marker_size_cm = world_coords['marker_size']
            self.vertical_markers = world_coords['vertical_markers']
            self.camera_marker_visibility = world_coords['camera_visibility']
            self._reference_levels = world_coords['reference_levels']
            self._camera_positions = world_coords['camera_positions']
            # NEW: Load intrinsics, rotations, and corner offsets
            self._camera_intrinsics_json = world_coords.get('camera_intrinsics', {})
            self._camera_rotations = world_coords.get('camera_rotations', {})
            self._corner_offsets = world_coords.get('corner_offsets', {
                'horizontal': [[-7.5, 0, -7.5], [7.5, 0, -7.5], [7.5, 0, 7.5], [-7.5, 0, 7.5]],
                'vertical': [[-7.5, 7.5, 0], [7.5, 7.5, 0], [7.5, -7.5, 0], [-7.5, -7.5, 0]],
            })
        else:
            # Fallback to hardcoded V2 coordinates
            self.marker_size_cm = 15.0  # 15cm markers
            
            self.marker_world_positions_3d = {
                # FRONT ROW (Z = 168 cm, Y = street level)
                0: (-30.0, STREET_LEVEL_Y, 168.0),
                1: (-150.0, STREET_LEVEL_Y, 168.0),
                2: (-270.0, STREET_LEVEL_Y, 168.0),
                # BACK ROW (Z = 219 cm, Y = street level)
                3: (-30.0, STREET_LEVEL_Y, 219.0),
                4: (-270.0, STREET_LEVEL_Y, 219.0),
                6: (-150.0, STREET_LEVEL_Y, 219.0),
                # VERTICAL MARKER
                5: (-150.0, CAMERA_LEDGE_Y, 628.0),
            }
            
            self.vertical_markers = {5}
            
            self.camera_marker_visibility = {
                'Camera 1': [0, 1, 3, 5, 6],
                'Camera 2': [1, 2, 4, 5, 6],
            }
            
            self._reference_levels = {
                'floor': 0,
                'street': -66,
                'camera_ledge': -15,
            }
            
            self._camera_positions = {
                'camera_1': (-30, -15, 78),
                'camera_2': (-270, -15, 78),
            }
            # Fallback for new fields
            self._camera_intrinsics_json = {}
            self._camera_rotations = {}
            self._corner_offsets = {
                'horizontal': [[-7.5, 0, -7.5], [7.5, 0, -7.5], [7.5, 0, 7.5], [-7.5, 0, 7.5]],
                'vertical': [[-7.5, 7.5, 0], [7.5, 7.5, 0], [7.5, -7.5, 0], [-7.5, -7.5, 0]],
            }
        
        # The shared marker IDs that must be visible from all cameras
        self.shared_marker_id = 1  # Primary shared marker
        self.shared_markers = {1, 5, 6}  # All shared markers (for validation)
        
        # Camera intrinsics storage (per camera)
        self.camera_intrinsics = {}  # camera_name -> {matrix, dist_coeffs, size}
        
        # Detected markers per camera
        self.detected_markers = {}  # camera_name -> {marker_id: corner_points}
        self.active_camera = 0
        
        # Auto-calibration state
        self.auto_calibrating = False
        self.auto_calib_step = 0
        self.auto_calib_messages = []
        
        # Calibration result overlay state
        self._calib_result = None       # 'success' or 'fail'
        self._calib_result_msg = ''     # summary message
        self._calib_result_details = [] # per-camera detail lines
        self._calib_result_time = 0     # time.time() when result was set
        self._calib_result_duration = 5.0  # seconds to show overlay
        
        self.instructions = [
            "3D CALIBRATION - V2 COORDINATE SYSTEM",
            "----------------------------------------",
            "Origin: Back right corner of Panel 0",
            "X: Negative toward left (Unit 3)",
            "Z: Positive into tracking zone",
            "",
            "Floor markers (flat): 0-4, 6 on street",
            "Vertical marker: 5 on subway wall",
            "",
            "Front row (Z=168):  0---1---2",
            "Back row (Z=219):   3---6---4",
            "Subway wall (Z=628):  --5--  ",
            "",
            "Camera 1 (RIGHT): 0, 1, 3, 5, 6",
            "Camera 2 (LEFT):  1, 2, 4, 5, 6",
            "",
            ">>> Press A for AUTO-CALIBRATE <<<",
            "",
            "Manual: SPACE=detect, ENTER=compute, S=save",
            "Press ESC to exit",
            f"Marker size: {self.marker_size_cm:.0f} cm",
        ]
    
    def get_camera_intrinsics(self, camera_name, image_size):
        """
        Get camera intrinsics - prefer loaded values from world_coordinates.json,
        fall back to estimation if not available.
        
        Returns: (camera_matrix, dist_coeffs)
        """
        width, height = image_size
        
        # Map display name to JSON key
        cam_key = 'camera_1' if 'Camera 1' in camera_name or camera_name == 'Camera 1' else 'camera_2'
        
        # Try to use intrinsics from world_coordinates.json
        if hasattr(self, '_camera_intrinsics_json') and cam_key in self._camera_intrinsics_json:
            intr = self._camera_intrinsics_json[cam_key]
            stored_size = intr.get('image_size', [2048, 1536])
            
            # Get stored values
            fx, fy = intr.get('focal_length_px', [1740.8, 1740.8])
            cx, cy = intr.get('principal_point', [stored_size[0]/2, stored_size[1]/2])
            dist = intr.get('dist_coeffs', [0, 0, 0, 0, 0])
            
            # Scale if current image size differs from stored size
            scale_x = width / stored_size[0]
            scale_y = height / stored_size[1]
            
            camera_matrix = np.array([
                [fx * scale_x, 0, cx * scale_x],
                [0, fy * scale_y, cy * scale_y],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.array(dist, dtype=np.float64)
            
            print(f"  📷 Using intrinsics from world_coordinates.json for {camera_name}")
            print(f"     Focal length: ({fx * scale_x:.1f}, {fy * scale_y:.1f})")
            return camera_matrix, dist_coeffs
        
        # Fall back to estimation
        return self.estimate_camera_intrinsics(image_size)
    
    def estimate_camera_intrinsics(self, image_size):
        """
        Estimate camera intrinsics based on image size.
        Uses reasonable defaults for typical surveillance cameras.
        For better accuracy, use intrinsics from world_coordinates.json.
        """
        width, height = image_size
        
        # Estimate focal length for 80 degree horizontal FOV (Reolink RLC-520A)
        # focal_length = width / (2 * tan(fov/2))
        # For 80 degree FOV: focal_length ≈ width * 0.839
        fov_h_rad = math.radians(80)
        focal_length = width / (2 * math.tan(fov_h_rad / 2))
        
        # Principal point at image center
        cx = width / 2
        cy = height / 2
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume minimal distortion for modern IP cameras
        dist_coeffs = np.zeros(5, dtype=np.float64)
        
        print(f"  📷 Using ESTIMATED intrinsics (no JSON data available)")
        print(f"     Focal length: {focal_length:.1f} (based on 80° HFOV)")
        
        return camera_matrix, dist_coeffs
    
    def detect_markers(self, frame, camera_name):
        """Detect ArUco markers in frame and store corner points.
        
        Applies sub-pixel corner refinement after detection for improved
        localization accuracy (~0.1px vs ~0.5-1px without refinement).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Store image size for intrinsics - use JSON values if available
        h, w = frame.shape[:2]
        if camera_name not in self.camera_intrinsics:
            matrix, dist = self.get_camera_intrinsics(camera_name, (w, h))
            self.camera_intrinsics[camera_name] = {
                'matrix': matrix,
                'dist_coeffs': dist,
                'size': (w, h)
            }
        
        # Apply additional sub-pixel refinement on top of ArUco's built-in refinement
        if corners is not None and len(corners) > 0:
            corners = list(corners)  # detectMarkers returns a tuple; convert so we can update in-place
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
            for i in range(len(corners)):
                corners[i] = cv2.cornerSubPix(
                    gray, corners[i],
                    winSize=(3, 3), zeroZone=(-1, -1),
                    criteria=criteria
                )
        
        if ids is not None:
            if camera_name not in self.detected_markers:
                self.detected_markers[camera_name] = {}
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.marker_world_positions_3d:
                    # Store all 4 corner points for the marker
                    self.detected_markers[camera_name][marker_id] = corners[i][0]
        
        return corners, ids
    
    def compute_3d_pose(self, camera_name):
        """Compute camera 3D pose using solvePnP.
        
        Works with 3+ markers (3 markers × 4 corners = 12 points).
        For limited visibility setups, validates that the shared marker is detected.
        """
        if camera_name not in self.detected_markers:
            return False, "No markers detected for this camera"
        
        if camera_name not in self.camera_intrinsics:
            return False, "Camera intrinsics not available"
        
        markers = self.detected_markers[camera_name]
        if len(markers) < self.MIN_MARKERS_REQUIRED:
            return False, f"Need {self.MIN_MARKERS_REQUIRED}+ markers, have {len(markers)}"
        
        # Check if shared marker is detected (required for multi-camera consistency)
        if self.shared_marker_id not in markers:
            return False, f"SHARED marker {self.shared_marker_id} not detected! Required for multi-camera calibration."
        
        # Validate expected markers for this camera if configured
        if camera_name in self.camera_marker_visibility:
            expected = set(self.camera_marker_visibility[camera_name])
            detected = set(markers.keys())
            missing = expected - detected
            if missing:
                print(f"⚠️ Warning: {camera_name} missing expected markers: {missing}")
        
        intrinsics = self.camera_intrinsics[camera_name]
        camera_matrix = intrinsics['matrix']
        dist_coeffs = intrinsics['dist_coeffs']
        image_size = intrinsics['size']
        
        # Build 3D-2D point correspondences using ALL 4 CORNERS per marker
        # This gives 4x more points for better solvePnP accuracy
        #
        # CORNER ORDERING: ArUco returns corners in a canonical order based on
        # the marker's encoded pattern. When a marker is placed flat on the ground,
        # the mapping between ArUco corners and our 3D offsets depends on the
        # marker's physical rotation. We try all 4 rotations and pick the best.
        
        # Try all 4 cyclic rotations of corner offsets to find the correct mapping
        # Test horizontal and vertical rotations independently (4 × 4 = 16 combos)
        best_reproj = float('inf')
        best_obj_pts = None
        best_img_pts = None
        best_h_rot = 0
        best_v_rot = 0
        
        for h_rot in range(4):
            for v_rot in range(4):
                object_points_trial = []
                image_points_trial = []
                
                for marker_id, corners_2d in markers.items():
                    center_3d = self.marker_world_positions_3d[marker_id]
                    is_vertical = marker_id in self.vertical_markers
                    offset_key = 'vertical' if is_vertical else 'horizontal'
                    base_offsets = self._corner_offsets.get(offset_key, self._corner_offsets['horizontal'])
                    
                    # Apply rotation based on marker type
                    rot = v_rot if is_vertical else h_rot
                    offsets = base_offsets[rot:] + base_offsets[:rot]
                    
                    for i, offset in enumerate(offsets):
                        corner_3d = (
                            center_3d[0] + offset[0],
                            center_3d[1] + offset[1],
                            center_3d[2] + offset[2]
                        )
                        object_points_trial.append(corner_3d)
                        image_points_trial.append(corners_2d[i])
                
                obj_arr = np.array(object_points_trial, dtype=np.float64)
                img_arr = np.array(image_points_trial, dtype=np.float64)
                
                if len(obj_arr) < 4:
                    continue
                
                # Quick solvePnP to test this rotation
                ok, rv, tv = cv2.solvePnP(obj_arr, img_arr, camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    proj, _ = cv2.projectPoints(obj_arr, rv, tv, camera_matrix, dist_coeffs)
                    err = np.sqrt(np.mean(np.sum((img_arr - proj.reshape(-1, 2))**2, axis=1)))
                    if err < best_reproj:
                        best_reproj = err
                        best_obj_pts = obj_arr
                        best_img_pts = img_arr
                        best_h_rot = h_rot
                        best_v_rot = v_rot
        
        if best_obj_pts is None:
            return False, "Could not solve for any corner rotation"
        
        object_points = best_obj_pts
        image_points = best_img_pts
        
        if best_h_rot != 0 or best_v_rot != 0:
            print(f"  Corner rotation: horizontal={best_h_rot}, vertical={best_v_rot} (reproj={best_reproj:.1f}px)")
        else:
            print(f"  Default corner ordering OK (reproj={best_reproj:.1f}px)")
        
        # Need at least 4 points for solvePnP (1 marker = 4 corners)
        if len(object_points) < 4:
            return False, f"Need at least 4 points, have {len(object_points)}"
        
        print(f"  Using {len(object_points)} corner points from {len(markers)} markers")
        
        # Provide initial guess based on expected camera position
        # V2 COORDINATE SYSTEM:
        #   Camera 1 (RIGHT): X=-30, Y=-15 (ledge height), Z=78 (front of active zone)
        #   Camera 2 (LEFT):  X=-270, Y=-15, Z=78
        # Both looking toward positive Z (into the tracking zone)
        
        # Use loaded camera positions if available
        if hasattr(self, '_camera_positions') and self._camera_positions:
            if 'Camera 1' in camera_name or camera_name == 'Camera 1':
                cam_key = 'camera_1'
            else:
                cam_key = 'camera_2'
            
            if cam_key in self._camera_positions:
                cam_pos = self._camera_positions[cam_key]
                init_camera_pos = np.array([float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])])
            else:
                # Fallback
                if 'Camera 1' in camera_name:
                    init_camera_pos = np.array([-30.0, -15.0, 78.0])
                else:
                    init_camera_pos = np.array([-270.0, -15.0, 78.0])
        else:
            if 'Camera 1' in camera_name or camera_name == 'Camera 1':
                init_camera_pos = np.array([-30.0, -15.0, 78.0])
            else:
                init_camera_pos = np.array([-270.0, -15.0, 78.0])
        
        # Build initial rotation guess using look-at toward center of tracking zone.
        # Exact Euler angles from config are not used as initial guess — cameras are
        # manually aimed in the same general direction (down, toward center) and
        # solvePnP computes the precise rotation from detected markers.
        init_target = np.array([-150.0, -66.0, 200.0])
        forward = init_target - init_camera_pos
        forward = forward / np.linalg.norm(forward)
        
        # Camera Z-axis points toward target
        # Camera Y-axis points down (OpenCV convention)
        # World Y is positive upward, so world "up" = [0, +1, 0]
        up_world = np.array([0.0, 1.0, 0.0])  # World up is +Y
        right = np.cross(forward, up_world)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        
        R_init = np.column_stack([right, down, forward])
        print(f"  Using look-at initial rotation toward tracking zone center")
        
        init_rvec, _ = cv2.Rodrigues(R_init)
        init_tvec = -R_init @ init_camera_pos
        
        # Try SQPNP first (robust, no initial guess needed, good for 4+ points)
        print(f"  Trying SQPNP solver (robust, no initial guess)...")
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        
        if success:
            # Check if SQPNP solution is reasonable
            R_check, _ = cv2.Rodrigues(rvec)
            cam_check = -R_check.T @ tvec.flatten()
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
            sqpnp_err = np.sqrt(np.mean(np.sum((image_points - projected.reshape(-1, 2))**2, axis=1)))
            print(f"  SQPNP result: Camera at ({cam_check[0]:.1f}, {cam_check[1]:.1f}, {cam_check[2]:.1f}) cm, reproj={sqpnp_err:.2f}px")
            
            if sqpnp_err > 10.0:
                # SQPNP didn't converge well, try iterative with initial guess
                print(f"  SQPNP reproj too high ({sqpnp_err:.1f}px), trying ITERATIVE with initial guess...")
                success2, rvec2, tvec2 = cv2.solvePnP(
                    object_points, image_points,
                    camera_matrix, dist_coeffs,
                    rvec=init_rvec.astype(np.float64),
                    tvec=init_tvec.reshape(3,1).astype(np.float64),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success2:
                    R_check2, _ = cv2.Rodrigues(rvec2)
                    cam_check2 = -R_check2.T @ tvec2.flatten()
                    projected2, _ = cv2.projectPoints(object_points, rvec2, tvec2, camera_matrix, dist_coeffs)
                    iter_err = np.sqrt(np.mean(np.sum((image_points - projected2.reshape(-1, 2))**2, axis=1)))
                    print(f"  ITERATIVE result: Camera at ({cam_check2[0]:.1f}, {cam_check2[1]:.1f}, {cam_check2[2]:.1f}) cm, reproj={iter_err:.2f}px")
                    
                    # Use whichever solution has lower reprojection error
                    if iter_err < sqpnp_err:
                        print(f"  Using ITERATIVE solution (better)")
                        rvec, tvec = rvec2, tvec2
                    else:
                        print(f"  Keeping SQPNP solution (better)")
        
        if not success:
            # Fallback to iterative with initial guess
            print(f"  SQPNP failed, trying ITERATIVE with initial guess...")
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs,
                rvec=init_rvec.astype(np.float64),
                tvec=init_tvec.reshape(3,1).astype(np.float64),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        if not success:
            return False, "solvePnP failed"
        
        # Check camera position
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec.flatten()
        print(f"  Solution: Camera at X={camera_pos[0]:.1f}, Y={camera_pos[1]:.1f}, Z={camera_pos[2]:.1f} cm")
        
        rvecs, tvecs = [rvec], [tvec]
        
        # Skip the IPPE branch since we're using centers which are non-coplanar
        if False:  # Disabled - was for corner-based calibration
            # Use IPPE for coplanar points (floor markers only)
            # This gives us both possible solutions for the planar ambiguity
            print(f"  Using IPPE (coplanar floor markers)")
            success, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
                object_points, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE
            )
            
            if not success or len(rvecs) == 0:
                return False, "solvePnP IPPE failed"
        
        # Find the solution where camera is at POSITIVE Z (in front of storefront)
        # The cameras are mounted on the storefront looking toward the street (positive Z)
        best_idx = 0
        best_z = float('-inf')
        
        print(f"  Found {len(rvecs)} possible pose(s):")
        for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rv)
            cam_pos = -R.T @ tv.flatten()
            print(f"    Solution {i+1}: Camera at Z={cam_pos[2]:.1f} cm")
            if cam_pos[2] > best_z:
                best_z = cam_pos[2]
                best_idx = i
        
        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]
        
        print(f"  Selected solution {best_idx+1} with Z={best_z:.1f} cm")
        
        # Refine with iterative optimization
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points,
            camera_matrix, dist_coeffs,
            rvec, tvec
        )
        
        # Compute camera position
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec.flatten()
        
        # ================================================================
        # VALIDATION GATE — reject bad calibrations before saving
        # ================================================================
        
        # 1. Compute per-corner reprojection errors to identify outlier markers
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        per_point_errors = np.sqrt(np.sum((image_points - projected.reshape(-1, 2))**2, axis=1))
        reproj_error = np.sqrt(np.mean(per_point_errors**2))  # RMS
        
        # Per-marker error analysis (4 corners per marker)
        marker_ids_list = list(markers.keys())
        marker_errors = {}
        for idx, mid in enumerate(marker_ids_list):
            corner_errors = per_point_errors[idx*4:(idx+1)*4]
            marker_errors[mid] = float(np.mean(corner_errors))
        
        # Report per-marker errors
        worst_marker = max(marker_errors, key=marker_errors.get)
        best_marker = min(marker_errors, key=marker_errors.get)
        print(f"  Per-marker reproj errors:")
        for mid in sorted(marker_errors.keys()):
            err = marker_errors[mid]
            flag = " <-- WORST" if mid == worst_marker and err > 3.0 else ""
            print(f"    Marker {mid}: {err:.2f}px{flag}")
        
        # 2. Reprojection error threshold — reject clearly bad calibrations
        # Note: with uncalibrated lens distortion (dist_coeffs=0), errors of
        # 20-50px are typical for surveillance cameras.  Only reject if clearly wrong.
        MAX_REPROJ_ERROR_PX = 100.0  # pixels — generous for uncalibrated distortion
        if reproj_error > MAX_REPROJ_ERROR_PX:
            return False, (f"Reprojection error too high: {reproj_error:.2f}px "
                          f"(max {MAX_REPROJ_ERROR_PX}px). "
                          f"Worst marker: #{worst_marker} at {marker_errors[worst_marker]:.1f}px")
        
        # 3. Camera Z must be positive (in front of storefront)
        if camera_pos[2] < 0:
            return False, (f"Camera Z={camera_pos[2]:.1f}cm is behind storefront. "
                          f"Marker positions or corner ordering may be wrong.")
        
        # 4. Validate camera position against expected physical location
        cam_key = 'camera_1' if 'Camera 1' in camera_name or camera_name == 'Camera 1' else 'camera_2'
        if hasattr(self, '_camera_positions') and cam_key in self._camera_positions:
            expected = np.array([float(c) for c in self._camera_positions[cam_key]])
            pos_delta = np.linalg.norm(camera_pos - expected)
            MAX_POSITION_DRIFT_CM = 50.0  # generous tolerance for imprecise mounting
            if pos_delta > MAX_POSITION_DRIFT_CM:
                print(f"  ⚠️ WARNING: Computed position is {pos_delta:.0f}cm from expected {tuple(expected.astype(int))}")
                print(f"     This may indicate a detection/correspondence error.")
                # Warning only — don't reject, since cameras may be repositioned
            else:
                print(f"  Position delta from expected: {pos_delta:.1f}cm (OK)")
        
        # 5. Camera Y should be near mounting height (-15 ± tolerance)
        expected_y = self._reference_levels.get('camera_ledge', -15) if hasattr(self, '_reference_levels') else -15
        y_delta = abs(camera_pos[1] - expected_y)
        if y_delta > 30:
            print(f"  ⚠️ WARNING: Camera Y={camera_pos[1]:.1f}cm, expected ~{expected_y}cm (delta={y_delta:.0f}cm)")
        
        # ================================================================
        # PASSED — store calibration
        # ================================================================
        self.calibration.set_calibration(
            camera_name, rvec, tvec, camera_matrix, dist_coeffs, image_size
        )
        
        quality = "excellent" if reproj_error < 5.0 else "good" if reproj_error < 20.0 else "acceptable" if reproj_error < 50.0 else "rough"
        return True, (f"3D pose computed from {len(markers)} markers, "
                     f"reproj error: {reproj_error:.2f}px ({quality}), "
                     f"camera at ({camera_pos[0]:.0f}, {camera_pos[1]:.0f}, {camera_pos[2]:.0f}) cm")
    
    def render_overlay(self, frame, camera_name, detected_corners, detected_ids):
        """Render calibration overlay on frame"""
        # Draw detected markers
        if detected_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)
            
            # Draw world coordinates next to each marker
            for i, marker_id in enumerate(detected_ids.flatten()):
                if marker_id in self.marker_world_positions_3d:
                    center = detected_corners[i][0].mean(axis=0).astype(int)
                    world = self.marker_world_positions_3d[marker_id]
                    label = f"({world[0]:.1f},{world[1]:.1f},{world[2]:.1f})m"
                    cv2.putText(frame, label, (center[0] + 30, center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Show detection count
        count = len(self.detected_markers.get(camera_name, {}))
        detected_ids = list(self.detected_markers.get(camera_name, {}).keys())
        has_shared = self.shared_marker_id in detected_ids
        status_color = (0, 255, 0) if count >= self.MIN_MARKERS_REQUIRED and has_shared else (0, 255, 255)
        cv2.putText(frame, f"Markers: {count}/{self.MIN_MARKERS_REQUIRED}+ (IDs: {detected_ids})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show shared marker status
        shared_status = "SHARED #1 detected" if has_shared else "SHARED #1 MISSING!"
        shared_color = (0, 255, 0) if has_shared else (0, 0, 255)
        cv2.putText(frame, shared_status, (10, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, shared_color, 2)
        
        # Show calibration status
        if camera_name in self.calibration.calibrations:
            cv2.putText(frame, "3D CALIBRATED", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show camera position in cm
            cam_pos = self.calibration.get_camera_position(camera_name)
            if cam_pos is not None:
                pos_text = f"Cam pos: ({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}) cm"
                cv2.putText(frame, pos_text, (10, 98),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show expected markers for this camera
        if camera_name in self.camera_marker_visibility:
            expected = self.camera_marker_visibility[camera_name]
            cv2.putText(frame, f"Expected markers: {expected}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return frame
    
    def render_instructions(self, width, height):
        """Render instruction panel for limited visibility calibration"""
        panel = np.zeros((height, 450, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        y = 30
        for line in self.instructions:
            cv2.putText(panel, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            y += 16
        
        # Show camera-marker assignments
        y += 10
        cv2.putText(panel, "Camera -> Marker Assignments:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        y += 20
        
        for cam_name, markers in self.camera_marker_visibility.items():
            # Highlight shared marker
            marker_str = ', '.join([f"*{m}*" if m == self.shared_marker_id else str(m) for m in markers])
            cv2.putText(panel, f"  {cam_name}: [{marker_str}]", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 150), 1)
            y += 16
        
        # Show marker positions
        y += 10
        cv2.putText(panel, "Marker 3D Positions (meters):", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        y += 20
        
        # Show shared marker first with highlight
        if self.shared_marker_id in self.marker_world_positions_3d:
            pos = self.marker_world_positions_3d[self.shared_marker_id]
            cv2.putText(panel, f"  ID {self.shared_marker_id} [SHARED]: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            y += 16
        
        # Show other markers
        for marker_id, pos in sorted(self.marker_world_positions_3d.items()):
            if marker_id == self.shared_marker_id:
                continue
            cv2.putText(panel, f"  ID {marker_id}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
            y += 16
            if y > height - 100:
                break
        
        # Show detected markers summary
        y += 10
        cv2.putText(panel, "Detection Status:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        y += 20
        
        for cam_name, markers in self.detected_markers.items():
            marker_ids = sorted(markers.keys())
            has_shared = self.shared_marker_id in marker_ids
            status_icon = "✓" if has_shared and len(marker_ids) >= self.MIN_MARKERS_REQUIRED else "○"
            color = (0, 255, 0) if has_shared else (0, 150, 255)
            cv2.putText(panel, f"  {status_icon} {cam_name}: {marker_ids}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            y += 16
        
        # Show auto-calibration status
        if self.auto_calibrating:
            y += 10
            cv2.putText(panel, "AUTO-CALIBRATING...", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 20
            for msg in self.auto_calib_messages[-5:]:  # Show last 5 messages
                cv2.putText(panel, f"  {msg}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y += 14
        
        return panel
    
    def render_calibration_result_overlay(self, frame):
        """Render a prominent on-screen banner showing calibration success or failure.
        
        Returns the frame with overlay drawn, or the original frame if no result
        is active or the display duration has expired.
        """
        import time as _time
        if self._calib_result is None:
            return frame
        
        elapsed = _time.time() - self._calib_result_time
        if elapsed > self._calib_result_duration:
            self._calib_result = None  # expire
            return frame
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Banner dimensions
        banner_h = 50 + 28 * len(self._calib_result_details)
        banner_w = min(w - 40, 500)
        x0 = (w - banner_w) // 2
        y0 = (h - banner_h) // 2
        
        # Colors based on result
        if self._calib_result == 'success':
            bg_color = (30, 120, 30)      # dark green
            border_color = (80, 255, 80)   # bright green
            title = "CALIBRATION SUCCESSFUL"
        else:
            bg_color = (30, 30, 120)       # dark red
            border_color = (80, 80, 255)   # bright red
            title = "CALIBRATION FAILED"
        
        # Draw filled banner with border
        cv2.rectangle(overlay, (x0, y0), (x0 + banner_w, y0 + banner_h), bg_color, -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + banner_w, y0 + banner_h), border_color, 3)
        
        # Blend overlay (semi-transparent background)
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Title text
        cv2.putText(frame, title, (x0 + 20, y0 + 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
        
        # Summary line
        cv2.putText(frame, self._calib_result_msg, (x0 + 20, y0 + 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        # Detail lines (camera positions)
        for i, detail in enumerate(self._calib_result_details):
            color = (200, 255, 200) if self._calib_result == 'success' else (200, 200, 255)
            cv2.putText(frame, detail, (x0 + 30, y0 + 86 + i * 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Countdown bar at bottom of banner
        remaining = max(0, 1.0 - elapsed / self._calib_result_duration)
        bar_y = y0 + banner_h - 6
        bar_w = int((banner_w - 4) * remaining)
        cv2.rectangle(frame, (x0 + 2, bar_y), (x0 + 2 + bar_w, bar_y + 4), border_color, -1)
        
        return frame
    
    # Number of frames to accumulate for robust corner averaging
    AUTO_CALIB_NUM_FRAMES = 10
    AUTO_CALIB_FRAME_DELAY = 0.05  # seconds between frame captures
    
    def run_auto_calibration(self, raw_frames, camera_configs, calibration_manager, camera_readers=None):
        """
        Run automatic calibration for all cameras with multi-frame accumulation.
        
        Captures AUTO_CALIB_NUM_FRAMES frames per camera, detects markers on each,
        and uses median corner positions across frames.  This eliminates single-frame
        noise from motion blur, compression artefacts, or lighting flicker.
        
        Args:
            raw_frames: dict of camera_name -> current frame (used as fallback)
            camera_configs: list of camera config dicts with 'name' key
            calibration_manager: CalibrationManager instance to save to
            camera_readers: optional list of CameraReader objects for multi-frame capture
            
        Returns:
            (success, message) tuple
        """
        self.auto_calibrating = True
        self.auto_calib_messages = []
        results = []
        
        N = self.AUTO_CALIB_NUM_FRAMES
        N = 10  # frames for multi-frame averaging
        use_multi = camera_readers is not None and len(camera_readers) > 0
        
        print("\n" + "=" * 50)
        print("🎯 AUTO-CALIBRATION STARTED")
        if use_multi:
            print(f"   Accumulating {N} frames per camera for robust detection")
        print("=" * 50)
        
        # ---------------------------------------------------------------
        # Step 1: Multi-frame marker detection with corner accumulation
        # ---------------------------------------------------------------
        step1_label = f"Step 1: Capturing {N} frames per camera..." if use_multi else "Step 1: Detecting markers..."
        self.auto_calib_messages.append(step1_label)
        print(f"\n📍 {step1_label}")
        
        # corner_accumulator[cam_name][marker_id] = list of (4,2) arrays
        corner_accumulator = {}
        
        for cam_idx, cfg in enumerate(camera_configs):
            cam_name = cfg['name']
            corner_accumulator[cam_name] = {}
            frame_ok_count = 0
            
            num_frames = N if use_multi else 1
            
            for frame_i in range(num_frames):
                # Get a fresh frame from the camera reader if available
                frame = None
                if use_multi and cam_idx < len(camera_readers):
                    ret, f, _, _ = camera_readers[cam_idx].read_new()
                    if not ret:
                        ret, f, _ = camera_readers[cam_idx].read()
                    if ret and f is not None:
                        frame = f
                
                # Fallback to raw_frames on first iteration
                if frame is None and frame_i == 0:
                    frame = raw_frames.get(cam_name)
                
                if frame is None:
                    continue
                
                # Detect markers (without storing — we accumulate ourselves)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)
                
                # Sub-pixel refine
                if corners is not None and len(corners) > 0:
                    corners = list(corners)  # detectMarkers returns a tuple; convert for in-place update
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
                    for ci in range(len(corners)):
                        corners[ci] = cv2.cornerSubPix(
                            gray, corners[ci],
                            winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                
                if ids is not None:
                    frame_ok_count += 1
                    for ci, mid in enumerate(ids.flatten()):
                        mid = int(mid)
                        if mid in self.marker_world_positions_3d:
                            if mid not in corner_accumulator[cam_name]:
                                corner_accumulator[cam_name][mid] = []
                            corner_accumulator[cam_name][mid].append(corners[ci][0].copy())
                
                # Brief pause to get a different frame from the camera thread
                # Use cv2.waitKey instead of time.sleep to keep GUI responsive
                if frame_i < num_frames - 1 and use_multi:
                    delay_ms = max(1, int(self.AUTO_CALIB_FRAME_DELAY * 1000))
                    cv2.waitKey(delay_ms)
            
            # Now compute median corners and store into self.detected_markers
            self.detected_markers[cam_name] = {}
            
            # Ensure camera intrinsics are initialized
            if cam_name not in self.camera_intrinsics:
                frame = raw_frames.get(cam_name)
                if frame is not None:
                    h, w = frame.shape[:2]
                    matrix, dist = self.get_camera_intrinsics(cam_name, (w, h))
                    self.camera_intrinsics[cam_name] = {
                        'matrix': matrix, 'dist_coeffs': dist, 'size': (w, h)
                    }
            
            # Require marker detected in at least half the frames to count as real
            min_detections = max(3, num_frames // 2)
            
            for mid, corner_list in corner_accumulator[cam_name].items():
                if len(corner_list) >= min_detections:
                    # Use median across frames (robust to single-frame outliers)
                    stacked = np.array(corner_list)  # (num_frames, 4, 2)
                    median_corners = np.median(stacked, axis=0)  # (4, 2)
                    
                    # Compute corner spread (std) as quality metric
                    corner_std = np.mean(np.std(stacked, axis=0))
                    if corner_std > 5.0:
                        print(f"  ⚠️ {cam_name} marker {mid}: rejected — corner jitter too high ({corner_std:.1f}px)")
                        continue  # Skip unreliable detections
                    elif corner_std > 3.0:
                        print(f"  ⚠️ {cam_name} marker {mid}: high corner jitter ({corner_std:.1f}px) across {len(corner_list)} frames")
                    
                    self.detected_markers[cam_name][mid] = median_corners.astype(np.float32)
                else:
                    # Not enough consistent detections — likely a false positive
                    if len(corner_list) > 0:
                        print(f"  ⚠️ {cam_name} marker {mid}: rejected — only {len(corner_list)}/{num_frames} detections (need {min_detections})")
            
            detected_ids = sorted(self.detected_markers.get(cam_name, {}).keys())
            if detected_ids:
                msg = f"  ✓ {cam_name}: Markers {detected_ids} ({frame_ok_count}/{num_frames} frames used)"
            else:
                msg = f"  ✗ {cam_name}: No markers detected"
            self.auto_calib_messages.append(msg)
            print(msg)
        
        # Step 2: Compute 3D pose for each camera
        self.auto_calib_messages.append("Step 2: Computing 3D poses...")
        print("\n🔧 Step 2: Computing 3D poses...")
        cv2.waitKey(1)  # Keep GUI responsive
        
        success_count = 0
        for cam_idx, cfg in enumerate(camera_configs):
            cam_name = cfg['name']
            cv2.waitKey(1)  # Keep GUI responsive between cameras
            
            success, msg = self.compute_3d_pose(cam_name)
            
            if success:
                success_count += 1
                cam_pos = calibration_manager.get_camera_position(cam_name)
                if cam_pos is not None:
                    pos_msg = f"  ✓ {cam_name}: Calibrated at ({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}) cm"
                else:
                    pos_msg = f"  ✓ {cam_name}: Calibrated"
                self.auto_calib_messages.append(pos_msg)
                print(pos_msg)
                results.append((cam_name, True, msg))
            else:
                err_msg = f"  ✗ {cam_name}: {msg}"
                self.auto_calib_messages.append(err_msg)
                print(err_msg)
                results.append((cam_name, False, msg))
        
        # Step 3: Save calibration if any succeeded
        if success_count > 0:
            self.auto_calib_messages.append(f"Step 3: Saving calibration ({success_count} cameras)...")
            print(f"\n💾 Step 3: Saving calibration...")
            calibration_manager.save()
            
            final_msg = f"AUTO-CALIBRATION COMPLETE: {success_count}/{len(camera_configs)} cameras calibrated"
            self.auto_calib_messages.append(final_msg)
            print(f"\n✅ {final_msg}")
            print("=" * 50 + "\n")
            
            # Set on-screen success overlay
            self._calib_result = 'success'
            self._calib_result_msg = f"{success_count}/{len(camera_configs)} cameras calibrated successfully"
            self._calib_result_details = []
            for cam_name, ok, msg in results:
                cam_pos = calibration_manager.get_camera_position(cam_name) if ok else None
                if ok and cam_pos is not None:
                    # Extract reproj error from msg (format: "...reproj error: X.XXpx (quality)...")
                    import re as _re
                    reproj_match = _re.search(r'reproj error: ([\d.]+)px \((\w+)\)', msg)
                    reproj_info = f" | {reproj_match.group(2)} ({reproj_match.group(1)}px)" if reproj_match else ""
                    self._calib_result_details.append(
                        f"{cam_name}: ({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}) cm{reproj_info}")
                elif ok:
                    self._calib_result_details.append(f"{cam_name}: calibrated")
                else:
                    self._calib_result_details.append(f"{cam_name}: FAILED - {msg}")
            import time as _time
            self._calib_result_time = _time.time()
            
            self.auto_calibrating = False
            return True, final_msg
        else:
            final_msg = "AUTO-CALIBRATION FAILED: No cameras could be calibrated"
            self.auto_calib_messages.append(final_msg)
            print(f"\n❌ {final_msg}")
            print("   Make sure markers are visible to all cameras")
            print("=" * 50 + "\n")
            
            # Set on-screen failure overlay
            self._calib_result = 'fail'
            self._calib_result_msg = "No cameras could be calibrated"
            self._calib_result_details = ["Ensure markers are visible to all cameras"]
            for cam_name, ok, msg in results:
                self._calib_result_details.append(f"{cam_name}: {msg}")
            import time as _time
            self._calib_result_time = _time.time()
            
            self.auto_calibrating = False
            return False, final_msg
    
    @classmethod
    def generate_markers(cls, output_dir='aruco_markers', num_markers=7):
        """Generate printable ArUco marker images with 3D position labels.
        
        Loads positions from world_coordinates.json if available.
        
        V2 COORDINATE SYSTEM:
        - Origin at back right corner of Panel 0
        - Markers 1, 5, 6 are SHARED (visible from both cameras)
        - Camera 1 (RIGHT): sees 0, 1, 3, 5, 6
        - Camera 2 (LEFT): sees 1, 2, 4, 5, 6
        """
        import os as _os
        _os.makedirs(output_dir, exist_ok=True)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Try to load from world_coordinates.json
        world_coords = cls.load_world_coordinates()
        
        if world_coords is not None:
            default_positions = world_coords['marker_positions']
            # Build camera assignments from visibility data
            visibility = world_coords['camera_visibility']
            camera_assignments = {}
            for marker_id in default_positions.keys():
                in_cam1 = marker_id in visibility.get('Camera 1', [])
                in_cam2 = marker_id in visibility.get('Camera 2', [])
                if in_cam1 and in_cam2:
                    camera_assignments[marker_id] = "SHARED (both cameras)"
                elif in_cam1:
                    camera_assignments[marker_id] = "Camera 1 (RIGHT)"
                elif in_cam2:
                    camera_assignments[marker_id] = "Camera 2 (LEFT)"
                else:
                    camera_assignments[marker_id] = "Unknown"
        else:
            # Fallback to hardcoded V2 positions
            default_positions = {
                0: (-30.0, -66.0, 168.0),
                1: (-150.0, -66.0, 168.0),
                2: (-270.0, -66.0, 168.0),
                3: (-30.0, -66.0, 219.0),
                4: (-270.0, -66.0, 219.0),
                5: (-150.0, -15.0, 628.0),
                6: (-150.0, -66.0, 219.0),
            }
            camera_assignments = {
                0: "Camera 1 (RIGHT front)",
                1: "SHARED (both cameras)",
                2: "Camera 2 (LEFT front)",
                3: "Camera 1 (RIGHT back)",
                4: "Camera 2 (LEFT back)",
                5: "SHARED (subway wall)",
                6: "SHARED (center back)",
            }
        
        for marker_id in range(num_markers):
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
            # Add white border for printing
            bordered = cv2.copyMakeBorder(marker_img, 50, 100, 50, 50,
                                         cv2.BORDER_CONSTANT, value=255)
            # Add ID label
            cv2.putText(bordered, f"ID: {marker_id}", (180, 480),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            # Add camera assignment
            if marker_id in camera_assignments:
                cv2.putText(bordered, camera_assignments[marker_id],
                           (100, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
            # Add suggested position (in cm)
            if marker_id in default_positions:
                pos = default_positions[marker_id]
                cv2.putText(bordered, f"Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) cm", 
                           (80, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            cv2.imwrite(f"{output_dir}/marker_{marker_id}.png", bordered)
        
        print(f"✅ Generated {num_markers} markers in {output_dir}/")
        print(f"   V2 COORDINATE SYSTEM - Origin at back right of Panel 0")
        print(f"   Front row (Z=168): Marker 0 @ X=-30, Marker 1 @ X=-150 (SHARED), Marker 2 @ X=-270")
        print(f"   Back row (Z=219):  Marker 3 @ X=-30, Marker 6 @ X=-150 (SHARED), Marker 4 @ X=-270")
        print(f"   Subway wall (Z=628): Marker 5 @ X=-150 (SHARED, vertical)")
        print(f"   Camera 1 (RIGHT): sees 0, 1, 3, 5, 6")
        print(f"   Camera 2 (LEFT):  sees 1, 2, 4, 5, 6")


class RobustCamera:
    """
    Reliable camera capture with automatic reconnection and frame caching.
    Optimized for multi-camera synchronized tracking.
    
    Features:
    - Automatic reconnection on stream failure
    - Frame caching for display continuity
    - Health monitoring and statistics
    - Thread-safe frame access
    - Configurable timeouts and retry logic
    """
    def __init__(self, name, src, target_fps=25):
        self.name = name
        self.src = src
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Frame storage
        self.frame = None
        self.cached_frame = None  # Last good frame for display continuity
        self.frame_time = 0
        self.frame_number = 0
        self.last_returned_frame_number = -1
        
        # State
        self.running = True
        self.connected = False
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'frames_dropped': 0,
            'reconnect_count': 0,
            'last_error': None,
            'connect_time': None,
        }
        
        # Connection
        self.cap = None
        self.width = 0
        self.height = 0
        
        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"Camera-{name}")
        self.thread.start()
    
    def _connect(self):
        """Establish connection to camera with proper settings"""
        try:
            # Set FFmpeg options for low latency RTSP
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'fflags;nobuffer+discardcorrupt|'
                'flags;low_delay|'
                'max_delay;500000|'
                'stimeout;5000000'
            )
            
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Wait for connection with timeout
            start_time = time.time()
            while time.time() - start_time < CONNECTION_TIMEOUT:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.width = frame.shape[1]
                        self.height = frame.shape[0]
                        self.connected = True
                        self.stats['connect_time'] = time.time()
                        print(f"   ✓ {self.name} connected: {self.width}x{self.height}")
                        
                        # Store initial frame
                        with self.lock:
                            self.frame = frame
                            self.cached_frame = frame.copy()
                            self.frame_time = time.time()
                            self.frame_number = 1
                            self.stats['frames_received'] = 1
                        return True
                time.sleep(0.1)
            
            self.stats['last_error'] = "Connection timeout"
            return False
            
        except Exception as e:
            self.stats['last_error'] = str(e)
            print(f"   ✗ {self.name} connection error: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop with automatic reconnection"""
        # Initial connection
        if not self._connect():
            print(f"   ✗ {self.name} failed initial connection, will retry...")
        
        consecutive_failures = 0
        
        while self.running:
            if not self.connected:
                # Attempt reconnection
                time.sleep(RECONNECT_DELAY)
                self.stats['reconnect_count'] += 1
                print(f"   🔄 {self.name} reconnecting (attempt {self.stats['reconnect_count']})...")
                if self._connect():
                    consecutive_failures = 0
                continue
            
            # Read frame
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
                    if consecutive_failures > 30:  # ~1 second of failures
                        print(f"   ⚠️ {self.name} stream lost, reconnecting...")
                        self.connected = False
                        consecutive_failures = 0
                    time.sleep(0.01)
                    
            except Exception as e:
                self.stats['last_error'] = str(e)
                consecutive_failures += 1
                if consecutive_failures > 10:
                    self.connected = False
                time.sleep(0.01)
    
    def read(self):
        """
        Get the most recent frame.
        Returns: (success, frame, latency_ms)
        """
        with self.lock:
            if self.frame is not None:
                age = time.time() - self.frame_time
                if age < MAX_FRAME_AGE:
                    latency_ms = age * 1000
                    return True, self.frame.copy(), latency_ms
            
            # Return cached frame if available (for display continuity)
            if FRAME_CACHE_ENABLED and self.cached_frame is not None:
                return True, self.cached_frame.copy(), -1  # -1 indicates cached
            
            return False, None, 0
    
    def read_new(self):
        """
        Get frame only if it's new since last read_new() call.
        Essential for synchronized multi-camera processing.
        Returns: (success, frame, latency_ms, is_new)
        """
        with self.lock:
            # Check if we have a new frame
            is_new = self.frame_number > self.last_returned_frame_number
            
            if self.frame is not None:
                age = time.time() - self.frame_time
                
                if age < MAX_FRAME_AGE:
                    if is_new:
                        self.last_returned_frame_number = self.frame_number
                    latency_ms = age * 1000
                    return True, self.frame.copy(), latency_ms, is_new
            
            # Return cached frame for display (but mark as not new)
            if FRAME_CACHE_ENABLED and self.cached_frame is not None:
                return True, self.cached_frame.copy(), -1, False
            
            return False, None, 0, False
    
    def is_healthy(self):
        """Check if camera is connected and receiving frames"""
        with self.lock:
            if not self.connected:
                return False
            age = time.time() - self.frame_time
            return age < MAX_FRAME_AGE * 2
    
    def get_stats(self):
        """Get camera statistics"""
        with self.lock:
            return {
                'name': self.name,
                'connected': self.connected,
                'frames_received': self.stats['frames_received'],
                'frames_dropped': self.stats['frames_dropped'],
                'reconnect_count': self.stats['reconnect_count'],
                'frame_age': time.time() - self.frame_time if self.frame_time > 0 else -1,
                'last_error': self.stats['last_error'],
            }
    
    def isOpened(self):
        return self.connected or self.cached_frame is not None
    
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap is not None:
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
    print("Multi-Camera YOLO Person Tracking - CUDA Optimized")
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
    
    # Initialize calibration and synthesized view
    calibration = CalibrationManager(CALIBRATION_FILE)
    synth_view = SynthesizedView(SYNTH_VIEW_WIDTH, SYNTH_VIEW_HEIGHT, calibration)
    calib_mode = CalibrationMode(calibration, SYNTH_VIEW_WIDTH, SYNTH_VIEW_HEIGHT)
    
    # Connect to cameras from CAMERAS config
    cameras = []
    camera_configs = []
    raw_frames = {}  # Store raw frames for calibration
    
    enabled_cameras = [c for c in CAMERAS if c.get('enabled', True)]
    print(f"\n📹 Connecting to {len(enabled_cameras)} cameras...")
    
    for cam_cfg in CAMERAS:
        if not cam_cfg.get('enabled', True):
            continue
        
        name = cam_cfg['name']
        url = cam_cfg['url']
        fps = cam_cfg.get('fps', 25)
        
        # Use RobustCamera for reliable capture with auto-reconnection
        cap = RobustCamera(name, url, target_fps=fps)
        
        # Wait for initial connection
        connect_start = time.time()
        while time.time() - connect_start < CONNECTION_TIMEOUT:
            if cap.isOpened() and cap.width > 0:
                break
            time.sleep(0.1)
        
        if cap.isOpened() and cap.width > 0:
            cameras.append(cap)
            camera_configs.append({
                'name': name,
                'fps': fps,
                'width': cap.width,
                'height': cap.height,
                'scale': PROCESS_WIDTH / cap.width,
                'process_height': int(cap.height * (PROCESS_WIDTH / cap.width)),
                'track_colors': {},
                'fps_history': [],
                'last_boxes': [],
                'current_fps': 0,
                'current_latency': 0,
            })
            print(f"   ✅ {name} ready: {cap.width}x{cap.height}")
        else:
            print(f"   ⚠️ {name} initial connection pending (will auto-reconnect)")
            # Still add it - RobustCamera will keep trying
            cameras.append(cap)
            camera_configs.append({
                'name': name,
                'fps': fps,
                'width': 1920,  # Default until connected
                'height': 1080,
                'scale': PROCESS_WIDTH / 1920,
                'process_height': int(1080 * (PROCESS_WIDTH / 1920)),
                'track_colors': {},
                'fps_history': [],
                'last_boxes': [],
                'current_fps': 0,
                'current_latency': 0,
            })
    
    if len(cameras) == 0:
        print("❌ ERROR: No cameras available")
        sys.exit(1)
    
    num_cameras = len(cameras)
    print(f"\n🎬 Running with {num_cameras} camera(s)")
    
    # View mode state
    view_mode = ViewMode.SIDE_BY_SIDE if num_cameras > 1 else ViewMode.INDIVIDUAL
    individual_cam_idx = 0
    global_track_colors = {}  # Shared colors for synthesized view
    
    # Calculate display sizes for each mode
    def calculate_display_configs(mode, selected_cam=0):
        configs = []
        
        if mode == ViewMode.INDIVIDUAL:
            # Single camera fullscreen
            cfg = camera_configs[selected_cam]
            display_scale = DISPLAY_WIDTH / cfg['width']
            display_height = int(cfg['height'] * display_scale)
            for i, c in enumerate(camera_configs):
                if i == selected_cam:
                    configs.append({
                        'width': DISPLAY_WIDTH,
                        'height': display_height,
                        'scale': display_scale,
                        'target_height': display_height,
                    })
                else:
                    configs.append(None)  # Not displayed
        
        elif mode == ViewMode.SIDE_BY_SIDE:
            per_camera_width = DISPLAY_WIDTH // num_cameras
            max_height = 0
            for cfg in camera_configs:
                display_scale = per_camera_width / cfg['width']
                display_height = int(cfg['height'] * display_scale)
                max_height = max(max_height, display_height)
                configs.append({
                    'width': per_camera_width,
                    'height': display_height,
                    'scale': display_scale,
                })
            for c in configs:
                c['target_height'] = max_height
        
        elif mode == ViewMode.GRID:
            # Calculate grid dimensions
            grid_cols = math.ceil(math.sqrt(num_cameras))
            grid_rows = math.ceil(num_cameras / grid_cols)
            per_camera_width = DISPLAY_WIDTH // grid_cols
            max_height = 0
            for cfg in camera_configs:
                display_scale = per_camera_width / cfg['width']
                display_height = int(cfg['height'] * display_scale)
                max_height = max(max_height, display_height)
                configs.append({
                    'width': per_camera_width,
                    'height': display_height,
                    'scale': display_scale,
                    'grid_cols': grid_cols,
                    'grid_rows': grid_rows,
                })
            for c in configs:
                c['target_height'] = max_height
        
        elif mode in (ViewMode.SYNTHESIZED, ViewMode.CALIBRATION):
            # Show cameras smaller on left, synth view on right
            per_camera_width = 320
            for cfg in camera_configs:
                display_scale = per_camera_width / cfg['width']
                display_height = int(cfg['height'] * display_scale)
                configs.append({
                    'width': per_camera_width,
                    'height': display_height,
                    'scale': display_scale,
                    'target_height': SYNTH_VIEW_HEIGHT // num_cameras,
                })
        
        return configs
    
    display_configs = calculate_display_configs(view_mode)
    
    print(f"   Processing at: {PROCESS_WIDTH}px width")
    print(f"\n🎬 Starting tracking...")
    print("   Keys: 1-9=Camera, S=Side-by-side, G=Grid, T=Synthesized, C=Calibrate, Q=Quit")
    print("   In Calibration mode: A=Auto-calibrate (one button!)\n")
    
    # Tracking state
    frame_count = 0
    start_time = time.time()
    
    # Frame pacing - use TARGET_FPS for synchronized mode
    if SYNC_CAMERAS:
        frame_interval = 1.0 / TARGET_FPS
    else:
        max_fps = max(cfg['fps'] for cfg in camera_configs)
        frame_interval = 1.0 / max_fps
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
        all_detections = []  # For synthesized view
        total_people = 0
        
        # Process each camera
        frames_ready = []
        for cam_idx, (cap, cfg) in enumerate(zip(cameras, camera_configs)):
            # Use read_new() for synchronized multi-camera processing
            # This returns cached frames for display continuity but tracks new frames for processing
            ret, frame, latency_ms, is_new = cap.read_new()
            
            if not ret or frame is None:
                raw_frames[cfg['name']] = None
                cfg['last_boxes'] = []
                continue
            
            # Update camera config dimensions if they changed (e.g., after reconnection)
            if cap.width > 0 and cap.width != cfg['width']:
                cfg['width'] = cap.width
                cfg['height'] = cap.height
                cfg['scale'] = PROCESS_WIDTH / cap.width
                cfg['process_height'] = int(cap.height * cfg['scale'])
            
            raw_frames[cfg['name']] = frame.copy()
            
            # Only run YOLO on new frames (saves GPU cycles, maintains sync)
            if is_new:
                # Run YOLO tracking
                small_frame = processor.resize(frame, (PROCESS_WIDTH, cfg['process_height']))
                
                results = model.track(
                    small_frame,
                    persist=True,
                    classes=TRACKED_CLASSES,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    imgsz=PROCESS_WIDTH,
                    tracker="bytetrack.yaml",
                    device=device
                )
                
                # Extract detections
                raw_detections = []
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, x2 = int(x1 / cfg['scale']), int(x2 / cfg['scale'])
                        y1, y2 = int(y1 / cfg['scale']), int(y2 / cfg['scale'])
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -1
                        class_id = int(box.cls[0])  # Get the detected class
                        raw_detections.append((x1, y1, x2, y2, conf, track_id, class_id))
                
                # Merge person + bicycle detections into cyclists
                cfg['last_boxes'] = merge_cyclists(raw_detections)
            
            all_detections.append((cfg['name'], cfg['last_boxes']))
            total_people += len(cfg['last_boxes'])
            
            # Calculate FPS (only count new frames)
            if is_new:
                process_time = time.time() - frame_start
                instant_fps = 1.0 / max(process_time, 0.001)
                cfg['fps_history'].append(instant_fps)
                if len(cfg['fps_history']) > 25:  # Use fixed window
                    cfg['fps_history'].pop(0)
                fps = sum(cfg['fps_history']) / len(cfg['fps_history'])
                cfg['current_fps'] = fps
            
            # Update latency (even for cached frames, show last known)
            if latency_ms >= 0:
                cfg['current_latency'] = latency_ms
        
        # Render based on view mode
        if view_mode == ViewMode.INDIVIDUAL:
            dcfg = display_configs[individual_cam_idx]
            if dcfg and raw_frames.get(camera_configs[individual_cam_idx]['name']) is not None:
                combined = render_camera_frame(
                    raw_frames[camera_configs[individual_cam_idx]['name']],
                    camera_configs[individual_cam_idx],
                    dcfg, processor, individual_cam_idx, global_track_colors
                )
            else:
                combined = np.zeros((400, DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.putText(combined, "No Signal", (DISPLAY_WIDTH//2 - 50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        elif view_mode == ViewMode.SIDE_BY_SIDE:
            for cam_idx, (cfg, dcfg) in enumerate(zip(camera_configs, display_configs)):
                frame = raw_frames.get(cfg['name'])
                if frame is not None:
                    display_frames.append(render_camera_frame(
                        frame, cfg, dcfg, processor, cam_idx, global_track_colors
                    ))
                else:
                    placeholder = np.zeros((dcfg['target_height'], dcfg['width'], 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"{cfg['name']} - No Signal", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    display_frames.append(placeholder)
            combined = cv2.hconcat(display_frames) if display_frames else np.zeros((400, DISPLAY_WIDTH, 3), dtype=np.uint8)
        
        elif view_mode == ViewMode.GRID:
            grid_cols = display_configs[0]['grid_cols'] if display_configs else 2
            grid_rows = display_configs[0]['grid_rows'] if display_configs else 2
            target_height = display_configs[0]['target_height'] if display_configs else 300
            
            rows = []
            for row in range(grid_rows):
                row_frames = []
                for col in range(grid_cols):
                    idx = row * grid_cols + col
                    if idx < num_cameras:
                        cfg = camera_configs[idx]
                        dcfg = display_configs[idx]
                        frame = raw_frames.get(cfg['name'])
                        if frame is not None:
                            row_frames.append(render_camera_frame(
                                frame, cfg, dcfg, processor, idx, global_track_colors
                            ))
                        else:
                            placeholder = np.zeros((dcfg['target_height'], dcfg['width'], 3), dtype=np.uint8)
                            cv2.putText(placeholder, f"{cfg['name']}", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                            row_frames.append(placeholder)
                    else:
                        # Empty cell
                        row_frames.append(np.zeros((target_height, display_configs[0]['width'], 3), dtype=np.uint8))
                rows.append(cv2.hconcat(row_frames))
            combined = cv2.vconcat(rows) if rows else np.zeros((400, DISPLAY_WIDTH, 3), dtype=np.uint8)
        
        elif view_mode == ViewMode.SYNTHESIZED:
            # Camera thumbnails on left, synth view on right
            thumb_frames = []
            for cam_idx, (cfg, dcfg) in enumerate(zip(camera_configs, display_configs)):
                frame = raw_frames.get(cfg['name'])
                if frame is not None:
                    thumb = render_camera_frame(frame, cfg, dcfg, processor, cam_idx, global_track_colors, compact=True)
                else:
                    thumb = np.zeros((dcfg['target_height'], dcfg['width'], 3), dtype=np.uint8)
                thumb_frames.append(thumb)
            
            # Stack thumbnails vertically
            thumb_column = cv2.vconcat(thumb_frames) if thumb_frames else np.zeros((SYNTH_VIEW_HEIGHT, 320, 3), dtype=np.uint8)
            
            # Render synthesized view
            synth_frame = synth_view.render(all_detections, global_track_colors)
            
            # Pad synth view to match height
            if synth_frame.shape[0] < thumb_column.shape[0]:
                padding = thumb_column.shape[0] - synth_frame.shape[0]
                synth_frame = cv2.copyMakeBorder(synth_frame, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif thumb_column.shape[0] < synth_frame.shape[0]:
                padding = synth_frame.shape[0] - thumb_column.shape[0]
                thumb_column = cv2.copyMakeBorder(thumb_column, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            combined = cv2.hconcat([thumb_column, synth_frame])
        
        elif view_mode == ViewMode.CALIBRATION:
            # Similar to synthesized but with calibration overlay
            thumb_frames = []
            for cam_idx, (cfg, dcfg) in enumerate(zip(camera_configs, display_configs)):
                frame = raw_frames.get(cfg['name'])
                if frame is not None:
                    # Detect markers for selected camera
                    if cam_idx == calib_mode.active_camera:
                        corners, ids = calib_mode.detect_markers(frame, cfg['name'])
                        thumb = processor.resize(frame, (dcfg['width'], dcfg['height']))
                        thumb = calib_mode.render_overlay(thumb, cfg['name'], 
                            [c * dcfg['scale'] for c in corners] if corners else None, ids)
                        # Highlight active camera
                        cv2.rectangle(thumb, (0, 0), (thumb.shape[1]-1, thumb.shape[0]-1), (0, 255, 255), 3)
                    else:
                        thumb = processor.resize(frame, (dcfg['width'], dcfg['height']))
                else:
                    thumb = np.zeros((dcfg['target_height'], dcfg['width'], 3), dtype=np.uint8)
                thumb_frames.append(thumb)
            
            thumb_column = cv2.vconcat(thumb_frames) if thumb_frames else np.zeros((SYNTH_VIEW_HEIGHT, 320, 3), dtype=np.uint8)
            
            # Render instruction panel
            instr_panel = calib_mode.render_instructions(SYNTH_VIEW_WIDTH, thumb_column.shape[0])
            
            # Pad if needed
            if instr_panel.shape[0] < thumb_column.shape[0]:
                padding = thumb_column.shape[0] - instr_panel.shape[0]
                instr_panel = cv2.copyMakeBorder(instr_panel, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            elif thumb_column.shape[0] < instr_panel.shape[0]:
                padding = instr_panel.shape[0] - thumb_column.shape[0]
                thumb_column = cv2.copyMakeBorder(thumb_column, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            combined = cv2.hconcat([thumb_column, instr_panel])
        
        # Add global status bar at bottom (only if overlay is enabled)
        if SHOW_GUI_OVERLAY:
            mode_name = view_mode.value.replace('_', ' ').title()
            status_bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
            status_text = f"Mode: {mode_name} | People: {total_people} | Cameras: {num_cameras}"
            if view_mode == ViewMode.INDIVIDUAL:
                status_text += f" | Showing: {camera_configs[individual_cam_idx]['name']}"
            status_text += " | Keys: 1-9/S/G/T/C/Q"
            cv2.putText(status_bar, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            combined = cv2.vconcat([combined, status_bar])
        
        window_title = "Multi-Camera Person Tracker"
        
        # Draw calibration result overlay (success/fail banner) if active
        if calib_mode is not None:
            combined = calib_mode.render_calibration_result_overlay(combined)
        
        cv2.imshow(window_title, combined)
        
        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        # Calibration mode specific keys (check FIRST before generic view switching)
        elif key == ord('a') and view_mode == ViewMode.CALIBRATION:
            # AUTO-CALIBRATE: detect all cameras, compute poses, save
            print("\n🎯 Starting auto-calibration...")
            success, msg = calib_mode.run_auto_calibration(raw_frames, camera_configs, calibration, cameras)
            if success:
                # Switch to synthesized view to show results
                view_mode = ViewMode.SYNTHESIZED
                display_configs = calculate_display_configs(view_mode)
        elif key == ord(' ') and view_mode == ViewMode.CALIBRATION:
            # Capture markers for active camera (manual mode)
            cam_name = camera_configs[calib_mode.active_camera]['name']
            frame = raw_frames.get(cam_name)
            if frame is not None:
                calib_mode.detect_markers(frame, cam_name)
                print(f"📍 Captured markers for {cam_name}")
        elif key == 13 and view_mode == ViewMode.CALIBRATION:  # Enter - compute 3D pose
            cam_name = camera_configs[calib_mode.active_camera]['name']
            success, msg = calib_mode.compute_3d_pose(cam_name)
            print(f"{'✅' if success else '❌'} {msg}")
            if success:
                print(f"   Calibrated cameras: {list(calibration.calibrations.keys())}")
        elif key == ord('s') and view_mode == ViewMode.CALIBRATION:
            # Save calibration (in calibration mode only)
            if len(calibration.calibrations) > 0:
                calibration.save()
            else:
                print("❌ No calibrations to save! Press ENTER after detecting markers to compute pose first.")
        elif key == 27:  # ESC - exit calibration
            if view_mode == ViewMode.CALIBRATION:
                view_mode = ViewMode.SIDE_BY_SIDE
                display_configs = calculate_display_configs(view_mode)
        # Generic view mode switching (only when NOT in calibration mode)
        elif key == ord('s') and view_mode != ViewMode.CALIBRATION:
            view_mode = ViewMode.SIDE_BY_SIDE
            display_configs = calculate_display_configs(view_mode)
        elif key == ord('g'):
            view_mode = ViewMode.GRID
            display_configs = calculate_display_configs(view_mode)
        elif key == ord('t'):
            view_mode = ViewMode.SYNTHESIZED
            display_configs = calculate_display_configs(view_mode)
        elif key == ord('c'):
            view_mode = ViewMode.CALIBRATION
            display_configs = calculate_display_configs(view_mode)
        elif ord('1') <= key <= ord('9'):
            cam_num = key - ord('1')  # 0-indexed
            if cam_num < num_cameras:
                if view_mode == ViewMode.CALIBRATION:
                    calib_mode.active_camera = cam_num
                else:
                    individual_cam_idx = cam_num
                    view_mode = ViewMode.INDIVIDUAL
                    display_configs = calculate_display_configs(view_mode, individual_cam_idx)
    
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
    print("\\n📊 Camera Statistics:")
    for i, cap in enumerate(cameras):
        stats = cap.get_stats()
        print(f"   {stats['name']}: {stats['frames_received']} received, {stats['frames_dropped']} dropped, {stats['reconnect_count']} reconnects")


def render_camera_frame(frame, cfg, dcfg, processor, cam_idx, track_colors, compact=False, is_cached=False):
    """Render a single camera frame with detections and overlays"""
    import random
    
    display_frame = processor.resize(frame, (dcfg['width'], dcfg['height']))
    
    # Pad to target height if needed
    if display_frame.shape[0] < dcfg['target_height']:
        padding = dcfg['target_height'] - display_frame.shape[0]
        display_frame = cv2.copyMakeBorder(
            display_frame, 0, padding, 0, 0, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    
    # If GUI overlay is disabled, return clean frame
    if not SHOW_GUI_OVERLAY:
        return display_frame
    
    # If showing cached frame, add indicator
    if is_cached:
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), (0, 100, 255), 3)
    
    # Draw detections
    for detection in cfg['last_boxes']:
        # Handle both old format (6 elements) and new format (7 elements with class_id)
        if len(detection) == 7:
            x1, y1, x2, y2, conf, track_id, class_id = detection
        else:
            x1, y1, x2, y2, conf, track_id = detection
            class_id = PERSON_CLASS_ID  # Default to person for backward compatibility
        
        dx1 = int(x1 * dcfg['scale'])
        dy1 = int(y1 * dcfg['scale'])
        dx2 = int(x2 * dcfg['scale'])
        dy2 = int(y2 * dcfg['scale'])
        
        # Determine label prefix and base color by class
        if class_id == CYCLIST_CLASS_ID:
            label_prefix = "Cy"  # Cyclist
            base_color = (255, 150, 0)  # Orange for cyclists
        elif class_id == BICYCLE_CLASS_ID:
            label_prefix = "B"  # Standalone bicycle (parked, etc.)
            base_color = (200, 200, 0)  # Yellow for bikes
        elif class_id == CAR_CLASS_ID:
            label_prefix = "Ca"  # Car
            base_color = (255, 100, 50)  # Blue-ish for cars
        else:
            label_prefix = "P"  # Pedestrian (person not on bike)
            base_color = (50, 255, 50)  # Green-ish for people
        
        # Unique color per track ID (offset by camera to avoid collisions)
        color_key = f"{cfg['name']}_{class_id}_{track_id}"
        if color_key not in track_colors:
            random.seed(hash(color_key))
            # Blend base color with random variation
            track_colors[color_key] = (
                min(255, base_color[0] + random.randint(-30, 30)),
                min(255, base_color[1] + random.randint(-30, 30)),
                min(255, base_color[2] + random.randint(-30, 30))
            )
        
        color = track_colors.get(color_key, base_color)
        thickness = 1 if compact else 2
        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, thickness)
        
        if not compact:
            label = f"{label_prefix}{track_id} ({conf:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(display_frame, (dx1, dy1 - 16), (dx1 + label_size[0], dy1), color, -1)
            cv2.putText(display_frame, label, (dx1, dy1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw info overlay
    # Count by category
    pedestrian_count = sum(1 for d in cfg['last_boxes'] if (len(d) < 7 or d[6] == PERSON_CLASS_ID))
    cyclist_count = sum(1 for d in cfg['last_boxes'] if len(d) == 7 and d[6] == CYCLIST_CLASS_ID)
    car_count = sum(1 for d in cfg['last_boxes'] if len(d) == 7 and d[6] == CAR_CLASS_ID)
    bike_count = sum(1 for d in cfg['last_boxes'] if len(d) == 7 and d[6] == BICYCLE_CLASS_ID)  # Standalone bikes
    fps = cfg.get('current_fps', 0)
    latency_ms = cfg.get('current_latency', 0)
    
    if compact:
        # Minimal overlay for thumbnails
        status = "⚠" if is_cached else ""
        count_text = f"{pedestrian_count}p"
        if cyclist_count > 0:
            count_text += f" {cyclist_count}cy"
        if car_count > 0:
            count_text += f" {car_count}ca"
        cv2.putText(display_frame, f"{cfg['name']}: {count_text} {status}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    else:
        info_bg = display_frame.copy()
        cv2.rectangle(info_bg, (5, 5), (180, 126), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 1, info_bg, 0.5, 0)
        
        latency_color = (0, 255, 0) if latency_ms < 50 else (0, 255, 255) if latency_ms < 100 else (0, 100, 255)
        if is_cached or latency_ms < 0:
            latency_color = (0, 100, 255)
            latency_text = "Cached"
        else:
            latency_text = f"{latency_ms:.0f}ms"
        
        cv2.putText(display_frame, cfg['name'], (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Pedestrians: {pedestrian_count}", (10, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Cyclists: {cyclist_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
        cv2.putText(display_frame, f"Cars: {car_count}", (10, 78),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 50), 1)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 96),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Latency: {latency_text}", (10, 114),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, latency_color, 1)
    
    return display_frame


if __name__ == "__main__":
    main()
