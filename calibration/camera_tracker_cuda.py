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
#   - Camera ledge: 16 cm below floor → Y = -16 (50 cm above street)
#   - Street level: 66 cm below floor → Y = -66 (where people walk)
#
STREET_LEVEL_Y = -66.0     # Where people walk (66 cm below floor)
CAMERA_LEDGE_Y = -16.0     # Where cameras are mounted (50 cm above street)
CAMERA_HEIGHT_ABOVE_STREET = 50.0  # Physical camera mount height

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
        # Offset so panels (X=0-240, Z=0) appear at a reasonable position
        # Add offset to center the view
        offset_x = 150  # pixels from left edge for X=0
        offset_y = 50   # pixels from top edge for Z=0
        
        px = int(offset_x + world_x / SYNTH_CM_PER_PIXEL)
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
        
        # Offset for coordinate system (panels at X=0-240, Z=0)
        offset_x = 150  # pixels from left for X=0
        offset_y = 50   # pixels from top for Z=0
        
        # Draw grid lines every 100cm (1 meter)
        grid_spacing_cm = 100
        grid_spacing_px = int(grid_spacing_cm / SYNTH_CM_PER_PIXEL)
        
        # Vertical lines (X axis)
        for x_cm in range(-100, 400, grid_spacing_cm):
            x_px = int(offset_x + x_cm / SYNTH_CM_PER_PIXEL)
            if 0 <= x_px < self.width:
                cv2.line(self.background, (x_px, 0), (x_px, self.height), (60, 60, 60), 1)
                if x_cm % 100 == 0:
                    cv2.putText(self.background, f"X={x_cm}", (x_px + 2, 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Horizontal lines (Z axis - depth)
        for z_cm in range(0, 400, grid_spacing_cm):
            y_px = int(offset_y + z_cm / SYNTH_CM_PER_PIXEL)
            if 0 <= y_px < self.height:
                cv2.line(self.background, (0, y_px), (self.width, y_px), (60, 60, 60), 1)
                cv2.putText(self.background, f"Z={z_cm}", (5, y_px - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Draw panel positions (X=0 to 240, Z=0)
        panel_y = int(offset_y)
        for unit in range(4):
            unit_x = unit * 80  # 80cm spacing
            px = int(offset_x + unit_x / SYNTH_CM_PER_PIXEL)
            cv2.rectangle(self.background, (px - 25, panel_y - 5), (px + 25, panel_y + 5), 
                         (100, 100, 150), -1)
        cv2.putText(self.background, "PANELS", (offset_x + 60, panel_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # Mark origin
        origin_px = (offset_x, offset_y)
        cv2.circle(self.background, origin_px, 5, (100, 100, 100), -1)
        cv2.putText(self.background, "(0,0)", (offset_x + 8, offset_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
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
    
    LIMITED VISIBILITY CALIBRATION:
    Designed for setups where each camera can only see 3 markers.
    Marker 1 is the SHARED marker visible from all camera views.
    This provides a common reference frame for multi-camera fusion.
    
    Setup (5 markers on floor, all at Y=0):
    - Front row (Z=90cm): markers 0, 1, 2
    - Back row (Z=141cm): markers 3, 4
    - Camera 1 sees: Marker 0, 1, 3 (left + shared)
    - Camera 2 sees: Marker 1, 2, 4 (right + shared)
    - Each camera needs 3 markers minimum (3 markers × 4 corners = 12 points for solvePnP)
    """
    # Minimum markers required per camera (3 markers × 4 corners = 12 points)
    MIN_MARKERS_REQUIRED = 3
    
    def __init__(self, calibration_manager, synth_width, synth_height):
        self.calibration = calibration_manager
        self.synth_width = synth_width
        self.synth_height = synth_height
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Physical marker size in CENTIMETERS (outer edge length)
        self.marker_size_cm = 15.0  # 15cm markers
        
        # Marker 3D world positions in CENTIMETERS (x, y, z)
        # Coordinates matching pointLightController3D system
        # X = left-right along panel array (panels at X=0 to X=240)
        # Y = height (storefront floor = 0, street level = STREET_LEVEL_Y)
        # Z = depth from panels (Z=90 front row, Z=141 back row)
        #
        # IMPORTANT: Street level is 66 cm below storefront floor
        # Markers are placed ON THE STREET, so Y = STREET_LEVEL_Y
        #
        # LIMITED VISIBILITY SETUP:
        # - Marker 1 is the SHARED reference marker (must be visible from all cameras)
        # - Place markers on tile lines for easy alignment
        # - Each camera sees marker 1 plus 2 additional markers in its zone
        
        self.marker_world_positions_3d = {
            # FRONT ROW (Z = 90 cm - on tile line, Y = street level)
            0: (-40.0, STREET_LEVEL_Y, 90.0),     # Left front - left tile seam
            1: (120.0, STREET_LEVEL_Y, 90.0),     # Center front - SHARED REFERENCE
            2: (280.0, STREET_LEVEL_Y, 90.0),     # Right front - right tile seam
            
            # BACK ROW (Z = 141 cm - one tile further out, Y = street level)
            3: (-40.0, STREET_LEVEL_Y, 141.0),    # Left back
            4: (280.0, STREET_LEVEL_Y, 141.0),    # Right back
            6: (120.0, STREET_LEVEL_Y, 141.0),    # Center back - SHARED (completes grid)
            
            # VERTICAL MARKER on subway entrance wall (facing toward storefront)
            5: (120.0, CAMERA_LEDGE_Y, 550.0),    # Center, at camera height, on subway wall
        }
        
        # Markers that are VERTICAL (facing outward) instead of flat on ground
        # These need different corner calculations
        self.vertical_markers = {5}  # Marker 5 is on subway entrance wall
        
        # Define which markers each camera should see
        # This helps validate calibration and guide marker placement
        self.camera_marker_visibility = {
            'Camera 1': [1, 0, 3, 5, 6],   # Shared markers 1, 6 + left markers 0, 3 + far 5
            'Camera 2': [1, 2, 4, 5, 6],   # Shared markers 1, 6 + right markers 2, 4 + far 5
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
        
        self.instructions = [
            "3D CALIBRATION - 6 MARKER LAYOUT",
            "----------------------------------------",
            "Floor markers (flat): 0-4 on street level",
            "Vertical marker: 5 on subway wall (at camera height)",
            "",
            "Front row (Z=90):   0--1--2  (near panels)",
            "Back row (Z=141):   3-----4  (one tile back)",
            "Subway wall (Z=550): ---5--- (vertical, on wall)",
            "",
            "Camera 1: markers 0, 1, 3, 5",
            "Camera 2: markers 1, 2, 4, 5",
            "",
            ">>> Press A for AUTO-CALIBRATE <<<",
            "(Detects all cameras and saves)",
            "",
            "Manual: SPACE=detect, ENTER=compute, S=save",
            "Press ESC to exit",
            "",
            f"Marker size: {self.marker_size_cm:.0f} cm",
        ]
    
    def estimate_camera_intrinsics(self, image_size):
        """
        Estimate camera intrinsics based on image size.
        Uses reasonable defaults for typical surveillance cameras.
        For better accuracy, perform proper camera calibration.
        """
        width, height = image_size
        
        # Estimate focal length (assume ~60-70 degree horizontal FOV)
        # focal_length = width / (2 * tan(fov/2))
        # For 65 degree FOV: focal_length ≈ width * 0.87
        focal_length = width * 0.85
        
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
        
        return camera_matrix, dist_coeffs
    
    def detect_markers(self, frame, camera_name):
        """Detect ArUco markers in frame and store corner points"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Store image size for intrinsics estimation
        h, w = frame.shape[:2]
        if camera_name not in self.camera_intrinsics:
            matrix, dist = self.estimate_camera_intrinsics((w, h))
            self.camera_intrinsics[camera_name] = {
                'matrix': matrix,
                'dist_coeffs': dist,
                'size': (w, h)
            }
        
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
        
        # Build 3D-2D point correspondences using marker CENTERS only
        # This avoids issues with marker rotation on floor when cameras are angled
        object_points = []  # 3D world points (in cm)
        image_points = []   # 2D image points (marker centers)
        
        for marker_id, corners in markers.items():
            # Get marker center in world coordinates (cm)
            world_pos = self.marker_world_positions_3d[marker_id]
            
            # Calculate image center from detected corners
            center_2d = corners.mean(axis=0)  # Average of 4 corners
            
            object_points.append(world_pos)
            image_points.append(center_2d)
        
        object_points = np.array(object_points, dtype=np.float64)
        image_points = np.array(image_points, dtype=np.float64)
        
        # Need at least 4 points for solvePnP
        if len(object_points) < 4:
            return False, f"Need at least 4 marker centers, have {len(object_points)}"
        
        # Provide initial guess based on expected camera position
        # Cameras are on the storefront at roughly:
        #   Camera 1: X~0, Y~-16 (ledge height), Z~0 (at storefront)
        #   Camera 2: X~240, Y~-16, Z~0
        # Looking toward positive Z (the street)
        if 'Camera 1' in camera_name or camera_name == 'Camera 1':
            # Camera 1 is on left side
            init_camera_pos = np.array([0.0, -16.0, 0.0])
            # Looking toward street center (positive Z, slight right)
            init_target = np.array([120.0, -66.0, 200.0])
        else:
            # Camera 2 is on right side
            init_camera_pos = np.array([240.0, -16.0, 0.0])
            # Looking toward street center (positive Z, slight left)
            init_target = np.array([120.0, -66.0, 200.0])
        
        # Calculate initial rotation from camera position looking at target
        forward = init_target - init_camera_pos
        forward = forward / np.linalg.norm(forward)
        
        # Camera Z-axis points toward target
        # Camera Y-axis points down (OpenCV convention)
        # Camera X-axis points right
        up_world = np.array([0.0, -1.0, 0.0])  # World up is -Y in our system
        right = np.cross(forward, up_world)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        
        # Rotation matrix: columns are camera axes in world coords
        R_init = np.column_stack([right, down, forward])
        init_rvec, _ = cv2.Rodrigues(R_init)
        init_tvec = -R_init @ init_camera_pos
        
        # Use iterative solver with initial guess
        print(f"  Using ITERATIVE solver with initial guess")
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            camera_matrix, dist_coeffs,
            rvec=init_rvec.astype(np.float64),
            tvec=init_tvec.reshape(3,1).astype(np.float64),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            # Fallback to SQPNP without guess
            print(f"  Iterative failed, trying SQPNP")
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
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
        
        # Compute camera position and validate it makes physical sense
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec.flatten()
        
        # VALIDATION: Camera should be at positive Z (in front of storefront, looking at street)
        if camera_pos[2] < 0:
            print(f"  ⚠️ WARNING: Camera Z={camera_pos[2]:.1f} still negative after selection!")
            print(f"     The marker positions or corner ordering may need adjustment")
        
        # Compute reprojection error
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_error = np.sqrt(np.mean((image_points - projected.reshape(-1, 2))**2))
        
        # Store calibration
        self.calibration.set_calibration(
            camera_name, rvec, tvec, camera_matrix, dist_coeffs, image_size
        )
        
        # Compute camera position for display
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec.flatten()
        
        return True, (f"3D pose computed from {len(markers)} markers, "
                     f"reproj error: {reproj_error:.2f}px, "
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
    
    def run_auto_calibration(self, raw_frames, camera_configs, calibration_manager):
        """
        Run automatic calibration for all cameras.
        
        Args:
            raw_frames: dict of camera_name -> current frame
            camera_configs: list of camera config dicts with 'name' key
            calibration_manager: CalibrationManager instance to save to
            
        Returns:
            (success, message) tuple
        """
        self.auto_calibrating = True
        self.auto_calib_messages = []
        results = []
        
        print("\n" + "=" * 50)
        print("🎯 AUTO-CALIBRATION STARTED")
        print("=" * 50)
        
        # Step 1: Detect markers on all cameras
        self.auto_calib_messages.append("Step 1: Detecting markers on all cameras...")
        print("\n📍 Step 1: Detecting markers on all cameras...")
        
        for cam_idx, cfg in enumerate(camera_configs):
            cam_name = cfg['name']
            frame = raw_frames.get(cam_name)
            
            if frame is None:
                msg = f"  ⚠️ {cam_name}: No frame available"
                self.auto_calib_messages.append(msg)
                print(msg)
                continue
            
            # Clear previous detections for this camera
            self.detected_markers[cam_name] = {}
            
            # Detect markers
            corners, ids = self.detect_markers(frame, cam_name)
            
            if ids is not None:
                detected_ids = sorted(self.detected_markers.get(cam_name, {}).keys())
                msg = f"  ✓ {cam_name}: Found markers {detected_ids}"
                self.auto_calib_messages.append(msg)
                print(msg)
            else:
                msg = f"  ✗ {cam_name}: No markers detected"
                self.auto_calib_messages.append(msg)
                print(msg)
        
        # Step 2: Compute 3D pose for each camera
        self.auto_calib_messages.append("Step 2: Computing 3D poses...")
        print("\n🔧 Step 2: Computing 3D poses...")
        
        success_count = 0
        for cam_idx, cfg in enumerate(camera_configs):
            cam_name = cfg['name']
            
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
            
            final_msg = f"✅ AUTO-CALIBRATION COMPLETE: {success_count}/{len(camera_configs)} cameras calibrated"
            self.auto_calib_messages.append(final_msg)
            print(f"\n{final_msg}")
            print("=" * 50 + "\n")
            
            self.auto_calibrating = False
            return True, final_msg
        else:
            final_msg = "❌ AUTO-CALIBRATION FAILED: No cameras could be calibrated"
            self.auto_calib_messages.append(final_msg)
            print(f"\n{final_msg}")
            print("   Make sure markers are visible to all cameras")
            print("=" * 50 + "\n")
            
            self.auto_calibrating = False
            return False, final_msg
    
    @staticmethod
    def generate_markers(output_dir='aruco_markers', num_markers=5):
        """Generate printable ArUco marker images with 3D position labels.
        
        LIMITED VISIBILITY SETUP (coordinates in cm):
        - Marker 1 is the SHARED marker (visible from all cameras)
        - Markers 0, 2 are for Camera 1 only (left side)
        - Markers 3, 4 are for Camera 2 only (right side)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Positions in CENTIMETERS - aligned to tile lines
        default_positions = {
            # Front row (Z = 90 cm)
            0: (-40.0, 0.0, 90.0),     # Left front
            1: (120.0, 0.0, 90.0),     # Center front - SHARED
            2: (280.0, 0.0, 90.0),     # Right front
            # Back row (Z = 141 cm)
            3: (-40.0, 0.0, 141.0),    # Left back
            4: (280.0, 0.0, 141.0),    # Right back
        }
        
        # Camera assignments for labeling
        camera_assignments = {
            0: "Camera 1 (left front)",
            1: "SHARED (ALL CAMERAS)",
            2: "Camera 2 (right front)",
            3: "Camera 1 (left back)",
            4: "Camera 2 (right back)",
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
        print(f"   Front row (Z=90): Marker 0 @ X=-40, Marker 1 @ X=120 (SHARED), Marker 2 @ X=280")
        print(f"   Back row (Z=141): Marker 3 @ X=-40, Marker 4 @ X=280")
        print(f"   Camera 1 sees: 0, 1, 3 (left side)")
        print(f"   Camera 2 sees: 1, 2, 4 (right side)")


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
        cv2.imshow(window_title, combined)
        
        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        # Calibration mode specific keys (check FIRST before generic view switching)
        elif key == ord('a') and view_mode == ViewMode.CALIBRATION:
            # AUTO-CALIBRATE: detect all cameras, compute poses, save
            print("\n🎯 Starting auto-calibration...")
            success, msg = calib_mode.run_auto_calibration(raw_frames, camera_configs, calibration)
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
