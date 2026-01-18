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
        'enabled': True,
    },
]

# Camera network info (for reference):
# - RTSP: port 555
# - HTTPS: port 443
# - ONVIF: port 8000

# Calibration file path (created by calibration mode)
CALIBRATION_FILE = 'camera_calibration.json'

# Synthesized view settings
SYNTH_VIEW_WIDTH = 800   # Width of bird's eye view in pixels
SYNTH_VIEW_HEIGHT = 600  # Height of bird's eye view
SYNTH_METERS_PER_PIXEL = 0.05  # 5cm per pixel = 40m x 30m coverage

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
            'meters_per_pixel': SYNTH_METERS_PER_PIXEL,
            'calibration_type': '3d_pose',
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
    
    def image_to_floor(self, camera_name, img_x, img_y, floor_z=0.0):
        """
        Project image point to floor plane (z=floor_z) using ray intersection.
        Returns (world_x, world_y) in meters, or None if invalid.
        """
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
        
        # Intersect ray with floor plane z = floor_z
        # Ray: P = camera_pos + t * ray_world
        # Plane: z = floor_z
        # Solve: camera_pos[2] + t * ray_world[2] = floor_z
        
        if abs(ray_world[2]) < 1e-6:
            return None  # Ray parallel to floor
        
        t = (floor_z - camera_pos[2]) / ray_world[2]
        
        if t < 0:
            return None  # Intersection behind camera
        
        # World intersection point
        world_pt = camera_pos + t * ray_world
        
        return float(world_pt[0]), float(world_pt[1])
    
    def transform_bbox_center(self, camera_name, x1, y1, x2, y2, person_height_estimate=0.0):
        """
        Transform bounding box to floor position.
        Uses bottom center of bbox (feet) projected to floor plane.
        """
        foot_x = (x1 + x2) / 2
        foot_y = y2  # Bottom of bounding box
        return self.image_to_floor(camera_name, foot_x, foot_y, floor_z=person_height_estimate)
    
    def world_to_synth_pixels(self, world_x, world_y):
        """Convert world coordinates (meters) to synthesized view pixels"""
        px = int(world_x / SYNTH_METERS_PER_PIXEL)
        py = int(world_y / SYNTH_METERS_PER_PIXEL)
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
    
    def load_or_create_background(self):
        """Load floor plan image or create grid"""
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
        
        # Draw grid lines (1 meter spacing at default scale)
        grid_spacing = int(1.0 / SYNTH_METERS_PER_PIXEL)
        for x in range(0, self.width, grid_spacing):
            cv2.line(self.background, (x, 0), (x, self.height), (60, 60, 60), 1)
            # Label every 5 meters
            meters = x * SYNTH_METERS_PER_PIXEL
            if meters % 5 == 0 and meters > 0:
                cv2.putText(self.background, f"{int(meters)}m", (x + 2, 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(self.background, (0, y), (self.width, y), (60, 60, 60), 1)
            meters = y * SYNTH_METERS_PER_PIXEL
            if meters % 5 == 0 and meters > 0:
                cv2.putText(self.background, f"{int(meters)}m", (2, y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Mark origin
        cv2.circle(self.background, (0, 0), 5, (100, 100, 100), -1)
        cv2.putText(self.background, "Origin", (8, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
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
        
        # Draw camera positions
        for cam_name in self.calibration.calibrations:
            cam_pos = self.calibration.get_camera_position(cam_name)
            if cam_pos is not None:
                px, py = self.calibration.world_to_synth_pixels(cam_pos[0], cam_pos[1])
                if 0 <= px < self.width and 0 <= py < self.height:
                    cv2.drawMarker(frame, (px, py), (0, 200, 255), cv2.MARKER_DIAMOND, 15, 2)
                    cv2.putText(frame, cam_name, (px + 10, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        
        # Draw all detected people
        for camera_name, boxes in all_detections:
            for (x1, y1, x2, y2, conf, track_id) in boxes:
                world_pos = self.calibration.transform_bbox_center(camera_name, x1, y1, x2, y2)
                if world_pos is None:
                    continue
                
                world_x, world_y = world_pos
                px, py = self.calibration.world_to_synth_pixels(world_x, world_y)
                
                # Clamp to view bounds
                px = max(0, min(self.width - 1, px))
                py = max(0, min(self.height - 1, py))
                
                # Get color for this track
                color_key = f"{camera_name}_{track_id}"
                if color_key not in track_colors:
                    import random
                    random.seed(hash(color_key))
                    track_colors[color_key] = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)
                    )
                color = track_colors[color_key]
                
                # Draw person as circle with ID
                cv2.circle(frame, (px, py), 12, color, -1)
                cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)
                cv2.putText(frame, str(track_id), (px - 6, py + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add info overlay
        cv2.putText(frame, "SYNTHESIZED VIEW (3D Calibrated)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show scale
        scale_text = f"Scale: {SYNTH_METERS_PER_PIXEL*100:.0f}cm/px"
        cv2.putText(frame, scale_text, (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame


class CalibrationMode:
    """
    Interactive 3D calibration using ArUco markers.
    Place markers at known 3D positions, detect them, compute camera pose via solvePnP.
    """
    def __init__(self, calibration_manager, synth_width, synth_height):
        self.calibration = calibration_manager
        self.synth_width = synth_width
        self.synth_height = synth_height
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Physical marker size in meters (outer edge length)
        self.marker_size_meters = 0.15  # 15cm markers
        
        # Marker 3D world positions in METERS (x, y, z)
        # Z = height above floor (0 = on floor)
        # Configure these based on your physical setup!
        self.marker_world_positions_3d = {
            0: (2.0, 2.0, 0.0),      # Floor marker, 2m from origin in x and y
            1: (8.0, 2.0, 0.0),      # Floor marker
            2: (2.0, 8.0, 0.0),      # Floor marker
            3: (8.0, 8.0, 0.0),      # Floor marker
            4: (5.0, 5.0, 0.0),      # Floor marker, center
            5: (2.0, 5.0, 1.5),      # Wall marker at 1.5m height
            6: (8.0, 5.0, 1.5),      # Wall marker at 1.5m height
            7: (5.0, 2.0, 1.0),      # Marker on stand at 1m height
            8: (5.0, 8.0, 1.0),      # Marker on stand at 1m height
            # Add more markers for better coverage
        }
        
        # Camera intrinsics storage (per camera)
        self.camera_intrinsics = {}  # camera_name -> {matrix, dist_coeffs, size}
        
        # Detected markers per camera
        self.detected_markers = {}  # camera_name -> {marker_id: corner_points}
        self.active_camera = 0
        self.instructions = [
            "3D CALIBRATION MODE",
            "-------------------",
            "1. Print ArUco markers and place at known 3D positions",
            "2. Measure marker positions in meters (x, y, z)",
            "3. Update marker_world_positions_3d in code",
            "4. Press 1-9 to select camera",
            "5. Press SPACE to detect markers",
            "6. Press ENTER to compute 3D pose (need 4+ markers)",
            "7. Press S to save calibration",
            "8. Press ESC to exit",
            "",
            f"Marker size: {self.marker_size_meters*100:.0f}cm",
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
        """Compute camera 3D pose using solvePnP"""
        if camera_name not in self.detected_markers:
            return False, "No markers detected for this camera"
        
        if camera_name not in self.camera_intrinsics:
            return False, "Camera intrinsics not available"
        
        markers = self.detected_markers[camera_name]
        if len(markers) < 4:
            return False, f"Need 4+ markers, have {len(markers)}"
        
        intrinsics = self.camera_intrinsics[camera_name]
        camera_matrix = intrinsics['matrix']
        dist_coeffs = intrinsics['dist_coeffs']
        image_size = intrinsics['size']
        
        # Build 3D-2D point correspondences using marker corners
        object_points = []  # 3D world points
        image_points = []   # 2D image points
        
        half_size = self.marker_size_meters / 2
        
        for marker_id, corners in markers.items():
            # Get marker center in world coordinates
            world_pos = self.marker_world_positions_3d[marker_id]
            wx, wy, wz = world_pos
            
            # Define 4 corners of marker in world coordinates
            # Corners are: top-left, top-right, bottom-right, bottom-left
            # Assuming marker lies in XY plane at height wz, facing up (or facing camera)
            marker_corners_3d = np.array([
                [wx - half_size, wy - half_size, wz],  # Top-left
                [wx + half_size, wy - half_size, wz],  # Top-right
                [wx + half_size, wy + half_size, wz],  # Bottom-right
                [wx - half_size, wy + half_size, wz],  # Bottom-left
            ], dtype=np.float64)
            
            # corners is 4x2 array of image points
            for j in range(4):
                object_points.append(marker_corners_3d[j])
                image_points.append(corners[j])
        
        object_points = np.array(object_points, dtype=np.float64)
        image_points = np.array(image_points, dtype=np.float64)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return False, "solvePnP failed"
        
        # Refine with iterative optimization
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points,
            camera_matrix, dist_coeffs,
            rvec, tvec
        )
        
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
                     f"camera at ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f})m")
    
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
        status_color = (0, 255, 0) if count >= 4 else (0, 255, 255)
        cv2.putText(frame, f"Markers: {count}/4+", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show calibration status
        if camera_name in self.calibration.calibrations:
            cv2.putText(frame, "3D CALIBRATED", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show camera position
            cam_pos = self.calibration.get_camera_position(camera_name)
            if cam_pos is not None:
                pos_text = f"Cam pos: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})m"
                cv2.putText(frame, pos_text, (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def render_instructions(self, width, height):
        """Render instruction panel"""
        panel = np.zeros((height, 400, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        y = 30
        for line in self.instructions:
            cv2.putText(panel, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += 20
        
        # Show marker positions
        y += 10
        cv2.putText(panel, "Marker 3D Positions (meters):", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        y += 20
        
        for marker_id, pos in sorted(self.marker_world_positions_3d.items()):
            cv2.putText(panel, f"  ID {marker_id}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
            y += 16
            if y > height - 80:
                break
        
        # Show detected markers summary
        y += 10
        cv2.putText(panel, "Detected:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        y += 20
        
        for cam_name, markers in self.detected_markers.items():
            marker_ids = sorted(markers.keys())
            cv2.putText(panel, f"  {cam_name}: {marker_ids}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
            y += 16
        
        return panel
    
    @staticmethod
    def generate_markers(output_dir='aruco_markers', num_markers=9):
        """Generate printable ArUco marker images with 3D position labels"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Default 3D positions (should match marker_world_positions_3d)
        default_positions = {
            0: (2.0, 2.0, 0.0),
            1: (8.0, 2.0, 0.0),
            2: (2.0, 8.0, 0.0),
            3: (8.0, 8.0, 0.0),
            4: (5.0, 5.0, 0.0),
            5: (2.0, 5.0, 1.5),
            6: (8.0, 5.0, 1.5),
            7: (5.0, 2.0, 1.0),
            8: (5.0, 8.0, 1.0),
        }
        
        for marker_id in range(num_markers):
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
            # Add white border for printing
            bordered = cv2.copyMakeBorder(marker_img, 50, 80, 50, 50,
                                         cv2.BORDER_CONSTANT, value=255)
            # Add ID label
            cv2.putText(bordered, f"ID: {marker_id}", (180, 480),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            # Add suggested position
            if marker_id in default_positions:
                pos = default_positions[marker_id]
                cv2.putText(bordered, f"Default: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})m", 
                           (80, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            cv2.imwrite(f"{output_dir}/marker_{marker_id}.png", bordered)
        
        print(f"✅ Generated {num_markers} markers in {output_dir}/")


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
    
    print(f"\n📹 Connecting to {len(CAMERAS)} cameras...")
    
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
    print("   Keys: 1-9=Camera, S=Side-by-side, G=Grid, T=Synthesized, C=Calibrate, Q=Quit\n")
    
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
            
            all_detections.append((cfg['name'], cfg['last_boxes'])
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
        
        # Add global status bar at bottom
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
        elif key == ord('s'):
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
        elif key == 27:  # ESC - exit calibration
            if view_mode == ViewMode.CALIBRATION:
                view_mode = ViewMode.SIDE_BY_SIDE
                display_configs = calculate_display_configs(view_mode)
        elif key == ord(' ') and view_mode == ViewMode.CALIBRATION:
            # Capture markers for active camera
            cam_name = camera_configs[calib_mode.active_camera]['name']
            frame = raw_frames.get(cam_name)
            if frame is not None:
                calib_mode.detect_markers(frame, cam_name)
                print(f"📍 Captured markers for {cam_name}")
        elif key == 13 and view_mode == ViewMode.CALIBRATION:  # Enter - compute 3D pose
            cam_name = camera_configs[calib_mode.active_camera]['name']
            success, msg = calib_mode.compute_3d_pose(cam_name)
            print(f"{'✅' if success else '❌'} {msg}")
        elif key == ord('s') and view_mode == ViewMode.CALIBRATION:
            calibration.save()
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
