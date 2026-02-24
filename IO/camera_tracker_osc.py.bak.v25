#!/usr/bin/env python3
"""
Camera Tracker V2.5 — Refactored for Speed & Simplicity

Tracks people using YOLO and sends floor positions via OSC.
Refactored from V2 with the following changes:
  - Removed zone sorting (handled by lightController)
  - Reduced from 9 tunable parameters to 3 live sliders + 2 config
  - Eliminated double world-coordinate transforms
  - Reduced frame copies in camera pipeline
  - Uses grab() for RTSP buffer flushing (no decode overhead)
  - Uses model.predict() by default (no double-tracking)
  - FrameProcessor indirection removed
  - Monolithic main() broken into Tracker class

OSC Messages Sent:
  /tracker/person/<id> <x> <z>  — Position in world cm
  /tracker/count <n>            — Number of tracked people

Usage:
    python camera_tracker_osc.py [--headless] [--process-width 416]

Press 'q' to quit, 's' to save settings
"""

import cv2
import numpy as np
import time
import json
import threading
import os
import signal
import sys
import logging
import argparse
import fcntl
import atexit
from collections import deque
from datetime import timedelta
from typing import Optional, List, Dict, Tuple

# OSC
from pythonosc import udp_client

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# FILE PATHS
# ==============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Config files live in IO/ directory (same as this script)
CALIBRATION_FILE = os.path.join(_SCRIPT_DIR, 'camera_calibration.json')
SETTINGS_FILE = os.path.join(_SCRIPT_DIR, 'tracker_settings.json')

# ==============================================================================
# SINGLE INSTANCE LOCK
# ==============================================================================

LOCK_FILE = "/tmp/camera_tracker_osc_v25.lock"
_lock_fd = None


def acquire_lock() -> bool:
    global _lock_fd
    try:
        _lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        return True
    except (IOError, OSError):
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = f.read().strip()
            print(f"Another tracker is already running (PID: {pid})")
        except Exception:
            print("Another tracker is already running")
        return False


def release_lock():
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
        except Exception:
            pass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class TrackerConfig:
    """
    All configuration in one place. Immutable after init.
    Live-tunable parameters are in TrackerSettings.
    """

    # OSC
    osc_ip: str = "127.0.0.1"
    osc_port: int = 7000

    # Cameras
    cameras: list = [
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
    ]

    # YOLO
    model_name: str = "yolo11n.pt"
    person_class_id: int = 0
    process_width: int = 416

    # Display
    headless: bool = False
    display_width: int = 480

    # Timing
    target_fps: int = 25

    # Camera reliability
    connection_timeout: float = 10.0
    reconnect_delay: float = 2.0
    max_frame_age: float = 0.5

    # Health
    health_log_interval: int = 300
    tracker_reset_interval: int = 3600

    # Calibration floor level
    floor_y: float = -66.0

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ==============================================================================
# LIVE-TUNABLE SETTINGS (3 sliders + 1 config-only)
# ==============================================================================

class TrackerSettings:
    """
    Minimal parameter set. Values stored as actual floats/ints (no scaling tricks).

    Live sliders (3):
        confidence    — YOLO detection threshold (0.10 – 0.80)
        fusion_dist   — Max distance (cm) to merge cross-camera detections (50 – 300)
        smoothing     — Position EMA alpha, higher = more responsive (0.01 – 0.20)

    Config-only (1):
        max_lost_frames — Frames before dropping a track (15 – 150)
    """

    DEFAULTS = {
        'confidence': 0.40,
        'fusion_dist': 150.0,
        'smoothing': 0.03,
        'max_lost_frames': 60,
    }

    # Slider ranges: key -> (min, max, scale_factor)
    # Sliders are integers; actual = slider_value / scale_factor
    SLIDER_DEFS = {
        'confidence':  {'min': 10, 'max': 95, 'scale': 100, 'label': 'Confidence'},
        'fusion_dist': {'min': 50, 'max': 500, 'scale': 1,  'label': 'Fusion Dist cm'},
        'smoothing':   {'min': 1,  'max': 50,  'scale': 100, 'label': 'Smoothing'},
    }

    def __init__(self, settings_file: str):
        self.settings_file = settings_file
        self.values = dict(self.DEFAULTS)
        self._dirty = False
        self._load()

    def _load(self):
        try:
            with open(self.settings_file, 'r') as f:
                saved = json.load(f)
            for key in self.DEFAULTS:
                if key in saved:
                    self.values[key] = saved[key]
            # Migrate from V2 format if needed
            if 'confidence_threshold' in saved and 'confidence' not in saved:
                self.values['confidence'] = saved['confidence_threshold'] / 100.0
            if 'fusion_threshold_cm' in saved and 'fusion_dist' not in saved:
                self.values['fusion_dist'] = float(saved['fusion_threshold_cm'])
            if 'position_smoothing' in saved and 'smoothing' not in saved:
                self.values['smoothing'] = saved['position_smoothing'] / 100.0
            if 'max_track_age_frames' in saved and 'max_lost_frames' not in saved:
                self.values['max_lost_frames'] = int(saved['max_track_age_frames'])
            logger.info(f"Loaded settings from {self.settings_file}")
        except FileNotFoundError:
            logger.info("No settings file found, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

    def save(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.values, f, indent=2)
            self._dirty = False
            logger.info(f"Saved settings to {self.settings_file}")
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")

    def get(self, key: str):
        return self.values.get(key, self.DEFAULTS.get(key))

    def set(self, key: str, value):
        if key in self.values:
            self.values[key] = value
            self._dirty = True

    # Convenience accessors
    @property
    def confidence(self) -> float:
        return float(self.values['confidence'])

    @property
    def fusion_dist(self) -> float:
        return float(self.values['fusion_dist'])

    @property
    def smoothing(self) -> float:
        return float(self.values['smoothing'])

    @property
    def max_lost_frames(self) -> int:
        return int(self.values['max_lost_frames'])

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def to_slider_value(self, key: str) -> int:
        """Convert internal float to integer slider value."""
        d = self.SLIDER_DEFS[key]
        return int(self.values[key] * d['scale'])

    def from_slider_value(self, key: str, slider_val: int):
        """Update internal value from integer slider."""
        d = self.SLIDER_DEFS[key]
        self.set(key, slider_val / d['scale'])


# ==============================================================================
# CALIBRATION — Projects image pixels to world floor coordinates
# ==============================================================================

class CalibrationManager:
    """
    Loads camera calibration and projects bounding-box feet to floor plane.
    Pre-computes R^T and K^-1 at load time to avoid per-call overhead.
    """

    def __init__(self, calibration_file: str, floor_y: float = -66.0):
        self.calibrations: Dict[str, dict] = {}
        self.is_calibrated = False
        self.floor_y = floor_y

        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            for name, cd in data.get('cameras', {}).items():
                K = np.array(cd['camera_matrix'])
                R, _ = cv2.Rodrigues(np.array(cd['rvec']))
                tvec = np.array(cd['tvec']).flatten()
                cam_pos = -R.T @ tvec
                K_inv = np.linalg.inv(K)

                self.calibrations[name] = {
                    'K': K,
                    'K_inv': K_inv,
                    'R_T': R.T,
                    'cam_pos': cam_pos,
                    'dist': np.array(cd['dist_coeffs']),
                }
            self.is_calibrated = len(self.calibrations) > 0
            if self.is_calibrated:
                logger.info(f"Loaded calibration for {len(self.calibrations)} cameras")
        except FileNotFoundError:
            logger.warning(f"Calibration file not found: {calibration_file}")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")

    def feet_to_floor(self, camera_name: str, foot_x: float, foot_y: float) -> Optional[Tuple[float, float]]:
        """
        Project a single image point (feet position) to the world floor plane.
        Returns (world_x, world_z) in cm, or None if projection fails.
        """
        cal = self.calibrations.get(camera_name)
        if cal is None:
            return None

        # Undistort
        pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, cal['K'], cal['dist'], P=cal['K'])
        ux, uy = und[0, 0]

        # Ray in world coordinates
        ray_cam = cal['K_inv'] @ np.array([ux, uy, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = cal['R_T'] @ ray_cam

        # Intersect floor plane y = floor_y
        if abs(ray_world[1]) < 1e-6:
            return None
        t = (self.floor_y - cal['cam_pos'][1]) / ray_world[1]
        if t < 0:
            return None

        hit = cal['cam_pos'] + t * ray_world
        return (float(hit[0]), float(hit[2]))

    def bbox_to_floor(self, camera_name: str, x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
        """Project bounding box bottom-center (feet) to floor."""
        return self.feet_to_floor(camera_name, (x1 + x2) / 2.0, y2)


# ==============================================================================
# ROBUST CAMERA — Threaded RTSP capture with minimal copies
# ==============================================================================

class RobustCamera:
    """
    Reliable RTSP camera with:
    - Single-copy frame buffer (not double-copy like V2)
    - grab()-based buffer flushing (no decode overhead)
    - Auto-reconnect on failure
    """

    def __init__(self, name: str, url: str, config: TrackerConfig):
        self.name = name
        self.url = url
        self.config = config

        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0
        self._frame_num: int = 0
        self._last_returned: int = -1
        self.width: int = 0
        self.height: int = 0
        self.connected: bool = False
        self._running: bool = False
        self._lock = threading.Lock()

        self.stats = {'received': 0, 'dropped': 0, 'reconnects': 0}

    def _connect(self) -> bool:
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        deadline = time.time() + self.config.connection_timeout
        while time.time() < deadline:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.width = frame.shape[1]
                self.height = frame.shape[0]
                self.connected = True
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.time()
                    self._frame_num += 1
                return True
            time.sleep(0.1)
        return False

    def start(self) -> bool:
        self._running = True
        ok = self._connect()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return ok

    def _capture_loop(self):
        fails = 0
        while self._running:
            if not self.connected:
                time.sleep(self.config.reconnect_delay)
                self.stats['reconnects'] += 1
                if self._connect():
                    fails = 0
                continue

            try:
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    fails = 0
                    with self._lock:
                        self._frame = frame  # Single reference — no extra copy
                        self._frame_time = time.time()
                        self._frame_num += 1
                        self.stats['received'] += 1

                    # Flush RTSP buffer using grab() — no decode cost
                    for _ in range(3):
                        if not self.cap.grab():
                            break
                        self.stats['dropped'] += 1
                else:
                    fails += 1
                    if fails > 30:
                        self.connected = False
                        fails = 0
                    time.sleep(0.01)
            except Exception as e:
                fails += 1
                if fails == 1 or fails % 10 == 0:
                    logger.warning(f"{self.name}: Capture error ({fails}x): {e}")
                if fails > 10:
                    self.connected = False
                    logger.error(f"{self.name}: Too many failures, reconnecting")
                time.sleep(0.01)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], bool]:
        """
        Get latest frame.
        Returns (ok, frame_copy, is_new_frame).
        Only copies when returning — single copy total.
        """
        with self._lock:
            if self._frame is None:
                return False, None, False

            age = time.time() - self._frame_time
            if age > self.config.max_frame_age:
                return False, None, False

            is_new = self._frame_num > self._last_returned
            if is_new:
                self._last_returned = self._frame_num
            return True, self._frame.copy(), is_new

    def release(self):
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


# ==============================================================================
# TRACKING FUSION — Cross-camera merge + temporal smoothing
# ==============================================================================

class TrackingFusion:
    """
    Merges detections from multiple cameras and maintains smooth tracks.
    No zone logic — outputs raw (x, z) positions for consumer to classify.

    Track matching uses 60% of fusion_dist to prevent cross-person jumps
    while allowing natural movement between frames.
    """

    def __init__(self, settings: TrackerSettings):
        self.settings = settings
        self.tracks: Dict[int, dict] = {}
        self._next_id = 1
        self._frame = 0

    def process(self, detections: List[dict]) -> List[Tuple[int, float, float]]:
        """
        Full pipeline: fuse cross-camera detections, then match to tracks.

        Args:
            detections: List of {'x': float, 'z': float, 'camera': str, 'conf': float}

        Returns:
            List of (track_id, world_x, world_z)
        """
        self._frame += 1
        fused = self._fuse(detections)
        return self._match_and_smooth(fused)

    def _fuse(self, detections: List[dict]) -> List[Tuple[float, float]]:
        """Merge detections from different cameras that are close together."""
        if not detections:
            return []

        n = len(detections)
        used = [False] * n
        threshold_sq = self.settings.fusion_dist ** 2
        result = []

        for i in range(n):
            if used[i]:
                continue
            used[i] = True

            sum_x = detections[i]['x']
            sum_z = detections[i]['z']
            count = 1
            cam_i = detections[i]['camera']

            for j in range(i + 1, n):
                if used[j] or detections[j]['camera'] == cam_i:
                    continue
                dx = detections[i]['x'] - detections[j]['x']
                dz = detections[i]['z'] - detections[j]['z']
                if dx * dx + dz * dz < threshold_sq:
                    sum_x += detections[j]['x']
                    sum_z += detections[j]['z']
                    count += 1
                    used[j] = True

            result.append((sum_x / count, sum_z / count))

        return result

    def _match_and_smooth(self, positions: List[Tuple[float, float]]) -> List[Tuple[int, float, float]]:
        """Match fused positions to existing tracks, create new ones, prune stale."""
        alpha = self.settings.smoothing
        match_dist_sq = (self.settings.fusion_dist * 0.6) ** 2
        matched = set()
        output = []

        for (raw_x, raw_z) in positions:
            best_id = None
            best_d2 = match_dist_sq

            for tid, t in self.tracks.items():
                if tid in matched:
                    continue
                # Predict using velocity
                px = t['x'] + t['vx']
                pz = t['z'] + t['vz']
                d2 = (raw_x - px) ** 2 + (raw_z - pz) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = tid

            if best_id is not None:
                matched.add(best_id)
                t = self.tracks[best_id]

                # EMA smoothing: blend predicted position with raw observation
                pred_x = t['x'] + t['vx']
                pred_z = t['z'] + t['vz']
                new_x = pred_x + alpha * (raw_x - pred_x)
                new_z = pred_z + alpha * (raw_z - pred_z)

                # Update velocity estimate (same alpha)
                t['vx'] += alpha * ((new_x - t['x']) - t['vx'])
                t['vz'] += alpha * ((new_z - t['z']) - t['vz'])
                t['x'] = new_x
                t['z'] = new_z
                t['last_seen'] = self._frame

                output.append((best_id, new_x, new_z))
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {
                    'x': raw_x, 'z': raw_z,
                    'vx': 0.0, 'vz': 0.0,
                    'last_seen': self._frame,
                }
                output.append((tid, raw_x, raw_z))

        # Prune stale tracks
        max_age = self.settings.max_lost_frames
        stale = [k for k, v in self.tracks.items()
                 if self._frame - v['last_seen'] > max_age]
        for k in stale:
            del self.tracks[k]

        return output


# ==============================================================================
# TRACKER — Main pipeline class
# ==============================================================================

class Tracker:
    """
    Encapsulates the full tracking pipeline:
    cameras -> YOLO detect -> calibrate to world -> fuse -> smooth -> OSC

    Responsibilities:
    - Camera I/O and YOLO inference
    - Projecting detections to world coordinates (single pass)
    - Cross-camera fusion and temporal smoothing
    - Sending positions via OSC

    NOT responsible for:
    - Zone classification (handled by lightController)
    - Light behavior decisions
    - Database recording
    """

    def __init__(self, config: TrackerConfig, settings: TrackerSettings):
        self.config = config
        self.settings = settings
        self.shutdown = False

        # Components (initialized in start())
        self.osc: Optional[udp_client.SimpleUDPClient] = None
        self.model = None
        self.device: str = "cpu"
        self.calibration: Optional[CalibrationManager] = None
        self.fusion: Optional[TrackingFusion] = None
        self.cameras: List[RobustCamera] = []
        self.cam_configs: List[dict] = []

        # Stats
        self.frame_count = 0
        self.start_time = 0.0
        self.osc_errors = 0
        self.total_tracked = 0
        self._yolo_times: deque = deque(maxlen=100)

    def start(self):
        """Initialize all components and enter main loop."""
        logger.info("=" * 50)
        logger.info("Camera Tracker V2.5")
        logger.info("=" * 50)

        self._init_osc()
        self._init_yolo()
        self._init_calibration()
        self.fusion = TrackingFusion(self.settings)
        self._init_cameras()

        if not self.cameras:
            logger.error("No cameras connected!")
            return

        self._setup_signals()
        if not self.config.headless:
            self._setup_gui()

        self.start_time = time.time()
        logger.info("Entering main loop")
        self._run()
        self._cleanup()

    def _init_osc(self):
        self.osc = udp_client.SimpleUDPClient(self.config.osc_ip, self.config.osc_port)
        logger.info(f"OSC output: {self.config.osc_ip}:{self.config.osc_port}")

    def _init_yolo(self):
        import torch
        from ultralytics import YOLO

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU")

        logger.info(f"Loading YOLO model: {self.config.model_name}")
        self.model = YOLO(self.config.model_name)
        self.model.to(self.device)

        # Warmup on GPU
        if self.device.startswith("cuda"):
            w = self.config.process_width
            dummy = np.zeros((w, w, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, classes=[self.config.person_class_id])
        logger.info("Model loaded")

    def _init_calibration(self):
        self.calibration = CalibrationManager(CALIBRATION_FILE, floor_y=self.config.floor_y)
        if not self.calibration.is_calibrated:
            logger.warning("No calibration loaded! Floor positions will be inaccurate.")

    def _init_cameras(self):
        enabled = [c for c in self.config.cameras if c.get('enabled', True)]
        logger.info(f"Connecting to {len(enabled)} cameras...")

        for cfg in enabled:
            cam = RobustCamera(cfg['name'], cfg['url'], self.config)
            if cam.start():
                logger.info(f"  {cfg['name']}: {cam.width}x{cam.height}")
                scale = self.config.process_width / cam.width
                self.cam_configs.append({
                    'name': cfg['name'],
                    'camera': cam,
                    'scale': scale,
                    'process_h': int(cam.height * scale),
                    'fps_hist': deque(maxlen=30),
                    'current_fps': 0.0,
                    '_last_boxes': [],  # cached for display reuse
                })
                self.cameras.append(cam)
            else:
                logger.error(f"  {cfg['name']}: FAILED to connect")

    def _setup_signals(self):
        def handler(signum, _frame):
            sig = signal.Signals(signum).name
            logger.info(f"Received {sig}, shutting down...")
            self.shutdown = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _setup_gui(self):
        cv2.namedWindow("Tracker V2.5", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Settings", 400, 200)

        for key, sdef in TrackerSettings.SLIDER_DEFS.items():
            cv2.createTrackbar(
                sdef['label'], "Settings",
                self.settings.to_slider_value(key),
                sdef['max'],
                lambda val, k=key: self.settings.from_slider_value(k, val)
            )

    # ---------- MAIN LOOP ----------

    def _run(self):
        frame_interval = 1.0 / self.config.target_fps
        last_process = 0.0
        last_health = time.time()
        last_save = time.time()
        last_reset = time.time()

        while not self.shutdown:
            now = time.time()
            if now - last_process < frame_interval:
                time.sleep(0.001)
                continue
            last_process = now
            self.frame_count += 1

            # 1) Detect + project to world (single pass, cached for display)
            world_dets, display_frames = self._detect_all()

            # 2) Fuse + track
            tracked = self.fusion.process(world_dets)

            # 3) Send OSC
            self._send_osc(tracked)

            # 4) Display (if GUI enabled)
            if not self.config.headless and display_frames:
                self._render(display_frames, tracked)

            # 5) Periodic maintenance
            if self.settings.is_dirty and now - last_save > 5.0:
                self.settings.save()
                last_save = now

            if now - last_health >= self.config.health_log_interval:
                self._log_health()
                last_health = now

            if now - last_reset >= self.config.tracker_reset_interval:
                self._reset_yolo()
                last_reset = now

            # 6) Handle keyboard input
            if not self.config.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown = True
                elif key == ord('s'):
                    self.settings.save()
                    print("Settings saved")

    def _detect_all(self) -> Tuple[List[dict], List[np.ndarray]]:
        """
        Run YOLO on all cameras, project to world, return detections and display frames.
        World coordinates are computed ONCE per detection and cached for display reuse.
        """
        world_dets = []
        display_frames = []
        pw = self.config.process_width
        person_cls = self.config.person_class_id

        for cfg in self.cam_configs:
            cam: RobustCamera = cfg['camera']
            ok, frame, is_new = cam.get_frame()
            if not ok or frame is None:
                continue

            if is_new:
                t0 = time.time()

                # Resize for YOLO
                small = cv2.resize(frame, (pw, cfg['process_h']), interpolation=cv2.INTER_LINEAR)

                # Detect only — no internal tracking (we do our own fusion)
                results = self.model.predict(
                    small,
                    verbose=False,
                    conf=self.settings.confidence,
                    classes=[person_cls],
                    imgsz=pw,
                    device=self.device,
                )

                dt = time.time() - t0
                self._yolo_times.append(dt)
                cfg['fps_hist'].append(1.0 / max(dt, 0.001))
                cfg['current_fps'] = sum(cfg['fps_hist']) / len(cfg['fps_hist'])

                # Extract boxes with world positions (computed ONCE)
                boxes_with_world = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    scale_inv = 1.0 / cfg['scale']

                    for i in range(len(boxes)):
                        bx = boxes.xyxy[i].cpu().numpy()
                        x1 = bx[0] * scale_inv
                        y1 = bx[1] * scale_inv
                        x2 = bx[2] * scale_inv
                        y2 = bx[3] * scale_inv
                        conf = float(boxes.conf[i])

                        # Project to world ONCE — result is cached in tuple
                        wp = self.calibration.bbox_to_floor(cfg['name'], x1, y1, x2, y2)
                        wx, wz = wp if wp else (None, None)

                        boxes_with_world.append((x1, y1, x2, y2, conf, wx, wz))

                        if wp is not None:
                            world_dets.append({
                                'x': wx, 'z': wz,
                                'camera': cfg['name'],
                                'conf': conf,
                            })

                cfg['_last_boxes'] = boxes_with_world

            # Build display frame reusing cached world positions
            if not self.config.headless:
                dw = self.config.display_width
                ds = dw / frame.shape[1]
                dframe = cv2.resize(frame, (dw, int(frame.shape[0] * ds)), interpolation=cv2.INTER_LINEAR)

                for (x1, y1, x2, y2, conf, wx, wz) in cfg.get('_last_boxes', []):
                    dx1, dy1 = int(x1 * ds), int(y1 * ds)
                    dx2, dy2 = int(x2 * ds), int(y2 * ds)

                    if wx is not None:
                        color = (0, 255, 0)
                        label = f"{conf:.2f} X:{wx:.0f} Z:{wz:.0f}"
                    else:
                        color = (128, 128, 128)
                        label = f"{conf:.2f} no calib"

                    cv2.rectangle(dframe, (dx1, dy1), (dx2, dy2), color, 2)
                    cv2.putText(dframe, label, (dx1, dy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                cv2.putText(dframe, f"{cfg['name']} FPS:{cfg['current_fps']:.1f}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                display_frames.append(dframe)

        return world_dets, display_frames

    def _send_osc(self, tracked: List[Tuple[int, float, float]]):
        """Send tracked positions via OSC. Format matches lightController expectations."""
        try:
            self.osc.send_message("/tracker/count", len(tracked))
            for tid, wx, wz in tracked:
                self.osc.send_message(f"/tracker/person/{tid}", [float(wx), float(wz)])
            self.total_tracked += len(tracked)
        except Exception as e:
            self.osc_errors += 1
            if self.osc_errors == 1 or self.osc_errors % 100 == 0:
                logger.warning(f"OSC error ({self.osc_errors}x): {e}")

    def _render(self, frames: List[np.ndarray], tracked: List[Tuple[int, float, float]]):
        """Compose camera views side-by-side with status bar."""
        if len(frames) == 1:
            combined = frames[0]
        else:
            max_h = max(f.shape[0] for f in frames)
            padded = []
            for f in frames:
                if f.shape[0] < max_h:
                    pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                    f = cv2.vconcat([f, pad])
                padded.append(f)
            combined = cv2.hconcat(padded)

        status = f"OSC -> {self.config.osc_ip}:{self.config.osc_port} | People: {len(tracked)} | Frame: {self.frame_count}"
        cv2.putText(combined, status, (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Tracker V2.5", combined)

        # Keep settings window alive
        simg = np.zeros((50, 400, 3), dtype=np.uint8)
        cv2.putText(simg, "Adjust sliders - auto-saves", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Settings", simg)

    def _log_health(self):
        elapsed = time.time() - self.start_time
        uptime = timedelta(seconds=int(elapsed))
        connected = sum(1 for c in self.cameras if c.connected)
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_yolo = (sum(self._yolo_times) / len(self._yolo_times) * 1000) if self._yolo_times else 0

        logger.info(
            f"HEALTH: uptime={uptime}, frames={self.frame_count}, fps={avg_fps:.1f}, "
            f"yolo={avg_yolo:.1f}ms, cameras={connected}/{len(self.cameras)}, "
            f"osc_errors={self.osc_errors}, tracked={self.total_tracked}"
        )
        for cfg in self.cam_configs:
            c = cfg['camera']
            logger.info(
                f"  {cfg['name']}: connected={c.connected}, fps={cfg['current_fps']:.1f}, "
                f"frames={c.stats['received']}, reconnects={c.stats['reconnects']}"
            )

    def _reset_yolo(self):
        """Periodically reset to prevent memory buildup."""
        logger.info("Resetting YOLO model state...")
        try:
            w = self.config.process_width
            dummy = np.zeros((w, w, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, classes=[self.config.person_class_id])
            logger.info("YOLO reset complete")
        except Exception as e:
            logger.warning(f"YOLO reset failed: {e}")

    def _cleanup(self):
        if self.settings.is_dirty:
            self.settings.save()

        for cam in self.cameras:
            logger.info(f"Releasing {cam.name}...")
            cam.release()

        if not self.config.headless:
            cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        uptime = timedelta(seconds=int(elapsed))
        logger.info("Shutdown complete")
        logger.info(f"  Uptime: {uptime}")
        logger.info(f"  Frames: {self.frame_count}")
        if elapsed > 0:
            logger.info(f"  Avg FPS: {self.frame_count / elapsed:.1f}")
        logger.info(f"  OSC errors: {self.osc_errors}")
        logger.info(f"  People tracked: {self.total_tracked}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Camera Tracker V2.5")
    parser.add_argument('--headless', action='store_true',
                        default=os.environ.get('HEADLESS', '').strip() in ('1', 'true', 'yes'),
                        help="Run without GUI (also set via HEADLESS=1 env var)")
    parser.add_argument('--process-width', type=int, default=416, help="YOLO input width (default: 416)")
    parser.add_argument('--osc-ip', default="127.0.0.1", help="OSC target IP")
    parser.add_argument('--osc-port', type=int, default=7000, help="OSC target port")
    parser.add_argument('--fps', type=int, default=25, help="Target FPS")
    args = parser.parse_args()

    if not acquire_lock():
        sys.exit(1)
    atexit.register(release_lock)

    config = TrackerConfig(
        headless=args.headless,
        process_width=args.process_width,
        osc_ip=args.osc_ip,
        osc_port=args.osc_port,
        target_fps=args.fps,
    )
    settings = TrackerSettings(SETTINGS_FILE)
    tracker = Tracker(config, settings)
    tracker.start()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal: {e}")
    finally:
        release_lock()
