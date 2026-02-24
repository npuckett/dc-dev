#!/usr/bin/env python3
"""
Camera Tracker V4 — FPS-Optimized Build

Based on camera_tracker_osc.py (V2.5). Implements:
  Stage 1: Per-stage timing instrumentation with percentile reporting
  Stage 2: Batched YOLO inference, vectorized projection, reduced GPU→CPU transfers

Same calibration, same cameras, same OSC output contract.

OSC Messages Sent:
  /tracker/person/<id> <x> <z>  — Position in world cm
  /tracker/count <n>            — Number of tracked people

Usage:
    python camera_tracker_v4.py [--headless] [--process-width 416] [--benchmark-interval 60]

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
# FILE PATHS — Reference IO/ directory for calibration & settings
# ==============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IO_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'IO')

# Calibration and settings live in IO/ (shared with production tracker)
CALIBRATION_FILE = os.path.join(_IO_DIR, 'camera_calibration.json')
SETTINGS_FILE = os.path.join(_IO_DIR, 'tracker_settings.json')

# ==============================================================================
# SINGLE INSTANCE LOCK (separate from production tracker)
# ==============================================================================

LOCK_FILE = "/tmp/camera_tracker_v4.lock"
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
            print(f"Another V4 tracker is already running (PID: {pid})")
        except Exception:
            print("Another V4 tracker is already running")
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
# STAGE 1: TIMING INSTRUMENTATION
# ==============================================================================

class StageTimer:
    """
    Collects per-stage timing samples and reports percentiles.
    Each stage is a named bucket. Call .start(name) / .stop(name) around code,
    or use .measure(name) as a context manager.
    """

    def __init__(self, window: int = 500):
        self.window = window
        self._buckets: Dict[str, deque] = {}
        self._active: Dict[str, float] = {}

    def start(self, name: str):
        self._active[name] = time.perf_counter()

    def stop(self, name: str):
        t0 = self._active.pop(name, None)
        if t0 is None:
            return 0.0
        dt = time.perf_counter() - t0
        if name not in self._buckets:
            self._buckets[name] = deque(maxlen=self.window)
        self._buckets[name].append(dt)
        return dt

    class _Ctx:
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name
        def __enter__(self):
            self.timer.start(self.name)
            return self
        def __exit__(self, *_):
            self.timer.stop(self.name)

    def measure(self, name: str):
        """Context manager: with timer.measure('stage'): ..."""
        return self._Ctx(self, name)

    def percentiles(self, name: str) -> Dict[str, float]:
        """Return p50, p95, p99, mean, count for a stage (in ms)."""
        buf = self._buckets.get(name)
        if not buf or len(buf) == 0:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0, 'count': 0}
        arr = sorted(buf)
        n = len(arr)
        return {
            'p50': arr[int(n * 0.50)] * 1000,
            'p95': arr[int(min(n * 0.95, n - 1))] * 1000,
            'p99': arr[int(min(n * 0.99, n - 1))] * 1000,
            'mean': (sum(arr) / n) * 1000,
            'count': n,
        }

    def all_stages(self) -> List[str]:
        return list(self._buckets.keys())

    def report(self) -> str:
        """Formatted benchmark report for all stages."""
        lines = ["=== BENCHMARK REPORT ==="]
        total_mean = 0
        for name in self.all_stages():
            p = self.percentiles(name)
            lines.append(
                f"  {name:20s}  mean={p['mean']:6.2f}ms  "
                f"p50={p['p50']:6.2f}ms  p95={p['p95']:6.2f}ms  "
                f"p99={p['p99']:6.2f}ms  (n={p['count']})"
            )
            total_mean += p['mean']
        lines.append(f"  {'TOTAL':20s}  mean={total_mean:6.2f}ms  → max theoretical FPS={1000/total_mean:.1f}" if total_mean > 0 else "  TOTAL  (no data)")
        return '\n'.join(lines)

    def reset(self):
        self._buckets.clear()
        self._active.clear()


# ==============================================================================
# CONFIGURATION (identical to V2.5)
# ==============================================================================

class TrackerConfig:
    """All configuration in one place. Immutable after init."""

    # OSC
    osc_ip: str = "127.0.0.1"
    osc_port: int = 7000

    # Cameras — same as production
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

    # V4: Benchmark reporting interval (seconds)
    benchmark_interval: int = 60

    # Calibration floor level
    floor_y: float = -66.0

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ==============================================================================
# LIVE-TUNABLE SETTINGS (identical to V2.5)
# ==============================================================================

class TrackerSettings:
    """
    Minimal parameter set.

    Live sliders (3):
        confidence    — YOLO detection threshold
        fusion_dist   — Cross-camera merge distance (cm)
        smoothing     — Position EMA alpha

    Config-only (1):
        max_lost_frames — Frames before dropping a track
    """

    DEFAULTS = {
        'confidence': 0.40,
        'fusion_dist': 150.0,
        'smoothing': 0.03,
        'max_lost_frames': 60,
    }

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
        d = self.SLIDER_DEFS[key]
        return int(self.values[key] * d['scale'])

    def from_slider_value(self, key: str, slider_val: int):
        d = self.SLIDER_DEFS[key]
        self.set(key, slider_val / d['scale'])


# ==============================================================================
# CALIBRATION — STAGE 2: Vectorized floor projection
# ==============================================================================

class CalibrationManager:
    """
    Loads camera calibration and projects bounding-box feet to floor plane.
    Pre-computes R^T and K^-1 at load time.

    V4 addition: batch_feet_to_floor() for vectorized projection of N points.
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
        """Single-point projection (kept for compatibility)."""
        cal = self.calibrations.get(camera_name)
        if cal is None:
            return None

        pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, cal['K'], cal['dist'], P=cal['K'])
        ux, uy = und[0, 0]

        ray_cam = cal['K_inv'] @ np.array([ux, uy, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = cal['R_T'] @ ray_cam

        if abs(ray_world[1]) < 1e-6:
            return None
        t = (self.floor_y - cal['cam_pos'][1]) / ray_world[1]
        if t < 0:
            return None

        hit = cal['cam_pos'] + t * ray_world
        return (float(hit[0]), float(hit[2]))

    def batch_feet_to_floor(self, camera_name: str, feet: np.ndarray) -> np.ndarray:
        """
        STAGE 2: Vectorized projection of N foot points to floor plane.

        Args:
            camera_name: Camera identifier
            feet: (N, 2) array of image coordinates [foot_x, foot_y]

        Returns:
            (N, 3) array where columns are [world_x, world_z, valid].
            valid=1.0 if projection succeeded, 0.0 otherwise.
        """
        n = len(feet)
        result = np.zeros((n, 3), dtype=np.float64)  # [wx, wz, valid]

        cal = self.calibrations.get(camera_name)
        if cal is None or n == 0:
            return result

        # Undistort all points at once — cv2.undistortPoints handles (N,1,2)
        pts = feet.reshape(-1, 1, 2).astype(np.float32)
        und = cv2.undistortPoints(pts, cal['K'], cal['dist'], P=cal['K'])
        und = und.reshape(-1, 2)  # (N, 2)

        # Build rays in camera space: K_inv @ [ux, uy, 1]^T for each point
        # Stack as (N, 3) homogeneous: [ux, uy, 1]
        ones = np.ones((n, 1), dtype=np.float64)
        homog = np.hstack([und.astype(np.float64), ones])  # (N, 3)

        # rays_cam = homog @ K_inv^T  → each row is K_inv @ [ux, uy, 1]
        rays_cam = homog @ cal['K_inv'].T  # (N, 3)

        # Normalize each ray
        norms = np.linalg.norm(rays_cam, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0  # avoid div by zero
        rays_cam /= norms

        # Rotate to world: rays_world = rays_cam @ R_T^T = rays_cam @ R
        rays_world = rays_cam @ cal['R_T'].T  # (N, 3)

        # Intersect floor plane y = floor_y
        # t = (floor_y - cam_pos[1]) / ray_world[:, 1]
        ray_y = rays_world[:, 1]
        valid_mask = np.abs(ray_y) > 1e-6
        t = np.zeros(n, dtype=np.float64)
        t[valid_mask] = (self.floor_y - cal['cam_pos'][1]) / ray_y[valid_mask]

        # Only keep positive t (ray going toward floor)
        valid_mask &= (t > 0)

        # Compute hit points: cam_pos + t * ray_world
        hits = cal['cam_pos'][np.newaxis, :] + t[:, np.newaxis] * rays_world  # (N, 3)

        result[valid_mask, 0] = hits[valid_mask, 0]  # world_x
        result[valid_mask, 1] = hits[valid_mask, 2]  # world_z
        result[valid_mask, 2] = 1.0                   # valid flag

        return result

    def bbox_to_floor(self, camera_name: str, x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
        """Project bounding box bottom-center (feet) to floor."""
        return self.feet_to_floor(camera_name, (x1 + x2) / 2.0, y2)


# ==============================================================================
# ROBUST CAMERA (identical to V2.5)
# ==============================================================================

class RobustCamera:
    """Reliable RTSP camera with auto-reconnect."""

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
                        self._frame = frame
                        self._frame_time = time.time()
                        self._frame_num += 1
                        self.stats['received'] += 1

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
# TRACKING FUSION (identical to V2.5)
# ==============================================================================

class TrackingFusion:
    """Merges detections from multiple cameras and maintains smooth tracks."""

    def __init__(self, settings: TrackerSettings):
        self.settings = settings
        self.tracks: Dict[int, dict] = {}
        self._next_id = 1
        self._frame = 0

    def process(self, detections: List[dict]) -> List[Tuple[int, float, float]]:
        self._frame += 1
        fused = self._fuse(detections)
        return self._match_and_smooth(fused)

    def _fuse(self, detections: List[dict]) -> List[Tuple[float, float]]:
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
                px = t['x'] + t['vx']
                pz = t['z'] + t['vz']
                d2 = (raw_x - px) ** 2 + (raw_z - pz) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = tid

            if best_id is not None:
                matched.add(best_id)
                t = self.tracks[best_id]

                pred_x = t['x'] + t['vx']
                pred_z = t['z'] + t['vz']
                new_x = pred_x + alpha * (raw_x - pred_x)
                new_z = pred_z + alpha * (raw_z - pred_z)

                t['vx'] += alpha * ((new_x - t['x']) - t['vx'])
                t['vz'] += alpha * ((new_z - t['z']) - t['vz'])
                t['x'] = new_x
                t['z'] = new_z
                t['last_seen'] = self._frame

                output.append((best_id, new_x, new_z))
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {
                    'x': raw_x, 'z': raw_z,
                    'vx': 0.0, 'vz': 0.0,
                    'last_seen': self._frame,
                }
                output.append((tid, raw_x, raw_z))

        max_age = self.settings.max_lost_frames
        stale = [k for k, v in self.tracks.items()
                 if self._frame - v['last_seen'] > max_age]
        for k in stale:
            del self.tracks[k]

        return output


# ==============================================================================
# TRACKER V4 — Batched inference + instrumented pipeline
# ==============================================================================

class Tracker:
    """
    V4 tracker pipeline with Stage 1 + Stage 2 optimizations:

    Stage 1 additions:
      - StageTimer instruments every pipeline phase
      - Periodic benchmark report (configurable interval)
      - Per-frame total timing

    Stage 2 changes:
      - _detect_all_batched(): collects frames from all cameras, resizes them,
        then runs ONE batched YOLO inference call instead of per-camera calls
      - Single tensor→CPU transfer for all boxes/scores
      - Vectorized floor projection via CalibrationManager.batch_feet_to_floor()
      - Same OSC output contract
    """

    def __init__(self, config: TrackerConfig, settings: TrackerSettings):
        self.config = config
        self.settings = settings
        self.shutdown = False

        # Components
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

        # Stage 1: Instrumentation
        self.timer = StageTimer(window=500)

    def start(self):
        """Initialize all components and enter main loop."""
        logger.info("=" * 50)
        logger.info("Camera Tracker V4 — FPS Optimized")
        logger.info("  Stage 1: Per-stage timing instrumentation")
        logger.info("  Stage 2: Batched inference + vectorized projection")
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

        # Warmup
        if self.device.startswith("cuda"):
            w = self.config.process_width
            dummy = np.zeros((w, w, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, classes=[self.config.person_class_id])
            logger.info("GPU warmup complete")
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
                    '_last_boxes': [],
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
        cv2.namedWindow("Tracker V4", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Settings", 400, 200)

        for key, sdef in TrackerSettings.SLIDER_DEFS.items():
            cv2.createTrackbar(
                sdef['label'], "Settings",
                self.settings.to_slider_value(key),
                sdef['max'],
                lambda val, k=key: self.settings.from_slider_value(k, val)
            )

    # ---------- MAIN LOOP (Stage 1: instrumented) ----------

    def _run(self):
        frame_interval = 1.0 / self.config.target_fps
        last_process = 0.0
        last_health = time.time()
        last_save = time.time()
        last_reset = time.time()
        last_benchmark = time.time()

        while not self.shutdown:
            now = time.time()
            if now - last_process < frame_interval:
                time.sleep(0.001)
                continue
            last_process = now
            self.frame_count += 1

            # Full frame timing
            self.timer.start('frame_total')

            # Stage 2: Batched detect + project
            world_dets, display_frames = self._detect_all_batched()

            # Fusion + tracking
            with self.timer.measure('fusion_smooth'):
                tracked = self.fusion.process(world_dets)

            # OSC send
            with self.timer.measure('osc_send'):
                self._send_osc(tracked)

            # Display
            if not self.config.headless and display_frames:
                with self.timer.measure('render'):
                    self._render(display_frames, tracked)

            self.timer.stop('frame_total')

            # Periodic maintenance
            if self.settings.is_dirty and now - last_save > 5.0:
                self.settings.save()
                last_save = now

            if now - last_health >= self.config.health_log_interval:
                self._log_health()
                last_health = now

            if now - last_reset >= self.config.tracker_reset_interval:
                self._reset_yolo()
                last_reset = now

            # Stage 1: Periodic benchmark report
            if now - last_benchmark >= self.config.benchmark_interval:
                logger.info('\n' + self.timer.report())
                last_benchmark = now

            # Keyboard input
            if not self.config.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown = True
                elif key == ord('s'):
                    self.settings.save()
                    print("Settings saved")
                elif key == ord('b'):
                    # Manual benchmark print
                    print(self.timer.report())

    # ---------- STAGE 2: BATCHED DETECT + VECTORIZED PROJECTION ----------

    def _detect_all_batched(self) -> Tuple[List[dict], List[np.ndarray]]:
        """
        Stage 2 optimization: Collect frames from all cameras, resize them,
        run ONE batched YOLO call, then vectorize the floor projection.

        Key differences from V2.5 _detect_all():
          1. Single model.predict() call with list of images (batch inference)
          2. Single .cpu().numpy() transfer for all results
          3. Vectorized batch_feet_to_floor() instead of per-box loop
        """
        world_dets: List[dict] = []
        display_frames: List[np.ndarray] = []
        pw = self.config.process_width
        person_cls = self.config.person_class_id

        # --- Phase 1: Decode/read frames from all cameras ---
        self.timer.start('decode_read')
        batch_images = []   # resized images for YOLO
        batch_cfgs = []     # corresponding cam_configs (only cameras with new frames)
        batch_frames = []   # original full frames (for display)

        for cfg in self.cam_configs:
            cam: RobustCamera = cfg['camera']
            ok, frame, is_new = cam.get_frame()
            if not ok or frame is None:
                continue
            if not is_new:
                # Still use cached boxes for display, but skip inference
                if not self.config.headless:
                    dw = self.config.display_width
                    ds = dw / frame.shape[1]
                    dframe = cv2.resize(frame, (dw, int(frame.shape[0] * ds)),
                                        interpolation=cv2.INTER_LINEAR)
                    self._draw_cached_boxes(dframe, cfg, ds)
                    display_frames.append(dframe)
                continue

            batch_images_entry = None  # will be set after resize
            batch_cfgs.append(cfg)
            batch_frames.append(frame)
        self.timer.stop('decode_read')

        if not batch_cfgs:
            return world_dets, display_frames

        # --- Phase 2: Resize/preprocess ---
        self.timer.start('resize_preprocess')
        for i, (cfg, frame) in enumerate(zip(batch_cfgs, batch_frames)):
            small = cv2.resize(frame, (pw, cfg['process_h']),
                               interpolation=cv2.INTER_LINEAR)
            batch_images.append(small)
        self.timer.stop('resize_preprocess')

        # --- Phase 3: Batched YOLO inference (single call) ---
        self.timer.start('inference')
        results = self.model.predict(
            batch_images,
            verbose=False,
            conf=self.settings.confidence,
            classes=[person_cls],
            imgsz=pw,
            device=self.device,
        )
        dt_infer = self.timer.stop('inference')
        self._yolo_times.append(dt_infer)

        # --- Phase 4: Postprocess + vectorized projection ---
        self.timer.start('postprocess_project')

        for result_idx, (cfg, frame) in enumerate(zip(batch_cfgs, batch_frames)):
            result = results[result_idx]
            scale_inv = 1.0 / cfg['scale']

            boxes_with_world = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # Stage 2: Single CPU transfer for all boxes from this camera
                xyxy_all = boxes.xyxy.cpu().numpy()  # (N, 4)
                conf_all = boxes.conf.cpu().numpy()   # (N,)

                # Scale back to original image coordinates
                xyxy_orig = xyxy_all * scale_inv  # (N, 4): [x1, y1, x2, y2]

                n_boxes = len(xyxy_orig)

                # Stage 2: Vectorized foot positions → floor projection
                # Feet = bottom-center of each bbox: ((x1+x2)/2, y2)
                feet = np.empty((n_boxes, 2), dtype=np.float64)
                feet[:, 0] = (xyxy_orig[:, 0] + xyxy_orig[:, 2]) / 2.0  # foot_x
                feet[:, 1] = xyxy_orig[:, 3]                              # foot_y (bottom)

                # Batch project all feet to floor in one vectorized call
                floor_pts = self.calibration.batch_feet_to_floor(cfg['name'], feet)
                # floor_pts is (N, 3): [world_x, world_z, valid]

                for i in range(n_boxes):
                    x1, y1, x2, y2 = xyxy_orig[i]
                    conf = float(conf_all[i])
                    wx = float(floor_pts[i, 0]) if floor_pts[i, 2] > 0.5 else None
                    wz = float(floor_pts[i, 1]) if floor_pts[i, 2] > 0.5 else None

                    boxes_with_world.append((x1, y1, x2, y2, conf, wx, wz))

                    if wx is not None:
                        world_dets.append({
                            'x': wx, 'z': wz,
                            'camera': cfg['name'],
                            'conf': conf,
                        })

            cfg['_last_boxes'] = boxes_with_world

            # Per-camera FPS
            if dt_infer > 0:
                per_cam_dt = dt_infer / len(batch_cfgs)
                cfg['fps_hist'].append(1.0 / max(per_cam_dt, 0.001))
                cfg['current_fps'] = sum(cfg['fps_hist']) / len(cfg['fps_hist'])

        self.timer.stop('postprocess_project')

        # --- Phase 5: Build display frames (reusing cached world positions) ---
        if not self.config.headless:
            self.timer.start('display_compose')
            for cfg, frame in zip(batch_cfgs, batch_frames):
                dw = self.config.display_width
                ds = dw / frame.shape[1]
                dframe = cv2.resize(frame, (dw, int(frame.shape[0] * ds)),
                                    interpolation=cv2.INTER_LINEAR)
                self._draw_cached_boxes(dframe, cfg, ds)
                display_frames.append(dframe)
            self.timer.stop('display_compose')

        return world_dets, display_frames

    def _draw_cached_boxes(self, dframe: np.ndarray, cfg: dict, ds: float):
        """Draw cached bounding boxes onto a display frame."""
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

    # ---------- OSC (unchanged contract) ----------

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

    # ---------- DISPLAY ----------

    def _render(self, frames: List[np.ndarray], tracked: List[Tuple[int, float, float]]):
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

        # Status bar with FPS from timing data
        frame_p = self.timer.percentiles('frame_total')
        fps_str = f"{1000/frame_p['mean']:.1f}" if frame_p['mean'] > 0 else "?"
        status = (
            f"V4 | OSC -> {self.config.osc_ip}:{self.config.osc_port} | "
            f"People: {len(tracked)} | FPS: {fps_str} | Frame: {self.frame_count}"
        )
        cv2.putText(combined, status, (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Tracker V4", combined)

        simg = np.zeros((50, 400, 3), dtype=np.uint8)
        cv2.putText(simg, "Adjust sliders | 'b'=benchmark", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Settings", simg)

    # ---------- HEALTH / MAINTENANCE ----------

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
        logger.info("Resetting YOLO model state...")
        try:
            w = self.config.process_width
            dummy = np.zeros((w, w, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, classes=[self.config.person_class_id])
            logger.info("YOLO reset complete")
        except Exception as e:
            logger.warning(f"YOLO reset failed: {e}")

    def _cleanup(self):
        # Final benchmark report
        logger.info('\n' + self.timer.report())

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
    parser = argparse.ArgumentParser(description="Camera Tracker V4 — FPS Optimized")
    parser.add_argument('--headless', action='store_true',
                        default=os.environ.get('HEADLESS', '').strip() in ('1', 'true', 'yes'),
                        help="Run without GUI (also set via HEADLESS=1 env var)")
    parser.add_argument('--process-width', type=int, default=416,
                        help="YOLO input width (default: 416)")
    parser.add_argument('--osc-ip', default="127.0.0.1", help="OSC target IP")
    parser.add_argument('--osc-port', type=int, default=7000, help="OSC target port")
    parser.add_argument('--fps', type=int, default=25, help="Target FPS")
    parser.add_argument('--benchmark-interval', type=int, default=60,
                        help="Seconds between benchmark reports (default: 60)")
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
        benchmark_interval=args.benchmark_interval,
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
