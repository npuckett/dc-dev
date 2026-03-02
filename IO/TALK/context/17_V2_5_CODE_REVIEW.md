# Camera Tracker V2 — Formal Code Review

**Reviewer:** GitHub Copilot  
**Date:** 2026-02-09  
**File reviewed:** `IO/camera_tracker_osc.py` (1,322 lines)  
**Goal:** Maximum speed, dynamic parameter adjustment, minimal parameter count, no zone sorting (delegated to lightController)

---

## Executive Summary

The V2 tracker works but was built incrementally, accumulating redundant responsibilities and unnecessary overhead. Key issues:

1. **Zone sorting is duplicated** — the tracker classifies zones that the lightController ignores and recomputes anyway
2. **Too many tuning parameters** (9 slider params) — several are tightly coupled or rarely need adjustment
3. **Frame processing is inefficient** — double world-coordinate transforms per detection (once for fusion, once for display), wasteful copy in camera buffer
4. **Cyclist merging adds complexity for minimal value** in this installation context
5. **Monolithic `main()` function** (300+ lines) mixes camera I/O, YOLO inference, fusion, OSC output, display rendering, health monitoring, and settings management

---

## Detailed Findings

### 1. ZONE SORTING — REMOVE (Critical)

**Lines 395–449: `ZoneChecker` class**  
**Lines 638–646: zone OSC messages**

The tracker sends `/tracker/zone/<id>` messages, but the lightController's `handle_zone()` is literally:
```python
def handle_zone(self, address, *args):
    # Intentionally do nothing - zone is computed locally
    pass
```

The `ZoneChecker` class, zone confidence weighting, `passive_zone_confidence` parameter, `zone_filter_enabled` parameter, and the zone-based dynamic fusion threshold are all dead weight from the lightController's perspective. The tracker should output raw positions and let the consumer decide.

**Impact:** Removing zone logic eliminates 2 parameters, ~100 lines of code, and per-detection zone lookups.

### 2. PARAMETER BLOAT — REDUCE

Current 9 parameters:
| Parameter | Purpose | Verdict |
|---|---|---|
| `confidence_threshold` | YOLO min confidence | **KEEP** — essential |
| `fusion_threshold_cm` | Cross-camera merge distance | **KEEP** — essential |
| `fusion_threshold_far_cm` | Far-object merge distance | **REMOVE** — coupled to zone system |
| `track_match_threshold_cm` | Frame-to-frame matching | **MERGE** with fusion threshold (use 60% of fusion threshold) |
| `position_smoothing` | EMA alpha for position | **KEEP** — essential |
| `velocity_smoothing` | EMA alpha for velocity | **REMOVE** — derive from position smoothing |
| `max_track_age_frames` | Track timeout | **KEEP** but move to config (rarely adjusted live) |
| `zone_filter_enabled` | Toggle zone filtering | **REMOVE** — no zones |
| `passive_zone_confidence` | Zone confidence multiplier | **REMOVE** — no zones |

**Recommended live parameters: 3** (confidence, fusion threshold, position smoothing)  
**Recommended config-only parameters: 2** (max track age, process resolution)

### 3. PERFORMANCE ISSUES

#### 3a. Double world-coordinate transform (Lines 1112–1165)
Every detection is projected to world coordinates for fusion, then *the same boxes* are projected again for display rendering. This doubles the expensive `cv2.undistortPoints` + matrix inversion calls.

**Fix:** Cache world position alongside box data.

#### 3b. Redundant frame copies in `RobustCamera` (Lines 807–812)
The camera thread does `self.cached_frame = frame.copy()` on every frame AND `self.frame = frame`. Then `read_new()` does another `.copy()`. That's 2–3 copies per frame per camera.

**Fix:** Single copy strategy — keep one buffer, return reference with lock held or do a single copy on read.

#### 3c. Buffer flush loop (Lines 816–826)
After reading a frame, the camera flushes up to 3 more frames. This means reading up to 4 frames per cycle, with full copy overhead on each. While the intent (drain RTSP buffer) is correct, the implementation is wasteful.

**Fix:** Use `CAP_PROP_BUFFERSIZE=1` (already set) and `grab()` instead of `read()` for flushing — `grab()` doesn't decode the frame.

#### 3d. YOLO model.track() overhead
The tracker calls `model.track()` with `bytetrack.yaml` which maintains internal state. But the tracker also maintains its own external tracking via `TrackingFusionV2`. This is **double tracking** — YOLO's ByteTrack assigns IDs, then fusion reassigns stable IDs.

**Fix:** Use `model.predict()` instead of `model.track()` since we do our own cross-camera tracking. Or use YOLO's track IDs directly and skip the fusion step. Pick one.

#### 3e. `FrameProcessor` class is a wrapper around single `cv2.resize` call
The class checks for CUDA support but never uses CUDA resize. It's 12 lines of indirection for `cv2.resize()`.

**Fix:** Inline the resize call.

### 4. CYCLIST MERGING — QUESTIONABLE VALUE

**Lines 375–410: `merge_cyclists()`**

This IoU-based person+bicycle merger adds complexity but:
- The installation tracks pedestrians on a sidewalk
- Cyclists would be rare and pass quickly
- The merged class ID `CYCLIST_CLASS_ID = -1` is checked in the main loop but treated identically to persons

**Recommendation:** Remove or make opt-in. Track `PERSON_CLASS_ID` only by default.

### 5. ARCHITECTURAL ISSUES

#### 5a. Monolithic `main()` — 300+ lines
The main function handles:
- Initialization (cameras, YOLO, calibration, settings)
- Frame acquisition
- YOLO inference
- Detection fusion
- OSC output
- Display rendering
- Health monitoring
- Settings auto-save
- Signal handling
- Cleanup

This should be split into a `Tracker` class with clear lifecycle methods.

#### 5b. Global constants mixed with runtime config
`PROCESS_WIDTH`, `DISPLAY_WIDTH`, `HEADLESS_MODE`, `SYNC_CAMERAS`, `TARGET_FPS` etc. are module-level constants that should be in a config object, some adjustable at runtime.

#### 5c. Settings scaling is confusing
`position_smoothing` is stored as integer 1–20, divided by 100 to get 0.01–0.20. The slider shows "3" but the actual value is 0.03. This makes debugging and documentation harder.

### 6. OSC OUTPUT FORMAT — CONFIRMED COMPATIBLE

The lightController expects:
- `/tracker/person/<id>` with args `[float x, float z]` — world coordinates in cm
- `/tracker/count` with arg `int n`

The zone message `/tracker/zone/<id>` is sent but ignored. The lightController takes raw (x, z), applies its own offset/scale/invert calibration, and determines zones locally. This confirms the tracker should NOT do zone sorting.

---

## V2.5 Refactoring Plan

### Output Contract (unchanged)
```
/tracker/person/<id>  float(x_cm)  float(z_cm)
/tracker/count        int(n)
```

### What to Remove
- `ZoneChecker` class and all zone logic
- `/tracker/zone` OSC messages
- `zone_filter_enabled`, `passive_zone_confidence`, `fusion_threshold_far_cm` parameters
- `merge_cyclists()` and bicycle tracking
- `FrameProcessor` wrapper class
- Double world-coordinate transforms
- Redundant frame copies

### What to Add
- `Tracker` class encapsulating the pipeline
- Cached world positions on detections
- `grab()` for buffer flushing instead of `read()`
- Option to use `model.predict()` instead of `model.track()` (configurable)
- Runtime-adjustable `process_width` via slider
- Cleaner parameter naming (actual float values, not scaled integers)

### Target Parameters (5 total, 3 live-adjustable)
| Parameter | Type | Range | Live? |
|---|---|---|---|
| `confidence` | float | 0.10–0.80 | Yes (slider) |
| `fusion_distance` | float cm | 50–300 | Yes (slider) |
| `smoothing` | float | 0.01–0.20 | Yes (slider) |
| `max_lost_frames` | int | 15–150 | Config file |
| `process_width` | int | 320–640 | Config file |

### Expected Performance Gains
- **~15–20% faster per frame** from eliminating double transforms, reducing copies, using grab() for flush
- **Simpler mental model** — 3 sliders instead of 9
- **Smaller codebase** — target ~700 lines (from 1,322)
- **Clearer separation** — tracker tracks, controller controls
