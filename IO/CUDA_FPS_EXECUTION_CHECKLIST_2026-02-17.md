# CUDA Tracker FPS Execution Checklist (Production)

## Scope
- Target environment: NVIDIA CUDA Linux production machine
- Invariants: same two camera inputs, same OSC outputs (schema/fields/rates)
- Goal: maximize sustained FPS while preserving behavior contract

## Stage 0 — Preflight
- Confirm GPU/runtime health: nvidia-smi, CUDA visible to PyTorch, model loads on CUDA
- Confirm tracker runs with both RTSP feeds on current production settings
- Freeze baseline parameters (model, confidence thresholds, process width, headless mode)
- Record git commit hash and config snapshot before any changes

## Stage 1 — Baseline Measurement (No behavior change)
- Add/enable stage timing in [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L790-L939):
  - decode/read
  - resize/preprocess
  - inference
  - postprocess/projection
  - fusion/smoothing
  - OSC send
  - render
- Run three 5-minute scenarios:
  - empty scene
  - sparse occupancy
  - dense occupancy
- Capture metrics:
  - total FPS
  - per-camera FPS
  - inference p50/p95
  - decode p50/p95
  - reconnect/dropped-frame stats from [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L920-L934)

## Stage 2 — Fast ROI Optimizations
- Batch two camera frames into one YOLO call in [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L801-L881)
- Reduce repeated tensor-to-CPU transfers for boxes/scores/classes
- Vectorize floor projection math currently tied to [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L309-L339)
- Keep OSC contract unchanged at [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L883-L893)
- Re-run Stage 1 benchmarks and compare

## Stage 3 — Decode/Preprocess Efficiency
- Improve capture latency settings in RobustCamera at [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L373-L456)
- Reuse low-latency FFmpeg/OpenCV patterns proven in [IO/camera_calibration.py](IO/camera_calibration.py#L2210-L2272)
- Preserve “latest frame wins” behavior and reconnect robustness
- Re-benchmark and compare against Stage 2

## Stage 4 — Production Runtime Profile
- Enforce headless production execution path (skip GUI rendering path in [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py#L895-L918))
- Verify no change to OSC payload cadence/content
- Re-benchmark and compare against Stage 3

## Stage 5 — Medium Refactor (Only if needed)
- Introduce bounded pipeline queues in [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py):
  - decode/preprocess queue
  - inference queue
  - postprocess/OSC queue
- Define explicit queue limits and frame-drop policy (newest frame priority)
- Validate stability under prolonged load (no queue growth, no lag accumulation)

## Stage 6 — Major Backend Evaluation (Only if needed)
- Track A: ONNX Runtime CUDA path for detector
- Track B: TensorRT engine path (start with FP16)
- Track C: Hardware decode path (NVDEC via GStreamer/FFmpeg)
- For each track:
  - run the same benchmark protocol
  - validate detection parity tolerance
  - validate unchanged OSC contract and controller stability

## Stage 7 — End-to-End Validation
- Soak test 2–4 hours with representative occupancy
- Verify tracker health metrics remain stable
- Verify controller ingest/render loop remains stable in [IO/lightController_osc.py](IO/lightController_osc.py#L4330-L5055)
- Verify downstream DB pipeline remains stable in [IO/tracking_database.py](IO/tracking_database.py#L467-L524)

## Acceptance Gates
- Keep a stage only if all are true:
  - same two camera inputs
  - same OSC schema/fields/rates
  - measurable FPS improvement
  - reduced p95 inference and/or decode latency
  - no regressions in reconnect behavior or long-run stability

## Rollback Rules
- One stage per commit
- If a stage fails acceptance, revert that stage immediately
- Do not stack unverified stages

## Priority Order
- Implement first: Stage 1 → Stage 2 → Stage 4
- Then: Stage 3
- Then (if still needed): Stage 5
- Last resort: Stage 6 backend tracks
