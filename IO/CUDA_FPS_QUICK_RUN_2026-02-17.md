# CUDA Tracker FPS Quick Run (Production)

## Objective
Increase tracker FPS on NVIDIA CUDA Linux while keeping:
- same two camera inputs
- same OSC outputs (schema/fields/rates)

## Run Order (stop if acceptance fails)
1. Baseline measure
2. Fast ROI code changes
3. Headless/runtime tightening
4. Decode/preprocess tuning
5. Queue pipeline refactor (only if needed)
6. ONNX/TensorRT/NVDEC trials (last resort)

## 1) Baseline Measure (No behavior changes)
- Use fixed settings: model/conf/process width/headless flag
- Run 3 scenarios for 5 min each:
  - empty
  - sparse
  - dense
- Capture:
  - total FPS
  - per-camera FPS
  - inference p50/p95
  - decode p50/p95
  - reconnect/drop stats

## 2) Fast ROI Changes
- Batch both camera frames into one YOLO inference call
- Reduce repeated tensor→CPU transfers
- Vectorize floor projection path
- Keep OSC payload contract exactly unchanged

## 3) Headless/Runtime Tightening
- Run production headless (disable tracker GUI rendering)
- Re-run same 3 scenarios and compare against baseline

## 4) Decode/Preprocess Tuning
- Apply low-latency camera capture options in RobustCamera
- Preserve latest-frame semantics and reconnect behavior
- Re-run benchmark set

## 5) Pipeline Refactor (Only if still needed)
- Add bounded queues:
  - decode/preprocess
  - inference
  - postprocess/OSC
- Frame policy: newest frame wins
- Verify no lag accumulation over long run

## 6) Major Backend Trials (Only if still needed)
- ONNX Runtime CUDA
- TensorRT (start FP16)
- NVDEC hardware decode
- Promote only if FPS gain is clear and OSC contract remains equivalent

## Acceptance Gate (every stage)
Pass only if all true:
- same two camera inputs
- same OSC schema/fields/rates
- measurable FPS gain
- improved p95 infer/decode latency
- no reconnect/stability regressions

## Rollback Rule
- One stage per commit
- Revert immediately on acceptance failure
- Do not stack unverified stages

## File Pointers
- Tracker main: [IO/camera_tracker_osc.py](IO/camera_tracker_osc.py)
- Controller load check: [IO/lightController_osc.py](IO/lightController_osc.py)
- DB stability check: [IO/tracking_database.py](IO/tracking_database.py)
