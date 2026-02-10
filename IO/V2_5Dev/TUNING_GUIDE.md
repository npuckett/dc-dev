# Camera Tracker V2.5 — Tuning Guide

Three live sliders, one config-only value. That's it.

---

## Live Sliders (adjustable while running)

### Confidence (0.10 – 0.80)
**What it does:** Minimum YOLO detection score to count as a person. Lower = more sensitive, higher = fewer false positives.

**When to adjust:**
- **Too many ghost detections** (shadows, signs, bags) → raise toward 0.50–0.60
- **Missing people who are far away or partially hidden** → lower toward 0.25–0.35
- **Night vs day** — lower at night when visibility drops

**Default:** 0.40 — good general-purpose starting point.

---

### Fusion Distance (50 – 300 cm)
**What it does:** Maximum distance (in cm) between detections from different cameras to merge them as the same person. Also controls track matching (frame-to-frame matching uses 60% of this value automatically).

**When to adjust:**
- **Seeing duplicate people** (same person counted twice, one per camera) → increase toward 180–250
- **Two real people merging into one** (standing close together) → decrease toward 80–120
- **After re-calibrating cameras** — calibration accuracy affects how well positions align between cameras

**Default:** 150 cm — works well for typical sidewalk spacing.

**Note:** Track matching threshold is derived from this (60% = 90cm at default). You don't need to tune it separately.

---

### Smoothing (0.01 – 0.20)
**What it does:** How quickly tracked positions respond to new detections. This is an EMA (exponential moving average) alpha. Lower = smoother/slower, higher = snappier/noisier.

**When to adjust:**
- **People appear jittery** (positions jumping around) → lower toward 0.02–0.03
- **People appear to lag behind their actual position** → raise toward 0.08–0.12
- **Fast-moving people** (joggers, cyclists) → raise slightly
- **Slow-moving or standing people** → lower for stability

**Default:** 0.03 — prioritizes smooth positions over instant response. Good for an art installation where visual smoothness matters.

---

## Config-Only (in tracker_settings.json)

### max_lost_frames (15 – 150)
**What it does:** How many frames a person's track survives without any matching detection before being removed. At 25 FPS, `60` frames = 2.4 seconds.

**When to adjust:**
- **People disappear and immediately reappear with a new ID** → increase (try 90–120)
- **"Phantom" tracks persist long after someone has left** → decrease (try 30–45)
- **Brief occlusions** (person walks behind a pole) — higher values help maintain their ID

**Default:** 60 frames (~2.4 seconds at 25 FPS)

**How to change:** Edit `tracker_settings.json`:
```json
{
  "max_lost_frames": 60
}
```

---

## Quick Reference

| Situation | Adjust | Direction |
|---|---|---|
| Ghost detections | Confidence | ↑ raise |
| Missing distant people | Confidence | ↓ lower |
| Same person counted twice | Fusion Distance | ↑ raise |
| Two people merging into one | Fusion Distance | ↓ lower |
| Jittery positions | Smoothing | ↓ lower |
| Laggy positions | Smoothing | ↑ raise |
| Tracks dropping too early | max_lost_frames | ↑ raise |
| Phantom tracks linger | max_lost_frames | ↓ lower |
