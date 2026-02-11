#!/usr/bin/env python3
"""
Generate ArUco markers optimized for US Letter paper (8.5" x 11").

Markers:
  - Dictionary: DICT_4X4_50
  - IDs: 0–6 (same as calibration setup)
  - Physical size: 20 cm × 20 cm (fits letter paper with margins)
  - Output: high-res PNG at 300 DPI for crisp printing
  - Each marker gets a full-page PDF-ready PNG with:
    - Centered marker on white background
    - Crop marks for cutting
    - ID label and size annotation
    - 1-inch margins on all sides

Coordinate reference (unchanged):
  ID 0: (-30, -66, 168)   Right front     - Camera 1 only
  ID 1: (-150, -66, 168)  Center front    - SHARED
  ID 2: (-270, -66, 168)  Left front      - Camera 2 only
  ID 3: (-30, -66, 219)   Right back      - Camera 1 only
  ID 4: (-270, -66, 219)  Left back       - Camera 2 only
  ID 5: (-150, -15, 628)  Subway wall     - SHARED (vertical)
  ID 6: (-150, -66, 219)  Center back     - SHARED

Usage:
    python generate_markers.py
"""

import cv2
import numpy as np
import os

# =============================================================================
# Configuration
# =============================================================================
MARKER_IDS = [0, 1, 2, 3, 4, 5, 6]
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKER_SIZE_CM = 20.0  # Physical printed size: 20 cm

# US Letter paper: 8.5" x 11" = 21.59 cm x 27.94 cm
PAPER_WIDTH_IN = 8.5
PAPER_HEIGHT_IN = 11.0
DPI = 300  # Print resolution

# Paper dimensions in pixels at 300 DPI
PAPER_WIDTH_PX = int(PAPER_WIDTH_IN * DPI)   # 2550
PAPER_HEIGHT_PX = int(PAPER_HEIGHT_IN * DPI)  # 3300

# Marker size in pixels at 300 DPI  (20 cm = 7.874 inches)
MARKER_SIZE_IN = MARKER_SIZE_CM / 2.54
MARKER_SIZE_PX = int(MARKER_SIZE_IN * DPI)  # ~2362 px

# Quiet zone: white border around marker (at least 1 cell width)
# For 4x4 markers, 1 cell = marker_px/6 (4 data cells + 1 border on each side)
QUIET_ZONE_CELLS = 1  # Extra white cells around marker

# Marker world positions for annotation
MARKER_INFO = {
    0: {"pos": "(-30, -66, 168)",   "desc": "Right front",   "cam": "Camera 1"},
    1: {"pos": "(-150, -66, 168)",  "desc": "Center front",  "cam": "SHARED"},
    2: {"pos": "(-270, -66, 168)",  "desc": "Left front",    "cam": "Camera 2"},
    3: {"pos": "(-30, -66, 219)",   "desc": "Right back",    "cam": "Camera 1"},
    4: {"pos": "(-270, -66, 219)",  "desc": "Left back",     "cam": "Camera 2"},
    5: {"pos": "(-150, -15, 628)",  "desc": "Subway wall (VERTICAL)", "cam": "SHARED"},
    6: {"pos": "(-150, -66, 219)",  "desc": "Center back",   "cam": "SHARED"},
}

# =============================================================================
# Generate markers
# =============================================================================
def generate_marker_page(marker_id, output_dir):
    """Generate a single marker on a letter-sized page, ready to print."""
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    
    # Generate the raw marker image (high res for crisp edges)
    # Use a large pixel size for the raw marker, then scale
    raw_marker_px = MARKER_SIZE_PX
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, raw_marker_px)
    
    # Create white letter-size page
    page = np.ones((PAPER_HEIGHT_PX, PAPER_WIDTH_PX), dtype=np.uint8) * 255
    
    # Center the marker on the page
    x_offset = (PAPER_WIDTH_PX - raw_marker_px) // 2
    y_offset = (PAPER_HEIGHT_PX - raw_marker_px) // 2
    
    # Place marker
    page[y_offset:y_offset + raw_marker_px, x_offset:x_offset + raw_marker_px] = marker_img
    
    # Convert to BGR for colored annotations
    page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    
    # --- Add crop marks ---
    mark_len = int(0.25 * DPI)  # 0.25 inch crop marks
    mark_gap = int(0.1 * DPI)   # 0.1 inch gap from marker edge
    color_crop = (150, 150, 150)  # Light gray
    thickness = 2
    
    corners = [
        (x_offset - mark_gap, y_offset - mark_gap),  # Top-left
        (x_offset + raw_marker_px + mark_gap, y_offset - mark_gap),  # Top-right
        (x_offset - mark_gap, y_offset + raw_marker_px + mark_gap),  # Bottom-left
        (x_offset + raw_marker_px + mark_gap, y_offset + raw_marker_px + mark_gap),  # Bottom-right
    ]
    
    for cx, cy in corners:
        # Horizontal line
        dx = -mark_len if cx < PAPER_WIDTH_PX // 2 else mark_len
        cv2.line(page_color, (cx, cy), (cx + dx, cy), color_crop, thickness)
        # Vertical line
        dy = -mark_len if cy < PAPER_HEIGHT_PX // 2 else mark_len
        cv2.line(page_color, (cx, cy), (cx, cy + dy), color_crop, thickness)
    
    # --- Add text annotations (below marker, won't interfere with detection) ---
    info = MARKER_INFO[marker_id]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title: large ID text above marker
    title = f"ArUco ID: {marker_id}"
    title_scale = 2.5
    title_thick = 4
    (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thick)
    title_x = (PAPER_WIDTH_PX - tw) // 2
    title_y = y_offset - int(0.6 * DPI)
    if title_y > th:
        cv2.putText(page_color, title, (title_x, title_y), font, title_scale, (0, 0, 0), title_thick)
    
    # Details below marker
    detail_y = y_offset + raw_marker_px + int(0.5 * DPI)
    detail_scale = 1.2
    detail_thick = 2
    
    lines = [
        f"Size: {MARKER_SIZE_CM:.0f} cm x {MARKER_SIZE_CM:.0f} cm",
        f"Position: {info['pos']}",
        f"{info['desc']} - {info['cam']}",
        f"Dict: DICT_4X4_50 | Print at 100% scale",
    ]
    
    for i, line in enumerate(lines):
        (lw, lh), _ = cv2.getTextSize(line, font, detail_scale, detail_thick)
        lx = (PAPER_WIDTH_PX - lw) // 2
        ly = detail_y + i * int(lh * 2.2)
        if ly < PAPER_HEIGHT_PX - 50:
            cv2.putText(page_color, line, (lx, ly), font, detail_scale, (80, 80, 80), detail_thick)
    
    # Save
    filename = f"marker_{marker_id}_letter.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, page_color)
    print(f"  ✅ {filename} ({PAPER_WIDTH_PX}x{PAPER_HEIGHT_PX} px @ {DPI} DPI, marker={MARKER_SIZE_CM:.0f}cm)")
    
    # Also save a raw marker PNG (no annotations, just the marker at print resolution)
    raw_filename = f"marker_{marker_id}_raw.png"
    raw_filepath = os.path.join(output_dir, raw_filename)
    cv2.imwrite(raw_filepath, marker_img)
    
    return filepath


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("  ArUco Marker Generator — Letter Paper (20 cm markers)")
    print("=" * 60)
    print(f"  Dictionary:    DICT_4X4_50")
    print(f"  Marker size:   {MARKER_SIZE_CM:.0f} cm ({MARKER_SIZE_IN:.2f} inches)")
    print(f"  Paper:         {PAPER_WIDTH_IN}\" x {PAPER_HEIGHT_IN}\" (US Letter)")
    print(f"  Resolution:    {DPI} DPI")
    print(f"  Marker pixels: {MARKER_SIZE_PX} x {MARKER_SIZE_PX}")
    print(f"  Page pixels:   {PAPER_WIDTH_PX} x {PAPER_HEIGHT_PX}")
    print(f"  Output:        {output_dir}")
    print()
    
    print("Generating markers...")
    for mid in MARKER_IDS:
        generate_marker_page(mid, output_dir)
    
    print()
    print("=" * 60)
    print("  PRINTING INSTRUCTIONS")
    print("=" * 60)
    print("  1. Print each _letter.png at 100% scale (no fit-to-page)")
    print("  2. Use 'Actual Size' or '100%' in print settings")
    print("  3. Cut along crop marks for exact 20cm x 20cm markers")
    print("  4. Mount flat on ground (or vertical for marker 5)")
    print("  5. Ensure markers are oriented correctly (text readable)")
    print()
    print(f"  ⚠️  Marker size changed: 15 cm → {MARKER_SIZE_CM:.0f} cm")
    print(f"  ⚠️  Update world_coordinates.json marker_size accordingly")
    print()


if __name__ == "__main__":
    main()
