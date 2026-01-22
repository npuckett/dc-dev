#!/usr/bin/env python3
"""Show camera positions from calibration data."""

import json
import numpy as np
import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Physical setup constants (positive Y = UP)
STREET_LEVEL_Y = -66.0      # Street is 66cm below floor (Y=-66)
CAMERA_LEDGE_Y = -16.0      # Camera ledge is 16cm below floor (Y=-16)
EXPECTED_CAM_HEIGHT = 50.0  # Cameras should be 50cm above street

with open(os.path.join(SCRIPT_DIR, 'camera_calibration.json'), 'r') as f:
    data = json.load(f)

print('Camera positions in world coordinates:')
print('Coordinate system: positive Y = UP')
print()
print('Physical setup:')
print(f'  Storefront floor: Y = 0')
print(f'  Camera ledge:     Y = {CAMERA_LEDGE_Y:.0f} cm (expected)')
print(f'  Street level:     Y = {STREET_LEVEL_Y:.0f} cm')
print()

for name, calib in data['cameras'].items():
    rvec = np.array(calib['rvec'])
    tvec = np.array(calib['tvec'])
    R, _ = cv2.Rodrigues(rvec)
    
    # Camera position in world coords: cam_pos = -R^T @ tvec
    cam_pos = -R.T @ tvec.flatten()
    
    # Height calculations (positive Y = up)
    height_above_street = cam_pos[1] - STREET_LEVEL_Y
    height_below_floor = -cam_pos[1]  # positive means below floor
    
    expected_y = CAMERA_LEDGE_Y
    discrepancy = cam_pos[1] - expected_y
    
    print(f'{name}:')
    print(f'  Calibrated position: X={cam_pos[0]:.1f}, Y={cam_pos[1]:.1f}, Z={cam_pos[2]:.1f} cm')
    print(f'  Height above street: {height_above_street:.1f} cm ({height_above_street/2.54:.1f} inches)')
    print(f'  Expected height:     {EXPECTED_CAM_HEIGHT:.1f} cm ({EXPECTED_CAM_HEIGHT/2.54:.1f} inches)')
    if abs(discrepancy) > 5:
        print(f'  ⚠️  Discrepancy: {abs(discrepancy):.1f} cm from expected ledge position')
    print()
