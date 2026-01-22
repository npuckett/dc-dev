#!/usr/bin/env python3
"""
Force-correct camera Y positions to match known physical height.

Use this if you trust your physical measurements more than the calibration's
computed camera height. This adjusts only the Y component while preserving
the rotation and X/Z positioning from calibration.

Physical setup:
- Street level: Y = +66 (66 cm below floor)
- Camera ledge: Y = +16 (50 cm above street, 16 cm below floor)
"""

import json
import numpy as np
import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, 'camera_calibration.json')

# Known physical values (positive Y = UP)
STREET_LEVEL_Y = -66.0      # Street is 66cm below floor
CAMERA_HEIGHT_ABOVE_STREET = 50.0  # Cameras are 50cm above street
TARGET_CAMERA_Y = STREET_LEVEL_Y + CAMERA_HEIGHT_ABOVE_STREET  # = -16

def force_correct_camera_height():
    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)
    
    print("Force-correcting camera Y positions to match physical mount height")
    print(f"Target: {CAMERA_HEIGHT_ABOVE_STREET} cm above street (Y = +{TARGET_CAMERA_Y})")
    print()
    
    for camera_name, calib in data['cameras'].items():
        rvec = np.array(calib['rvec'])
        tvec = np.array(calib['tvec'])
        R, _ = cv2.Rodrigues(rvec)
        
        # Current camera position
        cam_pos = -R.T @ tvec.flatten()
        
        print(f"{camera_name}:")
        print(f"  Current Y: {cam_pos[1]:.1f} cm ({STREET_LEVEL_Y - cam_pos[1]:.1f} cm above street)")
        
        # Adjust Y to target
        y_correction = TARGET_CAMERA_Y - cam_pos[1]
        cam_pos[1] = TARGET_CAMERA_Y
        
        # Recompute tvec from corrected camera position
        # tvec = -R @ cam_pos
        tvec_new = -R @ cam_pos.reshape(3, 1)
        
        print(f"  Corrected Y: {cam_pos[1]:.1f} cm ({STREET_LEVEL_Y - cam_pos[1]:.1f} cm above street)")
        print(f"  Y correction applied: {y_correction:.1f} cm")
        
        calib['tvec'] = tvec_new.tolist()
    
    # Mark as height-corrected
    data['height_corrected'] = True
    data['camera_height_above_street'] = CAMERA_HEIGHT_ABOVE_STREET
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print()
    print("✅ Camera heights corrected and saved!")

if __name__ == '__main__':
    response = input("This will modify calibration to force camera Y to 50cm above street. Continue? [y/N]: ")
    if response.lower() == 'y':
        force_correct_camera_height()
    else:
        print("Cancelled.")
