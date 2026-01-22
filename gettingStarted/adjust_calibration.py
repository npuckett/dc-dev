#!/usr/bin/env python3
"""
Adjust existing camera calibration for street level offset.

The calibration was originally done with markers at Y=0, but they are actually
on the street which is 66 cm below the storefront floor level (Y=-66).

This script adjusts the tvec values to account for this coordinate shift.
"""

import json
import numpy as np
import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, 'camera_calibration.json')

# The Y offset: markers were calibrated at Y=0, but should be at Y=-66
# Standard convention: positive Y = UP, so street (below floor) = -66
STREET_LEVEL_Y = -66.0

def adjust_calibration():
    # Load existing calibration
    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)
    
    # Check if already adjusted
    if data.get('adjusted_for_street_level'):
        print("⚠️  Calibration was already adjusted for street level!")
        print("    If you want to re-adjust, manually set 'adjusted_for_street_level' to false")
        return
    
    print(f"Adjusting calibration for street level (Y = {STREET_LEVEL_Y} cm)")
    print()
    
    for camera_name, calib in data['cameras'].items():
        rvec = np.array(calib['rvec'])
        tvec = np.array(calib['tvec'])
        
        print(f"{camera_name}:")
        print(f"  Old tvec: [{tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f}]")
        
        # Convert rvec to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate adjustment: tvec_new = tvec_old - R @ offset
        # offset = [0, y_offset, 0] (we're shifting world Y down by 66)
        offset = np.array([[0], [STREET_LEVEL_Y], [0]])
        tvec_new = tvec - R @ offset
        
        print(f"  New tvec: [{tvec_new[0,0]:.2f}, {tvec_new[1,0]:.2f}, {tvec_new[2,0]:.2f}]")
        
        # Update the calibration
        calib['tvec'] = tvec_new.tolist()
    
    # Add metadata about the adjustment
    data['street_level_y'] = STREET_LEVEL_Y
    data['adjusted_for_street_level'] = True
    
    # Save the adjusted calibration
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print()
    print("✅ Calibration adjusted and saved!")
    print(f"   Markers now at Y = {STREET_LEVEL_Y} cm (street level)")

if __name__ == '__main__':
    adjust_calibration()
