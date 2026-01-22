#!/usr/bin/env python3
"""Compare camera positions: original vs adjusted calibration."""

import numpy as np
import cv2

# Original calibration (markers at Y=0)
original = {
    'Camera 1': {
        'rvec': [[2.4010185518837655], [1.7891903942839014], [2.1348711042452293]],
        'tvec': [[-8.271782423081207], [-104.77949752667223], [-38.7823880470258]],
    },
    'Camera 2': {
        'rvec': [[2.4323719952756018], [-2.036762899326048], [-2.2021242589464967]],
        'tvec': [[4.733811749897546], [-55.45193204203457], [187.6152167117355]],
    }
}

# Adjusted calibration (markers at Y=-66)
adjusted = {
    'Camera 1': {
        'rvec': [[2.4010185518837655], [1.7891903942839014], [2.1348711042452293]],
        'tvec': [[50.282996109150304], [-132.47210327498163], [-26.115206122461053]],
    },
    'Camera 2': {
        'rvec': [[2.4323719952756018], [-2.036762899326048], [-2.2021242589464967]],
        'tvec': [[-58.49203730847344], [-72.90215102681677], [194.96245212590557]],
    }
}

def get_camera_pos(calib):
    rvec = np.array(calib['rvec'])
    tvec = np.array(calib['tvec'])
    R, _ = cv2.Rodrigues(rvec)
    return -R.T @ tvec.flatten()

print("="*60)
print("ORIGINAL Calibration (markers at Y=0 = ground)")
print("="*60)
for name, calib in original.items():
    pos = get_camera_pos(calib)
    print(f"{name}:")
    print(f"  Position: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f} cm")
    print(f"  Height above ground (Y=0): {pos[1]:.1f} cm ({pos[1]/2.54:.1f} inches)")
    print()

print("="*60)
print("ADJUSTED Calibration (markers at Y=-66 = street level)")
print("="*60)
for name, calib in adjusted.items():
    pos = get_camera_pos(calib)
    height_above_street = pos[1] - (-66)
    print(f"{name}:")
    print(f"  Position: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f} cm")
    print(f"  Height above street (Y=-66): {height_above_street:.1f} cm ({height_above_street/2.54:.1f} inches)")
    print(f"  Height above storefront floor (Y=0): {pos[1]:.1f} cm")
    print()

print("="*60)
print("INTERPRETATION:")
print("="*60)
print("If cameras are mounted 20 inches (~51cm) above street level,")
print("we expect camera Y position to be around -66 + 51 = -15 cm")
print("(relative to storefront floor at Y=0)")
