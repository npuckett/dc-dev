#!/usr/bin/env python3
"""
OSC Receiver Test Script
Monitors and displays OSC messages from the person tracker

Usage:
    python test_osc_receiver.py

Press Ctrl+C to stop
"""

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import sys

def tracking_count_handler(address, *args):
    """Handle /tracking/count messages"""
    print(f"📊 People tracked: {args[0]}")

def person_position_handler(address, *args):
    """Handle /person/{id}/x and /person/{id}/y messages"""
    # Extract person ID from address (e.g., /person/1/x -> ID=1)
    parts = address.split('/')
    person_id = parts[2]
    coord_type = parts[3]
    value = args[0]
    
    # Only print complete coordinates (when we get 'y' after 'x')
    if coord_type == 'y':
        print(f"  👤 Person {person_id}: position updated")

def person_normalized_handler(address, *args):
    """Handle /person/{id}/norm_x and /person/{id}/norm_y messages"""
    parts = address.split('/')
    person_id = parts[2]
    coord_type = parts[3]
    value = args[0]
    
    if coord_type == 'norm_y':
        print(f"     Normalized position: {value:.3f}")

def person_time_handler(address, *args):
    """Handle /person/{id}/time messages"""
    parts = address.split('/')
    person_id = parts[2]
    time_in_frame = args[0]
    print(f"     Time in frame: {time_in_frame:.1f}s")

def person_ended_handler(address, *args):
    """Handle /person/{id}/ended messages"""
    parts = address.split('/')
    person_id = parts[2]
    print(f"🚪 Person {person_id} left the frame")

def person_data_handler(address, *args):
    """Handle /person/data bundled messages"""
    if len(args) >= 6:
        person_id, x, y, norm_x, norm_y, time_val = args[0:6]
        print(f"📦 Bundled data - ID:{person_id} Pos:({x},{y}) Norm:({norm_x:.2f},{norm_y:.2f}) Time:{time_val:.1f}s")

def default_handler(address, *args):
    """Handle any unmatched messages"""
    # Uncomment to see all messages:
    # print(f"🔔 {address}: {args}")
    pass

def main():
    print("=" * 60)
    print("OSC Receiver Test - Person Tracker Monitor")
    print("=" * 60)
    print()
    print("Listening for OSC messages on 127.0.0.1:8000")
    print()
    print("Expected messages:")
    print("  /tracking/count         - Number of people tracked")
    print("  /person/{id}/x          - Pixel X coordinate")
    print("  /person/{id}/y          - Pixel Y coordinate")
    print("  /person/{id}/norm_x     - Normalized X (0.0-1.0)")
    print("  /person/{id}/norm_y     - Normalized Y (0.0-1.0)")
    print("  /person/{id}/time       - Seconds in frame")
    print("  /person/{id}/ended      - Person left frame")
    print("  /person/data            - Bundled data array")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    # Create dispatcher and map message handlers
    dispatcher = Dispatcher()
    dispatcher.map("/tracking/count", tracking_count_handler)
    dispatcher.map("/person/*/x", person_position_handler)
    dispatcher.map("/person/*/y", person_position_handler)
    dispatcher.map("/person/*/norm_x", person_normalized_handler)
    dispatcher.map("/person/*/norm_y", person_normalized_handler)
    dispatcher.map("/person/*/time", person_time_handler)
    dispatcher.map("/person/*/ended", person_ended_handler)
    dispatcher.map("/person/data", person_data_handler)
    dispatcher.set_default_handler(default_handler)
    
    # Create and start server
    try:
        server = BlockingOSCUDPServer(("127.0.0.1", 8000), dispatcher)
        print("✅ Server started successfully")
        print("🎬 Waiting for messages from tracker...\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
        print("=" * 60)
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            print("\n❌ ERROR: Port 8000 already in use!")
            print("\nSolution:")
            print("  1. Check if tracker is already running")
            print("  2. Or kill process using port 8000:")
            print("     lsof -i :8000")
            print("     kill -9 <PID>")
            print("\n  3. Or change port in config.json:")
            print("     'port': 8001")
        else:
            print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
