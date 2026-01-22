#!/usr/bin/env python3
"""Analyze behavior system performance from simulation data."""
import sqlite3
import os

def analyze():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'tracking_history.db')
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("BEHAVIOR SYSTEM ANALYSIS")
    print("=" * 60)
    
    cursor.execute("SELECT COUNT(*) FROM light_behavior")
    total = cursor.fetchone()[0]
    print(f"\nTotal records: {total}")
    
    cursor.execute("SELECT MIN(datetime), MAX(datetime) FROM light_behavior")
    start, end = cursor.fetchone()
    print(f"Time span: {start[:19]} to {end[:19]}")
    
    print("\n--- MODE DISTRIBUTION ---")
    cursor.execute("SELECT mode, COUNT(*) as cnt FROM light_behavior GROUP BY mode ORDER BY cnt DESC")
    for mode, cnt in cursor.fetchall():
        pct = 100.0 * cnt / total
        print(f"  {mode}: {cnt} ({pct:.1f}%)")
    
    print("\n--- GESTURE TYPES ---")
    cursor.execute("SELECT gesture_type, COUNT(*) as cnt FROM light_behavior WHERE gesture_type IS NOT NULL GROUP BY gesture_type ORDER BY cnt DESC")
    gestures = cursor.fetchall()
    if gestures:
        for gesture, cnt in gestures:
            print(f"  {gesture}: {cnt}")
    else:
        print("  No gestures recorded")
    
    print("\n--- PEOPLE COUNT DISTRIBUTION ---")
    cursor.execute("SELECT people_count, COUNT(*) as cnt FROM light_behavior GROUP BY people_count ORDER BY people_count")
    for people, cnt in cursor.fetchall():
        pct = 100.0 * cnt / total
        print(f"  {people} people: {cnt} ({pct:.1f}%)")
    
    print("\n--- ACTIVE vs PASSIVE (when people present) ---")
    cursor.execute("SELECT AVG(active_count), AVG(passive_count), MAX(active_count), MAX(passive_count) FROM light_behavior WHERE people_count > 0")
    row = cursor.fetchone()
    if row[0]:
        print(f"  Avg active: {row[0]:.2f}, Avg passive: {row[1]:.2f}")
        print(f"  Max active: {row[2]}, Max passive: {row[3]}")
    
    print("\n--- BRIGHTNESS STATS ---")
    cursor.execute("SELECT MIN(brightness), MAX(brightness), AVG(brightness) FROM light_behavior")
    min_b, max_b, avg_b = cursor.fetchone()
    print(f"  Min: {min_b:.1f}, Max: {max_b:.1f}, Avg: {avg_b:.1f}")
    
    print("\n--- LIGHT POSITION RANGES ---")
    cursor.execute("SELECT MIN(position_x), MAX(position_x), AVG(position_x) FROM light_behavior")
    r = cursor.fetchone()
    print(f"  X: {r[0]:.0f} to {r[1]:.0f} (avg: {r[2]:.0f})")
    cursor.execute("SELECT MIN(position_y), MAX(position_y), AVG(position_y) FROM light_behavior")
    r = cursor.fetchone()
    print(f"  Y: {r[0]:.0f} to {r[1]:.0f} (avg: {r[2]:.0f})")
    cursor.execute("SELECT MIN(position_z), MAX(position_z), AVG(position_z) FROM light_behavior")
    r = cursor.fetchone()
    print(f"  Z: {r[0]:.0f} to {r[1]:.0f} (avg: {r[2]:.0f})")
    
    print("\n--- MODE TRANSITIONS ---")
    cursor.execute("SELECT mode, LAG(mode) OVER (ORDER BY timestamp) as prev_mode FROM light_behavior")
    transitions = {}
    for mode, prev_mode in cursor.fetchall():
        if prev_mode and prev_mode != mode:
            key = f"{prev_mode} -> {mode}"
            transitions[key] = transitions.get(key, 0) + 1
    for trans, cnt in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"  {trans}: {cnt}")
    
    print("\n--- SPEED STATS ---")
    cursor.execute("SELECT MIN(move_speed), MAX(move_speed), AVG(move_speed) FROM light_behavior")
    r = cursor.fetchone()
    print(f"  Move speed: {r[0]:.0f} to {r[1]:.0f} (avg: {r[2]:.0f})")
    cursor.execute("SELECT MIN(pulse_speed), MAX(pulse_speed), AVG(pulse_speed) FROM light_behavior")
    r = cursor.fetchone()
    print(f"  Pulse speed: {r[0]:.0f} to {r[1]:.0f} (avg: {r[2]:.0f})")
    
    print("\n--- STATUS TEXT SAMPLES ---")
    cursor.execute("SELECT DISTINCT status_text FROM light_behavior LIMIT 15")
    for row in cursor.fetchall():
        if row[0]:
            print(f"  * {row[0][:70]}")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze()
