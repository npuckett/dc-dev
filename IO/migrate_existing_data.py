#!/usr/bin/env python3
"""
One-time migration script to aggregate existing raw data into hourly_stats.

Run this ONCE to backfill historical data before the new pruning takes effect.

Usage:
    python migrate_existing_data.py [path_to_database]

If no path provided, uses default: tracking_history.db
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking_database import TrackingDatabase


def migrate_existing_data(db_path: str = "tracking_history.db"):
    """Aggregate all existing raw data into hourly_stats table."""
    
    print(f"Opening database: {db_path}")
    db = TrackingDatabase(db_path)
    
    # Get date range of existing data
    cursor = db.conn.cursor()
    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM tracking_events')
    row = cursor.fetchone()
    
    if not row or not row[0]:
        print("No existing data to migrate")
        return
    
    start_dt = datetime.fromtimestamp(row[0])
    end_dt = datetime.fromtimestamp(row[1])
    
    print(f"Found data from {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
    
    # Count total events
    cursor.execute('SELECT COUNT(*) FROM tracking_events')
    total_events = cursor.fetchone()[0]
    print(f"Total raw events: {total_events:,}")
    
    # Check how many hours already aggregated
    cursor.execute('SELECT COUNT(*) FROM hourly_stats')
    existing_hours = cursor.fetchone()[0]
    print(f"Already aggregated hours: {existing_hours}")
    
    # Calculate expected hours
    hours_span = int((end_dt - start_dt).total_seconds() / 3600) + 1
    print(f"Expected hours to process: {hours_span}")
    
    # Confirm before proceeding
    response = input("\nProceed with migration? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Iterate through each hour
    current = start_dt.replace(minute=0, second=0, microsecond=0)
    hours_processed = 0
    hours_with_data = 0
    
    print("\nMigrating...")
    
    while current <= end_dt:
        date_str = current.strftime('%Y-%m-%d')
        hour = current.hour
        
        # Check if already exists
        cursor.execute(
            'SELECT 1 FROM hourly_stats WHERE date = ? AND hour = ?',
            (date_str, hour)
        )
        if cursor.fetchone():
            current += timedelta(hours=1)
            continue
        
        try:
            stats = db.aggregate_hour(date_str, hour)
            hours_processed += 1
            
            if stats['total_events'] > 0:
                hours_with_data += 1
                if hours_with_data % 10 == 0:
                    print(f"  Processed {hours_with_data} hours with data...")
                    
        except Exception as e:
            print(f"  Error aggregating {date_str} hour {hour}: {e}")
        
        current += timedelta(hours=1)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Migration complete!")
    print(f"  Hours processed: {hours_processed}")
    print(f"  Hours with data: {hours_with_data}")
    
    # Verify
    cursor.execute('SELECT COUNT(*) FROM hourly_stats')
    total_hourly = cursor.fetchone()[0]
    print(f"  Total hourly_stats rows: {total_hourly}")
    
    # Show sample
    print(f"\nSample hourly_stats:")
    cursor.execute('''
        SELECT date, hour, unique_people, active_count, passive_count 
        FROM hourly_stats 
        ORDER BY date DESC, hour DESC 
        LIMIT 5
    ''')
    for row in cursor.fetchall():
        print(f"  {row[0]} {row[1]:02d}:00 - {row[2]} people, {row[3]} active, {row[4]} passive")
    
    db.close()
    print("\nDone!")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "tracking_history.db"
    migrate_existing_data(db_path)
