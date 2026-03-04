#!/usr/bin/env python3
"""
Generate Daily Reports from Tracking Data

Reads hourly_stats (and raw tracking_events where available) to produce
JSON daily reports for each day with data. Reports are saved to:
  IO/reports/daily/YYYY-MM-DD.json

These JSON files are designed to be expanded with graphics/charts later.

Usage:
    python generate_reports.py                  # All available days
    python generate_reports.py --days 7         # Last 7 days
    python generate_reports.py --date 2026-02-22  # Specific date
"""

import sqlite3
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path


DB_PATH = Path(__file__).parent / "tracking_history.db"
REPORTS_DIR = Path(__file__).parent / "reports" / "daily"


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_available_dates(conn: sqlite3.Connection) -> list:
    """Get all dates that have hourly_stats or tracking_events data."""
    cursor = conn.cursor()
    
    dates = set()
    
    # Dates with hourly stats (aggregated data, even if raw events pruned)
    cursor.execute("SELECT DISTINCT date FROM hourly_stats ORDER BY date")
    for row in cursor.fetchall():
        dates.add(row['date'])
    
    # Dates with raw events
    cursor.execute("""
        SELECT DISTINCT date(timestamp, 'unixepoch', 'localtime') as day 
        FROM tracking_events
    """)
    for row in cursor.fetchall():
        dates.add(row['day'])
    
    return sorted(dates)


def generate_report_from_hourly_stats(conn: sqlite3.Connection, date_str: str) -> dict:
    """
    Generate a daily report from hourly_stats table.
    This works even when raw tracking_events have been pruned.
    """
    cursor = conn.cursor()
    
    # Get all hourly stats for the date
    cursor.execute("""
        SELECT hour, total_events, unique_people, active_count, passive_count,
               avg_speed, left_to_right, right_to_left, bloom_count,
               dominant_mode, avg_brightness
        FROM hourly_stats 
        WHERE date = ? 
        ORDER BY hour
    """, (date_str,))
    
    hourly_rows = cursor.fetchall()
    
    if not hourly_rows:
        return None
    
    # Build hourly trends
    hourly_trends = []
    total_events = 0
    total_unique = 0
    total_active = 0
    total_passive = 0
    total_ltr = 0
    total_rtl = 0
    total_blooms = 0
    speed_sum = 0.0
    speed_count = 0
    peak_hour = 0
    peak_count = 0
    quietest_hour = 0
    quietest_count = float('inf')
    
    mode_counts = {}
    brightness_sum = 0.0
    brightness_count = 0
    
    for row in hourly_rows:
        hour = row['hour']
        people = row['unique_people'] or 0
        active = row['active_count'] or 0
        passive = row['passive_count'] or 0
        avg_speed = row['avg_speed'] or 0.0
        ltr = row['left_to_right'] or 0
        rtl = row['right_to_left'] or 0
        events = row['total_events'] or 0
        blooms = row['bloom_count'] or 0
        mode = row['dominant_mode'] or 'unknown'
        brightness = row['avg_brightness'] or 0.0
        
        hourly_trends.append({
            'hour': hour,
            'total_events': events,
            'total_people': people,
            'active_count': active,
            'passive_count': passive,
            'avg_speed': round(avg_speed, 1),
            'flow_ltr': ltr,
            'flow_rtl': rtl,
            'bloom_count': blooms,
            'dominant_mode': mode,
            'avg_brightness': round(brightness, 2),
        })
        
        total_events += events
        total_unique += people  # Note: this over-counts across hours
        total_active += active
        total_passive += passive
        total_ltr += ltr
        total_rtl += rtl
        total_blooms += blooms
        
        if avg_speed > 0:
            speed_sum += avg_speed
            speed_count += 1
        
        if brightness > 0:
            brightness_sum += brightness
            brightness_count += 1
        
        # Track mode distribution
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Peak / quietest (by unique people)
        if people > peak_count:
            peak_count = people
            peak_hour = hour
        if people < quietest_count:
            quietest_count = people
            quietest_hour = hour
    
    # Get true unique people for the day from raw events if available
    cursor.execute("""
        SELECT COUNT(DISTINCT person_id) as unique_people
        FROM tracking_events
        WHERE date(timestamp, 'unixepoch', 'localtime') = ?
    """, (date_str,))
    row = cursor.fetchone()
    true_unique = row['unique_people'] if row and row['unique_people'] else total_unique
    
    # Flow analysis
    total_flow = total_ltr + total_rtl
    flow_balance = 0.0
    if total_flow > 0:
        flow_balance = (total_ltr - total_rtl) / total_flow
    
    dominant_flow = 'balanced'
    if flow_balance > 0.3:
        dominant_flow = 'left_to_right'
    elif flow_balance < -0.3:
        dominant_flow = 'right_to_left'
    
    # Mode distribution (normalize)
    total_mode_hours = sum(mode_counts.values())
    mode_distribution = {}
    if total_mode_hours > 0:
        mode_distribution = {k: round(v / total_mode_hours, 3) for k, v in mode_counts.items()}
    
    # Check for auto-tuning data
    tuning_analysis = {}
    try:
        cursor.execute("""
            SELECT total_adjustments, optimal_values_json, strategy_summary
            FROM autotune_daily_learnings
            WHERE date = ?
        """, (date_str,))
        tuning_row = cursor.fetchone()
        if tuning_row:
            tuning_analysis = {
                'total_adjustments': tuning_row['total_adjustments'] or 0,
                'optimal_values': json.loads(tuning_row['optimal_values_json']) if tuning_row['optimal_values_json'] else {},
                'strategy_summary': tuning_row['strategy_summary'] or '',
            }
    except Exception:
        pass
    
    report = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'data_source': 'hourly_stats',
        'summary': {
            'total_unique_people': true_unique,
            'total_events': total_events,
            'total_active_zone_visits': total_active,
            'total_passive_zone_count': total_passive,
            'total_blooms': total_blooms,
            'overall_avg_speed': round(speed_sum / speed_count, 1) if speed_count > 0 else 0.0,
            'avg_brightness': round(brightness_sum / brightness_count, 2) if brightness_count > 0 else 0.0,
            'hours_with_data': len(hourly_rows),
        },
        'peak_times': {
            'peak_hour': peak_hour,
            'peak_hour_count': peak_count,
            'quietest_hour': quietest_hour,
            'quietest_hour_count': int(quietest_count) if quietest_count != float('inf') else 0,
        },
        'flow': {
            'total_left_to_right': total_ltr,
            'total_right_to_left': total_rtl,
            'dominant_flow': dominant_flow,
            'flow_balance': round(flow_balance, 3),
        },
        'hourly_trends': hourly_trends,
        'light_behavior': {
            'mode_distribution': mode_distribution,
        },
        'auto_tuning': tuning_analysis,
    }
    
    return report


def save_report(report: dict, output_dir: Path) -> Path:
    """Save report as formatted JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{report['date']}.json"
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filepath


def persist_to_database(conn: sqlite3.Connection, report: dict):
    """Save report summary to daily_summary and daily_stats_v2 tables."""
    cursor = conn.cursor()
    s = report['summary']
    p = report['peak_times']
    f = report['flow']
    
    # daily_summary table
    cursor.execute("""
        INSERT OR REPLACE INTO daily_summary 
        (date, total_people, active_zone_visits, passive_zone_count,
         avg_speed, dominant_flow, busiest_hour, peak_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report['date'],
        s['total_unique_people'],
        s['total_active_zone_visits'],
        s['total_passive_zone_count'],
        s['overall_avg_speed'],
        f['dominant_flow'],
        p['peak_hour'],
        p['peak_hour_count'],
    ))
    
    # daily_stats_v2 table
    cursor.execute("""
        INSERT OR REPLACE INTO daily_stats_v2
        (date, total_events, unique_people, total_active, total_passive,
         total_blooms, peak_hour, peak_count, quietest_hour,
         dominant_flow, flow_balance, avg_speed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report['date'],
        s['total_events'],
        s['total_unique_people'],
        s['total_active_zone_visits'],
        s['total_passive_zone_count'],
        s.get('total_blooms', 0),
        p['peak_hour'],
        p['peak_hour_count'],
        p['quietest_hour'],
        f['dominant_flow'],
        f['flow_balance'],
        s['overall_avg_speed'],
    ))
    
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Generate daily reports from tracking data')
    parser.add_argument('--days', type=int, help='Generate reports for last N days')
    parser.add_argument('--date', type=str, help='Generate report for specific date (YYYY-MM-DD)')
    parser.add_argument('--db', type=str, default=str(DB_PATH), help='Path to database')
    parser.add_argument('--output', type=str, default=str(REPORTS_DIR), help='Output directory')
    parser.add_argument('--no-persist', action='store_true', help='Skip writing to database tables')
    args = parser.parse_args()
    
    db_path = Path(args.db)
    output_dir = Path(args.output)
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    conn = get_connection(db_path)
    
    # Determine which dates to process
    available = get_available_dates(conn)
    
    if not available:
        print("❌ No tracking data found in database.")
        sys.exit(1)
    
    if args.date:
        dates_to_process = [args.date]
    elif args.days:
        cutoff = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        dates_to_process = [d for d in available if d >= cutoff]
    else:
        dates_to_process = available
    
    print(f"📊 Generating reports for {len(dates_to_process)} day(s)")
    print(f"   Database: {db_path}")
    print(f"   Output:   {output_dir}")
    print()
    
    generated = 0
    for date_str in dates_to_process:
        report = generate_report_from_hourly_stats(conn, date_str)
        
        if report is None:
            print(f"   ⚠️  {date_str}: No data available, skipping")
            continue
        
        # Save JSON file
        filepath = save_report(report, output_dir)
        
        # Persist to database tables
        if not args.no_persist:
            persist_to_database(conn, report)
        
        s = report['summary']
        p = report['peak_times']
        f = report['flow']
        print(f"   ✅ {date_str}: {s['total_unique_people']:,} people, "
              f"{s['total_events']:,} events, "
              f"peak hour {p['peak_hour']}:00 ({p['peak_hour_count']} people), "
              f"flow: {f['dominant_flow']}")
        generated += 1
    
    # Also generate an index file listing all reports
    index = {
        'generated_at': datetime.now().isoformat(),
        'total_reports': generated,
        'database': str(db_path),
        'date_range': {
            'first': dates_to_process[0] if dates_to_process else None,
            'last': dates_to_process[-1] if dates_to_process else None,
        },
        'reports': []
    }
    
    # Read back all report files for the index
    for json_file in sorted(output_dir.glob('*.json')):
        if json_file.name == '_index.json':
            continue
        try:
            with open(json_file) as f:
                r = json.load(f)
            summary = r.get('summary', {})
            peak_times = r.get('peak_times', {})
            index['reports'].append({
                'date': r['date'],
                'file': json_file.name,
                'total_people': summary.get('total_unique_people', summary.get('unique_people', 0)),
                'total_events': summary.get('total_events', summary.get('total_passive_zone_count', 0)),
                'peak_hour': peak_times.get('peak_hour', 0),
                'dominant_flow': r.get('flow', {}).get('dominant_flow', 'unknown'),
            })
        except Exception as e:
            print(f"   ⚠️  Error reading {json_file.name}: {e}")
    
    index_path = output_dir / '_index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print()
    print(f"✅ Generated {generated} report(s)")
    print(f"📁 Reports saved to: {output_dir}/")
    print(f"📋 Index saved to: {index_path}")
    
    conn.close()


if __name__ == '__main__':
    main()
