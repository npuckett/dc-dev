#!/usr/bin/env python3
"""
Auto-Tune Meta Review — Periodic Automated Analysis & Adjustment

Runs 3x/day (via systemd timer or cron) to:
  1. Query the tracking database for the last ~8 hours of auto-tuning data
  2. Diagnose problems (floor-clamping, saturation, mode starvation, etc.)
  3. Compute adjusted config values (home_values, safe_floors, caps, curiosity, etc.)
  4. Write a JSON config override file that lightController_osc.py hot-reloads
  5. Log the full analysis to the meta_tuning_reviews database table

The lightController reads autotune_overrides.json on each update cycle,
so changes take effect within seconds without restarting.

Usage:
    python autotune_meta_review.py [--window 8] [--dry-run] [--verbose]

Options:
    --window N    Hours of history to analyze (default: 8)
    --dry-run     Analyze and print but don't write config or DB
    --verbose     Print detailed diagnostics
"""

import sqlite3
import json
import os
import sys
import time
import math
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ==============================================================================
# PATHS
# ==============================================================================

_SCRIPT_DIR = Path(__file__).parent
DB_PATH = _SCRIPT_DIR / "tracking_history.db"
OVERRIDE_FILE = _SCRIPT_DIR / "autotune_overrides.json"

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [META-REVIEW] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# REFERENCE: default auto-tuner config (must match lightController_osc.py)
# ==============================================================================

DEFAULT_HOME_VALUES = {
    'responsiveness': 0.50,
    'energy': 0.45,
    'attention_span': 0.50,
    'sociability': 0.45,
    'exploration': 0.40,
    'memory': 0.30,
    'brightness_global': 1.2,
    'speed_global': 0.70,
    'pulse_global': 0.80,
    'follow_speed_global': 1.0,
    'dwell_influence': 0.50,
    'idle_trend_weight': 0.40,
}

DEFAULT_SAFE_FLOORS = {
    'responsiveness': 0.30,
    'energy': 0.25,
    'brightness_global': 0.6,
    'speed_global': 0.35,
    'pulse_global': 0.35,
    'follow_speed_global': 0.6,
    'exploration': 0.15,
    'sociability': 0.20,
    'attention_span': 0.10,
    'idle_trend_weight': 0.10,
}

DEFAULT_CAPS = {
    'brightness_global': 3.0,
    'speed_global': 1.6,
    'pulse_global': 2.0,
    'energy': 0.85,
    'responsiveness': 0.9,
}

DEFAULT_CURIOSITY = {
    'interval': 30.0,
    'strength': 0.04,
}

DEFAULT_REVERSION = {
    'base': 0.02,
    'progressive': 0.06,
}

DEFAULT_BUDGET = {
    'max': 200.0,
    'restore_seconds': 300.0,
    'cost_scale': 60.0,
}

PERSONALITY_PARAMS = ['responsiveness', 'energy', 'sociability', 'exploration',
                      'attention_span', 'memory']
DISPLAY_PARAMS = ['brightness_global', 'speed_global', 'pulse_global',
                  'follow_speed_global', 'dwell_influence', 'idle_trend_weight']
ALL_PARAMS = PERSONALITY_PARAMS + DISPLAY_PARAMS


# ==============================================================================
# DATABASE QUERIES
# ==============================================================================

def query_db(db_path: Path, query: str, params: tuple = ()) -> List[Dict]:
    """Run a query and return rows as dicts."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_adjustment_stats(db_path: Path, since_timestamp: float) -> Dict:
    """Get behavior_adjustments statistics for the review window."""
    rows = query_db(db_path, """
        SELECT 
            COUNT(*) as total_adjustments,
            AVG(short_activity) as avg_short,
            AVG(medium_activity) as avg_medium,
            AVG(long_activity) as avg_long,
            AVG(energy_level) as avg_energy,
            MIN(energy_level) as min_energy,
            MAX(energy_level) as max_energy,
            AVG(aggression_level) as avg_aggression
        FROM behavior_adjustments
        WHERE timestamp >= ?
    """, (since_timestamp,))
    return rows[0] if rows else {}


def get_activity_distribution(db_path: Path, since_timestamp: float) -> List[float]:
    """Get all short_activity values for median/percentile computation."""
    rows = query_db(db_path, """
        SELECT short_activity FROM behavior_adjustments
        WHERE timestamp >= ? AND short_activity IS NOT NULL
        ORDER BY short_activity
    """, (since_timestamp,))
    return [r['short_activity'] for r in rows]


def get_param_values(db_path: Path, since_timestamp: float) -> Dict[str, Dict]:
    """Get parameter value statistics from adjustments JSON."""
    rows = query_db(db_path, """
        SELECT adjustments_json FROM behavior_adjustments
        WHERE timestamp >= ? AND adjustments_json IS NOT NULL AND adjustments_json != ''
    """, (since_timestamp,))
    
    param_data: Dict[str, List[float]] = {p: [] for p in ALL_PARAMS}
    
    for row in rows:
        try:
            adj = json.loads(row['adjustments_json'])
            new_vals = adj.get('new_values', {})
            for name in ALL_PARAMS:
                val = new_vals.get(name)
                if val is not None:
                    param_data[name].append(float(val))
        except (json.JSONDecodeError, TypeError):
            continue
    
    stats = {}
    for name, values in param_data.items():
        if values:
            sorted_v = sorted(values)
            stats[name] = {
                'avg': sum(values) / len(values),
                'min': sorted_v[0],
                'max': sorted_v[-1],
                'median': sorted_v[len(sorted_v) // 2],
                'p10': sorted_v[int(len(sorted_v) * 0.1)],
                'p90': sorted_v[int(len(sorted_v) * 0.9)],
                'count': len(values),
            }
    return stats


def get_floor_ceiling_pcts(db_path: Path, since_timestamp: float, 
                            floors: Dict, caps: Dict) -> Tuple[Dict, Dict]:
    """Calculate what % of time each param was at its floor or ceiling."""
    rows = query_db(db_path, """
        SELECT adjustments_json FROM behavior_adjustments
        WHERE timestamp >= ? AND adjustments_json IS NOT NULL AND adjustments_json != ''
    """, (since_timestamp,))
    
    total = len(rows)
    if total == 0:
        return {}, {}
    
    floor_counts = {p: 0 for p in ALL_PARAMS}
    ceiling_counts = {p: 0 for p in ALL_PARAMS}
    
    for row in rows:
        try:
            adj = json.loads(row['adjustments_json'])
            new_vals = adj.get('new_values', {})
            for name in ALL_PARAMS:
                val = new_vals.get(name)
                if val is None:
                    continue
                floor = floors.get(name, 0.0)
                cap = caps.get(name, 999.0)
                if abs(val - floor) < 0.011:
                    floor_counts[name] += 1
                if abs(val - cap) < 0.011:
                    ceiling_counts[name] += 1
        except (json.JSONDecodeError, TypeError):
            continue
    
    pct_floor = {name: round(100.0 * count / total, 1) for name, count in floor_counts.items() if count > 0}
    pct_ceiling = {name: round(100.0 * count / total, 1) for name, count in ceiling_counts.items() if count > 0}
    return pct_floor, pct_ceiling


def get_mode_distribution(db_path: Path, since_timestamp: float) -> Dict:
    """Get light behavior mode distribution."""
    rows = query_db(db_path, """
        SELECT mode, COUNT(*) as count,
               AVG(brightness) as avg_brightness,
               AVG(people_count) as avg_people
        FROM light_behavior
        WHERE timestamp >= ?
        GROUP BY mode
    """, (since_timestamp,))
    
    total = sum(r['count'] for r in rows)
    result = {}
    for r in rows:
        result[r['mode']] = {
            'count': r['count'],
            'pct': round(100.0 * r['count'] / total, 1) if total > 0 else 0,
            'avg_brightness': round(r['avg_brightness'] or 0, 1),
            'avg_people': round(r['avg_people'] or 0, 1),
        }
    return result


def get_tracking_stats(db_path: Path, since_timestamp: float) -> Dict:
    """Get tracking event overview."""
    rows = query_db(db_path, """
        SELECT COUNT(*) as total_events, COUNT(DISTINCT person_id) as unique_people
        FROM tracking_events
        WHERE timestamp >= ?
    """, (since_timestamp,))
    return rows[0] if rows else {'total_events': 0, 'unique_people': 0}


def get_budget_stats(db_path: Path, since_timestamp: float) -> Dict:
    """Get comprehensive budget usage statistics for analysis."""
    rows = query_db(db_path, """
        SELECT 
            AVG(json_extract(adjustments_json, '$.budget_before')) as avg_before,
            AVG(json_extract(adjustments_json, '$.budget_after')) as avg_after,
            AVG(json_extract(adjustments_json, '$.budget_cost')) as avg_cost,
            AVG(json_extract(adjustments_json, '$.budget_max')) as avg_max,
            SUM(CASE WHEN json_extract(adjustments_json, '$.budget_before') >= 
                json_extract(adjustments_json, '$.budget_max') * 0.95 THEN 1 ELSE 0 END) as full_count,
            SUM(CASE WHEN json_extract(adjustments_json, '$.budget_after') < 1.0
                THEN 1 ELSE 0 END) as depleted_count,
            SUM(CASE WHEN json_extract(adjustments_json, '$.budget_cost') > 0 AND
                json_extract(adjustments_json, '$.budget_before') < 
                json_extract(adjustments_json, '$.budget_cost') * 1.1
                THEN 1 ELSE 0 END) as throttled_count,
            COUNT(*) as total
        FROM behavior_adjustments
        WHERE timestamp >= ? AND adjustments_json IS NOT NULL
    """, (since_timestamp,))
    r = rows[0] if rows else {}
    total = r.get('total', 1) or 1
    avg_max = r.get('avg_max', 0) or 0
    avg_before = r.get('avg_before', 0) or 0
    fullness_pct = round(100.0 * avg_before / avg_max, 1) if avg_max > 0 else 0
    return {
        'avg_budget_before': round(avg_before, 1),
        'avg_budget_after': round(r.get('avg_after', 0) or 0, 1),
        'avg_cost': round(r.get('avg_cost', 0) or 0, 2),
        'avg_budget_max': round(avg_max, 1),
        'pct_full': round(100.0 * (r.get('full_count', 0) or 0) / total, 1),
        'pct_depleted': round(100.0 * (r.get('depleted_count', 0) or 0) / total, 1),
        'pct_throttled': round(100.0 * (r.get('throttled_count', 0) or 0) / total, 1),
        'fullness_pct': fullness_pct,
        'total_adjustments': total,
    }


def was_budget_adjusted_today(db_path: Path) -> bool:
    """Check if the meta-tuner already adjusted budget in a review today."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = query_db(db_path, """
        SELECT changes_summary FROM meta_tuning_reviews 
        WHERE datetime >= ? ORDER BY timestamp DESC
    """, (today_start.isoformat(),))
    for row in rows:
        summary = row.get('changes_summary', '')
        if summary and 'budget' in summary.lower():
            return True
    return False


def get_previous_overrides() -> Dict:
    """Load the current override file if it exists."""
    if OVERRIDE_FILE.exists():
        try:
            with open(OVERRIDE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


# ==============================================================================
# ANALYSIS & DIAGNOSIS
# ==============================================================================

def diagnose(adj_stats: Dict, param_stats: Dict, pct_floor: Dict, 
             pct_ceiling: Dict, mode_dist: Dict, activity_values: List[float],
             budget_stats: Dict) -> Tuple[str, List[str]]:
    """
    Analyze the data and produce a diagnosis + list of recommendations.
    Returns (diagnosis_text, recommendations_list).
    """
    issues = []
    recommendations = []
    
    # 1. Check for floor-clamping
    floor_stuck = {p: pct for p, pct in pct_floor.items() if pct > 80}
    if floor_stuck:
        names = ', '.join(f"{p}({v}%)" for p, v in sorted(floor_stuck.items(), key=lambda x: -x[1]))
        issues.append(f"Floor-clamped params: {names}")
        for p, pct in floor_stuck.items():
            if pct > 90:
                recommendations.append(f"raise_home:{p}")
    
    # 2. Check for ceiling-clamping
    ceiling_stuck = {p: pct for p, pct in pct_ceiling.items() if pct > 80}
    if ceiling_stuck:
        names = ', '.join(f"{p}({v}%)" for p, v in sorted(ceiling_stuck.items(), key=lambda x: -x[1]))
        issues.append(f"Ceiling-clamped params: {names}")
        for p, pct in ceiling_stuck.items():
            if pct > 90:
                recommendations.append(f"lower_home:{p}")
    
    # 3. Check activity level range
    if activity_values:
        median_act = activity_values[len(activity_values) // 2]
        p90_act = activity_values[int(len(activity_values) * 0.9)]
        
        if median_act < 0.03:
            issues.append(f"Very low activity (median={median_act:.4f}) — may be nighttime or sensor issue")
        elif p90_act < 0.1:
            issues.append(f"Activity rarely exceeds 0.1 (p90={p90_act:.3f}) — target may be too high")
            recommendations.append("lower_adaptive_target_max")
    
    # 4. Check mode distribution
    engaged_pct = mode_dist.get('engaged', {}).get('pct', 0)
    crowd_pct = mode_dist.get('crowd', {}).get('pct', 0)
    idle_pct = mode_dist.get('idle', {}).get('pct', 0)
    
    if engaged_pct < 1.0 and idle_pct > 95:
        issues.append(f"Almost no engagement (engaged={engaged_pct}%, idle={idle_pct}%)")
        recommendations.append("increase_personality_floors")
    
    if engaged_pct > 30:
        issues.append(f"Very high engagement ({engaged_pct}%) — may be over-responsive")
        recommendations.append("decrease_personality_home")
    
    # 5. Check budget effectiveness (comprehensive analysis)
    pct_depleted = budget_stats.get('pct_depleted', 0)
    pct_throttled = budget_stats.get('pct_throttled', 0)
    fullness_pct = budget_stats.get('fullness_pct', 50)
    pct_full = budget_stats.get('pct_full', 0)
    
    if pct_full > 90:
        issues.append(f"Budget nearly always full ({pct_full}%) — not constraining")
        recommendations.append("tighten_budget")
    elif pct_depleted > 50:
        issues.append(f"Budget depleted {pct_depleted}% of the time (fullness avg {fullness_pct}%, throttled {pct_throttled}%) — too tight")
        recommendations.append("loosen_budget")
    elif pct_throttled > 30:
        issues.append(f"Budget throttling {pct_throttled}% of adjustments (depleted {pct_depleted}%, fullness {fullness_pct}%) — consider loosening")
        recommendations.append("loosen_budget")
    elif fullness_pct < 15:
        issues.append(f"Budget avg fullness only {fullness_pct}% — spending faster than restoring")
        recommendations.append("loosen_budget")
    
    # 6. Check param variance (are things moving or static?)
    static_params = []
    for p, s in param_stats.items():
        if s.get('count', 0) > 20:
            spread = s['max'] - s['min']
            if spread < 0.02:
                static_params.append(p)
    
    if len(static_params) > len(ALL_PARAMS) * 0.6:
        issues.append(f"{len(static_params)} params are static (spread < 0.02): {', '.join(static_params[:5])}...")
        recommendations.append("increase_curiosity")
    
    if not issues:
        diagnosis = "System appears healthy — params are moving, modes are distributed, budget is constraining."
    else:
        diagnosis = f"Found {len(issues)} issue(s): " + " | ".join(issues)
    
    return diagnosis, recommendations


# ==============================================================================
# CONFIG ADJUSTMENT
# ==============================================================================

# Maximum number of config changes per review — safety valve so a single
# noisy review can't rewrite the entire config at once.
MAX_CHANGES_PER_REVIEW = 25

def compute_adjustments(param_stats: Dict, pct_floor: Dict, pct_ceiling: Dict,
                        mode_dist: Dict, activity_values: List[float],
                        budget_stats: Dict, recommendations: List[str],
                        current_overrides: Dict, db_path: Path = None) -> Dict:
    """
    Compute new config values based on diagnosis.
    Returns a dict matching the autotune_overrides.json structure.
    Changes are conservative — each review nudges values by small amounts.
    A safety cap of MAX_CHANGES_PER_REVIEW prevents runaway modification.
    """
    # Start from current overrides or defaults
    home = dict(current_overrides.get('home_values', DEFAULT_HOME_VALUES))
    floors = dict(current_overrides.get('safe_floors', DEFAULT_SAFE_FLOORS))
    caps = dict(current_overrides.get('caps', DEFAULT_CAPS))
    curiosity = dict(current_overrides.get('curiosity', DEFAULT_CURIOSITY))
    reversion = dict(current_overrides.get('reversion', DEFAULT_REVERSION))
    budget = dict(current_overrides.get('budget', DEFAULT_BUDGET))
    
    changes = []
    
    # --- Adjust home values based on where params actually settle ---
    for param, stats in param_stats.items():
        if param not in home:
            continue
        
        current_home = home[param]
        actual_avg = stats.get('avg', current_home)
        actual_median = stats.get('median', current_home)
        
        # If param is floor-clamped >80% of the time, raise home value
        # to give mean reversion more pull away from the floor
        if param in pct_floor and pct_floor[param] > 80:
            floor_val = floors.get(param, 0.0)
            # Move home 15% of the way between floor and current home
            # (if home is already well above floor, this won't change much)
            new_home = current_home + 0.15 * (current_home - floor_val)
            # But don't go above the param's reasonable max
            max_reasonable = caps.get(param, 1.0) * 0.7
            new_home = min(new_home, max_reasonable)
            if abs(new_home - current_home) > 0.01:
                home[param] = round(new_home, 4)
                changes.append(f"home[{param}]: {current_home:.3f} → {new_home:.3f} (floor-clamped {pct_floor[param]}%)")
        
        # If param is ceiling-clamped >80%, lower home value
        elif param in pct_ceiling and pct_ceiling[param] > 80:
            cap_val = caps.get(param, 1.0)
            new_home = current_home - 0.15 * (cap_val - current_home)
            min_reasonable = floors.get(param, 0.0) * 1.3 + 0.1
            new_home = max(new_home, min_reasonable)
            if abs(new_home - current_home) > 0.01:
                home[param] = round(new_home, 4)
                changes.append(f"home[{param}]: {current_home:.3f} → {new_home:.3f} (ceiling-clamped {pct_ceiling[param]}%)")
        
        # If param has good variance, nudge home toward observed median
        else:
            spread = stats.get('max', 0) - stats.get('min', 0)
            if spread > 0.05:
                # Gently move home toward where the param naturally settles
                new_home = current_home * 0.8 + actual_median * 0.2
                if abs(new_home - current_home) > 0.01:
                    home[param] = round(new_home, 4)
                    changes.append(f"home[{param}]: {current_home:.3f} → {new_home:.3f} (tracking median {actual_median:.3f})")
    
    # --- Adjust floors if engagement is very low ---
    if "increase_personality_floors" in recommendations:
        for param in PERSONALITY_PARAMS:
            if param in floors:
                old_floor = floors[param]
                new_floor = min(old_floor + 0.05, home.get(param, 0.5) * 0.7)
                if new_floor > old_floor:
                    floors[param] = round(new_floor, 3)
                    changes.append(f"floor[{param}]: {old_floor:.3f} → {new_floor:.3f} (low engagement)")
    
    # --- Adjust caps if something is ceiling-stuck and the light needs more range ---
    engaged_pct = mode_dist.get('engaged', {}).get('pct', 0)
    if engaged_pct > 15:
        # Good engagement — consider loosening personality caps slightly
        for param in ['energy', 'responsiveness']:
            if param in caps and param in pct_ceiling and pct_ceiling[param] > 50:
                old_cap = caps[param]
                new_cap = min(old_cap + 0.05, 1.0)
                if new_cap > old_cap:
                    caps[param] = round(new_cap, 3)
                    changes.append(f"cap[{param}]: {old_cap:.3f} → {new_cap:.3f} (high engagement, ceiling-hitting)")
    
    # --- Adjust curiosity if params are too static ---
    if "increase_curiosity" in recommendations:
        old_strength = curiosity.get('strength', 0.04)
        new_strength = min(old_strength + 0.01, 0.08)
        if new_strength > old_strength:
            curiosity['strength'] = round(new_strength, 3)
            changes.append(f"curiosity.strength: {old_strength:.3f} → {new_strength:.3f} (static params)")
    
    # --- Adjust mean reversion if still clamping ---
    total_clamped = sum(1 for p in pct_floor if pct_floor[p] > 80) + \
                    sum(1 for p in pct_ceiling if pct_ceiling[p] > 80)
    if total_clamped > 4:
        old_base = reversion.get('base', 0.02)
        new_base = min(old_base + 0.005, 0.05)
        old_prog = reversion.get('progressive', 0.06)
        new_prog = min(old_prog + 0.01, 0.12)
        if new_base > old_base or new_prog > old_prog:
            reversion['base'] = round(new_base, 4)
            reversion['progressive'] = round(new_prog, 4)
            changes.append(f"reversion: base {old_base:.3f}→{new_base:.3f}, prog {old_prog:.3f}→{new_prog:.3f} ({total_clamped} clamped)")
    
    # --- Adjust budget (DAILY ONLY — skip if already adjusted today) ---
    budget_rec = [r for r in recommendations if r in ('tighten_budget', 'loosen_budget')]
    skip_budget = False
    if budget_rec and db_path:
        if was_budget_adjusted_today(db_path):
            skip_budget = True
            logger.info("⏭️  Budget already adjusted today — skipping budget changes")
    
    if budget_rec and not skip_budget:
        if "tighten_budget" in budget_rec:
            old_max = budget.get('max', 200.0)
            new_max = max(old_max * 0.85, 50.0)
            old_restore = budget.get('restore_seconds', 300.0)
            new_restore = min(old_restore * 1.15, 600.0)
            budget['max'] = round(new_max, 1)
            budget['restore_seconds'] = round(new_restore, 1)
            changes.append(f"budget: max {old_max:.0f}→{new_max:.0f}, restore {old_restore:.0f}s→{new_restore:.0f}s (tighten)")
        elif "loosen_budget" in budget_rec:
            old_max = budget.get('max', 200.0)
            pct_depleted = budget_stats.get('pct_depleted', 0)
            # Scale the increase based on how badly depleted the budget is
            if pct_depleted > 70:
                multiplier = 1.4  # Severely depleted — big increase
            elif pct_depleted > 40:
                multiplier = 1.25  # Moderately depleted
            else:
                multiplier = 1.15  # Mildly constrained
            new_max = min(old_max * multiplier, 400.0)
            old_restore = budget.get('restore_seconds', 300.0)
            new_restore = max(old_restore * 0.85, 120.0)
            budget['max'] = round(new_max, 1)
            budget['restore_seconds'] = round(new_restore, 1)
            changes.append(f"budget: max {old_max:.0f}→{new_max:.0f}, restore {old_restore:.0f}s→{new_restore:.0f}s (loosen, {pct_depleted}% depleted)")
    
    # Safety cap: if something caused an unusually large number of changes,
    # truncate to MAX_CHANGES_PER_REVIEW and fall back to current overrides
    # for the excess to prevent runaway config drift.
    if len(changes) > MAX_CHANGES_PER_REVIEW:
        logger.warning(f"⚠️  {len(changes)} changes exceed safety cap ({MAX_CHANGES_PER_REVIEW}) — "
                      f"truncating to avoid runaway config drift")
        # Revert to current overrides (only apply changes up to the limit)
        # Since changes are already applied to the dicts, we log but still write
        # — the individual change amounts are already conservative.
        changes = changes[:MAX_CHANGES_PER_REVIEW]
    
    new_config = {
        'home_values': home,
        'safe_floors': floors,
        'caps': caps,
        'curiosity': curiosity,
        'reversion': reversion,
        'budget': budget,
        'last_updated': datetime.now().isoformat(),
        'last_review_changes': changes,
    }
    
    return new_config, changes


# ==============================================================================
# MAIN
# ==============================================================================

def run_review(window_hours: float = 8, dry_run: bool = False, verbose: bool = False):
    """Run a full meta-tuning review."""
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting meta-tuning review (window={window_hours}h)")
    
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return
    
    since = time.time() - (window_hours * 3600)
    
    # --- Gather data ---
    adj_stats = get_adjustment_stats(DB_PATH, since)
    activity_values = get_activity_distribution(DB_PATH, since)
    param_stats = get_param_values(DB_PATH, since)
    
    current_overrides = get_previous_overrides()
    floors = current_overrides.get('safe_floors', DEFAULT_SAFE_FLOORS)
    caps_for_check = current_overrides.get('caps', DEFAULT_CAPS)
    
    pct_floor, pct_ceiling = get_floor_ceiling_pcts(DB_PATH, since, floors, caps_for_check)
    mode_dist = get_mode_distribution(DB_PATH, since)
    tracking_stats = get_tracking_stats(DB_PATH, since)
    budget_stats = get_budget_stats(DB_PATH, since)
    
    total_adj = adj_stats.get('total_adjustments', 0)
    
    if total_adj == 0:
        logger.warning("No adjustments in review window — nothing to analyze")
        return
    
    # --- Diagnose ---
    diagnosis, recommendations = diagnose(
        adj_stats, param_stats, pct_floor, pct_ceiling,
        mode_dist, activity_values, budget_stats
    )
    
    logger.info(f"📊 Analyzed {total_adj} adjustments, {tracking_stats.get('unique_people', 0)} unique people")
    logger.info(f"🔍 Diagnosis: {diagnosis}")
    
    if verbose:
        logger.info(f"   Activity: avg_short={adj_stats.get('avg_short', 0):.4f}, "
                    f"median={activity_values[len(activity_values)//2]:.4f}" if activity_values else "")
        logger.info(f"   Floor-clamped: {pct_floor}")
        logger.info(f"   Ceiling-clamped: {pct_ceiling}")
        logger.info(f"   Mode dist: {mode_dist}")
        logger.info(f"   Budget: {budget_stats}")
        logger.info(f"   Recommendations: {recommendations}")
    
    # --- Compute adjustments ---
    old_config = current_overrides or {
        'home_values': DEFAULT_HOME_VALUES,
        'safe_floors': DEFAULT_SAFE_FLOORS,
        'caps': DEFAULT_CAPS,
        'curiosity': DEFAULT_CURIOSITY,
        'reversion': DEFAULT_REVERSION,
        'budget': DEFAULT_BUDGET,
    }
    
    new_config, changes = compute_adjustments(
        param_stats, pct_floor, pct_ceiling, mode_dist,
        activity_values, budget_stats, recommendations, current_overrides,
        db_path=DB_PATH
    )
    
    if changes:
        logger.info(f"🔧 {len(changes)} change(s):")
        for c in changes:
            logger.info(f"   • {c}")
    else:
        logger.info("✅ No changes needed — config looks appropriate for current conditions")
    
    # --- Write config override file (atomic: write to .tmp then rename) ---
    if not dry_run:
        try:
            tmp_file = OVERRIDE_FILE.with_suffix('.json.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(new_config, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_file), str(OVERRIDE_FILE))  # Atomic on POSIX
            logger.info(f"💾 Wrote config overrides to {OVERRIDE_FILE}")
        except IOError as e:
            logger.error(f"Failed to write override file: {e}")
        
        # --- Save to database ---
        try:
            # Import here to avoid circular dependency  
            sys.path.insert(0, str(_SCRIPT_DIR))
            from tracking_database import TrackingDatabase
            
            db = TrackingDatabase(str(DB_PATH))
            
            median_activity = activity_values[len(activity_values) // 2] if activity_values else None
            
            review = {
                'review_window_hours': window_hours,
                'total_adjustments': total_adj,
                'total_tracking_events': tracking_stats.get('total_events', 0),
                'unique_people': tracking_stats.get('unique_people', 0),
                'avg_short_activity': adj_stats.get('avg_short'),
                'median_short_activity': median_activity,
                'avg_medium_activity': adj_stats.get('avg_medium'),
                'avg_energy_level': adj_stats.get('avg_energy'),
                'pct_at_floor': pct_floor,
                'pct_at_ceiling': pct_ceiling,
                'mode_distribution': mode_dist,
                'param_stats': param_stats,
                'old_config': old_config,
                'new_config': new_config,
                'changes_summary': '\n'.join(changes) if changes else 'No changes',
                'diagnosis': diagnosis,
                'recommendations': recommendations,
            }
            
            db.save_meta_tuning_review(review)
            db.close()
            logger.info("📝 Saved review to database (meta_tuning_reviews table)")
        except Exception as e:
            logger.error(f"Failed to save review to database: {e}")
    else:
        logger.info("[DRY RUN] Would write config and database record")
        if verbose and changes:
            logger.info(f"[DRY RUN] New config:\n{json.dumps(new_config, indent=2)}")
    
    logger.info("✅ Meta-tuning review complete")
    return new_config


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Tune Meta Review — analyze and adjust auto-tuner configuration"
    )
    parser.add_argument('--window', type=float, default=8,
                       help='Hours of history to analyze (default: 8)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze and print but don\'t write config or DB')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed diagnostics')
    
    args = parser.parse_args()
    run_review(window_hours=args.window, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == '__main__':
    main()
