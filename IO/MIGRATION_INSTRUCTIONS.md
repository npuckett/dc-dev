# Database Migration Instructions (Production)

**Date Created:** January 29, 2026  
**Purpose:** Migrate existing tracking data to new aggregation schema

## What Changed

The database system now:
1. Keeps raw events for only **48 hours** (was 7 days)
2. Aggregates data into **hourly_stats** (kept forever)
3. Aggregates daily summaries into **daily_stats_v2** (kept forever)

This preserves historical patterns while reducing database size by ~70%.

## Migration Steps

### 1. Stop the running controller
```bash
sudo systemctl stop light-controller
# or if running manually:
pkill -f lightController_osc.py
```

### 2. Pull the updated code
```bash
cd ~/Drop-Ceiling  # or wherever the repo is
git pull origin main
```

### 3. Backup the existing database (recommended)
```bash
cp IO/tracking_history.db IO/tracking_history_backup_$(date +%Y%m%d).db
```

### 4. Run the migration script
```bash
cd IO
python3 migrate_existing_data.py tracking_history.db
```

The script will:
- Show you the date range of existing data
- Count total events to migrate
- Ask for confirmation before proceeding
- Aggregate all historical hours into `hourly_stats`

### 5. Restart the controller
```bash
sudo systemctl start light-controller
# or if running manually:
python3 lightController_osc.py
```

### 6. Verify it's working
Check the logs for aggregation messages:
```bash
journalctl -u light-controller -f
# Look for: "📊 Hourly aggregate" and "📊 DB maintenance"
```

## What Happens After Migration

- Every hour, the completed hour gets aggregated to `hourly_stats`
- Every hour, raw events older than 48 hours are deleted (after aggregation)
- `hourly_stats` and `daily_stats_v2` are kept **forever**
- Database size will stabilize around ~700MB instead of growing unbounded

## Rollback (if needed)

If something goes wrong:
```bash
# Stop controller
sudo systemctl stop light-controller

# Restore backup
cp IO/tracking_history_backup_YYYYMMDD.db IO/tracking_history.db

# Revert code
git checkout HEAD~1 -- IO/tracking_database.py IO/lightController_osc.py

# Restart
sudo systemctl start light-controller
```
