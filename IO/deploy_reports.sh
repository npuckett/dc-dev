#!/usr/bin/env bash
# ============================================================================
# Drop Ceiling — Deploy Reports
# ============================================================================
# Run this after generate_reports.py to push new report data to GitHub
# and trigger a Pages deployment.
#
# Usage:
#   ./deploy_reports.sh              # commit, push, and trigger deploy
#   ./deploy_reports.sh --push-only  # push without triggering dispatch
#   ./deploy_reports.sh --dry-run    # show what would be committed
#
# Requirements:
#   - git configured with push access to npuckett/dc-dev
#   - GITHUB_TOKEN env var set (for repository_dispatch trigger)
#     OR gh CLI authenticated (falls back to gh api)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORTS_DIR="$SCRIPT_DIR/reports/daily"

DRY_RUN=false
PUSH_ONLY=false

for arg in "$@"; do
    case $arg in
        --dry-run)   DRY_RUN=true ;;
        --push-only) PUSH_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--push-only]"
            exit 0
            ;;
    esac
done

cd "$REPO_ROOT"

echo "📊 Deploy Reports — $(date '+%Y-%m-%d %H:%M')"
echo "   Repo: $REPO_ROOT"
echo ""

# Stage report files + viewer files
git add IO/reports/daily/*.json
git add IO/public-viewer/ 2>/dev/null || true

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "ℹ️  No changes to deploy."
    exit 0
fi

echo "📝 Staged changes:"
git diff --cached --stat
echo ""

if $DRY_RUN; then
    echo "🏁 Dry run — nothing committed."
    git reset HEAD -- . >/dev/null
    exit 0
fi

# Commit
DATE_STR=$(date '+%Y-%m-%d')
git commit -m "📊 Daily reports update — $DATE_STR" --quiet

# Push
echo "⬆️  Pushing to origin/main..."
git push origin main --quiet
echo "   ✅ Pushed."

if $PUSH_ONLY; then
    echo "🏁 Push complete (deploy skipped — push triggers workflow on path change)."
    exit 0
fi

# Trigger repository_dispatch to deploy Pages
echo "🚀 Triggering Pages deployment..."

if [ -n "${GITHUB_TOKEN:-}" ]; then
    curl -s -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        https://api.github.com/repos/npuckett/dc-dev/dispatches \
        -d '{"event_type":"deploy-reports"}' \
        && echo "   ✅ Dispatch triggered." \
        || echo "   ⚠️  Dispatch failed (Pages will still deploy on push)."
elif command -v gh &>/dev/null; then
    gh api repos/npuckett/dc-dev/dispatches \
        -f event_type=deploy-reports \
        && echo "   ✅ Dispatch triggered via gh CLI." \
        || echo "   ⚠️  Dispatch failed (Pages will still deploy on push)."
else
    echo "   ℹ️  No GITHUB_TOKEN or gh CLI — skipping dispatch."
    echo "      Pages will still deploy automatically from the push."
fi

echo ""
echo "🏁 Done. Site will update at:"
echo "   https://npuckett.github.io/dc-dev/"
