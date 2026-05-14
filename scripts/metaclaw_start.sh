#!/bin/bash
# Start MetaClaw proxy for AutoResearchClaw integration.
#
# Usage:
#   ./scripts/metaclaw_start.sh              # skills_only mode (default)
#   ./scripts/metaclaw_start.sh madmax       # madmax mode (with RL training)
#   ./scripts/metaclaw_start.sh skills_only  # skills_only mode (explicit)

set -e

MODE="${1:-skills_only}"
PORT="${2:-30000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
METACLAW_BIN="${METACLAW_BIN:-$REPO_ROOT/.venv/bin/metaclaw}"

if [ ! -x "$METACLAW_BIN" ]; then
    echo "ERROR: metaclaw executable not found at $METACLAW_BIN"
    echo "Run: $REPO_ROOT/.venv/bin/python -m pip install -e $REPO_ROOT/.external/MetaClaw"
    exit 1
fi

echo "Starting MetaClaw in ${MODE} mode on port ${PORT}..."
exec "$METACLAW_BIN" start --mode "$MODE" --port "$PORT"
