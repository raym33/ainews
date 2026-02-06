#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BASE_DIR/logs"
CONFIG="$BASE_DIR/config.json"
INTERVAL_SECONDS=300

mkdir -p "$LOG_DIR"

while true; do
  /usr/bin/python3 "$BASE_DIR/runner.py" --config "$CONFIG" --min-words 900 >> "$LOG_DIR/daemon.out.log" 2>> "$LOG_DIR/daemon.err.log" || true
  sleep "$INTERVAL_SECONDS"
done
