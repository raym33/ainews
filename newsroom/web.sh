#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="${WEB_DIR:-$(cd "$BASE_DIR/../web" && pwd)}"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$WEB_DIR"
exec /opt/homebrew/bin/python3 -m http.server 8080 --bind 0.0.0.0
