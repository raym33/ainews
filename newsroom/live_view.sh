#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BASE_DIR/logs"
FEED_PATH="${FEED_PATH:-$(cd "$BASE_DIR/../web/data" && pwd)/articles.json}"
PUBLISHER_LABEL="com.la-aurora.publisher"

print_status() {
  local uid
  uid="$(id -u)"
  local job="gui/${uid}/${PUBLISHER_LABEL}"
  local launch
  launch="$(launchctl print "$job" 2>/dev/null || true)"
  local state pid
  state="$(printf "%s\n" "$launch" | awk -F'= ' '/state =/{print $2; exit}' | xargs || true)"
  pid="$(printf "%s\n" "$launch" | awk -F'= ' '/pid =/{print $2; exit}' | xargs || true)"
  if [[ -z "${state}" ]]; then
    state="unknown"
  fi
  if [[ -z "${pid}" ]]; then
    pid="-"
  fi
  printf "publisher: %s | pid: %s\n" "$state" "$pid"
}

print_json_snapshots() {
  FEED_PATH_ENV="$FEED_PATH" LOG_DIR_ENV="$LOG_DIR" /usr/bin/python3 - <<'PY'
import json
import os
import re

feed_path = os.environ.get("FEED_PATH_ENV", "")
log_dir = os.environ.get("LOG_DIR_ENV", "")
rewrite_state = os.path.join(log_dir, "rewrite.state.json")
health_path = os.path.join(log_dir, "health.json")

def load(path, default):
  if not os.path.exists(path):
    return default
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return default

feed = load(feed_path, {"articles": [], "generated_at": ""})
arts = feed.get("articles") if isinstance(feed.get("articles"), list) else []
latest = arts[0] if arts else {}
title = (latest.get("title") or "").strip()
section = (latest.get("section") or "").strip()
published = (latest.get("published") or "").strip()
body = re.sub(r"<[^>]+>", " ", latest.get("body_html", ""))
words = len(re.findall(r"\b\w+\b", body))

rw = load(rewrite_state, {})
rw_last = rw.get("last_run", "-")
rw_changed = rw.get("last_changed", "-")

health = load(health_path, {})
health_status = health.get("status", "-")
health_age = health.get("feed_age_seconds", "-")

print(f"health: {health_status} | feed_age_s: {health_age}")
print(f"rewrite: last_run={rw_last} | last_changed={rw_changed}")
print(f"feed_generated_at: {feed.get('generated_at','-')}")
if title:
  print(f"latest: [{section}] {title}")
  print(f"latest_published: {published} | words: {words}")
else:
  print("latest: -")
PY
}

while true; do
  if [[ -t 1 ]]; then
    clear 2>/dev/null || printf "\033c"
  fi
  echo "Metropolis Live View"
  echo "===================="
  date "+%Y-%m-%d %H:%M:%S %Z"
  print_status
  print_json_snapshots
  echo
  echo "--- publisher.log (last 8) ---"
  tail -n 8 "$LOG_DIR/publisher.log" 2>/dev/null || true
  echo
  echo "--- daemon.out.log (last 8) ---"
  tail -n 8 "$LOG_DIR/daemon.out.log" 2>/dev/null || true
  if [[ "${LIVE_VIEW_ONCE:-0}" == "1" ]]; then
    break
  fi
  sleep 5
done
