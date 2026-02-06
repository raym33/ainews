#!/usr/bin/env python3
import datetime as dt
import json
import os
import subprocess
import sys
import time

BASE_DIR = "/Users/c/Library/LaAurora/newsroom"
CONFIG = os.path.join(BASE_DIR, "config.json")
STATE = os.path.join(BASE_DIR, ".state.json")
TOPICS = os.path.join(BASE_DIR, "topics.json")
FEED = "/Users/c/Library/LaAurora/web/data/articles.json"
TARGET = 10
MAX_ATTEMPTS = 35
MIN_WORDS = 1200
TIMEOUT_SEC = 420


def log(msg):
  stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  print(f"[{stamp}] {msg}", flush=True)


def load_json(path, default):
  if not os.path.exists(path):
    return default
  with open(path, "r", encoding="utf-8") as f:
    try:
      return json.load(f)
    except Exception:
      return default


def save_json(path, data):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


def count_articles():
  data = load_json(FEED, {"articles": []})
  return len(data.get("articles", []))


def choose_topic(topics):
  state = load_json(STATE, {"index": 0, "last_run": ""})
  idx = int(state.get("index", 0))
  if not topics:
    return None
  chosen = topics[idx % len(topics)]
  state["index"] = (idx + 1) % len(topics)
  state["last_run"] = dt.datetime.utcnow().isoformat() + "Z"
  save_json(STATE, state)
  return chosen


def stop_publisher():
  subprocess.run(["launchctl", "bootout", "gui/501/com.la-aurora.publisher"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_publisher():
  subprocess.run(["launchctl", "bootstrap", "gui/501", "/Users/c/Library/LaunchAgents/com.la-aurora.publisher.plist"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(["launchctl", "enable", "gui/501/com.la-aurora.publisher"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(["launchctl", "kickstart", "-k", "gui/501/com.la-aurora.publisher"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_once(topic, region):
  cmd = [
    "/usr/bin/python3",
    os.path.join(BASE_DIR, "ai_publisher.py"),
    "--config", CONFIG,
    "--topic", topic,
    "--region", region,
    "--min-words", str(MIN_WORDS),
    "--max-sources", "4"
  ]
  return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)


def main():
  topics = load_json(TOPICS, [])
  if not topics:
    log("ERROR: topics.json is empty")
    return 1

  stop_publisher()
  log("Publisher paused for initial seeding")

  start_count = count_articles()
  log(f"Initial article count: {start_count}")

  for attempt in range(1, MAX_ATTEMPTS + 1):
    current = count_articles()
    if current >= TARGET:
      log(f"Target reached: {current} articles")
      break

    item = choose_topic(topics)
    if not item:
      log("No topic available")
      time.sleep(2)
      continue

    topic = item.get("topic", "")
    region = item.get("region", "es")
    log(f"Attempt {attempt}/{MAX_ATTEMPTS} | current={current} | topic='{topic}' ({region})")

    try:
      result = run_once(topic, region)
      out = (result.stdout or "").strip().replace("\n", " | ")
      err = (result.stderr or "").strip().replace("\n", " | ")
      log(f"ret={result.returncode}")
      if out:
        log(f"stdout: {out[:700]}")
      if err:
        log(f"stderr: {err[:700]}")
    except subprocess.TimeoutExpired:
      log(f"TIMEOUT > {TIMEOUT_SEC}s on topic '{topic}'")
    except Exception as exc:
      log(f"ERROR on attempt: {exc}")

    updated = count_articles()
    log(f"count after attempt: {updated}")
    time.sleep(2)

  final_count = count_articles()
  log(f"Seeding finished with {final_count} articles")
  start_publisher()
  log("Publisher 24/7 resumed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
