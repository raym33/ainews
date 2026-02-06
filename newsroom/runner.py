#!/usr/bin/env python3
import argparse
import datetime as dt
import fcntl
import json
import logging
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.json")
TOPICS_PATH = os.path.join(BASE_DIR, "topics.json")
STATE_PATH = os.path.join(BASE_DIR, ".state.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOCK_PATH = os.path.join(BASE_DIR, ".runner.lock")
REWRITE_SCRIPT = os.path.join(BASE_DIR, "rewrite_feed.py")
REWRITE_STATE_PATH = os.path.join(LOG_DIR, "rewrite.state.json")


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


def choose_topic(topics):
  state = load_json(STATE_PATH, {"index": 0})
  idx = int(state.get("index", 0))
  if not topics:
    return None
  topic = topics[idx % len(topics)]
  state["index"] = (idx + 1) % len(topics)
  state["last_run"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
  save_json(STATE_PATH, state)
  return topic


def setup_logging():
  os.makedirs(LOG_DIR, exist_ok=True)
  log_path = os.path.join(LOG_DIR, "publisher.log")
  logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
  )
  logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def parse_iso(raw):
  if not raw or not isinstance(raw, str):
    return None
  try:
    value = raw.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(value)
  except Exception:
    return None


def run_publisher(config, topic, region, min_words):
  cmd = [
    sys.executable,
    os.path.join(BASE_DIR, "ai_publisher.py"),
    "--config",
    config,
    "--topic",
    topic,
    "--region",
    region,
    "--min-words",
    str(min_words),
    "--max-sources",
    "4"
  ]
  # Prevent long stalls: if one run gets stuck, stop it and continue with next topic.
  return subprocess.run(cmd, capture_output=True, text=True, timeout=480)


def maybe_run_rewrite(config, interval_sec, max_items, min_score, min_words):
  if interval_sec <= 0:
    return
  if not os.path.exists(REWRITE_SCRIPT):
    logging.warning("rewrite_feed.py not found; skipping legacy rewrite.")
    return

  now = dt.datetime.now(dt.timezone.utc)
  state = load_json(REWRITE_STATE_PATH, {})
  last = parse_iso(state.get("last_run"))
  if last is not None:
    elapsed = (now - last).total_seconds()
    if elapsed < interval_sec:
      return

  cmd = [
    sys.executable,
    REWRITE_SCRIPT,
    "--config",
    config,
    "--max-items",
    str(max_items),
    "--skip-newest",
    "3",
    "--min-score",
    str(min_score),
    "--min-words",
    str(min_words),
    "--workers",
    "1"
  ]
  logging.info(
    "Running legacy feed rewrite (interval=%ss, max_items=%s, min_score=%s, min_words=%s)",
    interval_sec,
    max_items,
    min_score,
    min_words
  )
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
  except subprocess.TimeoutExpired:
    logging.error("Legacy rewrite timed out after 900s.")
    return

  output = (result.stdout or "").strip()
  err_output = (result.stderr or "").strip()
  if result.returncode != 0:
    logging.error("Legacy rewrite failed: %s", err_output or output or f"returncode={result.returncode}")
    return

  changed = 0
  match = None
  try:
    import re
    match = re.search(r"changed=(\\d+)", output)
  except Exception:
    match = None
  if match:
    changed = int(match.group(1))
  state["last_run"] = now.isoformat().replace("+00:00", "Z")
  state["last_changed"] = changed
  save_json(REWRITE_STATE_PATH, state)

  if output:
    logging.info("Legacy rewrite output: %s", output.replace("\\n", " | "))


def acquire_lock():
  lock_fd = open(LOCK_PATH, "w", encoding="utf-8")
  try:
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    lock_fd.close()
    return None
  lock_fd.write(str(os.getpid()))
  lock_fd.flush()
  return lock_fd


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=DEFAULT_CONFIG)
  parser.add_argument("--min-words", type=int, default=900)
  parser.add_argument("--rewrite-interval-sec", type=int, default=3600)
  parser.add_argument("--rewrite-max-items", type=int, default=1)
  parser.add_argument("--rewrite-min-score", type=int, default=2)
  parser.add_argument("--rewrite-min-words", type=int, default=780)
  return parser.parse_args()


def main():
  setup_logging()
  args = parse_args()
  lock_fd = acquire_lock()
  if lock_fd is None:
    logging.info("Skipping run: another execution is already active.")
    return

  try:
    if not os.path.exists(args.config):
      logging.error("config.json does not exist. Copy config.example.json to config.json and configure r server and LM Studio.")
      sys.exit(1)

    topics = load_json(TOPICS_PATH, [])
    if not topics:
      logging.error("topics.json is empty. Add topics to publish.")
      sys.exit(1)

    max_attempts = 2
    published = False
    for attempt in range(1, max_attempts + 1):
      entry = choose_topic(topics)
      if not entry:
        logging.error("Could not select topic.")
        sys.exit(1)

      topic = entry.get("topic", "")
      region = entry.get("region", "both")
      logging.info("Generating article: %s (%s) [attempt %d/%d]", topic, region, attempt, max_attempts)

      try:
        result = run_publisher(args.config, topic, region, args.min_words)
      except subprocess.TimeoutExpired:
        logging.error("Publisher error: timeout > 480s")
        continue

      if result.returncode != 0:
        err_msg = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
        logging.error("Publisher error: %s", err_msg)
        continue

      logging.info("OK: %s", result.stdout.strip())
      published = True
      break

    if not published:
      logging.error("No article was published in this cycle after %d attempts.", max_attempts)
      sys.exit(1)

    maybe_run_rewrite(
      args.config,
      interval_sec=args.rewrite_interval_sec,
      max_items=args.rewrite_max_items,
      min_score=args.rewrite_min_score,
      min_words=args.rewrite_min_words
    )
  finally:
    try:
      fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    except Exception:
      pass
    lock_fd.close()


if __name__ == "__main__":
  main()
