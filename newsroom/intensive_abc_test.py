#!/usr/bin/env python3
import argparse
import datetime as dt
import html
import json
import os
import re
import subprocess
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
PUBLISHER_LABEL = "com.la-aurora.publisher"
AI_PUBLISHER = os.path.join(BASE_DIR, "ai_publisher.py")
FORUM_SCRIPT = os.path.join(BASE_DIR, "generate_forum_debate.py")
AGENTBOOK_SYNC_SCRIPT = os.path.join(BASE_DIR, "sync_agentbook_forum.py")
REPORT_DIR = os.path.join(BASE_DIR, "logs")
ABC_URL = "https://www.abc.es/"


def utc_now_iso():
  return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def now_stamp():
  return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def normalize_key(text):
  return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean_text(text):
  value = html.unescape(text or "")
  value = re.sub(r"<[^>]+>", " ", value)
  value = re.sub(r"\s+", " ", value).strip()
  return value


def fetch_abc_headlines(limit=4):
  raw = urllib.request.urlopen(ABC_URL, timeout=25).read().decode("utf-8", errors="ignore")
  blocks = re.findall(r"<h2[^>]*>(.*?)</h2>", raw, flags=re.I | re.S)
  out = []
  seen = set()
  for block in blocks:
    text = clean_text(block)
    if len(text) < 42:
      continue
    key = normalize_key(text)
    if not key or key in seen:
      continue
    seen.add(key)
    out.append(text)
    if len(out) >= limit:
      break
  return out


def run_cmd(cmd, timeout_sec):
  start = time.time()
  result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
  elapsed = round(time.time() - start, 2)
  return {
    "returncode": result.returncode,
    "stdout": (result.stdout or "").strip(),
    "stderr": (result.stderr or "").strip(),
    "elapsed_sec": elapsed
  }


def launchctl_bootout(label):
  uid = str(os.getuid())
  cmd = ["launchctl", "bootout", f"gui/{uid}/{label}"]
  subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def launchctl_start(label):
  uid = str(os.getuid())
  plist = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
  subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", plist], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(["launchctl", "enable", f"gui/{uid}/{label}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def atomic_write_json(path, payload):
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)
  tmp = f"{path}.tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=CONFIG_PATH)
  parser.add_argument("--count", type=int, default=4)
  parser.add_argument("--min-words", type=int, default=1400)
  parser.add_argument("--max-sources", type=int, default=5)
  parser.add_argument("--timeout", type=int, default=780)
  return parser.parse_args()


def main():
  args = parse_args()
  report = {
    "started_at": utc_now_iso(),
    "abc_url": ABC_URL,
    "topics": [],
    "article_runs": [],
    "forum_run": {},
    "agentbook_sync_run": {},
    "publisher_paused": False,
    "publisher_resumed": False
  }

  topics = fetch_abc_headlines(limit=max(4, args.count))
  if len(topics) < args.count:
    raise RuntimeError(f"Could not extract {args.count} current topics from abc.es (got {len(topics)}).")
  selected_topics = topics[: args.count]
  report["topics"] = selected_topics

  launchctl_bootout(PUBLISHER_LABEL)
  report["publisher_paused"] = True
  time.sleep(1.2)

  try:
    for idx, topic in enumerate(selected_topics, start=1):
      cmd = [
        "/usr/bin/python3",
        AI_PUBLISHER,
        "--config",
        args.config,
        "--topic",
        topic,
        "--region",
        "es",
        "--min-words",
        str(args.min_words),
        "--max-sources",
        str(args.max_sources)
      ]
      run = run_cmd(cmd, timeout_sec=args.timeout)
      run["topic"] = topic
      run["index"] = idx
      report["article_runs"].append(run)

      # Cool-down to avoid saturating remote nodes between long-form generations.
      time.sleep(1.0)

    forum_cmd = [
      "/usr/bin/python3",
      FORUM_SCRIPT,
      "--config",
      args.config,
      "--topic",
      "Can human consciousness be replicated by AI systems?"
    ]
    report["forum_run"] = run_cmd(forum_cmd, timeout_sec=420)

    sync_cmd = [
      "/usr/bin/python3",
      AGENTBOOK_SYNC_SCRIPT,
      "--agentbook-api",
      "http://127.0.0.1:8000/api"
    ]
    report["agentbook_sync_run"] = run_cmd(sync_cmd, timeout_sec=60)
  finally:
    launchctl_start(PUBLISHER_LABEL)
    report["publisher_resumed"] = True
    report["finished_at"] = utc_now_iso()

  report_name = f"intensive_abc_test_{now_stamp()}.json"
  report_path = os.path.join(REPORT_DIR, report_name)
  atomic_write_json(report_path, report)
  print(f"OK: {report_path}")

  successes = sum(1 for r in report["article_runs"] if r.get("returncode") == 0)
  print(f"Articles generated: {successes}/{len(report['article_runs'])}")
  print(f"Forum return code: {report['forum_run'].get('returncode')}")


if __name__ == "__main__":
  main()
