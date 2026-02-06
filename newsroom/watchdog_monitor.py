#!/usr/bin/env python3
import datetime as dt
import json
import os
import signal
import subprocess
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
HEALTH_PATH = os.path.join(LOG_DIR, "health.json")
ALERT_LOG_PATH = os.path.join(LOG_DIR, "health.alerts.log")
ALERT_STATE_PATH = os.path.join(LOG_DIR, "health.alerts.state.json")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOCK_PATH = os.path.join(BASE_DIR, ".runner.lock")
FEED_PATH = "/Users/c/Library/LaAurora/web/data/articles.json"
QUARANTINE_PATH = os.path.join(LOG_DIR, "route_quarantine.json")
PUBLISHER_LABEL = "com.la-aurora.publisher"

HUNG_PROCESS_SEC = 900
FEED_STALE_WARN_SEC = 45 * 60
FEED_STALE_CRIT_SEC = 2 * 60 * 60
ALERT_COOLDOWN_SEC = 15 * 60
HTTP_TIMEOUT_SEC = 4
USER_AGENT = "MetropolisWatchdog/1.0"


def utc_now_iso():
  return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def now_ts():
  return time.time()


def load_json(path, default):
  if not os.path.exists(path):
    return default
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return data
  except Exception:
    return default


def atomic_write_json(path, data):
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)
  tmp = f"{path}.tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)


def append_line(path, line):
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)
  with open(path, "a", encoding="utf-8") as f:
    f.write(line.rstrip() + "\n")


def normalize_base_url(url):
  if not url:
    return ""
  return str(url).strip().rstrip("/")


def parse_iso_utc(value):
  if not value:
    return None
  text = str(value).strip()
  if text.endswith("Z"):
    text = text[:-1] + "+00:00"
  try:
    dtv = dt.datetime.fromisoformat(text)
    if dtv.tzinfo is None:
      dtv = dtv.replace(tzinfo=dt.timezone.utc)
    return dtv.astimezone(dt.timezone.utc)
  except Exception:
    return None


def parse_elapsed_to_sec(text):
  raw = (text or "").strip()
  if not raw:
    return 0
  days = 0
  clock = raw
  if "-" in raw:
    left, right = raw.split("-", 1)
    try:
      days = int(left)
    except Exception:
      days = 0
    clock = right
  parts = clock.split(":")
  try:
    if len(parts) == 3:
      hours = int(parts[0])
      minutes = int(parts[1])
      seconds = int(parts[2])
    elif len(parts) == 2:
      hours = 0
      minutes = int(parts[0])
      seconds = int(parts[1])
    else:
      hours = 0
      minutes = 0
      seconds = int(parts[0])
    return days * 86400 + hours * 3600 + minutes * 60 + seconds
  except Exception:
    return 0


def pid_exists(pid):
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  except Exception:
    return False


def list_ai_publisher_processes():
  cmd = ["ps", "-axo", "pid=,etime=,command="]
  res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
  rows = []
  for raw in res.stdout.splitlines():
    line = raw.strip()
    if not line:
      continue
    parts = line.split(None, 2)
    if len(parts) < 3:
      continue
    pid_text, etime_text, command = parts[0], parts[1], parts[2]
    if "ai_publisher.py" not in command:
      continue
    if BASE_DIR not in command:
      continue
    try:
      pid = int(pid_text)
    except Exception:
      continue
    rows.append({
      "pid": pid,
      "elapsed_sec": parse_elapsed_to_sec(etime_text),
      "command": command
    })
  return rows


def kill_hung_processes(threshold_sec):
  killed = []
  hung = []
  for proc in list_ai_publisher_processes():
    if proc["elapsed_sec"] <= threshold_sec:
      continue
    hung.append(proc)
    pid = proc["pid"]
    try:
      os.kill(pid, signal.SIGTERM)
      time.sleep(0.6)
      if pid_exists(pid):
        os.kill(pid, signal.SIGKILL)
      killed.append(pid)
    except Exception:
      continue
  return hung, killed


def remove_stale_runner_lock():
  if not os.path.exists(LOCK_PATH):
    return False, ""
  try:
    with open(LOCK_PATH, "r", encoding="utf-8") as f:
      raw = f.read().strip()
    pid = int(raw)
  except Exception:
    pid = 0
  if pid > 0 and pid_exists(pid):
    return False, f"lock_owned_by_pid_{pid}"
  try:
    os.remove(LOCK_PATH)
    return True, "stale_lock_removed"
  except Exception as exc:
    return False, f"stale_lock_remove_failed:{exc}"


def launchctl_print(label):
  uid = os.getuid()
  job = f"gui/{uid}/{label}"
  cmd = ["launchctl", "print", job]
  res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
  output = (res.stdout or "") + "\n" + (res.stderr or "")
  running = "state = running" in output
  pid = 0
  for line in output.splitlines():
    line = line.strip()
    if line.startswith("pid = "):
      try:
        pid = int(line.split("=", 1)[1].strip())
      except Exception:
        pid = 0
      break
  return {
    "job": job,
    "ok": res.returncode == 0,
    "running": running,
    "pid": pid,
    "raw": output[:1200]
  }


def launchctl_kickstart(label):
  uid = os.getuid()
  job = f"gui/{uid}/{label}"
  cmd = ["launchctl", "kickstart", "-k", job]
  res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
  msg = (res.stdout or "").strip() or (res.stderr or "").strip()
  return res.returncode == 0, msg[:300]


def feed_health():
  if not os.path.exists(FEED_PATH):
    return {
      "exists": False,
      "article_count": 0,
      "generated_at": "",
      "age_seconds": None,
      "stale_warn": True,
      "stale_critical": True
    }
  payload = load_json(FEED_PATH, {})
  generated_at = ""
  article_count = 0
  if isinstance(payload, dict):
    generated_at = str(payload.get("generated_at", "")).strip()
    articles = payload.get("articles", [])
    if isinstance(articles, list):
      article_count = len(articles)
  stamp = parse_iso_utc(generated_at)
  if stamp is None:
    try:
      mtime = os.path.getmtime(FEED_PATH)
      stamp = dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc)
    except Exception:
      stamp = None
  age_seconds = None
  if stamp is not None:
    age_seconds = int(max(0, (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()))
  stale_warn = age_seconds is None or age_seconds >= FEED_STALE_WARN_SEC
  stale_critical = age_seconds is None or age_seconds >= FEED_STALE_CRIT_SEC
  return {
    "exists": True,
    "article_count": article_count,
    "generated_at": generated_at,
    "age_seconds": age_seconds,
    "stale_warn": stale_warn,
    "stale_critical": stale_critical
  }


def http_get_json(url, timeout_sec):
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
    return json.loads(resp.read().decode("utf-8", errors="ignore"))


def fetch_model_ids(base_url, timeout_sec):
  url = normalize_base_url(base_url)
  if not url:
    return None, "empty_base_url"
  models_url = f"{url}/models"
  try:
    payload = http_get_json(models_url, max(1, int(timeout_sec)))
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    ids = set()
    for row in rows:
      if not isinstance(row, dict):
        continue
      model_id = str(row.get("id", "")).strip()
      if model_id:
        ids.add(model_id)
    if not ids:
      return None, "models_endpoint_empty"
    return ids, ""
  except Exception as exc:
    return None, str(exc)


def route_health(cfg):
  routes = cfg.get("routes", {}) if isinstance(cfg.get("routes"), dict) else {}
  models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
  lm_cfg = cfg.get("lmstudio", {}) if isinstance(cfg.get("lmstudio"), dict) else {}
  default_base = normalize_base_url(lm_cfg.get("base_url", ""))
  default_timeout = int(lm_cfg.get("timeout_sec", 30))
  roles = ["chief", "research", "fact", "tagger", "embedding"]

  out = []
  model_cache = {}
  for role in roles:
    route = routes.get(role, {}) if isinstance(routes.get(role), dict) else {}
    base_url = normalize_base_url(route.get("base_url", "")) or default_base
    model = str(route.get("model", "")).strip() or str(models.get(role, "")).strip()
    timeout_sec = int(route.get("timeout_sec", default_timeout) or default_timeout)
    row = {
      "role": role,
      "base_url": base_url,
      "model": model,
      "timeout_sec": timeout_sec,
      "reachable": False,
      "model_loaded": False,
      "error": ""
    }
    if not base_url or not model:
      row["error"] = "missing_route_or_model"
      out.append(row)
      continue
    key = (base_url, timeout_sec)
    if key not in model_cache:
      model_cache[key] = fetch_model_ids(base_url, timeout_sec=min(timeout_sec, HTTP_TIMEOUT_SEC))
    ids, error = model_cache[key]
    if ids is None:
      row["error"] = f"unreachable:{error[:160]}"
      out.append(row)
      continue
    row["reachable"] = True
    if model in ids:
      row["model_loaded"] = True
    else:
      row["error"] = "model_not_loaded"
    out.append(row)
  return out


def load_alert_state():
  data = load_json(ALERT_STATE_PATH, {"last_sent": {}})
  if not isinstance(data, dict):
    return {"last_sent": {}}
  if not isinstance(data.get("last_sent"), dict):
    data["last_sent"] = {}
  return data


def send_alert(alert_state, key, severity, message, cooldown_sec=ALERT_COOLDOWN_SEC):
  now = now_ts()
  last = float(alert_state["last_sent"].get(key, 0))
  if now - last < cooldown_sec:
    return False
  line = f"{utc_now_iso()} | {severity.upper()} | {key} | {message}"
  append_line(ALERT_LOG_PATH, line)
  alert_state["last_sent"][key] = now
  return True


def quarantine_overview():
  data = load_json(QUARANTINE_PATH, {"entries": {}})
  entries = data.get("entries", {}) if isinstance(data.get("entries"), dict) else {}
  now = now_ts()
  active = []
  for _, item in entries.items():
    if not isinstance(item, dict):
      continue
    until_ts = float(item.get("until_ts", 0))
    if until_ts <= now:
      continue
    active.append({
      "role": str(item.get("role", "")).strip(),
      "base_url": str(item.get("base_url", "")).strip(),
      "model": str(item.get("model", "")).strip(),
      "remaining_seconds": int(max(0, round(until_ts - now))),
      "reason": str(item.get("reason", "")).strip()[:160],
      "fail_count": int(item.get("fail_count", 0))
    })
  return active


def main():
  os.makedirs(LOG_DIR, exist_ok=True)
  alert_state = load_alert_state()

  launchd = launchctl_print(PUBLISHER_LABEL)
  restart_triggered = False
  restart_result = {"ok": True, "message": ""}
  if not launchd.get("running"):
    restart_triggered = True
    ok, msg = launchctl_kickstart(PUBLISHER_LABEL)
    restart_result = {"ok": ok, "message": msg}
    send_alert(alert_state, "publisher_not_running", "critical", "Publisher launchd job was not running. Kickstart attempted.")

  hung, killed = kill_hung_processes(HUNG_PROCESS_SEC)
  if killed:
    restart_triggered = True
    ok, msg = launchctl_kickstart(PUBLISHER_LABEL)
    restart_result = {"ok": ok, "message": msg}
    send_alert(
      alert_state,
      "hung_ai_publisher",
      "critical",
      f"Killed hung ai_publisher processes: {','.join(str(x) for x in killed)}"
    )

  stale_removed, stale_message = remove_stale_runner_lock()
  if stale_removed:
    send_alert(alert_state, "stale_runner_lock", "warning", "Stale .runner.lock removed.")

  feed = feed_health()
  if feed["stale_critical"]:
    age_txt = str(feed.get("age_seconds"))
    send_alert(alert_state, "feed_stale_critical", "critical", f"Feed is stale ({age_txt}s).")
  elif feed["stale_warn"]:
    age_txt = str(feed.get("age_seconds"))
    send_alert(alert_state, "feed_stale_warn", "warning", f"Feed age warning ({age_txt}s).")

  cfg = load_json(CONFIG_PATH, {})
  routes = route_health(cfg if isinstance(cfg, dict) else {})
  route_issues = []
  critical_roles = {"chief", "research", "fact"}
  for row in routes:
    if row["reachable"] and row["model_loaded"]:
      continue
    route_issues.append(row)
    sev = "critical" if row["role"] in critical_roles else "warning"
    reason = row["error"] or "unknown_route_issue"
    send_alert(
      alert_state,
      f"route_{row['role']}_{row['base_url']}_{row['model']}_{reason}",
      sev,
      f"Role={row['role']} base={row['base_url']} model={row['model']} issue={reason}"
    )

  quarantine = quarantine_overview()
  if quarantine:
    send_alert(
      alert_state,
      "routes_quarantined",
      "warning",
      f"Active quarantined routes: {len(quarantine)}"
    )

  status = "ok"
  if feed["stale_critical"] or not launchd.get("running") or any(
    (not row["reachable"] or not row["model_loaded"]) and row["role"] in {"chief", "research", "fact"}
    for row in routes
  ):
    status = "critical"
  elif feed["stale_warn"] or route_issues or stale_removed or killed or quarantine:
    status = "degraded"

  health = {
    "checked_at": utc_now_iso(),
    "status": status,
    "publisher": {
      "launchd_running": bool(launchd.get("running")),
      "launchd_pid": int(launchd.get("pid", 0) or 0),
      "launchd_ok": bool(launchd.get("ok")),
      "restart_triggered": restart_triggered,
      "restart_result": restart_result,
      "hung_processes": [{"pid": p["pid"], "elapsed_sec": p["elapsed_sec"]} for p in hung],
      "killed_pids": killed,
      "stale_lock_removed": stale_removed,
      "stale_lock_note": stale_message
    },
    "feed": feed,
    "routes": routes,
    "route_quarantine": quarantine,
    "metrics": {
      "route_issues": len(route_issues),
      "quarantined_routes": len(quarantine)
    }
  }

  atomic_write_json(HEALTH_PATH, health)
  atomic_write_json(ALERT_STATE_PATH, alert_state)
  print(f"watchdog status={status} routes={len(routes)} issues={len(route_issues)}")


if __name__ == "__main__":
  main()
