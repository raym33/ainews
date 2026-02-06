#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/Users/c/Library/LaAurora/newsroom/config.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: no existe config: $CONFIG" >&2
  exit 1
fi

python3 - "$CONFIG" <<'PY'
import json
import sys
import urllib.request
import urllib.error

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
  cfg = json.load(f)

routes = cfg.get("routes", {}) if isinstance(cfg.get("routes"), dict) else {}
if not routes:
  print("WARN: no hay 'routes' en config.json")
  sys.exit(0)

for role in ["chief", "research", "fact", "tagger", "embedding"]:
  r = routes.get(role, {}) if isinstance(routes.get(role), dict) else {}
  base = str(r.get("base_url", "")).rstrip("/")
  model = str(r.get("model", ""))
  if not base:
    print(f"{role}: SIN base_url")
    continue
  url = f"{base}/models"
  try:
    req = urllib.request.Request(url, headers={"User-Agent": "LaAuroraClusterCheck/1.0"})
    with urllib.request.urlopen(req, timeout=4) as resp:
      data = json.loads(resp.read().decode("utf-8", "ignore"))
    ids = [x.get("id") for x in data.get("data", []) if isinstance(x, dict)]
    ok = "OK" if (not model or model in ids) else "MODEL_MISSING"
    print(f"{role}: {ok} | {url} | model={model} | disponibles={len(ids)}")
  except Exception as exc:
    print(f"{role}: ERROR | {url} | {exc}")
PY
