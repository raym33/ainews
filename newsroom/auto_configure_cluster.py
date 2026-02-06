#!/usr/bin/env python3
import argparse
import concurrent.futures
import ipaddress
import json
import os
import socket
import urllib.error
import urllib.request

ROLE_ORDER = ["chief", "research", "fact", "tagger", "embedding"]
DEFAULT_MODELS = {
  "chief": "qwen/qwen3-8b",
  "research": "mistralai/ministral-3-3b",
  "fact": "inference-net.schematron-3b",
  "tagger": "liquid/lfm2.5-1.2b",
  "embedding": "text-embedding-qwen3-embedding-8b"
}
DEFAULT_TIMEOUTS = {
  "chief": 90,
  "research": 90,
  "fact": 90,
  "tagger": 60,
  "embedding": 60
}


def parse_args():
  p = argparse.ArgumentParser(description="Auto-configura rutas LM Studio por rol.")
  p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
  p.add_argument("--cidr", default="", help="Rango CIDR; por defecto autodetección /24")
  p.add_argument("--ips", default="", help="IPs separadas por comas")
  p.add_argument("--port", type=int, default=1234)
  p.add_argument("--timeout", type=float, default=0.9)
  p.add_argument("--max-workers", type=int, default=128)
  p.add_argument("--strict-models", action="store_true", help="Falla si falta modelo de rol")
  p.add_argument("--dry-run", action="store_true")
  return p.parse_args()


def default_cidr():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
  finally:
    s.close()
  return str(ipaddress.ip_network(f"{local_ip}/24", strict=False))


def list_targets(args):
  if args.ips.strip():
    return [x.strip() for x in args.ips.split(",") if x.strip()]
  net = ipaddress.ip_network(args.cidr.strip() or default_cidr(), strict=False)
  return [str(ip) for ip in net.hosts()]


def fetch_models(ip, port, timeout):
  url = f"http://{ip}:{port}/v1/models"
  req = urllib.request.Request(url, headers={"User-Agent": "MetropolisAutoConfig/1.0"})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      data = json.loads(resp.read().decode("utf-8", "ignore"))
    ids = [x.get("id") for x in data.get("data", []) if isinstance(x, dict) and x.get("id")]
    if not ids:
      return None
    return {"ip": ip, "base_url": f"http://{ip}:{port}/v1", "models": ids}
  except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
    return None


def discover_workers(args):
  targets = list_targets(args)
  if not targets:
    return []
  found = []
  workers = min(args.max_workers, max(8, len(targets)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
    futures = [ex.submit(fetch_models, ip, args.port, args.timeout) for ip in targets]
    for fut in concurrent.futures.as_completed(futures):
      item = fut.result()
      if item:
        found.append(item)
  found.sort(key=lambda x: x["ip"])
  return found


def choose_best(worker_list, model, assigned_count, prefer_unused=True):
  candidates = [w for w in worker_list if model in w["models"]]
  if not candidates:
    return None
  if prefer_unused:
    free = [w for w in candidates if assigned_count.get(w["ip"], 0) == 0]
    if free:
      candidates = free
  candidates.sort(key=lambda w: (assigned_count.get(w["ip"], 0), len(w["models"])))
  return candidates[0]


def assign_routes(cfg, workers, strict_models=False):
  models_cfg = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
  for role, model in DEFAULT_MODELS.items():
    if not models_cfg.get(role):
      models_cfg[role] = model
  cfg["models"] = models_cfg

  assigned_count = {}
  routes = {}

  for role in ROLE_ORDER:
    wanted = models_cfg.get(role, DEFAULT_MODELS[role])
    pick = choose_best(workers, wanted, assigned_count, prefer_unused=True)
    if pick is None and not strict_models:
      # fallback: cualquier worker disponible
      all_workers = sorted(workers, key=lambda w: assigned_count.get(w["ip"], 0))
      pick = all_workers[0] if all_workers else None
    if pick is None:
      raise RuntimeError(f"No hay worker para rol '{role}' (modelo '{wanted}').")

    chosen_model = wanted if wanted in pick["models"] else (pick["models"][0] if pick["models"] else wanted)
    routes[role] = {
      "base_url": pick["base_url"],
      "model": chosen_model,
      "timeout_sec": DEFAULT_TIMEOUTS.get(role, 90)
    }
    assigned_count[pick["ip"]] = assigned_count.get(pick["ip"], 0) + 1

  cfg["routes"] = routes
  return cfg


def load_json(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def save_json(path, data):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")


def main():
  args = parse_args()
  if not os.path.exists(args.config):
    raise SystemExit(f"No existe config: {args.config}")

  cfg = load_json(args.config)
  workers = discover_workers(args)
  if not workers:
    raise SystemExit("No se detectaron workers LM Studio en la red.")

  cfg = assign_routes(cfg, workers, strict_models=args.strict_models)

  if not args.dry_run:
    save_json(args.config, cfg)

  print(f"Workers detectados: {len(workers)}")
  for w in workers:
    print(f"- {w['ip']} | modelos={len(w['models'])} | {', '.join(w['models'][:5])}")
  print("Rutas asignadas:")
  for role in ROLE_ORDER:
    r = cfg["routes"][role]
    print(f"- {role}: {r['base_url']} | {r['model']}")
  if args.dry_run:
    print("DRY RUN: no se escribieron cambios.")
  else:
    print(f"OK: config actualizada en {args.config}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
