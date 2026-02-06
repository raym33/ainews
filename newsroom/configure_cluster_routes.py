#!/usr/bin/env python3
import argparse
import json
import os


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
  p = argparse.ArgumentParser(description="Configura rutas LM Studio por rol para 4 Macs.")
  p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
  p.add_argument("--chief-ip", required=True)
  p.add_argument("--research-ip", required=True)
  p.add_argument("--fact-ip", required=True)
  p.add_argument("--tagger-ip", required=True)
  p.add_argument("--embedding-ip", default=None, help="Si se omite, usa la IP de tagger.")
  return p.parse_args()


def load_json(path):
  if not os.path.exists(path):
    raise SystemExit(f"No existe config: {path}")
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def save_json(path, data):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")


def main():
  args = parse_args()
  cfg = load_json(args.config)
  models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
  for role, model in DEFAULT_MODELS.items():
    if not models.get(role):
      models[role] = model
  cfg["models"] = models

  embedding_ip = args.embedding_ip or args.tagger_ip
  role_ips = {
    "chief": args.chief_ip,
    "research": args.research_ip,
    "fact": args.fact_ip,
    "tagger": args.tagger_ip,
    "embedding": embedding_ip
  }

  routes = {}
  for role, ip in role_ips.items():
    routes[role] = {
      "base_url": f"http://{ip}:1234/v1",
      "model": models.get(role, DEFAULT_MODELS[role]),
      "timeout_sec": DEFAULT_TIMEOUTS.get(role, 90)
    }
  cfg["routes"] = routes
  save_json(args.config, cfg)

  print(f"OK: rutas de clúster guardadas en {args.config}")
  for role in ["chief", "research", "fact", "tagger", "embedding"]:
    info = routes[role]
    print(f"- {role}: {info['base_url']} | {info['model']}")


if __name__ == "__main__":
  main()
