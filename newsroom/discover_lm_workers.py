#!/usr/bin/env python3
import argparse
import concurrent.futures
import ipaddress
import json
import socket
import urllib.error
import urllib.request


def parse_args():
  p = argparse.ArgumentParser(description="Descubre workers LM Studio en red local.")
  p.add_argument("--cidr", default="", help="Rango CIDR a escanear, ej: 10.211.0.0/24")
  p.add_argument("--ips", default="", help="Lista IP separada por comas")
  p.add_argument("--port", type=int, default=1234)
  p.add_argument("--timeout", type=float, default=0.7)
  p.add_argument("--max-workers", type=int, default=128)
  p.add_argument("--json", action="store_true", help="Salida JSON")
  return p.parse_args()


def default_cidr():
  # Detecta la IP local de salida y asume /24.
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
  finally:
    s.close()
  net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
  return str(net)


def list_targets(args):
  if args.ips.strip():
    out = []
    for item in args.ips.split(","):
      ip = item.strip()
      if ip:
        out.append(ip)
    return out
  cidr = args.cidr.strip() or default_cidr()
  net = ipaddress.ip_network(cidr, strict=False)
  return [str(ip) for ip in net.hosts()]


def fetch_models(ip, port, timeout):
  url = f"http://{ip}:{port}/v1/models"
  req = urllib.request.Request(url, headers={"User-Agent": "LaAuroraDiscover/1.0"})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      data = json.loads(resp.read().decode("utf-8", "ignore"))
    ids = [x.get("id") for x in data.get("data", []) if isinstance(x, dict) and x.get("id")]
    if not ids:
      return None
    return {
      "ip": ip,
      "base_url": f"http://{ip}:{port}/v1",
      "models": ids
    }
  except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
    return None


def main():
  args = parse_args()
  targets = list_targets(args)
  if not targets:
    print("No hay objetivos para escanear.")
    return 0

  found = []
  workers = min(args.max_workers, max(8, len(targets)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
    futures = [ex.submit(fetch_models, ip, args.port, args.timeout) for ip in targets]
    for fut in concurrent.futures.as_completed(futures):
      item = fut.result()
      if item:
        found.append(item)

  found.sort(key=lambda x: x["ip"])
  if args.json:
    print(json.dumps(found, ensure_ascii=False, indent=2))
  else:
    for item in found:
      print(f"{item['ip']}\t{','.join(item['models'][:6])}")
    print(f"FOUND {len(found)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
