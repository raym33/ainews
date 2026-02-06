#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

DEFAULT_CONFIG = {
  "lmstudio": {
    "base_url": "http://10.211.0.240:1234/v1",
    "timeout_sec": 60
  },
  "models": {
    "chief": "qwen/qwen3-8b",
    "research": "mistralai/ministral-3-3b",
    "fact": "inference-net.schematron-3b",
    "tagger": "liquid/lfm2.5-1.2b"
  },
  "search": {
    "provider": "rss",
    "r_cli_base_url": "http://localhost:8765",
    "skill": "websearch",
    "tool": "search",
    "args": {
      "num_results": 8,
      "lang": "es"
    }
  },
  "rss_feeds": {
    "spain": [],
    "world": []
  }
}

USER_AGENT = "NewsroomBot/1.0 (+https://example.local)"


def load_config(path):
  cfg = json.loads(json.dumps(DEFAULT_CONFIG))
  if not path:
    return cfg
  with open(path, "r", encoding="utf-8") as f:
    user_cfg = json.load(f)
  deep_merge(cfg, user_cfg)
  return cfg


def deep_merge(base, updates):
  for key, value in updates.items():
    if isinstance(value, dict) and isinstance(base.get(key), dict):
      deep_merge(base[key], value)
    else:
      base[key] = value


def http_post_json(url, payload, timeout_sec=60):
  data = json.dumps(payload).encode("utf-8")
  req = urllib.request.Request(url, data=data, headers={
    "Content-Type": "application/json",
    "User-Agent": USER_AGENT
  })
  with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
    return json.loads(resp.read().decode("utf-8"))


def http_get(url, timeout_sec=30):
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
    return resp.read()


class LMStudioClient:
  def __init__(self, base_url, timeout_sec=60):
    self.base_url = base_url.rstrip("/")
    self.timeout_sec = timeout_sec

  def chat(self, model, messages, temperature=0.2, max_tokens=800):
    payload = {
      "model": model,
      "messages": messages,
      "temperature": temperature,
      "max_tokens": max_tokens
    }
    data = http_post_json(f"{self.base_url}/chat/completions", payload, self.timeout_sec)
    return data["choices"][0]["message"]["content"]


class RCLISearchProvider:
  def __init__(self, base_url, skill, tool, args, timeout_sec=30):
    self.base_url = base_url.rstrip("/")
    self.skill = skill
    self.tool = tool
    self.args = args
    self.timeout_sec = timeout_sec

  def search(self, query):
    payload = {
      "skill": self.skill,
      "tool": self.tool,
      "arguments": {**self.args, "query": query}
    }
    data = http_post_json(f"{self.base_url}/v1/skills/call", payload, self.timeout_sec)
    return normalize_sources(data)


class RSSProvider:
  def __init__(self, feeds, timeout_sec=20):
    self.feeds = feeds
    self.timeout_sec = timeout_sec

  def search(self, query):
    terms = [t.strip().lower() for t in query.split() if t.strip()]
    results = []
    for feed in self.feeds:
      try:
        raw = http_get(feed, self.timeout_sec)
        results.extend(parse_rss(raw, terms))
      except Exception:
        continue
    return results


def parse_rss(raw, terms):
  out = []
  try:
    root = ET.fromstring(raw)
  except Exception:
    return out
  items = root.findall(".//item")
  if not items:
    items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
  for item in items:
    title = find_text(item, ["title", "{http://www.w3.org/2005/Atom}title"]) or ""
    link = find_text(item, ["link", "{http://www.w3.org/2005/Atom}link"]) or ""
    if not link:
      link_el = item.find("{http://www.w3.org/2005/Atom}link")
      if link_el is not None:
        link = link_el.attrib.get("href", "")
    desc = find_text(item, ["description", "{http://www.w3.org/2005/Atom}summary"]) or ""
    pub = find_text(item, ["pubDate", "{http://www.w3.org/2005/Atom}updated"]) or ""
    text_blob = f"{title} {desc}".lower()
    if terms and not all(t in text_blob for t in terms):
      continue
    out.append({
      "title": sanitize(title),
      "url": sanitize(link),
      "snippet": sanitize(desc),
      "published": sanitize(pub),
      "source": "rss"
    })
  return out


def find_text(node, tags):
  for tag in tags:
    el = node.find(tag)
    if el is not None and el.text:
      return el.text
  return None


def normalize_sources(data):
  if isinstance(data, dict):
    if "result" in data:
      data = data["result"]
    elif "output" in data:
      data = data["output"]
  if isinstance(data, str):
    return [{"title": "", "url": "", "snippet": data, "source": "r_cli"}]
  if isinstance(data, list):
    return data
  return []


def sanitize(text):
  if not text:
    return ""
  text = html.unescape(text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


def fetch_article_text(url, timeout_sec=20, limit_chars=8000):
  if not url:
    return ""
  try:
    raw = http_get(url, timeout_sec)
    html_text = raw.decode("utf-8", errors="ignore")
    html_text = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.I)
    html_text = re.sub(r"<style[\s\S]*?</style>", " ", html_text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = sanitize(text)
    return text[:limit_chars]
  except Exception:
    return ""


def chunk_sources(sources, max_items):
  seen = set()
  out = []
  for src in sources:
    url = src.get("url", "")
    if url and url in seen:
      continue
    if url:
      seen.add(url)
    out.append(src)
    if len(out) >= max_items:
      break
  return out


def llm_json_or_text(text):
  match = re.search(r"\{[\s\S]*\}", text)
  if not match:
    return None
  try:
    return json.loads(match.group(0))
  except Exception:
    return None


def build_prompts(topic, sources):
  source_lines = []
  for i, src in enumerate(sources, 1):
    source_lines.append(f"[{i}] {src.get('title','')} | {src.get('url','')} | {src.get('published','')}")
  sources_block = "\n".join(source_lines)

  prompts = {
    "summarize": [
      {
        "role": "system",
        "content": "Eres un periodista de datos. Resume fuentes con precision. No inventes."
      },
      {
        "role": "user",
        "content": f"Tema: {topic}\nFuentes:\n{sources_block}\nDevuelve 5-7 puntos clave en espanol."
      }
    ],
    "extract": [
      {
        "role": "system",
        "content": "Extrae hechos verificables. Responde solo JSON."
      },
      {
        "role": "user",
        "content": "Devuelve JSON con claves: claims (lista de objetos con claim, fuente, evidencia_resumida)."
      }
    ],
    "verify": [
      {
        "role": "system",
        "content": "Eres verificador. Cruzas hechos y detectas contradicciones."
      },
      {
        "role": "user",
        "content": "Consolida claims. Devuelve JSON con verified (lista) y disputed (lista)."
      }
    ],
    "write": [
      {
        "role": "system",
        "content": "Eres editor jefe. Escribes notas claras, sobrias y verificadas. No especules."
      },
      {
        "role": "user",
        "content": "Escribe una nota en espanol con: Titular, Bajada, Cuerpo, Lo que se sabe, Lo que falta, Fuentes (lista)."
      }
    ],
    "tag": [
      {
        "role": "system",
        "content": "Etiquetador editorial. Devuelve JSON." 
      },
      {
        "role": "user",
        "content": "Devuelve JSON con categoria y tags (lista corta)."
      }
    ]
  }
  return prompts


def run_pipeline(cfg, topic, region, max_sources):
  lm = LMStudioClient(cfg["lmstudio"]["base_url"], cfg["lmstudio"]["timeout_sec"])

  provider = cfg["search"]["provider"]
  if provider == "r_cli":
    searcher = RCLISearchProvider(
      cfg["search"]["r_cli_base_url"],
      cfg["search"]["skill"],
      cfg["search"]["tool"],
      cfg["search"].get("args", {}),
      timeout_sec=30
    )
  else:
    feeds = []
    if region in ("es", "both"):
      feeds.extend(cfg["rss_feeds"].get("spain", []))
    if region in ("world", "both"):
      feeds.extend(cfg["rss_feeds"].get("world", []))
    searcher = RSSProvider(feeds)

  queries = [topic]
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    search_results = list(ex.map(searcher.search, queries))

  sources = []
  for batch in search_results:
    sources.extend(batch)

  sources = chunk_sources(sources, max_sources)
  if not sources:
    raise RuntimeError("No se encontraron fuentes para el tema.")

  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    fetched = list(ex.map(lambda s: fetch_article_text(s.get("url", "")), sources))

  for src, text in zip(sources, fetched):
    src["content"] = text or src.get("snippet", "")

  prompts = build_prompts(topic, sources)

  # Parallel: resumen, extraccion de hechos, etiquetas
  def call(model_key, prompt_key, extra_text=None, max_tokens=600, temperature=0.2):
    messages = list(prompts[prompt_key])
    if extra_text:
      messages.append({"role": "user", "content": extra_text})
    return lm.chat(cfg["models"][model_key], messages, temperature=temperature, max_tokens=max_tokens)

  sources_blob = "\n\n".join(
    [f"Fuente {i+1}: {s.get('title','')}\nURL: {s.get('url','')}\nTexto: {s.get('content','')[:4000]}" for i, s in enumerate(sources)]
  )

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    summarize_fut = ex.submit(call, "research", "summarize", sources_blob, 700, 0.2)
    extract_fut = ex.submit(call, "fact", "extract", sources_blob, 800, 0.1)
    tag_fut = ex.submit(call, "tagger", "tag", sources_blob, 200, 0.2)

  summary = summarize_fut.result()
  extracted = extract_fut.result()
  tags = tag_fut.result()

  extracted_json = llm_json_or_text(extracted) or {"claims": []}
  tag_json = llm_json_or_text(tags) or {"categoria": "", "tags": []}

  verify_input = json.dumps(extracted_json, ensure_ascii=False)
  verified = call("chief", "verify", verify_input, 400, 0.1)
  verified_json = llm_json_or_text(verified) or {"verified": [], "disputed": []}

  write_input = (
    f"Resumen:\n{summary}\n\nClaims verificados:\n{json.dumps(verified_json, ensure_ascii=False)}\n\n"
    f"Fuentes:\n" + "\n".join([s.get("url", "") for s in sources])
  )
  article = call("chief", "write", write_input, 900, 0.2)

  output = {
    "topic": topic,
    "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    "sources": sources,
    "summary": summary,
    "claims": extracted_json,
    "verified": verified_json,
    "tags": tag_json,
    "article": article
  }

  return output


def write_output(output, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
  slug = re.sub(r"[^a-z0-9]+", "-", output["topic"].lower()).strip("-")[:48] or "nota"
  base = f"{ts}-{slug}"
  json_path = os.path.join(out_dir, f"{base}.json")
  md_path = os.path.join(out_dir, f"{base}.md")

  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

  md = output["article"]
  md += "\n\n## Fuentes\n" + "\n".join([f"- {s.get('url','')}" for s in output["sources"]])
  with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)

  return json_path, md_path


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None)
  parser.add_argument("--topic", required=True)
  parser.add_argument("--region", choices=["es", "world", "both"], default="both")
  parser.add_argument("--max-sources", type=int, default=8)
  parser.add_argument("--out", default="output")
  return parser.parse_args()


def main():
  args = parse_args()
  cfg = load_config(args.config)
  output = run_pipeline(cfg, args.topic, args.region, args.max_sources)
  json_path, md_path = write_output(output, args.out)
  print(f"OK: {json_path}")
  print(f"OK: {md_path}")


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    sys.exit(1)
