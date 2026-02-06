#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime as dt
import difflib
import html
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

USER_AGENT = "MetropolisBot/2.0 (+https://metropolis.local)"
MASTER_EDITORIAL_PROMPT = """
You are the editorial core of Metropolis, an English-language digital newspaper.
Global non-negotiable rules:
- Output must be in natural, publication-ready English.
- Write with a professional newsroom tone: analytical, concise, and precise.
- Originality is mandatory: do not copy headlines or sentences from source material.
- Never reuse long strings from sources; avoid reproducing 7+ consecutive source words.
- Preserve factual meaning, but rewrite with new structure and wording.
- Do not include URLs, citations, bracket references, or source labels in public copy.
- No chain-of-thought, no meta-commentary, no drafting notes.
""".strip()

DEFAULT_CONFIG = {
  "lmstudio": {
    "base_url": "http://10.211.0.240:1234/v1",
    "timeout_sec": 120
  },
  "models": {
    "chief": "openai/gpt-oss-20b",
    "research": "openai/gpt-oss-20b",
    "fact": "openai/gpt-oss-20b",
    "tagger": "openai/gpt-oss-20b",
    "embedding": ""
  },
  "routes": {},
  "search": {
    "provider": "r_cli_local",
    "script_path": None,
    "args": {
      "num_results": 8,
      "lang": "en"
    },
    "preferred_domains": []
  },
  "pipeline_max_sec": 260,
  "quarantine_seconds": 900,
  "rss_feeds": {
    "spain": [],
    "world": []
  }
}


def utc_now_iso():
  return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class LMStudioClient:
  def __init__(self, base_url, timeout_sec=90):
    self.base_url = base_url.rstrip("/")
    self.timeout_sec = timeout_sec

  def chat(self, model, messages, temperature=0.2, max_tokens=900):
    payload = {
      "model": model,
      "messages": messages,
      "temperature": temperature,
      "max_tokens": max_tokens
    }
    data = http_post_json(f"{self.base_url}/chat/completions", payload, timeout_sec=self.timeout_sec)
    return data["choices"][0]["message"].get("content", "")

  def embeddings(self, model, inputs):
    payload = {
      "model": model,
      "input": inputs if isinstance(inputs, list) else [inputs]
    }
    data = http_post_json(f"{self.base_url}/embeddings", payload, timeout_sec=self.timeout_sec)
    rows = data.get("data", [])
    size = len(payload["input"])
    vectors = [None] * size
    for row in rows:
      if not isinstance(row, dict):
        continue
      idx = row.get("index")
      emb = row.get("embedding")
      if isinstance(idx, int) and 0 <= idx < size and isinstance(emb, list):
        vectors[idx] = emb
    if any(v is None for v in vectors):
      raise RuntimeError("Incomplete embeddings response.")
    return vectors


def log_warn(message):
  print(f"WARN: {message}", file=sys.stderr)


def http_post_json(url, payload, timeout_sec=90):
  data = json.dumps(payload).encode("utf-8")
  req = urllib.request.Request(
    url,
    data=data,
    headers={
      "Content-Type": "application/json",
      "User-Agent": USER_AGENT
    }
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
      return json.loads(resp.read().decode("utf-8"))
  except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def http_get(url, timeout_sec=30):
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
    return resp.read()


def chat_with_retry(lm, model, messages, temperature=0.2, max_tokens=900, retries=3):
  last_exc = None
  token_budget = max_tokens
  for attempt in range(retries + 1):
    try:
      return lm.chat(model, messages, temperature=temperature, max_tokens=token_budget)
    except Exception as exc:
      last_exc = exc
      if attempt >= retries:
        break
      token_budget = max(220, int(token_budget * 0.75))
      time.sleep(1.4 * (attempt + 1))
  raise last_exc


def sanitize(text):
  if not text:
    return ""
  out = html.unescape(str(text))
  out = re.sub(r"\s+", " ", out).strip()
  return out


def normalize_url(url):
  clean = sanitize(url)
  if not clean:
    return ""
  if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", clean):
    return clean
  return f"https://{clean.lstrip('/')}"


def domain_from_url(url):
  try:
    parsed = urllib.parse.urlparse(normalize_url(url))
    return parsed.netloc.lower().replace("www.", "")
  except Exception:
    return ""


def strip_internal_reasoning(text):
  if not text:
    return ""
  out = text.replace("\r\n", "\n").replace("\r", "\n")
  out = re.sub(r"(?is)<think>[\s\S]*?</think>", " ", out)
  out = re.sub(r"(?is)&lt;think&gt;[\s\S]*?&lt;/think&gt;", " ", out)
  out = re.sub(r"(?is)<analysis>[\s\S]*?</analysis>", " ", out)
  out = re.sub(r"(?is)&lt;analysis&gt;[\s\S]*?&lt;/analysis&gt;", " ", out)
  out = re.sub(r"(?im)^\s*(okay|vale|i need to|let me|the user wants|el usuario pide)\b.*$", "", out)
  out = re.sub(r"(?im)^\s*(first|second|third|next|finally),?\b.*$", "", out)
  out = re.sub(r"(?im)^\s*(debo|voy a|primero voy a)\b.*$", "", out)
  out = re.sub(r"(?im)^\s*(the title is|the section is|rules say|now using|now structuring|alternatively)\b.*$", "", out)
  out = re.sub(r"(?im)^\s*(?:<|&lt;)/?(?:think|analysis)(?:>|&gt;)\s*$", "", out)
  out = re.sub(r"\n{3,}", "\n\n", out)
  return out.strip()


def has_internal_markers(text):
  if not text:
    return False
  return bool(
    re.search(
      r"(?is)(?:<|&lt;)\s*(?:think|analysis)\b|\b(?:i need to|the user wants|el usuario pide|el usuario quiere|voy a|debo|the title is|the section is|rules say)\b",
      text
    )
  )


def looks_like_meta_paragraph(text):
  low = normalize_text_key(text)
  if not low:
    return True
  if re.fullmatch(r"(references?|referencias?|fuentes verificadas|verified sources)", low):
    return True
  if re.search(r"\b(i need to|the user wants|let me|first|second|third|finally)\b", low):
    return True
  if re.search(r"\b(the title is|the section is|rules say|this section|now using|now structuring|alternatively)\b", low):
    return True
  if re.search(r"\b(el usuario|voy a|debo|primero voy a)\b", low):
    return True
  if re.search(r"\b(okay|vale)\b", low):
    return True
  return False


def dedupe_paragraphs(text):
  if not text:
    return ""
  chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
  out = []
  seen = set()
  for chunk in chunks:
    if looks_like_meta_paragraph(chunk):
      continue
    key = normalize_text_key(chunk)
    if not key:
      continue
    if key in seen:
      continue
    seen.add(key)
    out.append(chunk)
  return "\n\n".join(out).strip()


def strip_embedded_headings(text):
  if not text:
    return ""
  out = re.sub(r"(?m)^\s*#{2,6}\s+.+$", "", text)
  inline = re.search(r"\s##\s+[A-Za-z0-9]", out)
  if inline:
    out = out[:inline.start()].strip()
  out = re.sub(r"\n{3,}", "\n\n", out)
  return out.strip()


def trim_to_complete_sentence(text, min_words=80):
  if not text:
    return ""
  out = text.strip()
  if re.search(r"[.!?]\s*$", out):
    return out
  matches = list(re.finditer(r"[.!?]", out))
  if matches:
    candidate = out[:matches[-1].end()].strip()
    if word_count(candidate) >= min_words:
      return candidate
  if "," in out:
    candidate = out.rsplit(",", 1)[0].strip()
    if word_count(candidate) >= min_words:
      return candidate + "."
  return out


def normalize_text_key(value):
  base = unicodedata.normalize("NFKD", sanitize(value))
  base = "".join(c for c in base if not unicodedata.combining(c))
  base = base.lower()
  base = re.sub(r"^\s*(?:\d+[\.\)]\s*)+", "", base)
  base = re.sub(r"[^a-z0-9]+", " ", base).strip()
  return base


def tokenize_similarity_text(text, limit_tokens=1800):
  key = normalize_text_key(text)
  if not key:
    return []
  tokens = [t for t in key.split() if len(t) >= 2]
  if len(tokens) > limit_tokens:
    tokens = tokens[:limit_tokens]
  return tokens


def ngram_set(tokens, n):
  if n <= 0 or len(tokens) < n:
    return set()
  return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def build_source_overlap_index(sources):
  source_titles = []
  chunks = []
  for src in sources or []:
    if not isinstance(src, dict):
      continue
    title = sanitize(src.get("title", ""))[:220]
    snippet = sanitize(src.get("snippet", ""))[:420]
    content = sanitize(src.get("content", ""))[:2600]
    if title:
      source_titles.append(title)
      chunks.append(title)
    if snippet:
      chunks.append(snippet)
    if content:
      chunks.append(content)
  tokens = tokenize_similarity_text(" ".join(chunks), limit_tokens=6500)
  return {
    "titles": source_titles,
    "ngrams_5": ngram_set(tokens, 5),
    "ngrams_7": ngram_set(tokens, 7)
  }


def text_source_overlap_metrics(text, source_index):
  if not source_index:
    return {"overlap_5": 0.0, "copied_run_7": False}
  tokens = tokenize_similarity_text(text, limit_tokens=2200)
  cand_5 = ngram_set(tokens, 5)
  cand_7 = ngram_set(tokens, 7)
  src_5 = source_index.get("ngrams_5") or set()
  src_7 = source_index.get("ngrams_7") or set()
  overlap_5 = 0.0
  if cand_5:
    overlap_5 = len(cand_5 & src_5) / max(1, len(cand_5))
  copied_run_7 = bool(cand_7 & src_7) if cand_7 and src_7 else False
  return {"overlap_5": overlap_5, "copied_run_7": copied_run_7}


def title_similarity_score(left, right):
  left_key = normalize_text_key(left)
  right_key = normalize_text_key(right)
  if not left_key or not right_key:
    return 0.0
  ratio = difflib.SequenceMatcher(None, left_key, right_key).ratio()
  left_tokens = left_key.split()
  right_tokens = right_key.split()
  union = set(left_tokens) | set(right_tokens)
  inter = set(left_tokens) & set(right_tokens)
  jaccard = len(inter) / max(1, len(union))
  same_prefix = 0.0
  if len(left_tokens) >= 4 and len(right_tokens) >= 4:
    if " ".join(left_tokens[:4]) == " ".join(right_tokens[:4]):
      same_prefix = 1.0
  return max(ratio, jaccard, same_prefix)


def title_too_close_to_sources(title, source_titles):
  key = normalize_text_key(title)
  if not key:
    return True
  for candidate in source_titles or []:
    score = title_similarity_score(title, candidate)
    if score >= 0.76:
      return True
  return False


def looks_invalid_short_text(text):
  value = sanitize(text)
  if not value:
    return True
  if len(value) < 4:
    return True
  if "\n" in value:
    return True
  if re.match(r"(?i)^```", value):
    return True
  if re.match(r"(?i)^(title|deck|section|category|tags?)\s*[:=]", value):
    return True
  if value.startswith("{") or value.startswith("["):
    return True
  if re.search(r"[{}\[\]]", value) and ":" in value:
    return True
  return False


def has_source_attribution(text):
  value = sanitize(text).lower()
  if not value:
    return False
  if "http://" in value or "https://" in value or "www." in value:
    return True
  if re.search(r"\[(\d{1,3})\]", text or ""):
    return True
  if re.search(r"\b(according to|as reported by|reported by|per|via)\b", value):
    return True
  if re.search(r"\b(wire service|newswire)\b", value):
    return True
  return False


def looks_mostly_spanish(text):
  raw = f" {sanitize(text).lower()} "
  if not raw.strip():
    return False
  es_markers = [
    " el ", " la ", " los ", " las ", " del ", " una ", " un ", " para ", " con ", " que ", " como ",
    " espana", " españa", " mercado ", " salarios ", " politica ", " economia ", " sociedad "
  ]
  en_markers = [
    " the ", " and ", " of ", " to ", " in ", " for ", " with ", " from ", " on ",
    " market ", " wages ", " spain ", " policy ", " business ", " technology "
  ]
  score_es = sum(raw.count(m) for m in es_markers)
  score_en = sum(raw.count(m) for m in en_markers)
  return score_es >= max(3, score_en + 2)


def looks_spanish_title(title):
  raw = f" {sanitize(title).lower()} "
  if not raw.strip():
    return False
  if re.search(r"[¿¡áéíóúñ]", raw):
    return True
  markers = [
    " el ", " la ", " los ", " las ", " del ", " de ", " y ", " en ", " para ",
    " con ", " que ", " gobierno ", " huelga ", " campana ", " campaña "
  ]
  score = sum(raw.count(m) for m in markers)
  return score >= 2


def normalize_public_section(value, region):
  key = normalize_text_key(value)
  if not key:
    return "Spain" if region == "es" else "World"
  mapping = [
    ("espana", "Spain"),
    ("spain", "Spain"),
    ("mundo", "World"),
    ("world", "World"),
    ("economia", "Business"),
    ("business", "Business"),
    ("politica", "Politics"),
    ("politics", "Politics"),
    ("tecnologia", "Technology"),
    ("technology", "Technology"),
    ("sociedad", "Society"),
    ("society", "Society"),
    ("ciencia", "Science"),
    ("science", "Science"),
    ("cultura", "Culture"),
    ("culture", "Culture"),
    ("deportes", "Sports"),
    ("sports", "Sports"),
    ("opinion", "Opinion"),
  ]
  for token, canonical in mapping:
    if token in key:
      return canonical
  return "Spain" if region == "es" else "World"


def normalize_prose_block(text):
  if not text:
    return ""
  lines = []
  for raw in text.splitlines():
    line = raw.strip()
    if not line:
      lines.append("")
      continue
    line = re.sub(r"^\s*(?:[-*]+|\d+[\.\)])\s+", "", line)
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = line.replace("__", "")
    lines.append(line)
  out = "\n".join(lines)
  out = re.sub(r"\n{3,}", "\n\n", out)
  return out.strip()


def remove_redundant_intro(text, title, heading):
  if not text:
    return ""
  title_key = normalize_text_key(title)
  heading_key = normalize_text_key(heading)
  chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
  trimmed = []
  for i, chunk in enumerate(chunks):
    key = normalize_text_key(chunk)
    if i < 3 and (key == title_key or key == heading_key):
      continue
    if i < 3 and title_key and key.startswith(title_key):
      continue
    if i < 3 and heading_key and key.startswith(heading_key):
      continue
    trimmed.append(chunk)
  return "\n\n".join(trimmed).strip()


def clean_generated_text(text, heading=None):
  out = strip_internal_reasoning(text)
  out = normalize_prose_block(out)
  if heading:
    escaped = re.escape(heading)
    out = re.sub(rf"(?is)^\s*(?:#+\s*)?(?:\*\*)?{escaped}(?:\*\*)?(?:\s*[:\-].*)?\s*", "", out, count=1)
  chunks = [c.strip() for c in re.split(r"\n\s*\n", out) if c.strip()]
  cleaned = []
  for chunk in chunks:
    low = chunk.lower()
    if re.match(r"^(okay|vale|let me|i need to|the user wants|el usuario pide)\b", low):
      continue
    if re.match(r"^(first|second|third|next|finally),?\b", low):
      continue
    if re.match(r"(?i)^(references?|referencias?|verified sources|fuentes verificadas)$", chunk.strip()):
      continue
    if re.match(r"(?i)^(the title is|the section is|rules say|this section|alternatively)\b", chunk.strip()):
      continue
    if re.match(r"(?i)^\[\d+\]\s*https?://", chunk.strip()):
      continue
    if "http://" in chunk or "https://" in chunk:
      if word_count(chunk) < 30:
        continue
    if looks_like_meta_paragraph(chunk):
      continue
    cleaned.append(chunk)
  out = "\n\n".join(cleaned).strip()
  out = dedupe_paragraphs(out)
  # Remove methodological caveats that leak into publication text.
  sentences = re.split(r"(?<=[.!?])\s+", out)
  kept_sentences = []
  for sentence in sentences:
    low = sentence.lower()
    mentions_sources = re.search(r"\b(source|sources|catalog|catalogue|evidence)\b", low)
    caveat_tone = re.search(
      r"\b(available|provided|at hand|source set|cannot|can't|lack|insufficient|speculative|verifiable information)\b",
      low
    )
    if mentions_sources and caveat_tone:
      continue
    if re.search(r"\b(any (detailed|precise) analysis remains speculative|would be speculative)\b", low):
      continue
    kept_sentences.append(sentence)
  out = " ".join(kept_sentences).strip()
  out = re.sub(r"\n{3,}", "\n\n", out)
  out = re.sub(r"\*\*(.*?)\*\*", r"\1", out)
  out = re.sub(r"\[(\d{1,3})\]", "", out)
  out = re.sub(r"https?://\S+", "", out)
  out = re.sub(r"[ \t]{2,}", " ", out)
  out = re.sub(r" +([,.;:!?])", r"\1", out)
  out = strip_embedded_headings(out)
  return out


def clean_body_html(body_html):
  if not body_html:
    return ""
  blocks = re.findall(r"(?is)<h[23][^>]*>[\s\S]*?</h[23]>|<p[^>]*>[\s\S]*?</p>|<ul>[\s\S]*?</ul>", body_html)
  if not blocks:
    return re.sub(r"\[(\d{1,3})\]", "", body_html.strip())
  out = []
  seen_heading = set()
  seen_paragraph = set()
  for block in blocks:
    plain = html.unescape(re.sub(r"<[^>]+>", " ", block))
    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\[(\d{1,3})\]", "", plain)
    plain = re.sub(r"https?://\S+", "", plain)
    if not plain:
      continue
    key = normalize_text_key(plain)
    if not key:
      continue
    if has_internal_markers(plain) or looks_like_meta_paragraph(plain):
      continue
    if re.match(r"(?i)^(okay|vale|let me|i need to|the user wants|el usuario pide|the title is|the section is|rules say)\b", plain):
      continue
    if re.match(r"(?is)<h[23][^>]*>", block):
      if key in seen_heading:
        continue
      seen_heading.add(key)
      out.append(block.strip())
      continue
    if key in seen_paragraph:
      continue
    seen_paragraph.add(key)
    out.append(block.strip())
  return "\n".join(out)


def extract_json_candidates(text):
  if not text:
    return []
  src = strip_internal_reasoning(text)
  candidates = []
  stack = []
  start = None
  in_string = False
  escape = False
  for i, ch in enumerate(src):
    if in_string:
      if escape:
        escape = False
      elif ch == "\\":
        escape = True
      elif ch == '"':
        in_string = False
      continue
    if ch == '"':
      in_string = True
      continue
    if ch == "{":
      if not stack:
        start = i
      stack.append(ch)
    elif ch == "}":
      if stack:
        stack.pop()
        if not stack and start is not None:
          candidates.append(src[start:i + 1])
          start = None
  return candidates


def json_from_text(text):
  if not text:
    return None
  cleaned = strip_internal_reasoning(text)
  try:
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
      return parsed
  except Exception:
    pass
  for candidate in extract_json_candidates(cleaned):
    try:
      parsed = json.loads(candidate)
      if isinstance(parsed, dict):
        return parsed
    except Exception:
      continue
  return None


def safe_json_dumps(data, fallback="{}"):
  try:
    return json.dumps(data, ensure_ascii=False)
  except Exception:
    return fallback


def compact_claims_for_verify(claims, max_items=10, max_chars=220):
  if not isinstance(claims, list):
    return []
  out = []
  for item in claims:
    if not isinstance(item, dict):
      continue
    claim = clean_generated_text(sanitize(item.get("claim", "")) or sanitize(item.get("hecho", "")))
    ref = sanitize(item.get("source_ref", "")) or sanitize(item.get("fuente_ref", ""))
    url = normalize_url(item.get("source_url", "")) or normalize_url(item.get("fuente_url", ""))
    support = clean_generated_text(sanitize(item.get("support", "")) or sanitize(item.get("soporte", "")))
    if not claim:
      continue
    out.append({
      "claim": claim[:max_chars],
      "source_ref": ref[:8],
      "source_url": url[:220],
      "support": support[:max_chars]
    })
    if len(out) >= max_items:
      break
  return out


def word_count(text):
  return len(re.findall(r"\b\w+\b", text or ""))


def extract_numeric_tokens(text):
  tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", text or "")
  out = set()
  for token in tokens:
    out.add(token.replace(",", "."))
  return out


def strip_invalid_refs(text, max_ref):
  if not text:
    return ""

  def replace_ref(match):
    try:
      ref = int(match.group(1))
    except Exception:
      return ""
    if 1 <= ref <= max_ref:
      return f"[{ref}]"
    return ""

  cleaned = re.sub(r"\[(\d{1,3})\]", replace_ref, text)
  cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
  cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
  return cleaned.strip()


def slugify(text):
  base = unicodedata.normalize("NFKD", text or "")
  base = "".join(c for c in base if not unicodedata.combining(c))
  base = re.sub(r"[^a-zA-Z0-9]+", "-", base.lower()).strip("-")
  return base[:72] or "nota"


def deep_merge(base, updates):
  for key, value in updates.items():
    if isinstance(value, dict) and isinstance(base.get(key), dict):
      deep_merge(base[key], value)
    else:
      base[key] = value


def load_config(path):
  cfg = json.loads(json.dumps(DEFAULT_CONFIG))
  if path and os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
      user_cfg = json.load(f)
    deep_merge(cfg, user_cfg)
  if not cfg["search"].get("script_path"):
    cfg["search"]["script_path"] = os.path.join(os.path.dirname(__file__), "r_cli_websearch.py")
  return cfg


def atomic_write_json(path, data):
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)
  tmp_path = f"{path}.tmp"
  with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  os.replace(tmp_path, path)


def load_quarantine_state(path):
  if not os.path.exists(path):
    return {"entries": {}}
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
      if isinstance(data, dict) and isinstance(data.get("entries"), dict):
        return data
  except Exception:
    pass
  return {"entries": {}}


def save_quarantine_state(path, state):
  payload = state if isinstance(state, dict) else {"entries": {}}
  if not isinstance(payload.get("entries"), dict):
    payload["entries"] = {}
  atomic_write_json(path, payload)


class RCLILocalProvider:
  def __init__(self, script_path, num_results=8):
    self.script_path = script_path
    self.num_results = num_results

  def search(self, query):
    cmd = [self.script_path, query, str(self.num_results)]
    try:
      result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
      if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"return={result.returncode}")
      data = json.loads(result.stdout)
      return normalize_sources(data)
    except Exception as exc:
      log_warn(f"r_cli_local failed: {exc}")
      return []


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
    try:
      data = http_post_json(f"{self.base_url}/v1/skills/call", payload, timeout_sec=self.timeout_sec)
      return normalize_sources(data)
    except Exception as exc:
      log_warn(f"r_cli API failed: {exc}")
      return []


class RSSProvider:
  def __init__(self, feeds, timeout_sec=20):
    self.feeds = feeds
    self.timeout_sec = timeout_sec

  def search(self, query):
    terms = [t.strip().lower() for t in query.split() if t.strip()]
    results = []
    for feed in self.feeds:
      try:
        raw = http_get(feed, timeout_sec=self.timeout_sec)
        results.extend(parse_rss(raw, terms))
      except Exception:
        continue
    return results


def normalize_sources(data):
  if isinstance(data, dict):
    if "result" in data:
      data = data["result"]
    elif "output" in data:
      data = data["output"]
  if isinstance(data, str):
    return [{"title": "", "url": "", "snippet": data, "published": "", "source": "text"}]
  out = []
  if isinstance(data, list):
    for item in data:
      if not isinstance(item, dict):
        continue
      out.append({
        "title": sanitize(item.get("title", "")),
        "url": normalize_url(item.get("url", "")),
        "snippet": sanitize(item.get("snippet", "") or item.get("description", "")),
        "published": sanitize(item.get("published", "")),
        "source": sanitize(item.get("source", "r_cli"))
      })
  return out


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
    blob = f"{title} {desc}".lower()
    if terms:
      hits = sum(1 for t in terms if t in blob)
      min_hits = 1 if len(terms) <= 3 else max(2, len(terms) // 4)
      if hits < min_hits:
        continue
    out.append({
      "title": sanitize(title),
      "url": normalize_url(link),
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


def dedupe_sources(sources, max_items=8):
  out = []
  seen_url = set()
  for src in sources:
    url = normalize_url(src.get("url", ""))
    title = sanitize(src.get("title", ""))
    snippet = sanitize(src.get("snippet", ""))
    if not url or url in seen_url:
      continue
    if not title and not snippet:
      continue
    seen_url.add(url)
    out.append({
      "title": title,
      "url": url,
      "snippet": snippet,
      "published": sanitize(src.get("published", "")),
      "source": sanitize(src.get("source", "web"))
    })
    if len(out) >= max_items:
      break
  return out


def sort_sources_by_preferred_domains(sources, preferred_domains):
  if not sources:
    return sources
  preferred = [sanitize(d).lower().replace("www.", "") for d in (preferred_domains or []) if sanitize(d)]
  if not preferred:
    return sources
  rank_map = {domain: idx for idx, domain in enumerate(preferred)}

  def score(src):
    domain = domain_from_url(src.get("url", ""))
    if domain in rank_map:
      return (0, rank_map[domain])
    return (1, 10**6)

  return sorted(sources, key=score)


def cosine_similarity(vec_a, vec_b):
  if not isinstance(vec_a, list) or not isinstance(vec_b, list):
    return 0.0
  size = min(len(vec_a), len(vec_b))
  if size == 0:
    return 0.0
  dot = 0.0
  norm_a = 0.0
  norm_b = 0.0
  for i in range(size):
    try:
      a = float(vec_a[i])
      b = float(vec_b[i])
    except Exception:
      continue
    dot += a * b
    norm_a += a * a
    norm_b += b * b
  if norm_a <= 0.0 or norm_b <= 0.0:
    return 0.0
  return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def rerank_sources_by_embedding(lm, embedding_model, query, sources, preferred_domains=None):
  if not embedding_model or not sources:
    return sources
  query_text = sanitize(query)
  if not query_text:
    return sources
  texts = []
  for src in sources:
    title = sanitize(src.get("title", ""))
    snippet = sanitize(src.get("snippet", ""))
    domain = domain_from_url(src.get("url", ""))
    text = f"{title}. {snippet}. dominio {domain}".strip()
    texts.append(text[:1400])
  try:
    vectors = lm.embeddings(embedding_model, [query_text] + texts)
  except Exception as exc:
    log_warn(f"embedding rerank failed: {exc}")
    return sources
  if len(vectors) < len(texts) + 1:
    return sources

  preferred = [sanitize(d).lower().replace("www.", "") for d in (preferred_domains or []) if sanitize(d)]
  rank_map = {dom: idx for idx, dom in enumerate(preferred)}

  query_vec = vectors[0]
  scored = []
  for i, src in enumerate(sources, start=1):
    score = cosine_similarity(query_vec, vectors[i])
    dom = domain_from_url(src.get("url", ""))
    if dom in rank_map:
      bonus = 0.05 * ((len(preferred) - rank_map[dom]) / max(1, len(preferred)))
      score += bonus
    if sanitize(src.get("source", "")) == "market_api":
      score += 0.08
    item = dict(src)
    item["_score"] = round(score, 6)
    scored.append(item)
  scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
  return scored


def fetch_article_text(url, timeout_sec=20, limit_chars=220):
  if not url:
    return ""
  try:
    raw = http_get(url, timeout_sec=timeout_sec)
    html_text = raw.decode("utf-8", errors="ignore")
    html_text = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.I)
    html_text = re.sub(r"<style[\s\S]*?</style>", " ", html_text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = sanitize(text)
    return text[:limit_chars]
  except Exception:
    return ""


def is_market_topic(topic):
  text = normalize_text_key(topic)
  keywords = [
    "bolsa", "ibex", "mercado", "acciones", "stock", "indices", "indice",
    "cripto", "crypto", "bitcoin", "ethereum", "solana", "xrp", "bnb",
    "tipos de interes", "forex", "commodities", "materias primas"
  ]
  return any(k in text for k in keywords)


def pick_crypto_ids(topic):
  text = normalize_text_key(topic)
  mapping = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "cardano": "cardano",
    "dogecoin": "dogecoin",
    "xrp": "ripple",
    "bnb": "binancecoin"
  }
  ids = []
  for key, coin_id in mapping.items():
    if key in text and coin_id not in ids:
      ids.append(coin_id)
  if not ids and ("cripto" in text or "crypto" in text):
    ids = ["bitcoin", "ethereum", "solana"]
  return ids[:6]


def pick_stock_symbols(topic, region):
  text = normalize_text_key(topic)
  mapping = {
    "ibex": "^IBEX",
    "banco santander": "SAN.MC",
    "bbva": "BBVA.MC",
    "caixabank": "CABK.MC",
    "telefonica": "TEF.MC",
    "inditex": "ITX.MC",
    "repsol": "REP.MC",
    "aapl": "AAPL",
    "apple": "AAPL",
    "msft": "MSFT",
    "microsoft": "MSFT",
    "nvda": "NVDA",
    "nvidia": "NVDA",
    "amzn": "AMZN",
    "googl": "GOOGL",
    "meta": "META",
    "tsla": "TSLA"
  }
  symbols = []
  for key, symbol in mapping.items():
    if key in text and symbol not in symbols:
      symbols.append(symbol)
  if not symbols and is_market_topic(topic):
    if region == "es":
      symbols = ["^IBEX", "SAN.MC", "BBVA.MC"]
    else:
      symbols = ["^GSPC", "^IXIC", "^DJI"]
  return symbols[:6]


def fetch_json_url(url, timeout_sec=20):
  try:
    raw = http_get(url, timeout_sec=timeout_sec)
    return json.loads(raw.decode("utf-8", errors="ignore"))
  except Exception:
    return None


def fetch_crypto_realtime(ids):
  if not ids:
    return []
  ids_csv = ",".join(ids)
  url = (
    "https://api.coingecko.com/api/v3/simple/price"
    f"?ids={urllib.parse.quote(ids_csv)}&vs_currencies=usd"
    "&include_24hr_change=true&include_market_cap=true"
  )
  data = fetch_json_url(url, timeout_sec=20)
  if not isinstance(data, dict) or not data:
    return []
  now_iso = utc_now_iso()
  out = []
  for coin_id, info in data.items():
    if not isinstance(info, dict):
      continue
    price = info.get("usd")
    change = info.get("usd_24h_change")
    mcap = info.get("usd_market_cap")
    parts = []
    if isinstance(price, (int, float)):
      parts.append(f"precio USD {round(float(price), 4)}")
    if isinstance(change, (int, float)):
      parts.append(f"variacion 24h {round(float(change), 2)}%")
    if isinstance(mcap, (int, float)):
      parts.append(f"capitalizacion {int(mcap)}")
    if not parts:
      continue
    snippet = f"{coin_id}: " + ", ".join(parts) + "."
    out.append({
      "title": f"Cotizacion cripto en tiempo real: {coin_id}",
      "url": url,
      "snippet": snippet,
      "published": now_iso,
      "source": "market_api"
    })
  return out


def fetch_yahoo_quote(symbol):
  enc_symbol = urllib.parse.quote(symbol, safe="")
  url = f"https://query1.finance.yahoo.com/v8/finance/chart/{enc_symbol}?interval=1d&range=5d"
  data = fetch_json_url(url, timeout_sec=20)
  try:
    chart = data.get("chart", {}).get("result", [{}])[0]
    meta = chart.get("meta", {})
    price = meta.get("regularMarketPrice")
    prev = meta.get("previousClose", price)
    currency = sanitize(meta.get("currency", "USD"))
    state = sanitize(meta.get("marketState", ""))
    exch = sanitize(meta.get("exchangeName", ""))
    if not isinstance(price, (int, float)):
      return None
    change = 0.0
    if isinstance(prev, (int, float)) and prev:
      change = ((float(price) - float(prev)) / float(prev)) * 100.0
    snippet = (
      f"{symbol}: precio {round(float(price), 4)} {currency}, "
      f"variacion diaria {round(change, 2)}%, mercado {state or 'N/A'}, "
      f"bolsa {exch or 'N/A'}."
    )
    return {
      "title": f"Cotizacion bursatil en tiempo real: {symbol}",
      "url": url,
      "snippet": snippet,
      "published": utc_now_iso(),
      "source": "market_api"
    }
  except Exception:
    return None


def fetch_market_indices():
  symbols = ["^IBEX", "^GSPC", "^IXIC", "^DJI", "^VIX"]
  out = []
  for symbol in symbols:
    quote = fetch_yahoo_quote(symbol)
    if quote:
      out.append(quote)
  return out


def fetch_forex_snapshot():
  pair_symbols = ["EURUSD=X", "GBPUSD=X", "JPY=X"]
  out = []
  for symbol in pair_symbols:
    quote = fetch_yahoo_quote(symbol)
    if quote:
      quote["title"] = f"Divisas en tiempo real: {symbol}"
      out.append(quote)
  return out


def realtime_market_sources(topic, region, max_items=8):
  if not is_market_topic(topic):
    return []
  out = []
  crypto_ids = pick_crypto_ids(topic)
  stock_symbols = pick_stock_symbols(topic, region)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [
      ex.submit(fetch_crypto_realtime, crypto_ids),
      ex.submit(lambda: [q for q in (fetch_yahoo_quote(s) for s in stock_symbols) if q]),
      ex.submit(fetch_market_indices),
      ex.submit(fetch_forex_snapshot)
    ]
    for fut in concurrent.futures.as_completed(futures):
      try:
        items = fut.result() or []
        if isinstance(items, list):
          out.extend(items)
      except Exception:
        continue
  out = dedupe_sources(out, max_items=max_items)
  return out[:max_items]


def fallback_outline(topic):
  return {
    "title": topic.title(),
    "deck": "In-depth analysis for readers in Spain.",
    "sections": [
      {"heading": "Key Facts", "focus": "Main verified facts and immediate context"},
      {"heading": "Spain Impact", "focus": "Concrete effects on households, companies and institutions"},
      {"heading": "International Context", "focus": "Comparison with Europe and global markets"},
      {"heading": "Risks And Outlook", "focus": "Threats, opportunities and short-term watchpoints"}
    ]
  }


def normalize_sections_from_outline(outline_json, topic):
  fallback = fallback_outline(topic)["sections"][:4]
  raw_sections = outline_json.get("sections") if isinstance(outline_json, dict) else None
  if not isinstance(raw_sections, list):
    return fallback
  cleaned = []
  seen = set()
  for item in raw_sections:
    if not isinstance(item, dict):
      continue
    heading = clean_generated_text(sanitize(item.get("heading", "")))
    focus = clean_generated_text(sanitize(item.get("focus", "")))
    if not heading or not focus:
      continue
    if looks_mostly_spanish(f"{heading} {focus}"):
      continue
    heading = re.sub(r"[:\-]+$", "", heading).strip()
    key = normalize_text_key(heading)
    if not key or key in seen:
      continue
    seen.add(key)
    cleaned.append({
      "heading": heading[:68],
      "focus": focus[:220]
    })
    if len(cleaned) >= 4:
      break
  if len(cleaned) < 3:
    return fallback
  while len(cleaned) < 4:
    cleaned.append(fallback[len(cleaned)])
  return cleaned[:4]


def fallback_section_text(section, summary, topic):
  focus = section.get("focus", "context")
  base = (
    f"The analysis examines {focus.lower()} in relation to {topic}. "
    "The practical interpretation for Spain is to distinguish structural shifts from short-term volatility, "
    "and to evaluate outcomes through measurable indicators rather than headlines alone. "
    "Public policy, financing conditions, and execution by institutions and companies will determine real impact over the next quarters. "
    "A robust editorial approach is to track implementation quality, compare scenarios, and keep uncertainty explicit where evidence is still developing."
  )
  return clean_generated_text(base)


def dedupe_markdown_sections(md_text):
  if not md_text:
    return ""
  sections = []
  current_heading = None
  current_lines = []
  for raw in md_text.splitlines():
    if raw.startswith("## "):
      if current_heading:
        sections.append((current_heading, "\n".join(current_lines).strip()))
      current_heading = raw[3:].strip()
      current_lines = []
    else:
      current_lines.append(raw)
  if current_heading:
    sections.append((current_heading, "\n".join(current_lines).strip()))

  out = []
  seen_heading = set()
  global_seen = set()
  for heading, content in sections:
    heading_key = normalize_text_key(heading)
    if not heading_key or heading_key in seen_heading:
      continue
    seen_heading.add(heading_key)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    clean_paragraphs = []
    for paragraph in paragraphs:
      cleaned = clean_generated_text(paragraph, heading=heading)
      if not cleaned or looks_like_meta_paragraph(cleaned):
        continue
      key = normalize_text_key(cleaned)
      if not key or key in global_seen:
        continue
      global_seen.add(key)
      clean_paragraphs.append(cleaned)
    if clean_paragraphs:
      out.append(f"## {heading}\n\n" + "\n\n".join(clean_paragraphs))
  return "\n\n".join(out).strip()


def markdown_to_html(md_text):
  lines = (md_text or "").splitlines()
  parts = []
  paragraph = []
  in_list = False

  def flush_paragraph():
    nonlocal paragraph
    if paragraph:
      txt = " ".join(paragraph).strip()
      if txt:
        parts.append(f"<p>{html.escape(txt)}</p>")
      paragraph = []

  def close_list():
    nonlocal in_list
    if in_list:
      parts.append("</ul>")
      in_list = False

  for raw in lines:
    line = raw.strip()
    if not line:
      flush_paragraph()
      close_list()
      continue
    if line.startswith("## "):
      flush_paragraph()
      close_list()
      parts.append(f"<h3>{html.escape(line[3:].strip())}</h3>")
      continue
    if line.startswith("# "):
      flush_paragraph()
      close_list()
      parts.append(f"<h2>{html.escape(line[2:].strip())}</h2>")
      continue
    if line.startswith("- "):
      flush_paragraph()
      if not in_list:
        parts.append("<ul>")
        in_list = True
      parts.append(f"<li>{html.escape(line[2:].strip())}</li>")
      continue
    close_list()
    paragraph.append(line)

  flush_paragraph()
  close_list()
  html_text = "\n".join(parts)
  return clean_body_html(html_text)


def run_pipeline(cfg, topic, region, max_sources, min_words):
  lm_cfg = cfg.get("lmstudio", {})
  default_base_url = sanitize(lm_cfg.get("base_url", ""))
  default_timeout = int(lm_cfg.get("timeout_sec", 120))
  pipeline_max_sec = int(cfg.get("pipeline_max_sec", 260))
  quarantine_seconds = int(cfg.get("quarantine_seconds", 900))
  quarantine_file = sanitize(cfg.get("quarantine_file", "")) or os.path.join(
    os.path.dirname(__file__),
    "logs",
    "route_quarantine.json"
  )
  pipeline_deadline = time.time() + max(90, pipeline_max_sec)
  models_cfg = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
  routes_cfg = cfg.get("routes", {}) if isinstance(cfg.get("routes"), dict) else {}

  client_cache = {}
  model_catalog_cache = {}
  skipped_route_warned = set()
  quarantine = load_quarantine_state(quarantine_file)
  quarantined_warned = set()

  def route_key(role, base_url, model):
    return f"{sanitize(role)}|{sanitize(base_url).rstrip('/')}|{sanitize(model)}"

  def prune_quarantine(now_ts):
    entries = quarantine.get("entries", {})
    changed = False
    if not isinstance(entries, dict):
      quarantine["entries"] = {}
      changed = True
      try:
        save_quarantine_state(quarantine_file, quarantine)
      except Exception:
        pass
      return
    remove = []
    for key, item in entries.items():
      if not isinstance(item, dict):
        remove.append(key)
        continue
      if float(item.get("until_ts", 0)) <= now_ts:
        remove.append(key)
    for key in remove:
      entries.pop(key, None)
      changed = True
    if changed:
      try:
        save_quarantine_state(quarantine_file, quarantine)
      except Exception:
        pass

  def is_quarantined(role, base_url, model, now_ts):
    entries = quarantine.get("entries", {})
    if not isinstance(entries, dict):
      return False
    key = route_key(role, base_url, model)
    entry = entries.get(key)
    if not isinstance(entry, dict):
      return False
    return float(entry.get("until_ts", 0)) > now_ts

  def mark_quarantine(role, base_url, model, reason):
    if quarantine_seconds <= 0:
      return
    now_ts = time.time()
    entries = quarantine.get("entries", {})
    if not isinstance(entries, dict):
      entries = {}
      quarantine["entries"] = entries
    key = route_key(role, base_url, model)
    prev = entries.get(key, {}) if isinstance(entries.get(key), dict) else {}
    fail_count = int(prev.get("fail_count", 0)) + 1
    cooldown = min(quarantine_seconds * max(1, fail_count), quarantine_seconds * 4)
    entries[key] = {
      "role": sanitize(role),
      "base_url": sanitize(base_url).rstrip("/"),
      "model": sanitize(model),
      "reason": sanitize(reason)[:220],
      "fail_count": fail_count,
      "last_fail_ts": now_ts,
      "until_ts": now_ts + cooldown
    }
    try:
      save_quarantine_state(quarantine_file, quarantine)
    except Exception as exc:
      log_warn(f"could not persist quarantine state: {exc}")

  def clear_quarantine(role, base_url, model):
    entries = quarantine.get("entries", {})
    if not isinstance(entries, dict):
      return
    key = route_key(role, base_url, model)
    if key in entries:
      entries.pop(key, None)
      quarantined_warned.discard(key)
      try:
        save_quarantine_state(quarantine_file, quarantine)
      except Exception:
        pass

  def get_client(base_url, timeout_sec):
    base = sanitize(base_url).rstrip("/")
    if not base:
      raise RuntimeError("LM Studio base_url is empty.")
    timeout = int(timeout_sec) if timeout_sec else default_timeout
    key = (base, timeout)
    if key not in client_cache:
      client_cache[key] = LMStudioClient(base, timeout_sec=timeout)
    return client_cache[key]

  def loaded_models(base_url, timeout_sec):
    base = sanitize(base_url).rstrip("/")
    timeout = int(timeout_sec) if timeout_sec else default_timeout
    key = (base, timeout)
    if key in model_catalog_cache:
      return model_catalog_cache[key]
    models = None
    try:
      raw = http_get(f"{base}/models", timeout_sec=min(timeout, 8))
      parsed = json.loads(raw.decode("utf-8", errors="ignore"))
      rows = parsed.get("data", []) if isinstance(parsed, dict) else []
      ids = set()
      for row in rows:
        if isinstance(row, dict):
          model_id = sanitize(row.get("id", ""))
          if model_id:
            ids.add(model_id)
      if ids:
        models = ids
    except Exception:
      models = None
    model_catalog_cache[key] = models
    return models

  def invalidate_model_catalog(base_url):
    base = sanitize(base_url).rstrip("/")
    if not base:
      return
    drop_keys = [k for k in model_catalog_cache.keys() if k[0] == base]
    for key in drop_keys:
      model_catalog_cache.pop(key, None)

  def is_embedding_model(model_name):
    key = normalize_text_key(model_name)
    if not key:
      return False
    return "embedding" in key or key.startswith("embed ")

  def is_model_unavailable_error(exc):
    low = sanitize(str(exc)).lower()
    tokens = [
      "model unloaded",
      "model not found",
      "unknown model",
      "failed to load model",
      "is not loaded",
      "model does not exist",
    ]
    return any(token in low for token in tokens)

  def role_model_rank(role, model_name):
    key = normalize_text_key(model_name)
    if not key:
      return 9999
    if is_embedding_model(model_name):
      return 9999

    score = 90
    if "gpt oss 20b" in key:
      score = 10
    elif "ministral" in key or "mistral" in key:
      score = 20
    elif "qwen" in key:
      score = 30
    elif "gemma" in key:
      score = 40
    elif "glm" in key:
      score = 50
    elif "phi" in key:
      score = 60

    if "mini" in key or "flash" in key:
      score += 8

    if role == "fact" and ("mistral" in key or "ministral" in key):
      score -= 3
    if role in ("chief", "research") and "gpt oss 20b" in key:
      score -= 2
    return max(1, score)

  def known_generation_routes():
    routes = []
    if default_base_url:
      routes.append({"base_url": default_base_url, "timeout_sec": default_timeout})

    for role_name, route in routes_cfg.items():
      if role_name == "embedding":
        continue
      if not isinstance(route, dict):
        continue
      base = sanitize(route.get("base_url", "")) or default_base_url
      timeout = int(route.get("timeout_sec", default_timeout))
      model_hint = sanitize(route.get("model", ""))
      if not base:
        continue
      if model_hint and is_embedding_model(model_hint):
        continue
      routes.append({"base_url": base.rstrip("/"), "timeout_sec": timeout})

    seen = set()
    deduped = []
    for item in routes:
      key = (item["base_url"], int(item["timeout_sec"]))
      if key in seen:
        continue
      seen.add(key)
      deduped.append(item)
    return deduped

  def dynamic_role_attempts(role, max_models_per_base=3):
    out = []
    for route in known_generation_routes():
      models = loaded_models(route["base_url"], route["timeout_sec"])
      if not isinstance(models, set) or not models:
        continue
      candidates = [m for m in models if not is_embedding_model(m)]
      if not candidates:
        continue
      candidates.sort(key=lambda m: role_model_rank(role, m))
      for model_name in candidates[:max_models_per_base]:
        out.append({
          "base_url": route["base_url"],
          "model": model_name,
          "timeout_sec": route["timeout_sec"]
        })
    return out

  def filter_loaded_attempts(attempts):
    out = []
    for attempt in attempts:
      models = loaded_models(attempt["base_url"], attempt["timeout_sec"])
      if models is not None and attempt["model"] not in models:
        key = (attempt["base_url"], attempt["model"])
        if key not in skipped_route_warned:
          skipped_route_warned.add(key)
          log_warn(f"skip route {attempt['base_url']} model '{attempt['model']}' not loaded")
        continue
      out.append(attempt)
    return out

  def deadline_remaining():
    return pipeline_deadline - time.time()

  def dedupe_attempts(attempts):
    seen = set()
    out = []
    for item in attempts:
      base = sanitize(item.get("base_url", "")).rstrip("/")
      model = sanitize(item.get("model", ""))
      timeout = int(item.get("timeout_sec", default_timeout))
      if not base or not model:
        continue
      key = (base, model, timeout)
      if key in seen:
        continue
      seen.add(key)
      out.append({"base_url": base, "model": model, "timeout_sec": timeout})
    return out

  def role_attempts(role):
    attempts = []
    route = routes_cfg.get(role, {}) if isinstance(routes_cfg.get(role), dict) else {}
    route_model = sanitize(route.get("model", "")) or sanitize(models_cfg.get(role, ""))
    route_base = sanitize(route.get("base_url", "")) or default_base_url
    route_timeout = int(route.get("timeout_sec", default_timeout))
    if route_base and route_model:
      attempts.append({"base_url": route_base, "model": route_model, "timeout_sec": route_timeout})

    default_model = sanitize(models_cfg.get(role, "")) or sanitize(models_cfg.get("chief", ""))
    if default_base_url and default_model:
      attempts.append({"base_url": default_base_url, "model": default_model, "timeout_sec": default_timeout})

    if role != "chief":
      chief_route = routes_cfg.get("chief", {}) if isinstance(routes_cfg.get("chief"), dict) else {}
      chief_model = sanitize(chief_route.get("model", "")) or sanitize(models_cfg.get("chief", ""))
      chief_base = sanitize(chief_route.get("base_url", "")) or default_base_url
      chief_timeout = int(chief_route.get("timeout_sec", default_timeout))
      if chief_base and chief_model:
        attempts.append({"base_url": chief_base, "model": chief_model, "timeout_sec": chief_timeout})

    # Automatic model fallback: if primary model is unavailable/unloaded, use loaded models on known nodes.
    attempts.extend(dynamic_role_attempts(role))

    now_ts = time.time()
    prune_quarantine(now_ts)
    active_attempts = []
    for attempt in filter_loaded_attempts(dedupe_attempts(attempts)):
      key = route_key(role, attempt["base_url"], attempt["model"])
      if is_quarantined(role, attempt["base_url"], attempt["model"], now_ts):
        if key not in quarantined_warned:
          quarantined_warned.add(key)
          entries = quarantine.get("entries", {})
          item = entries.get(key, {}) if isinstance(entries, dict) else {}
          until_ts = float(item.get("until_ts", 0)) if isinstance(item, dict) else 0
          remain = int(max(0, round(until_ts - now_ts)))
          log_warn(
            f"route quarantined role={role} base={attempt['base_url']} model={attempt['model']} "
            f"remaining={remain}s"
          )
        continue
      active_attempts.append(attempt)
    return active_attempts

  def embedding_attempts():
    attempts = []
    route = routes_cfg.get("embedding", {}) if isinstance(routes_cfg.get("embedding"), dict) else {}
    route_model = sanitize(route.get("model", "")) or sanitize(models_cfg.get("embedding", ""))
    route_base = sanitize(route.get("base_url", "")) or default_base_url
    route_timeout = int(route.get("timeout_sec", default_timeout))
    if route_base and route_model:
      attempts.append({"base_url": route_base, "model": route_model, "timeout_sec": route_timeout})

    default_model = sanitize(models_cfg.get("embedding", ""))
    if default_base_url and default_model:
      attempts.append({"base_url": default_base_url, "model": default_model, "timeout_sec": default_timeout})

    now_ts = time.time()
    prune_quarantine(now_ts)
    active_attempts = []
    for attempt in filter_loaded_attempts(dedupe_attempts(attempts)):
      key = route_key("embedding", attempt["base_url"], attempt["model"])
      if is_quarantined("embedding", attempt["base_url"], attempt["model"], now_ts):
        if key not in quarantined_warned:
          quarantined_warned.add(key)
          entries = quarantine.get("entries", {})
          item = entries.get(key, {}) if isinstance(entries, dict) else {}
          until_ts = float(item.get("until_ts", 0)) if isinstance(item, dict) else 0
          remain = int(max(0, round(until_ts - now_ts)))
          log_warn(
            f"route quarantined role=embedding base={attempt['base_url']} "
            f"model={attempt['model']} remaining={remain}s"
          )
        continue
      active_attempts.append(attempt)
    return active_attempts

  provider_name = cfg.get("search", {}).get("provider", "r_cli_local")
  if provider_name == "r_cli":
    searcher = RCLISearchProvider(
      cfg["search"].get("r_cli_base_url", "http://localhost:8765"),
      cfg["search"].get("skill", "websearch"),
      cfg["search"].get("tool", "search"),
      cfg["search"].get("args", {}),
      timeout_sec=30
    )
  elif provider_name == "rss":
    feeds = []
    if region in ("es", "both"):
      feeds.extend(cfg.get("rss_feeds", {}).get("spain", []))
    if region in ("world", "both"):
      feeds.extend(cfg.get("rss_feeds", {}).get("world", []))
    searcher = RSSProvider(feeds)
  else:
    script_path = cfg.get("search", {}).get("script_path") or os.path.join(os.path.dirname(__file__), "r_cli_websearch.py")
    num_results = int(cfg.get("search", {}).get("args", {}).get("num_results", 8))
    searcher = RCLILocalProvider(script_path, num_results=num_results)

  query = topic
  if region == "es":
    query = f"{topic} Espana"
  elif region == "world":
    query = f"{topic} Europe and world"

  preferred_domains = cfg.get("search", {}).get("preferred_domains", [])
  queries = [query]
  for domain in preferred_domains[:8]:
    dom = sanitize(domain).replace("www.", "")
    if not dom:
      continue
    queries.append(f"{query} site:{dom}")

  gathered = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, max(1, len(queries)))) as ex:
    futures = [ex.submit(searcher.search, q) for q in queries]
    for fut in concurrent.futures.as_completed(futures):
      try:
        chunk = fut.result() or []
        if isinstance(chunk, list):
          gathered.extend(chunk)
      except Exception:
        continue

  # If query variants are too narrow, perform a relaxed second pass.
  if len(gathered) < max(4, max_sources * 2):
    relaxed_queries = [topic]
    if region == "es":
      relaxed_queries.extend([f"{topic} Espana", f"{topic} España"])
    elif region == "world":
      relaxed_queries.extend([f"{topic} global", f"{topic} international"])
    unique_relaxed = []
    seen_relaxed = set()
    for q in relaxed_queries:
      cleaned = sanitize(q)
      if cleaned and cleaned not in seen_relaxed:
        seen_relaxed.add(cleaned)
        unique_relaxed.append(cleaned)
    if unique_relaxed:
      with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(unique_relaxed))) as ex:
        futures = [ex.submit(searcher.search, q) for q in unique_relaxed]
        for fut in concurrent.futures.as_completed(futures):
          try:
            chunk = fut.result() or []
            if isinstance(chunk, list):
              gathered.extend(chunk)
          except Exception:
            continue

  candidate_pool = max(max_sources * 4, 24)
  sources = dedupe_sources(gathered, max_items=candidate_pool)
  sources = sort_sources_by_preferred_domains(sources, preferred_domains)

  # Robust fallback: if R CLI does not return enough, backfill with RSS.
  if len(sources) < min(3, max_sources):
    feeds = []
    if region in ("es", "both"):
      feeds.extend(cfg.get("rss_feeds", {}).get("spain", []))
    if region in ("world", "both"):
      feeds.extend(cfg.get("rss_feeds", {}).get("world", []))
    if feeds:
      rss_searcher = RSSProvider(feeds)
      rss_results = rss_searcher.search(query)
      if not rss_results:
        rss_results = rss_searcher.search("")
      sources = dedupe_sources(sources + rss_results, max_items=max(candidate_pool, max_sources * 5))
      sources = sort_sources_by_preferred_domains(sources, preferred_domains)

  # Real-time market data: CoinGecko for crypto + Yahoo Finance for stocks/indices/forex.
  market_sources = realtime_market_sources(topic, region, max_items=max(4, max_sources))
  if market_sources:
    # Priorizamos datos live para temas de bolsa/cripto.
    sources = dedupe_sources(market_sources + sources, max_items=max(candidate_pool, max_sources * 5))

  embed_routes = embedding_attempts()
  if embed_routes:
    for attempt in embed_routes:
      try:
        embed_client = get_client(attempt["base_url"], attempt["timeout_sec"])
        sources = rerank_sources_by_embedding(
          embed_client,
          embedding_model=attempt["model"],
          query=f"{topic} {region}",
          sources=sources,
          preferred_domains=preferred_domains
        )
        clear_quarantine("embedding", attempt["base_url"], attempt["model"])
        break
      except Exception as exc:
        log_warn(f"embedding failed on {attempt['base_url']} ({attempt['model']}): {exc}")
        mark_quarantine("embedding", attempt["base_url"], attempt["model"], str(exc))

  sources = sources[:max_sources]

  if not sources:
    fallback_note = {
      "title": f"Coverage setup note: {topic}",
      "url": "https://www.reuters.com/world/",
      "snippet": "Automated fallback activated because primary search returned no usable results.",
      "published": utc_now_iso(),
      "source": "system_fallback"
    }
    sources = [fallback_note]

  fetch_workers = min(6, max(1, len(sources)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_workers) as ex:
    fetched = list(ex.map(lambda s: fetch_article_text(s.get("url", "")), sources))

  for i, (src, content) in enumerate(zip(sources, fetched), 1):
    src["rank"] = i
    src["domain"] = domain_from_url(src.get("url", ""))
    src["content"] = content or src.get("snippet", "")

  source_lines = []
  evidence_blocks = []
  for src in sources:
    i = src["rank"]
    source_lines.append(f"[{i}] {src.get('title', '')} ({src.get('domain', '')})")
    evidence_blocks.append(
      f"Source [{i}]\n"
      f"Title: {src.get('title', '')}\n"
      f"Snippet: {src.get('snippet', '')}\n"
      f"Text: {src.get('content', '')[:140]}"
    )
  sources_catalog = "\n".join(source_lines)
  evidence_blob = "\n\n".join(evidence_blocks)
  prompt_catalog = sources_catalog[:1800]
  prompt_evidence = evidence_blob[:3000]
  source_overlap_index = build_source_overlap_index(sources)
  source_titles = source_overlap_index.get("titles", [])

  def model_json(model_key, system_prompt, user_prompt, fallback):
    editorial_system = f"{MASTER_EDITORIAL_PROMPT}\n\n{system_prompt}".strip()
    messages = [
      {"role": "system", "content": editorial_system},
      {"role": "user", "content": user_prompt}
    ]
    attempts = role_attempts(model_key)
    if not attempts:
      log_warn(f"{model_key} has no configured route/model; using fallback")
      return fallback
    if deadline_remaining() <= 0:
      log_warn(f"{model_key} skipped due to pipeline deadline; using fallback")
      return fallback
    for attempt in attempts:
      if deadline_remaining() <= 0:
        log_warn(f"{model_key} stopped by pipeline deadline; using fallback")
        return fallback
      try:
        client = get_client(attempt["base_url"], attempt["timeout_sec"])
        raw = chat_with_retry(
          client,
          attempt["model"],
          messages,
          temperature=0.05,
          max_tokens=520,
          retries=1
        )
        parsed = json_from_text(raw)
        if isinstance(parsed, dict):
          clear_quarantine(model_key, attempt["base_url"], attempt["model"])
          return parsed
        log_warn(
          f"{model_key} returned invalid JSON on {attempt['base_url']} "
          f"({attempt['model']}); trying route fallback"
        )
        # Do not quarantine on format errors; keep the route available for prose tasks.
        continue
      except Exception as exc:
        log_warn(f"{model_key} failed on {attempt['base_url']} ({attempt['model']}): {exc}")
        if is_model_unavailable_error(exc):
          invalidate_model_catalog(attempt["base_url"])
          continue
        mark_quarantine(model_key, attempt["base_url"], attempt["model"], str(exc))
    return fallback

  def model_text(model_key, system_prompt, user_prompt, temperature=0.12, max_tokens=720, retries=1):
    editorial_system = f"{MASTER_EDITORIAL_PROMPT}\n\n{system_prompt}".strip()
    messages = [
      {"role": "system", "content": editorial_system},
      {"role": "user", "content": user_prompt}
    ]
    attempts = role_attempts(model_key)
    if not attempts:
      raise RuntimeError(f"{model_key} has no configured route/model.")
    if deadline_remaining() <= 0:
      raise RuntimeError(f"{model_key} aborted due to pipeline deadline.")
    last_exc = None
    for attempt in attempts:
      if deadline_remaining() <= 0:
        raise RuntimeError(f"{model_key} aborted due to pipeline deadline.")
      try:
        client = get_client(attempt["base_url"], attempt["timeout_sec"])
        result = chat_with_retry(
          client,
          attempt["model"],
          messages,
          temperature=temperature,
          max_tokens=max_tokens,
          retries=retries
        )
        clear_quarantine(model_key, attempt["base_url"], attempt["model"])
        return result
      except Exception as exc:
        last_exc = exc
        log_warn(f"{model_key} failed on {attempt['base_url']} ({attempt['model']}): {exc}")
        if is_model_unavailable_error(exc):
          invalidate_model_catalog(attempt["base_url"])
          continue
        mark_quarantine(model_key, attempt["base_url"], attempt["model"], str(exc))
    raise last_exc or RuntimeError(f"{model_key} has no valid response.")

  summary_fallback = {
    "bullets": [
      "Structured summary was unavailable; evidence-based synthesis is being used.",
      "Coverage prioritizes verifiable facts and practical implications for Spain."
    ],
    "spain_context": "The story affects households, companies and institutions in Spain in different ways.",
    "tone": "analytical"
  }
  facts_fallback = {"claims": []}
  tag_fallback = {
    "section": "Spain" if region == "es" else "World",
    "category": "General",
    "tags": ["spain", "news"],
    "kicker": "Editorial Analysis"
  }
  outline_fallback = fallback_outline(topic)

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    f_summary = ex.submit(
      model_json,
      "research",
      "You are a newsroom research editor. Return ONLY valid JSON with no visible reasoning.",
      (
        "Task: summarize the sources for front-page writing in English for readers in Spain.\n"
        f"Topic: {topic}\nEditorial region: {region}\n"
        "Return JSON with keys: bullets (5 to 7), spain_context (string), tone (string).\n"
        "Do not invent facts. If certainty is low, state it clearly.\n"
        f"Source catalog:\n{prompt_catalog}\n\nEvidence:\n{prompt_evidence}"
      ),
      summary_fallback
    )
    f_facts = ex.submit(
      model_json,
      "fact",
      "You are a fact-check editor. Return ONLY valid JSON with no visible reasoning.",
      (
        "Extract verifiable claims using ONLY the provided evidence.\n"
        "Return JSON with claims: list of objects {claim, source_ref, source_url, support}.\n"
        "source_ref must be a number between 1 and N.\n"
        "Do not add information that is not in the sources.\n"
        f"Catalog:\n{prompt_catalog}\n\nEvidence:\n{prompt_evidence}"
      ),
      facts_fallback
    )
    f_tag = ex.submit(
      model_json,
      "tagger",
      "You are a section editor. Return ONLY valid JSON with no visible reasoning.",
      (
        "Classify the piece for an English-language newspaper focused on Spain.\n"
        "Return JSON with section, category, tags (short list), kicker.\n"
        "Allowed sections: Spain, World, Business, Politics, Society, Technology, Science, Culture, Sports, Opinion.\n"
        f"Topic: {topic}\nSource summary:\n{prompt_catalog}"
      ),
      tag_fallback
    )
    f_outline = ex.submit(
      model_json,
      "chief",
      "You are the editorial director. Return ONLY valid JSON with no visible reasoning.",
      (
        "Design a long-form newsroom piece in English for readers in Spain.\n"
        "Return JSON with title, deck and sections (4 objects with heading and focus).\n"
        "Style: analytical, direct, verifiable, clear and concise.\n"
        f"Topic: {topic}\nRegion: {region}\n"
        f"Source catalog:\n{prompt_catalog}"
      ),
      outline_fallback
    )

  summary_json = f_summary.result()
  facts_json = f_facts.result()
  tag_json = f_tag.result()
  outline_json = f_outline.result()

  verify_seed_claims = compact_claims_for_verify(facts_json.get("claims", []), max_items=10, max_chars=220)
  verify_fallback = {"verified": verify_seed_claims, "unverified": []}
  verify_json = model_json(
    "fact",
    "You are a senior fact-checker. Return ONLY valid JSON with no visible reasoning.",
    (
      "Validate and refine the extracted claims.\n"
      "Return JSON with verified and unverified.\n"
      "Each verified item must include: claim, source_ref, source_url, short_justification.\n"
      "Do not invent facts.\n"
      f"Candidate claims:\n{safe_json_dumps({'claims': verify_seed_claims}, fallback='{}')}\n\nCatalog:\n{prompt_catalog}\n"
    ),
    verify_fallback
  )

  def ensure_english_short(text, fallback, role="chief", max_words=22):
    value = clean_generated_text(sanitize(text))
    value = re.sub(r"\s+", " ", value).strip()
    if looks_invalid_short_text(value):
      value = ""
    if value and word_count(value) > max_words:
      value = " ".join(value.split()[:max_words]).strip(" ,;:-")
    if not value:
      value = fallback
    if not looks_mostly_spanish(value) and not looks_invalid_short_text(value):
      return value
    try:
      translated = model_text(
        role,
        "You are a newsroom language editor. Return only final English text.",
        (
          "Rewrite this short newsroom line in natural English.\n"
          "Keep meaning and editorial tone.\n"
          "No citations, no URLs, no brackets.\n\n"
          f"Text: {value}"
        ),
        temperature=0.05,
        max_tokens=90,
        retries=0
      )
      translated = clean_generated_text(translated)
      translated = re.sub(r"\s+", " ", translated).strip()
      if translated and word_count(translated) > max_words:
        translated = " ".join(translated.split()[:max_words]).strip(" ,;:-")
      if translated and not looks_mostly_spanish(translated) and not looks_invalid_short_text(translated):
        return translated
    except Exception:
      pass
    return fallback

  title = ensure_english_short(outline_json.get("title", ""), "Metropolis Analysis", max_words=14)
  deck = ensure_english_short(
    outline_json.get("deck", ""),
    "In-depth analysis for readers in Spain.",
    max_words=28
  )

  if title_too_close_to_sources(title, source_titles):
    try:
      alt_title = model_text(
        "chief",
        "You are a newspaper headline editor. Return one original English headline.",
        (
          "Rewrite the headline so it is clearly original and newsroom-ready.\n"
          "Keep factual intent, but avoid wording similar to source titles.\n"
          "Do not use source wording, source names, or citations.\n"
          f"Topic: {topic}\n"
          f"Current headline: {title}\n"
          f"Source titles: {safe_json_dumps(source_titles[:8], fallback='[]')}\n"
          "Output only the new headline."
        ),
        temperature=0.25,
        max_tokens=70,
        retries=1
      )
      candidate = ensure_english_short(alt_title, title, role="chief", max_words=14)
      if candidate and not title_too_close_to_sources(candidate, source_titles):
        title = candidate
    except Exception:
      pass
  if title_too_close_to_sources(title, source_titles):
    title = ensure_english_short(f"{topic} outlook for Spain", "Metropolis Analysis", max_words=14)

  deck_metrics = text_source_overlap_metrics(deck, source_overlap_index)
  if deck_metrics.get("copied_run_7") or deck_metrics.get("overlap_5", 0.0) >= 0.34:
    try:
      alt_deck = model_text(
        "chief",
        "You are a newspaper deck editor. Return one original English deck.",
        (
          "Rewrite this deck so it is original, concise, and publication-ready.\n"
          "Do not mirror source phrasing, and do not cite or mention sources.\n"
          f"Topic: {topic}\n"
          f"Current deck: {deck}\n"
          "Output only the new deck."
        ),
        temperature=0.2,
        max_tokens=90,
        retries=1
      )
      candidate_deck = ensure_english_short(alt_deck, deck, role="chief", max_words=28)
      cand_metrics = text_source_overlap_metrics(candidate_deck, source_overlap_index)
      if candidate_deck and not cand_metrics.get("copied_run_7") and cand_metrics.get("overlap_5", 0.0) < 0.34:
        deck = candidate_deck
    except Exception:
      pass

  normalized_sections = normalize_sections_from_outline(outline_json, topic)

  summary_bullets = summary_json.get("bullets") if isinstance(summary_json.get("bullets"), list) else []
  summary_bullets = [clean_generated_text(sanitize(b)) for b in summary_bullets if sanitize(b)]
  summary_bullets = [b for b in summary_bullets if b][:10]
  spain_context = clean_generated_text(
    sanitize(summary_json.get("spain_context", "")) or sanitize(summary_json.get("contexto_es", ""))
  )

  verificados = (
    verify_json.get("verified")
    if isinstance(verify_json.get("verified"), list)
    else (verify_json.get("verificados") if isinstance(verify_json.get("verificados"), list) else [])
  )
  verified_lines = []
  for item in verificados:
    if not isinstance(item, dict):
      continue
    hecho = clean_generated_text(sanitize(item.get("claim", "")) or sanitize(item.get("hecho", "")))
    just = clean_generated_text(
      sanitize(item.get("short_justification", "")) or sanitize(item.get("justificacion_corta", ""))
    )
    if hecho:
      line = hecho
      if just:
        line += f" {just}"
      verified_lines.append(clean_generated_text(line))

  if not verified_lines:
    for src in sources[:5]:
      snip = sanitize(src.get("snippet", ""))
      if snip:
        verified_lines.append(clean_generated_text(snip[:240]))

  allowed_numeric_tokens = set()
  for src in sources:
    allowed_numeric_tokens |= extract_numeric_tokens(src.get("snippet", ""))
    allowed_numeric_tokens |= extract_numeric_tokens(src.get("content", ""))
  for line in verified_lines:
    allowed_numeric_tokens |= extract_numeric_tokens(line)

  def rotate_slice(values, start, size):
    if not values:
      return []
    out = []
    for i in range(size):
      out.append(values[(start + i) % len(values)])
    return out

  def write_section(section_index, section):
    heading = section.get("heading", "Section")
    focus = section.get("focus", "context")
    section_bullets = rotate_slice(summary_bullets, section_index * 2, min(5, max(1, len(summary_bullets)))) or summary_bullets[:5]
    section_claims = rotate_slice(verified_lines, section_index * 3, min(8, max(1, len(verified_lines)))) or verified_lines[:8]
    prompt = (
      "Write this section for a high-standards newspaper in English.\n"
      "Mandatory rules:\n"
      "- Between 170 and 240 words.\n"
      "- English only, analytical and direct.\n"
      "- Prioritize clarity and reader value.\n"
      "- No lists, no markdown, no meta-commentary, no <think> tags.\n"
      "- No template filler phrases.\n"
      "- Use only facts supported by sources.\n"
      "- Avoid repeating wording from other sections.\n"
      "- Rewrite in original wording; do not copy source phrasing.\n"
      "- Never reproduce 7 or more consecutive source words.\n"
      "- Do not include citations, source names, URLs, or bracket references.\n"
      "- Do not discuss evidence availability, verification mechanics, or source limitations.\n"
      f"Main title: {title}\n"
      f"Section: {heading}\n"
      f"Focus: {focus}\n"
      f"Spain context: {spain_context}\n"
      f"Summary bullets: {safe_json_dumps(section_bullets, fallback='[]')}\n"
      f"Verified facts:\n" + "\n".join(section_claims)
    )

    try:
      draft = model_text(
        "research",
        "You are a senior newsroom writer. Deliver publication-ready final text with no internal reasoning.",
        prompt,
        temperature=0.12,
        max_tokens=420,
        retries=1
      )
      draft = clean_generated_text(draft, heading=heading)
      draft = remove_redundant_intro(draft, title=title, heading=heading)
      draft = strip_embedded_headings(draft)
      draft = re.sub(r"\[(\d{1,3})\]", "", draft)
      draft = trim_to_complete_sentence(draft, min_words=70)
      if word_count(draft) < 90 or has_internal_markers(draft):
        rewrite_prompt = (
          "Rewrite this draft for immediate publication.\n"
          "Remove any internal reasoning or meta-text.\n"
          "Make it more analytical and direct.\n"
          "Keep facts, but do not include citations or source references.\n"
          "Output: final English text only.\n\n"
          f"Draft:\n{draft}"
        )
        repaired = model_text(
          "chief",
          "You are a newsroom style editor. Deliver final text only.",
          rewrite_prompt,
          temperature=0.1,
          max_tokens=420,
          retries=1
        )
        repaired = clean_generated_text(repaired, heading=heading)
        repaired = remove_redundant_intro(repaired, title=title, heading=heading)
        repaired = strip_embedded_headings(repaired)
        repaired = re.sub(r"\[(\d{1,3})\]", "", repaired)
        repaired = trim_to_complete_sentence(repaired, min_words=70)
        if word_count(repaired) > word_count(draft):
          draft = repaired
      unknown_numbers = sorted(
        n for n in extract_numeric_tokens(draft)
        if n not in allowed_numeric_tokens and not re.fullmatch(r"(?:19|20)\d\d", n)
      )
      if len(unknown_numbers) >= 2:
        try:
          numbers_prompt = (
            "Rewrite the text and remove figures or percentages not backed by sources.\n"
            "Keep only verified data and do not include citations.\n"
            "Final output in English, no lists.\n\n"
            f"Text:\n{draft}"
          )
          repaired_numbers = model_text(
            "fact",
            "You are a factual-precision editor. Deliver final text without internal reasoning.",
            numbers_prompt,
            temperature=0.05,
            max_tokens=320,
            retries=1
          )
          repaired_numbers = clean_generated_text(repaired_numbers, heading=heading)
          repaired_numbers = strip_embedded_headings(repaired_numbers)
          repaired_numbers = re.sub(r"\[(\d{1,3})\]", "", repaired_numbers)
          repaired_numbers = trim_to_complete_sentence(repaired_numbers, min_words=70)
          if word_count(repaired_numbers) >= 90 and "## " not in repaired_numbers:
            draft = repaired_numbers
        except Exception:
          pass
      if looks_mostly_spanish(draft):
        translate_prompt = (
          "Translate this newsroom section to fluent English.\n"
          "Keep meaning and factual content.\n"
          "Do not include citations or source references.\n"
          "Output only final publication-ready English text.\n\n"
          f"Text:\n{draft}"
        )
        translated = model_text(
          "research",
          "You are an editor-translator. Deliver only final English text.",
          translate_prompt,
          temperature=0.05,
          max_tokens=420,
          retries=0
        )
        translated = clean_generated_text(translated, heading=heading)
        translated = strip_embedded_headings(translated)
        translated = re.sub(r"\[(\d{1,3})\]", "", translated)
        translated = trim_to_complete_sentence(translated, min_words=70)
        if word_count(translated) >= 85 and not looks_mostly_spanish(translated):
          draft = translated
      metrics = text_source_overlap_metrics(draft, source_overlap_index)
      if metrics.get("copied_run_7") or metrics.get("overlap_5", 0.0) >= 0.20 or has_source_attribution(draft):
        originality_prompt = (
          "Rewrite this section to preserve facts while using original newsroom wording.\n"
          "Hard constraints:\n"
          "- Do not reuse source phrasing.\n"
          "- Do not copy any sequence of 7+ source words.\n"
          "- Do not mention or cite any source.\n"
          "- Keep analytical tone and publication quality.\n"
          "Output final English text only.\n\n"
          f"Text:\n{draft}"
        )
        rewritten = model_text(
          "chief",
          "You are a senior copy editor focused on originality and legal-safe newsroom style.",
          originality_prompt,
          temperature=0.18,
          max_tokens=420,
          retries=1
        )
        rewritten = clean_generated_text(rewritten, heading=heading)
        rewritten = remove_redundant_intro(rewritten, title=title, heading=heading)
        rewritten = strip_embedded_headings(rewritten)
        rewritten = re.sub(r"\[(\d{1,3})\]", "", rewritten)
        rewritten = trim_to_complete_sentence(rewritten, min_words=70)
        rewritten_metrics = text_source_overlap_metrics(rewritten, source_overlap_index)
        if (
          word_count(rewritten) >= 85 and
          not has_internal_markers(rewritten) and
          not looks_mostly_spanish(rewritten) and
          not has_source_attribution(rewritten) and
          not rewritten_metrics.get("copied_run_7") and
          rewritten_metrics.get("overlap_5", 0.0) < 0.20
        ):
          draft = rewritten
          metrics = rewritten_metrics
      if word_count(draft) < 85 or has_internal_markers(draft):
        raise RuntimeError("invalid draft after cleanup")
      if looks_mostly_spanish(draft):
        raise RuntimeError("non-english draft after cleanup")
      if has_source_attribution(draft):
        raise RuntimeError("source attribution leaked to public copy")
      final_metrics = text_source_overlap_metrics(draft, source_overlap_index)
      if final_metrics.get("copied_run_7") or final_metrics.get("overlap_5", 0.0) >= 0.20:
        raise RuntimeError("draft too close to source wording")
      return draft
    except Exception as exc:
      log_warn(f"section writing failed '{heading}': {exc}")
      return fallback_section_text(section, " ".join(summary_bullets[:4]), topic)

  section_texts = [""] * len(normalized_sections)
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, len(normalized_sections)))) as ex:
    future_map = {
      ex.submit(write_section, idx, section): idx
      for idx, section in enumerate(normalized_sections)
    }
    for future in concurrent.futures.as_completed(future_map):
      idx = future_map[future]
      try:
        section_texts[idx] = future.result()
      except Exception:
        section_texts[idx] = fallback_section_text(normalized_sections[idx], " ".join(summary_bullets[:4]), topic)

  body_parts = []
  for section, text in zip(normalized_sections, section_texts):
    body_parts.append(f"## {section.get('heading')}\n\n{text.strip()}")

  full_text = "\n\n".join(body_parts).strip()

  guard = 0
  extra_headings = ["What Comes Next", "Open Scenarios", "What To Watch"]
  used_extra_keys = set()
  while word_count(full_text) < min_words and guard < 3:
    extra_heading = extra_headings[guard % len(extra_headings)]
    extra_prompt = (
      "Add an additional 120 to 170 word block to close and contextualize the analysis.\n"
      "No lists, no markdown, no meta-commentary. English only.\n"
      "Tone must remain analytical and direct.\n"
      "Use original wording and avoid source phrase reuse.\n"
      "Do not copy any sequence of 7 or more source words.\n"
      "Keep factual rigor and avoid citations or source names.\n"
      f"Block subtitle: {extra_heading}\n"
      f"Key summary: {safe_json_dumps(summary_bullets, fallback='[]')}\n"
      f"Verified facts: {' | '.join(verified_lines[:8])}"
    )
    try:
      extra_text = model_text(
        "research",
        "You are an editorial closer. Deliver final text with no internal reasoning.",
        extra_prompt,
        temperature=0.12,
        max_tokens=220,
        retries=1
      )
      extra_text = clean_generated_text(extra_text, heading=extra_heading)
      extra_text = strip_embedded_headings(extra_text)
      extra_text = re.sub(r"\[(\d{1,3})\]", "", extra_text)
      extra_text = trim_to_complete_sentence(extra_text, min_words=60)
      extra_key = normalize_text_key(extra_text)
      if not extra_key or extra_key in used_extra_keys:
        raise RuntimeError("repeated additional block")
      used_extra_keys.add(extra_key)
      if word_count(extra_text) < 60 or has_internal_markers(extra_text):
        raise RuntimeError("invalid additional block")
      if looks_mostly_spanish(extra_text):
        raise RuntimeError("non-english additional block")
      if has_source_attribution(extra_text):
        raise RuntimeError("source attribution leaked to additional block")
      extra_metrics = text_source_overlap_metrics(extra_text, source_overlap_index)
      if extra_metrics.get("copied_run_7") or extra_metrics.get("overlap_5", 0.0) >= 0.20:
        raise RuntimeError("additional block too close to source wording")
    except Exception:
      extra_text = (
        "Following this topic requires continuous tracking of institutional decisions, pricing trends, household impact and business response. "
        "For readers in Spain, the key variable is whether announced measures translate into visible improvements in cost of living, regulatory stability "
        "and investment capacity. Comparison with other European countries remains useful to distinguish durable solutions from short-term reactions."
      )
    full_text += f"\n\n## {extra_heading}\n\n" + extra_text
    guard += 1

  full_text = dedupe_markdown_sections(full_text)
  full_text = strip_internal_reasoning(full_text)
  full_text = re.sub(r"\[(\d{1,3})\]", "", full_text)
  full_text = re.sub(r"https?://\S+", "", full_text)
  full_text = re.sub(r" +([,.;:!?])", r"\1", full_text)

  total_words = word_count(full_text)
  reading_time = max(5, int(round(total_words / 210.0)))

  section_name_raw = clean_generated_text(sanitize(tag_json.get("section", "")))
  section_name = normalize_public_section(section_name_raw, region)
  category = ensure_english_short(tag_json.get("category", ""), "General", role="tagger")
  kicker = ensure_english_short(tag_json.get("kicker", ""), "Editorial Analysis", role="tagger")
  tags = tag_json.get("tags") if isinstance(tag_json.get("tags"), list) else []
  clean_tags = []
  for tag in tags:
    value = clean_generated_text(sanitize(tag))
    key = normalize_text_key(value)
    if key == "espana":
      value = "spain"
    elif key == "economia":
      value = "business"
    elif key == "politica":
      value = "politics"
    elif key == "tecnologia":
      value = "technology"
    elif key == "mundo":
      value = "world"
    if value and value.lower() not in {t.lower() for t in clean_tags}:
      clean_tags.append(value)
  clean_tags = clean_tags[:8] if clean_tags else ["news", "spain"]

  article = {
    "id": slugify(title),
    "title": title,
    "deck": deck,
    "kicker": kicker,
    "section": section_name,
    "category": category,
    "tags": clean_tags,
    "author": "AI Desk",
    "published": utc_now_iso(),
    "reading_time": reading_time,
    "type": "report",
    "region": region,
    "body_html": markdown_to_html(full_text),
    "sources": []
  }
  return article


def update_web_feed(article, publish_dir):
  data_dir = os.path.join(publish_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  feed_path = os.path.join(data_dir, "articles.json")

  feed = {"generated_at": "", "breaking": "", "articles": []}
  if os.path.exists(feed_path):
    with open(feed_path, "r", encoding="utf-8") as f:
      try:
        feed = json.load(f)
      except Exception:
        feed = {"generated_at": "", "breaking": "", "articles": []}

  existing = feed.get("articles", []) if isinstance(feed.get("articles"), list) else []
  cleaned_existing = []
  for old in existing:
    if not isinstance(old, dict):
      continue
    title_probe = sanitize(old.get("title", ""))
    if looks_spanish_title(title_probe):
      continue
    body_probe = sanitize(re.sub(r"<[^>]+>", " ", old.get("body_html", "")))[:420]
    if looks_mostly_spanish(f"{title_probe} {body_probe}"):
      continue
    old["body_html"] = clean_body_html(old.get("body_html", ""))
    old["sources"] = []
    cleaned_existing.append(old)

  cleaned_existing = [a for a in cleaned_existing if a.get("id") != article.get("id")]
  cleaned_existing.insert(0, {
    "id": article["id"],
    "title": article["title"],
    "deck": article["deck"],
    "kicker": article.get("kicker", ""),
    "section": article.get("section", "General"),
    "category": article["category"],
    "tags": article["tags"],
    "author": article["author"],
    "published": article["published"],
    "reading_time": article["reading_time"],
    "type": article["type"],
    "region": article.get("region", "es"),
    "body_html": clean_body_html(article["body_html"]),
    "sources": []
  })

  feed["generated_at"] = utc_now_iso()
  feed["breaking"] = article["title"]
  feed["articles"] = cleaned_existing[:80]

  with open(feed_path, "w", encoding="utf-8") as f:
    json.dump(feed, f, ensure_ascii=False, indent=2)

  return feed_path


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None)
  parser.add_argument("--topic", required=True)
  parser.add_argument("--region", choices=["es", "world", "both"], default="both")
  parser.add_argument("--max-sources", type=int, default=4)
  parser.add_argument("--min-words", type=int, default=3000)
  default_publish = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
  parser.add_argument("--publish-dir", default=default_publish)
  return parser.parse_args()


def main():
  args = parse_args()
  cfg = load_config(args.config)
  article = run_pipeline(cfg, args.topic, args.region, args.max_sources, args.min_words)
  feed_path = update_web_feed(article, args.publish_dir)
  print(f"OK: {feed_path}")
  print(f"Article: {article['title']} ({article['reading_time']} min)")


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    sys.exit(1)
