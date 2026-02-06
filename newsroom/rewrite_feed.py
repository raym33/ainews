#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime as dt
import html as html_lib
import json
import os
import re
import difflib
from typing import Dict, List, Tuple, Optional

from ai_publisher import (
  MASTER_EDITORIAL_PROMPT,
  LMStudioClient,
  chat_with_retry,
  clean_body_html,
  clean_generated_text,
  dedupe_markdown_sections,
  has_internal_markers,
  has_source_attribution,
  json_from_text,
  load_config,
  looks_mostly_spanish,
  normalize_text_key,
  sanitize,
  strip_internal_reasoning,
  utc_now_iso,
  word_count,
  markdown_to_html,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.json")
DEFAULT_FEED = os.path.abspath(os.path.join(BASE_DIR, "..", "web", "data", "articles.json"))
LOG_DIR = os.path.join(BASE_DIR, "logs")
BACKUP_DIR = os.path.join(LOG_DIR, "feed_backups")


def load_json(path, default):
  if not os.path.exists(path):
    return default
  with open(path, "r", encoding="utf-8") as f:
    try:
      return json.load(f)
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


def html_to_text(body_html):
  if not body_html:
    return ""
  text = body_html
  text = re.sub(r"(?is)<br\\s*/?>", "\n", text)
  text = re.sub(r"(?is)</(p|h2|h3|li|ul|ol|div|section|article)>", "\n\n", text)
  text = re.sub(r"(?is)<li[^>]*>", "- ", text)
  text = re.sub(r"(?is)<[^>]+>", " ", text)
  text = html_lib.unescape(text)
  text = text.replace("\r\n", "\n").replace("\r", "\n")
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r" *\n *", "\n", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def html_to_markdown_like(body_html):
  if not body_html:
    return ""
  text = body_html
  text = re.sub(r"(?is)<h2[^>]*>(.*?)</h2>", lambda m: "\n\n## " + sanitize(re.sub(r"<[^>]+>", " ", m.group(1))) + "\n\n", text)
  text = re.sub(r"(?is)<h3[^>]*>(.*?)</h3>", lambda m: "\n\n## " + sanitize(re.sub(r"<[^>]+>", " ", m.group(1))) + "\n\n", text)
  text = re.sub(r"(?is)</p>", "\n\n", text)
  text = re.sub(r"(?is)<br\\s*/?>", "\n", text)
  text = re.sub(r"(?is)<li[^>]*>", "", text)
  text = re.sub(r"(?is)</li>", "\n", text)
  text = re.sub(r"(?is)</(ul|ol)>", "\n\n", text)
  text = re.sub(r"(?is)<[^>]+>", " ", text)
  text = html_lib.unescape(text)
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r" *\n *", "\n", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def clean_markdown_body(value):
  text = strip_internal_reasoning(value or "")
  text = text.replace("\r\n", "\n").replace("\r", "\n")
  text = re.sub(r"\[(\d{1,3})\]", "", text)
  text = re.sub(r"https?://\\S+", "", text)
  text = re.sub(r"(?im)^\s*(references?|sources?)\s*:?.*$", "", text)
  text = re.sub(r"(?im)^\s*[-*]\s+", "", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  text = dedupe_markdown_sections(text)
  if not re.search(r"(?m)^##\\s+", text):
    text = "## Analysis\n\n" + text.strip()
  return text.strip()


def reading_time_from_text(text):
  return max(5, int(round(word_count(text) / 210.0)))


def article_similarity(a, b):
  left = normalize_text_key(a)
  right = normalize_text_key(b)
  if not left or not right:
    return 0.0
  return difflib.SequenceMatcher(None, left, right).ratio()


def style_score(article):
  title = sanitize(article.get("title", ""))
  deck = sanitize(article.get("deck", ""))
  body_html = article.get("body_html", "")
  body_text = html_to_text(body_html)
  score = 0
  reasons = []

  if has_internal_markers(body_text):
    score += 5
    reasons.append("internal-markers")
  if has_source_attribution(body_text):
    score += 4
    reasons.append("source-attribution")
  if looks_mostly_spanish(f"{title} {deck} {body_text[:600]}"):
    score += 4
    reasons.append("non-english-traces")
  if word_count(body_text) < 780:
    score += 2
    reasons.append("short-body")
  h2_count = len(re.findall(r"(?is)<h2\\b", body_html))
  if h2_count < 3:
    score += 1
    reasons.append("weak-structure")
  if normalize_text_key(title) in {"metropolis analysis", "analysis"}:
    score += 2
    reasons.append("generic-title")
  if word_count(deck) < 12:
    score += 1
    reasons.append("weak-deck")
  generic_patterns = [
    r"\\bthis piece dissects\\b",
    r"\\bin this analysis\\b",
    r"\\bthe article examines\\b",
    r"\\bwhat comes next\\b",
  ]
  generic_hits = 0
  sample = f"{deck} {body_text[:1200]}".lower()
  for pattern in generic_patterns:
    if re.search(pattern, sample):
      generic_hits += 1
  if generic_hits:
    score += min(2, generic_hits)
    reasons.append("template-voice")

  return score, reasons


def unique_rewrite_routes(cfg):
  routes = []
  seen = set()
  route_cfg = cfg.get("routes", {}) if isinstance(cfg.get("routes"), dict) else {}
  models_cfg = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
  lm_cfg = cfg.get("lmstudio", {}) if isinstance(cfg.get("lmstudio"), dict) else {}

  for role in ("chief", "research", "tagger", "fact"):
    role_route = route_cfg.get(role, {}) if isinstance(route_cfg.get(role), dict) else {}
    base = sanitize(role_route.get("base_url", "") or lm_cfg.get("base_url", ""))
    model = sanitize(role_route.get("model", "") or models_cfg.get(role, ""))
    timeout = int(role_route.get("timeout_sec", lm_cfg.get("timeout_sec", 90)) or 90)
    timeout = max(20, min(90, timeout))
    if not base or not model:
      continue
    key = (base.rstrip("/"), model)
    if key in seen:
      continue
    seen.add(key)
    routes.append({"base_url": base, "model": model, "timeout_sec": timeout})

  if not routes:
    base = sanitize(lm_cfg.get("base_url", ""))
    model = sanitize(models_cfg.get("chief", ""))
    timeout = int(lm_cfg.get("timeout_sec", 90) or 90)
    timeout = max(20, min(90, timeout))
    if base and model:
      routes.append({"base_url": base, "model": model, "timeout_sec": timeout})
  return routes


def build_rewrite_prompt(article, issues, min_words):
  title = sanitize(article.get("title", ""))
  deck = sanitize(article.get("deck", ""))
  body_md = html_to_markdown_like(article.get("body_html", ""))
  if len(body_md) > 9000:
    body_md = body_md[:6200] + "\\n\\n[... trimmed for rewrite speed ...]\\n\\n" + body_md[-2400:]
  current_words = max(min_words, word_count(html_to_text(article.get("body_html", ""))))
  target_min = max(min_words, int(current_words * 0.9))
  target_max = min(1500, int(current_words * 1.2) + 120)
  issue_line = ", ".join(issues) if issues else "general consistency"

  return (
    "Rewrite this published Metropolis article to match a consistent house style.\n"
    "Return ONLY valid JSON with keys: title, deck, body_markdown.\n"
    "Rules:\n"
    "- English only.\n"
    "- Keep factual meaning from the provided draft only.\n"
    "- Do not invent facts, events, numbers, or quotes.\n"
    "- Create an original headline and original phrasing across the article.\n"
    "- Do not copy long strings from the input.\n"
    "- No URLs, citations, source mentions, bracket references, or attribution phrases.\n"
    "- No chain-of-thought or meta-commentary.\n"
    "- Body must be analytical and direct, with 4 to 6 H2 sections and paragraph prose.\n"
    f"- Body length target: {target_min} to {target_max} words.\n"
    f"Quality issues to fix: {issue_line}.\n\n"
    f"Current title: {title}\n"
    f"Current deck: {deck}\n"
    f"Current body:\n{body_md}"
  )


def rewrite_with_route(route, article, issues, min_words):
  system_prompt = (
    MASTER_EDITORIAL_PROMPT + "\n\n"
    "You are a senior newsroom editor for Metropolis. "
    "Rewrite copy to be publication-ready, legally safe, and style-consistent. "
    "Return JSON only."
  )
  user_prompt = build_rewrite_prompt(article, issues, min_words)
  client = LMStudioClient(route["base_url"], timeout_sec=route["timeout_sec"])
  raw = chat_with_retry(
    client,
    route["model"],
    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    temperature=0.14,
    max_tokens=1500,
    retries=0
  )
  parsed = json_from_text(raw)
  if not isinstance(parsed, dict):
    raise RuntimeError("invalid-json")
  return parsed


def validate_candidate(article, candidate, min_words):
  title = clean_generated_text(sanitize(candidate.get("title", "")))
  deck = clean_generated_text(sanitize(candidate.get("deck", "")))
  body_md_raw = candidate.get("body_markdown", "")
  if not isinstance(body_md_raw, str):
    body_md_raw = ""
  body_md = clean_markdown_body(body_md_raw)

  if not title:
    title = sanitize(article.get("title", "")) or "Metropolis Analysis"
  if not deck:
    deck = sanitize(article.get("deck", "")) or "In-depth analysis for readers in Spain."

  body_html = clean_body_html(markdown_to_html(body_md))
  body_text = html_to_text(body_html)

  if word_count(body_text) < min_words:
    return None, "too-short"
  if has_internal_markers(body_text):
    return None, "internal-markers"
  if has_source_attribution(body_text):
    return None, "source-attribution"
  if re.search(r"https?://|www\\.", body_text):
    return None, "url-leak"
  if re.search(r"\[(\d{1,3})\]", body_text):
    return None, "ref-leak"
  if looks_mostly_spanish(f"{title} {deck} {body_text[:900]}"):
    return None, "non-english"

  old_text = html_to_text(article.get("body_html", ""))
  if article_similarity(old_text, body_text) > 0.95:
    return None, "no-meaningful-rewrite"

  updated = dict(article)
  updated["title"] = title
  updated["deck"] = deck
  updated["body_html"] = body_html
  updated["reading_time"] = reading_time_from_text(body_text)
  updated["author"] = sanitize(article.get("author", "")) or "AI Desk"
  updated["sources"] = []
  updated["updated"] = utc_now_iso()
  return updated, None


def rewrite_article(article, routes, min_words, min_score):
  score, reasons = style_score(article)
  if score < min_score:
    return None, score, reasons, "score-below-threshold"

  last_err = "no-routes"
  for route in routes:
    try:
      parsed = rewrite_with_route(route, article, reasons, min_words)
      updated, err = validate_candidate(article, parsed, min_words)
      if updated is not None:
        return updated, score, reasons, None
      last_err = err or "invalid-candidate"
    except Exception as exc:
      last_err = sanitize(str(exc)) or "rewrite-failed"
      continue
  return None, score, reasons, last_err


def backup_feed(feed_path, feed_data):
  os.makedirs(BACKUP_DIR, exist_ok=True)
  stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
  path = os.path.join(BACKUP_DIR, f"articles-{stamp}.json")
  atomic_write_json(path, feed_data)
  return path


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=DEFAULT_CONFIG)
  parser.add_argument("--feed-path", default=DEFAULT_FEED)
  parser.add_argument("--max-items", type=int, default=6)
  parser.add_argument("--skip-newest", type=int, default=3)
  parser.add_argument("--min-score", type=int, default=2)
  parser.add_argument("--min-words", type=int, default=780)
  parser.add_argument("--workers", type=int, default=2)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main():
  args = parse_args()
  cfg = load_config(args.config)
  feed = load_json(args.feed_path, {"generated_at": "", "breaking": "", "articles": []})
  articles = feed.get("articles", []) if isinstance(feed.get("articles"), list) else []
  routes = unique_rewrite_routes(cfg)

  if not routes:
    raise RuntimeError("No rewrite routes available in config.")

  start_idx = max(0, int(args.skip_newest))
  candidates: List[Tuple[int, Dict]] = []
  for idx in range(len(articles) - 1, start_idx - 1, -1):
    item = articles[idx]
    if isinstance(item, dict):
      candidates.append((idx, item))
    if len(candidates) >= max(1, args.max_items):
      break

  if not candidates:
    print("No legacy articles selected for rewrite.")
    return

  results = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
    future_map = {
      ex.submit(rewrite_article, article, routes, int(args.min_words), int(args.min_score)): idx
      for idx, article in candidates
    }
    for future in concurrent.futures.as_completed(future_map):
      idx = future_map[future]
      try:
        updated, score, reasons, err = future.result()
      except Exception as exc:
        updated, score, reasons, err = None, 0, [], sanitize(str(exc))
      results[idx] = {
        "updated": updated,
        "score": score,
        "reasons": reasons,
        "error": err
      }
      status = "rewritten" if isinstance(updated, dict) else f"skipped:{err}"
      print(f"progress idx={idx} score={score} status={status}", flush=True)

  changed = 0
  for idx, _article in candidates:
    row = results.get(idx, {})
    updated = row.get("updated")
    if isinstance(updated, dict):
      articles[idx] = updated
      changed += 1

  summary_rows = []
  for idx, _article in candidates:
    row = results.get(idx, {})
    err = row.get("error")
    score = row.get("score", 0)
    reasons = ",".join(row.get("reasons", [])) if row.get("reasons") else "none"
    status = "rewritten" if isinstance(row.get("updated"), dict) else f"skipped:{err}"
    summary_rows.append(f"idx={idx} score={score} reasons={reasons} status={status}")

  feed["articles"] = articles
  feed["generated_at"] = utc_now_iso()

  backup_path = ""
  if changed > 0 and not args.dry_run:
    backup_path = backup_feed(args.feed_path, load_json(args.feed_path, feed))
    atomic_write_json(args.feed_path, feed)

  print(f"Rewriter routes={len(routes)} selected={len(candidates)} changed={changed}")
  if backup_path:
    print(f"Backup: {backup_path}")
  for line in summary_rows:
    print(line)


if __name__ == "__main__":
  main()
