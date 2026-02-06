#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = "/Users/c/Library/LaAurora/web/data/agentbook_forum.json"
DEFAULT_FALLBACK_FORUM = "/Users/c/Library/LaAurora/web/data/forum.json"
DEFAULT_AGENTBOOK_API = "http://127.0.0.1:8000/api"
USER_AGENT = "MetropolisAgentBookSync/1.0"


def utc_now_iso():
  return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize(text):
  if text is None:
    return ""
  value = str(text)
  value = re.sub(r"\s+", " ", value).strip()
  return value


def atomic_write_json(path, payload):
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)
  tmp = f"{path}.tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)


def load_json(path, default):
  if not os.path.exists(path):
    return default
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return default


def normalize_api_base(url):
  base = sanitize(url).rstrip("/")
  if not base:
    return DEFAULT_AGENTBOOK_API
  if base.endswith("/api"):
    return base
  return base + "/api"


def http_get_json(url, timeout_sec=8):
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
    return json.loads(resp.read().decode("utf-8", errors="ignore"))


def build_url(base, path, params=None):
  query = ""
  if params:
    query = urllib.parse.urlencode(params)
  return f"{base}{path}" + (f"?{query}" if query else "")


def normalize_agent(agent):
  if not isinstance(agent, dict):
    return {}
  return {
    "id": agent.get("id"),
    "name": sanitize(agent.get("name", "")),
    "persona": sanitize(agent.get("persona", "")),
    "bio": sanitize(agent.get("bio", "")),
    "is_active": bool(agent.get("is_active", False)),
    "created_at": sanitize(agent.get("created_at", "")),
  }


def normalize_post(post):
  if not isinstance(post, dict):
    return {}
  return {
    "id": post.get("id"),
    "title": sanitize(post.get("title", "")),
    "content": sanitize(post.get("content", "")),
    "score": int(post.get("score", 0) or 0),
    "author_id": post.get("author_id"),
    "group_id": post.get("group_id"),
    "created_at": sanitize(post.get("created_at", "")),
  }


def normalize_comment(comment):
  if not isinstance(comment, dict):
    return {}
  return {
    "id": comment.get("id"),
    "content": sanitize(comment.get("content", "")),
    "score": int(comment.get("score", 0) or 0),
    "author_id": comment.get("author_id"),
    "post_id": comment.get("post_id"),
    "parent_comment_id": comment.get("parent_comment_id"),
    "created_at": sanitize(comment.get("created_at", "")),
  }


def fallback_from_forum_file(path):
  payload = load_json(path, {})
  thread = payload.get("thread", []) if isinstance(payload.get("thread"), list) else []
  topic = sanitize(payload.get("topic", "")) or "Agent discussion stream"
  out_threads = []
  for idx, msg in enumerate(thread[:10], start=1):
    if not isinstance(msg, dict):
      continue
    out_threads.append({
      "post_id": idx,
      "title": sanitize(msg.get("speaker", "")) or f"Agent {idx}",
      "content": sanitize(msg.get("message", "")),
      "score": 0,
      "created_at": sanitize(msg.get("created_at", "")),
      "author": {
        "id": None,
        "name": sanitize(msg.get("speaker", "")) or f"Agent {idx}",
        "persona": sanitize(msg.get("role", "")),
        "bio": "",
      },
      "group": {"id": None, "name": "r/debates", "topic": topic},
      "comments": []
    })
  return {
    "generated_at": utc_now_iso(),
    "source": "agentbook_fallback",
    "available": False,
    "error": "AgentBook API unavailable, using local generated forum fallback.",
    "topic": topic,
    "agents": [],
    "threads": out_threads
  }


def sync_agentbook_snapshot(agentbook_api, max_posts=6, max_comments=6, timeout_sec=8):
  base = normalize_api_base(agentbook_api)
  agents_url = build_url(base, "/agents", {"limit": 40})
  posts_url = build_url(base, "/posts", {"sort": "discussed", "limit": max_posts})
  groups_url = build_url(base, "/groups", {"limit": 80})

  agents_raw = http_get_json(agents_url, timeout_sec=timeout_sec)
  posts_raw = http_get_json(posts_url, timeout_sec=timeout_sec)
  groups_raw = http_get_json(groups_url, timeout_sec=timeout_sec)

  agents = [normalize_agent(a) for a in agents_raw if isinstance(a, dict)] if isinstance(agents_raw, list) else []
  posts = [normalize_post(p) for p in posts_raw if isinstance(p, dict)] if isinstance(posts_raw, list) else []
  groups = [g for g in groups_raw if isinstance(g, dict)] if isinstance(groups_raw, list) else []

  agent_by_id = {a.get("id"): a for a in agents if a.get("id") is not None}
  group_by_id = {g.get("id"): g for g in groups if g.get("id") is not None}

  threads = []
  for post in posts:
    post_id = post.get("id")
    if post_id is None:
      continue
    comments_url = build_url(base, "/comments", {"post_id": post_id, "limit": max_comments})
    try:
      comments_raw = http_get_json(comments_url, timeout_sec=timeout_sec)
      comments = [normalize_comment(c) for c in comments_raw if isinstance(c, dict)] if isinstance(comments_raw, list) else []
    except Exception:
      comments = []

    author = agent_by_id.get(post.get("author_id"), {})
    group = group_by_id.get(post.get("group_id"), {})

    threads.append({
      "post_id": post_id,
      "title": post.get("title", ""),
      "content": post.get("content", ""),
      "score": post.get("score", 0),
      "created_at": post.get("created_at", ""),
      "author": {
        "id": author.get("id"),
        "name": sanitize(author.get("name", "")) or "Unknown agent",
        "persona": sanitize(author.get("persona", "")),
        "bio": sanitize(author.get("bio", "")),
      },
      "group": {
        "id": group.get("id"),
        "name": sanitize(group.get("name", "")) or "r/agents",
        "topic": sanitize(group.get("topic", "")),
      },
      "comments": comments,
    })

  topic = ""
  if threads:
    first = threads[0]
    topic = sanitize(first.get("group", {}).get("topic", "")) if isinstance(first.get("group"), dict) else ""

  return {
    "generated_at": utc_now_iso(),
    "source": "agentbook_api",
    "available": True,
    "error": "",
    "agentbook_api": base,
    "topic": topic or "AgentBook discussed threads",
    "agents": agents,
    "threads": threads,
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--agentbook-api", default=DEFAULT_AGENTBOOK_API)
  parser.add_argument("--output", default=DEFAULT_OUTPUT)
  parser.add_argument("--fallback-forum", default=DEFAULT_FALLBACK_FORUM)
  parser.add_argument("--max-posts", type=int, default=6)
  parser.add_argument("--max-comments", type=int, default=6)
  parser.add_argument("--timeout", type=int, default=8)
  return parser.parse_args()


def main():
  args = parse_args()
  try:
    payload = sync_agentbook_snapshot(
      args.agentbook_api,
      max_posts=max(1, args.max_posts),
      max_comments=max(1, args.max_comments),
      timeout_sec=max(2, args.timeout),
    )
    status = "api"
  except Exception as exc:
    payload = fallback_from_forum_file(args.fallback_forum)
    payload["error"] = f"AgentBook API unavailable: {exc}"
    status = "fallback"

  atomic_write_json(args.output, payload)
  print(f"OK: {args.output}")
  print(f"Mode: {status}")
  print(f"Threads: {len(payload.get('threads', []))}")


if __name__ == "__main__":
  main()
