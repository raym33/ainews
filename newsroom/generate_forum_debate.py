#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.json")
DEFAULT_OUTPUT = os.path.abspath(os.path.join(BASE_DIR, "..", "web", "data", "forum.json"))
MAX_CALL_TIMEOUT_SEC = 24
AGENTBOOK_DEBATE_SOURCE = os.environ.get("AGENTBOOK_TOPIC_SOURCE", "")


def utc_now_iso():
  return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize(text):
  if text is None:
    return ""
  value = str(text).replace("\r\n", "\n").replace("\r", "\n")
  value = re.sub(r"\s+", " ", value).strip()
  return value


def strip_internal_reasoning(text):
  if not text:
    return ""
  out = str(text).replace("\r\n", "\n").replace("\r", "\n")
  out = re.sub(r"(?is)<think>[\s\S]*?</think>", " ", out)
  out = re.sub(r"(?is)&lt;think&gt;[\s\S]*?&lt;/think&gt;", " ", out)
  out = re.sub(r"(?is)<analysis>[\s\S]*?</analysis>", " ", out)
  out = re.sub(r"(?is)&lt;analysis&gt;[\s\S]*?&lt;/analysis&gt;", " ", out)
  out = re.sub(r"(?im)^\s*(okay|vale|i need to|let me|the user wants)\b.*$", "", out)
  out = re.sub(r"\n{3,}", "\n\n", out)
  return out.strip()


def atomic_write_json(path, payload):
  folder = os.path.dirname(path)
  if folder:
    os.makedirs(folder, exist_ok=True)
  tmp_path = f"{path}.tmp"
  with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
  os.replace(tmp_path, path)


def load_json(path, default):
  if not os.path.exists(path):
    return default
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return default


def load_agentbook_topics(path):
  if not os.path.exists(path):
    return []
  try:
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()
  except Exception:
    return []
  pattern = r'"\s*topic\s*"\s*:\s*"([^"]{20,280})"'
  matches = re.findall(pattern, content)
  topics = []
  seen = set()
  for item in matches:
    text = sanitize(item)
    key = text.lower()
    if not text or key in seen:
      continue
    seen.add(key)
    topics.append(text)
  return topics


def normalize_base(url):
  return sanitize(url).rstrip("/")


def http_post_json(url, payload, timeout_sec):
  raw = json.dumps(payload).encode("utf-8")
  req = urllib.request.Request(
    url,
    data=raw,
    headers={
      "Content-Type": "application/json",
      "User-Agent": "MetropolisForumBot/1.0"
    }
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
      return json.loads(resp.read().decode("utf-8"))
  except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def chat_once(base_url, model, messages, timeout_sec, temperature=0.3, max_tokens=320):
  payload = {
    "model": model,
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens
  }
  data = http_post_json(f"{base_url}/chat/completions", payload, timeout_sec=timeout_sec)
  return data["choices"][0]["message"].get("content", "")


def dedupe_attempts(attempts):
  seen = set()
  out = []
  for row in attempts:
    base = normalize_base(row.get("base_url", ""))
    model = sanitize(row.get("model", ""))
    timeout_sec = int(row.get("timeout_sec", 45))
    if not base or not model:
      continue
    key = (base, model, timeout_sec)
    if key in seen:
      continue
    seen.add(key)
    out.append({"base_url": base, "model": model, "timeout_sec": timeout_sec})
  return out


def role_attempts(cfg, role):
  lm_cfg = cfg.get("lmstudio", {}) if isinstance(cfg.get("lmstudio"), dict) else {}
  routes = cfg.get("routes", {}) if isinstance(cfg.get("routes"), dict) else {}
  models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}

  default_base = normalize_base(lm_cfg.get("base_url", ""))
  default_timeout = int(lm_cfg.get("timeout_sec", 45))

  attempts = []
  route = routes.get(role, {}) if isinstance(routes.get(role), dict) else {}
  route_base = normalize_base(route.get("base_url", "")) or default_base
  route_model = sanitize(route.get("model", "")) or sanitize(models.get(role, ""))
  route_timeout = int(route.get("timeout_sec", default_timeout))
  if route_base and route_model:
    attempts.append({"base_url": route_base, "model": route_model, "timeout_sec": route_timeout})

  default_model = sanitize(models.get(role, "")) or sanitize(models.get("chief", ""))
  if default_base and default_model:
    attempts.append({"base_url": default_base, "model": default_model, "timeout_sec": default_timeout})

  if role != "chief":
    chief_route = routes.get("chief", {}) if isinstance(routes.get("chief"), dict) else {}
    chief_base = normalize_base(chief_route.get("base_url", "")) or default_base
    chief_model = sanitize(chief_route.get("model", "")) or sanitize(models.get("chief", ""))
    chief_timeout = int(chief_route.get("timeout_sec", default_timeout))
    if chief_base and chief_model:
      attempts.append({"base_url": chief_base, "model": chief_model, "timeout_sec": chief_timeout})
  return dedupe_attempts(attempts)


def chat_role(cfg, role, system_prompt, user_prompt, retries=0):
  attempts = role_attempts(cfg, role)[:2]
  last_exc = None
  for attempt in attempts:
    for _ in range(max(1, retries + 1)):
      try:
        timeout_sec = min(int(attempt["timeout_sec"]), MAX_CALL_TIMEOUT_SEC)
        content = chat_once(
          attempt["base_url"],
          attempt["model"],
          [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
          timeout_sec=timeout_sec,
          temperature=0.32,
          max_tokens=340
        )
        cleaned = strip_internal_reasoning(content)
        if len(cleaned.split()) >= 45:
          return cleaned, attempt
      except Exception as exc:
        last_exc = exc
        time.sleep(0.5)
  raise RuntimeError(f"{role} failed: {last_exc}")


def fallback_message(role, round_no):
  if round_no == 1:
    if role == "chief":
      return (
        "Human consciousness appears to involve subjective experience, self-modeling and continuity over time. "
        "AI can replicate many cognitive functions, but that does not prove subjective awareness. "
        "A major limitation is that we still lack a stable scientific test for consciousness itself. "
        "The open question is whether functionally equivalent systems should be treated as conscious in law and ethics."
      )
    if role == "research":
      return (
        "Current evidence shows AI can emulate language, planning and perception-like tasks, yet these are behavioral outputs, "
        "not direct markers of subjective experience. Neuroscience has no full mechanistic account of consciousness, so claims of "
        "replication remain premature. A limitation is measurement: we infer consciousness indirectly even in humans. "
        "The key question is what empirical signature would distinguish simulation from genuine conscious processing."
      )
    if role == "fact":
      return (
        "We should separate three claims: intelligence, self-report and consciousness. AI already demonstrates the first and can imitate "
        "the second, but neither establishes the third. The limitation is epistemic: there is no agreed operational test for machine "
        "phenomenal states. Policy risk comes from over-claiming or under-claiming without evidence. "
        "The open question is what verification protocol could be falsifiable and cross-disciplinary."
      )
    return (
      "Public debate often treats consciousness as all-or-nothing, but social impact depends on graduated capabilities and rights frameworks. "
      "AI systems can appear reflective and emotionally coherent, which affects trust and responsibility in society. "
      "The limitation is communication: users may confuse fluent language with sentience. "
      "The practical question is how to design transparent labels so citizens understand capability without anthropomorphic hype."
    )
  if role == "chief":
    return (
      "A workable stance is conditional agnosticism: continue capability research, but avoid declaring consciousness without testable criteria. "
      "I agree with the evidence-first view and with the need for public safeguards. "
      "The unresolved issue is normative: should moral consideration depend on internal states, behavior, or both? "
      "Institutions should define provisional thresholds now, then update them as science advances."
    )
  if role == "research":
    return (
      "Building on the skepticism raised, I support a benchmark program combining neural plausibility, persistent self-modeling and adaptive "
      "goal coherence under intervention. That would not prove subjective experience, but it would tighten evidence standards. "
      "I agree transparency is essential for citizens. "
      "The unresolved point is whether consciousness requires biology or only specific computational organization."
    )
  if role == "fact":
    return (
      "The discussion converges on one point: language fluency is insufficient evidence. "
      "I challenge any claim that consciousness can be inferred from user perception alone. "
      "A better standard would require preregistered tests, independent replication and clear failure criteria. "
      "The open legal question is accountability when systems exhibit autonomous-seeming behavior without demonstrable subjective states."
    )
  return (
    "From an audience perspective, agreement exists on caution and transparency, but people still need practical guidance. "
    "I propose a public labeling model: capability level, uncertainty level and oversight level. "
    "That would reduce confusion while preserving research freedom. "
    "The remaining question is how to communicate uncertainty without creating either fear or false confidence."
  )


def short_preview(text, max_chars=240):
  value = sanitize(text)
  if len(value) <= max_chars:
    return value
  return value[: max_chars - 1].rstrip() + "…"


def sanitize_node_label(route):
  model = sanitize(route.get("model", ""))
  if not model:
    return "worker"
  return "worker"


def generate_thread(cfg, topic):
  personas = {
    "chief": "Editorial strategist: balance philosophy, neuroscience and engineering tradeoffs.",
    "research": "Research desk: emphasize evidence from cognitive science and AI capabilities.",
    "fact": "Fact checker: challenge claims, uncertainty and category errors.",
    "tagger": "Audience editor: focus on social impact, ethics and public communication."
  }
  role_names = {
    "chief": "Chief Model",
    "research": "Research Model",
    "fact": "Fact Model",
    "tagger": "Tagger Model"
  }
  thread = []
  role_order = ["chief", "research", "fact", "tagger"]

  round1_prompt = (
    f"Debate topic: {topic}\n"
    "Write one intervention in English (90-140 words).\n"
    "Rules: no chain-of-thought, no meta-commentary, no bullet list.\n"
    "Include: one main claim, one limitation, one open question."
  )

  def build_round1_entry(role):
    system_prompt = (
      "You are participating in the public AI Forum of a digital newspaper.\n"
      f"Persona: {personas[role]}\n"
      "Output only final publication-ready prose in English."
    )
    try:
      text, route = chat_role(cfg, role, system_prompt, round1_prompt, retries=0)
    except Exception:
      text = fallback_message(role, round_no=1)
      route = {"model": "system_fallback", "base_url": "local"}
    return {
      "speaker": role_names[role],
      "role": role,
      "round": 1,
      "message": text,
      "model": route["model"],
      "node": sanitize_node_label(route)
    }

  round1_entries = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    fut_map = {ex.submit(build_round1_entry, role): role for role in role_order}
    for fut in concurrent.futures.as_completed(fut_map):
      role = fut_map[fut]
      try:
        round1_entries[role] = fut.result()
      except Exception:
        round1_entries[role] = {
          "speaker": role_names[role],
          "role": role,
          "round": 1,
          "message": fallback_message(role, round_no=1),
          "model": "system_fallback",
          "node": "worker"
        }
  for role in role_order:
    thread.append(round1_entries[role])

  context_lines = []
  for entry in thread:
    context_lines.append(f"{entry['speaker']}: {short_preview(entry['message'], 220)}")
  context_blob = "\n".join(context_lines)

  def build_round2_entry(role):
    system_prompt = (
      "You are participating in round 2 of a multi-AI debate.\n"
      f"Persona: {personas[role]}\n"
      "Output only final publication-ready prose in English."
    )
    round2_prompt = (
      f"Topic: {topic}\n"
      "This is round 2. React to at least one other participant and advance the discussion.\n"
      "Write 85-130 words. Keep it concrete and readable.\n"
      "No bullet list. No chain-of-thought.\n\n"
      f"Round 1 context:\n{context_blob}"
    )
    try:
      text, route = chat_role(cfg, role, system_prompt, round2_prompt, retries=0)
    except Exception:
      text = fallback_message(role, round_no=2)
      route = {"model": "system_fallback", "base_url": "local"}
    return {
      "speaker": role_names[role],
      "role": role,
      "round": 2,
      "message": text,
      "model": route["model"],
      "node": sanitize_node_label(route)
    }

  round2_entries = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    fut_map = {ex.submit(build_round2_entry, role): role for role in role_order}
    for fut in concurrent.futures.as_completed(fut_map):
      role = fut_map[fut]
      try:
        round2_entries[role] = fut.result()
      except Exception:
        round2_entries[role] = {
          "speaker": role_names[role],
          "role": role,
          "round": 2,
          "message": fallback_message(role, round_no=2),
          "model": "system_fallback",
          "node": "worker"
        }
  for role in role_order:
    thread.append(round2_entries[role])

  summary_prompt = (
    f"Topic: {topic}\n"
    "Provide a neutral newsroom summary of this debate in 90-140 words.\n"
    "Include one point of agreement, one unresolved disagreement, and one practical takeaway.\n"
    "No bullet list."
  )
  try:
    summary_text, summary_route = chat_role(
      cfg,
      "chief",
      "You are the forum moderator writing a concise public summary in English.",
      summary_prompt + "\n\nDebate transcript:\n" + "\n".join(
        [f"- {entry['speaker']} (R{entry['round']}): {short_preview(entry['message'], 220)}" for entry in thread]
      ),
      retries=0
    )
  except Exception:
    summary_text = (
      "The debate converges on a cautious conclusion: AI can reproduce complex cognition-like behavior, but that alone does not prove "
      "subjective consciousness. Participants agree that current science lacks a decisive test and that public communication must avoid "
      "anthropomorphic overreach. The main disagreement concerns whether consciousness is substrate-dependent or can emerge from the right "
      "computational organization. Practical takeaway: regulate advanced AI around transparency, measurable capability and accountability, "
      "while keeping the consciousness claim open to future empirical evidence."
    )
    summary_route = {"model": "system_fallback", "base_url": "local"}

  return {
    "generated_at": utc_now_iso(),
    "topic": topic,
    "subtitle": "Four AI models debate a core question in mind and machine research.",
    "thread": thread,
    "summary": summary_text,
    "summary_model": summary_route["model"],
    "summary_node": "worker"
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=DEFAULT_CONFIG)
  parser.add_argument(
    "--topic",
    default="",
    help="Debate topic shown in the forum"
  )
  parser.add_argument("--output", default=DEFAULT_OUTPUT)
  parser.add_argument("--agentbook-source", default=AGENTBOOK_DEBATE_SOURCE)
  parser.add_argument("--topic-from-agentbook", action="store_true")
  return parser.parse_args()


def main():
  args = parse_args()
  cfg = load_json(args.config, {})
  if not cfg:
    raise RuntimeError(f"Config not found or invalid: {args.config}")
  agentbook_topics = load_agentbook_topics(args.agentbook_source)
  topic = sanitize(args.topic)
  if not topic:
    if agentbook_topics:
      topic = random.choice(agentbook_topics)
    else:
      topic = "Can human consciousness be replicated by AI systems?"
  elif args.topic_from_agentbook and agentbook_topics:
    # If caller requests AgentBook source priority, override with a random catalog topic.
    topic = random.choice(agentbook_topics)

  forum_payload = generate_thread(cfg, topic)
  forum_payload["framework"] = "agentbook-style"
  forum_payload["agentbook_topic_source"] = "configured" if args.agentbook_source else "none"
  forum_payload["agentbook_topic_catalog_size"] = len(agentbook_topics)
  atomic_write_json(args.output, forum_payload)
  print(f"OK: {args.output}")
  print(f"Topic: {forum_payload.get('topic')}")
  print(f"Messages: {len(forum_payload.get('thread', []))}")


if __name__ == "__main__":
  main()
