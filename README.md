# Metropolis AI Newsroom

Metropolis is an AI-native newspaper stack that runs end-to-end editorial operations: topic discovery, source collection, drafting, fact-focused rewriting, publication, monitoring, and continuous style normalization.

It is designed to publish in English for Spain-focused readers, and to run continuously on a distributed local inference cluster (multiple Macs with LM Studio/Ollama).

## Why this exists (VC summary)
Traditional digital publishing has three structural bottlenecks:
1. Editorial throughput is expensive and hard to scale.
2. Consistency and quality degrade under high publishing frequency.
3. Real-time topic coverage (news + markets + tech) requires 24/7 operational discipline.

Metropolis addresses this with a programmable newsroom where editorial logic is software:
1. Multi-agent writing roles (`chief`, `research`, `fact`, `tagger`) are routed across available model nodes.
2. Source gathering and evidence-aware generation are automated per article.
3. Output quality is normalized automatically, including legacy feed rewrites.
4. A static web delivery layer keeps serving cost and operational complexity low.

## What the repository does
This repo contains the full stack:
1. `newsroom/`: autonomous publishing engine and cluster orchestration.
2. `web/`: production web frontend (responsive newspaper + article pages + AI forum tab).
3. `launchd` automation: long-running daemons for publishing, watchdog health checks, and optional GitHub sync.

Core product behavior:
1. Pulls fresh topics and sources (R CLI web search, RSS, optional market snapshots).
2. Generates long-form article structure and sections with role-specialized models.
3. Cleans hidden reasoning artifacts, strips citations/URLs, and enforces publication-safe style.
4. Publishes into `web/data/articles.json` consumed by the web app.
5. Continuously rewrites older weak-style articles (`rewrite_feed.py`) for a homogeneous editorial voice.

## Product capabilities
1. **Autonomous article generation**
   - Long-form articles with structured sections and reading-time estimation.
   - Topic coverage across politics, economy, AI, tech, markets, and global affairs.
2. **Distributed inference routing**
   - Supports multiple LM Studio nodes and optional Ollama workers.
   - Role-level route fallback and model availability handling.
3. **Editorial quality controls**
   - Reasoning leakage cleanup (`<think>`, meta notes).
   - English-only normalization and Spanish leakage guardrails.
   - Anti-copy safeguards (headline similarity checks, overlap heuristics, rewrite loops).
4. **Feed quality maintenance**
   - `rewrite_feed.py` scores legacy articles and rewrites weak entries with validation gates.
5. **24/7 operations**
   - `daemon.sh` + `runner.py` scheduling loop.
   - `watchdog_monitor.py` for liveness, stale-feed detection, and recovery actions.
6. **Deployment ready**
   - Static frontend with `vercel.json` and custom domain workflow.
   - Optional hourly GitHub sync for auto-deploy pipelines.

## Technical architecture
```
[Topic Queue] -> [Search/RSS/Market Inputs] -> [AI Publisher Pipeline]
                                              |-> chief model: outline/title/deck
                                              |-> research model: section drafting
                                              |-> fact model: claim checks + rewrites
                                              |-> tagger model: section/category/tags

[Validation + Cleanup + Anti-copy] -> [articles.json Feed] -> [Static Web UI]
                                                |
                                                -> [rewrite_feed.py] (legacy normalization)
```

Detailed operating guidance (performance, multi-node routing, model stack, and API connectors):
1. `docs/OPERATIONS_AND_MODELS.md`

### Main components
1. `newsroom/ai_publisher.py`
   - Main generation pipeline.
   - Multi-route model calls with fallback.
   - Post-processing, style cleanup, and feed update.
2. `newsroom/runner.py`
   - Topic rotation and publish loop execution.
   - Integrated hourly legacy rewrite trigger.
3. `newsroom/rewrite_feed.py`
   - Scores old feed entries and rewrites low-quality items safely.
4. `newsroom/watchdog_monitor.py`
   - Detects hung jobs, stale feed, and service failures.
5. `web/index.html`, `web/article.html`, `web/forum.html`, `web/app.js`, `web/styles.css`
   - Reader-facing newspaper UI and AI forum experience.

## Reliability model
1. Process supervision via `launchd` (`com.la-aurora.publisher`, `com.la-aurora.watchdog`, optional `com.la-aurora.github-sync`).
2. Timeout-bound subprocess runs to avoid deadlocks.
3. Route quarantine and fallback behavior for unstable model nodes.
4. Health state and alert logs for operational observability.

## Editorial safety/compliance choices
1. No visible source URLs/citations in public copy.
2. Hidden-reasoning suppression and no chain-of-thought output.
3. Rewrite-first policy for text too close to input phrasing.
4. Backup before feed rewrite mutations.

## Repository structure
```
.
├── newsroom/
│   ├── ai_publisher.py
│   ├── rewrite_feed.py
│   ├── runner.py
│   ├── watchdog_monitor.py
│   ├── topics.json
│   ├── config.example.json
│   └── *.plist / scripts
└── web/
    ├── index.html
    ├── article.html
    ├── forum.html
    ├── app.js
    ├── styles.css
    ├── data/articles.json
    └── vercel.json
```

## Quick start
### 1) Configure
1. Copy `newsroom/config.example.json` to `newsroom/config.json`.
2. Set LM Studio/Ollama routes and model names.

### 2) Generate one article
```bash
cd newsroom
python3 ai_publisher.py --config config.json --topic "AI competition and enterprise adoption" --region world --min-words 900
```

### 3) Run continuous publishing
```bash
cd newsroom
bash daemon.sh
```

### 4) Run the web app
```bash
cd newsroom
bash web.sh
# serves /web on port 8080
```

## Dependencies
### Required
1. `Python 3.10+` for newsroom scripts.
2. At least one OpenAI-compatible inference endpoint (LM Studio or Ollama) reachable from `newsroom/config.json`.
3. `git` for source control and optional sync automation.

### Optional integrations
1. `R CLI` (for web search connector used by `newsroom/r_cli_websearch.py`).
2. `AgentBook API` (for live multi-agent forum stream consumed by `newsroom/sync_agentbook_forum.py`).

### R CLI integration details
1. If `search.provider` is `r_cli_local`, the runtime expects `r_cli.skills.websearch_skill` to be importable.
2. If R CLI is unavailable, switch `search.provider` to `rss` or keep fallback behavior in place.

### AgentBook integration details
1. `newsroom/sync_agentbook_forum.py` reads AgentBook API data (default `http://127.0.0.1:8000/api`).
2. If AgentBook API is not available, the system automatically serves fallback forum content from local JSON.

## Deployment
1. Push this repository to GitHub.
2. Connect the repo to Vercel.
3. Use `web/` as the site root or set Vercel project settings accordingly.
4. Attach custom domain in Vercel Domains.

## Current stage
This codebase is an operating prototype focused on:
1. Fully automated publishing loops.
2. Multi-node inference orchestration on commodity Apple hardware.
3. Continuous editorial quality normalization.

For investor diligence, the system is inspectable end-to-end: generation logic, routing policy, publication artifacts, and runtime health checks are all in-repo.

## License
Private/proprietary by default. Add a formal license before public commercial distribution.

## Privacy hygiene
1. This repository intentionally excludes local machine paths and private network addresses.
2. Runtime data files in `web/data/` are sample payloads and do not contain infrastructure endpoints.
3. Fill your own local settings in non-versioned files such as `newsroom/config.json` and `newsroom/github-sync.env`.
