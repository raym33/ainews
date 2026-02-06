# Metropolis Newsroom

Automated newsroom engine for the Metropolis web product.

## What it does
1. Generates long-form news analysis with AI roles (`chief`, `research`, `fact`, `tagger`).
2. Collects source material via web search and RSS.
3. Applies post-processing to remove reasoning artifacts and enforce editorial style.
4. Publishes output to `web/data/articles.json`.
5. Continuously rewrites weak legacy articles for feed consistency.

## Quick start
```bash
cd newsroom
cp config.example.json config.json
python3 ai_publisher.py --config config.json --topic "AI competition and enterprise adoption" --region world --min-words 900
```

## Dependencies
### Required
1. `Python 3.10+`.
2. One OpenAI-compatible model endpoint (LM Studio or Ollama) configured in `config.json`.

### Optional
1. `R CLI` for `r_cli_local` search mode (`r_cli.skills.websearch_skill`).
2. `AgentBook API` for live forum sync via `sync_agentbook_forum.py`.

Fallback behavior:
1. If R CLI is missing, use RSS provider (`search.provider: rss`) or keep existing fallback logic.
2. If AgentBook API is unavailable, forum rendering falls back to local JSON snapshots.

## Continuous run
```bash
cd newsroom
bash daemon.sh
```

## Web server
```bash
cd newsroom
bash web.sh
# serves ../web on port 8080 by default
```

## Cluster routing
- Configure route/model assignments in `config.json`.
- Use `routes.4macs.example.json` as a template for multi-node setups.

Helpful scripts:
- `discover_lm_workers.py`
- `auto_configure_cluster.py`
- `configure_cluster_routes.py`
- `check_lm_cluster.sh`

## Health and monitoring
- `watchdog_monitor.py` monitors liveness and stale-feed risk.
- `live_view.sh` provides a terminal dashboard of publishing activity.

## GitHub/Vercel automation
- `github_hourly_sync.sh` can push snapshots on schedule.
- `setup_github_sync_launchd.sh` installs launchd job.
- `VERCEL_DOMAIN_CHECKLIST.md` describes deployment steps.

## Privacy note
This repository contains only template configuration and public-safe defaults.
Local machine paths, private network addresses, and personal runtime settings are intentionally excluded.
