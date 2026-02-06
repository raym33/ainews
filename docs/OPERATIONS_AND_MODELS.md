# Operations, Performance, and Model Planning

This document explains practical operating profiles for Metropolis on a small on-prem inference cluster, including:
1. expected generation times,
2. role routing across multiple Macs,
3. recommended open-source models,
4. optional external API connectors.

## 1) Performance expectations
Numbers below are field ranges for article jobs in the current pipeline (`topic -> draft -> cleanup -> publish`), not synthetic benchmark maxima.

Assumptions:
1. Apple Silicon machines with 16 GB RAM.
2. Quantized models (4-bit to 6-bit) loaded in LM Studio or Ollama.
3. Articles in the 900 to 1,600 word range.
4. Parallel role calls enabled where possible.

| Cluster shape | Typical article latency | Sustained throughput | Notes |
| --- | --- | --- | --- |
| 1 node, single 8B model | 5 to 11 min | 5 to 9 articles/day | Stable baseline, lowest complexity |
| 2 nodes (chief+research split) | 3.5 to 8 min | 9 to 16 articles/day | Better queue smoothing |
| 4 nodes (role-specialized) | 2.5 to 6.5 min | 16 to 34 articles/day | Best quality/latency balance |
| 5+ nodes with forum traffic | 2.5 to 7.5 min | 14 to 30 articles/day + forum turns | Throughput depends on debate load |

Why ranges are wide:
1. topic complexity varies,
2. retries increase when remote nodes timeout,
3. rewrite passes add quality but cost latency.

## 2) Recommended role topology (4-5 machines)
Use role specialization so each node keeps a stable KV cache profile instead of constant model swapping.

Example assignment:
1. `chief` node (LM Studio): headline, deck, outline, final editorial pass.
2. `research` node (LM Studio): first long draft.
3. `fact` node (LM Studio): consistency checks and claim softening.
4. `polish` node (Ollama or LM Studio): style normalization, anti-copy rewrite.
5. `forum` node (optional): AI debate generation isolated from newsroom traffic.

Operational rule:
1. keep at least one fallback route per role,
2. quarantine unstable nodes after repeated timeout failures,
3. keep forum/debate workloads off the article path when possible.

## 3) Model recommendations (16 GB Apple Silicon)
Pick a mixed stack by function instead of one model for every step.

### Primary newsroom models
1. `qwen/qwen3-8b`:
   - best for outline + long drafting under tight memory.
2. `mistralai/ministral-3b` (or similar 3-4B class):
   - efficient fact/style pass with low latency.
3. `gemma3:12b` (Ollama):
   - strong editorial rewrite quality when latency budget allows.

### Optional supporting models
1. small 1B-3B model for tagging/categorization only,
2. embedding model such as `text-embedding-qwen3-8b` for similarity checks and anti-copy guards.

Guidance:
1. use larger model only in the final rewrite stage if hardware is limited,
2. keep categorization/tagging on smaller models to preserve throughput,
3. avoid loading multiple heavy models per 16 GB node concurrently.

## 4) LM Studio + Ollama mixed operation
Metropolis supports OpenAI-compatible endpoints. In practice:
1. LM Studio nodes serve article and forum generation roles.
2. Ollama nodes can be used as fallback or dedicated rewrite workers.
3. Routing is configured per role in `newsroom/config.json`.

Best practice:
1. enable parallel requests in LM Studio,
2. keep request timeouts explicit and conservative,
3. tune max tokens per role so smaller passes do not block long jobs.

## 5) External API connectors (optional)
The pipeline can ingest non-LLM data from external APIs before drafting. This is useful for market and crypto sections.

Typical connector classes:
1. news APIs (headlines, metadata, recency filters),
2. market APIs (equities, FX, macro),
3. crypto APIs (spot prices, market cap, volume),
4. company/event APIs for tech product tracking.

Integration pattern:
1. pull structured snapshots,
2. normalize into a source bundle,
3. pass bundle to drafting prompt as factual constraints,
4. enforce rewrite policy so output is original and non-derivative.

This keeps article text original while using verified upstream facts.

## 6) Editorial quality policy for copyright safety
To reduce derivative risk:
1. never reuse external headlines verbatim,
2. force original title/deck generation from abstracted facts,
3. run overlap checks against source snippets,
4. trigger an automatic rewrite pass when overlap is above threshold.

This policy is compatible with the current feed rewriter (`newsroom/rewrite_feed.py`).

## 7) 24/7 viability checklist
For continuous operation:
1. run publisher + watchdog as supervised services,
2. separate website serving from generation workers,
3. keep hourly GitHub sync isolated from generation loop,
4. rotate logs and set disk growth limits,
5. keep one warm standby route per critical role.

With these controls, continuous operation is operationally viable on a small local cluster.

