# AI use disclosure

This project used AI tools under the course AI-use policy. Human review and decisions remain the author’s responsibility.

## Tools used

| Tool | Role |
|---|---|
| **Cursor (Composer / agent)** | Coding assistance for FastAPI, Next.js UI, Docker/compose, prediction-chat grounding, Overview shock-aware wiring, docs, and deploy config |
| **Groq-hosted LLM** (`qwen/qwen3.8-27b` via OpenAI-compatible API) | Optional natural-language explanations in Predictions follow-up chat only; receives structured Facts JSON (scores/attributions), not raw household interview dumps |

## What AI did **not** decide alone

- Problem framing (all-shocks vs drought-only), Stack A feature set (93 predictors), and champion vs logistic fallback selection were driven by scripted experiments and written reports under `all_shocks_framing_workbench/outputs/reports/`.
- Metrics, artefact versions, and serving contracts were verified from training/audit outputs and API tests, not accepted from chat alone.
- Secrets (API keys) are kept in local `.env` / host env vars and are not committed.

## Reviewer note

Prediction-chat replies may fall back to a **template** explanation if the LLM is unavailable; the UI labels the source (language model vs template).
