# Prototype — FinAccess all-shocks coping scorer

## Locked deployment candidates

| Role | Version | Path |
|---|---|---|
| Preferred | **v0.2.0 Stack A** | `artefacts/classifier/v0.2.0` |
| Fallback | v0.2.0 logistic | `artefacts/classifier/v0.2.0-logistic-fallback` |

Active pointer: `artefacts/metadata/ACTIVE_MODEL.json`.

**Features (93):** Model A household/agri + R2B main-shock one-hots + 13 financial access flags.  
**Population:** 5,241 shock-exposed farming households with valid four-class coping.  
**5-fold (first repeat) Stack A:** macro F1 ≈ 0.502; formal F1 ≈ 0.315.

## Run locally

```bash
# API (loads .env for Groq when prediction-chat uses the LLM)
cd backend && PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Web
cd frontend && npm run dev
```

Or `docker compose up --build` (API reads `.env` for `LLM_API_KEY` / Groq).

## Language model (backend only)

The standalone Assistant page was removed. Shared helpers live in `backend/app/llm.py` for the Predictions conversation (Change 2). Set in `.env` (gitignored):

- `LLM_API_KEY` — Groq key  
- `LLM_API_BASE=https://api.groq.com/openai/v1`  
- `LLM_MODEL=qwen/qwen3.8-27b` (free Groq-hosted; reasoning disabled in API calls)

## Notes

- Nav is Overview + Predictions only (standalone Assistant page removed).  
- `/api/v1/predict` serves Stack A (contract unchanged).  
- `/api/v1/explain` returns approximate question-level grouped Shapley (paired reversed permutations) for Stack A, or `explanation_mode=probability_only` if attribution fails. UI wording uses “among the main factors” and magnitude bands — never a single biggest factor. Never uses the logistic fallback to explain Stack A.  
- Follow-ups: `POST /api/v1/prediction-chat/follow-up` (LLM optional; keys stay on the API). Facts JSON is scores/attributions only — not raw interview answers.  
- Legacy `/assessment` and `/prediction` redirect to `/predictions`.  
- Stack B was stopped incomplete and is not packaged.  
- Original drought notebook was not modified.
