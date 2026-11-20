# AML Capstone — FinAccess all-shocks coping scorer

Predict four-class **coping responses** (personal, informal, formal, did_nothing) for shock-exposed Kenyan farming households using FinAccess 2024 features. The repo includes a reproducible data/modelling workbench, serialized Stack A artefacts, a **FastAPI** scoring/explain API, and a **Next.js** Overview + Predictions UI.

**Live demo:** _pending Render deploy — URL will be updated here_

**GitHub:** https://github.com/jakeschege/AML_capstone

## Architecture

```text
Browser (Next.js)  --NEXT_PUBLIC_API_BASE-->  FastAPI (uvicorn)
                                              |-- artefacts/classifier/v0.2.0 (Stack A)
                                              |-- grouped Shapley explain
                                              |-- prediction-chat (optional Groq LLM)
                                              '-- dashboard JSON (shock-aware overview)
```

| Layer | Path |
|---|---|
| Data pipeline | `finaccess_coping_workbench/scripts/` (02–10 prep), `all_shocks_framing_workbench/` (01–04 cohort/features) |
| Training / selection | `all_shocks_framing_workbench/` (05–14), `finaccess_coping_workbench/scripts/` (19–21) |
| Serving artefacts | `artefacts/classifier/v0.2.0` (+ logistic fallback), `artefacts/dashboard/`, `artefacts/shap/` |
| API | `backend/app/` (FastAPI) |
| UI | `frontend/` (Next.js) |
| Docker | `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`, `render.yaml` |

**Active model:** Stack A, 93 features, `artefacts/metadata/ACTIVE_MODEL.json` (macro F1 ≈ 0.502 on 5-fold first-repeat; formal F1 ≈ 0.315). Population n = 5,241 shock-exposed farming households with valid four-class coping.

## Quick start (local)

### Prerequisites

- Python 3.12+, Node 22+, Docker optional
- Copy [`.env.example`](.env.example) → `.env` and set `LLM_API_KEY` (Groq) if you want LLM explanations in Predictions chat

### Docker (recommended)

```bash
docker compose up --build
```

- API: http://localhost:8000  (health: `/api/v1/health`)
- Web: http://localhost:3000

### Without Docker

```bash
# API
cd backend && PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Web (separate terminal)
cd frontend && npm install && npm run dev
```

Set `NEXT_PUBLIC_API_BASE=http://localhost:8000` for the frontend when not using Compose defaults.

## Reproduce the project

1. **Prepared matrices** used by the API live under `finaccess_coping_workbench/outputs/prepared/` (raw FinAccess microdata is **not** in this repo).
2. **Pipeline / audits:** run numbered scripts in `finaccess_coping_workbench/scripts/` (see that folder’s README and logs).
3. **All-shocks framing & model selection:** run `all_shocks_framing_workbench/` scripts 01→14; reports under `outputs/reports/` (e.g. `FINAL_MODEL_SELECTION_REPORT.md`).
4. **Serve:** load Stack A via FastAPI (`/api/v1/predict`, `/api/v1/explain`, `/api/v1/prediction-chat/*`, overview endpoints).

More prototype notes: [PROTOTYPE.md](PROTOTYPE.md). Methodology / limitations: [`knowledge_base/`](knowledge_base/).

## Course deliverables

| Item | Location |
|---|---|
| README | this file |
| AI use disclosure | [AI_USE.md](AI_USE.md) |
| Data pipeline | `finaccess_coping_workbench/`, `all_shocks_framing_workbench/` |
| Training + artefacts | scripts above + `artefacts/` |
| Docker | `docker-compose.yml`, Dockerfiles, `render.yaml` |
| API | `backend/` (FastAPI) |
| Final report / slides | [docs/](docs/) — **slide deck to be added** |

## Deploy (Render)

Blueprint: [`render.yaml`](render.yaml). After the API service is live, set the web service build/runtime `NEXT_PUBLIC_API_BASE` to the public API URL and redeploy the web service. Never commit `.env`; set `LLM_*` as Render secrets.

## Licence / data note

FinAccess survey microdata remains subject to its original access terms. This repository ships prepared modelling extracts and code only.
