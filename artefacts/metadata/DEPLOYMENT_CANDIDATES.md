# Deployment candidates (locked selection)

| Role | Artefact | Algorithm | 5-fold macro F1 | 5-fold formal F1 |
|---|---|---|---:|---:|
| **Preferred** | `artefacts/classifier/v0.2.0` | Stack A (logistic + HGB → logistic meta) | 0.502 | 0.315 |
| **Fallback** | `artefacts/classifier/v0.2.0-logistic-fallback` | Class-weighted logistic | 0.494 | 0.310 |

**Active pointer:** `artefacts/metadata/ACTIVE_MODEL.json` → preferred Stack A.

**Feature spec:** all-shocks financial expansion, **93** predictors (Model A + R2B main-shock one-hots + 13 financial access flags). No agricultural segment membership.

**Stack B:** incomplete (stopped after 3/5 folds); not packaged.

**Notes:** SHAP LinearExplainer is not shipped for Stack A. `/api/v1/explain` serves approximate question-level grouped Shapley (`artefacts/shap/v0.2.0`) with `probability_only` fallback on failure; never explain Stack A with the logistic fallback. `/api/v1/predict` contract is unchanged.
