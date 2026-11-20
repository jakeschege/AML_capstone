# Final stacking assessment

**Status:** Stack A **complete**. Stack B **stopped incomplete** (user decision before deployment).  
**Feature set:** financial-expanded matrix, **93** predictors (no agricultural segment).  
**Validation:** first stratified 5-fold repeat of `RepeatedStratifiedKFold(..., random_state=42)`.  
**Comparisons are 5-fold means only** — not equated with prior 25-fold averages.

---

## Experiment stop decision (verified)

| Item | Status |
|---|---|
| Stack A (logistic + HGB → logistic meta) | **Complete** (5/5 folds; ~1041 s) |
| Stack B (logistic + HGB + RF → logistic meta) | **Incomplete** (3/5 folds observed, then process killed) |
| Why stopped | Stack A already improved macro and formal F1 vs individuals; RF adds cost without demonstrated overall gain; move to deployment packaging |

Partial Stack B folds are preserved in `14_stack_b_incomplete_partial_folds.csv` and **must not** be treated as a finished evaluation.

---

## 5-fold scorecard (verified)

| Experiment | Folds | Macro F1 | Formal F1 | Status |
|---|---:|---:|---:|---|
| Financial logistic | 5 | 0.4937 | 0.3098 | complete |
| Histogram GB | 5 | 0.4959 | 0.2987 | complete |
| Random forest | 5 | 0.4677 | 0.3008 | complete |
| **Stack A** | 5 | **0.5023** | **0.3146** | **complete** |
| Stack B | 3 | ~0.4927 (partial) | ~0.3222 (partial) | **incomplete** |

Stack A fold-level macro F1: 0.4947, 0.5101, 0.4918, 0.5146, 0.5004.

---

## Interpretation

- Stack A improves **macro F1** (~0.502) beyond individual logistic (~0.494) and HGB (~0.496) on this 5-fold protocol.
- Stack A also improves **formal F1** (~0.315) vs logistic (~0.310) and HGB (~0.299).
- Stack B was **not** fully evaluated; no claim that Stack A is conclusively superior to Stack B.

---

## Deployment candidates (recommendation)

| Model | Role |
|---|---|
| **Stack A** | Preferred predictive candidate for packaging |
| **Financial-expanded logistic** | Simpler benchmark / fallback |

No further algorithm search. Next step: save Stack A artefact, prediction smoke test, then wire serving.

---

## Outputs

| File | Role |
|---|---|
| `14_stacking_results.csv` | Scorecard including incomplete Stack B row |
| `14_stacking_fold_metrics.csv` | Fold metrics (individuals + Stack A) |
| `14_stack_b_incomplete_partial_folds.csv` | Partial Stack B folds only |
| `14_experiment_stop_decision.csv` | Stop rationale |
| `FINAL_STACKING_ASSESSMENT.md` | This report |
