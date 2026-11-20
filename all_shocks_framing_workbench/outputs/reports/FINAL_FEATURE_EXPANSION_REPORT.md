# Final feature expansion report

**Status:** Scripts 10–12 complete. Notebook and deployment application **not** modified.  
**Workbench:** `all_shocks_framing_workbench/`  
**Population:** n = **5,241** shock-exposed farming households with valid four-class coping (non-shock 4,499 **not** included).  
**Baseline:** Model A + R2B class-weighted multinomial logistic regression (Script 07 macro F1 ≈ **0.4765**).  
**Protocol:** repeated stratified 5-fold CV × 5 repeats, seed **42**; class-weighted logistic only (no stacking rerun; no extensive tuning).

---

## How to read this report

| Type | Meaning |
|---|---|
| **Verified** | Measured availability, missingness, redundancy, CV metrics |
| **Recommendation** | Judgement for deployment locking |

---

## 1. Verified variable availability

Script 10 reviewed Script 09 inventories, the prepared modelling frame, extract crop detail, and Stata `Data 18+.dta` (pyreadstat labels; no external value-label file).

| Source | Finding |
|---|---|
| Prepared farming / modelling frame | Farm challenges, financial usage flags, membership flags present |
| Extract crop/livestock detail | Available but redundant with Model A portfolio counts |
| Stata welfare/food/remittance gaps | Join on `respondent_id` = `interview__id`: **5,172 / 5,241** matched (98.7%); **69** unmatched; **0** Stata ID duplicates |

Details: `FEATURE_EXPANSION_ELIGIBILITY_REPORT.md`, `10_feature_expansion_inventory.csv`, `10_stata_join_qc.csv`.

---

## 2. Variables approved for modelling

**31 approved candidates** (`10_approved_candidates.csv`):

| Group | n | Content |
|---|---:|---|
| Agricultural diversity | **0** | None — all diversity candidates redundant with Model A |
| Farm challenges | **11** | Inputs, financing, market access, land size, labour (not drought/flood/pest) |
| Financial | **13** | Savings/loan/SACCO/bank/mobile money/insurance/agri credit access flags |
| Welfare / support | **7** | `B1Di`, `B3A__9`, `nfhi_11`, `nfhi_12`, `mfhi_32`, farmer assoc / coop membership |

Expanded matrices (`11_*_features.parquet`) keep identical respondent order and target labels.

| Specification | Predictors | Added vs baseline |
|---|---:|---:|
| Baseline Model A + R2B | 80 | 0 |
| Agri diversity expansion | 80 | 0 (empty group) |
| Farm challenge expansion | 91 | 11 |
| Financial expansion | 93 | 13 |
| Welfare expansion | 87 | 7 |
| Combined expansion | 111 | 31 |

---

## 3. Variables excluded (redundancy, missingness, leakage)

| Reason | Examples |
|---|---|
| Redundant with Model A | `shannon_diversity`, `n_crop_livestock_items_positive`, detailed `crop_livestock_*`, `has_any_cash_crop` / `has_any_dairy` |
| Shock-module overlap | `farm_reported_drought_challenge`, flood / pests / theft / wildlife challenges |
| Near-zero prevalence | Rare farm-challenge flags (storage, low yields, extension, etc.) |
| >40% missing after join | HSNP, government assistance, remittance-use detail (`B3H4`, `B3H6`, `G7`, `G12`, `B1Diii`) |
| Target / coping leakage | All `coping_*`, cut-back / asset-sale / repayment-consequence Stata items |
| Duplicate financial coding | `informal_usage`, `used_*` mirrors of approved access flags |

Full list: `10_excluded_candidates.csv`.

---

## 4. Performance (verified CV results)

Class-weighted logistic regression; 25 folds; same seed/folds across specs.

| Specification | Macro F1 mean (sd) | Formal F1 mean | Balanced accuracy | Δ macro vs baseline |
|---|---:|---:|---:|---:|
| Baseline A+R2B | **0.4765** (0.012) | 0.254 | 0.527 | — |
| Agri diversity | 0.4765 (0.012) | 0.254 | 0.527 | 0.000 |
| Farm challenges | 0.4763 (0.014) | 0.252 | 0.525 | −0.0002 |
| Financial | **0.4930** (0.014) | **0.303** | 0.547 | **+0.016** |
| Welfare | 0.4873 (0.013) | 0.274 | 0.539 | +0.011 |
| Combined | **0.4963** (0.014) | 0.301 | 0.548 | **+0.020** |

Fold consistency vs baseline (descriptive counts only — folds are **not** independent tests):

| Specification | Folds with higher macro F1 | Folds with higher formal F1 |
|---|---:|---:|
| Farm challenges | 15 / 25 (60%) | 10 / 25 |
| Financial | **24 / 25 (96%)** | **24 / 25** |
| Welfare | 23 / 25 (92%) | 21 / 25 |
| Combined | **24 / 25 (96%)** | **24 / 25** |

Chart: `outputs/figures/12_feature_expansion_macro_f1.png`.  
Tables: `12_feature_expansion_scorecard.csv`, `12_feature_expansion_fold_metrics.csv`, `12_feature_expansion_fold_differences.csv`, `12_feature_expansion_confusion_matrices.csv`.

---

## 5. Answers to the evaluation questions

1. **Which additional group contributes most?**  
   **Financial characteristics** (largest single-group macro and formal gains).

2. **Does agricultural diversity improve classification?**  
   **No** — no eligible non-redundant variables; matrix identical to baseline.

3. **Do farm challenges add useful information?**  
   **Not materially** on their own (macro ≈ flat; formal slightly worse).

4. **Do financial characteristics improve classification?**  
   **Yes** — macro +0.016; formal F1 +0.049; improved on 24/25 folds.

5. **Do welfare / support variables contribute?**  
   **Yes, moderately** — macro +0.011; formal +0.020; 23/25 folds.

6. **Does the combined set improve on baseline?**  
   **Yes** — best mean macro F1 (0.496); formal gain similar to financial alone.

7. **Are improvements consistent across folds?**  
   **Yes for financial and combined** (24/25). Farm challenges are inconsistent.

8. **Does expansion justify complexity?**  
   **Financial expansion: yes.** Combined adds +18 features beyond financial for only ~+0.003 macro and **no** formal-F1 gain over financial — weak complexity case.

---

## 6. Remaining limitations

- Cross-sectional FinAccess 2024 data: farm challenges and welfare flags may be concurrent with shocks, not proven pre-shock traits.
- 69 modelling IDs lack Stata welfare fields (imputed inside CV only).
- Evaluation used class-weighted logistic only (by design); stacking was not rerun.
- Repeated CV fold counts are descriptive stability checks, not significance tests.
- Formal class remains difficult (formal F1 ~0.30 even after expansion).

---

## 7. Deployment recommendation

**Recommendation: lock `financial_expansion` (Model A + R2B + 13 approved financial access flags) as the current deployment feature specification.**

Reasons:

1. **Verified** consistent improvement over the prior champion baseline (macro and formal; 24/25 folds).
2. **Best formal-class F1** among all specs (0.303 vs 0.254 baseline).
3. **Lower complexity** than combined (93 vs 111 predictors) while capturing most of the gain.
4. Farm-challenge-only expansion did not help; including those flags in combined adds complexity without improving formal recognition beyond financial alone.

**Alternative (if maximising macro F1 is the sole criterion):** `combined_expansion` (0.496). Not preferred here because the incremental macro gain over financial is small and formal F1 is not better.

**Do not retain baseline A+R2B** as the locked deployment matrix — financial expansion is a sufficiently consistent improvement under the study’s own criteria.

**Do not** add the 4,499 non-shock households to this classifier without a comparable four-class outcome.

---

## Artefacts

| Path | Role |
|---|---|
| `10_*.py` / `10_*.csv` / `FEATURE_EXPANSION_ELIGIBILITY_REPORT.md` | Eligibility |
| `11_*_features.parquet` | Expanded matrices (new files only) |
| `12_feature_expansion_*.csv` / `12_feature_expansion_macro_f1.png` | Evaluation |
| `FINAL_FEATURE_EXPANSION_REPORT.md` | This report |

*Original notebook, prior Model A/B matrices, Scripts 07–08 outputs, and the deployment application were not modified.*
