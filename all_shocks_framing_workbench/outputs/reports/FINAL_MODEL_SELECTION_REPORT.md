# Final model selection report

**Status:** Script 13 complete. Notebook and deployment application **not** modified.  
**Population:** n = **5,241** all-shocks modelling households.  
**Financial benchmark (Scripts 11–12):** class-weighted logistic, **93** predictors, macro F1 = **0.492970**, formal F1 = **0.303301**.  
**sklearn:** 1.9.1 (HistGradientBoostingClassifier supports `class_weight='balanced'`).

---

## Verified vs recommendation

| Type | Content |
|---|---|
| **Verified** | Segment absence in 93-feature matrix; Gower+PAM k-screen; CV metrics; fold-safe segment protocol timing |
| **Recommendation** | Selected deployment model / feature set |

---

## Part A — Is agricultural segment membership already a predictor?

**Verified: No.**

The financial-expansion matrix (`11_financial_expansion_features.parquet`) has **93** predictors and **zero** columns matching segment / medoid / cluster / pam. Segmentation has been used for descriptive typology work, not as a classification feature in Scripts 04–12.

### Recomputed all-shocks Spec A typology

- Features: **17** agricultural portfolio + ownership variables (acres **excluded**; shocks/coping **excluded**).
- Distance: Gower (`gower` package; binaries categorical, counts numeric).
- Algorithm: PAM (`segmentation_utils.pam_precomputed`), seed 42, k = 2…6.
- Full-sample Gower runtime: **9.9s**.

| k | Silhouette | Min n | Max n | Max/Min |
|---|---:|---:|---:|---:|
| 2 | 0.2572 | 1741 | 3500 | 2.01 |
| 3 | 0.1906 | 825 | 3114 | 3.775 |
| 4 | 0.1947 | 384 | 2809 | 7.315 |
| 5 | 0.2697 | 465 | 2179 | 4.686 |
| 6 | 0.1475 | 410 | 1151 | 2.807 |

**Working typology selected: k = 5** (best silhouette among k with each cluster ≥3% and n≥100; eligible k=[2, 3, 4, 5, 6]).

Working labels (descriptive profile shorthand — **not** reused drought-only names):

```
 k  cluster                                     working_label  n_households
 5        0 k5_c0_diversified-crops_low-livestock_family-land          2179
 5        1       k5_c1_mixed-crops_low-livestock_family-land          1479
 5        2               k5_c2_narrow-crops_livestock-active           473
 5        3   k5_c3_diversified-crops_low-livestock_own-title           645
 5        4          k5_c4_diversified-crops_livestock-active           465
```

Drought-only k=5 assignments were **not** reused.

Artefacts: `13_all_shocks_segmentation_results.csv`, `13_all_shocks_cluster_sizes.csv`, `13_all_shocks_cluster_profiles.csv`, `13_working_typology_labels.csv`.

---

## Parts B–D — Classification comparison

External protocol: repeated stratified 5-fold × 5 repeats, seed **42** (same as Scripts 07/12).

**Segment feature protocol (fold-safe):** within each external training fold, Gower+PAM is fit on **training households only**; held-out households are assigned to the **nearest training medoid** via Gower. Segment membership is one-hot encoded (`agri_segment_0…k-1`). This completed successfully (not abandoned for full-sample leakage-prone labels).

Timing note: fold-safe segmentation adds substantial runtime (see `13_fold_safe_segmentation_timing.csv`) but remained feasible.

### Scorecard

| Experiment | p | Macro F1 (sd) | Formal F1 (sd) | Balanced acc | Δ macro vs fin-logit |
|---|---:|---:|---:|---:|---:|
| `hist_gradient_boosting_class_weighted__financial` | 93 | 0.4948 (0.0131) | 0.2878 (0.0364) | 0.5158 | +0.0018 |
| `hist_gradient_boosting_class_weighted__financial_plus_segment` | 98 | 0.4933 (0.0131) | 0.2882 (0.0426) | 0.5143 | +0.0003 |
| `logistic_class_weighted__financial_plus_segment` | 98 | 0.4931 (0.0142) | 0.3028 (0.0281) | 0.5474 | +0.0001 |
| `logistic_class_weighted__financial` | 93 | 0.4930 (0.0144) | 0.3033 (0.0274) | 0.5474 | +0.0000 |
| `random_forest_class_weighted__financial` | 93 | 0.4722 (0.0148) | 0.3106 (0.0295) | 0.5283 | -0.0207 |
| `random_forest_class_weighted__financial_plus_segment` | 98 | 0.4710 (0.0136) | 0.3079 (0.0275) | 0.5252 | -0.0220 |

Chart: `outputs/figures/13_final_model_comparison.png`.

Fold-level files: `13_final_fold_metrics.csv`, `13_final_confusion_matrices.csv`, `13_fold_deltas_vs_financial_logistic.csv`.

### Segment membership effect (logistic)

| Metric | Financial logistic | + fold-safe segment | Difference |
|---|---:|---:|---:|
| Macro F1 | 0.4930 | 0.4931 | +0.0001 |
| Formal F1 | 0.3033 | 0.3028 | -0.0005 |
| Folds with higher macro F1 | — | 13 / 25 (52%) | — |

---

## Part E — Selection answers

1. **Does random forest improve on financial logistic?** **No overall.** Macro F1 falls to 0.472 (−0.021; better on only 2/25 folds). Formal F1 rises slightly to 0.311, but the macro drop is too large to prefer RF.
2. **Does histogram gradient boosting improve on it?** **Not enough to switch.** Macro F1 edges up to 0.495 (+0.002; 15/25 folds), but formal F1 falls to 0.288 (−0.015). Net trade-off is unfavourable given priority on formal recognition.
3. **Does agricultural segment membership improve classification?** **No material gain.** Logistic + fold-safe segment: Δmacro ≈ +0.0001, Δformal ≈ −0.0005, higher macro on only 13/25 folds. Same pattern for RF/HGB with segment.
4. **Consistency across folds?** Segment/HGB gains are weak; RF macro losses are consistent. Fold-win counts are descriptive only (not independent significance tests).
5. **Formal-class recognition?** Best formal F1 is RF (0.311), then financial logistic (0.303). HGB and +segment variants do not improve formal recognition vs the financial logistic benchmark.
6. **Is additional complexity justified?** **No.** Segment adds 5 one-hots plus costly fold-safe PAM for ~zero gain. RF/HGB add complexity without a clean win on both macro and formal.
7. **Selected model for deployment:** **`logistic_class_weighted__financial`**

### Selection rationale

- Default champion remains financial-expanded class-weighted logistic regression unless a challenger shows consistent gains on macro F1 and formal F1 without unjustified complexity.
- No challenger delivered a sufficiently consistent improvement on formal and/or macro F1 to justify replacing financial logistic.

Selected validation performance (verified):

| Metric | Value |
|---|---:|
| Experiment | `logistic_class_weighted__financial` |
| Predictors | 93 |
| Macro F1 | 0.492970 ± 0.014359 |
| Formal F1 | 0.303301 ± 0.027424 |
| Balanced accuracy | 0.547440 |
| Weighted F1 | 0.542090 |

Highest macro F1 experiment was `hist_gradient_boosting_class_weighted__financial` (0.4948); highest formal F1 was `random_forest_class_weighted__financial` (0.3106). Selection did **not** automatically equal max macro F1.

---

## Remaining limitations

- Spec A typology silhouette values are modest (typical for socio-agricultural Gower/PAM); clusters are working typology labels, not ground truth.
- Fold-safe segment assignment is rigorous but computationally heavy; full-sample segment labels were not used as CV predictors.
- No extensive hyperparameter tuning; Script 07 initial configs reused.
- Stacking was not rerun.
- Formal coping remains the hardest class.
- Deployment packaging is out of scope for this script (application not modified).

---

## Outputs

| File | Description |
|---|---|
| `13_all_shocks_segmentation_results.csv` | k-screen silhouette / sizes |
| `13_financial_model_comparison.csv` | Model scorecard |
| `13_segment_feature_comparison.csv` | With/without segment summary |
| `13_final_fold_metrics.csv` | 25-fold metrics |
| `13_final_confusion_matrices.csv` | Confusion matrices |
| `13_final_model_comparison.png` | Macro F1 chart |
| `FINAL_MODEL_SELECTION_REPORT.md` | This report |
