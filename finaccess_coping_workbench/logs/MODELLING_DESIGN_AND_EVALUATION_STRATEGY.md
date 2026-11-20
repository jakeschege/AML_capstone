# Modelling Design and Evaluation Strategy

Living design document for the CRISP-DM Modelling phase. This document establishes objectives and evaluation rules before algorithms are fitted.

**Status: C2/C3 four-class primary nonlinear experiments completed (Script 21). Acres-excluded default. Three-class and extended tracks remain deferred pending review.**

Prepared inputs (Data Preparation closed):

| Feature set | Features | Artefact |
|---|---:|---|
| Segmentation | 18 | `outputs/prepared/09_X_segmentation.parquet` |
| Primary classification | 69 | `outputs/prepared/09_X_primary_classification.parquet` |
| Extended classification | 92 | `outputs/prepared/09_X_extended_classification.parquet` |
| Shannon sensitivity | 1 | `outputs/prepared/09_X_sensitivity_shannon.parquet` |
| Targets | four- and three-class | `outputs/prepared/09_y_targets.parquet` |

Sample: `analysis_2024`, n = 1,552 drought-exposed households.

**Principle:** Keep segmentation and classification as separate experimental workflows. Segmentation first establishes agricultural household groups. Classification then examines coping behaviour, including whether it varies across those groups.

---

## 1. Modelling objectives and evaluation strategy

### Segmentation objective

Identify empirically meaningful agricultural and land-based household groups among drought-exposed households using the 18-feature segmentation matrix (portfolio indicators, crop/livestock counts, land size in acres, land ownership indicators).

Success is judged by:

- Stability and interpretability of groups
- Agricultural/land coherence of profiles
- Usefulness for later description of coping patterns across groups

Segmentation is unsupervised. It does not predict coping class.

### Classification objectives

1. **Primary classification** — predict coping category from household demographics and agricultural/land characteristics available in the primary matrix, without retrospective shock indicators.
2. **Extended classification** — examine whether adding retrospective same-window shock information improves discrimination in a clearly labelled retrospective analysis.

Targets:

- Four-class: `coping_category` (personal, informal, formal, did_nothing) — **primary target**
- Three-class alternative: `coping_category_collapsed`

### Shared evaluation rules

- Any imputation, scaling, or other data-dependent transforms fit **within training folds only**
- Stratified validation required for classification, with explicit checks on formal-class representation per fold
- Extended results must not be described as prospective prediction
- Missing `land_size_acres` never automatically set to zero; missingness indicator only if separately justified

---

## 2. Locked segmentation design decisions

| Design area | Decision |
|---|---|
| Distance | Gower |
| Primary algorithm | PAM (Partitioning Around Medoids) |
| Comparison algorithm | Hierarchical clustering using Gower distance |
| Candidate cluster counts | \(k = 2\) to \(k = 6\) |
| Primary cluster metric | Average silhouette width |
| Additional evaluation | Cluster sizes, stability under resampling, interpretability (not silhouette alone) |
| Spec A (primary baseline) | Exclude `land_size_acres`; **17 features**; equal feature weights; retain nine agri indicators and both portfolio counts |
| Spec B (acreage sensitivity) | Retain valid acres; keep community-land acres conceptually missing; impute the 16 non-community special-response cases with the **median of observed acres**; add indicator for those 16; add `is_community_land` from unit code 5; **never zero-fill** |
| Feature weighting | Equal feature weights primary; domain-weighted sensitivity later |
| Portfolio counts | Retain in primary; exclude both counts in a sensitivity |
| Shannon diversity | Sensitivity sidecar only; not default segmentation input |
| Coping outcomes | Descriptive profiling only after segmentation; never used to form clusters or choose \(k\) |
| Random seed (S1) | 42 |

### Spec B acreage treatment — locked (Script 15)

The installed `gower` package propagates NaNs into distances, so Spec B uses a **missing-aware Gower** implementation (`segmentation_utils.gower_matrix_missing_aware`) that excludes features missing in either member of a pair from that pair’s average distance.

| Element | Treatment |
|---|---|
| Valid measured acres | Retain `land_size_acres` |
| Community land (unit 5, n=182) | Acres remain **missing** (not zero-filled); `is_community_land=1` |
| Special response 98/99, non-community (n=16) | Impute acres with **median of observed acres only**; `acres_special_imputed=1` |
| Zero-fill | Prohibited |
| Spec A results | Unchanged; Spec B is a sensitivity comparison |
| Candidates | Same k ∈ {{2, 4, 5}} |
| Coping | Not used |

Comparability note: Spec B adds land-size information and two indicators; distances for community-land pairs omit the acres feature when it is missing, so community distinction is carried primarily by `is_community_land` and ownership/agri features.

### Experiment sequence

| Experiment | Objective | Main approach |
|---|---|---|
| S1 | Spec A baseline | Gower + PAM for \(k=2..6\) (Script 12) |
| S2 | Stability and profile coherence | Resampling / agreement for promising \(k\) |
| S3 | Spec B acreage contribution | After Gower missing-handling decision |
| S4 | Weighting and drop-counts sensitivity | Domain weights; exclude portfolio counts |

Stability means resampling agreement of assignments, not temporal behavioural stability.

---

## 3. Locked classification design decisions

| Design area | Decision |
|---|---|
| Validation | Repeated stratified five-fold CV, five repetitions |
| Fold control | Identical fold assignments for primary and extended; fixed random seeds; class distribution reported per fold |
| Nested CV | Only if extensive hyperparameter tuning becomes necessary; avoid for initial experiments |
| Four-class target | Primary target |
| Three-class target | Alternative target |
| Primary metric | Macro F1 |
| Additional metrics | Balanced accuracy; class-wise precision, recall, F1; weighted F1; confusion matrices; overall accuracy as descriptive secondary |
| Imbalance strategy | Compare unweighted vs class-weighted learning; no SMOTE by default |
| Scaling / imputation | Fit within training folds only |
| Land-size in classification | Compare exclude-acres baseline; retain with train-fold imputation; optional missingness indicator if justified |
| Primary vs extended | Identical validation folds and evaluation metrics; distinct interpretation |

### Baseline model progression (experiment sequence; not a commitment to fit all regardless of results)

| Model | Purpose |
|---|---|
| Dummy classifier | Basic reference |
| Multinomial logistic regression | Interpretable linear baseline |
| Decision tree | Simple nonlinear relationships |
| Random forest | Nonlinear interactions |
| Gradient boosting | Flexible ensemble |

Do not begin with deep learning for this sample size.

### Experiment groups (design structure)

| Experiment | Objective | Main approach |
|---|---|---|
| C1 | Establish classification baseline | Dummy and multinomial logistic regression |
| C2 | Evaluate nonlinear models | Decision tree and random forest |
| C3 | Evaluate ensemble performance | Gradient boosting |
| C4 | Assess class imbalance | Unweighted versus class-weighted learning |
| C5 | Compare target definitions | Four-class versus three-class |
| C6 | Compare information sets | Primary versus extended (identical folds/metrics) |

---

## 4. Comparative experiments

- Primary classification evaluates household characteristics vs coping categories.
- Extended classification examines whether retrospective shock information improves classification within the same retrospective survey context.
- An improvement in extended classification does **not** establish prospective prediction before shock or coping response.
- If segmentation groups are later used with classification, keep them as descriptive strata or separately derived features without using target labels to form clusters.

---

## 5. Modelling carry-forward from Data Preparation

1. Missing `land_size_acres` (n = 198) require appropriate handling (fold-wise for classification; documented pre-clustering treatment for segmentation Spec B).
2. Formal class n = 75 requires stratified validation and careful evaluation; consider the collapsed three-class target.
3. Extended shock information is same-window retrospective; interpret accordingly.

---

## 6. Segmentation design audit (Script 11) — approved

Audit report: `outputs/reports/11_segmentation_design_audit.md`.

Approved findings:

- Features are not 18 independent dimensions; indicator–count overlap is material.
- Equal-weight Gower gives agri groups and ownership most column share.
- Missing acreage is largely community land and is associated with community ownership.
- Full Gower + PAM is computationally feasible at n=1,552.

## 7. S1 status

Script 12 completed Spec A (Gower + PAM, \(k=2..6\)). Report: `outputs/reports/12_segmentation_s1_spec_a.md`.

Profile review retained **k = 2, 4 and 5** as S2 candidates (coarse / intermediate / detailed typologies). No final \(k\) selected. Coping outcomes were not used.

## 8. S2 stability design (Script 13) — locked

| Parameter | Value |
|---|---|
| Candidates | k = 2, 4, 5 |
| Spec | A (17 features; acres excluded; equal weights) |
| Resample | Subsample without replacement, fraction 0.80 |
| Replicates | R = 50 |
| Shared indices | Same subsample index sets for all three k |
| Random seed | 42 |
| Agreement | Adjusted Rand Index vs S1 labels on the subsample |
| Cluster-level | Max Jaccard of each S1 cluster with resample clusters |
| Exclusions | No Spec B, no domain weights, no drop-counts, no coping, no final k |

Stability means resampling agreement of assignments, not temporal behavioural stability. High overall ARI alone does not decide k; inspect cluster-level stability (especially k=5 legumes vs cereals-without-legumes groups).

## 9. Script 14 — k=5 family crop robustness

Targeted investigation of whether k=5 `family_legumes_heavy` and `family_cereals_without_legumes` are reproducible subtypes or an unstable split of family crop agriculture, compared with k=4 family crop groups.

Report: `outputs/reports/14_segmentation_k5_family_crop_robustness.md`

Provisional preference remains k=5 subject to this evidence; k=4 retained as comparator. Spec B, coping profiling and final k selection remain on hold.

## 10. Spec B (Script 15) — run completed

Spec B land-size sensitivity completed for k=2, 4, 5 using missing-aware Gower. Spec A results unchanged. Final k still pending review of Spec A vs Spec B agreement and community/acreage cluster composition.

Report: `outputs/reports/15_segmentation_s3_spec_b.md`

## 11. Spec A/B correspondence (Script 16)

Membership overlap audit for k=4 and k=5 using greedy maximum-overlap matching. Supports the provisional working typology decision (k=5 preferred, k=4 alternative) once reviewed.

Report: `outputs/reports/16_segmentation_spec_ab_correspondence.md`

### Working Segmentation Typology Decision

Following the completion of the Spec A stability investigation, the k=5 family crop robustness investigation and the Spec A/B correspondence audit, the five-cluster solution is adopted as the working segmentation typology.

Spec A remains the primary segmentation specification. Spec B is retained as a sensitivity analysis incorporating observed land size, a community land indicator and a separate indicator for special non-community acreage imputations. The Spec A and Spec B solutions show strong agreement at k=5, with an Adjusted Rand Index of 0.92. All five cluster profiles remain recognisable under both specifications.

The working clusters are described as follows.

1. Family-owned legume-oriented crop households
2. Own-name crop households
3. Community-land livestock-oriented households
4. Family-owned mixed crop and livestock households
5. Family-owned cereal-oriented households

These labels summarise the dominant characteristics of each cluster and should not be interpreted as exhaustive descriptions of every household within a group.

The cereals-oriented group requires particular interpretive caution. Although its original profile is characterised by the absence of legumes, the earlier resampling investigation found weaker membership and medoid consistency than for the other groups. The distinction between this group and the legume-oriented group should therefore not be presented as an entirely sharp or permanent household boundary.

The k=4 solution is retained as a sensitivity alternative. Its weaker recovery of the lease-crop group provides an additional reason for preferring k=5 as the working solution.

The primary cluster assignments for subsequent analysis will come from Spec A at k=5. Spec B will be used to assess the sensitivity of the resulting household profiles to the inclusion and treatment of land size.

This decision establishes a working typology for the subsequent coping behaviour analysis. It does not establish that the clusters are universal household categories or that they predict coping behaviour. Coping outcomes remain excluded from segmentation construction and selection.

| Item | Locked choice |
|---|---|
| Working k | 5 |
| Primary assignments | Spec A, `cluster_k5` in `12_s1_spec_a_assignments_wide.csv` |
| Spec B role | Sensitivity only |
| k=4 role | Sensitivity alternative |
| Coping in segmentation | Excluded |

## 12. Coping profiling (Script 17) — descriptive

Descriptive coping distributions across Spec A k=5 working clusters completed (unweighted counts and household-weighted proportions; four- and three-class targets). Classification models not fitted. k=4 coping sensitivity completed in Script 18.

Report: `outputs/reports/17_coping_profile_by_segment.md`

## 13. k=4 coping sensitivity (Script 18) — decision record

The k=4 coping sensitivity analysis was completed using the existing Spec A cluster assignments. No re-clustering was performed and the primary k=5 assignments were unchanged.

The comparison showed that the distinctive coping profile of the community-land livestock-oriented segment remained recognisable under both segmentation resolutions. The corresponding k=4 and k=5 clusters shared 294 households, representing 99.3% of the k=5 cluster. Both solutions showed comparatively higher personal coping, lower informal coping and higher proportions of households reporting no coping action than the overall sample.

The k=4 solution consolidated several crop-oriented groups that were distinguished under k=5. This reduced the visibility of some finer coping differences between agricultural household types. The five-cluster solution was therefore retained as the primary descriptive typology, with k=4 retained as a sensitivity alternative.

These findings support the descriptive robustness of the main livestock/community coping contrast across the two segmentation resolutions. They do not establish causal relationships or demonstrate that agricultural segments predict coping behaviour.

The descriptive segmentation and coping profiling stages are considered complete. The study may proceed to classification design using the four-class coping category as the primary target and the three-class collapsed category as a secondary target. Classification results will be evaluated separately from the descriptive segmentation analysis.

Report: `outputs/reports/18_coping_k4_sensitivity.md`

| Item | Locked / next step |
|---|---|
| Primary descriptive typology | Spec A k=5 (unchanged) |
| k=4 role | Sensitivity alternative (completed) |
| Descriptive stages | Complete |
| Classification primary target | `coping_category` (four-class) |
| Classification secondary target | `coping_category_collapsed` (three-class) |
| Classification vs segmentation | Evaluated separately; models not yet fitted |

## 14. Classification design lock (Script 19)

Classification evaluation protocol locked and shared CV folds materialised. No models fitted.

| Item | Locked choice |
|---|---|
| Primary track | Four-class `coping_category` + primary matrix (69 features) |
| Secondary track | Three-class `coping_category_collapsed` + same primary matrix / folds |
| Extended track | Four-class + extended matrix (92); retrospective interpretation only |
| CV | RepeatedStratifiedKFold, 5×5, seed 42; identical folds across tracks |
| Primary metric | Macro F1; class-wise P/R/F1 and confusion matrices required |
| Imbalance | Compare unweighted vs class-weighted; no SMOTE by default |
| Land size | Baseline A exclude acres; Baseline B train-fold median impute |
| Cluster membership | Not a default C1 predictor |
| Formal test fold min / max | 15 / 15 |
| Next | C1 baselines (dummy + multinomial logistic) after review |

Report: `outputs/reports/19_classification_design_and_folds.md`

### Script 19 acceptance and clarifications (pre-C1)

Script 19 is **accepted**. The modelling protocol is fixed before model performance is observed. Subsequent scripts must **read** `19_classification_protocol.json` and `19_cv_folds_long.csv` rather than regenerate folds.

**A. Survey weights vs class weights (distinct concepts)**

| Concept | Role |
|---|---|
| `household_weight` | Survey representation for descriptive / population-oriented estimates |
| `class_weight='balanced'` (or equivalent) | Learning-objective adjustment for unequal target-class frequencies |

These are not interchangeable. Classification C1 retains the planned comparison of unweighted vs class-weighted model fitting. Household weights are **not** passed into model fitting or fold-level evaluation metrics by default. Survey-weighted estimation, if pursued later, is a separately justified sensitivity.

**B. Summarising repeated stratified CV (25 test folds)**

The 25 test folds are not 25 independent samples: within each repetition the five folds cover the full dataset once, and repetitions reuse observations.

For each model, report:

- Mean and standard deviation of macro F1 across the 25 test folds
- Mean balanced accuracy and mean weighted F1 across the 25 folds
- Mean class-wise precision, recall and F1 across the 25 folds
- Confusion matrices per fold; within each repetition, **sum** the five fold-level matrices to obtain one complete out-of-fold matrix; then summarise those five repetition matrices (mean/range as appropriate)

Do **not** treat the 25 fold scores as independent observations for conventional significance testing. Formal-class metrics (15 test cases per fold) may vary substantially; report variability, not averages alone.

**C1 scope (Script 20)**

Four-class primary track only. Dummy + multinomial logistic; unweighted vs class-weighted; acres excluded vs train-fold median imputation. Three-class secondary and extended tracks deferred until after the four-class primary benchmark.

## 15. C1 baselines (Script 20) — accepted

Four-class primary C1 completed using Script 19 folds (not regenerated). Dummy (most_frequent) and multinomial logistic regression compared under unweighted vs class-weighted learning and acres exclude vs train-fold median imputation. Household weights not used in fitting or metrics. Three-class and extended tracks deferred.

Report: `outputs/reports/20_classification_c1_baselines.md`

### Land-size handling decision

The acres-excluded and train-fold median-imputed logistic specifications produced nearly identical macro F1 (unweighted 0.3394 vs 0.3394; class-weighted 0.3500 vs 0.3515). Differences are negligible.

| Choice | Decision |
|---|---|
| Acres excluded | **Default** classification specification going forward |
| Acres imputed (train-fold median) | Documented sensitivity only (C1); not the default |
| Missingness indicator | Not introduced without separate justification |

Acreage missingness is strongly associated with community land ownership. Median imputation could obscure differences between households with measured acreage and those recorded as community land.

### C1 decision record

| Component | Decision |
|---|---|
| Dummy baseline | Retain |
| Unweighted logistic regression | Retain (benchmark) |
| Class-weighted logistic regression | Retain; class weighting worth investigating further |
| Acres excluded | Default specification |
| Acres imputed | Sensitivity specification |
| Formal-class performance | Exploratory, with substantial limitations (15 test cases per fold) |
| C1 overall | **Complete / accepted** |

### Repeated-CV variability (methodological note)

Reported standard deviations across the 25 test folds describe variation across the repeated validation splits. They are **not** confidence intervals and must **not** be used for significance testing.

### Model selection principle (going forward)

Do not select a model solely for highest macro F1. Final comparison must jointly consider macro F1, balanced accuracy, class-specific precision/recall (especially formal), confusion matrices, fold-to-fold variation, and the trade-off between minority-class detection and false positives.

### Next: C2 / C3 (Script 21)

Four-class primary track; Script 19 folds; primary matrix; **acres excluded**; unweighted vs class-weighted where supported; no household weights; no SMOTE; same metrics as Script 20. Logistic regression remains the benchmark. No extensive hyperparameter tuning. Three-class and extended tracks remain deferred.

## 16. C2/C3 nonlinear models (Script 21)

Four-class primary decision tree, random forest, and hist gradient boosting completed on Script 19 folds with acres excluded (default). Unweighted vs class-weighted compared; household weights unused; no SMOTE; no hyperparameter search. Logistic acres-excluded C1 results remain the benchmark. Three-class and extended tracks deferred.

Report: `outputs/reports/21_classification_c2_c3_nonlinear.md`
