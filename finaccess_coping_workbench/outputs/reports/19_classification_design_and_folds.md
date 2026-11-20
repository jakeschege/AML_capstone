# Classification Design Lock and Shared CV Folds (Script 19)

Generated: 2026-11-19 07:20 UTC

## Scope

Design and fold infrastructure only. **No classification models fitted.**

## Locked tracks

| Track | Target | Matrix | Features | Role |
|---|---|---|---:|---|
| Primary | `coping_category` (4-class) | `09_X_primary_classification.parquet` | 69 | Main analysis |
| Secondary | `coping_category_collapsed` (3-class) | same primary matrix | 69 | Alternative target; same folds |
| Extended | `coping_category` (4-class) | `09_X_extended_classification.parquet` | 92 | Retrospective shock extension only |

## Evaluation protocol

- Scheme: `RepeatedStratifiedKFold` — 5 folds × 5 repeats
- Stratification: four-class `coping_category`
- Seed: 42
- Identical fold assignments for primary, secondary, and extended
- Primary metric: macro F1
- Also report: class-wise precision / recall / F1, confusion matrices, balanced accuracy, weighted F1; overall accuracy secondary
- Imbalance: compare unweighted vs class-weighted learning; no SMOTE by default
- Preprocessing: imputation and scaling fit within training folds only

## Land-size handling (for upcoming fits)

- Baseline A: exclude `land_size_acres`
- Baseline B: retain with train-fold median imputation
- Missingness indicator only if separately justified
- Spec A `cluster_k5` is not a default C1 predictor

## Sample class counts (four-class)

```
coping_category
personal       615
informal       447
formal          75
did_nothing    415
```

## Predictor audit

Checks: 11 | FAIL: 0 | WARN: 0

```
                  check_id   matrix status                                                                      detail
        primary_n_features  primary   PASS                                       primary predictors n=69 (expected 69)
       extended_n_features extended   PASS                                      extended predictors n=92 (expected 92)
        primary_list_match  primary   PASS   prepared primary columns match 09_feature_list_primary_classification.csv
       extended_list_match extended   PASS prepared extended columns match 09_feature_list_extended_classification.csv
primary_subset_of_extended     both   PASS                            all primary predictors appear in extended matrix
       extended_only_count extended   PASS                                    extended-only columns n=23 (expected 23)
      primary_no_forbidden  primary   PASS   no coping/target/financial/sample-flag/weight/shock predictors in primary
       extended_no_leakage extended   PASS        no coping/target/financial/sample-flag/weight predictors in extended
   land_size_acres_present  primary   PASS                    land_size_acres in primary; missing n=198 (expected 198)
excluded_shannon_diversity     both   PASS                       shannon_diversity absent from classification matrices
     excluded_drought_flag     both   PASS                            drought_flag absent from classification matrices
```

### Extended-only columns (n=23)

These are retrospective shock indicators / `main_shock_type` dummies and must not be described as prospective predictors:

```
- main_shock_type_10_loss_or_damage_due_to_fire_or_violence_including_political_violence
- main_shock_type_11_loss_or_damage_due_to_human_wildlife_conflict
- main_shock_type_12_loss_of_income_or_major_financial_expenses_due_to_taking_on_care_for_someone_els
- main_shock_type_1_child_birth
- main_shock_type_2_major_sickness_health_problem_accident_injury
- main_shock_type_3_loss_or_damage_due_to_drought
- main_shock_type_4_high_rainfall_flooding_mudslide_lightning
- main_shock_type_5_death_of_main_income_earner_family_member
- main_shock_type_6_loss_or_damage_due_to_theft_e_g_property_assets_business_etc
- main_shock_type_7_loss_of_a_job_or_a_source_of_income_or_a_wage_cut
- main_shock_type_8_lost_a_home_or_land_e_g_due_to_eviction_demolitions_or_land_seizure
- main_shock_type_9_livestock_crop_failure_due_to_pest_and_diseases
- shock_child_birth
- shock_death_of_main_income_earner_family_member
- shock_high_rainfall_flooding_mudslide_lightning
- shock_livestock_crop_failure_due_to_pest_and_diseases
- shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut
- shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care
- shock_loss_or_damage_due_to_human_wildlife_conflict
- shock_loss_or_damage_due_tofire_or_violence_including_political_violen
- shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc
- shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions
- shock_major_sickness_health_problem_accident_injury
```

## Formal-class fold diagnostics (test splits)

| Statistic | Value |
|---|---:|
| Test folds | 25 |
| Formal test min | 15 |
| Formal test max | 15 |
| Formal test mean | 15.0 |
| Test folds with formal = 0 | 0 |
| Test folds with formal &lt; 5 | 0 |
| Formal train min | 60 |
| Formal train max | 60 |

Per test fold formal counts:

```
 repeat  fold  n_households  split_n  formal_sparse_under_5
      0     0            15      311                  False
      0     1            15      311                  False
      0     2            15      310                  False
      0     3            15      310                  False
      0     4            15      310                  False
      1     0            15      311                  False
      1     1            15      311                  False
      1     2            15      310                  False
      1     3            15      310                  False
      1     4            15      310                  False
      2     0            15      311                  False
      2     1            15      311                  False
      2     2            15      310                  False
      2     3            15      310                  False
      2     4            15      310                  False
      3     0            15      311                  False
      3     1            15      311                  False
      3     2            15      310                  False
      3     3            15      310                  False
      3     4            15      310                  False
      4     0            15      311                  False
      4     1            15      311                  False
      4     2            15      310                  False
      4     3            15      310                  False
      4     4            15      310                  False
```

Class-specific metrics and confusion matrices will be essential in later scripts because formal remains sparse even when every test fold contains at least some formal cases.

## Deferred

- C1: Dummy + multinomial logistic (unweighted vs class-weighted; acres exclude vs fold-impute)
- C2–C3: Decision tree, random forest, gradient boosting
- Full primary / secondary / extended scorecards under these shared folds

## Output files

- `outputs/tables/19_classification_protocol.json`
- `outputs/tables/19_cv_folds_long.csv`
- `outputs/tables/19_fold_class_counts.csv`
- `outputs/tables/19_predictor_audit.csv`
