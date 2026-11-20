# C1 Classification Baselines — Four-class Primary (Script 20)

Generated: 2026-11-19 07:32 UTC

## Scope

- Reads locked Script 19 protocol and `19_cv_folds_long.csv` (**folds not regenerated**)
- Target: four-class `coping_category` only
- Predictors: primary matrix; three-class and extended tracks deferred
- `household_weight` not used in fitting or evaluation metrics
- Class weighting compared separately from survey weights

## Experiments

| experiment_id | model | class_weight | land_size |
|---|---|---|---|
| dummy_most_frequent | Dummy (most_frequent) | none | exclude |
| logistic_unweighted_exclude | Multinomial logistic | unweighted | exclude |
| logistic_class_weighted_exclude | Multinomial logistic | class_weighted | exclude |
| logistic_unweighted_impute_median | Multinomial logistic | unweighted | train-fold median impute |
| logistic_class_weighted_impute_median | Multinomial logistic | class_weighted | train-fold median impute |

Preprocessing (logistic): continuous columns (`age_years`, `education`, `n_crop_types`, `n_livestock_types`, and acres when retained) median-imputed if needed and standardised **within each training fold**; binary/one-hot columns passed through; `C=1.0`, solver=`lbfgs`, `max_iter=2000`.

## Scorecard (mean ± variability across 25 test folds)

Means and standard deviations are descriptive summaries of the 25 fold scores. The folds are not independent samples for significance testing.

```
                        experiment_id  n_features  macro_f1_mean  macro_f1_std  balanced_accuracy_mean  weighted_f1_mean  f1_formal_mean  f1_formal_std  f1_personal_mean  f1_informal_mean  f1_did_nothing_mean  accuracy_mean
                  dummy_most_frequent          68         0.1419        0.0002                  0.2500            0.2249          0.0000         0.0000            0.5676            0.0000               0.0000         0.3963
          logistic_unweighted_exclude          68         0.3394        0.0206                  0.3451            0.4347          0.0147         0.0398            0.5150            0.4154               0.4123         0.4478
      logistic_class_weighted_exclude          68         0.3500        0.0217                  0.3809            0.4098          0.1206         0.0487            0.4116            0.4266               0.4413         0.3880
    logistic_unweighted_impute_median          69         0.3394        0.0208                  0.3456            0.4361          0.0097         0.0329            0.5175            0.4165               0.4138         0.4494
logistic_class_weighted_impute_median          69         0.3515        0.0191                  0.3858            0.4101          0.1272         0.0469            0.4136            0.4262               0.4389         0.3888
```

## Fit diagnostics

- Warning messages recorded: 0 (Convergence-related: 0)
- Failed fits in fold metrics table: 0

## Confusion matrices

Within each repetition, fold-level matrices are summed to one complete out-of-fold matrix. Below: mean cell counts across the five repetitions.

### Dummy (`dummy_most_frequent`)

```
pred_label   personal  informal  formal  did_nothing
true_label                                          
personal        615.0       0.0     0.0          0.0
informal        447.0       0.0     0.0          0.0
formal           75.0       0.0     0.0          0.0
did_nothing     415.0       0.0     0.0          0.0
```

### Best logistic by mean macro F1 (`logistic_class_weighted_impute_median`)

```
pred_label   personal  informal  formal  did_nothing
true_label                                          
personal        195.8     146.6   127.0        145.6
informal         69.8     190.0    79.4        107.8
formal           16.8      17.6    25.2         15.4
did_nothing      45.6      90.4    86.6        192.4
```

## Interpretation notes

- Macro F1 is the primary comparison metric; balanced accuracy and class-wise F1 (especially formal) are required reading.
- Formal support is 15 per test fold; formal metrics are expected to be noisy — inspect `macro_f1_std` and `f1_formal_std`.
- Class-weighted learning is not assumed to improve results; compare against unweighted directly.
- Land-size exclude vs impute is a sensitivity within the same folds.

## Deferred

- Three-class secondary target
- Extended retrospective matrix
- Decision tree / random forest / gradient boosting (C2–C3)

## Output files

- `outputs/tables/20_c1_run_config.json`
- `outputs/tables/20_c1_fold_metrics.csv`
- `outputs/tables/20_c1_scorecard.csv`
- `outputs/tables/20_c1_fold_confusion.csv`
- `outputs/tables/20_c1_repetition_confusion.csv`
- `outputs/tables/20_c1_mean_repetition_confusion.csv`
- `outputs/tables/20_c1_fit_warnings.csv`
- `outputs/tables/20_c1_oof_predictions.csv`
