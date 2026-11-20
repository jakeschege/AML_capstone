# C2/C3 Nonlinear Models — Four-class Primary (Script 21)

Generated: 2026-11-19 08:05 UTC

## Scope

- Reads locked Script 19 protocol and folds (**not regenerated**)
- Target: four-class `coping_category`
- Predictors: primary matrix with **`land_size_acres` excluded** (default)
- Unweighted vs class-weighted learning
- No household survey weights in fitting or metrics; no SMOTE
- No extensive hyperparameter tuning
- Three-class secondary and extended retrospective tracks deferred

## Models

| Stage | Model | class_weight |
|---|---|---|
| C2 | Decision tree | native `class_weight` |
| C3 | Random forest (200 trees) | native `class_weight` |
| C3 | Hist gradient boosting (200 iterations) | native `class_weight` |

Gradient boosting uses `HistGradientBoostingClassifier`, which supports `class_weight` natively. Training-fold sample-weight balancing was therefore not required.

## Scorecard (mean ± fold sd across 25 test folds)

Fold standard deviations describe variation across repeated validation splits. They are **not** confidence intervals and must not be used for significance testing.

```
                                experiment_id stage    source  macro_f1_mean  macro_f1_std  balanced_accuracy_mean  weighted_f1_mean  f1_formal_mean  f1_formal_std  f1_personal_mean  f1_informal_mean  f1_did_nothing_mean  accuracy_mean  train_seconds_mean  train_seconds_total
             decision_tree_unweighted_exclude    C2 script_21         0.2960        0.0244                  0.2971            0.3595          0.0873         0.0666            0.4262            0.3475               0.3229         0.3604              0.0477               1.1933
         decision_tree_class_weighted_exclude    C2 script_21         0.2816        0.0195                  0.2829            0.3492          0.0621         0.0360            0.4244            0.3370               0.3028         0.3459              0.0384               0.9592
             random_forest_unweighted_exclude    C3 script_21         0.3108        0.0202                  0.3171            0.4025          0.0092         0.0314            0.4952            0.3998               0.3390         0.4168              1.9870              49.6745
         random_forest_class_weighted_exclude    C3 script_21         0.3306        0.0269                  0.3319            0.4061          0.0702         0.0656            0.4600            0.4179               0.3741         0.4080              2.6398              65.9962
    hist_gradient_boosting_unweighted_exclude    C3 script_21         0.2997        0.0251                  0.3026            0.3809          0.0338         0.0456            0.4696            0.3520               0.3432         0.3894             10.7163             267.9072
hist_gradient_boosting_class_weighted_exclude    C3 script_21         0.3030        0.0273                  0.3048            0.3794          0.0473         0.0489            0.4514            0.3634               0.3500         0.3840             10.8799             271.9984
```

## Comparison with C1 acres-excluded benchmarks

```
                                experiment_id stage    source  macro_f1_mean  macro_f1_std  balanced_accuracy_mean  weighted_f1_mean  f1_formal_mean  f1_formal_std  f1_personal_mean  f1_informal_mean  f1_did_nothing_mean  accuracy_mean
                          dummy_most_frequent    C1 script_20         0.1419        0.0002                  0.2500            0.2249          0.0000         0.0000            0.5676            0.0000               0.0000         0.3963
                  logistic_unweighted_exclude    C1 script_20         0.3394        0.0206                  0.3451            0.4347          0.0147         0.0398            0.5150            0.4154               0.4123         0.4478
              logistic_class_weighted_exclude    C1 script_20         0.3500        0.0217                  0.3809            0.4098          0.1206         0.0487            0.4116            0.4266               0.4413         0.3880
             decision_tree_unweighted_exclude    C2 script_21         0.2960        0.0244                  0.2971            0.3595          0.0873         0.0666            0.4262            0.3475               0.3229         0.3604
         decision_tree_class_weighted_exclude    C2 script_21         0.2816        0.0195                  0.2829            0.3492          0.0621         0.0360            0.4244            0.3370               0.3028         0.3459
             random_forest_unweighted_exclude    C3 script_21         0.3108        0.0202                  0.3171            0.4025          0.0092         0.0314            0.4952            0.3998               0.3390         0.4168
         random_forest_class_weighted_exclude    C3 script_21         0.3306        0.0269                  0.3319            0.4061          0.0702         0.0656            0.4600            0.4179               0.3741         0.4080
    hist_gradient_boosting_unweighted_exclude    C3 script_21         0.2997        0.0251                  0.3026            0.3809          0.0338         0.0456            0.4696            0.3520               0.3432         0.3894
hist_gradient_boosting_class_weighted_exclude    C3 script_21         0.3030        0.0273                  0.3048            0.3794          0.0473         0.0489            0.4514            0.3634               0.3500         0.3840
```

## Fit diagnostics

- Warning messages recorded: 0
- Failed fits: 0
- Per-experiment mean training seconds are in the scorecard (`train_seconds_mean`)

## Confusion matrices (mean cell counts across five repetition OOF matrices)

### Best C2/C3 by mean macro F1 (`random_forest_class_weighted_exclude`)

```
pred_label   personal  informal  formal  did_nothing
true_label                                          
personal        267.8     171.0    18.2        158.0
informal        126.4     199.0    11.8        109.8
formal           29.8      20.2     4.2         20.8
did_nothing     120.8     114.8    17.2        162.2
```

### Random forest class-weighted (`random_forest_class_weighted_exclude`)

```
pred_label   personal  informal  formal  did_nothing
true_label                                          
personal        267.8     171.0    18.2        158.0
informal        126.4     199.0    11.8        109.8
formal           29.8      20.2     4.2         20.8
did_nothing     120.8     114.8    17.2        162.2
```

### Hist gradient boosting class-weighted (`hist_gradient_boosting_class_weighted_exclude`)

```
pred_label   personal  informal  formal  did_nothing
true_label                                          
personal        279.8     163.0    15.8        156.4
informal        160.8     165.0    11.2        110.0
formal           35.0      20.6     2.8         16.6
did_nothing     145.8     110.8    10.0        148.4
```

## Interpretation guide

Do not select a model solely for highest macro F1. Jointly inspect:

1. Macro F1 vs logistic benchmark
2. Balanced accuracy
3. Class-specific precision/recall (especially formal)
4. Confusion matrices
5. Fold-to-fold variation (`*_std`)
6. Minority-class detection vs false-positive trade-off

Formal metrics remain exploratory (15 test cases per fold).

## Deferred

- Three-class secondary target
- Extended retrospective matrix
- Extensive hyperparameter tuning / stacking

## Output files

- `outputs/tables/21_c2c3_run_config.json`
- `outputs/tables/21_c2c3_fold_metrics.csv`
- `outputs/tables/21_c2c3_scorecard.csv`
- `outputs/tables/21_c2c3_vs_c1_comparison.csv`
- `outputs/tables/21_c2c3_fold_confusion.csv`
- `outputs/tables/21_c2c3_repetition_confusion.csv`
- `outputs/tables/21_c2c3_mean_repetition_confusion.csv`
- `outputs/tables/21_c2c3_fit_warnings.csv`
- `outputs/tables/21_c2c3_oof_predictions.csv`
