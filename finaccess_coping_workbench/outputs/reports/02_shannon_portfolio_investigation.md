# Shannon Portfolio Investigation Report

Generated: 2026-11-18 06:39 UTC

## What was investigated

Relationship between:

- `n_crop_types`
- `n_livestock_types`
- `shannon_diversity`

Sample: drought-exposed households from the prepared 2024 modelling frame (`n = 1552`).

Additional checks used the nine binary `has_*` agricultural group flags to construct `n_active_groups`, because Shannon was originally computed from those flags.

## Observed findings

### Sample and missingness

- Observations analysed: 1552
- Missing values in the three portfolio measures: none in this sample (see descriptive table)

### Descriptive summary

```
         variable    n  missing  n_unique     mean   median      std  min      max
     n_crop_types 1552        0         6 2.075387 2.000000 1.397746  0.0 5.000000
n_livestock_types 1552        0         6 0.774485 0.000000 1.057066  0.0 5.000000
shannon_diversity 1552        0         5 0.709718 0.693147 0.504427 -0.0 1.609438
```

### Shannon value frequencies

```
 shannon_diversity  n_households   pct
         -0.000000           430 27.71
          0.693147           512 32.99
          1.098612           373 24.03
          1.386294           200 12.89
          1.609438            37  2.38
```

### Distinct crop and livestock combinations at each Shannon value

```
 shannon_diversity  n_households  distinct_portfolio_combinations
         -0.000000           430                                9
          0.693147           512                               17
          1.098612           373                               13
          1.386294           200                                8
          1.609438            37                                4
```

- Minimum distinct combinations at a Shannon value: 4
- Maximum distinct combinations at a Shannon value: 17

### Combined portfolio crosstab

- Full crosstab row combinations written to CSV: 21
- Filtered crosstab rows with at least 10 households: 19

### Determinism checks

```
                       predictor  n_groups  n_groups_with_single_shannon  n_groups_with_multiple_shannon  fully_deterministic  max_unique_shannon_within_group
                    n_crop_types         6                             0                               6                False                                5
               n_livestock_types         6                             1                               5                False                                5
n_crop_types_x_n_livestock_types        21                             6                              15                False                                4
                 n_active_groups         6                             4                               2                False                                2
```

- Fully deterministic given `n_crop_types` alone: False
- Fully deterministic given `n_livestock_types` alone: False
- Fully deterministic given `(n_crop_types, n_livestock_types)`: False
- Fully deterministic given `n_active_groups`: False
- Crop-livestock combinations with more than one Shannon value: 15
- Active-group counts with more than one Shannon value: 2

### Comparison with ln(k)

Observed Shannon values that match `ln(k)` for some integer k in 0..9: [-0.0, 0.693147, 1.098612, 1.386294, 1.609438]

Observed Shannon values that do not match those `ln(k)` targets: none

```
 shannon_diversity  n_households  modal_n_active_groups n_active_group_values matches_ln_k_for_k  expected_ln_of_modal_k  abs_diff_from_ln_modal_k
         -0.000000           430                      1               0, 1, 2               0, 1                0.000000                       0.0
          0.693147           512                      2                  2, 3                  2                0.693147                       0.0
          1.098612           373                      3                     3                  3                1.098612                       0.0
          1.386294           200                      4                     4                  4                1.386294                       0.0
          1.609438            37                      5                     5                  5                1.609438                       0.0
```

### Correlations

```
                   n_crop_types  n_livestock_types  shannon_diversity  n_active_groups
n_crop_types              1.000             -0.441              0.722            0.686
n_livestock_types        -0.441              1.000              0.072            0.131
shannon_diversity         0.722              0.072              1.000            0.970
n_active_groups           0.686              0.131              0.970            1.000
```

## Interpretation

Facts above show how Shannon varies with the portfolio counts. Interpretation should stay separate from those facts.

Shannon was constructed from equal-weight presence of agricultural group flags. Values near `ln(k)` are consistent with k equally weighted active groups. That is not the same as a classical Shannon index based on species abundance shares within a single taxonomic inventory.

If multiple crop and livestock count combinations share the same Shannon value, Shannon collapses distinct portfolio structures into a coarser richness summary. If a single count pair maps to more than one Shannon value, Shannon is carrying information not present in that pair alone.

## Limitations

- This investigation uses the drought-exposed analytical sample only.
- Determinism is assessed empirically within this sample. It is not a proof about other populations.
- `n_active_groups` is derived for audit from the same `has_*` flags used to build Shannon. It is not proposed here as a replacement feature.
- No modelling or feature-selection performance comparison was run.

## Recommended preparation decisions for review

1. Retain `n_crop_types` and `n_livestock_types` as primary agricultural portfolio measures.
2. Shannon diversity is not a deterministic function of the crop count, livestock count, their combination, or the active group count alone. Retain it provisionally alongside the portfolio counts and revisit redundancy during later feature assessment. Do not assume it is interchangeable with either count measure.
3. Do not recalculate or transform Shannon diversity.
4. Do not drop Shannon solely because its observed values resemble `ln(k)`.
5. Keep this decision open until feature assessment for segmentation and classification is completed separately.

## Output files

- `outputs/tables/02_portfolio_descriptive_summary.csv`
- `outputs/tables/02_shannon_value_counts.csv`
- `outputs/tables/02_portfolio_shannon_crosstab_full.csv`
- `outputs/tables/02_portfolio_shannon_crosstab_min10.csv`
- `outputs/tables/02_shannon_distinct_combinations.csv`
- `outputs/tables/02_shannon_determinism_summary.csv`
- `outputs/tables/02_shannon_unique_by_crop.csv`
- `outputs/tables/02_shannon_unique_by_livestock.csv`
- `outputs/tables/02_shannon_unique_by_crop_livestock.csv`
- `outputs/tables/02_shannon_unique_by_active_groups.csv`
- `outputs/tables/02_shannon_ln_k_comparison.csv`
- `outputs/tables/02_portfolio_correlations.csv`
