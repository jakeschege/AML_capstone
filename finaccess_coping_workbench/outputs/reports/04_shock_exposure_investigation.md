# Shock Exposure Investigation Report

Generated: 2026-11-18 07:21 UTC

## What was investigated

Shock variables for the drought-exposed analytical sample (`n = 1552`):

- all 12 binary `shock_*` indicators
- `main_shock_type` and available labels

The project feature grouping uses the 11 non-drought indicators plus `main_shock_type`, because `shock_drought_loss` defines the analytical sample.

No encoding, category collapsing, observation removal or dataset modification was performed.

## Observed findings

### Inventory and completeness

```
                                                              variable   dtype    n  missing  missing_pct  n_unique                       role  n_positive  pct_positive  is_constant
                                                     shock_child_birth float64 1552        0          0.0         2 additional_shock_indicator       107.0          6.89          0.0
                   shock_major_sickness_health_problem_accident_injury float64 1552        0          0.0         2 additional_shock_indicator       539.0         34.73          0.0
                                                    shock_drought_loss float64 1552        0          0.0         1   sample_defining_exposure      1552.0        100.00          1.0
                       shock_high_rainfall_flooding_mudslide_lightning float64 1552        0          0.0         2 additional_shock_indicator       552.0         35.57          0.0
                       shock_death_of_main_income_earner_family_member float64 1552        0          0.0         2 additional_shock_indicator       237.0         15.27          0.0
     shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc float64 1552        0          0.0         2 additional_shock_indicator       283.0         18.23          0.0
               shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut float64 1552        0          0.0         2 additional_shock_indicator       162.0         10.44          0.0
             shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions float64 1552        0          0.0         2 additional_shock_indicator        21.0          1.35          0.0
                 shock_livestock_crop_failure_due_to_pest_and_diseases float64 1552        0          0.0         2 additional_shock_indicator       710.0         45.75          0.0
shock_loss_or_damage_due_tofire_or_violence_including_political_violen float64 1552        0          0.0         2 additional_shock_indicator        26.0          1.68          0.0
                   shock_loss_or_damage_due_to_human_wildlife_conflict float64 1552        0          0.0         2 additional_shock_indicator       128.0          8.25          0.0
shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care float64 1552        0          0.0         2 additional_shock_indicator        54.0          3.48          0.0
                                                       main_shock_type float64 1552        0          0.0        12                 main_shock         NaN           NaN          NaN
                                                 main_shock_type_label     str 1552        0          0.0        12                 main_shock         NaN           NaN          NaN
```

All investigated shock fields are complete in this sample.

### Indicator frequencies

```
                                                              variable  n_positive  pct_positive  n_zero  pct_zero  n_other_or_missing  is_drought_indicator  is_additional_shock  rare_under_5pct
                                                    shock_drought_loss        1552        100.00       0      0.00                   0                  True                False            False
                 shock_livestock_crop_failure_due_to_pest_and_diseases         710         45.75     842     54.25                   0                 False                 True            False
                       shock_high_rainfall_flooding_mudslide_lightning         552         35.57    1000     64.43                   0                 False                 True            False
                   shock_major_sickness_health_problem_accident_injury         539         34.73    1013     65.27                   0                 False                 True            False
     shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc         283         18.23    1269     81.77                   0                 False                 True            False
                       shock_death_of_main_income_earner_family_member         237         15.27    1315     84.73                   0                 False                 True            False
               shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut         162         10.44    1390     89.56                   0                 False                 True            False
                   shock_loss_or_damage_due_to_human_wildlife_conflict         128          8.25    1424     91.75                   0                 False                 True            False
                                                     shock_child_birth         107          6.89    1445     93.11                   0                 False                 True            False
shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care          54          3.48    1498     96.52                   0                 False                 True             True
shock_loss_or_damage_due_tofire_or_violence_including_political_violen          26          1.68    1526     98.32                   0                 False                 True             True
             shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions          21          1.35    1531     98.65                   0                 False                 True             True
```

- `shock_drought_loss` positives: 1552 (100.0%)
- Additional indicators below 5.0% positive: 3

### Main shock type

```
 main_shock_type                                                                            label                                                       mapped_indicator  n_households   pct  rare_under_n30
               1                                                                      Child birth                                                      shock_child_birth            23  1.48            True
               2                                    Major sickness/health problem/accident injury                    shock_major_sickness_health_problem_accident_injury           238 15.34           False
               3                                                    Loss or damage due to drought                                                     shock_drought_loss           778 50.13           False
               4                                        High rainfall/flooding/mudslide/lightning                        shock_high_rainfall_flooding_mudslide_lightning           127  8.18           False
               5                                        Death of main income earner/family member                        shock_death_of_main_income_earner_family_member           100  6.44           False
               6                   Loss or damage due to theft(e.g property, assets, business etc      shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc            45  2.90           False
               7                                Loss of a job or a source of income or a wage cut                shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut            35  2.26           False
               8          Lost a home or land (e.g. due to eviction, demolitions or land seizure)              shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions             5  0.32            True
               9                                  Livestock/crop failure due to pest and diseases                  shock_livestock_crop_failure_due_to_pest_and_diseases           152  9.79           False
              10              Loss or damage due to fire or violence including political violence shock_loss_or_damage_due_tofire_or_violence_including_political_violen            11  0.71            True
              11                                    Loss or damage due to human wildlife conflict                    shock_loss_or_damage_due_to_human_wildlife_conflict            25  1.61            True
              12 Loss of income or major financial expenses due to taking on care for someone els shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care            13  0.84            True
```

### Multiple shock exposure

Summary:

```
               count_definition  mean  median  min  max pct_only_drought_among_additional pct_with_at_least_one_additional
              all_12_indicators 2.816     3.0    1    9                                                                   
additional_11_excluding_drought 1.816     2.0    0    8                             19.65                            80.35
```

Detailed counts:

```
               count_definition  n_shocks  n_households   pct
              all_12_indicators       1.0           305 19.65
              all_12_indicators       2.0           454 29.25
              all_12_indicators       3.0           394 25.39
              all_12_indicators       4.0           166 10.70
              all_12_indicators       5.0           137  8.83
              all_12_indicators       6.0            59  3.80
              all_12_indicators       7.0            26  1.68
              all_12_indicators       8.0             8  0.52
              all_12_indicators       9.0             3  0.19
additional_11_excluding_drought       0.0           305 19.65
additional_11_excluding_drought       1.0           454 29.25
additional_11_excluding_drought       2.0           394 25.39
additional_11_excluding_drought       3.0           166 10.70
additional_11_excluding_drought       4.0           137  8.83
additional_11_excluding_drought       5.0            59  3.80
additional_11_excluding_drought       6.0            26  1.68
additional_11_excluding_drought       7.0             8  0.52
additional_11_excluding_drought       8.0             3  0.19
```

- Households with at least one additional non-drought shock: 80.35%
- Households with drought only among the indicator set: 19.65%

### Consistency between main_shock_type and indicators

Household-level consistency where the mapped indicator equals 1: 100.0%  
Households inconsistent: 0

```
 main_shock_type                                                                            label                                                       mapped_indicator  n_households  n_indicator_positive  n_indicator_not_positive  pct_consistent
               1                                                                      Child birth                                                      shock_child_birth            23                    23                         0           100.0
               2                                    Major sickness/health problem/accident injury                    shock_major_sickness_health_problem_accident_injury           238                   238                         0           100.0
               3                                                    Loss or damage due to drought                                                     shock_drought_loss           778                   778                         0           100.0
               4                                        High rainfall/flooding/mudslide/lightning                        shock_high_rainfall_flooding_mudslide_lightning           127                   127                         0           100.0
               5                                        Death of main income earner/family member                        shock_death_of_main_income_earner_family_member           100                   100                         0           100.0
               6                   Loss or damage due to theft(e.g property, assets, business etc      shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc            45                    45                         0           100.0
               7                                Loss of a job or a source of income or a wage cut                shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut            35                    35                         0           100.0
               8          Lost a home or land (e.g. due to eviction, demolitions or land seizure)              shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions             5                     5                         0           100.0
               9                                  Livestock/crop failure due to pest and diseases                  shock_livestock_crop_failure_due_to_pest_and_diseases           152                   152                         0           100.0
              10              Loss or damage due to fire or violence including political violence shock_loss_or_damage_due_tofire_or_violence_including_political_violen            11                    11                         0           100.0
              11                                    Loss or damage due to human wildlife conflict                    shock_loss_or_damage_due_to_human_wildlife_conflict            25                    25                         0           100.0
              12 Loss of income or major financial expenses due to taking on care for someone els shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care            13                    13                         0           100.0
```

### Overlaps among additional shocks

Top co-occurring additional shock pairs:

```
                                                          shock_a                                                           shock_b  n_both_positive  pct_of_sample  pct_of_shock_a_also_b  pct_of_shock_b_also_a
              shock_major_sickness_health_problem_accident_injury                   shock_high_rainfall_flooding_mudslide_lightning              276          17.78                  51.21                  50.00
              shock_major_sickness_health_problem_accident_injury             shock_livestock_crop_failure_due_to_pest_and_diseases              260          16.75                  48.24                  36.62
                  shock_high_rainfall_flooding_mudslide_lightning             shock_livestock_crop_failure_due_to_pest_and_diseases              259          16.69                  46.92                  36.48
shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc             shock_livestock_crop_failure_due_to_pest_and_diseases              185          11.92                  65.37                  26.06
                  shock_death_of_main_income_earner_family_member             shock_livestock_crop_failure_due_to_pest_and_diseases              144           9.28                  60.76                  20.28
              shock_major_sickness_health_problem_accident_injury                   shock_death_of_main_income_earner_family_member              142           9.15                  26.35                  59.92
              shock_major_sickness_health_problem_accident_injury shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc              127           8.18                  23.56                  44.88
                  shock_high_rainfall_flooding_mudslide_lightning                   shock_death_of_main_income_earner_family_member              126           8.12                  22.83                  53.16
```

Full pairwise positive overlaps with at least one shared household are written to CSV.

### Rare categories

```
                 item_type                                                                                 item  n  pct                 flag
additional_shock_indicator               shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care 54 3.48 positive rate < 5.0%
additional_shock_indicator               shock_loss_or_damage_due_tofire_or_violence_including_political_violen 26 1.68 positive rate < 5.0%
additional_shock_indicator                            shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions 21 1.35 positive rate < 5.0%
           main_shock_type                                                                       1: Child birth 23 1.48               n < 30
           main_shock_type           8: Lost a home or land (e.g. due to eviction, demolitions or land seizure)  5 0.32               n < 30
           main_shock_type              10: Loss or damage due to fire or violence including political violence 11 0.71               n < 30
           main_shock_type                                    11: Loss or damage due to human wildlife conflict 25 1.61               n < 30
           main_shock_type 12: Loss of income or major financial expenses due to taking on care for someone els 13 0.84               n < 30
```

## Interpretation

`shock_drought_loss` is constant in this sample and should not be used as an ordinary predictor. The remaining indicators capture concurrent non-drought shocks. Many households report more than one shock, so the indicators are overlapping rather than mutually exclusive.

`main_shock_type` identifies the household's reported principal shock among the same concept set. Where consistency is high, the main shock is supported by the corresponding indicator. Rare main-shock categories and rare indicators remain informative for description but may be fragile in supervised models with many classes.

## Limitations

- Consistency uses the fixed R2B-to-indicator map and does not invent alternative mappings.
- Pairwise overlap does not imply causal dependence.
- Rare-category thresholds (5.0% / n < 30) are descriptive audit rules only.
- No relationship with the coping target was used to select or discard shock variables.

## Recommended preparation decisions for review

1. Retain the 11 additional shock indicators as provisional candidate predictors for the classification stage where conceptually appropriate.
2. Exclude `shock_drought_loss` from ordinary predictors because it is constant in the drought-exposed sample. Preserve it for traceability.
3. Retain `main_shock_type` as a provisional candidate, recognising high cardinality and several rare categories.
4. Keep a derived additional-shock count as an audit measure for now rather than automatically adding it as a predictor.
5. Do not collapse rare shock categories or encode variables in this step.
6. Revisit rare shocks and possible redundancy between indicators and `main_shock_type` during later feature assessment.

## Output files

- `outputs/tables/04_shock_inventory.csv`
- `outputs/tables/04_shock_indicator_frequencies.csv`
- `outputs/tables/04_main_shock_frequencies.csv`
- `outputs/tables/04_shock_count_distribution.csv`
- `outputs/tables/04_shock_count_summary.csv`
- `outputs/tables/04_main_shock_consistency.csv`
- `outputs/tables/04_shock_overlap_summary.csv`
- `outputs/tables/04_rare_shock_flags.csv`
