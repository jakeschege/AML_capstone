# Spec A / Spec B Correspondence Audit (Script 16)

Generated: 2026-11-19 06:34 UTC

## Purpose

Verify household-level correspondence between Spec A and Spec B for k=4 and k=5 using maximum-overlap cluster matching. No PAM rerun; no coping outcomes; no final k locked in this script.

## Cluster matches (greedy max overlap)

```
 k  spec_a_cluster                    spec_a_note  matched_spec_b_cluster  overlap_n  spec_a_n  spec_b_n  retention_pct_of_spec_a  precision_pct_of_spec_b
 4               0          family_crops_no_roots                       0        640       698       701                    91.69                    91.30
 4               1        family_crops_with_roots                       2        290       290       290                   100.00                   100.00
 4               2            livestock_community                       3        363       363       363                   100.00                   100.00
 4               3                    lease_crops                       1        140       201       198                    69.65                    70.71
 5               0           family_legumes_heavy                       0        427       427       427                   100.00                   100.00
 5               1                 own_name_crops                       1        270       270       273                   100.00                    98.90
 5               2            livestock_community                       2        251       296       254                    84.80                    98.82
 5               3    family_mixed_crop_livestock                       3        199       214       230                    92.99                    86.52
 5               4 family_cereals_without_legumes                       4        344       345       368                    99.71                    93.48
```

## Retention summary

```
 k  spec_a_cluster                    spec_a_note  matched_spec_b_cluster  overlap_n  spec_a_n  spec_b_n  retention_pct_of_spec_a  precision_pct_of_spec_b
 4               0          family_crops_no_roots                       0        640       698       701                    91.69                    91.30
 4               1        family_crops_with_roots                       2        290       290       290                   100.00                   100.00
 4               2            livestock_community                       3        363       363       363                   100.00                   100.00
 4               3                    lease_crops                       1        140       201       198                    69.65                    70.71
 5               0           family_legumes_heavy                       0        427       427       427                   100.00                   100.00
 5               1                 own_name_crops                       1        270       270       273                   100.00                    98.90
 5               2            livestock_community                       2        251       296       254                    84.80                    98.82
 5               3    family_mixed_crop_livestock                       3        199       214       230                    92.99                    86.52
 5               4 family_cereals_without_legumes                       4        344       345       368                    99.71                    93.48
```

## Contingency tables

```
 k spec_a_cluster   0   1   2   3  All     4
 4              0 640  58   0   0  698   NaN
 4              1   0   0 290   0  290   NaN
 4              2   0   0   0 363  363   NaN
 4              3  61 140   0   0  201   NaN
 4            All 701 198 290 363 1552   NaN
 5              0 427   0   0   0  427   0.0
 5              1   0 270   0   0  270   0.0
 5              2   0   3 251  31  296  11.0
 5              3   0   0   2 199  214  13.0
 5              4   0   0   1   0  345 344.0
 5            All 427 273 254 230 1552 368.0
```

## Profile recognisability heuristics

```
 k  spec_a_cluster                    spec_a_note                                              variable direction  threshold  spec_a_mean  spec_b_mean  spec_a_meets  spec_b_meets  both_meet
 4               0          family_crops_no_roots                                      has_roots_tubers       low       0.15       0.0000       0.0000          True          True       True
 4               1        family_crops_with_roots                                      has_roots_tubers      high       0.70       1.0000       1.0000          True          True       True
 4               2            livestock_community                                     is_community_land      high       0.30       0.4931       0.4931          True          True       True
 4               2            livestock_community                                   has_other_livestock      high       0.70       0.8788       0.8788          True          True       True
 4               3                    lease_crops                land_ownership_lease_rent_you_dont_own      high       0.30       0.4975       0.5000          True          True       True
 5               0           family_legumes_heavy                                           has_legumes      high       0.70       1.0000       1.0000          True          True       True
 5               0           family_legumes_heavy land_ownership_family_owned_e_g_inherited_family_land      high       0.60       0.8407       0.8407          True          True       True
 5               1                 own_name_crops                    land_ownership_own_under_your_name      high       0.50       0.8407       0.8352          True          True       True
 5               2            livestock_community                                     is_community_land      high       0.40       0.5946       0.7047          True          True       True
 5               2            livestock_community                                   has_other_livestock      high       0.70       0.9662       0.9764          True          True       True
 5               3    family_mixed_crop_livestock                                   has_other_livestock      high       0.70       1.0000       1.0000          True          True       True
 5               3    family_mixed_crop_livestock                                           has_cereals      high       0.70       0.9813       0.8609          True          True       True
 5               4 family_cereals_without_legumes                                           has_legumes       low       0.35       0.0000       0.0000          True          True       True
 5               4 family_cereals_without_legumes land_ownership_family_owned_e_g_inherited_family_land      high       0.60       0.7913       0.7880          True          True       True
```

## Key trait means for matched pairs

```
 k  spec_a_cluster                    spec_a_note  matched_spec_b_cluster                                              variable  spec_a_mean  spec_b_mean  abs_diff
 4               0          family_crops_no_roots                       0                                           has_cereals     0.873926     0.877318  0.003393
 4               0          family_crops_no_roots                       0                                           has_legumes     0.587393     0.606277  0.018884
 4               0          family_crops_no_roots                       0                                   has_other_livestock     0.243553     0.236805  0.006748
 4               0          family_crops_no_roots                       0                                      has_roots_tubers     0.000000     0.000000  0.000000
 4               0          family_crops_no_roots                       0                                     is_community_land     0.002865     0.004280  0.001414
 4               0          family_crops_no_roots                       0 land_ownership_family_owned_e_g_inherited_family_land     0.838109     0.835949  0.002160
 4               0          family_crops_no_roots                       0                land_ownership_lease_rent_you_dont_own     0.008596     0.009986  0.001390
 4               0          family_crops_no_roots                       0                    land_ownership_own_under_your_name     0.116046     0.122682  0.006636
 4               0          family_crops_no_roots                       0                                land_size_acres_spec_b     3.138309     3.409844  0.271534
 4               0          family_crops_no_roots                       0                                          n_crop_types     2.121777     2.261056  0.139279
 4               1        family_crops_with_roots                       2                                           has_cereals     0.944828     0.944828  0.000000
 4               1        family_crops_with_roots                       2                                           has_legumes     0.637931     0.637931  0.000000
 4               1        family_crops_with_roots                       2                                   has_other_livestock     0.144828     0.144828  0.000000
 4               1        family_crops_with_roots                       2                                      has_roots_tubers     1.000000     1.000000  0.000000
 4               1        family_crops_with_roots                       2                                     is_community_land     0.000000     0.000000  0.000000
 4               1        family_crops_with_roots                       2 land_ownership_family_owned_e_g_inherited_family_land     0.758621     0.758621  0.000000
 4               1        family_crops_with_roots                       2                land_ownership_lease_rent_you_dont_own     0.017241     0.017241  0.000000
 4               1        family_crops_with_roots                       2                    land_ownership_own_under_your_name     0.165517     0.165517  0.000000
 4               1        family_crops_with_roots                       2                                land_size_acres_spec_b     1.939053     1.939053  0.000000
 4               1        family_crops_with_roots                       2                                          n_crop_types     3.479310     3.479310  0.000000
 4               2            livestock_community                       3                                           has_cereals     0.201102     0.201102  0.000000
 4               2            livestock_community                       3                                           has_legumes     0.046832     0.046832  0.000000
 4               2            livestock_community                       3                                   has_other_livestock     0.878788     0.878788  0.000000
 4               2            livestock_community                       3                                      has_roots_tubers     0.002755     0.002755  0.000000
 4               2            livestock_community                       3                                     is_community_land     0.493113     0.493113  0.000000
 4               2            livestock_community                       3 land_ownership_family_owned_e_g_inherited_family_land     0.110193     0.110193  0.000000
 4               2            livestock_community                       3                land_ownership_lease_rent_you_dont_own     0.005510     0.005510  0.000000
 4               2            livestock_community                       3                    land_ownership_own_under_your_name     0.154270     0.154270  0.000000
 4               2            livestock_community                       3                                land_size_acres_spec_b     7.763786     7.763786  0.000000
 4               2            livestock_community                       3                                          n_crop_types     0.451791     0.451791  0.000000
 4               3                    lease_crops                       1                                           has_cereals     0.895522     0.883838  0.011684
 4               3                    lease_crops                       1                                           has_legumes     0.706468     0.641414  0.065054
 4               3                    lease_crops                       1                                   has_other_livestock     0.124378     0.146465  0.022087
 4               3                    lease_crops                       1                                      has_roots_tubers     0.124378     0.126263  0.001885
 4               3                    lease_crops                       1                                     is_community_land     0.004975     0.000000  0.004975
 4               3                    lease_crops                       1 land_ownership_family_owned_e_g_inherited_family_land     0.009950     0.005051  0.004900
 4               3                    lease_crops                       1                land_ownership_lease_rent_you_dont_own     0.497512     0.500000  0.002488
 4               3                    lease_crops                       1                    land_ownership_own_under_your_name     0.358209     0.338384  0.019825
 4               3                    lease_crops                       1                                land_size_acres_spec_b     2.251639     1.285456  0.966184
 4               3                    lease_crops                       1                                          n_crop_types     2.820896     2.338384  0.482512
 5               0           family_legumes_heavy                       0                                           has_cereals     0.943794     0.943794  0.000000
 5               0           family_legumes_heavy                       0                                           has_legumes     1.000000     1.000000  0.000000
 5               0           family_legumes_heavy                       0                                   has_other_livestock     0.000000     0.000000  0.000000
 5               0           family_legumes_heavy                       0                                      has_roots_tubers     0.346604     0.346604  0.000000
 5               0           family_legumes_heavy                       0                                     is_community_land     0.002342     0.002342  0.000000
 5               0           family_legumes_heavy                       0 land_ownership_family_owned_e_g_inherited_family_land     0.840749     0.840749  0.000000
 5               0           family_legumes_heavy                       0                land_ownership_lease_rent_you_dont_own     0.084309     0.084309  0.000000
 5               0           family_legumes_heavy                       0                    land_ownership_own_under_your_name     0.004684     0.004684  0.000000
 5               0           family_legumes_heavy                       0                                land_size_acres_spec_b     2.150884     2.150884  0.000000
 5               0           family_legumes_heavy                       0                                          n_crop_types     3.243560     3.243560  0.000000
 5               1                 own_name_crops                       1                                           has_cereals     0.885185     0.875458  0.009727
 5               1                 own_name_crops                       1                                           has_legumes     0.644444     0.644689  0.000244
 5               1                 own_name_crops                       1                                   has_other_livestock     0.207407     0.208791  0.001384
 5               1                 own_name_crops                       1                                      has_roots_tubers     0.177778     0.175824  0.001954
 5               1                 own_name_crops                       1                                     is_community_land     0.000000     0.000000  0.000000
 5               1                 own_name_crops                       1 land_ownership_family_owned_e_g_inherited_family_land     0.007407     0.007326  0.000081
 5               1                 own_name_crops                       1                land_ownership_lease_rent_you_dont_own     0.100000     0.102564  0.002564
 5               1                 own_name_crops                       1                    land_ownership_own_under_your_name     0.840741     0.835165  0.005576
 5               1                 own_name_crops                       1                                land_size_acres_spec_b     2.987438     2.973839  0.013598
 5               1                 own_name_crops                       1                                          n_crop_types     2.385185     2.384615  0.000570
 5               2            livestock_community                       2                                           has_cereals     0.118243     0.125984  0.007741
 5               2            livestock_community                       2                                           has_legumes     0.006757     0.007874  0.001117
 5               2            livestock_community                       2                                   has_other_livestock     0.966216     0.976378  0.010162
 5               2            livestock_community                       2                                      has_roots_tubers     0.003378     0.003937  0.000559
 5               2            livestock_community                       2                                     is_community_land     0.594595     0.704724  0.110130
 5               2            livestock_community                       2 land_ownership_family_owned_e_g_inherited_family_land     0.131757     0.019685  0.112072
 5               2            livestock_community                       2                land_ownership_lease_rent_you_dont_own     0.013514     0.003937  0.009577
 5               2            livestock_community                       2                    land_ownership_own_under_your_name     0.057432     0.051181  0.006251
 5               2            livestock_community                       2                                land_size_acres_spec_b    10.671970    12.419764  1.747793
 5               2            livestock_community                       2                                          n_crop_types     0.219595     0.204724  0.014870
 5               3    family_mixed_crop_livestock                       3                                           has_cereals     0.981308     0.860870  0.120439
 5               3    family_mixed_crop_livestock                       3                                           has_legumes     0.705607     0.647826  0.057781
 5               3    family_mixed_crop_livestock                       3                                   has_other_livestock     1.000000     1.000000  0.000000
 5               3    family_mixed_crop_livestock                       3                                      has_roots_tubers     0.163551     0.130435  0.033117
 5               3    family_mixed_crop_livestock                       3                                     is_community_land     0.018692     0.008696  0.009996
 5               3    family_mixed_crop_livestock                       3 land_ownership_family_owned_e_g_inherited_family_land     0.813084     0.830435  0.017351
 5               3    family_mixed_crop_livestock                       3                land_ownership_lease_rent_you_dont_own     0.037383     0.034783  0.002601
 5               3    family_mixed_crop_livestock                       3                    land_ownership_own_under_your_name     0.037383     0.047826  0.010443
 5               3    family_mixed_crop_livestock                       3                                land_size_acres_spec_b     5.686952     6.674085  0.987134
 5               3    family_mixed_crop_livestock                       3                                          n_crop_types     2.336449     2.065217  0.271231
 5               4 family_cereals_without_legumes                       4                                           has_cereals     0.724638     0.720109  0.004529
 5               4 family_cereals_without_legumes                       4                                           has_legumes     0.000000     0.000000  0.000000
 5               4 family_cereals_without_legumes                       4                                   has_other_livestock     0.000000     0.057065  0.057065
 5               4 family_cereals_without_legumes                       4                                      has_roots_tubers     0.243478     0.241848  0.001630
 5               4 family_cereals_without_legumes                       4                                     is_community_land     0.002899     0.000000  0.002899
 5               4 family_cereals_without_legumes                       4 land_ownership_family_owned_e_g_inherited_family_land     0.791304     0.788043  0.003261
 5               4 family_cereals_without_legumes                       4                land_ownership_lease_rent_you_dont_own     0.110145     0.108696  0.001449
 5               4 family_cereals_without_legumes                       4                    land_ownership_own_under_your_name     0.008696     0.008152  0.000543
 5               4 family_cereals_without_legumes                       4                                land_size_acres_spec_b     1.243232     1.206914  0.036318
 5               4 family_cereals_without_legumes                       4                                          n_crop_types     1.817391     1.788043  0.029348
```

## Review guidance

- High retention for all five k=5 matched pairs supports treating Spec B as preserving the Spec A typology.
- Low retention concentrated in one Spec A group would mean the ARI is driven by other clusters.
- Recognisability failures on Spec B means would argue against adopting the provisional labels.

Working typology remains **k=5 provisionally preferred**, with k=4 as alternative, until this audit is accepted and labels are confirmed.

## Output files

- `outputs/tables/16_ab_*.csv`
- `outputs/tables/16_ab_correspondence_parameters.json`
