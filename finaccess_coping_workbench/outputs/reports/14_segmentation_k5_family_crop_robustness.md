# k=5 Family Crop Robustness Investigation (Script 14)

Generated: 2026-11-19 05:49 UTC

## Purpose

Assess whether the Spec A k=5 subtypes **family_legumes_heavy** (cluster 0) and **family_cereals_without_legumes** (cluster 4) are meaningful and reproducible distinctions, or an unstable subdivision of broader family crop agriculture. Comparator: k=4 family crop groups.

No Spec B, coping profiling, or final k selection.

## Design

- Same Spec A features and Gower construction as Scripts 12–13
- 50 shared 80% subsamples, seed 42
- Focus clusters: k=5 {0, 4}; comparator k=4 family crop {0, 1}

## 1. Membership transitions

```
 s1_k5_cluster                     s1_k5_note  n_member_replicate_observations  pct_same_matched  pct_other_focus_matched  pct_elsewhere
             0           family_legumes_heavy                            17063             78.78                     4.89          16.33
             4 family_cereals_without_legumes                            13863             68.04                    12.02          19.94
```

## 2. Merge rates and union recovery

```
                                   metric     value
     pct_replicates_k5_focus_groups_merge 36.000000
    pct_replicates_k4_family_groups_merge 56.000000
 mean_k5_focus_union_jaccard_matched_pair  0.686944
mean_k4_family_union_jaccard_matched_pair  0.640811
                     mean_jaccard_legumes  0.509733
          mean_jaccard_cereals_no_legumes  0.460080
```

## 3. Profile persistence (matched resample clusters)

Top absolute deviations from S1 means:

```
 s1_k5_cluster                     s1_k5_note                                               variable  s1_mean  mean_resample_matched_mean  std_resample_matched_mean  abs_diff_from_s1
             0           family_legumes_heavy                                           n_crop_types 3.243560                    2.758217                   0.481604          0.485343
             4 family_cereals_without_legumes                                           n_crop_types 1.817391                    2.140014                   0.463979          0.322623
             4 family_cereals_without_legumes                                            has_legumes 0.000000                    0.292801                   0.272236          0.292801
             0           family_legumes_heavy                                            has_legumes 1.000000                    0.776749                   0.158093          0.223251
             4 family_cereals_without_legumes                                      n_livestock_types 0.110145                    0.264301                   0.150577          0.154156
             0           family_legumes_heavy                                      n_livestock_types 0.124122                    0.278201                   0.164817          0.154079
             0           family_legumes_heavy                                    has_other_livestock 0.000000                    0.105213                   0.090543          0.105213
             4 family_cereals_without_legumes                                    has_other_livestock 0.000000                    0.090657                   0.080665          0.090657
             0           family_legumes_heavy                                       has_roots_tubers 0.346604                    0.267200                   0.175010          0.079405
             0           family_legumes_heavy                     land_ownership_own_under_your_name 0.004684                    0.077554                   0.090750          0.072870
             4 family_cereals_without_legumes                     land_ownership_own_under_your_name 0.008696                    0.073047                   0.077192          0.064352
             0           family_legumes_heavy  land_ownership_family_owned_e_g_inherited_family_land 0.840749                    0.780846                   0.131233          0.059904
             4 family_cereals_without_legumes                                       has_roots_tubers 0.243478                    0.193338                   0.095533          0.050141
             4 family_cereals_without_legumes                                            has_cereals 0.724638                    0.772392                   0.150316          0.047754
             0           family_legumes_heavy                                         has_cash_crops 0.194379                    0.151662                   0.081764          0.042717
             4 family_cereals_without_legumes                 land_ownership_lease_rent_you_dont_own 0.110145                    0.079598                   0.045635          0.030547
             4 family_cereals_without_legumes                                       has_horticulture 0.272464                    0.250631                   0.179024          0.021833
             0           family_legumes_heavy                                            has_cereals 0.943794                    0.924054                   0.041473          0.019740
             4 family_cereals_without_legumes                                         has_cash_crops 0.281159                    0.298047                   0.254806          0.016887
             4 family_cereals_without_legumes land_ownership_community_public_owned_e_g_pastoralists 0.017391                    0.008613                   0.006817          0.008778
```

## 4. Medoid feature variation

```
 s1_k5_cluster                     s1_k5_note  pct_medoid_legumes_ge_0_5  pct_medoid_cereals_ge_0_5  pct_medoid_family_ge_0_5  pct_medoid_other_live_lt_0_5  mean_medoid_has_legumes  mean_medoid_has_cereals  mean_medoid_n_crop_types  mean_medoid_has_roots  pct_full_s1_pattern_match
             0           family_legumes_heavy                       98.0                      100.0                      96.0                         100.0                     0.98                     1.00                      2.48                   0.14                       94.0
             4 family_cereals_without_legumes                       34.0                       92.0                      98.0                         100.0                     0.34                     0.92                      2.10                   0.00                       58.0
```

## 5. Boundary households

```
                                    metric      value
                        n_focus_households 772.000000
       boundary_gap_10th_percentile_cutoff   0.047059
                      n_boundary_ambiguous 181.000000
              pct_focus_boundary_ambiguous  23.450000
pct_focus_nearer_medoid_matches_assignment 100.000000
   pct_ambiguous_that_are_legumes_assigned  62.430000
```

Ambiguous households are those among the two focus groups whose Gower distances to the two S1 medoids differ by at most the 10th percentile gap (0.0471).

## 6. Overlap with k=4 family crop groups

```
                    cluster_k5  family_crops_no_roots  family_crops_with_roots  k4_cluster_2  k4_cluster_3  All
family_cereals_without_legumes                    212                       79            13            41  345
          family_legumes_heavy                    241                      132             0            54  427
                           All                    453                      211            13            95  772
```

## Review guidance

- High `pct_other_focus_matched` or frequent focus-group merges weaken the case for keeping both k=5 subtypes separate.
- Higher k=4 family-union Jaccard than separate k=5 subtype Jaccard supports preferring a broader family-crop representation.
- Persistent profile differences (especially legumes) with moderate membership churn can still be substantively meaningful.

**No final k is selected here.** Hold Spec B and coping profiling until this evidence is reviewed.

## Output files

- `outputs/tables/14_k5_*.csv`
- `outputs/tables/14_k5_robustness_parameters.json`
