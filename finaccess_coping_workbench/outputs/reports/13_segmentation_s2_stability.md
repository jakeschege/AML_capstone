# S2 Stability Report — Spec A candidates k=2, 4, 5 (Script 13)

Generated: 2026-11-19 05:35 UTC

## Scope

Stability only for Specification A candidates from S1 profile review.

- Candidates: k = 2, 4, 5
- Subsample without replacement at 80% (n = 1242)
- Replicates: 50
- Shared subsample indices across k
- Random seed: 42
- No Spec B, no coping outcomes, **no final k selected**

## Overall assignment agreement (ARI)

```
 k  n_replicates  mean_ari  median_ari  std_ari   min_ari  max_ari   q25_ari  q75_ari  iqr_ari
 2            50  0.339550    0.170400 0.447542 -0.088118 1.000000 -0.020060 1.000000 1.020060
 4            50  0.374695    0.338121 0.139046  0.208345 0.794726  0.278708 0.408744 0.130036
 5            50  0.480270    0.415979 0.180040  0.238007 1.000000  0.364696 0.626932 0.262237
```

## Cluster-level max Jaccard stability

```
 k  s1_cluster                s1_cluster_note  mean_max_jaccard  median_max_jaccard  min_max_jaccard  max_max_jaccard  n_replicates
 2           0                    crop_family          0.700147            0.617538         0.434914         1.000000            50
 2           1            livestock_community          0.555904            0.449992         0.239485         1.000000            50
 4           0          family_crops_no_roots          0.502677            0.469183         0.305195         0.847934            50
 4           1        family_crops_with_roots          0.363720            0.257121         0.190635         0.942149            50
 4           2            livestock_community          0.853652            0.854728         0.750000         0.964052            50
 4           3                    lease_crops          0.215266            0.187797         0.130208         0.408000            50
 5           0           family_legumes_heavy          0.509733            0.474958         0.261959         1.000000            50
 5           1                 own_name_crops          0.517100            0.554599         0.151149         1.000000            50
 5           2            livestock_community          0.878842            0.885598         0.640625         1.000000            50
 5           3    family_mixed_crop_livestock          0.489906            0.395238         0.153101         1.000000            50
 5           4 family_cereals_without_legumes          0.460080            0.372369         0.205165         1.000000            50
```

## Focus: k=5 legumes vs cereals-without-legumes

```
 k  s1_cluster                s1_cluster_note  mean_max_jaccard  median_max_jaccard  min_max_jaccard  max_max_jaccard  n_replicates
 5           0           family_legumes_heavy          0.509733            0.474958         0.261959              1.0            50
 5           4 family_cereals_without_legumes          0.460080            0.372369         0.205165              1.0            50
```

## Cluster size variation (matched resample clusters)

```
 k  s1_cluster                s1_cluster_note  s1_n  s1_pct  mean_matched_resample_n  std_matched_resample_n  min_matched_resample_n  max_matched_resample_n  mean_matched_resample_pct
 2           0                    crop_family  1164   75.00                   789.94                  203.94                     410                    1137                      63.60
 2           1            livestock_community   388   25.00                   649.50                  263.29                     294                    1137                      52.29
 4           0          family_crops_no_roots   698   44.97                   577.06                  107.25                     331                     886                      46.46
 4           1        family_crops_with_roots   290   18.69                   428.84                  208.04                      65                     886                      34.53
 4           2            livestock_community   363   23.39                   260.06                   19.03                     218                     304                      20.94
 4           3                    lease_crops   201   12.95                   391.12                  210.17                      52                     748                      31.49
 5           0           family_legumes_heavy   427   27.51                   480.50                  111.16                     219                     736                      38.69
 5           1                 own_name_crops   270   17.40                   284.00                  150.69                      99                     736                      22.87
 5           2            livestock_community   296   19.07                   239.04                   22.90                     182                     285                      19.25
 5           3    family_mixed_crop_livestock   214   13.79                   242.38                  157.43                      66                     640                      19.52
 5           4 family_cereals_without_legumes   345   22.23                   362.20                  176.04                      88                     736                      29.16
```

## Medoid pattern consistency

```
 k  s1_cluster                s1_cluster_note  pct_replicates_medoid_matches_pattern  n_replicates
 2           0                    crop_family                                   88.0            50
 2           1            livestock_community                                   30.0            50
 4           0          family_crops_no_roots                                  100.0            50
 4           1        family_crops_with_roots                                   40.0            50
 4           2            livestock_community                                  100.0            50
 4           3                    lease_crops                                   10.0            50
 5           0           family_legumes_heavy                                   94.0            50
 5           1                 own_name_crops                                   58.0            50
 5           2            livestock_community                                  100.0            50
 5           3    family_mixed_crop_livestock                                   64.0            50
 5           4 family_cereals_without_legumes                                   58.0            50
```

## Interpretation guide

Overall ARI summarises assignment agreement but must not alone decide k. Inspect cluster-level Jaccard, especially whether k=5 family_legumes_heavy and family_cereals_without_legumes are consistently recovered or frequently fragmented.

Review these outputs before S3/Spec B, coping profiling or final k selection.

## Output files

- `outputs/tables/13_s2_parameters.json`
- `outputs/tables/13_s2_overall_ari_by_k.csv`
- `outputs/tables/13_s2_ari_by_replicate.csv`
- `outputs/tables/13_s2_cluster_jaccard_stability.csv`
- `outputs/tables/13_s2_cluster_size_variation.csv`
- `outputs/tables/13_s2_profile_persistence.csv`
- `outputs/tables/13_s2_medoid_consistency.csv`
- `outputs/tables/13_s2_medoid_detail_by_replicate.csv`
