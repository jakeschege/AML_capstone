# S3 Spec B Segmentation Report (Script 15)

Generated: 2026-11-19 06:04 UTC

## Scope

Land-size sensitivity for Spec A candidates k=2, 4, 5.

- Spec A 17 features retained
- `land_size_acres_spec_b`: observed acres retained; community acres left missing; 16 non-community specials imputed with median of observed acres (1.0)
- `is_community_land` and `acres_special_imputed` added
- Missing-aware Gower (pairwise skip of missing features); **no zero-fill**
- Spec A assignments unchanged; coping not used; **no final k selected**

## Acreage treatment counts

```
{
  "n_observed_acres": 1354,
  "median_observed_acres": 1.0,
  "n_community_land": 182,
  "n_special_imputed": 16,
  "n_acres_nan_in_spec_b": 182,
  "zero_fill": false
}
```

## Silhouette by k

```
specification  k  average_silhouette  pam_runtime_seconds  n_features  random_seed
            B  2            0.409276                0.217          20           42
            B  4            0.258209                0.069          20           42
            B  5            0.318096                0.075          20           42
```

## Cluster sizes

```
 k  cluster  n_households   pct
 2        0          1164 75.00
 2        1           388 25.00
 4        0           701 45.17
 4        1           198 12.76
 4        2           290 18.69
 4        3           363 23.39
 5        0           427 27.51
 5        1           273 17.59
 5        2           254 16.37
 5        3           230 14.82
 5        4           368 23.71
```

## Agreement with Spec A (same k)

```
 k  ari_spec_a_vs_spec_b
 2              1.000000
 4              0.819355
 5              0.921932
```

## Community land and acreage by Spec B cluster

```
 k  spec_b_cluster    n   pct  pct_is_community_land  mean_land_size_acres_spec_b  n_acres_missing_in_cluster  ari_vs_spec_a_same_k
 2               0 1164 75.00                   0.26                       2.7323                           3              1.000000
 2               1  388 25.00                  46.13                       6.9531                         179              1.000000
 4               0  701 45.17                   0.43                       3.4098                           3              0.819355
 4               1  198 12.76                   0.00                       1.2855                           0              0.819355
 4               2  290 18.69                   0.00                       1.9391                           0              0.819355
 4               3  363 23.39                  49.31                       7.7638                         179              0.819355
 5               0  427 27.51                   0.23                       2.1509                           1              0.921932
 5               1  273 17.59                   0.00                       2.9738                           0              0.921932
 5               2  254 16.37                  70.47                      12.4198                         179              0.921932
 5               3  230 14.82                   0.87                       6.6741                           2              0.921932
 5               4  368 23.71                   0.00                       1.2069                           0              0.921932
```

## Comparability

Community-land pairs omit acres from the pairwise Gower average when acres are missing. The community distinction is carried by `is_community_land` together with ownership and agricultural features. Spec B is a sensitivity analysis relative to Spec A, not a replacement until reviewed.

## Output files

- `outputs/tables/15_s3_spec_b_*.csv`
- `outputs/tables/15_s3_spec_b_parameters.json`
