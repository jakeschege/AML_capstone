# Coping Profiles by Spec A k=5 Segment (Script 17)

Generated: 2026-11-19 06:54 UTC

## Scope

Descriptive coping distributions across the locked Spec A five-cluster working typology.

- Assignments: Spec A `cluster_k5`
- Targets: four-class `coping_category` and three-class `coping_category_collapsed`
- Weights: `household_weight` for population-oriented proportions; unweighted counts retained
- No classification models, no re-clustering, no Spec B, no k=4 comparison

## Unweighted counts (four-class)

```
coping_category                                              personal  informal  formal  did_nothing
cluster_k5 cluster_label                                                                            
-1         All households                                         615       447      75          415
 0         Family-owned legume-oriented crop households           144       154      19          110
 1         Own-name crop households                               109        87      15           59
 2         Community-land livestock-oriented households           140        45      14           97
 3         Family-owned mixed crop and livestock households        94        56       9           55
 4         Family-owned cereal-oriented households                128       105      18           94
```

## Unweighted within-cluster percentages (four-class)

```
coping_category                                              personal  informal  formal  did_nothing
cluster_k5 cluster_label                                                                            
-1         All households                                        39.6      28.8     4.8         26.7
 0         Family-owned legume-oriented crop households          33.7      36.1     4.4         25.8
 1         Own-name crop households                              40.4      32.2     5.6         21.8
 2         Community-land livestock-oriented households          47.3      15.2     4.7         32.8
 3         Family-owned mixed crop and livestock households      43.9      26.2     4.2         25.7
 4         Family-owned cereal-oriented households               37.1      30.4     5.2         27.2
```

## Household-weighted within-cluster percentages (four-class)

```
coping_category                                              personal  informal  formal  did_nothing
cluster_k5 cluster_label                                                                            
-1         All households                                        37.8      29.6     4.7         27.8
 0         Family-owned legume-oriented crop households          33.4      34.3     4.5         27.8
 1         Own-name crop households                              38.6      33.8     3.8         23.9
 2         Community-land livestock-oriented households          47.0      15.0     5.3         32.6
 3         Family-owned mixed crop and livestock households      42.2      28.5     4.2         25.2
 4         Family-owned cereal-oriented households               34.0      31.1     5.7         29.2
```

## Formal coping by cluster

```
 cluster_k5                                    cluster_label  n_formal  cluster_n  pct_formal_unweighted  sparse_under_n30
          0     Family-owned legume-oriented crop households        19        427                   4.45              True
          1                         Own-name crop households        15        270                   5.56              True
          2     Community-land livestock-oriented households        14        296                   4.73              True
          3 Family-owned mixed crop and livestock households         9        214                   4.21              True
          4          Family-owned cereal-oriented households        18        345                   5.22              True
         -1                                   All households        75       1552                   4.83             False
```

All five clusters have formal n &lt; 30. Formal-class comparisons across segments are descriptive only and should not be over-interpreted.

## Three-class collapsed target

Unweighted percentages:

```
coping_category_collapsed                                    personal  formal_or_informal  did_nothing
cluster_k5 cluster_label                                                                              
-1         All households                                        39.6                33.6         26.7
 0         Family-owned legume-oriented crop households          33.7                40.5         25.8
 1         Own-name crop households                              40.4                37.8         21.8
 2         Community-land livestock-oriented households          47.3                19.9         32.8
 3         Family-owned mixed crop and livestock households      43.9                30.4         25.7
 4         Family-owned cereal-oriented households               37.1                35.6         27.2
```

Household-weighted percentages:

```
coping_category_collapsed                                    personal  formal_or_informal  did_nothing
cluster_k5 cluster_label                                                                              
-1         All households                                        37.8                34.4         27.8
 0         Family-owned legume-oriented crop households          33.4                38.8         27.8
 1         Own-name crop households                              38.6                37.5         23.9
 2         Community-land livestock-oriented households          47.0                20.3         32.6
 3         Family-owned mixed crop and livestock households      42.2                32.6         25.2
 4         Family-owned cereal-oriented households               34.0                36.8         29.2
```

## Association summary (secondary)

```
                   test          target   grouping    chi2  df  p_value  cramers_v  n_cells_expected_lt_5  formal_observed_min  formal_observed_max  formal_expected_min                                                                                                                                         warning
chi_square_independence coping_category cluster_k5 45.9192  12 0.000007     0.0993                      0                    9                   19                10.34 Formal observed counts are sparse (roughly 9–19 per cluster); p-value and Cramer’s V are secondary and do not establish causal segment effects.
```

## Descriptive profile notes

- **Cluster 0 — Family-owned legume-oriented crop households:** personal lower than sample (33.7% unweighted / 33.4% weighted vs 39.6% overall); informal higher than sample (36.1% unweighted / 34.3% weighted vs 28.8% overall).
- **Cluster 1 — Own-name crop households:** within-cluster coping mix close to the overall sample (±5 pp).
- **Cluster 2 — Community-land livestock-oriented households:** personal higher than sample (47.3% unweighted / 47.0% weighted vs 39.6% overall); informal lower than sample (15.2% unweighted / 15.0% weighted vs 28.8% overall); did_nothing higher than sample (32.8% unweighted / 32.6% weighted vs 26.7% overall).
- **Cluster 3 — Family-owned mixed crop and livestock households:** within-cluster coping mix close to the overall sample (±5 pp).
- **Cluster 4 — Family-owned cereal-oriented households:** within-cluster coping mix close to the overall sample (±5 pp).

These notes compare each cluster to the overall sample using a ±5 percentage-point threshold on unweighted shares. They do not imply that segments predict coping or that differences are causal.

## Deferred / follow-on

- k=4 coping sensitivity — completed in Script 18 (`18_coping_k4_sensitivity.md`)
- Classification model fitting
- Spec B coping cross-tabulation

## Output files

- `outputs/tables/17_coping_counts_unweighted_fourclass.csv`
- `outputs/tables/17_coping_props_weighted_fourclass.csv`
- `outputs/tables/17_coping_counts_unweighted_threeclass.csv`
- `outputs/tables/17_coping_props_weighted_threeclass.csv`
- `outputs/tables/17_formal_by_cluster.csv`
- `outputs/tables/17_association_summary.csv`
