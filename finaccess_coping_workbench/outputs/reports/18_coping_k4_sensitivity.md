# k=4 Coping Sensitivity vs Spec A k=5 (Script 18)

Generated: 2026-11-19 07:09 UTC

## Scope

Descriptive sensitivity only: whether four-class coping patterns depend heavily on choosing five Spec A clusters rather than four.

- Uses existing Spec A `cluster_k4` and `cluster_k5` from Script 12
- Does **not** re-run segmentation or change locked k=5 assignments
- Weights: `household_weight`; unweighted counts retained
- No classification models; no Spec B

## Membership context (k=5 × k=4)

```
cluster_k4    0    1    2    3   All
cluster_k5                          
0           241  132    0   54   427
1            92   45   39   94   270
2             0    0  294    2   296
3           153   34   17   10   214
4           212   79   13   41   345
All         698  290  363  201  1552
```

Focal correspondence: k=5 cluster 2 (Community-land livestock-oriented households) ↔ k=4 cluster 2 (Community-land livestock-oriented households).

## Four-class unweighted % within cluster — k=5 (working)

```
coping_category                                           personal  informal  formal  did_nothing
cluster cluster_label                                                                            
-1      All households                                        39.6      28.8     4.8         26.7
 0      Family-owned legume-oriented crop households          33.7      36.1     4.4         25.8
 1      Own-name crop households                              40.4      32.2     5.6         21.8
 2      Community-land livestock-oriented households          47.3      15.2     4.7         32.8
 3      Family-owned mixed crop and livestock households      43.9      26.2     4.2         25.7
 4      Family-owned cereal-oriented households               37.1      30.4     5.2         27.2
```

## Four-class unweighted % within cluster — k=4 (sensitivity)

```
coping_category                                           personal  informal  formal  did_nothing
cluster cluster_label                                                                            
-1      All households                                        39.6      28.8     4.8         26.7
 0      Family-owned crop households (no roots/tubers)        38.8      31.8     5.4         23.9
 1      Family-owned crop households (with roots/tubers)      33.1      38.3     2.4         26.2
 2      Community-land livestock-oriented households          47.4      16.2     5.0         31.4
 3      Lease / own-name crop households                      37.8      27.4     6.0         28.9
```

## Household-weighted % within cluster — k=5

```
coping_category                                           personal  informal  formal  did_nothing
cluster cluster_label                                                                            
-1      All households                                        37.8      29.6     4.7         27.8
 0      Family-owned legume-oriented crop households          33.4      34.3     4.5         27.8
 1      Own-name crop households                              38.6      33.8     3.8         23.9
 2      Community-land livestock-oriented households          47.0      15.0     5.3         32.6
 3      Family-owned mixed crop and livestock households      42.2      28.5     4.2         25.2
 4      Family-owned cereal-oriented households               34.0      31.1     5.7         29.2
```

## Household-weighted % within cluster — k=4

```
coping_category                                           personal  informal  formal  did_nothing
cluster cluster_label                                                                            
-1      All households                                        37.8      29.6     4.7         27.8
 0      Family-owned crop households (no roots/tubers)        37.7      31.9     5.5         25.0
 1      Family-owned crop households (with roots/tubers)      31.8      37.5     2.3         28.3
 2      Community-land livestock-oriented households          45.3      17.4     5.0         32.3
 3      Lease / own-name crop households                      36.6      27.1     5.6         30.8
```

## Focal livestock / community segment comparison

Unweighted and weighted percentage-point deltas (k=4 − k=5) for the matched community-livestock clusters:

```
coping_category  k5_cluster2_pct_unweighted  k4_cluster2_pct_unweighted  delta_unweighted_pp  k5_cluster2_pct_weighted  k4_cluster2_pct_weighted  delta_weighted_pp
       personal                       47.30                       47.38                 0.08                     47.02                     45.33              -1.69
       informal                       15.20                       16.25                 1.05                     15.04                     17.38               2.34
         formal                        4.73                        4.96                 0.23                      5.30                      5.01              -0.29
    did_nothing                       32.77                       31.40                -1.37                     32.65                     32.28              -0.37
```

## Interpretability notes

- **Focal livestock/community segment:** Spec A k=5 cluster 2 and k=4 cluster 2 share 294 households (99.3% of the k=5 cluster; 81.0% of the k=4 cluster).
- **Distinctive k=5 Cluster 2 coping pattern** (vs sample: personal 39.6%, informal 28.8%, did-nothing 26.7%): personal 47.3%, informal 15.2%, did-nothing 32.8% unweighted (informal weighted 15.0%).
- **Corresponding k=4 Cluster 2:** personal 47.4%, informal 16.2%, did-nothing 31.4% unweighted (informal weighted 17.4%). The elevated personal / depressed informal / elevated did-nothing signature remains recognisable under k=4.
- **Family-crop consolidation:** under k=4, most of k=5 clusters 0, 3, and 4 fold into the two family-crop groups (clusters 0–1). Coping mixes in those broader groups are closer to the sample average than the more differentiated k=5 crop subtypes.
- **Own-name crop (k=5 cluster 1):** largest overlap is with k=4 cluster 3 (Lease / own-name crop households; n=94, 34.8% of the k=5 cluster). Lease/own-name recovery under k=4 remains partial, consistent with the earlier typology decision.
- **Overall:** coping composition differences are not an artefact of k=5 alone—the livestock/community contrast is visible at k=4—but k=5 preserves finer crop-ownership distinctions that k=4 collapses.

## Decision implication

This sensitivity does not unlock classification by itself. Review whether the livestock/community coping contrast and the broader k=5 patterns remain sufficiently robust; then decide whether to proceed to classification modelling.

## Output files

- `outputs/tables/18_coping_counts_unweighted_k4_k5.csv`
- `outputs/tables/18_coping_props_weighted_k4_k5.csv`
- `outputs/tables/18_k4_k5_membership_overlap.csv`
- `outputs/tables/18_focal_livestock_coping_compare.csv`
- `outputs/tables/18_focal_livestock_coping_delta.csv`
