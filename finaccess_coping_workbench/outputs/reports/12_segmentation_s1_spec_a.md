# S1 Segmentation Report — Specification A (Script 12)

Generated: 2026-11-18 20:56 UTC

## Scope

Gower distance + PAM for Specification A only.

- Excludes `land_size_acres` (17 features)
- Equal feature weights
- Retains agricultural indicators and portfolio counts
- Candidate k = 2..6
- Random seed: 42
- Coping outcomes **not** used
- **No final k selected**

## Runtime

- Gower distance: 1.317 s
- Distance shape: [1552, 1552]

## Silhouette by k

```
specification  k  average_silhouette  pam_runtime_seconds  n_features  random_seed
            A  2            0.377860                0.148          17           42
            A  3            0.228632                0.145          17           42
            A  4            0.278390                0.046          17           42
            A  5            0.330039                0.063          17           42
            A  6            0.256057                0.039          17           42
```

## Cluster sizes

```
 k  cluster  n_households   pct
 2        0          1164 75.00
 2        1           388 25.00
 3        0           513 33.05
 3        1           362 23.32
 3        2           677 43.62
 4        0           698 44.97
 4        1           290 18.69
 4        2           363 23.39
 4        3           201 12.95
 5        0           427 27.51
 5        1           270 17.40
 5        2           296 19.07
 5        3           214 13.79
 5        4           345 22.23
 6        0           292 18.81
 6        1           313 20.17
 6        2           247 15.91
 6        3           103  6.64
 6        4           131  8.44
 6        5           466 30.03
```

## Interpretation guide

Average silhouette is a primary summary metric for comparing k, but cluster sizes and agricultural/ownership profile coherence must also be reviewed. The highest silhouette alone does not establish the preferred solution.

Review these results before S2 stability assessment. Spec B is not run in this script.

## Output files

- `outputs/tables/12_s1_spec_a_parameters.json`
- `outputs/tables/12_s1_spec_a_silhouette_by_k.csv`
- `outputs/tables/12_s1_spec_a_cluster_sizes.csv`
- `outputs/tables/12_s1_spec_a_assignments_long.csv`
- `outputs/tables/12_s1_spec_a_assignments_wide.csv`
- `outputs/tables/12_s1_spec_a_cluster_profiles.csv`
- `outputs/tables/12_s1_spec_a_medoids.csv`
