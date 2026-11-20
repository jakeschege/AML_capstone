# Segmentation Design Audit (Script 11)

Generated: 2026-11-18 20:45 UTC

## Purpose

Evidence for Spec A/B acreage design and feature weighting before S1. No S1–S4 fitting and no classification fitting were performed beyond a Gower/PAM smoke test on Spec A.

## 1. Feature inventory (18 segmentation predictors)

```
                                                      variable measurement_type          domain  n_missing  missing_pct  n_unique     mean
                                                has_cash_crops           binary      agri_group          0         0.00         2 0.178479
                                                   has_cereals           binary      agri_group          0         0.00         2 0.732603
                                           has_dairy_livestock           binary      agri_group          0         0.00         2 0.085696
                                              has_horticulture           binary      agri_group          0         0.00         2 0.144974
                                                   has_legumes           binary      agri_group          0         0.00         2 0.485825
                                                has_other_agri           binary      agri_group          0         0.00         2 0.003866
                                           has_other_livestock           binary      agri_group          0         0.00         2 0.358247
                                                   has_poultry           binary      agri_group          0         0.00         2 0.094716
                                              has_roots_tubers           binary      agri_group          0         0.00         2 0.203608
                                                  n_crop_types    numeric_count portfolio_count          0         0.00         6 2.075387
                                             n_livestock_types    numeric_count portfolio_count          0         0.00         6 0.774485
                                               land_size_acres       continuous       land_size        198        12.76        91 3.404323
                            land_ownership_own_under_your_name           binary       ownership          0         0.00         2 0.165593
                        land_ownership_own_jointly_with_spouse           binary       ownership          0         0.00         2 0.072165
land_ownership_own_jointly_with_someone_else_other_than_spouse           binary       ownership          0         0.00         2 0.005799
                        land_ownership_lease_rent_you_dont_own           binary       ownership          0         0.00         2 0.072809
        land_ownership_community_public_owned_e_g_pastoralists           binary       ownership          0         0.00         2 0.169459
         land_ownership_family_owned_e_g_inherited_family_land           binary       ownership          0         0.00         2 0.545747
```

## 2. Indicator–count overlap

Highest absolute correlations (including derived indicator sums vs counts):

```
                    variable_a          variable_b  pearson_corr  abs_corr              pair_type
     sum_crop_group_indicators        n_crop_types        0.9056    0.9056   derived_sum_vs_count
sum_livestock_group_indicators   n_livestock_types        0.8782    0.8782   derived_sum_vs_count
           has_other_livestock   n_livestock_types        0.8042    0.8042     indicator_vs_count
                   has_cereals        n_crop_types        0.6577    0.6577     indicator_vs_count
                   has_legumes        n_crop_types        0.6275    0.6275     indicator_vs_count
              has_roots_tubers        n_crop_types        0.4960    0.4960     indicator_vs_count
                   has_cereals         has_legumes        0.4882    0.4882 indicator_vs_indicator
           has_other_livestock        n_crop_types       -0.4481    0.4481     indicator_vs_count
                  n_crop_types   n_livestock_types       -0.4406    0.4406         count_vs_count
           has_dairy_livestock   n_livestock_types        0.3921    0.3921     indicator_vs_count
                   has_poultry   n_livestock_types        0.3647    0.3647     indicator_vs_count
                has_cash_crops        n_crop_types        0.3373    0.3373     indicator_vs_count
                   has_cereals   n_livestock_types       -0.3259    0.3259     indicator_vs_count
                   has_cereals has_other_livestock       -0.3228    0.3228 indicator_vs_indicator
           has_other_livestock    has_roots_tubers       -0.2309    0.2309 indicator_vs_indicator
```

Interpretation: crop and livestock counts partially overlap with agricultural group indicators. Equal-weight Gower may double-count portfolio breadth if both indicators and counts are retained without weighting or a drop-counts sensitivity.

## 3. Domain contribution

```
         domain  n_columns  share_pct                 equal_weight_gower_implication
     agri_group          9      50.00     High influence under equal feature weights
      ownership          6      33.33     High influence under equal feature weights
portfolio_count          2      11.11 Lower column share under equal feature weights
      land_size          1       5.56 Lower column share under equal feature weights
```

Under equal feature weights, agri_group (9) and ownership (6) dominate column share relative to portfolio counts (2) and land size (1).

## 4. Missing acreage profiles

```
                  profile_type                    category  n_households  pct_of_missing  pct_of_sample
                missing_reason              Community land           182           91.92          11.73
                missing_reason Special response code 98/99            16            8.08           1.03
          ownership_vs_missing  community_ownership_flag=1           263           92.93          16.95
        ownership_missing_rate  community_ownership_flag=1           184           69.96          11.86
          ownership_vs_missing  community_ownership_flag=0          1289            7.07          83.05
        ownership_missing_rate  community_ownership_flag=0            14            1.09           0.90
 agri_prevalence_acres_missing                 has_cereals           198           10.10          12.76
agri_prevalence_acres_observed                 has_cereals          1354           82.50          87.24
 agri_prevalence_acres_missing                 has_poultry           198            8.08          12.76
agri_prevalence_acres_observed                 has_poultry          1354            9.68          87.24
 agri_prevalence_acres_missing         has_other_livestock           198           95.96          12.76
agri_prevalence_acres_observed         has_other_livestock          1354           27.03          87.24
 agri_prevalence_acres_missing                 has_legumes           198            3.54          12.76
agri_prevalence_acres_observed                 has_legumes          1354           55.17          87.24
```

Missing acreage is concentrated in community-land measurement (primary reason) and is not missing completely at random relative to ownership. Spec B should distinguish community land from special-response missingness. Do not auto-zero acres.

## 5. Gower / PAM feasibility

```
                      check status                                                                                                                                                                                                                                                                                                                detail  runtime_seconds
             package_import   pass                                                                                                                                                                                                           gower imported; PAM via local precomputed implementation (sklearn_extra skipped: NumPy 2.x incompatibility)              NaN
              spec_A_matrix   pass                                                                                                                                                                                                                                                                               n=1552, p=17 (acres excluded); complete              NaN
          gower_sample_n200   pass                                                                                                                                                                                                                                                                                             distance shape=(200, 200)            0.037
           gower_full_n1552   pass                                                                                                                                                                                                                                                               shape=(1552, 1552); memory_mb=9.2; practical_for_S1=yes            0.883
        pam_smoke_k3_spec_A   pass                                                                                                                                                                                                                  backend=local_pam_precomputed; cluster_sizes={0: 513, 1: 362, 2: 677}; medoid_indices=[30, 193, 801]            0.141
spec_B_candidate_treatments   info Candidates for review (no zero-fill): (1) acres + is_community_land binary from unit==5; (2) acres + missingness indicator with median of observed acres only for non-community special codes; (3) exclude acres for community land rows in a hybrid sensitivity. Exact Spec B treatment remains open pending review.              NaN
          no_zero_fill_rule   pass                                                                                                                                                                                                                                                                      Audit does not replace missing acreage with zero              NaN
```

## Open decisions (require approval before S1)

1. Exact Spec B missingness treatment (candidates listed in feasibility table).
2. Equal feature weights vs domain-level weighting.
3. Whether to drop `n_crop_types` / `n_livestock_types` in a sensitivity specification.

## Output files

- `outputs/tables/11_segmentation_feature_types.csv`
- `outputs/tables/11_segmentation_correlations.csv`
- `outputs/tables/11_domain_contribution.csv`
- `outputs/tables/11_acreage_missing_profiles.csv`
- `outputs/tables/11_gower_pam_feasibility.csv`
