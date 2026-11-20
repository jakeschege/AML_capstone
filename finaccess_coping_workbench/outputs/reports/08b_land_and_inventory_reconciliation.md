# Land and Inventory Reconciliation (08b)

Generated: 2026-11-18 17:01 UTC

## 1. Missing land_size_unit observation

Count: 1

```
                   respondent_id  land_size  land_size_unit  land_size_acres  land_ownership_own_under_your_name  land_ownership_own_jointly_with_spouse  land_ownership_own_jointly_with_someone_else_other_than_spouse  land_ownership_lease_rent_you_dont_own  land_ownership_community_public_owned_e_g_pastoralists  land_ownership_family_owned_e_g_inherited_family_land  land_size_acres_missing  exclusion_of_land_size_unit_still_valid                                                                                                note
6dfb0f7b9411478f9382a752d1855cf4       98.0             NaN              NaN                                 0.0                                     0.0                                                             0.0                                     0.0                                                     0.0                                                    1.0                     True                                     True Unit missing; acres cannot be derived. Predictor exclusion unchanged; do not impute land_size_unit.
```

Conclusion: exclusion of `land_size_unit` as a predictor remains valid. The unit is not imputed.

## 2. Special codes versus community land

Households with missing `land_size_acres`: 198

Mutually exclusive primary reasons (community land takes priority, matching Script 08):

```
     primary_missing_reason  n_households  expected_n  matches_script08_expectation
             Community land           182         182                          True
Special response code 98/99            16          16                          True
```

Detailed distinction / overlap crosstab:

```
                         group  n_households                                                                                                   note
community_only_unit5_not_98_99            11                                                                      Unit 5 without special size codes
 community_and_special_overlap           171 Unit 5 with size 98/99 placeholder; counted as Community land in the mutually exclusive primary reason
    special_only_not_community            16                                               Size 98/99 with unit not 5; Script 08 special-code count
 neither_community_nor_special             0                                                        Should be empty if reasons are fully classified
 total_missing_land_size_acres           198                                                                                           Expected 198
        convertible_unit_codes             4                                                                                                1,2,3,4
```

Interpretation: 171 community-land households also carry size codes 98/99 as placeholders. They remain classified as community land. The Script 08 special-code count of 16 is the non-community remainder (unit not 5). Conversion applies only to unit codes {1, 2, 3, 4}. Unit 5 is not convertible.

## 3. Inventory alignment

Config checks:

```
                               check  value  expected  pass
                  segmentation_count     18        18  True
                       primary_count     23        23  True
                      extended_count     35        35  True
             land_size_unit_excluded      1         1  True
            shannon_sensitivity_only      1         1  True
script08_extras_removed_from_default      1         1  True
```

Variable-level alignment versus Script 08 segmentation inventory:

```
                                                      variable  in_script08_segmentation_inventory  in_current_SEGMENTATION_CANDIDATES               status
                                                has_cash_crops                                True                                True             retained
                                                   has_cereals                                True                                True             retained
                                           has_dairy_livestock                                True                                True             retained
                                              has_horticulture                                True                                True             retained
                                                   has_legumes                                True                                True             retained
                                                has_other_agri                                True                                True             retained
                                           has_other_livestock                                True                                True             retained
                                                   has_poultry                                True                                True             retained
                                              has_roots_tubers                                True                                True             retained
        land_ownership_community_public_owned_e_g_pastoralists                                True                                True             retained
         land_ownership_family_owned_e_g_inherited_family_land                                True                                True             retained
                        land_ownership_lease_rent_you_dont_own                                True                                True             retained
land_ownership_own_jointly_with_someone_else_other_than_spouse                                True                                True             retained
                        land_ownership_own_jointly_with_spouse                                True                                True             retained
                            land_ownership_own_under_your_name                                True                                True             retained
                                               land_size_acres                                True                                True             retained
                                                land_size_unit                                True                               False   excluded_predictor
                                                  n_crop_types                                True                                True             retained
                                             n_livestock_types                                True                                True             retained
                                             shannon_diversity                                True                               False moved_to_sensitivity
```

All reconciliation checks passed: True

## Output files

- `outputs/tables/08b_missing_land_unit_case.csv`
- `outputs/tables/08b_land_missing_reason_summary.csv`
- `outputs/tables/08b_land_missing_reason_crosstab.csv`
- `outputs/tables/08b_land_missing_reason_detail.csv`
- `outputs/tables/08b_inventory_alignment.csv`
- `outputs/tables/08b_inventory_checks.csv`
