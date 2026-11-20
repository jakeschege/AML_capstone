# Missingness and Encoding Audit Report

Generated: 2026-11-18 09:48 UTC

## Purpose

Audit preparation requirements for the segmentation, primary classification and extended classification candidate inventories. No encoding, imputation, scaling or rare-category removal was performed.

Land size unit decision used in this audit: **acres** (`land_size_acres`).

## Inventories

- Segmentation candidates: 20
- Primary classification candidates: 25
- Extended classification candidates: 37

Sample size: 1552

## Missingness

Variables with any missing values:

```
   inventory        variable    n  n_non_missing  n_missing  missing_pct  n_unique   dtype
segmentation land_size_acres 1552           1354        198        12.76        91 float64
segmentation  land_size_unit 1552           1551          1         0.06         5 float64
```

Land size missing reasons:

```
             missing_reason  n_households  pct_of_sample  pct_of_missing
             Community land           182          11.73           91.92
Special response code 98/99            16           1.03            8.08
```

## Binary coding

```
                                                              variable expected_values observed_values  n_unique  coding_consistent  n_missing
                                                        has_cash_crops             0,1             0,1         2               True          0
                                                           has_cereals             0,1             0,1         2               True          0
                                                   has_dairy_livestock             0,1             0,1         2               True          0
                                                      has_horticulture             0,1             0,1         2               True          0
                                                           has_legumes             0,1             0,1         2               True          0
                                                        has_other_agri             0,1             0,1         2               True          0
                                                   has_other_livestock             0,1             0,1         2               True          0
                                                           has_poultry             0,1             0,1         2               True          0
                                                      has_roots_tubers             0,1             0,1         2               True          0
                                    land_ownership_own_under_your_name             0,1             0,1         2               True          0
                                land_ownership_own_jointly_with_spouse             0,1             0,1         2               True          0
        land_ownership_own_jointly_with_someone_else_other_than_spouse             0,1             0,1         2               True          0
                                land_ownership_lease_rent_you_dont_own             0,1             0,1         2               True          0
                land_ownership_community_public_owned_e_g_pastoralists             0,1             0,1         2               True          0
                 land_ownership_family_owned_e_g_inherited_family_land             0,1             0,1         2               True          0
                                                     shock_child_birth             0,1             0,1         2               True          0
                   shock_major_sickness_health_problem_accident_injury             0,1             0,1         2               True          0
                       shock_high_rainfall_flooding_mudslide_lightning             0,1             0,1         2               True          0
                       shock_death_of_main_income_earner_family_member             0,1             0,1         2               True          0
     shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc             0,1             0,1         2               True          0
               shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut             0,1             0,1         2               True          0
             shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions             0,1             0,1         2               True          0
                 shock_livestock_crop_failure_due_to_pest_and_diseases             0,1             0,1         2               True          0
shock_loss_or_damage_due_tofire_or_violence_including_political_violen             0,1             0,1         2               True          0
                   shock_loss_or_damage_due_to_human_wildlife_conflict             0,1             0,1         2               True          0
shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care             0,1             0,1         2               True          0
                                                                   sex             1,2             1,2         2               True          0
                                                           urban_rural             1,2             1,2         2               True          0
```

## Education ordinal assessment

```
 variable observed_codes                                     labels code_1_label  provisional_ordinal                                    encoding_recommendation                                                                                           notes
education        1,2,3,4 1=None; 2=Primary; 3=Secondary; 4=Tertiary         None                 True ordinal_or_ordered_categorical_pending_feature_preparation Codes 1-4 correspond to None, Primary, Secondary, Tertiary. Encoding not applied in this audit.
```

## Continuous summaries

```
       variable  n_non_missing  n_missing  missing_pct      mean  median       std       min   p05  p95      max  skewness
      age_years           1552          0         0.00 43.798325    41.0 16.055470 16.000000 21.00 72.0 101.0000  0.497885
land_size_acres           1354        198        12.76  3.404323     1.0 13.593814  0.000023  0.25  8.0 232.2787 11.252510
```

## Target representation

```
                   target        class_label  n_households   pct  below_5pct  below_n30
          coping_category           personal           615 39.63       False      False
          coping_category           informal           447 28.80       False      False
          coping_category        did_nothing           415 26.74       False      False
          coping_category             formal            75  4.83        True      False
coping_category_collapsed           personal           615 39.63       False      False
coping_category_collapsed formal_or_informal           522 33.63       False      False
coping_category_collapsed        did_nothing           415 26.74       False      False
```

## Rare category flags

```
         item_type variable             category  n  pct          flag
predictor_category   county      24.0:West Pokot 77 4.96 rare_category
predictor_category   county          42.0:Kisumu 75 4.83 rare_category
predictor_category   county           40.0:Busia 74 4.77 rare_category
predictor_category   county         30.0:Baringo 63 4.06 rare_category
predictor_category   county         43.0:Homabay 63 4.06 rare_category
predictor_category   county           41.0:Siaya 61 3.93 rare_category
predictor_category   county         23.0:Turkana 60 3.87 rare_category
predictor_category   county        10.0:Marsabit 58 3.74 rare_category
predictor_category   county         17.0:Makueni 58 3.74 rare_category
predictor_category   county     6.0:Taita-Taveta 55 3.54 rare_category
predictor_category   county         39.0:Bungoma 53 3.41 rare_category
predictor_category   county        31.0:Laikipia 50 3.22 rare_category
predictor_category   county          22.0:Kiambu 39 2.51 rare_category
predictor_category   county           45.0:Kisii 39 2.51 rare_category
predictor_category   county         25.0:Samburu 37 2.38 rare_category
predictor_category   county          32.0:Nakuru 35 2.26 rare_category
predictor_category   county   13.0:Tharaka-Nithi 34 2.19 rare_category
predictor_category   county 28.0:Elgeyo-Marakwet 34 2.19 rare_category
predictor_category   county            12.0:Meru 33 2.13 rare_category
predictor_category   county           36.0:Bomet 27 1.74 rare_category
predictor_category   county            8.0:Wajir 25 1.61 rare_category
predictor_category   county          38.0:Vihiga 24 1.55 rare_category
predictor_category   county          7.0:Garissa 22 1.42 rare_category
predictor_category   county          11.0:Isiolo 19 1.22 rare_category
predictor_category   county           29.0:Nandi 18 1.16 rare_category
predictor_category   county          1.0:Mombasa 16 1.03 rare_category
predictor_category   county     27.0:Uasin Gishu 15 0.97 rare_category
predictor_category   county       4.0:Tana River 13 0.84 rare_category
predictor_category   county       18.0:Nyandarua 13 0.84 rare_category
predictor_category   county          44.0:Migori 13 0.84 rare_category
predictor_category   county            14.0:Embu 11 0.71 rare_category
predictor_category   county           15.0:Kitui 11 0.71 rare_category
predictor_category   county             5.0:Lamu 10 0.64 rare_category
predictor_category   county           33.0:Narok  9 0.58 rare_category
predictor_category   county        21.0:Murang'a  7 0.45 rare_category
predictor_category   county    47.0:Nairobi City  7 0.45 rare_category
predictor_category   county     26.0:Trans Nzoia  6 0.39 rare_category
predictor_category   county         35.0:Kericho  5 0.32 rare_category
predictor_category   county        37.0:Kakamega  5 0.32 rare_category
predictor_category   county          9.0:Mandera  4 0.26 rare_category
```

## Preparation recommendations

```
                 topic          variable                                    finding                                                             recommended_action  apply_now
           missingness   land_size_acres                          missing_pct=12.76 document_missingness_and_decide_imputation_or_indicator_in_feature_preparation      False
           missingness    land_size_unit                           missing_pct=0.06                                         investigate_before_feature_preparation      False
        land_size_unit   land_size_acres          standardised_unit_acres_confirmed                                     use_land_size_acres_in_feature_preparation      False
 land_size_missingness   land_size_acres                      Community land: n=182                            retain_missingness_without_imputation_in_this_audit      False
 land_size_missingness   land_size_acres          Special response code 98/99: n=16                            retain_missingness_without_imputation_in_this_audit      False
      ordinal_encoding         education 1=None; 2=Primary; 3=Secondary; 4=Tertiary                     ordinal_or_ordered_categorical_pending_feature_preparation      False
  categorical_encoding               sex  categorical_codes_require_encoding_scheme                        one_hot_or_target_aware_encoding_in_feature_preparation      False
  categorical_encoding       urban_rural  categorical_codes_require_encoding_scheme                        one_hot_or_target_aware_encoding_in_feature_preparation      False
  categorical_encoding            county  categorical_codes_require_encoding_scheme                        one_hot_or_target_aware_encoding_in_feature_preparation      False
  categorical_encoding    land_size_unit  categorical_codes_require_encoding_scheme                        one_hot_or_target_aware_encoding_in_feature_preparation      False
  categorical_encoding   main_shock_type  categorical_codes_require_encoding_scheme                        one_hot_or_target_aware_encoding_in_feature_preparation      False
            redundancy shannon_diversity    flagged_redundant_with_portfolio_counts                                    assess_in_or_out_during_feature_preparation      False
 target_representation   coping_category                      formal_class_pct=4.83                       plan_stratified_validation_and_consider_collapsed_target      False
continuous_preparation         age_years                   skewness=0.498; complete                             keep_numeric_decide_scaling_in_feature_preparation      False
continuous_preparation   land_size_acres         skewness=11.253; missing_pct=12.76                   decide_scaling_and_missing_indicator_later_do_not_impute_now      False
             hard_rule    all_candidates                                 audit_only                do_not_encode_impute_scale_or_drop_rare_categories_in_this_step      False
```

## Interpretation

Most candidate predictors are complete. The main missingness issue is standardised land size, largely due to community land and special response codes. Education is a provisional ordinal scale from None to Tertiary. The four-class target has a sparse formal class, so stratified validation and the collapsed three-class target remain important design considerations.

Encoding schemes, imputation choices and rare-category handling are deferred to feature preparation.

## Output files

- `outputs/tables/08_inventory_segmentation.csv`
- `outputs/tables/08_inventory_primary_classification.csv`
- `outputs/tables/08_inventory_extended_classification.csv`
- `outputs/tables/08_missingness_by_variable.csv`
- `outputs/tables/08_categorical_frequencies.csv`
- `outputs/tables/08_binary_coding_checks.csv`
- `outputs/tables/08_ordinal_education_assessment.csv`
- `outputs/tables/08_continuous_summaries.csv`
- `outputs/tables/08_land_size_missing_reasons.csv`
- `outputs/tables/08_rare_category_flags.csv`
- `outputs/tables/08_target_representation.csv`
- `outputs/tables/08_preparation_recommendations.csv`
