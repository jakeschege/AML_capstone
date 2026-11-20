# Predictor Leakage Audit Report

Generated: 2026-11-18 08:17 UTC

## What was audited

Variable groups investigated so far for the drought-exposed analytical sample (`n = 1552`):

- targets and coping action indicators
- agricultural portfolio and land characteristics
- demographics
- shock indicators and main shock type
- financial access variables
- identifiers and survey weights

No variables were dropped, encoded or transformed. Recommendations are for review before feature preparation.

## Recommendation counts

```
          summary_type                      category  n_variables
primary_recommendation           candidate_predictor           24
primary_recommendation    flag_temporal_verification           19
primary_recommendation exclude_direct_target_leakage           16
primary_recommendation       retain_descriptive_only            8
primary_recommendation               flag_redundancy            5
primary_recommendation        exclude_constant_value            2
     segmentation_role                       exclude           41
     segmentation_role             include_candidate           19
     segmentation_role                not_applicable           12
     segmentation_role                          flag            1
     segmentation_role              descriptive_only            1
   classification_role                       exclude           25
   classification_role             include_candidate           24
   classification_role                          flag           24
   classification_role              descriptive_only            1
```

## Confirmed exclusions

```
                                                               variable         group        primary_recommendation evidence_strength classification_role                                                                                                               notes
          coping_worked_more_got_additional_jobs_farmed_additional_land coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
coping_managing_techniques_got_assistance_gift_from_friends_family_comm coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
coping_sale_of_assets_e_g_livestocka_house_car_plot_of_land_shares_or_s coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                coping_cut_back_on_expenses_adjust_consumption_patterns coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                             coping_loans_credit_borrow coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                                         coping_savings coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                            coping_relocated_changed_place_of_residence coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                               coping_claimed_insurance coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                  coping_withdrew_transfered_child_children_from_school coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
coping_received_assistance_from_government_and_non_governmental_organiz coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                                     coping_did_nothing coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                                   coping_other_specify coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                          coping_made_a_drainage_system coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                      coping_improved_security_measures coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                      coping_reported_at_police_station coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                     coping_reinvested_business_profits coping_action exclude_direct_target_leakage         confirmed             exclude                                                     Direct R2C coping action used to characterise the coping target
                                                     shock_drought_loss         shock        exclude_constant_value         confirmed             exclude Constant in the drought-exposed analytical sample / sample-defining exposure; No variation in the analytical sample
                                                        drought_exposed   sample_flag        exclude_constant_value         confirmed             exclude Constant in the drought-exposed analytical sample / sample-defining exposure; No variation in the analytical sample
```

## Flags requiring review

```
                                                              variable        group     primary_recommendation evidence_strength segmentation_role classification_role                                                                                                                                                     notes
                                                     shannon_diversity agricultural            flag_redundancy         confirmed              flag                flag                                            Redundant richness summary vs portfolio measures (richness_summary_correlated_with_active_groups; metric=97.0)
                                                     shock_child_birth        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
                   shock_major_sickness_health_problem_accident_injury        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
                       shock_high_rainfall_flooding_mudslide_lightning        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
                       shock_death_of_main_income_earner_family_member        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
     shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
               shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
             shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
                 shock_livestock_crop_failure_due_to_pest_and_diseases        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
shock_loss_or_damage_due_tofire_or_violence_including_political_violen        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
                   shock_loss_or_damage_due_to_human_wildlife_conflict        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care        shock flag_temporal_verification          possible           exclude                flag                                                                        Additional shock indicators share R2 module reference period with coping responses
                                                       main_shock_type        shock flag_temporal_verification          possible           exclude                flag                                      Principal shock within indicator set; share R2 module timing with coping target; Mapped indicator consistency 100.0%
                                                 buyer_supplier_credit    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
                                                 formal_agri_borrowing    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
                                                    used_formal_credit    financial            flag_redundancy         confirmed           exclude                flag         documented_or_composite with loan_usage;loan_formal;formal_agri_borrowing (agreement 100.0%); Also subject to unresolved finance timing vs coping
                                                     used_buyer_credit    financial            flag_redundancy         confirmed           exclude                flag                                           exact_recode with buyer_supplier_credit (agreement 100.0%); Also subject to unresolved finance timing vs coping
                                                   used_digital_credit    financial            flag_redundancy         confirmed           exclude                flag                                                exact_recode with digital_app_loan (agreement 100.0%); Also subject to unresolved finance timing vs coping
                                                        used_insurance    financial            flag_redundancy         confirmed           exclude                flag                                             exact_recode with insurance_excl_nhif (agreement 100.0%); Also subject to unresolved finance timing vs coping
                                                            bank_usage    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
                                                         savings_usage    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
                                                            loan_usage    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
                                                    mobile_money_usage    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
                                                             mfi_usage    financial flag_temporal_verification        unresolved           exclude                flag Provisional classification candidate pending timing/leakage verification; Excluded from agricultural segmentation inputs by approved preparation decision
```

## Candidate predictors

```
                                                      variable        group segmentation_role classification_role                                                                                 notes
                                                has_cash_crops agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                                   has_cereals agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                           has_dairy_livestock agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                              has_horticulture agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                                   has_legumes agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                                has_other_agri agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                           has_other_livestock agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                                   has_poultry agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                              has_roots_tubers agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                                  n_crop_types agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                             n_livestock_types agricultural include_candidate   include_candidate             Agricultural portfolio characteristic for segmentation and classification
                                                     land_size         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
                                                land_size_unit         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
                            land_ownership_own_under_your_name         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
                        land_ownership_own_jointly_with_spouse         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
land_ownership_own_jointly_with_someone_else_other_than_spouse         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
                        land_ownership_lease_rent_you_dont_own         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
        land_ownership_community_public_owned_e_g_pastoralists         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
         land_ownership_family_owned_e_g_inherited_family_land         land include_candidate   include_candidate                Land size/ownership characteristic for segmentation and classification
                                                     age_years  demographic    not_applicable   include_candidate Demographic background for classification; not a core agricultural segmentation input
                                                           sex  demographic    not_applicable   include_candidate Demographic background for classification; not a core agricultural segmentation input
                                                     education  demographic    not_applicable   include_candidate Demographic background for classification; not a core agricultural segmentation input
                                                   urban_rural  demographic    not_applicable   include_candidate Demographic background for classification; not a core agricultural segmentation input
                                                        county  demographic    not_applicable   include_candidate Demographic background for classification; not a core agricultural segmentation input
```

## Direct leakage checks

```
                                            check  n_action_positive  n_target_class  n_both  n_action_without_class  n_class_without_action  pct_action_in_class evidence_strength                                                                                                            notes
coping_did_nothing_vs_coping_category_did_nothing                415             415     415                       0                       0                100.0         confirmed                                                            Action indicator aligns with did_nothing target class
       any_non_nothing_action_among_active_copers               1137            1137    1137                       0                       0                100.0         confirmed Coping action indicators belong to the same R2C response module used to characterise resilience/coping behaviour
   coping_category_collapsed_from_coping_category               1552            1552    1552                       0                       0                100.0         confirmed                                              Collapsed target is a deterministic transform of the primary target
```

## Construct overlap summary

```
                    pair_type            variable_a                 variable_b  n_both_positive  pct_both_positive evidence_strength                                                                                                                    notes
     finance_vs_coping_action            loan_usage coping_loans_credit_borrow               87               5.61          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action    used_formal_credit coping_loans_credit_borrow               87               5.61          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action     used_buyer_credit coping_loans_credit_borrow               27               1.74          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action buyer_supplier_credit coping_loans_credit_borrow               27               1.74          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action formal_agri_borrowing coping_loans_credit_borrow               20               1.29          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action   used_digital_credit coping_loans_credit_borrow                2               0.13          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action         savings_usage             coping_savings               88               5.67          possible                                                                  Conceptual overlap; not confirmed direct target leakage
     finance_vs_coping_action        used_insurance   coping_claimed_insurance                0               0.00          possible                                                                  Conceptual overlap; not confirmed direct target leakage
shock_indicator_vs_main_shock    shock_*_indicators            main_shock_type             1552             100.00         confirmed Main shock is the principal reported shock within the indicator set; related constructs, not automatic exclusion of both
```

## Redundancy register

```
           variable                                related_variables                                   relationship  pct_agreement evidence_strength source_investigation
  used_buyer_credit                            buyer_supplier_credit                                   exact_recode          100.0         confirmed  05_financial_access
 used_formal_credit     loan_usage;loan_formal;formal_agri_borrowing                        documented_or_composite          100.0         confirmed  05_financial_access
used_digital_credit                                 digital_app_loan                                   exact_recode          100.0         confirmed  05_financial_access
     used_insurance                              insurance_excl_nhif                                   exact_recode          100.0         confirmed  05_financial_access
  shannon_diversity n_crop_types;n_livestock_types;has_*_group_flags richness_summary_correlated_with_active_groups           97.0         confirmed 02_shannon_portfolio
    main_shock_type                           mapped_shock_indicator           principal_shock_within_indicator_set          100.0         confirmed    04_shock_exposure
 shock_drought_loss                                  drought_exposed    sample_defining_constant_in_analysis_sample          100.0         confirmed    04_shock_exposure
```

## Temporal flags

The temporal table flags unresolved or possible timing issues. These are not confirmed exclusions.

- Shock indicators and `main_shock_type`: possible contemporaneous measurement with coping responses in the R2 module
- Financial access variables: unresolved timing relative to drought coping

Rows written: 25

## Interpretation

Confirmed leakage is limited to the coping action indicators and the target fields themselves. Constant drought exposure is excluded because it has no variation in this sample.

Financial usage measures and shock indicators remain analytically interesting, but their timing relative to coping behaviour is not established as safe pre-outcome information. Exact financial duplicates and the Shannon richness summary are flagged for redundancy rather than removed automatically.

Agricultural portfolio and land variables remain the core segmentation candidates. Demographics and the non-excluded classification candidates remain provisional pending your review of the flags above.

## Recommended next step

Review the register and decide which flagged variables proceed to feature preparation. Do not treat every conceptual relationship as confirmed leakage.

## Output files

- `outputs/tables/06_leakage_variable_register.csv`
- `outputs/tables/06_direct_leakage_checks.csv`
- `outputs/tables/06_construct_overlap_summary.csv`
- `outputs/tables/06_redundancy_register.csv`
- `outputs/tables/06_temporal_flags.csv`
- `outputs/tables/06_recommendation_counts.csv`
