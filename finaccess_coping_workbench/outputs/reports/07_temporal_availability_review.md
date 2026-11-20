# Temporal Availability Review Report

Generated: 2026-11-18 09:31 UTC

## Purpose

This review organises questionnaire evidence for variables flagged in the leakage audit for temporal verification. It preserves the distinction between confirmed leakage, confirmed redundancy and unresolved predictor suitability.

No variables were automatically excluded. Timing was not inferred from variable names or correlations.

## Review scope

Variables reviewed: 23

```
                                                              variable     group     primary_recommendation evidence_strength
                                                            bank_usage financial flag_temporal_verification        unresolved
                                                 buyer_supplier_credit financial flag_temporal_verification        unresolved
                                                 formal_agri_borrowing financial flag_temporal_verification        unresolved
                                                            loan_usage financial flag_temporal_verification        unresolved
                                                             mfi_usage financial flag_temporal_verification        unresolved
                                                    mobile_money_usage financial flag_temporal_verification        unresolved
                                                         savings_usage financial flag_temporal_verification        unresolved
                                                     used_buyer_credit financial            flag_redundancy         confirmed
                                                   used_digital_credit financial            flag_redundancy         confirmed
                                                    used_formal_credit financial            flag_redundancy         confirmed
                                                        used_insurance financial            flag_redundancy         confirmed
                                                       main_shock_type     shock flag_temporal_verification          possible
                                                     shock_child_birth     shock flag_temporal_verification          possible
                       shock_death_of_main_income_earner_family_member     shock flag_temporal_verification          possible
                       shock_high_rainfall_flooding_mudslide_lightning     shock flag_temporal_verification          possible
                 shock_livestock_crop_failure_due_to_pest_and_diseases     shock flag_temporal_verification          possible
               shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut     shock flag_temporal_verification          possible
shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care     shock flag_temporal_verification          possible
                   shock_loss_or_damage_due_to_human_wildlife_conflict     shock flag_temporal_verification          possible
shock_loss_or_damage_due_tofire_or_violence_including_political_violen     shock flag_temporal_verification          possible
     shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc     shock flag_temporal_verification          possible
             shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions     shock flag_temporal_verification          possible
                   shock_major_sickness_health_problem_accident_injury     shock flag_temporal_verification          possible
```

## Quoted questionnaire anchors

### R2A

> - R2C SINGLE MENTION ONLY PER USE CASE R2A. In the past 12 months have you experienced high cost in managing [insert use case]? Kwa miezi 12 iliyopita umewahi kupata gharama kubwa kutoka kwa na kusimamia... (Ingiza kesi ya matumizi) IF ANY USE CASES MENTIONED AS CODE 1 “YES’’ IN R2A. IF ONLY ONE MENTION IN R2A, AUTOCODE AS MOST IMPORTANT AND GO TO R2C. CODE R2B SHOULD BE IN R2A R2B. Of the above mentioned, which
### R2B

> ND GO TO R2C. CODE R2B SHOULD BE IN R2A R2B. Of the above mentioned, which event/shock made the biggest impact on your household income/expenses in the past 12 months? Ni tukio / mshtuko gani ulioathiri zaidi mapato / matumiziya nyumba yako katika miezi 12 iliyopita? ASK FOR USE CASE MENTIONED IN R2B. IF NONE MENTIONED IN R2A OR R2B, GO TO SECTION R3 MULTIPLE MENTIONS
### R2C

> MULTIPLE MENTIONS POSSIBLE PER USE CASE R2C. What did you do to manage [insert use case from R2B]? [USE CODES IN R2Di] Ulifanya nini ilikudhibiti [USE CODES IN R2Di] R2A. Past 12 Month Shock R2B. Main Shock QR2C. Options Main Shock CODES R2iA Yes=1 No=2 Don't know (DO NOT READ OUT)=98 Borrowed from a bank / Sacco / microfinance/
### Q12

> o Answer Kataa kujibu (DO NOT READ OUT) Q12 In the past 12 months, did you give customers/buyers money, goods or services on credit? Kwa miezi 12 iliyopita, uliwakopesha wateja pesa, bidhaa au huduma? Q12. Give Credit No La Q13 In the past 12 months, did you receive goods or services from your input suppliers/buyer
### Q13

> idhaa au huduma? Q12. Give Credit No La Q13 In the past 12 months, did you receive goods or services from your input suppliers/buyers on credit? Kwa miezi 12 iliyopita, ulipokea bidhaa au huduma kutoka kwa wasambazaji wako kwa mkopo? Q13. Receive Credit Yes, Ndio No, La Don't know, Sijui, (DO NOT READ OUT) Refused to A


## Availability status counts

```
                   availability_status  n_variables
same_reference_window_as_coping_module           12
   insufficient_questionnaire_evidence            7
  documented_past_12_months_outside_R2            3
      unresolved_after_document_review            1
```

## Review recommendation counts

```
              review_recommendation  n_variables
candidate_after_documented_ordering           12
 insufficient_evidence_keep_flagged            7
      keep_flagged_pending_decision            4
```

## Assessments

```
                                                              variable     group leakage_audit_recommendation                    availability_status               review_recommendation  n_evidence_rows                                                 documented_reference_period                                                                                                                                                                                                                  rationale  auto_excluded
                                                            bank_usage financial   flag_temporal_verification    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                1                                                                                                                                                               Variable labels or construction notes exist, but no explicit questionnaire reference period relative to the coping module was recovered.          False
                                                 buyer_supplier_credit financial   flag_temporal_verification   documented_past_12_months_outside_R2       keep_flagged_pending_decision                2                                                       In the past 12 months Questionnaire documents a past-12-months reference period for the agri-credit item. That window overlaps the R2 coping window, so pre-shock availability is not established even though the item is outside the R2 module.          False
                                                 formal_agri_borrowing financial   flag_temporal_verification   documented_past_12_months_outside_R2       keep_flagged_pending_decision                2                                                       In the past 12 months Questionnaire documents a past-12-months reference period for the agri-credit item. That window overlaps the R2 coping window, so pre-shock availability is not established even though the item is outside the R2 module.          False
                                                            loan_usage financial   flag_temporal_verification    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                1                                                                                                                                                               Variable labels or construction notes exist, but no explicit questionnaire reference period relative to the coping module was recovered.          False
                                                             mfi_usage financial   flag_temporal_verification    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                1                                                                                                                                                               Variable labels or construction notes exist, but no explicit questionnaire reference period relative to the coping module was recovered.          False
                                                    mobile_money_usage financial   flag_temporal_verification    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                1                                                                                                                                                               Variable labels or construction notes exist, but no explicit questionnaire reference period relative to the coping module was recovered.          False
                                                         savings_usage financial   flag_temporal_verification    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                1                                                                                                                                                               Variable labels or construction notes exist, but no explicit questionnaire reference period relative to the coping module was recovered.          False
                                                     used_buyer_credit financial              flag_redundancy   documented_past_12_months_outside_R2       keep_flagged_pending_decision                1                                                       In the past 12 months Questionnaire documents a past-12-months reference period for the agri-credit item. That window overlaps the R2 coping window, so pre-shock availability is not established even though the item is outside the R2 module.          False
                                                   used_digital_credit financial              flag_redundancy    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                1                                                                                                                                                               Variable labels or construction notes exist, but no explicit questionnaire reference period relative to the coping module was recovered.          False
                                                    used_formal_credit financial              flag_redundancy       unresolved_after_document_review       keep_flagged_pending_decision                1                                                       In the past 12 months                                                                            Composite includes a documented past-12-months agri-credit component, but other source components lack recovered questionnaire timing evidence.          False
                                                        used_insurance financial              flag_redundancy    insufficient_questionnaire_evidence  insufficient_evidence_keep_flagged                0                                                                                                                                                                                                                        No questionnaire reference-period excerpt was recovered for this usage measure.          False
                                                       main_shock_type     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                4 In the past 12 months; in the past 12 months; response to use case from R2B                                                                        Questionnaire documents R2B as selecting the main shock within the R2A past-12-month shock set, before R2C asks what was done to manage that shock.          False
                                                     shock_child_birth     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
                       shock_death_of_main_income_earner_family_member     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
                       shock_high_rainfall_flooding_mudslide_lightning     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
                 shock_livestock_crop_failure_due_to_pest_and_diseases     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
               shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
                   shock_loss_or_damage_due_to_human_wildlife_conflict     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
shock_loss_or_damage_due_tofire_or_violence_including_political_violen     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
     shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
             shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
                   shock_major_sickness_health_problem_accident_injury     shock   flag_temporal_verification same_reference_window_as_coping_module candidate_after_documented_ordering                3                        In the past 12 months; response to use case from R2B                        R2A documents a past-12-month shock window. R2C asks for actions taken to manage the R2B shock. Same retrospective module as the coping target, with documented question order shock then response.          False
```

## Interpretation

Documented questionnaire text shows that Section R2 asks about shocks in the past 12 months and then asks what the household did to manage the main shock. That supports treating shock indicators and main shock type as same-window measures with a documented within-module order, not as confirmed post-outcome leakage.

Q12 and Q13 also use a past-12-months window. Because that window overlaps the R2 coping window, agri-credit items remain flagged pending decision even though they sit outside the coping module.

Several broader financial usage measures lacked recoverable questionnaire reference-period wording in this review. They remain insufficiently evidenced for timing clearance.

## What this review does not decide

- Final classification predictor inclusions or exclusions
- Whether same-window shock measures should be used in supervised models
- Encoding, missingness handling or feature preparation

## Output files

- `outputs/tables/07_temporal_review_scope.csv`
- `outputs/tables/07_questionnaire_evidence.csv`
- `outputs/tables/07_temporal_availability_assessments.csv`
