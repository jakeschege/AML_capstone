# Financial Access Investigation Report

Generated: 2026-11-18 07:48 UTC

## What was investigated

Eleven financial access variables for the drought-exposed analytical sample (`n = 1552`):

```
buyer_supplier_credit, formal_agri_borrowing, used_formal_credit, used_buyer_credit, used_digital_credit, used_insurance, bank_usage, savings_usage, loan_usage, mobile_money_usage, mfi_usage
```

The investigation distinguished agri-trade credit questions, derived usage flags and broader service usage measures. Variables were not removed, combined, encoded or transformed.

## Observed findings

### Inventory

```
             variable            construct_group   dtype    n  missing  missing_pct  n_unique  min  max
buyer_supplier_credit agri_trade_credit_question float64 1552        0          0.0         2  1.0  2.0
formal_agri_borrowing agri_trade_credit_question float64 1552        0          0.0         2  1.0  2.0
   used_formal_credit         derived_usage_flag   int64 1552        0          0.0         2  0.0  1.0
    used_buyer_credit         derived_usage_flag   int64 1552        0          0.0         2  0.0  1.0
  used_digital_credit         derived_usage_flag   int64 1552        0          0.0         2  0.0  1.0
       used_insurance         derived_usage_flag   int64 1552        0          0.0         2  0.0  1.0
           bank_usage              service_usage   int64 1552        0          0.0         2  0.0  1.0
        savings_usage              service_usage   int64 1552        0          0.0         2  0.0  1.0
           loan_usage              service_usage   int64 1552        0          0.0         2  0.0  1.0
   mobile_money_usage              service_usage float64 1552        0          0.0         3  1.0  3.0
            mfi_usage              service_usage float64 1552        0          0.0         3  1.0  3.0
```

Total missing values across the 11 variables: 0

### Frequencies

```
             variable            construct_group  code         label  n_households   pct
buyer_supplier_credit agri_trade_credit_question     1           Yes           197 12.69
buyer_supplier_credit agri_trade_credit_question     2            No          1355 87.31
formal_agri_borrowing agri_trade_credit_question     1           Yes           129  8.31
formal_agri_borrowing agri_trade_credit_question     2            No          1423 91.69
   used_formal_credit         derived_usage_flag     0            No           443 28.54
   used_formal_credit         derived_usage_flag     1           Yes          1109 71.46
    used_buyer_credit         derived_usage_flag     0            No          1355 87.31
    used_buyer_credit         derived_usage_flag     1           Yes           197 12.69
  used_digital_credit         derived_usage_flag     0            No          1531 98.65
  used_digital_credit         derived_usage_flag     1           Yes            21  1.35
       used_insurance         derived_usage_flag     0            No          1490 96.01
       used_insurance         derived_usage_flag     1           Yes            62  3.99
           bank_usage              service_usage     0     Non-usage          1042 67.14
           bank_usage              service_usage     1         Usage           510 32.86
        savings_usage              service_usage     0     Non-usage           425 27.38
        savings_usage              service_usage     1         Usage          1127 72.62
           loan_usage              service_usage     0     Non-usage           454 29.25
           loan_usage              service_usage     1         Usage          1098 70.75
   mobile_money_usage              service_usage     1 Currently use          1278 82.35
   mobile_money_usage              service_usage     2   Used to use            75  4.83
   mobile_money_usage              service_usage     3    Never used           199 12.82
            mfi_usage              service_usage     1 Currently use           175 11.28
            mfi_usage              service_usage     2   Used to use           144  9.28
            mfi_usage              service_usage     3    Never used          1233 79.45
```

### Redundancy checks

```
                                                             check    n  n_agree  pct_agree  fully_deterministic                                                                           notes
               used_buyer_credit_equals_buyer_supplier_credit_eq_1 1552     1552     100.00                 True                                         Exact recode of Q12 Yes into a 0/1 flag
used_formal_credit_equals_loan_usage_or_loan_formal_or_formal_agri 1552     1552     100.00                 True                     Documented OR construction from notebook/feature dictionary
                  used_digital_credit_equals_digital_app_loan_eq_1 1552     1552     100.00                 True                                                   Derived from digital_app_loan
                    used_insurance_equals_insurance_excl_nhif_eq_1 1552     1552     100.00                 True                                                Derived from insurance_excl_nhif
                          loan_usage_vs_used_formal_credit_overlap 1552     1098      70.75                False both=1: 1098; loan_usage=1 & used_formal=0: 0; loan_usage=0 & used_formal=1: 11
```

Fully deterministic constructions confirmed: 4

### Pairwise agreement among binary views

Top pairs by exact agreement:

```
               variable_a                variable_b  pct_exact_agreement  n_both_positive  pct_both_positive
buyer_supplier_credit_yes         used_buyer_credit               100.00              197              12.69
       used_formal_credit                loan_usage                99.29             1098              70.75
      used_digital_credit            used_insurance                94.78                1               0.06
formal_agri_borrowing_yes       used_digital_credit                90.59                2               0.13
      used_digital_credit               mfi_current                90.08               21               1.35
formal_agri_borrowing_yes            used_insurance                88.34                5               0.32
buyer_supplier_credit_yes formal_agri_borrowing_yes                88.02               70               4.51
formal_agri_borrowing_yes         used_buyer_credit                88.02               70               4.51
buyer_supplier_credit_yes       used_digital_credit                86.34                3               0.19
        used_buyer_credit       used_digital_credit                86.34                3               0.19
```

### Leakage and coping overlap flags

Conceptual overlap checks:

```
   financial_variable    related_coping_variable coping_var_present n_both_positive pct_both_positive                                   risk_flag                                                                 notes
           loan_usage coping_loans_credit_borrow               True              87              5.61 conceptual_overlap_and_temporal_uncertainty Loan usage may overlap conceptually with borrowing as a coping action
   used_formal_credit coping_loans_credit_borrow               True              87              5.61 conceptual_overlap_and_temporal_uncertainty           Derived formal credit use may overlap with coping borrowing
    used_buyer_credit coping_loans_credit_borrow               True              27              1.74 conceptual_overlap_and_temporal_uncertainty            Buyer/supplier credit may overlap with credit-based coping
buyer_supplier_credit coping_loans_credit_borrow               True              27              1.74 conceptual_overlap_and_temporal_uncertainty            Q12 agri-trade credit may overlap with credit-based coping
formal_agri_borrowing coping_loans_credit_borrow               True              20              1.29 conceptual_overlap_and_temporal_uncertainty         Q13 agri credit question may overlap with credit-based coping
  used_digital_credit coping_loans_credit_borrow               True               2              0.13 conceptual_overlap_and_temporal_uncertainty               Digital credit use may overlap with borrowing as coping
        savings_usage             coping_savings               True              88              5.67 conceptual_overlap_and_temporal_uncertainty       Savings usage may overlap with using savings as a coping action
       used_insurance   coping_claimed_insurance               True               0               0.0 conceptual_overlap_and_temporal_uncertainty    Insurance use/access may overlap with claiming insurance as coping
```

Temporal availability of these finance measures relative to drought coping remains unresolved and is deferred to the predictor leakage audit.

### Suitability notes

```
             variable            construct_group  positive_or_current_use_pct segmentation_suitability                  classification_suitability                                                                                                                                        segmentation_note                                                                                               classification_note                                                                                                                         cautions
buyer_supplier_credit agri_trade_credit_question                        12.69    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                                                                                 
formal_agri_borrowing agri_trade_credit_question                         8.31    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                                                                                 
   used_formal_credit         derived_usage_flag                        71.46    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response. Deterministic composite of loan_usage/loan_formal/formal_agri_borrowing; Derived flag; not interchangeable with source questions
    used_buyer_credit         derived_usage_flag                        12.69    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                    Deterministic duplicate of buyer_supplier_credit Yes; Derived flag; not interchangeable with source questions
  used_digital_credit         derived_usage_flag                         1.35    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                               Low positive rate (1.35%); Derived flag; not interchangeable with source questions
       used_insurance         derived_usage_flag                         3.99    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                               Low positive rate (3.99%); Derived flag; not interchangeable with source questions
           bank_usage              service_usage                        32.86    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                                                                                 
        savings_usage              service_usage                        72.62    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                                                                                 
           loan_usage              service_usage                        70.75    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                                                                                 
   mobile_money_usage              service_usage                        82.35    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                   Three-level usage status, not a simple binary access indicator
            mfi_usage              service_usage                        11.28    provisional_candidate provisional_candidate_pending_leakage_audit May help distinguish households by financial service profile for agricultural segmentation only if treated as finance capacity, not as a coping outcome. Candidate classifier predictor only after leakage audit confirms the measure is not a post-shock coping response.                                                                   Three-level usage status, not a simple binary access indicator
```

Low positive or current-use rates under 5%: used_digital_credit, used_insurance

## Interpretation

Access questions, derived usage flags and service usage measures are related but not interchangeable constructs.

`used_buyer_credit` is an exact 0/1 recode of `buyer_supplier_credit == 1`. `used_formal_credit` is a composite of broader loan usage and agri-credit indicators. These relationships create redundancy for modelling even though the original and derived fields can remain useful for audit.

`mobile_money_usage` and `mfi_usage` retain former-user categories and therefore measure usage history, not simple binary access.

Several finance variables conceptually overlap with coping actions such as borrowing, using savings or claiming insurance. That does not by itself prove leakage, but it does mean they cannot be treated as automatically valid pre-shock predictors.

## Limitations

- Codebook labels were used for interpretation only.
- Pairwise agreement uses binary views of multi-level variables where needed.
- No target-based feature selection was performed.
- Questionnaire timing of finance measures versus coping responses was not resolved here.

## Recommended preparation decisions for review

1. Retain all 11 financial access variables as provisional candidates.
2. Flag `used_buyer_credit` as a deterministic duplicate of `buyer_supplier_credit` for later feature assessment. Do not drop either field yet.
3. Flag `used_formal_credit` as a derived composite overlapping `loan_usage` and `formal_agri_borrowing`.
4. Keep segmentation and classification uses distinct: finance capacity may be relevant for both, but classification use depends on the leakage audit.
5. Do not encode, combine or transform these variables in this step.
6. Carry conceptual overlap with coping actions into script 06 leakage audit.

## Output files

- `outputs/tables/05_financial_inventory.csv`
- `outputs/tables/05_financial_frequencies.csv`
- `outputs/tables/05_financial_redundancy_checks.csv`
- `outputs/tables/05_financial_pairwise_agreement.csv`
- `outputs/tables/05_financial_leakage_flags.csv`
- `outputs/tables/05_financial_suitability_notes.csv`
