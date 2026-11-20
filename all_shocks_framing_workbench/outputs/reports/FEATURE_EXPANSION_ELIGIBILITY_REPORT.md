# Feature expansion eligibility report

**Status:** Script 10 complete (verification only — no classifiers).  
**Population:** existing modelling cohort n = **5,241** (confirmed shock + valid four-class coping).  
**Non-shock households (4,499):** not added (no comparable coping outcome).  
**Stata join QC:** matched **5172** / 5,241; unmatched **69**; Stata ID duplicates = **0** (one-to-one among matched).  
**Outputs:** `10_feature_expansion_inventory.csv`, `10_approved_candidates.csv`, `10_review_candidates.csv`, `10_excluded_candidates.csv`, `10_stata_join_qc.csv`, `prepared/10_stata_welfare_extract_modelling.parquet`.

---

## Verified vs recommendation

| Type | Content |
|---|---|
| **Verified** | Presence, missingness on 5,241, redundancy correlations, Stata join match rates, overlap with Model A/B and R2 modules |
| **Recommendation** | Approved / review / exclude status for the controlled expansion in Scripts 11–12 |

---

## A. Agricultural diversity — approved=0, review=0, exclude=5

**Decision: no diversity variables approved for initial expansion.**

| Variable | Status | Why |
|---|---|---|
| `shannon_diversity` | Exclude | Corr ≈ 0.85 with `n_crop_types + n_livestock_types` already in Model A |
| `n_crop_livestock_items_positive` | Exclude | Corr ≈ 0.96 with the same Model A counts |
| `has_any_cash_crop` / `has_any_dairy` | Exclude | Duplicate of Model A `has_cash_crops` / `has_dairy_livestock` |
| Detailed `crop_livestock_*` | Exclude | High-dimensional overlap with Model A portfolio aggregates |

---

## B. Farm challenges — approved=11, review=0, exclude=14

**Retained:** structural constraints with adequate prevalence (inputs, financing, markets, land, labour).  
**Excluded:** drought / flood / pests / theft / wildlife challenges (shock-module overlap) and near-zero-prevalence items.

Approved examples include input-cost and financing constraints, market-access difficulties, small land sizes and high labour costs.  
These are **not** treated as proven pre-shock traits; they are concurrent agricultural-module flags used cautiously as context.

---

## C. Financial characteristics — approved=13, review=5, exclude=13

**Retained:** general access/usage flags outside R2C — savings, loans (formal/informal), SACCO, bank, mobile money, non-NHIF insurance, buyer-supplier credit, formal agri borrowing.  

**Excluded:**
- All `coping_*` financial actions (target pathway / leakage).
- Redundant composites (`informal_usage`, `used_*` mirrors).
- Ultra-sparse digital-agri service fields.

These approved flags are interpreted as **financial capacity / access**, not as observed shock-coping decisions.

---

## D. Previously unextracted welfare / support — approved=7, review=6, exclude=29

Extracted from `Data 18+.dta` via pyreadstat and joined on `respondent_id` = `interview__id`.

| Variable | Status | Notes |
|---|---|---|
| `B1Di` | Approved | Ate fewer meals (past month) |
| `B3A__9` | Approved | Remittance / family support source |
| `nfhi_11` | Approved | Food security (12 months) |
| `nfhi_12` | Approved | Managed non-food spending |
| `mfhi_32` | Approved | Savings for productive assets |
| `member_any_farmers_association` / `member_any_producer_coop` | Approved | Prepared-frame membership |
| HSNP / gov assistance / remittance-use details | Exclude | >40% missing after join |
| Asset-sale / cut-back / guarantor-loss items | Exclude | Coping-like / consequence leakage |

Unmatched modelling IDs (69) remain missing on Stata-only fields and will be imputed inside CV pipelines — not filled as zero outside the pipeline.

---

## Approved candidate count

**31** variables approved for Script 11.

```
                                                                       variable expansion_group  pct_missing_modelling_5241
                                    farm_challenge_difficulty_finding_customers  farm_challenge                       0.000
                             farm_challenge_difficulty_finding_inputs_seeds_etc  farm_challenge                       0.000
farm_challenge_difficulty_getting_products_to_market_transport_road_infrastruct  farm_challenge                       0.000
                               farm_challenge_difficulty_in_accessing_financing  farm_challenge                       0.000
   farm_challenge_difficulty_starting_financingbecause_of_lack_of_money_capital  farm_challenge                       0.000
                                   farm_challenge_high_increased_cost_of_inputs  farm_challenge                       0.000
                                                farm_challenge_high_labor_costs  farm_challenge                       0.000
                                         farm_challenge_late_delivery_of_inputs  farm_challenge                       0.000
                                              farm_challenge_low_quality_inputs  farm_challenge                       0.000
                                                farm_challenge_small_land_sizes  farm_challenge                       0.000
                     farm_challenge_unscrupulous_middlepersons_price_distortion  farm_challenge                       0.000
                                                                     bank_usage       financial                       0.000
                                                          buyer_supplier_credit       financial                       0.000
                                                          formal_agri_borrowing       financial                       0.000
                                                            insurance_excl_nhif       financial                       0.000
                                                                    loan_formal       financial                       0.000
                                                                  loan_informal       financial                       0.000
                                                                     loan_usage       financial                       0.000
                                                              mobile_bank_usage       financial                       0.000
                                                             mobile_money_usage       financial                       0.000
                                                                    sacco_usage       financial                       0.000
                                                                 savings_formal       financial                       0.000
                                                               savings_informal       financial                       0.000
                                                                  savings_usage       financial                       0.000
                                                 member_any_farmers_association welfare_support                       0.000
                                                       member_any_producer_coop welfare_support                       0.000
                                                                           B1Di welfare_support                       1.317
                                                                         B3A__9 welfare_support                       1.317
                                                                        nfhi_11 welfare_support                       1.317
                                                                        nfhi_12 welfare_support                       1.317
                                                                        mfhi_32 welfare_support                       1.317
```

---

## Next step

Script 11 builds expanded matrices from **Model A + R2B** baseline plus these approved groups only.
