# Non-shock household signal audit

**Status:** Script 09 complete. Descriptive / discovery only — **no new classifiers**.  
**Workbench:** `all_shocks_framing_workbench/`  
**Populations (verified):** farming frame n = **9,756**; shock-exposed = **5,248**; complete R2A with no shocks = **4,499**; incomplete R2A with no confirmed positive = **9**.  
**Rule:** missing R2A indicators were **not** treated as zero. The incomplete group is retained separately and excluded from pairwise comparisons below.  
**Sources:** prepared farming modelling frame (+ selected extract-only portfolio/association columns for the same IDs); Stata metadata via **pyreadstat** (no external value-label file).

---

## Verified findings vs recommendations

| Type | Content |
|---|---|
| **Verified** | Population counts; variable presence/missingness; descriptive group differences; Model A/B membership; coping availability among non-shock households |
| **Recommendation** | Which candidates to consider later; leakage caution; whether non-shock households can enter the four-class classifier |

---

## 1. Meaningful differences between shock-exposed and non-shock households

Descriptive thresholds (not p-values): numeric standardised absolute mean difference ≥ 0.15, or categorical max percentage-point gap ≥ 5.

**Largest descriptive contrasts** (see `09_meaningful_group_differences.csv` for the full list):

| Variable | Domain | Contrast summary |
|---|---|---|
| `loan_informal` | financial_position | max category gap=10.1 pp |
| `informal_usage` | financial_position | max category gap=8.4 pp |
| `sacco_usage` | financial_position | max category gap=5.8 pp |
| `n_crop_livestock_items_positive` | agricultural_characteristics | max category gap=5.7 pp |
| `loan_usage` | financial_position | max category gap=5.1 pp |
| `savings_informal` | financial_position | max category gap=5.1 pp |
| `farm_challenge_floods_heavy_rains` | agricultural_characteristics | max category gap=8.2 pp |
| `farm_challenge_drought` | agricultural_characteristics | max category gap=8.2 pp |
| `farm_reported_drought_challenge` | agricultural_characteristics | max category gap=8.2 pp |
| `farm_challenge_pests_and_diseases` | agricultural_characteristics | max category gap=6.8 pp |
| `shannon_diversity` | other_unclassified | max category gap=5.4 pp |
| `has_roots_tubers` | agricultural_characteristics | max category gap=5.5 pp |
| `has_legumes` | agricultural_characteristics | max category gap=5.4 pp |
| `n_crop_types` | agricultural_characteristics | max category gap=5.2 pp |

**Interpretation (verified pattern, not causal claim):** shock-exposed farming households differ from complete non-shock households on several agricultural-challenge and financial-usage markers already present in the prepared frame. Differences on demographics (age/sex/education/urban-rural) are generally smaller than differences on farm-challenge and inclusion markers.

---

## 2. Additional financial indicators available

In the prepared farming dataset, financial-domain columns include (among others): savings / loan / bank / mobile-money usage flags, formal vs informal savings and loans, insurance (with/without NHIF), buyer-supplier credit, formal agricultural borrowing, and digital-app loan usage.

| Metric | n |
|---|---:|
| Variables tagged `financial_position` | 32 |
| Of which not in Model A | 32 |
| Of which in neither Model A nor B | 32 |

Many of these are **already outside** the current Model A/B matrices (Model A focuses on demographics + agri portfolio + land ownership). Detailed Stata savings/loan product grids (C1_*, E*) are largely **not extracted** into the prepared farming frame (see gap file).

---

## 3. Additional agricultural indicators available

Agricultural domain coverage is relatively rich in the prepared frame:

- Land size / ownership
- Crop and livestock portfolio aggregates (`has_*`, `n_crop_types`, `n_livestock_types`)
- Extensive **farm challenge** indicators (inputs, finance, markets, yields, pests, storage, extension access, etc.)
- Extract-only crop/livestock item detail (summarised as `n_crop_livestock_items_positive` plus top-frequency items)

| Metric | n |
|---|---:|
| Variables tagged `agricultural_characteristics` | 75 |
| Not in Model A | 58 |

**Verified gap relative to Model A:** farm-challenge indicators and land-size acres are prepared but mostly **not** in the primary Model A feature list (acres intentionally excluded earlier; challenges never added).

---

## 4. Household welfare and food security indicators

Within the **prepared** farming frame, dedicated food-security / dietary-diversity / consumption modules are **sparse or absent** (domain tag `food_security_and_welfare`: **0** columns).

Stata metadata does contain related survey content (e.g. hunger/safety-net / remittance / assistance wording). Those variables are catalogued as **unextracted gap candidates** in `09_stata_unextracted_gap_candidates.csv` (n = 37). They are **not** verified as available in the current modelling frame.

---

## 5. Potential indicators of unclassified household stress

Domain `other_household_stress` and farm-challenge items can mark production, market, financing and management difficulties **outside** the 12 R2A shock labels.

| Metric | n |
|---|---:|
| Variables tagged `other_household_stress` | 0 |

**Important distinction (recommendation):**
1. **Reported shock (R2A/R2B)** — explicit shock module.
2. **Vulnerability / capacity** — education, land, portfolio, membership, financial access.
3. **Possible consequences / concurrent stress** — farm challenges, hardship-like items.
4. **Coping behaviour** — R2C / `coping_*` / `coping_category` (exclude from predictors).

A farm challenge or hardship marker is **not** proof of an unreported R2A shock.

---

## 6. Candidates not currently in Model A or Model B

Total audited columns (prepared + selected extract extras): **183**.  
Neither Model A nor Model B: **147**.

Notable groups not in A/B:
- Most `farm_challenge_*` indicators
- Financial usage / insurance / credit flags (`savings_usage`, `loan_usage`, `insurance_*`, etc.)
- Some association membership fields (where not already engineered into Model A)
- Land size in acres (prepared but excluded from primary Model A by prior decision)
- Extract crop-item detail

Full listing: `09_variable_audit_full.csv`.

---

## 7. Candidates suitable for further consideration

### Inventory A — potential household / agricultural characteristics
(Top by descriptive rank; see `09_inventory_a_household_characteristics.csv`)

| Variable | Domain | Suitability | Notes |
|---|---|---|---|
| `loan_informal` | financial_position | consider | max category gap=10.1 pp |
| `informal_usage` | financial_position | consider | max category gap=8.4 pp |
| `sacco_usage` | financial_position | consider | max category gap=5.8 pp |
| `n_crop_livestock_items_positive` | agricultural_characteristics | consider | max category gap=5.7 pp |
| `crop_livestock_beans` | agricultural_characteristics | consider | max category gap=5.3 pp |
| `loan_usage` | financial_position | consider | max category gap=5.1 pp |
| `savings_informal` | financial_position | consider | max category gap=5.1 pp |
| `used_formal_credit` | financial_position | consider | max category gap=4.9 pp |
| `crop_livestock_cassava` | financial_position | consider | max category gap=4.7 pp |
| `crop_livestock_maize` | agricultural_characteristics | consider | max category gap=4.1 pp |
| `crop_livestock_tea` | agricultural_characteristics | consider | max category gap=3.8 pp |
| `buyer_supplier_credit` | financial_position | consider | max category gap=3.4 pp |
| `used_buyer_credit` | financial_position | consider | max category gap=3.4 pp |
| `farm_decision_maker_secondary` | agricultural_characteristics | consider | exposed mean=2.240 med=1.000; non_shock mean=2.094 med=1.000 |
| `crop_livestock_chicken_layers` | agricultural_characteristics | consider | max category gap=2.5 pp |

### Inventory B — potential contextual indicators
(Top by descriptive rank; see `09_inventory_b_contextual_indicators.csv`)

| Variable | Domain | Suitability | Notes |
|---|---|---|---|
| `loan_informal` | financial_position | consider | max category gap=10.1 pp |
| `informal_usage` | financial_position | consider | max category gap=8.4 pp |
| `sacco_usage` | financial_position | consider | max category gap=5.8 pp |
| `n_crop_livestock_items_positive` | agricultural_characteristics | consider | max category gap=5.7 pp |
| `crop_livestock_beans` | agricultural_characteristics | consider | max category gap=5.3 pp |
| `loan_usage` | financial_position | consider | max category gap=5.1 pp |
| `savings_informal` | financial_position | consider | max category gap=5.1 pp |
| `used_formal_credit` | financial_position | consider | max category gap=4.9 pp |
| `crop_livestock_cassava` | financial_position | consider | max category gap=4.7 pp |
| `crop_livestock_maize` | agricultural_characteristics | consider | max category gap=4.1 pp |
| `crop_livestock_tea` | agricultural_characteristics | consider | max category gap=3.8 pp |
| `buyer_supplier_credit` | financial_position | consider | max category gap=3.4 pp |
| `used_buyer_credit` | financial_position | consider | max category gap=3.4 pp |
| `farm_decision_maker_secondary` | agricultural_characteristics | consider | exposed mean=2.240 med=1.000; non_shock mean=2.094 med=1.000 |
| `crop_livestock_chicken_layers` | agricultural_characteristics | consider | max category gap=2.5 pp |
| `has_any_cash_crop` | agricultural_characteristics | consider | max category gap=2.3 pp |
| `savings_usage` | financial_position | consider | max category gap=2.1 pp |
| `formal_agri_borrowing` | financial_position | consider | max category gap=2.1 pp |
| `crop_livestock_sweet_potatoes` | agricultural_characteristics | consider | max category gap=2.0 pp |
| `crop_livestock_goat_indegenious` | agricultural_characteristics | consider | max category gap=2.0 pp |

**Recommendation:** prioritise low-missingness vulnerability/capacity features (financial access, membership, selected agri challenges with clear definitions) for a *future* feature-expansion experiment. Do **not** add them yet.

---

## 8. Variables to exclude (missingness, redundancy, leakage)

Exclude from predictor consideration now:

| Reason | Examples / scope |
|---|---|
| Target / coping construction | all `coping_*`, `coping_category*` |
| Cohort definition flags | `any_shock_exposed`, `number_of_shocks_reported`, `drought_exposed` |
| High missingness in either group (>40%) | see `09_exclusion_candidates.csv` |
| Redundant with Model A | county one-hots / sex / education / `has_*` already used |
| Shock module already in Model B | `shock_*`, `main_shock_*` |

Farm-challenge variables are **not automatic excludes**, but they carry **temporal / consequence risk** and need case-by-case review before modelling.

---

## 9. Can the 4,499 non-shock households contribute directly to the four-class coping classifier?

**Verified answer: No — not under the current target definition.**

The four-class outcome (`Classification_A_Resilience` → `coping_category`) is defined in the R2 shock/coping module for households with shock experience. Among complete non-shock households:

```
coping_category
NaN    4499
```

Non-shock households therefore lack a comparable, valid four-class coping label for the same classification problem. Including them with an imputed class (e.g. forcing `did_nothing`) would change the estimand and is **not** supported by the current audit.

They **can** contribute to:
- descriptive contrasts for feature discovery (this script);
- models of shock exposure itself (different target);
- future multi-task / two-stage designs (exposure then coping) — recommendation only.

---

## 10. What additional information would make them eligible for that target?

To use non-shock households **in the same four-class coping classifier**, one would need **verified** outcome information that is definitionally comparable, for example:

1. A survey measure of coping/resilience classification collected for households **without** R2A shocks (currently not available in the prepared frame); or
2. An explicitly redesigned target that does not require shock exposure (new research question); or
3. Linkage to another wave/module with aligned outcome labels (not available here).

Extracting additional **predictors** (food security, remittances, household size, etc.) from Stata would enrich Model A/B for the **existing** 5,241 modelling households, but would **not** by itself make the 4,499 eligible for the current four-class shock-coping target.

---

## Incomplete R2A group (n = 9)

Retained separately (`09_incomplete_group_profile.csv`). Too small for inference; do not collapse into “no shock”.

---

## Outputs

| File | Description |
|---|---|
| `outputs/tables/09_population_counts.csv` | Group sizes |
| `outputs/tables/09_variable_audit_full.csv` | Full variable audit |
| `outputs/tables/09_group_distributions.csv` | Distribution summaries |
| `outputs/tables/09_meaningful_group_differences.csv` | Highlighted contrasts |
| `outputs/tables/09_domain_coverage.csv` | Domain counts |
| `outputs/tables/09_inventory_a_household_characteristics.csv` | Inventory A |
| `outputs/tables/09_inventory_b_contextual_indicators.csv` | Inventory B |
| `outputs/tables/09_candidate_inventories.csv` | Combined inventories |
| `outputs/tables/09_exclusion_candidates.csv` | Exclusion list |
| `outputs/tables/09_incomplete_group_profile.csv` | Incomplete n=9 profile |
| `outputs/tables/09_stata_unextracted_gap_candidates.csv` | Stata gaps not in prepared frame |
| `outputs/reports/NON_SHOCK_HOUSEHOLD_SIGNAL_AUDIT.md` | This report |

---

*Script 09 does not modify the notebook, Model A/B matrices, or Scripts 07–08 outputs.*
