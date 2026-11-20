# All-shocks framing audit report

**Status:** Scripts 01–06 completed. Original Jupyter notebook **not** modified.  
**Workbench:** `all_shocks_framing_workbench/`  
**Source farming frame:** `~/Documents/items/fsd/finaccess/data/processed/finaccess_2024_modelling_frame.parquet` (n = **9,756** farming households with reported land size; drought-exposed subset still n = 1,552).  
**Stata metadata:** `~/Documents/items/fsd/finaccess/2024/Data 18+.dta` via **pyreadstat** (variable/value labels extracted in-script; no dependence on `value_labels.csv`).

---

## 1. Does the 2024 dataset support an all-shocks analytical population?

**Yes (verified).**

| Stage | n |
|---|---:|
| Farming households (reported land size) | 9,756 |
| Confirmed ≥1 R2A shock (`any_shock_exposed`) | 5,248 |
| Complete information, no shocks (all R2A = 0) | 4,499 |
| Incomplete shock info, no confirmed positive | 9 |
| Shock-exposed with valid four-class coping | **5,241** |
| Shock-exposed missing/invalid coping | 7 |

Primary modelling population recommended: **n = 5,241** (confirmed shock + valid `coping_category`).  
Missing R2A cells were **not** treated as zero. The nine incomplete cases were excluded from “exposed” unless they had at least one confirmed `1`.

---

## 2. How many households report each shock?

Multi-response R2A (households may appear in several rows). Base = 5,248 shock-exposed.

| Shock (renamed) | n | % of exposed |
|---|---:|---:|
| Major sickness / accident / injury | 2,282 | 43.5% |
| Livestock/crop failure (pests/diseases) | 1,816 | 34.6% |
| High rainfall / flooding / mudslide / lightning | 1,573 | 30.0% |
| Drought loss | 1,552 | 29.6% |
| Death of main income earner / family member | 796 | 15.2% |
| Theft | 641 | 12.2% |
| Job / income / wage cut | 453 | 8.6% |
| Child birth | 418 | 8.0% |
| Human–wildlife conflict | 242 | 4.6% |
| Care-related income/expense shock | 202 | 3.9% |
| Fire / violence | 49 | 0.9% |
| Lost home/land (eviction etc.) | 39 | 0.7% |

Drought remains important (~30% of the shock-exposed pool) but is **not** the modal shock.

---

## 3. How many households report multiple shocks?

Among shock-exposed (n = 5,248):

| Number of shocks | n | % |
|---:|---:|---:|
| 1 | 2,690 | 51.3% |
| 2 | 1,296 | 24.7% |
| 3 | 708 | 13.5% |
| 4+ | 554 | 10.6% |

About half report a single shock; nearly half report two or more.

---

## 4. Coping category distribution (revised modelling population)

Four-class target retained (`Classification_A_Resilience` → `coping_category`). Stata codes: 1 Formal, 2 Informal, 3 Personal, 4 Did Nothing. Code↔string agreement on the farming frame: **9,756 / 9,756**.

| Class | n | % |
|---|---:|---:|
| informal | 2,094 | 40.0% |
| personal | 1,782 | 34.0% |
| did_nothing | 994 | 19.0% |
| formal | 371 | 7.1% |

Compared with the drought-only sample (formal was ~4.8% unweighted), **formal is still the minority class** but absolute support is larger (371 vs 75), which helps CV stability.

**Interpretation of the target (from labels + preparation, not name-guessing):**  
R2C items are coping/management actions after shock experience. `Classification_A_Resilience` is a derived resilience/coping classification with value labels Formal / Informal / Personal / Did Nothing. It is a **single household-level** label tied to the R2 shock module—not one row per shock. The Stata variable label text reads “Device used for Resilience” (likely a survey-bank artefact); value labels and existing preparation are the operational definition used here.

---

## 5. Can we examine coping against individual shock categories?

**Yes, for descriptive analysis**, with an explicit multi-response caveat.

- Tables: `03_coping_by_r2a_shock.csv`  
- A household with several shocks appears in several shock-specific groups.  
- This is valid for **pattern description**, not for causal “effect of shock type on coping” claims without further design.

R2B main-shock analysis is the cleaner **single-classification** companion (`03_coping_by_main_shock_r2b.csv`).

---

## 6. What does R2B add vs R2A?

| Aspect | R2A | R2B |
|---|---|---|
| Nature | Multi-response inventory (12 indicators) | Single “main shock” |
| Label | e.g. `R2A. Child birth` | `R2B. Main Shock` |
| Consistency | — | When main shock = *k*, corresponding R2A indicator = 1 for **100%** of farming-frame cases checked |

R2B is useful for mutually exclusive main-shock profiles. R2A is required for co-occurrence and “any shock” definition. They are complementary, not substitutes.

---

## 7. Is the four-class target still suitable?

**Yes, as the primary target for this audit**, with documented imbalance.

- Formal n = 371 supports stratified 5-fold (≫ 5).  
- Three-class collapsed target retained in the modelling cohort for sensitivity (`formal_or_informal` / `personal` / `did_nothing`).  
- Do **not** merge classes solely to raise scores without a separate justification.

---

## 8. Missingness and imbalance issues

**Verified**

- Shock indicators: ~9 households with NA on R2A; not imputed as zero.  
- Coping: ~46% missing on R2C/`Classification_A` in the **full farming** frame (mostly non-shocked). Among shock-exposed, only 7 lack a valid class.  
- Formal remains rare (~7%). Class-weighted logistic helps formal F1 but does not remove the weakness.

---

## 9. Primary classification feature set (recommendation)

**Model A — household & agricultural characteristics (acres excluded)**  
- Demographics (`age_years`, `sex_bin`, `education`, `urban_rural_bin`)  
- County one-hots  
- Agricultural portfolio + crop/livestock counts  
- Land ownership  
- **Exclude** `land_size_acres` (follows locked acres-excluded decision)  
- **Exclude** R2C, targets, ids, weights  

**Matrix:** n = 5,241 · **p = 68** predictors.

---

## 10. Should shock indicators / main shock enter an extended set?

**Yes, as Model B (contextual), with leakage caveats.**

| Inclusion | Status |
|---|---|
| R2A indicators | Model B only — shock context; same R2 module as coping (contemporaneous, not outcome-derived codes) |
| R2B main-shock one-hots | Model B only — not `Classification_A`, but jointly situated in the shock narrative |
| R2C / Classification_A / collapsed target | **Excluded** as predictors |

**Matrix:** n = 5,241 · **p = 92** predictors.

---

## 11. Baseline feasibility results (Script 05)

Protocol: repeated stratified **5×5** CV, seed **42**, identical folds within each matrix. Preprocessing inside the pipeline (median impute + scale continuous; most-frequent impute for other columns **inside CV only**).

| Model | Macro F1 mean (sd) | Bal. acc. mean | Formal F1 mean |
|---|---:|---:|---:|
| A · Dummy most_frequent | 0.143 (0.000) | 0.250 | 0.000 |
| A · Logistic unweighted | 0.345 (0.014) | 0.354 | 0.070 |
| A · Logistic class-weighted | **0.375 (0.012)** | 0.422 | 0.235 |
| B · Dummy most_frequent | 0.143 (0.000) | 0.250 | 0.000 |
| B · Logistic unweighted | 0.462 (0.013) | 0.467 | 0.097 |
| B · Logistic class-weighted | **0.475 (0.010)** | 0.525 | 0.251 |

**Findings**

- All-shocks Model A class-weighted (~0.375) is in a similar ballpark to the drought-only champion (~0.350), with much larger n.  
- Adding shock context (Model B) **raises** macro F1 substantially (~0.475). Part of this may reflect shared R2-module structure—treat as **associative feasibility**, not proven causal lift.  
- Formal remains the weakest class in every specification.

---

## 12. Implications for agricultural segmentation

| Item | Value |
|---|---|
| Current Spec A n | 1,552 |
| All-shocks modelling n | 5,241 |
| Spec A features | 17 (no acres; no shock/coping) |
| Dense pairwise ratio vs drought | ~**11.4×** |

**Recommendation:** Recompute Spec A (Gower + PAM) on the all-shocks modelling or shock-exposed population using the **same agricultural features**. Drought-only medoids are not valid typology labels for non-drought shock households. Defer full bootstrap stability until after a single k-screen. No full PAM was run in this audit.

---

## 13. What must change in the original notebook? (proposed sequence — not done yet)

1. Replace `analysis_2024 = df24[df24["drought_exposed"] == 1]` with the all-shocks cohort construction (confirmed R2A ≥ 1 + valid coping), keeping the farming land-size filter.  
2. Update Data Understanding / sample narrative from “drought-exposed” to “shock-exposed farming households”.  
3. Rebuild descriptive shock/coping sections using Scripts 02–03 logic.  
4. Rebuild primary/extended matrices as Model A / Model B.  
5. Re-run classification design + baselines; reconsider champion selection.  
6. Recompute Spec A segmentation on the new population.  
7. Keep **2021** as a separate validation track (coping classification not equivalent)—do not pool into 2024 fitting.  
8. Update deployment artefacts only after research lock (out of scope here).

---

## Recommended analytical framing

**Primary population:** FinAccess 2024 farming households (reported land size) with **at least one confirmed R2A shock** and a valid four-class `coping_category` (n ≈ 5,241).  

**Primary question:** How do these households’ agricultural/demographic profiles (and optionally observed shock context) relate to reported coping routes after household and agricultural shocks—not drought alone.  

**Primary predictors:** Model A (acres-excluded).  
**Extended predictors:** Model B (+ R2A + R2B), documented as contextual.  
**Segmentation:** Agricultural Spec A recomputed on the new population.  
**Target:** Keep four-class; three-class as sensitivity.

---

## Verified vs recommendation vs unresolved

| Item | Type |
|---|---|
| Farming frame n=9,756; modelling n=5,241 | Verified |
| Shock frequencies, co-occurrence, multi-shock mix | Verified |
| R2B↔R2A 100% consistency on checked codes | Verified |
| Four-class distribution and baseline scorecards | Verified |
| Stata label “Device used for Resilience” wording | Verified oddity — value labels + prep define the target |
| Prefer Model A primary / Model B extended | Recommendation |
| Recompute Spec A; defer heavy stability | Recommendation |
| How much of Model B gain is module-structure vs useful context | Unresolved |
| Exact questionnaire timing of R2B vs R2C in field | Partially documented — revisit questionnaire text when revising notebook |
| Whether deployment prototype should switch cohorts | Unresolved (research decision) |

---

## Output paths (all under `all_shocks_framing_workbench/`)

### Reports
- `outputs/reports/ALL_SHOCKS_FRAMING_AUDIT_REPORT.md` (this file)
- `outputs/reports/05_run_log.txt`

### Key tables
- `outputs/tables/cohort_flow.csv`
- `outputs/tables/target_distribution.csv`
- `outputs/tables/shock_variable_inventory.csv`
- `outputs/tables/main_shock_inventory.csv`
- `outputs/tables/coping_variable_inventory.csv`
- `outputs/tables/stata_value_label_mappings.csv`
- `outputs/tables/03_shock_frequencies.csv`
- `outputs/tables/03_shock_cooccurrence_counts.csv`
- `outputs/tables/03_coping_by_r2a_shock.csv`
- `outputs/tables/03_coping_by_main_shock_r2b.csv`
- `outputs/tables/05_scorecard_combined.csv`
- `outputs/tables/06_segmentation_implications.csv`

### Prepared data
- `outputs/prepared/all_shocks_farming_cohort.parquet` (descriptive shock-exposed)
- `outputs/prepared/all_shocks_modelling_cohort.parquet` (primary modelling)
- `outputs/prepared/model_a_features.parquet` / `model_b_features.parquet`

### Figures
- `outputs/figures/03_number_of_shocks.png`
- `outputs/figures/03_shock_cooccurrence.png`
- `outputs/figures/03_coping_distribution.png`

---

## Execution notes

| Script | Result |
|---|---|
| 01 audit | Success (Stata metadata via pyreadstat) |
| 02 cohort | Success |
| 03 descriptive | Success |
| 04 features | Success |
| 05 baselines | Success (~20–25 min wall time) |
| 06 segmentation implications | Success (no PAM) |

**No notebook edits. No overwrite of drought workbench artefacts.**
