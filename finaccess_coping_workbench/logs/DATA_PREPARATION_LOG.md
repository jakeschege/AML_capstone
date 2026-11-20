# Data Preparation Log

Living record of Data Preparation investigations and decisions for the FinAccess 2024 drought-exposed household analysis.

Sample context unless otherwise stated: `analysis_2024`, n = 1,552 drought-exposed households.

### Agricultural portfolio diversity investigation

The relationship between crop counts, livestock counts and Shannon diversity was investigated using the 1,552 drought exposed households in the FinAccess 2024 analytical sample.

The results showed that Shannon diversity was positively associated with the number of crop types, with a correlation of 0.722, but had a weak relationship with livestock counts, with a correlation of 0.072. Shannon diversity was not a deterministic function of either count or their combination.

An additional audit using the nine binary agricultural group indicators showed a correlation of 0.970 between Shannon diversity and the number of active agricultural groups. The observed Shannon values corresponded to the natural logarithm of the number of active groups. This suggests that the measure captures the breadth of agricultural activity across the recorded groups rather than conventional diversity based on abundance shares.

The investigation also found that different combinations of crop and livestock counts could produce the same Shannon value. The measure therefore provides a different summary of agricultural activity from the two portfolio counts, although some redundancy is expected.

The decision is to retain `n_crop_types`, `n_livestock_types`, `shannon_diversity` and the nine binary agricultural indicators as provisional candidate variables. The derived `n_active_groups` variable will remain an audit measure rather than an additional predictor. The usefulness of Shannon diversity will be assessed separately during feature assessment for household segmentation and supervised classification.

No observations were removed, no missing values were introduced and no transformations were applied.

### Demographic variable investigation

The demographic variables `age_years`, `sex`, `education`, `urban_rural` and `county` were investigated using the 1,552 drought exposed households in the FinAccess 2024 analytical sample.

None of the five variables had missing values. `age_years` ranged from 16 to 101 years, with a mean of 43.8 and a median of 41. The distribution was mildly right skewed (skewness 0.50). No ages fell below the adult survey floor of 16. Seven households were aged 90 or above, including five ages outside the IQR fence. These high ages were flagged for later sensitivity review and were not removed.

`sex` was binary coded as male (690 households, 44.5%) and female (862 households, 55.5%). `urban_rural` was binary coded as rural (1,324 households, 85.3%) and urban (228 households, 14.7%).

`education` contained four observed codes in this sample: 1 None (382 households), 2 Primary (651), 3 Secondary (389) and 4 Tertiary (130). Code 5 Other was absent. The label for code 1 was initially reported as missing because default pandas NA parsing treated the codebook text `None` as a missing value. Reading `value_labels.csv` with `keep_default_na=False` recovers the label **None**. The related A20 education codebook also uses code 1 as None, which supports interpreting this as no formal schooling. Codes 1 to 4 therefore form a provisional ordinal schooling ladder. Encoding was not applied in this step.

All 47 counties were represented. Household counts ranged from 1 to 95. Five counties had fewer than 5 households and 11 counties had fewer than 10 households. These sparse counties account for only a small share of the sample and were retained without collapsing.

The decision is to retain all five demographic variables as provisional candidate variables. Encoding, age transformation and any county aggregation remain deferred to the later feature preparation stage.

No observations were removed, no missing values were introduced and no transformations or encodings were applied.

### Shock exposure investigation

The 12 binary shock indicators and `main_shock_type` were investigated using the 1,552 drought exposed households in the FinAccess 2024 analytical sample.

All investigated variables were complete. The drought indicator was positive for every household and therefore had no variation within the analytical sample. It will be excluded from ordinary predictors while being retained in the underlying dataset for traceability.

The remaining 11 indicators captured additional household shocks. Livestock and crop failure due to pests and diseases was the most frequently reported, followed by high rainfall or flooding and major sickness or health problems. Approximately 80.35% of households reported at least one additional shock beyond drought.

Multiple shock exposure was common. The indicators were therefore treated as overlapping measures rather than mutually exclusive categories. Three additional shock indicators had positive frequencies below 5%. These were retained provisionally because low frequency alone does not establish that a variable lacks analytical value.

The `main_shock_type` variable was also complete and represented 12 categories. Every recorded category mapped to a positive corresponding shock indicator. However, this consistency check only establishes agreement under the specified mapping. The main shock variable represents the principal reported shock, whereas the binary indicators capture all reported shocks.

The decision is to retain the 11 additional shock indicators and `main_shock_type` as provisional candidates. A derived count of additional shocks will remain an audit measure. The usefulness of the indicators and main shock representation will be assessed separately during later feature assessment.

The drought indicator will not be used as an ordinary predictor because it is constant in the drought exposed sample. No categories were collapsed, no observations were removed and no transformations or imputations were applied.

The temporal relationship between shock exposure and coping responses remains an issue for the final predictor leakage audit. The available information must be checked to establish whether the shock variables would be available before the coping behaviour being predicted.

### Financial access investigation

**Status: Investigation completed. Preparation decisions provisionally approved.**

Eleven financial access variables were examined for completeness, distributions, category structure, redundancy and possible overlap with coping actions. All eleven variables had complete observations across the 1,552 households.

The variables represented three broad constructs. These were agricultural trade credit questions, derived financial usage flags and broader financial service usage measures. The investigation confirmed that some derived variables reproduce or overlap with other financial measures.

In particular, `used_buyer_credit` was an exact recode of `buyer_supplier_credit`. `used_formal_credit` was a documented composite involving `loan_usage`, `loan_formal` and `formal_agri_borrowing`. The remaining derived flags also matched their documented source indicators.

The investigation identified possible conceptual overlap between borrowing and savings usage measures and the corresponding coping actions. The timing of these financial measures relative to drought exposure and coping responses remains unresolved.

All eleven variables will be retained in the prepared dataset. The exact duplicate and overlapping composite measures will be flagged for later feature assessment. No encoding, transformation or removal will be performed at this stage.

Financial access variables will not be included in the agricultural segmentation inputs because the segmentation is defined using agricultural portfolio composition, land size and land ownership. They remain provisional candidates for the supervised coping classification, subject to the predictor leakage audit.

**Outstanding issues**

- Establish the temporal relationship between financial service usage and coping responses.
- Assess redundancy among financial predictors before finalising the classification feature set.
- Consider the low prevalence of digital credit use and insurance use when designing model validation.

### Temporal availability review

**Status: Questionnaire review completed. Predictor suitability decisions remain conditional.**

The temporal review examined 23 variables previously flagged during the predictor leakage audit. Questionnaire evidence was recovered for the R2 shock and coping module and for the agricultural credit questions.

The R2 questionnaire establishes a retrospective past 12-month reference period for shock exposure. It then asks respondents to identify the main shock affecting household income or expenses before asking what they did to manage that shock. This establishes a documented question sequence but does not independently establish the chronological order of every shock and coping action.

The 11 additional shock indicators and `main_shock_type` will therefore remain in the prepared dataset. They are not classified as confirmed leakage. Their use in supervised classification will be examined through an extended model, with the same-window retrospective nature of the information clearly acknowledged.

The agricultural credit questions also refer to the past 12 months. Their reference period overlaps the R2 coping window, so their availability before the coping response cannot be established. They remain excluded from the provisional primary classification predictor set.

The timing of broader financial service usage measures could not be established from the recovered questionnaire evidence. These variables remain flagged and will not be included in the primary classification predictor set without further justification.

Confirmed coping action indicators, target variables and constant drought exposure variables remain excluded from predictor matrices. Confirmed redundant financial measures will be retained in the prepared dataset but flagged for feature assessment.

The agricultural segmentation design remains based on agricultural portfolio composition, land size and land ownership. Financial and demographic measures will not be added to the segmentation inputs.

**Outstanding issues**

- Retain the distinction between primary household characteristic models and extended same-window shock models.
- Do not describe the classification framework as prospective or operationally validated based on this cross-sectional retrospective evidence.
- Land size unit and education encoding decisions are recorded below under the missingness and encoding audit.

### Land size unit reconciliation

**Status: Decision recorded. Standardised unit is acres.**

Data Understanding had recommended converting conventional land measurements into a common unit, preferably hectares. Data Preparation later created and audited `land_size_acres` because acres dominate the reported units in the drought-exposed sample.

The reconciled decision is to retain acres as the standardised land size unit for feature preparation. `land_size_acres` will be the numerical land measure carried forward. Original `land_size` and `land_size_unit` values remain preserved. Special codes 98 and 99 and community land observations without convertible acreage remain missing in the standardised measure.

Education code 1 was previously resolved as None.

### Missingness and encoding audit decisions

**Status: Preparation decisions recorded after Script 08. Encoding and imputation not yet applied.**

Script 08 audited missingness, coding consistency, categorical frequencies, continuous summaries and target representation for the segmentation, primary classification and extended classification inventories. The following decisions were recorded for feature preparation:

| Variable or feature group | Decision |
|---|---|
| `land_size_acres` | Retain as the standardised land measure |
| `land_size_unit` | Exclude as a predictor after resolving the single missing-unit observation |
| `education` | Treat as ordered categorical |
| Agricultural binary indicators | Retain |
| Land ownership indicators | Retain separately |
| `n_crop_types` and `n_livestock_types` | Retain provisionally |
| `shannon_diversity` | Reserve for sensitivity analysis |
| `county` | Retain provisionally, no category collapsing |
| `age_years` | Retain as numeric |
| `has_other_agri` | Retain despite low prevalence |
| Four-category target | Retain |
| Three-category target | Retain as an alternative |
| Missing land size | Do not impute at this stage |
| Financial variables | Continue to respect the temporal restrictions from Script 07 |
| Shock variables | Keep out of primary classification; consider in extended classification |

Additional notes from the audit:

- The single missing `land_size_unit` observation (n = 1) does not change the decision to exclude the unit code as a predictor once acres are used.
- Missing `land_size_acres` remains concentrated in community land (182) and special response codes 98/99 (16). These values stay missing for now; no imputation is applied.
- `sex` and `urban_rural` remain binary coded predictors for later encoding in feature preparation.
- Financial variables remain in the prepared dataset but stay outside primary classification inputs under the Script 07 temporal restrictions.
- Shock indicators and `main_shock_type` remain extended-classification candidates only, with same-window retrospective status acknowledged.

**Outstanding before Script 09 / feature preparation**

- Apply the encoding schemes consistent with these decisions (no scaling; no land missingness indicator; no imputation).
- Keep `shannon_diversity` available for sensitivity runs rather than the default feature matrices.

### Land and inventory reconciliation (Script 08b)

**Status: Reconciliation completed. Cleared for Script 09.**

The single missing `land_size_unit` observation was inspected. Acres cannot be derived for that household. Exclusion of `land_size_unit` as a predictor remains valid and the unit is not imputed.

Missing `land_size_acres` (n = 198) was re-checked. Using the mutually exclusive primary-reason rule from Script 08 (community land first), counts remain 182 community land and 16 special response codes 98/99. A detailed crosstab shows that 171 community-land households also carry size codes 98/99 as placeholders; they stay classified as community land. The 16 special-code cases are the non-community remainder. Conversion remains limited to unit codes 1–4.

Default inventories in config were confirmed: segmentation 18, primary 23, extended 35. `land_size_unit` is excluded; `shannon_diversity` is reserved for sensitivity analysis only.

### Feature preparation (Script 09)

**Status: Deterministic feature sets prepared. No imputation, scaling or data split.**

Script 09 produced separate prepared matrices for agricultural/land segmentation, primary coping classification and extended coping classification, plus a Shannon sensitivity sidecar.

Preparation actions applied:

- Derived `land_size_acres` from `land_size` and `land_size_unit`; retained missing values; did not create a missingness indicator.
- Remapped `sex` to `sex_bin` (Male 1→0, Female 2→1) and `urban_rural` to `urban_rural_bin` (Rural 1→0, Urban 2→1); originals retained.
- Kept `education` as ordered integer codes 1–4.
- One-hot encoded `county` and `main_shock_type` using fixed codebook categories (`label_set=county` and `label_set=R2B`).
- Excluded `land_size_unit` from default predictors; placed `shannon_diversity` only in the sensitivity sidecar.
- Excluded financial-access variables from default classification matrices under Script 07 temporal restrictions.
- Excluded shock indicators and `main_shock_type` one-hots from the primary classification matrix; included them in the extended matrix only.

Prepared feature counts: segmentation 18; primary 69; extended 92; sensitivity 1.

Land-size missingness retained in prepared matrices:

```
                                     topic   n   pct                                                                                                                                                                               note
             land_size_acres_missing_total 198 12.76 Missing values retained. No imputation and no missingness indicator in Script 09. Not all modelling algorithms can consume NaNs directly; handle later within training folds only.
             primary_reason_community_land 182 11.73                                                                                                                 Unit code 5; includes rows that also carry size 98/99 placeholders
primary_reason_special_98_99_non_community  16  1.03                                                                                                                                                Size 98/99 with unit not equal to 5
                             other_missing   0  0.00                                                                                                                            Residual missing after community/special classification
```

Original source columns and target variables were preserved in `outputs/prepared/09_prepared_analysis_frame.parquet` and `outputs/prepared/09_y_targets.parquet`. The external FinAccess modelling frame was not modified.

**Future fitted preprocessing rule**

Any future imputation, scaling, feature selection based on learned statistics, or other data-dependent transformations must be fitted using training data within each validation fold and then applied to the corresponding validation data. Script 09 itself does not create a train/validation split and does not fit encoders, imputers or scalers.

### Final Preparation Validation

The final preparation validation was completed using `10_final_preparation_validation.py`. All 11 validation checks passed.

The checks confirmed that all prepared feature matrices contain the same 1,552 respondents in identical order. Feature membership matched the approved inventories, and no duplicate columns or unintended dual representations were identified.

The validation also confirmed that the expected 198 missing values in `land_size_acres` were preserved, while no unexpected missingness was introduced. Binary and ordinal representations were consistent with the approved coding decisions. One-hot encoded variables covered the fixed codebook categories, with valid row sums and successful decoding checks.

The prepared targets matched the original source data, including the 75 observations in the formal coping category. Independently re-derived transformations also matched the prepared matrices.

The final prepared feature sets are:

- Segmentation: 18 features
- Primary classification: 69 features
- Extended classification: 92 features
- Shannon diversity sensitivity set: 1 feature

No imputation, scaling, train/validation splitting or model fitting was performed during preparation.

**Outcome: All 11 validation checks passed. The Data Preparation phase is considered complete.**

The following considerations will be carried into Modelling:

1. Missing `land_size_acres` values require appropriate handling within training folds.
2. The relatively small formal coping class requires stratified validation and careful evaluation. The collapsed three-class target will also be considered.
3. The extended classification analysis uses retrospective shock information from the same reference period as the coping module and must be interpreted as a retrospective analysis rather than prospective prediction.

Validation report: `outputs/reports/10_final_preparation_validation.md`

Modelling has not yet started.
