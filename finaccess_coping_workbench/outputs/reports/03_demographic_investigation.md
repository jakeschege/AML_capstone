# Demographic Investigation Report

Generated: 2026-11-18 06:54 UTC

## What was investigated

Demographic variables for the drought-exposed analytical sample (`n = 1552`):

- `age_years`
- `sex`
- `education`
- `urban_rural`
- `county`

Encoding was not performed. No observations were removed and no values were imputed or transformed.

## Observed findings

### Inventory

```
   variable dtype    n  missing  missing_pct  n_unique  min   max
  age_years int64 1552        0          0.0        77 16.0 101.0
        sex int64 1552        0          0.0         2  1.0   2.0
  education int64 1552        0          0.0         4  1.0   4.0
urban_rural int64 1552        0          0.0         2  1.0   2.0
     county int64 1552        0          0.0        47  1.0  47.0
```

Missingness: none of the five demographic variables has missing values in this sample.

### Age

```
 variable    n  missing      mean  median      std  skewness  min  p01  p05  p25  p50  p75  p95   p99   max  iqr  iqr_lower_fence  iqr_upper_fence  n_below_iqr_fence  n_above_iqr_fence  n_below_adult_floor  n_at_or_above_high_review
age_years 1552        0 43.798325    41.0 16.05547  0.497885 16.0 18.0 21.0 31.0 41.0 55.0 72.0 83.49 101.0 24.0             -5.0             91.0                  0                  5                    0                          7
```

- Skewness: 0.498
- Range: 16 to 101 years
- Households below adult floor (16): 0
- Households at or above high-age review threshold (90): 7
- Households above IQR upper fence: 5
- Distinct flagged age values written for review: 5

Age is mildly right-skewed. Extreme high ages exist but were not removed. They remain candidates for later sensitivity checks rather than automatic exclusion.

### Sex

```
variable  code  label  n_households   pct
     sex     1   Male           690 44.46
     sex     2 Female           862 55.54
```

### Education

```
 variable  code                       label  n_households   pct
education     1 [label missing in codebook]           382 24.61
education     2                     Primary           651 41.95
education     3                   Secondary           389 25.06
education     4                    Tertiary           130  8.38
```

Ordinal assessment:

```
observed_codes codebook_codes  code_5_other_present unexpected_codes_present  code_1_label_missing_in_codebook  provisional_ordinal_candidate                                                                                                                                                                                     assessment_note
    1, 2, 3, 4  1, 2, 3, 4, 5                 False                                                       True                           True Observed codes are 1-4 only. If code 1 is the lowest schooling level, the observed categories form an ordered schooling ladder. Code 5 Other is absent from this sample. Encoding remains deferred.
```

- Observed codes in this sample: 1, 2, 3, 4
- Code 5 Other present: False
- Provisional ordinal candidate before encoding: True
- Code 1 has a missing label in the codebook file and needs confirmation against the questionnaire before final ordinal treatment.

### Urban / rural

```
   variable  code label  n_households   pct
urban_rural     1 Rural          1324 85.31
urban_rural     2 Urban           228 14.69
```

The sample is predominantly rural.

### County

- Counties represented: 47
- Households per county range from 1 to 95

Small-sample summary:

```
threshold  n_counties_below  pct_of_counties_below  n_households_in_those_counties  pct_households_in_those_counties  min_county_n  max_county_n  n_counties_represented
    n < 5                 5                  10.64                              14                              0.90             1            95                      47
   n < 10                11                  23.40                              53                              3.41             1            95                      47
```

Sparse counties are documented for later modelling awareness. Counties were not collapsed or dropped in this step.

## Interpretation

The five demographic fields are complete in the drought-exposed sample and use survey codes that can be interpreted with the FinAccess codebook.

Education codes 1 to 4 are consistent with an ordered schooling ladder if code 1 is the lowest schooling category. That interpretation remains provisional until the blank codebook label for code 1 is confirmed. Sex and urban_rural are binary coded categories. County is a high-cardinality geographic identifier with uneven sample sizes.

## Limitations

- Codebook labels were joined for interpretation only and were not written back into the analytical dataset.
- Education code 1 label text is missing in `value_labels.csv`.
- Age extremes were flagged using simple descriptive rules, not clinical or survey-design exclusion rules.
- No relationship with the coping target was used to select or discard demographics.

## Recommended preparation decisions for review

1. Retain `age_years`, `sex`, `education`, `urban_rural` and `county` as provisional candidate variables.
2. Do not encode these variables yet. Encoding belongs in the later feature preparation stage.
3. Do not drop or winsorise high ages solely on the IQR screen.
4. Do not collapse sparse counties yet. Record the small-sample distribution for later classification design.
5. Confirm the meaning of education code 1 before finalising ordinal versus nominal treatment.

## Output files

- `outputs/tables/03_demographic_inventory.csv`
- `outputs/tables/03_age_descriptives.csv`
- `outputs/tables/03_age_extreme_flags.csv`
- `outputs/tables/03_sex_frequencies.csv`
- `outputs/tables/03_education_frequencies.csv`
- `outputs/tables/03_education_ordinal_assessment.csv`
- `outputs/tables/03_urban_rural_frequencies.csv`
- `outputs/tables/03_county_frequencies.csv`
- `outputs/tables/03_county_small_sample_summary.csv`
- `outputs/tables/03_age_by_sex.csv`
- `outputs/tables/03_age_by_urban_rural.csv`
- `outputs/tables/03_education_by_urban_rural.csv`
