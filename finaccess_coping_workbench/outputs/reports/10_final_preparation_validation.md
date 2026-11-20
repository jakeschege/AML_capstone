# Final Preparation Validation Report (Script 10)

Generated: 2026-11-18 17:19 UTC

## Overall result: PASS

Checks: 11 pass, 0 fail, 0 warn.

## Checklist

```
check_id                                 check_name status                                                                                 detail
      01        row_counts_and_respondent_alignment   pass                       n=1552; identical ordered respondent_id across matrices/y/source
      02      duplicate_columns_and_representations   pass                                                   no duplicate names or dual encodings
      03                             missing_values   pass                          only expected 198 missing land_size_acres in default matrices
      04                         feature_membership   pass                                       matrices match feature lists and expected counts
      05                       inventory_separation   pass      segmentation ⊂ primary ⊂ extended; shocks extended-only; shannon sensitivity-only
      06       leakage_and_non_predictor_exclusions   pass no targets/coping/weights/finance/sample-flags as predictors; id is alignment key only
      07  explicit_exclusions_land_unit_and_shannon   pass                                   land_size_unit excluded; shannon only in sensitivity
      08         binary_and_ordinal_representations   pass                     sex_bin/urban_rural_bin/education and binary indicators consistent
      09                   onehot_codebook_coverage   pass             fixed codebook columns present; row sums equal 1; decode matches originals
      10                         target_consistency   pass                           targets match source/prepared; formal=75; no missing targets
      11 reproducibility_from_documented_transforms   pass                                             re-derived columns match prepared matrices
```

## Missingness summary

```
                 matrix          column  n_missing  expected
           segmentation land_size_acres        198      True
 primary_classification land_size_acres        198      True
extended_classification land_size_acres        198      True
```

## Unexpected / duplicate flags

```
None
```

## Target consistency

```
                   target        class_label  n_households   pct
          coping_category           personal           615 39.63
          coping_category           informal           447 28.80
          coping_category        did_nothing           415 26.74
          coping_category             formal            75  4.83
coping_category_collapsed           personal           615 39.63
coping_category_collapsed formal_or_informal           522 33.63
coping_category_collapsed        did_nothing           415 26.74
          coping_category       formal_check            75  4.83
```

## Modelling carry-forward notes

These are not validation failures. They remain for the Modelling phase:

1. **Land size missingness** — 198 missing `land_size_acres` values must be handled within each training fold. Compare defensible approaches; do not automatically treat missing acreage as zero. Any missingness indicator is a separate modelling decision.
2. **Formal coping class** — four-class target retains 75 formal observations. Preserve the class and assess whether stratified validation maintains adequate representation in every fold. The three-class target remains an important alternative.
3. **Extended classification** — shock variables are retrospective and share the R2 reference period with coping. Interpret extended-matrix results as extended retrospective classification, not prospective prediction from pre-coping information.

## Review gate

Data Preparation is substantially complete pending review of this report. **Do not begin modelling until this validation report has been reviewed and any failures resolved.**

## Output files

- `outputs/tables/10_validation_checklist.csv`
- `outputs/tables/10_matrix_column_inventory.csv`
- `outputs/tables/10_unexpected_or_duplicate_flags.csv`
- `outputs/tables/10_missingness_summary.csv`
- `outputs/tables/10_target_consistency.csv`
