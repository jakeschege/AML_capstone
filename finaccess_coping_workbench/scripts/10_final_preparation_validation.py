#!/usr/bin/env python3
"""Final validation of Script 09 prepared feature matrices.

Validation only: no encoding changes, imputation, scaling or modelling.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    ADDITIONAL_SHOCK_VARS,
    AGRICULTURAL_HAS_VARS,
    COPING_ACTION_VARS,
    EXPECTED_DROUGHT_N,
    FINANCIAL_ACCESS_VARS,
    ID_VARS,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    LOGS_DIR,
    PREPARED_DIR,
    REPORTS_DIR,
    SAMPLE_FLAG_VARS,
    SEX_CODE_TO_BIN,
    TABLES_DIR,
    TARGET_VARS,
    URBAN_RURAL_CODE_TO_BIN,
    WEIGHT_VARS,
    add_land_size_acres,
    ensure_output_dirs,
    load_analysis_2024,
    load_codebook_categories,
)

EXPECTED_N_FEATURES = {
    "segmentation": 18,
    "primary_classification": 69,
    "extended_classification": 92,
    "sensitivity": 1,
}

MATRIX_FILES = {
    "segmentation": "09_X_segmentation.parquet",
    "primary_classification": "09_X_primary_classification.parquet",
    "extended_classification": "09_X_extended_classification.parquet",
    "sensitivity": "09_X_sensitivity_shannon.parquet",
}


def load_script09_helpers():
    path = WORKBENCH_ROOT / "scripts" / "09_feature_preparation.py"
    spec = importlib.util.spec_from_file_location("feature_prep_09", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Script 09 helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predictor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_VARS]


def add_check(
    rows: list[dict],
    check_id: str,
    name: str,
    status: str,
    detail: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "check_name": name,
            "status": status,
            "detail": detail,
        }
    )


def load_matrices() -> dict[str, pd.DataFrame]:
    out = {}
    for name, filename in MATRIX_FILES.items():
        path = PREPARED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared matrix: {path}")
        out[name] = pd.read_parquet(path)
    return out


def check_row_alignment(
    matrices: dict[str, pd.DataFrame],
    y: pd.DataFrame,
    source_ids: pd.Series,
    rows: list[dict],
) -> None:
    issues = []
    for name, df in {**matrices, "targets": y}.items():
        if len(df) != EXPECTED_DROUGHT_N:
            issues.append(f"{name}_n={len(df)}")
        if "respondent_id" not in df.columns:
            issues.append(f"{name}_missing_respondent_id")
    ids = {name: df["respondent_id"].astype(str).tolist() for name, df in matrices.items()}
    ids["targets"] = y["respondent_id"].astype(str).tolist()
    ids["source"] = source_ids.astype(str).tolist()
    ref = ids["segmentation"]
    for name, seq in ids.items():
        if seq != ref:
            issues.append(f"id_order_mismatch:{name}")
    if len(set(ref)) != len(ref):
        issues.append("duplicate_respondent_ids")
    status = "pass" if not issues else "fail"
    detail = (
        f"n={EXPECTED_DROUGHT_N}; identical ordered respondent_id across matrices/y/source"
        if status == "pass"
        else "; ".join(issues)
    )
    add_check(rows, "01", "row_counts_and_respondent_alignment", status, detail)


def check_duplicates(
    matrices: dict[str, pd.DataFrame],
    rows: list[dict],
    flags: list[dict],
) -> None:
    issues = []
    for name, df in matrices.items():
        preds = predictor_cols(df)
        if len(preds) != len(set(preds)):
            issues.append(f"{name}:duplicate_column_names")
            flags.append(
                {
                    "matrix": name,
                    "flag_type": "duplicate_column_names",
                    "detail": str(pd.Series(preds)[pd.Series(preds).duplicated()].tolist()),
                }
            )
        if "sex" in preds and "sex_bin" in preds:
            issues.append(f"{name}:sex_and_sex_bin")
            flags.append(
                {
                    "matrix": name,
                    "flag_type": "duplicate_representation",
                    "detail": "sex and sex_bin both present",
                }
            )
        if "urban_rural" in preds and "urban_rural_bin" in preds:
            issues.append(f"{name}:urban_rural_and_bin")
            flags.append(
                {
                    "matrix": name,
                    "flag_type": "duplicate_representation",
                    "detail": "urban_rural and urban_rural_bin both present",
                }
            )
        county_oh = [c for c in preds if re.match(r"^county_\d+_", c)]
        if "county" in preds and county_oh:
            issues.append(f"{name}:raw_county_with_onehots")
            flags.append(
                {
                    "matrix": name,
                    "flag_type": "duplicate_representation",
                    "detail": "raw county with county one-hots",
                }
            )
        shock_oh = [c for c in preds if re.match(r"^main_shock_type_\d+_", c)]
        if "main_shock_type" in preds and shock_oh:
            issues.append(f"{name}:raw_main_shock_with_onehots")
            flags.append(
                {
                    "matrix": name,
                    "flag_type": "duplicate_representation",
                    "detail": "raw main_shock_type with one-hots",
                }
            )
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "02",
        "duplicate_columns_and_representations",
        status,
        "no duplicate names or dual encodings" if status == "pass" else "; ".join(issues),
    )


def check_missingness(
    matrices: dict[str, pd.DataFrame],
    rows: list[dict],
    missing_summary: list[dict],
) -> None:
    issues = []
    for name, df in matrices.items():
        preds = predictor_cols(df)
        for col in preds:
            n_miss = int(df[col].isna().sum())
            if n_miss == 0:
                continue
            expected = name != "sensitivity" and col == LAND_SIZE_STANDARDISED and n_miss == 198
            missing_summary.append(
                {
                    "matrix": name,
                    "column": col,
                    "n_missing": n_miss,
                    "expected": expected,
                }
            )
            if not expected:
                issues.append(f"{name}.{col}:n_missing={n_miss}")
        if name != "sensitivity":
            if LAND_SIZE_STANDARDISED not in df.columns:
                issues.append(f"{name}:missing_land_size_acres_column")
            elif int(df[LAND_SIZE_STANDARDISED].isna().sum()) != 198:
                issues.append(
                    f"{name}:land_size_acres_missing="
                    f"{int(df[LAND_SIZE_STANDARDISED].isna().sum())}"
                )
        else:
            if int(df[preds].isna().sum().sum()) != 0:
                issues.append("sensitivity:unexpected_missing")
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "03",
        "missing_values",
        status,
        "only expected 198 missing land_size_acres in default matrices"
        if status == "pass"
        else "; ".join(issues),
    )


def check_feature_membership(
    matrices: dict[str, pd.DataFrame],
    expected_lists: dict[str, list[str]],
    rows: list[dict],
    flags: list[dict],
) -> None:
    issues = []
    for name, df in matrices.items():
        actual = predictor_cols(df)
        listed = expected_lists[name]
        if actual != listed:
            missing = sorted(set(listed) - set(actual))
            extra = sorted(set(actual) - set(listed))
            issues.append(f"{name}:mismatch")
            if missing:
                flags.append(
                    {
                        "matrix": name,
                        "flag_type": "missing_expected_feature",
                        "detail": ";".join(missing),
                    }
                )
            if extra:
                flags.append(
                    {
                        "matrix": name,
                        "flag_type": "unexpected_feature",
                        "detail": ";".join(extra),
                    }
                )
        if len(actual) != EXPECTED_N_FEATURES[name]:
            issues.append(f"{name}:n={len(actual)} expected={EXPECTED_N_FEATURES[name]}")
        # Compare to Script 09 feature list CSV
        list_path = TABLES_DIR / f"09_feature_list_{name}.csv"
        if list_path.exists():
            csv_cols = pd.read_csv(list_path)["column"].astype(str).tolist()
            if actual != csv_cols:
                issues.append(f"{name}:differs_from_09_feature_list_csv")
                flags.append(
                    {
                        "matrix": name,
                        "flag_type": "feature_list_csv_mismatch",
                        "detail": f"matrix_n={len(actual)} csv_n={len(csv_cols)}",
                    }
                )
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "04",
        "feature_membership",
        status,
        "matrices match feature lists and expected counts"
        if status == "pass"
        else "; ".join(issues),
    )


def check_separation(
    matrices: dict[str, pd.DataFrame],
    rows: list[dict],
) -> None:
    seg = set(predictor_cols(matrices["segmentation"]))
    pri = set(predictor_cols(matrices["primary_classification"]))
    ext = set(predictor_cols(matrices["extended_classification"]))
    sens = set(predictor_cols(matrices["sensitivity"]))
    issues = []
    if not seg.issubset(pri):
        issues.append(f"seg_not_subset_primary:{sorted(seg - pri)}")
    if not pri.issubset(ext):
        issues.append(f"primary_not_subset_extended:{sorted(pri - ext)[:10]}")
    shock_in_pri = sorted(set(ADDITIONAL_SHOCK_VARS) & pri)
    shock_oh_pri = sorted(c for c in pri if re.match(r"^main_shock_type_\d+_", c))
    if shock_in_pri or shock_oh_pri:
        issues.append(f"shocks_in_primary:{shock_in_pri + shock_oh_pri}")
    shock_in_seg = sorted(set(ADDITIONAL_SHOCK_VARS) & seg)
    if shock_in_seg:
        issues.append(f"shocks_in_segmentation:{shock_in_seg}")
    for name, cols in [
        ("segmentation", seg),
        ("primary_classification", pri),
        ("extended_classification", ext),
    ]:
        if "shannon_diversity" in cols:
            issues.append(f"shannon_in_{name}")
    if sens != {"shannon_diversity"}:
        issues.append(f"sensitivity_unexpected:{sorted(sens)}")
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "05",
        "inventory_separation",
        status,
        "segmentation ⊂ primary ⊂ extended; shocks extended-only; shannon sensitivity-only"
        if status == "pass"
        else "; ".join(issues),
    )


def check_leakage_exclusions(
    matrices: dict[str, pd.DataFrame],
    rows: list[dict],
    flags: list[dict],
) -> None:
    banned = set(
        TARGET_VARS
        + COPING_ACTION_VARS
        + WEIGHT_VARS
        + FINANCIAL_ACCESS_VARS
        + SAMPLE_FLAG_VARS
    )
    issues = []
    for name, df in matrices.items():
        preds = set(predictor_cols(df))
        hit = sorted(preds & banned)
        if hit:
            issues.append(f"{name}:{hit}")
            flags.append(
                {
                    "matrix": name,
                    "flag_type": "forbidden_predictor",
                    "detail": ";".join(hit),
                }
            )
        # respondent_id must not be treated as predictor; presence as key is OK
        if "respondent_id" in predictor_cols(df):
            issues.append(f"{name}:respondent_id_listed_as_predictor")
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "06",
        "leakage_and_non_predictor_exclusions",
        status,
        "no targets/coping/weights/finance/sample-flags as predictors; id is alignment key only"
        if status == "pass"
        else "; ".join(issues),
    )


def check_explicit_exclusions(
    matrices: dict[str, pd.DataFrame],
    rows: list[dict],
) -> None:
    issues = []
    for name in ["segmentation", "primary_classification", "extended_classification"]:
        preds = set(predictor_cols(matrices[name]))
        if "land_size_unit" in preds:
            issues.append(f"{name}:land_size_unit")
        if "shannon_diversity" in preds:
            issues.append(f"{name}:shannon_diversity")
    sens = set(predictor_cols(matrices["sensitivity"]))
    if sens != {"shannon_diversity"}:
        issues.append(f"sensitivity_contents:{sorted(sens)}")
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "07",
        "explicit_exclusions_land_unit_and_shannon",
        status,
        "land_size_unit excluded; shannon only in sensitivity"
        if status == "pass"
        else "; ".join(issues),
    )


def check_binary_ordinal(
    matrices: dict[str, pd.DataFrame],
    prepared: pd.DataFrame,
    rows: list[dict],
) -> None:
    issues = []
    pri = matrices["primary_classification"]
    for col, allowed in [("sex_bin", {0, 1}), ("urban_rural_bin", {0, 1}), ("education", {1, 2, 3, 4})]:
        vals = set(pd.to_numeric(pri[col], errors="coerce").dropna().unique().tolist())
        if not vals.issubset(allowed):
            issues.append(f"{col}_values={sorted(vals)}")
    # Remap consistency vs originals
    sex_ok = (
        pd.to_numeric(prepared["sex"], errors="coerce").map(SEX_CODE_TO_BIN)
        == pd.to_numeric(prepared["sex_bin"], errors="coerce")
    ).all()
    urban_ok = (
        pd.to_numeric(prepared["urban_rural"], errors="coerce").map(URBAN_RURAL_CODE_TO_BIN)
        == pd.to_numeric(prepared["urban_rural_bin"], errors="coerce")
    ).all()
    if not sex_ok:
        issues.append("sex_bin_mismatch_vs_sex")
    if not urban_ok:
        issues.append("urban_rural_bin_mismatch_vs_urban_rural")
    # Binary indicators
    binary_cols = (
        list(AGRICULTURAL_HAS_VARS)
        + list(LAND_OWNERSHIP_VARS)
        + list(ADDITIONAL_SHOCK_VARS)
    )
    ext = matrices["extended_classification"]
    for col in binary_cols:
        if col not in ext.columns:
            continue
        vals = set(pd.to_numeric(ext[col], errors="coerce").dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            issues.append(f"{col}_not_binary:{sorted(vals)}")
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "08",
        "binary_and_ordinal_representations",
        status,
        "sex_bin/urban_rural_bin/education and binary indicators consistent"
        if status == "pass"
        else "; ".join(issues),
    )


def check_onehot_coverage(
    matrices: dict[str, pd.DataFrame],
    prepared: pd.DataFrame,
    rows: list[dict],
) -> None:
    issues = []
    onehot_dict = pd.read_csv(TABLES_DIR / "09_onehot_category_dictionary.csv")
    county_cats = load_codebook_categories("county")
    shock_cats = load_codebook_categories("R2B")

    pri = matrices["primary_classification"]
    ext = matrices["extended_classification"]
    county_cols = [c for c in predictor_cols(pri) if re.match(r"^county_\d+_", c)]
    shock_cols = [c for c in predictor_cols(ext) if re.match(r"^main_shock_type_\d+_", c)]

    expected_county = [f"county_{code}_{_slug(label)}" for code, label in county_cats]
    expected_shock = [
        f"main_shock_type_{code}_{_slug(label)}" for code, label in shock_cats
    ]
    # Prefer dictionary file names if present
    dict_county = onehot_dict.loc[
        onehot_dict["source_variable"] == "county", "output_column"
    ].tolist()
    dict_shock = onehot_dict.loc[
        onehot_dict["source_variable"] == "main_shock_type", "output_column"
    ].tolist()
    if county_cols != dict_county:
        issues.append("county_columns_differ_from_dictionary")
    if shock_cols != dict_shock:
        issues.append("main_shock_columns_differ_from_dictionary")
    if len(county_cols) != len(county_cats):
        issues.append(f"county_onehot_n={len(county_cols)} codebook_n={len(county_cats)}")
    if len(shock_cols) != len(shock_cats):
        issues.append(f"shock_onehot_n={len(shock_cols)} codebook_n={len(shock_cats)}")

    county_sum = pri[county_cols].sum(axis=1)
    if not (county_sum == 1).all():
        issues.append(
            f"county_row_sum_not_1:min={county_sum.min()} max={county_sum.max()} "
            f"n_bad={int((county_sum != 1).sum())}"
        )
    shock_sum = ext[shock_cols].sum(axis=1)
    if not (shock_sum == 1).all():
        issues.append(
            f"main_shock_row_sum_not_1:min={shock_sum.min()} max={shock_sum.max()} "
            f"n_bad={int((shock_sum != 1).sum())}"
        )
    # Prepared frame originals should match one-hot reconstruction
    county_from_oh = pri[county_cols].idxmax(axis=1).str.extract(r"^county_(\d+)_")[0].astype(int)
    if not (
        county_from_oh.reset_index(drop=True)
        == pd.to_numeric(prepared["county"], errors="coerce").reset_index(drop=True)
    ).all():
        issues.append("county_onehot_decode_mismatch")
    shock_from_oh = (
        ext[shock_cols].idxmax(axis=1).str.extract(r"^main_shock_type_(\d+)_")[0].astype(int)
    )
    if not (
        shock_from_oh.reset_index(drop=True)
        == pd.to_numeric(prepared["main_shock_type"], errors="coerce").reset_index(drop=True)
    ).all():
        issues.append("main_shock_onehot_decode_mismatch")

    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "09",
        "onehot_codebook_coverage",
        status,
        "fixed codebook columns present; row sums equal 1; decode matches originals"
        if status == "pass"
        else "; ".join(issues),
    )


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "unknown"


def check_target_consistency(
    y: pd.DataFrame,
    prepared: pd.DataFrame,
    source: pd.DataFrame,
    rows: list[dict],
    target_table: list[dict],
) -> None:
    issues = []
    for col in ["coping_category", "coping_category_collapsed"]:
        if col not in y.columns:
            issues.append(f"y_missing_{col}")
            continue
        n_miss = int(y[col].isna().sum())
        if n_miss:
            issues.append(f"{col}_missing={n_miss}")
        match_prepared = y[col].astype(str).tolist() == prepared[col].astype(str).tolist()
        match_source = y[col].astype(str).tolist() == source[col].astype(str).tolist()
        if not match_prepared:
            issues.append(f"{col}_differs_from_prepared_frame")
        if not match_source:
            issues.append(f"{col}_differs_from_source_analysis")
        vc = y[col].value_counts(dropna=False)
        for label, n in vc.items():
            target_table.append(
                {
                    "target": col,
                    "class_label": label if pd.notna(label) else "[missing]",
                    "n_households": int(n),
                    "pct": round(100.0 * n / len(y), 2),
                }
            )
    formal_n = int((y["coping_category"] == "formal").sum()) if "coping_category" in y.columns else -1
    if formal_n != 75:
        issues.append(f"formal_n={formal_n} expected=75")
    else:
        target_table.append(
            {
                "target": "coping_category",
                "class_label": "formal_check",
                "n_households": formal_n,
                "pct": round(100.0 * formal_n / len(y), 2),
            }
        )
    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "10",
        "target_consistency",
        status,
        "targets match source/prepared; formal=75; no missing targets"
        if status == "pass"
        else "; ".join(issues),
    )


def check_reproducibility(
    matrices: dict[str, pd.DataFrame],
    prepared: pd.DataFrame,
    source: pd.DataFrame,
    rows: list[dict],
) -> None:
    issues = []
    helpers = load_script09_helpers()
    rebuilt, _, _, _ = helpers.prepare_frame(source)

    # Compare key derived columns in prepared frame
    for col in [LAND_SIZE_STANDARDISED, "sex_bin", "urban_rural_bin"]:
        a = pd.to_numeric(prepared[col], errors="coerce")
        b = pd.to_numeric(rebuilt[col], errors="coerce")
        if not a.equals(b) and not np.isclose(a.fillna(-999), b.fillna(-999)).all():
            issues.append(f"prepared_frame_{col}_not_reproducible")

    # Compare matrix columns to rebuilt
    lists = helpers.feature_lists(rebuilt)
    for name, expected_cols in [
        ("segmentation", lists["segmentation"]),
        ("primary_classification", lists["primary_classification"]),
        ("extended_classification", lists["extended_classification"]),
        ("sensitivity", lists["sensitivity"]),
    ]:
        mat = matrices[name]
        for col in expected_cols:
            left = mat[col]
            right = rebuilt[col]
            if left.dtype.kind in "fc" or right.dtype.kind in "fc":
                ok = np.isclose(
                    pd.to_numeric(left, errors="coerce").fillna(-999999),
                    pd.to_numeric(right, errors="coerce").fillna(-999999),
                ).all()
            else:
                ok = left.reset_index(drop=True).equals(right.reset_index(drop=True))
            if not ok:
                issues.append(f"{name}.{col}_mismatch")
                break  # one failure per matrix is enough signal

    # Direct remap from prepared originals into matrix columns
    sex_re = pd.to_numeric(prepared["sex"], errors="coerce").map(SEX_CODE_TO_BIN)
    if not sex_re.reset_index(drop=True).equals(
        pd.to_numeric(
            matrices["primary_classification"]["sex_bin"], errors="coerce"
        ).reset_index(drop=True)
    ):
        issues.append("sex_bin_matrix_not_reproducible")

    slim = prepared[["land_size", "land_size_unit"]].copy()
    acres = add_land_size_acres(slim)[LAND_SIZE_STANDARDISED]
    mat_acres = matrices["segmentation"][LAND_SIZE_STANDARDISED]
    if not np.isclose(acres.fillna(-1), mat_acres.fillna(-1)).all():
        issues.append("land_size_acres_matrix_not_reproducible")

    status = "pass" if not issues else "fail"
    add_check(
        rows,
        "11",
        "reproducibility_from_documented_transforms",
        status,
        "re-derived columns match prepared matrices"
        if status == "pass"
        else "; ".join(issues[:20]),
    )


def build_column_inventory(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in matrices.items():
        for col in df.columns:
            role = "id" if col in ID_VARS else "predictor"
            rows.append({"matrix": name, "column": col, "role": role})
    return pd.DataFrame(rows)


def write_report(
    checklist: pd.DataFrame,
    missing_summary: pd.DataFrame,
    targets: pd.DataFrame,
    flags: pd.DataFrame,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pass = int((checklist["status"] == "pass").sum())
    n_fail = int((checklist["status"] == "fail").sum())
    n_warn = int((checklist["status"] == "warn").sum())
    overall = "PASS" if n_fail == 0 else "FAIL"
    report = f"""# Final Preparation Validation Report (Script 10)

Generated: {timestamp}

## Overall result: {overall}

Checks: {n_pass} pass, {n_fail} fail, {n_warn} warn.

## Checklist

```
{checklist.to_string(index=False)}
```

## Missingness summary

```
{missing_summary.to_string(index=False) if len(missing_summary) else 'No missing predictor values recorded.'}
```

## Unexpected / duplicate flags

```
{flags.to_string(index=False) if len(flags) else 'None'}
```

## Target consistency

```
{targets.to_string(index=False)}
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
"""
    path = REPORTS_DIR / "10_final_preparation_validation.md"
    path.write_text(report, encoding="utf-8")
    return path


def append_log(checklist: pd.DataFrame, overall: str) -> None:
    log_path = LOGS_DIR / "DATA_PREPARATION_LOG.md"
    n_fail = int((checklist["status"] == "fail").sum())
    section = f"""

### Final preparation validation (Script 10)

**Status: Validation {overall}. Modelling not started.**

Script 10 validated the Script 09 prepared matrices against approved inventories and transformation rules.

Checklist summary: {(checklist['status'] == 'pass').sum()} pass, {n_fail} fail, {(checklist['status'] == 'warn').sum()} warn.

Validated items included row alignment, duplicate representations, expected land-size missingness (198), feature membership and inventory separation, exclusion of targets/coping actions/weights/financial variables/`land_size_unit`/default Shannon, binary and ordinal coding, fixed codebook one-hot coverage, target consistency (formal = 75), and reproducibility of documented transforms.

Modelling carry-forward notes recorded in the validation report: fold-wise land-size missingness handling; sparse formal class and stratified validation; extended shock matrix as retrospective same-window analysis only.

Data Preparation closes only after review of `outputs/reports/10_final_preparation_validation.md`. No modelling was started by Script 10.
"""
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    marker = "### Final preparation validation (Script 10)"
    if marker not in existing:
        log_path.write_text(existing.rstrip() + section, encoding="utf-8")
    else:
        pre, _, rest = existing.partition(marker)
        parts = rest.split("\n### ")
        tail = ""
        if len(parts) > 1:
            tail = "\n### " + "\n### ".join(parts[1:])
        log_path.write_text(pre.rstrip() + section + tail, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    helpers = load_script09_helpers()

    source = load_analysis_2024(
        required_columns=sorted(
            set(
                ID_VARS
                + TARGET_VARS
                + ["land_size", "land_size_unit", "sex", "urban_rural", "education", "county", "main_shock_type"]
                + AGRICULTURAL_HAS_VARS
                + LAND_OWNERSHIP_VARS
                + ADDITIONAL_SHOCK_VARS
                + ["n_crop_types", "n_livestock_types", "shannon_diversity", "age_years"]
            )
        )
    )
    prepared = pd.read_parquet(PREPARED_DIR / "09_prepared_analysis_frame.parquet")
    y = pd.read_parquet(PREPARED_DIR / "09_y_targets.parquet")
    matrices = load_matrices()

    # Expected feature lists regenerated from Script 09 rules on prepared frame
    expected_lists = helpers.feature_lists(prepared)

    checklist_rows: list[dict] = []
    flags: list[dict] = []
    missing_summary: list[dict] = []
    target_table: list[dict] = []

    check_row_alignment(matrices, y, source["respondent_id"], checklist_rows)
    check_duplicates(matrices, checklist_rows, flags)
    check_missingness(matrices, checklist_rows, missing_summary)
    check_feature_membership(matrices, expected_lists, checklist_rows, flags)
    check_separation(matrices, checklist_rows)
    check_leakage_exclusions(matrices, checklist_rows, flags)
    check_explicit_exclusions(matrices, checklist_rows)
    check_binary_ordinal(matrices, prepared, checklist_rows)
    check_onehot_coverage(matrices, prepared, checklist_rows)
    check_target_consistency(y, prepared, source, checklist_rows, target_table)
    check_reproducibility(matrices, prepared, source, checklist_rows)

    checklist = pd.DataFrame(checklist_rows)
    inventory = build_column_inventory(matrices)
    flags_df = pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["matrix", "flag_type", "detail"]
    )
    missing_df = pd.DataFrame(missing_summary) if missing_summary else pd.DataFrame(
        columns=["matrix", "column", "n_missing", "expected"]
    )
    targets_df = pd.DataFrame(target_table)

    outputs = {
        "10_validation_checklist.csv": checklist,
        "10_matrix_column_inventory.csv": inventory,
        "10_unexpected_or_duplicate_flags.csv": flags_df,
        "10_missingness_summary.csv": missing_df,
        "10_target_consistency.csv": targets_df,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    report_path = write_report(checklist, missing_df, targets_df, flags_df)
    print(f"Wrote {report_path}")

    overall = "PASS" if (checklist["status"] != "fail").all() else "FAIL"
    append_log(checklist, overall)
    print(f"Updated {LOGS_DIR / 'DATA_PREPARATION_LOG.md'}")

    print("\nChecklist")
    print(checklist.to_string(index=False))
    print(f"\nOverall: {overall}")
    print("\nDone. Review the validation report before modelling.")


if __name__ == "__main__":
    main()
