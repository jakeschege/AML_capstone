#!/usr/bin/env python3
"""Deterministic feature preparation for segmentation and classification.

Produces separate prepared matrices without imputation, scaling, train/val
splitting, or fitted transformers. Original source columns and targets are
preserved in a companion prepared analysis frame.
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    ADDITIONAL_SHOCK_VARS,
    AGRICULTURAL_HAS_VARS,
    DEMOGRAPHIC_VARS,
    EXPECTED_DROUGHT_N,
    ID_VARS,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    LAND_SIZE_TO_ACRES,
    LAND_SIZE_VARS,
    LOGS_DIR,
    PORTFOLIO_COUNT_VARS,
    PREPARED_DIR,
    REPORTS_DIR,
    SENSITIVITY_CANDIDATES,
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


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "unknown"


def one_hot_from_codebook(
    series: pd.Series,
    categories: list[tuple[int, str]],
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic one-hot columns for fixed codebook categories."""
    values = pd.to_numeric(series, errors="coerce")
    cols = {}
    dictionary_rows = []
    for code, label in categories:
        col = f"{prefix}_{code}_{slugify(label)}"
        cols[col] = (values == code).astype(int)
        dictionary_rows.append(
            {
                "source_variable": series.name,
                "code": code,
                "label": label,
                "output_column": col,
                "n_positive": int(cols[col].sum()),
            }
        )
    return pd.DataFrame(cols, index=series.index), pd.DataFrame(dictionary_rows)


def prepare_frame(analysis: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame]:
    transform_rows: list[dict] = []
    out = analysis.copy()

    # Land size acres
    out = add_land_size_acres(out)
    transform_rows.append(
        {
            "step": "derive_land_size_acres",
            "input_columns": "land_size;land_size_unit",
            "output_columns": LAND_SIZE_STANDARDISED,
            "rule": (
                f"size masked for 98/99; multiply by unit map {LAND_SIZE_TO_ACRES}; "
                "unit 5 and missing unit remain NaN; no imputation; no missingness indicator"
            ),
            "fit_on_data": False,
        }
    )

    # Binary remaps
    sex = pd.to_numeric(out["sex"], errors="coerce")
    out["sex_bin"] = sex.map(SEX_CODE_TO_BIN)
    transform_rows.append(
        {
            "step": "remap_sex",
            "input_columns": "sex",
            "output_columns": "sex_bin",
            "rule": "Male(1)->0; Female(2)->1; original sex retained",
            "fit_on_data": False,
        }
    )

    urban = pd.to_numeric(out["urban_rural"], errors="coerce")
    out["urban_rural_bin"] = urban.map(URBAN_RURAL_CODE_TO_BIN)
    transform_rows.append(
        {
            "step": "remap_urban_rural",
            "input_columns": "urban_rural",
            "output_columns": "urban_rural_bin",
            "rule": "Rural(1)->0; Urban(2)->1; original urban_rural retained",
            "fit_on_data": False,
        }
    )

    # Education ordered codes retained as-is
    out["education"] = pd.to_numeric(out["education"], errors="coerce")
    transform_rows.append(
        {
            "step": "education_ordered_codes",
            "input_columns": "education",
            "output_columns": "education",
            "rule": "Ordered integers 1-4 (None, Primary, Secondary, Tertiary); no fitted encoder",
            "fit_on_data": False,
        }
    )

    county_cats = load_codebook_categories("county")
    county_oh, county_dict = one_hot_from_codebook(out["county"], county_cats, "county")
    out = pd.concat([out, county_oh], axis=1)
    transform_rows.append(
        {
            "step": "onehot_county",
            "input_columns": "county",
            "output_columns": ";".join(county_oh.columns),
            "rule": "Fixed codebook one-hot (label_set=county); unused levels allowed as zeros",
            "fit_on_data": False,
        }
    )

    shock_cats = load_codebook_categories("R2B")
    shock_oh, shock_dict = one_hot_from_codebook(
        out["main_shock_type"], shock_cats, "main_shock_type"
    )
    out = pd.concat([out, shock_oh], axis=1)
    transform_rows.append(
        {
            "step": "onehot_main_shock_type",
            "input_columns": "main_shock_type",
            "output_columns": ";".join(shock_oh.columns),
            "rule": "Fixed codebook one-hot (label_set=R2B); used in extended matrix only",
            "fit_on_data": False,
        }
    )

    transform_rows.append(
        {
            "step": "pass_through_binaries_and_counts",
            "input_columns": ";".join(
                AGRICULTURAL_HAS_VARS
                + PORTFOLIO_COUNT_VARS
                + LAND_OWNERSHIP_VARS
                + ADDITIONAL_SHOCK_VARS
                + ["age_years"]
            ),
            "output_columns": "same",
            "rule": "No scaling, no imputation, no dropping",
            "fit_on_data": False,
        }
    )
    transform_rows.append(
        {
            "step": "exclude_from_default_matrices",
            "input_columns": "land_size_unit;shannon_diversity;financial_access_vars",
            "output_columns": "none_in_default_matrices",
            "rule": (
                "land_size_unit excluded; shannon_diversity sensitivity sidecar only; "
                "financial vars excluded per Script 07 temporal restrictions"
            ),
            "fit_on_data": False,
        }
    )

    binary_dict = pd.DataFrame(
        [
            {
                "variable": "sex",
                "original_code": 1,
                "original_label": "Male",
                "prepared_column": "sex_bin",
                "prepared_code": 0,
            },
            {
                "variable": "sex",
                "original_code": 2,
                "original_label": "Female",
                "prepared_column": "sex_bin",
                "prepared_code": 1,
            },
            {
                "variable": "urban_rural",
                "original_code": 1,
                "original_label": "Rural",
                "prepared_column": "urban_rural_bin",
                "prepared_code": 0,
            },
            {
                "variable": "urban_rural",
                "original_code": 2,
                "original_label": "Urban",
                "prepared_column": "urban_rural_bin",
                "prepared_code": 1,
            },
        ]
    )
    onehot_dict = pd.concat([county_dict, shock_dict], ignore_index=True)
    return out, transform_rows, binary_dict, onehot_dict


def feature_lists(prepared: pd.DataFrame) -> dict[str, list[str]]:
    # Only codebook one-hots (exclude source label columns such as main_shock_type_label)
    county_cols = [
        c for c in prepared.columns if re.match(r"^county_\d+_", c)
    ]
    shock_type_cols = [
        c for c in prepared.columns if re.match(r"^main_shock_type_\d+_", c)
    ]

    segmentation = (
        list(AGRICULTURAL_HAS_VARS)
        + list(PORTFOLIO_COUNT_VARS)
        + [LAND_SIZE_STANDARDISED]
        + list(LAND_OWNERSHIP_VARS)
    )
    primary = (
        ["age_years", "sex_bin", "education", "urban_rural_bin"]
        + county_cols
        + list(AGRICULTURAL_HAS_VARS)
        + list(PORTFOLIO_COUNT_VARS)
        + [LAND_SIZE_STANDARDISED]
        + list(LAND_OWNERSHIP_VARS)
    )
    extended = primary + list(ADDITIONAL_SHOCK_VARS) + shock_type_cols
    sensitivity = list(SENSITIVITY_CANDIDATES)

    for name, cols in {
        "segmentation": segmentation,
        "primary": primary,
        "extended": extended,
        "sensitivity": sensitivity,
    }.items():
        missing = [c for c in cols if c not in prepared.columns]
        if missing:
            raise KeyError(f"Prepared frame missing {name} columns: {missing}")
    return {
        "segmentation": segmentation,
        "primary_classification": primary,
        "extended_classification": extended,
        "sensitivity": sensitivity,
    }


def land_missingness_note(prepared: pd.DataFrame) -> pd.DataFrame:
    acres = prepared[LAND_SIZE_STANDARDISED]
    unit = pd.to_numeric(prepared["land_size_unit"], errors="coerce")
    size = pd.to_numeric(prepared["land_size"], errors="coerce")
    missing = acres.isna()
    community = missing & unit.eq(5)
    special = missing & size.isin([98, 99]) & ~unit.eq(5)
    other = missing & ~community & ~special
    return pd.DataFrame(
        [
            {
                "topic": "land_size_acres_missing_total",
                "n": int(missing.sum()),
                "pct": round(100.0 * float(missing.mean()), 2),
                "note": (
                    "Missing values retained. No imputation and no missingness indicator "
                    "in Script 09. Not all modelling algorithms can consume NaNs directly; "
                    "handle later within training folds only."
                ),
            },
            {
                "topic": "primary_reason_community_land",
                "n": int(community.sum()),
                "pct": round(100.0 * float(community.mean()), 2),
                "note": "Unit code 5; includes rows that also carry size 98/99 placeholders",
            },
            {
                "topic": "primary_reason_special_98_99_non_community",
                "n": int(special.sum()),
                "pct": round(100.0 * float(special.mean()), 2),
                "note": "Size 98/99 with unit not equal to 5",
            },
            {
                "topic": "other_missing",
                "n": int(other.sum()),
                "pct": round(100.0 * float(other.mean()), 2),
                "note": "Residual missing after community/special classification",
            },
        ]
    )


def write_feature_list(name: str, columns: list[str]) -> Path:
    path = TABLES_DIR / f"09_feature_list_{name}.csv"
    pd.DataFrame({"feature_set": name, "column": columns, "order": range(1, len(columns) + 1)}).to_csv(
        path, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    return path


def write_report(
    lists: dict[str, list[str]],
    transforms: pd.DataFrame,
    land_note: pd.DataFrame,
    binary_dict: pd.DataFrame,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Feature Preparation Report (Script 09)

Generated: {timestamp}

## Scope

Deterministic feature preparation for the drought-exposed sample (n = {EXPECTED_DROUGHT_N}).
No imputation, scaling, train/validation split, or fitted transformers were applied.

## Prepared feature sets

| Set | n_features |
|---|---|
| Segmentation | {len(lists['segmentation'])} |
| Primary classification | {len(lists['primary_classification'])} |
| Extended classification | {len(lists['extended_classification'])} |
| Sensitivity sidecar | {len(lists['sensitivity'])} |

## Binary remaps

```
{binary_dict.to_string(index=False)}
```

## Land-size missingness

```
{land_note.to_string(index=False)}
```

Leaving `land_size_acres` missing does not mean every modelling algorithm can consume the matrix directly. Any later imputation or algorithm-specific missing-value handling must be addressed separately and fitted on training data within each validation fold only.

## Transformation register

```
{transforms[['step', 'input_columns', 'output_columns', 'fit_on_data']].to_string(index=False)}
```

## Future fitted preprocessing rule

Any future imputation, scaling, feature selection based on learned statistics, or other data-dependent transformations must be fitted using training data within each validation fold and then applied to the corresponding validation data.

## Output files

- `outputs/prepared/09_prepared_analysis_frame.parquet`
- `outputs/prepared/09_X_segmentation.parquet`
- `outputs/prepared/09_X_primary_classification.parquet`
- `outputs/prepared/09_X_extended_classification.parquet`
- `outputs/prepared/09_X_sensitivity_shannon.parquet`
- `outputs/prepared/09_y_targets.parquet`
- `outputs/tables/09_transformation_register.csv`
- `outputs/tables/09_feature_list_*.csv`
- `outputs/tables/09_binary_remap_dictionary.csv`
- `outputs/tables/09_onehot_category_dictionary.csv`
- `outputs/tables/09_land_size_missingness_note.csv`
"""
    path = REPORTS_DIR / "09_feature_preparation.md"
    path.write_text(report, encoding="utf-8")
    return path


def append_log(
    lists: dict[str, list[str]],
    land_note: pd.DataFrame,
) -> None:
    log_path = LOGS_DIR / "DATA_PREPARATION_LOG.md"
    section = f"""

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

Prepared feature counts: segmentation {len(lists['segmentation'])}; primary {len(lists['primary_classification'])}; extended {len(lists['extended_classification'])}; sensitivity {len(lists['sensitivity'])}.

Land-size missingness retained in prepared matrices:

```
{land_note.to_string(index=False)}
```

Original source columns and target variables were preserved in `outputs/prepared/09_prepared_analysis_frame.parquet` and `outputs/prepared/09_y_targets.parquet`. The external FinAccess modelling frame was not modified.

**Future fitted preprocessing rule**

Any future imputation, scaling, feature selection based on learned statistics, or other data-dependent transformations must be fitted using training data within each validation fold and then applied to the corresponding validation data. Script 09 itself does not create a train/validation split and does not fit encoders, imputers or scalers.
"""
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if "### Feature preparation (Script 09)" not in existing:
        log_path.write_text(existing.rstrip() + section, encoding="utf-8")
    else:
        # Replace existing Script 09 section if re-run
        pre, _, rest = existing.partition("### Feature preparation (Script 09)")
        # Drop from that heading to EOF or next ### at start of line after
        tail = ""
        parts = rest.split("\n### ")
        if len(parts) > 1:
            # first chunk is old 09 body without heading; remaining headed sections
            tail = "\n### " + "\n### ".join(parts[1:])
        log_path.write_text(pre.rstrip() + section + tail, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()

    required = sorted(
        set(
            ID_VARS
            + WEIGHT_VARS
            + DEMOGRAPHIC_VARS
            + AGRICULTURAL_HAS_VARS
            + PORTFOLIO_COUNT_VARS
            + SENSITIVITY_CANDIDATES
            + LAND_SIZE_VARS
            + LAND_OWNERSHIP_VARS
            + ADDITIONAL_SHOCK_VARS
            + ["main_shock_type"]
            + TARGET_VARS
        )
    )
    analysis = load_analysis_2024(required_columns=required)
    prepared, transform_rows, binary_dict, onehot_dict = prepare_frame(analysis)
    lists = feature_lists(prepared)
    land_note = land_missingness_note(prepared)
    transforms = pd.DataFrame(transform_rows)

    # Companion targets and matrices
    id_cols = [c for c in ID_VARS if c in prepared.columns]
    y = prepared[id_cols + [c for c in TARGET_VARS if c in prepared.columns]].copy()
    X_seg = prepared[id_cols + lists["segmentation"]].copy()
    X_pri = prepared[id_cols + lists["primary_classification"]].copy()
    X_ext = prepared[id_cols + lists["extended_classification"]].copy()
    X_sens = prepared[id_cols + lists["sensitivity"]].copy()

    assert len(prepared) == EXPECTED_DROUGHT_N
    assert prepared[LAND_SIZE_STANDARDISED].isna().sum() == 198
    assert "land_size_unit" not in lists["segmentation"]
    assert "shannon_diversity" not in lists["segmentation"]
    assert "shannon_diversity" not in lists["primary_classification"]
    assert all(c not in lists["primary_classification"] for c in ADDITIONAL_SHOCK_VARS)

    # Write prepared artefacts
    prepared_path = PREPARED_DIR / "09_prepared_analysis_frame.parquet"
    prepared.to_parquet(prepared_path, index=False)
    print(f"Wrote {prepared_path}")

    matrix_map = {
        "09_X_segmentation.parquet": X_seg,
        "09_X_primary_classification.parquet": X_pri,
        "09_X_extended_classification.parquet": X_ext,
        "09_X_sensitivity_shannon.parquet": X_sens,
        "09_y_targets.parquet": y,
    }
    for name, frame in matrix_map.items():
        path = PREPARED_DIR / name
        frame.to_parquet(path, index=False)
        print(f"Wrote {path}")

    table_map = {
        "09_transformation_register.csv": transforms,
        "09_binary_remap_dictionary.csv": binary_dict,
        "09_onehot_category_dictionary.csv": onehot_dict,
        "09_land_size_missingness_note.csv": land_note,
    }
    for name, table in table_map.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    for name, cols in lists.items():
        path = write_feature_list(name, cols)
        print(f"Wrote {path}")

    report_path = write_report(lists, transforms, land_note, binary_dict)
    print(f"Wrote {report_path}")
    append_log(lists, land_note)
    print(f"Updated {LOGS_DIR / 'DATA_PREPARATION_LOG.md'}")

    print("\nFeature counts")
    for name, cols in lists.items():
        print(f"  {name}: {len(cols)}")
    print("\nLand missingness")
    print(land_note.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
