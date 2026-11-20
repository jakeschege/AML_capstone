#!/usr/bin/env python3
"""Missingness and encoding audit for final candidate feature inventories.

Audits segmentation, primary classification and extended classification
candidate sets. Does not encode, impute, scale or remove rare categories.
"""

from __future__ import annotations

import csv
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
    DEMOGRAPHIC_VARS,
    EXTENDED_CLASSIFICATION_CANDIDATES,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    PRIMARY_CLASSIFICATION_CANDIDATES,
    REPORTS_DIR,
    SEGMENTATION_CANDIDATES,
    TABLES_DIR,
    TARGET_VARS,
    VALUE_LABELS_PATH,
    add_land_size_acres,
    ensure_output_dirs,
    load_analysis_2024,
)

RARE_PCT = 5.0
RARE_N = 30

EDUCATION_LABELS = {
    1: "None",
    2: "Primary",
    3: "Secondary",
    4: "Tertiary",
    5: "Other",
}

BINARY_EXPECTED = {
    **{c: (0, 1) for c in AGRICULTURAL_HAS_VARS},
    **{c: (0, 1) for c in LAND_OWNERSHIP_VARS},
    **{c: (0, 1) for c in ADDITIONAL_SHOCK_VARS},
    "sex": (1, 2),
    "urban_rural": (1, 2),
}

CATEGORICAL_LABEL_SETS = {
    "sex": "Sex",
    "education": "Education",
    "urban_rural": "Cluster_Type",
    "county": "county",
    "land_size_unit": None,  # manual map
    "main_shock_type": "R2B",
}

LAND_UNIT_LABELS = {
    1: "Acres",
    2: "Hectares",
    3: "Square metres",
    4: "Square feet",
    5: "Community land",
}


def load_label_maps() -> dict[str, dict[int, str]]:
    maps: dict[str, dict[int, str]] = {
        "education": EDUCATION_LABELS.copy(),
        "land_size_unit": LAND_UNIT_LABELS.copy(),
    }
    if not VALUE_LABELS_PATH.exists():
        return maps
    labels = pd.read_csv(VALUE_LABELS_PATH, keep_default_na=False)
    for var, label_set in CATEGORICAL_LABEL_SETS.items():
        if label_set is None:
            continue
        subset = labels.loc[labels["label_set"] == label_set, ["code", "text"]]
        mapping: dict[int, str] = {}
        for _, row in subset.iterrows():
            try:
                code = int(float(row["code"]))
            except (TypeError, ValueError):
                continue
            text = str(row["text"]).strip()
            if text.lower() == "none":
                mapping[code] = "None"
            elif text:
                mapping[code] = text
            else:
                mapping[code] = "[blank label]"
        if mapping:
            maps[var] = mapping
    # Ensure education uses resolved None label
    maps["education"] = EDUCATION_LABELS.copy()
    return maps


def inventory_table(name: str, variables: list[str], notes: dict[str, str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        rows.append(
            {
                "inventory": name,
                "variable": var,
                "preparation_note": notes.get(var, ""),
            }
        )
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame, inventories: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for inv_name, variables in inventories.items():
        for var in variables:
            s = df[var]
            miss = int(s.isna().sum())
            rows.append(
                {
                    "inventory": inv_name,
                    "variable": var,
                    "n": len(df),
                    "n_non_missing": int(s.notna().sum()),
                    "n_missing": miss,
                    "missing_pct": round(100.0 * miss / len(df), 2),
                    "n_unique": int(s.nunique(dropna=True)),
                    "dtype": str(s.dtype),
                }
            )
    return pd.DataFrame(rows)


def categorical_frequencies(
    df: pd.DataFrame,
    variables: list[str],
    label_maps: dict[str, dict[int, str]],
) -> pd.DataFrame:
    rows = []
    for var in variables:
        label_map = label_maps.get(var, {})
        vc = df[var].value_counts(dropna=False)
        for code, n in vc.items():
            if pd.isna(code):
                code_out = np.nan
                label = "[missing]"
            else:
                try:
                    code_out = int(code) if float(code).is_integer() else float(code)
                    label = label_map.get(int(code), "[no codebook label]")
                except (TypeError, ValueError):
                    code_out = code
                    label = "[no codebook label]"
            rows.append(
                {
                    "variable": var,
                    "code": code_out,
                    "label": label,
                    "n_households": int(n),
                    "pct": round(100.0 * n / len(df), 2),
                    "rare_under_5pct": (100.0 * n / len(df)) < RARE_PCT,
                    "rare_under_n30": int(n) < RARE_N,
                }
            )
    return pd.DataFrame(rows)


def binary_coding_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, expected in BINARY_EXPECTED.items():
        if var not in df.columns:
            continue
        vals = sorted(pd.to_numeric(df[var], errors="coerce").dropna().unique().tolist())
        unexpected = [v for v in vals if v not in expected]
        rows.append(
            {
                "variable": var,
                "expected_values": ",".join(str(int(v)) for v in expected),
                "observed_values": ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in vals),
                "n_unique": len(vals),
                "coding_consistent": len(unexpected) == 0 and set(vals).issubset(set(expected)),
                "n_missing": int(df[var].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def ordinal_education_assessment(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["education"].value_counts(dropna=False).sort_index()
    observed = [int(c) for c in vc.index if pd.notna(c)]
    return pd.DataFrame(
        [
            {
                "variable": "education",
                "observed_codes": ",".join(str(c) for c in observed),
                "labels": "; ".join(f"{c}={EDUCATION_LABELS.get(c, '?')}" for c in observed),
                "code_1_label": EDUCATION_LABELS[1],
                "provisional_ordinal": set(observed).issubset({1, 2, 3, 4}) and 5 not in observed,
                "encoding_recommendation": (
                    "ordinal_or_ordered_categorical_pending_feature_preparation"
                    if set(observed).issubset({1, 2, 3, 4})
                    else "treat_as_nominal_until_other_resolved"
                ),
                "notes": (
                    "Codes 1-4 correspond to None, Primary, Secondary, Tertiary. "
                    "Encoding not applied in this audit."
                ),
            }
        ]
    )


def continuous_summaries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["age_years", LAND_SIZE_STANDARDISED]:
        s = pd.to_numeric(df[var], errors="coerce")
        rows.append(
            {
                "variable": var,
                "n_non_missing": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(100.0 * float(s.isna().mean()), 2),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "median": float(s.median()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().any() else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "p05": float(s.quantile(0.05)) if s.notna().any() else np.nan,
                "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
                "skewness": float(s.skew()) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def land_size_missing_reasons(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[df[LAND_SIZE_STANDARDISED].isna()].copy()
    unit = pd.to_numeric(missing["land_size_unit"], errors="coerce")
    size = pd.to_numeric(missing["land_size"], errors="coerce")
    reason = np.select(
        [
            unit.eq(5),
            size.isin([98, 99]),
            unit.isna(),
        ],
        [
            "Community land",
            "Special response code 98/99",
            "Missing measurement unit",
        ],
        default="Other or unresolved",
    )
    out = (
        pd.Series(reason)
        .value_counts(dropna=False)
        .rename_axis("missing_reason")
        .reset_index(name="n_households")
    )
    out["pct_of_sample"] = (100.0 * out["n_households"] / len(df)).round(2)
    out["pct_of_missing"] = (
        100.0 * out["n_households"] / max(len(missing), 1)
    ).round(2)
    return out


def rare_category_flags(cat_freq: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rare = cat_freq.loc[cat_freq["rare_under_5pct"] | cat_freq["rare_under_n30"]]
    for _, row in rare.iterrows():
        rows.append(
            {
                "item_type": "predictor_category",
                "variable": row["variable"],
                "category": f"{row['code']}:{row['label']}",
                "n": int(row["n_households"]),
                "pct": float(row["pct"]),
                "flag": "rare_category",
            }
        )
    for _, row in targets.iterrows():
        if row["pct"] < RARE_PCT or row["n_households"] < RARE_N:
            rows.append(
                {
                    "item_type": "target_class",
                    "variable": row["target"],
                    "category": row["class_label"],
                    "n": int(row["n_households"]),
                    "pct": float(row["pct"]),
                    "flag": "sparse_target_class",
                }
            )
    # rare shocks by positive rate
    return pd.DataFrame(rows)


def target_representation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["coping_category", "coping_category_collapsed"]:
        vc = df[target].value_counts(dropna=False)
        for label, n in vc.items():
            rows.append(
                {
                    "target": target,
                    "class_label": label if pd.notna(label) else "[missing]",
                    "n_households": int(n),
                    "pct": round(100.0 * n / len(df), 2),
                    "below_5pct": (100.0 * n / len(df)) < RARE_PCT,
                    "below_n30": int(n) < RARE_N,
                }
            )
    return pd.DataFrame(rows)


def preparation_recommendations(
    miss: pd.DataFrame,
    binary: pd.DataFrame,
    education: pd.DataFrame,
    continuous: pd.DataFrame,
    land_missing: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Missingness
    high_miss = miss.loc[miss["missing_pct"] > 0].drop_duplicates("variable")
    for _, row in high_miss.iterrows():
        rows.append(
            {
                "topic": "missingness",
                "variable": row["variable"],
                "finding": f"missing_pct={row['missing_pct']}",
                "recommended_action": (
                    "document_missingness_and_decide_imputation_or_indicator_in_feature_preparation"
                    if row["variable"] == LAND_SIZE_STANDARDISED
                    else "investigate_before_feature_preparation"
                ),
                "apply_now": False,
            }
        )

    rows.append(
        {
            "topic": "land_size_unit",
            "variable": LAND_SIZE_STANDARDISED,
            "finding": "standardised_unit_acres_confirmed",
            "recommended_action": "use_land_size_acres_in_feature_preparation",
            "apply_now": False,
        }
    )

    for _, row in land_missing.iterrows():
        rows.append(
            {
                "topic": "land_size_missingness",
                "variable": LAND_SIZE_STANDARDISED,
                "finding": f"{row['missing_reason']}: n={row['n_households']}",
                "recommended_action": "retain_missingness_without_imputation_in_this_audit",
                "apply_now": False,
            }
        )

    inconsistent = binary.loc[~binary["coding_consistent"]]
    for _, row in inconsistent.iterrows():
        rows.append(
            {
                "topic": "binary_coding",
                "variable": row["variable"],
                "finding": f"observed={row['observed_values']}",
                "recommended_action": "resolve_coding_before_encoding",
                "apply_now": False,
            }
        )

    edu = education.iloc[0]
    rows.append(
        {
            "topic": "ordinal_encoding",
            "variable": "education",
            "finding": edu["labels"],
            "recommended_action": edu["encoding_recommendation"],
            "apply_now": False,
        }
    )

    for var in ["sex", "urban_rural", "county", "land_size_unit", "main_shock_type"]:
        rows.append(
            {
                "topic": "categorical_encoding",
                "variable": var,
                "finding": "categorical_codes_require_encoding_scheme",
                "recommended_action": (
                    "one_hot_or_target_aware_encoding_in_feature_preparation"
                    if var != "education"
                    else "see_ordinal_encoding"
                ),
                "apply_now": False,
            }
        )

    rows.append(
        {
            "topic": "redundancy",
            "variable": "shannon_diversity",
            "finding": "flagged_redundant_with_portfolio_counts",
            "recommended_action": "assess_in_or_out_during_feature_preparation",
            "apply_now": False,
        }
    )

    formal = targets.loc[
        (targets["target"] == "coping_category") & (targets["class_label"] == "formal")
    ]
    if len(formal):
        rows.append(
            {
                "topic": "target_representation",
                "variable": "coping_category",
                "finding": f"formal_class_pct={formal.iloc[0]['pct']}",
                "recommended_action": "plan_stratified_validation_and_consider_collapsed_target",
                "apply_now": False,
            }
        )

    for _, row in continuous.iterrows():
        if row["variable"] == LAND_SIZE_STANDARDISED:
            rows.append(
                {
                    "topic": "continuous_preparation",
                    "variable": row["variable"],
                    "finding": f"skewness={row['skewness']:.3f}; missing_pct={row['missing_pct']}",
                    "recommended_action": "decide_scaling_and_missing_indicator_later_do_not_impute_now",
                    "apply_now": False,
                }
            )
        if row["variable"] == "age_years":
            rows.append(
                {
                    "topic": "continuous_preparation",
                    "variable": row["variable"],
                    "finding": f"skewness={row['skewness']:.3f}; complete",
                    "recommended_action": "keep_numeric_decide_scaling_in_feature_preparation",
                    "apply_now": False,
                }
            )

    rows.append(
        {
            "topic": "hard_rule",
            "variable": "all_candidates",
            "finding": "audit_only",
            "recommended_action": "do_not_encode_impute_scale_or_drop_rare_categories_in_this_step",
            "apply_now": False,
        }
    )
    return pd.DataFrame(rows)


def write_report(
    n: int,
    inv_seg: pd.DataFrame,
    inv_pri: pd.DataFrame,
    inv_ext: pd.DataFrame,
    miss: pd.DataFrame,
    cat_freq: pd.DataFrame,
    binary: pd.DataFrame,
    education: pd.DataFrame,
    continuous: pd.DataFrame,
    land_missing: pd.DataFrame,
    rare: pd.DataFrame,
    targets: pd.DataFrame,
    recs: pd.DataFrame,
) -> Path:
    miss_nonzero = (
        miss.drop_duplicates("variable")
        .loc[lambda d: d["n_missing"] > 0]
        .sort_values("missing_pct", ascending=False)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Missingness and Encoding Audit Report

Generated: {timestamp}

## Purpose

Audit preparation requirements for the segmentation, primary classification and extended classification candidate inventories. No encoding, imputation, scaling or rare-category removal was performed.

Land size unit decision used in this audit: **acres** (`land_size_acres`).

## Inventories

- Segmentation candidates: {len(inv_seg)}
- Primary classification candidates: {len(inv_pri)}
- Extended classification candidates: {len(inv_ext)}

Sample size: {n}

## Missingness

Variables with any missing values:

```
{miss_nonzero.to_string(index=False) if len(miss_nonzero) else 'None'}
```

Land size missing reasons:

```
{land_missing.to_string(index=False)}
```

## Binary coding

```
{binary.to_string(index=False)}
```

## Education ordinal assessment

```
{education.to_string(index=False)}
```

## Continuous summaries

```
{continuous.to_string(index=False)}
```

## Target representation

```
{targets.to_string(index=False)}
```

## Rare category flags

```
{rare.head(40).to_string(index=False) if len(rare) else 'None'}
```

## Preparation recommendations

```
{recs.to_string(index=False)}
```

## Interpretation

Most candidate predictors are complete. The main missingness issue is standardised land size, largely due to community land and special response codes. Education is a provisional ordinal scale from None to Tertiary. The four-class target has a sparse formal class, so stratified validation and the collapsed three-class target remain important design considerations.

Encoding schemes, imputation choices and rare-category handling are deferred to feature preparation.

## Output files

- `outputs/tables/08_inventory_segmentation.csv`
- `outputs/tables/08_inventory_primary_classification.csv`
- `outputs/tables/08_inventory_extended_classification.csv`
- `outputs/tables/08_missingness_by_variable.csv`
- `outputs/tables/08_categorical_frequencies.csv`
- `outputs/tables/08_binary_coding_checks.csv`
- `outputs/tables/08_ordinal_education_assessment.csv`
- `outputs/tables/08_continuous_summaries.csv`
- `outputs/tables/08_land_size_missing_reasons.csv`
- `outputs/tables/08_rare_category_flags.csv`
- `outputs/tables/08_target_representation.csv`
- `outputs/tables/08_preparation_recommendations.csv`
"""
    path = REPORTS_DIR / "08_missingness_and_encoding_audit.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()

    base_cols = sorted(
        set(
            DEMOGRAPHIC_VARS
            + AGRICULTURAL_HAS_VARS
            + ["n_crop_types", "n_livestock_types", "shannon_diversity"]
            + ["land_size", "land_size_unit"]
            + LAND_OWNERSHIP_VARS
            + ADDITIONAL_SHOCK_VARS
            + ["main_shock_type"]
            + TARGET_VARS
        )
    )
    analysis = load_analysis_2024(required_columns=base_cols)
    analysis = add_land_size_acres(analysis)

    notes = {
        "shannon_diversity": "flagged_redundancy",
        LAND_SIZE_STANDARDISED: "standardised_acres",
        "education": "ordinal_candidate_code_1_None",
        "main_shock_type": "extended_same_window_model",
    }
    for shock in ADDITIONAL_SHOCK_VARS:
        notes[shock] = "extended_same_window_model"

    inventories = {
        "segmentation": list(SEGMENTATION_CANDIDATES),
        "primary_classification": list(PRIMARY_CLASSIFICATION_CANDIDATES),
        "extended_classification": list(EXTENDED_CLASSIFICATION_CANDIDATES),
    }
    inv_seg = inventory_table("segmentation", inventories["segmentation"], notes)
    inv_pri = inventory_table(
        "primary_classification", inventories["primary_classification"], notes
    )
    inv_ext = inventory_table(
        "extended_classification", inventories["extended_classification"], notes
    )

    label_maps = load_label_maps()
    miss = missingness_table(analysis, inventories)

    cat_vars = [
        "sex",
        "education",
        "urban_rural",
        "county",
        "land_size_unit",
        "main_shock_type",
    ]
    cat_freq = categorical_frequencies(analysis, cat_vars, label_maps)
    binary = binary_coding_checks(analysis)
    education = ordinal_education_assessment(analysis)
    continuous = continuous_summaries(analysis)
    land_missing = land_size_missing_reasons(analysis)
    targets = target_representation(analysis)

    # Add rare positive shocks to rare flags
    rare = rare_category_flags(cat_freq, targets)
    shock_rows = []
    for var in ADDITIONAL_SHOCK_VARS:
        n_pos = int((pd.to_numeric(analysis[var], errors="coerce") == 1).sum())
        pct = 100.0 * n_pos / len(analysis)
        if pct < RARE_PCT or n_pos < RARE_N:
            shock_rows.append(
                {
                    "item_type": "shock_indicator_positive",
                    "variable": var,
                    "category": "positive=1",
                    "n": n_pos,
                    "pct": round(pct, 2),
                    "flag": "rare_positive_rate",
                }
            )
    if shock_rows:
        rare = pd.concat([rare, pd.DataFrame(shock_rows)], ignore_index=True)

    recs = preparation_recommendations(
        miss, binary, education, continuous, land_missing, targets
    )

    outputs = {
        "08_inventory_segmentation.csv": inv_seg,
        "08_inventory_primary_classification.csv": inv_pri,
        "08_inventory_extended_classification.csv": inv_ext,
        "08_missingness_by_variable.csv": miss,
        "08_categorical_frequencies.csv": cat_freq,
        "08_binary_coding_checks.csv": binary,
        "08_ordinal_education_assessment.csv": education,
        "08_continuous_summaries.csv": continuous,
        "08_land_size_missing_reasons.csv": land_missing,
        "08_rare_category_flags.csv": rare,
        "08_target_representation.csv": targets,
        "08_preparation_recommendations.csv": recs,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        # Quote non-numeric fields so labels such as education "None" survive re-reads
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    report_path = write_report(
        n=len(analysis),
        inv_seg=inv_seg,
        inv_pri=inv_pri,
        inv_ext=inv_ext,
        miss=miss,
        cat_freq=cat_freq,
        binary=binary,
        education=education,
        continuous=continuous,
        land_missing=land_missing,
        rare=rare,
        targets=targets,
        recs=recs,
    )
    print(f"Wrote {report_path}")

    print("\nInventory sizes")
    print(f"segmentation={len(inv_seg)} primary={len(inv_pri)} extended={len(inv_ext)}")
    print("\nMissing variables")
    print(
        miss.drop_duplicates("variable")
        .loc[lambda d: d["n_missing"] > 0][
            ["variable", "n_missing", "missing_pct"]
        ]
        .to_string(index=False)
    )
    print("\nTargets")
    print(targets.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
