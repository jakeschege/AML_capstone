#!/usr/bin/env python3
"""Investigate shock exposure variables for the drought-exposed analytical sample.

Covers all shock_* indicators and main_shock_type. Does not encode variables,
collapse categories, remove observations or modify the analytical dataset.
"""

from __future__ import annotations

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
    MAIN_SHOCK_TO_INDICATOR,
    REPORTS_DIR,
    SHOCK_INDICATOR_VARS,
    TABLES_DIR,
    VALUE_LABELS_PATH,
    ensure_output_dirs,
    load_analysis_2024,
)

RARE_INDICATOR_PCT = 5.0
RARE_MAIN_SHOCK_N = 30


def load_r2b_labels() -> dict[int, str]:
    """Load main shock labels from the R2B codebook set."""
    if not VALUE_LABELS_PATH.exists():
        return {}
    labels = pd.read_csv(VALUE_LABELS_PATH, keep_default_na=False)
    subset = labels.loc[labels["label_set"] == "R2B", ["code", "text"]]
    mapping: dict[int, str] = {}
    for _, row in subset.iterrows():
        try:
            code = int(float(row["code"]))
        except (TypeError, ValueError):
            continue
        text = str(row["text"]).strip()
        mapping[code] = text if text else "[blank label]"
    return mapping


def shock_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Completeness and uniqueness inventory for shock variables."""
    rows = []
    cols = SHOCK_INDICATOR_VARS + ["main_shock_type"]
    if "main_shock_type_label" in df.columns:
        cols.append("main_shock_type_label")

    for col in cols:
        s = df[col]
        row = {
            "variable": col,
            "dtype": str(s.dtype),
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100.0 * float(s.isna().mean()), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "role": (
                "sample_defining_exposure"
                if col == "shock_drought_loss"
                else (
                    "additional_shock_indicator"
                    if col.startswith("shock_")
                    else "main_shock"
                )
            ),
        }
        if col.startswith("shock_"):
            positives = pd.to_numeric(s, errors="coerce") == 1
            row["n_positive"] = int(positives.sum())
            row["pct_positive"] = round(100.0 * float(positives.mean()), 2)
            row["is_constant"] = int(s.nunique(dropna=True) == 1)
        rows.append(row)
    return pd.DataFrame(rows)


def indicator_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Positive/negative frequencies for each shock indicator."""
    rows = []
    for col in SHOCK_INDICATOR_VARS:
        s = pd.to_numeric(df[col], errors="coerce")
        n_pos = int((s == 1).sum())
        n_zero = int((s == 0).sum())
        n_other = int((~s.isin([0, 1]) & s.notna()).sum())
        rows.append(
            {
                "variable": col,
                "n_positive": n_pos,
                "pct_positive": round(100.0 * n_pos / len(df), 2),
                "n_zero": n_zero,
                "pct_zero": round(100.0 * n_zero / len(df), 2),
                "n_other_or_missing": n_other + int(s.isna().sum()),
                "is_drought_indicator": col == "shock_drought_loss",
                "is_additional_shock": col != "shock_drought_loss",
                "rare_under_5pct": (
                    col != "shock_drought_loss" and (100.0 * n_pos / len(df)) < RARE_INDICATOR_PCT
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "pct_positive", ascending=False
    ).reset_index(drop=True)


def main_shock_frequencies(df: pd.DataFrame, label_map: dict[int, str]) -> pd.DataFrame:
    """Frequency distribution of main_shock_type with labels."""
    counts = df["main_shock_type"].value_counts(dropna=False).sort_index()
    rows = []
    for code, n in counts.items():
        if pd.isna(code):
            code_out = np.nan
            label = "[missing]"
            mapped_indicator = ""
        else:
            code_out = int(code)
            label = df.loc[df["main_shock_type"] == code, "main_shock_type_label"].iloc[0] if (
                "main_shock_type_label" in df.columns
                and df.loc[df["main_shock_type"] == code, "main_shock_type_label"].notna().any()
            ) else label_map.get(code_out, "[no codebook label]")
            mapped_indicator = MAIN_SHOCK_TO_INDICATOR.get(code_out, "")
        rows.append(
            {
                "main_shock_type": code_out,
                "label": label,
                "mapped_indicator": mapped_indicator,
                "n_households": int(n),
                "pct": round(100.0 * n / len(df), 2),
                "rare_under_n30": int(n) < RARE_MAIN_SHOCK_N,
            }
        )
    return pd.DataFrame(rows)


def shock_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of total and additional shock counts."""
    all_count = df[SHOCK_INDICATOR_VARS].sum(axis=1)
    add_count = df[ADDITIONAL_SHOCK_VARS].sum(axis=1)

    def _dist(series: pd.Series, label: str) -> pd.DataFrame:
        vc = series.value_counts().sort_index()
        out = vc.rename_axis("n_shocks").reset_index(name="n_households")
        out["pct"] = (100.0 * out["n_households"] / len(df)).round(2)
        out.insert(0, "count_definition", label)
        return out

    summary_rows = [
        {
            "count_definition": "all_12_indicators",
            "mean": float(all_count.mean()),
            "median": float(all_count.median()),
            "min": int(all_count.min()),
            "max": int(all_count.max()),
            "pct_with_only_drought": round(100.0 * float((add_count == 0).mean()), 2),
            "pct_with_at_least_one_additional": round(
                100.0 * float((add_count >= 1).mean()), 2
            ),
        },
        {
            "count_definition": "additional_11_excluding_drought",
            "mean": float(add_count.mean()),
            "median": float(add_count.median()),
            "min": int(add_count.min()),
            "max": int(add_count.max()),
            "pct_with_only_drought": round(100.0 * float((add_count == 0).mean()), 2),
            "pct_with_at_least_one_additional": round(
                100.0 * float((add_count >= 1).mean()), 2
            ),
        },
    ]
    # Return detailed distribution; summary is written separately via report text
    detail = pd.concat(
        [
            _dist(all_count, "all_12_indicators"),
            _dist(add_count, "additional_11_excluding_drought"),
        ],
        ignore_index=True,
    )
    detail.attrs["summary"] = pd.DataFrame(summary_rows)
    return detail


def shock_count_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact summary statistics for shock counts."""
    all_count = df[SHOCK_INDICATOR_VARS].sum(axis=1)
    add_count = df[ADDITIONAL_SHOCK_VARS].sum(axis=1)
    return pd.DataFrame(
        [
            {
                "count_definition": "all_12_indicators",
                "mean": round(float(all_count.mean()), 3),
                "median": float(all_count.median()),
                "min": int(all_count.min()),
                "max": int(all_count.max()),
                "pct_only_drought_among_additional": "",
                "pct_with_at_least_one_additional": "",
            },
            {
                "count_definition": "additional_11_excluding_drought",
                "mean": round(float(add_count.mean()), 3),
                "median": float(add_count.median()),
                "min": int(add_count.min()),
                "max": int(add_count.max()),
                "pct_only_drought_among_additional": round(
                    100.0 * float((add_count == 0).mean()), 2
                ),
                "pct_with_at_least_one_additional": round(
                    100.0 * float((add_count >= 1).mean()), 2
                ),
            },
        ]
    )


def main_shock_consistency(df: pd.DataFrame, label_map: dict[int, str]) -> pd.DataFrame:
    """Check whether main_shock_type maps to an indicated shock equal to 1."""
    rows = []
    for code, indicator in MAIN_SHOCK_TO_INDICATOR.items():
        subset = df.loc[df["main_shock_type"] == code].copy()
        n = len(subset)
        if n == 0:
            rows.append(
                {
                    "main_shock_type": code,
                    "label": label_map.get(code, ""),
                    "mapped_indicator": indicator,
                    "n_households": 0,
                    "n_indicator_positive": 0,
                    "n_indicator_not_positive": 0,
                    "pct_consistent": np.nan,
                }
            )
            continue
        positive = pd.to_numeric(subset[indicator], errors="coerce") == 1
        n_pos = int(positive.sum())
        rows.append(
            {
                "main_shock_type": code,
                "label": subset["main_shock_type_label"].iloc[0]
                if "main_shock_type_label" in subset.columns
                else label_map.get(code, ""),
                "mapped_indicator": indicator,
                "n_households": n,
                "n_indicator_positive": n_pos,
                "n_indicator_not_positive": n - n_pos,
                "pct_consistent": round(100.0 * n_pos / n, 2),
            }
        )

    # Overall household-level consistency
    mapped = df["main_shock_type"].map(MAIN_SHOCK_TO_INDICATOR)
    consistent = []
    for idx, indicator in mapped.items():
        if pd.isna(indicator):
            consistent.append(False)
        else:
            consistent.append(
                pd.to_numeric(df.at[idx, indicator], errors="coerce") == 1
            )
    overall = pd.DataFrame(rows)
    overall.attrs["overall_pct"] = round(100.0 * float(np.mean(consistent)), 2)
    overall.attrs["n_inconsistent"] = int((~np.array(consistent)).sum())
    return overall


def overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise co-occurrence counts among additional shock indicators."""
    cols = ADDITIONAL_SHOCK_VARS
    mat = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    rows = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            both = int(((mat[left] == 1) & (mat[right] == 1)).sum())
            if both == 0:
                continue
            rows.append(
                {
                    "shock_a": left,
                    "shock_b": right,
                    "n_both_positive": both,
                    "pct_of_sample": round(100.0 * both / len(df), 2),
                    "pct_of_shock_a_also_b": round(
                        100.0 * both / max(int((mat[left] == 1).sum()), 1), 2
                    ),
                    "pct_of_shock_b_also_a": round(
                        100.0 * both / max(int((mat[right] == 1).sum()), 1), 2
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("n_both_positive", ascending=False).reset_index(drop=True)
    return out


def rare_shock_flags(
    indicator_freq: pd.DataFrame,
    main_freq: pd.DataFrame,
) -> pd.DataFrame:
    """Compile rare indicator and rare main-shock flags."""
    rows = []
    for _, row in indicator_freq.iterrows():
        if row["rare_under_5pct"]:
            rows.append(
                {
                    "item_type": "additional_shock_indicator",
                    "item": row["variable"],
                    "n": int(row["n_positive"]),
                    "pct": float(row["pct_positive"]),
                    "flag": f"positive rate < {RARE_INDICATOR_PCT}%",
                }
            )
    for _, row in main_freq.iterrows():
        if bool(row["rare_under_n30"]):
            rows.append(
                {
                    "item_type": "main_shock_type",
                    "item": f"{row['main_shock_type']}: {row['label']}",
                    "n": int(row["n_households"]),
                    "pct": float(row["pct"]),
                    "flag": f"n < {RARE_MAIN_SHOCK_N}",
                }
            )
    return pd.DataFrame(rows)


def write_report(
    n: int,
    inventory: pd.DataFrame,
    indicator_freq: pd.DataFrame,
    main_freq: pd.DataFrame,
    count_dist: pd.DataFrame,
    count_summary: pd.DataFrame,
    consistency: pd.DataFrame,
    overlap: pd.DataFrame,
    rare_flags: pd.DataFrame,
    overall_consistency_pct: float,
    n_inconsistent: int,
) -> Path:
    """Write the shock exposure Markdown report."""
    drought_row = indicator_freq.loc[
        indicator_freq["variable"] == "shock_drought_loss"
    ].iloc[0]
    rare_inds = indicator_freq.loc[indicator_freq["rare_under_5pct"]]
    top_overlap = overlap.head(8) if len(overlap) else overlap
    add_summary = count_summary.loc[
        count_summary["count_definition"] == "additional_11_excluding_drought"
    ].iloc[0]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Shock Exposure Investigation Report

Generated: {timestamp}

## What was investigated

Shock variables for the drought-exposed analytical sample (`n = {n}`):

- all 12 binary `shock_*` indicators
- `main_shock_type` and available labels

The project feature grouping uses the 11 non-drought indicators plus `main_shock_type`, because `shock_drought_loss` defines the analytical sample.

No encoding, category collapsing, observation removal or dataset modification was performed.

## Observed findings

### Inventory and completeness

```
{inventory.to_string(index=False)}
```

All investigated shock fields are complete in this sample.

### Indicator frequencies

```
{indicator_freq.to_string(index=False)}
```

- `shock_drought_loss` positives: {int(drought_row['n_positive'])} ({drought_row['pct_positive']}%)
- Additional indicators below {RARE_INDICATOR_PCT}% positive: {len(rare_inds)}

### Main shock type

```
{main_freq.to_string(index=False)}
```

### Multiple shock exposure

Summary:

```
{count_summary.to_string(index=False)}
```

Detailed counts:

```
{count_dist.to_string(index=False)}
```

- Households with at least one additional non-drought shock: {add_summary['pct_with_at_least_one_additional']}%
- Households with drought only among the indicator set: {add_summary['pct_only_drought_among_additional']}%

### Consistency between main_shock_type and indicators

Household-level consistency where the mapped indicator equals 1: {overall_consistency_pct}%  
Households inconsistent: {n_inconsistent}

```
{consistency.to_string(index=False)}
```

### Overlaps among additional shocks

Top co-occurring additional shock pairs:

```
{top_overlap.to_string(index=False)}
```

Full pairwise positive overlaps with at least one shared household are written to CSV.

### Rare categories

```
{rare_flags.to_string(index=False) if len(rare_flags) else 'None flagged under the stated thresholds.'}
```

## Interpretation

`shock_drought_loss` is constant in this sample and should not be used as an ordinary predictor. The remaining indicators capture concurrent non-drought shocks. Many households report more than one shock, so the indicators are overlapping rather than mutually exclusive.

`main_shock_type` identifies the household's reported principal shock among the same concept set. Where consistency is high, the main shock is supported by the corresponding indicator. Rare main-shock categories and rare indicators remain informative for description but may be fragile in supervised models with many classes.

## Limitations

- Consistency uses the fixed R2B-to-indicator map and does not invent alternative mappings.
- Pairwise overlap does not imply causal dependence.
- Rare-category thresholds ({RARE_INDICATOR_PCT}% / n < {RARE_MAIN_SHOCK_N}) are descriptive audit rules only.
- No relationship with the coping target was used to select or discard shock variables.

## Recommended preparation decisions for review

1. Retain the 11 additional shock indicators as provisional candidate predictors for the classification stage where conceptually appropriate.
2. Exclude `shock_drought_loss` from ordinary predictors because it is constant in the drought-exposed sample. Preserve it for traceability.
3. Retain `main_shock_type` as a provisional candidate, recognising high cardinality and several rare categories.
4. Keep a derived additional-shock count as an audit measure for now rather than automatically adding it as a predictor.
5. Do not collapse rare shock categories or encode variables in this step.
6. Revisit rare shocks and possible redundancy between indicators and `main_shock_type` during later feature assessment.

## Output files

- `outputs/tables/04_shock_inventory.csv`
- `outputs/tables/04_shock_indicator_frequencies.csv`
- `outputs/tables/04_main_shock_frequencies.csv`
- `outputs/tables/04_shock_count_distribution.csv`
- `outputs/tables/04_shock_count_summary.csv`
- `outputs/tables/04_main_shock_consistency.csv`
- `outputs/tables/04_shock_overlap_summary.csv`
- `outputs/tables/04_rare_shock_flags.csv`
"""
    path = REPORTS_DIR / "04_shock_exposure_investigation.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()
    required = SHOCK_INDICATOR_VARS + ["main_shock_type"]
    analysis = load_analysis_2024(required_columns=required)
    label_map = load_r2b_labels()

    print("=" * 72)
    print("04 Shock exposure investigation")
    print("=" * 72)
    print(f"Observations: {len(analysis)}")
    print(f"Shock indicators: {len(SHOCK_INDICATOR_VARS)}")
    print(f"Additional non-drought indicators: {len(ADDITIONAL_SHOCK_VARS)}")

    inventory = shock_inventory(analysis)
    ind_freq = indicator_frequencies(analysis)
    main_freq = main_shock_frequencies(analysis, label_map)
    count_dist = shock_count_distribution(analysis)
    count_summary = shock_count_summary(analysis)
    consistency = main_shock_consistency(analysis, label_map)
    overlap = overlap_summary(analysis)
    rare_flags = rare_shock_flags(ind_freq, main_freq)

    overall_pct = float(consistency.attrs.get("overall_pct", np.nan))
    n_inconsistent = int(consistency.attrs.get("n_inconsistent", 0))

    outputs = {
        "04_shock_inventory.csv": inventory,
        "04_shock_indicator_frequencies.csv": ind_freq,
        "04_main_shock_frequencies.csv": main_freq,
        "04_shock_count_distribution.csv": count_dist,
        "04_shock_count_summary.csv": count_summary,
        "04_main_shock_consistency.csv": consistency,
        "04_shock_overlap_summary.csv": overlap,
        "04_rare_shock_flags.csv": rare_flags,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    report_path = write_report(
        n=len(analysis),
        inventory=inventory,
        indicator_freq=ind_freq,
        main_freq=main_freq,
        count_dist=count_dist,
        count_summary=count_summary,
        consistency=consistency,
        overlap=overlap,
        rare_flags=rare_flags,
        overall_consistency_pct=overall_pct,
        n_inconsistent=n_inconsistent,
    )
    print(f"Wrote {report_path}")

    print("\nIndicator frequencies")
    print(ind_freq.to_string(index=False))
    print("\nMain shock frequencies")
    print(main_freq.to_string(index=False))
    print("\nShock count summary")
    print(count_summary.to_string(index=False))
    print(f"\nMain-shock consistency: {overall_pct}% ({n_inconsistent} inconsistent)")
    print("\nConsistency by main shock")
    print(consistency.to_string(index=False))
    print("\nRare flags")
    print(rare_flags.to_string(index=False) if len(rare_flags) else "None")
    print("\nDone.")


if __name__ == "__main__":
    main()
