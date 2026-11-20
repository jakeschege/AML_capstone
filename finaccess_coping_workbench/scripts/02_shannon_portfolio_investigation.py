#!/usr/bin/env python3
"""Investigate Shannon diversity against crop and livestock portfolio counts.

This script continues Data Preparation for the FinAccess 2024 drought-exposed
sample. It does not modify source data or the main notebook.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    AGRICULTURAL_HAS_VARS,
    PORTFOLIO_CORE_VARS,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    load_analysis_2024,
)

MIN_COMBO_COUNT = 10
SHANNON_ROUND = 6


def round_shannon(series: pd.Series) -> pd.Series:
    """Round Shannon values for stable grouping and display."""
    return series.round(SHANNON_ROUND)


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the three portfolio measures."""
    rows = []
    for col in PORTFOLIO_CORE_VARS:
        s = df[col]
        rows.append(
            {
                "variable": col,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "n_unique": int(s.nunique(dropna=True)),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def shannon_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency of each observed Shannon value."""
    rounded = round_shannon(df["shannon_diversity"])
    counts = (
        rounded.value_counts(dropna=False)
        .sort_index()
        .rename_axis("shannon_diversity")
        .reset_index(name="n_households")
    )
    counts["pct"] = (100.0 * counts["n_households"] / len(df)).round(2)
    return counts


def portfolio_crosstab(df: pd.DataFrame, min_count: int | None = None) -> pd.DataFrame:
    """Crosstab of crop and livestock counts against Shannon diversity."""
    work = df.copy()
    work["shannon_rounded"] = round_shannon(work["shannon_diversity"])
    table = pd.crosstab(
        [work["n_crop_types"], work["n_livestock_types"]],
        work["shannon_rounded"],
    )
    table["row_total"] = table.sum(axis=1)
    if min_count is not None:
        table = table.loc[table["row_total"] >= min_count].copy()
    return table


def distinct_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct crop/livestock count pairs at each Shannon value."""
    work = df.copy()
    work["shannon_rounded"] = round_shannon(work["shannon_diversity"])
    rows = []
    for value, group in work.groupby("shannon_rounded", dropna=False):
        n_combos = group[["n_crop_types", "n_livestock_types"]].drop_duplicates().shape[0]
        rows.append(
            {
                "shannon_diversity": value,
                "n_households": len(group),
                "distinct_portfolio_combinations": n_combos,
            }
        )
    return pd.DataFrame(rows).sort_values("shannon_diversity").reset_index(drop=True)


def unique_shannon_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    label: str,
) -> pd.DataFrame:
    """For each grouping key, count unique Shannon values and list them."""
    work = df.copy()
    work["shannon_rounded"] = round_shannon(work["shannon_diversity"])
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_households=("shannon_rounded", "size"),
            n_unique_shannon=("shannon_rounded", "nunique"),
            shannon_values=(
                "shannon_rounded",
                lambda s: ", ".join(str(v) for v in sorted(s.unique())),
            ),
        )
        .reset_index()
    )
    grouped.insert(0, "grouping", label)
    return grouped


def determinism_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise whether Shannon is constant within candidate predictors."""
    checks = [
        (["n_crop_types"], "n_crop_types"),
        (["n_livestock_types"], "n_livestock_types"),
        (["n_crop_types", "n_livestock_types"], "n_crop_types_x_n_livestock_types"),
        (["n_active_groups"], "n_active_groups"),
    ]
    rows = []
    for cols, label in checks:
        detail = unique_shannon_by_group(df, cols, label)
        n_groups = len(detail)
        n_deterministic = int((detail["n_unique_shannon"] == 1).sum())
        rows.append(
            {
                "predictor": label,
                "n_groups": n_groups,
                "n_groups_with_single_shannon": n_deterministic,
                "n_groups_with_multiple_shannon": n_groups - n_deterministic,
                "fully_deterministic": n_deterministic == n_groups,
                "max_unique_shannon_within_group": int(detail["n_unique_shannon"].max()),
            }
        )
    return pd.DataFrame(rows)


def ln_k_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare observed Shannon values with ln(k) for integer richness k."""
    work = df.copy()
    work["shannon_rounded"] = round_shannon(work["shannon_diversity"])
    work["n_active_groups"] = work[AGRICULTURAL_HAS_VARS].sum(axis=1).astype(int)

    observed = (
        work.groupby("shannon_rounded", dropna=False)
        .agg(
            n_households=("shannon_rounded", "size"),
            modal_n_active_groups=("n_active_groups", lambda s: int(s.mode().iloc[0])),
            n_active_group_values=("n_active_groups", lambda s: ", ".join(str(v) for v in sorted(s.unique()))),
        )
        .reset_index()
        .rename(columns={"shannon_rounded": "shannon_diversity"})
    )

    rows = []
    for _, row in observed.iterrows():
        shannon = float(row["shannon_diversity"])
        matches = []
        for k in range(0, 10):
            expected = 0.0 if k <= 1 else math.log(k)
            if abs(shannon - expected) < 1e-5:
                matches.append(k)
        rows.append(
            {
                "shannon_diversity": shannon,
                "n_households": int(row["n_households"]),
                "modal_n_active_groups": int(row["modal_n_active_groups"]),
                "n_active_group_values": row["n_active_group_values"],
                "matches_ln_k_for_k": ", ".join(str(k) for k in matches) if matches else "",
                "expected_ln_of_modal_k": (
                    0.0
                    if row["modal_n_active_groups"] <= 1
                    else round(math.log(int(row["modal_n_active_groups"])), SHANNON_ROUND)
                ),
                "abs_diff_from_ln_modal_k": round(
                    abs(
                        shannon
                        - (
                            0.0
                            if row["modal_n_active_groups"] <= 1
                            else math.log(int(row["modal_n_active_groups"]))
                        )
                    ),
                    SHANNON_ROUND,
                ),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation among portfolio measures and active group count."""
    cols = PORTFOLIO_CORE_VARS + ["n_active_groups"]
    return df[cols].corr().round(3)


def write_report(
    analysis: pd.DataFrame,
    desc: pd.DataFrame,
    value_counts: pd.DataFrame,
    combinations: pd.DataFrame,
    determinism: pd.DataFrame,
    ln_compare: pd.DataFrame,
    corr: pd.DataFrame,
    crosstab_full_rows: int,
    crosstab_filtered_rows: int,
) -> Path:
    """Write a concise Markdown report distinguishing facts from recommendations."""
    n = len(analysis)
    fully_det_crop = bool(
        determinism.loc[
            determinism["predictor"] == "n_crop_types", "fully_deterministic"
        ].iloc[0]
    )
    fully_det_livestock = bool(
        determinism.loc[
            determinism["predictor"] == "n_livestock_types", "fully_deterministic"
        ].iloc[0]
    )
    fully_det_combo = bool(
        determinism.loc[
            determinism["predictor"] == "n_crop_types_x_n_livestock_types",
            "fully_deterministic",
        ].iloc[0]
    )
    fully_det_groups = bool(
        determinism.loc[
            determinism["predictor"] == "n_active_groups", "fully_deterministic"
        ].iloc[0]
    )

    combo_multi = int(
        determinism.loc[
            determinism["predictor"] == "n_crop_types_x_n_livestock_types",
            "n_groups_with_multiple_shannon",
        ].iloc[0]
    )
    groups_multi = int(
        determinism.loc[
            determinism["predictor"] == "n_active_groups",
            "n_groups_with_multiple_shannon",
        ].iloc[0]
    )

    max_combos = int(combinations["distinct_portfolio_combinations"].max())
    min_combos = int(combinations["distinct_portfolio_combinations"].min())

    ln_matches = ln_compare.loc[ln_compare["matches_ln_k_for_k"] != "", "shannon_diversity"].tolist()
    ln_nonmatches = ln_compare.loc[
        ln_compare["matches_ln_k_for_k"] == "", "shannon_diversity"
    ].tolist()

    # Provisional recommendation based on observed determinism evidence
    if fully_det_combo and not fully_det_crop and not fully_det_livestock:
        recommendation = (
            "Shannon diversity is constant within each crop and livestock count "
            "combination in this sample. That means it does not add independent "
            "portfolio-composition information beyond the joint count pair for "
            "classification or segmentation that already includes both counts. "
            "Retain it provisionally only if a single richness-style summary is "
            "preferred for interpretation. Otherwise prefer the two count variables "
            "and revisit redundancy during feature assessment."
        )
    elif fully_det_groups and not fully_det_combo:
        recommendation = (
            "Shannon diversity tracks the number of active agricultural group flags "
            "more closely than either count alone or their pair. Retain the counts "
            "as primary portfolio measures. Keep Shannon provisional only if a "
            "compact richness summary across crop and livestock groups is useful. "
            "Do not treat it as a conventional species-abundance diversity index."
        )
    elif not fully_det_combo and not fully_det_groups:
        recommendation = (
            "Shannon diversity is not a deterministic function of the crop count, "
            "livestock count, their combination, or the active group count alone. "
            "Retain it provisionally alongside the portfolio counts and revisit "
            "redundancy during later feature assessment. Do not assume it is "
            "interchangeable with either count measure."
        )
    else:
        recommendation = (
            "Review the determinism tables carefully. Retain n_crop_types and "
            "n_livestock_types as primary portfolio measures. Keep shannon_diversity "
            "provisional pending feature assessment rather than dropping it solely "
            "on correlation evidence."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Shannon Portfolio Investigation Report

Generated: {timestamp}

## What was investigated

Relationship between:

- `n_crop_types`
- `n_livestock_types`
- `shannon_diversity`

Sample: drought-exposed households from the prepared 2024 modelling frame (`n = {n}`).

Additional checks used the nine binary `has_*` agricultural group flags to construct `n_active_groups`, because Shannon was originally computed from those flags.

## Observed findings

### Sample and missingness

- Observations analysed: {n}
- Missing values in the three portfolio measures: none in this sample (see descriptive table)

### Descriptive summary

```
{desc.to_string(index=False)}
```

### Shannon value frequencies

```
{value_counts.to_string(index=False)}
```

### Distinct crop and livestock combinations at each Shannon value

```
{combinations.to_string(index=False)}
```

- Minimum distinct combinations at a Shannon value: {min_combos}
- Maximum distinct combinations at a Shannon value: {max_combos}

### Combined portfolio crosstab

- Full crosstab row combinations written to CSV: {crosstab_full_rows}
- Filtered crosstab rows with at least {MIN_COMBO_COUNT} households: {crosstab_filtered_rows}

### Determinism checks

```
{determinism.to_string(index=False)}
```

- Fully deterministic given `n_crop_types` alone: {fully_det_crop}
- Fully deterministic given `n_livestock_types` alone: {fully_det_livestock}
- Fully deterministic given `(n_crop_types, n_livestock_types)`: {fully_det_combo}
- Fully deterministic given `n_active_groups`: {fully_det_groups}
- Crop-livestock combinations with more than one Shannon value: {combo_multi}
- Active-group counts with more than one Shannon value: {groups_multi}

### Comparison with ln(k)

Observed Shannon values that match `ln(k)` for some integer k in 0..9: {ln_matches if ln_matches else "none"}

Observed Shannon values that do not match those `ln(k)` targets: {ln_nonmatches if ln_nonmatches else "none"}

```
{ln_compare.to_string(index=False)}
```

### Correlations

```
{corr.to_string()}
```

## Interpretation

Facts above show how Shannon varies with the portfolio counts. Interpretation should stay separate from those facts.

Shannon was constructed from equal-weight presence of agricultural group flags. Values near `ln(k)` are consistent with k equally weighted active groups. That is not the same as a classical Shannon index based on species abundance shares within a single taxonomic inventory.

If multiple crop and livestock count combinations share the same Shannon value, Shannon collapses distinct portfolio structures into a coarser richness summary. If a single count pair maps to more than one Shannon value, Shannon is carrying information not present in that pair alone.

## Limitations

- This investigation uses the drought-exposed analytical sample only.
- Determinism is assessed empirically within this sample. It is not a proof about other populations.
- `n_active_groups` is derived for audit from the same `has_*` flags used to build Shannon. It is not proposed here as a replacement feature.
- No modelling or feature-selection performance comparison was run.

## Recommended preparation decisions for review

1. Retain `n_crop_types` and `n_livestock_types` as primary agricultural portfolio measures.
2. {recommendation}
3. Do not recalculate or transform Shannon diversity.
4. Do not drop Shannon solely because its observed values resemble `ln(k)`.
5. Keep this decision open until feature assessment for segmentation and classification is completed separately.

## Output files

- `outputs/tables/02_portfolio_descriptive_summary.csv`
- `outputs/tables/02_shannon_value_counts.csv`
- `outputs/tables/02_portfolio_shannon_crosstab_full.csv`
- `outputs/tables/02_portfolio_shannon_crosstab_min10.csv`
- `outputs/tables/02_shannon_distinct_combinations.csv`
- `outputs/tables/02_shannon_determinism_summary.csv`
- `outputs/tables/02_shannon_unique_by_crop.csv`
- `outputs/tables/02_shannon_unique_by_livestock.csv`
- `outputs/tables/02_shannon_unique_by_crop_livestock.csv`
- `outputs/tables/02_shannon_unique_by_active_groups.csv`
- `outputs/tables/02_shannon_ln_k_comparison.csv`
- `outputs/tables/02_portfolio_correlations.csv`
"""

    path = REPORTS_DIR / "02_shannon_portfolio_investigation.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()

    required = PORTFOLIO_CORE_VARS + AGRICULTURAL_HAS_VARS
    analysis = load_analysis_2024(required_columns=required)
    analysis["n_active_groups"] = analysis[AGRICULTURAL_HAS_VARS].sum(axis=1).astype(int)

    print("=" * 72)
    print("02 Shannon portfolio investigation")
    print("=" * 72)
    print(f"Observations: {len(analysis)}")
    print(f"Variables loaded for checks: {len(required)} required + n_active_groups")

    desc = descriptive_summary(analysis)
    value_counts = shannon_value_counts(analysis)
    crosstab_full = portfolio_crosstab(analysis, min_count=None)
    crosstab_min10 = portfolio_crosstab(analysis, min_count=MIN_COMBO_COUNT)
    combinations = distinct_combinations(analysis)
    determinism = determinism_summary(analysis)
    by_crop = unique_shannon_by_group(analysis, ["n_crop_types"], "n_crop_types")
    by_livestock = unique_shannon_by_group(
        analysis, ["n_livestock_types"], "n_livestock_types"
    )
    by_combo = unique_shannon_by_group(
        analysis,
        ["n_crop_types", "n_livestock_types"],
        "n_crop_types_x_n_livestock_types",
    )
    by_groups = unique_shannon_by_group(
        analysis, ["n_active_groups"], "n_active_groups"
    )
    ln_compare = ln_k_comparison(analysis)
    corr = correlation_table(analysis)

    outputs = {
        "02_portfolio_descriptive_summary.csv": desc,
        "02_shannon_value_counts.csv": value_counts,
        "02_portfolio_shannon_crosstab_full.csv": crosstab_full.reset_index(),
        "02_portfolio_shannon_crosstab_min10.csv": crosstab_min10.reset_index(),
        "02_shannon_distinct_combinations.csv": combinations,
        "02_shannon_determinism_summary.csv": determinism,
        "02_shannon_unique_by_crop.csv": by_crop,
        "02_shannon_unique_by_livestock.csv": by_livestock,
        "02_shannon_unique_by_crop_livestock.csv": by_combo,
        "02_shannon_unique_by_active_groups.csv": by_groups,
        "02_shannon_ln_k_comparison.csv": ln_compare,
        "02_portfolio_correlations.csv": corr.reset_index().rename(
            columns={"index": "variable"}
        ),
    }

    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    report_path = write_report(
        analysis=analysis,
        desc=desc,
        value_counts=value_counts,
        combinations=combinations,
        determinism=determinism,
        ln_compare=ln_compare,
        corr=corr,
        crosstab_full_rows=len(crosstab_full),
        crosstab_filtered_rows=len(crosstab_min10),
    )
    print(f"Wrote {report_path}")

    print("\nDeterminism summary")
    print(determinism.to_string(index=False))
    print("\nDistinct portfolio combinations by Shannon value")
    print(combinations.to_string(index=False))
    print("\nln(k) comparison")
    print(ln_compare.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
