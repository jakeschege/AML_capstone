#!/usr/bin/env python3
"""Descriptive coping profiles across Spec A k=5 working segments.

Unweighted counts and household-weighted proportions for four- and three-class
targets. No classification fitting, re-clustering, Spec B, or k=4 comparison.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    EXPECTED_DROUGHT_N,
    LOGS_DIR,
    PREPARED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)

CLUSTER_LABELS = {
    0: "Family-owned legume-oriented crop households",
    1: "Own-name crop households",
    2: "Community-land livestock-oriented households",
    3: "Family-owned mixed crop and livestock households",
    4: "Family-owned cereal-oriented households",
}

FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]
THREE_ORDER = ["personal", "formal_or_informal", "did_nothing"]
SPARSE_N = 30


def unweighted_crosstab(
    df: pd.DataFrame, target: str, categories: list[str]
) -> pd.DataFrame:
    rows = []
    for c in range(5):
        sub = df.loc[df["cluster_k5"] == c]
        n_c = len(sub)
        for cat in categories:
            n = int((sub[target] == cat).sum())
            rows.append(
                {
                    "cluster_k5": c,
                    "cluster_label": CLUSTER_LABELS[c],
                    "coping_category": cat,
                    "n_households": n,
                    "pct_within_cluster": round(100.0 * n / max(n_c, 1), 2),
                    "pct_of_sample": round(100.0 * n / len(df), 2),
                    "cluster_n": n_c,
                }
            )
        rows.append(
            {
                "cluster_k5": c,
                "cluster_label": CLUSTER_LABELS[c],
                "coping_category": "TOTAL",
                "n_households": n_c,
                "pct_within_cluster": 100.0,
                "pct_of_sample": round(100.0 * n_c / len(df), 2),
                "cluster_n": n_c,
            }
        )
    # Overall margins
    for cat in categories:
        n = int((df[target] == cat).sum())
        rows.append(
            {
                "cluster_k5": -1,
                "cluster_label": "All households",
                "coping_category": cat,
                "n_households": n,
                "pct_within_cluster": round(100.0 * n / len(df), 2),
                "pct_of_sample": round(100.0 * n / len(df), 2),
                "cluster_n": len(df),
            }
        )
    return pd.DataFrame(rows)


def weighted_props(
    df: pd.DataFrame, target: str, categories: list[str], weight_col: str
) -> pd.DataFrame:
    rows = []
    total_w = float(df[weight_col].sum())
    for c in range(5):
        sub = df.loc[df["cluster_k5"] == c]
        w_c = float(sub[weight_col].sum())
        for cat in categories:
            w = float(sub.loc[sub[target] == cat, weight_col].sum())
            rows.append(
                {
                    "cluster_k5": c,
                    "cluster_label": CLUSTER_LABELS[c],
                    "coping_category": cat,
                    "weighted_sum": w,
                    "pct_within_cluster_weighted": round(100.0 * w / max(w_c, 1e-12), 2),
                    "pct_of_sample_weighted": round(100.0 * w / max(total_w, 1e-12), 2),
                    "cluster_weight_sum": w_c,
                    "weight_variable": weight_col,
                }
            )
    for cat in categories:
        w = float(df.loc[df[target] == cat, weight_col].sum())
        rows.append(
            {
                "cluster_k5": -1,
                "cluster_label": "All households",
                "coping_category": cat,
                "weighted_sum": w,
                "pct_within_cluster_weighted": round(100.0 * w / max(total_w, 1e-12), 2),
                "pct_of_sample_weighted": round(100.0 * w / max(total_w, 1e-12), 2),
                "cluster_weight_sum": total_w,
                "weight_variable": weight_col,
            }
        )
    return pd.DataFrame(rows)


def formal_focus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in range(5):
        sub = df.loc[df["cluster_k5"] == c]
        n = int((sub["coping_category"] == "formal").sum())
        rows.append(
            {
                "cluster_k5": c,
                "cluster_label": CLUSTER_LABELS[c],
                "n_formal": n,
                "cluster_n": len(sub),
                "pct_formal_unweighted": round(100.0 * n / max(len(sub), 1), 2),
                "sparse_under_n30": n < SPARSE_N,
            }
        )
    n_all = int((df["coping_category"] == "formal").sum())
    rows.append(
        {
            "cluster_k5": -1,
            "cluster_label": "All households",
            "n_formal": n_all,
            "cluster_n": len(df),
            "pct_formal_unweighted": round(100.0 * n_all / len(df), 2),
            "sparse_under_n30": False,
        }
    )
    return pd.DataFrame(rows)


def association_summary(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df["cluster_k5"], df["coping_category"])
    ct = ct.reindex(columns=FOUR_ORDER)
    chi2, p, dof, expected = chi2_contingency(ct.values)
    n = ct.values.sum()
    r, k = ct.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
    n_small_expected = int((expected < 5).sum())
    formal_idx = list(ct.columns).index("formal")
    formal_expected = expected[:, formal_idx]
    formal_observed = ct["formal"].to_numpy()
    return pd.DataFrame(
        [
            {
                "test": "chi_square_independence",
                "target": "coping_category",
                "grouping": "cluster_k5",
                "chi2": round(float(chi2), 4),
                "df": int(dof),
                "p_value": float(p),
                "cramers_v": round(float(cramers_v), 4),
                "n_cells_expected_lt_5": n_small_expected,
                "formal_observed_min": int(formal_observed.min()),
                "formal_observed_max": int(formal_observed.max()),
                "formal_expected_min": round(float(formal_expected.min()), 2),
                "warning": (
                    "Formal observed counts are sparse (roughly 9–19 per cluster); "
                    "p-value and Cramer’s V are secondary and do not establish "
                    "causal segment effects."
                ),
            }
        ]
    )


def profile_notes(
    unweighted: pd.DataFrame, weighted: pd.DataFrame
) -> list[str]:
    notes = []
    overall = {
        cat: float(
            unweighted.loc[
                (unweighted["cluster_k5"] == -1)
                & (unweighted["coping_category"] == cat),
                "pct_within_cluster",
            ].iloc[0]
        )
        for cat in FOUR_ORDER
    }
    for c in range(5):
        parts = []
        for cat in FOUR_ORDER:
            pct_u = float(
                unweighted.loc[
                    (unweighted["cluster_k5"] == c)
                    & (unweighted["coping_category"] == cat),
                    "pct_within_cluster",
                ].iloc[0]
            )
            pct_w = float(
                weighted.loc[
                    (weighted["cluster_k5"] == c)
                    & (weighted["coping_category"] == cat),
                    "pct_within_cluster_weighted",
                ].iloc[0]
            )
            delta = pct_u - overall[cat]
            if abs(delta) >= 5:
                direction = "higher" if delta > 0 else "lower"
                parts.append(
                    f"{cat} {direction} than sample "
                    f"({pct_u:.1f}% unweighted / {pct_w:.1f}% weighted vs "
                    f"{overall[cat]:.1f}% overall)"
                )
        if not parts:
            parts.append("within-cluster coping mix close to the overall sample (±5 pp)")
        notes.append(f"- **Cluster {c} — {CLUSTER_LABELS[c]}:** " + "; ".join(parts) + ".")
    return notes


def main() -> None:
    ensure_output_dirs()
    assign = pd.read_csv(TABLES_DIR / "12_s1_spec_a_assignments_wide.csv")
    targets = pd.read_parquet(PREPARED_DIR / "09_y_targets.parquet")
    prepared = pd.read_parquet(PREPARED_DIR / "09_prepared_analysis_frame.parquet")

    df = (
        assign[["respondent_id", "cluster_k5"]]
        .merge(
            targets[
                [
                    "respondent_id",
                    "coping_category",
                    "coping_category_collapsed",
                ]
            ],
            on="respondent_id",
            how="inner",
        )
        .merge(
            prepared[["respondent_id", "household_weight", "individual_weight"]],
            on="respondent_id",
            how="inner",
        )
    )
    if len(df) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected merged n={len(df)}")
    if df[["coping_category", "coping_category_collapsed", "household_weight"]].isna().any().any():
        raise ValueError("Missing targets or household_weight")

    counts4 = unweighted_crosstab(df, "coping_category", FOUR_ORDER)
    props4 = weighted_props(df, "coping_category", FOUR_ORDER, "household_weight")
    counts3 = unweighted_crosstab(df, "coping_category_collapsed", THREE_ORDER)
    # rename column for three-class clarity in file
    counts3 = counts3.rename(columns={"coping_category": "coping_category_collapsed"})
    props3 = weighted_props(
        df, "coping_category_collapsed", THREE_ORDER, "household_weight"
    ).rename(columns={"coping_category": "coping_category_collapsed"})
    formal = formal_focus(df)
    assoc = association_summary(df)
    notes = profile_notes(counts4, props4)

    outputs = {
        "17_coping_counts_unweighted_fourclass.csv": counts4,
        "17_coping_props_weighted_fourclass.csv": props4,
        "17_coping_counts_unweighted_threeclass.csv": counts3,
        "17_coping_props_weighted_threeclass.csv": props3,
        "17_formal_by_cluster.csv": formal,
        "17_association_summary.csv": assoc,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    # Compact display tables for report
    pivot_u = (
        counts4.loc[counts4["coping_category"] != "TOTAL"]
        .pivot_table(
            index=["cluster_k5", "cluster_label"],
            columns="coping_category",
            values="pct_within_cluster",
            aggfunc="first",
        )
        .reindex(columns=FOUR_ORDER)
    )
    pivot_n = (
        counts4.loc[counts4["coping_category"] != "TOTAL"]
        .pivot_table(
            index=["cluster_k5", "cluster_label"],
            columns="coping_category",
            values="n_households",
            aggfunc="first",
        )
        .reindex(columns=FOUR_ORDER)
    )
    pivot_w = props4.pivot_table(
        index=["cluster_k5", "cluster_label"],
        columns="coping_category",
        values="pct_within_cluster_weighted",
        aggfunc="first",
    ).reindex(columns=FOUR_ORDER)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Coping Profiles by Spec A k=5 Segment (Script 17)

Generated: {timestamp}

## Scope

Descriptive coping distributions across the locked Spec A five-cluster working typology.

- Assignments: Spec A `cluster_k5`
- Targets: four-class `coping_category` and three-class `coping_category_collapsed`
- Weights: `household_weight` for population-oriented proportions; unweighted counts retained
- No classification models, no re-clustering, no Spec B, no k=4 comparison

## Unweighted counts (four-class)

```
{pivot_n.to_string()}
```

## Unweighted within-cluster percentages (four-class)

```
{pivot_u.round(1).to_string()}
```

## Household-weighted within-cluster percentages (four-class)

```
{pivot_w.round(1).to_string()}
```

## Formal coping by cluster

```
{formal.to_string(index=False)}
```

All five clusters have formal n &lt; 30. Formal-class comparisons across segments are descriptive only and should not be over-interpreted.

## Three-class collapsed target

Unweighted percentages:

```
{counts3.loc[counts3['coping_category_collapsed'] != 'TOTAL'].pivot_table(index=['cluster_k5','cluster_label'], columns='coping_category_collapsed', values='pct_within_cluster', aggfunc='first').reindex(columns=THREE_ORDER).round(1).to_string()}
```

Household-weighted percentages:

```
{props3.pivot_table(index=['cluster_k5','cluster_label'], columns='coping_category_collapsed', values='pct_within_cluster_weighted', aggfunc='first').reindex(columns=THREE_ORDER).round(1).to_string()}
```

## Association summary (secondary)

```
{assoc.to_string(index=False)}
```

## Descriptive profile notes

{chr(10).join(notes)}

These notes compare each cluster to the overall sample using a ±5 percentage-point threshold on unweighted shares. They do not imply that segments predict coping or that differences are causal.

## Deferred

- k=4 coping sensitivity
- Classification model fitting
- Spec B coping cross-tabulation

## Output files

- `outputs/tables/17_coping_counts_unweighted_fourclass.csv`
- `outputs/tables/17_coping_props_weighted_fourclass.csv`
- `outputs/tables/17_coping_counts_unweighted_threeclass.csv`
- `outputs/tables/17_coping_props_weighted_threeclass.csv`
- `outputs/tables/17_formal_by_cluster.csv`
- `outputs/tables/17_association_summary.csv`
"""
    path = REPORTS_DIR / "17_coping_profile_by_segment.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")

    design = LOGS_DIR / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    note = """

## 12. Coping profiling (Script 17) — descriptive

Descriptive coping distributions across Spec A k=5 working clusters completed (unweighted counts and household-weighted proportions; four- and three-class targets). Classification models not fitted. k=4 coping sensitivity deferred.

Report: `outputs/reports/17_coping_profile_by_segment.md`
"""
    if "## 12. Coping profiling (Script 17)" not in text:
        design.write_text(text.rstrip() + note, encoding="utf-8")
        print(f"Updated {design}")

    print("\nUnweighted % within cluster")
    print(pivot_u.round(1).to_string())
    print("\nWeighted % within cluster")
    print(pivot_w.round(1).to_string())
    print("\nFormal")
    print(formal.to_string(index=False))
    print("\nAssociation")
    print(assoc.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
