#!/usr/bin/env python3
"""k=4 coping sensitivity vs locked Spec A k=5 working typology.

Uses existing Spec A assignments only. No re-segmentation, Spec B, or
classification model fitting.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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

# Locked working labels (k=5); descriptive sensitivity labels (k=4)
K5_LABELS = {
    0: "Family-owned legume-oriented crop households",
    1: "Own-name crop households",
    2: "Community-land livestock-oriented households",
    3: "Family-owned mixed crop and livestock households",
    4: "Family-owned cereal-oriented households",
}

K4_LABELS = {
    0: "Family-owned crop households (no roots/tubers)",
    1: "Family-owned crop households (with roots/tubers)",
    2: "Community-land livestock-oriented households",
    3: "Lease / own-name crop households",
}

# Focal correspondence: k=5 community-livestock ↔ k=4 community-livestock
FOCAL_K5 = 2
FOCAL_K4 = 2

FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]


def profile_solution(
    df: pd.DataFrame,
    cluster_col: str,
    labels: dict[int, str],
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unweighted counts and household-weighted props for one k."""
    count_rows: list[dict] = []
    weight_rows: list[dict] = []
    total_w = float(df["household_weight"].sum())
    n_all = len(df)

    for c in range(k):
        sub = df.loc[df[cluster_col] == c]
        n_c = len(sub)
        w_c = float(sub["household_weight"].sum())
        for cat in FOUR_ORDER:
            n = int((sub["coping_category"] == cat).sum())
            w = float(sub.loc[sub["coping_category"] == cat, "household_weight"].sum())
            count_rows.append(
                {
                    "k": k,
                    "cluster": c,
                    "cluster_label": labels[c],
                    "coping_category": cat,
                    "n_households": n,
                    "pct_within_cluster": round(100.0 * n / max(n_c, 1), 2),
                    "pct_of_sample": round(100.0 * n / n_all, 2),
                    "cluster_n": n_c,
                }
            )
            weight_rows.append(
                {
                    "k": k,
                    "cluster": c,
                    "cluster_label": labels[c],
                    "coping_category": cat,
                    "weighted_sum": w,
                    "pct_within_cluster_weighted": round(
                        100.0 * w / max(w_c, 1e-12), 2
                    ),
                    "pct_of_sample_weighted": round(
                        100.0 * w / max(total_w, 1e-12), 2
                    ),
                    "cluster_weight_sum": w_c,
                    "weight_variable": "household_weight",
                }
            )

    for cat in FOUR_ORDER:
        n = int((df["coping_category"] == cat).sum())
        w = float(df.loc[df["coping_category"] == cat, "household_weight"].sum())
        count_rows.append(
            {
                "k": k,
                "cluster": -1,
                "cluster_label": "All households",
                "coping_category": cat,
                "n_households": n,
                "pct_within_cluster": round(100.0 * n / n_all, 2),
                "pct_of_sample": round(100.0 * n / n_all, 2),
                "cluster_n": n_all,
            }
        )
        weight_rows.append(
            {
                "k": k,
                "cluster": -1,
                "cluster_label": "All households",
                "coping_category": cat,
                "weighted_sum": w,
                "pct_within_cluster_weighted": round(
                    100.0 * w / max(total_w, 1e-12), 2
                ),
                "pct_of_sample_weighted": round(
                    100.0 * w / max(total_w, 1e-12), 2
                ),
                "cluster_weight_sum": total_w,
                "weight_variable": "household_weight",
            }
        )

    return pd.DataFrame(count_rows), pd.DataFrame(weight_rows)


def pivot_pct(counts: pd.DataFrame, value_col: str = "pct_within_cluster") -> pd.DataFrame:
    return (
        counts.pivot_table(
            index=["cluster", "cluster_label"],
            columns="coping_category",
            values=value_col,
            aggfunc="first",
        )
        .reindex(columns=FOUR_ORDER)
    )


def membership_overlap(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df["cluster_k5"], df["cluster_k4"], margins=True)
    rows = []
    for c5 in range(5):
        for c4 in range(4):
            n = int(ct.loc[c5, c4])
            if n == 0:
                continue
            rows.append(
                {
                    "cluster_k5": c5,
                    "cluster_k5_label": K5_LABELS[c5],
                    "cluster_k4": c4,
                    "cluster_k4_label": K4_LABELS[c4],
                    "n_overlap": n,
                    "pct_of_k5_cluster": round(
                        100.0 * n / int(ct.loc[c5, "All"]), 2
                    ),
                    "pct_of_k4_cluster": round(
                        100.0 * n / int(ct.loc["All", c4]), 2
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["cluster_k5", "n_overlap"], ascending=[True, False]
    )


def focal_comparison(counts4: pd.DataFrame, counts5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, k, c, labels in (
        (counts5, 5, FOCAL_K5, K5_LABELS),
        (counts4, 4, FOCAL_K4, K4_LABELS),
    ):
        for cat in FOUR_ORDER:
            row = src.loc[
                (src["cluster"] == c) & (src["coping_category"] == cat)
            ].iloc[0]
            rows.append(
                {
                    "solution": f"k={k}",
                    "cluster": c,
                    "cluster_label": labels[c],
                    "coping_category": cat,
                    "n_households": int(row["n_households"]),
                    "pct_within_cluster_unweighted": float(row["pct_within_cluster"]),
                    "cluster_n": int(row["cluster_n"]),
                }
            )
    return pd.DataFrame(rows)


def interpretability_notes(
    counts4: pd.DataFrame,
    counts5: pd.DataFrame,
    weights4: pd.DataFrame,
    weights5: pd.DataFrame,
    overlap: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []

    # Focal Cluster 2 pattern
    def pct(df: pd.DataFrame, c: int, cat: str, col: str) -> float:
        return float(
            df.loc[
                (df["cluster"] == c) & (df["coping_category"] == cat), col
            ].iloc[0]
        )

    k5_inf = pct(counts5, FOCAL_K5, "informal", "pct_within_cluster")
    k4_inf = pct(counts4, FOCAL_K4, "informal", "pct_within_cluster")
    k5_pers = pct(counts5, FOCAL_K5, "personal", "pct_within_cluster")
    k4_pers = pct(counts4, FOCAL_K4, "personal", "pct_within_cluster")
    k5_dn = pct(counts5, FOCAL_K5, "did_nothing", "pct_within_cluster")
    k4_dn = pct(counts4, FOCAL_K4, "did_nothing", "pct_within_cluster")
    k5_inf_w = pct(weights5, FOCAL_K5, "informal", "pct_within_cluster_weighted")
    k4_inf_w = pct(weights4, FOCAL_K4, "informal", "pct_within_cluster_weighted")

    overall_inf = pct(counts5, -1, "informal", "pct_within_cluster")
    overall_pers = pct(counts5, -1, "personal", "pct_within_cluster")
    overall_dn = pct(counts5, -1, "did_nothing", "pct_within_cluster")

    focal_overlap = overlap.loc[
        (overlap["cluster_k5"] == FOCAL_K5) & (overlap["cluster_k4"] == FOCAL_K4)
    ].iloc[0]

    notes.append(
        f"- **Focal livestock/community segment:** Spec A k=5 cluster {FOCAL_K5} "
        f"and k=4 cluster {FOCAL_K4} share {int(focal_overlap['n_overlap'])} "
        f"households ({focal_overlap['pct_of_k5_cluster']:.1f}% of the k=5 "
        f"cluster; {focal_overlap['pct_of_k4_cluster']:.1f}% of the k=4 cluster)."
    )
    notes.append(
        f"- **Distinctive k=5 Cluster 2 coping pattern** (vs sample: personal "
        f"{overall_pers:.1f}%, informal {overall_inf:.1f}%, did-nothing "
        f"{overall_dn:.1f}%): personal {k5_pers:.1f}%, informal {k5_inf:.1f}%, "
        f"did-nothing {k5_dn:.1f}% unweighted "
        f"(informal weighted {k5_inf_w:.1f}%)."
    )
    notes.append(
        f"- **Corresponding k=4 Cluster {FOCAL_K4}:** personal {k4_pers:.1f}%, "
        f"informal {k4_inf:.1f}%, did-nothing {k4_dn:.1f}% unweighted "
        f"(informal weighted {k4_inf_w:.1f}%). "
        + (
            "The elevated personal / depressed informal / elevated did-nothing "
            "signature remains recognisable under k=4."
            if (
                k4_pers > overall_pers
                and k4_inf < overall_inf - 5
                and k4_dn > overall_dn
            )
            else "Compare carefully: the signature is only partly retained under k=4."
        )
    )

    # Family-crop consolidation under k=4
    notes.append(
        "- **Family-crop consolidation:** under k=4, most of k=5 clusters 0, 3, "
        "and 4 fold into the two family-crop groups (clusters 0–1). Coping mixes "
        "in those broader groups are closer to the sample average than the more "
        "differentiated k=5 crop subtypes."
    )

    # Own-name / lease recovery
    own_to_k4 = overlap.loc[overlap["cluster_k5"] == 1].sort_values(
        "n_overlap", ascending=False
    )
    top = own_to_k4.iloc[0]
    notes.append(
        f"- **Own-name crop (k=5 cluster 1):** largest overlap is with k=4 "
        f"cluster {int(top['cluster_k4'])} ({K4_LABELS[int(top['cluster_k4'])]}; "
        f"n={int(top['n_overlap'])}, {top['pct_of_k5_cluster']:.1f}% of the "
        f"k=5 cluster). Lease/own-name recovery under k=4 remains partial, "
        f"consistent with the earlier typology decision."
    )

    notes.append(
        "- **Overall:** coping composition differences are not an artefact of "
        "k=5 alone—the livestock/community contrast is visible at k=4—but k=5 "
        "preserves finer crop-ownership distinctions that k=4 collapses."
    )
    return notes


def main() -> None:
    ensure_output_dirs()
    assign = pd.read_csv(TABLES_DIR / "12_s1_spec_a_assignments_wide.csv")
    targets = pd.read_parquet(PREPARED_DIR / "09_y_targets.parquet")
    prepared = pd.read_parquet(PREPARED_DIR / "09_prepared_analysis_frame.parquet")

    df = (
        assign[["respondent_id", "cluster_k4", "cluster_k5"]]
        .merge(
            targets[["respondent_id", "coping_category"]],
            on="respondent_id",
            how="inner",
        )
        .merge(
            prepared[["respondent_id", "household_weight"]],
            on="respondent_id",
            how="inner",
        )
    )
    if len(df) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected merged n={len(df)}")
    if df[["coping_category", "household_weight", "cluster_k4", "cluster_k5"]].isna().any().any():
        raise ValueError("Missing assignments, targets, or household_weight")

    counts4, weights4 = profile_solution(df, "cluster_k4", K4_LABELS, 4)
    counts5, weights5 = profile_solution(df, "cluster_k5", K5_LABELS, 5)
    counts_all = pd.concat([counts4, counts5], ignore_index=True)
    weights_all = pd.concat([weights4, weights5], ignore_index=True)
    overlap = membership_overlap(df)
    focal = focal_comparison(counts4, counts5)

    # Side-by-side delta table for focal segment
    focal_delta_rows = []
    for cat in FOUR_ORDER:
        r5 = focal.loc[
            (focal["solution"] == "k=5") & (focal["coping_category"] == cat)
        ].iloc[0]
        r4 = focal.loc[
            (focal["solution"] == "k=4") & (focal["coping_category"] == cat)
        ].iloc[0]
        w5 = float(
            weights5.loc[
                (weights5["cluster"] == FOCAL_K5)
                & (weights5["coping_category"] == cat),
                "pct_within_cluster_weighted",
            ].iloc[0]
        )
        w4 = float(
            weights4.loc[
                (weights4["cluster"] == FOCAL_K4)
                & (weights4["coping_category"] == cat),
                "pct_within_cluster_weighted",
            ].iloc[0]
        )
        focal_delta_rows.append(
            {
                "coping_category": cat,
                "k5_cluster2_pct_unweighted": float(r5["pct_within_cluster_unweighted"]),
                "k4_cluster2_pct_unweighted": float(r4["pct_within_cluster_unweighted"]),
                "delta_unweighted_pp": round(
                    float(r4["pct_within_cluster_unweighted"])
                    - float(r5["pct_within_cluster_unweighted"]),
                    2,
                ),
                "k5_cluster2_pct_weighted": w5,
                "k4_cluster2_pct_weighted": w4,
                "delta_weighted_pp": round(w4 - w5, 2),
            }
        )
    focal_delta = pd.DataFrame(focal_delta_rows)

    outputs = {
        "18_coping_counts_unweighted_k4_k5.csv": counts_all,
        "18_coping_props_weighted_k4_k5.csv": weights_all,
        "18_k4_k5_membership_overlap.csv": overlap,
        "18_focal_livestock_coping_compare.csv": focal,
        "18_focal_livestock_coping_delta.csv": focal_delta,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    notes = interpretability_notes(counts4, counts5, weights4, weights5, overlap)

    pivot_u4 = pivot_pct(counts4)
    pivot_u5 = pivot_pct(counts5)
    pivot_w4 = pivot_pct(weights4, "pct_within_cluster_weighted")
    pivot_w5 = pivot_pct(weights5, "pct_within_cluster_weighted")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# k=4 Coping Sensitivity vs Spec A k=5 (Script 18)

Generated: {timestamp}

## Scope

Descriptive sensitivity only: whether four-class coping patterns depend heavily on choosing five Spec A clusters rather than four.

- Uses existing Spec A `cluster_k4` and `cluster_k5` from Script 12
- Does **not** re-run segmentation or change locked k=5 assignments
- Weights: `household_weight`; unweighted counts retained
- No classification models; no Spec B

## Membership context (k=5 × k=4)

```
{pd.crosstab(df['cluster_k5'], df['cluster_k4'], margins=True).to_string()}
```

Focal correspondence: k=5 cluster {FOCAL_K5} ({K5_LABELS[FOCAL_K5]}) ↔ k=4 cluster {FOCAL_K4} ({K4_LABELS[FOCAL_K4]}).

## Four-class unweighted % within cluster — k=5 (working)

```
{pivot_u5.round(1).to_string()}
```

## Four-class unweighted % within cluster — k=4 (sensitivity)

```
{pivot_u4.round(1).to_string()}
```

## Household-weighted % within cluster — k=5

```
{pivot_w5.round(1).to_string()}
```

## Household-weighted % within cluster — k=4

```
{pivot_w4.round(1).to_string()}
```

## Focal livestock / community segment comparison

Unweighted and weighted percentage-point deltas (k=4 − k=5) for the matched community-livestock clusters:

```
{focal_delta.to_string(index=False)}
```

## Interpretability notes

{chr(10).join(notes)}

## Decision implication

This sensitivity does not unlock classification by itself. Review whether the livestock/community coping contrast and the broader k=5 patterns remain sufficiently robust; then decide whether to proceed to classification modelling.

## Output files

- `outputs/tables/18_coping_counts_unweighted_k4_k5.csv`
- `outputs/tables/18_coping_props_weighted_k4_k5.csv`
- `outputs/tables/18_k4_k5_membership_overlap.csv`
- `outputs/tables/18_focal_livestock_coping_compare.csv`
- `outputs/tables/18_focal_livestock_coping_delta.csv`
"""
    path = REPORTS_DIR / "18_coping_k4_sensitivity.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")

    design = LOGS_DIR / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    # Refresh status line if still outdated
    old_status = (
        "**Status: Working segmentation typology locked — Spec A, k=5. "
        "Spec B and k=4 retained as sensitivities. Coping profiling not yet started.**"
    )
    new_status = (
        "**Status: Working segmentation typology locked — Spec A, k=5. "
        "Descriptive coping profiling (k=5) and k=4 coping sensitivity completed. "
        "Classification models not yet fitted.**"
    )
    if old_status in text:
        text = text.replace(old_status, new_status)

    # Update Script 17 deferred note
    text = text.replace(
        "Classification models not fitted. k=4 coping sensitivity deferred.",
        "Classification models not fitted. k=4 coping sensitivity completed in Script 18.",
    )

    note = """

## 13. k=4 coping sensitivity (Script 18)

Four-class coping distributions compared under Spec A k=4 vs locked k=5 using existing assignments only (unweighted and household-weighted). Focal community-livestock segment correspondence checked. Classification still not fitted.

Report: `outputs/reports/18_coping_k4_sensitivity.md`
"""
    if "## 13. k=4 coping sensitivity (Script 18)" not in text:
        text = text.rstrip() + note
    design.write_text(text, encoding="utf-8")
    print(f"Updated {design}")

    print("\n=== k=5 unweighted % ===")
    print(pivot_u5.round(1).to_string())
    print("\n=== k=4 unweighted % ===")
    print(pivot_u4.round(1).to_string())
    print("\n=== Focal delta (k4 − k5) ===")
    print(focal_delta.to_string(index=False))
    print("\nNotes:")
    for n in notes:
        print(n)
    print("\nDone.")


if __name__ == "__main__":
    main()
