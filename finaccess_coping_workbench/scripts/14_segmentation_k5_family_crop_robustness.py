#!/usr/bin/env python3
"""Focused robustness check for k=5 family crop subtypes (Spec A).

Investigates legumes-heavy vs cereals-without-legumes membership transitions,
profile persistence, medoid variation, boundary households, and comparison
with k=4 family crop groups. No Spec B, coping, or final k selection.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gower
import numpy as np
import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    AGRICULTURAL_HAS_VARS,
    EXPECTED_DROUGHT_N,
    LAND_OWNERSHIP_VARS,
    PORTFOLIO_COUNT_VARS,
    PREPARED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)
from segmentation_utils import pam_precomputed  # noqa: E402

RANDOM_SEED = 42
N_REPLICATES = 50
SUBSAMPLE_FRACTION = 0.80
PAM_MAX_ITER = 100

# S1 k=5 family crop subtypes of interest
K5_LEGUMES = 0
K5_CEREALS_NO_LEGUMES = 4
K5_FOCUS = {K5_LEGUMES, K5_CEREALS_NO_LEGUMES}

# S1 k=4 broader family crop groups
K4_FAMILY_NO_ROOTS = 0
K4_FAMILY_ROOTS = 1
K4_FAMILY = {K4_FAMILY_NO_ROOTS, K4_FAMILY_ROOTS}

NOTES = {
    ("k5", K5_LEGUMES): "family_legumes_heavy",
    ("k5", K5_CEREALS_NO_LEGUMES): "family_cereals_without_legumes",
    ("k4", K4_FAMILY_NO_ROOTS): "family_crops_no_roots",
    ("k4", K4_FAMILY_ROOTS): "family_crops_with_roots",
}

DEFINING_TRAITS = [
    "has_cereals",
    "has_legumes",
    "has_roots_tubers",
    "has_cash_crops",
    "has_horticulture",
    "has_other_livestock",
    "n_crop_types",
    "n_livestock_types",
    "land_ownership_own_under_your_name",
    "land_ownership_family_owned_e_g_inherited_family_land",
    "land_ownership_community_public_owned_e_g_pastoralists",
    "land_ownership_lease_rent_you_dont_own",
]


def spec_a_features() -> list[str]:
    return (
        list(AGRICULTURAL_HAS_VARS)
        + list(PORTFOLIO_COUNT_VARS)
        + list(LAND_OWNERSHIP_VARS)
    )


def categorical_mask(feature_cols: list[str]) -> np.ndarray:
    return np.array([c not in PORTFOLIO_COUNT_VARS for c in feature_cols], dtype=bool)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def max_jaccard_matches(
    base_labels: np.ndarray, resample_labels: np.ndarray, k: int
) -> dict[int, tuple[int, float]]:
    out = {}
    for c in range(k):
        base = base_labels == c
        best_j, best_r = -1.0, -1
        for r in range(k):
            j = jaccard(base, resample_labels == r)
            if j > best_j:
                best_j, best_r = j, r
        out[c] = (best_r, best_j)
    return out


def main() -> None:
    ensure_output_dirs()
    feature_cols = spec_a_features()
    X_full = pd.read_parquet(PREPARED_DIR / "09_X_segmentation.parquet")
    assign = pd.read_csv(TABLES_DIR / "12_s1_spec_a_assignments_wide.csv")
    profiles = pd.read_csv(TABLES_DIR / "12_s1_spec_a_cluster_profiles.csv")
    medoids_s1 = pd.read_csv(TABLES_DIR / "12_s1_spec_a_medoids.csv")

    X = X_full.merge(assign, on="respondent_id", how="inner")
    if len(X) != EXPECTED_DROUGHT_N:
        raise ValueError("Unexpected merged sample size")
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    cat_features = categorical_mask(feature_cols)
    distance = gower.gower_matrix(
        X[feature_cols].to_numpy(dtype=float), cat_features=cat_features
    )
    if np.isnan(distance).any():
        raise ValueError("Gower distance contains NaNs")

    n = len(X)
    subsample_n = int(round(SUBSAMPLE_FRACTION * n))
    rng = np.random.default_rng(RANDOM_SEED)
    replicate_indices = [
        np.sort(rng.choice(n, size=subsample_n, replace=False))
        for _ in range(N_REPLICATES)
    ]

    # Full-sample S1 membership
    lab5 = X["cluster_k5"].to_numpy(dtype=int)
    lab4 = X["cluster_k4"].to_numpy(dtype=int)
    focus_mask = np.isin(lab5, list(K5_FOCUS))
    focus_ids = X.loc[focus_mask, "respondent_id"].tolist()

    # --- Boundary households (full sample, k=5 medoids) ---
    m5 = medoids_s1.loc[medoids_s1["k"] == 5].set_index("cluster")
    idx_leg = int(m5.loc[K5_LEGUMES, "medoid_index"])
    idx_cer = int(m5.loc[K5_CEREALS_NO_LEGUMES, "medoid_index"])
    d_leg = distance[:, idx_leg]
    d_cer = distance[:, idx_cer]
    # Among focus households, how close to each medoid
    boundary_rows = []
    for i in np.where(focus_mask)[0]:
        dl, dc = float(d_leg[i]), float(d_cer[i])
        gap = abs(dl - dc)
        assigned = int(lab5[i])
        nearer = K5_LEGUMES if dl <= dc else K5_CEREALS_NO_LEGUMES
        boundary_rows.append(
            {
                "respondent_id": X.iloc[i]["respondent_id"],
                "s1_k5_cluster": assigned,
                "s1_k5_note": NOTES[("k5", assigned)],
                "dist_to_legumes_medoid": round(dl, 6),
                "dist_to_cereals_no_legumes_medoid": round(dc, 6),
                "abs_dist_gap": round(gap, 6),
                "nearer_medoid_cluster": nearer,
                "nearer_matches_assignment": nearer == assigned,
                "has_cereals": int(X.iloc[i]["has_cereals"]),
                "has_legumes": int(X.iloc[i]["has_legumes"]),
                "has_roots_tubers": int(X.iloc[i]["has_roots_tubers"]),
                "n_crop_types": float(X.iloc[i]["n_crop_types"]),
                "family": int(
                    X.iloc[i]["land_ownership_family_owned_e_g_inherited_family_land"]
                ),
                "s1_k4_cluster": int(lab4[i]),
            }
        )
    boundary_df = pd.DataFrame(boundary_rows)
    # Flag ambiguous: gap below 10th percentile among focus
    gap_cut = float(boundary_df["abs_dist_gap"].quantile(0.10))
    boundary_df["is_boundary_ambiguous"] = boundary_df["abs_dist_gap"] <= gap_cut

    # Cross-tab S1 k=5 focus vs k=4 family groups
    crosstab = (
        pd.crosstab(
            X.loc[focus_mask, "cluster_k5"].map(
                lambda c: NOTES[("k5", int(c))]
            ),
            X.loc[focus_mask, "cluster_k4"].map(
                lambda c: NOTES.get(("k4", int(c)), f"k4_cluster_{int(c)}")
            ),
            margins=True,
        )
        .reset_index()
    )

    # --- Resample transition / persistence loop ---
    transition_rows = []
    merge_rows = []
    profile_rows = []
    medoid_feature_rows = []

    s1_means = {}
    for c in K5_FOCUS:
        for trait in DEFINING_TRAITS:
            val = profiles.loc[
                (profiles["k"] == 5)
                & (profiles["cluster"] == c)
                & (profiles["variable"] == trait),
                "mean",
            ]
            s1_means[(c, trait)] = float(val.iloc[0]) if len(val) else np.nan

    profile_acc: dict[tuple[int, str], list[float]] = {}
    # Pairwise membership: for each focus HH, track destination relative to other focus group
    # destination categories per S1 member in subsample
    dest_counts = {
        K5_LEGUMES: {"same_matched": 0, "other_focus_matched": 0, "elsewhere": 0, "n": 0},
        K5_CEREALS_NO_LEGUMES: {
            "same_matched": 0,
            "other_focus_matched": 0,
            "elsewhere": 0,
            "n": 0,
        },
    }

    for rep, idx in enumerate(replicate_indices):
        d_sub = distance[np.ix_(idx, idx)]
        X_sub = X.iloc[idx].reset_index(drop=True)
        base5 = X_sub["cluster_k5"].to_numpy(dtype=int)
        base4 = X_sub["cluster_k4"].to_numpy(dtype=int)

        labels5, medoids5 = pam_precomputed(
            d_sub, n_clusters=5, random_state=RANDOM_SEED, max_iter=PAM_MAX_ITER
        )
        labels4, _ = pam_precomputed(
            d_sub, n_clusters=4, random_state=RANDOM_SEED, max_iter=PAM_MAX_ITER
        )

        match5 = max_jaccard_matches(base5, labels5, 5)
        match4 = max_jaccard_matches(base4, labels4, 4)

        r_leg, j_leg = match5[K5_LEGUMES]
        r_cer, j_cer = match5[K5_CEREALS_NO_LEGUMES]
        merged = r_leg == r_cer
        merge_rows.append(
            {
                "replicate": rep,
                "matched_cluster_legumes": r_leg,
                "matched_cluster_cereals_no_legumes": r_cer,
                "jaccard_legumes": round(j_leg, 6),
                "jaccard_cereals_no_legumes": round(j_cer, 6),
                "focus_groups_merge_to_same_resample_cluster": merged,
            }
        )

        # Membership transitions for S1 focus members in this subsample
        for c in K5_FOCUS:
            members = np.where(base5 == c)[0]
            other = K5_CEREALS_NO_LEGUMES if c == K5_LEGUMES else K5_LEGUMES
            r_same = match5[c][0]
            r_other = match5[other][0]
            for loc in members:
                dest = int(labels5[loc])
                dest_counts[c]["n"] += 1
                if dest == r_same:
                    dest_counts[c]["same_matched"] += 1
                    dest_cat = "same_matched"
                elif dest == r_other:
                    dest_counts[c]["other_focus_matched"] += 1
                    dest_cat = "other_focus_matched"
                else:
                    dest_counts[c]["elsewhere"] += 1
                    dest_cat = "elsewhere"
                transition_rows.append(
                    {
                        "replicate": rep,
                        "respondent_id": X_sub.iloc[loc]["respondent_id"],
                        "s1_k5_cluster": c,
                        "s1_k5_note": NOTES[("k5", c)],
                        "resample_cluster": dest,
                        "destination_category": dest_cat,
                        "matched_same_cluster": r_same,
                        "matched_other_focus_cluster": r_other,
                        "focus_groups_merged": merged,
                    }
                )

            # Profile persistence on matched resample cluster
            member_mask = labels5 == r_same
            for trait in DEFINING_TRAITS:
                mean_val = (
                    float(X_sub.loc[member_mask, trait].mean())
                    if member_mask.any()
                    else np.nan
                )
                profile_acc.setdefault((c, trait), []).append(mean_val)

            # Medoid feature vector for matched resample cluster
            med_idx = int(medoids5[r_same])
            med_row = X_sub.iloc[med_idx]
            medoid_feature_rows.append(
                {
                    "replicate": rep,
                    "s1_k5_cluster": c,
                    "s1_k5_note": NOTES[("k5", c)],
                    "matched_resample_cluster": r_same,
                    "jaccard": round(match5[c][1], 6),
                    **{trait: med_row[trait] for trait in DEFINING_TRAITS},
                    "pattern_legumes_ge_0_5": float(med_row["has_legumes"]) >= 0.5,
                    "pattern_cereals_ge_0_5": float(med_row["has_cereals"]) >= 0.5,
                    "pattern_family_ge_0_5": float(
                        med_row[
                            "land_ownership_family_owned_e_g_inherited_family_land"
                        ]
                    )
                    >= 0.5,
                    "pattern_other_live_lt_0_5": float(med_row["has_other_livestock"])
                    < 0.5,
                }
            )

        # k=4: do the two family crop groups merge?
        r_a, j_a = match4[K4_FAMILY_NO_ROOTS]
        r_b, j_b = match4[K4_FAMILY_ROOTS]
        merge_rows[-1]["k4_family_groups_merge"] = r_a == r_b
        merge_rows[-1]["k4_jaccard_no_roots"] = round(j_a, 6)
        merge_rows[-1]["k4_jaccard_with_roots"] = round(j_b, 6)

        # Combined k=4 family-crop recovery: Jaccard of union of S1 family crop vs best resample
        s1_family4 = np.isin(base4, list(K4_FAMILY))
        best_j_union = 0.0
        for r in range(4):
            best_j_union = max(best_j_union, jaccard(s1_family4, labels4 == r))
        # Also union of two matched clusters
        union_match = np.isin(labels4, [r_a, r_b])
        merge_rows[-1]["k4_family_union_jaccard_vs_best_single"] = round(best_j_union, 6)
        merge_rows[-1]["k4_family_union_jaccard_vs_matched_pair"] = round(
            jaccard(s1_family4, union_match), 6
        )

        # Combined k=5 focus union recovery
        s1_focus = np.isin(base5, list(K5_FOCUS))
        best_j_focus = max(jaccard(s1_focus, labels5 == r) for r in range(5))
        union_focus = np.isin(labels5, [r_leg, r_cer])
        merge_rows[-1]["k5_focus_union_jaccard_vs_best_single"] = round(best_j_focus, 6)
        merge_rows[-1]["k5_focus_union_jaccard_vs_matched_pair"] = round(
            jaccard(s1_focus, union_focus), 6
        )

    # Summaries
    transition_summary = []
    for c, counts in dest_counts.items():
        n_c = max(counts["n"], 1)
        transition_summary.append(
            {
                "s1_k5_cluster": c,
                "s1_k5_note": NOTES[("k5", c)],
                "n_member_replicate_observations": counts["n"],
                "pct_same_matched": round(100.0 * counts["same_matched"] / n_c, 2),
                "pct_other_focus_matched": round(
                    100.0 * counts["other_focus_matched"] / n_c, 2
                ),
                "pct_elsewhere": round(100.0 * counts["elsewhere"] / n_c, 2),
            }
        )

    merge_df = pd.DataFrame(merge_rows)
    merge_summary = pd.DataFrame(
        [
            {
                "metric": "pct_replicates_k5_focus_groups_merge",
                "value": round(
                    100.0
                    * float(merge_df["focus_groups_merge_to_same_resample_cluster"].mean()),
                    2,
                ),
            },
            {
                "metric": "pct_replicates_k4_family_groups_merge",
                "value": round(
                    100.0 * float(merge_df["k4_family_groups_merge"].mean()), 2
                ),
            },
            {
                "metric": "mean_k5_focus_union_jaccard_matched_pair",
                "value": round(
                    float(merge_df["k5_focus_union_jaccard_vs_matched_pair"].mean()), 6
                ),
            },
            {
                "metric": "mean_k4_family_union_jaccard_matched_pair",
                "value": round(
                    float(merge_df["k4_family_union_jaccard_vs_matched_pair"].mean()), 6
                ),
            },
            {
                "metric": "mean_jaccard_legumes",
                "value": round(float(merge_df["jaccard_legumes"].mean()), 6),
            },
            {
                "metric": "mean_jaccard_cereals_no_legumes",
                "value": round(
                    float(merge_df["jaccard_cereals_no_legumes"].mean()), 6
                ),
            },
        ]
    )

    for (c, trait), vals in sorted(profile_acc.items()):
        arr = np.asarray(vals, dtype=float)
        s1 = s1_means[(c, trait)]
        profile_rows.append(
            {
                "s1_k5_cluster": c,
                "s1_k5_note": NOTES[("k5", c)],
                "variable": trait,
                "s1_mean": round(s1, 6),
                "mean_resample_matched_mean": round(float(np.nanmean(arr)), 6),
                "std_resample_matched_mean": round(float(np.nanstd(arr)), 6),
                "abs_diff_from_s1": round(abs(float(np.nanmean(arr)) - s1), 6),
            }
        )
    profile_df = pd.DataFrame(profile_rows)

    medoid_detail = pd.DataFrame(medoid_feature_rows)
    # Medoid variation: rate of key pattern flags + mean feature values
    medoid_summary = []
    for c in K5_FOCUS:
        sub = medoid_detail.loc[medoid_detail["s1_k5_cluster"] == c]
        row = {
            "s1_k5_cluster": c,
            "s1_k5_note": NOTES[("k5", c)],
            "pct_medoid_legumes_ge_0_5": round(
                100.0 * float(sub["pattern_legumes_ge_0_5"].mean()), 2
            ),
            "pct_medoid_cereals_ge_0_5": round(
                100.0 * float(sub["pattern_cereals_ge_0_5"].mean()), 2
            ),
            "pct_medoid_family_ge_0_5": round(
                100.0 * float(sub["pattern_family_ge_0_5"].mean()), 2
            ),
            "pct_medoid_other_live_lt_0_5": round(
                100.0 * float(sub["pattern_other_live_lt_0_5"].mean()), 2
            ),
            "mean_medoid_has_legumes": round(float(sub["has_legumes"].mean()), 4),
            "mean_medoid_has_cereals": round(float(sub["has_cereals"].mean()), 4),
            "mean_medoid_n_crop_types": round(float(sub["n_crop_types"].mean()), 4),
            "mean_medoid_has_roots": round(float(sub["has_roots_tubers"].mean()), 4),
        }
        # Full S1 pattern match rates
        if c == K5_LEGUMES:
            match = (
                sub["pattern_family_ge_0_5"]
                & sub["pattern_legumes_ge_0_5"]
                & sub["pattern_other_live_lt_0_5"]
            )
        else:
            match = (
                sub["pattern_family_ge_0_5"]
                & sub["pattern_cereals_ge_0_5"]
                & ~sub["pattern_legumes_ge_0_5"]
            )
        row["pct_full_s1_pattern_match"] = round(100.0 * float(match.mean()), 2)
        medoid_summary.append(row)
    medoid_summary_df = pd.DataFrame(medoid_summary)

    boundary_summary = pd.DataFrame(
        [
            {
                "metric": "n_focus_households",
                "value": int(focus_mask.sum()),
            },
            {
                "metric": "boundary_gap_10th_percentile_cutoff",
                "value": round(gap_cut, 6),
            },
            {
                "metric": "n_boundary_ambiguous",
                "value": int(boundary_df["is_boundary_ambiguous"].sum()),
            },
            {
                "metric": "pct_focus_boundary_ambiguous",
                "value": round(
                    100.0 * float(boundary_df["is_boundary_ambiguous"].mean()), 2
                ),
            },
            {
                "metric": "pct_focus_nearer_medoid_matches_assignment",
                "value": round(
                    100.0 * float(boundary_df["nearer_matches_assignment"].mean()), 2
                ),
            },
            {
                "metric": "pct_ambiguous_that_are_legumes_assigned",
                "value": round(
                    100.0
                    * float(
                        (
                            boundary_df.loc[
                                boundary_df["is_boundary_ambiguous"], "s1_k5_cluster"
                            ]
                            == K5_LEGUMES
                        ).mean()
                    ),
                    2,
                ),
            },
        ]
    )

    params = {
        "experiment": "S2b_k5_family_crop_robustness",
        "specification": "A",
        "focus_k5_clusters": {
            str(K5_LEGUMES): NOTES[("k5", K5_LEGUMES)],
            str(K5_CEREALS_NO_LEGUMES): NOTES[("k5", K5_CEREALS_NO_LEGUMES)],
        },
        "comparator_k4_family_clusters": {
            str(K4_FAMILY_NO_ROOTS): NOTES[("k4", K4_FAMILY_NO_ROOTS)],
            str(K4_FAMILY_ROOTS): NOTES[("k4", K4_FAMILY_ROOTS)],
        },
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "subsample_n": subsample_n,
        "n_replicates": N_REPLICATES,
        "random_seed": RANDOM_SEED,
        "boundary_gap_cutoff_rule": "10th_percentile_of_abs_dist_gap_among_focus",
        "coping_used": False,
        "final_k_selected": None,
    }

    # Write outputs
    (TABLES_DIR / "14_k5_robustness_parameters.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    outputs = {
        "14_k5_membership_transition_summary.csv": pd.DataFrame(transition_summary),
        "14_k5_membership_transitions_long.csv": pd.DataFrame(transition_rows),
        "14_k5_focus_merge_by_replicate.csv": merge_df,
        "14_k5_focus_merge_summary.csv": merge_summary,
        "14_k5_profile_persistence.csv": profile_df,
        "14_k5_medoid_feature_summary.csv": medoid_summary_df,
        "14_k5_medoid_features_by_replicate.csv": medoid_detail,
        "14_k5_boundary_households.csv": boundary_df,
        "14_k5_boundary_summary.csv": boundary_summary,
        "14_k5_vs_k4_crosstab.csv": crosstab,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")
    print(f"Wrote {TABLES_DIR / '14_k5_robustness_parameters.json'}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# k=5 Family Crop Robustness Investigation (Script 14)

Generated: {timestamp}

## Purpose

Assess whether the Spec A k=5 subtypes **family_legumes_heavy** (cluster 0) and **family_cereals_without_legumes** (cluster 4) are meaningful and reproducible distinctions, or an unstable subdivision of broader family crop agriculture. Comparator: k=4 family crop groups.

No Spec B, coping profiling, or final k selection.

## Design

- Same Spec A features and Gower construction as Scripts 12–13
- 50 shared 80% subsamples, seed 42
- Focus clusters: k=5 {{0, 4}}; comparator k=4 family crop {{0, 1}}

## 1. Membership transitions

```
{pd.DataFrame(transition_summary).to_string(index=False)}
```

## 2. Merge rates and union recovery

```
{merge_summary.to_string(index=False)}
```

## 3. Profile persistence (matched resample clusters)

Top absolute deviations from S1 means:

```
{profile_df.sort_values('abs_diff_from_s1', ascending=False).head(20).to_string(index=False)}
```

## 4. Medoid feature variation

```
{medoid_summary_df.to_string(index=False)}
```

## 5. Boundary households

```
{boundary_summary.to_string(index=False)}
```

Ambiguous households are those among the two focus groups whose Gower distances to the two S1 medoids differ by at most the 10th percentile gap ({gap_cut:.4f}).

## 6. Overlap with k=4 family crop groups

```
{crosstab.to_string(index=False)}
```

## Review guidance

- High `pct_other_focus_matched` or frequent focus-group merges weaken the case for keeping both k=5 subtypes separate.
- Higher k=4 family-union Jaccard than separate k=5 subtype Jaccard supports preferring a broader family-crop representation.
- Persistent profile differences (especially legumes) with moderate membership churn can still be substantively meaningful.

**No final k is selected here.** Hold Spec B and coping profiling until this evidence is reviewed.

## Output files

- `outputs/tables/14_k5_*.csv`
- `outputs/tables/14_k5_robustness_parameters.json`
"""
    path = REPORTS_DIR / "14_segmentation_k5_family_crop_robustness.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")

    # Update design doc pointer
    design = WORKBENCH_ROOT / "logs" / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    note = """

## 9. Script 14 — k=5 family crop robustness

Targeted investigation of whether k=5 `family_legumes_heavy` and `family_cereals_without_legumes` are reproducible subtypes or an unstable split of family crop agriculture, compared with k=4 family crop groups.

Report: `outputs/reports/14_segmentation_k5_family_crop_robustness.md`

Provisional preference remains k=5 subject to this evidence; k=4 retained as comparator. Spec B, coping profiling and final k selection remain on hold.
"""
    if "## 9. Script 14" not in text:
        design.write_text(text.rstrip() + note, encoding="utf-8")
        print(f"Updated {design}")

    print("\nTransition summary")
    print(pd.DataFrame(transition_summary).to_string(index=False))
    print("\nMerge summary")
    print(merge_summary.to_string(index=False))
    print("\nMedoid summary")
    print(medoid_summary_df.to_string(index=False))
    print("\nBoundary summary")
    print(boundary_summary.to_string(index=False))
    print("\nDone. Review before selecting k.")


if __name__ == "__main__":
    main()
