#!/usr/bin/env python3
"""S2 stability investigation for Spec A candidates k=2, 4 and 5.

Shared 80% subsamples (R=50). Reports ARI and cluster-level Jaccard stability.
Does not select a final k, run Spec B, or use coping outcomes.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gower
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    AGRICULTURAL_HAS_VARS,
    EXPECTED_DROUGHT_N,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    PORTFOLIO_COUNT_VARS,
    PREPARED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)
from segmentation_utils import pam_precomputed  # noqa: E402

RANDOM_SEED = 42
CANDIDATE_K = [2, 4, 5]
N_REPLICATES = 50
SUBSAMPLE_FRACTION = 0.80
PAM_MAX_ITER = 100
SPEC = "A"

DEFINING_TRAITS = [
    "has_cereals",
    "has_legumes",
    "has_roots_tubers",
    "has_other_livestock",
    "n_crop_types",
    "n_livestock_types",
    "land_ownership_own_under_your_name",
    "land_ownership_lease_rent_you_dont_own",
    "land_ownership_community_public_owned_e_g_pastoralists",
    "land_ownership_family_owned_e_g_inherited_family_land",
]

# S1 interpretive labels for report callouts (not used in fitting)
CLUSTER_NOTES = {
    2: {
        0: "crop_family",
        1: "livestock_community",
    },
    4: {
        0: "family_crops_no_roots",
        1: "family_crops_with_roots",
        2: "livestock_community",
        3: "lease_crops",
    },
    5: {
        0: "family_legumes_heavy",
        1: "own_name_crops",
        2: "livestock_community",
        3: "family_mixed_crop_livestock",
        4: "family_cereals_without_legumes",
    },
}


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
    if union == 0:
        return 0.0
    return float(inter / union)


def max_jaccard_match(
    base_labels: np.ndarray,
    resample_labels: np.ndarray,
    k: int,
) -> dict[int, tuple[int, float, int]]:
    """Map each base cluster -> (best resample cluster, jaccard, resample size)."""
    matches = {}
    for c in range(k):
        base_mask = base_labels == c
        best_j = -1.0
        best_r = -1
        best_n = 0
        for r in range(k):
            res_mask = resample_labels == r
            j = jaccard(base_mask, res_mask)
            if j > best_j:
                best_j = j
                best_r = r
                best_n = int(res_mask.sum())
        matches[c] = (best_r, best_j, best_n)
    return matches


def medoid_pattern_flags(row: pd.Series, k: int, cluster: int) -> dict[str, bool]:
    """Heuristic flags for whether a medoid matches S1 defining patterns."""
    community = float(row["land_ownership_community_public_owned_e_g_pastoralists"])
    family = float(row["land_ownership_family_owned_e_g_inherited_family_land"])
    own_name = float(row["land_ownership_own_under_your_name"])
    lease = float(row["land_ownership_lease_rent_you_dont_own"])
    legumes = float(row["has_legumes"])
    cereals = float(row["has_cereals"])
    roots = float(row["has_roots_tubers"])
    other_live = float(row["has_other_livestock"])

    note = CLUSTER_NOTES[k][cluster]
    flags = {
        "is_community_medoid": community >= 0.5,
        "is_family_medoid": family >= 0.5,
        "is_own_name_medoid": own_name >= 0.5,
        "is_lease_medoid": lease >= 0.5,
        "has_legumes_medoid": legumes >= 0.5,
        "has_cereals_medoid": cereals >= 0.5,
        "has_roots_medoid": roots >= 0.5,
        "has_other_livestock_medoid": other_live >= 0.5,
    }
    if note == "livestock_community":
        flags["matches_s1_pattern"] = community >= 0.5 and other_live >= 0.5
    elif note == "crop_family":
        flags["matches_s1_pattern"] = family >= 0.5 and cereals >= 0.5
    elif note == "family_crops_no_roots":
        flags["matches_s1_pattern"] = family >= 0.5 and roots < 0.5 and cereals >= 0.5
    elif note == "family_crops_with_roots":
        flags["matches_s1_pattern"] = family >= 0.5 and roots >= 0.5
    elif note == "lease_crops":
        flags["matches_s1_pattern"] = lease >= 0.5 and cereals >= 0.5
    elif note == "family_legumes_heavy":
        flags["matches_s1_pattern"] = family >= 0.5 and legumes >= 0.5 and other_live < 0.5
    elif note == "own_name_crops":
        flags["matches_s1_pattern"] = own_name >= 0.5 and cereals >= 0.5
    elif note == "family_mixed_crop_livestock":
        flags["matches_s1_pattern"] = family >= 0.5 and other_live >= 0.5 and cereals >= 0.5
    elif note == "family_cereals_without_legumes":
        flags["matches_s1_pattern"] = family >= 0.5 and cereals >= 0.5 and legumes < 0.5
    else:
        flags["matches_s1_pattern"] = False
    return flags


def main() -> None:
    ensure_output_dirs()
    feature_cols = spec_a_features()
    assert LAND_SIZE_STANDARDISED not in feature_cols
    assert len(feature_cols) == 17

    X_full = pd.read_parquet(PREPARED_DIR / "09_X_segmentation.parquet")
    assign_wide = pd.read_csv(TABLES_DIR / "12_s1_spec_a_assignments_wide.csv")
    profiles_s1 = pd.read_csv(TABLES_DIR / "12_s1_spec_a_cluster_profiles.csv")

    if len(X_full) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected n={len(X_full)}")

    X = X_full.merge(assign_wide, on="respondent_id", how="inner")
    if len(X) != EXPECTED_DROUGHT_N:
        raise ValueError("Assignment merge changed sample size")

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if int(X[feature_cols].isna().sum().sum()) != 0:
        raise ValueError("Spec A features must be complete")

    # Build full Spec A Gower matrix once (same as S1)
    cat_features = categorical_mask(feature_cols)
    t0 = time.perf_counter()
    distance = gower.gower_matrix(
        X[feature_cols].to_numpy(dtype=float),
        cat_features=cat_features,
    )
    gower_seconds = time.perf_counter() - t0
    if np.isnan(distance).any():
        raise ValueError("Gower distance contains NaNs")

    n = len(X)
    subsample_n = int(round(SUBSAMPLE_FRACTION * n))
    rng = np.random.default_rng(RANDOM_SEED)
    replicate_indices = [
        np.sort(rng.choice(n, size=subsample_n, replace=False))
        for _ in range(N_REPLICATES)
    ]

    params = {
        "experiment": "S2",
        "specification": SPEC,
        "candidate_k": CANDIDATE_K,
        "n_households": n,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "subsample_n": subsample_n,
        "n_replicates": N_REPLICATES,
        "random_seed": RANDOM_SEED,
        "shared_indices_across_k": True,
        "distance": "gower_subset_of_full_spec_a_matrix",
        "gower_runtime_seconds_full": round(gower_seconds, 3),
        "pam_implementation": "segmentation_utils.pam_precomputed",
        "pam_max_iter": PAM_MAX_ITER,
        "pam_random_seed": RANDOM_SEED,
        "agreement_metric": "adjusted_rand_index",
        "cluster_stability_metric": "max_jaccard_vs_s1_cluster",
        "coping_used": False,
        "final_k_selected": None,
        "defining_traits": DEFINING_TRAITS,
    }

    ari_rows = []
    jaccard_rows = []
    size_rows = []
    profile_rows = []
    medoid_rows = []

    s1_means = {}
    for k in CANDIDATE_K:
        for c in range(k):
            for trait in DEFINING_TRAITS:
                val = profiles_s1.loc[
                    (profiles_s1["k"] == k)
                    & (profiles_s1["cluster"] == c)
                    & (profiles_s1["variable"] == trait),
                    "mean",
                ]
                s1_means[(k, c, trait)] = float(val.iloc[0]) if len(val) else np.nan

    # Accumulators for profile means across replicates (matched clusters)
    profile_acc: dict[tuple[int, int, str], list[float]] = {}
    jaccard_acc: dict[tuple[int, int], list[float]] = {}
    size_acc: dict[tuple[int, int], list[int]] = {}
    medoid_match_acc: dict[tuple[int, int], list[bool]] = {}

    t_loop = time.perf_counter()
    for rep, idx in enumerate(replicate_indices):
        d_sub = distance[np.ix_(idx, idx)]
        X_sub = X.iloc[idx].reset_index(drop=True)

        for k in CANDIDATE_K:
            base_labels = X_sub[f"cluster_k{k}"].to_numpy(dtype=int)
            labels, medoids = pam_precomputed(
                d_sub,
                n_clusters=k,
                random_state=RANDOM_SEED,
                max_iter=PAM_MAX_ITER,
            )
            ari = float(adjusted_rand_score(base_labels, labels))
            ari_rows.append(
                {
                    "replicate": rep,
                    "k": k,
                    "ari": ari,
                    "subsample_n": subsample_n,
                }
            )

            matches = max_jaccard_match(base_labels, labels, k)
            for c in range(k):
                r_lab, j, r_n = matches[c]
                jaccard_acc.setdefault((k, c), []).append(j)
                size_acc.setdefault((k, c), []).append(r_n)

                # Profile persistence on matched resample cluster
                member_mask = labels == r_lab
                for trait in DEFINING_TRAITS:
                    mean_val = float(X_sub.loc[member_mask, trait].mean()) if member_mask.any() else np.nan
                    profile_acc.setdefault((k, c, trait), []).append(mean_val)

                # Medoid consistency for S1 cluster via matched resample medoid
                # Find medoid index among subsample for matched resample label
                # medoids array is ordered by cluster id 0..k-1 after pam_precomputed sorts medoids
                # but labels are assigned by argmin to medoids in sorted order - cluster id i corresponds to medoids[i]
                medoid_local_idx = int(medoids[r_lab])
                medoid_row = X_sub.iloc[medoid_local_idx]
                flags = medoid_pattern_flags(medoid_row, k, c)
                medoid_match_acc.setdefault((k, c), []).append(bool(flags["matches_s1_pattern"]))
                medoid_rows.append(
                    {
                        "replicate": rep,
                        "k": k,
                        "s1_cluster": c,
                        "s1_cluster_note": CLUSTER_NOTES[k][c],
                        "matched_resample_cluster": r_lab,
                        "jaccard": round(j, 6),
                        "medoid_respondent_id": medoid_row["respondent_id"],
                        **{f"medoid_{t}": medoid_row[t] for t in DEFINING_TRAITS},
                        **flags,
                    }
                )

        if (rep + 1) % 10 == 0:
            print(f"Completed replicate {rep + 1}/{N_REPLICATES}")

    loop_seconds = time.perf_counter() - t_loop
    params["stability_loop_runtime_seconds"] = round(loop_seconds, 3)

    ari_by_rep = pd.DataFrame(ari_rows)
    overall_ari = (
        ari_by_rep.groupby("k")["ari"]
        .agg(
            n_replicates="count",
            mean_ari="mean",
            median_ari="median",
            std_ari="std",
            min_ari="min",
            max_ari="max",
            q25_ari=lambda s: s.quantile(0.25),
            q75_ari=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    overall_ari["iqr_ari"] = overall_ari["q75_ari"] - overall_ari["q25_ari"]
    for col in ["mean_ari", "median_ari", "std_ari", "min_ari", "max_ari", "q25_ari", "q75_ari", "iqr_ari"]:
        overall_ari[col] = overall_ari[col].round(6)

    for (k, c), js in sorted(jaccard_acc.items()):
        arr = np.asarray(js, dtype=float)
        sizes = np.asarray(size_acc[(k, c)], dtype=float)
        jaccard_rows.append(
            {
                "k": k,
                "s1_cluster": c,
                "s1_cluster_note": CLUSTER_NOTES[k][c],
                "mean_max_jaccard": round(float(arr.mean()), 6),
                "median_max_jaccard": round(float(np.median(arr)), 6),
                "min_max_jaccard": round(float(arr.min()), 6),
                "max_max_jaccard": round(float(arr.max()), 6),
                "n_replicates": len(arr),
            }
        )
        size_rows.append(
            {
                "k": k,
                "s1_cluster": c,
                "s1_cluster_note": CLUSTER_NOTES[k][c],
                "s1_n": int((X[f"cluster_k{k}"] == c).sum()),
                "s1_pct": round(100.0 * float((X[f"cluster_k{k}"] == c).mean()), 2),
                "mean_matched_resample_n": round(float(sizes.mean()), 2),
                "std_matched_resample_n": round(float(sizes.std(ddof=0)), 2),
                "min_matched_resample_n": int(sizes.min()),
                "max_matched_resample_n": int(sizes.max()),
                "mean_matched_resample_pct": round(
                    100.0 * float(sizes.mean()) / subsample_n, 2
                ),
            }
        )

    for (k, c, trait), vals in sorted(profile_acc.items()):
        arr = np.asarray(vals, dtype=float)
        s1 = s1_means[(k, c, trait)]
        profile_rows.append(
            {
                "k": k,
                "s1_cluster": c,
                "s1_cluster_note": CLUSTER_NOTES[k][c],
                "variable": trait,
                "s1_mean": round(s1, 6),
                "mean_resample_matched_mean": round(float(np.nanmean(arr)), 6),
                "abs_diff_from_s1": round(abs(float(np.nanmean(arr)) - s1), 6),
            }
        )

    medoid_consistency = []
    for (k, c), flags in sorted(medoid_match_acc.items()):
        arr = np.asarray(flags, dtype=float)
        medoid_consistency.append(
            {
                "k": k,
                "s1_cluster": c,
                "s1_cluster_note": CLUSTER_NOTES[k][c],
                "pct_replicates_medoid_matches_pattern": round(100.0 * float(arr.mean()), 2),
                "n_replicates": len(arr),
            }
        )

    jaccard_df = pd.DataFrame(jaccard_rows)
    size_df = pd.DataFrame(size_rows)
    profile_df = pd.DataFrame(profile_rows)
    medoid_cons_df = pd.DataFrame(medoid_consistency)
    medoid_detail_df = pd.DataFrame(medoid_rows)

    params_path = TABLES_DIR / "13_s2_parameters.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote {params_path}")

    outputs = {
        "13_s2_overall_ari_by_k.csv": overall_ari,
        "13_s2_ari_by_replicate.csv": ari_by_rep,
        "13_s2_cluster_jaccard_stability.csv": jaccard_df,
        "13_s2_cluster_size_variation.csv": size_df,
        "13_s2_profile_persistence.csv": profile_df,
        "13_s2_medoid_consistency.csv": medoid_cons_df,
        "13_s2_medoid_detail_by_replicate.csv": medoid_detail_df,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    # Highlight k=5 legumes vs cereals-without-legumes
    k5 = jaccard_df.loc[jaccard_df["k"] == 5].copy()
    k5_focus = k5.loc[
        k5["s1_cluster_note"].isin(
            ["family_legumes_heavy", "family_cereals_without_legumes"]
        )
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# S2 Stability Report — Spec A candidates k=2, 4, 5 (Script 13)

Generated: {timestamp}

## Scope

Stability only for Specification A candidates from S1 profile review.

- Candidates: k = 2, 4, 5
- Subsample without replacement at 80% (n = {subsample_n})
- Replicates: {N_REPLICATES}
- Shared subsample indices across k
- Random seed: {RANDOM_SEED}
- No Spec B, no coping outcomes, **no final k selected**

## Overall assignment agreement (ARI)

```
{overall_ari.to_string(index=False)}
```

## Cluster-level max Jaccard stability

```
{jaccard_df.to_string(index=False)}
```

## Focus: k=5 legumes vs cereals-without-legumes

```
{k5_focus.to_string(index=False) if len(k5_focus) else 'n/a'}
```

## Cluster size variation (matched resample clusters)

```
{size_df.to_string(index=False)}
```

## Medoid pattern consistency

```
{medoid_cons_df.to_string(index=False)}
```

## Interpretation guide

Overall ARI summarises assignment agreement but must not alone decide k. Inspect cluster-level Jaccard, especially whether k=5 family_legumes_heavy and family_cereals_without_legumes are consistently recovered or frequently fragmented.

Review these outputs before S3/Spec B, coping profiling or final k selection.

## Output files

- `outputs/tables/13_s2_parameters.json`
- `outputs/tables/13_s2_overall_ari_by_k.csv`
- `outputs/tables/13_s2_ari_by_replicate.csv`
- `outputs/tables/13_s2_cluster_jaccard_stability.csv`
- `outputs/tables/13_s2_cluster_size_variation.csv`
- `outputs/tables/13_s2_profile_persistence.csv`
- `outputs/tables/13_s2_medoid_consistency.csv`
- `outputs/tables/13_s2_medoid_detail_by_replicate.csv`
"""
    report_path = REPORTS_DIR / "13_segmentation_s2_stability.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    print("\nOverall ARI")
    print(overall_ari.to_string(index=False))
    print("\nCluster Jaccard")
    print(jaccard_df.to_string(index=False))
    print("\nDone. Review S2 before selecting k.")


if __name__ == "__main__":
    main()
