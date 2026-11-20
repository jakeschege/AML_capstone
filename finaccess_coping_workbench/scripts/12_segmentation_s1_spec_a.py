#!/usr/bin/env python3
"""S1 segmentation: Gower distance + PAM for Specification A.

Specification A excludes land_size_acres (17 features). Candidate k = 2..6.
Does not select a final k, compare coping outcomes, or run Spec B.
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
from segmentation_utils import (  # noqa: E402
    average_silhouette_precomputed,
    cluster_feature_profiles,
    cluster_size_table,
    pam_precomputed,
)

RANDOM_SEED = 42
CANDIDATE_K = list(range(2, 7))
SPEC = "A"
PAM_MAX_ITER = 100


def spec_a_features() -> list[str]:
    return (
        list(AGRICULTURAL_HAS_VARS)
        + list(PORTFOLIO_COUNT_VARS)
        + list(LAND_OWNERSHIP_VARS)
    )


def categorical_mask(feature_cols: list[str]) -> np.ndarray:
    """Treat agri/ownership binaries as categorical for Gower; counts as numeric."""
    return np.array(
        [c not in PORTFOLIO_COUNT_VARS for c in feature_cols],
        dtype=bool,
    )


def main() -> None:
    ensure_output_dirs()
    X_full = pd.read_parquet(PREPARED_DIR / "09_X_segmentation.parquet")
    if len(X_full) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected n={len(X_full)}")

    feature_cols = spec_a_features()
    if LAND_SIZE_STANDARDISED in feature_cols:
        raise AssertionError("Spec A must exclude land_size_acres")
    if len(feature_cols) != 17:
        raise AssertionError(f"Spec A expects 17 features, got {len(feature_cols)}")

    X = X_full[["respondent_id"] + feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if int(X[feature_cols].isna().sum().sum()) != 0:
        raise ValueError("Spec A matrix must be complete")

    cat_features = categorical_mask(feature_cols)
    params = {
        "experiment": "S1",
        "specification": SPEC,
        "n_households": EXPECTED_DROUGHT_N,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "excluded": [LAND_SIZE_STANDARDISED],
        "distance": "gower",
        "gower_package": "gower",
        "cat_features_rule": "binaries_categorical_counts_numeric",
        "algorithm": "PAM",
        "pam_implementation": "segmentation_utils.pam_precomputed",
        "pam_max_iter": PAM_MAX_ITER,
        "candidate_k": CANDIDATE_K,
        "random_seed": RANDOM_SEED,
        "feature_weights": "equal",
        "zero_fill": False,
        "coping_used_in_clustering": False,
        "final_k_selected": None,
    }

    t0 = time.perf_counter()
    distance = gower.gower_matrix(
        X[feature_cols].to_numpy(dtype=float),
        cat_features=cat_features,
    )
    gower_seconds = time.perf_counter() - t0
    if np.isnan(distance).any():
        raise ValueError("Gower distance contains NaNs; Spec A should be complete")
    params["gower_runtime_seconds"] = round(gower_seconds, 3)
    params["distance_shape"] = list(distance.shape)

    assignment_frames = []
    size_frames = []
    silhouette_rows = []
    profile_frames = []
    medoid_rows = []

    for k in CANDIDATE_K:
        t1 = time.perf_counter()
        labels, medoids = pam_precomputed(
            distance,
            n_clusters=k,
            random_state=RANDOM_SEED,
            max_iter=PAM_MAX_ITER,
        )
        pam_seconds = time.perf_counter() - t1
        sil = average_silhouette_precomputed(distance, labels)

        assign = pd.DataFrame(
            {
                "respondent_id": X["respondent_id"].values,
                "specification": SPEC,
                "k": k,
                "cluster": labels,
            }
        )
        assignment_frames.append(assign)
        size_frames.append(cluster_size_table(labels, k))
        silhouette_rows.append(
            {
                "specification": SPEC,
                "k": k,
                "average_silhouette": round(sil, 6),
                "pam_runtime_seconds": round(pam_seconds, 3),
                "n_features": len(feature_cols),
                "random_seed": RANDOM_SEED,
            }
        )
        profile_frames.append(
            cluster_feature_profiles(X, labels, feature_cols, k)
        )
        for cluster_id, medoid_idx in enumerate(medoids):
            row = {
                "specification": SPEC,
                "k": k,
                "cluster": int(cluster_id),
                "medoid_index": int(medoid_idx),
                "respondent_id": X.iloc[int(medoid_idx)]["respondent_id"],
            }
            for col in feature_cols:
                row[col] = X.iloc[int(medoid_idx)][col]
            medoid_rows.append(row)

        print(
            f"k={k}: silhouette={sil:.4f}; "
            f"sizes={pd.Series(labels).value_counts().sort_index().to_dict()}"
        )

    assignments = pd.concat(assignment_frames, ignore_index=True)
    sizes = pd.concat(size_frames, ignore_index=True)
    silhouettes = pd.DataFrame(silhouette_rows)
    profiles = pd.concat(profile_frames, ignore_index=True)
    medoids_df = pd.DataFrame(medoid_rows)

    # Wide assignment helper: one column per k
    wide = X[["respondent_id"]].copy()
    for k in CANDIDATE_K:
        wide[f"cluster_k{k}"] = assignments.loc[
            assignments["k"] == k, "cluster"
        ].to_numpy()

    outputs = {
        "12_s1_spec_a_parameters.json": None,  # handled below
        "12_s1_spec_a_silhouette_by_k.csv": silhouettes,
        "12_s1_spec_a_cluster_sizes.csv": sizes,
        "12_s1_spec_a_assignments_long.csv": assignments,
        "12_s1_spec_a_assignments_wide.csv": wide,
        "12_s1_spec_a_cluster_profiles.csv": profiles,
        "12_s1_spec_a_medoids.csv": medoids_df,
    }
    params_path = TABLES_DIR / "12_s1_spec_a_parameters.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote {params_path}")

    for name, table in outputs.items():
        if table is None:
            continue
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    # Also store distance metadata only (not full matrix) — already in params
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# S1 Segmentation Report — Specification A (Script 12)

Generated: {timestamp}

## Scope

Gower distance + PAM for Specification A only.

- Excludes `land_size_acres` (17 features)
- Equal feature weights
- Retains agricultural indicators and portfolio counts
- Candidate k = 2..6
- Random seed: {RANDOM_SEED}
- Coping outcomes **not** used
- **No final k selected**

## Runtime

- Gower distance: {params['gower_runtime_seconds']} s
- Distance shape: {params['distance_shape']}

## Silhouette by k

```
{silhouettes.to_string(index=False)}
```

## Cluster sizes

```
{sizes.to_string(index=False)}
```

## Interpretation guide

Average silhouette is a primary summary metric for comparing k, but cluster sizes and agricultural/ownership profile coherence must also be reviewed. The highest silhouette alone does not establish the preferred solution.

Review these results before S2 stability assessment. Spec B is not run in this script.

## Output files

- `outputs/tables/12_s1_spec_a_parameters.json`
- `outputs/tables/12_s1_spec_a_silhouette_by_k.csv`
- `outputs/tables/12_s1_spec_a_cluster_sizes.csv`
- `outputs/tables/12_s1_spec_a_assignments_long.csv`
- `outputs/tables/12_s1_spec_a_assignments_wide.csv`
- `outputs/tables/12_s1_spec_a_cluster_profiles.csv`
- `outputs/tables/12_s1_spec_a_medoids.csv`
"""
    report_path = REPORTS_DIR / "12_segmentation_s1_spec_a.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    print("\nDone. Review S1 before selecting k or running S2.")


if __name__ == "__main__":
    main()
