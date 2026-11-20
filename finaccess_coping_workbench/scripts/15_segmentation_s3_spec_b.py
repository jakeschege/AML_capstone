#!/usr/bin/env python3
"""S3 Spec B sensitivity: land size with missing-aware Gower.

Adds land_size_acres (community acres left missing; 16 specials median-imputed),
is_community_land, and acres_special_imputed. Fits PAM for k=2,4,5.
Does not alter Spec A outputs, use coping, or select a final k.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
from segmentation_utils import (  # noqa: E402
    average_silhouette_precomputed,
    cluster_feature_profiles,
    cluster_size_table,
    gower_matrix_missing_aware,
    pam_precomputed,
)

RANDOM_SEED = 42
CANDIDATE_K = [2, 4, 5]
PAM_MAX_ITER = 100
SPEC = "B"


def spec_a_features() -> list[str]:
    return (
        list(AGRICULTURAL_HAS_VARS)
        + list(PORTFOLIO_COUNT_VARS)
        + list(LAND_OWNERSHIP_VARS)
    )


def build_spec_b_frame(prepared: pd.DataFrame, X_seg: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feature_a = spec_a_features()
    out = X_seg[["respondent_id"] + feature_a].copy()
    meta = prepared[
        ["respondent_id", "land_size", "land_size_unit", LAND_SIZE_STANDARDISED]
    ].copy()
    out = out.merge(meta, on="respondent_id", how="left")

    acres = pd.to_numeric(out[LAND_SIZE_STANDARDISED], errors="coerce")
    unit = pd.to_numeric(out["land_size_unit"], errors="coerce")
    size = pd.to_numeric(out["land_size"], errors="coerce")

    is_community = unit.eq(5)
    special_non_community = size.isin([98, 99]) & ~is_community
    observed = acres.notna()
    median_observed = float(acres.loc[observed].median())

    acres_b = acres.copy()
    acres_b.loc[special_non_community] = median_observed
    # Community acres remain NaN (not zero-filled)
    acres_b.loc[is_community] = np.nan

    out["is_community_land"] = is_community.astype(int)
    out["acres_special_imputed"] = special_non_community.astype(int)
    out["land_size_acres_spec_b"] = acres_b

    # Safety: no zeros introduced for missing community
    assert int((is_community & acres_b.notna()).sum()) == 0
    assert int(special_non_community.sum()) == 16
    assert int(is_community.sum()) == 182

    info = {
        "n_observed_acres": int(observed.sum()),
        "median_observed_acres": median_observed,
        "n_community_land": int(is_community.sum()),
        "n_special_imputed": int(special_non_community.sum()),
        "n_acres_nan_in_spec_b": int(acres_b.isna().sum()),
        "zero_fill": False,
    }
    return out, info


def main() -> None:
    ensure_output_dirs()
    X_seg = pd.read_parquet(PREPARED_DIR / "09_X_segmentation.parquet")
    prepared = pd.read_parquet(PREPARED_DIR / "09_prepared_analysis_frame.parquet")
    assign_a = pd.read_csv(TABLES_DIR / "12_s1_spec_a_assignments_wide.csv")

    if len(X_seg) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected n={len(X_seg)}")

    Xb, acre_info = build_spec_b_frame(prepared, X_seg)
    feature_cols = spec_a_features() + [
        "land_size_acres_spec_b",
        "is_community_land",
        "acres_special_imputed",
    ]
    assert len(feature_cols) == 20

    for col in feature_cols:
        if col != "land_size_acres_spec_b":
            if int(pd.to_numeric(Xb[col], errors="coerce").isna().sum()) != 0:
                raise ValueError(f"Unexpected missing values in {col}")

    # Categorical: binaries including new indicators; numeric: counts + acres
    cat_features = np.array(
        [
            c
            not in (
                list(PORTFOLIO_COUNT_VARS) + ["land_size_acres_spec_b"]
            )
            for c in feature_cols
        ],
        dtype=bool,
    )

    mat = Xb[feature_cols].to_numpy(dtype=float)
    t0 = time.perf_counter()
    distance = gower_matrix_missing_aware(mat, cat_features=cat_features)
    gower_seconds = time.perf_counter() - t0
    if np.isnan(distance).any():
        raise ValueError("Missing-aware Gower still produced NaN distances")

    params = {
        "experiment": "S3",
        "specification": SPEC,
        "n_households": EXPECTED_DROUGHT_N,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "acreage_treatment": acre_info,
        "distance": "gower_matrix_missing_aware",
        "gower_runtime_seconds": round(gower_seconds, 3),
        "algorithm": "PAM",
        "pam_implementation": "segmentation_utils.pam_precomputed",
        "pam_max_iter": PAM_MAX_ITER,
        "candidate_k": CANDIDATE_K,
        "random_seed": RANDOM_SEED,
        "feature_weights": "equal",
        "zero_fill": False,
        "spec_a_unchanged": True,
        "coping_used": False,
        "final_k_selected": None,
        "comparability_note": (
            "Community-land pairs omit acres from the pairwise Gower average when "
            "acres are missing; is_community_land carries the community distinction."
        ),
    }

    assignment_frames = []
    size_frames = []
    silhouette_rows = []
    profile_frames = []
    medoid_rows = []
    compare_rows = []

    Xb = Xb.merge(assign_a, on="respondent_id", how="left")

    for k in CANDIDATE_K:
        t1 = time.perf_counter()
        labels, medoids = pam_precomputed(
            distance, n_clusters=k, random_state=RANDOM_SEED, max_iter=PAM_MAX_ITER
        )
        pam_seconds = time.perf_counter() - t1
        sil = average_silhouette_precomputed(distance, labels)

        assign = pd.DataFrame(
            {
                "respondent_id": Xb["respondent_id"].values,
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
            cluster_feature_profiles(Xb, labels, feature_cols, k)
        )
        for cluster_id, medoid_idx in enumerate(medoids):
            row = {
                "specification": SPEC,
                "k": k,
                "cluster": int(cluster_id),
                "medoid_index": int(medoid_idx),
                "respondent_id": Xb.iloc[int(medoid_idx)]["respondent_id"],
            }
            for col in feature_cols:
                row[col] = Xb.iloc[int(medoid_idx)][col]
            medoid_rows.append(row)

        labels_a = Xb[f"cluster_k{k}"].to_numpy(dtype=int)
        ari = float(adjusted_rand_score(labels_a, labels))
        # Community composition by Spec B cluster
        for c in range(k):
            mask = labels == c
            compare_rows.append(
                {
                    "k": k,
                    "spec_b_cluster": c,
                    "n": int(mask.sum()),
                    "pct": round(100.0 * float(mask.mean()), 2),
                    "pct_is_community_land": round(
                        100.0 * float(Xb.loc[mask, "is_community_land"].mean()), 2
                    ),
                    "mean_land_size_acres_spec_b": round(
                        float(Xb.loc[mask, "land_size_acres_spec_b"].mean(skipna=True)),
                        4,
                    )
                    if mask.any()
                    else np.nan,
                    "n_acres_missing_in_cluster": int(
                        Xb.loc[mask, "land_size_acres_spec_b"].isna().sum()
                    ),
                    "ari_vs_spec_a_same_k": round(ari, 6),
                }
            )

        print(
            f"k={k}: sil={sil:.4f}; ARI vs Spec A={ari:.4f}; "
            f"sizes={pd.Series(labels).value_counts().sort_index().to_dict()}"
        )

    assignments = pd.concat(assignment_frames, ignore_index=True)
    sizes = pd.concat(size_frames, ignore_index=True)
    silhouettes = pd.DataFrame(silhouette_rows)
    profiles = pd.concat(profile_frames, ignore_index=True)
    medoids_df = pd.DataFrame(medoid_rows)
    compare_df = pd.DataFrame(compare_rows)

    wide = Xb[["respondent_id"]].copy()
    for k in CANDIDATE_K:
        wide[f"cluster_k{k}"] = assignments.loc[
            assignments["k"] == k, "cluster"
        ].to_numpy()

    # Spec A vs B agreement summary
    agree = (
        compare_df.groupby("k")["ari_vs_spec_a_same_k"]
        .first()
        .reset_index()
        .rename(columns={"ari_vs_spec_a_same_k": "ari_spec_a_vs_spec_b"})
    )

    params_path = TABLES_DIR / "15_s3_spec_b_parameters.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote {params_path}")

    outputs = {
        "15_s3_spec_b_silhouette_by_k.csv": silhouettes,
        "15_s3_spec_b_cluster_sizes.csv": sizes,
        "15_s3_spec_b_assignments_long.csv": assignments,
        "15_s3_spec_b_assignments_wide.csv": wide,
        "15_s3_spec_b_cluster_profiles.csv": profiles,
        "15_s3_spec_b_medoids.csv": medoids_df,
        "15_s3_spec_b_community_acreage_by_cluster.csv": compare_df,
        "15_s3_spec_a_vs_b_ari.csv": agree,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# S3 Spec B Segmentation Report (Script 15)

Generated: {timestamp}

## Scope

Land-size sensitivity for Spec A candidates k=2, 4, 5.

- Spec A 17 features retained
- `land_size_acres_spec_b`: observed acres retained; community acres left missing; 16 non-community specials imputed with median of observed acres ({acre_info['median_observed_acres']})
- `is_community_land` and `acres_special_imputed` added
- Missing-aware Gower (pairwise skip of missing features); **no zero-fill**
- Spec A assignments unchanged; coping not used; **no final k selected**

## Acreage treatment counts

```
{json.dumps(acre_info, indent=2)}
```

## Silhouette by k

```
{silhouettes.to_string(index=False)}
```

## Cluster sizes

```
{sizes.to_string(index=False)}
```

## Agreement with Spec A (same k)

```
{agree.to_string(index=False)}
```

## Community land and acreage by Spec B cluster

```
{compare_df.to_string(index=False)}
```

## Comparability

Community-land pairs omit acres from the pairwise Gower average when acres are missing. The community distinction is carried by `is_community_land` together with ownership and agricultural features. Spec B is a sensitivity analysis relative to Spec A, not a replacement until reviewed.

## Output files

- `outputs/tables/15_s3_spec_b_*.csv`
- `outputs/tables/15_s3_spec_b_parameters.json`
"""
    report_path = REPORTS_DIR / "15_segmentation_s3_spec_b.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    # Design doc pointer
    design = WORKBENCH_ROOT / "logs" / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    note = """

## 10. Spec B (Script 15) — run completed

Spec B land-size sensitivity completed for k=2, 4, 5 using missing-aware Gower. Spec A results unchanged. Final k still pending review of Spec A vs Spec B agreement and community/acreage cluster composition.

Report: `outputs/reports/15_segmentation_s3_spec_b.md`
"""
    if "## 10. Spec B (Script 15)" not in text:
        design.write_text(text.rstrip() + note, encoding="utf-8")
        print(f"Updated {design}")

    print("\nDone. Review Spec B before final k or coping profiling.")


if __name__ == "__main__":
    main()
