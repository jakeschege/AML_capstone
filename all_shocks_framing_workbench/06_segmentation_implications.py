#!/usr/bin/env python3
"""Script 06 — Light assessment of segmentation implications (no full PAM rerun)."""

from __future__ import annotations

import pandas as pd

from config import (
    AGRICULTURAL_HAS_VARS,
    LAND_OWNERSHIP_VARS,
    PORTFOLIO_COUNT_VARS,
    PREPARED,
    TABLES,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    model = pd.read_parquet(PREPARED / "all_shocks_modelling_cohort.parquet")
    drought_n = 1552
    all_n = len(model)

    seg_features = AGRICULTURAL_HAS_VARS + PORTFOLIO_COUNT_VARS + LAND_OWNERSHIP_VARS
    # Spec A excludes acres; Spec B includes acres — note both
    n_feat_a = len(seg_features)
    # Gower distance matrix is O(n^2) in memory/time for naive dense storage
    pairs_drought = drought_n * (drought_n - 1) / 2
    pairs_all = all_n * (all_n - 1) / 2

    rows = [
        {
            "item": "drought_exposed_segmentation_n",
            "value": drought_n,
            "note": "Existing Spec A PAM sample",
        },
        {
            "item": "all_shocks_modelling_n",
            "value": all_n,
            "note": "Proposed recomputation sample if segmentation follows modelling cohort",
        },
        {
            "item": "spec_a_n_features",
            "value": n_feat_a,
            "note": "Agri + counts + ownership; acres excluded (Spec A)",
        },
        {
            "item": "pairwise_comparisons_drought",
            "value": int(pairs_drought),
            "note": "Approx pairs for dense Gower",
        },
        {
            "item": "pairwise_comparisons_all_shocks",
            "value": int(pairs_all),
            "note": "Approx pairs for dense Gower",
        },
        {
            "item": "pair_ratio_vs_drought",
            "value": round(pairs_all / pairs_drought, 2),
            "note": "Relative compute/memory pressure vs current Spec A",
        },
        {
            "item": "recommendation",
            "value": "recompute_spec_a_on_all_shocks_cohort",
            "note": (
                "Cluster assignments from drought-only PAM are not valid out-of-sample "
                "labels for non-drought shock households. Recompute Spec A on the "
                "all-shocks modelling (or shock-exposed) population using the same "
                "agricultural features; exclude shock/coping from clustering. "
                "Defer full bootstrap stability until after a single k-screen."
            ),
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "06_segmentation_implications.csv", index=False)
    print(out.to_string(index=False))
    print("Script 06 complete (no PAM run).")


if __name__ == "__main__":
    main()
