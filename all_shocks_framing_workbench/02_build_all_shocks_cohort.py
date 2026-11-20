#!/usr/bin/env python3
"""Script 02 — Build all-shocks analytical population (one row per household)."""

from __future__ import annotations

import pandas as pd

from config import (
    CLASS_ORDER,
    PREPARED,
    SHOCK_INDICATOR_VARS,
    TABLES,
    ensure_dirs,
    load_farming_frame,
)


def main() -> None:
    ensure_dirs()
    df = load_farming_frame().copy()
    n_farming = len(df)

    shocks = [c for c in SHOCK_INDICATOR_VARS if c in df.columns]
    shock_mat = df[shocks].apply(pd.to_numeric, errors="coerce")

    any_positive = (shock_mat == 1).any(axis=1)
    all_zero_no_na = (shock_mat == 0).all(axis=1) & shock_mat.notna().all(axis=1)
    # Incomplete: no confirmed positive, but at least one NA among shock indicators
    incomplete = (~any_positive) & shock_mat.isna().any(axis=1)

    df["any_shock_exposed"] = any_positive.astype(int)
    # Count only confirmed positives; do not treat NA as zero in the sum
    df["number_of_shocks_reported"] = (shock_mat == 1).sum(axis=1).astype(int)

    has_coping = df["coping_category"].notna() & df["coping_category"].isin(CLASS_ORDER)

    flow = [
        {"stage": "farming_households_reported_land_size", "n": n_farming},
        {"stage": "any_shock_exposed_confirmed_positive", "n": int(any_positive.sum())},
        {"stage": "no_shocks_complete_information_all_zero", "n": int(all_zero_no_na.sum())},
        {"stage": "incomplete_shock_information_no_confirmed_positive", "n": int(incomplete.sum())},
        {
            "stage": "shock_exposed_with_valid_coping_category",
            "n": int((any_positive & has_coping).sum()),
        },
        {
            "stage": "shock_exposed_missing_or_invalid_coping",
            "n": int((any_positive & ~has_coping).sum()),
        },
        {
            "stage": "primary_modelling_population",
            "n": int((any_positive & has_coping).sum()),
            "definition": "any_shock_exposed==1 AND coping_category in four-class set",
        },
    ]
    flow_df = pd.DataFrame(flow)
    flow_df.to_csv(TABLES / "cohort_flow.csv", index=False)
    print(flow_df.to_string(index=False))

    # Broader shock-exposed for descriptive analysis
    exposed = df.loc[any_positive].copy()
    # Primary modelling cohort
    model = df.loc[any_positive & has_coping].copy()

    # Collapsed three-class (existing convention)
    model["coping_category_collapsed"] = model["coping_category"].map(
        {
            "formal": "formal_or_informal",
            "informal": "formal_or_informal",
            "personal": "personal",
            "did_nothing": "did_nothing",
        }
    )

    tgt = (
        model["coping_category"]
        .value_counts()
        .reindex(CLASS_ORDER)
        .rename_axis("coping_category")
        .reset_index(name="n")
    )
    tgt["pct"] = (100 * tgt["n"] / len(model)).round(2)
    tgt.to_csv(TABLES / "target_distribution.csv", index=False)
    print(tgt)

    # Save full farming frame with flags + shock-exposed + modelling cohort
    keep_cols = list(df.columns)
    exposed.to_parquet(PREPARED / "all_shocks_farming_cohort.parquet", index=False)
    exposed.to_csv(PREPARED / "all_shocks_farming_cohort.csv", index=False)
    model.to_parquet(PREPARED / "all_shocks_modelling_cohort.parquet", index=False)
    model.to_csv(PREPARED / "all_shocks_modelling_cohort.csv", index=False)

    # Also save farming frame with flags for transparency
    df[keep_cols].to_parquet(PREPARED / "farming_frame_with_shock_flags.parquet", index=False)

    meta = {
        "n_farming": n_farming,
        "n_shock_exposed_descriptive": len(exposed),
        "n_modelling": len(model),
        "shock_columns_used": shocks,
        "missingness_rule": (
            "any_shock_exposed if any R2A indicator == 1; "
            "NA not treated as 0; households with only NA/0 and some NA flagged incomplete"
        ),
    }
    pd.Series(meta).to_csv(TABLES / "02_cohort_meta.csv")
    print(f"Saved exposed n={len(exposed)}, modelling n={len(model)}")
    print("Script 02 complete.")


if __name__ == "__main__":
    main()
