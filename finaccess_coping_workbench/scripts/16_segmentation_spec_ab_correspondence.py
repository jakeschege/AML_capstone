#!/usr/bin/env python3
"""Spec A vs Spec B membership correspondence audit for k=4 and k=5.

Matches clusters by maximum overlap (not assumed label equality), reports
retention/movement and profile comparison. Does not rerun PAM or use coping.
"""

from __future__ import annotations

import csv
import json
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
    EXPECTED_DROUGHT_N,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    PORTFOLIO_COUNT_VARS,
    PREPARED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    add_land_size_acres,
    ensure_output_dirs,
)

CANDIDATE_K = [4, 5]

# Provisional Spec A profile notes from prior review
PROVISIONAL_NOTES = {
    5: {
        0: "family_legumes_heavy",
        1: "own_name_crops",
        2: "livestock_community",
        3: "family_mixed_crop_livestock",
        4: "family_cereals_without_legumes",
    },
    4: {
        0: "family_crops_no_roots",
        1: "family_crops_with_roots",
        2: "livestock_community",
        3: "lease_crops",
    },
}

PROFILE_VARS = (
    list(AGRICULTURAL_HAS_VARS)
    + list(PORTFOLIO_COUNT_VARS)
    + list(LAND_OWNERSHIP_VARS)
    + ["is_community_land", "land_size_acres_spec_b", "acres_special_imputed"]
)


def match_clusters(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> pd.DataFrame:
    """Greedy max-overlap matching of Spec A clusters to Spec B clusters."""
    rows = []
    used_b = set()
    # Score matrix
    scores = np.zeros((k, k), dtype=int)
    for a in range(k):
        for b in range(k):
            scores[a, b] = int(np.logical_and(labels_a == a, labels_b == b).sum())

    # Greedy: repeatedly take highest remaining overlap
    remaining_a = set(range(k))
    remaining_b = set(range(k))
    matches = {}
    while remaining_a and remaining_b:
        best = (-1, -1, -1)
        for a in remaining_a:
            for b in remaining_b:
                if scores[a, b] > best[0]:
                    best = (scores[a, b], a, b)
        _, a, b = best
        matches[a] = b
        remaining_a.remove(a)
        remaining_b.remove(b)

    for a in range(k):
        b = matches[a]
        n_a = int((labels_a == a).sum())
        n_b = int((labels_b == b).sum())
        n_both = int(scores[a, b])
        rows.append(
            {
                "k": k,
                "spec_a_cluster": a,
                "spec_a_note": PROVISIONAL_NOTES[k].get(a, ""),
                "matched_spec_b_cluster": b,
                "overlap_n": n_both,
                "spec_a_n": n_a,
                "spec_b_n": n_b,
                "retention_pct_of_spec_a": round(100.0 * n_both / max(n_a, 1), 2),
                "precision_pct_of_spec_b": round(100.0 * n_both / max(n_b, 1), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("spec_a_cluster")


def contingency(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> pd.DataFrame:
    ct = pd.crosstab(
        pd.Series(labels_a, name="spec_a_cluster"),
        pd.Series(labels_b, name="spec_b_cluster"),
        margins=True,
    )
    ct = ct.reset_index()
    ct.insert(0, "k", k)
    return ct


def movement_from_a(
    labels_a: np.ndarray, labels_b: np.ndarray, match_df: pd.DataFrame, k: int
) -> pd.DataFrame:
    """Where Spec A members go in Spec B relative to matched cluster."""
    a_to_b = {
        int(r.spec_a_cluster): int(r.matched_spec_b_cluster)
        for r in match_df.itertuples()
    }
    rows = []
    for a in range(k):
        members = labels_a == a
        n = int(members.sum())
        matched_b = a_to_b[a]
        stay = int(np.logical_and(members, labels_b == matched_b).sum())
        # movements to each other B cluster
        for b in range(k):
            n_ab = int(np.logical_and(members, labels_b == b).sum())
            rows.append(
                {
                    "k": k,
                    "spec_a_cluster": a,
                    "spec_a_note": PROVISIONAL_NOTES[k].get(a, ""),
                    "spec_b_cluster": b,
                    "is_matched_pair": b == matched_b,
                    "n_households": n_ab,
                    "pct_of_spec_a_cluster": round(100.0 * n_ab / max(n, 1), 2),
                }
            )
        rows.append(
            {
                "k": k,
                "spec_a_cluster": a,
                "spec_a_note": PROVISIONAL_NOTES[k].get(a, ""),
                "spec_b_cluster": "left_matched_pair",
                "is_matched_pair": False,
                "n_households": n - stay,
                "pct_of_spec_a_cluster": round(100.0 * (n - stay) / max(n, 1), 2),
            }
        )
    return pd.DataFrame(rows)


def profile_compare(
    X: pd.DataFrame,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    match_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    rows = []
    for r in match_df.itertuples():
        a, b = int(r.spec_a_cluster), int(r.matched_spec_b_cluster)
        mask_a = labels_a == a
        mask_b = labels_b == b
        for var in PROFILE_VARS:
            if var not in X.columns:
                continue
            sa = pd.to_numeric(X.loc[mask_a, var], errors="coerce")
            sb = pd.to_numeric(X.loc[mask_b, var], errors="coerce")
            ma = float(sa.mean(skipna=True)) if mask_a.any() else np.nan
            mb = float(sb.mean(skipna=True)) if mask_b.any() else np.nan
            rows.append(
                {
                    "k": k,
                    "spec_a_cluster": a,
                    "spec_a_note": PROVISIONAL_NOTES[k].get(a, ""),
                    "matched_spec_b_cluster": b,
                    "variable": var,
                    "spec_a_mean": round(ma, 6) if pd.notna(ma) else np.nan,
                    "spec_b_mean": round(mb, 6) if pd.notna(mb) else np.nan,
                    "abs_diff": round(abs(ma - mb), 6)
                    if pd.notna(ma) and pd.notna(mb)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def recognisability(
    profile_df: pd.DataFrame, k: int
) -> pd.DataFrame:
    """Heuristic: key trait means still align with provisional Spec A notes."""
    checks = []
    if k == 5:
        rules = [
            (0, "has_legumes", "high", 0.7),
            (0, "land_ownership_family_owned_e_g_inherited_family_land", "high", 0.6),
            (1, "land_ownership_own_under_your_name", "high", 0.5),
            (2, "is_community_land", "high", 0.4),
            (2, "has_other_livestock", "high", 0.7),
            (3, "has_other_livestock", "high", 0.7),
            (3, "has_cereals", "high", 0.7),
            (4, "has_legumes", "low", 0.35),
            (4, "land_ownership_family_owned_e_g_inherited_family_land", "high", 0.6),
        ]
    else:
        rules = [
            (0, "has_roots_tubers", "low", 0.15),
            (1, "has_roots_tubers", "high", 0.7),
            (2, "is_community_land", "high", 0.3),
            (2, "has_other_livestock", "high", 0.7),
            (3, "land_ownership_lease_rent_you_dont_own", "high", 0.3),
        ]

    sub = profile_df.loc[profile_df["k"] == k]
    for a, var, direction, thresh in rules:
        row = sub.loc[
            (sub["spec_a_cluster"] == a) & (sub["variable"] == var)
        ]
        if row.empty:
            continue
        mb = float(row.iloc[0]["spec_b_mean"])
        ma = float(row.iloc[0]["spec_a_mean"])
        if direction == "high":
            ok_b = mb >= thresh
            ok_a = ma >= thresh
        else:
            ok_b = mb <= thresh
            ok_a = ma <= thresh
        checks.append(
            {
                "k": k,
                "spec_a_cluster": a,
                "spec_a_note": PROVISIONAL_NOTES[k].get(a, ""),
                "variable": var,
                "direction": direction,
                "threshold": thresh,
                "spec_a_mean": round(ma, 4),
                "spec_b_mean": round(mb, 4),
                "spec_a_meets": ok_a,
                "spec_b_meets": ok_b,
                "both_meet": ok_a and ok_b,
            }
        )
    return pd.DataFrame(checks)


def main() -> None:
    ensure_output_dirs()
    assign_a = pd.read_csv(TABLES_DIR / "12_s1_spec_a_assignments_wide.csv")
    assign_b = pd.read_csv(TABLES_DIR / "15_s3_spec_b_assignments_wide.csv")
    X_seg = pd.read_parquet(PREPARED_DIR / "09_X_segmentation.parquet")
    prepared = pd.read_parquet(PREPARED_DIR / "09_prepared_analysis_frame.parquet")

    X = X_seg.merge(assign_a, on="respondent_id", suffixes=("", "_a"))
    X = X.merge(
        assign_b.rename(
            columns={f"cluster_k{k}": f"spec_b_k{k}" for k in CANDIDATE_K}
        ),
        on="respondent_id",
    )
    land = add_land_size_acres(
        prepared[["respondent_id", "land_size", "land_size_unit"]].copy()
    )
    # Segmentation matrix already has land_size_acres; take unit/size from prepared
    X = X.drop(columns=[LAND_SIZE_STANDARDISED], errors="ignore")
    X = X.merge(
        land[["respondent_id", "land_size", "land_size_unit", LAND_SIZE_STANDARDISED]],
        on="respondent_id",
        how="left",
    )
    acres = pd.to_numeric(X[LAND_SIZE_STANDARDISED], errors="coerce")
    unit = pd.to_numeric(X["land_size_unit"], errors="coerce")
    size = pd.to_numeric(X["land_size"], errors="coerce")
    is_community = unit.eq(5)
    special = size.isin([98, 99]) & ~is_community
    median_obs = float(acres.loc[acres.notna()].median())
    acres_b = acres.copy()
    acres_b.loc[special] = median_obs
    acres_b.loc[is_community] = np.nan
    X["is_community_land"] = is_community.astype(int)
    X["acres_special_imputed"] = special.astype(int)
    X["land_size_acres_spec_b"] = acres_b

    if len(X) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected n={len(X)}")

    match_frames = []
    ct_frames = []
    move_frames = []
    profile_frames = []
    recog_frames = []

    for k in CANDIDATE_K:
        labels_a = X[f"cluster_k{k}"].to_numpy(dtype=int)
        labels_b = X[f"spec_b_k{k}"].to_numpy(dtype=int)
        match_df = match_clusters(labels_a, labels_b, k)
        match_frames.append(match_df)
        ct_frames.append(contingency(labels_a, labels_b, k))
        move_frames.append(movement_from_a(labels_a, labels_b, match_df, k))
        prof = profile_compare(X, labels_a, labels_b, match_df, k)
        profile_frames.append(prof)
        recog_frames.append(recognisability(prof, k))

        print(f"\nk={k} matches")
        print(match_df.to_string(index=False))

    matches = pd.concat(match_frames, ignore_index=True)
    contingencies = pd.concat(ct_frames, ignore_index=True)
    movements = pd.concat(move_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    recog = pd.concat(recog_frames, ignore_index=True)

    # Compact retention summary
    retention = matches[
        [
            "k",
            "spec_a_cluster",
            "spec_a_note",
            "matched_spec_b_cluster",
            "overlap_n",
            "spec_a_n",
            "spec_b_n",
            "retention_pct_of_spec_a",
            "precision_pct_of_spec_b",
        ]
    ].copy()

    params = {
        "experiment": "Spec_A_B_correspondence",
        "candidate_k": CANDIDATE_K,
        "matching": "greedy_maximum_overlap",
        "pam_rerun": False,
        "coping_used": False,
        "final_k_selected": None,
        "working_typology_status": "k5_provisionally_preferred_pending_this_audit",
    }
    (TABLES_DIR / "16_ab_correspondence_parameters.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )

    outputs = {
        "16_ab_cluster_matches.csv": matches,
        "16_ab_contingency_tables.csv": contingencies,
        "16_ab_membership_movement.csv": movements,
        "16_ab_retention_summary.csv": retention,
        "16_ab_matched_profile_comparison.csv": profiles,
        "16_ab_profile_recognisability.csv": recog,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    # Key trait diffs for report
    key_vars = [
        "has_legumes",
        "has_cereals",
        "has_roots_tubers",
        "has_other_livestock",
        "n_crop_types",
        "land_ownership_own_under_your_name",
        "land_ownership_family_owned_e_g_inherited_family_land",
        "land_ownership_lease_rent_you_dont_own",
        "is_community_land",
        "land_size_acres_spec_b",
    ]
    key_prof = profiles.loc[profiles["variable"].isin(key_vars)].copy()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Spec A / Spec B Correspondence Audit (Script 16)

Generated: {timestamp}

## Purpose

Verify household-level correspondence between Spec A and Spec B for k=4 and k=5 using maximum-overlap cluster matching. No PAM rerun; no coping outcomes; no final k locked in this script.

## Cluster matches (greedy max overlap)

```
{matches.to_string(index=False)}
```

## Retention summary

```
{retention.to_string(index=False)}
```

## Contingency tables

```
{contingencies.to_string(index=False)}
```

## Profile recognisability heuristics

```
{recog.to_string(index=False)}
```

## Key trait means for matched pairs

```
{key_prof.sort_values(['k','spec_a_cluster','variable']).to_string(index=False)}
```

## Review guidance

- High retention for all five k=5 matched pairs supports treating Spec B as preserving the Spec A typology.
- Low retention concentrated in one Spec A group would mean the ARI is driven by other clusters.
- Recognisability failures on Spec B means would argue against adopting the provisional labels.

Working typology remains **k=5 provisionally preferred**, with k=4 as alternative, until this audit is accepted and labels are confirmed.

## Output files

- `outputs/tables/16_ab_*.csv`
- `outputs/tables/16_ab_correspondence_parameters.json`
"""
    path = REPORTS_DIR / "16_segmentation_spec_ab_correspondence.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")

    design = WORKBENCH_ROOT / "logs" / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    note = """

## 11. Spec A/B correspondence (Script 16)

Membership overlap audit for k=4 and k=5 using greedy maximum-overlap matching. Supports the provisional working typology decision (k=5 preferred, k=4 alternative) once reviewed.

Report: `outputs/reports/16_segmentation_spec_ab_correspondence.md`
"""
    if "## 11. Spec A/B correspondence" not in text:
        design.write_text(text.rstrip() + note, encoding="utf-8")
        print(f"Updated {design}")

    print("\nRecognisability")
    print(recog.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
