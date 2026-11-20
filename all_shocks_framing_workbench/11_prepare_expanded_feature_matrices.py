#!/usr/bin/env python3
"""Script 11 — Build expanded feature matrices from Script 10 approved candidates.

Does not overwrite model_a_features / model_b_features.
"""

from __future__ import annotations

import pandas as pd

from config import MAIN_SHOCK_TO_INDICATOR, PREPARED, TABLES, ensure_dirs
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_build_base():
    path = Path(__file__).resolve().parent / "04_prepare_all_shocks_features.py"
    spec = spec_from_file_location("prep04", path)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_base_features


def main() -> None:
    ensure_dirs()
    build_base_features = load_build_base()

    approved = pd.read_csv(TABLES / "10_approved_candidates.csv")
    print("Approved candidates:")
    print(approved[["variable", "expansion_group"]].to_string(index=False))

    model = pd.read_parquet(PREPARED / "all_shocks_modelling_cohort.parquet")
    assert len(model) == 5241
    # Preserve respondent order from existing Model A matrix
    ref = pd.read_parquet(PREPARED / "model_a_features.parquet")[["respondent_id", "coping_category"]]
    assert len(ref) == 5241
    model = ref.merge(model.drop(columns=["coping_category"], errors="ignore"), on="respondent_id", how="left")
    assert len(model) == 5241
    assert model["coping_category"].notna().all()

    welfare = pd.read_parquet(PREPARED / "10_stata_welfare_extract_modelling.parquet")
    welfare = welfare.drop(columns=["interview__id"], errors="ignore")
    model = model.merge(welfare, on="respondent_id", how="left")

    base, _ = build_base_features(model)
    # Model A columns (exclude acres)
    model_a_cols = [
        c
        for c in base.columns
        if c != "land_size_acres"
    ]
    # Align with existing saved Model A feature list where possible
    existing_a = [c for c in ref.columns if c not in ("respondent_id", "coping_category")]
    # Prefer exact existing Model A columns from saved matrix for fair baseline
    X_existing_a = pd.read_parquet(PREPARED / "model_a_features.parquet").drop(
        columns=["respondent_id", "coping_category"]
    )

    # R2B main-shock one-hots (same coding as Script 04)
    r2b = pd.DataFrame(index=model.index)
    for code, _ind in MAIN_SHOCK_TO_INDICATOR.items():
        r2b[f"main_shock_{code}"] = (
            pd.to_numeric(model["main_shock_type"], errors="coerce") == code
        ).astype(float)

    baseline = pd.concat([X_existing_a.reset_index(drop=True), r2b.reset_index(drop=True)], axis=1)

    # Approved sets by group
    by_group = {
        g: approved.loc[approved["expansion_group"] == g, "variable"].tolist()
        for g in approved["expansion_group"].unique()
    }
    agri = by_group.get("agricultural_diversity", [])
    farm = by_group.get("farm_challenge", [])
    fin = by_group.get("financial", [])
    welfare_vars = by_group.get("welfare_support", [])

    def pull(cols: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=model.index)
        for c in cols:
            if c not in model.columns:
                raise KeyError(f"Approved variable missing from modelling frame/extract: {c}")
            out[c] = pd.to_numeric(model[c], errors="coerce")
        return out.reset_index(drop=True)

    specs = {
        "baseline_model_a_plus_r2b": baseline.copy(),
        "agri_diversity_expansion": baseline.copy(),  # may stay identical
        "farm_challenge_expansion": pd.concat([baseline, pull(farm)], axis=1) if farm else baseline.copy(),
        "financial_expansion": pd.concat([baseline, pull(fin)], axis=1) if fin else baseline.copy(),
        "welfare_expansion": pd.concat([baseline, pull(welfare_vars)], axis=1)
        if welfare_vars
        else baseline.copy(),
    }
    if agri:
        specs["agri_diversity_expansion"] = pd.concat([baseline, pull(agri)], axis=1)

    combined_cols = []
    for g in ("agricultural_diversity", "farm_challenge", "financial", "welfare_support"):
        combined_cols.extend(by_group.get(g, []))
    # unique preserve order
    seen = set()
    combined_unique = []
    for c in combined_cols:
        if c not in seen:
            seen.add(c)
            combined_unique.append(c)
    specs["combined_expansion"] = (
        pd.concat([baseline, pull(combined_unique)], axis=1) if combined_unique else baseline.copy()
    )

    dictionary_rows = []
    summary_rows = []
    for name, X in specs.items():
        assert len(X) == 5241
        # Drop duplicated columns if any
        X = X.loc[:, ~X.columns.duplicated()].copy()
        save = X.copy()
        save.insert(0, "respondent_id", ref["respondent_id"].values)
        save["coping_category"] = ref["coping_category"].values
        out_path = PREPARED / f"11_{name}_features.parquet"
        save.to_parquet(out_path, index=False)
        save.to_csv(PREPARED / f"11_{name}_features.csv", index=False)

        added = [c for c in X.columns if c not in baseline.columns]
        summary_rows.append(
            {
                "specification": name,
                "n_observations": len(save),
                "n_predictors": X.shape[1],
                "n_added_vs_baseline": len(added),
                "added_variables": ";".join(added) if added else "",
                "note": (
                    "No approved agricultural diversity variables; identical to baseline."
                    if name == "agri_diversity_expansion" and not agri
                    else ""
                ),
            }
        )
        for c in X.columns:
            dictionary_rows.append(
                {
                    "specification": name,
                    "feature": c,
                    "in_baseline": c in baseline.columns,
                    "expansion_group": (
                        approved.loc[approved["variable"] == c, "expansion_group"].iloc[0]
                        if c in set(approved["variable"])
                        else ("baseline_model_a_plus_r2b" if c in baseline.columns else "other")
                    ),
                }
            )
        print(f"{name}: p={X.shape[1]} added={len(added)}")

    pd.DataFrame(summary_rows).to_csv(TABLES / "11_expanded_matrix_summary.csv", index=False)
    pd.DataFrame(dictionary_rows).to_csv(TABLES / "11_expanded_feature_dictionary.csv", index=False)
    approved.to_csv(TABLES / "11_approved_variables_used.csv", index=False)
    print("Script 11 complete.")


if __name__ == "__main__":
    main()
