#!/usr/bin/env python3
"""Script 04 — Prepare Model A / Model B feature matrices for all-shocks cohort."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import (
    AGRICULTURAL_HAS_VARS,
    LAND_OWNERSHIP_VARS,
    MAIN_SHOCK_TO_INDICATOR,
    PORTFOLIO_COUNT_VARS,
    PREPARED,
    SEX_CODE_TO_BIN,
    SHOCK_INDICATOR_VARS,
    TABLES,
    URBAN_RURAL_CODE_TO_BIN,
    add_land_size_acres,
    ensure_dirs,
)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    return cleaned.strip("_") or "unknown"


def county_categories_from_frame(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Use one-hot dictionary from drought workbench if present; else codes only."""
    path = (
        PREPARED.parent.parent.parent
        / "finaccess_coping_workbench"
        / "outputs"
        / "tables"
        / "09_onehot_category_dictionary.csv"
    )
    if path.exists():
        oh = pd.read_csv(path)
        rows = []
        for _, r in oh.loc[oh["source_variable"] == "county"].iterrows():
            rows.append((int(r["code"]), str(r["label"])))
        rows.sort(key=lambda x: x[0])
        return rows
    codes = sorted(int(x) for x in df["county"].dropna().unique())
    return [(c, f"county_{c}") for c in codes]


def build_base_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Demographics + agri + ownership; acres computed but not in Model A."""
    out = pd.DataFrame(index=df.index)
    dictionary = []

    out["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    dictionary.append({"feature": "age_years", "group": "demographic", "in_model_a": True, "in_model_b": True})

    out["sex_bin"] = df["sex"].map(SEX_CODE_TO_BIN).astype(float)
    dictionary.append({"feature": "sex_bin", "group": "demographic", "in_model_a": True, "in_model_b": True})

    out["education"] = pd.to_numeric(df["education"], errors="coerce")
    dictionary.append({"feature": "education", "group": "demographic", "in_model_a": True, "in_model_b": True})

    out["urban_rural_bin"] = df["urban_rural"].map(URBAN_RURAL_CODE_TO_BIN).astype(float)
    dictionary.append(
        {"feature": "urban_rural_bin", "group": "demographic", "in_model_a": True, "in_model_b": True}
    )

    cats = county_categories_from_frame(df)
    for code, label in cats:
        col = f"county_{code}_{slugify(label)}"
        out[col] = (pd.to_numeric(df["county"], errors="coerce") == code).astype(float)
        dictionary.append({"feature": col, "group": "county_onehot", "in_model_a": True, "in_model_b": True})

    for c in AGRICULTURAL_HAS_VARS + PORTFOLIO_COUNT_VARS + LAND_OWNERSHIP_VARS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
        dictionary.append(
            {
                "feature": c,
                "group": "agricultural" if c.startswith("has_") or c.startswith("n_") else "land_ownership",
                "in_model_a": True,
                "in_model_b": True,
            }
        )

    # Acres for documentation / optional segmentation — NOT in Model A primary
    acres_df = add_land_size_acres(df.copy())
    out["land_size_acres"] = pd.to_numeric(acres_df["land_size_acres"], errors="coerce")
    dictionary.append(
        {
            "feature": "land_size_acres",
            "group": "land_size",
            "in_model_a": False,
            "in_model_b": False,
            "note": "Excluded from primary Model A following acres-excluded decision; retained for audit",
        }
    )
    return out, dictionary


def main() -> None:
    ensure_dirs()
    model = pd.read_parquet(PREPARED / "all_shocks_modelling_cohort.parquet")
    print(f"Modelling cohort n={len(model)}")

    base, dictionary = build_base_features(model)

    # Model A: drop acres
    model_a_cols = [d["feature"] for d in dictionary if d.get("in_model_a")]
    X_a = base[model_a_cols].copy()

    # Model B: A + all R2A shocks + main_shock one-hots
    X_b = X_a.copy()
    for c in SHOCK_INDICATOR_VARS:
        if c not in model.columns:
            continue
        X_b[c] = pd.to_numeric(model[c], errors="coerce")
        dictionary.append(
            {
                "feature": c,
                "group": "r2a_shock",
                "in_model_a": False,
                "in_model_b": True,
                "leakage_note": (
                    "Same R2 module as coping (R2C/Classification). Contemporaneous survey "
                    "context — not derived from the coping outcome codes, but temporally "
                    "linked. Treat as shock-context predictors with caution."
                ),
            }
        )

    # Main shock one-hots for codes present
    for code, ind in MAIN_SHOCK_TO_INDICATOR.items():
        col = f"main_shock_{code}"
        X_b[col] = (pd.to_numeric(model["main_shock_type"], errors="coerce") == code).astype(float)
        dictionary.append(
            {
                "feature": col,
                "group": "r2b_main_shock",
                "in_model_a": False,
                "in_model_b": True,
                "leakage_note": (
                    "R2B main shock is asked within the shock module before/with coping. "
                    "It does not encode Classification_A_Resilience, but may be jointly "
                    "determined with coping narrative. Document as contextual, not causal."
                ),
            }
        )

    y = model["coping_category"].astype(str)
    ids = model["respondent_id"] if "respondent_id" in model.columns else model.index.astype(str)

    # Drop rows with NA in Model A predictors (strict)
    mask_a = X_a.notna().all(axis=1) & y.notna()
    X_a_f = X_a.loc[mask_a].reset_index(drop=True)
    y_a = y.loc[mask_a].reset_index(drop=True)
    id_a = ids.loc[mask_a].reset_index(drop=True)

    # Model B: shock NAs — keep confirmed shocks but leave NA as NA for imputer in CV later;
    # for matrix save, require Model A complete; allow shock NA (will impute in pipeline as 0 only inside CV? 
    # User said do not treat missing as zero. So for Model B fitting we should impute carefully.
    # For saved matrix: keep NaN in shock columns; CV pipeline will use SimpleImputer.
    mask_b = X_a.notna().all(axis=1) & y.notna()  # same base completeness
    X_b_f = X_b.loc[mask_b].reset_index(drop=True)
    y_b = y.loc[mask_b].reset_index(drop=True)

    # Attach ids/target for saved frames
    save_a = X_a_f.copy()
    save_a.insert(0, "respondent_id", id_a.values)
    save_a["coping_category"] = y_a.values

    save_b = X_b_f.copy()
    save_b.insert(0, "respondent_id", ids.loc[mask_b].reset_index(drop=True).values)
    save_b["coping_category"] = y_b.values

    save_a.to_parquet(PREPARED / "model_a_features.parquet", index=False)
    save_b.to_parquet(PREPARED / "model_b_features.parquet", index=False)
    save_a.to_csv(PREPARED / "model_a_features.csv", index=False)
    save_b.to_csv(PREPARED / "model_b_features.csv", index=False)

    dict_df = pd.DataFrame(dictionary)
    dict_df.to_csv(TABLES / "04_feature_dictionary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "matrix": "Model_A",
                "n_observations": len(save_a),
                "n_predictors": len(model_a_cols),
                "excluded": "land_size_acres; R2C; targets; ids; weights; coping actions",
            },
            {
                "matrix": "Model_B",
                "n_observations": len(save_b),
                "n_predictors": len(save_b.columns) - 2,
                "excluded": "land_size_acres; R2C; targets; ids; weights; coping actions",
                "added": "R2A shock indicators + R2B main_shock one-hots",
            },
        ]
    )
    summary.to_csv(TABLES / "04_matrix_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Leakage register short
    leak = pd.DataFrame(
        [
            {
                "variable_group": "R2C coping actions",
                "status": "excluded",
                "reason": "Direct coping responses used to characterise the target",
            },
            {
                "variable_group": "Classification_A / coping_category*",
                "status": "excluded_as_predictor",
                "reason": "Target (or derived from target)",
            },
            {
                "variable_group": "R2A shock indicators",
                "status": "Model_B_only",
                "reason": "Shock context; same module as coping — contextual not outcome-derived",
            },
            {
                "variable_group": "R2B main_shock",
                "status": "Model_B_only",
                "reason": "Main shock among experienced shocks; not the resilience classification",
            },
            {
                "variable_group": "land_size_acres",
                "status": "excluded_from_A_and_B_primary",
                "reason": "Follows acres-excluded primary specification from drought work",
            },
        ]
    )
    leak.to_csv(TABLES / "04_leakage_and_inclusion_notes.csv", index=False)
    print("Script 04 complete.")


if __name__ == "__main__":
    main()
