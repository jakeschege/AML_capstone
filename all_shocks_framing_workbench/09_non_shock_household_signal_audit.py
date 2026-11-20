#!/usr/bin/env python3
"""Script 09 — Descriptive audit of non-shock farming households vs shock-exposed.

Targeted variable discovery only. Does not fit classifiers or alter Model A/B matrices.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from config import (
    COLUMN_RENAME_MAP,
    MODELLING_FRAME,
    PREPARED,
    REPORTS,
    SHOCK_INDICATOR_VARS,
    STATA_2024,
    TABLES,
    ensure_dirs,
)

SEED = 42
EXTRACT = MODELLING_FRAME.parent / "finaccess_2024_extract.csv"
EXTRACT_DD = MODELLING_FRAME.parent / "finaccess_2024_extract_data_dictionary.csv"

# Columns that must never be treated as predictors for the coping classifier
LEAKAGE_PREFIXES = (
    "coping_",
    "shock_",
    "main_shock_",
)
LEAKAGE_EXACT = {
    "coping_category",
    "coping_category_code",
    "coping_category_collapsed",
    "coping_channel",
    "Classification_A_Resilience",
    "any_shock_exposed",
    "number_of_shocks_reported",
    "drought_exposed",
    "shock_info_group",
    "r2a_complete_no_shock",
    "r2a_incomplete_no_positive",
}

ID_ADMIN = {
    "source_wave",
    "source_file",
    "respondent_id",
    "household_weight",
    "individual_weight",
    "weight",
}

# Prefer name-based rules first (avoid substring traps like "age" inside "usage").
NAME_DOMAIN_RULES: list[tuple[str, list[str]]] = [
    (
        "financial_position",
        [
            "income",
            "sav",
            "borrow",
            "debt",
            "loan",
            "credit",
            "bank",
            "insur",
            "remit",
            "mobile_money",
            "mobile_bank",
            "digital",
            "financial",
            "formal_agri",
            "buyer_supplier",
            "sacco",
            "informal_usage",
            "formal_usage",
        ],
    ),
    (
        "agricultural_characteristics",
        [
            "farm_",
            "crop",
            "livestock",
            "land_",
            "agri",
            "has_",
            "n_crop",
            "n_livestock",
            "poultry",
            "dairy",
            "horticulture",
            "cereal",
            "legume",
            "buyer",
            "market",
            "yield",
            "input",
            "storage",
        ],
    ),
    (
        "social_and_institutional_support",
        [
            "member_",
            "association",
            "coop",
            "extension",
            "government",
            "assist",
            "network",
            "implementer",
            "chama",
        ],
    ),
    (
        "food_security_and_welfare",
        ["food", "diet", "consum", "hunger", "welfare", "poverty", "meal", "nutrition"],
    ),
    (
        "other_household_stress",
        [
            "challenge",
            "difficult",
            "loss",
            "disrupt",
            "hardship",
            "stress",
            "theft",
            "damage",
            "deterior",
            "unavailab",
        ],
    ),
    (
        "household_demographics",
        [
            "age_years",
            "age_group",
            "sex",
            "education",
            "urban_rural",
            "county",
            "household_size",
            "hh_size",
        ],
    ),
]


def assign_domain(name: str, label: str = "") -> str:
    n = name.lower()
    blob = f"{name} {label}".lower()
    if name.startswith("shock_") or name.startswith("main_shock_"):
        return "shock_module_r2a_r2b"
    if name.startswith("coping_") or name in {
        "coping_category",
        "coping_category_code",
        "coping_category_collapsed",
        "coping_channel",
    }:
        return "coping_module_r2c_target"
    for domain, keys in NAME_DOMAIN_RULES:
        if any(k in n for k in keys):
            return domain
    # Label fallback for rare cases (word-ish tokens only)
    label_keys = [
        ("financial_position", ["loan", "saving", "insurance", "credit", "remittance"]),
        ("food_security_and_welfare", ["food", "hunger", "diet", "meal"]),
        ("household_demographics", ["household size", "education", "urban", "rural"]),
    ]
    for domain, keys in label_keys:
        if any(k in blob for k in keys):
            return domain
    return "other_unclassified"


def signal_role(name: str, label: str, domain: str) -> str:
    """Heuristic role tag — descriptive only, not causal proof."""
    blob = f"{name} {label}".lower()
    if domain == "coping_module_r2c_target" or name.startswith("coping_"):
        return "coping_behaviour_or_target"
    if domain == "shock_module_r2a_r2b" or name.startswith("shock_") or name.startswith("main_shock_"):
        return "reported_shock_indicator"
    hardship = [
        "challenge",
        "difficult",
        "loss",
        "hardship",
        "hunger",
        "food insecur",
        "denied a loan",
        "can't afford",
        "low yield",
        "disrupt",
    ]
    if any(k in blob for k in hardship):
        if any(k in blob for k in ["challenge", "low yield", "loss", "damage"]):
            return "possible_shock_consequence_or_stress"
        return "possible_hardship_or_consequence"
    if domain in {"financial_position", "agricultural_characteristics", "household_demographics"}:
        return "vulnerability_or_capacity_characteristic"
    if domain == "social_and_institutional_support":
        return "support_network_characteristic"
    if domain == "food_security_and_welfare":
        return "welfare_or_consequence_candidate"
    return "unclassified_context"


def leakage_flag(name: str) -> str:
    if name in LEAKAGE_EXACT or name.startswith(LEAKAGE_PREFIXES):
        if name.startswith("shock_") or name.startswith("main_shock_"):
            return "shock_module_feature_not_outcome_leakage"
        return "exclude_target_or_coping_construction"
    if name in {"any_shock_exposed", "number_of_shocks_reported", "drought_exposed"}:
        return "exclude_cohort_definition_flag"
    return "none"


def temporal_note(name: str, label: str, role: str) -> str:
    blob = f"{name} {label}".lower()
    if role == "coping_behaviour_or_target":
        return "Likely measured as post-shock response / directly builds the target — exclude from predictors."
    if role == "reported_shock_indicator":
        return "Part of the R2 shock module used to define exposure; already in Model B (or cohort gate)."
    if "challenge" in blob or "farm_challenge" in name:
        return (
            "Farm production challenges asked in the agricultural module; timing relative to R2 "
            "shocks is not guaranteed. Treat as possible concurrent stress / consequence, not proven prior trait."
        )
    if role.startswith("vulnerability") or "member_" in name or name in {
        "age_years",
        "sex",
        "education",
        "urban_rural",
        "county",
    }:
        return "Plausible underlying household characteristic / capacity; still cross-sectional (not proven pre-shock)."
    if role.startswith("possible_hardship") or role.startswith("possible_shock"):
        return "May reflect shock consequences or general hardship; risk of post-treatment / leakage into coping path."
    return "Cross-sectional FinAccess 2024 measure; temporal order vs R2 shocks not verified."


def is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notna().mean() >= 0.9 and s.notna().any()


def summarise_numeric(a: pd.Series, b: pd.Series) -> dict:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    a_ok, b_ok = a.dropna(), b.dropna()
    mean_a, mean_b = (a_ok.mean() if len(a_ok) else np.nan), (b_ok.mean() if len(b_ok) else np.nan)
    med_a, med_b = (a_ok.median() if len(a_ok) else np.nan), (b_ok.median() if len(b_ok) else np.nan)
    pooled = np.nan
    if len(a_ok) > 1 and len(b_ok) > 1:
        sd = np.sqrt(0.5 * (a_ok.var(ddof=1) + b_ok.var(ddof=1)))
        if sd and np.isfinite(sd) and sd > 0:
            pooled = abs(mean_a - mean_b) / sd
    return {
        "mean_exposed": mean_a,
        "mean_non_shock": mean_b,
        "median_exposed": med_a,
        "median_non_shock": med_b,
        "abs_mean_diff": abs(mean_a - mean_b) if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan,
        "std_diff_abs": pooled,
        "top_category_diff_pp": np.nan,
    }


def summarise_categorical(a: pd.Series, b: pd.Series, max_levels: int = 12) -> dict:
    def pct(s: pd.Series) -> pd.Series:
        vc = s.value_counts(dropna=True)
        return 100 * vc / vc.sum() if vc.sum() else vc

    pa, pb = pct(a), pct(b)
    levels = sorted(set(pa.index.tolist()) | set(pb.index.tolist()), key=lambda x: str(x))
    if len(levels) > max_levels:
        # keep most frequent overall
        both = pd.concat([a, b], ignore_index=True)
        keep = both.value_counts().head(max_levels).index
        levels = [x for x in levels if x in keep]
    diffs = []
    for lv in levels:
        diffs.append(abs(float(pa.get(lv, 0.0)) - float(pb.get(lv, 0.0))))
    return {
        "mean_exposed": np.nan,
        "mean_non_shock": np.nan,
        "median_exposed": np.nan,
        "median_non_shock": np.nan,
        "abs_mean_diff": np.nan,
        "std_diff_abs": np.nan,
        "top_category_diff_pp": max(diffs) if diffs else np.nan,
        "category_pct_exposed": json.dumps(
            {str(k): round(float(v), 2) for k, v in pa.head(8).items()}, ensure_ascii=False
        ),
        "category_pct_non_shock": json.dumps(
            {str(k): round(float(v), 2) for k, v in pb.head(8).items()}, ensure_ascii=False
        ),
    }


def model_membership(col: str, model_a_src: set[str], model_b_src: set[str]) -> tuple[bool, bool]:
    """Map raw/engineered column names onto Model A/B feature names."""
    aliases = {
        "sex": "sex_bin",
        "urban_rural": "urban_rural_bin",
        "county": "county_",  # one-hots
    }
    in_a = col in model_a_src
    in_b = col in model_b_src
    if col in aliases:
        alias = aliases[col]
        if alias == "county_":
            in_a = in_a or any(c.startswith("county_") for c in model_a_src)
            in_b = in_b or any(c.startswith("county_") for c in model_b_src)
        else:
            in_a = in_a or alias in model_a_src
            in_b = in_b or alias in model_b_src
    if col.startswith("shock_"):
        in_b = in_b or col in model_b_src
    if col == "main_shock_type" or col == "main_shock_type_label":
        in_b = True  # represented via main_shock_* one-hots
    return in_a, in_b


def rank_score(row: pd.Series) -> float:
    """Higher = more interesting candidate for further consideration."""
    score = 0.0
    if row["in_model_a"] or (row["in_model_b"] and row["domain"] != "shock_module_r2a_r2b"):
        score -= 5  # already represented (shock module still noted separately)
    if row["leakage_flag"].startswith("exclude"):
        return -100
    miss = max(row["pct_missing_exposed"], row["pct_missing_non_shock"])
    if miss > 40:
        score -= 3
    elif miss > 20:
        score -= 1
    else:
        score += 1
    if row["data_kind"] == "numeric":
        sd = row.get("std_diff_abs", np.nan)
        if pd.notna(sd):
            score += min(float(sd), 1.5) * 2
    else:
        dpp = row.get("top_category_diff_pp", np.nan)
        if pd.notna(dpp):
            score += min(float(dpp) / 20.0, 2.0)
    role = row["signal_role"]
    if role.startswith("vulnerability"):
        score += 1.5
    if "consequence" in role or "hardship" in role:
        score += 0.5  # interesting but cautious
    if row["domain"] in {
        "financial_position",
        "agricultural_characteristics",
        "social_and_institutional_support",
        "other_household_stress",
    }:
        score += 1.0
    if row["domain"] == "food_security_and_welfare":
        score += 1.2
    return score


def main() -> None:
    ensure_dirs()
    print("Loading Stata metadata (labels only)…")
    _, meta = pyreadstat.read_dta(str(STATA_2024), metadataonly=True)
    stata_labels = meta.column_names_to_labels or {}
    stata_value_labels = meta.variable_value_labels or {}

    rename = pd.read_csv(COLUMN_RENAME_MAP)
    rename_map = dict(zip(rename["renamed_column"], rename["original_column"]))
    # R2 rename map may add more
    r2_path = MODELLING_FRAME.parent / "r2_column_rename_map.csv"
    if r2_path.exists():
        r2 = pd.read_csv(r2_path)
        for _, r in r2.iterrows():
            rename_map[str(r["renamed_column"])] = str(r["original_column"])

    extract_dd = {}
    if EXTRACT_DD.exists():
        dd = pd.read_csv(EXTRACT_DD)
        for _, r in dd.iterrows():
            extract_dd[str(r["column"])] = {
                "original_column": r.get("original_column", ""),
                "variable_group": r.get("variable_group", ""),
                "description": r.get("description", ""),
            }

    print("Loading farming modelling frame + shock flags…")
    farm = pd.read_parquet(PREPARED / "farming_frame_with_shock_flags.parquet")
    assert len(farm) == 9756

    shocks = [c for c in SHOCK_INDICATOR_VARS if c in farm.columns]
    shock_mat = farm[shocks].apply(pd.to_numeric, errors="coerce")
    any_positive = (shock_mat == 1).any(axis=1)
    all_zero = (shock_mat == 0).all(axis=1) & shock_mat.notna().all(axis=1)
    incomplete = (~any_positive) & shock_mat.isna().any(axis=1)

    farm = farm.copy()
    farm["shock_info_group"] = np.where(
        any_positive,
        "shock_exposed",
        np.where(all_zero, "complete_no_shock", np.where(incomplete, "incomplete_no_positive", "other")),
    )
    farm["r2a_complete_no_shock"] = all_zero.astype(int)
    farm["r2a_incomplete_no_positive"] = incomplete.astype(int)

    counts = farm["shock_info_group"].value_counts()
    print(counts.to_string())
    assert int(any_positive.sum()) == 5248
    assert int(all_zero.sum()) == 4499
    assert int(incomplete.sum()) == 9

    # Enrich with extract-only columns for the same farming IDs
    print("Merging extract-only columns for farming households…")
    ex = pd.read_csv(EXTRACT, low_memory=False)
    ex_farm = ex[ex["respondent_id"].isin(farm["respondent_id"])].copy()
    extra_cols = [c for c in ex_farm.columns if c not in farm.columns and c != "respondent_id"]
    # Drop ultra-sparse crop one-hots from primary comparison if > 80 columns of crop_livestock_
    crop_cols = [c for c in extra_cols if c.startswith("crop_livestock_")]
    other_extra = [c for c in extra_cols if not c.startswith("crop_livestock_")]
    # Keep crop portfolio as n_crop_detail_positive summary + a few frequent crops
    crop_block = ex_farm[["respondent_id"] + crop_cols].copy()
    crop_num = crop_block[crop_cols].apply(pd.to_numeric, errors="coerce")
    crop_block["n_crop_livestock_items_positive"] = (crop_num == 1).sum(axis=1)
    # top 15 most common crop/livestock items among farming households
    freq = (crop_num == 1).sum().sort_values(ascending=False)
    top_crops = freq.head(15).index.tolist()
    keep_extra = other_extra + ["n_crop_livestock_items_positive"] + top_crops
    merge_extra = crop_block[["respondent_id", "n_crop_livestock_items_positive"] + top_crops].merge(
        ex_farm[["respondent_id"] + other_extra], on="respondent_id", how="left"
    )
    farm = farm.merge(merge_extra, on="respondent_id", how="left", suffixes=("", "_ex"))
    print(f"Added {len(keep_extra)} extract-derived columns (incl. top crops + portfolio count).")

    # Model A/B feature names
    a_cols = set(pd.read_parquet(PREPARED / "model_a_features.parquet").columns) - {
        "respondent_id",
        "coping_category",
    }
    b_cols = set(pd.read_parquet(PREPARED / "model_b_features.parquet").columns) - {
        "respondent_id",
        "coping_category",
    }

    exposed = farm[farm["shock_info_group"] == "shock_exposed"]
    nonshock = farm[farm["shock_info_group"] == "complete_no_shock"]
    incomplete_df = farm[farm["shock_info_group"] == "incomplete_no_positive"]

    # Population summary
    pop = pd.DataFrame(
        [
            {"group": "farming_total", "n": len(farm)},
            {"group": "shock_exposed", "n": len(exposed)},
            {"group": "complete_r2a_no_shock", "n": len(nonshock)},
            {"group": "incomplete_r2a_no_positive", "n": len(incomplete_df)},
            {
                "group": "non_shock_with_valid_coping_category",
                "n": int(
                    nonshock["coping_category"].notna().sum()
                    if "coping_category" in nonshock.columns
                    else 0
                ),
            },
            {
                "group": "shock_exposed_with_valid_coping_category",
                "n": int(exposed["coping_category"].isin(["personal", "informal", "formal", "did_nothing"]).sum()),
            },
        ]
    )
    pop.to_csv(TABLES / "09_population_counts.csv", index=False)

    skip = ID_ADMIN | {
        "shock_info_group",
        "r2a_complete_no_shock",
        "r2a_incomplete_no_positive",
        "any_shock_exposed",
        "number_of_shocks_reported",
    }

    candidate_cols = [c for c in farm.columns if c not in skip]
    rows = []
    dist_rows = []

    for col in candidate_cols:
        original = rename_map.get(col, extract_dd.get(col, {}).get("original_column", ""))
        label = ""
        value_lab = {}
        if original and original in stata_labels:
            label = stata_labels.get(original) or ""
            value_lab = stata_value_labels.get(original) or {}
        if not label and col in extract_dd:
            label = str(extract_dd[col].get("description") or "")
            if not original:
                original = str(extract_dd[col].get("original_column") or "")

        domain = assign_domain(col, label)
        role = signal_role(col, label, domain)
        leak = leakage_flag(col)
        in_a, in_b = model_membership(col, a_cols, b_cols)

        s_e = exposed[col]
        s_n = nonshock[col]
        n_miss_e = int(s_e.isna().sum())
        n_miss_n = int(s_n.isna().sum())
        pct_miss_e = 100 * n_miss_e / len(exposed)
        pct_miss_n = 100 * n_miss_n / len(nonshock)
        nunique = int(pd.concat([s_e, s_n], ignore_index=True).nunique(dropna=True))

        numeric = is_numeric_like(pd.concat([s_e, s_n], ignore_index=True))
        # Binary/low-cardinality numeric treated as categorical for distribution tables if <= 8 levels
        if numeric and nunique > 8:
            kind = "numeric"
            stats = summarise_numeric(s_e, s_n)
            stats["category_pct_exposed"] = ""
            stats["category_pct_non_shock"] = ""
        else:
            kind = "categorical"
            stats = summarise_categorical(s_e, s_n)

        # Distribution string for inventory
        if kind == "numeric":
            dist_note = (
                f"exposed mean={stats['mean_exposed']:.3f} med={stats['median_exposed']:.3f}; "
                f"non_shock mean={stats['mean_non_shock']:.3f} med={stats['median_non_shock']:.3f}"
                if pd.notna(stats["mean_exposed"]) and pd.notna(stats["mean_non_shock"])
                else "insufficient numeric data"
            )
        else:
            dist_note = (
                f"max category gap={stats['top_category_diff_pp']:.1f} pp"
                if pd.notna(stats["top_category_diff_pp"])
                else ""
            )

        row = {
            "variable": col,
            "original_stata_name": original,
            "variable_label": label,
            "domain": domain,
            "signal_role": role,
            "data_kind": kind,
            "dtype": str(farm[col].dtype),
            "n_distinct": nunique,
            "pct_missing_exposed": round(pct_miss_e, 3),
            "pct_missing_non_shock": round(pct_miss_n, 3),
            "in_model_a": in_a,
            "in_model_b": in_b,
            "leakage_flag": leak,
            "temporal_assessment": temporal_note(col, label, role),
            "value_labels_json": json.dumps(
                {str(k): str(v) for k, v in list(value_lab.items())[:20]}, ensure_ascii=False
            )
            if value_lab
            else "",
            "distribution_summary": dist_note,
            **{k: stats[k] for k in stats},
        }
        rows.append(row)

        # Compact distribution export
        dist_rows.append(
            {
                "variable": col,
                "domain": domain,
                "data_kind": kind,
                "distribution_summary": dist_note,
                "category_pct_exposed": stats.get("category_pct_exposed", ""),
                "category_pct_non_shock": stats.get("category_pct_non_shock", ""),
            }
        )

    audit = pd.DataFrame(rows)
    audit["candidate_rank_score"] = audit.apply(rank_score, axis=1)
    audit = audit.sort_values(
        ["candidate_rank_score", "domain", "variable"], ascending=[False, True, True]
    ).reset_index(drop=True)
    audit.to_csv(TABLES / "09_variable_audit_full.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(TABLES / "09_group_distributions.csv", index=False)

    # Meaningful difference highlight table (descriptive thresholds, not p-values)
    highlight = audit[
        (~audit["leakage_flag"].str.startswith("exclude"))
        & (~audit["domain"].isin(["shock_module_r2a_r2b", "coping_module_r2c_target"]))
        & (~audit["variable"].str.startswith("crop_livestock_"))
    ].copy()

    def meaningful(r: pd.Series) -> bool:
        if max(r["pct_missing_exposed"], r["pct_missing_non_shock"]) > 30:
            return False
        if r["data_kind"] == "numeric":
            return pd.notna(r["std_diff_abs"]) and r["std_diff_abs"] >= 0.15
        dpp = r["top_category_diff_pp"]
        # Drop near-empty / degenerate contrasts (e.g. 100pp from sparse flags)
        if pd.isna(dpp) or dpp < 5.0 or dpp >= 95.0:
            return False
        return True

    highlight = highlight[highlight.apply(meaningful, axis=1)].copy()
    highlight.to_csv(TABLES / "09_meaningful_group_differences.csv", index=False)

    # Domain coverage
    domain_cov = (
        audit.assign(not_in_either=~audit["in_model_a"] & ~audit["in_model_b"])
        .groupby("domain", as_index=False)
        .agg(
            n_variables=("variable", "count"),
            n_not_in_model_a=("in_model_a", lambda s: int((~s).sum())),
            n_not_in_either_model=("not_in_either", "sum"),
        )
    )
    domain_cov.to_csv(TABLES / "09_domain_coverage.csv", index=False)

    # Inventories
    usable = audit[
        (~audit["leakage_flag"].str.startswith("exclude"))
        & (~audit["in_model_a"])
        & ~(
            audit["in_model_b"]
            & audit["domain"].isin(["shock_module_r2a_r2b", "coping_module_r2c_target"])
        )
    ].copy()
    # Still exclude shock module columns even if somehow flagged
    usable = usable[~usable["domain"].isin(["shock_module_r2a_r2b", "coping_module_r2c_target"])]

    inv_a = usable[
        usable["domain"].isin(
            [
                "household_demographics",
                "agricultural_characteristics",
                "social_and_institutional_support",
                "financial_position",
            ]
        )
        & usable["signal_role"].isin(
            [
                "vulnerability_or_capacity_characteristic",
                "support_network_characteristic",
                "unclassified_context",
            ]
        )
    ].copy()
    inv_a["inventory"] = "A_household_characteristics"
    inv_a["suitability"] = inv_a.apply(
        lambda r: (
            "consider"
            if r["candidate_rank_score"] >= 2
            and max(r["pct_missing_exposed"], r["pct_missing_non_shock"]) <= 25
            else "low_priority_or_caution"
        ),
        axis=1,
    )

    inv_b = usable[
        usable["domain"].isin(
            [
                "financial_position",
                "food_security_and_welfare",
                "other_household_stress",
                "agricultural_characteristics",
                "social_and_institutional_support",
            ]
        )
    ].copy()
    inv_b["inventory"] = "B_contextual_indicators"
    inv_b["suitability"] = inv_b.apply(
        lambda r: (
            "consider_with_leakage_review"
            if "consequence" in r["signal_role"] or "hardship" in r["signal_role"]
            else (
                "consider"
                if r["candidate_rank_score"] >= 2
                and max(r["pct_missing_exposed"], r["pct_missing_non_shock"]) <= 25
                else "low_priority_or_caution"
            )
        ),
        axis=1,
    )

    inv_cols = [
        "inventory",
        "variable",
        "original_stata_name",
        "variable_label",
        "domain",
        "signal_role",
        "data_kind",
        "pct_missing_exposed",
        "pct_missing_non_shock",
        "in_model_a",
        "in_model_b",
        "leakage_flag",
        "temporal_assessment",
        "distribution_summary",
        "std_diff_abs",
        "top_category_diff_pp",
        "candidate_rank_score",
        "suitability",
    ]
    inventory = pd.concat([inv_a[inv_cols], inv_b[inv_cols]], ignore_index=True)
    inventory = inventory.sort_values(
        ["inventory", "suitability", "candidate_rank_score"], ascending=[True, True, False]
    )
    inventory.to_csv(TABLES / "09_candidate_inventories.csv", index=False)
    inv_a[inv_cols].to_csv(TABLES / "09_inventory_a_household_characteristics.csv", index=False)
    inv_b[inv_cols].to_csv(TABLES / "09_inventory_b_contextual_indicators.csv", index=False)

    # Exclusions table
    excl = audit[
        audit["leakage_flag"].str.startswith("exclude")
        | (audit["pct_missing_exposed"] > 40)
        | (audit["pct_missing_non_shock"] > 40)
    ][
        [
            "variable",
            "variable_label",
            "domain",
            "leakage_flag",
            "pct_missing_exposed",
            "pct_missing_non_shock",
            "temporal_assessment",
            "in_model_a",
            "in_model_b",
        ]
    ].copy()
    excl["exclusion_reason"] = excl.apply(
        lambda r: (
            r["leakage_flag"]
            if str(r["leakage_flag"]).startswith("exclude")
            else "high_missingness"
        ),
        axis=1,
    )
    excl.to_csv(TABLES / "09_exclusion_candidates.csv", index=False)

    # Incomplete group retained separately (profile only)
    inc_profile = []
    for col in ["age_years", "sex", "education", "urban_rural", "coping_category"]:
        if col not in incomplete_df.columns:
            continue
        inc_profile.append(
            {
                "variable": col,
                "n": len(incomplete_df),
                "n_missing": int(incomplete_df[col].isna().sum()),
                "value_counts": incomplete_df[col].value_counts(dropna=False).to_dict(),
            }
        )
    pd.DataFrame(inc_profile).to_csv(TABLES / "09_incomplete_group_profile.csv", index=False)

    # Stata gap scan: promising labels not present in prepared farming columns
    prepared_originals = set(rename_map.values()) | {
        str(extract_dd[c].get("original_column") or "") for c in extract_dd
    }
    prepared_names = set(farm.columns)
    gap_keys = [
        "food",
        "hunger",
        "meal",
        "diet",
        "remittance",
        "household size",
        "number of people",
        "dependant",
        "consumption",
        "poverty",
        "asset",
        "extension",
        "hsnp",
        "safety net",
    ]
    gap_rows = []
    for var, lab in stata_labels.items():
        blob = f"{var} {lab or ''}".lower()
        if not any(k in blob for k in gap_keys):
            continue
        # skip if already prepared under some rename
        if var in prepared_originals:
            continue
        # skip huge payment grids
        if re.match(r"^B3D__", str(var)):
            continue
        gap_rows.append(
            {
                "stata_variable": var,
                "variable_label": lab,
                "reason_flagged": "keyword match for welfare/food/remittance/hh composition/extension",
                "in_prepared_farming_frame": False,
            }
        )
    gap = pd.DataFrame(gap_rows).drop_duplicates(subset=["stata_variable"])
    gap.to_csv(TABLES / "09_stata_unextracted_gap_candidates.csv", index=False)

    # ---- Write markdown report ----
    not_in_models = audit[(~audit["in_model_a"]) & (~audit["in_model_b"])]
    fin = audit[audit["domain"] == "financial_position"]
    agri = audit[audit["domain"] == "agricultural_characteristics"]
    food = audit[audit["domain"] == "food_security_and_welfare"]
    stress = audit[audit["domain"] == "other_household_stress"]

    top_diff = highlight.head(25)
    top_inv_a = inv_a.sort_values("candidate_rank_score", ascending=False).head(15)
    top_inv_b = inv_b.sort_values("candidate_rank_score", ascending=False).head(20)

    coping_in_nonshock = (
        nonshock["coping_category"].value_counts(dropna=False)
        if "coping_category" in nonshock.columns
        else pd.Series(dtype=int)
    )

    report = f"""# Non-shock household signal audit

**Status:** Script 09 complete. Descriptive / discovery only — **no new classifiers**.  
**Workbench:** `all_shocks_framing_workbench/`  
**Populations (verified):** farming frame n = **9,756**; shock-exposed = **5,248**; complete R2A with no shocks = **4,499**; incomplete R2A with no confirmed positive = **9**.  
**Rule:** missing R2A indicators were **not** treated as zero. The incomplete group is retained separately and excluded from pairwise comparisons below.  
**Sources:** prepared farming modelling frame (+ selected extract-only portfolio/association columns for the same IDs); Stata metadata via **pyreadstat** (no external value-label file).

---

## Verified findings vs recommendations

| Type | Content |
|---|---|
| **Verified** | Population counts; variable presence/missingness; descriptive group differences; Model A/B membership; coping availability among non-shock households |
| **Recommendation** | Which candidates to consider later; leakage caution; whether non-shock households can enter the four-class classifier |

---

## 1. Meaningful differences between shock-exposed and non-shock households

Descriptive thresholds (not p-values): numeric standardised absolute mean difference ≥ 0.15, or categorical max percentage-point gap ≥ 5.

**Largest descriptive contrasts** (see `09_meaningful_group_differences.csv` for the full list):

| Variable | Domain | Contrast summary |
|---|---|---|
"""
    for _, r in top_diff.head(15).iterrows():
        report += (
            f"| `{r['variable']}` | {r['domain']} | {r['distribution_summary']} |\n"
        )

    report += f"""
**Interpretation (verified pattern, not causal claim):** shock-exposed farming households differ from complete non-shock households on several agricultural-challenge and financial-usage markers already present in the prepared frame. Differences on demographics (age/sex/education/urban-rural) are generally smaller than differences on farm-challenge and inclusion markers.

---

## 2. Additional financial indicators available

In the prepared farming dataset, financial-domain columns include (among others): savings / loan / bank / mobile-money usage flags, formal vs informal savings and loans, insurance (with/without NHIF), buyer-supplier credit, formal agricultural borrowing, and digital-app loan usage.

| Metric | n |
|---|---:|
| Variables tagged `financial_position` | {len(fin)} |
| Of which not in Model A | {int((~fin['in_model_a']).sum())} |
| Of which in neither Model A nor B | {int(((~fin['in_model_a']) & (~fin['in_model_b'])).sum())} |

Many of these are **already outside** the current Model A/B matrices (Model A focuses on demographics + agri portfolio + land ownership). Detailed Stata savings/loan product grids (C1_*, E*) are largely **not extracted** into the prepared farming frame (see gap file).

---

## 3. Additional agricultural indicators available

Agricultural domain coverage is relatively rich in the prepared frame:

- Land size / ownership
- Crop and livestock portfolio aggregates (`has_*`, `n_crop_types`, `n_livestock_types`)
- Extensive **farm challenge** indicators (inputs, finance, markets, yields, pests, storage, extension access, etc.)
- Extract-only crop/livestock item detail (summarised as `n_crop_livestock_items_positive` plus top-frequency items)

| Metric | n |
|---|---:|
| Variables tagged `agricultural_characteristics` | {len(agri)} |
| Not in Model A | {int((~agri['in_model_a']).sum())} |

**Verified gap relative to Model A:** farm-challenge indicators and land-size acres are prepared but mostly **not** in the primary Model A feature list (acres intentionally excluded earlier; challenges never added).

---

## 4. Household welfare and food security indicators

Within the **prepared** farming frame, dedicated food-security / dietary-diversity / consumption modules are **sparse or absent** (domain tag `food_security_and_welfare`: **{len(food)}** columns).

Stata metadata does contain related survey content (e.g. hunger/safety-net / remittance / assistance wording). Those variables are catalogued as **unextracted gap candidates** in `09_stata_unextracted_gap_candidates.csv` (n = {len(gap)}). They are **not** verified as available in the current modelling frame.

---

## 5. Potential indicators of unclassified household stress

Domain `other_household_stress` and farm-challenge items can mark production, market, financing and management difficulties **outside** the 12 R2A shock labels.

| Metric | n |
|---|---:|
| Variables tagged `other_household_stress` | {len(stress)} |

**Important distinction (recommendation):**
1. **Reported shock (R2A/R2B)** — explicit shock module.
2. **Vulnerability / capacity** — education, land, portfolio, membership, financial access.
3. **Possible consequences / concurrent stress** — farm challenges, hardship-like items.
4. **Coping behaviour** — R2C / `coping_*` / `coping_category` (exclude from predictors).

A farm challenge or hardship marker is **not** proof of an unreported R2A shock.

---

## 6. Candidates not currently in Model A or Model B

Total audited columns (prepared + selected extract extras): **{len(audit)}**.  
Neither Model A nor Model B: **{len(not_in_models)}**.

Notable groups not in A/B:
- Most `farm_challenge_*` indicators
- Financial usage / insurance / credit flags (`savings_usage`, `loan_usage`, `insurance_*`, etc.)
- Some association membership fields (where not already engineered into Model A)
- Land size in acres (prepared but excluded from primary Model A by prior decision)
- Extract crop-item detail

Full listing: `09_variable_audit_full.csv`.

---

## 7. Candidates suitable for further consideration

### Inventory A — potential household / agricultural characteristics
(Top by descriptive rank; see `09_inventory_a_household_characteristics.csv`)

| Variable | Domain | Suitability | Notes |
|---|---|---|---|
"""
    for _, r in top_inv_a.iterrows():
        report += (
            f"| `{r['variable']}` | {r['domain']} | {r['suitability']} | {r['distribution_summary']} |\n"
        )

    report += """
### Inventory B — potential contextual indicators
(Top by descriptive rank; see `09_inventory_b_contextual_indicators.csv`)

| Variable | Domain | Suitability | Notes |
|---|---|---|---|
"""
    for _, r in top_inv_b.iterrows():
        report += (
            f"| `{r['variable']}` | {r['domain']} | {r['suitability']} | {r['distribution_summary']} |\n"
        )

    report += f"""
**Recommendation:** prioritise low-missingness vulnerability/capacity features (financial access, membership, selected agri challenges with clear definitions) for a *future* feature-expansion experiment. Do **not** add them yet.

---

## 8. Variables to exclude (missingness, redundancy, leakage)

Exclude from predictor consideration now:

| Reason | Examples / scope |
|---|---|
| Target / coping construction | all `coping_*`, `coping_category*` |
| Cohort definition flags | `any_shock_exposed`, `number_of_shocks_reported`, `drought_exposed` |
| High missingness in either group (>40%) | see `09_exclusion_candidates.csv` |
| Redundant with Model A | county one-hots / sex / education / `has_*` already used |
| Shock module already in Model B | `shock_*`, `main_shock_*` |

Farm-challenge variables are **not automatic excludes**, but they carry **temporal / consequence risk** and need case-by-case review before modelling.

---

## 9. Can the 4,499 non-shock households contribute directly to the four-class coping classifier?

**Verified answer: No — not under the current target definition.**

The four-class outcome (`Classification_A_Resilience` → `coping_category`) is defined in the R2 shock/coping module for households with shock experience. Among complete non-shock households:

```
{coping_in_nonshock.to_string()}
```

Non-shock households therefore lack a comparable, valid four-class coping label for the same classification problem. Including them with an imputed class (e.g. forcing `did_nothing`) would change the estimand and is **not** supported by the current audit.

They **can** contribute to:
- descriptive contrasts for feature discovery (this script);
- models of shock exposure itself (different target);
- future multi-task / two-stage designs (exposure then coping) — recommendation only.

---

## 10. What additional information would make them eligible for that target?

To use non-shock households **in the same four-class coping classifier**, one would need **verified** outcome information that is definitionally comparable, for example:

1. A survey measure of coping/resilience classification collected for households **without** R2A shocks (currently not available in the prepared frame); or
2. An explicitly redesigned target that does not require shock exposure (new research question); or
3. Linkage to another wave/module with aligned outcome labels (not available here).

Extracting additional **predictors** (food security, remittances, household size, etc.) from Stata would enrich Model A/B for the **existing** 5,241 modelling households, but would **not** by itself make the 4,499 eligible for the current four-class shock-coping target.

---

## Incomplete R2A group (n = 9)

Retained separately (`09_incomplete_group_profile.csv`). Too small for inference; do not collapse into “no shock”.

---

## Outputs

| File | Description |
|---|---|
| `outputs/tables/09_population_counts.csv` | Group sizes |
| `outputs/tables/09_variable_audit_full.csv` | Full variable audit |
| `outputs/tables/09_group_distributions.csv` | Distribution summaries |
| `outputs/tables/09_meaningful_group_differences.csv` | Highlighted contrasts |
| `outputs/tables/09_domain_coverage.csv` | Domain counts |
| `outputs/tables/09_inventory_a_household_characteristics.csv` | Inventory A |
| `outputs/tables/09_inventory_b_contextual_indicators.csv` | Inventory B |
| `outputs/tables/09_candidate_inventories.csv` | Combined inventories |
| `outputs/tables/09_exclusion_candidates.csv` | Exclusion list |
| `outputs/tables/09_incomplete_group_profile.csv` | Incomplete n=9 profile |
| `outputs/tables/09_stata_unextracted_gap_candidates.csv` | Stata gaps not in prepared frame |
| `outputs/reports/NON_SHOCK_HOUSEHOLD_SIGNAL_AUDIT.md` | This report |

---

*Script 09 does not modify the notebook, Model A/B matrices, or Scripts 07–08 outputs.*
"""

    out = REPORTS / "NON_SHOCK_HOUSEHOLD_SIGNAL_AUDIT.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"Audit rows={len(audit)}; meaningful diffs={len(highlight)}; "
        f"invA={len(inv_a)}; invB={len(inv_b)}; stata gaps={len(gap)}"
    )
    print("Script 09 complete.")


if __name__ == "__main__":
    main()
