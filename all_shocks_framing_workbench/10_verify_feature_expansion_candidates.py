#!/usr/bin/env python3
"""Script 10 — Verify feature-expansion candidates from Script 09 (no modelling)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyreadstat

from config import (
    COLUMN_RENAME_MAP,
    MAIN_SHOCK_TO_INDICATOR,
    MODELLING_FRAME,
    PREPARED,
    REPORTS,
    SHOCK_INDICATOR_VARS,
    STATA_2024,
    TABLES,
    ensure_dirs,
)

EXTRACT = MODELLING_FRAME.parent / "finaccess_2024_extract.csv"

# Near-zero-variance threshold on modelling cohort (positive rate)
MIN_POSITIVE_RATE = 0.02


def load_stata_labels():
    _, meta = pyreadstat.read_dta(str(STATA_2024), metadataonly=True)
    return meta.column_names_to_labels or {}, meta.variable_value_labels or {}, list(meta.column_names)


def corr_abs(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    if x.notna().sum() < 50 or y.notna().sum() < 50:
        return np.nan
    return float(x.corr(y))


def missing_pct(s: pd.Series) -> float:
    return float(100 * s.isna().mean())


def n_distinct(s: pd.Series) -> int:
    return int(s.nunique(dropna=True))


def positive_rate(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float((x == 1).mean())


def row(
    variable: str,
    label: str,
    domain: str,
    dtype: str,
    miss: float,
    distinct: int,
    in_a: bool,
    in_b: bool,
    redundant: str,
    period: str,
    relation: str,
    status: str,
    recommendation: str,
    notes: str = "",
    group: str = "",
) -> dict:
    return {
        "variable": variable,
        "original_survey_label": label,
        "domain": domain,
        "expansion_group": group,
        "data_type": dtype,
        "pct_missing_modelling_5241": round(miss, 3),
        "n_distinct": distinct,
        "already_in_model_a": in_a,
        "already_in_model_b": in_b,
        "redundant_with": redundant,
        "questionnaire_reference_period": period,
        "relationship_to_coping_target": relation,
        "status": status,
        "recommendation": recommendation,
        "notes": notes,
    }


def main() -> None:
    ensure_dirs()
    print("Loading Script 09 tables and modelling cohort…")
    audit09 = pd.read_csv(TABLES / "09_variable_audit_full.csv")
    inv = pd.read_csv(TABLES / "09_candidate_inventories.csv")
    gaps = pd.read_csv(TABLES / "09_stata_unextracted_gap_candidates.csv")
    excl09 = pd.read_csv(TABLES / "09_exclusion_candidates.csv")

    model = pd.read_parquet(PREPARED / "all_shocks_modelling_cohort.parquet")
    assert len(model) == 5241
    model_a = pd.read_parquet(PREPARED / "model_a_features.parquet")
    model_b = pd.read_parquet(PREPARED / "model_b_features.parquet")
    a_feats = set(model_a.columns) - {"respondent_id", "coping_category"}
    b_feats = set(model_b.columns) - {"respondent_id", "coping_category"}

    labels, value_labels, stata_names = load_stata_labels()
    rename = pd.read_csv(COLUMN_RENAME_MAP)
    rename_map = dict(zip(rename["renamed_column"], rename["original_column"]))
    r2_path = MODELLING_FRAME.parent / "r2_column_rename_map.csv"
    if r2_path.exists():
        r2 = pd.read_csv(r2_path)
        for _, r in r2.iterrows():
            rename_map[str(r["renamed_column"])] = str(r["original_column"])

    # Extract crop/livestock detail for diversity checks
    print("Loading extract crop/livestock detail…")
    ex = pd.read_csv(EXTRACT, low_memory=False)
    ex_m = ex[ex["respondent_id"].isin(model["respondent_id"])].copy()
    crop_cols = [c for c in ex_m.columns if c.startswith("crop_livestock_")]
    crop_num = ex_m[crop_cols].apply(pd.to_numeric, errors="coerce")
    ex_m["n_crop_livestock_items_positive"] = (crop_num == 1).sum(axis=1)
    model = model.merge(
        ex_m[["respondent_id", "n_crop_livestock_items_positive"] + crop_cols],
        on="respondent_id",
        how="left",
    )

    inventory = []

    # -------------------------------------------------------------------------
    # A. Agricultural diversity
    # -------------------------------------------------------------------------
    print("A. Agricultural diversity…")
    shannon_corr = corr_abs(model["shannon_diversity"], model["n_crop_types"] + model["n_livestock_types"])
    nitems_corr = corr_abs(
        model["n_crop_livestock_items_positive"], model["n_crop_types"] + model["n_livestock_types"]
    )
    inventory.append(
        row(
            "shannon_diversity",
            labels.get(rename_map.get("shannon_diversity", ""), "Shannon diversity of crop/livestock portfolio"),
            "agricultural_diversity",
            str(model["shannon_diversity"].dtype),
            missing_pct(model["shannon_diversity"]),
            n_distinct(model["shannon_diversity"]),
            "shannon_diversity" in a_feats,
            "shannon_diversity" in b_feats,
            "n_crop_types + n_livestock_types",
            "Cross-sectional agri portfolio (FinAccess 2024)",
            "Underlying capacity / diversification; not a coping action",
            "Exclude",
            "Redundant with existing Model A portfolio counts "
            f"(corr with n_crop_types+n_livestock_types ≈ {shannon_corr:.2f}).",
            group="agricultural_diversity",
        )
    )
    inventory.append(
        row(
            "n_crop_livestock_items_positive",
            "Count of positive detailed crop/livestock item flags (extract)",
            "agricultural_diversity",
            "int",
            missing_pct(model["n_crop_livestock_items_positive"]),
            n_distinct(model["n_crop_livestock_items_positive"]),
            False,
            False,
            "n_crop_types + n_livestock_types",
            "Cross-sectional agri portfolio",
            "Underlying capacity; not coping",
            "Exclude",
            "Near-duplicate of Model A counts "
            f"(corr ≈ {nitems_corr:.2f}). Adds little beyond existing diversity measures.",
            group="agricultural_diversity",
        )
    )
    # has_any_* duplicates
    for var, twin in [("has_any_cash_crop", "has_cash_crops"), ("has_any_dairy", "has_dairy_livestock")]:
        if var not in model.columns or twin not in model.columns:
            continue
        c = corr_abs(model[var], model[twin])
        inventory.append(
            row(
                var,
                labels.get(rename_map.get(var, ""), var),
                "agricultural_diversity",
                str(model[var].dtype),
                missing_pct(model[var]),
                n_distinct(model[var]),
                twin in a_feats or var in a_feats,
                twin in b_feats or var in b_feats,
                twin,
                "Cross-sectional agri portfolio",
                "Underlying capacity",
                "Exclude",
                f"Effectively identical to Model A `{twin}` (corr ≈ {c:.2f}).",
                group="agricultural_diversity",
            )
        )
    # Detailed crop/livestock items — summarise as one inventory decision
    inventory.append(
        row(
            "crop_livestock_*_detail_block",
            f"Detailed crop/livestock item indicators (n={len(crop_cols)} in extract)",
            "agricultural_diversity",
            "binary_block",
            0.0,
            len(crop_cols),
            False,
            False,
            "has_* portfolio aggregates already in Model A",
            "Cross-sectional agri portfolio",
            "Underlying capacity",
            "Exclude",
            "High dimensionality and substantial overlap with Model A has_*/n_* aggregates. "
            "Not approved for initial expansion.",
            group="agricultural_diversity",
        )
    )

    # -------------------------------------------------------------------------
    # B. Farm challenges
    # -------------------------------------------------------------------------
    print("B. Farm challenges…")
    shock_overlap = {
        "farm_reported_drought_challenge": "shock_drought_loss / R2B drought main-shock",
        "farm_challenge_floods_heavy_rains": "shock_high_rainfall_flooding_* / R2B flood",
        "farm_challenge_pests_and_diseases": "shock_livestock_crop_failure_due_to_pest_*",
        "farm_challenge_theft": "shock_*theft*",
        "farm_challenge_invasion_by_wild_animals": "shock human-wildlife conflict",
    }
    # curated structural challenges with enough prevalence
    approve_challenges = [
        "farm_challenge_high_increased_cost_of_inputs",
        "farm_challenge_difficulty_finding_inputs_seeds_etc",
        "farm_challenge_low_quality_inputs",
        "farm_challenge_late_delivery_of_inputs",
        "farm_challenge_difficulty_starting_financingbecause_of_lack_of_money_capital",
        "farm_challenge_difficulty_in_accessing_financing",
        "farm_challenge_difficulty_finding_customers",
        "farm_challenge_difficulty_getting_products_to_market_transport_road_infrastruct",
        "farm_challenge_unscrupulous_middlepersons_price_distortion",
        "farm_challenge_small_land_sizes",
        "farm_challenge_high_labor_costs",
    ]
    review_challenges = [
        "farm_challenge_lack_of_storage_facilities_for_your_harvest",
        "farm_challenge_difficulty_getting_paid_for_products_sold_on_credit",
        "farm_challenge_low_yields",
        "farm_challenge_limited_skills_and_inaccessible_agricultural_experts_vets_extens",
        "farm_challenge_weeds",
        "farm_challenge_reduced_soil_quality",
        "farm_challenge_delay_by_buyer_to_collect_produce_compromising_quality",
        "farm_challenge_difficult_in_getting_licence_permits",
        "farm_challenge_difficulty_in_management_unavailablity_of_owner",
    ]

    challenge_cols = [
        c
        for c in model.columns
        if c.startswith("farm_challenge_") or c == "farm_reported_drought_challenge"
    ]
    for col in sorted(challenge_cols):
        orig = rename_map.get(col, "")
        lab = labels.get(orig, col)
        miss = missing_pct(model[col])
        dist = n_distinct(model[col])
        pr = positive_rate(model[col])
        in_a = col in a_feats
        in_b = col in b_feats

        if col in shock_overlap:
            status, rec = (
                "Exclude",
                "Likely overlaps R2 shock content (weather/pest/theft/wildlife). "
                f"Related to: {shock_overlap[col]}. Not automatic constraint measures.",
            )
            relation = "Possible concurrent shock consequence / duplicated shock signal"
            period = "Agricultural challenges module; timing vs R2 shocks not guaranteed"
            red = shock_overlap[col]
        elif miss > 40:
            status, rec = "Exclude", f"Missingness {miss:.1f}% exceeds 40% rule."
            relation = "Unclear"
            period = "Agricultural challenges module"
            red = ""
        elif pr is not None and pr < MIN_POSITIVE_RATE:
            status, rec = (
                "Exclude",
                f"Near-zero prevalence on modelling cohort (positive rate={pr:.3f}); unstable signal.",
            )
            relation = "Possible production constraint"
            period = "Agricultural challenges module"
            red = ""
        elif col in approve_challenges:
            status, rec = (
                "Approved candidate",
                "Structural agri constraint (finance/inputs/market/land/labour) with adequate support; "
                "not an R2C coping item. Retain for farm-challenge expansion.",
            )
            relation = "Possible vulnerability / concurrent production constraint (not target-derived)"
            period = "Agricultural challenges module (cross-sectional; may be concurrent with shocks)"
            red = ""
        elif col in review_challenges:
            status, rec = (
                "Requires further review",
                "Thematically relevant but very rare or weakly measured in this cohort; "
                "defer from initial expansion.",
            )
            relation = "Possible constraint / consequence"
            period = "Agricultural challenges module"
            red = ""
        else:
            status, rec = "Requires further review", "Not in curated initial set."
            relation = "Possible constraint"
            period = "Agricultural challenges module"
            red = ""

        inventory.append(
            row(
                col,
                lab,
                "farm_challenge",
                str(model[col].dtype),
                miss,
                dist,
                in_a,
                in_b,
                red,
                period,
                relation,
                status,
                rec,
                notes=f"positive_rate={pr:.4f}" if pr == pr else "",
                group="farm_challenge",
            )
        )

    # -------------------------------------------------------------------------
    # C. Financial characteristics
    # -------------------------------------------------------------------------
    print("C. Financial characteristics…")
    finance_approve = [
        "savings_usage",
        "savings_formal",
        "savings_informal",
        "loan_usage",
        "loan_formal",
        "loan_informal",
        "sacco_usage",
        "bank_usage",
        "mobile_money_usage",
        "mobile_bank_usage",
        "insurance_excl_nhif",
        "buyer_supplier_credit",
        "formal_agri_borrowing",
    ]
    finance_exclude_redundant = {
        "informal_usage": "loan_informal | savings_informal (corr≈1.0)",
        "used_formal_credit": "loan_formal / loan_usage",
        "used_buyer_credit": "buyer_supplier_credit",
        "used_insurance": "insurance_excl_nhif",
        "used_digital_credit": "digital_app_loan",
        "insurance_incl_nhif": "broader than excl_nhif; keep excl_nhif for non-NHIF cover signal",
        "digital_app_loan": "optional digital credit; defer (review) to limit correlated loan flags",
    }
    finance_exclude_leak = [c for c in model.columns if c.startswith("coping_")]
    finance_exclude_missing = [
        "digital_agri_service_purpose_other",
        "digital_agri_service_purpose",
        "uses_digital_agri_service",
    ]

    # candidates from 09 financial domain + prepared frame
    fin_vars = sorted(
        set(
            audit09.loc[audit09["domain"] == "financial_position", "variable"].tolist()
        )
        | set(finance_approve)
        | set(finance_exclude_redundant)
        | set(finance_exclude_missing)
    )
    for col in fin_vars:
        if col not in model.columns:
            # may be extract-only / missing
            if col in finance_exclude_missing or col.startswith("digital_agri"):
                inventory.append(
                    row(
                        col,
                        col,
                        "financial_position",
                        "unknown",
                        99.0,
                        0,
                        False,
                        False,
                        "",
                        "Digital agri module",
                        "Unclear / sparse",
                        "Exclude",
                        "High missingness in Script 09 exclusion list; not usable for initial expansion.",
                        group="financial",
                    )
                )
            continue
        if col.startswith("coping_"):
            continue  # handled as leak below
        if col.startswith("farm_challenge_"):
            continue

        orig = rename_map.get(col, "")
        lab = labels.get(orig, col)
        miss = missing_pct(model[col])
        dist = n_distinct(model[col])

        if col in finance_exclude_leak or col.startswith("coping_"):
            status, rec = "Exclude", "R2C coping action / target construction — leakage."
            relation = "Directly encodes coping behaviour"
            red = "coping_category construction"
            period = "R2C coping module"
        elif miss > 40:
            status, rec = "Exclude", f"Missingness {miss:.1f}% > 40%."
            relation = "Financial access / behaviour"
            red = ""
            period = "Financial inclusion / usage modules"
        elif col in finance_exclude_redundant:
            status, rec = (
                "Exclude",
                f"Redundant with approved financial set: {finance_exclude_redundant[col]}.",
            )
            relation = "Financial access (duplicate coding)"
            red = finance_exclude_redundant[col]
            period = "Financial inclusion / usage modules"
        elif col == "digital_app_loan":
            status, rec = (
                "Requires further review",
                "Relevant digital credit access but correlated with other loan flags; defer.",
            )
            relation = "Financial access (not R2C)"
            red = "loan_usage / loan_formal"
            period = "Financial inclusion / usage modules"
        elif col in finance_approve:
            status, rec = (
                "Approved candidate",
                "General financial access / usage capacity measured outside R2C. "
                "Not an observed shock-coping action item.",
            )
            relation = "Vulnerability / capacity — financial access (not target-derived)"
            red = ""
            period = "Financial inclusion / usage modules (cross-sectional)"
        else:
            status, rec = (
                "Requires further review",
                "Financial-domain candidate not selected for the lean approved set.",
            )
            relation = "Possible financial access"
            red = ""
            period = "Financial inclusion / usage modules"

        inventory.append(
            row(
                col,
                lab,
                "financial_position",
                str(model[col].dtype),
                miss,
                dist,
                col in a_feats,
                col in b_feats,
                red,
                period,
                relation,
                status,
                rec,
                group="financial",
            )
        )

    # Explicit coping financial leaks
    for col in ["coping_loans_credit_borrow", "coping_savings", "coping_claimed_insurance"]:
        if col not in model.columns:
            continue
        inventory.append(
            row(
                col,
                labels.get(rename_map.get(col, ""), col),
                "coping_module_r2c_target",
                str(model[col].dtype),
                missing_pct(model[col]),
                n_distinct(model[col]),
                False,
                False,
                "coping_category",
                "R2C after shock",
                "Direct coping behaviour / target pathway",
                "Exclude",
                "Hard exclude — observed coping response.",
                group="financial",
            )
        )

    # -------------------------------------------------------------------------
    # D. Unextracted Stata welfare / support variables
    # -------------------------------------------------------------------------
    print("D. Stata gap candidates — extract and join…")
    extract_vars = [
        "B1Di",  # eat fewer meals (past month) — better coverage than B1Diii
        "B1Diii",
        "B3A__9",
        "B3H4",
        "B3H6",
        "G7",
        "G12",
        "U24_1",
        "U24_6",
        "mfhi_13",
        "nfhi_11",
        "nfhi_12",
        "mfhi_32",
        "mfhi_32_1",
    ]
    usecols = ["interview__id"] + [v for v in extract_vars if v in stata_names]
    stata_df, _ = pyreadstat.read_dta(str(STATA_2024), usecols=usecols)
    stata_df["interview__id"] = stata_df["interview__id"].astype(str)
    assert stata_df["interview__id"].duplicated().sum() == 0

    join = model[["respondent_id"]].copy()
    join["respondent_id"] = join["respondent_id"].astype(str)
    joined = join.merge(
        stata_df, left_on="respondent_id", right_on="interview__id", how="left", indicator=True
    )
    n_matched = int((joined["_merge"] == "both").sum())
    n_unmatched = int((joined["_merge"] == "left_only").sum())
    join_report = pd.DataFrame(
        [
            {"metric": "modelling_n", "value": len(model)},
            {"metric": "stata_rows_read", "value": len(stata_df)},
            {"metric": "stata_duplicate_ids", "value": int(stata_df["interview__id"].duplicated().sum())},
            {"metric": "one_to_one_matched", "value": n_matched},
            {"metric": "unmatched_modelling_ids", "value": n_unmatched},
            {"metric": "match_rate_pct", "value": round(100 * n_matched / len(model), 3)},
        ]
    )
    join_report.to_csv(TABLES / "10_stata_join_qc.csv", index=False)
    print(join_report.to_string(index=False))

    # Save extracted columns for Script 11 (new file only; do not alter modelling cohort parquet)
    extract_out = joined.drop(columns=["_merge"]).copy()
    extract_out.to_parquet(PREPARED / "10_stata_welfare_extract_modelling.parquet", index=False)
    extract_out.to_csv(PREPARED / "10_stata_welfare_extract_modelling.csv", index=False)

    # Decisions for each gap + key extras
    stata_decisions = {
        "B1Di": (
            "Approved candidate",
            "Food hardship (eat fewer meals, past month). Low missingness after join; "
            "welfare signal not in prepared frame. Mild consequence risk but not R2C.",
        ),
        "B1Diii": (
            "Exclude",
            "Related food item with high missingness on modelling join (>40%). Prefer B1Di.",
        ),
        "B3A__9": (
            "Approved candidate",
            "Remittance / family-friend support income source flag. Capacity/support context.",
        ),
        "B3H4": ("Exclude", "HSNP — missingness ~96% on modelling join."),
        "B3H6": ("Exclude", "Government assistance — missingness ~96% on modelling join."),
        "G7": ("Exclude", "Remittance sent use — missingness ~99%."),
        "G12": ("Exclude", "Remittance received use — missingness ~95%."),
        "U24_1": (
            "Requires further review",
            "Weekly food expenditure — available but noisy continuous spend; defer from initial lean set.",
        ),
        "U24_6": ("Exclude", "Related food spend with higher missingness."),
        "mfhi_13": (
            "Exclude",
            "Near-duplicate of nfhi_11 on matched rows; keep a single food-security flag.",
        ),
        "nfhi_11": (
            "Approved candidate",
            "Food security in 12 months (index flag). Low missingness among matched IDs.",
        ),
        "nfhi_12": (
            "Approved candidate",
            "Managed non-food spending (index flag). Household financial circumstance / welfare.",
        ),
        "mfhi_32": (
            "Approved candidate",
            "Savings for productive assets — financial capacity (not R2C coping item).",
        ),
        "mfhi_32_1": (
            "Requires further review",
            "Closely related to mfhi_32; keep one savings-asset flag for lean expansion.",
        ),
    }

    # Membership from prepared frame → welfare/support
    for col in ["member_any_farmers_association", "member_any_producer_coop", "member_named_implementer_program"]:
        if col not in model.columns:
            continue
        pr = positive_rate(model[col])
        if col == "member_named_implementer_program" or (pr is not None and pr < MIN_POSITIVE_RATE):
            status, rec = (
                "Exclude",
                f"Too rare for stable modelling (positive rate={pr:.4f}).",
            )
        else:
            status, rec = (
                "Approved candidate",
                "Farmer organisation / coop membership — social/institutional support capacity.",
            )
        inventory.append(
            row(
                col,
                labels.get(rename_map.get(col, ""), col),
                "social_and_institutional_support",
                str(model[col].dtype),
                missing_pct(model[col]),
                n_distinct(model[col]),
                col in a_feats,
                col in b_feats,
                "",
                "Association / programme membership module",
                "Support network / capacity (not coping action)",
                status,
                rec,
                group="welfare_support",
            )
        )

    for var in extract_vars:
        if var not in joined.columns:
            inventory.append(
                row(
                    var,
                    labels.get(var, gaps.loc[gaps["stata_variable"] == var, "variable_label"].iloc[0]
                    if (gaps["stata_variable"] == var).any()
                    else var),
                    "welfare_stata_gap",
                    "missing_in_stata_read",
                    100.0,
                    0,
                    False,
                    False,
                    "",
                    "See Stata label",
                    "Unknown",
                    "Exclude",
                    "Variable not available in Stata read.",
                    group="welfare_support",
                )
            )
            continue
        s = joined[var]
        miss = missing_pct(s)
        status, rec = stata_decisions.get(
            var, ("Requires further review", "Listed in Script 09 gap file; not in lean approved set.")
        )
        # override by missingness rule unless already exclude
        if miss > 40 and status == "Approved candidate":
            status, rec = "Exclude", f"Missingness {miss:.1f}% exceeds 40% after join (incl. unmatched IDs)."
        if miss > 40 and status == "Requires further review":
            status, rec = "Exclude", f"Missingness {miss:.1f}% exceeds 40% after join."

        # leakage-like gap rows from 09 (R1Ci, R3C, etc.) — mark exclude if present in gaps as coping-like
        lab = labels.get(var, var)
        if any(k in str(lab).lower() for k in ["cut back on expenses", "sale of assets", "repay the loan"]):
            status, rec = "Exclude", "Likely hardship consequence / coping-like behaviour — leakage risk."

        inventory.append(
            row(
                var,
                lab,
                "welfare_stata_gap",
                str(s.dtype),
                miss,
                n_distinct(s),
                False,
                False,
                "nfhi_11" if var == "mfhi_13" else ("mfhi_32" if var == "mfhi_32_1" else ""),
                "See FinAccess questionnaire / derived index documentation",
                "Welfare / support / food security context; not R2C target codes",
                status,
                rec + f" Join: matched={n_matched}, unmatched={n_unmatched}.",
                notes=json.dumps(
                    {str(k): str(v) for k, v in list((value_labels.get(var) or {}).items())[:12]},
                    ensure_ascii=False,
                ),
                group="welfare_support",
            )
        )

    # Remaining gap file vars not explicitly extracted → exclude / review note
    for _, g in gaps.iterrows():
        var = str(g["stata_variable"])
        if var in extract_vars:
            continue
        lab = str(g["variable_label"])
        blob = lab.lower()
        if any(k in blob for k in ["cut back", "sale of assets", "repay the loan", "lost money/asset as guarantor"]):
            status, rec = "Exclude", "Hardship/coping-like or repayment consequence — leakage risk."
        elif any(k in blob for k in ["digital", "crypto", "bitcoin", "capital markets"]):
            status, rec = "Exclude", "Not central to farming coping classifier; out of scope for this expansion."
        else:
            status, rec = (
                "Requires further review",
                "In Script 09 gap list but not selected for extraction in this controlled pass.",
            )
        inventory.append(
            row(
                var,
                lab,
                "welfare_stata_gap",
                "not_extracted",
                np.nan,
                0,
                False,
                False,
                "",
                "Not extracted in Script 10",
                "Unknown without extraction",
                status,
                rec,
                group="welfare_support",
            )
        )

    inv_df = pd.DataFrame(inventory).drop_duplicates(subset=["variable"], keep="first")
    inv_df.to_csv(TABLES / "10_feature_expansion_inventory.csv", index=False)
    approved = inv_df[inv_df["status"] == "Approved candidate"].copy()
    review = inv_df[inv_df["status"] == "Requires further review"].copy()
    excluded = inv_df[inv_df["status"] == "Exclude"].copy()
    approved.to_csv(TABLES / "10_approved_candidates.csv", index=False)
    review.to_csv(TABLES / "10_review_candidates.csv", index=False)
    excluded.to_csv(TABLES / "10_excluded_candidates.csv", index=False)

    # Group counts for report
    def grp_status(g: str) -> str:
        sub = inv_df[inv_df["expansion_group"] == g]
        return (
            f"approved={int((sub.status=='Approved candidate').sum())}, "
            f"review={int((sub.status=='Requires further review').sum())}, "
            f"exclude={int((sub.status=='Exclude').sum())}"
        )

    report = f"""# Feature expansion eligibility report

**Status:** Script 10 complete (verification only — no classifiers).  
**Population:** existing modelling cohort n = **5,241** (confirmed shock + valid four-class coping).  
**Non-shock households (4,499):** not added (no comparable coping outcome).  
**Stata join QC:** matched **{n_matched}** / 5,241; unmatched **{n_unmatched}**; Stata ID duplicates = **0** (one-to-one among matched).  
**Outputs:** `10_feature_expansion_inventory.csv`, `10_approved_candidates.csv`, `10_review_candidates.csv`, `10_excluded_candidates.csv`, `10_stata_join_qc.csv`, `prepared/10_stata_welfare_extract_modelling.parquet`.

---

## Verified vs recommendation

| Type | Content |
|---|---|
| **Verified** | Presence, missingness on 5,241, redundancy correlations, Stata join match rates, overlap with Model A/B and R2 modules |
| **Recommendation** | Approved / review / exclude status for the controlled expansion in Scripts 11–12 |

---

## A. Agricultural diversity — {grp_status("agricultural_diversity")}

**Decision: no diversity variables approved for initial expansion.**

| Variable | Status | Why |
|---|---|---|
| `shannon_diversity` | Exclude | Corr ≈ {shannon_corr:.2f} with `n_crop_types + n_livestock_types` already in Model A |
| `n_crop_livestock_items_positive` | Exclude | Corr ≈ {nitems_corr:.2f} with the same Model A counts |
| `has_any_cash_crop` / `has_any_dairy` | Exclude | Duplicate of Model A `has_cash_crops` / `has_dairy_livestock` |
| Detailed `crop_livestock_*` | Exclude | High-dimensional overlap with Model A portfolio aggregates |

---

## B. Farm challenges — {grp_status("farm_challenge")}

**Retained:** structural constraints with adequate prevalence (inputs, financing, markets, land, labour).  
**Excluded:** drought / flood / pests / theft / wildlife challenges (shock-module overlap) and near-zero-prevalence items.

Approved examples include input-cost and financing constraints, market-access difficulties, small land sizes and high labour costs.  
These are **not** treated as proven pre-shock traits; they are concurrent agricultural-module flags used cautiously as context.

---

## C. Financial characteristics — {grp_status("financial")}

**Retained:** general access/usage flags outside R2C — savings, loans (formal/informal), SACCO, bank, mobile money, non-NHIF insurance, buyer-supplier credit, formal agri borrowing.  

**Excluded:**
- All `coping_*` financial actions (target pathway / leakage).
- Redundant composites (`informal_usage`, `used_*` mirrors).
- Ultra-sparse digital-agri service fields.

These approved flags are interpreted as **financial capacity / access**, not as observed shock-coping decisions.

---

## D. Previously unextracted welfare / support — {grp_status("welfare_support")}

Extracted from `Data 18+.dta` via pyreadstat and joined on `respondent_id` = `interview__id`.

| Variable | Status | Notes |
|---|---|---|
| `B1Di` | Approved | Ate fewer meals (past month) |
| `B3A__9` | Approved | Remittance / family support source |
| `nfhi_11` | Approved | Food security (12 months) |
| `nfhi_12` | Approved | Managed non-food spending |
| `mfhi_32` | Approved | Savings for productive assets |
| `member_any_farmers_association` / `member_any_producer_coop` | Approved | Prepared-frame membership |
| HSNP / gov assistance / remittance-use details | Exclude | >40% missing after join |
| Asset-sale / cut-back / guarantor-loss items | Exclude | Coping-like / consequence leakage |

Unmatched modelling IDs ({n_unmatched}) remain missing on Stata-only fields and will be imputed inside CV pipelines — not filled as zero outside the pipeline.

---

## Approved candidate count

**{len(approved)}** variables approved for Script 11.

```
{approved[["variable", "expansion_group", "pct_missing_modelling_5241"]].to_string(index=False) if len(approved) else "None"}
```

---

## Next step

Script 11 builds expanded matrices from **Model A + R2B** baseline plus these approved groups only.
"""
    (REPORTS / "FEATURE_EXPANSION_ELIGIBILITY_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Approved={len(approved)} review={len(review)} excluded={len(excluded)}")
    print("Script 10 complete.")


if __name__ == "__main__":
    main()
