#!/usr/bin/env python3
"""Predictor leakage audit for FinAccess drought-exposed coping analysis.

Classifies investigated variables by confirmed leakage, possible leakage,
redundancy, constancy and stage-specific suitability.

Does not drop, encode or transform variables. Does not modify the notebook.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    ADDITIONAL_SHOCK_VARS,
    AGRICULTURAL_HAS_VARS,
    COPING_ACTION_VARS,
    DEMOGRAPHIC_VARS,
    FINANCIAL_ACCESS_VARS,
    ID_VARS,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_VARS,
    MAIN_SHOCK_TO_INDICATOR,
    PORTFOLIO_CORE_VARS,
    REPORTS_DIR,
    SAMPLE_FLAG_VARS,
    SHOCK_INDICATOR_VARS,
    TABLES_DIR,
    TARGET_VARS,
    WEIGHT_VARS,
    ensure_output_dirs,
    load_analysis_2024,
)


def build_variable_groups() -> list[tuple[str, str, list[str]]]:
    """Return ordered (group, subgroup, variables) for the audit universe."""
    return [
        ("target", "target", TARGET_VARS),
        ("coping_action", "direct_coping_action", COPING_ACTION_VARS),
        ("agricultural", "portfolio_indicator", AGRICULTURAL_HAS_VARS),
        ("agricultural", "portfolio_summary", PORTFOLIO_CORE_VARS),
        ("land", "land_size", LAND_SIZE_VARS),
        ("land", "land_ownership", LAND_OWNERSHIP_VARS),
        ("demographic", "demographic", DEMOGRAPHIC_VARS),
        ("shock", "shock_indicator", SHOCK_INDICATOR_VARS),
        ("shock", "main_shock", ["main_shock_type", "main_shock_type_label"]),
        ("financial", "financial_access", FINANCIAL_ACCESS_VARS),
        ("id_weight", "identifier", ID_VARS),
        ("id_weight", "survey_weight", WEIGHT_VARS),
        ("sample_flag", "sample_defining", ["drought_exposed"]),
    ]


def direct_leakage_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical checks linking coping actions to the target."""
    rows = []

    # did_nothing action vs category
    if "coping_did_nothing" in df.columns:
        pos = pd.to_numeric(df["coping_did_nothing"], errors="coerce") == 1
        cat_nothing = df["coping_category"] == "did_nothing"
        n_pos = int(pos.sum())
        n_agree = int((pos & cat_nothing).sum())
        n_pos_not_cat = int((pos & ~cat_nothing).sum())
        n_cat_not_pos = int((~pos & cat_nothing).sum())
        rows.append(
            {
                "check": "coping_did_nothing_vs_coping_category_did_nothing",
                "n_action_positive": n_pos,
                "n_target_class": int(cat_nothing.sum()),
                "n_both": n_agree,
                "n_action_without_class": n_pos_not_cat,
                "n_class_without_action": n_cat_not_pos,
                "pct_action_in_class": round(100.0 * n_agree / max(n_pos, 1), 2),
                "evidence_strength": "confirmed",
                "notes": "Action indicator aligns with did_nothing target class",
            }
        )

    # Any positive coping action excluding did_nothing among non-did_nothing classes
    action_cols = [c for c in COPING_ACTION_VARS if c != "coping_did_nothing" and c in df.columns]
    if action_cols:
        any_action = df[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0).eq(1).any(axis=1)
        active = df["coping_category"] != "did_nothing"
        rows.append(
            {
                "check": "any_non_nothing_action_among_active_copers",
                "n_action_positive": int(any_action.sum()),
                "n_target_class": int(active.sum()),
                "n_both": int((any_action & active).sum()),
                "n_action_without_class": int((any_action & ~active).sum()),
                "n_class_without_action": int((~any_action & active).sum()),
                "pct_action_in_class": round(
                    100.0 * int((any_action & active).sum()) / max(int(active.sum()), 1), 2
                ),
                "evidence_strength": "confirmed",
                "notes": (
                    "Coping action indicators belong to the same R2C response module "
                    "used to characterise resilience/coping behaviour"
                ),
            }
        )

    # collapsed target is a deterministic map of primary target
    expected = df["coping_category"].map(
        {
            "formal": "formal_or_informal",
            "informal": "formal_or_informal",
            "personal": "personal",
            "did_nothing": "did_nothing",
        }
    )
    agree = int((expected == df["coping_category_collapsed"]).sum())
    rows.append(
        {
            "check": "coping_category_collapsed_from_coping_category",
            "n_action_positive": len(df),
            "n_target_class": len(df),
            "n_both": agree,
            "n_action_without_class": 0,
            "n_class_without_action": len(df) - agree,
            "pct_action_in_class": round(100.0 * agree / len(df), 2),
            "evidence_strength": "confirmed",
            "notes": "Collapsed target is a deterministic transform of the primary target",
        }
    )

    return pd.DataFrame(rows)


def redundancy_register(df: pd.DataFrame) -> pd.DataFrame:
    """Confirmed duplicates and composites from prior investigations."""
    rows = []

    # Financial duplicates
    buyer_yes = (pd.to_numeric(df["buyer_supplier_credit"], errors="coerce") == 1).astype(int)
    used_buyer = pd.to_numeric(df["used_buyer_credit"], errors="coerce").astype(int)
    agree = int((buyer_yes == used_buyer).sum())
    rows.append(
        {
            "variable": "used_buyer_credit",
            "related_variables": "buyer_supplier_credit",
            "relationship": "exact_recode",
            "pct_agreement": round(100.0 * agree / len(df), 2),
            "evidence_strength": "confirmed",
            "source_investigation": "05_financial_access",
        }
    )

    loan_usage = pd.to_numeric(df["loan_usage"], errors="coerce") == 1
    loan_formal = (
        pd.to_numeric(df["loan_formal"], errors="coerce") == 1
        if "loan_formal" in df.columns
        else pd.Series(False, index=df.index)
    )
    formal_agri = pd.to_numeric(df["formal_agri_borrowing"], errors="coerce") == 1
    recon = (loan_usage | loan_formal | formal_agri).astype(int)
    used_formal = pd.to_numeric(df["used_formal_credit"], errors="coerce").astype(int)
    agree = int((recon == used_formal).sum())
    rows.append(
        {
            "variable": "used_formal_credit",
            "related_variables": "loan_usage;loan_formal;formal_agri_borrowing",
            "relationship": "documented_or_composite",
            "pct_agreement": round(100.0 * agree / len(df), 2),
            "evidence_strength": "confirmed",
            "source_investigation": "05_financial_access",
        }
    )

    if "digital_app_loan" in df.columns:
        recon = (pd.to_numeric(df["digital_app_loan"], errors="coerce") == 1).astype(int)
        agree = int(
            (recon == pd.to_numeric(df["used_digital_credit"], errors="coerce").astype(int)).sum()
        )
        rows.append(
            {
                "variable": "used_digital_credit",
                "related_variables": "digital_app_loan",
                "relationship": "exact_recode",
                "pct_agreement": round(100.0 * agree / len(df), 2),
                "evidence_strength": "confirmed",
                "source_investigation": "05_financial_access",
            }
        )

    if "insurance_excl_nhif" in df.columns:
        recon = (pd.to_numeric(df["insurance_excl_nhif"], errors="coerce") == 1).astype(int)
        agree = int(
            (recon == pd.to_numeric(df["used_insurance"], errors="coerce").astype(int)).sum()
        )
        rows.append(
            {
                "variable": "used_insurance",
                "related_variables": "insurance_excl_nhif",
                "relationship": "exact_recode",
                "pct_agreement": round(100.0 * agree / len(df), 2),
                "evidence_strength": "confirmed",
                "source_investigation": "05_financial_access",
            }
        )

    # Agricultural redundancy
    active = df[AGRICULTURAL_HAS_VARS].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    corr = float(pd.Series(active).corr(df["shannon_diversity"]))
    rows.append(
        {
            "variable": "shannon_diversity",
            "related_variables": "n_crop_types;n_livestock_types;has_*_group_flags",
            "relationship": "richness_summary_correlated_with_active_groups",
            "pct_agreement": round(100.0 * abs(corr), 2),  # store |corr|*100 as summary metric
            "evidence_strength": "confirmed",
            "source_investigation": "02_shannon_portfolio",
        }
    )

    # Shock construct overlap: main shock maps to indicators
    mapped = df["main_shock_type"].map(MAIN_SHOCK_TO_INDICATOR)
    consistent = []
    for idx, indicator in mapped.items():
        if pd.isna(indicator) or indicator not in df.columns:
            consistent.append(False)
        else:
            consistent.append(pd.to_numeric(df.at[idx, indicator], errors="coerce") == 1)
    rows.append(
        {
            "variable": "main_shock_type",
            "related_variables": "mapped_shock_indicator",
            "relationship": "principal_shock_within_indicator_set",
            "pct_agreement": round(100.0 * float(np.mean(consistent)), 2),
            "evidence_strength": "confirmed",
            "source_investigation": "04_shock_exposure",
        }
    )

    # Constant drought exposure
    rows.append(
        {
            "variable": "shock_drought_loss",
            "related_variables": "drought_exposed",
            "relationship": "sample_defining_constant_in_analysis_sample",
            "pct_agreement": 100.0,
            "evidence_strength": "confirmed",
            "source_investigation": "04_shock_exposure",
        }
    )

    return pd.DataFrame(rows)


def construct_overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Finance-coping and shock construct overlap summaries."""
    pairs = [
        ("loan_usage", "coping_loans_credit_borrow"),
        ("used_formal_credit", "coping_loans_credit_borrow"),
        ("used_buyer_credit", "coping_loans_credit_borrow"),
        ("buyer_supplier_credit", "coping_loans_credit_borrow"),
        ("formal_agri_borrowing", "coping_loans_credit_borrow"),
        ("used_digital_credit", "coping_loans_credit_borrow"),
        ("savings_usage", "coping_savings"),
        ("used_insurance", "coping_claimed_insurance"),
    ]
    rows = []
    for fin, coping in pairs:
        if fin not in df.columns or coping not in df.columns:
            continue
        if fin in ("buyer_supplier_credit", "formal_agri_borrowing"):
            fin_pos = pd.to_numeric(df[fin], errors="coerce") == 1
        else:
            fin_pos = pd.to_numeric(df[fin], errors="coerce") == 1
        coping_pos = pd.to_numeric(df[coping], errors="coerce") == 1
        both = int((fin_pos & coping_pos).sum())
        rows.append(
            {
                "pair_type": "finance_vs_coping_action",
                "variable_a": fin,
                "variable_b": coping,
                "n_both_positive": both,
                "pct_both_positive": round(100.0 * both / len(df), 2),
                "evidence_strength": "possible",
                "notes": "Conceptual overlap; not confirmed direct target leakage",
            }
        )

    rows.append(
        {
            "pair_type": "shock_indicator_vs_main_shock",
            "variable_a": "shock_*_indicators",
            "variable_b": "main_shock_type",
            "n_both_positive": len(df),
            "pct_both_positive": 100.0,
            "evidence_strength": "confirmed",
            "notes": (
                "Main shock is the principal reported shock within the indicator set; "
                "related constructs, not automatic exclusion of both"
            ),
        }
    )
    return pd.DataFrame(rows)


def temporal_flags() -> pd.DataFrame:
    """Timing concerns that are unresolved or possible, not confirmed exclusions."""
    rows = []
    for var in SHOCK_INDICATOR_VARS + ["main_shock_type", "main_shock_type_label"]:
        rows.append(
            {
                "variable": var,
                "timing_issue": "same_R2_shock_coping_module_reference_period",
                "evidence_strength": "possible",
                "recommendation_hint": "flag_temporal_verification",
                "notes": (
                    "R2A/R2B shocks and R2C coping responses share the survey shock module. "
                    "Shocks logically precede coping, but measurement is contemporaneous in the questionnaire."
                ),
            }
        )
    for var in FINANCIAL_ACCESS_VARS:
        rows.append(
            {
                "variable": var,
                "timing_issue": "finance_usage_timing_vs_drought_coping_unresolved",
                "evidence_strength": "unresolved",
                "recommendation_hint": "flag_temporal_verification",
                "notes": (
                    "Usage/access measures may reflect current or recent behaviour. "
                    "Pre-shock availability is not established."
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_variable(
    variable: str,
    group: str,
    subgroup: str,
    df: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> dict:
    """Assign primary recommendation and stage roles for one variable."""
    nunique = int(df[variable].nunique(dropna=True)) if variable in df.columns else 0
    missing = int(df[variable].isna().sum()) if variable in df.columns else np.nan

    # Defaults
    rec = "candidate_predictor"
    strength = "confirmed"
    seg = "not_applicable"
    clf = "include_candidate"
    notes = []

    if group == "target":
        rec = "retain_descriptive_only"
        strength = "confirmed"
        seg = "not_applicable"
        clf = "exclude"
        notes.append("Target variable; not a predictor")
    elif group == "coping_action":
        rec = "exclude_direct_target_leakage"
        strength = "confirmed"
        seg = "exclude"
        clf = "exclude"
        notes.append("Direct R2C coping action used to characterise the coping target")
    elif group == "id_weight":
        rec = "retain_descriptive_only"
        strength = "confirmed"
        seg = "not_applicable"
        clf = "exclude"
        notes.append("Identifier or survey weight; not an ordinary predictor")
    elif variable in ("drought_exposed", "shock_drought_loss"):
        rec = "exclude_constant_value"
        strength = "confirmed"
        seg = "exclude"
        clf = "exclude"
        notes.append("Constant in the drought-exposed analytical sample / sample-defining exposure")
    elif variable == "main_shock_type_label":
        rec = "retain_descriptive_only"
        strength = "confirmed"
        seg = "descriptive_only"
        clf = "descriptive_only"
        notes.append("Label companion to main_shock_type")
    elif variable in redundancy["variable"].values:
        rel = redundancy.loc[redundancy["variable"] == variable].iloc[0]
        if variable == "shock_drought_loss":
            rec = "exclude_constant_value"
            clf = "exclude"
            seg = "exclude"
        elif variable == "shannon_diversity":
            rec = "flag_redundancy"
            strength = "confirmed"
            seg = "flag"
            clf = "flag"
            notes.append(
                f"Redundant richness summary vs portfolio measures "
                f"({rel['relationship']}; metric={rel['pct_agreement']})"
            )
        elif variable == "main_shock_type":
            # Related to indicators but still a distinct principal-shock construct;
            # also temporal flag from R2 module.
            rec = "flag_temporal_verification"
            strength = "possible"
            seg = "exclude"
            clf = "flag"
            notes.append(
                "Principal shock within indicator set; share R2 module timing with coping target"
            )
            notes.append(f"Mapped indicator consistency {rel['pct_agreement']}%")
        else:
            # Financial derived duplicates/composites: redundancy + temporal
            rec = "flag_redundancy"
            strength = "confirmed"
            seg = "exclude"
            clf = "flag"
            notes.append(
                f"{rel['relationship']} with {rel['related_variables']} "
                f"(agreement {rel['pct_agreement']}%)"
            )
            notes.append("Also subject to unresolved finance timing vs coping")
    elif group == "financial":
        rec = "flag_temporal_verification"
        strength = "unresolved"
        seg = "exclude"
        clf = "flag"
        notes.append("Provisional classification candidate pending timing/leakage verification")
        notes.append("Excluded from agricultural segmentation inputs by approved preparation decision")
    elif group == "shock":
        rec = "flag_temporal_verification"
        strength = "possible"
        seg = "exclude"
        clf = "flag"
        notes.append("Additional shock indicators share R2 module reference period with coping responses")
        if variable == "shock_drought_loss":
            rec = "exclude_constant_value"
            clf = "exclude"
            strength = "confirmed"
    elif group == "agricultural":
        rec = "candidate_predictor"
        strength = "confirmed"
        seg = "include_candidate"
        clf = "include_candidate"
        notes.append("Agricultural portfolio characteristic for segmentation and classification")
        if variable == "shannon_diversity":
            rec = "flag_redundancy"
            seg = "flag"
            clf = "flag"
    elif group == "land":
        rec = "candidate_predictor"
        strength = "confirmed"
        seg = "include_candidate"
        clf = "include_candidate"
        notes.append("Land size/ownership characteristic for segmentation and classification")
    elif group == "demographic":
        rec = "candidate_predictor"
        strength = "confirmed"
        seg = "not_applicable"
        clf = "include_candidate"
        notes.append(
            "Demographic background for classification; not a core agricultural segmentation input"
        )
    else:
        rec = "candidate_predictor"
        notes.append("Default candidate pending further preparation")

    # Constancy override for any unexpected constants among candidates
    if (
        variable in df.columns
        and nunique <= 1
        and group not in ("target", "coping_action", "id_weight")
        and variable not in ("main_shock_type_label",)
    ):
        rec = "exclude_constant_value"
        strength = "confirmed"
        clf = "exclude"
        if seg == "include_candidate":
            seg = "exclude"
        notes.append("No variation in the analytical sample")

    return {
        "variable": variable,
        "group": group,
        "subgroup": subgroup,
        "present_in_sample": variable in df.columns,
        "n_unique": nunique if variable in df.columns else np.nan,
        "missing": missing,
        "primary_recommendation": rec,
        "evidence_strength": strength,
        "segmentation_role": seg,
        "classification_role": clf,
        "notes": "; ".join(notes),
    }


def build_register(df: pd.DataFrame, redundancy: pd.DataFrame) -> pd.DataFrame:
    """Build the full variable recommendation register."""
    rows = []
    seen = set()
    for group, subgroup, variables in build_variable_groups():
        for var in variables:
            if var in seen:
                continue
            seen.add(var)
            rows.append(classify_variable(var, group, subgroup, df, redundancy))
    return pd.DataFrame(rows)


def recommendation_counts(register: pd.DataFrame) -> pd.DataFrame:
    """Count primary recommendations and stage roles."""
    primary = (
        register["primary_recommendation"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="n_variables")
    )
    primary.insert(0, "summary_type", "primary_recommendation")
    seg = (
        register["segmentation_role"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="n_variables")
    )
    seg.insert(0, "summary_type", "segmentation_role")
    clf = (
        register["classification_role"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="n_variables")
    )
    clf.insert(0, "summary_type", "classification_role")
    return pd.concat([primary, seg, clf], ignore_index=True)


def write_report(
    n: int,
    register: pd.DataFrame,
    direct: pd.DataFrame,
    overlap: pd.DataFrame,
    redundancy: pd.DataFrame,
    temporal: pd.DataFrame,
    counts: pd.DataFrame,
) -> Path:
    """Write the leakage audit Markdown report."""
    confirmed_exclude = register.loc[
        register["primary_recommendation"].isin(
            ["exclude_direct_target_leakage", "exclude_constant_value"]
        )
    ]
    flagged = register.loc[
        register["primary_recommendation"].isin(
            ["flag_temporal_verification", "flag_redundancy"]
        )
    ]
    candidates = register.loc[register["primary_recommendation"] == "candidate_predictor"]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Predictor Leakage Audit Report

Generated: {timestamp}

## What was audited

Variable groups investigated so far for the drought-exposed analytical sample (`n = {n}`):

- targets and coping action indicators
- agricultural portfolio and land characteristics
- demographics
- shock indicators and main shock type
- financial access variables
- identifiers and survey weights

No variables were dropped, encoded or transformed. Recommendations are for review before feature preparation.

## Recommendation counts

```
{counts.to_string(index=False)}
```

## Confirmed exclusions

```
{confirmed_exclude[['variable','group','primary_recommendation','evidence_strength','classification_role','notes']].to_string(index=False)}
```

## Flags requiring review

```
{flagged[['variable','group','primary_recommendation','evidence_strength','segmentation_role','classification_role','notes']].to_string(index=False)}
```

## Candidate predictors

```
{candidates[['variable','group','segmentation_role','classification_role','notes']].to_string(index=False)}
```

## Direct leakage checks

```
{direct.to_string(index=False)}
```

## Construct overlap summary

```
{overlap.to_string(index=False)}
```

## Redundancy register

```
{redundancy.to_string(index=False)}
```

## Temporal flags

The temporal table flags unresolved or possible timing issues. These are not confirmed exclusions.

- Shock indicators and `main_shock_type`: possible contemporaneous measurement with coping responses in the R2 module
- Financial access variables: unresolved timing relative to drought coping

Rows written: {len(temporal)}

## Interpretation

Confirmed leakage is limited to the coping action indicators and the target fields themselves. Constant drought exposure is excluded because it has no variation in this sample.

Financial usage measures and shock indicators remain analytically interesting, but their timing relative to coping behaviour is not established as safe pre-outcome information. Exact financial duplicates and the Shannon richness summary are flagged for redundancy rather than removed automatically.

Agricultural portfolio and land variables remain the core segmentation candidates. Demographics and the non-excluded classification candidates remain provisional pending your review of the flags above.

## Recommended next step

Review the register and decide which flagged variables proceed to feature preparation. Do not treat every conceptual relationship as confirmed leakage.

## Output files

- `outputs/tables/06_leakage_variable_register.csv`
- `outputs/tables/06_direct_leakage_checks.csv`
- `outputs/tables/06_construct_overlap_summary.csv`
- `outputs/tables/06_redundancy_register.csv`
- `outputs/tables/06_temporal_flags.csv`
- `outputs/tables/06_recommendation_counts.csv`
"""
    path = REPORTS_DIR / "06_predictor_leakage_audit.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()

    # Required core columns; optional helpers attached if present
    required = (
        TARGET_VARS
        + COPING_ACTION_VARS
        + AGRICULTURAL_HAS_VARS
        + PORTFOLIO_CORE_VARS
        + LAND_SIZE_VARS
        + LAND_OWNERSHIP_VARS
        + DEMOGRAPHIC_VARS
        + SHOCK_INDICATOR_VARS
        + ["main_shock_type"]
        + FINANCIAL_ACCESS_VARS
        + ID_VARS
        + WEIGHT_VARS
        + ["drought_exposed"]
    )
    analysis = load_analysis_2024(required_columns=required)

    from config import load_modelling_frame

    full = load_modelling_frame()
    drought = full.loc[full["drought_exposed"] == 1].copy()
    for col in ["main_shock_type_label", "loan_formal", "digital_app_loan", "insurance_excl_nhif"]:
        if col in drought.columns and col not in analysis.columns:
            analysis[col] = drought[col].values

    print("=" * 72)
    print("06 Predictor leakage audit")
    print("=" * 72)
    print(f"Observations: {len(analysis)}")

    direct = direct_leakage_checks(analysis)
    redundancy = redundancy_register(analysis)
    overlap = construct_overlap_summary(analysis)
    temporal = temporal_flags()
    register = build_register(analysis, redundancy)
    counts = recommendation_counts(register)

    outputs = {
        "06_leakage_variable_register.csv": register,
        "06_direct_leakage_checks.csv": direct,
        "06_construct_overlap_summary.csv": overlap,
        "06_redundancy_register.csv": redundancy,
        "06_temporal_flags.csv": temporal,
        "06_recommendation_counts.csv": counts,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    report_path = write_report(
        n=len(analysis),
        register=register,
        direct=direct,
        overlap=overlap,
        redundancy=redundancy,
        temporal=temporal,
        counts=counts,
    )
    print(f"Wrote {report_path}")

    print("\nRecommendation counts")
    print(counts.to_string(index=False))
    print("\nPrimary recommendation by group")
    print(
        register.groupby(["group", "primary_recommendation"])
        .size()
        .rename("n")
        .reset_index()
        .to_string(index=False)
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
