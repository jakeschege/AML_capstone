#!/usr/bin/env python3
"""Stage 3a: precompute Overview payload for All shocks + each R2A shock slice."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COHORT = (
    ROOT
    / "all_shocks_framing_workbench"
    / "outputs"
    / "prepared"
    / "all_shocks_modelling_cohort.parquet"
)
ASSIGN = (
    ROOT
    / "all_shocks_framing_workbench"
    / "outputs"
    / "tables"
    / "13_all_shocks_segment_assignments.csv"
)
WB_TABLES = ROOT / "all_shocks_framing_workbench" / "outputs" / "tables"
DROUGHT_TABLES = ROOT / "finaccess_coping_workbench" / "outputs" / "tables"
OUT_DIR = ROOT / "artefacts" / "dashboard" / "v0.2.0"
CLASS_MAP = ROOT / "artefacts" / "classifier" / "v0.2.0" / "class_mapping.json"
SEG_LABELS_DROUGHT = ROOT / "artefacts" / "segmentation" / "v0.1.0" / "cluster_labels.json"

MIN_N = 30
SMALL_SAMPLE_N = 100
CLASS_ORDER = ["personal", "informal", "formal", "did_nothing"]

SHOCK_PLAIN = {
    "shock_child_birth": "Child birth",
    "shock_major_sickness_health_problem_accident_injury": "Major sickness or injury",
    "shock_drought_loss": "Drought loss",
    "shock_high_rainfall_flooding_mudslide_lightning": "High rainfall or flooding",
    "shock_death_of_main_income_earner_family_member": "Death of main income earner",
    "shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc": "Theft or property loss",
    "shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut": "Job or income loss / wage cut",
    "shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions": "Lost home or land",
    "shock_livestock_crop_failure_due_to_pest_and_diseases": "Livestock or crop pest failure",
    "shock_loss_or_damage_due_tofire_or_violence_including_political_violen": "Fire or violence",
    "shock_loss_or_damage_due_to_human_wildlife_conflict": "Human–wildlife conflict",
    "shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care": "Care-related income or expense",
}

MAIN_SHOCK_PLAIN = {
    1: "Child birth",
    2: "Major sickness or injury",
    3: "Drought loss",
    4: "High rainfall or flooding",
    5: "Death of main income earner",
    6: "Theft or property loss",
    7: "Job or income loss / wage cut",
    8: "Lost home or land",
    9: "Livestock or crop pest failure",
    10: "Fire or violence",
    11: "Human–wildlife conflict",
    12: "Care-related income or expense",
}

DECISION_LABELS = {
    1: "Self",
    2: "Spouse or partner",
    3: "Jointly with spouse",
    4: "Jointly with other household member",
    5: "Mother",
    6: "Father",
    7: "Daughter",
    8: "Son",
    9: "Other relative",
    10: "Non-relative",
    11: "No one",
}

CHALLENGE_PLAIN = {
    "farm_challenge_pests_and_diseases": "Pests and diseases",
    "farm_challenge_floods_heavy_rains": "Floods or heavy rains",
    "farm_challenge_high_increased_cost_of_inputs": "High cost of inputs",
    "farm_challenge_difficulty_finding_inputs_seeds_etc": "Hard to find inputs",
    "farm_challenge_low_quality_inputs": "Low quality inputs",
    "farm_challenge_late_delivery_of_inputs": "Late delivery of inputs",
    "farm_challenge_difficulty_starting_financingbecause_of_lack_of_money_capital": "Lack of start-up capital",
    "farm_challenge_difficulty_finding_customers": "Hard to find customers",
    "farm_challenge_small_land_sizes": "Small land sizes",
    "farm_challenge_high_labor_costs": "High labour costs",
    "farm_challenge_unscrupulous_middlepersons_price_distortion": "Unfair middleperson prices",
    "farm_challenge_theft": "Theft",
    "farm_challenge_difficulty_getting_products_to_market_transport_road_infrastruct": "Hard to get produce to market",
    "farm_challenge_invasion_by_wild_animals": "Wild animal invasion",
    "farm_challenge_difficulty_getting_paid_for_products_sold_on_credit": "Hard to get paid on credit",
}

FINANCIAL_FLAGS = [
    "bank_usage",
    "mobile_money_usage",
    "sacco_usage",
    "savings_usage",
    "loan_usage",
    "loan_formal",
    "loan_informal",
    "savings_formal",
    "savings_informal",
    "buyer_supplier_credit",
    "formal_agri_borrowing",
    "insurance_excl_nhif",
    "mobile_bank_usage",
]


def _age_band(age: float) -> str:
    if age < 35:
        return "Under 35"
    if age < 55:
        return "35 to 54"
    return "55 and over"


def _mix_shares(sub: pd.DataFrame) -> dict[str, float]:
    w = float(sub["household_weight"].sum())
    if w <= 0:
        return {c: 0.0 for c in CLASS_ORDER}
    return {
        c: float(sub.loc[sub["coping_category"] == c, "household_weight"].sum()) / w
        for c in CLASS_ORDER
    }


def _none_share(sub: pd.DataFrame) -> float:
    w = float(sub["household_weight"].sum())
    if w <= 0:
        return 0.0
    return float(sub.loc[sub["coping_category"] == "did_nothing", "household_weight"].sum()) / w


def _county_labels() -> dict[int, str]:
    path = DROUGHT_TABLES / "09_onehot_category_dictionary.csv"
    oh = pd.read_csv(path)
    rows = oh.loc[oh["source_variable"] == "county", ["code", "label"]]
    return {int(r.code): str(r.label) for _, r in rows.iterrows()}


def _segment_labels() -> dict[str, str]:
    # Prefer drought Spec A labels as readable names for matching k=5 ids when present;
    # fall back to generic Segment k.
    if SEG_LABELS_DROUGHT.exists():
        return {str(k): str(v) for k, v in json.loads(SEG_LABELS_DROUGHT.read_text()).items()}
    return {}


def _who_panel(df: pd.DataFrame, group_col: str, label_map: dict | None = None) -> list[dict]:
    rows = []
    for key, sub in df.groupby(group_col, dropna=True):
        n = int(len(sub))
        label = label_map.get(key, str(key)) if label_map else str(key)
        rows.append(
            {
                "label": label,
                "none_share": round(_none_share(sub), 4) if n >= MIN_N else None,
                "n": n,
                "weighted": round(float(sub["household_weight"].sum()), 1),
                "suppressed": n < MIN_N,
                "small_sample": n < SMALL_SAMPLE_N,
            }
        )
    rows.sort(key=lambda r: (-(r["n"]), r["label"]))
    return rows


def _slice_payload(
    df: pd.DataFrame,
    *,
    shock_key: str,
    shock_label: str,
    county_lab: dict[int, str],
    seg_lab: dict[str, str],
    mapping: dict,
    full_main_shock_chart: list[dict],
) -> dict:
    total_w = float(df["household_weight"].sum())
    n = int(len(df))
    mix = _mix_shares(df)
    mix_weighted = {c: round(mix[c] * total_w, 1) for c in CLASS_ORDER}
    mix_unweighted = {c: int((df["coping_category"] == c).sum()) for c in CLASS_ORDER}

    county_rows = []
    for code, sub in df.groupby("county"):
        n_c = int(len(sub))
        w_c = float(sub["household_weight"].sum())
        shares = _mix_shares(sub)
        county_rows.append(
            {
                "code": int(code),
                "name": county_lab.get(int(code), f"County {int(code)}"),
                "mix": [round(shares[c], 4) for c in CLASS_ORDER],
                "mix_by_class": {c: round(shares[c], 4) for c in CLASS_ORDER},
                "affected_weighted": round(w_c, 1),
                "n": n_c,
                "suppressed": n_c < MIN_N,
                "small_sample": n_c < SMALL_SAMPLE_N,
            }
        )
    eligible = [r for r in county_rows if not r["suppressed"]]
    eligible.sort(key=lambda r: -r["affected_weighted"])
    top_counties = eligible[:8]

    segments = []
    for cid, sub in df.groupby("cluster_k5"):
        if pd.isna(cid):
            continue
        cid_i = int(cid)
        n_s = int(len(sub))
        w_s = float(sub["household_weight"].sum())
        shares = _mix_shares(sub)
        segments.append(
            {
                "id": cid_i,
                "name": seg_lab.get(str(cid_i), f"Segment {cid_i}"),
                "size_weighted": round(w_s, 1),
                "size_share": round(w_s / total_w, 4) if total_w else 0.0,
                "n": n_s,
                "mix": [round(shares[c], 4) for c in CLASS_ORDER],
                "mix_by_class": {c: round(shares[c], 4) for c in CLASS_ORDER},
                "suppressed": n_s < MIN_N,
                "small_sample": n_s < SMALL_SAMPLE_N,
            }
        )
    segments.sort(key=lambda r: r["id"])

    work = df.copy()
    work["age_band"] = work["age_years"].map(_age_band)
    work["decision_label"] = work["farm_decision_maker"].map(
        lambda x: DECISION_LABELS.get(int(x), "Other") if pd.notna(x) else "Unknown"
    )
    dec_counts = work["decision_label"].value_counts()
    rare = set(dec_counts[dec_counts < MIN_N].index)
    work["decision_grouped"] = work["decision_label"].map(
        lambda x: "Other / small groups" if x in rare else x
    )
    work["farmer_group"] = work["member_any_farmers_association"].map(
        {0: "Not a member", 1: "Member"}
    )
    dig_cols = [c for c in work.columns if c.startswith("digital_agri_")]
    if dig_cols:
        work["digital_any"] = (work[dig_cols].fillna(0).astype(float).max(axis=1) >= 1).map(
            {
                True: "Uses digital agri services",
                False: "Does not use digital agri services",
            }
        )
    who = {
        "Farm decision maker": _who_panel(work, "decision_grouped"),
        "Age of household head": _who_panel(
            work,
            "age_band",
            {"Under 35": "Under 35", "35 to 54": "35 to 54", "55 and over": "55 and over"},
        ),
        "Farmer group membership": _who_panel(work, "farmer_group"),
    }
    if "digital_any" in work.columns:
        who["Digital agricultural services"] = _who_panel(work, "digital_any")

    challenges = []
    for col, label in CHALLENGE_PLAIN.items():
        if col not in work.columns:
            continue
        flag = work[col].fillna(0).astype(float)
        n_pos = int((flag >= 1).sum())
        if n_pos < MIN_N:
            continue
        share = float((flag * work["household_weight"]).sum()) / total_w if total_w else 0.0
        challenges.append(
            {
                "label": label,
                "share": round(share, 4),
                "n": n_pos,
                "weighted": round(float((flag * work["household_weight"]).sum()), 1),
                "suppressed": False,
                "small_sample": n_pos < SMALL_SAMPLE_N,
            }
        )
    challenges.sort(key=lambda r: -r["share"])
    challenges = challenges[:8]

    financial = []
    for col in FINANCIAL_FLAGS:
        if col not in work.columns:
            continue
        flag = work[col].fillna(0).astype(float)
        n_pos = int((flag >= 1).sum())
        share = float((flag * work["household_weight"]).sum()) / total_w if total_w else 0.0
        financial.append(
            {
                "key": col,
                "label": col.replace("_", " "),
                "share": round(share, 4),
                "n": n_pos,
                "weighted": round(float((flag * work["household_weight"]).sum()), 1),
                "suppressed": n_pos < MIN_N,
                "small_sample": n_pos < SMALL_SAMPLE_N,
            }
        )

    none_w = mix_weighted["did_nothing"]
    top_group = None
    top_share = -1.0
    for panel, rows in who.items():
        for r in rows:
            if r["suppressed"] or r["none_share"] is None:
                continue
            if r["none_share"] > top_share:
                top_share = r["none_share"]
                top_group = {"panel": panel, "label": r["label"], "none_share": r["none_share"]}

    suppressed = {
        "counties_total": len(county_rows),
        "counties_suppressed_n_lt_30": sum(1 for r in county_rows if r["suppressed"]),
        "counties_small_sample_n_lt_100": sum(1 for r in county_rows if r["small_sample"]),
        "segments_total": len(segments),
        "segments_suppressed_n_lt_30": sum(1 for r in segments if r["suppressed"]),
        "segments_small_sample_n_lt_100": sum(1 for r in segments if r["small_sample"]),
    }

    return {
        "shock_key": shock_key,
        "shock_label": shock_label,
        "filter_definition": "r2a_any_shock",
        "n_respondents": n,
        "affected_weighted": round(total_w, 1),
        "affected_unweighted": n,
        "small_sample": n < SMALL_SAMPLE_N,
        "mix": {c: round(mix[c], 4) for c in CLASS_ORDER},
        "mix_weighted": mix_weighted,
        "mix_unweighted": mix_unweighted,
        "geography_level": "county",
        "geography_note": (
            "Top counties by survey-weighted households in this shock slice (n ≥ 30 shown)."
        ),
        "counties": top_counties,
        "counties_all": county_rows,
        "segments": segments,
        "who": who,
        "challenges": challenges,
        "financial_access": financial,
        "coping_by_main_shock": full_main_shock_chart,
        "coping_by_main_shock_note": (
            "This chart always uses the household’s main reported shock (R2B), not the "
            "any-shock filter. Coping route is the household’s overall route, not specific "
            "to the selected Overview shock. Main-shock groups with fewer than 30 households "
            "are marked unavailable."
        ),
        "takeaway": {
            "none_weighted": round(none_w, 1),
            "none_share": round(mix["did_nothing"], 4),
            "highest_none_group": top_group,
        },
        "suppressed_cells": suppressed,
        "class_order": CLASS_ORDER,
        "display_labels": mapping.get("display_labels", {}),
        "category_colours": {
            "personal": "#1f8bd1",
            "informal": "#f2a516",
            "formal": "#1a6b3c",
            "did_nothing": "#c0562b",
        },
    }


def _main_shock_chart(df: pd.DataFrame) -> list[dict]:
    """Full R2B main-shock coping mix for all 12 codes (n from full modelling cohort)."""
    rows = []
    for code in range(1, 13):
        sub = df.loc[df["main_shock_type"] == code]
        n = int(len(sub))
        unavailable = n < MIN_N
        mix = _mix_shares(sub) if n else {c: 0.0 for c in CLASS_ORDER}
        rows.append(
            {
                "main_shock_code": code,
                "label": MAIN_SHOCK_PLAIN[code],
                "n": n,
                "weighted": round(float(sub["household_weight"].sum()), 1) if n else 0.0,
                "mix": {c: round(mix[c], 4) for c in CLASS_ORDER},
                "mix_unweighted": {c: int((sub["coping_category"] == c).sum()) for c in CLASS_ORDER},
                "unavailable": unavailable,
                "unavailable_reason": (
                    f"Fewer than {MIN_N} households with this main shock (n={n})."
                    if unavailable
                    else None
                ),
                "small_sample": n < SMALL_SAMPLE_N,
            }
        )
    return rows


def _reconcile(df: pd.DataFrame, slices: dict[str, dict]) -> dict:
    drought = slices["shock_drought_loss"]
    all_slice = slices["all"]
    wb_shock = pd.read_csv(WB_TABLES / "03_shock_frequencies_weighted.csv")
    wb_mix = pd.read_csv(WB_TABLES / "03_coping_distribution_fourclass.csv")
    wb_main = pd.read_csv(WB_TABLES / "03_coping_by_main_shock_r2b.csv")

    drought_w_target = 985365.4
    drought_n_target = 1552
    drought_ok = (
        drought["n_respondents"] == drought_n_target
        and abs(drought["affected_weighted"] - drought_w_target) < 1.0
    )

    # All shocks: each household once
    all_ok = all_slice["n_respondents"] == len(df) == 5241

    # Overlap: sum of R2A ns > all n
    r2a_ns = {
        k: slices[k]["n_respondents"]
        for k in SHOCK_PLAIN
    }
    overlap_ok = sum(r2a_ns.values()) > all_slice["n_respondents"]

    # Weighted shock frequencies: Script 03 table is on the descriptive exposed
    # cohort (n=5,248). Overview uses the modelling cohort (n=5,241; 7 households
    # dropped for missing/invalid coping). Reconcile unweighted n to modelling,
    # and document weighted diffs vs the descriptive table.
    wb_shock_n = pd.read_csv(WB_TABLES / "03_shock_frequencies.csv")
    farm = pd.read_parquet(
        ROOT
        / "all_shocks_framing_workbench"
        / "outputs"
        / "prepared"
        / "all_shocks_farming_cohort.parquet"
    )
    shock_freq_checks = []
    for _, row in wb_shock.iterrows():
        key = str(row["shock"])
        got_w = slices[key]["affected_weighted"]
        exp_w = float(row["weighted_sum"])
        got_n = slices[key]["n_respondents"]
        # modelling unweighted should match df flag counts
        exp_n_model = int((df[key] == 1).sum()) if key in df.columns else None
        exp_n_desc = (
            int(wb_shock_n.loc[wb_shock_n["shock"] == key, "n_households_reporting"].iloc[0])
            if key in set(wb_shock_n["shock"])
            else None
        )
        shock_freq_checks.append(
            {
                "shock": key,
                "payload_n": got_n,
                "modelling_n": exp_n_model,
                "descriptive_n_script03": exp_n_desc,
                "n_ok": got_n == exp_n_model,
                "payload_weighted": got_w,
                "descriptive_weighted_script03": round(exp_w, 1),
                "weighted_abs_diff_vs_descriptive": round(abs(got_w - exp_w), 2),
                "weighted_match_descriptive": abs(got_w - exp_w) < 1.0,
                "note": (
                    "Script 03 weighted frequencies use descriptive exposed n=5,248; "
                    "Overview modelling n=5,241. Drought is identical in both; other "
                    "shocks lose a few households without valid coping labels."
                ),
            }
        )

    # Unweighted mix vs workbench
    mix_checks = []
    for _, row in wb_mix.iterrows():
        cat = str(row["category"])
        got = all_slice["mix_unweighted"][cat]
        exp = int(row["n"])
        mix_checks.append({"class": cat, "payload_n": got, "workbench_n": exp, "ok": got == exp})

    # Main shock group sizes
    main_checks = []
    for code in range(1, 13):
        payload_n = next(
            r["n"] for r in all_slice["coping_by_main_shock"] if r["main_shock_code"] == code
        )
        wb_n = int(
            wb_main.loc[wb_main["main_shock_code"] == float(code), "main_shock_group_n"].iloc[0]
        )
        main_checks.append(
            {
                "main_shock_code": code,
                "payload_n": payload_n,
                "workbench_n": wb_n,
                "ok": payload_n == wb_n,
            }
        )

    strata_cols = [
        c
        for c in df.columns
        if any(x in c.lower() for x in ("strata", "stratum", "psu", "cluster_id", "enumeration"))
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drought_slice": {
            "target_n": drought_n_target,
            "target_weighted": drought_w_target,
            "payload_n": drought["n_respondents"],
            "payload_weighted": drought["affected_weighted"],
            "matches_legacy_overview": drought_ok,
            "note": (
                "Drought R2A slice in the all-shocks modelling cohort reproduces the legacy "
                "Overview totals (n=1,552; weighted≈985,365)."
                if drought_ok
                else "Mismatch — investigate cohort filters."
            ),
        },
        "all_shocks_counted_once": {
            "n": all_slice["n_respondents"],
            "ok": all_ok,
        },
        "r2a_overlap": {
            "sum_of_shock_ns": sum(r2a_ns.values()),
            "all_shocks_n": all_slice["n_respondents"],
            "overlap_expected": overlap_ok,
            "per_shock_n": r2a_ns,
        },
        "shock_weighted_vs_workbench": shock_freq_checks,
        "coping_mix_unweighted_vs_workbench": mix_checks,
        "main_shock_n_vs_workbench": main_checks,
        "suppressed_cells_by_shock": {
            k: slices[k]["suppressed_cells"] for k in ["all", *SHOCK_PLAIN]
        },
        "design_confidence_intervals": {
            "strata_or_cluster_ids_available": bool(strata_cols),
            "columns_found": strata_cols,
            "recommendation": (
                "No survey strata/PSU/cluster identifiers are in the modelling cohort, so "
                "design-based confidence intervals are not computed. Describe weighted shares "
                "as survey-weighted estimates within this FinAccess 2024 analysis sample "
                "(household_weight), not as official national headcounts with sampling-error bands."
            ),
        },
        "all_checks_ok": drought_ok
        and all_ok
        and overlap_ok
        and all(c["n_ok"] for c in shock_freq_checks)
        and all(c["ok"] for c in mix_checks)
        and all(c["ok"] for c in main_checks),
        "weighted_vs_descriptive_note": (
            "Exact weighted match to 03_shock_frequencies_weighted.csv is expected only "
            "where the descriptive and modelling cohorts coincide (e.g. drought). Other "
            "shocks show small weight gaps from the 7 modelling exclusions."
        ),
    }


def main() -> None:
    mapping = json.loads(CLASS_MAP.read_text())
    county_lab = _county_labels()
    seg_lab = _segment_labels()

    df = pd.read_parquet(COHORT)
    assign = pd.read_csv(ASSIGN)[["respondent_id", "cluster_k5"]]
    df = df.merge(assign, on="respondent_id", how="left")
    assert len(df) == 5241

    main_chart = _main_shock_chart(df)
    slices: dict[str, dict] = {}
    slices["all"] = _slice_payload(
        df,
        shock_key="all",
        shock_label="All shocks",
        county_lab=county_lab,
        seg_lab=seg_lab,
        mapping=mapping,
        full_main_shock_chart=main_chart,
    )
    for key, label in SHOCK_PLAIN.items():
        sub = df.loc[df[key] == 1].copy()
        slices[key] = _slice_payload(
            sub,
            shock_key=key,
            shock_label=label,
            county_lab=county_lab,
            seg_lab=seg_lab,
            mapping=mapping,
            full_main_shock_chart=main_chart,
        )

    payload = {
        "version": "v0.2.0",
        "updated": date.today().isoformat(),
        "wave": 2024,
        "source": "precomputed",
        "filter_definition": "r2a_any_shock",
        "filter_note": (
            "The Overview filter uses any reported shock (R2A). A household can appear under "
            "more than one shock. Coping route is the household’s overall route, not specific "
            "to the selected shock. The coping-by-main-shock chart uses main shock (R2B) and "
            "always shows all main-shock groups."
        ),
        "weight_variable": "household_weight",
        "weight_note": (
            "household_weight supports survey-weighted shares within this FinAccess 2024 "
            "analysis sample. No strata/PSU identifiers are available here, so figures are "
            "point estimates represented by the survey — not design-based intervals or an "
            "official national headcount."
        ),
        "min_cell_n": MIN_N,
        "small_sample_n": SMALL_SAMPLE_N,
        "shock_options": [
            {"key": "all", "label": "All shocks", "n": slices["all"]["n_respondents"]},
            *[
                {
                    "key": k,
                    "label": lab,
                    "n": slices[k]["n_respondents"],
                    "disabled": False,
                }
                for k, lab in SHOCK_PLAIN.items()
            ],
        ],
        "methods_note": (
            "All-shocks modelling cohort (n=5,241). A planned 2021 wave comparison is not "
            "included. Financial access charts show patterns across groups, not effects of a shock."
        ),
        "slices": slices,
    }

    recon = _reconcile(df, slices)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "overview_by_shock.json").write_text(json.dumps(payload, indent=2))
    (OUT_DIR / "overview_reconciliation.json").write_text(json.dumps(recon, indent=2))
    print(
        "Wrote",
        OUT_DIR / "overview_by_shock.json",
        "all_checks_ok=",
        recon["all_checks_ok"],
        "drought=",
        recon["drought_slice"],
    )


if __name__ == "__main__":
    main()
