"""Overview dashboard aggregates.

Stage 3a: serves precomputed all-shocks slices from artefacts/dashboard/v0.2.0.
Legacy on-the-fly drought builder is retained only as a fallback if the artefact
is missing.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTED = ROOT / "artefacts" / "dashboard" / "v0.2.0" / "overview_by_shock.json"
PREPARED = ROOT / "finaccess_coping_workbench" / "outputs" / "prepared"
TABLES = ROOT / "finaccess_coping_workbench" / "outputs" / "tables"
SEG = ROOT / "artefacts" / "segmentation" / "v0.1.0"
CLASS_MAP = ROOT / "artefacts" / "classifier" / "v0.1.0" / "class_mapping.json"

MIN_N = 30
CLASS_ORDER = ["personal", "informal", "formal", "did_nothing"]

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


def _class_mapping() -> dict:
    return json.loads(CLASS_MAP.read_text())


def _county_labels() -> dict[int, str]:
    oh = pd.read_csv(TABLES / "09_onehot_category_dictionary.csv")
    rows = oh.loc[oh["source_variable"] == "county", ["code", "label"]]
    return {int(r.code): str(r.label) for _, r in rows.iterrows()}


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
    out = {}
    for c in CLASS_ORDER:
        out[c] = float(sub.loc[sub["coping_category"] == c, "household_weight"].sum()) / w
    return out


def _none_share(sub: pd.DataFrame) -> float:
    w = float(sub["household_weight"].sum())
    if w <= 0:
        return 0.0
    return float(sub.loc[sub["coping_category"] == "did_nothing", "household_weight"].sum()) / w


def _who_panel(df: pd.DataFrame, group_col: str, label_map: dict | None = None) -> list[dict]:
    rows = []
    for key, sub in df.groupby(group_col, dropna=True):
        n = int(len(sub))
        label = label_map.get(key, str(key)) if label_map else str(key)
        item = {
            "label": label,
            "none_share": round(_none_share(sub), 4) if n >= MIN_N else None,
            "n": n,
            "weighted": round(float(sub["household_weight"].sum()), 1),
            "suppressed": n < MIN_N,
        }
        rows.append(item)
    rows.sort(key=lambda r: (-(r["n"]), r["label"]))
    return rows


def _load_precomputed() -> dict[str, Any]:
    if not PRECOMPUTED.exists():
        raise FileNotFoundError(str(PRECOMPUTED))
    return json.loads(PRECOMPUTED.read_text())


def build_dashboard(shock: str | None = None) -> dict:
    """Return Overview payload for shock filter (default: all)."""
    key = (shock or "all").strip() or "all"
    try:
        raw = _load_precomputed()
    except FileNotFoundError:
        if key not in {"all", "shock_drought_loss", "drought"}:
            raise
        # Legacy drought-only builder for local fallback
        legacy = _build_dashboard_legacy_drought()
        legacy["shock_key"] = "shock_drought_loss"
        legacy["shock_label"] = "Drought loss"
        legacy["filter_definition"] = "legacy_drought_only"
        return legacy

    if key == "drought":
        key = "shock_drought_loss"
    slices = raw.get("slices") or {}
    if key not in slices:
        raise KeyError(f"Unknown shock filter: {key}")

    slice_body = dict(slices[key])
    # Highlight selected shock on the always-complete main-shock chart
    selected_main = None
    # Map R2A key → main shock code when names align
    r2a_to_main = {
        "shock_child_birth": 1,
        "shock_major_sickness_health_problem_accident_injury": 2,
        "shock_drought_loss": 3,
        "shock_high_rainfall_flooding_mudslide_lightning": 4,
        "shock_death_of_main_income_earner_family_member": 5,
        "shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc": 6,
        "shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut": 7,
        "shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions": 8,
        "shock_livestock_crop_failure_due_to_pest_and_diseases": 9,
        "shock_loss_or_damage_due_tofire_or_violence_including_political_violen": 10,
        "shock_loss_or_damage_due_to_human_wildlife_conflict": 11,
        "shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care": 12,
    }
    selected_main = r2a_to_main.get(key)
    chart = []
    for row in slice_body.get("coping_by_main_shock") or []:
        item = dict(row)
        item["highlighted"] = selected_main is not None and item.get("main_shock_code") == selected_main
        chart.append(item)
    slice_body["coping_by_main_shock"] = chart
    slice_body["highlighted_main_shock_code"] = selected_main

    return {
        "updated": raw.get("updated"),
        "wave": raw.get("wave", 2024),
        "source": raw.get("source", "precomputed"),
        "version": raw.get("version"),
        "filter_definition": raw.get("filter_definition"),
        "filter_note": raw.get("filter_note"),
        "weight_variable": raw.get("weight_variable"),
        "weight_note": raw.get("weight_note"),
        "min_cell_n": raw.get("min_cell_n", MIN_N),
        "small_sample_n": raw.get("small_sample_n", 100),
        "shock_options": raw.get("shock_options"),
        "methods_note": raw.get("methods_note"),
        "selected_shock": key,
        **slice_body,
        # Drop bulky all-county list from default API response (kept in artefact)
        "counties_all": None,
    }


def _build_dashboard_legacy_drought() -> dict:
    """Original drought Overview builder (n=1,552)."""
    mapping = _class_mapping()
    df = pd.read_parquet(PREPARED / "09_prepared_analysis_frame.parquet")
    assign = pd.read_csv(TABLES / "12_s1_spec_a_assignments_wide.csv")[
        ["respondent_id", "cluster_k5"]
    ]
    df = df.merge(assign, on="respondent_id", how="left")
    labels = json.loads((SEG / "cluster_labels.json").read_text())
    county_lab = _county_labels()

    total_w = float(df["household_weight"].sum())
    n = int(len(df))
    mix = _mix_shares(df)
    mix_weighted = {c: round(mix[c] * total_w, 1) for c in CLASS_ORDER}
    mix_unweighted = {
        c: int((df["coping_category"] == c).sum()) for c in CLASS_ORDER
    }

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
            }
        )

    eligible = [r for r in county_rows if not r["suppressed"]]
    eligible.sort(key=lambda r: -r["affected_weighted"])
    top_counties = eligible[:8]
    geography_level = "county"
    geography_note = "Top counties by survey-weighted drought-exposed households (n ≥ 30)."
    if len(eligible) < 5:
        geography_level = "limited"
        geography_note = (
            "Few counties meet the n ≥ 30 rule; showing all eligible counties only."
        )

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
                "name": labels.get(str(cid_i), f"Segment {cid_i}"),
                "size_weighted": round(w_s, 1),
                "size_share": round(w_s / total_w, 4) if total_w else 0.0,
                "n": n_s,
                "mix": [round(shares[c], 4) for c in CLASS_ORDER],
                "mix_by_class": {c: round(shares[c], 4) for c in CLASS_ORDER},
                "suppressed": n_s < MIN_N,
            }
        )
    segments.sort(key=lambda r: r["id"])

    df = df.copy()
    df["age_band"] = df["age_years"].map(_age_band)
    df["decision_label"] = df["farm_decision_maker"].map(
        lambda x: DECISION_LABELS.get(int(x), "Other") if pd.notna(x) else "Unknown"
    )
    dec_counts = df["decision_label"].value_counts()
    rare = set(dec_counts[dec_counts < MIN_N].index)
    df["decision_grouped"] = df["decision_label"].map(
        lambda x: "Other / small groups" if x in rare else x
    )
    df["farmer_group"] = df["member_any_farmers_association"].map(
        {0: "Not a member", 1: "Member"}
    )
    dig_cols = [c for c in df.columns if c.startswith("digital_agri_")]
    df["digital_any"] = (df[dig_cols].fillna(0).astype(float).max(axis=1) >= 1).map(
        {True: "Uses digital agri services", False: "Does not use digital agri services"}
    )

    who = {
        "Farm decision maker": _who_panel(df, "decision_grouped"),
        "Age of household head": _who_panel(
            df, "age_band", {"Under 35": "Under 35", "35 to 54": "35 to 54", "55 and over": "55 and over"}
        ),
        "Farmer group membership": _who_panel(df, "farmer_group"),
        "Digital agricultural services": _who_panel(df, "digital_any"),
    }

    challenges = []
    for col, label in CHALLENGE_PLAIN.items():
        if col not in df.columns:
            continue
        flag = df[col].fillna(0).astype(float)
        n_pos = int((flag >= 1).sum())
        if n_pos < MIN_N:
            continue
        share = float((flag * df["household_weight"]).sum()) / total_w if total_w else 0.0
        challenges.append(
            {
                "label": label,
                "share": round(share, 4),
                "n": n_pos,
                "weighted": round(float((flag * df["household_weight"]).sum()), 1),
                "suppressed": False,
            }
        )
    challenges.sort(key=lambda r: -r["share"])
    challenges = challenges[:8]

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

    takeaway = {
        "none_weighted": round(none_w, 1),
        "none_share": round(mix["did_nothing"], 4),
        "highest_none_group": top_group,
    }

    return {
        "updated": date.today().isoformat(),
        "wave": 2024,
        "source": "api_legacy_drought",
        "weight_variable": "household_weight",
        "weight_note": (
            "household_weight sums to a survey-weighted total for drought-exposed "
            "households in this FinAccess 2024 analysis sample. It supports "
            "population-oriented shares within the sample. It is not independently "
            "verified here as an official national headcount, so figures are described "
            "as represented by the survey."
        ),
        "n_respondents": n,
        "affected_weighted": round(total_w, 1),
        "affected_unweighted": n,
        "class_order": CLASS_ORDER,
        "display_labels": mapping["display_labels"],
        "category_colours": {
            "personal": "#1f8bd1",
            "informal": "#f2a516",
            "formal": "#1a6b3c",
            "did_nothing": "#c0562b",
        },
        "mix": {c: round(mix[c], 4) for c in CLASS_ORDER},
        "mix_weighted": mix_weighted,
        "mix_unweighted": mix_unweighted,
        "geography_level": geography_level,
        "geography_note": geography_note,
        "counties": top_counties,
        "segments": segments,
        "who": who,
        "challenges": challenges,
        "takeaway": takeaway,
        "methods_note": (
            "A planned comparison with the 2021 FinAccess wave is not included in this release. "
            "Classifier evaluation uses repeated stratified 5×5 CV on the 2024 drought-exposed frame "
            "(macro F1 ≈ 0.35); see Predictions → About this model."
        ),
        "min_cell_n": MIN_N,
    }
