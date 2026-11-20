"""Survey/domain fields → model matrix (v0.1 drought 68 feats or v0.2 all-shocks 93)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SEX_CODE_TO_BIN = {1: 0, 2: 1}  # Male→0, Female→1
URBAN_RURAL_CODE_TO_BIN = {1: 0, 2: 1}  # Rural→0, Urban→1

AGRICULTURAL_HAS_VARS = [
    "has_cash_crops",
    "has_cereals",
    "has_dairy_livestock",
    "has_horticulture",
    "has_legumes",
    "has_other_agri",
    "has_other_livestock",
    "has_poultry",
    "has_roots_tubers",
]
PORTFOLIO_COUNT_VARS = ["n_crop_types", "n_livestock_types"]
LAND_OWNERSHIP_VARS = [
    "land_ownership_own_under_your_name",
    "land_ownership_own_jointly_with_spouse",
    "land_ownership_own_jointly_with_someone_else_other_than_spouse",
    "land_ownership_lease_rent_you_dont_own",
    "land_ownership_community_public_owned_e_g_pastoralists",
    "land_ownership_family_owned_e_g_inherited_family_land",
]
FINANCIAL_ACCESS_VARS = [
    "bank_usage",
    "buyer_supplier_credit",
    "formal_agri_borrowing",
    "insurance_excl_nhif",
    "loan_formal",
    "loan_informal",
    "loan_usage",
    "mobile_bank_usage",
    "mobile_money_usage",
    "sacco_usage",
    "savings_formal",
    "savings_informal",
    "savings_usage",
]
MAIN_SHOCK_CODES = list(range(1, 13))
CONTINUOUS_FEATURES = ["age_years", "education", "n_crop_types", "n_livestock_types"]
SPEC_A_FEATURES = (
    list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS) + list(LAND_OWNERSHIP_VARS)
)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    return cleaned.strip("_") or "unknown"


def load_county_categories(value_labels_path: Path) -> list[tuple[int, str]]:
    labels = pd.read_csv(value_labels_path, keep_default_na=False)
    subset = labels.loc[labels["label_set"] == "county", ["code", "text"]].copy()
    rows: list[tuple[int, str]] = []
    for _, row in subset.iterrows():
        try:
            code = int(float(row["code"]))
        except (TypeError, ValueError):
            continue
        text = str(row["text"]).strip() or f"code_{code}"
        rows.append((code, text))
    rows.sort(key=lambda x: x[0])
    return rows


def county_one_hot_columns(categories: list[tuple[int, str]]) -> list[str]:
    return [f"county_{code}_{slugify(label)}" for code, label in categories]


def primary_feature_names(categories: list[tuple[int, str]]) -> list[str]:
    return (
        ["age_years", "sex_bin", "education", "urban_rural_bin"]
        + county_one_hot_columns(categories)
        + list(AGRICULTURAL_HAS_VARS)
        + list(PORTFOLIO_COUNT_VARS)
        + list(LAND_OWNERSHIP_VARS)
    )


def encode_household(
    payload: dict,
    feature_names: list[str],
    county_categories: list[tuple[int, str]],
) -> pd.DataFrame:
    """Build a single-row DataFrame in model feature order (acres excluded)."""
    row: dict[str, float] = {name: 0.0 for name in feature_names}

    age = payload.get("age_years")
    if age is None:
        raise ValueError("age_years is required")
    row["age_years"] = float(age)

    education = payload.get("education")
    if education is None:
        raise ValueError("education is required (1–4)")
    education = int(education)
    if education not in (1, 2, 3, 4):
        raise ValueError("education must be 1–4")
    row["education"] = float(education)

    sex = payload.get("sex")
    if sex is None and payload.get("sex_bin") is not None:
        row["sex_bin"] = float(payload["sex_bin"])
    elif sex is not None:
        sex_i = int(sex)
        if sex_i not in SEX_CODE_TO_BIN:
            raise ValueError("sex must be 1 (Male) or 2 (Female)")
        row["sex_bin"] = float(SEX_CODE_TO_BIN[sex_i])
    else:
        raise ValueError("sex (1/2) or sex_bin is required")

    urban = payload.get("urban_rural")
    if urban is None and payload.get("urban_rural_bin") is not None:
        row["urban_rural_bin"] = float(payload["urban_rural_bin"])
    elif urban is not None:
        urban_i = int(urban)
        if urban_i not in URBAN_RURAL_CODE_TO_BIN:
            raise ValueError("urban_rural must be 1 (Rural) or 2 (Urban)")
        row["urban_rural_bin"] = float(URBAN_RURAL_CODE_TO_BIN[urban_i])
    else:
        raise ValueError("urban_rural (1/2) or urban_rural_bin is required")

    county = payload.get("county")
    if county is None:
        raise ValueError("county code is required")
    county_i = int(county)
    valid_codes = {code for code, _ in county_categories}
    if county_i not in valid_codes:
        raise ValueError(f"unknown county code: {county_i}")
    for code, label in county_categories:
        col = f"county_{code}_{slugify(label)}"
        if col in row:
            row[col] = 1.0 if code == county_i else 0.0

    for col in AGRICULTURAL_HAS_VARS + LAND_OWNERSHIP_VARS:
        if col not in payload:
            raise ValueError(f"{col} is required (0/1)")
        val = int(payload[col])
        if val not in (0, 1):
            raise ValueError(f"{col} must be 0 or 1")
        row[col] = float(val)

    for col in PORTFOLIO_COUNT_VARS:
        if col not in payload:
            raise ValueError(f"{col} is required")
        row[col] = float(payload[col])

    # Optional for v0.1; required when present in the active feature schema (v0.2+)
    needs_main_shock = any(n.startswith("main_shock_") for n in feature_names)
    if needs_main_shock:
        if payload.get("main_shock_type") is not None:
            code = int(payload["main_shock_type"])
            if code not in MAIN_SHOCK_CODES:
                raise ValueError("main_shock_type must be an integer 1–12")
            for k in MAIN_SHOCK_CODES:
                col = f"main_shock_{k}"
                if col in row:
                    row[col] = 1.0 if k == code else 0.0
        else:
            for k in MAIN_SHOCK_CODES:
                col = f"main_shock_{k}"
                if col not in feature_names:
                    continue
                if col not in payload:
                    raise ValueError("main_shock_type (1–12) or main_shock_* one-hots are required")
                val = int(payload[col])
                if val not in (0, 1):
                    raise ValueError(f"{col} must be 0 or 1")
                row[col] = float(val)

    for col in FINANCIAL_ACCESS_VARS:
        if col not in feature_names:
            continue
        if col not in payload:
            raise ValueError(f"{col} is required (0/1) for the financial-expanded model")
        val = int(payload[col])
        if val not in (0, 1):
            raise ValueError(f"{col} must be 0 or 1")
        row[col] = float(val)

    return pd.DataFrame([[row[c] for c in feature_names]], columns=feature_names)


def encode_segmentation(payload: dict) -> pd.DataFrame:
    """Build Spec A 17-feature row (no acres)."""
    row = {}
    for col in SPEC_A_FEATURES:
        if col not in payload:
            raise ValueError(f"{col} is required for segmentation")
        row[col] = float(payload[col])
    return pd.DataFrame([row], columns=SPEC_A_FEATURES)
