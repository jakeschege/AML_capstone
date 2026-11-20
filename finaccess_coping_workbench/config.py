"""Shared paths and data-loading helpers for the FinAccess coping workbench.

All workbench outputs stay inside capstone_project. The external FinAccess
store is used only as a read-only source for the prepared modelling frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKBENCH_ROOT = Path(__file__).resolve().parent

# Read-only external prepared data
FINACCESS_ROOT = Path.home() / "Documents" / "items" / "fsd" / "finaccess"
MODELLING_FRAME = (
    FINACCESS_ROOT / "data" / "processed" / "finaccess_2024_modelling_frame.parquet"
)

OUTPUTS_DIR = WORKBENCH_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREPARED_DIR = OUTPUTS_DIR / "prepared"
LOGS_DIR = WORKBENCH_ROOT / "logs"

# Deterministic binary remaps for feature preparation (original columns retained)
SEX_CODE_TO_BIN = {1: 0, 2: 1}  # Male→0, Female→1
URBAN_RURAL_CODE_TO_BIN = {1: 0, 2: 1}  # Rural→0, Urban→1

EXPECTED_DROUGHT_N = 1552

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

PORTFOLIO_COUNT_VARS = [
    "n_crop_types",
    "n_livestock_types",
]

# Reserved for sensitivity analysis; not in default predictor inventories
SENSITIVITY_CANDIDATES = [
    "shannon_diversity",
]

# Legacy audit set used by Scripts 02 and 06
PORTFOLIO_CORE_VARS = PORTFOLIO_COUNT_VARS + SENSITIVITY_CANDIDATES

DEMOGRAPHIC_VARS = [
    "age_years",
    "sex",
    "education",
    "urban_rural",
    "county",
]

# All binary shock indicators in the modelling frame
SHOCK_INDICATOR_VARS = [
    "shock_child_birth",
    "shock_major_sickness_health_problem_accident_injury",
    "shock_drought_loss",
    "shock_high_rainfall_flooding_mudslide_lightning",
    "shock_death_of_main_income_earner_family_member",
    "shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc",
    "shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut",
    "shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions",
    "shock_livestock_crop_failure_due_to_pest_and_diseases",
    "shock_loss_or_damage_due_tofire_or_violence_including_political_violen",
    "shock_loss_or_damage_due_to_human_wildlife_conflict",
    "shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care",
]

# Project feature grouping excludes the sample-defining drought indicator
ADDITIONAL_SHOCK_VARS = [
    c for c in SHOCK_INDICATOR_VARS if c != "shock_drought_loss"
]

# R2B main_shock_type codes mapped to corresponding shock indicators
MAIN_SHOCK_TO_INDICATOR = {
    1: "shock_child_birth",
    2: "shock_major_sickness_health_problem_accident_injury",
    3: "shock_drought_loss",
    4: "shock_high_rainfall_flooding_mudslide_lightning",
    5: "shock_death_of_main_income_earner_family_member",
    6: "shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc",
    7: "shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut",
    8: "shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions",
    9: "shock_livestock_crop_failure_due_to_pest_and_diseases",
    10: "shock_loss_or_damage_due_tofire_or_violence_including_political_violen",
    11: "shock_loss_or_damage_due_to_human_wildlife_conflict",
    12: "shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care",
}

FINANCIAL_ACCESS_VARS = [
    "buyer_supplier_credit",
    "formal_agri_borrowing",
    "used_formal_credit",
    "used_buyer_credit",
    "used_digital_credit",
    "used_insurance",
    "bank_usage",
    "savings_usage",
    "loan_usage",
    "mobile_money_usage",
    "mfi_usage",
]

LAND_SIZE_VARS = [
    "land_size",
    "land_size_unit",
]

LAND_SIZE_STANDARDISED = "land_size_acres"

# Conversion factors into acres (unit codes 1-4). Unit 5 community land is not converted.
LAND_SIZE_TO_ACRES = {
    1: 1.0,
    2: 2.47105,
    3: 0.000247105,
    4: 1.0 / 43560.0,
}

LAND_OWNERSHIP_VARS = [
    "land_ownership_own_under_your_name",
    "land_ownership_own_jointly_with_spouse",
    "land_ownership_own_jointly_with_someone_else_other_than_spouse",
    "land_ownership_lease_rent_you_dont_own",
    "land_ownership_community_public_owned_e_g_pastoralists",
    "land_ownership_family_owned_e_g_inherited_family_land",
]

# land_size_unit excluded as a predictor after the Script 08 decision.
# shannon_diversity reserved in SENSITIVITY_CANDIDATES.
SEGMENTATION_CANDIDATES = (
    AGRICULTURAL_HAS_VARS
    + PORTFOLIO_COUNT_VARS
    + [LAND_SIZE_STANDARDISED]
    + LAND_OWNERSHIP_VARS
)

PRIMARY_CLASSIFICATION_CANDIDATES = (
    DEMOGRAPHIC_VARS
    + AGRICULTURAL_HAS_VARS
    + PORTFOLIO_COUNT_VARS
    + [LAND_SIZE_STANDARDISED]
    + LAND_OWNERSHIP_VARS
)

EXTENDED_CLASSIFICATION_CANDIDATES = (
    PRIMARY_CLASSIFICATION_CANDIDATES
    + ADDITIONAL_SHOCK_VARS
    + ["main_shock_type"]
)

# Predictors retained in the dataset but excluded from default modelling matrices
EXCLUDED_PREDICTORS = [
    "land_size_unit",
]

TARGET_VARS = [
    "coping_category",
    "coping_category_code",
    "coping_category_collapsed",
    "coping_channel",
]

COPING_ACTION_VARS = [
    "coping_worked_more_got_additional_jobs_farmed_additional_land",
    "coping_managing_techniques_got_assistance_gift_from_friends_family_comm",
    "coping_sale_of_assets_e_g_livestocka_house_car_plot_of_land_shares_or_s",
    "coping_cut_back_on_expenses_adjust_consumption_patterns",
    "coping_loans_credit_borrow",
    "coping_savings",
    "coping_relocated_changed_place_of_residence",
    "coping_claimed_insurance",
    "coping_withdrew_transfered_child_children_from_school",
    "coping_received_assistance_from_government_and_non_governmental_organiz",
    "coping_did_nothing",
    "coping_other_specify",
    "coping_made_a_drainage_system",
    "coping_improved_security_measures",
    "coping_reported_at_police_station",
    "coping_reinvested_business_profits",
]

ID_VARS = ["respondent_id"]
WEIGHT_VARS = ["individual_weight", "household_weight"]
SAMPLE_FLAG_VARS = ["drought_exposed", "shock_drought_loss"]

# Read-only codebook for interpreting coded demographics and shocks
VALUE_LABELS_PATH = FINACCESS_ROOT / "2024" / "value_labels.csv"
VARIABLE_LABELS_PATH = FINACCESS_ROOT / "2024" / "variable_labels.csv"
QUESTIONNAIRE_XLSX = FINACCESS_ROOT / "2024" / "2024 FinAccess Questionnaire _Final.xlsx"
COLUMN_RENAME_MAP = (
    FINACCESS_ROOT / "data" / "processed" / "finaccess_2024_column_rename_map.csv"
)
R2_RENAME_MAP = FINACCESS_ROOT / "data" / "processed" / "r2_column_rename_map.csv"
FEATURE_DICTIONARY = (
    FINACCESS_ROOT / "data" / "processed" / "finaccess_2024_feature_dictionary.csv"
)
LEAKAGE_REGISTER = TABLES_DIR / "06_leakage_variable_register.csv"
TEMPORAL_FLAGS = TABLES_DIR / "06_temporal_flags.csv"


def ensure_output_dirs() -> None:
    """Create workbench output directories if they do not already exist."""
    for path in (TABLES_DIR, REPORTS_DIR, FIGURES_DIR, PREPARED_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_codebook_categories(label_set: str) -> list[tuple[int, str]]:
    """Return ordered (code, text) pairs for a value-label set.

    Uses keep_default_na=False so labels such as education 'None' are preserved.
    """
    labels = pd.read_csv(VALUE_LABELS_PATH, keep_default_na=False)
    subset = labels.loc[labels["label_set"] == label_set, ["code", "text"]].copy()
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


def load_modelling_frame() -> pd.DataFrame:
    """Load the prepared 2024 modelling frame from the external store."""
    if not MODELLING_FRAME.exists():
        raise FileNotFoundError(
            f"Modelling frame not found at read-only path: {MODELLING_FRAME}"
        )
    return pd.read_parquet(MODELLING_FRAME)


def load_analysis_2024(
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return the drought-exposed analytical sample used in Data Preparation.

    Recreates analysis_2024 as modelling_frame[drought_exposed == 1].
    Validates observation count and required columns.
    """
    frame = load_modelling_frame()

    if "drought_exposed" not in frame.columns:
        raise KeyError("Column 'drought_exposed' is missing from the modelling frame.")

    analysis = frame.loc[frame["drought_exposed"] == 1].copy()

    if len(analysis) != EXPECTED_DROUGHT_N:
        raise ValueError(
            "Unexpected drought-exposed sample size: "
            f"expected {EXPECTED_DROUGHT_N}, found {len(analysis)}"
        )

    cols = required_columns or []
    missing = [c for c in cols if c not in analysis.columns]
    if missing:
        raise KeyError(f"Missing required columns in analysis sample: {missing}")

    return analysis


def add_land_size_acres(df: pd.DataFrame) -> pd.DataFrame:
    """Derive land_size_acres in memory without modifying the source store.

    Special codes 98/99 and community land (unit 5) remain missing.
    """
    out = df.copy()
    unit = pd.to_numeric(out["land_size_unit"], errors="coerce")
    size = pd.to_numeric(out["land_size"], errors="coerce")
    size = size.mask(size.isin([98, 99]))
    out["land_size_unit_code"] = unit
    out[LAND_SIZE_STANDARDISED] = size * unit.map(LAND_SIZE_TO_ACRES)
    return out
