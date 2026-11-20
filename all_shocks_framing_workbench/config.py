"""Paths for the all-shocks framing audit workbench (read-only inputs)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKBENCH_ROOT = Path(__file__).resolve().parent

FINACCESS_ROOT = Path.home() / "Documents" / "items" / "fsd" / "finaccess"
MODELLING_FRAME = (
    FINACCESS_ROOT / "data" / "processed" / "finaccess_2024_modelling_frame.parquet"
)
STATA_2024 = FINACCESS_ROOT / "2024" / "Data 18+.dta"
R2_RENAME_MAP = FINACCESS_ROOT / "data" / "processed" / "r2_column_rename_map.csv"
COLUMN_RENAME_MAP = (
    FINACCESS_ROOT / "data" / "processed" / "finaccess_2024_column_rename_map.csv"
)

# Load drought workbench config under an explicit module name (avoid circular import)
_WB_CONFIG = PROJECT_ROOT / "finaccess_coping_workbench" / "config.py"
_spec = importlib.util.spec_from_file_location("finaccess_coping_workbench_config", _WB_CONFIG)
_wb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_wb)

ADDITIONAL_SHOCK_VARS = _wb.ADDITIONAL_SHOCK_VARS
AGRICULTURAL_HAS_VARS = _wb.AGRICULTURAL_HAS_VARS
COPING_ACTION_VARS = _wb.COPING_ACTION_VARS
DEMOGRAPHIC_VARS = _wb.DEMOGRAPHIC_VARS
LAND_OWNERSHIP_VARS = _wb.LAND_OWNERSHIP_VARS
LAND_SIZE_STANDARDISED = _wb.LAND_SIZE_STANDARDISED
LAND_SIZE_TO_ACRES = _wb.LAND_SIZE_TO_ACRES
MAIN_SHOCK_TO_INDICATOR = _wb.MAIN_SHOCK_TO_INDICATOR
PORTFOLIO_COUNT_VARS = _wb.PORTFOLIO_COUNT_VARS
SHOCK_INDICATOR_VARS = _wb.SHOCK_INDICATOR_VARS
SEX_CODE_TO_BIN = _wb.SEX_CODE_TO_BIN
URBAN_RURAL_CODE_TO_BIN = _wb.URBAN_RURAL_CODE_TO_BIN
WEIGHT_VARS = _wb.WEIGHT_VARS
add_land_size_acres = _wb.add_land_size_acres

OUTPUTS = WORKBENCH_ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
PREPARED = OUTPUTS / "prepared"
REPORTS = OUTPUTS / "reports"

CLASS_ORDER = ["personal", "informal", "formal", "did_nothing"]
# Classification_A_Resilience codes from Stata metadata
CODE_TO_CLASS = {1: "formal", 2: "informal", 3: "personal", 4: "did_nothing"}


def ensure_dirs() -> None:
    for p in (TABLES, FIGURES, PREPARED, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def load_farming_frame() -> pd.DataFrame:
    if not MODELLING_FRAME.exists():
        raise FileNotFoundError(MODELLING_FRAME)
    return pd.read_parquet(MODELLING_FRAME)
