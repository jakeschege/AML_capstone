# FinAccess Coping Workbench

Scripts-first Data Preparation investigations for the FinAccess 2024 drought-exposed household analysis.

All workbench files live inside the open `capstone_project` folder. The external FinAccess store is read only for loading prepared data.

## Layout

```text
finaccess_coping_workbench/
├── config.py
├── scripts/
├── outputs/
│   ├── tables/
│   ├── reports/
│   └── figures/
└── logs/
```

## Data source

Scripts load:

`~/Documents/items/fsd/finaccess/data/processed/finaccess_2024_modelling_frame.parquet`

The analytical sample is the drought-exposed subset (`drought_exposed == 1`, n = 1,552).

## Current scripts

- `scripts/02_shannon_portfolio_investigation.py` — relationship between crop counts, livestock counts, and Shannon diversity
- `scripts/03_demographic_investigation.py` — age, sex, education, urban/rural and county inspection
- `scripts/04_shock_exposure_investigation.py` — shock indicators and main_shock_type
- `scripts/05_financial_access_investigation.py` — financial access, usage and credit variables
- `scripts/06_predictor_leakage_audit.py` — leakage, redundancy and stage-suitability audit
- `scripts/07_temporal_availability_review.py` — questionnaire evidence for temporally flagged predictors

## How to run

From `capstone_project`:

```bash
.venv/bin/python finaccess_coping_workbench/scripts/02_shannon_portfolio_investigation.py
```

## Principles

1. One responsibility per script
2. Load prepared data from disk rather than notebook session state
3. Preserve original variables
4. Save findings as CSV tables and Markdown reports
5. Report evidence and recommend decisions rather than silently changing methodology
6. Do not modify the main notebook until script outputs have been reviewed
