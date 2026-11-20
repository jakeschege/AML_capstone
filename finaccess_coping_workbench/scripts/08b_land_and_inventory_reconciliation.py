#!/usr/bin/env python3
"""Reconcile land-size edge cases and post-decision inventory alignment.

Confirms the single missing land_size_unit case, distinguishes community land
from special land-size codes, and verifies default inventories exclude
land_size_unit and shannon_diversity.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    EXCLUDED_PREDICTORS,
    EXTENDED_CLASSIFICATION_CANDIDATES,
    ID_VARS,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    LAND_SIZE_TO_ACRES,
    PRIMARY_CLASSIFICATION_CANDIDATES,
    REPORTS_DIR,
    SEGMENTATION_CANDIDATES,
    SENSITIVITY_CANDIDATES,
    TABLES_DIR,
    add_land_size_acres,
    ensure_output_dirs,
    load_analysis_2024,
)

EXPECTED_COMMUNITY = 182
EXPECTED_SPECIAL = 16
SCRIPT08_SEG_EXTRA = {"land_size_unit", "shannon_diversity"}


def classify_missing_reason(unit: float | None, size: float | None) -> str:
    if pd.notna(unit) and int(unit) == 5:
        return "Community land"
    if pd.notna(size) and int(size) in {98, 99}:
        return "Special response code 98/99"
    if pd.isna(unit):
        return "Missing measurement unit"
    return "Other or unresolved"


def missing_unit_case(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.to_numeric(df["land_size_unit"], errors="coerce").isna()
    case = df.loc[mask].copy()
    cols = (
        [c for c in ID_VARS if c in case.columns]
        + ["land_size", "land_size_unit", LAND_SIZE_STANDARDISED]
        + [c for c in LAND_OWNERSHIP_VARS if c in case.columns]
    )
    out = case[cols].copy()
    out["land_size_acres_missing"] = out[LAND_SIZE_STANDARDISED].isna()
    out["exclusion_of_land_size_unit_still_valid"] = True
    out["note"] = (
        "Unit missing; acres cannot be derived. Predictor exclusion unchanged; "
        "do not impute land_size_unit."
    )
    return out


def land_missing_crosstab(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = df.loc[df[LAND_SIZE_STANDARDISED].isna()].copy()
    unit = pd.to_numeric(missing["land_size_unit"], errors="coerce")
    size = pd.to_numeric(missing["land_size"], errors="coerce")

    community = unit.eq(5)
    special = size.isin([98, 99])
    overlap = community & special

    rows = []
    for idx in missing.index:
        u = unit.loc[idx]
        s = size.loc[idx]
        is_community = bool(pd.notna(u) and int(u) == 5)
        is_special = bool(pd.notna(s) and int(s) in {98, 99})
        rows.append(
            {
                "respondent_id": (
                    missing.loc[idx, "respondent_id"]
                    if "respondent_id" in missing.columns
                    else idx
                ),
                "land_size": s,
                "land_size_unit": u,
                "is_community_land": is_community,
                "is_special_98_99": is_special,
                # Mutually exclusive primary reason: community land takes priority
                "primary_missing_reason": classify_missing_reason(
                    None if pd.isna(u) else float(u),
                    None if pd.isna(s) else float(s),
                ),
            }
        )
    detail = pd.DataFrame(rows)

    summary = (
        detail["primary_missing_reason"]
        .value_counts(dropna=False)
        .rename_axis("primary_missing_reason")
        .reset_index(name="n_households")
    )
    summary["expected_n"] = summary["primary_missing_reason"].map(
        {
            "Community land": EXPECTED_COMMUNITY,
            "Special response code 98/99": EXPECTED_SPECIAL,
            "Missing measurement unit": 0,
            "Other or unresolved": 0,
        }
    )
    summary["matches_script08_expectation"] = summary.apply(
        lambda r: (
            pd.isna(r["expected_n"])
            or int(r["n_households"]) == int(r["expected_n"])
        ),
        axis=1,
    )

    crosstab = pd.DataFrame(
        [
            {
                "group": "community_only_unit5_not_98_99",
                "n_households": int((community & ~special).sum()),
                "note": "Unit 5 without special size codes",
            },
            {
                "group": "community_and_special_overlap",
                "n_households": int(overlap.sum()),
                "note": (
                    "Unit 5 with size 98/99 placeholder; counted as Community land "
                    "in the mutually exclusive primary reason"
                ),
            },
            {
                "group": "special_only_not_community",
                "n_households": int((~community & special).sum()),
                "note": "Size 98/99 with unit not 5; Script 08 special-code count",
            },
            {
                "group": "neither_community_nor_special",
                "n_households": int((~community & ~special).sum()),
                "note": "Should be empty if reasons are fully classified",
            },
            {
                "group": "total_missing_land_size_acres",
                "n_households": int(len(missing)),
                "note": "Expected 198",
            },
            {
                "group": "convertible_unit_codes",
                "n_households": len(LAND_SIZE_TO_ACRES),
                "note": ",".join(str(k) for k in sorted(LAND_SIZE_TO_ACRES)),
            },
        ]
    )
    return summary, detail, crosstab


def inventory_alignment() -> pd.DataFrame:
    script08_seg_path = TABLES_DIR / "08_inventory_segmentation.csv"
    script08_vars = set()
    if script08_seg_path.exists():
        script08_vars = set(pd.read_csv(script08_seg_path)["variable"].tolist())

    current_seg = set(SEGMENTATION_CANDIDATES)
    rows = []

    for var in sorted(script08_vars | current_seg | SCRIPT08_SEG_EXTRA):
        in_08 = var in script08_vars
        in_current = var in current_seg
        status = "retained"
        if in_08 and not in_current:
            if var == "shannon_diversity":
                status = "moved_to_sensitivity"
            elif var == "land_size_unit":
                status = "excluded_predictor"
            else:
                status = "removed"
        elif in_current and not in_08:
            status = "added_after_script08"
        rows.append(
            {
                "variable": var,
                "in_script08_segmentation_inventory": in_08,
                "in_current_SEGMENTATION_CANDIDATES": in_current,
                "status": status,
            }
        )

    checks = pd.DataFrame(
        [
            {
                "check": "segmentation_count",
                "value": len(SEGMENTATION_CANDIDATES),
                "expected": 18,
                "pass": len(SEGMENTATION_CANDIDATES) == 18,
            },
            {
                "check": "primary_count",
                "value": len(PRIMARY_CLASSIFICATION_CANDIDATES),
                "expected": 23,
                "pass": len(PRIMARY_CLASSIFICATION_CANDIDATES) == 23,
            },
            {
                "check": "extended_count",
                "value": len(EXTENDED_CLASSIFICATION_CANDIDATES),
                "expected": 35,
                "pass": len(EXTENDED_CLASSIFICATION_CANDIDATES) == 35,
            },
            {
                "check": "land_size_unit_excluded",
                "value": int("land_size_unit" in EXCLUDED_PREDICTORS),
                "expected": 1,
                "pass": "land_size_unit" in EXCLUDED_PREDICTORS
                and "land_size_unit" not in SEGMENTATION_CANDIDATES,
            },
            {
                "check": "shannon_sensitivity_only",
                "value": int(
                    "shannon_diversity" in SENSITIVITY_CANDIDATES
                    and "shannon_diversity" not in SEGMENTATION_CANDIDATES
                ),
                "expected": 1,
                "pass": "shannon_diversity" in SENSITIVITY_CANDIDATES
                and "shannon_diversity" not in SEGMENTATION_CANDIDATES
                and "shannon_diversity" not in PRIMARY_CLASSIFICATION_CANDIDATES,
            },
            {
                "check": "script08_extras_removed_from_default",
                "value": int(not (SCRIPT08_SEG_EXTRA & current_seg)),
                "expected": 1,
                "pass": not bool(SCRIPT08_SEG_EXTRA & current_seg),
            },
        ]
    )
    return pd.DataFrame(rows), checks


def write_report(
    case: pd.DataFrame,
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    crosstab: pd.DataFrame,
    alignment: pd.DataFrame,
    checks: pd.DataFrame,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_missing = int(detail.shape[0]) if len(detail) else 0
    all_pass = bool(checks["pass"].all())
    overlap_n = int(
        crosstab.loc[
            crosstab["group"] == "community_and_special_overlap", "n_households"
        ].iloc[0]
    )
    report = f"""# Land and Inventory Reconciliation (08b)

Generated: {timestamp}

## 1. Missing land_size_unit observation

Count: {len(case)}

```
{case.to_string(index=False) if len(case) else 'No missing land_size_unit cases.'}
```

Conclusion: exclusion of `land_size_unit` as a predictor remains valid. The unit is not imputed.

## 2. Special codes versus community land

Households with missing `land_size_acres`: {n_missing}

Mutually exclusive primary reasons (community land takes priority, matching Script 08):

```
{summary.to_string(index=False)}
```

Detailed distinction / overlap crosstab:

```
{crosstab.to_string(index=False)}
```

Interpretation: {overlap_n} community-land households also carry size codes 98/99 as placeholders. They remain classified as community land. The Script 08 special-code count of 16 is the non-community remainder (unit not 5). Conversion applies only to unit codes {{{', '.join(str(k) for k in sorted(LAND_SIZE_TO_ACRES))}}}. Unit 5 is not convertible.

## 3. Inventory alignment

Config checks:

```
{checks.to_string(index=False)}
```

Variable-level alignment versus Script 08 segmentation inventory:

```
{alignment.to_string(index=False)}
```

All reconciliation checks passed: {all_pass}

## Output files

- `outputs/tables/08b_missing_land_unit_case.csv`
- `outputs/tables/08b_land_missing_reason_summary.csv`
- `outputs/tables/08b_land_missing_reason_crosstab.csv`
- `outputs/tables/08b_land_missing_reason_detail.csv`
- `outputs/tables/08b_inventory_alignment.csv`
- `outputs/tables/08b_inventory_checks.csv`
"""
    path = REPORTS_DIR / "08b_land_and_inventory_reconciliation.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()
    required = (
        ["land_size", "land_size_unit"]
        + LAND_OWNERSHIP_VARS
        + [c for c in ID_VARS]
    )
    analysis = load_analysis_2024(required_columns=required)
    analysis = add_land_size_acres(analysis)

    case = missing_unit_case(analysis)
    summary, detail, crosstab = land_missing_crosstab(analysis)
    alignment, checks = inventory_alignment()

    reason_counts = {
        r["primary_missing_reason"]: int(r["n_households"])
        for _, r in summary.iterrows()
    }
    if reason_counts.get("Community land", 0) != EXPECTED_COMMUNITY:
        raise AssertionError(
            f"Community land missing count expected {EXPECTED_COMMUNITY}, "
            f"found {reason_counts.get('Community land', 0)}"
        )
    if reason_counts.get("Special response code 98/99", 0) != EXPECTED_SPECIAL:
        raise AssertionError(
            f"Special 98/99 missing count expected {EXPECTED_SPECIAL}, "
            f"found {reason_counts.get('Special response code 98/99', 0)}"
        )
    if int(crosstab.loc[crosstab["group"] == "total_missing_land_size_acres", "n_households"].iloc[0]) != 198:
        raise AssertionError("Expected 198 missing land_size_acres rows.")
    if not checks["pass"].all():
        failed = checks.loc[~checks["pass"], "check"].tolist()
        raise AssertionError(f"Inventory checks failed: {failed}")

    outputs = {
        "08b_missing_land_unit_case.csv": case,
        "08b_land_missing_reason_summary.csv": summary,
        "08b_land_missing_reason_crosstab.csv": crosstab,
        "08b_land_missing_reason_detail.csv": detail,
        "08b_inventory_alignment.csv": alignment,
        "08b_inventory_checks.csv": checks,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    report_path = write_report(case, summary, detail, crosstab, alignment, checks)
    print(f"Wrote {report_path}")
    print("\nMissing unit cases:", len(case))
    print(summary.to_string(index=False))
    print("\nCrosstab:")
    print(crosstab.to_string(index=False))
    print("\nInventory checks:")
    print(checks.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
