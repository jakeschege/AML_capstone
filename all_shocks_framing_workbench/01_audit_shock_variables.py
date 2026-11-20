#!/usr/bin/env python3
"""Script 01 — Audit R2A / R2B / R2C / Classification_A_Resilience using Stata metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyreadstat

from config import (
    CODE_TO_CLASS,
    COPING_ACTION_VARS,
    MAIN_SHOCK_TO_INDICATOR,
    MODELLING_FRAME,
    R2_RENAME_MAP,
    SHOCK_INDICATOR_VARS,
    STATA_2024,
    TABLES,
    ensure_dirs,
    load_farming_frame,
)


def freq_table(series: pd.Series) -> str:
    vc = series.value_counts(dropna=False)
    parts = []
    for k, v in vc.items():
        parts.append(f"{k!r}:{int(v)}")
    return "; ".join(parts)


def main() -> None:
    ensure_dirs()
    rename = pd.read_csv(R2_RENAME_MAP)
    rename_map = dict(zip(rename["original_column"], rename["renamed_column"]))
    inv_rename = {v: k for k, v in rename_map.items()}

    print("Loading Stata metadata (pyreadstat, metadata only)…")
    _, meta = pyreadstat.read_dta(str(STATA_2024), metadataonly=True)
    labels = meta.column_names_to_labels or {}
    value_labels = meta.variable_value_labels or {}

    # Persist value-label mappings extracted from Stata
    vl_rows = []
    for var, mapping in value_labels.items():
        if not str(var).startswith(("R2A", "R2B", "R2C", "Classification_A_Resilience")):
            continue
        for code, text in mapping.items():
            vl_rows.append(
                {
                    "stata_variable": var,
                    "value_code": code,
                    "value_label": text,
                }
            )
    vl_df = pd.DataFrame(vl_rows)
    vl_df.to_csv(TABLES / "stata_value_label_mappings.csv", index=False)
    print(f"Wrote {TABLES / 'stata_value_label_mappings.csv'} ({len(vl_df)} rows)")

    print("Loading farming modelling frame…")
    df = load_farming_frame()
    print(f"Farming frame n={len(df)} cols={df.shape[1]} path={MODELLING_FRAME}")
    assert len(df) > 1552, "Expected full farming sample, not drought-only"

    # --- R2A shock indicators ---
    shock_rows = []
    for renamed in SHOCK_INDICATOR_VARS:
        original = inv_rename.get(renamed, "")
        series = df[renamed] if renamed in df.columns else pd.Series(dtype=float)
        n_miss = int(series.isna().sum())
        distinct = sorted(series.dropna().unique().tolist(), key=lambda x: (str(type(x)), str(x)))
        # Binary check: after dropping NA, only {0,1}?
        non_na = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
        genuinely_binary = non_na.issubset({0.0, 1.0, 0, 1})
        shock_rows.append(
            {
                "original_stata_name": original,
                "renamed_name": renamed,
                "variable_label": labels.get(original, ""),
                "value_labels": json.dumps(value_labels.get(original) or {}, ensure_ascii=False),
                "dtype": str(series.dtype),
                "n_non_missing": int(series.notna().sum()),
                "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / len(df), 4),
                "distinct_values": json.dumps([str(x) for x in distinct]),
                "value_frequencies": freq_table(series),
                "genuinely_binary_ignoring_na": genuinely_binary,
                "n_positive_1": int((pd.to_numeric(series, errors="coerce") == 1).sum()),
                "n_zero_0": int((pd.to_numeric(series, errors="coerce") == 0).sum()),
                "missingness_note": (
                    "Missing is rare (~9 households across shocks). "
                    "Do not treat missing as zero without justification."
                    if n_miss > 0
                    else "No missing in farming frame."
                ),
            }
        )
    shock_inv = pd.DataFrame(shock_rows)
    shock_inv.to_csv(TABLES / "shock_variable_inventory.csv", index=False)
    print(f"Wrote shock_variable_inventory.csv ({len(shock_inv)} indicators)")

    # --- R2B main shock ---
    main = df["main_shock_type"] if "main_shock_type" in df.columns else pd.Series(dtype=float)
    main_label_col = df["main_shock_type_label"] if "main_shock_type_label" in df.columns else None
    r2b_vl = value_labels.get("R2B") or {}
    main_rows = []
    for code, count in main.value_counts(dropna=False).items():
        if pd.isna(code):
            lab = "(missing)"
            mapped_ind = ""
        else:
            code_i = int(code)
            lab = r2b_vl.get(code_i, r2b_vl.get(float(code_i), str(code)))
            mapped_ind = MAIN_SHOCK_TO_INDICATOR.get(code_i, "")
        main_rows.append(
            {
                "main_shock_code": code if not pd.isna(code) else None,
                "value_label": lab,
                "mapped_r2a_indicator": mapped_ind,
                "n_households": int(count),
                "pct": round(100 * int(count) / len(df), 4),
            }
        )
    main_inv = pd.DataFrame(main_rows)
    # Consistency: when main shock = k, is corresponding R2A == 1?
    consistency = []
    for code, ind in MAIN_SHOCK_TO_INDICATOR.items():
        if ind not in df.columns:
            continue
        sub = df.loc[df["main_shock_type"] == code]
        if len(sub) == 0:
            continue
        match = int((pd.to_numeric(sub[ind], errors="coerce") == 1).sum())
        consistency.append(
            {
                "main_shock_code": code,
                "indicator": ind,
                "n_with_main_shock": len(sub),
                "n_indicator_equals_1": match,
                "pct_consistent": round(100 * match / len(sub), 2),
            }
        )
    cons_df = pd.DataFrame(consistency)
    cons_df.to_csv(TABLES / "main_shock_r2a_consistency.csv", index=False)

    main_summary = {
        "stata_variable": "R2B",
        "renamed_name": "main_shock_type",
        "variable_label": labels.get("R2B", ""),
        "value_labels_json": json.dumps(r2b_vl, ensure_ascii=False),
        "n_missing": int(main.isna().sum()),
        "pct_missing": round(100 * main.isna().sum() / len(df), 4),
        "n_non_missing": int(main.notna().sum()),
        "note": (
            "R2B asks for the main shock among those experienced. "
            "It is not a multi-response inventory; use with R2A multi-response carefully."
        ),
    }
    pd.DataFrame([main_summary]).to_csv(TABLES / "main_shock_inventory_meta.csv", index=False)
    main_inv.to_csv(TABLES / "main_shock_inventory.csv", index=False)
    print("Wrote main_shock_inventory.csv and consistency table")

    # --- R2C + Classification_A_Resilience ---
    class_vl = value_labels.get("Classification_A_Resilience") or {}
    coping_rows = []
    for renamed in COPING_ACTION_VARS:
        original = inv_rename.get(renamed, "")
        series = df[renamed] if renamed in df.columns else pd.Series(dtype=float)
        n_miss = int(series.isna().sum())
        coping_rows.append(
            {
                "module": "R2C",
                "original_stata_name": original,
                "renamed_name": renamed,
                "variable_label": labels.get(original, ""),
                "value_labels": json.dumps(value_labels.get(original) or {}, ensure_ascii=False),
                "dtype": str(series.dtype) if renamed in df.columns else "missing_from_frame",
                "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / len(df), 4) if renamed in df.columns else None,
                "value_frequencies": freq_table(series) if renamed in df.columns else "",
                "n_positive_1": int((pd.to_numeric(series, errors="coerce") == 1).sum())
                if renamed in df.columns
                else None,
            }
        )

    # Classification target
    code = df["coping_category_code"] if "coping_category_code" in df.columns else None
    cat = df["coping_category"] if "coping_category" in df.columns else None
    coping_rows.append(
        {
            "module": "Classification_A_Resilience",
            "original_stata_name": "Classification_A_Resilience",
            "renamed_name": "coping_category_code / coping_category",
            "variable_label": labels.get("Classification_A_Resilience", ""),
            "value_labels": json.dumps(class_vl, ensure_ascii=False),
            "dtype": str(code.dtype) if code is not None else "",
            "n_missing": int(code.isna().sum()) if code is not None else None,
            "pct_missing": round(100 * code.isna().sum() / len(df), 4) if code is not None else None,
            "value_frequencies": freq_table(code) if code is not None else "",
            "n_positive_1": None,
            "construction_note": (
                "Existing preparation maps Classification_A_Resilience codes to "
                f"{CODE_TO_CLASS}. Stata value labels: {class_vl}. "
                "Questionnaire/context: R2C asks how the household managed/coped after "
                "the shock experience; Classification_A_Resilience is a derived resilience "
                "classification over coping responses. In the drought notebook this was treated "
                "as the household's coping category for the analysis sample. "
                "It is associated with the shock module (R2), not a free-standing annual coping "
                "measure — but it is a single household-level classification, not one row per shock."
            ),
        }
    )
    # Cross-check code vs string label
    if code is not None and cat is not None:
        mapped = code.map(CODE_TO_CLASS)
        agree = int(((mapped == cat) | (code.isna() & cat.isna())).sum())
        print(f"Code↔string agreement (incl. both-missing): {agree}/{len(df)}")

    coping_inv = pd.DataFrame(coping_rows)
    coping_inv.to_csv(TABLES / "coping_variable_inventory.csv", index=False)
    print(f"Wrote coping_variable_inventory.csv ({len(coping_inv)} rows)")

    # Short markdown note
    note = TABLES / "01_audit_notes.md"
    note.write_text(
        "\n".join(
            [
                "# Script 01 audit notes",
                "",
                f"- Farming frame n = {len(df)} (full farming sample with reported land size).",
                f"- Drought-exposed within frame: {int((df['drought_exposed']==1).sum())}.",
                f"- R2A indicators: {len(SHOCK_INDICATOR_VARS)}; Stata names R2A__1 … R2A__12.",
                f"- R2B label: {labels.get('R2B', '')}.",
                f"- Classification_A_Resilience labels: {class_vl}.",
                f"- R2A↔R2B consistency mean pct: {cons_df['pct_consistent'].mean():.1f}%.",
                "- Missing shock indicators: do not impute as zero.",
                "",
            ]
        )
    )
    print("Script 01 complete.")


if __name__ == "__main__":
    main()
