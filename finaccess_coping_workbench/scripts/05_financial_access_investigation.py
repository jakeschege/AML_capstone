#!/usr/bin/env python3
"""Investigate financial access variables for the drought-exposed sample.

Does not remove, combine, encode or transform variables.
Does not modify the analytical dataset or the main notebook.
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
    FINANCIAL_ACCESS_VARS,
    REPORTS_DIR,
    TABLES_DIR,
    VALUE_LABELS_PATH,
    ensure_output_dirs,
    load_analysis_2024,
)

# Codebook label sets for interpretation
LABEL_SETS = {
    "buyer_supplier_credit": "Q12",
    "formal_agri_borrowing": "Q13",
    "bank_usage": "bank_usage",
    "savings_usage": "Savings_usage",
    "loan_usage": "Loan_usage",
    "mobile_money_usage": "mobile_money_use",
    "mfi_usage": "MFIs_use",
}

CONSTRUCT_GROUP = {
    "buyer_supplier_credit": "agri_trade_credit_question",
    "formal_agri_borrowing": "agri_trade_credit_question",
    "used_formal_credit": "derived_usage_flag",
    "used_buyer_credit": "derived_usage_flag",
    "used_digital_credit": "derived_usage_flag",
    "used_insurance": "derived_usage_flag",
    "bank_usage": "service_usage",
    "savings_usage": "service_usage",
    "loan_usage": "service_usage",
    "mobile_money_usage": "service_usage",
    "mfi_usage": "service_usage",
}

COPING_OVERLAP_CANDIDATES = [
    "coping_loans_credit_borrow",
    "coping_savings",
    "coping_claimed_insurance",
]

COMPONENT_COLS = ["loan_formal", "digital_app_loan", "insurance_excl_nhif"]


def load_label_maps() -> dict[str, dict[int, str]]:
    """Load selected value labels with None preserved as text."""
    maps = {var: {} for var in FINANCIAL_ACCESS_VARS}
    if not VALUE_LABELS_PATH.exists():
        return maps
    labels = pd.read_csv(VALUE_LABELS_PATH, keep_default_na=False)
    for var, label_set in LABEL_SETS.items():
        subset = labels.loc[labels["label_set"] == label_set, ["code", "text"]]
        mapping: dict[int, str] = {}
        for _, row in subset.iterrows():
            try:
                code = int(float(row["code"]))
            except (TypeError, ValueError):
                continue
            text = str(row["text"]).strip()
            mapping[code] = text if text else "[blank label]"
        maps[var] = mapping
    # Binary derived flags: document intended meaning
    for var in [
        "used_formal_credit",
        "used_buyer_credit",
        "used_digital_credit",
        "used_insurance",
    ]:
        maps[var] = {0: "No", 1: "Yes"}
    return maps


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Completeness and uniqueness inventory."""
    rows = []
    for col in FINANCIAL_ACCESS_VARS:
        s = df[col]
        rows.append(
            {
                "variable": col,
                "construct_group": CONSTRUCT_GROUP[col],
                "dtype": str(s.dtype),
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(100.0 * float(s.isna().mean()), 2),
                "n_unique": int(s.nunique(dropna=True)),
                "min": float(pd.to_numeric(s, errors="coerce").min()),
                "max": float(pd.to_numeric(s, errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def frequencies(df: pd.DataFrame, label_maps: dict[str, dict[int, str]]) -> pd.DataFrame:
    """Category frequencies with labels where available."""
    rows = []
    for col in FINANCIAL_ACCESS_VARS:
        label_map = label_maps.get(col, {})
        vc = df[col].value_counts(dropna=False).sort_index()
        for code, n in vc.items():
            if pd.isna(code):
                code_out = np.nan
                label = "[missing]"
            else:
                code_out = int(code) if float(code).is_integer() else float(code)
                try:
                    label = label_map.get(int(code), "[no codebook label]")
                except (TypeError, ValueError):
                    label = "[no codebook label]"
            rows.append(
                {
                    "variable": col,
                    "construct_group": CONSTRUCT_GROUP[col],
                    "code": code_out,
                    "label": label,
                    "n_households": int(n),
                    "pct": round(100.0 * n / len(df), 2),
                }
            )
    return pd.DataFrame(rows)


def redundancy_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Verify documented derived-flag constructions empirically."""
    rows = []

    # used_buyer_credit == (buyer_supplier_credit == 1)
    buyer_yes = (pd.to_numeric(df["buyer_supplier_credit"], errors="coerce") == 1).astype(int)
    used_buyer = pd.to_numeric(df["used_buyer_credit"], errors="coerce").astype(int)
    agree_buyer = int((buyer_yes == used_buyer).sum())
    rows.append(
        {
            "check": "used_buyer_credit_equals_buyer_supplier_credit_eq_1",
            "n": len(df),
            "n_agree": agree_buyer,
            "pct_agree": round(100.0 * agree_buyer / len(df), 2),
            "fully_deterministic": agree_buyer == len(df),
            "notes": "Exact recode of Q12 Yes into a 0/1 flag",
        }
    )

    # used_formal_credit from documented components
    loan_usage = pd.to_numeric(df["loan_usage"], errors="coerce") == 1
    loan_formal = (
        pd.to_numeric(df["loan_formal"], errors="coerce") == 1
        if "loan_formal" in df.columns
        else pd.Series(False, index=df.index)
    )
    formal_agri = pd.to_numeric(df["formal_agri_borrowing"], errors="coerce") == 1
    recon_formal = (loan_usage | loan_formal | formal_agri).astype(int)
    used_formal = pd.to_numeric(df["used_formal_credit"], errors="coerce").astype(int)
    agree_formal = int((recon_formal == used_formal).sum())
    rows.append(
        {
            "check": "used_formal_credit_equals_loan_usage_or_loan_formal_or_formal_agri",
            "n": len(df),
            "n_agree": agree_formal,
            "pct_agree": round(100.0 * agree_formal / len(df), 2),
            "fully_deterministic": agree_formal == len(df),
            "notes": "Documented OR construction from notebook/feature dictionary",
        }
    )

    # used_digital_credit from digital_app_loan
    if "digital_app_loan" in df.columns:
        recon_digital = (
            pd.to_numeric(df["digital_app_loan"], errors="coerce") == 1
        ).astype(int)
        used_digital = pd.to_numeric(df["used_digital_credit"], errors="coerce").astype(int)
        agree_digital = int((recon_digital == used_digital).sum())
        rows.append(
            {
                "check": "used_digital_credit_equals_digital_app_loan_eq_1",
                "n": len(df),
                "n_agree": agree_digital,
                "pct_agree": round(100.0 * agree_digital / len(df), 2),
                "fully_deterministic": agree_digital == len(df),
                "notes": "Derived from digital_app_loan",
            }
        )
    else:
        rows.append(
            {
                "check": "used_digital_credit_equals_digital_app_loan_eq_1",
                "n": len(df),
                "n_agree": np.nan,
                "pct_agree": np.nan,
                "fully_deterministic": False,
                "notes": "digital_app_loan not present in analytical sample columns",
            }
        )

    # used_insurance from insurance_excl_nhif
    if "insurance_excl_nhif" in df.columns:
        recon_ins = (
            pd.to_numeric(df["insurance_excl_nhif"], errors="coerce") == 1
        ).astype(int)
        used_ins = pd.to_numeric(df["used_insurance"], errors="coerce").astype(int)
        agree_ins = int((recon_ins == used_ins).sum())
        rows.append(
            {
                "check": "used_insurance_equals_insurance_excl_nhif_eq_1",
                "n": len(df),
                "n_agree": agree_ins,
                "pct_agree": round(100.0 * agree_ins / len(df), 2),
                "fully_deterministic": agree_ins == len(df),
                "notes": "Derived from insurance_excl_nhif",
            }
        )
    else:
        rows.append(
            {
                "check": "used_insurance_equals_insurance_excl_nhif_eq_1",
                "n": len(df),
                "n_agree": np.nan,
                "pct_agree": np.nan,
                "fully_deterministic": False,
                "notes": "insurance_excl_nhif not present in analytical sample columns",
            }
        )

    # loan_usage vs used_formal_credit overlap (not expected to be identical)
    both = int(((df["loan_usage"] == 1) & (df["used_formal_credit"] == 1)).sum())
    loan_only = int(((df["loan_usage"] == 1) & (df["used_formal_credit"] == 0)).sum())
    formal_only = int(((df["loan_usage"] == 0) & (df["used_formal_credit"] == 1)).sum())
    rows.append(
        {
            "check": "loan_usage_vs_used_formal_credit_overlap",
            "n": len(df),
            "n_agree": both,
            "pct_agree": round(100.0 * both / len(df), 2),
            "fully_deterministic": False,
            "notes": (
                f"both=1: {both}; loan_usage=1 & used_formal=0: {loan_only}; "
                f"loan_usage=0 & used_formal=1: {formal_only}"
            ),
        }
    )

    return pd.DataFrame(rows)


def pairwise_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise exact agreement and positive co-occurrence for binary-like fields."""
    # Standardise to comparable binary where coding allows
    binary_views = {
        "buyer_supplier_credit_yes": (
            pd.to_numeric(df["buyer_supplier_credit"], errors="coerce") == 1
        ).astype(int),
        "formal_agri_borrowing_yes": (
            pd.to_numeric(df["formal_agri_borrowing"], errors="coerce") == 1
        ).astype(int),
        "used_formal_credit": pd.to_numeric(df["used_formal_credit"], errors="coerce").astype(int),
        "used_buyer_credit": pd.to_numeric(df["used_buyer_credit"], errors="coerce").astype(int),
        "used_digital_credit": pd.to_numeric(df["used_digital_credit"], errors="coerce").astype(int),
        "used_insurance": pd.to_numeric(df["used_insurance"], errors="coerce").astype(int),
        "bank_usage": pd.to_numeric(df["bank_usage"], errors="coerce").astype(int),
        "savings_usage": pd.to_numeric(df["savings_usage"], errors="coerce").astype(int),
        "loan_usage": pd.to_numeric(df["loan_usage"], errors="coerce").astype(int),
        "mobile_money_current": (
            pd.to_numeric(df["mobile_money_usage"], errors="coerce") == 1
        ).astype(int),
        "mfi_current": (pd.to_numeric(df["mfi_usage"], errors="coerce") == 1).astype(int),
    }
    names = list(binary_views.keys())
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = binary_views[left]
            b = binary_views[right]
            agree = int((a == b).sum())
            both = int(((a == 1) & (b == 1)).sum())
            rows.append(
                {
                    "variable_a": left,
                    "variable_b": right,
                    "pct_exact_agreement": round(100.0 * agree / len(df), 2),
                    "n_both_positive": both,
                    "pct_both_positive": round(100.0 * both / len(df), 2),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["pct_exact_agreement", "pct_both_positive"], ascending=False)
        .reset_index(drop=True)
    )


def leakage_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag conceptual overlaps with coping actions and temporal uncertainty."""
    rows = []
    mapping = [
        (
            "loan_usage",
            "coping_loans_credit_borrow",
            "Loan usage may overlap conceptually with borrowing as a coping action",
        ),
        (
            "used_formal_credit",
            "coping_loans_credit_borrow",
            "Derived formal credit use may overlap with coping borrowing",
        ),
        (
            "used_buyer_credit",
            "coping_loans_credit_borrow",
            "Buyer/supplier credit may overlap with credit-based coping",
        ),
        (
            "buyer_supplier_credit",
            "coping_loans_credit_borrow",
            "Q12 agri-trade credit may overlap with credit-based coping",
        ),
        (
            "formal_agri_borrowing",
            "coping_loans_credit_borrow",
            "Q13 agri credit question may overlap with credit-based coping",
        ),
        (
            "used_digital_credit",
            "coping_loans_credit_borrow",
            "Digital credit use may overlap with borrowing as coping",
        ),
        (
            "savings_usage",
            "coping_savings",
            "Savings usage may overlap with using savings as a coping action",
        ),
        (
            "used_insurance",
            "coping_claimed_insurance",
            "Insurance use/access may overlap with claiming insurance as coping",
        ),
    ]

    for fin_var, coping_var, note in mapping:
        if coping_var not in df.columns:
            rows.append(
                {
                    "financial_variable": fin_var,
                    "related_coping_variable": coping_var,
                    "coping_var_present": False,
                    "n_both_positive": np.nan,
                    "pct_both_positive": np.nan,
                    "risk_flag": "conceptual_overlap_and_temporal_uncertainty",
                    "notes": note + "; coping variable absent from sample",
                }
            )
            continue

        # Positive definitions differ by coding
        if fin_var in ("buyer_supplier_credit", "formal_agri_borrowing"):
            fin_pos = pd.to_numeric(df[fin_var], errors="coerce") == 1
        else:
            fin_pos = pd.to_numeric(df[fin_var], errors="coerce") == 1
        coping_pos = pd.to_numeric(df[coping_var], errors="coerce") == 1
        both = int((fin_pos & coping_pos).sum())
        rows.append(
            {
                "financial_variable": fin_var,
                "related_coping_variable": coping_var,
                "coping_var_present": True,
                "n_both_positive": both,
                "pct_both_positive": round(100.0 * both / len(df), 2),
                "risk_flag": "conceptual_overlap_and_temporal_uncertainty",
                "notes": note,
            }
        )

    # Always record unresolved timing for all finance vars
    for var in FINANCIAL_ACCESS_VARS:
        rows.append(
            {
                "financial_variable": var,
                "related_coping_variable": "",
                "coping_var_present": "",
                "n_both_positive": "",
                "pct_both_positive": "",
                "risk_flag": "temporal_availability_unresolved",
                "notes": (
                    "Whether the measure reflects pre-shock access or "
                    "post-shock response remains for the leakage audit"
                ),
            }
        )
    return pd.DataFrame(rows)


def suitability_notes(df: pd.DataFrame, redundancy: pd.DataFrame) -> pd.DataFrame:
    """Separate segmentation vs classification suitability notes."""
    rare = []
    for col in FINANCIAL_ACCESS_VARS:
        s = pd.to_numeric(df[col], errors="coerce")
        if col in ("mobile_money_usage", "mfi_usage"):
            # currently use share
            pos_pct = 100.0 * float((s == 1).mean())
        elif col in ("buyer_supplier_credit", "formal_agri_borrowing"):
            pos_pct = 100.0 * float((s == 1).mean())
        else:
            pos_pct = 100.0 * float((s == 1).mean())
        rare.append((col, pos_pct))
    rare_map = dict(rare)

    buyer_det = bool(
        redundancy.loc[
            redundancy["check"] == "used_buyer_credit_equals_buyer_supplier_credit_eq_1",
            "fully_deterministic",
        ].iloc[0]
    )
    formal_det = bool(
        redundancy.loc[
            redundancy["check"]
            == "used_formal_credit_equals_loan_usage_or_loan_formal_or_formal_agri",
            "fully_deterministic",
        ].iloc[0]
    )

    rows = []
    for col in FINANCIAL_ACCESS_VARS:
        group = CONSTRUCT_GROUP[col]
        pos_pct = rare_map[col]
        seg_note = (
            "May help distinguish households by financial service profile "
            "for agricultural segmentation only if treated as finance capacity, "
            "not as a coping outcome."
        )
        clf_note = (
            "Candidate classifier predictor only after leakage audit confirms "
            "the measure is not a post-shock coping response."
        )
        caution = []
        if col == "used_buyer_credit" and buyer_det:
            caution.append("Deterministic duplicate of buyer_supplier_credit Yes")
        if col == "used_formal_credit" and formal_det:
            caution.append(
                "Deterministic composite of loan_usage/loan_formal/formal_agri_borrowing"
            )
        if pos_pct < 5:
            caution.append(f"Low positive rate ({pos_pct:.2f}%)")
        if group == "derived_usage_flag":
            caution.append("Derived flag; not interchangeable with source questions")
        if col in ("mobile_money_usage", "mfi_usage"):
            caution.append("Three-level usage status, not a simple binary access indicator")

        rows.append(
            {
                "variable": col,
                "construct_group": group,
                "positive_or_current_use_pct": round(pos_pct, 2),
                "segmentation_suitability": "provisional_candidate",
                "classification_suitability": "provisional_candidate_pending_leakage_audit",
                "segmentation_note": seg_note,
                "classification_note": clf_note,
                "cautions": "; ".join(caution) if caution else "",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    n: int,
    inv: pd.DataFrame,
    freq: pd.DataFrame,
    redundancy: pd.DataFrame,
    pairwise: pd.DataFrame,
    leakage: pd.DataFrame,
    suitability: pd.DataFrame,
) -> Path:
    """Write Markdown report from actual investigation outputs."""
    missing_total = int(inv["missing"].sum())
    det_rows = redundancy.loc[redundancy["fully_deterministic"] == True]  # noqa: E712
    rare = suitability.loc[suitability["positive_or_current_use_pct"] < 5]
    top_pairs = pairwise.head(10)
    conceptual = leakage.loc[
        leakage["risk_flag"] == "conceptual_overlap_and_temporal_uncertainty"
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Financial Access Investigation Report

Generated: {timestamp}

## What was investigated

Eleven financial access variables for the drought-exposed analytical sample (`n = {n}`):

```
{', '.join(FINANCIAL_ACCESS_VARS)}
```

The investigation distinguished agri-trade credit questions, derived usage flags and broader service usage measures. Variables were not removed, combined, encoded or transformed.

## Observed findings

### Inventory

```
{inv.to_string(index=False)}
```

Total missing values across the 11 variables: {missing_total}

### Frequencies

```
{freq.to_string(index=False)}
```

### Redundancy checks

```
{redundancy.to_string(index=False)}
```

Fully deterministic constructions confirmed: {len(det_rows)}

### Pairwise agreement among binary views

Top pairs by exact agreement:

```
{top_pairs.to_string(index=False)}
```

### Leakage and coping overlap flags

Conceptual overlap checks:

```
{conceptual.to_string(index=False)}
```

Temporal availability of these finance measures relative to drought coping remains unresolved and is deferred to the predictor leakage audit.

### Suitability notes

```
{suitability.to_string(index=False)}
```

Low positive or current-use rates under 5%: {', '.join(rare['variable'].tolist()) if len(rare) else 'none'}

## Interpretation

Access questions, derived usage flags and service usage measures are related but not interchangeable constructs.

`used_buyer_credit` is an exact 0/1 recode of `buyer_supplier_credit == 1`. `used_formal_credit` is a composite of broader loan usage and agri-credit indicators. These relationships create redundancy for modelling even though the original and derived fields can remain useful for audit.

`mobile_money_usage` and `mfi_usage` retain former-user categories and therefore measure usage history, not simple binary access.

Several finance variables conceptually overlap with coping actions such as borrowing, using savings or claiming insurance. That does not by itself prove leakage, but it does mean they cannot be treated as automatically valid pre-shock predictors.

## Limitations

- Codebook labels were used for interpretation only.
- Pairwise agreement uses binary views of multi-level variables where needed.
- No target-based feature selection was performed.
- Questionnaire timing of finance measures versus coping responses was not resolved here.

## Recommended preparation decisions for review

1. Retain all 11 financial access variables as provisional candidates.
2. Flag `used_buyer_credit` as a deterministic duplicate of `buyer_supplier_credit` for later feature assessment. Do not drop either field yet.
3. Flag `used_formal_credit` as a derived composite overlapping `loan_usage` and `formal_agri_borrowing`.
4. Keep segmentation and classification uses distinct: finance capacity may be relevant for both, but classification use depends on the leakage audit.
5. Do not encode, combine or transform these variables in this step.
6. Carry conceptual overlap with coping actions into script 06 leakage audit.

## Output files

- `outputs/tables/05_financial_inventory.csv`
- `outputs/tables/05_financial_frequencies.csv`
- `outputs/tables/05_financial_redundancy_checks.csv`
- `outputs/tables/05_financial_pairwise_agreement.csv`
- `outputs/tables/05_financial_leakage_flags.csv`
- `outputs/tables/05_financial_suitability_notes.csv`
"""
    path = REPORTS_DIR / "05_financial_access_investigation.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()
    required = list(FINANCIAL_ACCESS_VARS)
    # Optional component/coping columns for checks
    optional = COMPONENT_COLS + COPING_OVERLAP_CANDIDATES
    frame_cols = required + optional
    # Load with required only, then reattach optional from modelling frame if present
    analysis = load_analysis_2024(required_columns=required)
    # Bring optional columns if available in the same filtered sample
    from config import load_modelling_frame

    full = load_modelling_frame()
    drought = full.loc[full["drought_exposed"] == 1].copy()
    for col in optional:
        if col in drought.columns and col not in analysis.columns:
            analysis[col] = drought[col].values

    label_maps = load_label_maps()

    print("=" * 72)
    print("05 Financial access investigation")
    print("=" * 72)
    print(f"Observations: {len(analysis)}")

    inv = inventory(analysis)
    freq = frequencies(analysis, label_maps)
    redundancy = redundancy_checks(analysis)
    pairwise = pairwise_agreement(analysis)
    leakage = leakage_flags(analysis)
    suitability = suitability_notes(analysis, redundancy)

    outputs = {
        "05_financial_inventory.csv": inv,
        "05_financial_frequencies.csv": freq,
        "05_financial_redundancy_checks.csv": redundancy,
        "05_financial_pairwise_agreement.csv": pairwise,
        "05_financial_leakage_flags.csv": leakage,
        "05_financial_suitability_notes.csv": suitability,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    report_path = write_report(
        n=len(analysis),
        inv=inv,
        freq=freq,
        redundancy=redundancy,
        pairwise=pairwise,
        leakage=leakage,
        suitability=suitability,
    )
    print(f"Wrote {report_path}")

    print("\nInventory")
    print(inv.to_string(index=False))
    print("\nRedundancy checks")
    print(redundancy.to_string(index=False))
    print("\nSuitability notes")
    print(suitability.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
