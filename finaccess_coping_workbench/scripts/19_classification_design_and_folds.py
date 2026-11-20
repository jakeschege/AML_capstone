#!/usr/bin/env python3
"""Classification design lock and shared CV fold materialisation.

Locks the evaluation protocol, audits predictors, and writes identical
repeated stratified five-fold assignments for primary / secondary / extended
tracks. No classification models are fitted.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    ADDITIONAL_SHOCK_VARS,
    COPING_ACTION_VARS,
    EXPECTED_DROUGHT_N,
    FINANCIAL_ACCESS_VARS,
    ID_VARS,
    LAND_SIZE_STANDARDISED,
    LOGS_DIR,
    PREPARED_DIR,
    REPORTS_DIR,
    SAMPLE_FLAG_VARS,
    TABLES_DIR,
    TARGET_VARS,
    WEIGHT_VARS,
    ensure_output_dirs,
)

RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 5
FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]
THREE_ORDER = ["personal", "formal_or_informal", "did_nothing"]


def predictor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_VARS]


def load_feature_list(name: str) -> list[str]:
    path = TABLES_DIR / f"09_feature_list_{name}.csv"
    fl = pd.read_csv(path)
    return fl["column"].tolist()


def audit_predictors(
    X_pri: pd.DataFrame, X_ext: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    pri_cols = predictor_cols(X_pri)
    ext_cols = predictor_cols(X_ext)
    list_pri = load_feature_list("primary_classification")
    list_ext = load_feature_list("extended_classification")

    forbidden_primary = (
        set(COPING_ACTION_VARS)
        | set(TARGET_VARS)
        | set(FINANCIAL_ACCESS_VARS)
        | set(SAMPLE_FLAG_VARS)
        | set(WEIGHT_VARS)
        | set(ID_VARS)
        | set(ADDITIONAL_SHOCK_VARS)
    )
    # main_shock_type dummies start with this prefix in the extended matrix
    shock_dummy_prefix = "main_shock_type_"

    rows: list[dict] = []

    def add(
        check_id: str,
        status: str,
        detail: str,
        *,
        matrix: str = "both",
    ) -> None:
        rows.append(
            {
                "check_id": check_id,
                "matrix": matrix,
                "status": status,
                "detail": detail,
            }
        )

    # Shape / list match
    add(
        "primary_n_features",
        "PASS" if len(pri_cols) == 69 else "FAIL",
        f"primary predictors n={len(pri_cols)} (expected 69)",
        matrix="primary",
    )
    add(
        "extended_n_features",
        "PASS" if len(ext_cols) == 92 else "FAIL",
        f"extended predictors n={len(ext_cols)} (expected 92)",
        matrix="extended",
    )
    add(
        "primary_list_match",
        "PASS" if pri_cols == list_pri else "FAIL",
        "prepared primary columns match 09_feature_list_primary_classification.csv"
        if pri_cols == list_pri
        else f"mismatch; prepared_only={sorted(set(pri_cols)-set(list_pri))}; "
        f"list_only={sorted(set(list_pri)-set(pri_cols))}",
        matrix="primary",
    )
    add(
        "extended_list_match",
        "PASS" if ext_cols == list_ext else "FAIL",
        "prepared extended columns match 09_feature_list_extended_classification.csv"
        if ext_cols == list_ext
        else f"mismatch; prepared_only={sorted(set(ext_cols)-set(list_ext))}; "
        f"list_only={sorted(set(list_ext)-set(ext_cols))}",
        matrix="extended",
    )

    # Primary must be a subset of extended; extended-only listed
    pri_set, ext_set = set(pri_cols), set(ext_cols)
    add(
        "primary_subset_of_extended",
        "PASS" if pri_set <= ext_set else "FAIL",
        "all primary predictors appear in extended matrix"
        if pri_set <= ext_set
        else f"missing_from_extended={sorted(pri_set - ext_set)}",
    )
    extended_only = sorted(ext_set - pri_set)
    add(
        "extended_only_count",
        "PASS" if len(extended_only) == 23 else "FAIL",
        f"extended-only columns n={len(extended_only)} (expected 23)",
        matrix="extended",
    )

    # Forbidden in primary
    bad_pri = sorted(forbidden_primary & pri_set)
    bad_pri += sorted(c for c in pri_cols if c.startswith(shock_dummy_prefix))
    add(
        "primary_no_forbidden",
        "PASS" if not bad_pri else "FAIL",
        "no coping/target/financial/sample-flag/weight/shock predictors in primary"
        if not bad_pri
        else f"forbidden_in_primary={bad_pri}",
        matrix="primary",
    )

    # Forbidden in extended (shocks allowed; coping/finance/targets still not)
    forbidden_ext = (
        set(COPING_ACTION_VARS)
        | set(TARGET_VARS)
        | set(FINANCIAL_ACCESS_VARS)
        | set(SAMPLE_FLAG_VARS)
        | set(WEIGHT_VARS)
        | set(ID_VARS)
    )
    bad_ext = sorted(forbidden_ext & ext_set)
    add(
        "extended_no_leakage",
        "PASS" if not bad_ext else "FAIL",
        "no coping/target/financial/sample-flag/weight predictors in extended"
        if not bad_ext
        else f"forbidden_in_extended={bad_ext}",
        matrix="extended",
    )

    # Acres present with known missingness
    if LAND_SIZE_STANDARDISED in pri_cols:
        n_miss = int(X_pri[LAND_SIZE_STANDARDISED].isna().sum())
        add(
            "land_size_acres_present",
            "PASS" if n_miss == 198 else "WARN",
            f"{LAND_SIZE_STANDARDISED} in primary; missing n={n_miss} (expected 198)",
            matrix="primary",
        )
    else:
        add(
            "land_size_acres_present",
            "FAIL",
            f"{LAND_SIZE_STANDARDISED} missing from primary matrix",
            matrix="primary",
        )

    # Shannon / drought sample flags not in matrices
    for name, cols in (
        ("shannon_diversity", ["shannon_diversity"]),
        ("drought_flag", ["drought_exposed", "shock_drought_loss"]),
    ):
        hit = [c for c in cols if c in pri_set or c in ext_set]
        add(
            f"excluded_{name}",
            "PASS" if not hit else "FAIL",
            f"{name} absent from classification matrices"
            if not hit
            else f"unexpectedly_present={hit}",
        )

    return pd.DataFrame(rows), extended_only


def materialise_folds(y_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long fold table and class-count diagnostics from four-class labels."""
    ids = y_frame["respondent_id"].to_numpy()
    y4 = y_frame["coping_category"].to_numpy()
    y3 = y_frame["coping_category_collapsed"].to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
    )

    fold_rows: list[dict] = []
    count_rows: list[dict] = []

    for split_id, (train_idx, test_idx) in enumerate(rskf.split(np.zeros(len(y4)), y4)):
        repeat = split_id // N_SPLITS
        fold = split_id % N_SPLITS
        for split_name, idx in (("train", train_idx), ("test", test_idx)):
            for i in idx:
                fold_rows.append(
                    {
                        "respondent_id": ids[i],
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": split_id,
                        "split": split_name,
                        "coping_category": y4[i],
                        "coping_category_collapsed": y3[i],
                    }
                )
            # Class counts for this split
            labels = y4[idx]
            for cat in FOUR_ORDER:
                n = int((labels == cat).sum())
                count_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": split_id,
                        "split": split_name,
                        "coping_category": cat,
                        "n_households": n,
                        "split_n": len(idx),
                        "formal_zero": cat == "formal" and n == 0,
                        "formal_sparse_under_5": cat == "formal" and n < 5,
                    }
                )

    folds = pd.DataFrame(fold_rows)
    counts = pd.DataFrame(count_rows)

    # Integrity: each respondent appears once per split_id as train or test
    chk = folds.groupby(["split_id", "respondent_id"]).size()
    if not (chk == 1).all():
        raise ValueError("Fold table has duplicate respondent assignments within a split")
    if len(folds) != EXPECTED_DROUGHT_N * N_SPLITS * N_REPEATS:
        # each household appears in exactly one of train/test per split_id → N rows per split
        # total rows = n * n_splits * n_repeats
        raise ValueError(
            f"Unexpected fold row count {len(folds)}; "
            f"expected {EXPECTED_DROUGHT_N * N_SPLITS * N_REPEATS}"
        )

    return folds, counts


def fold_summary(counts: pd.DataFrame) -> dict:
    test_formal = counts.loc[
        (counts["split"] == "test") & (counts["coping_category"] == "formal")
    ]
    train_formal = counts.loc[
        (counts["split"] == "train") & (counts["coping_category"] == "formal")
    ]
    return {
        "n_test_folds": int(len(test_formal)),
        "formal_test_min": int(test_formal["n_households"].min()),
        "formal_test_max": int(test_formal["n_households"].max()),
        "formal_test_mean": round(float(test_formal["n_households"].mean()), 2),
        "formal_test_zero_folds": int(test_formal["formal_zero"].sum()),
        "formal_test_under_5_folds": int(test_formal["formal_sparse_under_5"].sum()),
        "formal_train_min": int(train_formal["n_households"].min()),
        "formal_train_max": int(train_formal["n_households"].max()),
    }


def main() -> None:
    ensure_output_dirs()
    X_pri = pd.read_parquet(PREPARED_DIR / "09_X_primary_classification.parquet")
    X_ext = pd.read_parquet(PREPARED_DIR / "09_X_extended_classification.parquet")
    y = pd.read_parquet(PREPARED_DIR / "09_y_targets.parquet")

    if len(X_pri) != EXPECTED_DROUGHT_N or len(X_ext) != EXPECTED_DROUGHT_N:
        raise ValueError("Unexpected matrix length")
    if not X_pri["respondent_id"].equals(y["respondent_id"]):
        raise ValueError("Primary X and y respondent_id order mismatch")
    if not X_ext["respondent_id"].equals(y["respondent_id"]):
        raise ValueError("Extended X and y respondent_id order mismatch")
    if y["coping_category"].isna().any() or y["coping_category_collapsed"].isna().any():
        raise ValueError("Missing target labels")

    audit, extended_only = audit_predictors(X_pri, X_ext)
    if (audit["status"] == "FAIL").any():
        fails = audit.loc[audit["status"] == "FAIL", "check_id"].tolist()
        raise RuntimeError(f"Predictor audit failed: {fails}")

    folds, counts = materialise_folds(y)
    summary = fold_summary(counts)

    protocol = {
        "script": "19_classification_design_and_folds",
        "models_fitted": False,
        "random_seed": RANDOM_SEED,
        "cv": {
            "scheme": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "stratify_on": "coping_category",
            "shared_across_tracks": True,
        },
        "tracks": {
            "primary": {
                "target": "coping_category",
                "n_classes": 4,
                "matrix": "09_X_primary_classification.parquet",
                "n_features": 69,
                "interpretation": "household/agri/land predictors; no retrospective shocks",
            },
            "secondary": {
                "target": "coping_category_collapsed",
                "n_classes": 3,
                "matrix": "09_X_primary_classification.parquet",
                "n_features": 69,
                "interpretation": "same predictors and folds as primary; alternative target",
            },
            "extended": {
                "target": "coping_category",
                "n_classes": 4,
                "matrix": "09_X_extended_classification.parquet",
                "n_features": 92,
                "interpretation": (
                    "retrospective extension with same-window shock information; "
                    "not prospective prediction"
                ),
            },
        },
        "metrics": {
            "primary": "macro_f1",
            "additional": [
                "class_wise_precision",
                "class_wise_recall",
                "class_wise_f1",
                "confusion_matrix",
                "balanced_accuracy",
                "weighted_f1",
                "accuracy_secondary",
            ],
        },
        "imbalance_strategy": {
            "compare": ["unweighted", "class_weighted"],
            "smote_default": False,
        },
        "preprocessing": {
            "imputation_scaling": "fit_within_training_folds_only",
            "land_size_acres": {
                "baseline_a": "exclude",
                "baseline_b": "retain_with_train_fold_median_imputation",
                "missingness_indicator": "only_if_separately_justified",
                "n_missing": 198,
            },
            "cluster_k5_as_predictor": False,
        },
        "exclusions": {
            "coping_action_indicators": True,
            "financial_access_variables": True,
            "retrospective_shocks_in_primary": True,
            "sample_flags_in_predictors": True,
        },
        "extended_only_columns": extended_only,
        "class_counts_sample": {
            cat: int((y["coping_category"] == cat).sum()) for cat in FOUR_ORDER
        },
        "formal_fold_summary": summary,
        "deferred": [
            "C1_dummy_and_multinomial_logistic",
            "C2_decision_tree_random_forest",
            "C3_gradient_boosting",
            "full_primary_secondary_extended_scorecards",
        ],
    }

    # Write outputs
    protocol_path = TABLES_DIR / "19_classification_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {protocol_path}")

    for name, table in (
        ("19_cv_folds_long.csv", folds),
        ("19_fold_class_counts.csv", counts),
        ("19_predictor_audit.csv", audit),
    ):
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    # Compact formal test table for report
    formal_test = (
        counts.loc[
            (counts["split"] == "test") & (counts["coping_category"] == "formal"),
            ["repeat", "fold", "n_households", "split_n", "formal_sparse_under_5"],
        ]
        .sort_values(["repeat", "fold"])
        .reset_index(drop=True)
    )

    overall = y["coping_category"].value_counts().reindex(FOUR_ORDER)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fail_n = int((audit["status"] == "FAIL").sum())
    warn_n = int((audit["status"] == "WARN").sum())

    report = f"""# Classification Design Lock and Shared CV Folds (Script 19)

Generated: {timestamp}

## Scope

Design and fold infrastructure only. **No classification models fitted.**

## Locked tracks

| Track | Target | Matrix | Features | Role |
|---|---|---|---:|---|
| Primary | `coping_category` (4-class) | `09_X_primary_classification.parquet` | 69 | Main analysis |
| Secondary | `coping_category_collapsed` (3-class) | same primary matrix | 69 | Alternative target; same folds |
| Extended | `coping_category` (4-class) | `09_X_extended_classification.parquet` | 92 | Retrospective shock extension only |

## Evaluation protocol

- Scheme: `RepeatedStratifiedKFold` — {N_SPLITS} folds × {N_REPEATS} repeats
- Stratification: four-class `coping_category`
- Seed: {RANDOM_SEED}
- Identical fold assignments for primary, secondary, and extended
- Primary metric: macro F1
- Also report: class-wise precision / recall / F1, confusion matrices, balanced accuracy, weighted F1; overall accuracy secondary
- Imbalance: compare unweighted vs class-weighted learning; no SMOTE by default
- Preprocessing: imputation and scaling fit within training folds only

## Land-size handling (for upcoming fits)

- Baseline A: exclude `land_size_acres`
- Baseline B: retain with train-fold median imputation
- Missingness indicator only if separately justified
- Spec A `cluster_k5` is not a default C1 predictor

## Sample class counts (four-class)

```
{overall.to_string()}
```

## Predictor audit

Checks: {len(audit)} | FAIL: {fail_n} | WARN: {warn_n}

```
{audit.to_string(index=False)}
```

### Extended-only columns (n={len(extended_only)})

These are retrospective shock indicators / `main_shock_type` dummies and must not be described as prospective predictors:

```
{chr(10).join(f'- {c}' for c in extended_only)}
```

## Formal-class fold diagnostics (test splits)

| Statistic | Value |
|---|---:|
| Test folds | {summary['n_test_folds']} |
| Formal test min | {summary['formal_test_min']} |
| Formal test max | {summary['formal_test_max']} |
| Formal test mean | {summary['formal_test_mean']} |
| Test folds with formal = 0 | {summary['formal_test_zero_folds']} |
| Test folds with formal &lt; 5 | {summary['formal_test_under_5_folds']} |
| Formal train min | {summary['formal_train_min']} |
| Formal train max | {summary['formal_train_max']} |

Per test fold formal counts:

```
{formal_test.to_string(index=False)}
```

Class-specific metrics and confusion matrices will be essential in later scripts because formal remains sparse even when every test fold contains at least some formal cases.

## Deferred

- C1: Dummy + multinomial logistic (unweighted vs class-weighted; acres exclude vs fold-impute)
- C2–C3: Decision tree, random forest, gradient boosting
- Full primary / secondary / extended scorecards under these shared folds

## Output files

- `outputs/tables/19_classification_protocol.json`
- `outputs/tables/19_cv_folds_long.csv`
- `outputs/tables/19_fold_class_counts.csv`
- `outputs/tables/19_predictor_audit.csv`
"""
    path = REPORTS_DIR / "19_classification_design_and_folds.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")

    design = LOGS_DIR / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    old_status = (
        "**Status: Descriptive segmentation and coping profiling complete. "
        "Working typology Spec A k=5. Classification design may proceed "
        "(four-class primary; three-class secondary). Models not yet fitted.**"
    )
    new_status = (
        "**Status: Classification design locked (Script 19). Shared repeated "
        "stratified CV folds materialised. Models not yet fitted; next = C1 baselines.**"
    )
    if old_status in text:
        text = text.replace(old_status, new_status)

    note = f"""

## 14. Classification design lock (Script 19)

Classification evaluation protocol locked and shared CV folds materialised. No models fitted.

| Item | Locked choice |
|---|---|
| Primary track | Four-class `coping_category` + primary matrix (69 features) |
| Secondary track | Three-class `coping_category_collapsed` + same primary matrix / folds |
| Extended track | Four-class + extended matrix (92); retrospective interpretation only |
| CV | RepeatedStratifiedKFold, 5×5, seed {RANDOM_SEED}; identical folds across tracks |
| Primary metric | Macro F1; class-wise P/R/F1 and confusion matrices required |
| Imbalance | Compare unweighted vs class-weighted; no SMOTE by default |
| Land size | Baseline A exclude acres; Baseline B train-fold median impute |
| Cluster membership | Not a default C1 predictor |
| Formal test fold min / max | {summary['formal_test_min']} / {summary['formal_test_max']} |
| Next | C1 baselines (dummy + multinomial logistic) after review |

Report: `outputs/reports/19_classification_design_and_folds.md`
"""
    if "## 14. Classification design lock (Script 19)" not in text:
        text = text.rstrip() + note
    design.write_text(text, encoding="utf-8")
    print(f"Updated {design}")

    print("\n=== Predictor audit ===")
    print(audit.to_string(index=False))
    print("\n=== Formal test fold summary ===")
    print(summary)
    print("\nDone. No models fitted.")


if __name__ == "__main__":
    main()
