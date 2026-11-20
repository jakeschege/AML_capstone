#!/usr/bin/env python3
"""Export Stack A (preferred) + financial logistic fallback as classifier v0.2.0 artefacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
WB = ROOT / "all_shocks_framing_workbench"
PREPARED = WB / "outputs" / "prepared"
TABLES = WB / "outputs" / "tables"
OUT_STACK = ROOT / "artefacts" / "classifier" / "v0.2.0"
OUT_LOGIT = ROOT / "artefacts" / "classifier" / "v0.2.0-logistic-fallback"
ACTIVE = ROOT / "artefacts" / "metadata" / "ACTIVE_MODEL.json"

SEED = 42
CLASS_ORDER = ["personal", "informal", "formal", "did_nothing"]
DISPLAY_LABELS = {
    "personal": "Personal coping",
    "informal": "Informal support",
    "formal": "Formal support",
    "did_nothing": "No coping action reported",
}
CONTINUOUS = ["age_years", "education", "n_crop_types", "n_livestock_types"]

STACK_A_CV = {
    "macro_f1_mean": 0.5023,
    "macro_f1_std": 0.0097,
    "f1_formal_mean": 0.3146,
    "f1_formal_std": 0.0133,
    "protocol": "first_repeat_stratified_5fold_seed_42",
    "source": "script_14_stack_a_complete",
    "n_folds": 5,
}
LOGIT_CV = {
    "macro_f1_mean": 0.4937,
    "f1_formal_mean": 0.3098,
    "protocol": "first_repeat_stratified_5fold_seed_42",
    "source": "script_13_folds_0_4",
    "n_folds": 5,
}

LIMITATIONS_STACK = (
    "All-shocks financial-expanded Stack A (logistic + histogram GB → logistic meta). "
    "Validated on the first 5-fold repeat only (macro F1 ≈ 0.502; formal F1 ≈ 0.315). "
    "Not compared against the prior 25-fold averages in the stacking decision. "
    "Stack B was stopped incomplete and is not a deployment candidate. "
    "Serving artefact is refit on the full modelling cohort (n=5,241). "
    "Not for operational targeting or eligibility decisions. "
    "SHAP LinearExplainer is not supported for this stacked model."
)
LIMITATIONS_LOGIT = (
    "Financial-expanded class-weighted logistic fallback (93 predictors). "
    "5-fold macro F1 ≈ 0.494; formal F1 ≈ 0.310 on the same first repeat. "
    "Simpler alternative to Stack A. Not for operational targeting."
)


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def hash_list(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def hash_ids(ids: pd.Series) -> str:
    return hashlib.sha256(",".join(sorted(ids.astype(str))).encode()).hexdigest()


def make_pipe(feature_names: list[str], clf) -> Pipeline:
    cont = [c for c in CONTINUOUS if c in feature_names]
    other = [c for c in feature_names if c not in cont]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                cont,
            ),
            ("other", Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]), other),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_stack_a(feature_names: list[str]) -> StackingClassifier:
    estimators = [
        (
            "logistic_cw",
            make_pipe(
                feature_names,
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
                ),
            ),
        ),
        (
            "hist_gb_cw",
            make_pipe(
                feature_names,
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_depth=6,
                    learning_rate=0.1,
                    max_iter=150,
                    random_state=SEED,
                ),
            ),
        ),
    ]
    meta = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def build_schema(feature_names: list[str]) -> dict:
    # County codes from feature names
    counties = []
    for name in feature_names:
        if name.startswith("county_"):
            parts = name.split("_", 2)
            code = int(parts[1])
            label = parts[2].replace("_", " ") if len(parts) > 2 else str(code)
            counties.append({"code": code, "label": label, "one_hot_column": name})

    features = []
    for name in feature_names:
        if name in CONTINUOUS:
            ftype, group = "continuous", name
        elif name.startswith("county_"):
            ftype, group = "one_hot", "county"
        elif name.startswith("main_shock_"):
            ftype, group = "binary", "r2b_main_shock"
        elif name in {
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
        }:
            ftype, group = "binary", "financial_access"
        else:
            ftype, group = "binary", name
        features.append(
            {
                "name": name,
                "type": ftype,
                "group": group,
                "required_in_model_matrix": True,
            }
        )

    return {
        "version": "v0.2.0",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "features": features,
        "domain_inputs": [
            {
                "name": "county",
                "encoding": "one_hot",
                "allowed_codes": counties,
            },
            {
                "name": "main_shock_type",
                "encoding": "one_hot_to_main_shock_1_12",
                "allowed_codes": [{"code": i, "column": f"main_shock_{i}"} for i in range(1, 13)],
            },
            {
                "name": "financial_access_flags",
                "encoding": "binary_passthrough",
                "fields": [
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
                ],
            },
        ],
        "population": "all_shocks_modelling_n_5241",
        "excluded": [
            "land_size_acres",
            "R2C coping actions",
            "agricultural segment membership",
            "R2A individual shock indicators (baseline is A+R2B+financial)",
        ],
    }


def write_bundle(
    out: Path,
    model,
    feature_names: list[str],
    ids: pd.Series,
    version: str,
    algorithm: str,
    cv_metrics: dict,
    limitations: str,
    role: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    schema = build_schema(feature_names)
    schema["version"] = version
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    mapping = {
        "classes": CLASS_ORDER,
        "display_labels": DISPLAY_LABELS,
        "stata_codes": {"1": "formal", "2": "informal", "3": "personal", "4": "did_nothing"},
    }
    (out / "class_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    meta = {
        "name": "coping-classifier",
        "version": version,
        "role": role,
        "algorithm": algorithm,
        "predictor_spec_id": "all_shocks_financial_expansion_v1",
        "seed": SEED,
        "train_n": int(len(ids)),
        "n_features": len(feature_names),
        "dataset_hash": hash_ids(ids),
        "feature_list_hash": hash_list(feature_names),
        "package_versions": {
            "scikit-learn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "cv_metrics": cv_metrics,
        "limitations": limitations,
        "low_confidence_rule": {
            "description": "Flag when score margin (max - second_max) < 0.10 OR predicted class is formal",
            "margin_threshold": 0.1,
            "always_flag_classes": ["formal"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "fallback_path": "artefacts/classifier/v0.2.0-logistic-fallback",
        "preferred_path": "artefacts/classifier/v0.2.0",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "requirements_freeze.txt").write_text(
        f"scikit-learn=={__import__('sklearn').__version__}\n"
        f"numpy=={np.__version__}\npandas=={pd.__version__}\njoblib=={joblib.__version__}\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


def main() -> None:
    fin = pd.read_parquet(PREPARED / "11_financial_expansion_features.parquet")
    assert len(fin) == 5241
    y = fin["coping_category"].astype(str)
    ids = fin["respondent_id"]
    X = fin.drop(columns=["respondent_id", "coping_category"])
    feature_names = list(X.columns)
    assert len(feature_names) == 93

    print("Fitting Stack A on full modelling cohort…")
    stack = build_stack_a(feature_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack.fit(X, y)

    # Prediction smoke on first row
    proba = stack.predict_proba(X.iloc[[0]])[0]
    pred = stack.predict(X.iloc[[0]])[0]
    print(f"Smoke Stack A: pred={pred} proba_sum={proba.sum():.4f}")

    write_bundle(
        OUT_STACK,
        stack,
        feature_names,
        ids,
        version="v0.2.0",
        algorithm="stacking_logistic_hgb_meta_logistic",
        cv_metrics=STACK_A_CV,
        limitations=LIMITATIONS_STACK,
        role="preferred_stack_a",
    )

    print("Fitting logistic fallback…")
    logit = make_pipe(
        feature_names,
        LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit.fit(X, y)
    pred_l = logit.predict(X.iloc[[0]])[0]
    print(f"Smoke logistic: pred={pred_l}")

    write_bundle(
        OUT_LOGIT,
        logit,
        feature_names,
        ids,
        version="v0.2.0-logistic-fallback",
        algorithm="multinomial_logistic_regression_class_weighted",
        cv_metrics=LOGIT_CV,
        limitations=LIMITATIONS_LOGIT,
        role="fallback_financial_logistic",
    )

    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    active = {
        "alias": "champion",
        "classifier_version": "v0.2.0",
        "path": "artefacts/classifier/v0.2.0",
        "fallback_path": "artefacts/classifier/v0.2.0-logistic-fallback",
        "feature_spec": "all_shocks_financial_expansion_93",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "selection_note": (
            "Stack A preferred (5-fold macro F1≈0.502). "
            "Financial logistic retained as simpler fallback. "
            "Stack B incomplete — not packaged."
        ),
    }
    ACTIVE.write_text(json.dumps(active, indent=2), encoding="utf-8")
    print(f"Updated {ACTIVE}")
    print("Export complete.")


if __name__ == "__main__":
    main()
