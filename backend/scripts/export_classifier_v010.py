#!/usr/bin/env python3
"""Export acres-excluded class-weighted multinomial logistic classifier v0.1.0."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.encoding import (  # noqa: E402
    CONTINUOUS_FEATURES,
    AGRICULTURAL_HAS_VARS,
    LAND_OWNERSHIP_VARS,
    PORTFOLIO_COUNT_VARS,
    primary_feature_names,
)

WB = ROOT / "finaccess_coping_workbench"
PREPARED = WB / "outputs" / "prepared"
TABLES = WB / "outputs" / "tables"
OUT = ROOT / "artefacts" / "classifier" / "v0.1.0"
ACTIVE = ROOT / "artefacts" / "metadata" / "ACTIVE_MODEL.json"

CLASS_ORDER = ["personal", "informal", "formal", "did_nothing"]
DISPLAY_LABELS = {
    "personal": "Personal coping",
    "informal": "Informal support",
    "formal": "Formal support",
    "did_nothing": "No coping action reported",
}

CV_METRICS = {
    "macro_f1_mean": 0.3500,
    "macro_f1_std": 0.0217,
    "balanced_accuracy_mean": 0.3809,
    "weighted_f1_mean": 0.4098,
    "f1_formal_mean": 0.1206,
    "f1_formal_std": 0.0487,
    "f1_personal_mean": 0.4116,
    "f1_informal_mean": 0.4266,
    "f1_did_nothing_mean": 0.4413,
    "accuracy_mean": 0.3880,
    "protocol": "repeated_stratified_5x5_cv_seed_42",
    "source": "script_20 logistic_class_weighted_exclude",
}

LIMITATIONS = (
    "Modest discrimination (macro F1 ≈ 0.35 under repeated CV). "
    "Formal support is sparse (≈15 per test fold); formal precision ≈ 0.07 and recall ≈ 0.31. "
    "Fold standard deviations are not confidence intervals. "
    "Not for operational targeting or eligibility decisions. "
    "Serving artefact is refit on the full drought-exposed sample (n=1,552); "
    "do not treat in-sample scores as validation metrics."
)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _hash_ids(ids: pd.Series) -> str:
    joined = ",".join(sorted(ids.astype(str).tolist()))
    return hashlib.sha256(joined.encode()).hexdigest()


def _hash_list(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def load_county_categories() -> list[tuple[int, str]]:
    onehot = pd.read_csv(TABLES / "09_onehot_category_dictionary.csv")
    county = onehot.loc[onehot["source_variable"] == "county"].copy()
    rows: list[tuple[int, str]] = []
    for _, r in county.iterrows():
        rows.append((int(r["code"]), str(r["label"])))
    rows.sort(key=lambda x: x[0])
    return rows


def build_feature_schema(feature_names: list[str], county_categories: list[tuple[int, str]]) -> dict:
    features = []
    for name in feature_names:
        if name in CONTINUOUS_FEATURES:
            ftype = "continuous"
            group = name
        elif name.startswith("county_"):
            ftype = "one_hot"
            group = "county"
        elif name in ("sex_bin", "urban_rural_bin") or name in AGRICULTURAL_HAS_VARS or name in LAND_OWNERSHIP_VARS:
            ftype = "binary"
            group = name.split("_")[0] if name.startswith("has_") else name
        else:
            ftype = "binary"
            group = name
        features.append(
            {
                "name": name,
                "type": ftype,
                "group": group,
                "required_in_model_matrix": True,
            }
        )

    domain_inputs = [
        {
            "name": "age_years",
            "type": "continuous",
            "required": True,
            "label_key": "field.age_years",
            "helper_key": "help.age_years",
        },
        {
            "name": "sex",
            "type": "categorical_code",
            "required": True,
            "allowed_codes": [
                {"code": 1, "label": "Male"},
                {"code": 2, "label": "Female"},
            ],
            "maps_to": "sex_bin",
            "label_key": "field.sex",
        },
        {
            "name": "education",
            "type": "ordinal",
            "required": True,
            "allowed_codes": [
                {"code": 1, "label": "None / informal"},
                {"code": 2, "label": "Primary"},
                {"code": 3, "label": "Secondary"},
                {"code": 4, "label": "Tertiary"},
            ],
            "label_key": "field.education",
        },
        {
            "name": "urban_rural",
            "type": "categorical_code",
            "required": True,
            "allowed_codes": [
                {"code": 1, "label": "Rural"},
                {"code": 2, "label": "Urban"},
            ],
            "maps_to": "urban_rural_bin",
            "label_key": "field.urban_rural",
        },
        {
            "name": "county",
            "type": "categorical_code",
            "required": True,
            "allowed_codes": [
                {"code": c, "label": lab, "one_hot_column": f"county_{c}_{lab.lower().replace(' ', '_').replace('/', '_')}"}
                for c, lab in county_categories
            ],
            "one_hot_group": "county_*",
            "label_key": "field.county",
        },
    ]
    for col in AGRICULTURAL_HAS_VARS + LAND_OWNERSHIP_VARS:
        domain_inputs.append(
            {
                "name": col,
                "type": "binary",
                "required": True,
                "allowed_codes": [{"code": 0, "label": "No"}, {"code": 1, "label": "Yes"}],
                "label_key": f"field.{col}",
            }
        )
    for col in PORTFOLIO_COUNT_VARS:
        domain_inputs.append(
            {
                "name": col,
                "type": "continuous",
                "required": True,
                "label_key": f"field.{col}",
            }
        )

    # Fix county one_hot_column names to match actual feature names
    county_cols = {int(c): name for name in feature_names if name.startswith("county_") for c in [int(name.split("_")[1])]}
    for item in domain_inputs:
        if item["name"] == "county":
            for ac in item["allowed_codes"]:
                ac["one_hot_column"] = county_cols[ac["code"]]

    return {
        "version": "v0.1.0",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "features": features,
        "domain_inputs": domain_inputs,
        "excluded": ["land_size_acres"],
        "missing_rules": {
            "land_size_acres": "Not collected for classifier v0.1.0 (acres-excluded specification).",
            "model_matrix": "All 68 model features required after encoding; no imputation in serving pipeline.",
        },
        "continuous_scaled": CONTINUOUS_FEATURES,
    }


def fit_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    continuous = [c for c in CONTINUOUS_FEATURES if c in X.columns]
    passthrough = [c for c in X.columns if c not in continuous]
    pre = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), continuous),
            ("pass", "passthrough", passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    pipe = Pipeline([("preprocess", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


def maybe_log_mlflow(metadata: dict, model_path: Path, req_path: Path | None) -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("mlflow not installed; skipping MLflow logging")
        return

    tracking_dir = ROOT / "artefacts" / "mlflow"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # Local file-store deprecated in MLflow 3; use sqlite for prototype registry
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    mlflow.set_experiment("coping-classifier")
    with mlflow.start_run(run_name="export_v0.1.0") as run:
        mlflow.log_params(
            {
                "algorithm": metadata["algorithm"],
                "predictor_spec_id": metadata["predictor_spec_id"],
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "random_state": 42,
                "class_weight": "balanced",
                "n_features": metadata["n_features"],
                "train_n": metadata["train_n"],
            }
        )
        for k, v in CV_METRICS.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_artifact(str(model_path))
        if req_path and req_path.exists():
            mlflow.log_artifact(str(req_path))
        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/model.joblib"
        try:
            # Register via artefact path annotation; local file store prototype
            mlflow.set_tag("registered_name", "coping-classifier")
            mlflow.set_tag("registered_version", "v0.1.0")
            mlflow.set_tag("alias", "champion")
            mlflow.set_tag("artefact_path", str(model_path))
            mlflow.set_tag("model_uri_note", model_uri)
        except Exception as exc:
            print(f"MLflow register note skipped: {exc}")
        print(f"MLflow run: {run.info.run_id}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)

    county_categories = load_county_categories()
    feature_names = primary_feature_names(county_categories)

    fl = pd.read_csv(TABLES / "09_feature_list_primary_classification.csv")
    expected = [c for c in fl["column"] if c != "land_size_acres"]
    if feature_names != expected:
        # Prefer locked feature list order from workbench
        feature_names = expected

    X_full = pd.read_parquet(PREPARED / "09_X_primary_classification.parquet")
    y_df = pd.read_parquet(PREPARED / "09_y_targets.parquet")

    if "respondent_id" in X_full.columns:
        respondent_ids = X_full["respondent_id"]
        X = X_full[feature_names].copy()
    else:
        # IDs may live alongside targets
        analysis = pd.read_parquet(PREPARED / "09_prepared_analysis_frame.parquet")
        respondent_ids = analysis["respondent_id"]
        X = X_full[feature_names].copy()

    y = y_df["coping_category"].astype(str)
    assert len(X) == len(y) == 1552

    # Ensure class order for consistent probability columns
    classes_present = sorted(y.unique().tolist())
    assert set(classes_present) == set(CLASS_ORDER)

    pipe = fit_pipeline(X, y)

    import sklearn

    model_path = OUT / "model.joblib"
    joblib.dump(pipe, model_path)

    schema = build_feature_schema(feature_names, county_categories)
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    class_mapping = {
        "classes": CLASS_ORDER,
        "display_labels": DISPLAY_LABELS,
        "category_colours": {
            "personal": "#2F5D50",
            "informal": "#C4A574",
            "formal": "#3D5A80",
            "did_nothing": "#7A6F64",
        },
        "estimator_classes": list(pipe.named_steps["clf"].classes_),
    }
    (OUT / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2))

    pkg = {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }

    metadata = {
        "name": "coping-classifier",
        "version": "v0.1.0",
        "algorithm": "multinomial_logistic_regression",
        "predictor_spec_id": "primary_acres_excluded_v1",
        "hyperparameters": {
            "class_weight": "balanced",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 42,
        },
        "seed": 42,
        "train_n": int(len(X)),
        "n_features": len(feature_names),
        "dataset_hash": _hash_ids(respondent_ids),
        "feature_list_hash": _hash_list(feature_names),
        "package_versions": pkg,
        "cv_metrics": CV_METRICS,
        "limitations": LIMITATIONS,
        "low_confidence_rule": {
            "description": "Flag when score margin (max - second_max) < 0.10 OR predicted class is formal",
            "margin_threshold": 0.10,
            "always_flag_classes": ["formal"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    manifest = {
        "train_n": int(len(X)),
        "respondent_id_hash": metadata["dataset_hash"],
        "feature_list_hash": metadata["feature_list_hash"],
        "feature_names": feature_names,
        "class_counts": y.value_counts().to_dict(),
        "acres_excluded": True,
    }
    (OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2))

    ACTIVE.write_text(
        json.dumps(
            {
                "alias": "champion",
                "classifier_version": "v0.1.0",
                "path": "artefacts/classifier/v0.1.0",
                "updated_at": metadata["created_at"],
            },
            indent=2,
        )
    )

    # Smoke predict
    proba = pipe.predict_proba(X.iloc[:1])
    pred = pipe.predict(X.iloc[:1])[0]
    print(f"Wrote {OUT}")
    print(f"Smoke: pred={pred} proba_shape={proba.shape} classes={pipe.named_steps['clf'].classes_}")

    req_freeze = ROOT / "artefacts" / "classifier" / "v0.1.0" / "requirements_freeze.txt"
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], cwd=ROOT
        ).decode()
        req_freeze.write_text(freeze)
    except Exception:
        req_freeze = None

    maybe_log_mlflow(metadata, model_path, req_freeze)


if __name__ == "__main__":
    main()
