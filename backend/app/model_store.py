"""Load active classifier + SHAP + segmentation artefacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from .encoding import encode_household, load_county_categories, primary_feature_names
from .segment import Segmenter


@dataclass
class ModelBundle:
    root: Path
    version: str
    pipeline: object
    schema: dict
    mapping: dict
    metadata: dict
    feature_names: list[str]
    county_categories: list[tuple[int, str]]
    loaded_at: str
    shap_background: np.ndarray | None = None
    shap_background_frame: object | None = None  # pd.DataFrame for Stack A grouped Shapley
    shap_explainer: object | None = None
    shap_feature_names: list[str] = field(default_factory=list)
    segmenter: Segmenter | None = None
    global_importance: list[dict] = field(default_factory=list)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_active(root: Path | None = None) -> tuple[Path, dict]:
    root = root or project_root()
    active_path = root / "artefacts" / "metadata" / "ACTIVE_MODEL.json"
    active = json.loads(active_path.read_text())
    clf_dir = root / active["path"]
    return clf_dir, active


def load_bundle(root: Path | None = None) -> ModelBundle:
    root = root or project_root()
    clf_dir, active = resolve_active(root)
    pipeline = joblib.load(clf_dir / "model.joblib")
    schema = json.loads((clf_dir / "feature_schema.json").read_text())
    mapping = json.loads((clf_dir / "class_mapping.json").read_text())
    metadata = json.loads((clf_dir / "metadata.json").read_text())
    feature_names = schema["feature_names"]

    # County categories from schema domain inputs
    county_item = next(d for d in schema["domain_inputs"] if d["name"] == "county")
    county_categories = [(int(c["code"]), str(c["label"])) for c in county_item["allowed_codes"]]

    bundle = ModelBundle(
        root=root,
        version=active.get("classifier_version", metadata["version"]),
        pipeline=pipeline,
        schema=schema,
        mapping=mapping,
        metadata=metadata,
        feature_names=feature_names,
        county_categories=county_categories,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )

    # Stack A: grouped Shapley background (question-level approximate attributions)
    is_stack = pipeline.__class__.__name__ == "StackingClassifier" or not (
        hasattr(pipeline, "named_steps") and "clf" in getattr(pipeline, "named_steps", {})
    )
    if is_stack and len(feature_names) == 93:
        from .attribution import load_background

        bg = load_background(root, feature_names)
        if bg is not None:
            bundle.shap_background_frame = bg
            bundle.shap_feature_names = feature_names

    # Legacy LinearExplainer only for linear Pipeline classifiers (v0.1, 68 features)
    shap_dir = root / "artefacts" / "shap" / "v0.1.0"
    if (
        not is_stack
        and shap_dir.exists()
        and (shap_dir / "shap_values.npz").exists()
        and len(feature_names) == 68
    ):
        npz = np.load(shap_dir / "shap_values.npz", allow_pickle=True)
        bg_idx = npz["background_indices"]
        X = pd.read_parquet(
            root / "finaccess_coping_workbench" / "outputs" / "prepared" / "09_X_primary_classification.parquet"
        )[feature_names]
        Xt_bg = pipeline[:-1].transform(X.iloc[bg_idx])
        bundle.shap_background = Xt_bg
        bundle.shap_explainer = shap.LinearExplainer(pipeline.named_steps["clf"], Xt_bg)
        try:
            bundle.shap_feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        except Exception:
            bundle.shap_feature_names = feature_names
        gi = shap_dir / "global_importance_aggregated.csv"
        if gi.exists():
            bundle.global_importance = pd.read_csv(gi).to_dict(orient="records")

    seg_dir = root / "artefacts" / "segmentation" / "v0.1.0"
    if (seg_dir / "spec_a_k5_medoids.parquet").exists():
        bundle.segmenter = Segmenter(seg_dir)

    return bundle


def _model_classes(pipe) -> list:
    """Support Pipeline(... clf=...) and top-level StackingClassifier."""
    if hasattr(pipe, "named_steps") and "clf" in getattr(pipe, "named_steps", {}):
        return list(pipe.named_steps["clf"].classes_)
    if hasattr(pipe, "classes_"):
        return list(pipe.classes_)
    if hasattr(pipe, "final_estimator_") and hasattr(pipe.final_estimator_, "classes_"):
        return list(pipe.final_estimator_.classes_)
    raise AttributeError("Cannot resolve class labels from loaded model")


def predict_frame(bundle: ModelBundle, X: pd.DataFrame) -> dict:
    pipe = bundle.pipeline
    classes = _model_classes(pipe)
    proba = pipe.predict_proba(X)[0]
    pred = pipe.predict(X)[0]
    scores = {str(c): float(p) for c, p in zip(classes, proba)}
    # Align display order
    ordered = {c: scores.get(c, 0.0) for c in bundle.mapping["classes"]}
    sorted_scores = sorted(ordered.values(), reverse=True)
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0
    rule = bundle.metadata.get("low_confidence_rule", {})
    threshold = float(rule.get("margin_threshold", 0.10))
    always_flag = set(rule.get("always_flag_classes", ["formal"]))
    low_confidence = margin < threshold or str(pred) in always_flag
    warnings = []
    if low_confidence:
        warnings.append(
            "Low confidence: small score margin and/or predicted formal class (known weak class)."
        )
    warnings.append(bundle.metadata.get("limitations", ""))
    return {
        "prediction": str(pred),
        "display_label": bundle.mapping["display_labels"].get(str(pred), str(pred)),
        "scores": ordered,
        "model_version": bundle.version,
        "low_confidence": low_confidence,
        "score_margin": margin,
        "warnings": [w for w in warnings if w],
    }


def explain_frame(
    bundle: ModelBundle,
    X: pd.DataFrame,
    top_k: int = 12,
    *,
    payload_for_cache: dict | None = None,
    for_class: str | None = None,
) -> dict:
    """Explain Stack A via grouped Shapley, or legacy LinearExplainer for v0.1 pipelines."""
    base = predict_frame(bundle, X)

    # Stack A path — never use logistic fallback
    if bundle.shap_background_frame is not None:
        from .attribution import explain_with_grouped_shapley

        cache_payload = payload_for_cache or X.iloc[0].astype(float).to_dict()
        try:
            attr = explain_with_grouped_shapley(
                bundle.pipeline,
                bundle.feature_names,
                X,
                bundle.shap_background_frame,
                cache_payload,
                top_k=top_k,
                for_class=for_class,
            )
        except Exception as exc:
            base["top_features"] = []
            base["toward_features"] = []
            base["against_features"] = []
            base["attributions_by_class"] = {}
            base["baseline_scores"] = {}
            base["explanation_mode"] = "probability_only"
            base["attribution_method"] = None
            base["shap_note"] = (
                "Approximate attribution unavailable; showing model scores only. "
                f"({type(exc).__name__})"
            )
            base["predicted_class_for_shap"] = base["prediction"]
            return base

        base.update(
            {
                "top_features": attr["top_features"],
                "toward_features": attr.get("toward_features", []),
                "against_features": attr.get("against_features", []),
                "attributions_by_class": attr.get("attributions_by_class", {}),
                "baseline_scores": attr.get("baseline_scores", {}),
                "baseline_description": attr.get("baseline_description"),
                "explanation_mode": "approximate_attribution",
                "attribution_method": attr.get("attribution_method"),
                "shap_note": attr["shap_note"],
                "predicted_class_for_shap": attr["predicted_class_for_shap"],
                "additivity_abs_error": attr.get("additivity_abs_error"),
                "elapsed_seconds": attr.get("elapsed_seconds"),
            }
        )
        return base

    if bundle.shap_explainer is None:
        raise RuntimeError("SHAP explainer not loaded")

    Xt = bundle.pipeline[:-1].transform(X)
    sv = bundle.shap_explainer.shap_values(Xt)
    classes = list(bundle.pipeline.named_steps["clf"].classes_)
    pred = bundle.pipeline.predict(X)[0]
    class_j = list(classes).index(pred)

    if isinstance(sv, list):
        contrib = np.asarray(sv[class_j][0])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            if arr.shape[1] == len(bundle.shap_feature_names):
                contrib = arr[0, :, class_j]
            else:
                contrib = arr[0, class_j, :]
        else:
            contrib = arr[0]

    names = bundle.shap_feature_names
    order = np.argsort(-np.abs(contrib))[:top_k]
    top_features = []
    for i in order:
        val = float(contrib[i])
        top_features.append(
            {
                "name": names[i],
                "shap": val,
                "direction": "increases" if val >= 0 else "decreases",
            }
        )
    base["top_features"] = top_features
    base["shap_note"] = (
        "Local SHAP attributions for the predicted class on transformed features. "
        "Attribution is not causation."
    )
    base["predicted_class_for_shap"] = str(pred)
    base["explanation_mode"] = "legacy_linear"
    return base
