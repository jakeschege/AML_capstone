"""FastAPI service for FinAccess coping prototype."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .dashboard import build_dashboard
from .encoding import encode_household, encode_segmentation
from .model_store import ModelBundle, explain_frame, load_bundle, predict_frame
from .prediction_chat import router as prediction_chat_router
from .schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    HouseholdRequest,
    PredictResponse,
    SegmentRequest,
)

bundle: Optional[ModelBundle] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    try:
        bundle = load_bundle()
    except Exception as exc:
        bundle = None
        print(f"Model load failed: {exc}")
    yield


app = FastAPI(title="FinAccess Coping Prototype API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(prediction_chat_router)


def require_bundle() -> ModelBundle:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return bundle


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    b = require_bundle()
    return HealthResponse(status="ok", model_version=b.version, loaded_at=b.loaded_at)


@app.get("/api/v1/metadata")
def metadata():
    b = require_bundle()
    return {
        "model_version": b.version,
        "name": b.metadata.get("name"),
        "algorithm": b.metadata.get("algorithm"),
        "predictor_spec_id": b.metadata.get("predictor_spec_id"),
        "limitations": b.metadata.get("limitations"),
        "cv_metrics": b.metadata.get("cv_metrics"),
        "low_confidence_rule": b.metadata.get("low_confidence_rule"),
        "train_n": b.metadata.get("train_n"),
        "created_at": b.metadata.get("created_at"),
    }


@app.get("/api/v1/model-info")
def model_info():
    b = require_bundle()
    return {
        "model_version": b.version,
        "metrics": b.metadata.get("cv_metrics"),
        "class_mapping": b.mapping,
        "feature_schema": {
            "n_features": b.schema.get("n_features"),
            "feature_names": b.schema.get("feature_names"),
            "domain_inputs": b.schema.get("domain_inputs"),
            "excluded": b.schema.get("excluded"),
            "missing_rules": b.schema.get("missing_rules"),
        },
        "global_importance": b.global_importance[:20],
        "segmentation_available": b.segmenter is not None,
    }


@app.get("/api/v1/dashboard")
def dashboard(shock: Optional[str] = None):
    """Weighted overview aggregates (precomputed all-shocks slices). Optional ?shock=."""
    try:
        return build_dashboard(shock=shock)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard data unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dashboard build failed: {exc}") from exc


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: HouseholdRequest):
    b = require_bundle()
    payload = req.model_dump()
    include_segment = payload.pop("include_segment", False)
    try:
        X = encode_household(payload, b.feature_names, b.county_categories)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = predict_frame(b, X)
    segment = None
    if include_segment and b.segmenter is not None:
        try:
            S = encode_segmentation(payload)
            segment = b.segmenter.assign(S)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["segment"] = segment
    return result


@app.post("/api/v1/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """Approximate grouped Shapley for Stack A; probability-only fallback on failure.

    /predict contract unchanged. Never explains Stack A with the logistic fallback.
    """
    b = require_bundle()
    payload = req.model_dump()
    top_k = payload.pop("top_k", 12)
    for_class = payload.pop("for_class", None)
    payload.pop("include_segment", None)
    # Stack A uses shap_background_frame; legacy v0.1 uses shap_explainer
    if b.shap_background_frame is None and b.shap_explainer is None:
        # Still return scores via predict path embedded in probability_only
        try:
            X = encode_household(payload, b.feature_names, b.county_categories)
            result = predict_frame(b, X)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result.update(
            {
                "top_features": [],
                "shap_note": "Attribution artefacts unavailable; model scores only.",
                "predicted_class_for_shap": result["prediction"],
                "explanation_mode": "probability_only",
                "segment": None,
            }
        )
        return result
    try:
        X = encode_household(payload, b.feature_names, b.county_categories)
        result = explain_frame(
            b,
            X,
            top_k=top_k,
            payload_for_cache=payload,
            for_class=for_class,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        try:
            X = encode_household(payload, b.feature_names, b.county_categories)
            result = predict_frame(b, X)
            result.update(
                {
                    "top_features": [],
                    "shap_note": f"Attribution failed ({type(exc).__name__}); model scores only.",
                    "predicted_class_for_shap": result["prediction"],
                    "explanation_mode": "probability_only",
                }
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve)) from ve
    result["segment"] = None
    # Drop internal-only keys not in ExplainResponse
    for key in ("elapsed_seconds", "group_names", "n_perms", "seed", "attribution_raw"):
        result.pop(key, None)
    return result


@app.post("/api/v1/segment")
def segment(req: SegmentRequest):
    b = require_bundle()
    if b.segmenter is None:
        return {
            "assigned": False,
            "profiles": [],
            "note": "Segmentation artefacts not available.",
        }
    try:
        S = encode_segmentation(req.model_dump())
        out = b.segmenter.assign(S)
        out["profiles"] = b.segmenter.profiles()
        out["note"] = b.segmenter.metadata.get("note")
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
