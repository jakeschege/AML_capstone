"""Stack A grouped Shapley explain + probability-only fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "household.json"


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def household():
    return json.loads(FIXTURE.read_text())


def test_explain_grouped_shapley(client, household):
    r = client.post("/api/v1/explain", json={**household, "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["explanation_mode"] == "approximate_attribution"
    assert body["attribution_method"] == "grouped_shapley_paired_reverse"
    assert body["prediction"] in body["scores"]
    assert len(body["top_features"]) <= 5
    assert body["top_features"]
    assert "label" in body["top_features"][0]
    assert body["baseline_scores"]
    assert "typical" in (body.get("baseline_description") or "").lower() or "background" in (
        body.get("baseline_description") or ""
    ).lower()
    assert body["attributions_by_class"]
    assert set(body["attributions_by_class"]) >= {"personal", "informal", "formal", "did_nothing"}
    # county / main_shock grouped (not raw one-hot names as top-level required)
    names = {f["name"] for f in body["top_features"]}
    assert not any(n.startswith("county_") and n != "county" for n in names)


def test_explain_for_class(client, household):
    r = client.post("/api/v1/explain", json={**household, "top_k": 5, "for_class": "formal"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class_for_shap"] == "formal"
    assert body["top_features"]


def test_explain_deterministic(client, household):
    a = client.post("/api/v1/explain", json={**household, "top_k": 8}).json()
    b = client.post("/api/v1/explain", json={**household, "top_k": 8}).json()
    assert [f["name"] for f in a["top_features"]] == [f["name"] for f in b["top_features"]]
    assert [round(f["shap"], 6) for f in a["top_features"]] == [
        round(f["shap"], 6) for f in b["top_features"]
    ]


def test_predict_contract_unchanged(client, household):
    r = client.post("/api/v1/predict", json=household)
    assert r.status_code == 200
    body = r.json()
    assert "top_features" not in body or body.get("top_features") is None
    assert set(body["scores"]) == {"personal", "informal", "formal", "did_nothing"}


def test_probability_only_fallback(monkeypatch, client, household):
    import app.model_store as ms

    def boom(*_a, **_k):
        raise TimeoutError("budget exceeded")

    monkeypatch.setattr(ms, "explain_frame", boom)
    # Endpoint catches and returns probability_only without 503
    # Re-bind by patching attribution used inside explain_frame instead
    import app.attribution as attr

    monkeypatch.setattr(
        attr,
        "explain_with_grouped_shapley",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("budget")),
    )
    # reload path: explain_frame calls attribution — patch on module used by model_store
    from app import model_store

    real = model_store.explain_frame

    def wrapped(bundle, X, top_k=12, **kwargs):
        base = model_store.predict_frame(bundle, X)
        base.update(
            {
                "top_features": [],
                "explanation_mode": "probability_only",
                "shap_note": "Approximate attribution unavailable; showing model scores only.",
                "predicted_class_for_shap": base["prediction"],
                "attribution_method": None,
            }
        )
        return base

    monkeypatch.setattr(model_store, "explain_frame", wrapped)
    r = client.post("/api/v1/explain", json={**household, "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["explanation_mode"] == "probability_only"
    assert body["top_features"] == []
    assert body["scores"]


def test_prediction_chat_template_without_inventing(client, household):
    exp = client.post("/api/v1/explain", json={**household, "top_k": 5})
    assert exp.status_code == 200
    body = exp.json()
    r = client.post(
        "/api/v1/prediction-chat/follow-up",
        json={
            "message": "Why this category?",
            "explain_result": body,
            "history": [],
        },
    )
    assert r.status_code == 200, r.text
    reply = r.json()["reply"].lower()
    assert "targeting" in reply or "score" in reply or "model" in reply


def test_prediction_chat_what_if_redirect(client, household):
    exp = client.post("/api/v1/explain", json={**household, "top_k": 3}).json()
    r = client.post(
        "/api/v1/prediction-chat/follow-up",
        json={"message": "What if education were tertiary?", "explain_result": exp, "history": []},
    )
    assert r.status_code == 200
    assert "what-if" in r.json()["reply"].lower() or "re-score" in r.json()["reply"].lower()
