"""API smoke tests for classifier v0.2.0 (Stack A)."""

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


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v0.2.0"


def test_metadata_and_model_info(client):
    m = client.get("/api/v1/metadata")
    assert m.status_code == 200
    assert "limitations" in m.json()
    info = client.get("/api/v1/model-info")
    assert info.status_code == 200
    schema = info.json()["feature_schema"]
    assert schema["n_features"] == 93
    assert "land_size_acres" in schema["excluded"]


def test_predict_smoke(client, household):
    r = client.post("/api/v1/predict", json=household)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"personal", "informal", "formal", "did_nothing"}
    assert set(body["scores"]) == {"personal", "informal", "formal", "did_nothing"}
    assert body["model_version"] == "v0.2.0"
    assert isinstance(body["warnings"], list)


def test_reject_unknown_fields(client, household):
    bad = {**household, "land_size_acres": 2.5}
    r = client.post("/api/v1/predict", json=bad)
    assert r.status_code == 422


def test_explain_stack_a_grouped_shapley(client, household):
    """Stack A serves approximate grouped Shapley (not LinearExplainer)."""
    r = client.post("/api/v1/explain", json={**household, "top_k": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["explanation_mode"] in {"approximate_attribution", "probability_only"}
    if body["explanation_mode"] == "approximate_attribution":
        assert body["attribution_method"] == "grouped_shapley_paired_reverse"
        assert body["top_features"]


def test_segment(client, household):
    seg_payload = {
        k: household[k]
        for k in household
        if k.startswith("has_") or k.startswith("n_") or k.startswith("land_")
    }
    r = client.post("/api/v1/segment", json=seg_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["assigned"] is True
    assert body["cluster_id"] in range(5)
