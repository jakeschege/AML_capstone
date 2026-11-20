"""Dashboard aggregate smoke tests (Stage 3a precomputed payload)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_dashboard_all_shocks_default():
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/dashboard")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["selected_shock"] == "all"
        assert body["filter_definition"] == "r2a_any_shock"
        assert body["n_respondents"] == 5241
        assert body["affected_unweighted"] == 5241
        assert set(body["mix"]) == {"personal", "informal", "formal", "did_nothing"}
        assert abs(sum(body["mix"].values()) - 1.0) < 1e-3
        assert body["mix_unweighted"]["informal"] == 2094
        assert all(c["n"] >= 30 for c in body["counties"])
        assert "shock_options" in body and len(body["shock_options"]) == 13
        assert "coping_by_main_shock" in body
        assert len(body["coping_by_main_shock"]) == 12
        unavailable = [r for r in body["coping_by_main_shock"] if r["unavailable"]]
        assert {r["main_shock_code"] for r in unavailable} == {8, 10}
        assert "suppressed_cells" in body
        assert "represented by the survey" in body["weight_note"].lower() or "survey" in body[
            "weight_note"
        ].lower()
        for panel in body["who"].values():
            for row in panel:
                if row["suppressed"]:
                    assert row["none_share"] is None
                    assert row["n"] < 30


def test_dashboard_drought_slice_matches_legacy_totals():
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/dashboard", params={"shock": "shock_drought_loss"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["selected_shock"] == "shock_drought_loss"
        assert body["n_respondents"] == 1552
        assert body["affected_unweighted"] == 1552
        assert abs(body["affected_weighted"] - 985365.4) < 1.0
        # Main-shock drought row highlighted
        highlighted = [r for r in body["coping_by_main_shock"] if r.get("highlighted")]
        assert len(highlighted) == 1
        assert highlighted[0]["main_shock_code"] == 3


def test_dashboard_unknown_shock_400():
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/dashboard", params={"shock": "not_a_shock"})
        assert r.status_code == 400
