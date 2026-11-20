"""Tests for shared LLM / knowledge-base helpers (Change 1: assistant page removed)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_assistant_chat_endpoint_removed():
    from app.main import app

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/assistant/chat",
            json={"message": "How many households had no coping action?", "history": []},
        )
        assert r.status_code == 404


def test_kb_retrieve_returns_chunks():
    from app.llm import load_kb_chunks, retrieve

    chunks = load_kb_chunks()
    assert len(chunks) >= 1
    hits = retrieve("coping formal informal limitations targeting")
    assert len(hits) >= 1
    assert all(isinstance(name, str) and isinstance(text, str) for name, text in hits)


def test_llm_configured_false_without_key(monkeypatch):
    from app import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert llm.llm_configured() is False


def test_extract_assistant_text_prefers_content_over_reasoning():
    from app.llm import _extract_assistant_text

    text = _extract_assistant_text(
        {"content": "Visible answer 0.42", "reasoning": "private chain that invents 9.99"}
    )
    assert text == "Visible answer 0.42"


def test_extract_assistant_text_rejects_reasoning_only():
    from app.llm import _extract_assistant_text
    import pytest

    with pytest.raises(RuntimeError, match="empty assistant content"):
        _extract_assistant_text({"content": "", "reasoning": "only reasoning"})


def test_detect_comparison_and_why_not():
    from app.prediction_chat import detect_comparison_query, build_comparison_facts

    assert detect_comparison_query("Why not formal support?") == {
        "kind": "why_not",
        "focus_class": "formal",
    }
    assert detect_comparison_query("Is formal more likely than personal here?") == {
        "kind": "compare",
        "class_a": "formal",
        "class_b": "personal",
    }
    result = {
        "prediction": "informal",
        "scores": {"personal": 0.2, "informal": 0.56, "formal": 0.15, "did_nothing": 0.09},
        "attributions_by_class": {
            "formal": [
                {
                    "name": "main_shock_type",
                    "label": "Main reported shock",
                    "direction": "increases",
                    "magnitude_band": "moderate",
                }
            ]
        },
        "baseline_scores": {"personal": 0.36, "informal": 0.23, "formal": 0.19, "did_nothing": 0.22},
    }
    facts, nums, names = build_comparison_facts(
        result, {"kind": "why_not", "focus_class": "formal"}
    )
    assert facts["score_gap_predicted_minus_compared"] == 0.41
    assert "0.41" in nums
    assert facts["do_not_calculate"] is True
    assert facts["attributions_for_compared_class"][0]["label"] == "Main reported shock"
