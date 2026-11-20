"""Prediction follow-up chat: LLM may only paraphrase supplied scores/attributions."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from .attribution import (
    allowed_tokens_from_explain,
    grounded_reply_ok,
    template_explanation,
)
from .llm import TARGETING_PATTERNS, call_chat_completion, llm_configured

router = APIRouter(prefix="/api/v1/prediction-chat", tags=["prediction-chat"])

CLASS_IDS = ("personal", "informal", "formal", "did_nothing")
CLASS_PAT = r"(personal|informal|formal|did[_\s-]?nothing|no coping(?: action)?)"


class PredictionChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(..., min_length=1)
    explain_result: dict[str, Any]
    history: list[dict[str, str]] = Field(default_factory=list)


def _norm_class(raw: str) -> str | None:
    s = raw.lower().strip().replace("-", "_").replace(" ", "_")
    s = s.replace("did_nothing", "did_nothing")
    if s in {"did_nothing", "didnothing", "no_coping", "no_coping_action"}:
        return "did_nothing"
    if s in CLASS_IDS:
        return s
    return None


def detect_comparison_query(message: str) -> dict[str, str] | None:
    """Detect why-not / pairwise comparison questions."""
    q = message.lower().strip()
    m = re.search(rf"why\s+not\s+{CLASS_PAT}", q)
    if m:
        other = _norm_class(m.group(1))
        if other:
            return {"kind": "why_not", "focus_class": other}
    m = re.search(
        rf"(?:is|was)\s+{CLASS_PAT}\s+more\s+likely\s+than\s+{CLASS_PAT}",
        q,
    )
    if m:
        a, b = _norm_class(m.group(1)), _norm_class(m.group(2))
        if a and b:
            return {"kind": "compare", "class_a": a, "class_b": b}
    m = re.search(rf"compar(?:e|ison)\s+{CLASS_PAT}\s+(?:vs\.?|versus|and|with)\s+{CLASS_PAT}", q)
    if m:
        a, b = _norm_class(m.group(1)), _norm_class(m.group(2))
        if a and b:
            return {"kind": "compare", "class_a": a, "class_b": b}
    return None


def _round_score_map(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            out[str(k)] = round(float(v), 2)
        except (TypeError, ValueError):
            continue
    return out


def _slim_feats(feats: Any, limit: int = 5) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    if not isinstance(feats, list):
        return slim
    for f in feats[:limit]:
        slim.append(
            {
                "label": f.get("label") or f.get("name"),
                "direction": f.get("direction"),
                "magnitude_band": f.get("magnitude_band"),
            }
        )
    return slim


def _attrs_for_class(result: dict[str, Any], cls: str) -> list[dict[str, Any]]:
    by = result.get("attributions_by_class") or {}
    if cls in by:
        return _slim_feats(by[cls])
    if result.get("predicted_class_for_shap") == cls or result.get("prediction") == cls:
        return _slim_feats(result.get("top_features"))
    return []


def build_comparison_facts(
    result: dict[str, Any], spec: dict[str, str]
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Scores for both classes, attributions for focus class(es), precomputed gap."""
    scores = _round_score_map(result.get("scores"))
    pred = str(result.get("prediction") or "")
    extra_nums: set[str] = set()
    extra_names: set[str] = set()

    if spec["kind"] == "why_not":
        focus = spec["focus_class"]
        other = focus
        left, right = pred, other
        score_left = scores.get(left)
        score_right = scores.get(right)
        gap = None
        if score_left is not None and score_right is not None:
            gap = round(float(score_left) - float(score_right), 2)
            extra_nums.add(f"{gap:.2f}")
            extra_nums.add(f"{abs(gap):.2f}")
        facts = {
            "question_kind": "why_not",
            "predicted_class": pred,
            "compared_class": other,
            "score_predicted": score_left,
            "score_compared": score_right,
            "score_gap_predicted_minus_compared": gap,
            "do_not_calculate": True,
            "instruction": (
                "Use only these supplied scores and the precomputed gap. "
                "Do not subtract, rank, or invent any numbers yourself."
            ),
            "attributions_for_compared_class": _attrs_for_class(result, other),
            "baseline_scores": _round_score_map(result.get("baseline_scores")),
            "baseline_description": result.get("baseline_description"),
            "explanation_mode": result.get("explanation_mode"),
            "shap_note": result.get("shap_note"),
        }
        extra_names.update({pred, other, "predicted", "compared", "gap"})
        for f in facts["attributions_for_compared_class"]:
            if f.get("label"):
                extra_names.add(str(f["label"]))
            if f.get("magnitude_band"):
                extra_names.add(str(f["magnitude_band"]))
        return facts, extra_nums, extra_names

    # compare
    a, b = spec["class_a"], spec["class_b"]
    sa, sb = scores.get(a), scores.get(b)
    gap = None
    higher = None
    if sa is not None and sb is not None:
        gap = round(float(sa) - float(sb), 2)
        extra_nums.add(f"{gap:.2f}")
        extra_nums.add(f"{abs(gap):.2f}")
        if gap > 0:
            higher = a
        elif gap < 0:
            higher = b
        else:
            higher = "tie"
    facts = {
        "question_kind": "compare",
        "class_a": a,
        "class_b": b,
        "score_a": sa,
        "score_b": sb,
        "score_gap_a_minus_b": gap,
        "higher_score_class": higher,
        "predicted_class": pred,
        "do_not_calculate": True,
        "instruction": (
            "Use only these supplied scores and the precomputed gap. "
            "Do not subtract, rank, or invent any numbers yourself."
        ),
        "attributions_for_class_a": _attrs_for_class(result, a),
        "attributions_for_class_b": _attrs_for_class(result, b),
        "baseline_scores": _round_score_map(result.get("baseline_scores")),
        "baseline_description": result.get("baseline_description"),
        "explanation_mode": result.get("explanation_mode"),
        "shap_note": result.get("shap_note"),
    }
    extra_names.update({a, b, pred, "gap", "tie"})
    for key in ("attributions_for_class_a", "attributions_for_class_b"):
        for f in facts[key]:
            if f.get("label"):
                extra_names.add(str(f["label"]))
            if f.get("magnitude_band"):
                extra_names.add(str(f["magnitude_band"]))
    return facts, extra_nums, extra_names


@router.post("/follow-up")
def follow_up(req: PredictionChatRequest):
    """Answer follow-ups from stored explain/predict JSON. Never invents numbers."""
    result = req.explain_result
    scores = result.get("scores") or {}
    prediction = str(result.get("prediction") or "")
    display = str(result.get("display_label") or prediction)
    mode = str(result.get("explanation_mode") or "probability_only")
    top = result.get("top_features") or []

    if TARGETING_PATTERNS.search(req.message):
        text = (
            "I cannot recommend targeting or eligibility. "
            "I can only describe this model’s scores and approximate drivers for the "
            "household profile you entered. See the Overview for survey group patterns."
        )
        return {"reply": text, "source": "policy_refusal", "used_llm": False}

    template = template_explanation(prediction, display, scores, top, mode=mode)

    # Counterfactual / what-if: instruct client to re-run predict — do not guess
    q = req.message.lower()
    if any(p in q for p in ("what if", "what would", "if i change", "if we change", "different answer")):
        text = (
            "For a what-if question, change the answer in the conversation and re-score "
            "with the same prediction endpoint. I will not guess how the score would move. "
            "What-if results show how the model responds to a changed answer, not what "
            "would happen to the household. They are not advice to alter anyone’s situation."
        )
        return {"reply": text, "source": "what_if_redirect", "used_llm": False}

    # Off-topic → Overview
    on_topic = any(
        w in q
        for w in (
            "why",
            "score",
            "formal",
            "informal",
            "personal",
            "did_nothing",
            "did nothing",
            "feature",
            "driver",
            "attribut",
            "shap",
            "explain",
            "against",
            "toward",
            "baseline",
            "typical",
            "category",
            "class",
            "shock",
            "factor",
            "factors",
            "effect",
            "moderate",
            "small",
            "large",
            "among the main",
            "summar",
            "read these",
            "model",
            "more likely",
            "compar",
            "versus",
            " vs",
        )
    )
    if not on_topic:
        text = (
            "I can only discuss this prediction’s model scores and approximate drivers. "
            "For survey-wide patterns, open the Overview page."
        )
        return {"reply": text, "source": "redirect_overview", "used_llm": False}

    nums, names = allowed_tokens_from_explain(result)
    for feats in (result.get("attributions_by_class") or {}).values():
        for f in feats:
            names.add(str(f.get("label") or ""))
            names.add(str(f.get("name") or ""))

    if not llm_configured():
        return {"reply": template, "source": "template", "used_llm": False}

    comparison = detect_comparison_query(req.message)

    try:
        system = (
            "You are a careful research assistant in a chat. Write in short natural paragraphs "
            "(2–5 sentences), like a person speaking — not a report, not a bullet list, not a table, "
            "not markdown headings. "
            "Use only the JSON facts given. Call numbers model scores (not probabilities). "
            "The baseline is a typical household in the background sample. "
            "Attributions are approximate. Say 'among the main factors' — never claim a "
            "biggest, largest, top, or most important factor. Prefer magnitude bands "
            "(small / moderate / large) and direction over precise ranking. "
            "When you mention a score or gap, copy it exactly as given (two decimal places) — "
            "do not round further, approximate, subtract, or invent values. "
            "If Facts JSON includes score_gap_* or do_not_calculate=true, never calculate "
            "differences yourself — only restate the supplied gap. "
            "Do not invent numbers, features, or reasons. "
            "Do not advise the household to change anything. "
            "Put the answer only in the normal assistant content (not private reasoning). "
            "Close with one plain sentence: This describes what drove the model's score, not why "
            "the household actually coped the way it did, and it is not for targeting or "
            "eligibility decisions."
        )

        if comparison:
            facts, extra_nums, extra_names = build_comparison_facts(result, comparison)
            nums |= extra_nums
            names |= extra_names
        else:
            facts = {
                "prediction": result.get("prediction"),
                "display_label": result.get("display_label"),
                "scores": _round_score_map(result.get("scores")),
                "explanation_mode": result.get("explanation_mode"),
                "baseline_scores": _round_score_map(result.get("baseline_scores")),
                "baseline_description": result.get("baseline_description"),
                "predicted_class_for_shap": result.get("predicted_class_for_shap"),
                "shap_note": result.get("shap_note"),
            }
            for key in ("top_features", "toward_features", "against_features"):
                facts[key] = _slim_feats(result.get(key) or [])

        user = f"User question: {req.message}\n\nFacts JSON:\n{json.dumps(facts, default=str)}"
        reply = call_chat_completion(system=system, user=str(user), temperature=0.1, timeout=45.0)
        if not grounded_reply_ok(reply, nums, names):
            return {
                "reply": template,
                "source": "template_grounding_failed",
                "used_llm": True,
            }
        return {"reply": reply, "source": "llm", "used_llm": True}
    except Exception as exc:
        return {
            "reply": f"{template}\n\n(Language model unavailable: {type(exc).__name__}: {exc}.)",
            "source": "template_llm_error",
            "used_llm": False,
            "error_type": type(exc).__name__,
            "error_detail": str(exc)[:240],
        }
