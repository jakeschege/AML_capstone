"""Approximate question-level attributions for Stack A (grouped Shapley).

Never uses the logistic fallback. Fixed seed for reproducibility.
Uses paired reversed permutations for more stable rankings.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ATTRIBUTION_SEED = 42
# 40 random orders × reverse pair = 80 paths; keeps typical explain under ~1s
# while improving top-5 stability vs unpaired sampling.
DEFAULT_N_ORDERS = 40
EXPLAIN_TIMEOUT_S = 2.5

_QUESTION_LABELS: dict[str, str] = {
    "age_years": "Age of household head",
    "sex_bin": "Sex of respondent",
    "education": "Highest education level",
    "urban_rural_bin": "Urban or rural",
    "county": "County",
    "main_shock_type": "Main reported shock",
    "has_cash_crops": "Grows cash crops",
    "has_cereals": "Grows cereals",
    "has_dairy_livestock": "Keeps dairy livestock",
    "has_horticulture": "Grows horticulture crops",
    "has_legumes": "Grows legumes",
    "has_other_agri": "Other crop or agri activity",
    "has_other_livestock": "Keeps other livestock",
    "has_poultry": "Keeps poultry",
    "has_roots_tubers": "Grows roots or tubers",
    "n_crop_types": "Number of crop types",
    "n_livestock_types": "Number of livestock types",
    "land_ownership_own_under_your_name": "Land owned under own name",
    "land_ownership_own_jointly_with_spouse": "Land owned jointly with spouse",
    "land_ownership_own_jointly_with_someone_else_other_than_spouse": "Land owned jointly with someone else",
    "land_ownership_lease_rent_you_dont_own": "Land leased or rented",
    "land_ownership_community_public_owned_e_g_pastoralists": "Community or public land",
    "land_ownership_family_owned_e_g_inherited_family_land": "Family-owned / inherited land",
    "bank_usage": "Uses a bank account",
    "sacco_usage": "Uses a SACCO",
    "mobile_money_usage": "Uses mobile money",
    "mobile_bank_usage": "Uses mobile banking",
    "savings_usage": "Uses any savings channel",
    "savings_formal": "Uses formal savings",
    "savings_informal": "Uses informal savings",
    "loan_usage": "Uses any loan / credit",
    "loan_formal": "Uses formal loans",
    "loan_informal": "Uses informal loans",
    "insurance_excl_nhif": "Has insurance (excluding NHIF)",
    "buyer_supplier_credit": "Uses buyer / supplier credit",
    "formal_agri_borrowing": "Uses formal agricultural borrowing",
}

_SHAP_NOTE = (
    "Approximate question-level attributions for Stack A (grouped Shapley with "
    "paired reversed permutations). They describe factors that moved the model’s "
    "score relative to a typical household in the background sample — not why the "
    "household actually coped the way it did. Rankings are approximate; nothing here "
    "is claimed as the single biggest factor. Not for targeting or eligibility decisions."
)

_cache_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}
_CACHE_MAX = 128


def magnitude_band(value: float) -> str:
    a = abs(float(value))
    if a < 0.02:
        return "small"
    if a < 0.08:
        return "moderate"
    return "large"


def question_groups(feature_names: list[str]) -> tuple[list[str], list[list[int]]]:
    groups: dict[str, list[str]] = {}
    for f in feature_names:
        if f.startswith("county_"):
            groups.setdefault("county", []).append(f)
        elif f.startswith("main_shock_"):
            groups.setdefault("main_shock_type", []).append(f)
        else:
            groups[f] = [f]
    names = list(groups.keys())
    idxs = [[feature_names.index(c) for c in groups[g]] for g in names]
    return names, idxs


def plain_question_label(group_name: str) -> str:
    return _QUESTION_LABELS.get(group_name, group_name.replace("_", " "))


def _cache_key(payload: dict[str, Any], n_orders: int, seed: int, paired: bool) -> str:
    body = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(f"{body}|{n_orders}|{seed}|paired={paired}".encode()).hexdigest()


def load_background(root: Path, feature_names: list[str]) -> pd.DataFrame | None:
    path = root / "artefacts" / "shap" / "v0.2.0" / "background_sample.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return None
    return df[feature_names]


def _feature_dict(name: str, val: float) -> dict[str, Any]:
    return {
        "name": name,
        "label": plain_question_label(name),
        "shap": round(float(val), 2),
        "magnitude_band": magnitude_band(val),
        "direction": "increases" if val >= 0 else "decreases",
    }


def grouped_shapley_all_classes(
    pipeline,
    feature_names: list[str],
    X_row: pd.DataFrame,
    X_bg: pd.DataFrame,
    *,
    n_orders: int = DEFAULT_N_ORDERS,
    seed: int = ATTRIBUTION_SEED,
    paired_reverse: bool = True,
) -> dict[str, Any]:
    """Batched grouped Shapley for every class; additive vs background-mean baseline."""
    group_names, group_col_idx = question_groups(feature_names)
    n_g = len(group_names)
    classes = list(pipeline.classes_)
    n_c = len(classes)

    bg_mean = X_bg.values.astype(float).mean(axis=0)
    xq = X_row[feature_names].values.astype(float).reshape(-1)

    def predict_rows(mat: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(pd.DataFrame(mat, columns=feature_names))

    f_x = predict_rows(xq.reshape(1, -1))[0]
    f_base = predict_rows(bg_mean.reshape(1, -1))[0]
    pred_idx = int(np.argmax(f_x))

    order_rng = np.random.default_rng(seed)
    orders: list[np.ndarray] = []
    for _ in range(n_orders):
        o = order_rng.permutation(n_g)
        orders.append(o)
        if paired_reverse:
            orders.append(o[::-1].copy())

    mats: list[np.ndarray] = []
    for order in orders:
        row = bg_mean.copy()
        for gi in order:
            row = row.copy()
            idxs = group_col_idx[int(gi)]
            row[idxs] = xq[idxs]
            mats.append(row)
    probs = predict_rows(np.vstack(mats))

    shap = np.zeros((n_c, n_g))
    cursor = 0
    for order in orders:
        prev = f_base.copy()
        for gi in order:
            cur = probs[cursor]
            cursor += 1
            shap[:, int(gi)] += cur - prev
            prev = cur
    shap /= len(orders)

    baseline_scores = {str(c): round(float(p), 2) for c, p in zip(classes, f_base)}
    scores = {str(c): round(float(p), 2) for c, p in zip(classes, f_x)}
    attributions_by_class: dict[str, list[dict[str, Any]]] = {}
    additivity: dict[str, float] = {}

    for j, cls in enumerate(classes):
        gap = float(f_x[j] - f_base[j])
        s = float(shap[j].sum())
        additivity[str(cls)] = abs(s - gap)
        feats = []
        order = np.argsort(-np.abs(shap[j]))
        for i in order:
            feats.append(_feature_dict(group_names[int(i)], float(shap[j, i])))
        attributions_by_class[str(cls)] = feats

    return {
        "attribution_method": "grouped_shapley_paired_reverse",
        "n_orders": n_orders,
        "n_paths": len(orders),
        "paired_reverse": paired_reverse,
        "seed": seed,
        "baseline_scores": baseline_scores,
        "scores": scores,
        "prediction": str(classes[pred_idx]),
        "attributions_by_class": attributions_by_class,
        "additivity_abs_error": additivity,
        "group_names": group_names,
    }


def explain_with_grouped_shapley(
    pipeline,
    feature_names: list[str],
    X_row: pd.DataFrame,
    X_bg: pd.DataFrame,
    payload_for_cache: dict[str, Any],
    *,
    top_k: int = 12,
    n_orders: int = DEFAULT_N_ORDERS,
    seed: int = ATTRIBUTION_SEED,
    for_class: str | None = None,
) -> dict[str, Any]:
    key = _cache_key(payload_for_cache, n_orders, seed, True)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is None:
        t0 = time.perf_counter()
        raw = grouped_shapley_all_classes(
            pipeline,
            feature_names,
            X_row,
            X_bg,
            n_orders=n_orders,
            seed=seed,
            paired_reverse=True,
        )
        elapsed = time.perf_counter() - t0
        if elapsed > EXPLAIN_TIMEOUT_S:
            raise TimeoutError(f"Attribution exceeded {EXPLAIN_TIMEOUT_S}s ({elapsed:.2f}s)")
        raw["elapsed_seconds"] = elapsed
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX:
                _cache.pop(next(iter(_cache)))
            _cache[key] = raw
        cached = raw

    pred = cached["prediction"]
    target = for_class or pred
    if target not in cached["attributions_by_class"]:
        target = pred
    feats = cached["attributions_by_class"][target][:top_k]
    toward = [f for f in feats if f["direction"] == "increases"][:5]
    against = [f for f in feats if f["direction"] == "decreases"][:5]
    return {
        **cached,
        "top_features": feats[:top_k],
        "toward_features": toward,
        "against_features": against,
        "predicted_class_for_shap": target,
        "shap_note": _SHAP_NOTE,
        "explanation_mode": "approximate_attribution",
        "baseline_description": (
            "Typical household in the background sample (mean encoded profile over "
            f"{len(X_bg)} shock-exposed modelling households)."
        ),
    }


def template_explanation(
    prediction: str,
    display_label: str,
    scores: dict[str, float],
    top_features: list[dict[str, Any]] | None,
    *,
    mode: str,
) -> str:
    score_bits = ", ".join(f"{k} {float(scores.get(k, 0)):.2f}" for k in sorted(scores))
    parts = [
        f"The model’s highest score is for “{display_label}” ({prediction}). "
        f"The model scores (not calibrated probabilities) are {score_bits}."
    ]
    if mode == "approximate_attribution" and top_features:
        bits = []
        for f in top_features[:5]:
            label = f.get("label") or plain_question_label(str(f["name"]))
            direction = (
                "raised"
                if f.get("direction") == "increases" or float(f.get("shap", 0)) >= 0
                else "lowered"
            )
            band = f.get("magnitude_band") or magnitude_band(float(f.get("shap", 0)))
            bits.append(f"{label} {direction} this class score ({band})")
        parts.append(
            "Among the main factors relative to a typical household in the background sample "
            "(approximate), " + "; ".join(bits) + "."
        )
    else:
        parts.append(
            "Feature-level attribution was unavailable, so this note uses model scores only."
        )
    parts.append(
        "This describes what drove the model’s score, not why the household actually "
        "coped the way it did, and it is not for targeting or eligibility decisions."
    )
    return " ".join(parts)


def grounded_reply_ok(reply: str, allowed_numbers: set[str], allowed_names: set[str]) -> bool:
    """Reject LLM text that invents decimal scores/effects or unknown feature labels."""
    import re

    allowed_floats = set()
    for n in allowed_numbers:
        try:
            allowed_floats.add(round(float(n.replace("%", "").lstrip("+")), 2))
        except ValueError:
            continue

    for m in re.finditer(r"[+-]?\d+\.\d+", reply):
        val = round(float(m.group(0)), 2)
        if not any(abs(val - a) < 1e-2 for a in allowed_floats):
            return False

    known_labels = sorted({v.lower() for v in _QUESTION_LABELS.values()}, key=len, reverse=True)
    allowed_lower = {n.lower() for n in allowed_names if n}
    lower = reply.lower()
    # Forbid "biggest" / "top factor" style claims
    if re.search(r"\b(biggest|largest|top factor|most important factor|main reason)\b", lower):
        return False
    for lab in known_labels:
        if lab in lower and not any(lab == a or lab in a or a in lab for a in allowed_lower):
            return False
    return True


def allowed_tokens_from_explain(result: dict[str, Any]) -> tuple[set[str], set[str]]:
    nums: set[str] = set()
    names: set[str] = set()
    for k, v in (result.get("scores") or {}).items():
        nums.add(f"{float(v):.2f}")
        nums.add(f"{float(v):.3f}")
        nums.add(str(int(round(float(v) * 100))) + "%")
        names.add(str(k))
    for k, v in (result.get("baseline_scores") or {}).items():
        nums.add(f"{float(v):.2f}")
    for cls, feats in (result.get("attributions_by_class") or {}).items():
        names.add(str(cls))
        for f in feats[:12]:
            names.add(str(f["name"]))
            names.add(str(f.get("label") or ""))
            nums.add(f"{float(f['shap']):.2f}")
            nums.add(f"{float(f['shap']):+.2f}")
            if f.get("magnitude_band"):
                names.add(str(f["magnitude_band"]))
    names.update({"small", "moderate", "large", "among the main factors"})
    names.add(str(result.get("prediction") or ""))
    names.add(str(result.get("display_label") or ""))
    return nums, {n for n in names if n}
