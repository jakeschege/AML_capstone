#!/usr/bin/env python3
"""Simulated drift check (Evidently when compatible; mean-shift fallback).

Labelled **simulated** monitoring — do not auto-retrain on exit code 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "monitoring" / "outputs"
PREPARED = ROOT / "finaccess_coping_workbench" / "outputs" / "prepared"
SCHEMA = ROOT / "artefacts" / "classifier" / "v0.1.0" / "feature_schema.json"


def _patch_pandas_for_evidently() -> None:
    """Evidently legacy imports pandas.core.dtypes.common.infer_dtype_from_object (removed in pandas 3)."""
    import pandas.core.dtypes.common as common

    if not hasattr(common, "infer_dtype_from_object"):

        def infer_dtype_from_object(value):  # type: ignore[no-untyped-def]
            return np.dtype(type(value) if not hasattr(value, "dtype") else value.dtype)

        common.infer_dtype_from_object = infer_dtype_from_object  # type: ignore[attr-defined]


def _fallback_report(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str], exc: Exception | None) -> int:
    shifts = {c: abs(float(current[c].mean()) - float(reference[c].mean())) for c in cols}
    top = sorted(shifts.items(), key=lambda x: -x[1])[:10]
    drifted = bool(top and top[0][1] > 0.05)

    html = [
        "<html><head><title>Simulated drift (fallback)</title></head><body>",
        "<h1>Simulated feature drift — fallback mean-shift</h1>",
        "<p>Evidently unavailable or incompatible; using absolute mean shift.</p>",
        f"<p>Error: {exc}</p>" if exc else "",
        "<table border='1' cellpadding='4'><tr><th>feature</th><th>abs mean shift</th></tr>",
    ]
    for name, val in top:
        html.append(f"<tr><td>{name}</td><td>{val:.4f}</td></tr>")
    html.append("</table><p>Runbook: investigate; do not auto-retrain.</p></body></html>")
    html_path = OUT / "drift_report.html"
    json_path = OUT / "drift_report.json"
    html_path.write_text("\n".join(html))
    payload = {"simulated": True, "engine": "fallback_mean_shift", "top_mean_shifts": top, "drifted": drifted}
    json_path.write_text(json.dumps(payload, indent=2))

    summary = {
        "simulated": True,
        "engine": "fallback_mean_shift",
        "error": str(exc) if exc else None,
        "top_mean_shifts": top,
        "likely_drift_signal": drifted,
        "html": str(html_path),
        "json": str(json_path),
        "runbook": "Investigate feature/county mix shift; do not auto-retrain.",
    }
    (OUT / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if drifted else 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA.read_text())
    features = [f for f in schema["feature_names"] if not f.startswith("county_")][:20]
    counties = [f for f in schema["feature_names"] if f.startswith("county_")][:8]
    cols = features + counties

    X = pd.read_parquet(PREPARED / "09_X_primary_classification.parquet")[cols]
    reference = X.sample(n=min(600, len(X)), random_state=42)
    current = X.sample(n=min(600, len(X)), random_state=99).copy()

    # Simulate county mix / education skew
    if "education" in current.columns:
        current["education"] = current["education"].clip(lower=3)
    for c in counties[:3]:
        if c in current.columns:
            current[c] = 1
    for c in counties[3:]:
        if c in current.columns:
            current[c] = 0

    try:
        _patch_pandas_for_evidently()
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        html_path = OUT / "drift_report.html"
        json_path = OUT / "drift_report.json"
        # Evidently 0.7 API variants
        if hasattr(snapshot, "save_html"):
            snapshot.save_html(str(html_path))
            snapshot.save_json(str(json_path))
        elif hasattr(report, "save_html"):
            report.save_html(str(html_path))
            report.save_json(str(json_path))
        else:
            html_path.write_text("<html><body><p>Evidently report generated (no save_html).</p></body></html>")
            json_path.write_text(json.dumps({"note": "see drift_summary"}, indent=2))

        payload = json.loads(json_path.read_text()) if json_path.exists() else {}
        drifted = "drift" in json.dumps(payload).lower()
        summary = {
            "simulated": True,
            "engine": "evidently",
            "runbook": "Investigate feature/county mix shift; do not auto-retrain.",
            "html": str(html_path),
            "json": str(json_path),
            "likely_drift_signal": drifted,
        }
        (OUT / "drift_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 1 if drifted else 0
    except Exception as exc:
        return _fallback_report(reference, current, cols, exc)


if __name__ == "__main__":
    sys.exit(main())
