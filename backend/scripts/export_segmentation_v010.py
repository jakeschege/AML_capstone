#!/usr/bin/env python3
"""Package Spec A k=5 medoids for nearest-medoid assignment."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.encoding import SPEC_A_FEATURES  # noqa: E402

WB = ROOT / "finaccess_coping_workbench"
TABLES = WB / "outputs" / "tables"
OUT = ROOT / "artefacts" / "segmentation" / "v0.1.0"

CLUSTER_LABELS = {
    "0": "Family-owned legume-oriented crop households",
    "1": "Own-name crop households",
    "2": "Community-land livestock-oriented households",
    "3": "Family-owned mixed crop and livestock households",
    "4": "Family-owned cereal-oriented households",
}


def gower_distance_row(x: np.ndarray, medoid: np.ndarray, is_numeric: np.ndarray, ranges: np.ndarray) -> float:
    """Simple Gower distance for mixed binary/numeric Spec A features."""
    diffs = np.empty_like(x, dtype=float)
    # numeric: abs diff / range; binary/categorical: 0 if equal else 1
    numeric_mask = is_numeric
    cat_mask = ~numeric_mask
    diffs[numeric_mask] = np.abs(x[numeric_mask] - medoid[numeric_mask]) / np.maximum(ranges[numeric_mask], 1e-12)
    diffs[cat_mask] = (x[cat_mask] != medoid[cat_mask]).astype(float)
    return float(np.mean(diffs))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    medoids_all = pd.read_csv(TABLES / "12_s1_spec_a_medoids.csv")
    k5 = medoids_all.loc[(medoids_all["specification"] == "A") & (medoids_all["k"] == 5)].copy()
    if k5.empty:
        raise SystemExit("No Spec A k=5 medoids found")

    missing = [c for c in SPEC_A_FEATURES if c not in k5.columns]
    if missing:
        raise SystemExit(f"Medoid columns missing: {missing}")

    out_df = k5[["cluster", "respondent_id"] + SPEC_A_FEATURES].copy()
    out_df = out_df.rename(columns={"cluster": "cluster_id"})
    out_df["cluster_id"] = out_df["cluster_id"].astype(int)
    out_df["label"] = out_df["cluster_id"].astype(str).map(CLUSTER_LABELS)
    out_df.to_parquet(OUT / "spec_a_k5_medoids.parquet", index=False)
    out_df.to_csv(OUT / "spec_a_k5_medoids.csv", index=False)

    (OUT / "spec_a_feature_list.json").write_text(
        json.dumps({"features": SPEC_A_FEATURES, "n_features": len(SPEC_A_FEATURES), "acres_excluded": True}, indent=2)
    )
    (OUT / "cluster_labels.json").write_text(json.dumps(CLUSTER_LABELS, indent=2))

    # Ranges for numeric features from segmentation matrix if available
    seg_path = WB / "outputs" / "prepared" / "09_X_segmentation.parquet"
    if seg_path.exists():
        Xseg = pd.read_parquet(seg_path)
        cols = [c for c in SPEC_A_FEATURES if c in Xseg.columns]
        ranges = {}
        for c in cols:
            r = float(Xseg[c].max() - Xseg[c].min())
            ranges[c] = r if r > 0 else 1.0
    else:
        ranges = {c: 1.0 for c in SPEC_A_FEATURES}

    is_numeric = {c: c in ("n_crop_types", "n_livestock_types") for c in SPEC_A_FEATURES}
    (OUT / "gower_config.json").write_text(
        json.dumps({"is_numeric": is_numeric, "ranges": ranges, "distance": "gower"}, indent=2)
    )

    metadata = {
        "specification": "A",
        "k": 5,
        "method": "nearest_medoid_gower",
        "source_medoids": "12_s1_spec_a_medoids.csv",
        "n_medoids": int(len(out_df)),
        "note": "Segment assignment is descriptive typology, not a coping-class predictor.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Smoke: each medoid assigns to itself
    M = out_df[SPEC_A_FEATURES].to_numpy(dtype=float)
    num_mask = np.array([is_numeric[c] for c in SPEC_A_FEATURES])
    range_arr = np.array([ranges[c] for c in SPEC_A_FEATURES], dtype=float)
    for i in range(len(M)):
        dists = [gower_distance_row(M[i], M[j], num_mask, range_arr) for j in range(len(M))]
        assert int(np.argmin(dists)) == i, (i, dists)

    print(f"Wrote {OUT}; medoids={len(out_df)}")


if __name__ == "__main__":
    main()
