"""Nearest-medoid Spec A assignment under Gower distance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .encoding import SPEC_A_FEATURES


def gower_distance_row(x: np.ndarray, medoid: np.ndarray, is_numeric: np.ndarray, ranges: np.ndarray) -> float:
    diffs = np.empty_like(x, dtype=float)
    numeric_mask = is_numeric
    cat_mask = ~numeric_mask
    diffs[numeric_mask] = np.abs(x[numeric_mask] - medoid[numeric_mask]) / np.maximum(ranges[numeric_mask], 1e-12)
    diffs[cat_mask] = (x[cat_mask] != medoid[cat_mask]).astype(float)
    return float(np.mean(diffs))


class Segmenter:
    def __init__(self, artefact_dir: Path):
        self.artefact_dir = artefact_dir
        self.medoids = pd.read_parquet(artefact_dir / "spec_a_k5_medoids.parquet")
        self.labels = json.loads((artefact_dir / "cluster_labels.json").read_text())
        cfg = json.loads((artefact_dir / "gower_config.json").read_text())
        self.is_numeric = np.array([cfg["is_numeric"][c] for c in SPEC_A_FEATURES])
        self.ranges = np.array([cfg["ranges"][c] for c in SPEC_A_FEATURES], dtype=float)
        self.M = self.medoids[SPEC_A_FEATURES].to_numpy(dtype=float)
        self.cluster_ids = self.medoids["cluster_id"].astype(int).tolist()
        self.metadata = json.loads((artefact_dir / "metadata.json").read_text())

    def profiles(self) -> list[dict]:
        rows = []
        for _, r in self.medoids.iterrows():
            rows.append(
                {
                    "cluster_id": int(r["cluster_id"]),
                    "label": str(r["label"]),
                    "features": {c: float(r[c]) for c in SPEC_A_FEATURES},
                }
            )
        return rows

    def assign(self, row: pd.DataFrame) -> dict:
        x = row[SPEC_A_FEATURES].to_numpy(dtype=float)[0]
        distances = {
            str(cid): gower_distance_row(x, self.M[i], self.is_numeric, self.ranges)
            for i, cid in enumerate(self.cluster_ids)
        }
        best = min(distances, key=distances.get)
        return {
            "assigned": True,
            "cluster_id": int(best),
            "label": self.labels.get(best, self.labels.get(str(best))),
            "distances": distances,
        }
