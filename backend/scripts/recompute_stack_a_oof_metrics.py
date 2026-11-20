#!/usr/bin/env python3
"""Recompute Stack A per-class P/R from OOF (first 5 folds, seed 42). Does not change the served model."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
WB = ROOT / "all_shocks_framing_workbench"
sys.path.insert(0, str(WB))
from config import CLASS_ORDER, PREPARED, TABLES  # noqa: E402

SEED = 42
N_SPLITS = 5
OUT = ROOT / "artefacts" / "classifier" / "v0.2.0"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "s14", WB / "14_final_stacking_assessment.py"
    )
    assert spec and spec.loader
    s14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s14)

    fin = pd.read_parquet(PREPARED / "11_financial_expansion_features.parquet")
    y = fin["coping_category"].astype(str)
    X = fin.drop(columns=["respondent_id", "coping_category"])
    assert X.shape[1] == 93 and len(X) == 5241

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=5, random_state=SEED)
    folds = list(rskf.split(X, y))[:N_SPLITS]
    stack = s14.build_stack(list(X.columns), include_rf=False)
    print(f"Running Stack A OOF on {len(folds)} folds...", flush=True)
    t0 = time.perf_counter()
    fold_rows = []
    oof = pd.Series(index=y.index, dtype=object)
    for i, (tr, te) in enumerate(folds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack.fit(X.iloc[tr], y.iloc[tr])
            pred = stack.predict(X.iloc[te])
        oof.iloc[te] = pred
        row = s14.metrics_row("stack_a_logistic_hgb", i, y.iloc[te], pred)
        fold_rows.append(row)
        print(
            f" fold {i}: macro={row['macro_f1']:.4f} formal_f1={row['f1_formal']:.4f} "
            f"P_formal={row['precision_formal']:.3f} R_formal={row['recall_formal']:.3f}",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    fold_df = pd.DataFrame(fold_rows)
    p, r, f, _ = precision_recall_fscore_support(
        y, oof.astype(str), labels=CLASS_ORDER, zero_division=0
    )
    macro = f1_score(y, oof.astype(str), average="macro", zero_division=0)
    cm = confusion_matrix(y, oof.astype(str), labels=CLASS_ORDER)
    summary = {
        "experiment": "stack_a_logistic_hgb",
        "protocol": (
            "RepeatedStratifiedKFold first 5 folds, n_splits=5, n_repeats=5, random_state=42"
        ),
        "n": int(len(y)),
        "n_features": 93,
        "seconds_total": round(elapsed, 1),
        "macro_f1_oof": round(float(macro), 4),
        "macro_f1_mean_folds": round(float(fold_df["macro_f1"].mean()), 4),
        "macro_f1_std_folds": round(float(fold_df["macro_f1"].std()), 4),
        "per_class": {
            cls: {
                "precision_oof": round(float(p[j]), 4),
                "recall_oof": round(float(r[j]), 4),
                "f1_oof": round(float(f[j]), 4),
                "precision_mean_folds": round(float(fold_df[f"precision_{cls}"].mean()), 4),
                "recall_mean_folds": round(float(fold_df[f"recall_{cls}"].mean()), 4),
                "f1_mean_folds": round(float(fold_df[f"f1_{cls}"].mean()), 4),
            }
            for j, cls in enumerate(CLASS_ORDER)
        },
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(TABLES / "14_stack_a_fold_metrics_full.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        TABLES / "14_stack_a_confusion_oof.csv"
    )
    (OUT / "stack_a_oof_class_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE", elapsed, flush=True)


if __name__ == "__main__":
    main()
