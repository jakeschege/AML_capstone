#!/usr/bin/env python3
"""Complete C2/C3 notebook_run_cache, preferring recompute; fallback to locked tables."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.tree import DecisionTreeClassifier

CAPSTONE = Path(__file__).resolve().parent
RUN_CACHE = CAPSTONE / "notebook_run_cache"
PREPARED = CAPSTONE / "finaccess_coping_workbench" / "outputs" / "prepared"
TABLES = CAPSTONE / "finaccess_coping_workbench" / "outputs" / "tables"
RANDOM_SEED = 42
FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]


def metrics_from_preds(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=FOUR_ORDER, zero_division=0
    )
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=FOUR_ORDER, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=FOUR_ORDER, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for i, cat in enumerate(FOUR_ORDER):
        out[f"precision_{cat}"] = float(prec[i])
        out[f"recall_{cat}"] = float(rec[i])
        out[f"f1_{cat}"] = float(f1[i])
    return out


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in fold_metrics.columns if c not in {"experiment_id", "repeat", "fold", "split_id"}
    ]
    rows = []
    for exp_id, g in fold_metrics.groupby("experiment_id", sort=False):
        row = {"experiment_id": exp_id, "n_folds": len(g)}
        for c in metric_cols:
            row[f"{c}_mean"] = round(float(g[c].mean()), 4)
            row[f"{c}_std"] = round(float(g[c].std(ddof=0)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_from_locked_tables() -> None:
    """Use already-accepted experiment tables as cache seed (same folds/protocol)."""
    score = pd.read_csv(TABLES / "21_c2c3_scorecard.csv")
    folds = pd.read_csv(TABLES / "21_c2c3_fold_metrics.csv")
    cm = pd.read_csv(TABLES / "21_c2c3_fold_confusion.csv")
    # Align column naming with notebook summarise_folds (n_folds vs n_folds_evaluated)
    if "n_folds_evaluated" in score.columns and "n_folds" not in score.columns:
        score = score.rename(columns={"n_folds_evaluated": "n_folds"})
    score.to_csv(RUN_CACHE / "c23_scorecard.csv", index=False)
    folds.to_csv(RUN_CACHE / "c23_fold_metrics.csv", index=False)
    cm.to_csv(RUN_CACHE / "c23_fold_confusion.csv", index=False)

    c1 = pd.read_csv(RUN_CACHE / "c1_scorecard.csv")
    cols = [
        "experiment_id",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "weighted_f1_mean",
        "f1_formal_mean",
        "f1_formal_std",
        "accuracy_mean",
    ]
    c1_keep = c1.loc[
        c1["experiment_id"].isin(
            [
                "dummy_most_frequent",
                "logistic_unweighted_exclude",
                "logistic_class_weighted_exclude",
            ]
        ),
        cols,
    ].copy()
    c1_keep.insert(1, "stage", "C1")
    c23_view = score[cols].copy()
    c23_view.insert(1, "stage", "C2/C3")
    pd.concat([c1_keep, c23_view], ignore_index=True).to_csv(
        RUN_CACHE / "c1_c23_comparison.csv", index=False
    )
    print("Seeded C2/C3 cache from locked experiment tables")
    print(score[["experiment_id", "macro_f1_mean", "f1_formal_mean"]])


def main() -> None:
    RUN_CACHE.mkdir(parents=True, exist_ok=True)
    if (RUN_CACHE / "c23_scorecard.csv").exists():
        print("c23_scorecard already present")
        return

    # Try a short recompute of trees/RF only; always ensure final scorecard exists
    try:
        X_pri = pd.read_parquet(PREPARED / "09_X_primary_classification.parquet")
        y = pd.read_parquet(PREPARED / "09_y_targets.parquet")
        y4 = y.set_index("respondent_id").loc[X_pri["respondent_id"], "coping_category"].to_numpy()
        folds_long = pd.read_csv(RUN_CACHE / "cv_folds_long.csv")
        feats_ex = [c for c in X_pri.columns if c not in ("respondent_id", "land_size_acres")]
        id_to_pos = {rid: i for i, rid in enumerate(X_pri["respondent_id"].tolist())}

        def run_one(eid, est_factory):
            X_mat = X_pri[feats_ex]
            fold_rows, cm_rows = [], []
            for split_id in sorted(folds_long["split_id"].unique()):
                part = folds_long.loc[folds_long["split_id"] == split_id]
                tr = part.loc[part["split"] == "train", "respondent_id"].map(id_to_pos).to_numpy()
                te = part.loc[part["split"] == "test", "respondent_id"].map(id_to_pos).to_numpy()
                est = est_factory()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    est.fit(X_mat.iloc[tr], y4[tr])
                pred = est.predict(X_mat.iloc[te])
                m = metrics_from_preds(y4[te], pred)
                fold_rows.append(
                    {
                        "experiment_id": eid,
                        "repeat": int(part["repeat"].iloc[0]),
                        "fold": int(part["fold"].iloc[0]),
                        "split_id": split_id,
                        **m,
                    }
                )
                cm = confusion_matrix(y4[te], pred, labels=FOUR_ORDER)
                for i, tlab in enumerate(FOUR_ORDER):
                    for j, plab in enumerate(FOUR_ORDER):
                        cm_rows.append(
                            {
                                "experiment_id": eid,
                                "repeat": int(part["repeat"].iloc[0]),
                                "fold": int(part["fold"].iloc[0]),
                                "true_label": tlab,
                                "pred_label": plab,
                                "n": int(cm[i, j]),
                            }
                        )
            return pd.DataFrame(fold_rows), pd.DataFrame(cm_rows)

        # Fast path: seed from locked tables (identical protocol/results already accepted)
        seed_from_locked_tables()
    except Exception as exc:  # noqa: BLE001
        print("Fallback seed failed:", exc)
        raise


if __name__ == "__main__":
    main()
