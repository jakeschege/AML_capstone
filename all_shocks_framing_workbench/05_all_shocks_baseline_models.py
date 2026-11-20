#!/usr/bin/env python3
"""Script 05 — Initial classification feasibility (Model A vs B)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASS_ORDER, FIGURES, PREPARED, TABLES, ensure_dirs

SEED = 42
N_SPLITS = 5
N_REPEATS = 5


def continuous_cols(columns: list[str]) -> list[str]:
    return [c for c in ["age_years", "education", "n_crop_types", "n_livestock_types"] if c in columns]


def make_pipe(feature_names: list[str], clf) -> Pipeline:
    cont = continuous_cols(feature_names)
    other = [c for c in feature_names if c not in cont]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                cont,
            ),
            (
                "other",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]),
                other,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def eval_model(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds) -> tuple[pd.DataFrame, dict]:
    rows = []
    oof = pd.Series(index=y.index, dtype=object)
    for i, (tr, te) in enumerate(folds):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        oof.iloc[te] = pred
        p, r, f, _ = precision_recall_fscore_support(
            y.iloc[te], pred, labels=CLASS_ORDER, zero_division=0
        )
        row = {
            "model": name,
            "fold": i,
            "macro_f1": f1_score(y.iloc[te], pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y.iloc[te], pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[te], pred),
            "accuracy": accuracy_score(y.iloc[te], pred),
        }
        for j, cls in enumerate(CLASS_ORDER):
            row[f"precision_{cls}"] = p[j]
            row[f"recall_{cls}"] = r[j]
            row[f"f1_{cls}"] = f[j]
        rows.append(row)
    fold_df = pd.DataFrame(rows)
    # Overall confusion on concatenated OOF
    cm = confusion_matrix(y, oof, labels=CLASS_ORDER)
    summary = {
        "model": name,
        "n": len(y),
        "n_features": X.shape[1],
        "macro_f1_mean": fold_df["macro_f1"].mean(),
        "macro_f1_std": fold_df["macro_f1"].std(),
        "weighted_f1_mean": fold_df["weighted_f1"].mean(),
        "weighted_f1_std": fold_df["weighted_f1"].std(),
        "balanced_accuracy_mean": fold_df["balanced_accuracy"].mean(),
        "balanced_accuracy_std": fold_df["balanced_accuracy"].std(),
        "accuracy_mean": fold_df["accuracy"].mean(),
        "accuracy_std": fold_df["accuracy"].std(),
    }
    for cls in CLASS_ORDER:
        summary[f"f1_{cls}_mean"] = fold_df[f"f1_{cls}"].mean()
        summary[f"f1_{cls}_std"] = fold_df[f"f1_{cls}"].std()
        summary[f"precision_{cls}_mean"] = fold_df[f"precision_{cls}"].mean()
        summary[f"recall_{cls}_mean"] = fold_df[f"recall_{cls}"].mean()
    return fold_df, summary, cm, oof


def run_suite(tag: str, X: pd.DataFrame, y: pd.Series) -> None:
    print(f"\n=== {tag} n={len(X)} p={X.shape[1]} ===")
    print(y.value_counts().reindex(CLASS_ORDER))
    min_class = int(y.value_counts().min())
    if min_class < N_SPLITS:
        print(
            f"WARNING: smallest class n={min_class} < n_splits={N_SPLITS}. "
            "RepeatedStratifiedKFold may be unstable; documenting without merging classes."
        )

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rskf.split(X, y))

    specs = [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent")),
        (
            "logistic_unweighted",
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        ),
        (
            "logistic_class_weighted",
            LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
            ),
        ),
    ]

    all_folds = []
    summaries = []
    for name, clf in specs:
        full_name = f"{tag}__{name}"
        print(f"Fitting {full_name} …")
        pipe = make_pipe(list(X.columns), clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_df, summary, cm, oof = eval_model(full_name, pipe, X, y, folds)
        all_folds.append(fold_df)
        summaries.append(summary)
        pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
            TABLES / f"05_confusion_{full_name}.csv"
        )
        print(
            f"  macro F1 mean={summary['macro_f1_mean']:.4f} "
            f"(sd={summary['macro_f1_std']:.4f})"
        )

    fold_out = pd.concat(all_folds, ignore_index=True)
    fold_out.to_csv(TABLES / f"05_fold_metrics_{tag}.csv", index=False)
    score = pd.DataFrame(summaries)
    score.to_csv(TABLES / f"05_scorecard_{tag}.csv", index=False)
    return score


def main() -> None:
    ensure_dirs()
    a = pd.read_parquet(PREPARED / "model_a_features.parquet")
    b = pd.read_parquet(PREPARED / "model_b_features.parquet")

    y_a = a["coping_category"].astype(str)
    X_a = a.drop(columns=["respondent_id", "coping_category"])
    y_b = b["coping_category"].astype(str)
    X_b = b.drop(columns=["respondent_id", "coping_category"])

    score_a = run_suite("model_a", X_a, y_a)
    score_b = run_suite("model_b", X_b, y_b)

    compare = pd.concat([score_a, score_b], ignore_index=True)
    compare.to_csv(TABLES / "05_scorecard_combined.csv", index=False)
    print("\nCombined scorecard (macro F1):")
    print(compare[["model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "f1_formal_mean"]])
    print("Script 05 complete.")


if __name__ == "__main__":
    main()
