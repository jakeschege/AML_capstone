#!/usr/bin/env python3
"""Script 07 — Feature ablation (A / A+R2A / A+R2B / B) and nonlinear Model B models."""

from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def split_feature_blocks(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    shock = [c for c in columns if c.startswith("shock_")]
    main = [c for c in columns if c.startswith("main_shock_")]
    base = [c for c in columns if c not in shock and c not in main]
    return base, shock, main


def evaluate(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds) -> tuple[pd.DataFrame, dict, np.ndarray]:
    rows = []
    oof = pd.Series(index=y.index, dtype=object)
    t0 = time.perf_counter()
    for i, (tr, te) in enumerate(folds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
        oof.iloc[te] = pred
        p, r, f, _ = precision_recall_fscore_support(
            y.iloc[te], pred, labels=CLASS_ORDER, zero_division=0
        )
        row = {
            "experiment": name,
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
    elapsed = time.perf_counter() - t0
    fold_df = pd.DataFrame(rows)
    cm = confusion_matrix(y, oof, labels=CLASS_ORDER)
    summary = {
        "experiment": name,
        "n": len(y),
        "n_features": X.shape[1],
        "seconds_total": round(elapsed, 2),
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
        summary[f"precision_{cls}_std"] = fold_df[f"precision_{cls}"].std()
        summary[f"recall_{cls}_mean"] = fold_df[f"recall_{cls}"].mean()
        summary[f"recall_{cls}_std"] = fold_df[f"recall_{cls}"].std()
    return fold_df, summary, cm


def cm_long(name: str, cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, true_c in enumerate(CLASS_ORDER):
        for j, pred_c in enumerate(CLASS_ORDER):
            rows.append(
                {
                    "experiment": name,
                    "true_class": true_c,
                    "pred_class": pred_c,
                    "count": int(cm[i, j]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    print(f"scikit-learn {sklearn.__version__}")
    print("HistGradientBoostingClassifier supports class_weight:", True)

    a = pd.read_parquet(PREPARED / "model_a_features.parquet").set_index("respondent_id")
    b = pd.read_parquet(PREPARED / "model_b_features.parquet").set_index("respondent_id")
    assert list(a.index) == list(b.index)
    y = b["coping_category"].astype(str)
    assert (a["coping_category"].astype(str) == y).all()

    X_full = b.drop(columns=["coping_category"])
    base_cols, shock_cols, main_cols = split_feature_blocks(list(X_full.columns))
    print(f"Blocks: base={len(base_cols)} R2A={len(shock_cols)} R2B={len(main_cols)}")

    feature_sets = {
        "model_a": base_cols,
        "model_a_plus_r2a": base_cols + shock_cols,
        "model_a_plus_r2b": base_cols + main_cols,
        "model_b": base_cols + shock_cols + main_cols,
    }

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rskf.split(X_full, y))

    fold_parts = []
    summaries = []
    cm_parts = []

    # --- Part A: feature ablation with class-weighted logistic ---
    print("\n=== Part A: feature ablation (class-weighted logistic) ===")
    for spec_name, cols in feature_sets.items():
        X = X_full[cols]
        exp = f"ablation__{spec_name}__logistic_class_weighted"
        print(f"Running {exp} (p={len(cols)}) …")
        pipe = make_pipe(
            cols,
            LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
            ),
        )
        fold_df, summary, cm = evaluate(exp, pipe, X, y, folds)
        fold_parts.append(fold_df)
        summaries.append(summary)
        cm_parts.append(cm_long(exp, cm))
        print(f"  macro F1={summary['macro_f1_mean']:.4f} formal F1={summary['f1_formal_mean']:.4f}")

    # Unweighted logistic benchmarks for A and B
    print("\n=== Part A benchmarks: unweighted logistic A/B ===")
    for spec_name in ("model_a", "model_b"):
        cols = feature_sets[spec_name]
        exp = f"benchmark__{spec_name}__logistic_unweighted"
        print(f"Running {exp} …")
        pipe = make_pipe(
            cols,
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        )
        fold_df, summary, cm = evaluate(exp, pipe, X_full[cols], y, folds)
        fold_parts.append(fold_df)
        summaries.append(summary)
        cm_parts.append(cm_long(exp, cm))
        print(f"  macro F1={summary['macro_f1_mean']:.4f}")

    # --- Part B: nonlinear models on Model B ---
    print("\n=== Part B: nonlinear models on Model B ===")
    cols_b = feature_sets["model_b"]
    Xb = X_full[cols_b]

    nonlinear = [
        (
            "model_b__decision_tree_class_weighted",
            DecisionTreeClassifier(
                class_weight="balanced", max_depth=8, random_state=SEED
            ),
        ),
        (
            "model_b__random_forest_class_weighted",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
        (
            "model_b__hist_gradient_boosting_class_weighted",
            HistGradientBoostingClassifier(
                class_weight="balanced",
                max_depth=6,
                learning_rate=0.1,
                max_iter=150,
                random_state=SEED,
            ),
        ),
    ]

    for exp, clf in nonlinear:
        print(f"Running {exp} …")
        pipe = make_pipe(cols_b, clf)
        fold_df, summary, cm = evaluate(exp, pipe, Xb, y, folds)
        fold_parts.append(fold_df)
        summaries.append(summary)
        cm_parts.append(cm_long(exp, cm))
        print(
            f"  macro F1={summary['macro_f1_mean']:.4f} "
            f"formal F1={summary['f1_formal_mean']:.4f} "
            f"time={summary['seconds_total']}s"
        )

    fold_all = pd.concat(fold_parts, ignore_index=True)
    score = pd.DataFrame(summaries)
    cm_all = pd.concat(cm_parts, ignore_index=True)

    # Split scorecards for requested filenames
    ablation_mask = score["experiment"].str.startswith("ablation__") | score["experiment"].str.startswith(
        "benchmark__"
    )
    model_mask = score["experiment"].str.startswith("model_b__") | score["experiment"].isin(
        [
            "ablation__model_b__logistic_class_weighted",
            "benchmark__model_b__logistic_unweighted",
            "ablation__model_a__logistic_class_weighted",
        ]
    )

    fold_all.to_csv(TABLES / "07_fold_metrics.csv", index=False)
    score.loc[ablation_mask].to_csv(TABLES / "07_feature_ablation_results.csv", index=False)
    score.loc[model_mask].to_csv(TABLES / "07_model_comparison.csv", index=False)
    # Also save full scorecard for stacking script
    score.to_csv(TABLES / "07_scorecard_full.csv", index=False)
    cm_all.to_csv(TABLES / "07_confusion_matrices.csv", index=False)

    # Macro F1 chart
    plot_df = score.sort_values("macro_f1_mean")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["experiment"], plot_df["macro_f1_mean"], xerr=plot_df["macro_f1_std"], color="#1c6a84")
    plt.xlabel("Macro F1 (mean ± sd over 5×5 CV)")
    plt.title("Script 07 — Macro F1 comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "07_macro_f1_comparison.png", dpi=120)
    plt.close()

    print("\nScorecard (macro F1):")
    print(score[["experiment", "macro_f1_mean", "macro_f1_std", "f1_formal_mean", "seconds_total"]].to_string(index=False))
    print("Script 07 complete.")


if __name__ == "__main__":
    main()
