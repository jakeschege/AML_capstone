#!/usr/bin/env python3
"""Script 12 — Controlled evaluation of expanded feature specs (class-weighted logistic)."""

from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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

from config import CLASS_ORDER, FIGURES, PREPARED, REPORTS, TABLES, ensure_dirs

SEED = 42
N_SPLITS = 5
N_REPEATS = 5

SPECS = [
    "baseline_model_a_plus_r2b",
    "agri_diversity_expansion",
    "farm_challenge_expansion",
    "financial_expansion",
    "welfare_expansion",
    "combined_expansion",
]


def continuous_cols(columns: list[str]) -> list[str]:
    return [c for c in ["age_years", "education", "n_crop_types", "n_livestock_types"] if c in columns]


def make_pipe(feature_names: list[str]) -> Pipeline:
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
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(name: str, X: pd.DataFrame, y: pd.Series, folds) -> tuple[pd.DataFrame, dict, np.ndarray]:
    pipe = make_pipe(list(X.columns))
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
            "specification": name,
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
        print(f"  {name} fold {i}: macro F1={row['macro_f1']:.4f} formal F1={row['f1_formal']:.4f}")
    elapsed = time.perf_counter() - t0
    fold_df = pd.DataFrame(rows)
    cm = confusion_matrix(y, oof.astype(str), labels=CLASS_ORDER)
    summary = {
        "specification": name,
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
                    "specification": name,
                    "true_class": true_c,
                    "pred_class": pred_c,
                    "count": int(cm[i, j]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    matrices = {}
    y_ref = None
    for name in SPECS:
        df = pd.read_parquet(PREPARED / f"11_{name}_features.parquet")
        assert list(df["respondent_id"]) == list(
            pd.read_parquet(PREPARED / "11_baseline_model_a_plus_r2b_features.parquet")["respondent_id"]
        )
        y = df["coping_category"].astype(str)
        X = df.drop(columns=["respondent_id", "coping_category"])
        matrices[name] = (X, y)
        y_ref = y
        print(f"Loaded {name}: n={len(X)} p={X.shape[1]}")

    # Same external folds as prior experiments (seed 42, 5×5)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rskf.split(matrices["baseline_model_a_plus_r2b"][0], y_ref))

    fold_parts = []
    summaries = []
    cm_parts = []
    for name in SPECS:
        X, y = matrices[name]
        print(f"\n=== {name} ===")
        fold_df, summary, cm = evaluate(name, X, y, folds)
        fold_parts.append(fold_df)
        summaries.append(summary)
        cm_parts.append(cm_long(name, cm))

    fold_all = pd.concat(fold_parts, ignore_index=True)
    score = pd.DataFrame(summaries)
    cm_all = pd.concat(cm_parts, ignore_index=True)

    # Fold differences vs baseline
    base = fold_all[fold_all["specification"] == "baseline_model_a_plus_r2b"].set_index("fold")
    diff_rows = []
    for name in SPECS:
        if name == "baseline_model_a_plus_r2b":
            continue
        cur = fold_all[fold_all["specification"] == name].set_index("fold")
        d = (cur["macro_f1"] - base["macro_f1"]).rename("macro_f1_delta_vs_baseline")
        formal_d = (cur["f1_formal"] - base["f1_formal"]).rename("formal_f1_delta_vs_baseline")
        tmp = pd.concat([d, formal_d], axis=1).reset_index()
        tmp.insert(0, "specification", name)
        tmp["n_folds_macro_improved"] = int((tmp["macro_f1_delta_vs_baseline"] > 0).sum())
        tmp["pct_folds_macro_improved"] = 100 * tmp["n_folds_macro_improved"] / len(tmp)
        diff_rows.append(tmp)
    diffs = pd.concat(diff_rows, ignore_index=True)

    fold_all.to_csv(TABLES / "12_feature_expansion_fold_metrics.csv", index=False)
    score.to_csv(TABLES / "12_feature_expansion_scorecard.csv", index=False)
    cm_all.to_csv(TABLES / "12_feature_expansion_confusion_matrices.csv", index=False)
    diffs.to_csv(TABLES / "12_feature_expansion_fold_differences.csv", index=False)

    # Chart
    plot_df = score.sort_values("macro_f1_mean")
    plt.figure(figsize=(10, 5))
    plt.barh(
        plot_df["specification"],
        plot_df["macro_f1_mean"],
        xerr=plot_df["macro_f1_std"],
        color="#1c6a84",
    )
    plt.xlabel("Macro F1 (mean ± sd, 5×5 CV)")
    plt.title("Script 12 — Feature expansion macro F1")
    plt.tight_layout()
    plt.savefig(FIGURES / "12_feature_expansion_macro_f1.png", dpi=120)
    plt.close()

    print("\nScorecard:")
    print(
        score[
            [
                "specification",
                "n_features",
                "macro_f1_mean",
                "macro_f1_std",
                "f1_formal_mean",
                "balanced_accuracy_mean",
            ]
        ].to_string(index=False)
    )
    print("Script 12 complete.")


if __name__ == "__main__":
    main()
