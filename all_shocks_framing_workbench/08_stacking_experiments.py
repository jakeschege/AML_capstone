#!/usr/bin/env python3
"""Script 08 — Nested stacking on Model B, fold-level comparison, explainability."""

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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import CLASS_ORDER, FIGURES, PREPARED, REPORTS, TABLES, ensure_dirs

SEED = 42
N_SPLITS = 5
N_REPEATS = 5
INNER_CV = 5


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


def metrics_row(experiment: str, fold: int, y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    row = {
        "experiment": experiment,
        "fold": fold,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    for j, cls in enumerate(CLASS_ORDER):
        row[f"precision_{cls}"] = p[j]
        row[f"recall_{cls}"] = r[j]
        row[f"f1_{cls}"] = f[j]
    return row


def summarise(fold_df: pd.DataFrame, experiment: str, n: int, n_features: int, seconds: float) -> dict:
    sub = fold_df[fold_df["experiment"] == experiment]
    summary = {
        "experiment": experiment,
        "n": n,
        "n_features": n_features,
        "seconds_total": round(seconds, 2),
        "macro_f1_mean": sub["macro_f1"].mean(),
        "macro_f1_std": sub["macro_f1"].std(),
        "weighted_f1_mean": sub["weighted_f1"].mean(),
        "weighted_f1_std": sub["weighted_f1"].std(),
        "balanced_accuracy_mean": sub["balanced_accuracy"].mean(),
        "balanced_accuracy_std": sub["balanced_accuracy"].std(),
        "accuracy_mean": sub["accuracy"].mean(),
        "accuracy_std": sub["accuracy"].std(),
    }
    for cls in CLASS_ORDER:
        summary[f"f1_{cls}_mean"] = sub[f"f1_{cls}"].mean()
        summary[f"f1_{cls}_std"] = sub[f"f1_{cls}"].std()
        summary[f"precision_{cls}_mean"] = sub[f"precision_{cls}"].mean()
        summary[f"precision_{cls}_std"] = sub[f"precision_{cls}"].std()
        summary[f"recall_{cls}_mean"] = sub[f"recall_{cls}"].mean()
        summary[f"recall_{cls}_std"] = sub[f"recall_{cls}"].std()
    return summary


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


def build_stacking(feature_names: list[str], include_tree: bool = False) -> StackingClassifier:
    """Build stacking classifier with class-weighted bases + logistic meta-learner."""
    estimators = [
        (
            "logistic_cw",
            make_pipe(
                feature_names,
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
                ),
            ),
        ),
        (
            "random_forest_cw",
            make_pipe(
                feature_names,
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    max_depth=12,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ),
        (
            "hist_gb_cw",
            make_pipe(
                feature_names,
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_depth=6,
                    learning_rate=0.1,
                    max_iter=150,
                    random_state=SEED,
                ),
            ),
        ),
    ]
    if include_tree:
        estimators.append(
            (
                "decision_tree_cw",
                make_pipe(
                    feature_names,
                    DecisionTreeClassifier(
                        class_weight="balanced", max_depth=8, random_state=SEED
                    ),
                ),
            )
        )

    meta = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
    )
    inner = StratifiedKFold(n_splits=INNER_CV, shuffle=True, random_state=SEED)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=inner,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def run_cv(name: str, estimator, X: pd.DataFrame, y: pd.Series, folds, max_folds: int | None = None):
    rows = []
    oof = pd.Series(index=y.index, dtype=object)
    n_folds = len(folds) if max_folds is None else min(max_folds, len(folds))
    t0 = time.perf_counter()
    for i in range(n_folds):
        tr, te = folds[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X.iloc[tr], y.iloc[tr])
            pred = estimator.predict(X.iloc[te])
        oof.iloc[te] = pred
        rows.append(metrics_row(name, i, y.iloc[te], pred))
        print(f"  {name} fold {i}: macro F1={rows[-1]['macro_f1']:.4f}")
    elapsed = time.perf_counter() - t0
    fold_df = pd.DataFrame(rows)
    # OOF confusion only meaningful if all rows covered
    if max_folds is None or max_folds >= len(folds):
        cm = confusion_matrix(y, oof.astype(str), labels=CLASS_ORDER)
    else:
        # pilot: only scored rows
        mask = oof.notna()
        cm = confusion_matrix(y[mask], oof[mask].astype(str), labels=CLASS_ORDER)
    return fold_df, cm, elapsed


def plot_importance(imp_df: pd.DataFrame, title: str, out_path, top_n: int = 20) -> None:
    top = imp_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#1c6a84")
    plt.xlabel("Permutation importance (macro F1 drop)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    ensure_dirs()
    print(f"scikit-learn {sklearn.__version__}")

    b = pd.read_parquet(PREPARED / "model_b_features.parquet").set_index("respondent_id")
    y = b["coping_category"].astype(str)
    X_full = b.drop(columns=["coping_category"])
    base_cols, shock_cols, main_cols = split_feature_blocks(list(X_full.columns))
    cols_a = base_cols
    cols_b = base_cols + shock_cols + main_cols
    Xa = X_full[cols_a]
    Xb = X_full[cols_b]

    # Select base learners from Script 07
    score07 = pd.read_csv(TABLES / "07_scorecard_full.csv")
    print("\nScript 07 scorecard (macro F1):")
    print(score07[["experiment", "macro_f1_mean", "f1_formal_mean"]].to_string(index=False))

    nonlinear_names = [
        "model_b__decision_tree_class_weighted",
        "model_b__random_forest_class_weighted",
        "model_b__hist_gradient_boosting_class_weighted",
    ]
    nonlinear = score07[score07["experiment"].isin(nonlinear_names)].copy()
    best_nonlinear = nonlinear.sort_values("macro_f1_mean", ascending=False).iloc[0]
    best_nl_name = best_nonlinear["experiment"]
    print(f"\nStrongest individual nonlinear model: {best_nl_name} "
          f"(macro F1={best_nonlinear['macro_f1_mean']:.4f})")

    # Candidate architecture: logistic + RF + HGB (all evaluated in Script 07)
    # Drop decision tree from stacking unless it beats RF or HGB.
    include_tree = False
    tree_row = score07[score07["experiment"] == "model_b__decision_tree_class_weighted"]
    rf_row = score07[score07["experiment"] == "model_b__random_forest_class_weighted"]
    hgb_row = score07[score07["experiment"] == "model_b__hist_gradient_boosting_class_weighted"]
    if not tree_row.empty and not rf_row.empty and not hgb_row.empty:
        if tree_row.iloc[0]["macro_f1_mean"] > max(
            rf_row.iloc[0]["macro_f1_mean"], hgb_row.iloc[0]["macro_f1_mean"]
        ):
            include_tree = True
            print("Including decision tree in stacking because it beat RF and HGB individually.")
        else:
            print("Stacking bases: logistic_cw, random_forest_cw, hist_gb_cw "
                  "(decision tree omitted — weaker individually).")

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rskf.split(Xb, y))

    # Reuse Script 07 fold/scorecard/CM outputs for individual models (same seed + folds).
    # Only the stacking estimator is newly fitted here.
    fold07 = pd.read_csv(TABLES / "07_fold_metrics.csv")
    cm07 = pd.read_csv(TABLES / "07_confusion_matrices.csv")
    rename_map = {
        "ablation__model_a__logistic_class_weighted": "model_a__logistic_class_weighted",
        "ablation__model_b__logistic_class_weighted": "model_b__logistic_class_weighted",
    }
    comparator_exps = [
        "ablation__model_a__logistic_class_weighted",
        "ablation__model_b__logistic_class_weighted",
        "model_b__decision_tree_class_weighted",
        "model_b__random_forest_class_weighted",
        "model_b__hist_gradient_boosting_class_weighted",
    ]
    fold_comp = fold07[fold07["experiment"].isin(comparator_exps)].copy()
    fold_comp["experiment"] = fold_comp["experiment"].replace(rename_map)
    cm_comp = cm07[cm07["experiment"].isin(comparator_exps)].copy()
    cm_comp["experiment"] = cm_comp["experiment"].replace(rename_map)

    stack = build_stacking(cols_b, include_tree=include_tree)

    # --- Pilot: first 5 folds only ---
    print("\n=== Pilot: stacking on first 5 folds ===")
    pilot_folds = folds[:N_SPLITS]
    pilot_df, _, pilot_sec = run_cv("stacking_model_b_pilot", stack, Xb, y, pilot_folds)
    print(f"Pilot complete in {pilot_sec:.1f}s; mean macro F1={pilot_df['macro_f1'].mean():.4f}")

    # --- Full experiment ---
    print("\n=== Full repeated CV (5×5) — stacking only ===")
    print("Running stacking_model_b …")
    stack = build_stacking(cols_b, include_tree=include_tree)
    fold_df, cm, elapsed = run_cv("stacking_model_b", stack, Xb, y, folds)
    print(f"  stacking total time: {elapsed:.1f}s")

    all_fold_parts = [fold_df, fold_comp]
    all_cm_parts = [cm_long("stacking_model_b", cm), cm_comp]

    summaries = [summarise(fold_df, "stacking_model_b", len(y), len(cols_b), elapsed)]
    score07_lookup = score07.copy()
    score07_lookup["experiment"] = score07_lookup["experiment"].replace(rename_map)
    for name in [
        "model_a__logistic_class_weighted",
        "model_b__logistic_class_weighted",
        "model_b__decision_tree_class_weighted",
        "model_b__random_forest_class_weighted",
        "model_b__hist_gradient_boosting_class_weighted",
    ]:
        nfeat = len(cols_a) if name.startswith("model_a") else len(cols_b)
        src = score07_lookup[score07_lookup["experiment"] == name]
        sec = float(src.iloc[0]["seconds_total"]) if not src.empty else 0.0
        summaries.append(summarise(fold_comp, name, len(y), nfeat, sec))

    fold_all = pd.concat(all_fold_parts, ignore_index=True)
    score = pd.DataFrame(summaries)
    cm_all = pd.concat(all_cm_parts, ignore_index=True)

    # Fold-level stacking vs strongest nonlinear
    stack_folds = fold_all[fold_all["experiment"] == "stacking_model_b"].set_index("fold")
    best_folds = fold_all[fold_all["experiment"] == best_nl_name].set_index("fold")
    delta = (stack_folds["macro_f1"] - best_folds["macro_f1"]).rename("macro_f1_delta_vs_best_nl")
    n_better = int((delta > 0).sum())
    pct_better = 100.0 * n_better / len(delta)

    comparison_rows = score.copy()
    # attach fold-win summary as separate columns on stacking row
    comparison = comparison_rows.copy()
    comparison["best_nonlinear_experiment"] = best_nl_name
    comparison["stacking_beats_best_nl_folds"] = n_better
    comparison["stacking_beats_best_nl_pct"] = pct_better
    comparison["stacking_vs_best_nl_macro_f1_mean_delta"] = float(delta.mean())

    fold_all.to_csv(TABLES / "08_stacking_fold_metrics.csv", index=False)
    score.to_csv(TABLES / "08_stacking_scorecard.csv", index=False)
    cm_all.to_csv(TABLES / "08_stacking_confusion_matrices.csv", index=False)
    comparison.to_csv(TABLES / "08_stacking_comparison.csv", index=False)
    delta.reset_index().to_csv(TABLES / "08_stacking_vs_best_nl_fold_deltas.csv", index=False)

    print(f"\nStacking beats {best_nl_name} on {n_better}/{len(delta)} folds ({pct_better:.1f}%).")
    print(f"Mean macro-F1 delta (stack − best NL): {delta.mean():+.4f}")

    # Macro F1 chart
    plot_df = score.sort_values("macro_f1_mean")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["experiment"], plot_df["macro_f1_mean"], xerr=plot_df["macro_f1_std"], color="#1c6a84")
    plt.xlabel("Macro F1 (mean ± sd over 5×5 CV)")
    plt.title("Script 08 — Macro F1 comparison (stacking vs individuals)")
    plt.tight_layout()
    plt.savefig(FIGURES / "08_macro_f1_comparison.png", dpi=120)
    plt.close()

    # --- Explainability: permutation importance on a held-out fold ---
    print("\n=== Explainability (permutation importance, macro F1) ===")
    tr, te = folds[0]
    macro_scorer = make_scorer(f1_score, average="macro", zero_division=0)

    # Strongest individual model
    if "hist_gradient" in best_nl_name:
        champ_ind = make_pipe(
            cols_b,
            HistGradientBoostingClassifier(
                class_weight="balanced",
                max_depth=6,
                learning_rate=0.1,
                max_iter=150,
                random_state=SEED,
            ),
        )
    elif "random_forest" in best_nl_name:
        champ_ind = make_pipe(
            cols_b,
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=SEED,
            ),
        )
    else:
        champ_ind = make_pipe(
            cols_b,
            DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=SEED),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        champ_ind.fit(Xb.iloc[tr], y.iloc[tr])
        pi_ind = permutation_importance(
            champ_ind,
            Xb.iloc[te],
            y.iloc[te],
            scoring=macro_scorer,
            n_repeats=10,
            random_state=SEED,
            n_jobs=-1,
        )
    imp_ind = (
        pd.DataFrame(
            {
                "feature": cols_b,
                "importance_mean": pi_ind.importances_mean,
                "importance_std": pi_ind.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_ind.to_csv(TABLES / "08_perm_importance_best_individual.csv", index=False)
    plot_importance(
        imp_ind,
        f"Permutation importance — {best_nl_name}",
        FIGURES / "08_perm_importance_best_individual.png",
    )

    # Stacked model
    stack_fit = build_stacking(cols_b, include_tree=include_tree)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack_fit.fit(Xb.iloc[tr], y.iloc[tr])
        pi_stack = permutation_importance(
            stack_fit,
            Xb.iloc[te],
            y.iloc[te],
            scoring=macro_scorer,
            n_repeats=5,
            random_state=SEED,
            n_jobs=1,
        )
    imp_stack = (
        pd.DataFrame(
            {
                "feature": cols_b,
                "importance_mean": pi_stack.importances_mean,
                "importance_std": pi_stack.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_stack.to_csv(TABLES / "08_perm_importance_stacking.csv", index=False)
    plot_importance(
        imp_stack,
        "Permutation importance — stacking (original features)",
        FIGURES / "08_perm_importance_stacking.png",
    )

    # Role of base estimators: permute each base's contribution via leave-one-base-out
    # Fit stacking once, then measure macro F1 when each base is removed from meta features.
    print("Assessing base-estimator roles via leave-one-base-out on held-out fold …")
    # Rebuild meta features manually for transparency
    base_names = [n for n, _ in stack_fit.estimators]
    # After fit, stacked_estimators_ holds fitted bases; final_estimator_ is meta
    # Use sklearn's transform to get stacked features, then zero out blocks per base
    X_meta_te = stack_fit.transform(Xb.iloc[te])
    y_te = y.iloc[te]
    meta = stack_fit.final_estimator_
    full_pred = meta.predict(X_meta_te)
    full_f1 = f1_score(y_te, full_pred, average="macro", zero_division=0)

    # Each base contributes n_classes probability columns (sklearn class order)
    classes = list(stack_fit.classes_)
    n_classes = len(classes)
    role_rows = [{"component": "full_stack", "macro_f1": full_f1, "delta_vs_full": 0.0}]
    for i, bname in enumerate(base_names):
        X_drop = X_meta_te.copy()
        start = i * n_classes
        end = start + n_classes
        # Replace this base's probabilities with training-label priors
        priors = y.iloc[tr].value_counts(normalize=True).reindex(classes).fillna(0).values
        X_drop[:, start:end] = priors
        pred_drop = meta.predict(X_drop)
        f1_drop = f1_score(y_te, pred_drop, average="macro", zero_division=0)
        role_rows.append(
            {
                "component": f"without_{bname}",
                "macro_f1": f1_drop,
                "delta_vs_full": f1_drop - full_f1,
            }
        )

    # Meta-learner alone vs chance: also report base-only majority
    role_df = pd.DataFrame(role_rows)
    role_df.to_csv(TABLES / "08_stacking_base_roles.csv", index=False)

    plt.figure(figsize=(8, 4))
    plot_role = role_df[role_df["component"] != "full_stack"].copy()
    plt.barh(plot_role["component"], -plot_role["delta_vs_full"], color="#1c6a84")
    plt.xlabel("Macro F1 drop when base is replaced by training priors")
    plt.title("Stacking — contribution of each base learner")
    plt.tight_layout()
    plt.savefig(FIGURES / "08_stacking_base_roles.png", dpi=120)
    plt.close()

    print("\nScorecard:")
    print(score[["experiment", "macro_f1_mean", "macro_f1_std", "f1_formal_mean", "seconds_total"]].to_string(index=False))
    print(f"\nPilot seconds: {pilot_sec:.1f}; stacking full seconds: {summaries[0]['seconds_total']}")
    print("Script 08 complete.")


if __name__ == "__main__":
    main()
