#!/usr/bin/env python3
"""Script 14 — Final stacking assessment on financial-expanded features (5-fold only).

Compares Stack A (logistic + HGB) and Stack B (logistic + HGB + RF) against
individual models on the same first five external folds (seed 42).
"""

from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASS_ORDER, FIGURES, PREPARED, REPORTS, TABLES, ensure_dirs

SEED = 42
N_SPLITS = 5
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


def base_logistic(feature_names: list[str]) -> Pipeline:
    return make_pipe(
        feature_names,
        LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
        ),
    )


def base_hgb(feature_names: list[str]) -> Pipeline:
    return make_pipe(
        feature_names,
        HistGradientBoostingClassifier(
            class_weight="balanced",
            max_depth=6,
            learning_rate=0.1,
            max_iter=150,
            random_state=SEED,
        ),
    )


def base_rf(feature_names: list[str]) -> Pipeline:
    return make_pipe(
        feature_names,
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        ),
    )


def build_stack(feature_names: list[str], include_rf: bool) -> StackingClassifier:
    estimators = [
        ("logistic_cw", base_logistic(feature_names)),
        ("hist_gb_cw", base_hgb(feature_names)),
    ]
    if include_rf:
        estimators.append(("random_forest_cw", base_rf(feature_names)))
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


def summarise(fold_df: pd.DataFrame, experiment: str, seconds: float, n_features: int) -> dict:
    sub = fold_df[fold_df["experiment"] == experiment]
    out = {
        "experiment": experiment,
        "n_folds": len(sub),
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
        out[f"f1_{cls}_mean"] = sub[f"f1_{cls}"].mean()
        out[f"f1_{cls}_std"] = sub[f"f1_{cls}"].std()
        out[f"precision_{cls}_mean"] = sub[f"precision_{cls}"].mean()
        out[f"recall_{cls}_mean"] = sub[f"recall_{cls}"].mean()
    return out


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


def run_stack(name: str, stack: StackingClassifier, X: pd.DataFrame, y: pd.Series, folds):
    rows = []
    oof = pd.Series(index=y.index, dtype=object)
    # store base predictions on test for complementarity
    base_pred_parts = []
    t0 = time.perf_counter()
    for i, (tr, te) in enumerate(folds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack.fit(X.iloc[tr], y.iloc[tr])
            pred = stack.predict(X.iloc[te])
            # fitted base estimators on full external train (after stacking fit)
            base_cols = {}
            for est_name, est in stack.named_estimators_.items():
                base_cols[f"pred_{est_name}"] = est.predict(X.iloc[te])
        oof.iloc[te] = pred
        row = metrics_row(name, i, y.iloc[te], pred)
        rows.append(row)
        print(f"  {name} fold {i}: macro={row['macro_f1']:.4f} formal={row['f1_formal']:.4f}")
        part = pd.DataFrame(base_cols)
        part["fold"] = i
        part["y_true"] = y.iloc[te].values
        part["pred_stack"] = pred
        part["row_index"] = X.iloc[te].index
        base_pred_parts.append(part)
    elapsed = time.perf_counter() - t0
    fold_df = pd.DataFrame(rows)
    cm = confusion_matrix(y, oof.astype(str), labels=CLASS_ORDER)
    base_preds = pd.concat(base_pred_parts, ignore_index=True)
    return fold_df, cm, elapsed, base_preds


def complementarity_table(base_preds: pd.DataFrame, stack_name: str) -> pd.DataFrame:
    """Agreement / disagreement among base learners on external test predictions."""
    pred_cols = [c for c in base_preds.columns if c.startswith("pred_") and c != "pred_stack"]
    rows = []
    # pairwise agreement
    for i, a in enumerate(pred_cols):
        for b in pred_cols[i + 1 :]:
            agree = (base_preds[a].astype(str) == base_preds[b].astype(str)).mean()
            rows.append(
                {
                    "stack": stack_name,
                    "metric": f"agreement_{a}_vs_{b}",
                    "value": float(agree),
                }
            )
    # all bases agree
    if len(pred_cols) >= 2:
        all_agree = base_preds[pred_cols].astype(str).nunique(axis=1).eq(1)
        rows.append(
            {
                "stack": stack_name,
                "metric": "fraction_all_bases_agree",
                "value": float(all_agree.mean()),
            }
        )
        # when bases disagree, does stack match logistic?
        if "pred_logistic_cw" in pred_cols:
            disagree = ~all_agree
            if disagree.any():
                match_log = (
                    base_preds.loc[disagree, "pred_stack"].astype(str)
                    == base_preds.loc[disagree, "pred_logistic_cw"].astype(str)
                ).mean()
                match_hgb = (
                    base_preds.loc[disagree, "pred_stack"].astype(str)
                    == base_preds.loc[disagree, "pred_hist_gb_cw"].astype(str)
                ).mean()
                rows.append(
                    {
                        "stack": stack_name,
                        "metric": "when_disagree_stack_matches_logistic",
                        "value": float(match_log),
                    }
                )
                rows.append(
                    {
                        "stack": stack_name,
                        "metric": "when_disagree_stack_matches_hgb",
                        "value": float(match_hgb),
                    }
                )
    # class-wise: formal predictions among bases
    for col in pred_cols + ["pred_stack"]:
        rows.append(
            {
                "stack": stack_name,
                "metric": f"pct_pred_formal_{col}",
                "value": float((base_preds[col].astype(str) == "formal").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    t_all = time.perf_counter()

    fin = pd.read_parquet(PREPARED / "11_financial_expansion_features.parquet")
    assert len(fin) == 5241
    y = fin["coping_category"].astype(str)
    X = fin.drop(columns=["respondent_id", "coping_category"])
    assert X.shape[1] == 93
    print(f"Financial matrix n={len(X)} p={X.shape[1]}")

    # First complete five-fold repeat from the same RepeatedStratifiedKFold protocol
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=5, random_state=SEED)
    all_folds = list(rskf.split(X, y))
    folds = all_folds[:N_SPLITS]
    print(f"Using first {len(folds)} external folds (seed={SEED})")

    # Individual models on these folds: reuse Script 13 fold metrics
    fold13 = pd.read_csv(TABLES / "13_final_fold_metrics.csv")
    individual_exps = [
        "logistic_class_weighted__financial",
        "random_forest_class_weighted__financial",
        "hist_gradient_boosting_class_weighted__financial",
    ]
    ind_folds = fold13[
        fold13["experiment"].isin(individual_exps) & (fold13["fold"] < N_SPLITS)
    ].copy()
    assert len(ind_folds) == 15, f"Expected 15 individual fold rows, got {len(ind_folds)}"

    # Stacking
    print("\n=== Stack A: logistic + HGB → logistic meta ===")
    stack_a = build_stack(list(X.columns), include_rf=False)
    fold_a, cm_a, sec_a, base_a = run_stack("stack_a_logistic_hgb", stack_a, X, y, folds)
    print(f"Stack A total time: {sec_a:.1f}s")

    print("\n=== Stack B: logistic + HGB + RF → logistic meta ===")
    stack_b = build_stack(list(X.columns), include_rf=True)
    fold_b, cm_b, sec_b, base_b = run_stack("stack_b_logistic_hgb_rf", stack_b, X, y, folds)
    print(f"Stack B total time: {sec_b:.1f}s")

    fold_all = pd.concat([ind_folds, fold_a, fold_b], ignore_index=True)

    summaries = []
    for exp in individual_exps:
        summaries.append(summarise(ind_folds, exp, seconds=0.0, n_features=93))
    summaries.append(summarise(fold_a, "stack_a_logistic_hgb", sec_a, 93))
    summaries.append(summarise(fold_b, "stack_b_logistic_hgb_rf", sec_b, 93))
    score = pd.DataFrame(summaries)

    cm_parts = [cm_long("stack_a_logistic_hgb", cm_a), cm_long("stack_b_logistic_hgb_rf", cm_b)]
    # rebuild individual OOF CMs from fold predictions? We don't have OOF preds stored.
    # Save stacking CMs; for individuals note metrics-only on these 5 folds.
    cm_all = pd.concat(cm_parts, ignore_index=True)

    comp = pd.concat(
        [
            complementarity_table(base_a, "stack_a_logistic_hgb"),
            complementarity_table(base_b, "stack_b_logistic_hgb_rf"),
        ],
        ignore_index=True,
    )
    comp.to_csv(TABLES / "14_base_learner_complementarity.csv", index=False)

    # Fold deltas vs logistic on matching folds
    log = ind_folds[ind_folds["experiment"] == "logistic_class_weighted__financial"].set_index("fold")
    hgb = ind_folds[
        ind_folds["experiment"] == "hist_gradient_boosting_class_weighted__financial"
    ].set_index("fold")
    delta_rows = []
    for name, fdf in [
        ("stack_a_logistic_hgb", fold_a),
        ("stack_b_logistic_hgb_rf", fold_b),
        (
            "hist_gradient_boosting_class_weighted__financial",
            ind_folds[
                ind_folds["experiment"] == "hist_gradient_boosting_class_weighted__financial"
            ],
        ),
        (
            "random_forest_class_weighted__financial",
            ind_folds[ind_folds["experiment"] == "random_forest_class_weighted__financial"],
        ),
    ]:
        cur = fdf.set_index("fold")
        d = (cur["macro_f1"] - log["macro_f1"]).rename("macro_f1_delta_vs_logistic")
        fd = (cur["f1_formal"] - log["f1_formal"]).rename("formal_f1_delta_vs_logistic")
        dh = (cur["macro_f1"] - hgb["macro_f1"]).rename("macro_f1_delta_vs_hgb")
        tmp = pd.concat([d, fd, dh], axis=1).reset_index()
        tmp.insert(0, "experiment", name)
        tmp["n_folds_macro_gt_logistic"] = int((tmp["macro_f1_delta_vs_logistic"] > 0).sum())
        tmp["n_folds_formal_gt_logistic"] = int((tmp["formal_f1_delta_vs_logistic"] > 0).sum())
        delta_rows.append(tmp)
    deltas = pd.concat(delta_rows, ignore_index=True)
    deltas.to_csv(TABLES / "14_stacking_fold_deltas.csv", index=False)

    fold_all.to_csv(TABLES / "14_stacking_fold_metrics.csv", index=False)
    score.to_csv(TABLES / "14_stacking_results.csv", index=False)
    cm_all.to_csv(TABLES / "14_stacking_confusion_matrices.csv", index=False)

    # Chart — 5-fold means only
    plot_df = score.sort_values("macro_f1_mean")
    plt.figure(figsize=(9, 5))
    colors = [
        "#c47b2b" if "stack" in e else "#1c6a84" for e in plot_df["experiment"]
    ]
    plt.barh(
        plot_df["experiment"],
        plot_df["macro_f1_mean"],
        xerr=plot_df["macro_f1_std"],
        color=colors,
    )
    plt.xlabel("Macro F1 (mean ± sd over 5 external folds)")
    plt.title("Script 14 — Stacking vs individuals (first 5-fold repeat only)")
    plt.tight_layout()
    plt.savefig(FIGURES / "14_stacking_comparison.png", dpi=120)
    plt.close()

    # Formal F1 companion chart values in report
    log_row = score[score["experiment"] == "logistic_class_weighted__financial"].iloc[0]
    hgb_row = score[
        score["experiment"] == "hist_gradient_boosting_class_weighted__financial"
    ].iloc[0]
    rf_row = score[score["experiment"] == "random_forest_class_weighted__financial"].iloc[0]
    a_row = score[score["experiment"] == "stack_a_logistic_hgb"].iloc[0]
    b_row = score[score["experiment"] == "stack_b_logistic_hgb_rf"].iloc[0]

    def beats(row, ref, metric="macro_f1_mean"):
        return float(row[metric] - ref[metric])

    # Selection
    selected = "logistic_class_weighted__financial"
    reasons = []
    # Require meaningful improvement: >=0.01 macro OR >=0.01 formal, and not trading off the other badly
    for label, row in [("Stack A", a_row), ("Stack B", b_row)]:
        dm_log = beats(row, log_row, "macro_f1_mean")
        df_log = beats(row, log_row, "f1_formal_mean")
        dm_hgb = beats(row, hgb_row, "macro_f1_mean")
        df_hgb = beats(row, hgb_row, "f1_formal_mean")
        reasons.append(
            f"{label}: vs logistic Δmacro={dm_log:+.4f}, Δformal={df_log:+.4f}; "
            f"vs HGB Δmacro={dm_hgb:+.4f}, Δformal={df_hgb:+.4f}."
        )

    meaningful_a = (
        beats(a_row, log_row, "macro_f1_mean") >= 0.01
        or beats(a_row, log_row, "f1_formal_mean") >= 0.01
        or beats(a_row, hgb_row, "macro_f1_mean") >= 0.01
    ) and beats(a_row, log_row, "f1_formal_mean") >= -0.005
    meaningful_b = (
        beats(b_row, log_row, "macro_f1_mean") >= 0.01
        or beats(b_row, log_row, "f1_formal_mean") >= 0.01
        or beats(b_row, hgb_row, "macro_f1_mean") >= 0.01
    ) and beats(b_row, log_row, "f1_formal_mean") >= -0.005

    if meaningful_b and b_row["macro_f1_mean"] >= a_row["macro_f1_mean"]:
        selected = "stack_b_logistic_hgb_rf"
        reasons.append("Stack B meets the meaningful-improvement bar and is preferred over Stack A.")
    elif meaningful_a:
        selected = "stack_a_logistic_hgb"
        reasons.append("Stack A meets the meaningful-improvement bar.")
    else:
        reasons.append(
            "Neither stack shows a meaningful, non-trade-off improvement over individual "
            "logistic / HGB on this 5-fold protocol. Retain financial-expanded logistic regression."
        )

    total_sec = time.perf_counter() - t_all
    sa_agree = float(
        comp.loc[
            (comp["stack"] == "stack_a_logistic_hgb")
            & (comp["metric"] == "fraction_all_bases_agree"),
            "value",
        ].iloc[0]
    )
    sb_agree = float(
        comp.loc[
            (comp["stack"] == "stack_b_logistic_hgb_rf")
            & (comp["metric"] == "fraction_all_bases_agree"),
            "value",
        ].iloc[0]
    )

    report = f"""# Final stacking assessment

**Status:** Script 14 complete. Notebook and deployment application **not** modified.  
**Feature set:** financial-expanded matrix, **93** predictors (no agricultural segment).  
**Validation:** **single** stratified 5-fold external CV — the **first repeat** of `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)`.  
**Important:** all comparisons below are **5-fold means only**. They are **not** compared to the previous 25-fold averages.

---

## Verified vs recommendation

| Type | Content |
|---|---|
| **Verified** | 5-fold metrics for Stack A/B and reused Script 13 individual metrics on folds 0–4; complementarity rates; runtimes |
| **Recommendation** | Whether to stack or retain financial logistic for deployment |

---

## Protocol

- Base learners emit `predict_proba`; `passthrough=False`.
- Meta-learner: class-weighted multinomial logistic regression.
- Inner CV for stacking features: stratified 5-fold (`shuffle=True`, seed 42) **within each external training fold**.
- External test fold never fits bases or meta-learner.
- Preprocessing stays inside each estimator pipeline (same as Scripts 07/13).
- Individual logistic / RF / HGB metrics on these five folds reused from `13_final_fold_metrics.csv` (folds 0–4).

---

## 5-fold scorecard (verified)

| Experiment | Macro F1 (sd) | Formal F1 (sd) | Balanced acc | Weighted F1 | Seconds |
|---|---:|---:|---:|---:|---:|
| Logistic (financial) | {log_row['macro_f1_mean']:.4f} ({log_row['macro_f1_std']:.4f}) | {log_row['f1_formal_mean']:.4f} ({log_row['f1_formal_std']:.4f}) | {log_row['balanced_accuracy_mean']:.4f} | {log_row['weighted_f1_mean']:.4f} | — |
| HGB (financial) | {hgb_row['macro_f1_mean']:.4f} ({hgb_row['macro_f1_std']:.4f}) | {hgb_row['f1_formal_mean']:.4f} ({hgb_row['f1_formal_std']:.4f}) | {hgb_row['balanced_accuracy_mean']:.4f} | {hgb_row['weighted_f1_mean']:.4f} | — |
| RF (financial) | {rf_row['macro_f1_mean']:.4f} ({rf_row['macro_f1_std']:.4f}) | {rf_row['f1_formal_mean']:.4f} ({rf_row['f1_formal_std']:.4f}) | {rf_row['balanced_accuracy_mean']:.4f} | {rf_row['weighted_f1_mean']:.4f} | — |
| **Stack A** (logit+HGB) | {a_row['macro_f1_mean']:.4f} ({a_row['macro_f1_std']:.4f}) | {a_row['f1_formal_mean']:.4f} ({a_row['f1_formal_std']:.4f}) | {a_row['balanced_accuracy_mean']:.4f} | {a_row['weighted_f1_mean']:.4f} | {sec_a:.1f} |
| **Stack B** (logit+HGB+RF) | {b_row['macro_f1_mean']:.4f} ({b_row['macro_f1_std']:.4f}) | {b_row['f1_formal_mean']:.4f} ({b_row['f1_formal_std']:.4f}) | {b_row['balanced_accuracy_mean']:.4f} | {b_row['weighted_f1_mean']:.4f} | {sec_b:.1f} |

Chart: `outputs/figures/14_stacking_comparison.png`.

### Class-wise formal focus (5-fold means)

| Experiment | Formal precision | Formal recall | Formal F1 |
|---|---:|---:|---:|
| Logistic | {log_row['precision_formal_mean']:.4f} | {log_row['recall_formal_mean']:.4f} | {log_row['f1_formal_mean']:.4f} |
| HGB | {hgb_row['precision_formal_mean']:.4f} | {hgb_row['recall_formal_mean']:.4f} | {hgb_row['f1_formal_mean']:.4f} |
| Stack A | {a_row['precision_formal_mean']:.4f} | {a_row['recall_formal_mean']:.4f} | {a_row['f1_formal_mean']:.4f} |
| Stack B | {b_row['precision_formal_mean']:.4f} | {b_row['recall_formal_mean']:.4f} | {b_row['f1_formal_mean']:.4f} |

---

## Base-learner complementarity (verified)

On external test predictions from the fitted base estimators inside each stacking fold:

| Stack | Fraction where all bases agree |
|---|---:|
| Stack A | {sa_agree:.3f} |
| Stack B | {sb_agree:.3f} |

Full table: `14_base_learner_complementarity.csv`. High agreement implies limited stacking headroom; disagreement cases show whether the meta-learner leans toward logistic or HGB.

---

## Model selection

### Does stacking help?

{chr(10).join('- ' + r for r in reasons)}

### Final deployment recommendation

**Selected model: `{selected}`**

| Metric | Value (5-fold protocol) |
|---|---|
| Feature set | Financial expansion (93 predictors) |
| Algorithm | Class-weighted multinomial logistic regression (unless a stack was selected above) |
| Macro F1 | see scorecard row for selected experiment |
| Formal F1 | see scorecard row for selected experiment |

Stacking was **not** selected automatically based on highest macro F1. Selection required a meaningful gain without a material formal-class trade-off.

---

## Limitations

- This assessment uses **5 external folds only** (first repeat). Do not equate these means with the prior 25-fold averages.
- No hyperparameter tuning; Script 07/13 base configs reused.
- Agricultural segment membership excluded by design.
- Stacking CMs saved for Stack A/B; individual models contribute metrics from Script 13 on the same folds (confusion matrices for individuals not regenerated here).
- Total Script 14 wall time: **{total_sec:.1f}s** (Stack A {sec_a:.1f}s; Stack B {sec_b:.1f}s).

---

## Outputs

| File | Description |
|---|---|
| `14_stacking_results.csv` | 5-fold scorecard |
| `14_stacking_fold_metrics.csv` | Fold-level metrics (individuals + stacks) |
| `14_stacking_confusion_matrices.csv` | Stack A/B confusion matrices |
| `14_stacking_comparison.png` | Macro F1 chart |
| `14_base_learner_complementarity.csv` | Base agreement diagnostics |
| `14_stacking_fold_deltas.csv` | Fold deltas vs logistic / HGB |
| `FINAL_STACKING_ASSESSMENT.md` | This report |
"""

    (REPORTS / "FINAL_STACKING_ASSESSMENT.md").write_text(report, encoding="utf-8")
    print("\nScorecard (5-fold):")
    print(
        score[
            [
                "experiment",
                "macro_f1_mean",
                "f1_formal_mean",
                "balanced_accuracy_mean",
                "seconds_total",
            ]
        ].to_string(index=False)
    )
    print(f"\nSELECTED: {selected}")
    print(f"Total time: {total_sec:.1f}s")
    print("Script 14 complete.")


if __name__ == "__main__":
    main()
