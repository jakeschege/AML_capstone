#!/usr/bin/env python3
"""C2/C3 nonlinear four-class primary experiments (Script 21).

Decision tree, random forest, and histogram gradient boosting under the
locked Script 19 folds. Default land treatment: acres excluded.
Compares unweighted vs class-weighted learning. No household survey weights,
no SMOTE, no extensive hyperparameter tuning. Three-class and extended
tracks deferred.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    EXPECTED_DROUGHT_N,
    ID_VARS,
    LAND_SIZE_STANDARDISED,
    LOGS_DIR,
    PREPARED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)

FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]
RANDOM_SEED = 42
LAND_TREATMENT = "exclude"

# Default-ish settings; no grid search
DT_PARAMS = {"random_state": RANDOM_SEED}
RF_PARAMS = {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": -1}
HGB_PARAMS = {"max_iter": 200, "random_state": RANDOM_SEED, "early_stopping": False}


def load_locked_protocol() -> dict:
    path = TABLES_DIR / "19_classification_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    cv = protocol["cv"]
    if cv["n_splits"] != 5 or cv["n_repeats"] != 5:
        raise ValueError(f"Unexpected CV settings in protocol: {cv}")
    return protocol


def load_shared_folds() -> pd.DataFrame:
    folds = pd.read_csv(TABLES_DIR / "19_cv_folds_long.csv")
    expected_rows = EXPECTED_DROUGHT_N * 5 * 5
    if len(folds) != expected_rows:
        raise ValueError(f"Unexpected fold rows={len(folds)}; expected {expected_rows}")
    return folds


def predictor_columns(X: pd.DataFrame) -> list[str]:
    cols = [c for c in X.columns if c not in ID_VARS]
    return [c for c in cols if c != LAND_SIZE_STANDARDISED]


def experiment_grid() -> list[dict]:
    models = [
        ("decision_tree", "C2"),
        ("random_forest", "C3"),
        ("hist_gradient_boosting", "C3"),
    ]
    experiments = []
    for model_name, stage in models:
        for cw in ("unweighted", "class_weighted"):
            experiments.append(
                {
                    "experiment_id": f"{model_name}_{cw}_exclude",
                    "stage": stage,
                    "model_name": model_name,
                    "class_weight_mode": cw,
                    "land_treatment": LAND_TREATMENT,
                    # All three estimators support class_weight natively in sklearn 1.9
                    "class_weight_mechanism": "estimator_class_weight",
                }
            )
    return experiments


def build_estimator(model_name: str, class_weight_mode: str):
    cw = "balanced" if class_weight_mode == "class_weighted" else None
    if model_name == "decision_tree":
        return DecisionTreeClassifier(class_weight=cw, **DT_PARAMS)
    if model_name == "random_forest":
        return RandomForestClassifier(class_weight=cw, **RF_PARAMS)
    if model_name == "hist_gradient_boosting":
        # Native class_weight support — no sample_weight workaround required
        return HistGradientBoostingClassifier(class_weight=cw, **HGB_PARAMS)
    raise ValueError(f"Unknown model {model_name}")


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = FOUR_ORDER
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, average="weighted", labels=labels, zero_division=0
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for i, cat in enumerate(labels):
        out[f"precision_{cat}"] = float(prec[i])
        out[f"recall_{cat}"] = float(rec[i])
        out[f"f1_{cat}"] = float(f1[i])
        out[f"support_{cat}"] = int(support[i])
    return out


def run_experiment(
    exp: dict,
    X_mat: pd.DataFrame,
    y: pd.Series,
    folds: pd.DataFrame,
    id_to_pos: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_metric_rows: list[dict] = []
    fold_cm_rows: list[dict] = []
    warning_rows: list[dict] = []
    oof_rows: list[dict] = []

    for split_id in sorted(folds["split_id"].unique()):
        part = folds.loc[folds["split_id"] == split_id]
        train_ids = part.loc[part["split"] == "train", "respondent_id"]
        test_ids = part.loc[part["split"] == "test", "respondent_id"]
        train_idx = train_ids.map(id_to_pos).to_numpy()
        test_idx = test_ids.map(id_to_pos).to_numpy()
        repeat = int(part["repeat"].iloc[0])
        fold = int(part["fold"].iloc[0])

        X_train = X_mat.iloc[train_idx]
        X_test = X_mat.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        est = build_estimator(exp["model_name"], exp["class_weight_mode"])
        fit_ok = True
        warn_msgs: list[str] = []
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                est.fit(X_train, y_train)
            except Exception as exc:  # noqa: BLE001
                fit_ok = False
                warn_msgs.append(f"FIT_FAILED: {type(exc).__name__}: {exc}")
            for w in caught:
                warn_msgs.append(f"{w.category.__name__}: {w.message}")
        train_seconds = time.perf_counter() - t0

        for msg in warn_msgs:
            warning_rows.append(
                {
                    "experiment_id": exp["experiment_id"],
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "fit_ok": fit_ok,
                    "train_seconds": round(train_seconds, 4),
                    "message": str(msg),
                }
            )

        if not fit_ok:
            fold_metric_rows.append(
                {
                    "experiment_id": exp["experiment_id"],
                    "stage": exp["stage"],
                    "model_name": exp["model_name"],
                    "class_weight_mode": exp["class_weight_mode"],
                    "land_treatment": exp["land_treatment"],
                    "class_weight_mechanism": exp["class_weight_mechanism"],
                    "n_features": X_mat.shape[1],
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "fit_ok": False,
                    "train_seconds": round(train_seconds, 4),
                }
            )
            continue

        y_pred = est.predict(X_test)
        m = metrics_from_preds(y_test.to_numpy(), y_pred)
        fold_metric_rows.append(
            {
                "experiment_id": exp["experiment_id"],
                "stage": exp["stage"],
                "model_name": exp["model_name"],
                "class_weight_mode": exp["class_weight_mode"],
                "land_treatment": exp["land_treatment"],
                "class_weight_mechanism": exp["class_weight_mechanism"],
                "n_features": X_mat.shape[1],
                "repeat": repeat,
                "fold": fold,
                "split_id": split_id,
                "fit_ok": True,
                "train_seconds": round(train_seconds, 4),
                **m,
            }
        )

        cm = confusion_matrix(y_test, y_pred, labels=FOUR_ORDER)
        for i, true_lab in enumerate(FOUR_ORDER):
            for j, pred_lab in enumerate(FOUR_ORDER):
                fold_cm_rows.append(
                    {
                        "experiment_id": exp["experiment_id"],
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": split_id,
                        "true_label": true_lab,
                        "pred_label": pred_lab,
                        "n": int(cm[i, j]),
                    }
                )

        for rid, yt, yp in zip(
            test_ids.to_numpy(), y_test.to_numpy(), y_pred, strict=True
        ):
            oof_rows.append(
                {
                    "experiment_id": exp["experiment_id"],
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "respondent_id": rid,
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    return (
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(fold_cm_rows),
        pd.DataFrame(warning_rows),
        pd.DataFrame(oof_rows),
    )


def summarise_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    skip = {
        "experiment_id",
        "stage",
        "model_name",
        "class_weight_mode",
        "land_treatment",
        "class_weight_mechanism",
        "n_features",
        "repeat",
        "fold",
        "split_id",
        "fit_ok",
        "train_seconds",
    }
    metric_cols = [
        c
        for c in fold_metrics.columns
        if c not in skip and not c.startswith("support_")
    ]
    rows = []
    ok = fold_metrics.loc[fold_metrics["fit_ok"] == True]  # noqa: E712
    for exp_id, g in ok.groupby("experiment_id", sort=False):
        meta = g.iloc[0]
        row = {
            "experiment_id": exp_id,
            "stage": meta["stage"],
            "model_name": meta["model_name"],
            "class_weight_mode": meta["class_weight_mode"],
            "land_treatment": meta["land_treatment"],
            "class_weight_mechanism": meta["class_weight_mechanism"],
            "n_features": int(meta["n_features"]),
            "n_folds_evaluated": len(g),
            "train_seconds_mean": round(float(g["train_seconds"].mean()), 4),
            "train_seconds_total": round(float(g["train_seconds"].sum()), 4),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = round(float(g[col].mean()), 4)
            row[f"{col}_std"] = round(float(g[col].std(ddof=0)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def repetition_confusion(fold_cm: pd.DataFrame) -> pd.DataFrame:
    if fold_cm.empty:
        return fold_cm
    return (
        fold_cm.groupby(
            ["experiment_id", "repeat", "true_label", "pred_label"], as_index=False
        )["n"]
        .sum()
        .rename(columns={"n": "n_summed_across_folds"})
    )


def mean_repetition_confusion(rep_cm: pd.DataFrame) -> pd.DataFrame:
    if rep_cm.empty:
        return rep_cm
    g = rep_cm.groupby(
        ["experiment_id", "true_label", "pred_label"], as_index=False
    )["n_summed_across_folds"].agg(
        n_mean_across_repeats="mean",
        n_min_across_repeats="min",
        n_max_across_repeats="max",
    )
    g["n_mean_across_repeats"] = g["n_mean_across_repeats"].round(2)
    return g


def cm_matrix_string(mean_cm: pd.DataFrame, experiment_id: str) -> str:
    sub = mean_cm.loc[mean_cm["experiment_id"] == experiment_id]
    if sub.empty:
        return "(empty)"
    mat = (
        sub.pivot(
            index="true_label",
            columns="pred_label",
            values="n_mean_across_repeats",
        )
        .reindex(index=FOUR_ORDER, columns=FOUR_ORDER)
        .fillna(0)
    )
    return mat.to_string()


def load_c1_benchmark() -> pd.DataFrame:
    path = TABLES_DIR / "20_c1_scorecard.csv"
    c1 = pd.read_csv(path)
    keep = [
        "dummy_most_frequent",
        "logistic_unweighted_exclude",
        "logistic_class_weighted_exclude",
    ]
    return c1.loc[c1["experiment_id"].isin(keep)].copy()


def main() -> None:
    ensure_output_dirs()
    protocol = load_locked_protocol()
    folds = load_shared_folds()

    X_full = pd.read_parquet(PREPARED_DIR / "09_X_primary_classification.parquet")
    y_frame = pd.read_parquet(PREPARED_DIR / "09_y_targets.parquet")
    if len(X_full) != EXPECTED_DROUGHT_N:
        raise ValueError("Unexpected primary matrix length")
    if not X_full["respondent_id"].equals(y_frame["respondent_id"]):
        y_frame = (
            y_frame.set_index("respondent_id").loc[X_full["respondent_id"]].reset_index()
        )
    y = y_frame["coping_category"]
    if y.isna().any():
        raise ValueError("Missing four-class labels")

    feature_cols = predictor_columns(X_full)
    X_mat = X_full[feature_cols]
    if X_mat.isna().any().any():
        raise ValueError("Unexpected missing values after excluding land_size_acres")

    id_to_pos = {rid: i for i, rid in enumerate(X_full["respondent_id"].tolist())}

    all_fold_metrics = []
    all_fold_cm = []
    all_warnings = []
    all_oof = []

    for exp in experiment_grid():
        print(f"Running {exp['experiment_id']} ...", flush=True)
        fm, fcm, warns, oof = run_experiment(exp, X_mat, y, folds, id_to_pos)
        all_fold_metrics.append(fm)
        all_fold_cm.append(fcm)
        all_warnings.append(warns)
        all_oof.append(oof)
        n_ok = int(fm["fit_ok"].sum()) if len(fm) else 0
        print(
            f"  folds_ok={n_ok} warnings={len(warns)} "
            f"train_s_total={fm['train_seconds'].sum():.1f}",
            flush=True,
        )

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    fold_cm = pd.concat(all_fold_cm, ignore_index=True)
    warn_df = pd.concat(all_warnings, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)
    scorecard = summarise_fold_metrics(fold_metrics)
    rep_cm = repetition_confusion(fold_cm)
    mean_cm = mean_repetition_confusion(rep_cm)

    # Combined view with C1 acres-exclude benchmarks
    c1 = load_c1_benchmark()
    c1_view = c1[
        [
            "experiment_id",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "weighted_f1_mean",
            "f1_formal_mean",
            "f1_formal_std",
            "f1_personal_mean",
            "f1_informal_mean",
            "f1_did_nothing_mean",
            "accuracy_mean",
        ]
    ].copy()
    c1_view.insert(1, "stage", "C1")
    c1_view.insert(2, "source", "script_20")

    c23_view = scorecard[
        [
            "experiment_id",
            "stage",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "weighted_f1_mean",
            "f1_formal_mean",
            "f1_formal_std",
            "f1_personal_mean",
            "f1_informal_mean",
            "f1_did_nothing_mean",
            "accuracy_mean",
            "train_seconds_mean",
            "train_seconds_total",
        ]
    ].copy()
    c23_view.insert(2, "source", "script_21")

    comparison = pd.concat(
        [
            c1_view,
            c23_view[
                [
                    "experiment_id",
                    "stage",
                    "source",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "balanced_accuracy_mean",
                    "weighted_f1_mean",
                    "f1_formal_mean",
                    "f1_formal_std",
                    "f1_personal_mean",
                    "f1_informal_mean",
                    "f1_did_nothing_mean",
                    "accuracy_mean",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    outputs = {
        "21_c2c3_fold_metrics.csv": fold_metrics,
        "21_c2c3_scorecard.csv": scorecard,
        "21_c2c3_fold_confusion.csv": fold_cm,
        "21_c2c3_repetition_confusion.csv": rep_cm,
        "21_c2c3_mean_repetition_confusion.csv": mean_cm,
        "21_c2c3_fit_warnings.csv": warn_df,
        "21_c2c3_oof_predictions.csv": oof_df,
        "21_c2c3_vs_c1_comparison.csv": comparison,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}", flush=True)

    run_meta = {
        "script": "21_classification_c2_c3_nonlinear",
        "reads_protocol": "19_classification_protocol.json",
        "reads_folds": "19_cv_folds_long.csv",
        "regenerates_folds": False,
        "track": "primary_four_class",
        "land_treatment_default": LAND_TREATMENT,
        "household_weight_in_fitting": False,
        "household_weight_in_metrics": False,
        "smote": False,
        "hyperparameter_search": False,
        "models": {
            "decision_tree": DT_PARAMS,
            "random_forest": RF_PARAMS,
            "hist_gradient_boosting": {
                **HGB_PARAMS,
                "class_weight_support": "native",
                "note": (
                    "HistGradientBoostingClassifier used as the gradient-boosting "
                    "estimator; class_weight is supported natively so training-fold "
                    "sample_weight balancing was not required."
                ),
            },
        },
        "n_features": len(feature_cols),
        "protocol_seed": protocol["random_seed"],
        "benchmark_experiments_from_c1": [
            "dummy_most_frequent",
            "logistic_unweighted_exclude",
            "logistic_class_weighted_exclude",
        ],
    }
    meta_path = TABLES_DIR / "21_c2c3_run_config.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}", flush=True)

    # Spotlight CMs: best C2/C3 by macro F1 and class-weighted RF/HGB
    best_id = scorecard.sort_values("macro_f1_mean", ascending=False).iloc[0][
        "experiment_id"
    ]
    cw_rf = "random_forest_class_weighted_exclude"
    cw_hgb = "hist_gradient_boosting_class_weighted_exclude"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    warn_n = len(warn_df)
    fail_n = int((fold_metrics["fit_ok"] == False).sum())  # noqa: E712

    report = f"""# C2/C3 Nonlinear Models — Four-class Primary (Script 21)

Generated: {timestamp}

## Scope

- Reads locked Script 19 protocol and folds (**not regenerated**)
- Target: four-class `coping_category`
- Predictors: primary matrix with **`land_size_acres` excluded** (default)
- Unweighted vs class-weighted learning
- No household survey weights in fitting or metrics; no SMOTE
- No extensive hyperparameter tuning
- Three-class secondary and extended retrospective tracks deferred

## Models

| Stage | Model | class_weight |
|---|---|---|
| C2 | Decision tree | native `class_weight` |
| C3 | Random forest (200 trees) | native `class_weight` |
| C3 | Hist gradient boosting (200 iterations) | native `class_weight` |

Gradient boosting uses `HistGradientBoostingClassifier`, which supports `class_weight` natively. Training-fold sample-weight balancing was therefore not required.

## Scorecard (mean ± fold sd across 25 test folds)

Fold standard deviations describe variation across repeated validation splits. They are **not** confidence intervals and must not be used for significance testing.

```
{c23_view.to_string(index=False)}
```

## Comparison with C1 acres-excluded benchmarks

```
{comparison.to_string(index=False)}
```

## Fit diagnostics

- Warning messages recorded: {warn_n}
- Failed fits: {fail_n}
- Per-experiment mean training seconds are in the scorecard (`train_seconds_mean`)

## Confusion matrices (mean cell counts across five repetition OOF matrices)

### Best C2/C3 by mean macro F1 (`{best_id}`)

```
{cm_matrix_string(mean_cm, best_id)}
```

### Random forest class-weighted (`{cw_rf}`)

```
{cm_matrix_string(mean_cm, cw_rf)}
```

### Hist gradient boosting class-weighted (`{cw_hgb}`)

```
{cm_matrix_string(mean_cm, cw_hgb)}
```

## Interpretation guide

Do not select a model solely for highest macro F1. Jointly inspect:

1. Macro F1 vs logistic benchmark
2. Balanced accuracy
3. Class-specific precision/recall (especially formal)
4. Confusion matrices
5. Fold-to-fold variation (`*_std`)
6. Minority-class detection vs false-positive trade-off

Formal metrics remain exploratory (15 test cases per fold).

## Deferred

- Three-class secondary target
- Extended retrospective matrix
- Extensive hyperparameter tuning / stacking

## Output files

- `outputs/tables/21_c2c3_run_config.json`
- `outputs/tables/21_c2c3_fold_metrics.csv`
- `outputs/tables/21_c2c3_scorecard.csv`
- `outputs/tables/21_c2c3_vs_c1_comparison.csv`
- `outputs/tables/21_c2c3_fold_confusion.csv`
- `outputs/tables/21_c2c3_repetition_confusion.csv`
- `outputs/tables/21_c2c3_mean_repetition_confusion.csv`
- `outputs/tables/21_c2c3_fit_warnings.csv`
- `outputs/tables/21_c2c3_oof_predictions.csv`
"""
    path = REPORTS_DIR / "21_classification_c2_c3_nonlinear.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}", flush=True)

    design = LOGS_DIR / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    text = text.replace(
        "**Status: C1 accepted. Acres-excluded is the default classification specification. C2/C3 four-class primary experiments next (or in progress). Three-class and extended tracks remain deferred.**",
        "**Status: C2/C3 four-class primary nonlinear experiments completed (Script 21). Acres-excluded default. Three-class and extended tracks remain deferred pending review.**",
    )
    note = """

## 16. C2/C3 nonlinear models (Script 21)

Four-class primary decision tree, random forest, and hist gradient boosting completed on Script 19 folds with acres excluded (default). Unweighted vs class-weighted compared; household weights unused; no SMOTE; no hyperparameter search. Logistic acres-excluded C1 results remain the benchmark. Three-class and extended tracks deferred.

Report: `outputs/reports/21_classification_c2_c3_nonlinear.md`
"""
    if "## 16. C2/C3 nonlinear models (Script 21)" not in text:
        text = text.rstrip() + note
    design.write_text(text, encoding="utf-8")
    print(f"Updated {design}", flush=True)

    print("\n=== C2/C3 scorecard ===", flush=True)
    print(c23_view.to_string(index=False), flush=True)
    print("\n=== vs C1 ===", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
