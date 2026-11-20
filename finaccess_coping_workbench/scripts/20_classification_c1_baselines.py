#!/usr/bin/env python3
"""C1 classification baselines on the locked four-class primary track.

Reads Script 19 protocol and shared folds (does not regenerate them).
Fits Dummy and multinomial logistic regression under unweighted vs
class-weighted learning and acres-exclude vs train-fold median imputation.
No three-class, extended, tree, or boosting models in this script.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
CONTINUOUS_BASE = ["age_years", "education", "n_crop_types", "n_livestock_types"]
LR_MAX_ITER = 2000
LR_C = 1.0
LR_SOLVER = "lbfgs"


def load_locked_protocol() -> dict:
    path = TABLES_DIR / "19_classification_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    cv = protocol["cv"]
    if cv["n_splits"] != 5 or cv["n_repeats"] != 5:
        raise ValueError(f"Unexpected CV settings in protocol: {cv}")
    if protocol["tracks"]["primary"]["target"] != "coping_category":
        raise ValueError("Primary track target mismatch")
    return protocol


def load_shared_folds() -> pd.DataFrame:
    folds = pd.read_csv(TABLES_DIR / "19_cv_folds_long.csv")
    expected_rows = EXPECTED_DROUGHT_N * 5 * 5
    if len(folds) != expected_rows:
        raise ValueError(f"Unexpected fold rows={len(folds)}; expected {expected_rows}")
    return folds


def predictor_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ID_VARS]


def continuous_columns(feature_cols: list[str], land_treatment: str) -> list[str]:
    cols = [c for c in CONTINUOUS_BASE if c in feature_cols]
    if land_treatment == "impute_median" and LAND_SIZE_STANDARDISED in feature_cols:
        cols.append(LAND_SIZE_STANDARDISED)
    return cols


def build_estimator(
    feature_cols: list[str],
    *,
    model_name: str,
    class_weight_mode: str,
    land_treatment: str,
):
    """Build sklearn estimator / pipeline for one C1 configuration."""
    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")

    if model_name != "multinomial_logistic":
        raise ValueError(f"Unknown model {model_name}")

    cont = continuous_columns(feature_cols, land_treatment)
    binary = [c for c in feature_cols if c not in cont]

    transformers = []
    if cont:
        cont_steps = []
        if land_treatment == "impute_median" and LAND_SIZE_STANDARDISED in cont:
            cont_steps.append(("imputer", SimpleImputer(strategy="median")))
        cont_steps.append(("scaler", StandardScaler()))
        transformers.append(("continuous", Pipeline(cont_steps), cont))
    if binary:
        transformers.append(("binary", "passthrough", binary))

    pre = ColumnTransformer(transformers, remainder="drop")
    # sklearn >=1.8: multinomial logistic is the default multiclass behaviour
    clf = LogisticRegression(
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        C=LR_C,
        class_weight="balanced" if class_weight_mode == "class_weighted" else None,
        random_state=42,
    )
    return Pipeline([("preprocess", pre), ("clf", clf)])


def experiment_grid() -> list[dict]:
    experiments = [
        {
            "experiment_id": "dummy_most_frequent",
            "model_name": "dummy_most_frequent",
            "class_weight_mode": "none",
            "land_treatment": "exclude",
        }
    ]
    for land in ("exclude", "impute_median"):
        for cw in ("unweighted", "class_weighted"):
            experiments.append(
                {
                    "experiment_id": f"logistic_{cw}_{land}",
                    "model_name": "multinomial_logistic",
                    "class_weight_mode": cw,
                    "land_treatment": land,
                }
            )
    return experiments


def feature_set_for(land_treatment: str, all_cols: list[str]) -> list[str]:
    cols = list(all_cols)
    if land_treatment == "exclude":
        cols = [c for c in cols if c != LAND_SIZE_STANDARDISED]
    return cols


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = FOUR_ORDER
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
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
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.DataFrame,
    id_to_pos: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_cols = predictor_columns(X)
    feature_cols = feature_set_for(exp["land_treatment"], all_cols)
    X_mat = X[feature_cols]

    fold_metric_rows: list[dict] = []
    fold_cm_rows: list[dict] = []
    warning_rows: list[dict] = []
    oof_rows: list[dict] = []

    split_ids = sorted(folds["split_id"].unique())
    for split_id in split_ids:
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

        est = build_estimator(
            feature_cols,
            model_name=exp["model_name"],
            class_weight_mode=exp["class_weight_mode"],
            land_treatment=exp["land_treatment"],
        )

        fit_ok = True
        warn_msgs: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                est.fit(X_train, y_train)
            except Exception as exc:  # noqa: BLE001 — record failed fits
                fit_ok = False
                warn_msgs.append(f"FIT_FAILED: {type(exc).__name__}: {exc}")
            for w in caught:
                warn_msgs.append(f"{w.category.__name__}: {w.message}")

        for msg in warn_msgs:
            warning_rows.append(
                {
                    "experiment_id": exp["experiment_id"],
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "fit_ok": fit_ok,
                    "message": str(msg),
                }
            )

        if not fit_ok:
            continue

        y_pred = est.predict(X_test)
        m = metrics_from_preds(y_test.to_numpy(), y_pred)
        fold_metric_rows.append(
            {
                "experiment_id": exp["experiment_id"],
                "model_name": exp["model_name"],
                "class_weight_mode": exp["class_weight_mode"],
                "land_treatment": exp["land_treatment"],
                "n_features": len(feature_cols),
                "repeat": repeat,
                "fold": fold,
                "split_id": split_id,
                "fit_ok": True,
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
    metric_cols = [
        c
        for c in fold_metrics.columns
        if c
        not in {
            "experiment_id",
            "model_name",
            "class_weight_mode",
            "land_treatment",
            "n_features",
            "repeat",
            "fold",
            "split_id",
            "fit_ok",
        }
        and not c.startswith("support_")
    ]
    rows = []
    for exp_id, g in fold_metrics.groupby("experiment_id", sort=False):
        meta = g.iloc[0]
        row = {
            "experiment_id": exp_id,
            "model_name": meta["model_name"],
            "class_weight_mode": meta["class_weight_mode"],
            "land_treatment": meta["land_treatment"],
            "n_features": int(meta["n_features"]),
            "n_folds_evaluated": len(g),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = round(float(g[col].mean()), 4)
            row[f"{col}_std"] = round(float(g[col].std(ddof=0)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def repetition_confusion(fold_cm: pd.DataFrame) -> pd.DataFrame:
    """Sum five fold matrices within each repetition → one OOF matrix per repeat."""
    if fold_cm.empty:
        return fold_cm
    g = (
        fold_cm.groupby(
            ["experiment_id", "repeat", "true_label", "pred_label"], as_index=False
        )["n"]
        .sum()
    )
    g = g.rename(columns={"n": "n_summed_across_folds"})
    return g


def mean_repetition_confusion(rep_cm: pd.DataFrame) -> pd.DataFrame:
    if rep_cm.empty:
        return rep_cm
    g = (
        rep_cm.groupby(
            ["experiment_id", "true_label", "pred_label"], as_index=False
        )["n_summed_across_folds"]
        .agg(n_mean_across_repeats="mean", n_min_across_repeats="min", n_max_across_repeats="max")
    )
    for col in ("n_mean_across_repeats",):
        g[col] = g[col].round(2)
    return g


def cm_matrix_string(rep_cm: pd.DataFrame, experiment_id: str, repeat: int | None) -> str:
    if repeat is None:
        sub = (
            rep_cm.loc[rep_cm["experiment_id"] == experiment_id]
            .groupby(["true_label", "pred_label"], as_index=False)["n_mean_across_repeats"]
            .first()
            if "n_mean_across_repeats" in rep_cm.columns
            else pd.DataFrame()
        )
        value_col = "n_mean_across_repeats"
    else:
        sub = rep_cm.loc[
            (rep_cm["experiment_id"] == experiment_id) & (rep_cm["repeat"] == repeat)
        ]
        value_col = "n_summed_across_folds"
    if sub.empty:
        return "(empty)"
    mat = (
        sub.pivot(index="true_label", columns="pred_label", values=value_col)
        .reindex(index=FOUR_ORDER, columns=FOUR_ORDER)
        .fillna(0)
    )
    return mat.to_string()


def main() -> None:
    ensure_output_dirs()
    protocol = load_locked_protocol()
    folds = load_shared_folds()

    X = pd.read_parquet(PREPARED_DIR / "09_X_primary_classification.parquet")
    y_frame = pd.read_parquet(PREPARED_DIR / "09_y_targets.parquet")
    if len(X) != EXPECTED_DROUGHT_N:
        raise ValueError("Unexpected primary matrix length")
    if not X["respondent_id"].equals(y_frame["respondent_id"]):
        # align on id if order differs
        y_frame = y_frame.set_index("respondent_id").loc[X["respondent_id"]].reset_index()
    y = y_frame["coping_category"]
    if y.isna().any():
        raise ValueError("Missing four-class labels")

    id_to_pos = {rid: i for i, rid in enumerate(X["respondent_id"].tolist())}
    # Verify fold ids exist
    missing = set(folds["respondent_id"]) - set(id_to_pos)
    if missing:
        raise ValueError(f"Fold ids missing from matrix: {len(missing)}")

    all_fold_metrics = []
    all_fold_cm = []
    all_warnings = []
    all_oof = []

    for exp in experiment_grid():
        print(f"Running {exp['experiment_id']} ...")
        fm, fcm, warns, oof = run_experiment(exp, X, y, folds, id_to_pos)
        all_fold_metrics.append(fm)
        all_fold_cm.append(fcm)
        all_warnings.append(warns)
        all_oof.append(oof)
        print(f"  folds={len(fm)} warnings={len(warns)}")

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    fold_cm = pd.concat(all_fold_cm, ignore_index=True)
    warn_df = pd.concat(all_warnings, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)
    scorecard = summarise_fold_metrics(fold_metrics)
    rep_cm = repetition_confusion(fold_cm)
    mean_cm = mean_repetition_confusion(rep_cm)

    outputs = {
        "20_c1_fold_metrics.csv": fold_metrics,
        "20_c1_scorecard.csv": scorecard,
        "20_c1_fold_confusion.csv": fold_cm,
        "20_c1_repetition_confusion.csv": rep_cm,
        "20_c1_mean_repetition_confusion.csv": mean_cm,
        "20_c1_fit_warnings.csv": warn_df,
        "20_c1_oof_predictions.csv": oof_df,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    run_meta = {
        "script": "20_classification_c1_baselines",
        "reads_protocol": "19_classification_protocol.json",
        "reads_folds": "19_cv_folds_long.csv",
        "regenerates_folds": False,
        "track": "primary_four_class",
        "household_weight_in_fitting": False,
        "household_weight_in_metrics": False,
        "class_weight_modes": ["unweighted", "class_weighted"],
        "land_treatments": ["exclude", "impute_median"],
        "continuous_scaled": CONTINUOUS_BASE + [LAND_SIZE_STANDARDISED],
        "logistic": {
            "multi_class": "multinomial_default_sklearn_ge_1_8",
            "solver": LR_SOLVER,
            "C": LR_C,
            "max_iter": LR_MAX_ITER,
        },
        "n_experiments": len(experiment_grid()),
        "protocol_seed": protocol["random_seed"],
    }
    meta_path = TABLES_DIR / "20_c1_run_config.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")

    # Report
    score_view = scorecard[
        [
            "experiment_id",
            "n_features",
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

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    warn_n = len(warn_df)
    conv_n = int(warn_df["message"].str.contains("Convergence", na=False).sum()) if len(warn_df) else 0
    fail_n = int((~fold_metrics["fit_ok"]).sum()) if "fit_ok" in fold_metrics.columns else 0

    # Best logistic by macro F1 mean for spotlight CMs
    logistic_card = scorecard.loc[scorecard["model_name"] == "multinomial_logistic"]
    best_id = logistic_card.sort_values("macro_f1_mean", ascending=False).iloc[0][
        "experiment_id"
    ]
    dummy_id = "dummy_most_frequent"

    report = f"""# C1 Classification Baselines — Four-class Primary (Script 20)

Generated: {timestamp}

## Scope

- Reads locked Script 19 protocol and `19_cv_folds_long.csv` (**folds not regenerated**)
- Target: four-class `coping_category` only
- Predictors: primary matrix; three-class and extended tracks deferred
- `household_weight` not used in fitting or evaluation metrics
- Class weighting compared separately from survey weights

## Experiments

| experiment_id | model | class_weight | land_size |
|---|---|---|---|
| dummy_most_frequent | Dummy (most_frequent) | none | exclude |
| logistic_unweighted_exclude | Multinomial logistic | unweighted | exclude |
| logistic_class_weighted_exclude | Multinomial logistic | class_weighted | exclude |
| logistic_unweighted_impute_median | Multinomial logistic | unweighted | train-fold median impute |
| logistic_class_weighted_impute_median | Multinomial logistic | class_weighted | train-fold median impute |

Preprocessing (logistic): continuous columns (`age_years`, `education`, `n_crop_types`, `n_livestock_types`, and acres when retained) median-imputed if needed and standardised **within each training fold**; binary/one-hot columns passed through; `C={LR_C}`, solver=`{LR_SOLVER}`, `max_iter={LR_MAX_ITER}`.

## Scorecard (mean ± variability across 25 test folds)

Means and standard deviations are descriptive summaries of the 25 fold scores. The folds are not independent samples for significance testing.

```
{score_view.to_string(index=False)}
```

## Fit diagnostics

- Warning messages recorded: {warn_n} (Convergence-related: {conv_n})
- Failed fits in fold metrics table: {fail_n}

## Confusion matrices

Within each repetition, fold-level matrices are summed to one complete out-of-fold matrix. Below: mean cell counts across the five repetitions.

### Dummy (`{dummy_id}`)

```
{cm_matrix_string(mean_cm, dummy_id, None)}
```

### Best logistic by mean macro F1 (`{best_id}`)

```
{cm_matrix_string(mean_cm, best_id, None)}
```

## Interpretation notes

- Macro F1 is the primary comparison metric; balanced accuracy and class-wise F1 (especially formal) are required reading.
- Formal support is 15 per test fold; formal metrics are expected to be noisy — inspect `macro_f1_std` and `f1_formal_std`.
- Class-weighted learning is not assumed to improve results; compare against unweighted directly.
- Land-size exclude vs impute is a sensitivity within the same folds.

## Deferred

- Three-class secondary target
- Extended retrospective matrix
- Decision tree / random forest / gradient boosting (C2–C3)

## Output files

- `outputs/tables/20_c1_run_config.json`
- `outputs/tables/20_c1_fold_metrics.csv`
- `outputs/tables/20_c1_scorecard.csv`
- `outputs/tables/20_c1_fold_confusion.csv`
- `outputs/tables/20_c1_repetition_confusion.csv`
- `outputs/tables/20_c1_mean_repetition_confusion.csv`
- `outputs/tables/20_c1_fit_warnings.csv`
- `outputs/tables/20_c1_oof_predictions.csv`
"""
    path = REPORTS_DIR / "20_classification_c1_baselines.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")

    design = LOGS_DIR / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = design.read_text(encoding="utf-8")
    text = text.replace(
        "**Status: Classification design accepted (Script 19). Clarifications on survey vs class weights and repeated-CV summarisation locked. C1 baseline fitting in progress / next.**",
        "**Status: C1 baselines completed (Script 20). Four-class primary dummy + multinomial logistic scorecards available. C2–C3 and secondary/extended tracks deferred.**",
    )
    note = """

## 15. C1 baselines (Script 20)

Four-class primary C1 completed using Script 19 folds (not regenerated). Dummy (most_frequent) and multinomial logistic regression compared under unweighted vs class-weighted learning and acres exclude vs train-fold median imputation. Household weights not used in fitting or metrics. Three-class and extended tracks deferred.

Report: `outputs/reports/20_classification_c1_baselines.md`
"""
    if "## 15. C1 baselines (Script 20)" not in text:
        text = text.rstrip() + note
    design.write_text(text, encoding="utf-8")
    print(f"Updated {design}")

    print("\n=== Scorecard ===")
    print(score_view.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
