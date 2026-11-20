#!/usr/bin/env python3
"""Build SHAP artefacts for classifier v0.1.0 using LinearExplainer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

WB = ROOT / "finaccess_coping_workbench"
PREPARED = WB / "outputs" / "prepared"
TABLES = WB / "outputs" / "tables"
CLF_DIR = ROOT / "artefacts" / "classifier" / "v0.1.0"
OUT = ROOT / "artefacts" / "shap" / "v0.1.0"


def aggregate_feature_name(name: str) -> str:
    if name.startswith("county_"):
        return "county"
    if name.startswith("land_ownership_"):
        return "land_ownership"
    if name.startswith("has_"):
        return name
    return name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(CLF_DIR / "model.joblib")
    mapping = json.loads((CLF_DIR / "class_mapping.json").read_text())
    schema = json.loads((CLF_DIR / "feature_schema.json").read_text())
    feature_names = schema["feature_names"]
    classes = list(pipe.named_steps["clf"].classes_)

    X_full = pd.read_parquet(PREPARED / "09_X_primary_classification.parquet")
    y = pd.read_parquet(PREPARED / "09_y_targets.parquet")["coping_category"].astype(str)
    X = X_full[feature_names].copy()

    rng_bg = np.random.RandomState(42)
    rng_ex = np.random.RandomState(43)
    bg_idx = rng_bg.choice(len(X), size=min(100, len(X)), replace=False)
    # Stratified explain sample ~200
    explain_parts = []
    remaining = 200
    counts = y.value_counts()
    for cls, n_cls in counts.items():
        take = max(1, int(round(200 * (n_cls / len(y)))))
        take = min(take, int((y == cls).sum()), remaining)
        idx = rng_ex.choice(np.where(y == cls)[0], size=take, replace=False)
        explain_parts.append(idx)
        remaining -= take
    if remaining > 0:
        leftover = np.setdiff1d(np.arange(len(X)), np.concatenate(explain_parts))
        extra = rng_ex.choice(leftover, size=min(remaining, len(leftover)), replace=False)
        explain_parts.append(extra)
    ex_idx = np.concatenate(explain_parts)[:200]

    X_bg = X.iloc[bg_idx]
    X_ex = X.iloc[ex_idx]
    Xt_bg = pipe[:-1].transform(X_bg)
    Xt_ex = pipe[:-1].transform(X_ex)

    # Recover transformed feature names
    pre = pipe.named_steps["preprocess"]
    try:
        t_names = list(pre.get_feature_names_out())
    except Exception:
        t_names = feature_names

    clf = pipe.named_steps["clf"]
    explainer = shap.LinearExplainer(clf, Xt_bg)
    shap_values = explainer.shap_values(Xt_ex)

    # Normalise to (n, n_features, n_classes)
    if isinstance(shap_values, list):
        arr = np.stack(shap_values, axis=-1)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            # binary edge case — expand
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3 and arr.shape[1] == len(classes):
            # (n, n_classes, n_features) → transpose
            if arr.shape[2] == len(t_names):
                arr = np.transpose(arr, (0, 2, 1))

    if arr.shape[1] != len(t_names) and arr.shape[2] == len(t_names):
        arr = np.transpose(arr, (0, 2, 1))

    np.savez_compressed(
        OUT / "shap_values.npz",
        shap_values=arr,
        feature_names=np.array(t_names),
        classes=np.array(classes),
        row_indices=ex_idx,
        background_indices=bg_idx,
    )

    # Global mean |SHAP|
    mean_abs = np.mean(np.abs(arr), axis=0)  # (features, classes)
    global_abs = mean_abs.mean(axis=1)
    global_df = pd.DataFrame(
        {
            "feature": t_names,
            "mean_abs_shap": global_abs,
            "aggregated_feature": [aggregate_feature_name(f) for f in t_names],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_df.to_csv(OUT / "global_importance.csv", index=False)

    agg = (
        global_df.groupby("aggregated_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    agg.to_csv(OUT / "global_importance_aggregated.csv", index=False)

    # Class-specific
    rows = []
    for j, cls in enumerate(classes):
        for i, feat in enumerate(t_names):
            rows.append(
                {
                    "class": cls,
                    "feature": feat,
                    "aggregated_feature": aggregate_feature_name(feat),
                    "mean_abs_shap": float(mean_abs[i, j]),
                    "mean_shap": float(np.mean(arr[:, i, j])),
                }
            )
    class_df = pd.DataFrame(rows)
    class_df.to_csv(OUT / "class_specific_importance.csv", index=False)

    # Research figures
    try:
        plt.figure(figsize=(8, 6))
        top = agg.head(15)
        plt.barh(top["aggregated_feature"][::-1], top["mean_abs_shap"][::-1], color="#6B4F3A")
        plt.xlabel("mean |SHAP| (aggregated)")
        plt.title("Global feature importance (LinearExplainer)")
        plt.tight_layout()
        plt.savefig(OUT / "figures" / "global_importance_bar.png", dpi=120)
        plt.close()
    except Exception as exc:
        print(f"figure skipped: {exc}")

    # Dependence for top 5 aggregated — use first raw feature matching
    top5 = agg["aggregated_feature"].head(5).tolist()
    for agg_name in top5:
        raw_feats = [f for f in t_names if aggregate_feature_name(f) == agg_name]
        if not raw_feats:
            continue
        feat = raw_feats[0]
        fi = t_names.index(feat)
        try:
            plt.figure(figsize=(5, 4))
            # Use predicted class column 0 for illustration
            plt.scatter(Xt_ex[:, fi], arr[:, fi, 0], s=8, alpha=0.5, c="#B44A2B")
            plt.xlabel(feat)
            plt.ylabel(f"SHAP ({classes[0]})")
            plt.title(f"Dependence: {agg_name}")
            plt.tight_layout()
            plt.savefig(OUT / "figures" / f"dependence_{agg_name}.png", dpi=100)
            plt.close()
        except Exception:
            pass

    # Waterfall-ish bar examples: one per class
    examples = []
    for cls in classes:
        mask = y.iloc[ex_idx].values == cls
        if not mask.any():
            continue
        local_i = int(np.where(mask)[0][0])
        class_j = classes.index(cls)
        contrib = arr[local_i, :, class_j]
        order = np.argsort(-np.abs(contrib))[:12]
        fig_path = OUT / "figures" / f"waterfall_{cls}.png"
        plt.figure(figsize=(7, 5))
        plt.barh(
            [t_names[k] for k in order][::-1],
            contrib[order][::-1],
            color=["#2F5D50" if v >= 0 else "#B44A2B" for v in contrib[order][::-1]],
        )
        plt.title(f"Local SHAP example — true class {cls}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=100)
        plt.close()
        examples.append({"class": cls, "explain_row": int(ex_idx[local_i]), "figure": str(fig_path.name)})

    config = {
        "explainer": "shap.LinearExplainer",
        "background_n": int(len(bg_idx)),
        "background_seed": 42,
        "explain_n": int(len(ex_idx)),
        "explain_seed": 43,
        "stratified_by": "coping_category",
        "classes": classes,
        "display_labels": mapping.get("display_labels", {}),
        "caveat": "Attribution is not causation.",
        "waterfall_examples": examples,
        "model_version": "v0.1.0",
    }
    (OUT / "shap_run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote {OUT}; shap shape={arr.shape}")


if __name__ == "__main__":
    main()
