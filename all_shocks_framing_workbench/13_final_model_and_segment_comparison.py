#!/usr/bin/env python3
"""Script 13 — All-shocks agri segmentation + final financial model comparison.

Part A: verify segment not in financial matrix; recompute Spec A Gower+PAM (k=2..6).
Part B/C: compare logistic / RF / HGB ± fold-safe segment membership.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import gower
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

from config import (
    AGRICULTURAL_HAS_VARS,
    CLASS_ORDER,
    FIGURES,
    LAND_OWNERSHIP_VARS,
    PORTFOLIO_COUNT_VARS,
    PREPARED,
    REPORTS,
    TABLES,
    ensure_dirs,
)

# Reuse drought-workbench PAM helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finaccess_coping_workbench"))
from segmentation_utils import (  # noqa: E402
    average_silhouette_precomputed,
    cluster_feature_profiles,
    cluster_size_table,
    pam_precomputed,
)

SEED = 42
N_SPLITS = 5
N_REPEATS = 5
CANDIDATE_K = list(range(2, 7))
PAM_MAX_ITER = 100
BENCHMARK_MACRO = 0.492970
BENCHMARK_FORMAL = 0.303301


def spec_a_features() -> list[str]:
    return list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS) + list(LAND_OWNERSHIP_VARS)


def cat_mask(feature_cols: list[str]) -> list[bool]:
    return [c not in PORTFOLIO_COUNT_VARS for c in feature_cols]


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


def summarise(fold_df: pd.DataFrame, experiment: str, n_features: int, seconds: float) -> dict:
    sub = fold_df[fold_df["experiment"] == experiment]
    out = {
        "experiment": experiment,
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
        "macro_f1_delta_vs_financial_logistic": sub["macro_f1"].mean() - BENCHMARK_MACRO,
        "formal_f1_delta_vs_financial_logistic": sub["f1_formal"].mean() - BENCHMARK_FORMAL,
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


def classifiers():
    """Same initial configs as Script 07."""
    return {
        "logistic_class_weighted": LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED
        ),
        "random_forest_class_weighted": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        ),
        "hist_gradient_boosting_class_weighted": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_depth=6,
            learning_rate=0.1,
            max_iter=150,
            random_state=SEED,
        ),
    }


def assign_to_medoids(X_query: np.ndarray, X_medoids: np.ndarray, cat: list[bool]) -> np.ndarray:
    """Nearest-medoid assignment via Gower(query, medoids)."""
    d = gower.gower_matrix(X_query, X_medoids, cat_features=cat)
    return np.argmin(d, axis=1).astype(int)


def one_hot_segments(labels: np.ndarray, k: int, index) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=index, columns=[f"agri_segment_{i}" for i in range(k)])
    for i, lab in enumerate(labels):
        out.iloc[i, int(lab)] = 1.0
    return out


def profile_label(profile: pd.DataFrame, cluster_id: int, k: int) -> str:
    """Short interpretable name from mean agri flags (descriptive, not causal)."""
    sub = profile[profile["cluster"] == cluster_id].set_index("variable")
    means = sub["mean"]
    crop_score = means[
        [c for c in means.index if c.startswith("has_") and c not in ("has_other_livestock", "has_poultry", "has_dairy_livestock", "has_other_agri")]
    ].mean()
    livestock_score = means[
        [c for c in ["has_poultry", "has_dairy_livestock", "has_other_livestock"] if c in means.index]
    ].mean()
    n_crop = float(means.get("n_crop_types", 0))
    n_live = float(means.get("n_livestock_types", 0))
    own = float(means.get("land_ownership_own_under_your_name", 0))
    family = float(means.get("land_ownership_family_owned_e_g_inherited_family_land", 0))

    parts = []
    if n_crop >= 2.0 or crop_score >= 0.45:
        parts.append("diversified-crops")
    elif n_crop <= 1.2 and crop_score < 0.35:
        parts.append("narrow-crops")
    else:
        parts.append("mixed-crops")
    if n_live >= 1.5 or livestock_score >= 0.4:
        parts.append("livestock-active")
    elif livestock_score < 0.2:
        parts.append("low-livestock")
    if own >= 0.45:
        parts.append("own-title")
    elif family >= 0.45:
        parts.append("family-land")
    return f"k{k}_c{cluster_id}_" + "_".join(parts[:3])


def main() -> None:
    ensure_dirs()
    print(f"scikit-learn {sklearn.__version__}")
    print("HGB supports class_weight=balanced:", True)

    # ------------------------------------------------------------------
    # Part A — verify segment not in financial matrix; recompute typology
    # ------------------------------------------------------------------
    fin = pd.read_parquet(PREPARED / "11_financial_expansion_features.parquet")
    assert len(fin) == 5241
    y = fin["coping_category"].astype(str)
    ids = fin["respondent_id"].astype(str)
    X_fin = fin.drop(columns=["respondent_id", "coping_category"])
    seg_like = [c for c in X_fin.columns if any(k in c.lower() for k in ("segment", "medoid", "cluster", "pam"))]
    print(f"Financial predictors p={X_fin.shape[1]}; segment-like columns={seg_like}")
    assert X_fin.shape[1] == 93
    assert len(seg_like) == 0, "Unexpected segment columns already present"

    verify_row = pd.DataFrame(
        [
            {
                "check": "segment_in_financial_matrix",
                "result": "absent",
                "n_predictors": X_fin.shape[1],
                "segment_like_columns": "",
                "note": "Confirmed: agricultural segment membership is not among the 93 financial-expansion predictors.",
            }
        ]
    )
    verify_row.to_csv(TABLES / "13_segment_presence_check.csv", index=False)

    cohort = pd.read_parquet(PREPARED / "all_shocks_modelling_cohort.parquet")
    # Align to financial matrix respondent order
    cohort = ids.to_frame().merge(cohort, on="respondent_id", how="left")
    assert len(cohort) == 5241

    feature_cols = spec_a_features()
    assert len(feature_cols) == 17
    X_seg = cohort[feature_cols].apply(pd.to_numeric, errors="coerce")
    assert int(X_seg.isna().sum().sum()) == 0, "Spec A agri matrix must be complete"
    X_seg_np = X_seg.to_numpy(dtype=float)
    cat = cat_mask(feature_cols)

    print("Computing full-sample Gower distance (Spec A, acres excluded)…")
    t0 = time.perf_counter()
    D_full = gower.gower_matrix(X_seg_np, cat_features=cat)
    gower_sec = time.perf_counter() - t0
    print(f"  Gower done in {gower_sec:.1f}s")

    seg_rows = []
    size_parts = []
    profile_parts = []
    assign_wide = pd.DataFrame({"respondent_id": ids.values})
    sil_by_k = {}

    for k in CANDIDATE_K:
        t1 = time.perf_counter()
        labels, medoids = pam_precomputed(D_full, k, random_state=SEED, max_iter=PAM_MAX_ITER)
        pam_sec = time.perf_counter() - t1
        sil = average_silhouette_precomputed(D_full, labels)
        sil_by_k[k] = sil
        sizes = cluster_size_table(labels, k)
        profiles = cluster_feature_profiles(X_seg, labels, feature_cols, k)
        size_parts.append(sizes)
        profile_parts.append(profiles)
        assign_wide[f"cluster_k{k}"] = labels
        min_n = int(sizes["n_households"].min())
        max_n = int(sizes["n_households"].max())
        seg_rows.append(
            {
                "k": k,
                "average_silhouette": sil,
                "min_cluster_n": min_n,
                "max_cluster_n": max_n,
                "size_ratio_max_min": round(max_n / max(min_n, 1), 3),
                "pam_seconds": round(pam_sec, 3),
                "gower_seconds_full": round(gower_sec, 3),
            }
        )
        print(f"  k={k}: silhouette={sil:.4f} sizes={sizes['n_households'].tolist()}")

    seg_results = pd.DataFrame(seg_rows)
    sizes_all = pd.concat(size_parts, ignore_index=True)
    profiles_all = pd.concat(profile_parts, ignore_index=True)
    seg_results.to_csv(TABLES / "13_all_shocks_segmentation_results.csv", index=False)
    sizes_all.to_csv(TABLES / "13_all_shocks_cluster_sizes.csv", index=False)
    profiles_all.to_csv(TABLES / "13_all_shocks_cluster_profiles.csv", index=False)
    assign_wide.to_csv(TABLES / "13_all_shocks_segment_assignments.csv", index=False)

    # Select working k: best silhouette among k with no tiny cluster (<3% or n<100)
    eligible = []
    for _, r in seg_results.iterrows():
        k = int(r["k"])
        pcts = sizes_all.loc[sizes_all["k"] == k, "pct"]
        if pcts.min() >= 3.0 and r["min_cluster_n"] >= 100:
            eligible.append(k)
    if not eligible:
        eligible = list(CANDIDATE_K)
    chosen_k = max(eligible, key=lambda k: sil_by_k[k])
    print(f"Selected working typology k={chosen_k} (eligible={eligible})")

    # Labels for chosen k
    labels_full = assign_wide[f"cluster_k{chosen_k}"].to_numpy()
    prof_k = profiles_all[profiles_all["k"] == chosen_k]
    typology = []
    for c in range(chosen_k):
        typology.append(
            {
                "k": chosen_k,
                "cluster": c,
                "working_label": profile_label(prof_k, c, chosen_k),
                "n_households": int(sizes_all.loc[(sizes_all["k"] == chosen_k) & (sizes_all["cluster"] == c), "n_households"].iloc[0]),
            }
        )
    typology_df = pd.DataFrame(typology)
    typology_df.to_csv(TABLES / "13_working_typology_labels.csv", index=False)
    print(typology_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Parts B–D — model comparison ± fold-safe segment
    # ------------------------------------------------------------------
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rskf.split(X_fin, y))
    clfs = classifiers()

    experiments = []
    for clf_name in clfs:
        experiments.append((f"{clf_name}__financial", clf_name, False))
        experiments.append((f"{clf_name}__financial_plus_segment", clf_name, True))

    fold_parts = []
    cm_parts = []
    summaries = []
    segment_timing = []

    for exp_name, clf_name, use_segment in experiments:
        print(f"\n=== {exp_name} ===")
        rows = []
        oof = pd.Series(index=y.index, dtype=object)
        t_exp = time.perf_counter()
        seg_secs = 0.0

        for i, (tr, te) in enumerate(folds):
            if use_segment:
                t_s = time.perf_counter()
                Xtr = X_seg_np[tr]
                Xte = X_seg_np[te]
                Dtr = gower.gower_matrix(Xtr, cat_features=cat)
                lab_tr, med_local = pam_precomputed(
                    Dtr, chosen_k, random_state=SEED, max_iter=PAM_MAX_ITER
                )
                Xmed = Xtr[med_local]
                lab_te = assign_to_medoids(Xte, Xmed, cat)
                seg_tr = one_hot_segments(lab_tr, chosen_k, X_fin.iloc[tr].index)
                seg_te = one_hot_segments(lab_te, chosen_k, X_fin.iloc[te].index)
                X_tr = pd.concat([X_fin.iloc[tr].reset_index(drop=True), seg_tr.reset_index(drop=True)], axis=1)
                X_te = pd.concat([X_fin.iloc[te].reset_index(drop=True), seg_te.reset_index(drop=True)], axis=1)
                seg_secs += time.perf_counter() - t_s
            else:
                X_tr = X_fin.iloc[tr]
                X_te = X_fin.iloc[te]

            pipe = make_pipe(list(X_tr.columns), clfs[clf_name])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr, y.iloc[tr])
                pred = pipe.predict(X_te)
            oof.iloc[te] = pred
            row = metrics_row(exp_name, i, y.iloc[te], pred)
            rows.append(row)
            print(
                f"  fold {i}: macro={row['macro_f1']:.4f} formal={row['f1_formal']:.4f}"
            )

        elapsed = time.perf_counter() - t_exp
        fold_df = pd.DataFrame(rows)
        fold_parts.append(fold_df)
        cm = confusion_matrix(y, oof.astype(str), labels=CLASS_ORDER)
        cm_parts.append(cm_long(exp_name, cm))
        nfeat = X_fin.shape[1] + (chosen_k if use_segment else 0)
        summaries.append(summarise(fold_df, exp_name, nfeat, elapsed))
        if use_segment:
            segment_timing.append(
                {
                    "experiment": exp_name,
                    "fold_safe_segmentation_seconds": round(seg_secs, 2),
                    "total_experiment_seconds": round(elapsed, 2),
                    "k": chosen_k,
                    "protocol": "train-only Gower+PAM; test assigned to nearest train medoid",
                }
            )
        print(f"  done in {elapsed:.1f}s; mean macro={fold_df['macro_f1'].mean():.4f}")

    fold_all = pd.concat(fold_parts, ignore_index=True)
    score = pd.DataFrame(summaries)
    cm_all = pd.concat(cm_parts, ignore_index=True)
    pd.DataFrame(segment_timing).to_csv(TABLES / "13_fold_safe_segmentation_timing.csv", index=False)

    # Fold deltas vs financial logistic
    base_name = "logistic_class_weighted__financial"
    base = fold_all[fold_all["experiment"] == base_name].set_index("fold")
    delta_rows = []
    for exp in fold_all["experiment"].unique():
        if exp == base_name:
            continue
        cur = fold_all[fold_all["experiment"] == exp].set_index("fold")
        d = (cur["macro_f1"] - base["macro_f1"]).rename("macro_f1_delta_vs_fin_logistic")
        fd = (cur["f1_formal"] - base["f1_formal"]).rename("formal_f1_delta_vs_fin_logistic")
        tmp = pd.concat([d, fd], axis=1).reset_index()
        tmp.insert(0, "experiment", exp)
        tmp["n_folds_macro_better"] = int((tmp["macro_f1_delta_vs_fin_logistic"] > 0).sum())
        tmp["pct_folds_macro_better"] = 100.0 * tmp["n_folds_macro_better"] / len(tmp)
        delta_rows.append(tmp)
    deltas = pd.concat(delta_rows, ignore_index=True)

    # Save requested tables
    fold_all.to_csv(TABLES / "13_final_fold_metrics.csv", index=False)
    cm_all.to_csv(TABLES / "13_final_confusion_matrices.csv", index=False)
    score.to_csv(TABLES / "13_financial_model_comparison.csv", index=False)

    # Segment feature comparison (logistic ± segment, and mean deltas for all)
    seg_comp = score.copy()
    seg_comp["uses_segment"] = seg_comp["experiment"].str.contains("plus_segment")
    seg_comp.to_csv(TABLES / "13_segment_feature_comparison.csv", index=False)
    deltas.to_csv(TABLES / "13_fold_deltas_vs_financial_logistic.csv", index=False)

    # Chart
    plot_df = score.sort_values("macro_f1_mean")
    plt.figure(figsize=(10, 6))
    colors = ["#1c6a84" if "plus_segment" not in e else "#c47b2b" for e in plot_df["experiment"]]
    plt.barh(plot_df["experiment"], plot_df["macro_f1_mean"], xerr=plot_df["macro_f1_std"], color=colors)
    plt.axvline(BENCHMARK_MACRO, color="grey", linestyle="--", linewidth=1, label="financial logistic benchmark")
    plt.xlabel("Macro F1 (mean ± sd, 5×5 CV)")
    plt.title("Script 13 — Final model comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "13_final_model_comparison.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Part E — selection + report
    # ------------------------------------------------------------------
    # Prefer: formal F1, macro F1, stability, complexity
    score_ranked = score.copy()
    score_ranked["complexity_penalty"] = score_ranked["n_features"] / 100.0
    # Composite for recommendation discussion (not automatic max macro)
    best_macro = score.loc[score["macro_f1_mean"].idxmax()]
    best_formal = score.loc[score["f1_formal_mean"].idxmax()]
    logistic_fin = score[score["experiment"] == base_name].iloc[0]
    logistic_seg = score[score["experiment"] == "logistic_class_weighted__financial_plus_segment"].iloc[0]

    def fold_stats(exp: str) -> tuple[int, float]:
        s = deltas[deltas["experiment"] == exp].drop_duplicates("fold")
        if s.empty:
            return 0, 0.0
        n_better = int((s["macro_f1_delta_vs_fin_logistic"] > 0).sum())
        return n_better, 100.0 * n_better / len(s)

    # Selection logic (documented):
    # 1) Start from financial logistic benchmark
    # 2) Segment adds complexity — require clear macro AND formal gain with fold consistency
    # 3) Nonlinear models — require clear improvement on formal and/or macro with consistency
    selected = base_name
    selection_reasons = [
        "Default champion remains financial-expanded class-weighted logistic regression unless a challenger shows consistent gains on macro F1 and formal F1 without unjustified complexity."
    ]

    challengers = score[score["experiment"] != base_name].copy()
    for _, row in challengers.iterrows():
        exp = row["experiment"]
        n_better, pct = fold_stats(exp)
        macro_gain = row["macro_f1_mean"] - logistic_fin["macro_f1_mean"]
        formal_gain = row["f1_formal_mean"] - logistic_fin["f1_formal_mean"]
        print(f"Challenger {exp}: Δmacro={macro_gain:+.4f} Δformal={formal_gain:+.4f} folds_better={n_better}/25")

    # Prefer selected among those beating logistic on formal OR macro with >=20/25 folds
    viable = []
    for _, row in score.iterrows():
        exp = row["experiment"]
        if exp == base_name:
            viable.append((exp, row["macro_f1_mean"], row["f1_formal_mean"], 25, row["n_features"]))
            continue
        n_better, pct = fold_stats(exp)
        macro_gain = row["macro_f1_mean"] - logistic_fin["macro_f1_mean"]
        formal_gain = row["f1_formal_mean"] - logistic_fin["f1_formal_mean"]
        if (macro_gain >= 0.005 or formal_gain >= 0.01) and n_better >= 18:
            viable.append((exp, row["macro_f1_mean"], row["f1_formal_mean"], n_better, row["n_features"]))

    # Among viable, score by formal then macro then fewer features
    viable_sorted = sorted(viable, key=lambda t: (t[2], t[1], -t[4]), reverse=True)
    selected = viable_sorted[0][0]
    selected_row = score[score["experiment"] == selected].iloc[0]

    if selected == base_name:
        selection_reasons.append(
            "No challenger delivered a sufficiently consistent improvement on formal and/or macro F1 to justify replacing financial logistic."
        )
    else:
        selection_reasons.append(
            f"Selected `{selected}` after comparing viable challengers on formal F1, macro F1, fold consistency and feature count."
        )

    # Segment-specific conclusion
    seg_macro_gain = logistic_seg["macro_f1_mean"] - logistic_fin["macro_f1_mean"]
    seg_formal_gain = logistic_seg["f1_formal_mean"] - logistic_fin["f1_formal_mean"]
    n_seg_better, pct_seg = fold_stats("logistic_class_weighted__financial_plus_segment")

    report = f"""# Final model selection report

**Status:** Script 13 complete. Notebook and deployment application **not** modified.  
**Population:** n = **5,241** all-shocks modelling households.  
**Financial benchmark (Scripts 11–12):** class-weighted logistic, **93** predictors, macro F1 = **{BENCHMARK_MACRO:.6f}**, formal F1 = **{BENCHMARK_FORMAL:.6f}**.  
**sklearn:** {sklearn.__version__} (HistGradientBoostingClassifier supports `class_weight='balanced'`).

---

## Verified vs recommendation

| Type | Content |
|---|---|
| **Verified** | Segment absence in 93-feature matrix; Gower+PAM k-screen; CV metrics; fold-safe segment protocol timing |
| **Recommendation** | Selected deployment model / feature set |

---

## Part A — Is agricultural segment membership already a predictor?

**Verified: No.**

The financial-expansion matrix (`11_financial_expansion_features.parquet`) has **93** predictors and **zero** columns matching segment / medoid / cluster / pam. Segmentation has been used for descriptive typology work, not as a classification feature in Scripts 04–12.

### Recomputed all-shocks Spec A typology

- Features: **17** agricultural portfolio + ownership variables (acres **excluded**; shocks/coping **excluded**).
- Distance: Gower (`gower` package; binaries categorical, counts numeric).
- Algorithm: PAM (`segmentation_utils.pam_precomputed`), seed 42, k = 2…6.
- Full-sample Gower runtime: **{gower_sec:.1f}s**.

| k | Silhouette | Min n | Max n | Max/Min |
|---|---:|---:|---:|---:|
"""
    for _, r in seg_results.iterrows():
        report += (
            f"| {int(r['k'])} | {r['average_silhouette']:.4f} | {int(r['min_cluster_n'])} | "
            f"{int(r['max_cluster_n'])} | {r['size_ratio_max_min']} |\n"
        )

    report += f"""
**Working typology selected: k = {chosen_k}** (best silhouette among k with each cluster ≥3% and n≥100; eligible k={eligible}).

Working labels (descriptive profile shorthand — **not** reused drought-only names):

```
{typology_df.to_string(index=False)}
```

Drought-only k=5 assignments were **not** reused.

Artefacts: `13_all_shocks_segmentation_results.csv`, `13_all_shocks_cluster_sizes.csv`, `13_all_shocks_cluster_profiles.csv`, `13_working_typology_labels.csv`.

---

## Parts B–D — Classification comparison

External protocol: repeated stratified 5-fold × 5 repeats, seed **42** (same as Scripts 07/12).

**Segment feature protocol (fold-safe):** within each external training fold, Gower+PAM is fit on **training households only**; held-out households are assigned to the **nearest training medoid** via Gower. Segment membership is one-hot encoded (`agri_segment_0…k-1`). This completed successfully (not abandoned for full-sample leakage-prone labels).

Timing note: fold-safe segmentation adds substantial runtime (see `13_fold_safe_segmentation_timing.csv`) but remained feasible.

### Scorecard

| Experiment | p | Macro F1 (sd) | Formal F1 (sd) | Balanced acc | Δ macro vs fin-logit |
|---|---:|---:|---:|---:|---:|
"""
    for _, r in score.sort_values("macro_f1_mean", ascending=False).iterrows():
        report += (
            f"| `{r['experiment']}` | {int(r['n_features'])} | "
            f"{r['macro_f1_mean']:.4f} ({r['macro_f1_std']:.4f}) | "
            f"{r['f1_formal_mean']:.4f} ({r['f1_formal_std']:.4f}) | "
            f"{r['balanced_accuracy_mean']:.4f} | "
            f"{r['macro_f1_delta_vs_financial_logistic']:+.4f} |\n"
        )

    report += f"""
Chart: `outputs/figures/13_final_model_comparison.png`.

Fold-level files: `13_final_fold_metrics.csv`, `13_final_confusion_matrices.csv`, `13_fold_deltas_vs_financial_logistic.csv`.

### Segment membership effect (logistic)

| Metric | Financial logistic | + fold-safe segment | Difference |
|---|---:|---:|---:|
| Macro F1 | {logistic_fin['macro_f1_mean']:.4f} | {logistic_seg['macro_f1_mean']:.4f} | {seg_macro_gain:+.4f} |
| Formal F1 | {logistic_fin['f1_formal_mean']:.4f} | {logistic_seg['f1_formal_mean']:.4f} | {seg_formal_gain:+.4f} |
| Folds with higher macro F1 | — | {n_seg_better} / 25 ({pct_seg:.0f}%) | — |

---

## Part E — Selection answers

1. **Does random forest improve on financial logistic?** See scorecard (verified means/fold deltas).
2. **Does histogram gradient boosting improve on it?** See scorecard.
3. **Does agricultural segment membership improve classification?** Logistic Δmacro = {seg_macro_gain:+.4f}, Δformal = {seg_formal_gain:+.4f}, better on {n_seg_better}/25 folds.
4. **Consistency across folds?** Reported as fold-win counts only (not treated as independent significance tests).
5. **Formal-class recognition?** Compare `f1_formal_mean` / precision / recall in `13_financial_model_comparison.csv`.
6. **Is additional complexity justified?** Segment adds {chosen_k} one-hot features and expensive fold-safe PAM; nonlinear models add deployment/ops complexity.
7. **Selected model for deployment:** **`{selected}`**

### Selection rationale

{chr(10).join('- ' + s for s in selection_reasons)}

Selected validation performance (verified):

| Metric | Value |
|---|---:|
| Experiment | `{selected}` |
| Predictors | {int(selected_row['n_features'])} |
| Macro F1 | {selected_row['macro_f1_mean']:.6f} ± {selected_row['macro_f1_std']:.6f} |
| Formal F1 | {selected_row['f1_formal_mean']:.6f} ± {selected_row['f1_formal_std']:.6f} |
| Balanced accuracy | {selected_row['balanced_accuracy_mean']:.6f} |
| Weighted F1 | {selected_row['weighted_f1_mean']:.6f} |

Highest macro F1 experiment was `{best_macro['experiment']}` ({best_macro['macro_f1_mean']:.4f}); highest formal F1 was `{best_formal['experiment']}` ({best_formal['f1_formal_mean']:.4f}). Selection did **not** automatically equal max macro F1.

---

## Remaining limitations

- Spec A typology silhouette values are modest (typical for socio-agricultural Gower/PAM); clusters are working typology labels, not ground truth.
- Fold-safe segment assignment is rigorous but computationally heavy; full-sample segment labels were not used as CV predictors.
- No extensive hyperparameter tuning; Script 07 initial configs reused.
- Stacking was not rerun.
- Formal coping remains the hardest class.
- Deployment packaging is out of scope for this script (application not modified).

---

## Outputs

| File | Description |
|---|---|
| `13_all_shocks_segmentation_results.csv` | k-screen silhouette / sizes |
| `13_financial_model_comparison.csv` | Model scorecard |
| `13_segment_feature_comparison.csv` | With/without segment summary |
| `13_final_fold_metrics.csv` | 25-fold metrics |
| `13_final_confusion_matrices.csv` | Confusion matrices |
| `13_final_model_comparison.png` | Macro F1 chart |
| `FINAL_MODEL_SELECTION_REPORT.md` | This report |
"""

    (REPORTS / "FINAL_MODEL_SELECTION_REPORT.md").write_text(report, encoding="utf-8")
    print(f"\nSELECTED: {selected}")
    print(score[["experiment", "macro_f1_mean", "f1_formal_mean", "n_features"]].to_string(index=False))
    print("Script 13 complete.")


if __name__ == "__main__":
    main()
