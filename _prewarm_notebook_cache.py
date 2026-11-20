#!/usr/bin/env python3
"""Pre-warm notebook_run_cache using the same locked settings as the notebook cells."""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import gower
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CAPSTONE = Path(__file__).resolve().parent
RUN_CACHE = CAPSTONE / "notebook_run_cache"
RUN_CACHE.mkdir(parents=True, exist_ok=True)

# Prefer prepared frame for speed/parity
PREPARED = CAPSTONE / "finaccess_coping_workbench" / "outputs" / "prepared"
PROJECT_ROOT = Path.home() / "Documents" / "items" / "fsd" / "finaccess"
VALUE_LABELS_PATH = PROJECT_ROOT / "2024" / "value_labels.csv"

RANDOM_SEED = 42
EXPECTED_DROUGHT_N = 1552
FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]
CONTINUOUS_BASE = ["age_years", "education", "n_crop_types", "n_livestock_types"]

AGRICULTURAL_HAS_VARS = [
    "has_cash_crops", "has_cereals", "has_dairy_livestock", "has_horticulture",
    "has_legumes", "has_other_agri", "has_other_livestock", "has_poultry", "has_roots_tubers",
]
PORTFOLIO_COUNT_VARS = ["n_crop_types", "n_livestock_types"]
LAND_OWNERSHIP_VARS = [
    "land_ownership_own_under_your_name",
    "land_ownership_own_jointly_with_spouse",
    "land_ownership_own_jointly_with_someone_else_other_than_spouse",
    "land_ownership_lease_rent_you_dont_own",
    "land_ownership_community_public_owned_e_g_pastoralists",
    "land_ownership_family_owned_e_g_inherited_family_land",
]


def pam_precomputed(distance, n_clusters, random_state=42, max_iter=100):
    rng = np.random.default_rng(random_state)
    n = distance.shape[0]
    medoids = np.sort(rng.choice(n, size=n_clusters, replace=False))
    labels = np.argmin(distance[:, medoids], axis=1)
    for _ in range(max_iter):
        new_medoids = []
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                new_medoids.append(int(medoids[k]))
                continue
            block = distance[np.ix_(members, members)]
            costs = block.sum(axis=1)
            new_medoids.append(int(members[int(np.argmin(costs))]))
        new_medoids_arr = np.sort(np.asarray(new_medoids, dtype=int))
        labels = np.argmin(distance[:, new_medoids_arr], axis=1)
        if np.array_equal(new_medoids_arr, medoids):
            break
        medoids = new_medoids_arr
    return labels.astype(int), medoids


def average_silhouette_precomputed(distance, labels):
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    return float(silhouette_score(distance, labels, metric="precomputed"))


def gower_matrix_missing_aware(X, cat_features, weight=None):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    cat_features = np.asarray(cat_features, dtype=bool)
    weight = np.ones(p, dtype=float) if weight is None else np.asarray(weight, dtype=float)
    ranges = np.ones(p, dtype=float)
    for f in range(p):
        if cat_features[f]:
            continue
        col = X[:, f]
        observed = col[~np.isnan(col)]
        ranges[f] = 1.0 if len(observed) == 0 else max(float(observed.max() - observed.min()), 1e-12)
    present = ~np.isnan(X)
    num = np.zeros((n, n), dtype=float)
    den = np.zeros((n, n), dtype=float)
    for f in range(p):
        w = float(weight[f])
        if w == 0:
            continue
        col = X[:, f]
        both = present[:, f][:, None] & present[:, f][None, :]
        if not both.any():
            continue
        if cat_features[f]:
            delta = (col[:, None] != col[None, :]).astype(float)
        else:
            delta = np.abs(col[:, None] - col[None, :]) / ranges[f]
        delta = np.where(both, delta, 0.0)
        num += w * delta
        den += w * both.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.divide(num, den, out=np.ones((n, n), dtype=float), where=den > 0)
    np.fill_diagonal(dist, 0.0)
    return dist


def metrics_from_preds(y_true, y_pred):
    prec, rec, f1, support = precision_recall_fscore_support(
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


def summarise_folds(fold_metrics):
    metric_cols = [c for c in fold_metrics.columns if c not in {"experiment_id", "repeat", "fold", "split_id"}]
    rows = []
    for exp_id, g in fold_metrics.groupby("experiment_id", sort=False):
        row = {"experiment_id": exp_id, "n_folds": len(g)}
        for c in metric_cols:
            row[f"{c}_mean"] = round(float(g[c].mean()), 4)
            row[f"{c}_std"] = round(float(g[c].std(ddof=0)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    X_seg = pd.read_parquet(PREPARED / "09_X_segmentation.parquet")
    X_pri = pd.read_parquet(PREPARED / "09_X_primary_classification.parquet")
    y = pd.read_parquet(PREPARED / "09_y_targets.parquet")
    frame = pd.read_parquet(PREPARED / "09_prepared_analysis_frame.parquet")
    assert len(X_seg) == EXPECTED_DROUGHT_N

    SPEC_A_FEATURES = list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS) + list(LAND_OWNERSHIP_VARS)
    cat_mask_a = np.array([c not in PORTFOLIO_COUNT_VARS for c in SPEC_A_FEATURES], dtype=bool)
    X_a = X_seg[["respondent_id"] + SPEC_A_FEATURES].copy()
    for c in SPEC_A_FEATURES:
        X_a[c] = pd.to_numeric(X_a[c], errors="coerce")

    # Spec A
    spec_a_cache = RUN_CACHE / "spec_a_s1.npz"
    if not spec_a_cache.exists():
        print("Computing Spec A Gower+PAM...")
        t0 = time.perf_counter()
        dist_a = gower.gower_matrix(X_a[SPEC_A_FEATURES].to_numpy(dtype=float), cat_features=cat_mask_a)
        print(f"  distance {time.perf_counter()-t0:.1f}s")
        sil_rows = []
        assign = {"respondent_id": X_a["respondent_id"].to_numpy()}
        for k in range(2, 7):
            labels, _ = pam_precomputed(dist_a, k, random_state=RANDOM_SEED, max_iter=100)
            sil = average_silhouette_precomputed(dist_a, labels)
            sil_rows.append({"specification": "A", "k": k, "average_silhouette": sil, "n_features": 17, "random_seed": RANDOM_SEED})
            assign[f"cluster_k{k}"] = labels
            print(f"  k={k} sil={sil:.4f}")
        sil_table = pd.DataFrame(sil_rows)
        assign_wide = pd.DataFrame(assign)
        np.savez_compressed(
            spec_a_cache,
            distance=dist_a,
            assign_wide=assign_wide.to_dict(orient="list"),
            sil_table=sil_table.to_dict(orient="list"),
        )
        (RUN_CACHE / "spec_a_s1_meta.json").write_text(
            json.dumps({"random_seed": RANDOM_SEED, "features": SPEC_A_FEATURES, "n": len(X_a)}, indent=2),
            encoding="utf-8",
        )
    else:
        print("Spec A cache exists")
        blob = np.load(spec_a_cache, allow_pickle=True)
        dist_a = blob["distance"]
        assign_wide = pd.DataFrame(blob["assign_wide"].item())
        sil_table = pd.DataFrame(blob["sil_table"].item())

    # Spec B correspondence
    spec_b_cache = RUN_CACHE / "spec_b_correspondence.json"
    if not spec_b_cache.exists():
        print("Computing Spec B...")
        unit = pd.to_numeric(frame.set_index("respondent_id").loc[X_a["respondent_id"], "land_size_unit"].to_numpy(), errors="coerce")
        size = pd.to_numeric(frame.set_index("respondent_id").loc[X_a["respondent_id"], "land_size"].to_numpy(), errors="coerce")
        acres = pd.to_numeric(X_seg.set_index("respondent_id").loc[X_a["respondent_id"], "land_size_acres"].to_numpy(), errors="coerce")
        is_community = (unit == 5).astype(float)
        special_non_comm = np.isnan(acres) & np.isin(size, [98, 99]) & (unit != 5)
        med = float(np.median(acres[~np.isnan(acres)]))
        acres_b = acres.copy()
        acres_b[special_non_comm] = med
        acres_b[is_community == 1] = np.nan
        SPEC_B_FEATURES = SPEC_A_FEATURES + ["land_size_acres_b", "is_community_land", "acres_special_imputed"]
        Xb_mat = X_a[SPEC_A_FEATURES].copy()
        Xb_mat["land_size_acres_b"] = acres_b
        Xb_mat["is_community_land"] = is_community
        Xb_mat["acres_special_imputed"] = special_non_comm.astype(float)
        cat_mask_b = np.array([c not in PORTFOLIO_COUNT_VARS + ["land_size_acres_b"] for c in SPEC_B_FEATURES], dtype=bool)
        t0 = time.perf_counter()
        dist_b = gower_matrix_missing_aware(Xb_mat[SPEC_B_FEATURES].to_numpy(dtype=float), cat_mask_b)
        print(f"  Spec B distance {time.perf_counter()-t0:.1f}s")
        corr = {"ari": {}, "sizes_b": {}}
        for k in (4, 5):
            lab_b, _ = pam_precomputed(dist_b, k, random_state=RANDOM_SEED, max_iter=100)
            lab_a = assign_wide[f"cluster_k{k}"].to_numpy()
            corr["ari"][str(k)] = float(adjusted_rand_score(lab_a, lab_b))
            corr["sizes_b"][str(k)] = pd.Series(lab_b).value_counts().sort_index().astype(int).to_dict()
            assign_wide[f"spec_b_cluster_k{k}"] = lab_b
            print(f"  ARI k={k}: {corr['ari'][str(k)]:.4f}")
        spec_b_cache.write_text(json.dumps(corr, indent=2), encoding="utf-8")
        np.savez_compressed(
            spec_a_cache,
            distance=dist_a,
            assign_wide=assign_wide.to_dict(orient="list"),
            sil_table=sil_table.to_dict(orient="list"),
        )
    else:
        print("Spec B cache exists")

    # Stability
    stab_cache = RUN_CACHE / "spec_a_stability_summary.csv"
    if not stab_cache.exists():
        print("Computing stability (50 replicates)...")
        N_REPLICATES = 50
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(X_a)
        subsample_idx = [
            np.sort(rng.choice(n, size=int(round(0.8 * n)), replace=False))
            for _ in range(N_REPLICATES)
        ]
        X_np = X_a[SPEC_A_FEATURES].to_numpy(dtype=float)
        rows = []
        for k in (2, 4, 5):
            aris = []
            full_labels = assign_wide[f"cluster_k{k}"].to_numpy()
            for rep, idx in enumerate(subsample_idx):
                d_sub = gower.gower_matrix(X_np[idx], cat_features=cat_mask_a)
                lab_sub, _ = pam_precomputed(d_sub, k, random_state=RANDOM_SEED, max_iter=100)
                aris.append(adjusted_rand_score(full_labels[idx], lab_sub))
                if (rep + 1) % 10 == 0:
                    print(f"  k={k} rep {rep+1}/{N_REPLICATES}")
            rows.append({
                "k": k,
                "n_replicates": N_REPLICATES,
                "mean_ari_vs_full": float(np.mean(aris)),
                "std_ari_vs_full": float(np.std(aris)),
                "min_ari_vs_full": float(np.min(aris)),
                "max_ari_vs_full": float(np.max(aris)),
            })
        pd.DataFrame(rows).to_csv(stab_cache, index=False)
    else:
        print("Stability cache exists")

    # Folds
    folds_cache = RUN_CACHE / "cv_folds_long.csv"
    y4 = y.set_index("respondent_id").loc[X_pri["respondent_id"], "coping_category"].to_numpy()
    ids = X_pri["respondent_id"].to_numpy()
    if not folds_cache.exists():
        print("Building CV folds...")
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
        rows = []
        for split_id, (tr, te) in enumerate(rskf.split(np.zeros(len(y4)), y4)):
            repeat, fold = divmod(split_id, 5)
            for split_name, idx in (("train", tr), ("test", te)):
                for i in idx:
                    rows.append({
                        "respondent_id": ids[i],
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": split_id,
                        "split": split_name,
                        "coping_category": y4[i],
                    })
        pd.DataFrame(rows).to_csv(folds_cache, index=False)
    folds_long = pd.read_csv(folds_cache)

    primary_features = [c for c in X_pri.columns if c != "respondent_id"]
    id_to_pos = {rid: i for i, rid in enumerate(X_pri["respondent_id"].tolist())}

    def build_logistic(feature_cols, class_weight_mode, land_treatment):
        cont = [c for c in CONTINUOUS_BASE if c in feature_cols]
        if land_treatment == "impute_median" and "land_size_acres" in feature_cols:
            cont = cont + ["land_size_acres"]
        binary = [c for c in feature_cols if c not in cont]
        transformers = []
        if cont:
            steps = []
            if land_treatment == "impute_median" and "land_size_acres" in cont:
                steps.append(("imputer", SimpleImputer(strategy="median")))
            steps.append(("scaler", StandardScaler()))
            transformers.append(("continuous", Pipeline(steps), cont))
        if binary:
            transformers.append(("binary", "passthrough", binary))
        pre = ColumnTransformer(transformers, remainder="drop")
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0,
            class_weight="balanced" if class_weight_mode == "class_weighted" else None,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("preprocess", pre), ("clf", clf)])

    def run_cv_experiment(experiment_id, model_builder, feature_cols, X_df):
        X_mat = X_df[feature_cols]
        fold_rows, cm_rows = [], []
        for split_id in sorted(folds_long["split_id"].unique()):
            part = folds_long.loc[folds_long["split_id"] == split_id]
            tr = part.loc[part["split"] == "train", "respondent_id"].map(id_to_pos).to_numpy()
            te = part.loc[part["split"] == "test", "respondent_id"].map(id_to_pos).to_numpy()
            repeat = int(part["repeat"].iloc[0])
            fold = int(part["fold"].iloc[0])
            est = model_builder()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                est.fit(X_mat.iloc[tr], y4[tr])
            pred = est.predict(X_mat.iloc[te])
            m = metrics_from_preds(y4[te], pred)
            fold_rows.append({"experiment_id": experiment_id, "repeat": repeat, "fold": fold, "split_id": split_id, **m})
            cm = confusion_matrix(y4[te], pred, labels=FOUR_ORDER)
            for i, tlab in enumerate(FOUR_ORDER):
                for j, plab in enumerate(FOUR_ORDER):
                    cm_rows.append({
                        "experiment_id": experiment_id, "repeat": repeat, "fold": fold,
                        "true_label": tlab, "pred_label": plab, "n": int(cm[i, j]),
                    })
        return pd.DataFrame(fold_rows), pd.DataFrame(cm_rows)

    # C1
    c1_cache = RUN_CACHE / "c1_scorecard.csv"
    if not c1_cache.exists():
        print("Running C1...")
        experiments = []
        feats_ex = [c for c in primary_features if c != "land_size_acres"]
        experiments.append(("dummy_most_frequent", lambda: DummyClassifier(strategy="most_frequent"), feats_ex))
        for land in ("exclude", "impute_median"):
            feats = feats_ex if land == "exclude" else list(primary_features)
            for cw in ("unweighted", "class_weighted"):
                eid = f"logistic_{cw}_{'exclude' if land == 'exclude' else 'impute_median'}"
                experiments.append((eid, lambda feats=feats, cw=cw, land=land: build_logistic(feats, cw, land), feats))
        all_folds, all_cms = [], []
        for eid, builder, feats in experiments:
            print(" ", eid)
            fm, cm = run_cv_experiment(eid, builder, feats, X_pri)
            all_folds.append(fm)
            all_cms.append(cm)
        c1_folds = pd.concat(all_folds, ignore_index=True)
        c1_scorecard = summarise_folds(c1_folds)
        c1_folds.to_csv(RUN_CACHE / "c1_fold_metrics.csv", index=False)
        c1_scorecard.to_csv(c1_cache, index=False)
        pd.concat(all_cms, ignore_index=True).to_csv(RUN_CACHE / "c1_fold_confusion.csv", index=False)
        print(c1_scorecard[["experiment_id", "macro_f1_mean", "f1_formal_mean"]])
    else:
        print("C1 cache exists")
        c1_scorecard = pd.read_csv(c1_cache)

    # C2/C3
    c23_cache = RUN_CACHE / "c23_scorecard.csv"
    feats_ex = [c for c in primary_features if c != "land_size_acres"]

    def build_tree_models(name, cw_mode):
        cw = "balanced" if cw_mode == "class_weighted" else None
        if name == "decision_tree":
            return DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_SEED)
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1)
        if name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=200, class_weight=cw, random_state=RANDOM_SEED, early_stopping=False
            )
        raise ValueError(name)

    if not c23_cache.exists():
        print("Running C2/C3...")
        all_folds, all_cms = [], []
        for model_name in ("decision_tree", "random_forest", "hist_gradient_boosting"):
            for cw in ("unweighted", "class_weighted"):
                eid = f"{model_name}_{cw}_exclude"
                print(" ", eid)
                t0 = time.perf_counter()
                fm, cm = run_cv_experiment(
                    eid, lambda model_name=model_name, cw=cw: build_tree_models(model_name, cw), feats_ex, X_pri
                )
                print(f"    {time.perf_counter()-t0:.1f}s")
                all_folds.append(fm)
                all_cms.append(cm)
        c23_folds = pd.concat(all_folds, ignore_index=True)
        c23_scorecard = summarise_folds(c23_folds)
        c23_folds.to_csv(RUN_CACHE / "c23_fold_metrics.csv", index=False)
        c23_scorecard.to_csv(c23_cache, index=False)
        pd.concat(all_cms, ignore_index=True).to_csv(RUN_CACHE / "c23_fold_confusion.csv", index=False)
        print(c23_scorecard[["experiment_id", "macro_f1_mean", "f1_formal_mean"]])
    else:
        print("C2/C3 cache exists")
        c23_scorecard = pd.read_csv(c23_cache)

    cols = ["experiment_id", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "weighted_f1_mean", "f1_formal_mean", "f1_formal_std", "accuracy_mean"]
    c1_keep = c1_scorecard.loc[c1_scorecard["experiment_id"].isin([
        "dummy_most_frequent", "logistic_unweighted_exclude", "logistic_class_weighted_exclude"
    ]), cols].copy()
    c1_keep.insert(1, "stage", "C1")
    c23_view = c23_scorecard[cols].copy()
    c23_view.insert(1, "stage", "C2/C3")
    comparison = pd.concat([c1_keep, c23_view], ignore_index=True)
    comparison.to_csv(RUN_CACHE / "c1_c23_comparison.csv", index=False)
    print("Done. Cache at", RUN_CACHE)


if __name__ == "__main__":
    main()
