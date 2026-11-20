"""Shared helpers for Gower/PAM segmentation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gower_matrix_missing_aware(
    X: np.ndarray,
    cat_features: np.ndarray,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise Gower distance that skips features missing in either observation.

    Vectorised implementation. For each pair and feature:
    - If either value is NaN, that feature is omitted from the pair average.
    - Numeric: |x-y| / range_f (range from non-missing values; floor 1e-12).
    - Categorical/binary: 0 if equal else 1.
    Pairs with no overlapping non-missing features receive distance 1.0.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    cat_features = np.asarray(cat_features, dtype=bool)
    if cat_features.shape != (p,):
        raise ValueError("cat_features must have shape (n_features,)")
    if weight is None:
        weight = np.ones(p, dtype=float)
    else:
        weight = np.asarray(weight, dtype=float)
        if weight.shape != (p,):
            raise ValueError("weight must have shape (n_features,)")

    ranges = np.ones(p, dtype=float)
    for f in range(p):
        if cat_features[f]:
            continue
        col = X[:, f]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            ranges[f] = 1.0
        else:
            ranges[f] = max(float(observed.max() - observed.min()), 1e-12)

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
        # Zero out pairs missing this feature; diagonal handled later
        delta = np.where(both, delta, 0.0)
        num += w * delta
        den += w * both.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.divide(num, den, out=np.ones((n, n), dtype=float), where=den > 0)
    np.fill_diagonal(dist, 0.0)
    return dist


def pam_precomputed(
    distance: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """PAM on a precomputed distance matrix.

    Returns (labels, medoid_indices).
    """
    rng = np.random.default_rng(random_state)
    n = distance.shape[0]
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(f"n_clusters must be in 1..{n}, got {n_clusters}")
    medoids = np.sort(rng.choice(n, size=n_clusters, replace=False))
    labels = np.argmin(distance[:, medoids], axis=1)
    for _ in range(max_iter):
        new_medoids: list[int] = []
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


def average_silhouette_precomputed(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette width using a precomputed distance matrix."""
    from sklearn.metrics import silhouette_score

    # Silhouette requires at least 2 labels with members
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    return float(silhouette_score(distance, labels, metric="precomputed"))


def cluster_size_table(labels: np.ndarray, k: int) -> pd.DataFrame:
    vc = pd.Series(labels).value_counts().sort_index()
    rows = []
    for cluster_id in range(k):
        n = int(vc.get(cluster_id, 0))
        rows.append(
            {
                "k": k,
                "cluster": cluster_id,
                "n_households": n,
                "pct": round(100.0 * n / len(labels), 2),
            }
        )
    return pd.DataFrame(rows)


def cluster_feature_profiles(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    k: int,
) -> pd.DataFrame:
    work = X[feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    work["cluster"] = labels
    rows = []
    for cluster_id in range(k):
        subset = work.loc[work["cluster"] == cluster_id, feature_cols]
        for col in feature_cols:
            s = subset[col]
            rows.append(
                {
                    "k": k,
                    "cluster": cluster_id,
                    "variable": col,
                    "n": int(len(subset)),
                    "mean": float(s.mean()) if len(subset) else np.nan,
                    "std": float(s.std(ddof=0)) if len(subset) else np.nan,
                    "median": float(s.median()) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)
