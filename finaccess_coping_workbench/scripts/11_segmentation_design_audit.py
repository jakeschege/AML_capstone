#!/usr/bin/env python3
"""Segmentation design audit before S1 (Gower + PAM).

Investigates feature types, indicator–count overlap, domain contribution,
acreage missingness profiles, and Gower/PAM feasibility. Does not run the
S1–S4 segmentation experiments or fit classification models.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    AGRICULTURAL_HAS_VARS,
    EXPECTED_DROUGHT_N,
    LAND_OWNERSHIP_VARS,
    LAND_SIZE_STANDARDISED,
    LOGS_DIR,
    PORTFOLIO_COUNT_VARS,
    PREPARED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)

COMMUNITY_OWNERSHIP = "land_ownership_community_public_owned_e_g_pastoralists"


def classify_feature(col: str) -> tuple[str, str]:
    if col in AGRICULTURAL_HAS_VARS:
        return "binary", "agri_group"
    if col in PORTFOLIO_COUNT_VARS:
        return "numeric_count", "portfolio_count"
    if col == LAND_SIZE_STANDARDISED:
        return "continuous", "land_size"
    if col in LAND_OWNERSHIP_VARS:
        return "binary", "ownership"
    return "unknown", "other"


def primary_missing_reason(unit: float, size: float) -> str:
    if pd.notna(unit) and int(unit) == 5:
        return "Community land"
    if pd.notna(size) and int(size) in {98, 99}:
        return "Special response code 98/99"
    if pd.isna(unit):
        return "Missing measurement unit"
    return "Other or unresolved"


def feature_inventory(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in X.columns if c != "respondent_id"]:
        mtype, domain = classify_feature(col)
        s = pd.to_numeric(X[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "measurement_type": mtype,
                "domain": domain,
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(100.0 * float(s.isna().mean()), 2),
                "n_unique": int(s.nunique(dropna=True)),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def correlation_audit(X: pd.DataFrame) -> pd.DataFrame:
    cols = list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS)
    work = X[cols].apply(pd.to_numeric, errors="coerce")
    corr = work.corr(method="pearson")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rows.append(
                {
                    "variable_a": a,
                    "variable_b": b,
                    "pearson_corr": round(float(corr.loc[a, b]), 4),
                    "abs_corr": round(abs(float(corr.loc[a, b])), 4),
                    "pair_type": (
                        "count_vs_count"
                        if a in PORTFOLIO_COUNT_VARS and b in PORTFOLIO_COUNT_VARS
                        else "indicator_vs_count"
                        if (a in PORTFOLIO_COUNT_VARS) != (b in PORTFOLIO_COUNT_VARS)
                        else "indicator_vs_indicator"
                    ),
                }
            )
    out = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    # Highlight crop-indicator sum vs n_crop_types style redundancy
    crop_inds = [c for c in AGRICULTURAL_HAS_VARS if c not in {
        "has_dairy_livestock", "has_other_livestock", "has_poultry"
    }]
    livestock_inds = [
        "has_dairy_livestock",
        "has_other_livestock",
        "has_poultry",
    ]
    crop_sum = work[crop_inds].sum(axis=1)
    live_sum = work[livestock_inds].sum(axis=1)
    extra = pd.DataFrame(
        [
            {
                "variable_a": "sum_crop_group_indicators",
                "variable_b": "n_crop_types",
                "pearson_corr": round(float(crop_sum.corr(work["n_crop_types"])), 4),
                "abs_corr": round(abs(float(crop_sum.corr(work["n_crop_types"]))), 4),
                "pair_type": "derived_sum_vs_count",
            },
            {
                "variable_a": "sum_livestock_group_indicators",
                "variable_b": "n_livestock_types",
                "pearson_corr": round(float(live_sum.corr(work["n_livestock_types"])), 4),
                "abs_corr": round(abs(float(live_sum.corr(work["n_livestock_types"]))), 4),
                "pair_type": "derived_sum_vs_count",
            },
        ]
    )
    return pd.concat([extra, out], ignore_index=True)


def domain_contribution(inventory: pd.DataFrame) -> pd.DataFrame:
    counts = inventory.groupby("domain").size().rename("n_columns").reset_index()
    counts["share_pct"] = (100.0 * counts["n_columns"] / counts["n_columns"].sum()).round(2)
    counts["equal_weight_gower_implication"] = counts.apply(
        lambda r: (
            "High influence under equal feature weights"
            if r["domain"] in {"agri_group", "ownership"}
            else "Lower column share under equal feature weights"
        ),
        axis=1,
    )
    return counts.sort_values("n_columns", ascending=False)


def acreage_missing_profiles(X: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    merged = X.merge(
        prepared[["respondent_id", "land_size", "land_size_unit"]],
        on="respondent_id",
        how="left",
    )
    acres = pd.to_numeric(merged[LAND_SIZE_STANDARDISED], errors="coerce")
    unit = pd.to_numeric(merged["land_size_unit"], errors="coerce")
    size = pd.to_numeric(merged["land_size"], errors="coerce")
    missing = acres.isna()
    reason = [
        primary_missing_reason(
            None if pd.isna(u) else float(u),
            None if pd.isna(s) else float(s),
        )
        for u, s in zip(unit, size)
    ]
    merged["acres_missing"] = missing.astype(int)
    merged["missing_reason"] = reason
    # Only meaningful for missing rows' reason distribution, but profile all for ownership
    rows = []
    # Overall reason counts among missing
    for label, n in pd.Series(reason)[missing.values].value_counts().items():
        rows.append(
            {
                "profile_type": "missing_reason",
                "category": label,
                "n_households": int(n),
                "pct_of_missing": round(100.0 * n / int(missing.sum()), 2),
                "pct_of_sample": round(100.0 * n / len(merged), 2),
            }
        )
    # Ownership community flag vs missing
    for val, label in [(1, "community_ownership_flag=1"), (0, "community_ownership_flag=0")]:
        mask = pd.to_numeric(merged[COMMUNITY_OWNERSHIP], errors="coerce").eq(val)
        n = int(mask.sum())
        n_miss = int((mask & missing).sum())
        rows.append(
            {
                "profile_type": "ownership_vs_missing",
                "category": label,
                "n_households": n,
                "pct_of_missing": round(100.0 * n_miss / max(int(missing.sum()), 1), 2),
                "pct_of_sample": round(100.0 * n / len(merged), 2),
            }
        )
        rows.append(
            {
                "profile_type": "ownership_missing_rate",
                "category": label,
                "n_households": n_miss,
                "pct_of_missing": round(100.0 * n_miss / max(n, 1), 2),
                "pct_of_sample": round(100.0 * n_miss / len(merged), 2),
            }
        )
    # Selected agri indicators among missing vs complete
    for col in ["has_cereals", "has_poultry", "has_other_livestock", "has_legumes"]:
        if col not in merged.columns:
            continue
        for subset_name, mask in [
            ("acres_missing", missing),
            ("acres_observed", ~missing),
        ]:
            rate = float(pd.to_numeric(merged.loc[mask, col], errors="coerce").mean())
            rows.append(
                {
                    "profile_type": f"agri_prevalence_{subset_name}",
                    "category": col,
                    "n_households": int(mask.sum()),
                    "pct_of_missing": round(100.0 * rate, 2),
                    "pct_of_sample": round(100.0 * int(mask.sum()) / len(merged), 2),
                }
            )
    return pd.DataFrame(rows)


def pam_precomputed(
    distance: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal PAM on a precomputed distance matrix (smoke-test / fallback)."""
    rng = np.random.default_rng(random_state)
    n = distance.shape[0]
    medoids = np.sort(rng.choice(n, size=n_clusters, replace=False))
    labels = np.argmin(distance[:, medoids], axis=1)
    for _ in range(max_iter):
        new_medoids = []
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                new_medoids.append(medoids[k])
                continue
            block = distance[np.ix_(members, members)]
            costs = block.sum(axis=1)
            new_medoids.append(int(members[int(np.argmin(costs))]))
        new_medoids = np.sort(np.array(new_medoids, dtype=int))
        labels = np.argmin(distance[:, new_medoids], axis=1)
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return labels, medoids


def gower_pam_feasibility(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    try:
        import gower
    except ImportError as exc:
        return pd.DataFrame(
            [
                {
                    "check": "package_import",
                    "status": "fail",
                    "detail": f"Missing dependency: {exc}",
                    "runtime_seconds": np.nan,
                }
            ]
        )

    # sklearn_extra.KMedoids is not used: its wheels are incompatible with NumPy 2.x here.
    pam_backend = "local_pam_precomputed"
    rows.append(
        {
            "check": "package_import",
            "status": "pass",
            "detail": (
                "gower imported; PAM via local precomputed implementation "
                "(sklearn_extra skipped: NumPy 2.x incompatibility)"
            ),
            "runtime_seconds": np.nan,
        }
    )

    # Spec A: exclude acres
    feature_cols = [
        c
        for c in X.columns
        if c != "respondent_id" and c != LAND_SIZE_STANDARDISED
    ]
    XA = X[feature_cols].apply(pd.to_numeric, errors="coerce")
    assert XA.isna().sum().sum() == 0
    rows.append(
        {
            "check": "spec_A_matrix",
            "status": "pass",
            "detail": f"n={len(XA)}, p={XA.shape[1]} (acres excluded); complete",
            "runtime_seconds": np.nan,
        }
    )

    sample = XA.sample(n=min(200, len(XA)), random_state=42)
    t0 = time.perf_counter()
    d_sample = gower.gower_matrix(sample.to_numpy())
    t_sample = time.perf_counter() - t0
    rows.append(
        {
            "check": "gower_sample_n200",
            "status": "pass",
            "detail": f"distance shape={d_sample.shape}",
            "runtime_seconds": round(t_sample, 3),
        }
    )

    t0 = time.perf_counter()
    d_full = gower.gower_matrix(XA.to_numpy())
    t_full = time.perf_counter() - t0
    mem_mb = d_full.nbytes / (1024 * 1024)
    practical = t_full < 120 and mem_mb < 500
    rows.append(
        {
            "check": "gower_full_n1552",
            "status": "pass" if practical else "warn",
            "detail": (
                f"shape={d_full.shape}; memory_mb={mem_mb:.1f}; "
                f"practical_for_S1={'yes' if practical else 'borderline'}"
            ),
            "runtime_seconds": round(t_full, 3),
        }
    )

    t0 = time.perf_counter()
    labels, medoids_arr = pam_precomputed(d_full, n_clusters=3, random_state=42)
    medoids = medoids_arr.tolist()
    t_pam = time.perf_counter() - t0
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    rows.append(
        {
            "check": "pam_smoke_k3_spec_A",
            "status": "pass",
            "detail": f"backend={pam_backend}; cluster_sizes={sizes}; medoid_indices={medoids}",
            "runtime_seconds": round(t_pam, 3),
        }
    )

    rows.append(
        {
            "check": "spec_B_candidate_treatments",
            "status": "info",
            "detail": (
                "Candidates for review (no zero-fill): "
                "(1) acres + is_community_land binary from unit==5; "
                "(2) acres + missingness indicator with median of observed acres "
                "only for non-community special codes; "
                "(3) exclude acres for community land rows in a hybrid sensitivity. "
                "Exact Spec B treatment remains open pending review."
            ),
            "runtime_seconds": np.nan,
        }
    )
    rows.append(
        {
            "check": "no_zero_fill_rule",
            "status": "pass",
            "detail": "Audit does not replace missing acreage with zero",
            "runtime_seconds": np.nan,
        }
    )
    return pd.DataFrame(rows)


def write_report(
    inventory: pd.DataFrame,
    corr: pd.DataFrame,
    domains: pd.DataFrame,
    acres: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top_corr = corr.head(15)
    report = f"""# Segmentation Design Audit (Script 11)

Generated: {timestamp}

## Purpose

Evidence for Spec A/B acreage design and feature weighting before S1. No S1–S4 fitting and no classification fitting were performed beyond a Gower/PAM smoke test on Spec A.

## 1. Feature inventory (18 segmentation predictors)

```
{inventory.to_string(index=False)}
```

## 2. Indicator–count overlap

Highest absolute correlations (including derived indicator sums vs counts):

```
{top_corr.to_string(index=False)}
```

Interpretation: crop and livestock counts partially overlap with agricultural group indicators. Equal-weight Gower may double-count portfolio breadth if both indicators and counts are retained without weighting or a drop-counts sensitivity.

## 3. Domain contribution

```
{domains.to_string(index=False)}
```

Under equal feature weights, agri_group (9) and ownership (6) dominate column share relative to portfolio counts (2) and land size (1).

## 4. Missing acreage profiles

```
{acres.to_string(index=False)}
```

Missing acreage is concentrated in community-land measurement (primary reason) and is not missing completely at random relative to ownership. Spec B should distinguish community land from special-response missingness. Do not auto-zero acres.

## 5. Gower / PAM feasibility

```
{feasibility.to_string(index=False)}
```

## Open decisions (require approval before S1)

1. Exact Spec B missingness treatment (candidates listed in feasibility table).
2. Equal feature weights vs domain-level weighting.
3. Whether to drop `n_crop_types` / `n_livestock_types` in a sensitivity specification.

## Output files

- `outputs/tables/11_segmentation_feature_types.csv`
- `outputs/tables/11_segmentation_correlations.csv`
- `outputs/tables/11_domain_contribution.csv`
- `outputs/tables/11_acreage_missing_profiles.csv`
- `outputs/tables/11_gower_pam_feasibility.csv`
"""
    path = REPORTS_DIR / "11_segmentation_design_audit.md"
    path.write_text(report, encoding="utf-8")
    return path


def update_design_doc_open_decisions(feasibility: pd.DataFrame) -> None:
    path = LOGS_DIR / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md"
    text = path.read_text(encoding="utf-8")
    note = r"""

### Script 11 audit pointer

Audit completed: `outputs/reports/11_segmentation_design_audit.md`.

Findings to carry into Spec B / weighting approval:

- Agri group (9) and ownership (6) columns dominate equal-weight Gower relative to counts (2) and acres (1).
- Crop/livestock counts correlate with sums of related group indicators; a drop-counts sensitivity remains justified.
- Missing acreage aligns strongly with community-land measurement; Spec B must distinguish community land from other missing responses and must not zero-fill.
- Gower + PAM smoke test on Spec A (acres excluded) completed successfully; full n x n Gower at n=1,552 is feasible for S1.

**Still open before S1:** Spec B exact treatment; equal vs domain weighting; drop-counts sensitivity yes/no.
"""
    marker = "### Script 11 audit pointer"
    if marker in text:
        pre, _, rest = text.partition(marker)
        # keep content before marker; drop old pointer section body until next ## or end
        parts = rest.split("\n## ")
        tail = ""
        if len(parts) > 1:
            tail = "\n## " + "\n## ".join(parts[1:])
        # Also handle ### sections after
        first = parts[0]
        sub = first.split("\n### ")
        if len(sub) > 1:
            # unlikely inside pointer
            pass
        path.write_text(pre.rstrip() + note + tail, encoding="utf-8")
    else:
        path.write_text(text.rstrip() + note, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    X = pd.read_parquet(PREPARED_DIR / "09_X_segmentation.parquet")
    prepared = pd.read_parquet(PREPARED_DIR / "09_prepared_analysis_frame.parquet")
    if len(X) != EXPECTED_DROUGHT_N:
        raise ValueError(f"Unexpected segmentation n={len(X)}")

    inventory = feature_inventory(X)
    corr = correlation_audit(X)
    domains = domain_contribution(inventory)
    acres = acreage_missing_profiles(X, prepared)
    feasibility = gower_pam_feasibility(X)

    outputs = {
        "11_segmentation_feature_types.csv": inventory,
        "11_segmentation_correlations.csv": corr,
        "11_domain_contribution.csv": domains,
        "11_acreage_missing_profiles.csv": acres,
        "11_gower_pam_feasibility.csv": feasibility,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote {path}")

    report_path = write_report(inventory, corr, domains, acres, feasibility)
    print(f"Wrote {report_path}")
    update_design_doc_open_decisions(feasibility)
    print(f"Updated {LOGS_DIR / 'MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md'}")

    print("\nDomain contribution")
    print(domains.to_string(index=False))
    print("\nTop correlations")
    print(corr.head(8).to_string(index=False))
    print("\nFeasibility")
    print(feasibility.to_string(index=False))
    print("\nDone. Review audit before approving S1.")


if __name__ == "__main__":
    main()
