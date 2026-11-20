#!/usr/bin/env python3
"""Script 03 — Descriptive analysis of the all-shocks population."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CLASS_ORDER,
    FIGURES,
    PREPARED,
    SHOCK_INDICATOR_VARS,
    TABLES,
    ensure_dirs,
)


def pct_table(counts: pd.Series) -> pd.DataFrame:
    out = counts.rename_axis("category").reset_index(name="n")
    out["pct"] = (100 * out["n"] / out["n"].sum()).round(2)
    return out


def main() -> None:
    ensure_dirs()
    exposed = pd.read_parquet(PREPARED / "all_shocks_farming_cohort.parquet")
    model = pd.read_parquet(PREPARED / "all_shocks_modelling_cohort.parquet")
    shocks = [c for c in SHOCK_INDICATOR_VARS if c in exposed.columns]
    print(f"Descriptive cohort (shock-exposed) n={len(exposed)}; modelling n={len(model)}")

    # A. Shock frequencies (multi-response — households can appear in multiple)
    freq_rows = []
    for c in shocks:
        n = int((pd.to_numeric(exposed[c], errors="coerce") == 1).sum())
        freq_rows.append(
            {
                "shock": c,
                "n_households_reporting": n,
                "pct_of_shock_exposed": round(100 * n / len(exposed), 2),
                "note": "Multi-response: households may appear in several rows",
            }
        )
    freq = pd.DataFrame(freq_rows).sort_values("n_households_reporting", ascending=False)
    freq.to_csv(TABLES / "03_shock_frequencies.csv", index=False)

    # Weighted shares (supplementary)
    if "household_weight" in exposed.columns:
        w_rows = []
        tw = float(exposed["household_weight"].sum())
        for c in shocks:
            flag = (pd.to_numeric(exposed[c], errors="coerce") == 1).astype(float)
            w = float((flag * exposed["household_weight"]).sum())
            w_rows.append(
                {
                    "shock": c,
                    "weighted_sum": round(w, 1),
                    "pct_weighted_of_exposed": round(100 * w / tw, 2) if tw else None,
                }
            )
        pd.DataFrame(w_rows).to_csv(TABLES / "03_shock_frequencies_weighted.csv", index=False)

    # B. Number of shocks
    nshock = exposed["number_of_shocks_reported"].value_counts().sort_index()
    nshock_tbl = pct_table(nshock)
    nshock_tbl.to_csv(TABLES / "03_number_of_shocks_distribution.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(nshock.index.astype(str), nshock.values, color="#1c6a84")
    plt.xlabel("Number of different shocks reported")
    plt.ylabel("Households")
    plt.title("Distribution of number_of_shocks_reported (shock-exposed)")
    plt.tight_layout()
    plt.savefig(FIGURES / "03_number_of_shocks.png", dpi=120)
    plt.close()

    # C. Co-occurrence
    mat = exposed[shocks].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    # Only count confirmed 1s
    bin_mat = (mat == 1).astype(int)
    co = bin_mat.T.dot(bin_mat)
    co.to_csv(TABLES / "03_shock_cooccurrence_counts.csv")
    # Jaccard-ish pairwise for viz: co[i,j] / max
    plt.figure(figsize=(10, 8))
    plt.imshow(co.values, cmap="Blues")
    plt.colorbar(label="Households reporting both")
    short = [c.replace("shock_", "")[:28] for c in shocks]
    plt.xticks(range(len(short)), short, rotation=90, fontsize=7)
    plt.yticks(range(len(short)), short, fontsize=7)
    plt.title("Shock co-occurrence counts (multi-response)")
    plt.tight_layout()
    plt.savefig(FIGURES / "03_shock_cooccurrence.png", dpi=120)
    plt.close()

    # D. Coping distribution (modelling cohort — has outcomes)
    cope = model["coping_category"].value_counts().reindex(CLASS_ORDER)
    cope_tbl = pct_table(cope)
    cope_tbl.to_csv(TABLES / "03_coping_distribution_fourclass.csv", index=False)

    by_n = (
        model.groupby(["number_of_shocks_reported", "coping_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_ORDER)
    )
    by_n_pct = by_n.div(by_n.sum(axis=1), axis=0) * 100
    by_n.to_csv(TABLES / "03_coping_by_n_shocks_counts.csv")
    by_n_pct.round(2).to_csv(TABLES / "03_coping_by_n_shocks_pct.csv")

    # E. Coping by individual shock (multi-response groups)
    rows = []
    for c in shocks:
        sub = model.loc[pd.to_numeric(model[c], errors="coerce") == 1]
        if len(sub) == 0:
            continue
        for cat in CLASS_ORDER:
            n = int((sub["coping_category"] == cat).sum())
            rows.append(
                {
                    "shock": c,
                    "coping_category": cat,
                    "n": n,
                    "pct_within_shock_group": round(100 * n / len(sub), 2),
                    "shock_group_n": len(sub),
                    "note": "Households with multiple shocks appear in multiple shock groups",
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "03_coping_by_r2a_shock.csv", index=False)

    # F. Coping by main shock (R2B) — separate analysis
    main_rows = []
    for code, sub in model.groupby("main_shock_type"):
        label = (
            sub["main_shock_type_label"].iloc[0]
            if "main_shock_type_label" in sub.columns
            else str(code)
        )
        for cat in CLASS_ORDER:
            n = int((sub["coping_category"] == cat).sum())
            main_rows.append(
                {
                    "main_shock_code": code,
                    "main_shock_label": label,
                    "coping_category": cat,
                    "n": n,
                    "pct_within_main_shock": round(100 * n / len(sub), 2),
                    "main_shock_group_n": len(sub),
                }
            )
    pd.DataFrame(main_rows).to_csv(TABLES / "03_coping_by_main_shock_r2b.csv", index=False)

    # Bar of overall coping
    plt.figure(figsize=(6, 4))
    plt.bar([str(i) for i in CLASS_ORDER], cope.fillna(0).values, color="#1c6a84")
    plt.ylabel("Households")
    plt.title("Four-class coping (all-shocks modelling cohort)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES / "03_coping_distribution.png", dpi=120)
    plt.close()

    print("Top shocks:\n", freq.head(5).to_string(index=False))
    print("Coping:\n", cope_tbl.to_string(index=False))
    print("Script 03 complete.")


if __name__ == "__main__":
    main()
