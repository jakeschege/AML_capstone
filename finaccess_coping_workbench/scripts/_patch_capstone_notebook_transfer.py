#!/usr/bin/env python3
"""One-shot patch: transfer workbench narrative into the capstone notebook.

Does not re-run modelling. Appends markdown + artefact-loading cells.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(
    "/home/jakes/Documents/strathmore/Modules/Module 4/"
    "AppliedMachineLearning/capstone_project/"
    "capstone_data_preparation (1).ipynb"
)


def md(text: str) -> dict:
    # Ensure trailing newline like jupyter
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + (
            [text.split("\n")[-1] + "\n"] if text.split("\n")[-1] else []
        ),
    }


def code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1] != "":
        src.append(lines[-1] + "\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def fix_md_source(text: str) -> dict:
    """Markdown cell with clean source list."""
    text = text.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": [text] if "\n" not in text[:-1] else [l + "\n" for l in text.split("\n")[:-1]] + ([text.split("\n")[-1] + "\n"] if False else []),
    }


def markdown_cell(text: str) -> dict:
    text = text.strip("\n") + "\n"
    parts = text.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": parts,
    }


def code_cell(text: str) -> dict:
    text = text.strip("\n") + "\n"
    parts = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": parts,
    }


def set_markdown_source(cell: dict, text: str) -> None:
    text = text.strip("\n") + "\n"
    cell["source"] = text.splitlines(keepends=True)


INTRO_OVERVIEW = """## Notebook overview

This notebook presents the FinAccess farmer drought-coping capstone as a CRISP-DM workflow.

**Part A / Part B** build the 2024 modelling frame and the separate 2021 validation frame from the raw Stata files.

**Business Understanding** and **Data Understanding** establish the problem, sample and exploratory findings.

**Data Preparation** cleans and encodes features for the drought-exposed analytical sample (n = 1,552). Detailed preparation audits and locked feature matrices were produced in the companion `finaccess_coping_workbench/` and are summarised here by loading those artefacts.

**Modelling** covers unsupervised agricultural/land segmentation (Spec A working typology, k = 5) and supervised coping classification under a locked repeated stratified cross-validation protocol.

**Evaluation** synthesises what the segments and classifiers do and do not establish. Heavy computation (Gower + PAM, ensemble fits) is not re-run inside this notebook; results are loaded from reviewed workbench outputs.
"""

BRIDGE_MD = """## Workbench artefact bridge

The reviewed preparation and modelling pipeline lives in `finaccess_coping_workbench/`. The cells below locate that folder relative to the `capstone_project` directory and check that the locked artefacts exist.

This notebook **loads** tables and prepared matrices. It does **not** regenerate Script 19 cross-validation folds or re-fit C2/C3 ensembles.
"""

BRIDGE_CODE = """from pathlib import Path

# Resolve capstone_project root whether the kernel cwd is capstone_project or a subfolder
_cwd = Path.cwd().resolve()
if (_cwd / "finaccess_coping_workbench").is_dir():
    CAPSTONE_ROOT = _cwd
elif (_cwd.parent / "finaccess_coping_workbench").is_dir():
    CAPSTONE_ROOT = _cwd.parent
else:
    CAPSTONE_ROOT = Path("/home/jakes/Documents/strathmore/Modules/Module 4/AppliedMachineLearning/capstone_project")

WORKBENCH = CAPSTONE_ROOT / "finaccess_coping_workbench"
TABLES = WORKBENCH / "outputs" / "tables"
REPORTS = WORKBENCH / "outputs" / "reports"
PREPARED = WORKBENCH / "outputs" / "prepared"
LOGS = WORKBENCH / "logs"

required = [
    PREPARED / "09_X_primary_classification.parquet",
    PREPARED / "09_X_extended_classification.parquet",
    PREPARED / "09_X_segmentation.parquet",
    PREPARED / "09_y_targets.parquet",
    TABLES / "19_cv_folds_long.csv",
    TABLES / "19_classification_protocol.json",
    TABLES / "20_c1_scorecard.csv",
    TABLES / "21_c2c3_scorecard.csv",
    TABLES / "21_c2c3_vs_c1_comparison.csv",
    TABLES / "12_s1_spec_a_assignments_wide.csv",
    LOGS / "MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md",
]

missing = [str(p) for p in required if not p.exists()]
print("CAPSTONE_ROOT:", CAPSTONE_ROOT)
print("WORKBENCH:", WORKBENCH)
print("Artefacts OK:" if not missing else "MISSING ARTEFACTS:")
if missing:
    for m in missing:
        print(" -", m)
else:
    for p in required:
        print(" -", p.relative_to(CAPSTONE_ROOT))
assert not missing, "Workbench artefacts missing; run workbench scripts before continuing"
"""

# --- Data prep continuation cells ---
DP_CELLS = [
    markdown_cell(
        """### Shannon diversity conclusion (Script 02)

The crosstabs above show that Shannon values align closely with the number of active agricultural groups (approximately $\\ln(g)$), while different crop/livestock count combinations can share the same Shannon value.

**Decision carried into feature preparation**

- Retain `n_crop_types`, `n_livestock_types` and the nine agricultural binary indicators as default candidates.
- Keep `shannon_diversity` as a **sensitivity sidecar only**, not a default segmentation or classification predictor.
- Do not treat Shannon and the portfolio counts as interchangeable.

Full investigation: `finaccess_coping_workbench/outputs/reports/02_shannon_portfolio_investigation.md`."""
    ),
    markdown_cell(
        """### Demographic preparation notes (Script 03)

In the drought-exposed sample (n = 1,552):

- No missing values in `age_years`, `sex`, `education`, `urban_rural`, or `county`.
- `education` codes 1–4 form an ordinal schooling ladder (code 1 = None / no formal schooling).
- All 47 counties appear; sparse counties were retained without collapsing.

Encoding to modelling-ready columns (`sex_bin`, `urban_rural_bin`, county one-hots) was applied later in Script 09."""
    ),
    markdown_cell(
        """### Shock constructs (Script 04)

- `shock_drought_loss` is constant in the analytical sample and is not used as an ordinary predictor.
- Eleven additional shock indicators and `main_shock_type` are complete and remain candidates for the **extended** classification matrix only.
- Shock information shares the R2 retrospective window with coping responses and must not be described as prospective predictors."""
    ),
    markdown_cell(
        """### Financial access exclusion (Scripts 05 and 07)

Financial access and usage variables are complete but temporally unresolved relative to coping (overlapping past-12-month windows).

**Decision:** retain them in the prepared analysis frame for traceability, but **exclude** them from default primary and extended classification predictor matrices and from agricultural segmentation inputs."""
    ),
    markdown_cell(
        """### Leakage and stage suitability (Script 06)

Confirmed exclusions from predictor matrices:

- Coping action indicators and target columns
- Constant drought sample flags as ordinary predictors
- Financial access variables under the temporal restriction
- Shock indicators from the **primary** classification matrix (extended only)

Segmentation inputs remain agricultural portfolio, land size and land ownership only."""
    ),
    markdown_cell(
        """### Missingness, encoding and land reconciliation (Scripts 08 / 08b)

| Topic | Decision |
|---|---|
| Standard land measure | `land_size_acres` (missing retained; n = 198) |
| `land_size_unit` | Excluded as a predictor |
| Missing acres | 182 community land + 16 special codes 98/99 (non-community); never zero-filled |
| `education` | Ordered codes 1–4 |
| `shannon_diversity` | Sensitivity only |
| Imputation / scaling in preparation | Not applied |

Report: `outputs/reports/08_missingness_and_encoding_audit.md`."""
    ),
    markdown_cell(
        """### Feature preparation and validation (Scripts 09–10)

Script 09 built locked feature matrices without train/validation splits. Script 10 validated them (all checks passed).

| Feature set | Features | Artefact |
|---|---:|---|
| Segmentation | 18 | `outputs/prepared/09_X_segmentation.parquet` |
| Primary classification | 69 | `outputs/prepared/09_X_primary_classification.parquet` |
| Extended classification | 92 | `outputs/prepared/09_X_extended_classification.parquet` |
| Shannon sensitivity | 1 | `outputs/prepared/09_X_sensitivity_shannon.parquet` |
| Targets | four- and three-class | `outputs/prepared/09_y_targets.parquet` |

**Data Preparation gate:** phase closed. Modelling uses these artefacts. Any later imputation or scaling must be fit inside training folds only."""
    ),
    code_cell(
        """# Load prepared feature inventories and target counts from the workbench
feature_sets = {
    "segmentation": pd.read_csv(TABLES / "09_feature_list_segmentation.csv"),
    "primary_classification": pd.read_csv(TABLES / "09_feature_list_primary_classification.csv"),
    "extended_classification": pd.read_csv(TABLES / "09_feature_list_extended_classification.csv"),
    "sensitivity": pd.read_csv(TABLES / "09_feature_list_sensitivity.csv"),
}
y_prep = pd.read_parquet(PREPARED / "09_y_targets.parquet")

summary = pd.DataFrame(
    {
        "feature_set": list(feature_sets),
        "n_features": [len(df) for df in feature_sets.values()],
    }
)
display(summary)
print("\\nFour-class target counts")
display(y_prep["coping_category"].value_counts().rename_axis("coping_category").to_frame("n"))
print("Three-class target counts")
display(
    y_prep["coping_category_collapsed"]
    .value_counts()
    .rename_axis("coping_category_collapsed")
    .to_frame("n")
)
print("Sample n =", len(y_prep))
assert len(y_prep) == 1552"""
    ),
    markdown_cell(
        """## Data Preparation phase conclusion

Data Preparation is complete for the drought-exposed 2024 sample (n = 1,552).

Locked outputs live under `finaccess_coping_workbench/outputs/prepared/`. The external FinAccess modelling frame was not modified. Coping outcomes were not used to construct feature matrices.

Carry-forward issues for Modelling:

1. Handle missing `land_size_acres` within training folds (or exclude acres — later locked as the classification default).
2. Formal coping class is small (n = 75); use stratified validation and class-wise metrics.
3. Extended shock predictors are same-window retrospective information."""
    ),
]

SEG_CELLS = [
    markdown_cell(
        """# Modelling

Modelling is split into two workflows that were designed and evaluated separately:

1. **Segmentation** — unsupervised agricultural/land household groups (coping outcomes excluded).
2. **Classification** — supervised prediction of coping category from household/agri/land characteristics.

Design rules and experiment locks are recorded in `finaccess_coping_workbench/logs/MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md`. The notebook loads reviewed outputs rather than re-fitting Gower + PAM or ensemble models."""
    ),
    markdown_cell(
        """## Modelling — Segmentation

### Design summary

| Element | Locked choice |
|---|---|
| Distance | Gower |
| Algorithm | PAM (Partitioning Around Medoids) |
| Spec A (primary) | Exclude `land_size_acres`; 17 features; equal weights |
| Spec B | Acreage sensitivity with missing-aware Gower |
| Candidate k | 2–6 (focus on 4 and 5 after S1) |
| Coping in clustering | Excluded |

Script 11 audited feature overlap and computational feasibility before S1."""
    ),
    code_cell(
        """# Spec A silhouette and cluster sizes (Script 12)
sil = pd.read_csv(TABLES / "12_s1_spec_a_silhouette_by_k.csv")
sizes = pd.read_csv(TABLES / "12_s1_spec_a_cluster_sizes.csv")
print("Silhouette by k (Spec A)")
display(sil)
print("Cluster sizes for k in {2,4,5}")
display(sizes.loc[sizes["k"].isin([2, 4, 5])])"""
    ),
    markdown_cell(
        """### Stability, robustness and Spec B (Scripts 13–16)

- **S2 (Script 13):** resampling stability for promising k.
- **Script 14:** k = 5 family-crop subtypes are softer than the other groups; k = 4 remains a comparator.
- **Spec B (Script 15):** land-size sensitivity with community acres left missing (never zero-filled).
- **Correspondence (Script 16):** Spec A/B membership overlap; Spec B retained as sensitivity only.

### Working typology decision (locked)

**Primary assignments: Spec A, k = 5.**

| cluster_k5 | Working label |
|---|---|
| 0 | Family-owned legume-oriented crop households |
| 1 | Own-name crop households |
| 2 | Community-land livestock-oriented households |
| 3 | Family-owned mixed crop and livestock households |
| 4 | Family-owned cereal-oriented households |

k = 4 is retained as a sensitivity alternative. This typology is for description; it does not establish that segments predict coping."""
    ),
    code_cell(
        """# Spec A k=5 assignments and sizes
assign = pd.read_csv(TABLES / "12_s1_spec_a_assignments_wide.csv")
labels = {
    0: "Family-owned legume-oriented crop households",
    1: "Own-name crop households",
    2: "Community-land livestock-oriented households",
    3: "Family-owned mixed crop and livestock households",
    4: "Family-owned cereal-oriented households",
}
seg_sizes = (
    assign["cluster_k5"]
    .value_counts()
    .sort_index()
    .rename_axis("cluster_k5")
    .to_frame("n_households")
)
seg_sizes["working_label"] = seg_sizes.index.map(labels)
seg_sizes["pct"] = (100 * seg_sizes["n_households"] / len(assign)).round(2)
display(seg_sizes)
assert len(assign) == 1552"""
    ),
    markdown_cell(
        """### Coping profiles across segments (Scripts 17–18)

Descriptive only: unweighted counts and **household-weighted** proportions (`household_weight` for survey representation — not used later as a classification fitting weight).

Formal coping is sparse (about 9–19 households per cluster). Class-wise formal comparisons are exploratory.

Script 18 compared k = 4 vs k = 5. The distinctive community-livestock pattern (higher personal / lower informal / higher did-nothing) remained recognisable under k = 4 (294 overlapping households; 99.3% of the k = 5 cluster)."""
    ),
    code_cell(
        """# Four-class unweighted and household-weighted within-cluster percentages (Script 17)
counts4 = pd.read_csv(TABLES / "17_coping_counts_unweighted_fourclass.csv")
props4 = pd.read_csv(TABLES / "17_coping_props_weighted_fourclass.csv")
formal = pd.read_csv(TABLES / "17_formal_by_cluster.csv")

unw = (
    counts4.loc[counts4["coping_category"] != "TOTAL"]
    .pivot_table(
        index=["cluster_k5", "cluster_label"],
        columns="coping_category",
        values="pct_within_cluster",
        aggfunc="first",
    )
    .reindex(columns=["personal", "informal", "formal", "did_nothing"])
)
wt = props4.pivot_table(
    index=["cluster_k5", "cluster_label"],
    columns="coping_category",
    values="pct_within_cluster_weighted",
    aggfunc="first",
).reindex(columns=["personal", "informal", "formal", "did_nothing"])

print("Unweighted % within cluster")
display(unw.round(1))
print("Household-weighted % within cluster")
display(wt.round(1))
print("Formal counts (sparse flag n<30)")
display(formal)

# k=4 sensitivity focal delta (Script 18)
focal = pd.read_csv(TABLES / "18_focal_livestock_coping_delta.csv")
print("Focal livestock/community segment: k=4 minus k=5 (pp)")
display(focal)"""
    ),
]

CLS_CELLS = [
    markdown_cell(
        """## Modelling — Classification

### Protocol lock (Script 19) — accepted before fitting

| Track | Target | Predictors |
|---|---|---|
| Primary | Four-class `coping_category` | Primary matrix (69); acres handling decided after C1 |
| Secondary | Three-class collapsed | Same primary matrix / folds (**deferred** after C1–C3) |
| Extended | Four-class | Extended matrix (92) with retrospective shocks (**deferred**) |

- CV: repeated stratified 5-fold × 5 repeats, seed 42; **identical folds** for fair comparisons
- Primary metric: **macro F1**; also balanced accuracy, class-wise P/R/F1, confusion matrices
- Imbalance: compare unweighted vs `class_weight='balanced'`; no SMOTE
- **Survey weights ≠ class weights:** `household_weight` is for descriptive population estimates only; it is not passed into model fitting or fold metrics by default
- Fold SDs describe variation across the 25 splits; they are **not** confidence intervals and must not be used for significance testing
- Folds are loaded from Script 19 and must not be regenerated"""
    ),
    code_cell(
        """import json

protocol = json.loads((TABLES / "19_classification_protocol.json").read_text())
fold_counts = pd.read_csv(TABLES / "19_fold_class_counts.csv")
formal_test = fold_counts.loc[
    (fold_counts["split"] == "test") & (fold_counts["coping_category"] == "formal"),
    ["repeat", "fold", "n_households", "split_n"],
]

print("CV settings:", protocol["cv"])
print("Primary metric:", protocol["metrics"]["primary"])
print("\\nFormal class in test folds (should be 15 everywhere)")
display(formal_test.agg({"n_households": ["min", "max", "mean", "count"]}))
assert formal_test["n_households"].min() == 15
assert formal_test["n_households"].max() == 15"""
    ),
    markdown_cell(
        """### C1 baselines (Script 20) — accepted

Dummy (most frequent) and multinomial logistic regression on the four-class primary track.

**Land-size decision after C1:** acres-excluded is the **default** specification (nearly identical to train-fold median imputation). Imputed logistic remains a documented sensitivity. No missingness indicator without separate justification.

Class-weighted logistic improved formal F1 and balanced accuracy relative to unweighted logistic, at some cost to overall accuracy. Formal metrics remain exploratory (15 test cases per fold)."""
    ),
    code_cell(
        """c1 = pd.read_csv(TABLES / "20_c1_scorecard.csv")
cols = [
    "experiment_id",
    "land_treatment",
    "class_weight_mode",
    "macro_f1_mean",
    "macro_f1_std",
    "balanced_accuracy_mean",
    "weighted_f1_mean",
    "f1_formal_mean",
    "f1_formal_std",
    "accuracy_mean",
]
display(c1[cols])

# Mean repetition confusion for class-weighted acres-exclude logistic
mean_cm = pd.read_csv(TABLES / "20_c1_mean_repetition_confusion.csv")
best = "logistic_class_weighted_exclude"
cm = (
    mean_cm.loc[mean_cm["experiment_id"] == best]
    .pivot(index="true_label", columns="pred_label", values="n_mean_across_repeats")
    .reindex(index=["personal", "informal", "formal", "did_nothing"],
             columns=["personal", "informal", "formal", "did_nothing"])
)
print(f"\\nMean OOF confusion (across repetitions): {best}")
display(cm)"""
    ),
    markdown_cell(
        """### C2 / C3 nonlinear models (Script 21)

Decision tree, random forest and histogram gradient boosting on the same folds, **acres excluded**, unweighted vs class-weighted. No hyperparameter search. No household weights in fitting.

**Result:** no C2/C3 configuration beat acres-excluded class-weighted logistic on mean macro F1 or balanced accuracy. Class-weighted random forest was the strongest nonlinear model but remained below the logistic benchmark.

**Selection principle:** do not choose a model solely for highest macro F1; inspect balanced accuracy, class-wise precision/recall (especially formal), confusion matrices, fold variation and false-positive trade-offs."""
    ),
    code_cell(
        """comparison = pd.read_csv(TABLES / "21_c2c3_vs_c1_comparison.csv")
display(comparison.sort_values(["stage", "macro_f1_mean"], ascending=[True, False]))

mean_cm21 = pd.read_csv(TABLES / "21_c2c3_mean_repetition_confusion.csv")
rf_cw = "random_forest_class_weighted_exclude"
cm_rf = (
    mean_cm21.loc[mean_cm21["experiment_id"] == rf_cw]
    .pivot(index="true_label", columns="pred_label", values="n_mean_across_repeats")
    .reindex(index=["personal", "informal", "formal", "did_nothing"],
             columns=["personal", "informal", "formal", "did_nothing"])
)
print(f"\\nMean OOF confusion: {rf_cw}")
display(cm_rf)"""
    ),
]

EVAL_CELLS = [
    markdown_cell(
        """# Evaluation

### Segmentation

The Spec A five-cluster typology provides an interpretable agricultural/land working typology. Descriptive coping profiles differ across segments, most clearly for the community-land livestock-oriented group. The k = 4 sensitivity showed that this contrast is not an artefact of choosing five clusters. Segmentation did not use coping labels and does not, by itself, demonstrate predictive segment effects.

### Classification

Under the locked primary four-class protocol:

- Logistic regression clearly beats a most-frequent dummy baseline on macro F1.
- Class weighting helps minority (formal) detection but does not create reliable formal prediction (formal F1 remains low and variable).
- Tree and boosting ensembles did not improve on the logistic benchmark under default settings.
- Discrimination is modest overall; the study should not claim strong predictive performance for operational targeting.

Three-class secondary and extended retrospective tracks remain **deferred** until explicitly reviewed.

### What this study does not claim

- Causal effects of farm type on coping
- Prospective prediction before shock exposure
- Deployment-ready classification for programme targeting"""
    ),
    markdown_cell(
        """# Deployment and next steps

Operational deployment is **out of scope** for this capstone. The artefacts support academic reporting and further research design.

Possible next steps (not executed here):

- Three-class secondary classification under the same Script 19 folds
- Extended retrospective shock matrix comparison (clearly labelled retrospective)
- Survey-weighted estimation as a separately justified sensitivity
- Limited, pre-registered hyperparameter exploration if a stronger primary benchmark is required"""
    ),
    markdown_cell(
        """# Reproducibility appendix

Workbench scripts are the source of truth for computation. From `capstone_project`:

```bash
.venv/bin/python finaccess_coping_workbench/scripts/09_feature_preparation.py
.venv/bin/python finaccess_coping_workbench/scripts/10_final_preparation_validation.py
.venv/bin/python finaccess_coping_workbench/scripts/12_segmentation_s1_spec_a.py
# ... through ...
.venv/bin/python finaccess_coping_workbench/scripts/21_classification_c2_c3_nonlinear.py
```

Do **not** regenerate `19_cv_folds_long.csv` when comparing models; Scripts 20–21 read the locked fold file.

Design logs:

- `finaccess_coping_workbench/logs/DATA_PREPARATION_LOG.md`
- `finaccess_coping_workbench/logs/MODELLING_DESIGN_AND_EVALUATION_STRATEGY.md`"""
    ),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # 1) Update intro overview (cell 1)
    set_markdown_source(cells[1], INTRO_OVERVIEW)

    # 2) Insert workbench bridge after setup imports (after cell 5)
    # Avoid double-insert if re-run
    already = any(
        "Workbench artefact bridge" in "".join(c.get("source", []))
        for c in cells
    )
    if not already:
        bridge = [markdown_cell(BRIDGE_MD), code_cell(BRIDGE_CODE)]
        cells[6:6] = bridge

    # 3) Replace trailing empty code cell with transition, then append sections
    # Find empty trailing code cells
    while cells and cells[-1]["cell_type"] == "code" and "".join(cells[-1].get("source", [])).strip() == "":
        cells.pop()

    # Guard against duplicate append
    if not any(
        "Data Preparation phase conclusion" in "".join(c.get("source", []))
        for c in cells
    ):
        transition = markdown_cell(
            """### Continuing Data Preparation from workbench audits

The notebook investigation of Shannon diversity aligns with workbench Script 02. The following subsections summarise the remaining preparation decisions (Scripts 02–10) and load the locked prepared artefacts. Detailed numerical audits remain in the workbench reports."""
        )
        cells.extend([transition] + DP_CELLS + SEG_CELLS + CLS_CELLS + EVAL_CELLS)

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NB_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
