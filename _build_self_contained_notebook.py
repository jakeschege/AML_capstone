#!/usr/bin/env python3
"""Build self-contained modelling sections into the capstone notebook.

Presentation contains no workbench/script provenance language.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "capstone_data_preparation (1).ipynb"


def md(text: str) -> dict:
    text = text.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    text = text.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def set_src(cell: dict, text: str) -> None:
    text = text.strip("\n") + "\n"
    cell["source"] = text.splitlines(keepends=True)


OVERVIEW = """## Notebook overview

This notebook presents a CRISP-DM analysis of coping behaviour among Kenyan farming households that reported drought-related losses in FinAccess 2024.

**Part A / Part B** construct the 2024 modelling frame and a separate 2021 validation frame from the survey files.

**Business Understanding** and **Data Understanding** define the problem, describe the sample and document exploratory findings.

**Data Preparation** cleans land and portfolio measures, records encoding decisions and builds the feature matrices used for modelling (drought-exposed sample, n = 1,552).

**Modelling** has two separate workflows: unsupervised agricultural/land segmentation, then supervised classification of the four-class coping category under a fixed repeated stratified cross-validation protocol.

**Evaluation** synthesises what the segments and classifiers establish, including limitations for the sparse formal class.

Intermediate modelling artefacts are stored under `notebook_run_cache/` so expensive distance and cross-validation steps need not be recomputed on every re-run. Deleting that folder forces a full recompute.
"""

SETUP_CODE = r'''from pathlib import Path
import hashlib
import json
import math
import re
import time
import warnings

import numpy as np
import pandas as pd
import pyreadstat

from IPython.display import display

# Optional modelling imports (used from Data Preparation onwards)
import gower
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

# PROJECT_ROOT = Path("/content/drive/MyDrive/fsd/finaccess")  # Colab alternative
PROJECT_ROOT = Path.home() / "Documents" / "items" / "fsd" / "finaccess"
PATH_2024 = PROJECT_ROOT / "2024" / "Data 16+.dta"
PATH_2021 = PROJECT_ROOT / "2021" / "Data 2021 16+.dta"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
VALUE_LABELS_PATH = PROJECT_ROOT / "2024" / "value_labels.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Notebook-owned cache for expensive modelling intermediates
_cwd = Path.cwd().resolve()
if (_cwd / "capstone_data_preparation (1).ipynb").exists() or (_cwd / "notebook_run_cache").exists():
    CAPSTONE_ROOT = _cwd
elif (_cwd.parent / "capstone_data_preparation (1).ipynb").exists():
    CAPSTONE_ROOT = _cwd.parent
else:
    CAPSTONE_ROOT = Path("/home/jakes/Documents/strathmore/Modules/Module 4/AppliedMachineLearning/capstone_project")

RUN_CACHE = CAPSTONE_ROOT / "notebook_run_cache"
RUN_CACHE.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EXPECTED_DROUGHT_N = 1552

print("2024 exists:", PATH_2024.expanduser().exists() if False else PATH_2024.exists())
print("2021 exists:", PATH_2021.exists())
print("Processed dir:", OUT_DIR)
print("Notebook cache:", RUN_CACHE)


def cache_path(name: str) -> Path:
    return RUN_CACHE / name


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
'''


def build_append_cells() -> list[dict]:
    cells: list[dict] = []

    # ----- Shannon decision + DP decisions -----
    cells.append(md("""### Shannon diversity decision

The Shannon values observed in this sample align closely with the number of active agricultural groups and are not a deterministic function of the crop and livestock counts alone.

**Decision**

- Retain `n_crop_types`, `n_livestock_types` and the nine agricultural binary indicators in the default modelling inventories.
- Reserve `shannon_diversity` for sensitivity analysis only; do not place it in the default segmentation or classification matrices.
- Do not treat Shannon diversity and the portfolio counts as interchangeable measures.
"""))

    cells.append(md("""### Preparation decisions carried into feature construction

The following decisions close Data Preparation for the drought-exposed sample.

| Topic | Decision |
|---|---|
| Analytical sample | `drought_exposed == 1` (n = 1,552) |
| Land measure | `land_size_acres`; special codes 98/99 and community land (unit 5) remain missing; never zero-filled |
| `land_size_unit` | Not used as a predictor once acres are derived |
| Demographics | Retain age, sex, education (ordered 1–4), urban/rural, county |
| Financial access | Retained in the analysis frame but excluded from default classification predictors (timing unresolved relative to coping) |
| Shock indicators / main shock | Excluded from primary classification; available only in an extended retrospective matrix |
| Coping actions and targets | Excluded from all predictor matrices |
| Imputation / scaling at preparation | Not applied; any later fitted transforms belong inside training folds |

Light checks on the working sample follow.
"""))

    cells.append(code(r'''# Light verification on the drought-exposed working sample
assert "analysis_2024" in dir() or "analysis_2024" in globals()
print("n =", len(analysis_2024))
assert len(analysis_2024) == EXPECTED_DROUGHT_N

print("\nTarget distribution")
display(analysis_2024["coping_category"].value_counts().rename_axis("coping_category").to_frame("n"))

# Land acres missingness (using preparation rule)
_unit = pd.to_numeric(analysis_2024["land_size_unit"], errors="coerce")
_size = pd.to_numeric(analysis_2024["land_size"], errors="coerce")
_size_masked = _size.mask(_size.isin([98, 99]))
_acres = _size_masked * _unit.map({1: 1.0, 2: 2.47105, 3: 0.000247105, 4: 1.0 / 43560.0})
_missing = _acres.isna()
print("\nland_size_acres missing:", int(_missing.sum()))
print("  community unit 5:", int((_missing & _unit.eq(5)).sum()))
print("  special 98/99 non-community:", int((_missing & _size.isin([98, 99]) & ~_unit.eq(5)).sum()))
'''))

    cells.append(md("""### Feature matrix construction

The cells below construct the modelling matrices from `analysis_2024`:

- Segmentation inventory (agricultural indicators, portfolio counts, acres, ownership)
- Primary classification inventory (demographics + agriculture/land; no shocks; no financial access)
- Extended classification inventory (primary + additional shock indicators + main-shock one-hots)
- Target columns for four-class and collapsed three-class outcomes

Encoding uses fixed codebook categories for county and main shock type. No train/validation split is created here.
"""))

    cells.append(code(r'''# --- Feature preparation helpers ---
SEX_CODE_TO_BIN = {1: 0, 2: 1}
URBAN_RURAL_CODE_TO_BIN = {1: 0, 2: 1}
LAND_SIZE_TO_ACRES = {1: 1.0, 2: 2.47105, 3: 0.000247105, 4: 1.0 / 43560.0}

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
SHOCK_INDICATOR_VARS = [
    "shock_child_birth",
    "shock_major_sickness_health_problem_accident_injury",
    "shock_drought_loss",
    "shock_high_rainfall_flooding_mudslide_lightning",
    "shock_death_of_main_income_earner_family_member",
    "shock_loss_or_damage_due_totheft_e_g_property_assets_business_etc",
    "shock_loss_of_a_job_or_a_source_of_income_or_a_wage_cut",
    "shock_lost_a_home_or_land_e_g_due_to_eviction_demolitions",
    "shock_livestock_crop_failure_due_to_pest_and_diseases",
    "shock_loss_or_damage_due_tofire_or_violence_including_political_violen",
    "shock_loss_or_damage_due_to_human_wildlife_conflict",
    "shock_loss_of_income_or_major_financial_expenses_due_to_taking_on_care",
]
ADDITIONAL_SHOCK_VARS = [c for c in SHOCK_INDICATOR_VARS if c != "shock_drought_loss"]


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    return cleaned.strip("_") or "unknown"


def load_codebook_categories(label_set: str):
    labels = pd.read_csv(VALUE_LABELS_PATH, keep_default_na=False)
    subset = labels.loc[labels["label_set"] == label_set, ["code", "text"]].copy()
    rows = []
    for _, row in subset.iterrows():
        try:
            code = int(float(row["code"]))
        except (TypeError, ValueError):
            continue
        text = str(row["text"]).strip() or f"code_{code}"
        rows.append((code, text))
    rows.sort(key=lambda x: x[0])
    return rows


def one_hot_from_codebook(series: pd.Series, categories, prefix: str) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce")
    cols = {}
    for code, label in categories:
        col = f"{prefix}_{code}_{slugify(label)}"
        cols[col] = (values == code).astype(int)
    return pd.DataFrame(cols, index=series.index)


def prepare_modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    unit = pd.to_numeric(out["land_size_unit"], errors="coerce")
    size = pd.to_numeric(out["land_size"], errors="coerce")
    size = size.mask(size.isin([98, 99]))
    out["land_size_acres"] = size * unit.map(LAND_SIZE_TO_ACRES)

    sex = pd.to_numeric(out["sex"], errors="coerce")
    out["sex_bin"] = sex.map(SEX_CODE_TO_BIN)
    ur = pd.to_numeric(out["urban_rural"], errors="coerce")
    out["urban_rural_bin"] = ur.map(URBAN_RURAL_CODE_TO_BIN)
    out["education"] = pd.to_numeric(out["education"], errors="coerce")
    out["age_years"] = pd.to_numeric(out["age_years"], errors="coerce")

    county_oh = one_hot_from_codebook(out["county"], load_codebook_categories("county"), "county")
    shock_oh = one_hot_from_codebook(
        out["main_shock_type"], load_codebook_categories("R2B"), "main_shock_type"
    )
    out = pd.concat([out, county_oh, shock_oh], axis=1)
    return out


prepared = prepare_modelling_frame(analysis_2024)
assert len(prepared) == EXPECTED_DROUGHT_N

county_cols = [c for c in prepared.columns if re.match(r"^county_\d+_", c)]
shock_type_cols = [c for c in prepared.columns if re.match(r"^main_shock_type_\d+_", c)]

segmentation_features = (
    list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS)
    + ["land_size_acres"] + list(LAND_OWNERSHIP_VARS)
)
primary_features = (
    ["age_years", "sex_bin", "education", "urban_rural_bin"]
    + county_cols
    + list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS)
    + ["land_size_acres"] + list(LAND_OWNERSHIP_VARS)
)
extended_features = primary_features + list(ADDITIONAL_SHOCK_VARS) + shock_type_cols

X_segmentation = prepared[["respondent_id"] + segmentation_features].copy()
X_primary = prepared[["respondent_id"] + primary_features].copy()
X_extended = prepared[["respondent_id"] + extended_features].copy()
y_targets = prepared[[
    "respondent_id", "coping_category", "coping_category_code",
    "coping_category_collapsed", "coping_channel",
]].copy()
# weights for descriptive profiling
if "household_weight" in prepared.columns:
    household_weight = prepared[["respondent_id", "household_weight"]].copy()
else:
    household_weight = prepared[["respondent_id"]].copy()
    household_weight["household_weight"] = 1.0

feature_summary = pd.DataFrame({
    "feature_set": ["segmentation", "primary_classification", "extended_classification"],
    "n_features": [len(segmentation_features), len(primary_features), len(extended_features)],
})
display(feature_summary)
print("Primary feature count expected 69:", len(primary_features))
print("Extended feature count expected 92:", len(extended_features))
print("Segmentation feature count expected 18:", len(segmentation_features))
assert len(segmentation_features) == 18
assert len(primary_features) == 69
assert len(extended_features) == 92
print("\nFour-class targets")
display(y_targets["coping_category"].value_counts())
'''))

    cells.append(md("""## Data Preparation phase conclusion

Feature matrices for segmentation and classification are now available in memory (`X_segmentation`, `X_primary`, `X_extended`, `y_targets`).

Carry-forward points for Modelling:

1. Missing `land_size_acres` values require fold-wise handling or exclusion.
2. The formal coping class is small (n = 75) and needs stratified validation with class-wise metrics.
3. Extended shock predictors share the survey reference window with coping and are retrospective only.

The extended classification matrix is constructed for completeness; the completed modelling experiments reported below focus on the **primary four-class** track. The three-class target and extended track remain deferred.
"""))

    # ----- Modelling segmentation -----
    cells.append(md("""# Modelling

Modelling proceeds in two separate workflows.

1. **Segmentation** identifies agricultural and land-based household groups without using coping outcomes.
2. **Classification** predicts the four-class coping category from household and agricultural characteristics under a fixed validation protocol.

Segmentation results are later used only for descriptive coping profiles, not as classification labels during model training.
"""))

    cells.append(md("""## Modelling — Segmentation

### Design

| Element | Choice |
|---|---|
| Distance | Gower |
| Algorithm | PAM (Partitioning Around Medoids) |
| Spec A (primary) | Exclude `land_size_acres` (17 features); equal weights |
| Spec B | Acreage sensitivity with community acres left missing; never zero-filled |
| Candidate k | 2 to 6 |
| Random seed | 42 |
| Coping outcomes | Excluded from distance and clustering |

Spec A is the primary specification. Spec B is a sensitivity analysis.
"""))

    cells.append(code(r'''# --- Segmentation helpers ---

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


SPEC_A_FEATURES = (
    list(AGRICULTURAL_HAS_VARS) + list(PORTFOLIO_COUNT_VARS) + list(LAND_OWNERSHIP_VARS)
)
assert "land_size_acres" not in SPEC_A_FEATURES
assert len(SPEC_A_FEATURES) == 17
print("Spec A features (", len(SPEC_A_FEATURES), "):")
print(SPEC_A_FEATURES)
'''))

    cells.append(code(r'''# Spec A: Gower distance + PAM for k = 2..6
spec_a_cache = cache_path("spec_a_s1.npz")
spec_a_meta = cache_path("spec_a_s1_meta.json")

X_a = X_segmentation[["respondent_id"] + SPEC_A_FEATURES].copy()
for c in SPEC_A_FEATURES:
    X_a[c] = pd.to_numeric(X_a[c], errors="coerce")
assert int(X_a[SPEC_A_FEATURES].isna().sum().sum()) == 0

cat_mask_a = np.array([c not in PORTFOLIO_COUNT_VARS for c in SPEC_A_FEATURES], dtype=bool)

if spec_a_cache.exists() and spec_a_meta.exists():
    meta = load_json(spec_a_meta)
    blob = np.load(spec_a_cache, allow_pickle=True)
    dist_a = blob["distance"]
    assign_wide = pd.DataFrame(blob["assign_wide"].item())
    sil_table = pd.DataFrame(blob["sil_table"].item())
    print("Loaded Spec A results from notebook_run_cache")
else:
    t0 = time.perf_counter()
    dist_a = gower.gower_matrix(X_a[SPEC_A_FEATURES].to_numpy(dtype=float), cat_features=cat_mask_a)
    print(f"Gower distance computed in {time.perf_counter()-t0:.1f}s; shape={dist_a.shape}")
    sil_rows = []
    assign = {"respondent_id": X_a["respondent_id"].to_numpy()}
    for k in range(2, 7):
        labels, medoids = pam_precomputed(dist_a, k, random_state=RANDOM_SEED, max_iter=100)
        sil = average_silhouette_precomputed(dist_a, labels)
        sil_rows.append({"specification": "A", "k": k, "average_silhouette": sil, "n_features": 17, "random_seed": RANDOM_SEED})
        assign[f"cluster_k{k}"] = labels
        print(f"k={k}: silhouette={sil:.4f}")
    sil_table = pd.DataFrame(sil_rows)
    assign_wide = pd.DataFrame(assign)
    np.savez_compressed(
        spec_a_cache,
        distance=dist_a,
        assign_wide=assign_wide.to_dict(orient="list"),
        sil_table=sil_table.to_dict(orient="list"),
    )
    save_json(spec_a_meta, {"random_seed": RANDOM_SEED, "features": SPEC_A_FEATURES, "n": len(X_a)})
    print("Saved Spec A cache")

display(sil_table)
sizes_k5 = assign_wide["cluster_k5"].value_counts().sort_index().rename_axis("cluster_k5").to_frame("n")
sizes_k5["pct"] = (100 * sizes_k5["n"] / len(assign_wide)).round(2)
print("\nSpec A k=5 sizes")
display(sizes_k5)
'''))

    cells.append(md("""### Stability and land-size sensitivity (condensed)

Resampling stability was assessed for Spec A at k ∈ {2, 4, 5} using shared 80% subsamples (50 replicates, seed 42). Spec B repeats PAM after adding measured acres with community land left missing and a community-land indicator, using missing-aware Gower distances.

The completed study found strong Spec A/B agreement at k = 5 (Adjusted Rand Index ≈ 0.92). The notebook recomputes Spec B correspondence for k = 4 and k = 5 against the Spec A assignments above. Full 50-replicate stability summaries are recomputed when absent from `notebook_run_cache/` (this step is computationally heavy).
"""))

    cells.append(code(r'''# Spec B correspondence for k=4 and k=5 (sensitivity)
spec_b_cache = cache_path("spec_b_correspondence.json")

# Build Spec B matrix: acres with community missing; impute 16 special non-community; indicators
Xb = X_segmentation.merge(
    prepared[["respondent_id", "land_size_unit", "land_size"]],
    on="respondent_id",
    how="left",
    suffixes=("", "_raw"),
)
unit = pd.to_numeric(prepared.set_index("respondent_id").loc[Xb["respondent_id"], "land_size_unit"].to_numpy(), errors="coerce")
size = pd.to_numeric(prepared.set_index("respondent_id").loc[Xb["respondent_id"], "land_size"].to_numpy(), errors="coerce")
acres = Xb["land_size_acres"].to_numpy(dtype=float)
is_community = (unit == 5).astype(float)
special_non_comm = np.isnan(acres) & np.isin(size, [98, 99]) & (unit != 5)
obs_acres = acres[~np.isnan(acres)]
med = float(np.median(obs_acres))
acres_b = acres.copy()
acres_b[special_non_comm] = med
acres_special_imputed = special_non_comm.astype(float)
# community remains NaN
acres_b[is_community == 1] = np.nan

SPEC_B_FEATURES = SPEC_A_FEATURES + ["land_size_acres_b", "is_community_land", "acres_special_imputed"]
Xb_mat = Xb[SPEC_A_FEATURES].copy()
Xb_mat["land_size_acres_b"] = acres_b
Xb_mat["is_community_land"] = is_community
Xb_mat["acres_special_imputed"] = acres_special_imputed
cat_mask_b = np.array(
    [c not in PORTFOLIO_COUNT_VARS + ["land_size_acres_b"] for c in SPEC_B_FEATURES],
    dtype=bool,
)

if spec_b_cache.exists():
    corr = load_json(spec_b_cache)
    print("Loaded Spec B correspondence from cache")
else:
    t0 = time.perf_counter()
    dist_b = gower_matrix_missing_aware(Xb_mat[SPEC_B_FEATURES].to_numpy(dtype=float), cat_mask_b)
    print(f"Spec B distance in {time.perf_counter()-t0:.1f}s")
    corr = {"ari": {}, "sizes_b": {}}
    for k in (4, 5):
        lab_b, _ = pam_precomputed(dist_b, k, random_state=RANDOM_SEED, max_iter=100)
        lab_a = assign_wide[f"cluster_k{k}"].to_numpy()
        corr["ari"][str(k)] = float(adjusted_rand_score(lab_a, lab_b))
        corr["sizes_b"][str(k)] = pd.Series(lab_b).value_counts().sort_index().astype(int).to_dict()
        assign_wide[f"spec_b_cluster_k{k}"] = lab_b
    save_json(spec_b_cache, corr)
    # refresh cache assign with spec b labels
    if spec_a_cache.exists():
        blob = dict(np.load(spec_a_cache, allow_pickle=True))
        # keep distance; update assign
        np.savez_compressed(
            spec_a_cache,
            distance=dist_a if "dist_a" in dir() else np.load(spec_a_cache)["distance"],
            assign_wide=assign_wide.to_dict(orient="list"),
            sil_table=sil_table.to_dict(orient="list"),
        )

print("Spec A vs Spec B ARI:", corr["ari"])
'''))

    cells.append(code(r'''# Spec A resampling stability (k=2,4,5): 50 replicates of 80% subsamples
stab_cache = cache_path("spec_a_stability_summary.csv")
N_REPLICATES = 50
SUBSAMPLE_FRACTION = 0.80

if stab_cache.exists():
    stab_summary = pd.read_csv(stab_cache)
    print("Loaded stability summary from cache")
else:
    print("Computing stability replicates (this can take several minutes)...")
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(X_a)
    subsample_idx = [
        np.sort(rng.choice(n, size=int(round(SUBSAMPLE_FRACTION * n)), replace=False))
        for _ in range(N_REPLICATES)
    ]
    rows = []
    X_np = X_a[SPEC_A_FEATURES].to_numpy(dtype=float)
    for k in (2, 4, 5):
        aris = []
        full_labels = assign_wide[f"cluster_k{k}"].to_numpy()
        for rep, idx in enumerate(subsample_idx):
            d_sub = gower.gower_matrix(X_np[idx], cat_features=cat_mask_a)
            lab_sub, _ = pam_precomputed(d_sub, k, random_state=RANDOM_SEED, max_iter=100)
            aris.append(adjusted_rand_score(full_labels[idx], lab_sub))
            if (rep + 1) % 10 == 0:
                print(f"  k={k} replicate {rep+1}/{N_REPLICATES}")
        rows.append({
            "k": k,
            "n_replicates": N_REPLICATES,
            "mean_ari_vs_full": float(np.mean(aris)),
            "std_ari_vs_full": float(np.std(aris)),
            "min_ari_vs_full": float(np.min(aris)),
            "max_ari_vs_full": float(np.max(aris)),
        })
    stab_summary = pd.DataFrame(rows)
    stab_summary.to_csv(stab_cache, index=False)
    print("Saved stability summary")

display(stab_summary)
'''))

    cells.append(md("""### Working typology decision

Following silhouette, stability and Spec B correspondence evidence, **Spec A with k = 5** is adopted as the working segmentation typology. Spec B remains a land-size sensitivity. The k = 4 solution remains an alternative typology.

| cluster_k5 | Working label |
|---|---|
| 0 | Family-owned legume-oriented crop households |
| 1 | Own-name crop households |
| 2 | Community-land livestock-oriented households |
| 3 | Family-owned mixed crop and livestock households |
| 4 | Family-owned cereal-oriented households |

These labels summarise dominant characteristics. The cereals-oriented group is a softer distinction in resampling. Coping outcomes were not used to form clusters or choose k.
"""))

    cells.append(code(r'''CLUSTER_LABELS_K5 = {
    0: "Family-owned legume-oriented crop households",
    1: "Own-name crop households",
    2: "Community-land livestock-oriented households",
    3: "Family-owned mixed crop and livestock households",
    4: "Family-owned cereal-oriented households",
}
typology = (
    assign_wide["cluster_k5"].value_counts().sort_index().rename_axis("cluster_k5").to_frame("n_households")
)
typology["working_label"] = typology.index.map(CLUSTER_LABELS_K5)
typology["pct"] = (100 * typology["n_households"] / len(assign_wide)).round(2)
display(typology)
'''))

    cells.append(md("""### Coping profiles across segments

Coping category distributions are compared across Spec A clusters using unweighted percentages and household-weighted percentages. Household survey weights adjust descriptive composition toward population representation. They are **not** used later as classification fitting weights.

Formal coping remains sparse within clusters. The k = 4 solution is checked as a sensitivity for the community-livestock segment.
"""))

    cells.append(code(r'''# Merge assignments, targets and household weights
profile = (
    assign_wide[["respondent_id", "cluster_k4", "cluster_k5"]]
    .merge(y_targets[["respondent_id", "coping_category", "coping_category_collapsed"]], on="respondent_id")
    .merge(household_weight, on="respondent_id")
)
FOUR = ["personal", "informal", "formal", "did_nothing"]

def within_cluster_pct(df, cluster_col, weighted=False):
    rows = []
    for c in sorted(df[cluster_col].unique()):
        sub = df.loc[df[cluster_col] == c]
        if weighted:
            w = sub["household_weight"].to_numpy(dtype=float)
            wsum = w.sum()
            for cat in FOUR:
                rows.append({
                    cluster_col: c,
                    "coping_category": cat,
                    "pct": 100.0 * float(w[sub["coping_category"].to_numpy() == cat].sum()) / max(wsum, 1e-12),
                })
        else:
            n = len(sub)
            for cat in FOUR:
                rows.append({
                    cluster_col: c,
                    "coping_category": cat,
                    "pct": 100.0 * int((sub["coping_category"] == cat).sum()) / n,
                })
    return (
        pd.DataFrame(rows)
        .pivot(index=cluster_col, columns="coping_category", values="pct")
        .reindex(columns=FOUR)
    )

print("Unweighted % within Spec A k=5")
unw5 = within_cluster_pct(profile, "cluster_k5", weighted=False)
unw5.index = [f"{i} — {CLUSTER_LABELS_K5[i]}" for i in unw5.index]
display(unw5.round(1))

print("Household-weighted % within Spec A k=5")
wt5 = within_cluster_pct(profile, "cluster_k5", weighted=True)
wt5.index = [f"{i} — {CLUSTER_LABELS_K5[i]}" for i in wt5.index]
display(wt5.round(1))

formal_n = (
    profile.loc[profile["coping_category"] == "formal"]
    .groupby("cluster_k5").size()
    .reindex(range(5), fill_value=0)
    .rename("n_formal")
    .to_frame()
)
formal_n["sparse_under_30"] = formal_n["n_formal"] < 30
print("Formal counts by cluster")
display(formal_n)

# k=4 vs k=5 focal livestock/community overlap
overlap = int(((profile["cluster_k5"] == 2) & (profile["cluster_k4"] == 2)).sum())
n_k5_2 = int((profile["cluster_k5"] == 2).sum())
n_k4_2 = int((profile["cluster_k4"] == 2).sum())
print(f"\nFocal overlap cluster2: n={overlap} ({100*overlap/n_k5_2:.1f}% of k=5 c2; {100*overlap/n_k4_2:.1f}% of k=4 c2)")

print("\nUnweighted % within Spec A k=4 (sensitivity)")
unw4 = within_cluster_pct(profile, "cluster_k4", weighted=False)
display(unw4.round(1))
'''))

    # ----- Classification -----
    cells.append(md("""## Modelling — Classification

### Protocol

| Track | Target | Predictors | Status in this notebook |
|---|---|---|---|
| Primary | Four-class `coping_category` | Primary matrix | **Completed** |
| Secondary | Three-class collapsed | Same primary matrix / folds | Deferred |
| Extended | Four-class | Extended matrix (retrospective shocks) | Deferred |

| Element | Choice |
|---|---|
| Validation | Repeated stratified 5-fold CV × 5 repeats |
| Seed | 42 |
| Primary metric | Macro F1 |
| Additional metrics | Balanced accuracy; class-wise precision/recall/F1; weighted F1; confusion matrices |
| Class imbalance | Compare unweighted vs class-weighted learning; no SMOTE |
| Survey weights | `household_weight` is **not** used in fitting or fold metrics |
| Land size | Compare acres excluded vs train-fold median imputation; then adopt acres-excluded as default |

Fold-level standard deviations describe variation across the 25 test splits. They are not confidence intervals and are not used for significance testing.
"""))

    cells.append(code(r'''# Shared CV folds (built once; reused for all classifiers)
folds_cache = cache_path("cv_folds_long.csv")
y4 = y_targets.set_index("respondent_id").loc[X_primary["respondent_id"], "coping_category"].to_numpy()
ids = X_primary["respondent_id"].to_numpy()

if folds_cache.exists():
    folds_long = pd.read_csv(folds_cache)
    print("Loaded CV folds from cache")
else:
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
    folds_long = pd.DataFrame(rows)
    folds_long.to_csv(folds_cache, index=False)
    print("Saved CV folds")

formal_test = (
    folds_long.loc[(folds_long["split"] == "test") & (folds_long["coping_category"] == "formal")]
    .groupby(["repeat", "fold"]).size()
    .rename("n_formal_test")
    .reset_index()
)
print("Formal test counts across folds:")
display(formal_test["n_formal_test"].agg(["min", "max", "mean", "count"]).to_frame("value"))
assert formal_test["n_formal_test"].min() == 15
assert formal_test["n_formal_test"].max() == 15
'''))

    cells.append(md("""### C1 baselines — dummy and multinomial logistic regression

Experiments:

1. Most-frequent dummy classifier  
2. Multinomial logistic regression, unweighted vs class-weighted  
3. Land size excluded vs train-fold median imputation  

Continuous predictors (`age_years`, `education`, portfolio counts, and acres when retained) are standardised within each training fold. Binary and one-hot columns are passed through. Regularised multinomial logistic regression uses `C = 1`, solver `lbfgs`.
"""))

    cells.append(code(r'''# Classification evaluation helpers
FOUR_ORDER = ["personal", "informal", "formal", "did_nothing"]
CONTINUOUS_BASE = ["age_years", "education", "n_crop_types", "n_livestock_types"]
id_to_pos = {rid: i for i, rid in enumerate(X_primary["respondent_id"].tolist())}
y_series = pd.Series(y4, index=np.arange(len(y4)))


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
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        class_weight="balanced" if class_weight_mode == "class_weighted" else None,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", pre), ("clf", clf)])


def run_cv_experiment(experiment_id, model_builder, feature_cols, X_df):
    X_mat = X_df[feature_cols]
    fold_rows = []
    cm_rows = []
    for split_id in sorted(folds_long["split_id"].unique()):
        part = folds_long.loc[folds_long["split_id"] == split_id]
        tr_ids = part.loc[part["split"] == "train", "respondent_id"]
        te_ids = part.loc[part["split"] == "test", "respondent_id"]
        tr = tr_ids.map(id_to_pos).to_numpy()
        te = te_ids.map(id_to_pos).to_numpy()
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


c1_cache = cache_path("c1_scorecard.csv")
c1_fold_cache = cache_path("c1_fold_metrics.csv")

if c1_cache.exists() and c1_fold_cache.exists():
    c1_scorecard = pd.read_csv(c1_cache)
    c1_folds = pd.read_csv(c1_fold_cache)
    print("Loaded C1 results from cache")
else:
    experiments = []
    # dummy
    feats_ex = [c for c in primary_features if c != "land_size_acres"]
    experiments.append(("dummy_most_frequent", lambda: DummyClassifier(strategy="most_frequent"), feats_ex, X_primary))
    for land in ("exclude", "impute_median"):
        feats = [c for c in primary_features if c != "land_size_acres"] if land == "exclude" else list(primary_features)
        for cw in ("unweighted", "class_weighted"):
            eid = f"logistic_{cw}_{'exclude' if land=='exclude' else 'impute_median'}"
            experiments.append((
                eid,
                lambda feats=feats, cw=cw, land=land: build_logistic(feats, cw, land),
                feats,
                X_primary,
            ))

    all_folds = []
    all_cms = []
    for eid, builder, feats, Xdf in experiments:
        print("Running", eid, "...")
        fm, cm = run_cv_experiment(eid, builder, feats, Xdf)
        fm["experiment_id"] = eid
        all_folds.append(fm)
        all_cms.append(cm)
    c1_folds = pd.concat(all_folds, ignore_index=True)
    c1_scorecard = summarise_folds(c1_folds)
    c1_folds.to_csv(c1_fold_cache, index=False)
    c1_scorecard.to_csv(c1_cache, index=False)
    pd.concat(all_cms, ignore_index=True).to_csv(cache_path("c1_fold_confusion.csv"), index=False)
    print("Saved C1 cache")

cols_show = [
    "experiment_id", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean",
    "weighted_f1_mean", "f1_formal_mean", "f1_formal_std", "accuracy_mean",
]
display(c1_scorecard[cols_show])
'''))

    cells.append(md("""### Land-size decision after C1

Acres-excluded and train-fold median-imputed logistic models produce nearly identical macro F1. **Acres excluded** is adopted as the default classification specification. Train-fold imputation remains a documented sensitivity. A missingness indicator is not introduced.

Class-weighted logistic regression improves formal-class F1 and balanced accuracy relative to the unweighted logistic model, at some cost to overall accuracy. Formal metrics remain exploratory because each test fold contains only 15 formal observations.
"""))

    cells.append(md("""### C2 / C3 — decision tree, random forest and gradient boosting

Using the same folds and the acres-excluded primary matrix, the study compares:

- Decision tree  
- Random forest (200 trees)  
- Histogram gradient boosting (200 iterations)  

each under unweighted and class-weighted learning. No extensive hyperparameter search is undertaken. Models are judged against the class-weighted logistic benchmark using macro F1 together with balanced accuracy, class-wise metrics and confusion structure — not macro F1 alone.
"""))

    cells.append(code(r'''c23_cache = cache_path("c23_scorecard.csv")
c23_fold_cache = cache_path("c23_fold_metrics.csv")
feats_ex = [c for c in primary_features if c != "land_size_acres"]

def build_tree_models(name, cw_mode):
    cw = "balanced" if cw_mode == "class_weighted" else None
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, class_weight=cw, random_state=RANDOM_SEED, early_stopping=False)
    raise ValueError(name)

if c23_cache.exists() and c23_fold_cache.exists():
    c23_scorecard = pd.read_csv(c23_cache)
    c23_folds = pd.read_csv(c23_fold_cache)
    print("Loaded C2/C3 results from cache")
else:
    all_folds = []
    all_cms = []
    for model_name in ("decision_tree", "random_forest", "hist_gradient_boosting"):
        for cw in ("unweighted", "class_weighted"):
            eid = f"{model_name}_{cw}_exclude"
            print("Running", eid, "...")
            t0 = time.perf_counter()
            fm, cm = run_cv_experiment(
                eid,
                lambda model_name=model_name, cw=cw: build_tree_models(model_name, cw),
                feats_ex,
                X_primary,
            )
            print(f"  done in {time.perf_counter()-t0:.1f}s")
            all_folds.append(fm)
            all_cms.append(cm)
    c23_folds = pd.concat(all_folds, ignore_index=True)
    c23_scorecard = summarise_folds(c23_folds)
    c23_folds.to_csv(c23_fold_cache, index=False)
    c23_scorecard.to_csv(c23_cache, index=False)
    pd.concat(all_cms, ignore_index=True).to_csv(cache_path("c23_fold_confusion.csv"), index=False)
    print("Saved C2/C3 cache")

# Comparison with C1 acres-exclude benchmarks
c1_keep = c1_scorecard.loc[
    c1_scorecard["experiment_id"].isin([
        "dummy_most_frequent",
        "logistic_unweighted_exclude",
        "logistic_class_weighted_exclude",
    ]),
    cols_show,
].copy()
c1_keep.insert(1, "stage", "C1")
c23_view = c23_scorecard[cols_show].copy()
c23_view.insert(1, "stage", "C2/C3")
comparison = pd.concat([c1_keep, c23_view], ignore_index=True)
display(comparison.sort_values(["stage", "macro_f1_mean"], ascending=[True, False]))
comparison.to_csv(cache_path("c1_c23_comparison.csv"), index=False)
'''))

    cells.append(md("""# Evaluation

### Segmentation

The Spec A five-cluster typology provides an interpretable agricultural and land-based working typology for drought-exposed households. Descriptive coping mixes differ across segments, most clearly for the community-land livestock-oriented group (higher personal coping, lower informal coping, higher did-nothing shares). The k = 4 sensitivity shows that this contrast is not an artefact of choosing five clusters alone.

Segmentation did not use coping labels. Cluster membership therefore describes agricultural structure; it does not by itself demonstrate that segments cause or predict coping choices.

### Classification

Under the locked primary four-class protocol with acres excluded:

- Multinomial logistic regression clearly outperforms a most-frequent dummy baseline on macro F1.
- Class-weighted logistic regression is the strongest completed model on mean macro F1 and yields the highest formal-class F1 among the fitted models, while remaining limited by formal-class sparsity.
- Decision tree, random forest and histogram gradient boosting did not improve on the class-weighted logistic benchmark under the default settings used here.
- Overall discrimination is modest. The study should not claim strong predictive performance for operational targeting.

The three-class secondary target and the extended retrospective shock matrix were defined in preparation but were not fitted in the completed experiment sequence.

### What the study does not claim

- Causal effects of farm type on coping  
- Prospective prediction before shock exposure  
- Deployment-ready classification for programme targeting  
"""))

    cells.append(md("""# Limitations and next steps

Operational deployment is outside the scope of this notebook. Formal-class metrics should be treated as exploratory. Domain-weighted segmentation sensitivities and the deferred three-class / extended classification tracks remain available as future work if required.

To force recomputation of cached modelling intermediates, delete the `notebook_run_cache/` directory and re-run the Modelling cells.
"""))

    return cells


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Stage 0: overview, setup, remove bridge
    set_src(cells[1], OVERVIEW)
    set_src(cells[5], SETUP_CODE)

    # Remove workbench bridge cells if present
    new_cells = []
    skip_next_code = False
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "Workbench artefact bridge" in src:
            # skip this md and the following code cell
            skip_next_code = True
            continue
        if skip_next_code and c["cell_type"] == "code" and "WORKBENCH" in src:
            skip_next_code = False
            continue
        skip_next_code = False
        new_cells.append(c)
    cells = new_cells

    # Avoid duplicate append
    if not any("Data Preparation phase conclusion" in "".join(c.get("source", [])) for c in cells):
        cells.extend(build_append_cells())
    else:
        print("Append sections already present; skipping append")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NB_PATH} with {len(cells)} cells")


if __name__ == "__main__":
    main()
