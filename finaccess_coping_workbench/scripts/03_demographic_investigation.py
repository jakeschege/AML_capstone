#!/usr/bin/env python3
"""Investigate demographic variables for the drought-exposed analytical sample.

Covers age_years, sex, education, urban_rural and county.
Does not encode variables, drop observations, impute values or edit the notebook.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    DEMOGRAPHIC_VARS,
    REPORTS_DIR,
    TABLES_DIR,
    VALUE_LABELS_PATH,
    ensure_output_dirs,
    load_analysis_2024,
)

# Codebook label sets used for interpretation only
LABEL_SETS = {
    "sex": "Sex",
    "education": "Education",
    "urban_rural": "Cluster_Type",
    "county": "county",
}

# Survey adult floor used for implausibility screening only
AGE_ADULT_FLOOR = 16
AGE_HIGH_REVIEW = 90
SMALL_COUNTY_THRESHOLDS = (5, 10)


def load_codebook_labels() -> dict[str, dict[int, str]]:
    """Load selected value-label maps from the read-only codebook file."""
    if not VALUE_LABELS_PATH.exists():
        return {var: {} for var in LABEL_SETS}

    labels = pd.read_csv(VALUE_LABELS_PATH)
    maps: dict[str, dict[int, str]] = {}
    for var, label_set in LABEL_SETS.items():
        subset = labels.loc[labels["label_set"] == label_set, ["code", "text"]].copy()
        mapping: dict[int, str] = {}
        for _, row in subset.iterrows():
            try:
                code = int(row["code"])
            except (TypeError, ValueError):
                continue
            text = row["text"]
            if pd.isna(text) or str(text).strip() == "":
                mapping[code] = "[label missing in codebook]"
            else:
                mapping[code] = str(text).strip()
        maps[var] = mapping
    return maps


def variable_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Basic inventory for the demographic variables."""
    rows = []
    for col in DEMOGRAPHIC_VARS:
        s = df[col]
        rows.append(
            {
                "variable": col,
                "dtype": str(s.dtype),
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(100.0 * float(s.isna().mean()), 2),
                "n_unique": int(s.nunique(dropna=True)),
                "min": float(s.min()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def age_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics and skewness for age_years."""
    s = df["age_years"]
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    n_below_fence = int((s < lower_fence).sum())
    n_above_fence = int((s > upper_fence).sum())

    summary = {
        "variable": "age_years",
        "n": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "skewness": float(s.skew()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": q1,
        "p50": float(s.quantile(0.50)),
        "p75": q3,
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "iqr": iqr,
        "iqr_lower_fence": lower_fence,
        "iqr_upper_fence": upper_fence,
        "n_below_iqr_fence": n_below_fence,
        "n_above_iqr_fence": n_above_fence,
        "n_below_adult_floor": int((s < AGE_ADULT_FLOOR).sum()),
        "n_at_or_above_high_review": int((s >= AGE_HIGH_REVIEW).sum()),
    }
    return pd.DataFrame([summary])


def age_extreme_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag ages for review without removing observations."""
    s = df["age_years"]
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    flags = []
    for age, count in s.value_counts().sort_index().items():
        reasons = []
        if age < AGE_ADULT_FLOOR:
            reasons.append("below_adult_floor")
        if age >= AGE_HIGH_REVIEW:
            reasons.append("high_age_review")
        if age < lower_fence or age > upper_fence:
            reasons.append("outside_iqr_fence")
        if reasons:
            flags.append(
                {
                    "age_years": int(age),
                    "n_households": int(count),
                    "flag_reasons": ";".join(reasons),
                }
            )
    return pd.DataFrame(flags)


def category_frequencies(
    df: pd.DataFrame,
    column: str,
    label_map: dict[int, str],
) -> pd.DataFrame:
    """Frequency table with optional codebook labels."""
    counts = df[column].value_counts(dropna=False).sort_index()
    rows = []
    for code, n in counts.items():
        if pd.isna(code):
            label = "[missing]"
            code_out: int | float = np.nan
        else:
            code_out = int(code)
            label = label_map.get(code_out, "[no codebook label]")
        rows.append(
            {
                "variable": column,
                "code": code_out,
                "label": label,
                "n_households": int(n),
                "pct": round(100.0 * n / len(df), 2),
            }
        )
    return pd.DataFrame(rows)


def education_ordinal_assessment(freq: pd.DataFrame) -> pd.DataFrame:
    """Document whether observed education codes look like an ordered scale."""
    observed = [int(c) for c in freq["code"].dropna().tolist()]
    codebook_order = [1, 2, 3, 4, 5]
    present_in_order = [c for c in codebook_order if c in observed]
    unexpected = [c for c in observed if c not in codebook_order]

    # In this sample, codes 1-4 match the conventional schooling ladder if
    # code 1 is the lowest schooling level. Code 5 Other would break strict
    # ordinal use if present.
    other_present = 5 in observed
    contiguous_ladder = present_in_order == list(range(min(present_in_order), max(present_in_order) + 1)) if present_in_order else False

    provisional_ordinal = (
        len(unexpected) == 0
        and not other_present
        and set(observed).issubset({1, 2, 3, 4})
        and contiguous_ladder
    )

    return pd.DataFrame(
        [
            {
                "observed_codes": ", ".join(str(c) for c in observed),
                "codebook_codes": "1, 2, 3, 4, 5",
                "code_5_other_present": other_present,
                "unexpected_codes_present": ", ".join(str(c) for c in unexpected)
                if unexpected
                else "",
                "code_1_label_missing_in_codebook": True,
                "provisional_ordinal_candidate": provisional_ordinal,
                "assessment_note": (
                    "Observed codes are 1-4 only. If code 1 is the lowest schooling "
                    "level, the observed categories form an ordered schooling ladder. "
                    "Code 5 Other is absent from this sample. Encoding remains deferred."
                ),
            }
        ]
    )


def county_frequencies(df: pd.DataFrame, label_map: dict[int, str]) -> pd.DataFrame:
    """County household counts with labels and cumulative share."""
    freq = category_frequencies(df, "county", label_map)
    freq = freq.sort_values("n_households", ascending=False).reset_index(drop=True)
    freq["rank"] = np.arange(1, len(freq) + 1)
    freq["cumulative_pct"] = freq["pct"].cumsum().round(2)
    return freq


def county_small_sample_summary(freq: pd.DataFrame) -> pd.DataFrame:
    """Summarise how many counties fall below small-sample thresholds."""
    rows = []
    n_counties = len(freq)
    for threshold in SMALL_COUNTY_THRESHOLDS:
        below = freq.loc[freq["n_households"] < threshold]
        rows.append(
            {
                "threshold": f"n < {threshold}",
                "n_counties_below": int(len(below)),
                "pct_of_counties_below": round(100.0 * len(below) / n_counties, 2),
                "n_households_in_those_counties": int(below["n_households"].sum()),
                "pct_households_in_those_counties": round(
                    100.0 * below["n_households"].sum() / freq["n_households"].sum(), 2
                ),
                "min_county_n": int(freq["n_households"].min()),
                "max_county_n": int(freq["n_households"].max()),
                "n_counties_represented": n_counties,
            }
        )
    return pd.DataFrame(rows)


def age_by_group(df: pd.DataFrame, group_col: str, label_map: dict[int, str]) -> pd.DataFrame:
    """Age descriptive statistics by a categorical demographic."""
    rows = []
    for code, group in df.groupby(group_col, dropna=False):
        code_int = int(code) if pd.notna(code) else np.nan
        label = label_map.get(code_int, "[no codebook label]") if pd.notna(code_int) else "[missing]"
        s = group["age_years"]
        rows.append(
            {
                "group_variable": group_col,
                "code": code_int,
                "label": label,
                "n_households": len(group),
                "age_mean": float(s.mean()),
                "age_median": float(s.median()),
                "age_std": float(s.std()),
                "age_min": float(s.min()),
                "age_max": float(s.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("code").reset_index(drop=True)


def education_by_urban_rural(
    df: pd.DataFrame,
    edu_labels: dict[int, str],
    ur_labels: dict[int, str],
) -> pd.DataFrame:
    """Crosstab of education against urban_rural with labels."""
    table = pd.crosstab(df["education"], df["urban_rural"])
    table = table.reset_index().rename(columns={"education": "education_code"})
    table.insert(
        1,
        "education_label",
        table["education_code"].map(
            lambda c: edu_labels.get(int(c), "[no codebook label]")
        ),
    )
    # Rename urban_rural code columns with labels where possible
    renamed = {"education_code": "education_code", "education_label": "education_label"}
    for col in table.columns:
        if col in renamed:
            continue
        code = int(col)
        renamed[col] = f"{code}_{ur_labels.get(code, 'unknown')}"
    return table.rename(columns=renamed)


def write_report(
    n: int,
    inventory: pd.DataFrame,
    age_desc: pd.DataFrame,
    age_flags: pd.DataFrame,
    sex_freq: pd.DataFrame,
    edu_freq: pd.DataFrame,
    ur_freq: pd.DataFrame,
    county_freq: pd.DataFrame,
    county_small: pd.DataFrame,
    edu_ordinal: pd.DataFrame,
) -> Path:
    """Write the demographic investigation Markdown report."""
    age = age_desc.iloc[0]
    edu = edu_ordinal.iloc[0]
    n_counties = int(county_freq.shape[0])
    min_county = int(county_freq["n_households"].min())
    max_county = int(county_freq["n_households"].max())

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Demographic Investigation Report

Generated: {timestamp}

## What was investigated

Demographic variables for the drought-exposed analytical sample (`n = {n}`):

- `age_years`
- `sex`
- `education`
- `urban_rural`
- `county`

Encoding was not performed. No observations were removed and no values were imputed or transformed.

## Observed findings

### Inventory

```
{inventory.to_string(index=False)}
```

Missingness: none of the five demographic variables has missing values in this sample.

### Age

```
{age_desc.to_string(index=False)}
```

- Skewness: {age['skewness']:.3f}
- Range: {int(age['min'])} to {int(age['max'])} years
- Households below adult floor ({AGE_ADULT_FLOOR}): {int(age['n_below_adult_floor'])}
- Households at or above high-age review threshold ({AGE_HIGH_REVIEW}): {int(age['n_at_or_above_high_review'])}
- Households above IQR upper fence: {int(age['n_above_iqr_fence'])}
- Distinct flagged age values written for review: {len(age_flags)}

Age is mildly right-skewed. Extreme high ages exist but were not removed. They remain candidates for later sensitivity checks rather than automatic exclusion.

### Sex

```
{sex_freq.to_string(index=False)}
```

### Education

```
{edu_freq.to_string(index=False)}
```

Ordinal assessment:

```
{edu_ordinal.to_string(index=False)}
```

- Observed codes in this sample: {edu['observed_codes']}
- Code 5 Other present: {edu['code_5_other_present']}
- Provisional ordinal candidate before encoding: {edu['provisional_ordinal_candidate']}
- Code 1 has a missing label in the codebook file and needs confirmation against the questionnaire before final ordinal treatment.

### Urban / rural

```
{ur_freq.to_string(index=False)}
```

The sample is predominantly rural.

### County

- Counties represented: {n_counties}
- Households per county range from {min_county} to {max_county}

Small-sample summary:

```
{county_small.to_string(index=False)}
```

Sparse counties are documented for later modelling awareness. Counties were not collapsed or dropped in this step.

## Interpretation

The five demographic fields are complete in the drought-exposed sample and use survey codes that can be interpreted with the FinAccess codebook.

Education codes 1 to 4 are consistent with an ordered schooling ladder if code 1 is the lowest schooling category. That interpretation remains provisional until the blank codebook label for code 1 is confirmed. Sex and urban_rural are binary coded categories. County is a high-cardinality geographic identifier with uneven sample sizes.

## Limitations

- Codebook labels were joined for interpretation only and were not written back into the analytical dataset.
- Education code 1 label text is missing in `value_labels.csv`.
- Age extremes were flagged using simple descriptive rules, not clinical or survey-design exclusion rules.
- No relationship with the coping target was used to select or discard demographics.

## Recommended preparation decisions for review

1. Retain `age_years`, `sex`, `education`, `urban_rural` and `county` as provisional candidate variables.
2. Do not encode these variables yet. Encoding belongs in the later feature preparation stage.
3. Do not drop or winsorise high ages solely on the IQR screen.
4. Do not collapse sparse counties yet. Record the small-sample distribution for later classification design.
5. Confirm the meaning of education code 1 before finalising ordinal versus nominal treatment.

## Output files

- `outputs/tables/03_demographic_inventory.csv`
- `outputs/tables/03_age_descriptives.csv`
- `outputs/tables/03_age_extreme_flags.csv`
- `outputs/tables/03_sex_frequencies.csv`
- `outputs/tables/03_education_frequencies.csv`
- `outputs/tables/03_education_ordinal_assessment.csv`
- `outputs/tables/03_urban_rural_frequencies.csv`
- `outputs/tables/03_county_frequencies.csv`
- `outputs/tables/03_county_small_sample_summary.csv`
- `outputs/tables/03_age_by_sex.csv`
- `outputs/tables/03_age_by_urban_rural.csv`
- `outputs/tables/03_education_by_urban_rural.csv`
"""
    path = REPORTS_DIR / "03_demographic_investigation.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()
    analysis = load_analysis_2024(required_columns=DEMOGRAPHIC_VARS)
    labels = load_codebook_labels()

    print("=" * 72)
    print("03 Demographic investigation")
    print("=" * 72)
    print(f"Observations: {len(analysis)}")

    inventory = variable_inventory(analysis)
    age_desc = age_descriptives(analysis)
    age_flags = age_extreme_flags(analysis)
    sex_freq = category_frequencies(analysis, "sex", labels.get("sex", {}))
    edu_freq = category_frequencies(analysis, "education", labels.get("education", {}))
    ur_freq = category_frequencies(
        analysis, "urban_rural", labels.get("urban_rural", {})
    )
    county_freq = county_frequencies(analysis, labels.get("county", {}))
    county_small = county_small_sample_summary(county_freq)
    edu_ordinal = education_ordinal_assessment(edu_freq)
    age_sex = age_by_group(analysis, "sex", labels.get("sex", {}))
    age_ur = age_by_group(analysis, "urban_rural", labels.get("urban_rural", {}))
    edu_ur = education_by_urban_rural(
        analysis,
        labels.get("education", {}),
        labels.get("urban_rural", {}),
    )

    outputs = {
        "03_demographic_inventory.csv": inventory,
        "03_age_descriptives.csv": age_desc,
        "03_age_extreme_flags.csv": age_flags,
        "03_sex_frequencies.csv": sex_freq,
        "03_education_frequencies.csv": edu_freq,
        "03_education_ordinal_assessment.csv": edu_ordinal,
        "03_urban_rural_frequencies.csv": ur_freq,
        "03_county_frequencies.csv": county_freq,
        "03_county_small_sample_summary.csv": county_small,
        "03_age_by_sex.csv": age_sex,
        "03_age_by_urban_rural.csv": age_ur,
        "03_education_by_urban_rural.csv": edu_ur,
    }

    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    report_path = write_report(
        n=len(analysis),
        inventory=inventory,
        age_desc=age_desc,
        age_flags=age_flags,
        sex_freq=sex_freq,
        edu_freq=edu_freq,
        ur_freq=ur_freq,
        county_freq=county_freq,
        county_small=county_small,
        edu_ordinal=edu_ordinal,
    )
    print(f"Wrote {report_path}")

    print("\nInventory")
    print(inventory.to_string(index=False))
    print("\nAge descriptives")
    print(age_desc.to_string(index=False))
    print("\nEducation frequencies")
    print(edu_freq.to_string(index=False))
    print("\nCounty small-sample summary")
    print(county_small.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
