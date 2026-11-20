#!/usr/bin/env python3
"""Temporal availability review using questionnaire documentation only.

Organises unresolved temporally flagged predictors and quotes supporting
questionnaire evidence. Does not infer timing from variable names or
correlations. Does not exclude variables or finalise the predictor set.
"""

from __future__ import annotations

import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKBENCH_ROOT = Path(__file__).resolve().parent.parent
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from config import (  # noqa: E402
    COLUMN_RENAME_MAP,
    FEATURE_DICTIONARY,
    LEAKAGE_REGISTER,
    QUESTIONNAIRE_XLSX,
    R2_RENAME_MAP,
    REPORTS_DIR,
    TABLES_DIR,
    TEMPORAL_FLAGS,
    VALUE_LABELS_PATH,
    VARIABLE_LABELS_PATH,
    ensure_output_dirs,
)

QUESTIONNAIRE_SOURCE = str(QUESTIONNAIRE_XLSX)


def load_questionnaire_text(path: Path) -> str:
    """Extract plain text from xlsx shared strings without openpyxl."""
    if not path.exists():
        return ""
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", raw)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return re.sub(r"\s+", " ", text)


def excerpt_around(text: str, pattern: str, window: int = 280) -> str:
    """Return a short excerpt around the first regex match."""
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = max(0, match.start() - 40)
    end = min(len(text), match.end() + window)
    return text[start:end].strip()


def build_module_evidence(questionnaire_text: str) -> dict[str, dict]:
    """Collect module-level excerpts that can be attached to variables."""
    evidence: dict[str, dict] = {}

    r2a = excerpt_around(
        questionnaire_text,
        r"R2A\.\s*In the past 12 months have you experienced high cost",
        window=320,
    )
    r2b = excerpt_around(
        questionnaire_text,
        r"R2B\.\s*Of the above mentioned, which event/shock made the biggest impact",
        window=260,
    )
    r2c = excerpt_around(
        questionnaire_text,
        r"R2C\.\s*What did you do to manage",
        window=260,
    )
    q12 = excerpt_around(
        questionnaire_text,
        r"Q12\s+In the past 12 months, did you give customers/buyers",
        window=220,
    )
    q13 = excerpt_around(
        questionnaire_text,
        r"Q13\s+In the past 12 months, did you receive goods or services",
        window=220,
    )

    if r2a:
        evidence["R2A"] = {
            "evidence_type": "questionnaire_module_stem",
            "excerpt": r2a,
            "reference_period_text": "In the past 12 months",
            "source": QUESTIONNAIRE_SOURCE,
        }
    if r2b:
        evidence["R2B"] = {
            "evidence_type": "questionnaire_module_stem",
            "excerpt": r2b,
            "reference_period_text": "in the past 12 months"
            if "past 12 months" in r2b.lower() or "past 1" in r2b.lower()
            else "",
            "source": QUESTIONNAIRE_SOURCE,
        }
    if r2c:
        evidence["R2C"] = {
            "evidence_type": "questionnaire_module_stem",
            "excerpt": r2c,
            "reference_period_text": "response to use case from R2B",
            "source": QUESTIONNAIRE_SOURCE,
        }
    if q12:
        evidence["Q12"] = {
            "evidence_type": "questionnaire_item",
            "excerpt": q12,
            "reference_period_text": "In the past 12 months",
            "source": QUESTIONNAIRE_SOURCE,
        }
    if q13:
        evidence["Q13"] = {
            "evidence_type": "questionnaire_item",
            "excerpt": q13,
            "reference_period_text": "In the past 12 months",
            "source": QUESTIONNAIRE_SOURCE,
        }

    # Search for common usage-status wording if present
    for key, pat in {
        "usage_status_currently_use": r"Currently use.{0,80}Used to use.{0,80}Never used",
        "bank_usage_label": r"bank_usage.{0,120}|All commecial banks.{0,120}",
    }.items():
        ex = excerpt_around(questionnaire_text, pat, window=160)
        if ex:
            evidence[key] = {
                "evidence_type": "questionnaire_search",
                "excerpt": ex,
                "reference_period_text": "",
                "source": QUESTIONNAIRE_SOURCE,
            }

    return evidence


def load_rename_maps() -> dict[str, str]:
    """Map analysis variable -> original questionnaire/source column."""
    mapping: dict[str, str] = {}
    for path in (COLUMN_RENAME_MAP, R2_RENAME_MAP):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            mapping[str(row["renamed_column"])] = str(row["original_column"])

    # Feature dictionary may record constructed sources
    if FEATURE_DICTIONARY.exists():
        feat = pd.read_csv(FEATURE_DICTIONARY)
        if {"feature", "source_columns"}.issubset(feat.columns):
            for _, row in feat.iterrows():
                feature = str(row["feature"])
                sources = str(row["source_columns"])
                if feature not in mapping and sources and sources.lower() != "nan":
                    mapping[feature] = sources
    return mapping


def load_variable_labels() -> dict[str, str]:
    """Original variable -> label text."""
    labels: dict[str, str] = {}
    if not VARIABLE_LABELS_PATH.exists():
        return labels
    df = pd.read_csv(VARIABLE_LABELS_PATH)
    for _, row in df.iterrows():
        labels[str(row["variable"])] = str(row.get("label", ""))
    return labels


def select_scope(register: pd.DataFrame) -> pd.DataFrame:
    """Variables requiring temporal documentation review."""
    temporal = register["primary_recommendation"] == "flag_temporal_verification"
    financial_redundant = (
        (register["group"] == "financial")
        & (register["primary_recommendation"] == "flag_redundancy")
    )
    scope = register.loc[temporal | financial_redundant].copy()
    # Exclude agricultural redundancy-only items such as shannon_diversity
    scope = scope.loc[scope["group"].isin(["shock", "financial"])].copy()
    scope = scope.sort_values(["group", "variable"]).reset_index(drop=True)
    return scope


def evidence_rows_for_variable(
    variable: str,
    group: str,
    original: str,
    var_labels: dict[str, str],
    module_evidence: dict[str, dict],
) -> list[dict]:
    """Attach documentation excerpts relevant to one variable."""
    rows: list[dict] = []

    # Label evidence
    originals = [part.strip() for part in original.split(";") if part.strip()]
    for orig in originals or [original]:
        label = var_labels.get(orig, "")
        if label and label.lower() != "nan":
            rows.append(
                {
                    "variable": variable,
                    "original_item": orig,
                    "evidence_type": "variable_label",
                    "excerpt": label,
                    "reference_period_text": "",
                    "source": str(VARIABLE_LABELS_PATH),
                }
            )

    if group == "shock":
        if variable == "main_shock_type" and "R2B" in module_evidence:
            ev = module_evidence["R2B"]
            rows.append(
                {
                    "variable": variable,
                    "original_item": original or "R2B",
                    "evidence_type": ev["evidence_type"],
                    "excerpt": ev["excerpt"],
                    "reference_period_text": ev["reference_period_text"],
                    "source": ev["source"],
                }
            )
        if "R2A" in module_evidence:
            ev = module_evidence["R2A"]
            rows.append(
                {
                    "variable": variable,
                    "original_item": original or "R2A",
                    "evidence_type": ev["evidence_type"],
                    "excerpt": ev["excerpt"],
                    "reference_period_text": ev["reference_period_text"],
                    "source": ev["source"],
                }
            )
        if "R2C" in module_evidence:
            # Include R2C stem to document ordering relative to coping outcome
            ev = module_evidence["R2C"]
            rows.append(
                {
                    "variable": variable,
                    "original_item": "R2C",
                    "evidence_type": "related_outcome_module_stem",
                    "excerpt": ev["excerpt"],
                    "reference_period_text": ev["reference_period_text"],
                    "source": ev["source"],
                }
            )

    if variable == "buyer_supplier_credit" and "Q12" in module_evidence:
        ev = module_evidence["Q12"]
        rows.append(
            {
                "variable": variable,
                "original_item": "Q12",
                "evidence_type": ev["evidence_type"],
                "excerpt": ev["excerpt"],
                "reference_period_text": ev["reference_period_text"],
                "source": ev["source"],
            }
        )
    if variable == "formal_agri_borrowing" and "Q13" in module_evidence:
        ev = module_evidence["Q13"]
        rows.append(
            {
                "variable": variable,
                "original_item": "Q13",
                "evidence_type": ev["evidence_type"],
                "excerpt": ev["excerpt"],
                "reference_period_text": ev["reference_period_text"],
                "source": ev["source"],
            }
        )

    # Derived financial flags inherit source documentation where mapped
    if variable == "used_buyer_credit" and "Q12" in module_evidence:
        ev = module_evidence["Q12"]
        rows.append(
            {
                "variable": variable,
                "original_item": "Q12 via buyer_supplier_credit",
                "evidence_type": ev["evidence_type"],
                "excerpt": ev["excerpt"],
                "reference_period_text": ev["reference_period_text"],
                "source": ev["source"],
            }
        )
    if variable == "used_formal_credit" and "Q13" in module_evidence:
        ev = module_evidence["Q13"]
        rows.append(
            {
                "variable": variable,
                "original_item": "Q13 component via formal_agri_borrowing",
                "evidence_type": "partial_source_documentation",
                "excerpt": ev["excerpt"],
                "reference_period_text": ev["reference_period_text"],
                "source": ev["source"],
            }
        )

    return rows


def assess_variable(
    variable: str,
    group: str,
    leakage_recommendation: str,
    evidence: pd.DataFrame,
) -> dict:
    """Assign documentation-based availability status and review recommendation."""
    var_ev = evidence.loc[evidence["variable"] == variable]
    periods = [
        p
        for p in var_ev["reference_period_text"].fillna("").tolist()
        if str(p).strip()
    ]
    excerpts = var_ev["excerpt"].fillna("").tolist()
    has_q_excerpt = any(str(x).strip() for x in excerpts)

    if group == "shock":
        has_r2a = var_ev["original_item"].astype(str).str.contains("R2A", na=False).any() or (
            "In the past 12 months" in " ".join(periods)
        )
        has_r2c = (
            var_ev["evidence_type"].astype(str) == "related_outcome_module_stem"
        ).any()
        if variable == "main_shock_type" and has_q_excerpt:
            status = "same_reference_window_as_coping_module"
            review_rec = "candidate_after_documented_ordering"
            rationale = (
                "Questionnaire documents R2B as selecting the main shock within the "
                "R2A past-12-month shock set, before R2C asks what was done to manage that shock."
            )
        elif has_r2a and has_r2c:
            status = "same_reference_window_as_coping_module"
            review_rec = "candidate_after_documented_ordering"
            rationale = (
                "R2A documents a past-12-month shock window. R2C asks for actions taken "
                "to manage the R2B shock. Same retrospective module as the coping target, "
                "with documented question order shock then response."
            )
        elif has_q_excerpt:
            status = "unresolved_after_document_review"
            review_rec = "keep_flagged_pending_decision"
            rationale = "Some questionnaire text was found, but the shock-to-coping ordering evidence is incomplete."
        else:
            status = "insufficient_questionnaire_evidence"
            review_rec = "insufficient_evidence_keep_flagged"
            rationale = "No questionnaire excerpt locating this shock item was recovered."
    elif variable in ("buyer_supplier_credit", "formal_agri_borrowing", "used_buyer_credit"):
        if any("past 12 months" in str(p).lower() for p in periods):
            status = "documented_past_12_months_outside_R2"
            review_rec = "keep_flagged_pending_decision"
            rationale = (
                "Questionnaire documents a past-12-months reference period for the agri-credit "
                "item. That window overlaps the R2 coping window, so pre-shock availability "
                "is not established even though the item is outside the R2 module."
            )
        else:
            status = "insufficient_questionnaire_evidence"
            review_rec = "insufficient_evidence_keep_flagged"
            rationale = "Agri-credit item lacked recoverable reference-period wording."
    elif variable == "used_formal_credit":
        # Only partial Q13 evidence among components
        if any("past 12 months" in str(p).lower() for p in periods):
            status = "unresolved_after_document_review"
            review_rec = "keep_flagged_pending_decision"
            rationale = (
                "Composite includes a documented past-12-months agri-credit component, "
                "but other source components lack recovered questionnaire timing evidence."
            )
        else:
            status = "insufficient_questionnaire_evidence"
            review_rec = "insufficient_evidence_keep_flagged"
            rationale = "Composite source timing could not be documented from questionnaire excerpts."
    else:
        # bank/savings/loan/mobile/mfi/digital/insurance usage measures
        if has_q_excerpt and periods:
            status = "unresolved_after_document_review"
            review_rec = "keep_flagged_pending_decision"
            rationale = (
                "Some documentation was found, but an explicit pre-shock timing statement "
                "relative to drought coping was not established."
            )
        elif has_q_excerpt:
            status = "insufficient_questionnaire_evidence"
            review_rec = "insufficient_evidence_keep_flagged"
            rationale = (
                "Variable labels or construction notes exist, but no explicit questionnaire "
                "reference period relative to the coping module was recovered."
            )
        else:
            status = "insufficient_questionnaire_evidence"
            review_rec = "insufficient_evidence_keep_flagged"
            rationale = (
                "No questionnaire reference-period excerpt was recovered for this usage measure."
            )

    return {
        "variable": variable,
        "group": group,
        "leakage_audit_recommendation": leakage_recommendation,
        "availability_status": status,
        "review_recommendation": review_rec,
        "n_evidence_rows": int(len(var_ev)),
        "documented_reference_period": "; ".join(sorted(set(periods))) if periods else "",
        "rationale": rationale,
        "auto_excluded": False,
    }


def write_report(
    scope: pd.DataFrame,
    evidence: pd.DataFrame,
    assessments: pd.DataFrame,
    module_evidence: dict[str, dict],
) -> Path:
    """Write the temporal availability Markdown report."""
    counts = (
        assessments["availability_status"]
        .value_counts()
        .rename_axis("availability_status")
        .reset_index(name="n_variables")
    )
    review_counts = (
        assessments["review_recommendation"]
        .value_counts()
        .rename_axis("review_recommendation")
        .reset_index(name="n_variables")
    )

    quoted = []
    for key in ("R2A", "R2B", "R2C", "Q12", "Q13"):
        if key in module_evidence:
            quoted.append(f"### {key}\n\n> {module_evidence[key]['excerpt']}\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Temporal Availability Review Report

Generated: {timestamp}

## Purpose

This review organises questionnaire evidence for variables flagged in the leakage audit for temporal verification. It preserves the distinction between confirmed leakage, confirmed redundancy and unresolved predictor suitability.

No variables were automatically excluded. Timing was not inferred from variable names or correlations.

## Review scope

Variables reviewed: {len(scope)}

```
{scope[['variable','group','primary_recommendation','evidence_strength']].to_string(index=False)}
```

## Quoted questionnaire anchors

{''.join(quoted) if quoted else '_No module anchors recovered._'}

## Availability status counts

```
{counts.to_string(index=False)}
```

## Review recommendation counts

```
{review_counts.to_string(index=False)}
```

## Assessments

```
{assessments.to_string(index=False)}
```

## Interpretation

Documented questionnaire text shows that Section R2 asks about shocks in the past 12 months and then asks what the household did to manage the main shock. That supports treating shock indicators and main shock type as same-window measures with a documented within-module order, not as confirmed post-outcome leakage.

Q12 and Q13 also use a past-12-months window. Because that window overlaps the R2 coping window, agri-credit items remain flagged pending decision even though they sit outside the coping module.

Several broader financial usage measures lacked recoverable questionnaire reference-period wording in this review. They remain insufficiently evidenced for timing clearance.

## What this review does not decide

- Final classification predictor inclusions or exclusions
- Whether same-window shock measures should be used in supervised models
- Encoding, missingness handling or feature preparation

## Output files

- `outputs/tables/07_temporal_review_scope.csv`
- `outputs/tables/07_questionnaire_evidence.csv`
- `outputs/tables/07_temporal_availability_assessments.csv`
"""
    path = REPORTS_DIR / "07_temporal_availability_review.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dirs()

    if not LEAKAGE_REGISTER.exists():
        raise FileNotFoundError(
            f"Leakage register not found: {LEAKAGE_REGISTER}. Run script 06 first."
        )

    register = pd.read_csv(LEAKAGE_REGISTER)
    temporal_flags = (
        pd.read_csv(TEMPORAL_FLAGS) if TEMPORAL_FLAGS.exists() else pd.DataFrame()
    )
    scope = select_scope(register)
    if "evidence_strength" not in scope.columns and len(temporal_flags):
        scope = scope.merge(
            temporal_flags[["variable", "evidence_strength"]],
            on="variable",
            how="left",
        )

    rename_map = load_rename_maps()
    var_labels = load_variable_labels()
    q_text = load_questionnaire_text(QUESTIONNAIRE_XLSX)
    module_evidence = build_module_evidence(q_text)

    scope = scope.copy()
    scope["original_item"] = scope["variable"].map(lambda v: rename_map.get(v, ""))
    scope_out = scope[
        [
            c
            for c in [
                "variable",
                "group",
                "subgroup",
                "primary_recommendation",
                "evidence_strength",
                "classification_role",
                "notes",
                "original_item",
            ]
            if c in scope.columns
        ]
    ].copy()

    evidence_rows: list[dict] = []
    assessments: list[dict] = []
    for _, row in scope_out.iterrows():
        variable = row["variable"]
        group = row["group"]
        original = str(row.get("original_item", "") or "")
        ev_rows = evidence_rows_for_variable(
            variable=variable,
            group=group,
            original=original,
            var_labels=var_labels,
            module_evidence=module_evidence,
        )
        evidence_rows.extend(ev_rows)
        assessments.append(
            assess_variable(
                variable=variable,
                group=group,
                leakage_recommendation=row["primary_recommendation"],
                evidence=pd.DataFrame(ev_rows)
                if ev_rows
                else pd.DataFrame(
                    columns=[
                        "variable",
                        "original_item",
                        "evidence_type",
                        "excerpt",
                        "reference_period_text",
                        "source",
                    ]
                ),
            )
        )

    evidence_df = pd.DataFrame(evidence_rows)
    assessments_df = pd.DataFrame(assessments)

    outputs = {
        "07_temporal_review_scope.csv": scope_out,
        "07_questionnaire_evidence.csv": evidence_df,
        "07_temporal_availability_assessments.csv": assessments_df,
    }
    for name, table in outputs.items():
        path = TABLES_DIR / name
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    report_path = write_report(scope_out, evidence_df, assessments_df, module_evidence)
    print(f"Wrote {report_path}")

    print("\nScope")
    print(scope_out[["variable", "group", "primary_recommendation"]].to_string(index=False))
    print("\nAvailability status counts")
    print(assessments_df["availability_status"].value_counts().to_string())
    print("\nReview recommendations")
    print(assessments_df["review_recommendation"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
