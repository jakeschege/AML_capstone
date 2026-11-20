# Limitations (must surface in UI and prediction explanations)

1. Stack A (v0.2.0) on the all-shocks financial-expanded matrix: macro F1 ≈ 0.50 under the first 5-fold repeat (seed 42). Modest discrimination only — useful for exploring patterns, not for decisions about people.
2. Formal support is the weakest class in evaluation (formal F1 ≈ 0.31 on that same 5-fold protocol). Treat formal predictions cautiously; the app also flags low score margins.
3. Full per-class precision and recall for Stack A were not stored in the stacking summary table (only macro F1 and formal F1). Companion class-weighted logistic on the same 93 features (Script 13/14 folds) had roughly: personal P≈0.60 R≈0.39; informal P≈0.71 R≈0.57; formal P≈0.22 R≈0.54; did_nothing P≈0.47 R≈0.69 — indicative of class imbalance, not Stack A itself.
4. Low-confidence rule: score margin (max − second max) < 0.10 OR predicted class is formal.
5. Model scores are not calibrated probabilities.
6. Not for operational targeting or eligibility.
7. Approximate attributions ≠ causation. Rankings of factors are only partly stable; the UI speaks of “among the main factors,” not a single biggest driver.
8. Land size (acres) is excluded from the locked classifier.
