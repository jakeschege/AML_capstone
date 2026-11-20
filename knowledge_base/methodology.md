# Methodology (prototype knowledge base)

## Sample
Drought-exposed Kenyan households from FinAccess 2024, n = 1,552.

## Target
Four-class `coping_category`: personal, informal, formal, did_nothing.

## Classifier (champion v0.1.0)
Multinomial logistic regression with `class_weight="balanced"`, C=1, solver=lbfgs, max_iter=2000, random_state=42.
Predictor specification: `primary_acres_excluded_v1` (68 features; `land_size_acres` excluded).

## Evaluation
Repeated stratified 5×5 cross-validation, seed 42.
Mean macro F1 ≈ 0.350; balanced accuracy ≈ 0.381; formal F1 ≈ 0.121; accuracy ≈ 0.388.
Fold standard deviations are not confidence intervals.

## Serving
The serving artefact is refit on the full drought-exposed matrix. Do not present in-sample metrics as validation.

## Limitations
Modest discrimination. Formal support is sparse and weak. Not for operational targeting or eligibility decisions.
SHAP attributions are not causal claims.

## Segmentation
Spec A Gower + PAM, working k=5. Coping outcomes excluded from clustering.
Out-of-sample assignment uses nearest medoid under Gower distance. Segment is descriptive typology, not a coping predictor.
