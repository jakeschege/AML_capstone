"""Pydantic request/response models (extra fields forbidden)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HouseholdRequest(StrictModel):
    """Raw domain fields for drought-exposed household interview."""

    age_years: float = Field(..., ge=0, le=120)
    sex: Literal[1, 2]
    education: Literal[1, 2, 3, 4]
    urban_rural: Literal[1, 2]
    county: int
    has_cash_crops: Literal[0, 1]
    has_cereals: Literal[0, 1]
    has_dairy_livestock: Literal[0, 1]
    has_horticulture: Literal[0, 1]
    has_legumes: Literal[0, 1]
    has_other_agri: Literal[0, 1]
    has_other_livestock: Literal[0, 1]
    has_poultry: Literal[0, 1]
    has_roots_tubers: Literal[0, 1]
    n_crop_types: float = Field(..., ge=0)
    n_livestock_types: float = Field(..., ge=0)
    land_ownership_own_under_your_name: Literal[0, 1]
    land_ownership_own_jointly_with_spouse: Literal[0, 1]
    land_ownership_own_jointly_with_someone_else_other_than_spouse: Literal[0, 1]
    land_ownership_lease_rent_you_dont_own: Literal[0, 1]
    land_ownership_community_public_owned_e_g_pastoralists: Literal[0, 1]
    land_ownership_family_owned_e_g_inherited_family_land: Literal[0, 1]
    # All-shocks financial-expanded Stack A / logistic (v0.2+)
    main_shock_type: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bank_usage: Literal[0, 1]
    buyer_supplier_credit: Literal[0, 1]
    formal_agri_borrowing: Literal[0, 1]
    insurance_excl_nhif: Literal[0, 1]
    loan_formal: Literal[0, 1]
    loan_informal: Literal[0, 1]
    loan_usage: Literal[0, 1]
    mobile_bank_usage: Literal[0, 1]
    mobile_money_usage: Literal[0, 1]
    sacco_usage: Literal[0, 1]
    savings_formal: Literal[0, 1]
    savings_informal: Literal[0, 1]
    savings_usage: Literal[0, 1]
    include_segment: bool = False


class SegmentRequest(StrictModel):
    has_cash_crops: Literal[0, 1]
    has_cereals: Literal[0, 1]
    has_dairy_livestock: Literal[0, 1]
    has_horticulture: Literal[0, 1]
    has_legumes: Literal[0, 1]
    has_other_agri: Literal[0, 1]
    has_other_livestock: Literal[0, 1]
    has_poultry: Literal[0, 1]
    has_roots_tubers: Literal[0, 1]
    n_crop_types: float = Field(..., ge=0)
    n_livestock_types: float = Field(..., ge=0)
    land_ownership_own_under_your_name: Literal[0, 1]
    land_ownership_own_jointly_with_spouse: Literal[0, 1]
    land_ownership_own_jointly_with_someone_else_other_than_spouse: Literal[0, 1]
    land_ownership_lease_rent_you_dont_own: Literal[0, 1]
    land_ownership_community_public_owned_e_g_pastoralists: Literal[0, 1]
    land_ownership_family_owned_e_g_inherited_family_land: Literal[0, 1]


class ExplainRequest(HouseholdRequest):
    top_k: int = Field(12, ge=1, le=40)
    for_class: Optional[
        Literal["personal", "informal", "formal", "did_nothing"]
    ] = None


class HealthResponse(StrictModel):
    status: str
    model_version: str
    loaded_at: str


class PredictResponse(StrictModel):
    prediction: str
    display_label: str
    scores: dict[str, float]
    model_version: str
    low_confidence: bool
    score_margin: float
    warnings: list[str]
    segment: Optional[dict[str, Any]] = None


class ExplainResponse(PredictResponse):
    top_features: list[dict[str, Any]]
    shap_note: str
    predicted_class_for_shap: str
    explanation_mode: str = "approximate_attribution"
    attribution_method: Optional[str] = None
    baseline_scores: Optional[dict[str, float]] = None
    baseline_description: Optional[str] = None
    toward_features: Optional[list[dict[str, Any]]] = None
    against_features: Optional[list[dict[str, Any]]] = None
    attributions_by_class: Optional[dict[str, list[dict[str, Any]]]] = None
    additivity_abs_error: Optional[dict[str, float]] = None
