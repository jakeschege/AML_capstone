import { getApiBase } from "./config";

/** @deprecated prefer getApiBase() — kept for existing imports */
const API_BASE = getApiBase();

export type HouseholdPayload = {
  age_years: number;
  sex: 1 | 2;
  education: 1 | 2 | 3 | 4;
  urban_rural: 1 | 2;
  county: number;
  has_cash_crops: 0 | 1;
  has_cereals: 0 | 1;
  has_dairy_livestock: 0 | 1;
  has_horticulture: 0 | 1;
  has_legumes: 0 | 1;
  has_other_agri: 0 | 1;
  has_other_livestock: 0 | 1;
  has_poultry: 0 | 1;
  has_roots_tubers: 0 | 1;
  n_crop_types: number;
  n_livestock_types: number;
  land_ownership_own_under_your_name: 0 | 1;
  land_ownership_own_jointly_with_spouse: 0 | 1;
  land_ownership_own_jointly_with_someone_else_other_than_spouse: 0 | 1;
  land_ownership_lease_rent_you_dont_own: 0 | 1;
  land_ownership_community_public_owned_e_g_pastoralists: 0 | 1;
  land_ownership_family_owned_e_g_inherited_family_land: 0 | 1;
  main_shock_type: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
  bank_usage: 0 | 1;
  buyer_supplier_credit: 0 | 1;
  formal_agri_borrowing: 0 | 1;
  insurance_excl_nhif: 0 | 1;
  loan_formal: 0 | 1;
  loan_informal: 0 | 1;
  loan_usage: 0 | 1;
  mobile_bank_usage: 0 | 1;
  mobile_money_usage: 0 | 1;
  sacco_usage: 0 | 1;
  savings_formal: 0 | 1;
  savings_informal: 0 | 1;
  savings_usage: 0 | 1;
  include_segment?: boolean;
};

export type ShapFeature = {
  name: string;
  shap: number;
  direction: string;
  label?: string;
  magnitude_band?: string;
};

export type PredictResult = {
  prediction: string;
  display_label: string;
  scores: Record<string, number>;
  model_version: string;
  low_confidence: boolean;
  score_margin: number;
  warnings: string[];
  segment?: {
    assigned: boolean;
    cluster_id: number;
    label: string;
    distances?: Record<string, number>;
  } | null;
  top_features?: ShapFeature[];
  toward_features?: ShapFeature[];
  against_features?: ShapFeature[];
  shap_note?: string;
  predicted_class_for_shap?: string;
  explanation_mode?: string;
  attribution_method?: string | null;
  baseline_scores?: Record<string, number>;
  baseline_description?: string;
  attributions_by_class?: Record<string, ShapFeature[]>;
};

export type DomainInput = {
  name: string;
  type: string;
  required?: boolean;
  allowed_codes?: { code: number; label: string }[];
  label_key?: string;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function getMetadata() {
  return api<{
    model_version: string;
    limitations: string;
    cv_metrics: Record<string, number | string>;
    algorithm: string;
    predictor_spec_id?: string;
  }>("/api/v1/metadata");
}

export function getModelInfo() {
  return api<{
    model_version: string;
    class_mapping: {
      classes: string[];
      display_labels: Record<string, string>;
      category_colours: Record<string, string>;
    };
    feature_schema: {
      domain_inputs: DomainInput[];
      excluded: string[];
      n_features: number;
    };
    metrics: Record<string, number | string>;
    segmentation_available?: boolean;
  }>("/api/v1/model-info");
}

export function predict(payload: HouseholdPayload) {
  return api<PredictResult>("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function explain(
  payload: HouseholdPayload & {
    top_k?: number;
    for_class?: "personal" | "informal" | "formal" | "did_nothing";
  },
) {
  return api<PredictResult>("/api/v1/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function predictionChatFollowUp(message: string, explainResult: PredictResult) {
  return api<{ reply: string; source?: string; used_llm?: boolean }>(
    "/api/v1/prediction-chat/follow-up",
    {
      method: "POST",
      body: JSON.stringify({
        message,
        explain_result: explainResult,
        history: [],
      }),
    },
  );
}

export function segment(payload: Record<string, number>) {
  return api<{
    assigned: boolean;
    cluster_id?: number;
    label?: string;
    distances?: Record<string, number>;
    profiles?: unknown[];
    note?: string;
  }>("/api/v1/segment", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export { API_BASE };
