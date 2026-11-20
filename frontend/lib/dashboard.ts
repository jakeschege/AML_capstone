import { CLASS_ORDER, type ClassId } from "./constants";
import { getApiBase } from "./config";

/** @deprecated prefer getApiBase() */
const API_BASE = getApiBase();

export type WhoRow = {
  label: string;
  none_share: number | null;
  n: number;
  weighted: number;
  suppressed: boolean;
  small_sample?: boolean;
};

export type ShockOption = {
  key: string;
  label: string;
  n: number;
  disabled?: boolean;
};

export type MainShockRow = {
  main_shock_code: number;
  label: string;
  n: number;
  weighted: number;
  mix: Record<string, number>;
  mix_unweighted?: Record<string, number>;
  unavailable: boolean;
  unavailable_reason: string | null;
  small_sample?: boolean;
  highlighted?: boolean;
};

export type DashboardPayload = {
  updated: string;
  wave: number;
  source: string;
  version?: string;
  filter_definition?: string;
  filter_note?: string;
  weight_variable: string;
  weight_note: string;
  n_respondents: number;
  affected_weighted: number;
  affected_unweighted: number;
  small_sample?: boolean;
  selected_shock?: string;
  shock_key?: string;
  shock_label?: string;
  shock_options?: ShockOption[];
  class_order: ClassId[];
  display_labels: Record<string, string>;
  category_colours: Record<string, string>;
  mix: Record<ClassId, number>;
  mix_weighted: Record<ClassId, number>;
  mix_unweighted: Record<ClassId, number>;
  geography_level: string;
  geography_note: string;
  counties: Array<{
    name: string;
    mix: number[];
    mix_by_class: Record<string, number>;
    affected_weighted: number;
    n: number;
    suppressed: boolean;
    small_sample?: boolean;
  }>;
  segments: Array<{
    id: number;
    name: string;
    size_weighted: number;
    size_share: number;
    n: number;
    mix: number[];
    mix_by_class: Record<string, number>;
    suppressed: boolean;
    small_sample?: boolean;
  }>;
  who: Record<string, WhoRow[]>;
  challenges: Array<{
    label: string;
    share: number;
    n: number;
    weighted: number;
    small_sample?: boolean;
  }>;
  financial_access?: Array<{
    key: string;
    label: string;
    share: number;
    n: number;
    weighted: number;
    suppressed: boolean;
    small_sample?: boolean;
  }>;
  coping_by_main_shock?: MainShockRow[];
  coping_by_main_shock_note?: string;
  highlighted_main_shock_code?: number | null;
  suppressed_cells?: {
    counties_total: number;
    counties_suppressed_n_lt_30: number;
    counties_small_sample_n_lt_100: number;
    segments_total: number;
    segments_suppressed_n_lt_30: number;
    segments_small_sample_n_lt_100: number;
  };
  takeaway: {
    none_weighted: number;
    none_share: number;
    highest_none_group: { panel: string; label: string; none_share: number } | null;
  };
  methods_note: string;
  min_cell_n: number;
  small_sample_n?: number;
};

export async function fetchDashboard(shock = "all"): Promise<DashboardPayload> {
  const qs = shock && shock !== "all" ? `?shock=${encodeURIComponent(shock)}` : "";
  const res = await fetch(`${getApiBase()}/api/v1/dashboard${qs}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as DashboardPayload;
  if (!data.class_order?.length) data.class_order = [...CLASS_ORDER];
  return data;
}

export { API_BASE };
