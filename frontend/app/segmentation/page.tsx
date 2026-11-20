"use client";

import { useState } from "react";
import { segment } from "@/lib/api";

const DEFAULT = {
  has_cash_crops: 0 as 0 | 1,
  has_cereals: 1 as 0 | 1,
  has_dairy_livestock: 0 as 0 | 1,
  has_horticulture: 0 as 0 | 1,
  has_legumes: 1 as 0 | 1,
  has_other_agri: 0 as 0 | 1,
  has_other_livestock: 0 as 0 | 1,
  has_poultry: 0 as 0 | 1,
  has_roots_tubers: 0 as 0 | 1,
  n_crop_types: 2,
  n_livestock_types: 0,
  land_ownership_own_under_your_name: 0 as 0 | 1,
  land_ownership_own_jointly_with_spouse: 0 as 0 | 1,
  land_ownership_own_jointly_with_someone_else_other_than_spouse: 0 as 0 | 1,
  land_ownership_lease_rent_you_dont_own: 0 as 0 | 1,
  land_ownership_community_public_owned_e_g_pastoralists: 0 as 0 | 1,
  land_ownership_family_owned_e_g_inherited_family_land: 1 as 0 | 1,
};

export default function SegmentationPage() {
  const [form, setForm] = useState(DEFAULT);
  const [result, setResult] = useState<{
    assigned: boolean;
    cluster_id?: number;
    label?: string;
    note?: string;
    profiles?: Array<{ cluster_id: number; label: string }>;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setError(null);
    try {
      const out = await segment(form);
      setResult(out as typeof result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="font-display text-3xl">Spec A typology</h1>
      <p className="max-w-2xl text-ink/80">
        Nearest-medoid assignment under Gower distance (k=5). Descriptive only — not a coping predictor.
      </p>
      <button type="button" onClick={run} className="laterite-underline font-plex text-sm font-medium">
        Assign from sample profile
      </button>
      {error ? <p className="text-laterite">{error}</p> : null}
      {result?.assigned ? (
        <div className="border border-rule px-4 py-3">
          <p className="font-display text-xl">
            Cluster {result.cluster_id}: {result.label}
          </p>
          <p className="margin-note mt-2">{result.note}</p>
        </div>
      ) : result ? (
        <p>Not assigned. Profiles only.</p>
      ) : null}
      {result?.profiles ? (
        <div className="grid gap-3 md:grid-cols-2">
          {result.profiles.map((p) => (
            <div key={p.cluster_id} className="border-l-2 border-umber pl-3">
              <p className="font-plex text-xs uppercase tracking-wide text-umber">Cluster {p.cluster_id}</p>
              <p className="font-serif text-sm">{p.label}</p>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
