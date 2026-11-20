"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import type { HouseholdPayload } from "@/lib/api";
import { explain, predict } from "@/lib/api";

type DomainInput = {
  name: string;
  type: string;
  allowed_codes?: { code: number; label: string }[];
};

const STEPS: { title: string; fields: string[] }[] = [
  { title: "Demographics", fields: ["age_years", "sex", "education", "urban_rural", "county"] },
  {
    title: "Crops & livestock",
    fields: [
      "has_cash_crops",
      "has_cereals",
      "has_dairy_livestock",
      "has_horticulture",
      "has_legumes",
      "has_other_agri",
      "has_other_livestock",
      "has_poultry",
      "has_roots_tubers",
      "n_crop_types",
      "n_livestock_types",
    ],
  },
  {
    title: "Land ownership",
    fields: [
      "land_ownership_own_under_your_name",
      "land_ownership_own_jointly_with_spouse",
      "land_ownership_own_jointly_with_someone_else_other_than_spouse",
      "land_ownership_lease_rent_you_dont_own",
      "land_ownership_community_public_owned_e_g_pastoralists",
      "land_ownership_family_owned_e_g_inherited_family_land",
    ],
  },
];

const DEFAULTS: HouseholdPayload = {
  age_years: 40,
  sex: 1,
  education: 2,
  urban_rural: 1,
  county: 2,
  has_cash_crops: 0,
  has_cereals: 1,
  has_dairy_livestock: 0,
  has_horticulture: 0,
  has_legumes: 1,
  has_other_agri: 0,
  has_other_livestock: 0,
  has_poultry: 0,
  has_roots_tubers: 0,
  n_crop_types: 2,
  n_livestock_types: 0,
  land_ownership_own_under_your_name: 0,
  land_ownership_own_jointly_with_spouse: 0,
  land_ownership_own_jointly_with_someone_else_other_than_spouse: 0,
  land_ownership_lease_rent_you_dont_own: 0,
  land_ownership_community_public_owned_e_g_pastoralists: 0,
  land_ownership_family_owned_e_g_inherited_family_land: 1,
  main_shock_type: 3,
  bank_usage: 0,
  buyer_supplier_credit: 0,
  formal_agri_borrowing: 0,
  insurance_excl_nhif: 0,
  loan_formal: 0,
  loan_informal: 0,
  loan_usage: 0,
  mobile_bank_usage: 0,
  mobile_money_usage: 0,
  sacco_usage: 0,
  savings_formal: 0,
  savings_informal: 0,
  savings_usage: 0,
};

function labelize(name: string) {
  return name.replace(/^has_/, "").replace(/^land_ownership_/, "").replaceAll("_", " ");
}

export function InterviewForm({ domainInputs }: { domainInputs: DomainInput[] }) {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<HouseholdPayload>(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const byName = useMemo(() => {
    const m = new Map(domainInputs.map((d) => [d.name, d]));
    return m;
  }, [domainInputs]);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const payload = { ...form, include_segment: true };
      const pred = await predict(payload);
      let merged = { ...pred };
      try {
        const exp = await explain({ ...form, top_k: 12 });
        merged = { ...pred, top_features: exp.top_features, shap_note: exp.shap_note };
      } catch {
        merged = {
          ...pred,
          shap_note:
            "Local SHAP is not available for the stacked deployment model; scores above are still valid.",
        };
      }
      sessionStorage.setItem("coping_result", JSON.stringify(merged));
      sessionStorage.setItem("coping_household", JSON.stringify(form));
      router.push("/prediction");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const current = STEPS[step];

  return (
    <div className="step-enter grid gap-8 lg:grid-cols-[1.6fr_0.9fr]">
      <section className="space-y-6">
        <div className="flex items-center justify-between font-plex text-xs uppercase tracking-[0.16em] text-umber">
          <span className="border border-rule px-2 py-1">Interview step {step + 1}/{STEPS.length}</span>
          <span>{current.title}</span>
        </div>
        <div className="space-y-4">
          {current.fields.map((name) => {
            const meta = byName.get(name);
            const codes = meta?.allowed_codes;
            return (
              <label key={name} className="block font-plex text-sm">
                <span className="mb-1 block text-ink/80">{labelize(name)}</span>
                {codes ? (
                  <select
                    className="w-full border border-rule bg-paper px-3 py-2"
                    value={String(form[name as keyof HouseholdPayload])}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        [name]: Number(e.target.value),
                      }))
                    }
                  >
                    {codes.map((c) => (
                      <option key={c.code} value={c.code}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                ) : name.startsWith("has_") || name.startsWith("land_") ? (
                  <select
                    className="w-full border border-rule bg-paper px-3 py-2"
                    value={String(form[name as keyof HouseholdPayload])}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        [name]: Number(e.target.value) as 0 | 1,
                      }))
                    }
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                ) : (
                  <input
                    type="number"
                    className="w-full border border-rule bg-paper px-3 py-2 tabular-nums"
                    value={Number(form[name as keyof HouseholdPayload])}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        [name]: Number(e.target.value),
                      }))
                    }
                  />
                )}
              </label>
            );
          })}
        </div>
        <div className="flex gap-3">
          <button
            type="button"
            className="font-plex text-sm text-umber disabled:opacity-40"
            disabled={step === 0}
            onClick={() => setStep((s) => s - 1)}
          >
            Back
          </button>
          {step < STEPS.length - 1 ? (
            <button
              type="button"
              className="laterite-underline font-plex text-sm font-medium"
              onClick={() => setStep((s) => s + 1)}
            >
              Continue
            </button>
          ) : (
            <button
              type="button"
              className="laterite-underline font-plex text-sm font-medium disabled:opacity-50"
              disabled={busy}
              onClick={submit}
            >
              {busy ? "Scoring…" : "Score household"}
            </button>
          )}
        </div>
        {error ? <p className="font-plex text-sm text-laterite">{error}</p> : null}
      </section>
      <aside className="margin-note space-y-3 border-l border-rule pl-4">
        <p>Land size (acres) is omitted for classifier v0.1.0 — acres-excluded specification.</p>
        <p>Answers map to the same encoding used in packaging; the API rejects unknown fields.</p>
        <p>Scores are class probabilities from multinomial logistic regression with balanced class weights.</p>
      </aside>
    </div>
  );
}
