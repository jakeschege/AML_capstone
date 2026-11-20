/** Everyday labels for schema field names. */

const LABELS: Record<string, string> = {
  age_years: "Age of household head (years)",
  sex: "Sex of respondent",
  education: "Highest education level",
  urban_rural: "Urban or rural",
  county: "County",
  has_cash_crops: "Grows cash crops",
  has_cereals: "Grows cereals",
  has_dairy_livestock: "Keeps dairy livestock",
  has_horticulture: "Grows horticulture crops",
  has_legumes: "Grows legumes",
  has_other_agri: "Other crop or agri activity",
  has_other_livestock: "Keeps other livestock",
  has_poultry: "Keeps poultry",
  has_roots_tubers: "Grows roots or tubers",
  n_crop_types: "Number of crop types",
  n_livestock_types: "Number of livestock types",
  land_ownership_own_under_your_name: "Land owned under own name",
  land_ownership_own_jointly_with_spouse: "Land owned jointly with spouse",
  land_ownership_own_jointly_with_someone_else_other_than_spouse:
    "Land owned jointly with someone else",
  land_ownership_lease_rent_you_dont_own: "Land leased or rented",
  land_ownership_community_public_owned_e_g_pastoralists: "Community or public land",
  land_ownership_family_owned_e_g_inherited_family_land: "Family-owned / inherited land",
  main_shock_type: "Main reported shock",
  bank_usage: "Uses a bank account",
  sacco_usage: "Uses a SACCO",
  mobile_money_usage: "Uses mobile money",
  mobile_bank_usage: "Uses mobile banking",
  savings_usage: "Uses any savings channel",
  savings_formal: "Uses formal savings",
  savings_informal: "Uses informal savings",
  loan_usage: "Uses any loan / credit",
  loan_formal: "Uses formal loans",
  loan_informal: "Uses informal loans",
  insurance_excl_nhif: "Has insurance (excluding NHIF)",
  buyer_supplier_credit: "Uses buyer / supplier credit",
  formal_agri_borrowing: "Uses formal agricultural borrowing",
};

export function plainFieldLabel(name: string): string {
  return LABELS[name] ?? name.replaceAll("_", " ");
}

export function plainFeatureName(name: string): string {
  if (name.startsWith("county_")) {
    const parts = name.split("_").slice(2);
    return `County: ${parts.join(" ").replace(/\b\w/g, (c) => c.toUpperCase())}`;
  }
  return plainFieldLabel(name.replace(/_bin$/, ""));
}
