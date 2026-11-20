/** Ordered interview questions for the Predictions conversation. */

export type FieldKey =
  | "age_years"
  | "sex"
  | "education"
  | "urban_rural"
  | "county"
  | "has_cash_crops"
  | "has_cereals"
  | "has_dairy_livestock"
  | "has_horticulture"
  | "has_legumes"
  | "has_other_agri"
  | "has_other_livestock"
  | "has_poultry"
  | "has_roots_tubers"
  | "n_crop_types"
  | "n_livestock_types"
  | "land_ownership_own_under_your_name"
  | "land_ownership_own_jointly_with_spouse"
  | "land_ownership_own_jointly_with_someone_else_other_than_spouse"
  | "land_ownership_lease_rent_you_dont_own"
  | "land_ownership_community_public_owned_e_g_pastoralists"
  | "land_ownership_family_owned_e_g_inherited_family_land"
  | "main_shock_type"
  | "bank_usage"
  | "buyer_supplier_credit"
  | "formal_agri_borrowing"
  | "insurance_excl_nhif"
  | "loan_formal"
  | "loan_informal"
  | "loan_usage"
  | "mobile_bank_usage"
  | "mobile_money_usage"
  | "sacco_usage"
  | "savings_formal"
  | "savings_informal"
  | "savings_usage";

export type ChatQuestion = {
  id: string;
  section: string;
  prompt: string;
  kind: "number" | "choice" | "multi";
  fields: FieldKey[];
  /** For choice / multi option chips */
  options?: { value: number; label: string }[];
  min?: number;
  max?: number;
  help?: string;
};

export const MAIN_SHOCK_OPTIONS = [
  { value: 1, label: "Child birth" },
  { value: 2, label: "Major sickness / accident / injury" },
  { value: 3, label: "Drought loss" },
  { value: 4, label: "Flooding / heavy rain / mudslide" },
  { value: 5, label: "Death of main income earner / family member" },
  { value: 6, label: "Theft" },
  { value: 7, label: "Job / income / wage cut" },
  { value: 8, label: "Lost home or land" },
  { value: 9, label: "Pest / disease crop or livestock failure" },
  { value: 10, label: "Fire / violence" },
  { value: 11, label: "Human–wildlife conflict" },
  { value: 12, label: "Care-related income / expense shock" },
];

export const CHAT_QUESTIONS: ChatQuestion[] = [
  {
    id: "age",
    section: "Household",
    prompt: "What is the age of the household head, in years?",
    kind: "number",
    fields: ["age_years"],
    min: 0,
    max: 120,
  },
  {
    id: "sex",
    section: "Household",
    prompt: "What is the sex of the respondent?",
    kind: "choice",
    fields: ["sex"],
    options: [
      { value: 1, label: "Male" },
      { value: 2, label: "Female" },
    ],
  },
  {
    id: "education",
    section: "Household",
    prompt: "What is the highest education level completed?",
    kind: "choice",
    fields: ["education"],
    options: [
      { value: 1, label: "None / informal" },
      { value: 2, label: "Primary" },
      { value: 3, label: "Secondary" },
      { value: 4, label: "Tertiary" },
    ],
  },
  {
    id: "urban_rural",
    section: "Household",
    prompt: "Is the household in a rural or urban area?",
    kind: "choice",
    fields: ["urban_rural"],
    options: [
      { value: 1, label: "Rural" },
      { value: 2, label: "Urban" },
    ],
  },
  {
    id: "county",
    section: "Household",
    prompt: "Which county is the household in?",
    kind: "choice",
    fields: ["county"],
    // options filled from API domain_inputs
  },
  {
    id: "crops",
    section: "Crops and livestock",
    prompt: "Which of these crop activities does the household have? Tap all that apply.",
    kind: "multi",
    fields: [
      "has_cash_crops",
      "has_cereals",
      "has_horticulture",
      "has_legumes",
      "has_roots_tubers",
      "has_other_agri",
    ],
    options: [
      { value: 0, label: "Cash crops" }, // value unused for multi; field order maps
    ],
    help: "Select every matching activity, or None of these.",
  },
  {
    id: "livestock",
    section: "Crops and livestock",
    prompt: "Which livestock does the household keep? Tap all that apply.",
    kind: "multi",
    fields: ["has_dairy_livestock", "has_poultry", "has_other_livestock"],
    help: "Select every matching type, or None of these.",
  },
  {
    id: "n_crops",
    section: "Crops and livestock",
    prompt: "How many crop types does the household grow?",
    kind: "number",
    fields: ["n_crop_types"],
    min: 0,
    max: 50,
  },
  {
    id: "n_livestock",
    section: "Crops and livestock",
    prompt: "How many livestock types does the household keep?",
    kind: "number",
    fields: ["n_livestock_types"],
    min: 0,
    max: 50,
  },
  {
    id: "land",
    section: "Land ownership",
    prompt: "How is the household’s land owned or held? Tap all that apply.",
    kind: "multi",
    fields: [
      "land_ownership_own_under_your_name",
      "land_ownership_own_jointly_with_spouse",
      "land_ownership_own_jointly_with_someone_else_other_than_spouse",
      "land_ownership_lease_rent_you_dont_own",
      "land_ownership_community_public_owned_e_g_pastoralists",
      "land_ownership_family_owned_e_g_inherited_family_land",
    ],
    help: "Select every matching arrangement, or None of these. Land size in acres is not used by the model.",
  },
  {
    id: "main_shock",
    section: "Main shock",
    prompt: "What was the household’s main reported shock?",
    kind: "choice",
    fields: ["main_shock_type"],
    options: MAIN_SHOCK_OPTIONS,
  },
  {
    id: "fin_channels",
    section: "Financial access",
    prompt: "Which of these financial channels does the household use? Tap all that apply.",
    kind: "multi",
    fields: ["bank_usage", "sacco_usage", "mobile_money_usage", "mobile_bank_usage"],
    help: "General access patterns — not shock-coping actions.",
  },
  {
    id: "fin_savings",
    section: "Financial access",
    prompt: "Which savings options apply? Tap all that apply.",
    kind: "multi",
    fields: ["savings_usage", "savings_formal", "savings_informal"],
  },
  {
    id: "fin_loans",
    section: "Financial access",
    prompt: "Which loan or credit options apply? Tap all that apply.",
    kind: "multi",
    fields: [
      "loan_usage",
      "loan_formal",
      "loan_informal",
      "buyer_supplier_credit",
      "formal_agri_borrowing",
    ],
  },
  {
    id: "insurance",
    section: "Financial access",
    prompt: "Does the household have insurance excluding NHIF?",
    kind: "choice",
    fields: ["insurance_excl_nhif"],
    options: [
      { value: 1, label: "Yes" },
      { value: 0, label: "No" },
    ],
  },
];

export const SECTIONS_ORDER = [
  "Household",
  "Crops and livestock",
  "Land ownership",
  "Main shock",
  "Financial access",
] as const;

export const MULTI_FIELD_LABELS: Record<string, string> = {
  has_cash_crops: "Cash crops",
  has_cereals: "Cereals",
  has_horticulture: "Horticulture",
  has_legumes: "Legumes",
  has_roots_tubers: "Roots or tubers",
  has_other_agri: "Other agri",
  has_dairy_livestock: "Dairy livestock",
  has_poultry: "Poultry",
  has_other_livestock: "Other livestock",
  land_ownership_own_under_your_name: "Owned under own name",
  land_ownership_own_jointly_with_spouse: "Jointly with spouse",
  land_ownership_own_jointly_with_someone_else_other_than_spouse: "Jointly with someone else",
  land_ownership_lease_rent_you_dont_own: "Leased or rented",
  land_ownership_community_public_owned_e_g_pastoralists: "Community / public",
  land_ownership_family_owned_e_g_inherited_family_land: "Family / inherited",
  bank_usage: "Bank account",
  sacco_usage: "SACCO",
  mobile_money_usage: "Mobile money",
  mobile_bank_usage: "Mobile banking",
  savings_usage: "Any savings",
  savings_formal: "Formal savings",
  savings_informal: "Informal savings",
  loan_usage: "Any loan / credit",
  loan_formal: "Formal loans",
  loan_informal: "Informal loans",
  buyer_supplier_credit: "Buyer / supplier credit",
  formal_agri_borrowing: "Formal agri borrowing",
};

/** Every HouseholdPayload scoring field appears in exactly one question. */
export function assertAllFieldsCovered(): string[] {
  const covered = new Set(CHAT_QUESTIONS.flatMap((q) => q.fields));
  const expected: FieldKey[] = [
    "age_years",
    "sex",
    "education",
    "urban_rural",
    "county",
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
    "land_ownership_own_under_your_name",
    "land_ownership_own_jointly_with_spouse",
    "land_ownership_own_jointly_with_someone_else_other_than_spouse",
    "land_ownership_lease_rent_you_dont_own",
    "land_ownership_community_public_owned_e_g_pastoralists",
    "land_ownership_family_owned_e_g_inherited_family_land",
    "main_shock_type",
    "bank_usage",
    "buyer_supplier_credit",
    "formal_agri_borrowing",
    "insurance_excl_nhif",
    "loan_formal",
    "loan_informal",
    "loan_usage",
    "mobile_bank_usage",
    "mobile_money_usage",
    "sacco_usage",
    "savings_formal",
    "savings_informal",
    "savings_usage",
  ];
  return expected.filter((f) => !covered.has(f));
}
