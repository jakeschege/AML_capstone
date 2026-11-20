/** Copy resources — English now; Swahili keys reserved. */

export const en = {
  brand: "FinAccess Coping Field Notebook",
  overviewTitle: "A research interview ledger for shock coping",
  overviewLead:
    "Score a household under the locked acres-excluded class-weighted logistic model. Modest discrimination; not for operational targeting.",
  whatThisIsNot:
    "This is not a targeting tool, eligibility screen, or causal claim about why a household coped a certain way.",
  start: "Begin interview",
  assessment: "Interview",
  prediction: "Ledger scores",
  segmentation: "Typology strip",
  mostLikely: "Most likely category under this model",
  lowConfidence: "Low confidence — narrow score margin or formal class.",
  shapDisclaimer: "Feature attributions are not causal explanations.",
  acresNote: "Land size (acres) is not used in classifier v0.1.0.",
  modelBadge: "Model",
} as const;

export type Copy = typeof en;
