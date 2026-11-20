/** Shared coping-route labels and colours — artefact ids, four classes. */

export const CLASS_ORDER = ["personal", "informal", "formal", "did_nothing"] as const;
export type ClassId = (typeof CLASS_ORDER)[number];

/** Artefact display names (class_mapping.json). */
export const DISPLAY_LABELS: Record<ClassId, string> = {
  personal: "Personal coping",
  informal: "Informal support",
  formal: "Formal support",
  did_nothing: "No coping action reported",
};

/** Four distinct colours aligned with the dashboard brief. */
export const CLASS_COLOURS: Record<ClassId, string> = {
  personal: "#1f8bd1",
  informal: "#f2a516",
  formal: "#1a6b3c",
  did_nothing: "#c0562b",
};

export const MIN_CELL_N = 30;

export const NAV = [
  { href: "/overview", label: "Overview" },
  { href: "/predictions", label: "Predictions" },
] as const;
