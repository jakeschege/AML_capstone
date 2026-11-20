import { DISPLAY_LABELS, type ClassId } from "./constants";

/** Round for display only; keep raw fractions for math. Use digits, not word suffixes. */
export function formatWeightedCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n >= 1_000_000) {
    const millions = n / 1_000_000;
    return millions >= 10
      ? `${Math.round(millions).toLocaleString("en-KE")} million`
      : `${millions.toFixed(1)} million`;
  }
  return Math.round(n).toLocaleString("en-KE");
}

export function formatPct(share: number, digits = 0): string {
  if (!Number.isFinite(share)) return "—";
  return `${(share * 100).toFixed(digits)}%`;
}

export function formatUnweightedN(n: number): string {
  return `n = ${n.toLocaleString("en-KE")}`;
}

export function classLabel(id: string): string {
  return DISPLAY_LABELS[id as ClassId] ?? id;
}

export function buildScaleSentence(
  affectedWeighted: number,
  nRespondents: number,
  shockLabel = "shock-exposed",
): string {
  const who =
    shockLabel === "All shocks" || !shockLabel
      ? "shock-exposed farming households"
      : `farming households reporting ${shockLabel.toLowerCase()}`;
  return (
    `${formatWeightedCount(affectedWeighted)} ${who} ` +
    `are represented by the survey in this FinAccess 2024 analysis sample ` +
    `(${formatUnweightedN(nRespondents)} interviewed). ` +
    `Every figure below uses survey weights within this sample, not a claim about all farming households in Kenya.`
  );
}

export function buildTakeawaySentence(input: {
  noneWeighted: number;
  noneShare: number;
  groupLabel?: string | null;
  groupPanel?: string | null;
  groupShare?: number | null;
}): string {
  const base =
    `About ${formatPct(input.noneShare)} of these households — roughly ${formatWeightedCount(input.noneWeighted)} ` +
    `represented by the survey — reported no coping action.`;
  if (input.groupLabel && input.groupShare != null && input.groupPanel) {
    return (
      `${base} Among the groups shown, ${input.groupLabel.toLowerCase()} ` +
      `(${input.groupPanel.toLowerCase()}) had the highest share without a reported coping action ` +
      `(${formatPct(input.groupShare)}). These are patterns across groups, not reasons why any one household coped as it did.`
    );
  }
  return `${base} These are patterns across groups, not reasons why any one household coped as it did.`;
}

export function suppressIfSmall<T extends { n: number; suppressed?: boolean }>(
  rows: T[],
  minN = 30,
): T[] {
  return rows.map((r) => ({ ...r, suppressed: r.n < minN }));
}
