/** Fuzzy match typed interview answers to the current question's options. */

export type ChoiceOption = { value: number; label: string };

export type ParseChoiceResult =
  | { kind: "match"; value: number; label: string; exact: boolean }
  | { kind: "ambiguous"; candidates: ChoiceOption[] }
  | { kind: "none" };

function normalize(s: string): string {
  return s
    .trim()
    .toLowerCase()
    .replace(/[–—]/g, "/")
    .replace(/[^a-z0-9/\s]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function lev(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = i - 1;
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cur = row[j];
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + cost);
      prev = cur;
    }
  }
  return row[b.length];
}

/**
 * Match free text to choice options using the current question's labels
 * (prefix / word / substring / light edit distance). Unique hits only.
 */
export function parseChoice(
  text: string,
  options: ChoiceOption[],
): ParseChoiceResult {
  const t = normalize(text);
  if (!t || !options.length) return { kind: "none" };

  for (const o of options) {
    if (t === String(o.value) || t === normalize(o.label)) {
      return { kind: "match", value: o.value, label: o.label, exact: true };
    }
  }
  const asNum = Number(t);
  if (!Number.isNaN(asNum) && options.some((o) => o.value === asNum)) {
    const o = options.find((o) => o.value === asNum)!;
    return { kind: "match", value: o.value, label: o.label, exact: true };
  }

  const scored: { o: ChoiceOption; rank: number }[] = [];
  for (const o of options) {
    const lab = normalize(o.label);
    const words = lab.split(/[\s/]+/).filter(Boolean);
    if (lab.startsWith(t) || words.some((w) => w.startsWith(t))) {
      scored.push({ o, rank: 1 });
      continue;
    }
    if (t.length >= 3 && lab.includes(t)) {
      scored.push({ o, rank: 2 });
      continue;
    }
    if (t.length >= 3) {
      const d = Math.min(lev(t, lab), ...words.map((w) => lev(t, w)));
      const maxD = t.length <= 4 ? 1 : 2;
      if (d <= maxD) scored.push({ o, rank: 3 + d });
    }
  }

  if (!scored.length) return { kind: "none" };
  scored.sort((a, b) => a.rank - b.rank || a.o.label.localeCompare(b.o.label));
  const best = scored[0].rank;
  const top = scored.filter((s) => s.rank === best).map((s) => s.o);
  // de-dupe by value
  const uniq = [...new Map(top.map((o) => [o.value, o])).values()];
  if (uniq.length === 1) {
    return { kind: "match", value: uniq[0].value, label: uniq[0].label, exact: false };
  }
  return { kind: "ambiguous", candidates: uniq };
}
