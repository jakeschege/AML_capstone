import { describe, expect, it } from "vitest";
import {
  buildScaleSentence,
  buildTakeawaySentence,
  formatPct,
  formatWeightedCount,
  suppressIfSmall,
} from "./format";

describe("formatWeightedCount", () => {
  it("formats large survey-weighted totals with digits", () => {
    expect(formatWeightedCount(985365)).toBe("985,365");
    expect(formatWeightedCount(3_200_000)).toBe("3.2 million");
    expect(formatWeightedCount(985365)).not.toMatch(/thousand/i);
  });
});

describe("formatPct", () => {
  it("formats shares as percentages", () => {
    expect(formatPct(0.2784)).toBe("28%");
    expect(formatPct(0.2784, 1)).toBe("27.8%");
  });
});

describe("suppressIfSmall", () => {
  it("flags rows below n=30", () => {
    const out = suppressIfSmall([
      { n: 29, label: "a" },
      { n: 30, label: "b" },
    ]);
    expect(out[0].suppressed).toBe(true);
    expect(out[1].suppressed).toBe(false);
  });
});

describe("sentence builders", () => {
  it("uses represented-by-the-survey wording", () => {
    const s = buildScaleSentence(985365, 1552);
    expect(s).toContain("represented by the survey");
    expect(s).toContain("not a claim about all farming households");
    expect(s).not.toMatch(/share of all farming households/);
  });

  it("builds takeaway from data", () => {
    const s = buildTakeawaySentence({
      noneWeighted: 274289,
      noneShare: 0.2784,
      groupLabel: "Not a member",
      groupPanel: "Farmer group membership",
      groupShare: 0.38,
    });
    expect(s).toContain("no coping action");
    expect(s).toContain("Not a member".toLowerCase());
    expect(s).toContain("patterns across groups");
  });
});
