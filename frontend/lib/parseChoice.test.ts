import { describe, expect, it } from "vitest";
import { parseChoice } from "./parseChoice";

const education = [
  { value: 1, label: "None / informal" },
  { value: 2, label: "Primary" },
  { value: 3, label: "Secondary" },
  { value: 4, label: "Tertiary" },
];

describe("parseChoice", () => {
  it("matches second to Secondary", () => {
    const r = parseChoice("second", education);
    expect(r).toEqual({
      kind: "match",
      value: 3,
      label: "Secondary",
      exact: false,
    });
  });

  it("matches exact Secondary", () => {
    const r = parseChoice("Secondary", education);
    expect(r.kind).toBe("match");
    if (r.kind === "match") expect(r.exact).toBe(true);
  });

  it("matches prim to Primary", () => {
    const r = parseChoice("prim", education);
    expect(r).toMatchObject({ kind: "match", label: "Primary" });
  });
});
