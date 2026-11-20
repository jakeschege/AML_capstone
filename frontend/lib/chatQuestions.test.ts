import { describe, expect, it } from "vitest";
import { assertAllFieldsCovered, CHAT_QUESTIONS } from "./chatQuestions";

describe("chatQuestions", () => {
  it("covers every scoring field exactly once across questions", () => {
    expect(assertAllFieldsCovered()).toEqual([]);
    const all = CHAT_QUESTIONS.flatMap((q) => q.fields);
    expect(new Set(all).size).toBe(all.length);
  });

  it("asks a manageable number of paced questions", () => {
    expect(CHAT_QUESTIONS.length).toBe(15);
  });
});
