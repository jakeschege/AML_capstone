"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CLASS_COLOURS, CLASS_ORDER, DISPLAY_LABELS, type ClassId } from "@/lib/constants";
import {
  CHAT_QUESTIONS,
  MULTI_FIELD_LABELS,
  SECTIONS_ORDER,
  type ChatQuestion,
  type FieldKey,
} from "@/lib/chatQuestions";
import {
  explain,
  getMetadata,
  getModelInfo,
  predictionChatFollowUp,
  predict,
  type DomainInput,
  type HouseholdPayload,
  type PredictResult,
} from "@/lib/api";
import { plainFieldLabel } from "@/lib/plainLabels";
import { parseChoice } from "@/lib/parseChoice";

type Msg = {
  role: "bot" | "user" | "system";
  content: string;
  meta?: string;
  source?: string;
};

type Phase = "interview" | "summary" | "result" | "followup";

function emptyAnswers(): Partial<Record<FieldKey, number>> {
  return {};
}

function buildPayload(answers: Partial<Record<FieldKey, number>>): HouseholdPayload | null {
  const required = CHAT_QUESTIONS.flatMap((q) => q.fields);
  for (const f of required) {
    if (answers[f] === undefined || Number.isNaN(answers[f] as number)) return null;
  }
  return answers as HouseholdPayload;
}

export function PredictionsView() {
  const [domainInputs, setDomainInputs] = useState<DomainInput[]>([]);
  const [colours, setColours] = useState<Record<string, string>>(CLASS_COLOURS);
  const [labels, setLabels] = useState<Record<string, string>>(DISPLAY_LABELS);
  const [meta, setMeta] = useState<{
    model_version: string;
    limitations: string;
    cv_metrics: Record<string, number | string>;
    algorithm: string;
  } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>("interview");
  const [qIndex, setQIndex] = useState(0);
  const [answers, setAnswers] = useState<Partial<Record<FieldKey, number>>>(emptyAnswers);
  const [multiSel, setMultiSel] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "bot",
      content:
        "I will ask about the household one question at a time. Answers are validated before scoring. Model scores are not calibrated probabilities.",
    },
  ]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const messagesScrollRef = useRef<HTMLDivElement | null>(null);

  const questions = useMemo(() => {
    const county = domainInputs.find((d) => d.name === "county");
    return CHAT_QUESTIONS.map((q) => {
      if (q.id !== "county") return q;
      return {
        ...q,
        options: (county?.allowed_codes ?? []).map((c) => ({
          value: c.code,
          label: c.label,
        })),
      };
    });
  }, [domainInputs]);

  const current = questions[qIndex] as ChatQuestion | undefined;
  const sectionIdx = current ? SECTIONS_ORDER.indexOf(current.section as (typeof SECTIONS_ORDER)[number]) : -1;
  const sectionCount = SECTIONS_ORDER.length;

  useEffect(() => {
    const el = messagesScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, busy, phase]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [info, m] = await Promise.all([getModelInfo(), getMetadata()]);
        if (cancelled) return;
        setDomainInputs(info.feature_schema.domain_inputs ?? []);
        setColours({ ...CLASS_COLOURS, ...(info.class_mapping.category_colours ?? {}) });
        setLabels({ ...DISPLAY_LABELS, ...(info.class_mapping.display_labels ?? {}) });
        setMeta({
          model_version: m.model_version,
          limitations: m.limitations,
          cv_metrics: m.cv_metrics,
          algorithm: m.algorithm,
        });
        setMessages((prev) => [
          ...prev,
          {
            role: "bot",
            content: questions[0]?.prompt ?? CHAT_QUESTIONS[0].prompt,
            meta: `Section 1/${SECTIONS_ORDER.length} · Question 1/${CHAT_QUESTIONS.length}`,
          },
        ]);
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : "Failed to load model info");
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function push(msg: Msg) {
    setMessages((m) => [...m, msg]);
  }

  function progressMeta(qi: number) {
    const q = questions[qi] ?? CHAT_QUESTIONS[qi];
    const sIdx = SECTIONS_ORDER.indexOf(q.section as (typeof SECTIONS_ORDER)[number]) + 1;
    return `Section ${sIdx}/${sectionCount} · Question ${qi + 1}/${questions.length}`;
  }

  function askQuestion(qi: number) {
    const q = questions[qi];
    if (!q) return;
    const selected = new Set<string>();
    if (q.kind === "multi") {
      for (const f of q.fields) {
        if (answers[f] === 1) selected.add(f);
      }
    }
    setMultiSel(selected);
    push({ role: "bot", content: q.prompt, meta: progressMeta(qi) });
  }

  function applyAnswerAndAdvance(nextAnswers: Partial<Record<FieldKey, number>>, userText: string) {
    setAnswers(nextAnswers);
    push({ role: "user", content: userText });
    const next = qIndex + 1;
    if (next >= questions.length) {
      setPhase("summary");
      push({
        role: "bot",
        content: "Here is a summary of your answers. Confirm to score, or go back to change one.",
      });
      return;
    }
    setQIndex(next);
    const q = questions[next];
    const selected = new Set<string>();
    if (q.kind === "multi") {
      for (const f of q.fields) {
        if (nextAnswers[f] === 1) selected.add(f);
      }
    }
    setMultiSel(selected);
    push({ role: "bot", content: q.prompt, meta: progressMeta(next) });
  }

  function handleBack() {
    setError(null);
    if (phase === "summary") {
      setPhase("interview");
      const last = questions.length - 1;
      setQIndex(last);
      askQuestion(last);
      return;
    }
    if (phase === "followup" || phase === "result") {
      setPhase("summary");
      push({ role: "bot", content: "Back to the answer summary. Confirm again to re-score, or go back further." });
      return;
    }
    if (qIndex <= 0) return;
    const prev = qIndex - 1;
    setQIndex(prev);
    askQuestion(prev);
  }

  function toggleMulti(field: string) {
    setMultiSel((prev) => {
      const n = new Set(prev);
      if (n.has(field)) n.delete(field);
      else n.add(field);
      return n;
    });
  }

  function submitMulti() {
    if (!current || current.kind !== "multi") return;
    const next = { ...answers };
    for (const f of current.fields) {
      next[f] = multiSel.has(f) ? 1 : 0;
    }
    const labels = current.fields
      .filter((f) => multiSel.has(f))
      .map((f) => MULTI_FIELD_LABELS[f] ?? f);
    const text = labels.length ? labels.join(", ") : "None of these";
    applyAnswerAndAdvance(next, text);
  }

  function submitChoice(value: number, label: string) {
    if (!current || current.kind !== "choice") return;
    const field = current.fields[0];
    applyAnswerAndAdvance({ ...answers, [field]: value }, label);
  }

  function submitNumber(raw: string) {
    if (!current || current.kind !== "number") return;
    const n = Number(raw.trim());
    if (Number.isNaN(n)) {
      push({ role: "bot", content: "Please enter a number." });
      return;
    }
    if (current.min !== undefined && n < current.min) {
      push({ role: "bot", content: `Please enter a value of at least ${current.min}.` });
      return;
    }
    if (current.max !== undefined && n > current.max) {
      push({ role: "bot", content: `Please enter a value of at most ${current.max}.` });
      return;
    }
    applyAnswerAndAdvance({ ...answers, [current.fields[0]]: n }, String(n));
  }

  function onSend() {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    setError(null);

    if (phase === "followup" || phase === "result") {
      void handleFollowUp(text);
      return;
    }
    if (phase === "summary") {
      push({ role: "user", content: text });
      push({ role: "bot", content: "Use Confirm to score, or Back to change an answer." });
      return;
    }
    if (!current) return;

    if (current.kind === "number") {
      submitNumber(text);
      return;
    }
    if (current.kind === "choice" && current.options?.length) {
      const parsed = parseChoice(text, current.options);
      if (parsed.kind === "none") {
        push({ role: "user", content: text });
        push({
          role: "bot",
          content:
            "I did not recognise that answer. Tap an option below, or type a clearer match (for example “second” for Secondary).",
        });
        return;
      }
      if (parsed.kind === "ambiguous") {
        push({ role: "user", content: text });
        push({
          role: "bot",
          content:
            `That could mean more than one option: ${parsed.candidates.map((c) => c.label).join(", ")}. ` +
            "Please tap one, or type the full label.",
        });
        return;
      }
      // Fuzzy match: show what was typed, confirm the recorded label, then advance.
      if (!parsed.exact) {
        push({ role: "user", content: text });
        push({ role: "bot", content: `Got it — recorded as “${parsed.label}”.` });
        const field = current.fields[0];
        const nextAnswers = { ...answers, [field]: parsed.value };
        setAnswers(nextAnswers);
        const next = qIndex + 1;
        if (next >= questions.length) {
          setPhase("summary");
          push({
            role: "bot",
            content: "Here is a summary of your answers. Confirm to score, or go back to change one.",
          });
          return;
        }
        setQIndex(next);
        const q = questions[next];
        const selected = new Set<string>();
        if (q.kind === "multi") {
          for (const f of q.fields) {
            if (nextAnswers[f] === 1) selected.add(f);
          }
        }
        setMultiSel(selected);
        push({ role: "bot", content: q.prompt, meta: progressMeta(next) });
        return;
      }
      submitChoice(parsed.value, parsed.label);
      return;
    }
    if (current.kind === "multi") {
      push({ role: "user", content: text });
      push({
        role: "bot",
        content: "For this question, tap the options below, then Done — or tap None of these.",
      });
    }
  }

  async function confirmScore() {
    const payload = buildPayload(answers);
    if (!payload) {
      setError("Some answers are still missing. Go back and complete every question.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const pred = await predict({ ...payload, include_segment: false });
      let exp: PredictResult = {
        ...pred,
        explanation_mode: "probability_only",
        shap_note: "Attribution not loaded yet.",
      };
      try {
        exp = await explain({ ...payload, top_k: 8 });
      } catch {
        exp = {
          ...pred,
          top_features: [],
          explanation_mode: "probability_only",
          shap_note:
            "Approximate attribution unavailable; showing model scores only. These are model scores, not calibrated probabilities.",
        };
      }
      setResult(exp);
      setPhase("followup");
      const scoreBits = CLASS_ORDER.map(
        (id) => `${labels[id] ?? id} ${(exp.scores[id] ?? 0).toFixed(2)}`,
      ).join(", ");
      push({
        role: "bot",
        content: `The model’s top score is for ${exp.display_label}. Scores: ${scoreBits}.`,
      });
      try {
        const chat = await predictionChatFollowUp(
          "In plain language, explain these model scores and among the main factors that moved the predicted class relative to a typical household. Keep it conversational.",
          exp,
        );
        const src = chat.source ?? (chat.used_llm ? "llm" : "template");
        push({
          role: "bot",
          content: chat.reply,
          meta:
            src === "llm"
              ? "Language model"
              : src.startsWith("template")
                ? "Template fallback (LLM unavailable or not grounded)"
                : src,
          source: src,
        });
      } catch {
        push({
          role: "bot",
          content:
            exp.shap_note ??
            "I could not load a plain-language explanation just now. You can still ask a follow-up about the scores.",
          meta: "Template fallback",
        });
      }
      push({
        role: "bot",
        content:
          "Ask why another category scored lower, or change an answer for a what-if and we will re-score. What-if answers show how the model responds to a changed answer — not what would happen to the household.",
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scoring failed");
    } finally {
      setBusy(false);
    }
  }

  async function handleFollowUp(text: string) {
    push({ role: "user", content: text });
    const lower = text.toLowerCase();
    if (lower.includes("what if") || lower.includes("if i change") || lower.includes("if we change")) {
      push({
        role: "bot",
        content:
          "To run a what-if, tap Back to the summary, change the relevant answer, and Confirm again. " +
          "I will not guess the new scores. What-if results show how the model responds to a changed answer, not what would happen to the household.",
      });
      return;
    }

    // why not <class> — load that class's attributions, then ask the LLM with precomputed gaps
    const whyNot = lower.match(/why not (personal|informal|formal|did.?nothing|no coping)/);
    if (whyNot && result) {
      const map: Record<string, ClassId> = {
        personal: "personal",
        informal: "informal",
        formal: "formal",
        "did nothing": "did_nothing",
        did_nothing: "did_nothing",
        "no coping": "did_nothing",
      };
      const key = whyNot[1].replace(/\s+/g, "_");
      const cls =
        map[whyNot[1]] ||
        map[key] ||
        (CLASS_ORDER.includes(whyNot[1] as ClassId) ? (whyNot[1] as ClassId) : null);
      if (cls) {
        setBusy(true);
        try {
          const payload = buildPayload(answers);
          if (!payload) throw new Error("Incomplete answers");
          const exp = await explain({ ...payload, top_k: 8, for_class: cls });
          setResult(exp);
          const res = await predictionChatFollowUp(text, exp);
          const src = res.source ?? (res.used_llm ? "llm" : "template");
          push({
            role: "bot",
            content: res.reply,
            meta:
              src === "llm"
                ? "Language model"
                : src.startsWith("template")
                  ? "Template fallback (LLM unavailable or not grounded)"
                  : src,
            source: src,
          });
        } catch (e) {
          push({
            role: "bot",
            content: e instanceof Error ? e.message : "Could not load class attribution.",
          });
        } finally {
          setBusy(false);
        }
        return;
      }
    }

    if (!result) {
      push({ role: "bot", content: "Score a household first, then ask a follow-up." });
      return;
    }
    setBusy(true);
    try {
      const res = await predictionChatFollowUp(text, result);
      const src = res.source ?? (res.used_llm ? "llm" : "template");
      push({
        role: "bot",
        content: res.reply,
        meta:
          src === "llm"
            ? "Language model"
            : src.startsWith("template")
              ? "Template fallback (LLM unavailable or not grounded)"
              : src === "redirect_overview"
                ? "Out of scope for Predictions"
                : src,
        source: src,
      });
    } catch {
      push({
        role: "bot",
        content:
          "I can discuss the stored model scores and approximate drivers, or you can open the Overview for survey patterns.",
      });
    } finally {
      setBusy(false);
    }
  }

  function summaryRows() {
    return questions.map((q) => {
      if (q.kind === "multi") {
        const on = q.fields.filter((f) => answers[f] === 1).map((f) => MULTI_FIELD_LABELS[f] ?? f);
        return { id: q.id, section: q.section, label: q.prompt, value: on.length ? on.join(", ") : "None of these" };
      }
      const f = q.fields[0];
      const v = answers[f];
      if (q.kind === "choice" && q.options) {
        const lab = q.options.find((o) => o.value === v)?.label ?? String(v);
        return { id: q.id, section: q.section, label: plainFieldLabel(f), value: lab };
      }
      return { id: q.id, section: q.section, label: plainFieldLabel(f), value: String(v) };
    });
  }

  if (loadError) {
    return <p className="banner-warn">{loadError}</p>;
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <div className="card-panel flex h-[min(78vh,820px)] min-h-[560px] flex-col overflow-hidden">
        <div className="mb-3 shrink-0 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--line)] pb-2">
          <div>
            <p className="m-0 text-sm font-medium">Prediction conversation</p>
            <p className="m-0 text-xs text-[var(--mute)]">
              {phase === "interview" && current
                ? progressMeta(qIndex)
                : phase === "summary"
                  ? "Summary"
                  : "Results & follow-up"}
            </p>
          </div>
          <div className="flex gap-2">
            <button type="button" className="chip" onClick={handleBack} disabled={busy || (phase === "interview" && qIndex === 0)}>
              Back
            </button>
            <button type="button" className="chip" onClick={() => setAboutOpen((v) => !v)}>
              About this model
            </button>
          </div>
        </div>

        {phase === "interview" && current && (
          <div className="mb-2 h-1.5 w-full shrink-0 overflow-hidden rounded bg-[var(--line)]">
            <div
              className="h-full bg-[var(--teal)]"
              style={{ width: `${((qIndex + 1) / questions.length) * 100}%` }}
            />
          </div>
        )}

        <div
          ref={messagesScrollRef}
          className="min-h-0 flex-1 space-y-2 overflow-y-auto overscroll-contain pr-1"
        >
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex w-full ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`w-fit max-w-[min(85%,28rem)] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap break-words ${
                  m.role === "user"
                    ? "bg-[var(--teal)] text-white rounded-br-md"
                    : "bg-[var(--paper)] text-[var(--ink)] border border-[var(--line)] rounded-bl-md"
                }`}
              >
                {m.meta ? (
                  <p className="mb-1 text-[10px] uppercase tracking-wide opacity-70">{m.meta}</p>
                ) : null}
                {m.content}
              </div>
            </div>
          ))}
          {busy ? (
            <div className="flex w-full justify-start">
              <div className="w-fit max-w-[min(85%,28rem)] rounded-2xl rounded-bl-md border border-[var(--line)] bg-[var(--paper)] px-3 py-2 text-sm text-[var(--mute)]">
                Thinking…
              </div>
            </div>
          ) : null}
          <div ref={messagesEndRef} className="h-px w-full shrink-0" aria-hidden />
        </div>

        <div className="mt-3 shrink-0 border-t border-[var(--line)] pt-3">
          {phase === "interview" && current?.kind === "choice" && current.options && (
            <div className="mb-2 flex flex-wrap gap-2">
              {current.options.map((o) => (
                <button key={o.value} type="button" className="chip" disabled={busy} onClick={() => submitChoice(o.value, o.label)}>
                  {o.label}
                </button>
              ))}
            </div>
          )}

          {phase === "interview" && current?.kind === "multi" && (
            <div className="mb-2 space-y-2">
              {current.help ? <p className="text-xs text-[var(--mute)]">{current.help}</p> : null}
              <div className="flex flex-wrap gap-2">
                {current.fields.map((f) => (
                  <button
                    key={f}
                    type="button"
                    className={`chip ${multiSel.has(f) ? "ring-2 ring-[var(--teal)]" : ""}`}
                    onClick={() => toggleMulti(f)}
                  >
                    {MULTI_FIELD_LABELS[f] ?? f}
                  </button>
                ))}
                <button
                  type="button"
                  className="chip"
                  onClick={() => {
                    setMultiSel(new Set());
                  }}
                >
                  None of these
                </button>
                <button type="button" className="chip bg-[var(--teal)] text-white" onClick={submitMulti}>
                  Done
                </button>
              </div>
            </div>
          )}

          {phase === "summary" && (
            <div className="mb-2 space-y-2">
              <ul className="m-0 max-h-36 list-none space-y-1 overflow-y-auto p-0 text-sm">
                {summaryRows().map((row) => (
                  <li key={row.id} className="border-b border-[var(--line)] py-1">
                    <span className="text-[var(--mute)]">{row.section}: </span>
                    <strong>{row.label}</strong> — {row.value}
                  </li>
                ))}
              </ul>
              <button type="button" className="chip bg-[var(--teal)] text-white" disabled={busy} onClick={() => void confirmScore()}>
                {busy ? "Scoring…" : "Confirm and score"}
              </button>
            </div>
          )}

          {(phase === "interview" && current?.kind !== "multi") || phase === "followup" || phase === "result" ? (
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                onSend();
              }}
            >
              <input
                className="flex-1 rounded border border-[var(--line)] bg-[var(--card)] px-3 py-2 text-sm"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder={phase === "followup" ? "Ask a follow-up…" : "Type your answer…"}
                disabled={busy}
              />
              <button type="submit" className="chip" disabled={busy || !draft.trim()}>
                Send
              </button>
            </form>
          ) : null}

          {error ? <p className="banner-warn mt-2 mb-0">{error}</p> : null}
        </div>
      </div>

      <aside className="card-panel text-sm">
        <p className="mt-0 font-medium">Coping routes</p>
        <ul className="m-0 list-none space-y-2 p-0">
          {CLASS_ORDER.map((id) => (
            <li key={id} className="flex items-center gap-2">
              <i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: colours[id] }} />
              {labels[id] ?? id}
              {result ? (
                <span className="ml-auto tabular-nums text-[var(--mute)]">
                  {(result.scores[id] ?? 0).toFixed(3)}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
        {result?.low_confidence ? (
          <p className="mt-3 text-xs text-[var(--mute)]">Low confidence flag on this score.</p>
        ) : null}
        <div className="mt-4 border-t border-[var(--line)] pt-3 text-xs text-[var(--mute)]">
          <p className="mt-0 font-medium text-[var(--ink)]">How reliable are these scores?</p>
          <p>
            Stack A shows only modest discrimination (macro F1 about 0.50 on a 5-fold check). Formal
            support is the weakest class (formal F1 about 0.31). Full per-class precision and recall
            were not stored for Stack A itself; a companion class-weighted logistic on the same 93
            features (same fold protocol) had roughly: personal P≈0.60 R≈0.39; informal P≈0.71
            R≈0.57; formal P≈0.22 R≈0.54; did nothing P≈0.47 R≈0.69 — a reminder of class imbalance,
            not a substitute for Stack A metrics. Scores are model scores, not calibrated
            probabilities, and a low-confidence flag appears when the top two scores are close or
            the prediction is formal. Use them to explore patterns — not to judge households.
          </p>
        </div>
        {aboutOpen && meta ? (
          <div className="mt-4 border-t border-[var(--line)] pt-3 text-xs text-[var(--mute)]">
            <p className="mt-0">
              {meta.algorithm} · {meta.model_version}
            </p>
            <p>{meta.limitations}</p>
          </div>
        ) : null}
        <p className="mt-4 text-xs text-[var(--mute)]">
          Figures and scores are for learning patterns. Not for targeting or eligibility decisions.
        </p>
      </aside>
    </div>
  );
}
