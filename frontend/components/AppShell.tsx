"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "@/lib/constants";

const TITLES: Record<string, { h1: string; lede: string }> = {
  "/overview": {
    h1: "Shock-exposed farming households",
    lede:
      "How many shock-exposed households are represented by the survey, and how they coped — weighted within the FinAccess 2024 all-shocks analysis sample.",
  },
  "/predictions": {
    h1: "Predict a coping route",
    lede:
      "Answer one question at a time, then see model scores and an approximate explanation. For learning patterns, not targeting.",
  },
};

export function AppShell({
  children,
  stamp,
}: {
  children: React.ReactNode;
  stamp?: string;
}) {
  const path = usePathname() || "/overview";
  const meta = TITLES[path] ?? TITLES["/overview"];

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--card)]">
        <div className="shell flex min-h-16 flex-wrap items-center justify-between gap-4">
          <div>
            <p className="font-display text-[17px] font-bold text-[var(--teal)]">
              FinAccess Coping Notebook
            </p>
            <p className="text-xs text-[var(--mute)]">
              Shock-exposed farming households in Kenya
            </p>
          </div>
          <nav aria-label="Primary" className="flex gap-1">
            {NAV.map((item) => {
              const on = path === item.href || (item.href === "/overview" && path === "/");
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`border-b-[3px] px-3.5 py-4 text-sm font-medium font-display ${
                    on
                      ? "border-[var(--teal)] text-[var(--teal)]"
                      : "border-transparent text-[var(--ink)]"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="shell pb-10">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-4 border-b border-[var(--line)] pb-2.5 pt-6">
          <div>
            <h1 className="m-0 text-[clamp(26px,4vw,40px)] font-bold leading-tight tracking-[-0.02em]">
              {meta.h1}
            </h1>
            <p className="mt-2 max-w-[62ch] text-[var(--mute)]">{meta.lede}</p>
          </div>
          {stamp ? <span className="chip">{stamp}</span> : null}
        </div>
        {children}
      </main>

      <footer className="shell max-w-[80ch] pb-10 pt-8 text-[12.5px] text-[var(--mute)]">
        Figures are survey estimates, weighted so that each interviewed household stands for similar
        households represented by the survey within this shock-exposed analysis sample. They show
        patterns across groups. They do not explain why any one household coped the way it did, and
        they are not for targeting or eligibility decisions.
      </footer>
    </>
  );
}
