"use client";

type Props = {
  scores: Record<string, number>;
  prediction: string;
  colours: Record<string, string>;
  labels: Record<string, string>;
};

export function LedgerBar({ scores, prediction, colours, labels }: Props) {
  const entries = Object.entries(scores);
  return (
    <div className="space-y-3" role="img" aria-label="Class probability ledger">
      <div className="flex h-4 w-full overflow-hidden border border-rule bg-paper">
        {entries.map(([cls, p]) => (
          <div
            key={cls}
            style={{
              width: `${Math.max(p * 100, 0)}%`,
              background: colours[cls] ?? "#7A6F64",
              opacity: cls === prediction ? 1 : 0.55,
              transition: "width 400ms ease-out",
            }}
            title={`${labels[cls] ?? cls}: ${(p * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      <ul className="grid gap-2 font-plex text-sm sm:grid-cols-2">
        {entries.map(([cls, p]) => (
          <li
            key={cls}
            className={`flex items-center justify-between border-b border-rule/70 pb-1 ${
              cls === prediction ? "font-semibold" : ""
            }`}
          >
            <span className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5"
                style={{
                  background: colours[cls],
                  backgroundImage:
                    cls === "formal"
                      ? "repeating-linear-gradient(45deg,#fff0,#fff0 1px,#fff6 1px,#fff6 2px)"
                      : undefined,
                }}
                aria-hidden
              />
              {labels[cls] ?? cls}
              {cls === prediction ? " · predicted" : ""}
            </span>
            <span>{(p * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
