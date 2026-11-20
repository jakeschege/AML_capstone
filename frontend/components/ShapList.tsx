"use client";

type Item = { name: string; shap: number; direction: string };

export function ShapList({ items, note }: { items: Item[]; note?: string }) {
  const max = Math.max(...items.map((i) => Math.abs(i.shap)), 1e-9);
  return (
    <div className="space-y-3">
      {note ? <p className="margin-note font-serif italic">{note}</p> : null}
      <ul className="space-y-2 font-plex text-sm">
        {items.map((item, idx) => {
          const width = `${(Math.abs(item.shap) / max) * 100}%`;
          const positive = item.shap >= 0;
          return (
            <li
              key={item.name}
              className="step-enter grid grid-cols-[1fr_auto] items-center gap-3"
              style={{ animationDelay: `${idx * 40}ms` }}
            >
              <div>
                <div className="mb-1 flex justify-between gap-2">
                  <span className="truncate">{item.name}</span>
                  <span className="tabular-nums text-umber">
                    {positive ? "+" : ""}
                    {item.shap.toFixed(3)}
                  </span>
                </div>
                <div className="relative h-2 w-full bg-rule/40">
                  <div
                    className="absolute top-0 h-2"
                    style={{
                      width,
                      left: positive ? "50%" : undefined,
                      right: positive ? undefined : "50%",
                      background: positive ? "var(--leaf)" : "var(--laterite)",
                    }}
                  />
                  <div className="absolute left-1/2 top-0 h-2 w-px bg-ink/40" />
                </div>
              </div>
              <span className="text-xs text-margin">{item.direction}</span>
            </li>
          );
        })}
      </ul>
      <p className="sr-only">
        Horizontal list of SHAP contributions with positive and negative directions.
      </p>
    </div>
  );
}
