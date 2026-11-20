"use client";

import { useCallback, useEffect, useState } from "react";
import { CLASS_COLOURS, CLASS_ORDER, DISPLAY_LABELS, type ClassId } from "@/lib/constants";
import {
  fetchDashboard,
  type DashboardPayload,
  type MainShockRow,
  type ShockOption,
} from "@/lib/dashboard";
import {
  buildScaleSentence,
  buildTakeawaySentence,
  formatPct,
  formatUnweightedN,
  formatWeightedCount,
} from "@/lib/format";

function Dot({ colour }: { colour: string }) {
  return (
    <i
      className="inline-block h-2.5 w-2.5 rounded-full"
      style={{ background: colour }}
      aria-hidden
    />
  );
}

function StackedMix({
  mix,
  colours,
}: {
  mix: number[] | Record<string, number>;
  colours: Record<string, string>;
}) {
  const values = Array.isArray(mix)
    ? mix
    : CLASS_ORDER.map((id) => (mix as Record<string, number>)[id] ?? 0);
  return (
    <div className="track" role="img" aria-label="Coping mix stacked bar">
      {CLASS_ORDER.map((id, i) => (
        <span
          key={id}
          className="track-seg"
          style={{
            width: `${(values[i] ?? 0) * 100}%`,
            background: colours[id] ?? CLASS_COLOURS[id],
          }}
        />
      ))}
    </div>
  );
}

function Donut({ mix, colours }: { mix: Record<ClassId, number>; colours: Record<string, string> }) {
  const r = 15.9;
  const c = 2 * Math.PI * r;
  let off = 0;
  const arcs = CLASS_ORDER.map((k) => {
    const len = (mix[k] ?? 0) * c;
    const el = (
      <circle
        key={k}
        r={r}
        cx={21}
        cy={21}
        fill="none"
        stroke={colours[k] ?? CLASS_COLOURS[k]}
        strokeWidth={6}
        strokeDasharray={`${len} ${c - len}`}
        strokeDashoffset={-off}
        transform="rotate(-90 21 21)"
      />
    );
    off += len;
    return el;
  });
  const none = mix.did_nothing ?? 0;
  return (
    <svg width={170} height={170} viewBox="0 0 42 42" role="img" aria-label="Coping route shares">
      {arcs}
      <text
        x={21}
        y={21}
        textAnchor="middle"
        fontSize={7}
        fontWeight={700}
        fill="currentColor"
        fontFamily="var(--font-poppins)"
      >
        {formatPct(none)}
      </text>
      <text x={21} y={26.5} textAnchor="middle" fontSize={2.5} fill="currentColor">
        no coping action
      </text>
    </svg>
  );
}

function ShockFilterBar({
  options,
  selected,
  onSelect,
  busy,
}: {
  options: ShockOption[];
  selected: string;
  onSelect: (key: string) => void;
  busy: boolean;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="sec-title m-0">Shock filter</h2>
        <span className="text-[12.5px] text-[var(--mute)]">
          Any reported shock · swipe on small screens
        </span>
      </div>
      <div className="shock-filter-row" role="tablist" aria-label="Shock filter">
        {options.map((opt) => {
          const on = opt.key === selected;
          const disabled = Boolean(opt.disabled) || busy;
          return (
            <button
              key={opt.key}
              type="button"
              role="tab"
              aria-selected={on}
              disabled={disabled}
              title={`${opt.label} · n = ${opt.n}`}
              className={`chip ${on ? "chip-active" : ""} ${opt.disabled ? "chip-disabled" : ""}`}
              onClick={() => onSelect(opt.key)}
            >
              {opt.label}
              <span className="ml-1 opacity-70">({opt.n.toLocaleString("en-KE")})</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function MainShockChart({
  rows,
  colours,
  labels,
  note,
}: {
  rows: MainShockRow[];
  colours: Record<string, string>;
  labels: Record<string, string>;
  note?: string;
}) {
  return (
    <article className="card-panel">
      <h3 className="m-0 text-sm uppercase tracking-wide text-[var(--teal)]">
        Coping by main shock
      </h3>
      <p className="mb-3 mt-1 text-[13px] text-[var(--mute)]">
        Uses the household’s <strong>main reported shock</strong> (R2B), not the any-shock filter.
        All main-shock groups are listed; the selected Overview shock is highlighted when it maps
        to a main-shock code.
      </p>
      {note ? <p className="mb-3 text-[12.5px] text-[var(--mute)]">{note}</p> : null}
      <div className="mb-2 flex flex-wrap gap-3 text-[12.5px] text-[var(--mute)]">
        {CLASS_ORDER.map((id) => (
          <span key={id} className="flex items-center gap-1">
            <Dot colour={colours[id]} /> {labels[id]}
          </span>
        ))}
      </div>
      <div className="grid gap-2">
        {rows.map((row) => (
          <div
            key={row.main_shock_code}
            className={`grid grid-cols-[minmax(110px,160px)_1fr_56px] items-center gap-2 rounded px-1 py-1 text-[13px] ${
              row.unavailable ? "opacity-45" : ""
            }`}
            style={
              row.highlighted
                ? { background: "color-mix(in srgb, var(--teal) 10%, transparent)" }
                : undefined
            }
            title={
              row.unavailable
                ? row.unavailable_reason ?? "Too few households"
                : `n = ${row.n}${row.small_sample ? " · small sample" : ""}`
            }
          >
            <span>
              {row.label}
              {row.highlighted ? " · selected" : ""}
              {row.unavailable ? " · unavailable" : ""}
              {row.small_sample && !row.unavailable ? " · small sample" : ""}
            </span>
            {row.unavailable ? (
              <span className="text-[12px] text-[var(--mute)]">
                {row.unavailable_reason ?? `Fewer than 30 households (n=${row.n}).`}
              </span>
            ) : (
              <StackedMix mix={row.mix} colours={colours} />
            )}
            <b className="text-right font-display text-[12.5px]">
              {row.unavailable ? "—" : formatUnweightedN(row.n).replace("n = ", "")}
            </b>
          </div>
        ))}
      </div>
    </article>
  );
}

export function OverviewView() {
  const [shock, setShock] = useState("all");
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (key: string, initial: boolean) => {
    if (initial) setLoading(true);
    else setRefreshing(true);
    try {
      const d = await fetchDashboard(key);
      setData(d);
      setError(null);
      setShock(d.selected_shock ?? key);
    } catch {
      setData(null);
      setError(
        "The model service is not running, or the dashboard could not be loaded. Start the API and try again. No sample figures are shown.",
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void load("all", true);
  }, [load]);

  if (loading) {
    return (
      <div className="grid gap-4" aria-busy="true">
        <div className="card-panel h-24 animate-pulse bg-[var(--line)]/40" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="card-panel h-28 animate-pulse bg-[var(--line)]/40" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="banner-warn" role="alert">
        {error ?? "Dashboard unavailable."}
      </div>
    );
  }

  const colours = { ...CLASS_COLOURS, ...data.category_colours };
  const labels = { ...DISPLAY_LABELS, ...data.display_labels };
  const group = data.takeaway.highest_none_group;
  const options = data.shock_options ?? [{ key: "all", label: "All shocks", n: data.n_respondents }];
  const shockLabel = data.shock_label ?? options.find((o) => o.key === shock)?.label ?? "All shocks";
  const suppressed = data.suppressed_cells;

  return (
    <div className={`space-y-6 ${refreshing ? "opacity-80" : ""}`}>
      <div className="flex flex-wrap items-center gap-2">
        <p className="chip m-0 inline-block">
          Updated {data.updated} · wave {data.wave}
        </p>
        {data.small_sample ? (
          <p className="chip m-0 inline-block border-[var(--informal)] text-[var(--informal)]">
            Small sample (n &lt; {data.small_sample_n ?? 100})
          </p>
        ) : null}
      </div>

      <ShockFilterBar
        options={options}
        selected={shock}
        busy={refreshing}
        onSelect={(key) => {
          setShock(key);
          void load(key, false);
        }}
      />

      <p className="text-[13px] text-[var(--mute)]">
        {data.filter_note ??
          "Filter uses any reported shock. Households can appear under more than one shock. Coping route is the household’s overall route, not specific to the selected shock."}
      </p>

      <div className="card-panel grid gap-3 sm:grid-cols-[auto_1fr] sm:items-center">
        <div className="num text-[30px] font-bold text-[var(--formal)]">
          {formatWeightedCount(data.affected_weighted)}
        </div>
        <p>{buildScaleSentence(data.affected_weighted, data.n_respondents, shockLabel)}</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        <article className="card-panel">
          <div className="flex items-center gap-2 text-xs font-semibold text-[var(--mute)]">
            <Dot colour="var(--teal)" /> Households in this slice
          </div>
          <div className="num mt-1 text-[32px] font-bold text-[var(--teal)]">
            {formatWeightedCount(data.affected_weighted)}
          </div>
          <p className="text-[13px] text-[var(--mute)]">
            Represented by the survey · {formatUnweightedN(data.affected_unweighted)}
          </p>
        </article>
        {CLASS_ORDER.map((id) => (
          <article key={id} className="card-panel">
            <div className="flex items-center gap-2 text-xs font-semibold text-[var(--mute)]">
              <Dot colour={colours[id]} /> {labels[id]}
            </div>
            <div className="num mt-1 text-[32px] font-bold text-[var(--teal)]">
              {formatWeightedCount(data.mix_weighted[id])}
            </div>
            <p className="text-[13px] text-[var(--mute)]">
              {formatPct(data.mix[id])} of slice · {formatUnweightedN(data.mix_unweighted[id])}
            </p>
          </article>
        ))}
      </div>

      <div className="flex flex-wrap items-baseline justify-between gap-2 pt-2">
        <h2 className="sec-title">How households coped</h2>
        <span className="text-[13px] text-[var(--mute)]">
          Overall coping route · weighted within this shock slice
        </span>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <article className="card-panel">
          <h3 className="m-0 text-sm uppercase tracking-wide text-[var(--teal)]">Coping mix</h3>
          <p className="mb-3 mt-1 text-[13px] text-[var(--mute)]">
            Share of households in this slice by overall coping route — not a route specific to the
            selected shock alone.
          </p>
          <div className="flex flex-wrap items-center gap-5">
            <Donut mix={data.mix} colours={colours} />
            <ul className="min-w-[200px] flex-1 list-none space-y-0 p-0">
              {CLASS_ORDER.map((id) => (
                <li
                  key={id}
                  className="flex justify-between gap-3 border-b border-[var(--line)] py-1.5 text-sm last:border-0"
                >
                  <span className="flex items-center gap-2">
                    <Dot colour={colours[id]} /> {labels[id]}
                  </span>
                  <b className="font-display">{formatPct(data.mix[id])}</b>
                </li>
              ))}
            </ul>
          </div>
        </article>

        <article className="card-panel">
          <h3 className="m-0 text-sm uppercase tracking-wide text-[var(--teal)]">
            Where it is happening
          </h3>
          <p className="mb-3 mt-1 text-[13px] text-[var(--mute)]">{data.geography_note}</p>
          {suppressed ? (
            <p className="mb-2 text-[12px] text-[var(--mute)]">
              Counties suppressed (n &lt; {data.min_cell_n}):{" "}
              {suppressed.counties_suppressed_n_lt_30} of {suppressed.counties_total}
              {"; "}
              small-sample flag (n &lt; {data.small_sample_n ?? 100}):{" "}
              {suppressed.counties_small_sample_n_lt_100}
            </p>
          ) : null}
          <div className="mb-2 flex flex-wrap gap-3 text-[12.5px] text-[var(--mute)]">
            {CLASS_ORDER.map((id) => (
              <span key={id} className="flex items-center gap-1">
                <Dot colour={colours[id]} /> {labels[id]}
              </span>
            ))}
          </div>
          <div className="grid gap-2">
            {data.counties.map((c) => {
              const maxW = Math.max(...data.counties.map((x) => x.affected_weighted), 1);
              const width = Math.max(30, (c.affected_weighted / maxW) * 100);
              return (
                <div
                  key={c.name}
                  className="grid grid-cols-[minmax(90px,120px)_1fr_64px] items-center gap-2 text-[13px]"
                  title={`Unweighted respondents: ${c.n}${c.small_sample ? " · small sample" : ""}`}
                >
                  <span>
                    {c.name}
                    {c.small_sample ? " · small" : ""}
                  </span>
                  <div className="track" style={{ width: `${width}%`, minWidth: "30%" }}>
                    {CLASS_ORDER.map((id, i) => (
                      <span
                        key={id}
                        className="track-seg"
                        style={{
                          width: `${(c.mix[i] ?? 0) * 100}%`,
                          background: colours[id],
                        }}
                      />
                    ))}
                  </div>
                  <b className="text-right font-display text-[12.5px]">
                    {formatWeightedCount(c.affected_weighted)}
                  </b>
                </div>
              );
            })}
          </div>
        </article>
      </div>

      {data.coping_by_main_shock?.length ? (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-2 pt-2">
            <h2 className="sec-title">Coping by main shock</h2>
            <span className="text-[13px] text-[var(--mute)]">Main shock (R2B) · all groups shown</span>
          </div>
          <MainShockChart
            rows={data.coping_by_main_shock}
            colours={colours}
            labels={labels}
            note={data.coping_by_main_shock_note}
          />
        </>
      ) : null}

      <div className="flex flex-wrap items-baseline justify-between gap-2 pt-2">
        <h2 className="sec-title">Farm household segments</h2>
        <span className="text-[13px] text-[var(--mute)]">Spec A typology · k = 5 · descriptive only</span>
      </div>
      {suppressed ? (
        <p className="text-[12px] text-[var(--mute)]">
          Segments suppressed (n &lt; {data.min_cell_n}): {suppressed.segments_suppressed_n_lt_30} of{" "}
          {suppressed.segments_total}
          {"; "}
          small-sample flag (n &lt; {data.small_sample_n ?? 100}):{" "}
          {suppressed.segments_small_sample_n_lt_100}
        </p>
      ) : null}
      <div className="grid gap-3 md:grid-cols-2">
        {data.segments
          .filter((s) => !s.suppressed)
          .map((s) => (
            <article key={s.id} className="card-panel">
              <h3 className="m-0 text-sm text-[var(--teal)]">
                {s.name}
                {s.small_sample ? " · small sample" : ""}
              </h3>
              <p className="mb-2 mt-1 text-[13px] text-[var(--mute)]">
                {formatWeightedCount(s.size_weighted)} represented by the survey ·{" "}
                {formatPct(s.size_share)} of slice · {formatUnweightedN(s.n)}
              </p>
              <StackedMix mix={s.mix} colours={colours} />
            </article>
          ))}
      </div>

      <div className="flex flex-wrap items-baseline justify-between gap-2 pt-2">
        <h2 className="sec-title">Who is left without a reported coping action</h2>
        <span className="text-[13px] text-[var(--mute)]">
          Share with “{labels.did_nothing}” in each group (n ≥ {data.min_cell_n})
        </span>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(data.who).map(([panel, rows]) => (
          <article key={panel} className="card-panel">
            <h3 className="m-0 text-sm uppercase tracking-wide text-[var(--teal)]">{panel}</h3>
            <div className="mt-3 grid gap-2">
              {rows.map((r) => (
                <div
                  key={r.label}
                  className={`grid grid-cols-[minmax(90px,1fr)_1fr_52px] items-center gap-2 text-[13px] ${
                    r.suppressed ? "opacity-45" : ""
                  }`}
                >
                  <span>
                    {r.label}
                    {r.suppressed ? " · too few" : ""}
                    {r.small_sample && !r.suppressed ? " · small sample" : ""}
                  </span>
                  <div className="track">
                    {!r.suppressed && r.none_share != null ? (
                      <span
                        className="track-seg"
                        style={{
                          width: `${r.none_share * 100}%`,
                          background: colours.did_nothing,
                        }}
                      />
                    ) : null}
                  </div>
                  <b className="text-right font-display text-[12.5px]">
                    {r.suppressed || r.none_share == null ? "—" : formatPct(r.none_share)}
                  </b>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>

      <div className="flex flex-wrap items-baseline justify-between gap-2 pt-2">
        <h2 className="sec-title">Other farm challenges</h2>
        <span className="text-[13px] text-[var(--mute)]">
          Reported in this slice · weighted share · not effects of the selected shock
        </span>
      </div>
      <article className="card-panel">
        <div className="grid gap-2">
          {data.challenges.map((ch) => (
            <div
              key={ch.label}
              className="grid grid-cols-[minmax(120px,1fr)_1fr_52px] items-center gap-2 text-[13px]"
              title={formatUnweightedN(ch.n)}
            >
              <span>
                {ch.label}
                {ch.small_sample ? " · small sample" : ""}
              </span>
              <div className="track">
                <span
                  className="track-seg"
                  style={{ width: `${ch.share * 100}%`, background: "var(--teal)" }}
                />
              </div>
              <b className="text-right font-display text-[12.5px]">{formatPct(ch.share)}</b>
            </div>
          ))}
        </div>
      </article>

      {data.financial_access?.length ? (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-2 pt-2">
            <h2 className="sec-title">Financial access patterns</h2>
            <span className="text-[13px] text-[var(--mute)]">
              Group patterns in this slice · not effects of a shock
            </span>
          </div>
          <article className="card-panel">
            <div className="grid gap-2">
              {data.financial_access
                .filter((f) => !f.suppressed)
                .slice(0, 10)
                .map((f) => (
                  <div
                    key={f.key}
                    className="grid grid-cols-[minmax(120px,1fr)_1fr_52px] items-center gap-2 text-[13px]"
                    title={formatUnweightedN(f.n)}
                  >
                    <span className="capitalize">
                      {f.label}
                      {f.small_sample ? " · small sample" : ""}
                    </span>
                    <div className="track">
                      <span
                        className="track-seg"
                        style={{ width: `${f.share * 100}%`, background: "var(--formal)" }}
                      />
                    </div>
                    <b className="text-right font-display text-[12.5px]">{formatPct(f.share)}</b>
                  </div>
                ))}
            </div>
          </article>
        </>
      ) : null}

      <div className="take">
        <strong className="mb-1 block font-display">What this means</strong>
        {buildTakeawaySentence({
          noneWeighted: data.takeaway.none_weighted,
          noneShare: data.takeaway.none_share,
          groupLabel: group?.label,
          groupPanel: group?.panel,
          groupShare: group?.none_share,
        })}
      </div>

      <p className="border-t border-[var(--line)] pt-4 text-[12.5px] text-[var(--mute)]">
        Source: FinAccess 2024 analysis sample · weight: {data.weight_variable} ·{" "}
        {formatUnweightedN(data.n_respondents)}. {data.weight_note} {data.methods_note}
      </p>
    </div>
  );
}
