import Link from "next/link";

const links = [
  { href: "/overview", label: "Overview" },
  { href: "/predictions", label: "Predictions" },
  { href: "/segmentation", label: "Typology" },
];

export function SiteHeader() {
  return (
    <header className="notebook-rule border-b border-rule bg-paper/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-end justify-between gap-4 px-4 py-4 md:px-8">
        <div className="flex items-start gap-3">
          <span
            aria-hidden
            className="mt-1 inline-block h-8 w-1.5 bg-laterite"
            title="Notebook margin mark"
          />
          <div>
            <p className="font-display text-xl tracking-tight text-ink md:text-2xl">
              FinAccess Coping Field Notebook
            </p>
            <p className="font-plex text-xs uppercase tracking-[0.14em] text-umber">
              Prototype · v0.1.0 · acres excluded
            </p>
          </div>
        </div>
        <nav className="hidden gap-4 font-plex text-sm md:flex">
          {links.map((l) => (
            <Link key={l.href} href={l.href} className="text-ink/80 hover:text-laterite">
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
