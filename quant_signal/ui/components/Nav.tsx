import Link from "next/link";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/market", label: "Market" },
  { href: "/fundamentals", label: "Fundamentals" },
  { href: "/pead", label: "PEAD" },
  { href: "/macro", label: "Macro" },
  { href: "/pipeline", label: "Pipeline" },
];

export function Nav() {
  return (
    <header className="sticky top-0 z-10 border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-black/60">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
        <Link href="/" className="font-mono text-sm font-semibold tracking-tight">
          quant_signal
        </Link>
        <nav className="flex gap-1 overflow-x-auto">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="rounded-md px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100"
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
