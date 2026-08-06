import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <header className="mb-4">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h2>
        {subtitle ? <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">{subtitle}</p> : null}
      </header>
      {children}
    </section>
  );
}
