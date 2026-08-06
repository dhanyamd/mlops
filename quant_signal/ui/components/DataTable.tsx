"use client";

export function fmt(value: unknown): string {
  if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value)))
    return "—";
  if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(value);
}

export function DataTable({ columns, rows }: { columns: { key: string; label: string }[]; rows: Record<string, unknown>[] }) {
  if (rows.length === 0) {
    return <p className="py-8 text-center text-sm text-zinc-500">No data.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-zinc-200 dark:border-zinc-700">
            {columns.map((c) => (
              <th key={c.key} className="px-3 py-2 font-mono text-xs font-medium text-zinc-500 dark:text-zinc-400">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className="border-b border-zinc-100 dark:border-zinc-800/60 last:border-0 hover:bg-zinc-50 dark:hover:bg-zinc-800/40"
            >
              {columns.map((c) => (
                <td key={c.key} className="px-3 py-2 tabular-nums text-zinc-700 dark:text-zinc-300">
                  {fmt(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
