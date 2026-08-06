export function Select<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
      {label ? <span>{label}</span> : null}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
      >
        {options.map((o) => (
          <option key={String(o.value)} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
