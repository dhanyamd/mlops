"use client";

import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/Card";
import { LineChart } from "@/components/Charts";
import { Select } from "@/components/Select";
import { api } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

export default function MacroPage() {
  const [series, setSeries] = useState<string[]>([]);
  const [current, setCurrent] = useState("VIXCLS");

  useEffect(() => {
    api.macroSeries().then((s) => {
      setSeries(s.series);
      setCurrent((c) => (s.series.includes(c) ? c : (s.series[0] ?? "VIXCLS")));
    });
  }, []);

  const { data: points, error } = useQuery([current], async () =>
    (await api.macro(current)).points
  );

  const chart = useMemo(
    () =>
      (points ?? []).map((p) => ({
        DATE: p.DATE.slice(0, 10),
        VALUE: p.VALUE,
      })),
    [points]
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Macro</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">FRED series from SILVER_FRED_MACRO</p>
        </div>
        <Select label="Series" value={current} onChange={setCurrent} options={series.map((s) => ({ value: s, label: s }))} />
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      ) : null}

      <Card title={current} subtitle={points ? `${points.length} observations` : "…"}>
        {points === null ? (
          <p className="py-12 text-center text-sm text-zinc-500">Loading…</p>
        ) : (
          <LineChart data={chart} dataKey="VALUE" xKey="DATE" color="#f59e0b" />
        )}
      </Card>
    </div>
  );
}
