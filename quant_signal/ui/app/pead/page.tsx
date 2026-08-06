"use client";

import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/Card";
import { Bars } from "@/components/Charts";
import { DataTable } from "@/components/DataTable";
import { Select } from "@/components/Select";
import { api } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

const WINDOW_OPTIONS = ["1,5,20", "1,5", "5,20", "1,20"];

export default function PeadPage() {
  const [metrics, setMetrics] = useState<string[]>([]);
  const [metric, setMetric] = useState("NetIncomeLoss");
  const [windows, setWindows] = useState("1,5,20");

  useEffect(() => {
    api.metrics().then((m) => {
      setMetrics(m.metrics);
      if (!m.metrics.includes("NetIncomeLoss")) setMetric(m.metrics[0] ?? "NetIncomeLoss");
    });
  }, []);

  const { data: result, error } = useQuery([metric, windows], () =>
    api.pead(metric, windows)
  );

  const windowList = useMemo(() => windows.split(",").map(Number), [windows]);

  const tableColumns = useMemo(
    () => [
      { key: "quintile", label: "Quintile" },
      { key: "n", label: "n" },
      { key: "mean_sue", label: "Mean SUE" },
      ...windowList.flatMap((h) => [
        { key: `car${h}`, label: `CAR +${h}d` },
        { key: `t${h}`, label: `t(+${h}d)` },
      ]),
    ],
    [windowList]
  );

  const chartSeries = useMemo(() => {
    if (!result) return [];
    return windowList.map((h) => ({
      name: `+${h}d`,
      ...Object.fromEntries(
        result.quintiles.map((q) => [q.quintile, q[`car${h}`] ?? 0])
      ),
    }));
  }, [result, windowList]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">PEAD event study</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Post-earnings drift by SUE quintile — no-lookahead, filed_at-based
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <Select label="Metric" value={metric} onChange={setMetric} options={metrics.map((m) => ({ value: m, label: m }))} />
          <Select label="Windows" value={windows} onChange={setWindows} options={WINDOW_OPTIONS.map((w) => ({ value: w, label: w }))} />
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      ) : null}

      {result ? (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { label: "Events", value: result.n_events },
              { label: "Tickers", value: result.n_tickers },
              { label: "Unlabeled (early)", value: result.unlabeled },
              { label: "Metric", value: result.metric },
            ].map((s) => (
              <div key={s.label} className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
                <div className="text-xs text-zinc-500 dark:text-zinc-400">{s.label}</div>
                <div className="mt-1 font-mono text-lg tabular-nums">{s.value}</div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Mean CAR by SUE quintile" subtitle="Bars are window; colors are quintile">
              <Bars data={chartSeries} dataKey="Q1" height={320} />
            </Card>

            <Card title="Quintile drift table" subtitle="CAR = ticker buy-hold − equal-weight universe">
              <DataTable columns={tableColumns} rows={result.quintiles} />
            </Card>
          </div>

          <Card title="PEAD spread (highest minus lowest SUE quintile)" subtitle="Fractional CAR, per window">
            <div className="flex flex-wrap gap-6">
              {Object.entries(result.spread).map(([window, spread]) => (
                <div key={window} className="min-w-[140px]">
                  <div className="text-xs text-zinc-500 dark:text-zinc-400">Window {window}</div>
                  <div
                    className={`mt-1 font-mono text-xl tabular-nums ${
                      spread === null ? "text-zinc-400" : spread >= 0 ? "text-emerald-500" : "text-red-500"
                    }`}
                  >
                    {spread === null ? "—" : `${(spread * 100).toFixed(2)}%`}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      ) : (
        <p className="py-16 text-center text-sm text-zinc-500">
          {error ? "Failed to load." : "Loading PEAD study (~10s query)…"}
        </p>
      )}
    </div>
  );
}
