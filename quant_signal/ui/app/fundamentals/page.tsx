"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/Card";
import { DataTable } from "@/components/DataTable";
import { Select } from "@/components/Select";
import { api } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

export default function FundamentalsPage() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("AAPL");
  const [metric, setMetric] = useState("NetIncomeLoss");

  useEffect(() => {
    Promise.all([api.tickers(), api.metrics()]).then(([t, m]) => {
      setTickers(t.tickers);
      setMetrics(m.metrics);
      setSymbol((s) => (t.tickers.includes(s) ? s : (t.tickers[0] ?? s)));
      if (!m.metrics.includes("NetIncomeLoss")) setMetric(m.metrics[0] ?? "NetIncomeLoss");
    });
  }, []);

  const { data: facts, error } = useQuery([symbol, metric], async () =>
    (await api.fundamentals(symbol, metric)).facts
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Fundamentals</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Point-in-time US-GAAP facts as filed — FILED_AT is the as-known order
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <Select label="Ticker" value={symbol} onChange={setSymbol} options={tickers.map((t) => ({ value: t, label: t }))} />
          <Select label="Metric" value={metric} onChange={setMetric} options={metrics.map((m) => ({ value: m, label: m }))} />
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      ) : null}

      <Card title={`${symbol} — ${metric}`} subtitle={facts ? `${facts.length} filings` : "…"}>
        {facts === null ? (
          <p className="py-12 text-center text-sm text-zinc-500">Loading…</p>
        ) : (
          <DataTable
            columns={[
              { key: "FISCAL_YEAR", label: "Fiscal year" },
              { key: "VALUE", label: "Value" },
              { key: "UNIT", label: "Unit" },
              { key: "FILED_AT", label: "Filed at" },
            ]}
            rows={facts}
          />
        )}
      </Card>
    </div>
  );
}
