"use client";

import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/Card";
import { Bars, LineChart } from "@/components/Charts";
import { Select } from "@/components/Select";
import { api } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

const DAY_OPTIONS = [252, 750, 1500, 3000, 8000];

export default function MarketPage() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("AAPL");
  const [days, setDays] = useState(750);

  useEffect(() => {
    api.tickers().then((t) => {
      setTickers(t.tickers);
      setSymbol((s) => (t.tickers.includes(s) ? s : (t.tickers[0] ?? s)));
    });
  }, []);

  const { data: bars, error } = useQuery([symbol, days], async () =>
    (await api.market(symbol, days)).bars
  );

  const closeSeries = useMemo(
    () =>
      (bars ?? []).map((b) => ({
        TRADE_DATE: b.TRADE_DATE.slice(0, 10),
        DAY_CLOSE: b.DAY_CLOSE,
      })),
    [bars]
  );

  const volumeSeries = useMemo(
    () =>
      (bars ?? []).map((b) => ({
        TRADE_DATE: b.TRADE_DATE.slice(0, 10),
        VOLUME: b.VOLUME,
      })),
    [bars]
  );

  const last = bars?.[bars.length - 1];
  const first = bars?.[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Market</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Daily OHLCV from GOLD_DAILY_BARS
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <Select
            label="Symbol"
            value={symbol}
            onChange={setSymbol}
            options={tickers.map((t) => ({ value: t, label: t }))}
          />
          <Select
            label="Days"
            value={days}
            onChange={setDays}
            options={DAY_OPTIONS.map((d) => ({ value: d, label: String(d) }))}
          />
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card
            title={`${symbol} — close`}
            subtitle={last ? `${first?.TRADE_DATE.slice(0, 10)} → ${last.TRADE_DATE.slice(0, 10)}` : "…"}
          >
            {bars === null ? <p className="py-12 text-center text-sm text-zinc-500">Loading…</p> : (
              <LineChart data={closeSeries} dataKey="DAY_CLOSE" xKey="TRADE_DATE" />
            )}
          </Card>
        </div>
        <div>
          <Card title="Volume">
            {bars === null ? <p className="py-12 text-center text-sm text-zinc-500">Loading…</p> : (
              <Bars data={volumeSeries} dataKey="VOLUME" xKey="TRADE_DATE" height={320} />
            )}
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-6">
        {(["DAY_OPEN", "DAY_HIGH", "DAY_LOW", "DAY_CLOSE", "VOLUME"] as const).map((k) => (
          <div key={k} className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="text-xs text-zinc-500 dark:text-zinc-400">{k}</div>
            <div className="mt-1 font-mono text-lg tabular-nums">
              {last?.[k] === null || last?.[k] === undefined
                ? "—"
                : Number(last?.[k]).toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
