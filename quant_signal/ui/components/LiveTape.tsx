"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMarketStream } from "@/lib/useMarketStream";

function fmtPrice(v: number | null) {
  return v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/**
 * Live 1-minute tape straight off the Kafka stream (via the API's `/ws/market`
 * fan-out). Raw bars arrive every ~20s, so this chart re-forms continuously
 * while the 5-minute feature aggregates drive the rest of the page.
 */
export function LiveTape({
  symbol,
  height = 200,
}: {
  symbol: string;
  height?: number;
}) {
  const { bars, connected, retries } = useMarketStream(symbol);

  const last = bars[bars.length - 1] ?? null;
  const prev = bars[bars.length - 2] ?? null;
  const delta =
    last && prev && last.close !== prev.close ? last.close - prev.close : null;
  const deltaPct = delta != null && prev ? (delta / prev.close) * 100 : null;

  const series = useMemo(
    () =>
      bars.map((b) => ({
        ts: b.ts,
        close: b.close,
        label: new Date(b.ts).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      })),
    [bars]
  );

  return (
    <div className="w-full">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 font-mono text-sm">
          <span
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-emerald-500" : "bg-red-500"
            }`}
          />
          <span className="tabular-nums">
            {last ? fmtPrice(last.close) : "—"}
          </span>
          {delta != null ? (
            <span
              className={`tabular-nums text-xs ${
                delta >= 0 ? "text-emerald-500" : "text-red-500"
              }`}
            >
              {delta >= 0 ? "+" : ""}
              {deltaPct?.toFixed(3)}%
            </span>
          ) : null}
        </div>
        <div className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
          {connected
            ? `live · 1m bars · ${bars.length} in ring`
            : retries > 0
              ? `reconnecting (${retries})…`
              : "connecting…"}
        </div>
      </div>

      {series.length >= 2 ? (
        <div style={{ height }} className="w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                minTickGap={40}
              />
              <YAxis
                tick={{ fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={52}
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => fmtPrice(v)}
              />
              <Tooltip
                formatter={(v) => [fmtPrice(Number(v)), "close"]}
                labelFormatter={(label) => String(label)}
                contentStyle={{
                  background: "rgba(15,15,15,0.9)",
                  border: "none",
                  borderRadius: 8,
                  color: "#eee",
                  fontSize: 12,
                }}
              />
              <Line
                type="monotone"
                dataKey="close"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="grid place-items-center text-sm text-zinc-500" style={{ height }}>
          Waiting for live bars…
        </div>
      )}

      <div className="mt-1 grid grid-cols-4 gap-2 text-[11px] text-zinc-500 dark:text-zinc-400">
        {(["open", "high", "low", "volume"] as const).map((k) => (
          <div key={k} className="truncate font-mono tabular-nums">
            {k} {last ? (k === "volume" ? last[k].toFixed(4) : fmtPrice(last[k])) : "—"}
          </div>
        ))}
      </div>
    </div>
  );
}
