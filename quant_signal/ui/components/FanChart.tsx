"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function fmt(v: number) {
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

/**
 * Monte Carlo forward fan chart: 10–90 and 25–75 percentile bands drawn as
 * stacked areas (the quantpad-style "10k paths at once" view) around the
 * median path.
 */
export function FanChart({
  percentiles,
  medianPath,
  horizonSteps,
  height = 320,
}: {
  percentiles: Record<string, number[]>;
  medianPath: number[];
  horizonSteps: number;
  height?: number;
}) {
  const data = Array.from({ length: horizonSteps + 1 }, (_, step) => {
    const p10 = percentiles["10"][step];
    const p25 = percentiles["25"][step];
    const p75 = percentiles["75"][step];
    const p90 = percentiles["90"][step];
    return {
      step,
      p10,
      p25,
      p50: medianPath[step],
      p75,
      p90,
      outer: p90 - p10,
      inner: p75 - p25,
    };
  });

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="fan-outer" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.03} />
            </linearGradient>
            <linearGradient id="fan-inner" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.32} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.08} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis dataKey="step" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={72}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => fmt(v)}
          />
          <Tooltip
            labelFormatter={(label) => `step ${String(label)}`}
            formatter={(value, name) => {
              const v = Number(value);
              if (name === "p50") return [fmt(v), "median"];
              return [fmt(v), name === "outer" ? "10–90" : "25–75"];
            }}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <Area type="monotone" dataKey="p10" stackId="outer" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="outer" stackId="outer" stroke="none" fill="url(#fan-outer)" isAnimationActive={false} />
          <Area type="monotone" dataKey="p25" stackId="inner" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="inner" stackId="inner" stroke="none" fill="url(#fan-inner)" isAnimationActive={false} />
          <Area type="monotone" dataKey="p50" stroke="#3b82f6" strokeWidth={2} fill="none" dot={false} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
