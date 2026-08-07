"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Validation } from "@/lib/api";

const OUTCOME_STYLE: Record<string, string> = {
  passed: "#10b981",
  busted: "#ef4444",
  neutral: "#71717a",
};

const SAMPLE_LINES = 50;

function fmtPct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}

/**
 * QuantPad-style equity fan: percentile bands with the actual simulated
 * futures overlaid as outcome-colored thin lines (green = passed the prop-firm
 * rules, red = busted on max drawdown, gray = survived but missed the target),
 * plus the red max-drawdown bust line and green profit-target line.
 */
export function StrategyFan({
  validation,
  height = 320,
}: {
  validation: Validation;
  height?: number;
}) {
  const steps = validation.equity_fan["50"].length;
  const samples = validation.sample_paths.slice(0, SAMPLE_LINES);

  const loRef = 1 - validation.max_drawdown_rule;
  const hiRef = validation.target == null ? null : 1 + validation.target;

  const values: number[] = [loRef];
  if (hiRef != null) values.push(hiRef);
  for (const k of ["10", "25", "50", "75", "90"]) values.push(...validation.equity_fan[k]);
  for (const s of samples) values.push(...s.equity);

  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = (hi - lo) * 0.1 || 0.02;
  lo -= pad;
  hi += pad;

  const data = Array.from({ length: steps }, (_, step) => {
    const row: Record<string, number> = { step };
    row.p10 = validation.equity_fan["10"][step];
    row.p25 = validation.equity_fan["25"][step];
    row.p50 = validation.equity_fan["50"][step];
    row.p75 = validation.equity_fan["75"][step];
    row.p90 = validation.equity_fan["90"][step];
    row.outer = row.p90 - row.p10;
    row.inner = row.p75 - row.p25;
    samples.forEach((s, i) => {
      row[`s${i}`] = s.equity[step];
    });
    return row;
  });

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="vfan-outer" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.16} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.03} />
            </linearGradient>
            <linearGradient id="vfan-inner" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.08} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis dataKey="step" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={56}
            domain={[lo, hi]}
            tickFormatter={(v: number) => fmtPct(v)}
          />
          <Tooltip
            labelFormatter={(label) => `step ${String(label)}`}
            formatter={(value, name) => {
              const v = Number(value);
              if (name === "p50") return [fmtPct(v), "median future"];
              if (typeof name === "string" && name.startsWith("s")) return [fmtPct(v), "simulated future"];
              return [fmtPct(v), name === "outer" ? "10–90" : "25–75"];
            }}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          {samples.map((s, i) => (
            <Line
              key={i}
              type="monotone"
              dataKey={`s${i}`}
              stroke={OUTCOME_STYLE[s.outcome]}
              strokeWidth={0.8}
              dot={false}
              isAnimationActive={false}
              opacity={0.55}
            />
          ))}
          <ReferenceLine
            y={loRef}
            stroke="#ef4444"
            strokeDasharray="6 4"
            strokeWidth={1.5}
            label={{ value: `bust ${fmtPct(loRef)}`, position: "insideBottomRight", fill: "#ef4444", fontSize: 11 }}
          />
          {hiRef != null && (
            <ReferenceLine
              y={hiRef}
              stroke="#10b981"
              strokeDasharray="6 4"
              strokeWidth={1.5}
              label={{ value: `target ${fmtPct(hiRef)}`, position: "insideTopRight", fill: "#10b981", fontSize: 11 }}
            />
          )}
          <Area type="monotone" dataKey="p10" stackId="outer" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="outer" stackId="outer" stroke="none" fill="url(#vfan-outer)" isAnimationActive={false} />
          <Area type="monotone" dataKey="p25" stackId="inner" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="inner" stackId="inner" stroke="none" fill="url(#vfan-inner)" isAnimationActive={false} />
          <Area type="monotone" dataKey="p50" stroke="#3b82f6" strokeWidth={2} fill="none" dot={false} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
