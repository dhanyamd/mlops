"use client";

import {
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { Validation } from "@/lib/api";

const OUTCOME_COLOR: Record<string, string> = {
  passed: "#10b981",
  busted: "#ef4444",
  neutral: "#717171",
};

function fmtPct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}

/**
 * Efficiency cloud — the QuantPad/PropEdge pass-probability landscape.
 * Each dot is one sampled future positioned by its terminal return (x) vs.
 * its worst max drawdown (y). Green dots cleared the prop-firm gates and live
 * inside the target band; red dots busted on drawdown before reaching the
 * target; gray survived but missed the target. A 6% gap with no green means
 * the strategy's edge is insufficient for the selected rules — the honest
 * answer, rendered as a picture.
 *
 * Recharts renders SVG `<circle>` dots directly (no Canvas), which is crisp
 * at 100–200 paths and avoids WebGL/Canvas SSR pitfalls Next.js dislikes.
 */
export function EfficiencyCloud({
  validation,
  height = 300,
}: {
  validation: Validation;
  height?: number;
}) {
  const points = (validation.sample_paths ?? []).map((p) => ({
    x: p.terminal_return,
    y: p.max_drawdown,
    z: 1,
    outcome: p.outcome,
  }));
  const hasPassed = points.some((p) => p.outcome === "passed");
  const hasBusted = points.some((p) => p.outcome === "busted");
  const hasNeutral = points.some((p) => p.outcome === "neutral");

  const target = validation.target ?? 0;
  const ddRule = validation.max_drawdown_rule;

  const dataSets = Object.entries({ hasPassed, hasBusted, hasNeutral }).flatMap(
    ([outcome, present]) =>
      present
        ? [
            {
              label: outcome,
              data: points.filter((p) => p.outcome === outcome),
            },
          ]
        : []
  );

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis
            type="number"
            dataKey="x"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={fmtPct}
            label={{
              value: "terminal return",
              position: "bottom",
              offset: -4,
              fontSize: 11,
              fill: "currentColor",
              opacity: 0.6,
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[0, "dataMax"]}
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={fmtPct}
            label={{
              value: "max drawdown",
              position: "insideLeft",
              angle: -90,
              offset: 8,
              fontSize: 11,
              fill: "currentColor",
              opacity: 0.6,
            }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "currentColor", opacity: 0.2 }}
            formatter={(value: unknown) => [fmtPct(Number(value)), ""]}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <ZAxis type="number" dataKey="z" range={[4, 8]} />
          {dataSets.map((ds) => (
            <Scatter
              key={ds.label}
              name={ds.label}
              data={ds.data}
              fill={OUTCOME_COLOR[ds.label]}
              line={false}
            />
          ))}
          <ReferenceLine
            x={target}
            stroke="#10b981"
            strokeDasharray="6 4"
            strokeWidth={1.5}
            label={{
              value: `target ${fmtPct(target)}`,
              position: "top",
              fill: "#10b981",
              fontSize: 10,
            }}
          />
          <ReferenceLine
            y={ddRule}
            stroke="#ef4444"
            strokeDasharray="6 4"
            strokeWidth={1.5}
          />
          <ReferenceDot
            x={target}
            y={ddRule}
            r={0}
            fill="transparent"
            stroke="transparent"
          />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap items-center justify-end gap-4 text-xs text-zinc-500 dark:text-zinc-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-500" /> passed (hit {fmtPct(target)} target, DD &lt; {fmtPct(ddRule)})
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-red-500" /> busted (DD breached {fmtPct(ddRule)})
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-zinc-500" /> neutral (survived, missed target)
        </span>
      </div>
    </div>
  );
}
