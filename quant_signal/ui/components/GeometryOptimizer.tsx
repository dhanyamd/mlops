"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type EdgeSweep = {
  edges_bps: number[];
  pass: number[];
  current_edge_bps: number;
};

type GridResult = {
  grid: number[][];
  wr_axis: number[];
  rr_axis: number[];
  ev: number;
  sigma?: number;
  edge_sweep?: EdgeSweep;
};

/**
 * Geometry Optimizer heat grid + edge sweep.
 *
 * The heat grid holds the strategy's expected per-period return constant while
 * sweeping win-rate × R:R, showing pass probability across 49 configurations
 * (PropSim/QuantPad insight: trailing-DD rules are path-dependent, so geometry
 * matters as much as edge). A negative-EV strategy honestly renders as an
 * all-zero grid, so beneath it we draw the *edge sweep*: pass probability vs
 * per-period edge in bps (prop-ev style), which shows exactly where the
 * challenge flips from unbeatable to passable and where the live edge sits
 * today. That curve moves every window even when every heat cell is 0%.
 */
export function GeometryOptimizer({
  grid,
  height = 320,
}: {
  grid: GridResult | null;
  height?: number;
}) {
  const data = useMemo(() => {
    if (!grid || !grid.grid.length) return null;
    const { grid: cells, wr_axis, rr_axis, edge_sweep } = grid;
    const sweep =
      edge_sweep && edge_sweep.edges_bps.length === edge_sweep.pass.length
        ? edge_sweep.edges_bps.map((edge, i) => ({
            edge,
            pass: edge_sweep.pass[i] * 100,
          }))
        : null;
    return { cells, wr_axis, rr_axis, sweep };
  }, [grid]);

  if (!data) {
    return (
      <div className="py-12 text-center text-sm text-zinc-500" style={{ height }}>
        No realized strategy windows yet.
      </div>
    );
  }

  const { cells, wr_axis, rr_axis, sweep } = data;

  return (
    <div className="w-full">
      <div className="mb-2 flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
        <span>win rate →</span>
        <span>R:R ↓</span>
      </div>
      <div
        className="grid items-end gap-1"
        style={{
          gridTemplateColumns: `repeat(${wr_axis.length}, 1fr)`,
          gap: 4,
        }}
      >
        {cells.map((row, ri) =>
          row.map((prob, ci) => {
            const pct = prob * 100;
            const t = Math.pow(Math.min(1, pct / 100), 0.6);
            const hue = 140 * t;
            const sat = 40 + 50 * t;
            const light = 42 + 22 * t;
            const bg = `hsl(${hue.toFixed(0)}, ${sat.toFixed(0)}%, ${light.toFixed(0)}%)`;

            return (
              <div
                key={`${ri}-${ci}`}
                className="relative flex h-8 w-full flex-col items-center justify-end rounded"
                style={{
                  background: bg,
                  border: "1px solid rgba(0,0,0,0.08)",
                  transition: "background-color 700ms ease, opacity 400ms ease",
                }}
                title={`WR ${(wr_axis[ci] * 100).toFixed(0)}% · R:R ${rr_axis[ri]} · ${pct.toFixed(1)}% pass`}
              >
                <span className="font-mono text-[9px] font-semibold text-zinc-800 dark:text-zinc-100">
                  {pct.toFixed(0)}
                </span>
              </div>
            );
          })
        )}
      </div>

      <div className="mt-3 flex items-center justify-between text-xs">
        <span className="text-zinc-500">
          WR {Math.round(wr_axis[0] * 100)}%–{Math.round(wr_axis[wr_axis.length - 1] * 100)}%
        </span>
        <span className="text-zinc-500">
          R:R {rr_axis[0].toFixed(1)}–{rr_axis[rr_axis.length - 1].toFixed(1)}
        </span>
      </div>

      <div className="mt-2 flex items-center gap-3 text-xs text-zinc-600 dark:text-zinc-400">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-red-500/30" /> &lt;25%
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-amber-400/30" /> 25–50%
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-emerald-600/30" /> &gt;50%
        </span>
        {grid && (
          <span className="ml-auto font-mono">
            EV {grid.ev >= 0 ? "+" : ""}{(grid.ev * 100).toFixed(3)}%/period
          </span>
        )}
      </div>

      {sweep && (
        <div className="mt-4">
          <div className="mb-1 flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>pass probability vs per-period edge</span>
            <span className="font-mono">
              live edge {grid?.edge_sweep?.current_edge_bps ?? 0} bps
            </span>
          </div>
          <div style={{ height: 160 }} className="w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sweep} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                <XAxis
                  dataKey="edge"
                  tick={{ fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => `${v}bps`}
                  minTickGap={24}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  width={36}
                  domain={[0, 100]}
                  tickFormatter={(v: number) => `${v}%`}
                />
                <Tooltip
                  formatter={(v) => [`${Number(v).toFixed(1)}%`, "pass probability"]}
                  labelFormatter={(v) => `edge ${Number(v).toFixed(1)} bps`}
                  contentStyle={{
                    background: "rgba(15,15,15,0.9)",
                    border: "none",
                    borderRadius: 8,
                    color: "#eee",
                    fontSize: 12,
                  }}
                />
                <ReferenceLine
                  x={grid?.edge_sweep?.current_edge_bps ?? 0}
                  stroke="#06b6d4"
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{
                    value: "live edge",
                    position: "insideTopLeft",
                    fill: "#06b6d4",
                    fontSize: 10,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="pass"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
