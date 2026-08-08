"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
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
  breakeven_edge_bps: number;
  n_periods?: number;
  target?: number;
  max_drawdown?: number;
  seed?: number | null;
};

type GridResult = {
  grid: number[][];
  wr_axis: number[];
  rr_axis: number[];
  ev: number;
  sigma?: number;
  seed?: number | null;
  edge_sweep?: EdgeSweep;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Geometry Optimizer: edge sweep + R:R heat grid.
 *
 * The honest headline is the **edge sweep**: pass probability vs per-period
 * edge in bps, seeded from the current 5m window (so it visibly changes every
 * window while staying reproducible). It answers the one question a PM asks
 * first: *"my live edge is X bps — how far am I from the +Y bps/window this
 * volatility + drawdown rule demands for a 50% pass rate?"* The gap between
 * the live edge and the breakeven edge is shaded.
 *
 * The heat grid below holds edge constant and sweeps win-rate × R:R
 * (PropSim/QuantPad: trailing-DD rules are path-dependent). A negative-EV
 * strategy honestly renders as an all-zero grid — flagged as such, not hidden.
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
    const live = edge_sweep?.current_edge_bps ?? 0;
    const breakeven = edge_sweep?.breakeven_edge_bps ?? 0;
    const zeroGrid = cells.every((row) => row.every((c) => c === 0));
    return { cells, wr_axis, rr_axis, sweep, live, breakeven, zeroGrid };
  }, [grid]);

  if (!data) {
    return (
      <div className="py-12 text-center text-sm text-zinc-500" style={{ height }}>
        No realized strategy windows yet.
      </div>
    );
  }

  const { cells, wr_axis, rr_axis, sweep, live, breakeven, zeroGrid } = data;
  const gap = breakeven - live;

  return (
    <div className="w-full">
      <div className="mb-3 grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">live edge</div>
          <div className={`mt-0.5 font-mono text-sm font-semibold tabular-nums ${live >= 0 ? "text-emerald-500" : "text-red-500"}`}>
            {live >= 0 ? "+" : ""}
            {live.toFixed(2)} bps/w
          </div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">breakeven edge</div>
          <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-amber-500">
            {breakeven >= 0 ? "+" : ""}
            {breakeven.toFixed(2)} bps/w
          </div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">gap to 50% pass</div>
          <div className={`mt-0.5 font-mono text-sm font-semibold tabular-nums ${gap <= 0 ? "text-emerald-500" : "text-red-500"}`}>
            {gap >= 0 ? "+" : ""}
            {gap.toFixed(2)} bps/w
          </div>
        </div>
      </div>

      {sweep && (
        <div className="mb-4">
          <div className="mb-1 flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>pass probability vs per-period edge</span>
            {edge_sweep_note(grid)}
          </div>
          <div style={{ height: 180 }} className="w-full">
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
                <ReferenceArea
                  x1={clamp(Math.min(live, breakeven), -12, 12)}
                  x2={clamp(Math.max(live, breakeven), -12, 12)}
                  fill="#f59e0b"
                  fillOpacity={0.08}
                />
                <ReferenceLine
                  x={clamp(live, -12, 12)}
                  stroke="#06b6d4"
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{
                    value: `live ${live.toFixed(1)}`,
                    position: "insideTopLeft",
                    fill: "#06b6d4",
                    fontSize: 10,
                  }}
                />
                <ReferenceLine
                  x={clamp(breakeven, -12, 12)}
                  stroke="#f59e0b"
                  strokeDasharray="4 3"
                  strokeWidth={1.5}
                  label={{
                    value: `breakeven ${breakeven.toFixed(1)}`,
                    position: "insideTopRight",
                    fill: "#f59e0b",
                    fontSize: 10,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="pass"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive
                  animationDuration={800}
                  animationEasing="ease-out"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

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

      {zeroGrid && (
        <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600 dark:text-amber-400">
          Live edge {live.toFixed(2)} bps is below the {breakeven.toFixed(1)} bps breakeven for this
          volatility + {grid?.edge_sweep?.max_drawdown != null ? `${(grid.edge_sweep.max_drawdown * 100).toFixed(0)}%` : ""} max-DD rule, so
          no R:R geometry passes — the all-zero grid is the honest answer. The sweep above shows
          what edge clears it.
        </div>
      )}
    </div>
  );
}

function edge_sweep_note(grid: GridResult | null): React.ReactNode {
  if (!grid?.edge_sweep) return null;
  const { n_periods, seed } = grid.edge_sweep;
  const parts: string[] = [];
  if (n_periods != null) parts.push(`${n_periods} windows`);
  if (seed != null && seed !== 0) parts.push(`seed ${seed}`);
  return <span className="font-mono">{parts.join(" · ")}</span>;
}
