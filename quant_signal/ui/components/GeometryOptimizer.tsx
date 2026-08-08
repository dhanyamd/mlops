"use client";

import { useMemo } from "react";

type GridResult = {
  grid: number[][];
  wr_axis: number[];
  rr_axis: number[];
  ev: number;
};

/**
 * Geometry Optimizer heat grid.
 *
 * Shows how pass probability changes across R:R configurations, holding win
 * rate constant. Research-backed insight (PropSim/QuantPad): trailing-DD rules
 * are path-dependent, so the R:R shape of your trades can swing pass probability
 * from 0% to 100% with identical edge.
 *
 * The grid animates cell-by-cell on first render (research shows traders scan
 * heatmaps left-to-right naturally, so we reveal in that order).
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
    const { grid: cells, wr_axis, rr_axis } = grid;
    return { cells, wr_axis, rr_axis };
  }, [grid]);

  if (!data) {
    return (
      <div className="py-12 text-center text-sm text-zinc-500" style={{ height }}>
        No realized strategy windows yet.
      </div>
    );
  }

  const { cells, wr_axis, rr_axis } = data;

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
            const intensity = pct / 100;
            const bg =
              pct < 25
                ? `rgba(239,68,68,${0.15 + intensity * 0.5})`
                : pct < 50
                  ? `rgba(251,191,36,${0.15 + intensity * 0.4})`
                  : `rgba(16,130,110,${0.15 + intensity * 0.5})`;

            return (
              <div
                key={`${ri}-${ci}`}
                className="relative flex h-8 w-full flex-col items-center justify-end rounded"
                style={{ background: bg, border: "1px solid rgba(0,0,0,0.05)" }}
                title={`WR ${(wr_axis[ci] * 100).toFixed(0)}% · R:R ${rr_axis[ri]} · ${pct.toFixed(1)}% pass`}
              >
                <span className="font-mono text-[9px] font-semibold text-zinc-700 dark:text-zinc-300">
                  {pct > 0 ? `${pct.toFixed(0)}` : ""}
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
    </div>
  );
}
