"use client";

import { useMemo } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Reality } from "@/lib/api";

const pct0 = (v: number) => `${Math.round(v * 100)}%`;

function Verdict({ reject }: { reject: boolean }) {
  return reject ? (
    <span className="rounded bg-red-100 px-1.5 py-0.5 text-[10px] font-semibold text-red-700 dark:bg-red-950 dark:text-red-300">
      REJECT
    </span>
  ) : (
    <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
      ok
    </span>
  );
}

const dotTooltip = { fontSize: 12, background: "rgba(15,15,15,0.9)", border: "none", borderRadius: 8, color: "#eee" };

type HitDotProps = {
  cx?: number;
  cy?: number;
  payload?: { step?: number; inBand?: boolean };
};

function HitDot({ cx, cy, payload }: HitDotProps) {
  if (cx == null || cy == null || payload?.step === 0) return <circle cx={cx} cy={cy} r={0} fill="transparent" />;
  const hit = payload?.inBand;
  return (
    <circle cx={cx} cy={cy} r={3.5} fill={hit ? "#10b981" : "#ef4444"} stroke="#fff" strokeWidth={1} />
  );
}

/**
 * Reality check: replays the MC engine's 1-step-ahead predictive distribution
 * point-in-time over stored history and scores it against realized closes —
 * the honest "does the model mean what it says?" panel.
 */
export function RealityCheck({ reality }: { reality: Reality }) {
  const { coverage, pit, evalue } = reality;

  const fanData = useMemo(() => {
    const f = reality.fan;
    const pct = f?.percentiles;
    const realized = reality.realized;
    if (!f || !pct || !realized) return [];
    const steps = Math.max(0, realized.length - 1);
    return Array.from({ length: steps + 1 }, (_, i) => {
      const lo = pct["10"]?.[i];
      const hi = pct["90"]?.[i];
      const med = pct["50"]?.[i];
      return {
        step: i,
        lo: lo ?? f.base_price,
        spread: hi != null && lo != null ? hi - lo : 0,
        median: med ?? f.base_price,
        realized: realized[i]?.close ?? f.base_price,
        inBand: realized[i]?.in_band ?? false,
      };
    });
  }, [reality]);

  const pitData = useMemo(
    () =>
      pit.counts.map((c, i) => ({
        mid: (pit.edges[i] + pit.edges[i + 1]) / 2,
        count: c,
        expected: pit.expected,
        ci_lo: pit.ci_lo,
        ci_hi: pit.ci_hi,
      })),
    [pit]
  );

  const eData = useMemo(
    () => evalue.process.map((e, i) => ({ i, e })),
    [evalue]
  );

  const covTone =
    Math.abs(coverage.coverage - reality.nominal_coverage) <= 0.06
      ? "text-emerald-500"
      : "text-amber-500";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-md bg-zinc-100 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
          reality check · replay
        </span>
        <span className="rounded-md bg-zinc-100 px-2 py-1 font-mono text-[10px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
          1-step-ahead · no overlap
        </span>
        {evalue.alarm ? (
          <span className="rounded-md bg-red-100 px-2 py-1 font-mono text-[10px] font-semibold text-red-700 dark:bg-red-950 dark:text-red-300">
            ALARM · anytime p {evalue.anytime_p.toExponential(1)}
          </span>
        ) : (
          <span className="rounded-md bg-emerald-100 px-2 py-1 font-mono text-[10px] font-semibold text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
            no alarm
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">10–90 coverage</div>
          <div className={`font-mono text-base font-semibold tabular-nums ${covTone}`}>
            {pct0(coverage.coverage)}
            <span className="text-xs text-zinc-400"> / {pct0(reality.nominal_coverage)}</span>
          </div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">MCB (W₁)</div>
          <div className="font-mono text-base font-semibold tabular-nums">{pit.mcb.toFixed(3)}</div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">KS</div>
          <div className="font-mono text-base font-semibold tabular-nums">{pit.ks.toFixed(3)}</div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">anytime p</div>
          <div className="font-mono text-base font-semibold tabular-nums">{evalue.anytime_p.toExponential(1)}</div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Kupiec</div>
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-base font-semibold tabular-nums">
              {coverage.pof.p.toFixed(2)}
            </span>
            <Verdict reject={coverage.pof.reject} />
          </div>
        </div>
        <div className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Christoffersen</div>
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-base font-semibold tabular-nums">
              {coverage.cc ? coverage.cc.p.toFixed(2) : "—"}
            </span>
            {coverage.cc ? <Verdict reject={coverage.cc.reject} /> : <span className="text-xs text-zinc-400">n/a</span>}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div>
          <div className="mb-1.5 text-xs text-zinc-500 dark:text-zinc-400">
            fan vs realized · bands 10–90 · median · dots = in/out of band
          </div>
          <div style={{ height: 180 }} className="w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={fanData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                <XAxis dataKey="step" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={64}
                  domain={["auto", "auto"]}
                  tickFormatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                />
                <Tooltip
                  formatter={(v, name) => {
                    const labels: Record<string, string> = {
                      spread: "band",
                      median: "median",
                      realized: "realized",
                    };
                    const k = String(name ?? "");
                    return [Number(v).toLocaleString(), labels[k] ?? k];
                  }}
                  labelFormatter={(l) => `step ${String(l)}`}
                  contentStyle={dotTooltip}
                />
                <Area dataKey="lo" stackId="b" stroke="none" fill="rgba(59,130,246,0.10)" isAnimationActive={false} />
                <Area dataKey="spread" stackId="b" stroke="none" fill="rgba(59,130,246,0.10)" isAnimationActive={false} />
                <Line dataKey="median" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line
                  dataKey="realized"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={<HitDot />}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <div className="mb-1.5 text-xs text-zinc-500 dark:text-zinc-400">
            PIT histogram vs U(0,1) · bars = observed, line = 95% binomial CI
          </div>
          <div style={{ height: 180 }} className="w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={pitData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                <XAxis
                  dataKey="mid"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => v.toFixed(1)}
                />
                <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                <Tooltip
                  formatter={(v, name) => [
                    name === "count" ? `${Number(v)}` : Number(v).toFixed(1),
                    name === "count" ? "observed" : "expected",
                  ]}
                  labelFormatter={(l) => `PIT ~${Number(l).toFixed(2)}`}
                  contentStyle={dotTooltip}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} isAnimationActive={false} />
                <Line
                  dataKey="expected"
                  stroke="#a1a1aa"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="ci_lo"
                  stroke="#a1a1aa"
                  strokeWidth={1}
                  strokeDasharray="2 3"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="ci_hi"
                  stroke="#a1a1aa"
                  strokeWidth={1}
                  strokeDasharray="2 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
          <span>anytime-valid e-process · alarm at 1/α = {evalue.threshold}</span>
          <span className="font-mono">
            e = {evalue.e_value.toFixed(2)}
          </span>
        </div>
        <div style={{ height: 120 }} className="w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={eData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
              <XAxis dataKey="i" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={32} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={48}
                scale="log"
                domain={["auto", "auto"]}
                allowDataOverflow
              />
              <Tooltip
                formatter={(v, name) => [name === "e" ? Number(v).toFixed(3) : Number(v).toFixed(2), name === "e" ? "e-value" : "threshold"]}
                labelFormatter={(l) => `window ${String(l)}`}
                contentStyle={dotTooltip}
              />
              <Area dataKey="e" stroke="#8b5cf6" strokeWidth={1.8} fill="rgba(139,92,246,0.12)" dot={false} isAnimationActive={false} />
              <ReferenceLine
                y={evalue.threshold}
                stroke="#ef4444"
                strokeDasharray="4 3"
                label={{ value: "1/α", position: "insideTopRight", fontSize: 10, fill: "#ef4444" }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <p className="text-[11px] leading-relaxed text-zinc-400 dark:text-zinc-500">
        Point-in-time replay of the engine&apos;s exact 1-step predictive distribution (Student-t + EWMA,
        same code path as the live simulator) scored against realized closes — coverage backtests, PIT
        calibration, and an anytime-valid e-process (Arnold–Henzi–Ziegel 2021). A green fan that the
        realized line keeps missing is a model lying to you.
      </p>
    </div>
  );
}
