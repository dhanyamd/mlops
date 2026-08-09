"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/Card";
import { AllPathsFan } from "@/components/AllPathsFan";
import { LiveTape } from "@/components/LiveTape";
import { NextWindowCountdown } from "@/components/NextWindowCountdown";
import { PathDrawIn } from "@/components/PathDrawIn";
import { LiveBadge, PipelineHealth } from "@/components/PipelineHealth";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ExecutionPanel } from "@/components/ExecutionPanel";
import { GeometryOptimizer } from "@/components/GeometryOptimizer";
import { ProbSurface3D } from "@/components/ProbSurface3D";
import { RealizedReturns } from "@/components/RealizedReturns";
import { Select } from "@/components/Select";
import { StrategyFan } from "@/components/StrategyFan";
import { api, type FeatureWindow, type GeometryGrid, type Prediction, type Simulation, type Strategy, type Validation } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

const POLL_MS = 15000;

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

const DIRECTION_STYLE: Record<string, string> = {
  LONG: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  SHORT: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  FLAT: "bg-zinc-100 text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300",
};

function DirectionBadge({ direction }: { direction: string }) {
  return (
    <span
      className={`inline-block rounded-md px-2.5 py-1 font-mono text-xs font-semibold tracking-wider ${
        DIRECTION_STYLE[direction] ?? DIRECTION_STYLE.FLAT
      }`}
    >
      {direction}
    </span>
  );
}

function Stat({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-xs text-zinc-500 dark:text-zinc-400">{label}</div>
      <div className={`mt-1 font-mono text-lg tabular-nums ${tone}`}>{value}</div>
    </div>
  );
}

function SignalGauge({ prediction }: { prediction: Prediction }) {
  const lo = prediction.interval_low;
  const hi = prediction.interval_high;
  const ret = prediction.predicted_return;
  const min = Math.min(lo, 0, hi);
  const max = Math.max(lo, 0, hi);
  const span = max - min || 1;
  const pos = (v: number) => `${((v - min) / span) * 100}%`;
  const tone = prediction.direction === "LONG" ? "text-emerald-500" : prediction.direction === "SHORT" ? "text-red-500" : "text-zinc-400";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <DirectionBadge direction={prediction.direction} />
        <span className="font-mono text-xs text-zinc-500">
          ACI 90% · α<sub>t</sub> {prediction.alpha?.toFixed(3) ?? "—"}
        </span>
      </div>

      <div>
        <div className={`font-mono text-3xl font-semibold tabular-nums ${tone}`}>
          {ret >= 0 ? "+" : ""}
          {pct(ret)}
        </div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400">expected next-window return</div>
      </div>

      <div>
        <div className="mb-1 flex justify-between font-mono text-[11px] text-zinc-500">
          <span>{pct1(lo)}</span>
          <span>0%</span>
          <span>{pct1(hi)}</span>
        </div>
        <div className="relative h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
          <div
            className="absolute h-2 rounded-full bg-blue-500/30"
            style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }}
          />
          <div className="absolute top-[-2px] h-[12px] w-px bg-zinc-500" style={{ left: pos(0) }} />
          <div className="absolute top-[-3px] h-2 w-1 rounded-full bg-blue-600" style={{ left: pos(ret) }} />
        </div>
        <div className="mt-1 flex items-center justify-between text-xs">
          <span className="text-zinc-500 dark:text-zinc-400">
            conformal interval {pct1(lo)} → {pct1(hi)}
          </span>
          <span className="font-mono text-zinc-500">
            coverage {prediction.coverage == null ? "—" : pct1(prediction.coverage)}
          </span>
        </div>
      </div>
    </div>
  );
}

function McGauge({ simulation }: { simulation: Simulation }) {
  const up = simulation.prob_up;
  const tone = up >= 0.5 ? "text-emerald-500" : "text-red-500";
  const pctDeg = Math.round(up * 180);
  const arc = {
    background: `conic-gradient(currentColor ${pctDeg}deg, rgba(128,128,128,0.2) ${pctDeg}deg)`,
  };

  return (
    <div className="flex items-center gap-6">
      <div
        className={`grid h-32 w-32 shrink-0 place-items-center rounded-full ${tone}`}
        style={arc}
      >
        <div className="grid h-24 w-24 place-items-center rounded-full bg-white dark:bg-zinc-900">
          <div className="text-center">
            <div className="font-mono text-2xl font-semibold tabular-nums">
              {Math.round(up * 100)}
              <span className="text-sm">%</span>
            </div>
            <div className="text-[10px] text-zinc-500">P(up)</div>
          </div>
        </div>
      </div>
      <div className="space-y-1 font-mono text-xs text-zinc-500 dark:text-zinc-400">
        <div>
          median {pct1(simulation.median_path[simulation.median_path.length - 1] / simulation.base_price - 1)}
        </div>
        <div>
          90% CI {pct1(simulation.confidence_interval.p10 / simulation.base_price - 1)} →{" "}
          {pct1(simulation.confidence_interval.p90 / simulation.base_price - 1)}
        </div>
        <div>
          {simulation.n_paths.toLocaleString()} paths · {simulation.horizon_steps}×5m
        </div>
      </div>
    </div>
  );
}

function ReturnsHistogram({ simulation }: { simulation: Simulation }) {
  const { counts, edges } = simulation.returns_histogram;
  const data = counts.map((c, i) => ({
    mid: ((edges[i] + edges[i + 1]) / 2) * 100,
    count: c,
  }));
  return (
    <div style={{ height: 220 }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis
            dataKey="mid"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
          />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
          <Tooltip
            formatter={(v) => [`${Number(v)} paths`, "terminal return"]}
            labelFormatter={(v) => `~${Number(v).toFixed(2)}%`}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" fill="#3b82f6" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function PassArc({ validation }: { validation: Validation }) {
  const W = 280;
  const H = 160;
  const cx = W / 2;
  const cy = H - 6;
  const r = 108;
  const thick = 16;
  const C = Math.PI * r;
  const pass = validation.pass_probability;
  const bust = validation.bust_rate;
  const neutral = validation.neutral_rate;
  const track = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  const passLen = C * pass;
  const bustLen = C * bust;
  const tone = pass >= 0.5 ? "text-emerald-500" : pass >= 0.25 ? "text-amber-500" : "text-red-500";

  return (
    <div className="flex flex-col items-center">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="pass probability gauge">
        <path d={track} stroke="rgba(128,128,128,0.18)" strokeWidth={thick} fill="none" strokeLinecap="butt" />
        {passLen > 0 && (
          <path d={track} stroke="#10b981" strokeWidth={thick} fill="none" strokeLinecap="butt" strokeDasharray={`${passLen} ${C}`} />
        )}
        {bustLen > 0 && (
          <path d={track} stroke="#ef4444" strokeWidth={thick} fill="none" strokeLinecap="butt" strokeDasharray={`${bustLen} ${C}`} strokeDashoffset={-passLen} />
        )}
        {neutral > 0 && (
          <path d={track} stroke="#71717a" strokeWidth={thick} fill="none" strokeLinecap="butt" strokeDasharray={`${neutral * C} ${C}`} strokeDashoffset={-(passLen + bustLen)} />
        )}
      </svg>
      <div className="-mt-1 text-center">
        <div className={`font-mono text-4xl font-semibold tabular-nums ${tone}`}>
          {Math.round(pass * 100)}
          <span className="text-lg">%</span>
        </div>
        <div className="text-[10px] uppercase tracking-widest text-zinc-500">pass probability</div>
      </div>
      <div className="mt-3 grid w-full grid-cols-3 gap-2 text-center">
        <div className="rounded-lg bg-emerald-50 px-2 py-1.5 dark:bg-emerald-950/40">
          <div className="font-mono text-sm font-semibold text-emerald-600 dark:text-emerald-400">{pct1(pass)}</div>
          <div className="text-[10px] text-zinc-500">passed</div>
        </div>
        <div className="rounded-lg bg-zinc-100 px-2 py-1.5 dark:bg-zinc-900">
          <div className="font-mono text-sm font-semibold text-zinc-600 dark:text-zinc-400">{pct1(neutral)}</div>
          <div className="text-[10px] text-zinc-500">neutral</div>
        </div>
        <div className="rounded-lg bg-red-50 px-2 py-1.5 dark:bg-red-950/40">
          <div className="font-mono text-sm font-semibold text-red-600 dark:text-red-400">{pct1(bust)}</div>
          <div className="text-[10px] text-zinc-500">busted</div>
        </div>
      </div>
    </div>
  );
}

function OutcomeHistogram({ validation }: { validation: Validation }) {
  const h = validation.terminal_histogram;
  const data = h.counts.map((c, i) => ({
    mid: ((h.edges[i] + h.edges[i + 1]) / 2 - 1) * 100,
    passed: h.passed[i],
    neutral: h.neutral[i],
    busted: h.busted[i],
  }));
  return (
    <div style={{ height: 220 }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis
            dataKey="mid"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
          />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
          <Tooltip
            formatter={(v, name) => [Number(v), String(name)]}
            labelFormatter={(v) => `~${Number(v).toFixed(1)}% terminal`}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <Bar dataKey="passed" stackId="o" fill="#10b981" />
          <Bar dataKey="neutral" stackId="o" fill="#71717a" />
          <Bar dataKey="busted" stackId="o" fill="#ef4444" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function DrawdownHistogram({ validation }: { validation: Validation }) {
  const h = validation.drawdown_histogram;
  const data = h.counts.map((c, i) => ({
    mid: ((h.edges[i] + h.edges[i + 1]) / 2) * 100,
    count: c,
  }));
  return (
    <div style={{ height: 220 }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
          <XAxis
            dataKey="mid"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
          />
          <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
          <Tooltip
            formatter={(v) => [`${Number(v)} paths`, "max drawdown"]}
            labelFormatter={(v) => `~${Number(v).toFixed(1)}%`}
            contentStyle={{
              background: "rgba(15,15,15,0.9)",
              border: "none",
              borderRadius: 8,
              color: "#eee",
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" fill="#f59e0b" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ValidationPanel({
  validation,
  geometry,
  realized,
  stress = false,
}: {
  validation: Validation;
  geometry: GeometryGrid | null;
  realized?: number[];
  stress?: boolean;
}) {
  const rulePct = pct(validation.max_drawdown_rule);
  const targetPct = validation.target == null ? "—" : pct(validation.target);
  const ret = validation.expected_return;
  const simTag = stress ? "WHAT-IF · simulated · " : "";
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Pass probability" subtitle={`${simTag}QuantPad-style · ${validation.n_sims.toLocaleString()} futures`}>
          <PassArc validation={validation} />
        </Card>
        <div className="lg:col-span-2">
          <Card
            title="Simulated futures"
            subtitle={`${simTag}${validation.n_periods} realized windows resampled · ${rulePct} max DD · ${targetPct} target`}
          >
            <StrategyFan validation={validation} realized={realized} height={280} />
            <div className="mt-2 flex flex-wrap items-center justify-end gap-4 text-xs text-zinc-500 dark:text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="h-0.5 w-4 bg-emerald-500" /> passed (hit target, no DD breach)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-0.5 w-4 bg-red-500" /> busted (DD breach)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-0.5 w-4 bg-zinc-500" /> neutral
              </span>
              {realized && (
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-cyan-500" /> realized equity
                </span>
              )}
            </div>
          </Card>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card title="Terminal outcomes" subtitle={`${simTag}final equity across futures, colored by verdict`}>
            <OutcomeHistogram validation={validation} />
          </Card>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Max drawdown" subtitle={`${simTag}distribution of worst peak-to-trough · rule ${rulePct}`}>
          <DrawdownHistogram validation={validation} />
        </Card>
       </div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
        <Stat
          label="expected return"
          value={`${ret >= 0 ? "+" : ""}${pct(ret)}`}
          tone={ret >= 0 ? "text-emerald-500" : "text-red-500"}
        />
        <Stat label="median max DD" value={`-${pct(validation.median_max_drawdown)}`} tone="text-red-500" />
        <Stat label="p95 max DD" value={`-${pct(validation.p95_max_drawdown)}`} tone="text-red-500" />
        <Stat label="best 10% terminal" value={`+${pct(validation.best10_terminal - 1)}`} tone="text-emerald-500" />
        <Stat label="worst 10% terminal" value={pct(validation.worst10_terminal - 1)} tone="text-red-500" />
      </div>
      <Card
        title="Geometry optimizer"
        subtitle="pass probability across R:R configurations · same edge, different shape"
      >
        <GeometryOptimizer grid={geometry} height={320} />
      </Card>
    </div>
  );
}

function PnLStrip({ strategy }: { strategy: Strategy }) {
  const data = useMemo(
    () =>
      strategy.strategy_equity.map((e, i) => ({
        i,
        strategy: e - 1,
        buyhold: strategy.buyhold_equity[i] ?? e - 1,
      })),
    [strategy]
  );
  const strat = strategy.total_return_strategy;
  const buy = strategy.total_return_buyhold;
  const drawdown = useMemo(() => {
    let peak = -Infinity;
    let maxDd = 0;
    for (const e of strategy.strategy_equity) {
      peak = Math.max(peak, e);
      maxDd = Math.max(maxDd, peak > 0 ? (peak - e) / peak : 0);
    }
    return maxDd;
  }, [strategy]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="strategy return" value={`${strat >= 0 ? "+" : ""}${pct(strat)}`} tone={strat >= 0 ? "text-emerald-500" : "text-red-500"} />
        <Stat label="buy-and-hold" value={`${buy >= 0 ? "+" : ""}${pct(buy)}`} tone={buy >= 0 ? "text-emerald-500" : "text-red-500"} />
        <Stat label="win rate" value={strategy.win_rate == null ? "—" : pct1(strategy.win_rate)} />
        <Stat label="max drawdown" value={`-${pct(drawdown)}`} />
      </div>
      <div style={{ height: 220 }} className="w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="strat-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
            <XAxis dataKey="i" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={32} />
            <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={56} tickFormatter={(v: number) => pct(v)} domain={["auto", "auto"]} />
            <Tooltip
              formatter={(v, name) => [pct(Number(v)), name === "strategy" ? "strategy" : "buy-and-hold"]}
              labelFormatter={(label) => `window ${String(label)}`}
              contentStyle={{
                background: "rgba(15,15,15,0.9)",
                border: "none",
                borderRadius: 8,
                color: "#eee",
                fontSize: 12,
              }}
            />
            <Area type="monotone" dataKey="buyhold" stroke="#a1a1aa" strokeWidth={1.5} strokeDasharray="4 3" fill="none" dot={false} isAnimationActive={false} />
            <Area type="monotone" dataKey="strategy" stroke="#10b981" strokeWidth={2} fill="url(#strat-fill)" dot={false} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-end gap-4 text-xs text-zinc-500 dark:text-zinc-400">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-emerald-500" /> strategy (signals)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 border-t border-dashed border-zinc-400" /> buy-and-hold
        </span>
        <span>{strategy.n_trades} trades</span>
      </div>
    </div>
  );
}

function FeaturePanel({ features }: { features: FeatureWindow[] }) {
  const latest = features[0];
  if (!latest) return <p className="py-12 text-center text-sm text-zinc-500">No feature windows yet.</p>;

  const rows: { k: string; v: string }[] = [
    { k: "close", v: latest.close.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
    { k: "open", v: latest.open.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
    { k: "high", v: latest.high.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
    { k: "low", v: latest.low.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
    { k: "vwap", v: latest.vwap == null ? "—" : latest.vwap.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
    { k: "volume", v: latest.volume.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
    { k: "bar_count", v: String(latest.bar_count ?? "—") },
    { k: "window", v: new Date(latest.window_end_ms).toLocaleTimeString() },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {rows.map((r) => (
        <div key={r.k} className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-800">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">{r.k}</div>
          <div className="mt-0.5 truncate font-mono text-sm tabular-nums">{r.v}</div>
        </div>
      ))}
    </div>
  );
}

export default function SignalPage() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [tick, setTick] = useState(0);
  const [stress, setStress] = useState(false);
  const scenario = stress ? "stress" : undefined;

  useEffect(() => {
    api.marketSymbols().then((s) => {
      setSymbols(s.symbols);
      setSymbol((cur) => (s.symbols.includes(cur) ? cur : (s.symbols[0] ?? cur)));
    });
  }, []);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), POLL_MS);
    return () => clearInterval(id);
  }, []);

  const prediction = useQuery([symbol, tick], () => api.prediction(symbol));
  const simulation = useQuery([symbol, tick, stress], () => api.simulation(symbol, scenario));
  const strategy = useQuery([symbol, tick], () => api.strategy(symbol));
  const features = useQuery([symbol, tick], () => api.features(symbol, 12));
  const validation = useQuery([symbol, tick, stress], () => api.validation(symbol, scenario));
  const geometry = useQuery(["geometry", symbol, tick], () => api.geometry(symbol));
  const execution = useQuery([symbol, tick], () => api.execution(symbol));
  const portfolio = useQuery([tick], () => api.portfolio());
  const health = useQuery([tick], () => api.healthSummary());

  const error =
    prediction.error ??
    simulation.error ??
    strategy.error ??
    features.error ??
    validation.error ??
    geometry.error ??
    execution.error ??
    portfolio.error ??
    health.error;
  const pred = prediction.data?.prediction ?? null;
  const sim = simulation.data?.simulation ?? null;
  const strat = strategy.data?.strategy ?? null;
  const feat = features.data?.features ?? null;
  const val = validation.data?.validation ?? null;
  const geo = geometry.data?.grid ?? null;
  const lastUpdated = pred?.updated_at ?? strat?.updated_at ?? null;
  const simTag = stress ? "WHAT-IF · simulated · " : "";

  // Shared fixed price window (base ± 4σ·√steps) so the fan, draw-in and
  // surface all track the real calibrated volatility instead of auto-zooming
  // each ensemble to look identical every window.
  const mcDomain = useMemo(() => {
    if (!sim || !(sim.sigma > 0)) return undefined;
    const half = sim.base_price * 4 * sim.sigma * Math.sqrt(sim.horizon_steps);
    return [sim.base_price - half, sim.base_price + half] as [number, number];
  }, [sim]);
  const volLabel = sim
    ? `σ ${(sim.sigma_annualized * 100).toFixed(1)}%/yr · VaR95 ${pct(sim.var95)}`
    : "";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Signal Terminal</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            River online model + ACI conformal intervals + 10k-path Monte Carlo, refreshed every 5m window
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <LiveBadge summary={health.data} symbol={symbol} />
          <div
            role="group"
            aria-label="what-if stress preview"
            className="flex items-center rounded-lg border border-zinc-200 p-0.5 font-mono text-xs dark:border-zinc-800"
          >
            <button
              type="button"
              onClick={() => setStress(false)}
              className={`rounded-md px-2.5 py-1 font-semibold tracking-wide ${
                stress
                  ? "text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200"
                  : "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
              }`}
            >
              Live
            </button>
            <button
              type="button"
              onClick={() => setStress(true)}
              className={`rounded-md px-2.5 py-1 font-semibold tracking-wide ${
                stress
                  ? "bg-cyan-600 text-white"
                  : "text-zinc-500 hover:text-cyan-700 dark:text-zinc-400 dark:hover:text-cyan-300"
              }`}
            >
              Stress · what-if
            </button>
          </div>
          <Select
            label="Symbol"
            value={symbol}
            onChange={setSymbol}
            options={symbols.map((s) => ({ value: s, label: s }))}
          />
          <div className="text-right font-mono text-xs text-zinc-500 dark:text-zinc-400">
            <div>{lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : "—"}</div>
            <div className="text-zinc-400 dark:text-zinc-600">
              updates every 5m · <NextWindowCountdown />
            </div>
          </div>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </div>
      ) : null}

      <PipelineHealth summary={health.data} symbol={symbol} />

      <Card
        title="Live minute tape"
        subtitle="raw venue 1m bars straight off the Kafka stream · re-forms every ~20s, independent of the 5m windows below"
      >
        <LiveTape symbol={symbol} height={200} />
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="Signal" subtitle={pred ? `${pred.symbol} · ${pred.direction}` : "warming up"}>
          {pred ? <SignalGauge prediction={pred} /> : <p className="py-12 text-center text-sm text-zinc-500">No prediction yet — waiting for the next 5m window.</p>}
        </Card>
        <Card title="Monte Carlo" subtitle={`${simTag}10,000 paths · geometric Brownian motion`}>
          {sim ? <McGauge simulation={sim} /> : <p className="py-12 text-center text-sm text-zinc-500">Warming up — needs a few realized windows to calibrate volatility.</p>}
        </Card>
        <Card title="Risk" subtitle={`per-window, horizon ${sim?.horizon_steps ?? 12}×5m risk stats`}>
          {sim ? (
            <div className="grid grid-cols-2 gap-3">
              <Stat label="base price" value={sim.base_price.toLocaleString(undefined, { maximumFractionDigits: 0 })} />
              <Stat label="σ annualized" value={pct1(sim.sigma_annualized)} />
              <Stat label="VaR95 (1h)" value={pct1(sim.var95)} tone="text-red-500" />
              <Stat label="ES95 (CVaR)" value={pct1(sim.es95)} tone="text-red-500" />
            </div>
          ) : (
            <p className="py-12 text-center text-sm text-zinc-500">Warming up…</p>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card
            title="Probability surface"
            subtitle={
              sim
                ? `${simTag}return density per forward step · ${sim.n_paths.toLocaleString()} paths · base ${sim.base_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                : "Monte Carlo return density"
            }
          >
            {sim?.surface_grid ? (
              <ErrorBoundary fallback={<p className="py-12 text-center text-sm text-zinc-500">WebGL unavailable — 2D fan above still rendering.</p>}>
                <ProbSurface3D surface={sim.surface_grid} basePrice={sim.base_price} height={300} />
              </ErrorBoundary>
            ) : (
              <p className="py-12 text-center text-sm text-zinc-500">Warming up…</p>
            )}
          </Card>
        </div>
        <Card
          title="Path draw-in"
          subtitle={sim ? `${simTag}${sim.sample_paths?.length ?? 0} sampled paths · continuous replay` : "Monte Carlo path reveal"}
        >
            {sim?.sample_paths?.length ? (
              <PathDrawIn
                paths={sim.sample_paths}
                basePrice={sim.base_price}
                domain={mcDomain}
                height={300}
              />
            ) : (
            <p className="py-12 text-center text-sm text-zinc-500">Warming up…</p>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card
            title="All simulated futures"
            subtitle={
              sim
                ? `${simTag}${(sim.sample_paths?.length ?? 0).toLocaleString()} raw paths · percentile bands 10–90 / 25–75 · median · ${volLabel}`
                : "Monte Carlo all-paths fan"
            }
          >
            {sim && sim.sample_paths?.length ? (
              <AllPathsFan
                paths={sim.sample_paths}
                percentiles={sim.percentiles}
                basePrice={sim.base_price}
                domain={mcDomain}
                height={280}
              />
            ) : (
              <p className="py-12 text-center text-sm text-zinc-500">Warming up…</p>
            )}
          </Card>
        </div>
        <Card title="Terminal returns" subtitle={`${simTag}distribution of final path returns`}>
          {sim ? <ReturnsHistogram simulation={sim} /> : <p className="py-12 text-center text-sm text-zinc-500">Warming up…</p>}
        </Card>
      </div>

      <Card
        title="Paper execution"
        subtitle={
          execution.data?.execution
            ? `${execution.data.execution.symbol} · simulated fills on live signals · adverse-side slippage ${execution.data.execution.slippage_bps}bps · taker ${execution.data.execution.taker_fee_bps}bps · last ${execution.data.execution.fills.length} fills`
            : "paper book on the predictor's real signals — first position opens at the close of the window after a signal appears"
        }
      >
        <ExecutionPanel
          execution={execution.data?.execution ?? null}
          portfolio={portfolio.data ?? null}
        />
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card title="Live P&L vs buy-and-hold" subtitle={strat ? `${strat.n_windows} windows · ${strat.n_trades} trades` : "compounded from realized 5m signals"}>
            {strat ? <PnLStrip strategy={strat} /> : <p className="py-12 text-center text-sm text-zinc-500">Warming up — equity compounds from realized signals.</p>}
          </Card>
        </div>
        <div className="space-y-6">
          <Card title="Latest window features" subtitle="Flink 5m aggregates from Redis">
            {feat ? <FeaturePanel features={feat} /> : <p className="py-12 text-center text-sm text-zinc-500">No feature windows yet.</p>}
          </Card>
          <Card
            title="Realized 5m returns"
            subtitle="actual moves each window — the market the model is reacting to"
          >
            {feat && feat.length >= 2 ? (
              <RealizedReturns features={feat} height={200} />
            ) : (
              <p className="py-12 text-center text-sm text-zinc-500">Need at least two feature windows.</p>
            )}
          </Card>
        </div>
      </div>

      {val ? (
        <ValidationPanel validation={val} geometry={geo} realized={strat?.strategy_equity} stress={stress} />
      ) : (
        <Card title="Strategy validation" subtitle="QuantPad-style pass probability">
          <p className="py-12 text-center text-sm text-zinc-500">
            Warming up — needs at least a few realized strategy windows to bootstrap futures.
          </p>
        </Card>
      )}
    </div>
  );
}
