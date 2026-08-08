"use client";

import { useMemo } from "react";
import type { HealthStage, HealthSummary } from "@/lib/api";

const STAGE_ORDER: HealthStage["name"][] = [
  "produce",
  "features",
  "predict",
  "simulate",
  "strategy",
];

const STAGE_LABEL: Record<HealthStage["name"], string> = {
  produce: "ingest",
  features: "features",
  predict: "predict",
  simulate: "simulate",
  strategy: "strategy",
};

const DOT: Record<string, string> = {
  healthy: "bg-emerald-500",
  stale: "bg-red-500",
  warming: "bg-zinc-400 dark:bg-zinc-600",
};

function fmtAge(seconds: number | null): string {
  if (seconds == null) return "warming";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return rem ? `${m}m ${rem}s` : `${m}m`;
  }
  const h = Math.floor(s / 3600);
  return `${h}h ${Math.floor((s % 3600) / 60)}m`;
}

/**
 * Operator-status pill: "live · 0s" when every stage is fresh, "stale · 1h 3m"
 * when any stage lags. Age is event-time staleness from the health summary —
 * the same number the stream watchdog alerts on.
 */
export function LiveBadge({ summary, symbol }: { summary: HealthSummary | null; symbol: string }) {
  if (!summary?.enabled) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-zinc-100 px-2.5 py-1 font-mono text-xs text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
        <span className="h-1.5 w-1.5 rounded-full bg-zinc-400" />
        checking
      </span>
    );
  }
  const stages = summary.stages.filter((s) => s.symbol === symbol);
  const stale = stages.find((s) => s.status === "stale");
  const produce = stages.find((s) => s.name === "produce");
  const features = stages.find((s) => s.name === "features");

  if (stale) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-red-50 px-2.5 py-1 font-mono text-xs text-red-600 dark:bg-red-950/40 dark:text-red-300">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-500" />
        stale · {fmtAge(stale.age_seconds)}
      </span>
    );
  }
  const venue = produce?.venue ? ` · ${produce.venue}` : "";
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-50 px-2.5 py-1 font-mono text-xs text-emerald-600 dark:bg-emerald-950/40 dark:text-emerald-300">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
      live{venue} · {fmtAge(features?.age_seconds ?? null)}
    </span>
  );
}

/**
 * Per-stage freshness strip for one symbol, Observatory-Terminal style: an LED
 * dot (breathing when fresh) per pipeline stage with its event-time age. A
 * cyan pulse travels across the strip on every state change, so the terminal
 * visibly reacts the moment any stage advances.
 */
export function PipelineHealth({ summary, symbol }: { summary: HealthSummary | null; symbol: string }) {
  const stages = useMemo(
    () =>
      STAGE_ORDER.map((name) =>
        summary?.stages.find((s) => s.symbol === symbol && s.name === name)
      ),
    [summary, symbol]
  );
  const anyStale = stages.some((s) => s?.status === "stale");
  const warming = !summary?.enabled || stages.some((s) => s?.status === "warming");

  const overall = anyStale
    ? {
        label: "degraded",
        cls: "bg-red-50 text-red-600 dark:bg-red-950/40 dark:text-red-300",
      }
    : warming
      ? {
          label: "warming",
          cls: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300",
        }
      : {
          label: "operational",
          cls: "bg-emerald-50 text-emerald-600 dark:bg-emerald-950/40 dark:text-emerald-300",
        };

  const pulseKey = useMemo(() => JSON.stringify(stages), [stages]);

  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Pipeline health</h2>
          <span
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-xs font-semibold uppercase tracking-wider ${overall.cls}`}
          >
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            {overall.label}
          </span>
          <LiveBadge summary={summary} symbol={symbol} />
        </div>
        <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
          event-time staleness · alert ≥ {summary ? fmtAge(summary.threshold_seconds) : "—"}
        </div>
      </header>

      <div className="relative overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800">
        <div
          key={pulseKey}
          className="pulse-travel pointer-events-none absolute inset-y-0 left-0 z-10 w-1/3"
        />
        <div className="grid grid-cols-2 gap-px bg-zinc-200 sm:grid-cols-3 lg:grid-cols-5 dark:bg-zinc-800">
          {stages.map((stage, i) => (
            <StageCell key={`${stage?.name ?? i}-${i}`} stage={stage} />
          ))}
        </div>
      </div>

      <style>{`
        .pulse-travel {
          background: linear-gradient(90deg, transparent, rgba(34,211,238,0.18), transparent);
          animation: pulse-travel 1.3s cubic-bezier(0.4, 0, 0.2, 1) 1;
        }
        @keyframes pulse-travel {
          0% { transform: translateX(-120%); }
          100% { transform: translateX(340%); }
        }
        .led-breathe { animation: led-breathe 2.4s ease-in-out infinite; }
        @keyframes led-breathe {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
      `}</style>
    </section>
  );
}

function StageCell({ stage }: { stage: HealthStage | undefined }) {
  const label = stage ? STAGE_LABEL[stage.name] : "waiting";
  const dot = stage ? DOT[stage.status] : "bg-zinc-300 dark:bg-zinc-700";
  const sub = stage?.name === "produce" ? stage.venue ?? "waiting" : fmtAge(stage?.age_seconds ?? null);
  return (
    <div className="relative flex flex-col gap-1 bg-white px-3 py-2.5 dark:bg-zinc-900">
      <div className="flex items-center gap-1.5">
        <span className={`led-breathe h-2 w-2 rounded-full ${dot}`} />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-600 dark:text-zinc-300">
          {label}
        </span>
      </div>
      <div className="truncate font-mono text-[11px] text-zinc-500 dark:text-zinc-400">{sub}</div>
    </div>
  );
}
