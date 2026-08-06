"use client";

import Link from "next/link";
import { api } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

const tiles = [
  { href: "/market", title: "Market", desc: "Daily OHLCV from GOLD_DAILY_BARS", color: "text-blue-500" },
  { href: "/fundamentals", title: "Fundamentals", desc: "PIT US-GAAP facts as filed (SILVER)", color: "text-emerald-500" },
  { href: "/pead", title: "PEAD", desc: "Post-earnings drift by SUE quintile", color: "text-violet-500" },
  { href: "/macro", title: "Macro", desc: "FRED series (VIX, CPI, rates)", color: "text-amber-500" },
  { href: "/pipeline", title: "Pipeline", desc: "Per-stage latency telemetry", color: "text-rose-500" },
];

export default function DashboardPage() {
  const { data: tickers } = useQuery(["tickers"], () => api.tickers());
  const { data: pead, error: peadError } = useQuery(["pead-home"], () => api.pead("NetIncomeLoss", "1,5,20"));
  const error = peadError;

  const spread20 = pead?.spread["+20d"] ?? null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Quant signal dashboard</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          All numbers come from the live Silver/Gold layers over the FastAPI backend.
        </p>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
          Cannot reach the API ({error}). Start it with{" "}
          <code className="font-mono text-xs">uv run uvicorn api.main:app --port 8000</code>
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {tiles.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm transition hover:border-zinc-300 hover:shadow dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
          >
            <div className={`font-mono text-sm font-semibold ${t.color}`}>{t.title}</div>
            <div className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">{t.desc}</div>
          </Link>
        ))}

        <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="font-mono text-sm font-semibold text-zinc-700 dark:text-zinc-300">Live dataset</div>
          <dl className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-zinc-500 dark:text-zinc-400">Tickers</dt>
              <dd className="tabular-nums font-medium">{tickers?.tickers.length ?? "…"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-zinc-500 dark:text-zinc-400">PEAD events</dt>
              <dd className="tabular-nums font-medium">{pead?.n_events ?? "…"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-zinc-500 dark:text-zinc-400">Q5−Q1 drift (+20d)</dt>
              <dd className="tabular-nums font-medium">
                {pead ? (spread20 === null ? "—" : `${(spread20 * 100).toFixed(2)}%`) : "…"}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}
