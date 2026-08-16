"use client";

import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Execution, Portfolio, PortfolioRow } from "@/lib/api";
import { useMarketStream } from "@/lib/useMarketStream";

const money = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : `${v >= 0 ? "+" : "−"}$${Math.abs(v).toFixed(digits)}`;

const pct1 = (v: number | null | undefined) =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;

const POS_STYLE: Record<string, string> = {
  LONG: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  SHORT: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  FLAT: "bg-zinc-100 text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300",
};

function PositionBadge({ side }: { side: string }) {
  return (
    <span
      className={`inline-block rounded-md px-2.5 py-1 font-mono text-xs font-semibold tracking-wider ${
        POS_STYLE[side] ?? POS_STYLE.FLAT
      }`}
    >
      {side}
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

function fmtTime(ms: number | null | undefined) {
  if (ms == null) return "—";
  const d = new Date(ms);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Execution book for one symbol: fills on the predictor's real signals. On the
 * "paper" venue fills are simulated (market fill at the next window's close,
 * adverse-side slippage, taker fee both legs) and honestly labeled "not live".
 * On the "bybit-demo" venue the same state machine drives real market orders
 * on Bybit's free Demo account (virtual USDT) — fill prices/fees come from the
 * venue, still not live money. P&L is always honest about which venue it came
 * from; the fill ledger is the primary artifact, per the Sequence/ordersim
 * research ("the stream is the research object").
 */
export function ExecutionPanel({
  execution,
  portfolio,
  symbol,
}: {
  execution: Execution | null;
  portfolio: Portfolio | null;
  symbol?: string;
}) {
  const venueDemo = execution?.venue === "bybit-demo";
  const equityData = useMemo(
    () =>
      (execution?.equity ?? [1.0]).map((e, i) => ({
        i,
        ret: e - 1,
        zero: 0,
      })),
    [execution]
  );

  const pos = execution?.position;
  const posTone = pos
    ? pos.side === "LONG"
      ? "text-emerald-500"
      : "text-red-500"
    : "";

  // Live tick: the execution engine only marks positions to market once per
  // 1h bar close (stream/execution.py _advance). Between bars, re-mark the
  // open position against the live WebSocket tape (~20s cadence) so the panel
  // visibly moves instead of sitting frozen for up to an hour. Same formula
  // as the backend's _mark_pnl (pre-fees), computed client-side only for
  // display — the actual settled P&L still comes from the hourly bar.
  const { bars: liveBars, connected: liveConnected } = useMarketStream(symbol ?? "BTCUSDT");
  const livePrice = liveBars.length > 0 ? liveBars[liveBars.length - 1].close : null;
  const liveUnrealized =
    pos && livePrice != null
      ? pos.side === "LONG"
        ? pos.qty * (livePrice - pos.entry_price)
        : pos.qty * (pos.entry_price - livePrice)
      : null;
  const liveTone = liveUnrealized == null ? "" : liveUnrealized >= 0 ? "text-emerald-500" : "text-red-500";
  const netTone =
    (execution?.net_pnl ?? 0) >= 0 ? "text-emerald-500" : "text-red-500";
  const winTone =
    (execution?.win_rate ?? 0) >= 0.5 ? "text-emerald-500" : "text-red-500";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="inline-flex items-center gap-2 rounded-md bg-cyan-100 px-2.5 py-1 font-mono text-[11px] font-bold uppercase tracking-widest text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300">
          {venueDemo ? "Bybit Demo · virtual funds · not live money" : "Simulated · not live"}
        </span>
        <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
          {execution
            ? `${execution.signals_skipped} signal${execution.signals_skipped === 1 ? "" : "s"} skipped · ${execution.n_trades} closed`
            : "warming up"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          label={venueDemo ? "net P&L (demo)" : "net P&L (simulated)"}
          value={money(execution?.net_pnl)}
          tone={netTone}
        />
        <Stat label="realized" value={money(execution?.realized_pnl)} />
        <Stat label="unrealized" value={money(execution?.unrealized_pnl)} tone={posTone} />
        <Stat label="win rate" value={execution ? pct1(execution.win_rate) : "—"} tone={winTone} />
        <Stat label="closed trades" value={String(execution?.n_trades ?? 0)} />
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <PositionBadge side={pos?.side ?? "FLAT"} />
        {pos ? (
          <div className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-zinc-500 dark:text-zinc-400">
            <span>
              entry {pos.entry_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
            <span>
              mark (1h bar) {pos.mark_price?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? "—"}
            </span>
            <span>qty {pos.qty.toFixed(4)}</span>
            <span className={posTone}>
              unrealized (1h bar) {money(pos.unrealized_pnl)} ({pct1(pos.unrealized_pnl_pct)})
            </span>
          </div>
        ) : (
          <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
            flat · next signal fills at the following window close
          </span>
        )}
        <span className="ml-auto font-mono text-xs text-zinc-500 dark:text-zinc-400">
          {venueDemo
            ? `notional $${execution?.notional_usd.toLocaleString()} · venue Bybit Demo · actual fees`
            : `notional $${execution?.notional_usd.toLocaleString()} · slip ${
                execution?.slippage_bps
              }bps · taker ${execution?.taker_fee_bps}bps`}
        </span>
      </div>

      {pos ? (
        <div className="flex flex-wrap items-center gap-4 rounded-xl border border-cyan-200 bg-cyan-50/60 px-4 py-3 dark:border-cyan-900 dark:bg-cyan-950/20">
          <span
            className={`h-2 w-2 rounded-full ${liveConnected ? "animate-pulse bg-emerald-500" : "bg-zinc-400"}`}
          />
          <span className="font-mono text-[11px] uppercase tracking-widest text-cyan-700 dark:text-cyan-300">
            live tick
          </span>
          <span className="font-mono text-xs text-zinc-600 dark:text-zinc-300">
            price {livePrice != null ? livePrice.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
          </span>
          <span className={`font-mono text-sm font-semibold tabular-nums ${liveTone}`}>
            {liveUnrealized == null ? "—" : money(liveUnrealized)}
          </span>
          <span className="ml-auto font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
            {liveConnected ? "re-marked off the live WebSocket tape (~20s) between hourly bar closes" : "connecting to live tape…"}
          </span>
        </div>
      ) : null}

      <div style={{ height: 220 }} className="w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={equityData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="exec-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
            <XAxis dataKey="i" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={32} />
            <YAxis
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={56}
              tickFormatter={(v: number) => pct1(v)}
              domain={["auto", "auto"]}
            />
            <Tooltip
              formatter={(v) => [pct1(Number(v)), "compounded equity"]}
              labelFormatter={(label) => `window ${String(label)}`}
              contentStyle={{
                background: "rgba(15,15,15,0.9)",
                border: "none",
                borderRadius: 8,
                color: "#eee",
                fontSize: 12,
              }}
            />
            <ReferenceLine y={0} stroke="#71717a" strokeDasharray="4 4" strokeWidth={1} />
            <Area
              type="monotone"
              dataKey="ret"
              stroke="#06b6d4"
              strokeWidth={2}
              fill="url(#exec-fill)"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Fill ledger
          </h3>
          <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
            gross ${execution?.gross_volume.toLocaleString()} · fees{" "}
            {money(execution?.total_fees)} ·{" "}
            {execution?.fees_pct_of_gross_pnl == null
              ? "—"
              : `${execution.fees_pct_of_gross_pnl.toFixed(1)}% of gross P&L`}
          </span>
        </div>
        <FillsTable fills={execution?.fills ?? []} />
      </div>

      {portfolio && portfolio.enabled ? <PortfolioStrip portfolio={portfolio} /> : null}

      <div className="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-[11px] leading-relaxed text-cyan-800 dark:border-cyan-900 dark:bg-cyan-950/40 dark:text-cyan-200">
        <span className="font-semibold text-cyan-900 dark:text-cyan-100">
          {venueDemo
            ? "Real market orders on Bybit Demo (virtual funds) — not live money."
            : "Simulated execution — not live orders."}
        </span>{" "}
        {execution
          ? `${execution.assumptions?.fill_timing ?? "Market fills at the next window's close"}. ${
              execution.assumptions?.cost_model ?? "Adverse-side slippage plus a taker fee on both legs"
            }.`
          : "Market fills at the next window's close; adverse-side slippage plus a taker fee on both legs."}{" "}
        {venueDemo
          ? "Orders are placed against Bybit's Demo order book with virtual USDT, so no real money is at risk and results are demo only. Rejected or unfilled orders are skipped and counted."
          : "Trades are not actually executed, so these results may over- or under-compensate for market factors such as lack of liquidity. No representation is made that any account will or is likely to achieve profits or losses similar to these. Not modeled: margin, funding, partial fills, queue position, and market impact."}
      </div>
    </div>
  );
}

function FillsTable({ fills }: { fills: Execution["fills"] }) {
  if (fills.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-zinc-300 px-4 py-8 text-center text-sm text-zinc-500 dark:border-zinc-700">
        No fills yet — the book opens its first position at the close of the
        window after a signal appears.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-left font-mono text-xs">
        <thead className="border-b border-zinc-200 text-[11px] uppercase tracking-wide text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <tr>
            <th className="px-3 py-2">closed</th>
            <th className="px-3 py-2">side</th>
            <th className="px-3 py-2 text-right">entry</th>
            <th className="px-3 py-2 text-right">exit</th>
            <th className="px-3 py-2 text-right">qty</th>
            <th className="px-3 py-2 text-right">fees</th>
            <th className="px-3 py-2 text-right">net P&L</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
          {fills.slice(0, 8).map((f, i) => (
            <tr key={i} className="tabular-nums">
              <td className="px-3 py-2 text-zinc-500 dark:text-zinc-400">{fmtTime(f.window_end_ms)}</td>
              <td className="px-3 py-2">
                <PositionBadge side={f.side} />
              </td>
              <td className="px-3 py-2 text-right">{f.entry_price.toFixed(2)}</td>
              <td className="px-3 py-2 text-right">{f.exit_price.toFixed(2)}</td>
              <td className="px-3 py-2 text-right">{f.qty.toFixed(4)}</td>
              <td className="px-3 py-2 text-right text-zinc-500">{money(f.fees, 4)}</td>
              <td className={`px-3 py-2 text-right ${f.net_pnl >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                {money(f.net_pnl, 4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PortfolioStrip({ portfolio }: { portfolio: Portfolio }) {
  const total = portfolio.total_pnl ?? 0;
  const tone = total >= 0 ? "text-emerald-500" : "text-red-500";
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
          Portfolio
        </h3>
        <span className={`font-mono text-xs font-semibold tabular-nums ${tone}`}>
          {money(total)} · {portfolio.n_trades ?? 0} trades · fees {money(portfolio.total_fees)}
        </span>
      </div>
      <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
        <table className="w-full text-left font-mono text-xs">
          <thead className="border-b border-zinc-200 text-[11px] uppercase tracking-wide text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            <tr>
              <th className="px-3 py-2">symbol</th>
              <th className="px-3 py-2">position</th>
              <th className="px-3 py-2 text-right">net P&L</th>
              <th className="px-3 py-2 text-right">realized</th>
              <th className="px-3 py-2 text-right">win rate</th>
              <th className="px-3 py-2 text-right">trades</th>
              <th className="px-3 py-2 text-right">return</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {portfolio.rows.map((row) => (
              <PortfolioRow key={row.symbol} row={row} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PortfolioRow({ row }: { row: PortfolioRow }) {
  const side = row.position?.side ?? "FLAT";
  const pnl = row.total_pnl ?? 0;
  return (
    <tr className="tabular-nums">
      <td className="px-3 py-2 font-semibold text-zinc-700 dark:text-zinc-200">{row.symbol}</td>
      <td className="px-3 py-2">
        <PositionBadge side={side} />
      </td>
      <td className={`px-3 py-2 text-right ${pnl >= 0 ? "text-emerald-500" : "text-red-500"}`}>
        {money(pnl)}
      </td>
      <td className="px-3 py-2 text-right text-zinc-500 dark:text-zinc-400">
        {money(row.realized_pnl)}
      </td>
      <td className="px-3 py-2 text-right">{pct1(row.win_rate)}</td>
      <td className="px-3 py-2 text-right">{row.n_trades ?? 0}</td>
      <td className="px-3 py-2 text-right">{pct1(row.total_return)}</td>
    </tr>
  );
}
