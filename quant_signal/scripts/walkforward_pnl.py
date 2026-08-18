"""Historical walk-forward: prove the FAS+SMB+FACC book CLOSES for profit.

Replays the last ~12 weeks of REAL 1h bars (from the warm-start history: live
Bybit closes + keyless Binance funding) through the EXACT production classes the
live book uses -- AsymSignal (FAS_avg + SMB + FACC) + PaperExecutionSimulator --
in PAPER mode (fills at the real historical close, no venue, no Bybit). Each
signal flip / weekly rebalance CLOSES the prior position and records realized
P&L, so this is a faithful, real-data answer to "does the +2 Sharpe actually
close for profit?".

Uses ISOLATED Redis prefixes (wf:pred / wf:exec) so it never touches the live
book's keys. Nothing hardcoded: universe + settings from config, data from the
real warm-start. Run:

  cd /Users/dhanyamd/Projects/mlops/quant_signal && uv run python scripts/walkforward_pnl.py
"""

from __future__ import annotations

import os
import sys

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.asym_signal import _HOUR_MS, AsymSignal
from stream.execution import PaperExecutionSimulator
from stream.kv import FakeKV, RedisKV

logger = get_logger(__name__)

_WEEKS = 12


class _ReplayVenue:
    """Paper venue for historical replay: fills at the REAL historical close.

    The execution engine only records a trade when the venue returns a fill, so
    a real (or realistic) venue is required even in paper mode. We fill at the
    bar's actual close (passed in per symbol as we replay), so realized P&L is
    the model's true closed-trade P&L on real data -- no invented prices.
    """

    def __init__(self) -> None:
        self._close: dict[str, float] = {}

    def set_close(self, symbol: str, price: float) -> None:
        self._close[symbol] = price

    def open_market(self, symbol: str, side: str, notional_usd: float, maker_first: bool = False):
        p = self._close.get(symbol)
        if not p or p <= 0:
            return None
        qty = float(notional_usd) / float(p)
        return {"fill_price": p, "qty": qty, "fees": float(notional_usd) * 1e-4}

    def close_market(self, symbol: str, side: str, qty: float, maker_first: bool = False):
        p = self._close.get(symbol)
        if not p or p <= 0:
            return None
        return {"fill_price": p, "qty": qty, "fees": float(p) * float(qty) * 1e-4}


def main() -> None:
    configure_logging()
    settings = get_settings()
    # In-memory KV for replay: the walk-forward writes ~60k prediction keys;
    # Redis round-trips dominate runtime. FakeKV keeps it hermetic + fast.
    kv = FakeKV() if os.environ.get("QUANT_WF_MEM") == "1" else RedisKV(settings.stream_redis_url)
    universe = csv_list(settings.stream_xs_universe)

    # A/B test rebalance frequency without touching live config. Research says
    # crypto momentum is only profitable at LOW turnover (daily/2-week), not 4h.
    rb = int(os.environ.get("QUANT_REBALANCE_H", settings.stream_xs_rebalance_h))
    sig = AsymSignal(
        kv,
        prediction_prefix="wf:pred",
        universe=universe,
        rebalance_h=rb,
        quintile=settings.stream_asym_quintile,
        min_symbols=settings.stream_asym_min_symbols,
        regime=False,  # observation: trade through bear, same as live demo
        regime_slow_days=settings.stream_asym_regime_slow_days,
        market_symbol=settings.stream_asym_market_symbol,
        horizons=settings.stream_asym_horizons,
        accrual_weeks=settings.stream_asym_accrual_weeks,
        smb_weeks=settings.stream_asym_smb_weeks,
        use_facc=True,
        use_rev=False,
        replay=True,  # historical replay: warm-start seed already covers the window
    )
    print("warm-starting real history (Bybit 1h + keyless Bybit funding)...")
    sig.warm_start(settings)
    print("warm-start done; replaying last %d weeks..." % _WEEKS)

    venue = _ReplayVenue()
    sim = PaperExecutionSimulator(
        kv,
        execution_prefix="wf:exec",
        prediction_prefix="wf:pred",
        notional_usd=settings.stream_execution_notional_usd,
        slippage_bps=settings.stream_execution_slippage_bps,
        taker_fee_bps=settings.stream_execution_taker_fee_bps,
        ledger_maxlen=1000,
        max_trades=10_000,
        window_ms=settings.stream_window_ms,
        venue=venue,  # PAPER: fills at the real historical close
        cost_filter_lambda=settings.stream_execution_cost_filter_lambda,
        hold_until_decay=settings.stream_execution_hold_until_decay,
    )

    # All 1h window boundaries present in the warm-start history.
    windows_set: set[int] = set()
    for s in universe:
        for e, _c, _v in sig._closes.get(s, []):
            windows_set.add(e)
    windows = sorted(w for w in windows_set if w)
    # Robustness A/B: shift the tested window back by N weeks (QUANT_WF_OFFSET_W)
    # so we don't overfit to the single most-recent 12-week slice.
    offset_w = int(os.environ.get("QUANT_WF_OFFSET_W", "0"))
    if offset_w:
        end = windows[-1] - offset_w * 7 * 24 * _HOUR_MS
        windows = [w for w in windows if w <= end]
    cutoff = windows[-1] - _WEEKS * 7 * 24 * _HOUR_MS
    windows = [w for w in windows if w >= cutoff]
    print("replay windows: %d (from %d to %d)" % (len(windows), windows[0], windows[-1]))

    # Precompute window -> (close, volume) per symbol for O(1) replay lookup.
    bars_map: dict[str, dict[int, tuple[float, float]]] = {}
    for s in universe:
        bars_map[s] = {e: (c, v) for e, c, v in sig._closes.get(s, [])}

    # Two-pass per window: (1) signal publishes prediction for w, (2) engine
    # executes at w's close (falls back to prior-window prediction, like live).
    for w in windows:
        for s in universe:
            b = bars_map.get(s, {}).get(w)
            if b:
                sig.handle({"symbol": s, "close": b[0], "volume": b[1], "window_end_ms": w})
        for s in universe:
            b = bars_map.get(s, {}).get(w)
            if b:
                venue.set_close(s, b[0])  # replay fills at this bar's real close
                sim.handle({"symbol": s, "close": b[0], "window_end_ms": w})

    # Aggregate realized results across the universe.
    total_realized = sum(sim._realized_pnl.values())
    total_trades = sum(sim._n_trades.values())
    total_wins = sum(sim._n_wins.values())
    gross = sum(sim._gross_pnl.values())
    fees = sum(sim._total_fees.values())
    rejected = sum(sim._orders_rejected.values())

    # Combined per-trade P&L list for a quick Sharpe of closed trades.
    trade_pnl = []
    for s in universe:
        for f in sim._ledger.get(s, []):
            trade_pnl.append(f["net_pnl"])

    import statistics

    mean_pnl = statistics.fmean(trade_pnl) if trade_pnl else 0.0
    sd = statistics.pstdev(trade_pnl) if len(trade_pnl) > 1 else 0.0
    # Per-trade Sharpe (per-$1000 notional): mean / std of net P&L per trade.
    per_trade_sharpe = (mean_pnl / sd) if sd > 0 else 0.0

    print("\n==================== WALK-FORWARD RESULTS ====================")
    print("replay horizon      : %d weeks of REAL 1h bars" % _WEEKS)
    print("rebalance           : %dh (research: low turnover = profitable)" % rb)
    print("universe            : %d symbols" % len(universe))
    print("model               : FAS_avg + SMB + FACC (regime off / demo)")
    print("notional / trade    : $%.0f" % settings.stream_execution_notional_usd)
    print("-------------------------------------------------------------")
    print("CLOSED trades       : %d" % total_trades)
    print(
        "wins / losses       : %d / %d  (win rate %.1f%%)"
        % (
            total_wins,
            total_trades - total_wins,
            100.0 * total_wins / total_trades if total_trades else 0.0,
        )
    )
    print("REALIZED P&L        : $%.2f" % total_realized)
    print("gross P&L           : $%.2f" % gross)
    print("fees paid           : $%.2f" % fees)
    print("avg per-trade P&L   : $%.2f" % mean_pnl)
    print("per-trade Sharpe    : %.2f" % per_trade_sharpe)
    print("orders rejected     : %d" % rejected)
    print("=============================================================")

    # Show a few sample closed trades so the user SEES real closes.
    shown = 0
    for s in universe:
        for f in list(sim._ledger.get(s, []))[:3]:
            shown += 1
            if shown > 12:
                break
            print(
                "  %s %s  entry=%.4f exit=%.4f net=$%.2f (%.2f%%) bars=%d"
                % (
                    s,
                    f["side"],
                    f["entry_price"],
                    f["exit_price"],
                    f["net_pnl"],
                    f["net_pnl_pct"] * 100.0,
                    f["bars_held"],
                )
            )
        if shown > 12:
            break


if __name__ == "__main__":
    sys.exit(main())
