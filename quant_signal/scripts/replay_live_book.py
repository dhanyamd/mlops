"""Historical replay of the EXACT live FAS+SMB book (AsymSignal + PaperExecutionSimulator)
over the warm-start cache, to validate the open/close logic end-to-end on real past data.

Uses the SAME production classes the daemons use, with the SAME live config
(CGO dir=-1, SMB off, FCARRY+TSMOM on, regime off, hourly rebalance). Paper
venue (no real orders) — this validates the close DECISIONS (entry/exit/cost-band/
hold-until-decay), which are identical to the live Bybit book; only fill pricing
differs. In-memory KV so it never touches the live Redis ledger.

Run: python scripts/replay_live_book.py
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = "/tmp/quant_cache/asym_warm_start.json"

from stream.kv import KVStore  # module-level: MemKV subclasses it; reads no env

# NOTE: stream.asym_signal / stream.execution are imported INSIDE main() AFTER
# os.environ is set, because asym_signal.py reads QUANT_* env vars at MODULE
# IMPORT time (so the replay flags actually reach the live code path).

_BARS = " ▁▂▃▄▅▆▇█"


def sparkline(series: list[float], width: int = 60) -> str:
    if not series:
        return ""
    lo, hi = min(series), max(series)
    if hi <= lo:
        return "─" * width
    step = max(1, len(series) // width)
    out = []
    for i in range(0, len(series), step):
        v = series[i]
        out.append(_BARS[min(len(_BARS) - 1, int((v - lo) / (hi - lo) * (len(_BARS) - 1)))])
    return "".join(out)


class MemKV(KVStore):
    """In-memory KV so the replay never touches the live Redis ledger."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get_json(self, key: str):
        return self._d.get(key)

    def set_json(self, key: str, val) -> None:
        self._d[key] = val  # store raw — no JSON round-trip (replay speed)

    def get(self, key: str):
        return self._d.get(key)

    def set(self, key: str, val) -> None:
        self._d[key] = val


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay live FAS+SMB book over warm cache")
    parser.add_argument("--max-windows", type=int, default=0, help="cap to last N windows (0=all)")
    parser.add_argument(
        "--rebalance-h",
        type=int,
        default=24,
        help="rebalance hours (24=live default, 168=weekly validated edge)",
    )
    parser.add_argument(
        "--regime", action="store_true", help="enable BTC regime gate (default off in live)"
    )
    parser.add_argument(
        "--smb", action="store_true", default=True, help="enable SMB size leg (ON in live)"
    )
    parser.add_argument(
        "--fcarry", action="store_true", default=False, help="FACC+FCARRY leg (OFF in live)"
    )
    parser.add_argument(
        "--tsmom", action="store_true", default=False, help="TSMOM leg (OFF in live)"
    )
    parser.add_argument("--cgo-dir", type=int, default=-1, help="CGO filter direction")
    parser.add_argument("--rcgo-w", type=float, default=0.0, help="RCGO₿ blend weight (0=off)")
    parser.add_argument("--rcgo-dir", type=int, default=1, help="RCGO₿ tilt direction")
    parser.add_argument(
        "--rcgo-ortho", action="store_true", default=True, help="orthogonalize CGO on carry"
    )
    args = parser.parse_args()

    os.environ.update(
        {
            "QUANT_CGO_DIR": str(args.cgo_dir),
            "QUANT_CGO_L": "7",
            "QUANT_CGO_Q": "0.3",
            "QUANT_FCARRY_ON": "1" if args.fcarry else "0",
            "QUANT_REGIME_OFF": "0" if args.regime else "1",
            "QUANT_SMB_OFF": "0" if args.smb else "1",
            "QUANT_TSMOM_ON": "1" if args.tsmom else "0",
            "QUANT_RCGO_W": str(args.rcgo_w),
            "QUANT_RCGO_DIR": str(args.rcgo_dir),
            "QUANT_RCGO_ORTHO": "1" if args.rcgo_ortho else "0",
        }
    )

    # Imports AFTER env set: asym_signal.py reads QUANT_* at import time.
    from config.settings import csv_list, get_settings
    from stream.asym_signal import AsymSignal, prediction_key
    from stream.execution import PaperExecutionSimulator
    from stream.kv import KVStore

    settings = get_settings()
    kv = MemKV()
    universe = csv_list(settings.stream_xs_universe)

    cache = json.load(open(CACHE))
    bars = cache["bars"]
    funding = cache["funding"]

    # Index closes/volume by (symbol, window) for O(1) lookup.
    close_idx: dict[str, dict[int, tuple[float, float]]] = {}
    for s in universe:
        close_idx[s] = {
            int(row[0]): (float(row[1]), float(row[2] or 0.0)) for row in bars.get(s, [])
        }

    sig = AsymSignal(
        kv,
        prediction_prefix=settings.stream_asym_prediction_prefix,
        universe=universe,
        rebalance_h=args.rebalance_h,  # 1=hourly live, 168=weekly validated edge
        quintile=settings.stream_asym_quintile,
        min_symbols=settings.stream_asym_min_symbols,
        regime=False,  # QUANT_REGIME_OFF=1 -> observation
        regime_slow_days=settings.stream_asym_regime_slow_days,
        market_symbol=settings.stream_asym_market_symbol,
        horizons=settings.stream_asym_horizons,
        accrual_weeks=settings.stream_asym_accrual_weeks,
        smb_weeks=settings.stream_asym_smb_weeks,
        use_facc=args.fcarry,  # FAS+SMB+CGO only when --fcarry off
        use_rev=False,
    )
    sig._replay = True  # skip network funding fetches
    # Silence noisy factor-insufficiency warnings (expected during warm-up).
    import stream.asym_signal as _asym_mod

    _asym_mod.logger.warning = lambda *a, **k: None  # type: ignore[attr-defined]
    from stream.execution import logger as _exec_logger  # type: ignore

    _exec_logger.warning = lambda *a, **k: None  # type: ignore[attr-defined]

    full_windows = sorted({w for s in universe for w in close_idx[s]})
    n_cap = args.max_windows if args.max_windows and args.max_windows > 0 else len(full_windows)
    n_cap = min(n_cap, len(full_windows))
    # Warm-up portion is pre-seeded into _closes so factors have full lookback;
    # only the trailing n_cap windows run the (expensive) handle() loop.
    warmup = full_windows[: len(full_windows) - n_cap]
    trailing = full_windows[len(full_windows) - n_cap :]

    sig._funding = {s: [(int(ms), float(r)) for ms, r in (funding.get(s) or [])] for s in universe}
    sig._closes = {s: deque() for s in universe}
    sig._last_week = None
    for w in warmup:
        for s in universe:
            cv = close_idx[s].get(w)
            if cv is None:
                continue
            sig._record(s, w, cv[0], cv[1])
    print(
        f"warm-up seeded {len(warmup)} windows; replaying trailing {len(trailing)} (paper venue)",
        flush=True,
    )

    sim = PaperExecutionSimulator(
        kv,
        execution_prefix=settings.stream_redis_execution_prefix,
        prediction_prefix=settings.stream_asym_prediction_prefix,
        notional_usd=settings.stream_execution_notional_usd,
        slippage_bps=settings.stream_execution_slippage_bps,
        taker_fee_bps=settings.stream_execution_taker_fee_bps,
        ledger_maxlen=settings.stream_execution_ledger_maxlen,
        max_trades=settings.stream_execution_max_trades,
        window_ms=settings.stream_window_ms,
        venue=None,  # PAPER — validates close logic, no real orders
        cost_filter_lambda=settings.stream_execution_cost_filter_lambda,
        hold_until_decay=settings.stream_execution_hold_until_decay,
        max_hold_h=(settings.stream_execution_max_hold_h or args.rebalance_h),
        signal_max_stale_windows=settings.stream_execution_signal_max_stale_windows,
    )

    windows = trailing
    total = len(windows)
    print(f"replaying {total} hourly windows over {len(universe)} symbols (paper venue)...")

    equity_curve: list[float] = []
    weekly_equity: list[float] = []
    last_wk: int | None = None
    prev_equity: float = 0.0
    for i, w in enumerate(windows):
        for s in universe:
            cv = close_idx[s].get(w)
            if cv is None:
                continue
            close, vol = cv
            sig.handle({"symbol": s, "close": close, "volume": vol, "window_end_ms": w})
        for s in universe:
            cv = close_idx[s].get(w)
            if cv is None:
                continue
            close, vol = cv
            sim.handle({"symbol": s, "close": close, "volume": vol, "window_end_ms": w})
        equity_curve.append(
            sum(sim._realized_pnl.values())
            + sum(p.get("unrealized_pnl", 0.0) or 0.0 for p in sim._position.values() if p)
        )
        wk = w // (168 * 3_600_000)
        if last_wk is None:
            last_wk = wk
        elif wk != last_wk:
            # capture equity at END of the prior week (holding value, not the
            # flat rebalance instant), then advance.
            weekly_equity.append(prev_equity)
            last_wk = wk
        prev_equity = equity_curve[-1]
        if i % 1000 == 0:
            print(f"  window {i}/{total}  closed_trades={sum(sim._n_trades.values())}", flush=True)

    # ── Sharpe from the book mark-to-market equity curve ────────────────────────
    rets = [
        equity_curve[k] - equity_curve[k - 1]
        for k in range(1, len(equity_curve))
        if equity_curve[k - 1] != 0
    ]
    n = len(rets)
    if n > 1:
        mean_r = sum(rets) / n
        var = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
        sharpe_h = (mean_r / math.sqrt(var)) * math.sqrt(24 * 365) if var > 0 else 0.0
    else:
        sharpe_h = 0.0

    # ── Sharpe on WEEKLY portfolio returns (research methodology, √52) ──────────
    # weekly_equity is CUMULATIVE P&L in dollars starting at 0, so dividing a
    # weekly change by the previous cumulative value is not a return: at $5
    # cumulative a $10 move reads as +200%, and near zero it explodes. The
    # `!= 0` guard then silently dropped every pre-profit week, which is why
    # this reported only ~24 observations out of ~110 replayed weeks.
    # A return needs CAPITAL in the denominator: the book runs `notional` per
    # position across a long+short quintile, so gross deployed capital is
    # notional * (2 * quintile * universe).
    n_slots = max(1, int(round(2 * settings.stream_asym_quintile * len(universe))))
    gross_capital = settings.stream_execution_notional_usd * n_slots
    # Score only the ACTIVE period -- from the first week the book actually
    # deployed capital. Weeks before funding data exists produce no positions
    # and no P&L; including them pads the sample with structural zeros that
    # drag the mean toward zero and are not a property of the strategy.
    _first = next(
        (k for k in range(1, len(weekly_equity)) if weekly_equity[k] != weekly_equity[k - 1]),
        len(weekly_equity),
    )
    weekly_rets = [
        (weekly_equity[k] - weekly_equity[k - 1]) / gross_capital
        for k in range(max(1, _first), len(weekly_equity))
    ]
    nw = len(weekly_rets)
    if nw > 1:
        mean_w = sum(weekly_rets) / nw
        var_w = sum((r - mean_w) ** 2 for r in weekly_rets) / (nw - 1)
        sharpe_w = (mean_w / math.sqrt(var_w)) * math.sqrt(52) if var_w > 0 else 0.0
    else:
        sharpe_w = 0.0

    tot_real = sum(sim._realized_pnl.values())
    tot_trades = sum(sim._n_trades.values())
    tot_wins = sum(sim._n_wins.values())
    dir_count: dict = {}
    for s in universe:
        p = kv.get_json(prediction_key(settings.stream_asym_prediction_prefix, s))
        d = p.get("direction") if isinstance(p, dict) else None
        dir_count[d] = dir_count.get(d, 0) + 1
    print("FINAL prediction directions:", dir_count)

    print("\n=== REPLAY RESULTS (paper venue, EXACT live logic) ===")
    print(f"windows replayed      : {total}")
    print(f"total closed trades  : {tot_trades}")
    if tot_trades:
        print(f"total wins           : {tot_wins}  win_rate={tot_wins / tot_trades:.1%}")
    print(f"total realized PnL    : ${tot_real:,.2f}")
    print(f"book Sharpe (ann., hourly Mark-to-Mkt): {sharpe_h:.2f}")
    print(f"book Sharpe (ann., WEEKLY returns √52): {sharpe_w:.2f}  [{nw} weekly obs]")
    print(f"(notional ${settings.stream_execution_notional_usd:.0f}/position, {tot_trades} trades)")
    print(
        f"\nconfig: regime={'ON' if args.regime else 'OFF'} smb={'ON' if args.smb else 'OFF'} "
        f"fcarry={'ON' if args.fcarry else 'OFF'} tsmom={'ON' if args.tsmom else 'OFF'} "
        f"cgo_dir={args.cgo_dir} rebal_h={args.rebalance_h}"
    )
    eq = [round(v, 2) for v in equity_curve]
    print(f"equity curve (${settings.stream_execution_notional_usd:.0f}/pos, cum PnL):")
    print(f"  {sparkline(eq)}")
    print(f"  min=${min(eq):,.2f}  max=${max(eq):,.2f}  end=${eq[-1] if eq else 0:,.2f}")
    print("\nPer-symbol (traded):")
    for s in sorted(universe):
        if sim._n_trades.get(s, 0) > 0:
            print(
                f"  {s:10s} trades={sim._n_trades[s]:3d} "
                f"wins={sim._n_wins.get(s, 0):3d} "
                f"realized=${sim._realized_pnl.get(s, 0):8.2f}"
            )
    print("\nSample fills (first 12 across symbols):")
    cnt = 0
    for s in sorted(universe):
        for f in sim._ledger.get(s, deque()):
            print(
                f"  {s:10s} {f['side']:5s} entry={f['entry_price']:.5f} "
                f"exit={f['exit_price']:.5f} net=${f['net_pnl']:+.2f} "
                f"({f['bars_held']}h)"
            )
            cnt += 1
            if cnt >= 12:
                break
        if cnt >= 12:
            break


if __name__ == "__main__":
    main()
