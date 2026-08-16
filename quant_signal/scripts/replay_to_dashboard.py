"""Accelerated replay of REAL history through the live pipeline, onto the dashboard.

A weekly-horizon strategy cannot be proven live in an afternoon: positions are
held ~168h, and closing faster loses to fees by arithmetic. But the thing that
actually takes a week is CALENDAR time, not compute. Replaying recorded market
data through the production code path is the established substitute --
"reproducing market conditions by replaying data in the original sequence and
timing" (QuestDB), and "validation ... by replaying historical order flow,
recording simulated snapshots" (agent-based market-replay literature).

scripts/replay_live_book.py already does this, but into an in-memory store, so
it only prints a summary. This variant writes the SAME payloads to the SAME
Redis keys the dashboard reads, paced so positions visibly open, mark to
market, and close on localhost:3000/signal.

  * Uses the REAL AsymSignal + PaperExecutionSimulator (same classes as live).
  * venue=None (paper fills). It must NEVER place Bybit orders for historical
    prices -- those would be real orders at prices years out of date.
  * Writes under a REPLAY key prefix by default so the genuine live ledger is
    not polluted. --live-keys overwrites the real dashboard keys instead (the
    live daemons should be stopped first; the script refuses otherwise).

Run:
    uv run python -m scripts.replay_to_dashboard --speed 40
    uv run python -m scripts.replay_to_dashboard --live-keys --speed 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

CACHE = "/tmp/quant_cache/asym_warm_start.json.binance"


def _live_daemons_running() -> list[str]:
    out = []
    for name in ("stream.execution", "stream.asym_signal"):
        r = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
        if r.stdout.strip():
            out.append(name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--speed", type=float, default=40.0,
                    help="windows per second (higher = faster replay)")
    ap.add_argument("--weeks", type=int, default=54, help="most recent weeks to replay")
    ap.add_argument("--rebalance-h", type=int, default=168)
    ap.add_argument("--rcgo-w", type=float, default=1.0)
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--taker-bps", type=float, default=2.75)
    ap.add_argument("--live-keys", action="store_true",
                    help="write to the REAL dashboard keys (live daemons must be stopped)")
    a = ap.parse_args()

    os.environ.setdefault("QUANT_CGO_DIR", "1")
    os.environ.setdefault("QUANT_CGO_L", "7")
    os.environ.setdefault("QUANT_REGIME_OFF", "1")
    os.environ.setdefault("QUANT_SMB_OFF", "0")
    os.environ.setdefault("QUANT_FACC_OFF", "1")
    os.environ.setdefault("QUANT_RCGO_W", str(a.rcgo_w))
    os.environ.setdefault("QUANT_RCGO_DIR", "1")
    os.environ.setdefault("QUANT_RESEARCH_PARITY", "1")
    os.environ.setdefault("QUANT_CGO_GH", "1")
    os.environ.setdefault("QUANT_TRAIL_OFF", "1")

    from config.settings import csv_list, get_settings
    from stream.asym_signal import AsymSignal, prediction_key
    from stream.execution import PaperExecutionSimulator
    from stream.kv import RedisKV

    settings = get_settings()

    if a.live_keys:
        running = _live_daemons_running()
        if running:
            print(
                "REFUSING: --live-keys would fight the running daemons "
                f"({', '.join(running)}) for the same Redis keys.\n"
                "Stop them first:\n"
                "  launchctl unload ~/Library/LaunchAgents/com.quantsignal.execution.plist\n"
                "  launchctl unload ~/Library/LaunchAgents/com.quantsignal.signal.plist",
                file=sys.stderr,
            )
            raise SystemExit(2)
        exec_prefix = settings.stream_redis_execution_prefix
        pred_prefix = settings.stream_asym_prediction_prefix
    else:
        exec_prefix = "execution:replay:1h"
        pred_prefix = "prediction:replay:asym:1h"

    kv = RedisKV(settings.stream_redis_url)
    universe = csv_list(settings.stream_xs_universe)
    cache = json.load(open(a.cache))
    bars, funding = cache["bars"], cache["funding"]
    close_idx = {
        s: {int(r[0]): (float(r[1]), float(r[2] or 0.0)) for r in bars.get(s, [])}
        for s in universe
    }

    sig = AsymSignal(
        kv,
        prediction_prefix=pred_prefix,
        universe=universe,
        rebalance_h=a.rebalance_h,
        quintile=settings.stream_asym_quintile,
        min_symbols=settings.stream_asym_min_symbols,
        regime=False,
        regime_slow_days=settings.stream_asym_regime_slow_days,
        market_symbol=settings.stream_asym_market_symbol,
        horizons=settings.stream_asym_horizons,
        accrual_weeks=settings.stream_asym_accrual_weeks,
        smb_weeks=settings.stream_asym_smb_weeks,
        use_facc=False,
        use_rev=False,
    )
    sig._replay = True
    import stream.asym_signal as _m

    _m.logger.warning = lambda *x, **k: None
    _m.logger.info = lambda *x, **k: None

    sim = PaperExecutionSimulator(
        kv,
        execution_prefix=exec_prefix,
        prediction_prefix=pred_prefix,
        notional_usd=a.notional,
        slippage_bps=0.0,
        taker_fee_bps=a.taker_bps,
        window_ms=settings.stream_window_ms,
        venue=None,  # PAPER ONLY -- never send historical prices to a real venue
        hold_until_decay=True,
        max_hold_h=a.rebalance_h,
        durable_log=False,  # never write historical fills to the live ledger
    )

    sig._funding = {s: [(int(ms), float(r)) for ms, r in (funding.get(s) or [])] for s in universe}
    sig._closes = {s: deque() for s in universe}

    all_w = sorted({w for s in universe for w in close_idx[s]})
    hour = 3_600_000
    play = all_w[-(a.weeks * a.rebalance_h) :] if a.weeks else all_w
    warm = all_w[: len(all_w) - len(play)]
    for w in warm:
        for s in universe:
            cv = close_idx[s].get(w)
            if cv:
                sig._record(s, w, cv[0], cv[1])

    print(f"warm-up: {len(warm)} windows seeded")
    print(f"replaying {len(play)} hourly windows ({len(play) / 168:.0f} weeks) "
          f"at {a.speed:.0f} windows/sec -> ~{len(play) / a.speed / 60:.1f} min")
    print(f"writing to: {exec_prefix}:<SYMBOL>")
    if not a.live_keys:
        print("(replay keys -- pass --live-keys to drive the real dashboard)")
    print()

    delay = 1.0 / a.speed if a.speed > 0 else 0.0
    t0 = time.time()
    for i, w in enumerate(play):
        msgs = []
        for s in universe:
            cv = close_idx[s].get(w)
            if cv:
                msgs.append({"symbol": s, "close": cv[0], "volume": cv[1], "window_end_ms": w})
        for m in msgs:
            sig.handle(m)
        for m in msgs:
            sim.handle(m)
        if i % 168 == 0:
            trades = sum(sim._n_trades.values())
            pnl = sum(sim._realized_pnl.values())
            openp = sum(1 for p in sim._position.values() if p)
            print(
                f"  week {i // 168:3d}/{len(play) // 168}  "
                f"open={openp:2d}  closed={trades:4d}  realized=${pnl:>10,.2f}",
                flush=True,
            )
        if delay:
            time.sleep(delay)

    trades = sum(sim._n_trades.values())
    wins = sum(sim._n_wins.values())
    pnl = sum(sim._realized_pnl.values())
    print(
        f"\nDONE in {time.time() - t0:.0f}s -- {trades} closed trades, "
        f"win rate {wins / trades:.1%}, realized ${pnl:,.2f}"
        if trades
        else "\nDONE -- no trades"
    )


if __name__ == "__main__":
    main()
