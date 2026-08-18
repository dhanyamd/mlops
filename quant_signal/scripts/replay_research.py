"""Validate the LIVE research path end-to-end on history: stream.research_signal
(REAL research_novel ens_mcd) -> stream.research_executor (weekly weight rebal)
over the warm cache, paper venue. Proves the rewired live engine makes money.

Run: uv run python scripts/replay_research.py [--regime]   (--regime = true ENS_MCD_SLOW
slow-gate; needs ~200w history, so default off to trade on the 56w warm cache)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stream.kv import KVStore
from stream.research_executor import ResearchExecutor
from stream.research_signal import ResearchSignal

CACHE = "/tmp/quant_cache/asym_warm_start.json"
_BARS = " ▁▂▃▄▅▆▇█"


class MemKV(KVStore):
    def __init__(self) -> None:
        self._d: dict = {}

    def get_json(self, key):
        return self._d.get(key)

    def set_json(self, key, val) -> None:
        self._d[key] = val

    def get(self, key):
        return self._d.get(key)

    def set(self, key, val) -> None:
        self._d[key] = val


def sparkline(series, width=60):
    if not series:
        return ""
    lo, hi = min(series), max(series)
    if hi <= lo:
        return "─" * width
    out = []
    for i in range(0, len(series), max(1, len(series) // width)):
        v = series[i]
        out.append(_BARS[min(len(_BARS) - 1, int((v - lo) / (hi - lo) * (len(_BARS) - 1)))])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime", action="store_true", help="use true ENS_MCD_SLOW slow-gate (needs ~200w)"
    )
    ap.add_argument("--max-windows", type=int, default=0)
    args = ap.parse_args()

    cache = json.load(open(CACHE))
    universe = list(cache["bars"].keys())

    kv = MemKV()
    sig = ResearchSignal(
        kv, universe=universe, spec=dict(score="ens_mcd", regime=args.regime, regime_mode="slow")
    )
    ex = ResearchExecutor(
        kv, universe=universe, total_capital=8000.0, taker_fee_bps=5.0, venue=None
    )

    for s in universe:
        sig.seed_funding(s, [(int(ms), float(r)) for ms, r in (cache["funding"].get(s) or [])])
        ex._marks[s] = 0.0

    windows = sorted({int(w) for s in universe for w, _ in [(r[0], r) for r in cache["bars"][s]]})
    if args.max_windows and args.max_windows > 0:
        windows = windows[-args.max_windows :]
    total = len(windows)
    print(
        f"replaying {total} windows, regime={'ON' if args.regime else 'OFF'} (paper venue, REAL research signal)"
    )

    equity = []
    last_targets = 0
    for i, w in enumerate(windows):
        for s in universe:
            row = next((r for r in cache["bars"][s] if r[0] == w), None)
            if row is None:
                continue
            close, vol = row[1], row[2] or 0.0
            msg = {"symbol": s, "close": close, "volume": vol, "window_end_ms": w}
            sig.handle(msg)
            ex.handle(msg)
        t = kv.get_json("portfolio:targets:research")
        if t:
            last_targets = len(t.get("targets", {}))
        equity.append(ex.equity())
        if i % 2000 == 0:
            print(
                f"  window {i}/{total} equity=${equity[-1]:,.2f} targets={last_targets}", flush=True
            )

    rets = [equity[k] - equity[k - 1] for k in range(1, len(equity)) if equity[k - 1] != 0]
    n = len(rets)
    sharpe = 0.0
    if n > 1:
        mean_r = sum(rets) / n
        var = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
        sharpe = (mean_r / math.sqrt(var)) * math.sqrt(24 * 365) if var > 0 else 0.0

    tot_real = sum(ex._realized.values())
    tot_trades = sum(ex._n_trades.values())
    tot_wins = sum(ex._n_wins.values())
    eq = [round(v, 2) for v in equity]
    print("\n=== RESEARCH LIVE-PATH REPLAY (paper venue) ===")
    print(
        f"windows={total}  closed_trades={tot_trades}  win_rate={tot_wins / tot_trades:.1%}"
        if tot_trades
        else f"windows={total} trades=0"
    )
    print(f"realized PnL=${tot_real:,.2f}  book Sharpe(ann)={sharpe:.2f}")
    print(f"equity curve (start=$0):\n  {sparkline(eq)}")
    print(f"  min=${min(eq):,.2f} max=${max(eq):,.2f} end=${eq[-1] if eq else 0:,.2f}")


if __name__ == "__main__":
    main()
