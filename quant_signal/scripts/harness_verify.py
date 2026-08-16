"""Harness verification: feed the RESEARCH selections through the LIVE executor.

The research book (scripts/research_fas_invent.py) reports Sharpe ~2.28 using
its own vectorised return series. The replay harness
(scripts/replay_live_book.py) drives the live AsymSignal + PaperExecutionSimulator
and reports something far lower. Three things could explain that gap:

  A. the LIVE SIGNAL picks different symbols than the research book
  B. the EXECUTION ENGINE destroys the edge (fees / timing / position mgmt)
  C. the HARNESS METRIC is wrong and neither A nor B is real

This isolates them. It takes the research book's EXACT weekly long/short sets
and pushes them through the real PaperExecutionSimulator, then scores the
result two ways on the same trades:

  * research-style: weekly portfolio returns, equal-weight quintiles, the same
    construction research_fas_invent uses
  * executor-style: the fill ledger the live engine produced

If research-selections + live-executor reproduces ~2.28, the executor and the
metric are sound and the LIVE SIGNAL is the defect (A). If it does not, the
executor or metric is the defect (B/C) and no signal fix would have helped.

Run:  uv run python -m scripts.harness_verify
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from config.settings import get_settings
from scripts.research_fas_clean import _liquidity_mask, _rank_z, fas_scores, load, smb_scores
from scripts.research_fas_invent import rcgo_scores

CACHE = "/tmp/quant_cache/asym_warm_start.json.binance"


def research_weekly_returns(cw, fas, smb, rcgo, symbols, quintile, cost_bps, w_rcgo, rcgo_dir):
    """Exact research construction, returning the weekly return series."""
    fwd = cw[symbols].shift(-1) / cw[symbols] - 1.0
    base = (fas[symbols] + smb[symbols]).apply(_rank_z)
    tilt = (w_rcgo * rcgo_dir * rcgo[symbols]).apply(_rank_z)
    score = (base + tilt).apply(_rank_z)
    rets, idx, pos = [], [], None
    sels = {}
    for w in fas.index[:-1]:
        s = score.loc[w].dropna()
        if len(s) < 8:
            pos = None
            continue
        ranked = s.sort_values()
        n = max(2, int(round(quintile * len(ranked))))
        longs, shorts = list(ranked.index[-n:]), list(ranked.index[:n])
        sels[w] = (longs, shorts)
        w_pos = pd.Series(0.0, index=symbols)
        w_pos[longs], w_pos[shorts] = 1.0 / n, -1.0 / n
        r = float((w_pos * fwd.loc[w]).reindex(symbols).sum(skipna=True))
        if pos is not None:
            r -= cost_bps / 1e4 * float((w_pos.reindex(pos.index).fillna(0) - pos).abs().sum())
        rets.append(r)
        idx.append(w)
        pos = w_pos
    return pd.Series(rets, index=idx), sels


def sharpe(rets: pd.Series) -> tuple[float, float, int]:
    r = rets[rets != 0]
    if len(r) < 3:
        return 0.0, 0.0, len(r)
    ann = r.mean() * 52
    vol = r.std() * math.sqrt(52)
    return (ann / vol if vol > 0 else 0.0), ann, len(r)


def replay_selections_through_executor(sels, cw, symbols, notional, taker_bps):
    """Push the research selections through the real execution engine.

    Uses the research weekly close grid as the price path so the ONLY thing
    under test is the executor: same symbols, same weeks, same prices.
    """
    from stream.execution import PaperExecutionSimulator, execution_key
    from stream.kv import FakeKV
    from stream.predictor import prediction_key

    kv = FakeKV()
    week_ms = 7 * 24 * 3_600_000
    sim = PaperExecutionSimulator(
        kv,
        execution_prefix="execution:verify",
        prediction_prefix="prediction:verify",
        notional_usd=notional,
        slippage_bps=0.0,
        taker_fee_bps=taker_bps,
        window_ms=week_ms,          # one bar == one rebalance week
        hold_until_decay=True,
        max_hold_h=0,
        venue=None,
    )
    weeks = sorted(sels)
    equity, prev = [], 0.0
    for i, w in enumerate(weeks):
        longs, shorts = sels[w]
        wend = (i + 1) * week_ms
        for s in symbols:
            d = "LONG" if s in longs else "SHORT" if s in shorts else "FLAT"
            kv.set_json(
                prediction_key("prediction:verify", s),
                {
                    "symbol": s,
                    "window_end_ms": wend,
                    "predicted_return": 0.5 if d == "LONG" else -0.5 if d == "SHORT" else 0.0,
                    "direction": d,
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
            )
        for s in symbols:
            px = cw[s].get(w)
            if px is None or not np.isfinite(px) or px <= 0:
                continue
            sim.handle(
                {"symbol": s, "close": float(px), "volume": 1.0, "window_end_ms": wend}
            )
        eq = sum(sim._realized_pnl.values()) + sum(
            (p.get("unrealized_pnl") or 0.0) for p in sim._position.values() if p
        )
        equity.append(eq - prev)
        prev = eq
    gross_capital = notional * max(1, sum(len(v[0]) + len(v[1]) for v in sels.values()) // len(sels))
    ex_rets = pd.Series([e / gross_capital for e in equity], index=weeks)
    return ex_rets, sim, gross_capital


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--w-rcgo", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--taker-bps", type=float, default=2.75)
    a = ap.parse_args()

    s = get_settings()
    cw, vw, aw, dcl, dvl = load(a.cache)
    symbols = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, symbols)
    smb = smb_scores(vw, symbols)
    rcgo = rcgo_scores(dcl, dvl, aw, fas.index, symbols)

    r_rets, sels = research_weekly_returns(
        cw, fas, smb, rcgo, symbols, s.stream_asym_quintile, a.cost_bps, a.w_rcgo, 1
    )
    r_sh, r_ann, r_n = sharpe(r_rets)
    print("=== A. RESEARCH book, its own return series (the 2.28 benchmark) ===")
    print(f"    weeks={r_n}  ann_ret={r_ann:.1%}  Sharpe={r_sh:.2f}\n")

    ex_rets, sim, cap = replay_selections_through_executor(
        sels, cw, symbols, s.stream_execution_notional_usd, a.taker_bps
    )
    e_sh, e_ann, e_n = sharpe(ex_rets)
    trades = sum(sim._n_trades.values())
    wins = sum(sim._n_wins.values())
    realized = sum(sim._realized_pnl.values())
    print("=== B. SAME research selections, pushed through the LIVE EXECUTOR ===")
    print(f"    weeks={e_n}  ann_ret={e_ann:.1%}  Sharpe={e_sh:.2f}")
    print(f"    trades={trades}  wins={wins} ({wins / trades:.1%})" if trades else "    trades=0")
    print(f"    realized=${realized:,.2f}  gross capital=${cap:,.0f}\n")

    print("=== VERDICT ===")
    if r_sh <= 0.5:
        print("    Research benchmark itself did not reproduce -- the research")
        print("    construction or data load is the problem, not the executor.")
    elif e_sh >= r_sh * 0.6:
        print("    Executor REPRODUCES the research edge on research selections.")
        print("    => executor + metric are sound; the LIVE SIGNAL is the defect.")
    else:
        print("    Executor DESTROYS the edge even on research selections.")
        print("    => the defect is in EXECUTION or the metric, not the signal.")
        print(f"    (retained {e_sh / r_sh:.0%} of the research Sharpe)")


if __name__ == "__main__":
    main()
