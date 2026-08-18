"""Decisive test: is the live/backtest gap a WIRING bug or a SAMPLE-WINDOW artefact?

scripts/research_fas_invent.py reports Sharpe 1.93 over 55 weekly return
observations. scripts/replay_live_book.py -- which drives the REAL live
classes over the same cache -- only ever produces ~24 tradeable weeks,
because FAS needs ~26 weeks of horizon history plus a 12-week funding-accrual
warm-up before it emits its first selection. The two numbers were therefore
never measured on the same sample, so their difference cannot be attributed
to execution wiring without controlling for the window first.

This runs the RESEARCH book unchanged, then re-scores it on trailing windows
of decreasing length. If the research Sharpe collapses toward the replay's
on a matched ~24-week window, the remaining gap is sample, not wiring. If it
stays high, a real wiring defect is still hiding in the live path.

Run:  uv run python -m scripts.parity_window
"""

from __future__ import annotations

import argparse
import math

import pandas as pd

from scripts.research_fas_clean import _liquidity_mask, _rank_z, fas_scores, load, smb_scores
from scripts.research_fas_invent import rcgo_scores


def metrics(rets: pd.Series) -> dict:
    rets = rets[rets != 0]
    if len(rets) < 3:
        return {"weeks": len(rets), "ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "wealth": 1.0}
    ann = rets.mean() * 52
    vol = rets.std() * math.sqrt(52)
    wealth = float((1 + rets).cumprod().iloc[-1])
    return {
        "weeks": len(rets),
        "ann_ret": ann,
        "ann_vol": vol,
        "sharpe": ann / vol if vol > 0 else 0.0,
        "wealth": wealth,
    }


def weekly_returns(close_w, fas, smb, rcgo, symbols, quintile=0.20, cost_bps=10.0,
                   w_rcgo=0.5, rcgo_dir=1) -> pd.Series:
    """Research book's weekly return series (same construction as
    research_fas_invent.backtest_continuous, exposed as a series so it can be
    re-scored on sub-windows)."""
    weeks = fas.index
    fwd = close_w[symbols].shift(-1) / close_w[symbols] - 1.0
    base = (fas[symbols] + smb[symbols]).apply(_rank_z)
    tilt = (w_rcgo * rcgo_dir * rcgo[symbols]).apply(_rank_z)
    score = (base + tilt).apply(_rank_z)
    rets, pos = [], None
    idx = []
    for w in weeks[:-1]:
        s = score.loc[w].dropna()
        if len(s) < 8:
            pos = None
            continue
        ranked = s.sort_values()
        n = max(2, int(round(quintile * len(ranked))))
        longs, shorts = ranked.index[-n:], ranked.index[:n]
        w_pos = pd.Series(0.0, index=symbols)
        w_pos[longs], w_pos[shorts] = 1.0 / n, -1.0 / n
        r = float((w_pos * fwd.loc[w]).reindex(symbols).sum(skipna=True))
        if pos is not None:
            r -= cost_bps / 1e4 * float((w_pos.reindex(pos.index).fillna(0) - pos).abs().sum())
        rets.append(r)
        idx.append(w)
        pos = w_pos
    return pd.Series(rets, index=idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/quant_cache/asym_warm_start.json.binance")
    ap.add_argument("--w-rcgo", type=float, default=0.5)
    ap.add_argument("--rcgo-dir", type=int, default=1)
    ap.add_argument("--replay-weeks", type=int, default=24,
                    help="tradeable weeks the live replay actually produces")
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    symbols = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, symbols)
    smb = smb_scores(vw, symbols)
    rcgo = rcgo_scores(dcl, dvl, aw, fas.index, symbols)

    rets = weekly_returns(cw, fas, smb, rcgo, symbols, w_rcgo=a.w_rcgo, rcgo_dir=a.rcgo_dir)
    nz = rets[rets != 0]

    print("=== RESEARCH BOOK re-scored on trailing sub-windows ===")
    print("(same construction, same data -- only the evaluation window changes)\n")
    print(f"{'window':>22} {'weeks':>6} {'ann_ret':>9} {'ann_vol':>8} {'Sharpe':>8} {'wealth':>8}")
    print("-" * 68)
    for label, series in [
        ("FULL (published 1.93)", nz),
        (f"last {a.replay_weeks * 3} weeks", nz.tail(a.replay_weeks * 3)),
        (f"last {a.replay_weeks * 2} weeks", nz.tail(a.replay_weeks * 2)),
        (f"last {a.replay_weeks} wks (replay-matched)", nz.tail(a.replay_weeks)),
        (f"first {a.replay_weeks} weeks", nz.head(a.replay_weeks)),
    ]:
        m = metrics(series)
        print(
            f"{label:>22} {m['weeks']:>6} {m['ann_ret'] * 100:>8.2f}% "
            f"{m['ann_vol'] * 100:>7.1f}% {m['sharpe']:>8.2f} {m['wealth']:>8.3f}"
        )

    print("\nVERDICT")
    full = metrics(nz)["sharpe"]
    matched = metrics(nz.tail(a.replay_weeks))["sharpe"]
    print(f"  research Sharpe, FULL sample            : {full:.2f}")
    print(f"  research Sharpe, replay-matched window  : {matched:.2f}")
    print(f"  delta attributable to SAMPLE WINDOW     : {matched - full:+.2f}")
    print(
        "\n  If the matched-window Sharpe is near the replay's, the remaining\n"
        "  live/backtest gap is the evaluation period, not execution wiring.\n"
        "  If it stays high, a wiring defect is still unaccounted for."
    )


if __name__ == "__main__":
    main()
