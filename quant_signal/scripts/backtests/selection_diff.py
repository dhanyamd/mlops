"""Decisive parity test: does the LIVE signal pick the same basket as RESEARCH?

Everything downstream -- fills, fees, holding period -- is irrelevant if the
live book is not selecting the same symbols the Sharpe-2.28 research book
selects. This drives the REAL AsymSignal class over the same warm-start cache
the research uses, captures its long/short selection at every weekly
rebalance, computes the research selection for the same weeks, and reports
overlap.

Perfect overlap => the divergence is purely execution mechanics.
Poor overlap  => the live SIGNAL is the bug, and no execution fix can recover
                 the edge.

Run:  uv run python -m scripts.selection_diff
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import pandas as pd

CACHE = "/tmp/quant_cache/asym_warm_start.json.binance"


def research_selection(cw, fas, smb, rcgo, symbols, quintile=0.20, w_rcgo=0.5, rcgo_dir=1):
    """Research book's long/short sets per week (research_fas_invent construction)."""
    from scripts.research_fas_clean import _rank_z

    base = (fas[symbols] + smb[symbols]).apply(_rank_z)
    tilt = (w_rcgo * rcgo_dir * rcgo[symbols]).apply(_rank_z)
    score = (base + tilt).apply(_rank_z)
    out = {}
    for w in fas.index:
        s = score.loc[w].dropna()
        if len(s) < 8:
            continue
        ranked = s.sort_values()
        n = max(2, int(round(quintile * len(ranked))))
        out[w] = (set(ranked.index[-n:]), set(ranked.index[:n]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--rebalance-h", type=int, default=168)
    ap.add_argument("--rcgo-w", type=float, default=1.0)
    ap.add_argument("--weeks", type=int, default=12, help="most recent weeks to compare")
    a = ap.parse_args()

    # asym_signal reads QUANT_* at import time -- set before importing it.
    os.environ.update(
        {
            "QUANT_CGO_DIR": "1",
            "QUANT_CGO_L": "7",
            "QUANT_CGO_Q": "0.3",
            "QUANT_REGIME_OFF": "1",
            "QUANT_SMB_OFF": "0",
            "QUANT_FACC_OFF": "1",
            "QUANT_RCGO_W": str(a.rcgo_w),
            "QUANT_RCGO_DIR": "1",
            "QUANT_RCGO_ORTHO": "1",
        }
    )

    import json

    from config.settings import csv_list, get_settings
    from scripts.research_fas_clean import _liquidity_mask, fas_scores, load, smb_scores
    from scripts.research_fas_invent import rcgo_scores
    from stream.asym_signal import AsymSignal
    from stream.kv import FakeKV

    settings = get_settings()
    cw, vw, aw, dcl, dvl = load(a.cache)
    r_symbols = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, r_symbols)
    smb = smb_scores(vw, r_symbols)
    rcgo = rcgo_scores(dcl, dvl, aw, fas.index, r_symbols)
    r_sel = research_selection(cw, fas, smb, rcgo, r_symbols, w_rcgo=a.rcgo_w)

    universe = csv_list(settings.stream_xs_universe)
    print(f"[universe] research={len(r_symbols)}  live={len(universe)}")
    only_r = sorted(set(r_symbols) - set(universe))
    only_l = sorted(set(universe) - set(r_symbols))
    if only_r or only_l:
        print(f"[universe] research-only={only_r}  live-only={only_l}")

    # Drive the REAL live signal over the same cache.
    cache = json.load(open(a.cache))
    bars, funding = cache["bars"], cache["funding"]
    close_idx = {
        s: {int(r[0]): (float(r[1]), float(r[2] or 0.0)) for r in bars.get(s, [])}
        for s in universe
    }
    sig = AsymSignal(
        FakeKV(),
        prediction_prefix=settings.stream_asym_prediction_prefix,
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

    _m.logger.warning = lambda *x, **k: None  # silence warm-up warnings
    _m.logger.info = lambda *x, **k: None

    sig._funding = {s: [(int(ms), float(r)) for ms, r in (funding.get(s) or [])] for s in universe}
    sig._closes = {s: deque() for s in universe}
    windows = sorted({w for s in universe for w in close_idx[s]})
    for w in windows:
        for s in universe:
            cv = close_idx[s].get(w)
            if cv:
                sig._record(s, w, cv[0], cv[1])

    # Live selection at each weekly boundary, for the most recent N weeks.
    hour = 3_600_000
    boundaries = sorted({w for w in windows if w % (a.rebalance_h * hour) == 0})[-a.weeks :]
    print(f"\n{'week':>12} {'LONG overlap':>14} {'SHORT overlap':>14}  live L/S   research L/S")
    print("-" * 74)
    tot_l = tot_s = tot_n = 0
    for w in boundaries:
        sel = sig._selection(w)
        live_l = {s for s, (d, _) in sel.items() if d == "LONG"}
        live_s = {s for s, (d, _) in sel.items() if d == "SHORT"}
        # Compare the SAME week. Live rebalance boundaries are epoch-aligned
        # (Thursdays); research weeks are Monday-labelled. Nearest-match was
        # ambiguous by 3-4 days and could pair a live week against a DIFFERENT
        # research week, understating overlap. Map through the shared
        # Monday-anchored label instead.
        from stream.asym_signal import _week_end_ms

        scored = getattr(sig, "_last_scored_week", None)
        key = pd.Timestamp(scored if scored else _week_end_ms(w), unit="ms", tz="UTC")
        cands = [k for k in r_sel if (k.tz_localize("UTC") if k.tzinfo is None else k) == key]
        if not cands:
            continue
        r_l, r_s = r_sel[cands[0]]
        if not live_l and not live_s:
            continue
        ol = len(live_l & r_l) / max(1, len(r_l))
        os_ = len(live_s & r_s) / max(1, len(r_s))
        tot_l += len(live_l & r_l)
        tot_s += len(live_s & r_s)
        tot_n += len(r_l)
        print(
            f"{key:%Y-%m-%d}  {ol:>13.0%} {os_:>14.0%}  "
            f"{len(live_l)}/{len(live_s):<8} {len(r_l)}/{len(r_s)}"
        )
    if tot_n:
        print(
            f"\nOVERALL long overlap {tot_l / tot_n:.0%}  short overlap {tot_s / tot_n:.0%}"
            f"  (over {tot_n} research picks)"
        )
        print(
            "\nHigh overlap => the signal is faithful; the loss is execution mechanics.\n"
            "Low overlap  => the LIVE SIGNAL diverges and no execution fix recovers the edge."
        )


if __name__ == "__main__":
    main()
