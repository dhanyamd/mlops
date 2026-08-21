"""IC / ICIR test: does each signal actually predict forward returns?

Standard Chinese-quant factor-validation battery (信息系数 / IC, ICIR):
    IC_w   = cross-sectional Spearman corr( score_w , forward_return_w )
    ICIR   = mean(IC) / std(IC)
Interpretation used throughout that literature:
    |IC| > 0.05  -> effective alpha factor
    |IC| > 0.10  -> especially strong
    |IC| ~ 0     -> 无效因子 (no predictive power)

Runs in seconds and needs no trade simulation, so it isolates SIGNAL quality
from execution entirely -- the fastest way to answer "is the live signal
actually computing the validated factor?".
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QUANT_CGO_DIR","1"); os.environ.setdefault("QUANT_CGO_L","7")
os.environ.setdefault("QUANT_CGO_Q","0.3"); os.environ.setdefault("QUANT_REGIME_OFF","1")
os.environ.setdefault("QUANT_SMB_OFF","0"); os.environ.setdefault("QUANT_FACC_OFF","1")
os.environ.setdefault("QUANT_RCGO_W","1.0"); os.environ.setdefault("QUANT_RCGO_DIR","1")
os.environ.setdefault("QUANT_RCGO_ORTHO","1")
from collections import deque

import numpy as np
import pandas as pd

import stream.asym_signal as _m
from config.settings import csv_list, get_settings
from scripts.research_fas_clean import _liquidity_mask, _rank_z, fas_scores, load, smb_scores
from scripts.research_fas_invent import rcgo_scores
from stream.asym_signal import AsymSignal
from stream.kv import FakeKV

_m.logger.warning = lambda *a, **k: None
_m.logger.info = lambda *a, **k: None

CACHE = "/tmp/quant_cache/asym_warm_start.json.binance"


def ic_stats(scores: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    ics = []
    for w in scores.index:
        if w not in fwd.index:
            continue
        a, b = scores.loc[w], fwd.loc[w]
        common = [c for c in a.index if c in b.index and np.isfinite(a[c]) and np.isfinite(b[c])]
        if len(common) < 8:
            continue
        c = pd.Series({x: a[x] for x in common}).corr(
            pd.Series({x: b[x] for x in common}), method="spearman")
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return {"n": 0}
    ics = np.array(ics)
    return {"n": len(ics), "ic_mean": ics.mean(), "ic_std": ics.std(),
            "icir": ics.mean() / ics.std() if ics.std() > 0 else 0.0,
            "win": (np.sign(ics) == np.sign(ics.mean())).mean()}


def verdict(ic: float) -> str:
    a = abs(ic)
    if a > 0.10: return "STRONG alpha factor (>0.10)"
    if a > 0.05: return "effective alpha factor (>0.05)"
    if a > 0.02: return "weak"
    return "无效因子 — NO predictive power"


def main() -> None:
    s = get_settings()
    cw, vw, aw, dcl, dvl = load(CACHE)
    rs = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, rs); smb = smb_scores(vw, rs)
    rcgo = rcgo_scores(dcl, dvl, aw, fas.index, rs)
    fwd = (cw[rs].shift(-1) / cw[rs] - 1.0).reindex(fas.index)

    base = (fas[rs] + smb[rs]).apply(_rank_z)
    r_score = (base + (1.0 * rcgo[rs]).apply(_rank_z)).apply(_rank_z)

    # live signal over the same cache
    uni = csv_list(s.stream_xs_universe)
    cache = json.load(open(CACHE)); bars = cache["bars"]; fund = cache["funding"]
    ci = {x: {int(r[0]): (float(r[1]), float(r[2] or 0.0)) for r in bars.get(x, [])} for x in uni}
    sig = AsymSignal(FakeKV(), prediction_prefix=s.stream_asym_prediction_prefix, universe=uni,
        rebalance_h=168, quintile=s.stream_asym_quintile, min_symbols=s.stream_asym_min_symbols,
        regime=False, regime_slow_days=s.stream_asym_regime_slow_days,
        market_symbol=s.stream_asym_market_symbol, horizons=s.stream_asym_horizons,
        accrual_weeks=s.stream_asym_accrual_weeks, smb_weeks=s.stream_asym_smb_weeks,
        use_facc=False, use_rev=False)
    sig._replay = True
    sig._funding = {x: [(int(ms), float(r)) for ms, r in (fund.get(x) or [])] for x in uni}
    sig._closes = {x: deque() for x in uni}
    wins = sorted({w for x in uni for w in ci[x]})

    # POINT-IN-TIME: feed windows strictly in order and score at each weekly
    # boundary, so the signal's registry only ever holds data that existed at
    # that moment. Pre-loading the whole history first (the earlier harness)
    # can only ever flatter the factor -- any component that reads unbounded
    # history would be scoring on the future.
    H = 3_600_000
    bounds = {w for w in wins if w % (168 * H) == 0}
    rows = {}
    for w in wins:
        for x in uni:
            cv = ci[x].get(w)
            if cv: sig._record(x, w, cv[0], cv[1])
        if w in bounds:
            sc = sig._fas_scores(w)
            if sc:
                rows[pd.Timestamp(w, unit="ms", tz="UTC")] = pd.Series(sc)
    l_score = pd.DataFrame(rows).T.reindex(columns=rs)
    # align live weeks onto the research weekly grid
    l_score.index = [min(fas.index, key=lambda k:
        abs((k.tz_localize("UTC") if k.tzinfo is None else k) - t)) for t in l_score.index]

    # Lag sweep: a genuine forward-predictive factor peaks at lag 0 (score at
    # week w vs return w->w+1). If IC peaks at a NEGATIVE lag, the score is
    # being scored against a return period it already overlaps -- i.e. the
    # week alignment leaked contemporaneous information and the IC is fake.
    print("=== ALIGNMENT CHECK: IC by lag (should peak at lag 0) ===")
    print(f"{'signal':<26} " + " ".join(f"lag{l:+d}".rjust(8) for l in (-2,-1,0,1,2)))
    print("-" * 72)
    for name, sc in [("RESEARCH", r_score), ("LIVE", l_score)]:
        cells = []
        for lag in (-2,-1,0,1,2):
            f2 = (cw[rs].shift(-1 - lag) / cw[rs].shift(-lag) - 1.0).reindex(fas.index)
            st = ic_stats(sc, f2)
            cells.append(f"{st['ic_mean']:+.3f}".rjust(8) if st.get("n") else "  n/a  ")
        print(f"{name:<26} " + " ".join(cells))
    print()

    print("=== IC / ICIR — does each signal predict forward weekly returns? ===")
    print(f"{'signal':<26} {'weeks':>6} {'IC mean':>9} {'IC std':>8} {'ICIR':>7} {'win%':>6}  verdict")
    print("-" * 96)
    for name, sc in [("RESEARCH (Sharpe 2.28)", r_score), ("LIVE asym_signal.py", l_score)]:
        st = ic_stats(sc, fwd)
        if not st.get("n"):
            print(f"{name:<26} no overlapping weeks"); continue
        print(f"{name:<26} {st['n']:>6} {st['ic_mean']:>9.4f} {st['ic_std']:>8.4f} "
              f"{st['icir']:>7.2f} {st['win']:>5.0%}  {verdict(st['ic_mean'])}")


if __name__ == "__main__":
    main()
