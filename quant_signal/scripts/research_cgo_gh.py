"""Canonical Grinblatt-Han CGO vs the simplified CGO both books currently use.

Both stream/asym_signal.py::_cgo and research_fas_invent.py::cgo_daily_z compute

    CGO_t = Σ_s (P_t - P_{t-s})·V_{t-s} / (P_t · Σ_s V_{t-s})

i.e. a plain turnover-WEIGHTED AVERAGE of past gains. That drops the term that
defines the Grinblatt-Han reference price. Per 广发证券 "资本利得突出量CGO与风险
偏好" (行为金融因子研究之一) and its 2024 multi-frequency follow-up, the reference
price is

    RP_t = (1/k) Σ_{n=1..L} [ V_{t-n} · Π_{s=1..n-1} (1 - V_{t-n+s}) ] · P_{t-n}
    CGO_t = (P_t - RP_t) / RP_t

The Π(1 - V) factor is the SURVIVAL probability: the chance a unit bought at
t-n has not been traded away since. Without it, a day 100 bars back whose
holders have long since turned over still carries full weight, so the factor
stops measuring "unrealised gain still held" and becomes a generic
volume-weighted momentum average.

Crypto has no share count, so raw turnover-rate is unavailable. We proxy it as
each bar's share of lookback volume (V_n = vol_n / Σ vol), which is bounded in
[0,1) and preserves the construction's meaning: heavily-traded bars wash out
the holders that came before them.

Run:  uv run python -m scripts.research_cgo_gh
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from scripts.research_fas_clean import _liquidity_mask, _rank_z, fas_scores, load, smb_scores
from scripts.research_fas_invent import carry_scores
from scripts.parity_window import metrics, weekly_returns


def cgo_simplified(P: np.ndarray, V: np.ndarray, L: int) -> float | None:
    """What both books do today: turnover-weighted mean past gain."""
    if len(P) <= L:
        return None
    Pt = P[-1]
    if Pt <= 0 or not np.isfinite(Pt):
        return None
    num = sum((Pt - P[-1 - s]) * V[-1 - s] for s in range(1, L + 1))
    den = Pt * sum(V[-1 - s] for s in range(1, L + 1))
    return num / den if den > 0 else None


def cgo_grinblatt_han(P: np.ndarray, V: np.ndarray, L: int) -> float | None:
    """Canonical GH reference price with the Π(1-V) survival weighting."""
    if len(P) <= L:
        return None
    Pt = P[-1]
    if Pt <= 0 or not np.isfinite(Pt):
        return None
    px = P[-1 - L : -1]          # P_{t-L} .. P_{t-1}, oldest first
    vol = V[-1 - L : -1]
    tot = float(vol.sum())
    if tot <= 0 or not np.isfinite(tot):
        return None
    turn = vol / tot             # turnover proxy in [0,1)
    # weight_n = V_{t-n} * Π_{s<n, more recent} (1 - V_s): walk newest -> oldest
    weights = np.empty(len(px))
    survive = 1.0
    for i in range(len(px) - 1, -1, -1):
        weights[i] = turn[i] * survive
        survive *= 1.0 - turn[i]
    k = weights.sum()
    if k <= 0 or not np.isfinite(k):
        return None
    rp = float((weights * px).sum() / k)
    if rp <= 0 or not np.isfinite(rp):
        return None
    return (Pt - rp) / rp


def cgo_frame(daily_close, daily_vol, weekly_index, symbols, L, fn) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=weekly_index, columns=symbols)
    for wk in weekly_index:
        row = {}
        for s in symbols:
            c = daily_close[s][daily_close.index < wk].dropna().values
            v = daily_vol[s][daily_vol.index < wk].dropna().values
            n = min(len(c), len(v))
            if n <= L:
                continue
            val = fn(np.asarray(c[-(L + 1):], float), np.asarray(v[-(L + 1):], float), L)
            if val is not None and np.isfinite(val):
                row[s] = val
        if row:
            out.loc[wk] = _rank_z(pd.Series(row)).reindex(symbols).fillna(0.0)
    return out


def residualize_multi(cgo_z: pd.DataFrame, factors: dict, symbols, weekly_index) -> pd.DataFrame:
    """Residualise CGO on SEVERAL cross-sectional controls, not just carry.

    广发证券 "资本利得突出量的多频率测算" (2024) reports the CGO long-short book
    at Sharpe 2.27 raw and 3.56 after 行业市值中性化 (industry/market-cap
    neutralisation) -- the single largest reported lift in that study. Crypto
    has no industries, but it has size, which the book already computes as
    SMB. Adding size alongside funding-carry as a control is the direct analog.
    """
    rcgo = pd.DataFrame(0.0, index=weekly_index, columns=symbols)
    names = list(factors)
    for wk in weekly_index:
        data = {"cgo": cgo_z.loc[wk]}
        for n in names:
            data[n] = factors[n].loc[wk]
        df = pd.DataFrame(data).replace(0.0, np.nan).dropna()
        df = df[np.all(np.isfinite(df.values), axis=1)]
        if len(df) < 10:
            rcgo.loc[wk] = cgo_z.loc[wk].reindex(symbols).fillna(0.0)
            continue
        X = np.column_stack([np.ones(len(df))] + [df[n].values for n in names])
        beta, *_ = np.linalg.lstsq(X, df["cgo"].values, rcond=None)
        resid = df["cgo"].values - X @ beta
        rcgo.loc[wk] = _rank_z(pd.Series(resid, index=df.index)).reindex(symbols).fillna(0.0)
    return rcgo


def residualize(cgo_z: pd.DataFrame, carry_z: pd.DataFrame, symbols, weekly_index) -> pd.DataFrame:
    rcgo = pd.DataFrame(0.0, index=weekly_index, columns=symbols)
    for wk in weekly_index:
        cz, kz = cgo_z.loc[wk], carry_z.loc[wk]
        df = pd.DataFrame({"cgo": cz, "carry": kz}).replace(0.0, np.nan).dropna()
        df = df[np.isfinite(df["cgo"]) & np.isfinite(df["carry"])]
        if len(df) < 10:
            rcgo.loc[wk] = cz.reindex(symbols).fillna(0.0)
            continue
        X = np.column_stack([np.ones(len(df)), df["carry"].values])
        beta, *_ = np.linalg.lstsq(X, df["cgo"].values, rcond=None)
        resid = df["cgo"].values - X @ beta
        rcgo.loc[wk] = _rank_z(pd.Series(resid, index=df.index)).reindex(symbols).fillna(0.0)
    return rcgo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/quant_cache/asym_warm_start.json.binance")
    ap.add_argument("--lookbacks", default="5,7,14,20")
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    symbols = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, symbols)
    smb = smb_scores(vw, symbols)
    carry = carry_scores(aw, symbols).reindex(index=fas.index, columns=symbols)
    print(f"[data] {cw.shape}  tradable={len(symbols)}\n")

    print(f"{'CGO construction':<34} {'L':>3} {'weeks':>6} {'ann_ret':>9} {'Sharpe':>8} {'wealth':>8}")
    print("-" * 72)
    rows = []
    for L in [int(x) for x in a.lookbacks.split(",")]:
        for name, fn in [("simplified (current book)", cgo_simplified),
                         ("Grinblatt-Han survival RP", cgo_grinblatt_han)]:
            cz = cgo_frame(dcl, dvl, fas.index, symbols, L, fn)
            for ctrl, rc in [
                ("carry", residualize(cz, carry, symbols, fas.index)),
                ("carry+size", residualize_multi(
                    cz, {"carry": carry, "size": smb[symbols]}, symbols, fas.index)),
            ]:
                rets = weekly_returns(cw, fas, smb, rc, symbols, w_rcgo=0.5, rcgo_dir=1)
                m = metrics(rets[rets != 0])
                label = f"{name} [{ctrl}]"
                rows.append((label, L, m))
                print(
                    f"{label:<34} {L:>3} {m['weeks']:>6} {m['ann_ret'] * 100:>8.2f}% "
                    f"{m['sharpe']:>8.2f} {m['wealth']:>8.3f}"
                )

    best = max(rows, key=lambda r: r[2]["sharpe"])
    print(f"\nbest: {best[0]}  L={best[1]}  Sharpe={best[2]['sharpe']:.2f}")


if __name__ == "__main__":
    main()
