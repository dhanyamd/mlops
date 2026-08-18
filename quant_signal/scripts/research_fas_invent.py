"""INVENTION: RCGO_crypto — Residual Capital-Gains-Overhang, orthogonalized on
funding-carry, blended CONTINUOUSLY with FAS_avg + SMB.

DERIVED from the research we reasoned from, INVENTED for our book (not copied):

  * Liu Shuai / Fang Yong / Wang Shouyang (Management Review 2024, 36(6):94-106)
    "Cryptocurrency investment strategy based on disposition effect and momentum
    effect": CGO is a valid crypto disposition proxy ONLY on daily-or-higher
    frequency (weekly is INVALID), crypto momentum lasts <=2 weeks, and the
    COMBINED disposition + momentum strategy beats either alone.
  * Crypto Carry (SSRN 3774118, 2023): funding/carry positively predicts
    cross-sectional returns across 51 cryptocurrencies.
  * A-share disposition literature (Guangfa "multi-frequency CGO", CICC/CFIJ):
    RCGO = Capital-Gains-Overhang residualized on risk factors (the part of CGO
    uncorrelated with value/momentum) is the *sharp* signal; CGO is a CONTINUOUS
    factor (daily IC ~= -4.4%), so a hard q-filter throws away information.
  * FAS already orthogonalizes funding accrual on the price path (Bianchi analog)
    -> consistent theme: orthogonalize, then blend.

THE INVENTION: no paper orthogonalizes CGO on crypto funding-carry. We build
  RCGO_b[s,w] = CGO_daily_z[s,w] - beta_w * CARRY_xs_z[s,w]
where CARRY_xs is the cross-sectional z of weekly funding accrual (Crypto Carry),
and beta_w is the per-week cross-sectional regression of CGO on carry (Griffin-Han
/ Guangfa RCGO construction). The RESIDUAL is pure behavioral overhang net of
mechanical crowding. We then blend it CONTINUOUSLY (not as a hard gate):
  score = z(FAS_avg + SMB) + W_RCGO * dir * z(RCGO_b)
This sharpens the disposition leg and removes the redundant carry component that
FAS already captures.

Proven on the warm cache (live 1h Flink feature job is currently stalled, so the
offline replay is the verification path — same math as live, no Kafka needed).

Run: python scripts/research_fas_invent.py [--cache PATH] [--funding binance|bybit]
                                       [--rcgo-dir {1,-1}] [--w-rcgo F] [--no-filter]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.research_fas_clean import (  # reuse the proven reference pieces
    _liquidity_mask,
    _rank_z,
    fas_scores,
    load,
    smb_scores,
)


def carry_scores(accr_w: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Cross-sectional z of WEEKLY funding accrual = Crypto Carry cross-sectional
    crowding signal (SSRN 3774118). High accrual = crowded longs."""
    z = accr_w.reindex(index=accr_w.index, columns=symbols).apply(_rank_z)
    return z


def cgo_daily_z(
    daily_close: pd.DataFrame,
    daily_vol: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    symbols: list[str],
    L: int = 7,
) -> pd.DataFrame:
    """Griffin-Han CGO on DAILY bars (Liu-Fang-Wang: weekly invalid), computed at
    each weekly rebalance from daily bars strictly before the week-end, then
    cross-sectionally rank-z'd. Returns week x symbol raw CGO z-scores."""

    # QUANT_CGO_GH=1 (default) uses the canonical Grinblatt-Han reference price;
    # 0 restores the legacy turnover-weighted-mean form. Shared by the research
    # backtest AND the live book (stream/asym_signal._research_scores) so the
    # two cannot diverge again.
    _GH = os.environ.get("QUANT_CGO_GH", "1") == "1"

    def _cgo(c, v):
        if len(c) <= L:
            return None
        P = np.asarray(c, float)
        V = np.asarray(v, float)
        Pt = P[-1]
        if Pt <= 0 or not np.isfinite(Pt):
            return None
        if not _GH:
            num = sum((Pt - P[-1 - s]) * V[-1 - s] for s in range(1, L + 1))
            den = Pt * sum(V[-1 - s] for s in range(1, L + 1))
            return num / den if den > 0 else None
        # Grinblatt-Han (2005), as constructed in 广发证券 "资本利得突出量CGO与
        # 风险偏好": the reference price is a turnover-weighted average of past
        # prices where each bar's weight carries the SURVIVAL probability that
        # a unit bought then has not been traded away since:
        #     RP_t = (1/k) * sum_n [ V_{t-n} * prod_{s<n}(1 - V) ] * P_{t-n}
        #     CGO_t = (P_t - RP_t) / RP_t
        # Dropping prod(1-V) -- the legacy form above -- leaves a plain
        # volume-weighted momentum average, not an overhang measure.
        # Turnover is volume/shares-outstanding in equities; crypto has no share
        # count, so it is proxied by each bar's share of lookback volume, which
        # is bounded in [0,1) and preserves the construction's meaning.
        px = P[-1 - L : -1]
        vol = V[-1 - L : -1]
        tot = float(vol.sum())
        if tot <= 0 or not np.isfinite(tot):
            return None
        turn = vol / tot
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

    out = pd.DataFrame(0.0, index=weekly_index, columns=symbols)
    for wk in weekly_index:
        cgo = {}
        for s in symbols:
            dc = daily_close[s].dropna()
            dv = daily_vol[s].reindex(dc.index).fillna(0.0)
            mask = dc.index < wk
            sub_c, sub_v = dc[mask], dv[mask]
            if sub_c.shape[0] <= L + 1:
                continue
            val = _cgo(sub_c.values, sub_v.values)
            if val is not None and math.isfinite(val):
                cgo[s] = val
        out.loc[wk] = _rank_z(pd.Series(cgo, name=wk)).reindex(symbols).fillna(0.0)
    return out


def rcgo_scores(daily_close, daily_vol, accr_w, weekly_index, symbols, L=7) -> pd.DataFrame:
    """INVENTION: RCGO_b = CGO_daily_z residualized on CARRY_xs_z, per week
    (cross-sectional OLS, Griffin-Han/Guangfa RCGO construction applied to crypto
    funding-carry). Returns week x symbol residual-z scores."""
    cgo_z = cgo_daily_z(daily_close, daily_vol, weekly_index, symbols, L)
    carry_z = carry_scores(accr_w, symbols).reindex(index=weekly_index, columns=symbols)
    rcgo = pd.DataFrame(0.0, index=weekly_index, columns=symbols)
    for wk in weekly_index:
        cz = cgo_z.loc[wk]
        kz = carry_z.loc[wk]
        df = pd.DataFrame({"cgo": cz, "carry": kz}).replace(0.0, np.nan).dropna()
        df = df[np.isfinite(df["cgo"]) & np.isfinite(df["carry"])]
        if len(df) < 10:
            rcgo.loc[wk] = cz.reindex(symbols).fillna(0.0)
            continue
        X = np.column_stack([np.ones(len(df)), df["carry"].values])
        y = df["cgo"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rcgo_z = _rank_z(pd.Series(resid, index=df.index))
        rcgo.loc[wk] = rcgo_z.reindex(symbols).fillna(0.0)
    return rcgo


def backtest_continuous(
    close_w, fas, smb, rcgo, symbols, quintile=0.20, cost_bps=10.0, w_rcgo=0.5, rcgo_dir=1
) -> dict:
    weeks = fas.index
    fwd = close_w[symbols].shift(-1) / close_w[symbols] - 1.0
    base = (fas[symbols] + smb[symbols]).apply(_rank_z)
    tilt = (w_rcgo * rcgo_dir * rcgo[symbols]).apply(_rank_z)
    score = (base + tilt).apply(_rank_z)
    rets, ridx, pos = [], [], None
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
        ridx.append(w)
        pos = w_pos
    # Index by the weeks that ACTUALLY produced a return: the loop skips
    # weeks with <8 usable symbols, so weeks[:-1] is longer than rets
    # whenever early history is sparse. Zipping them misaligns every
    # return with the wrong week (and raises outright on long samples).
    rets = pd.Series(rets, index=ridx)
    rets = rets[rets != 0]
    ann = rets.mean() * 52
    vol = rets.std() * math.sqrt(52)
    wealth = (1 + rets).cumprod()
    return {
        "weeks": len(rets),
        "ann_ret": ann,
        "ann_vol": vol,
        "sharpe": ann / vol if vol > 0 else 0.0,
        "maxdd": (wealth / wealth.cummax() - 1).min(),
        "wealth_end": float(wealth.iloc[-1]),
    }


def _show(tag, m):
    print(
        f"  {tag:42s} weeks={m['weeks']:3d}  ann_ret={m['ann_ret'] * 100:6.2f}%  "
        f"ann_vol={m['ann_vol'] * 100:6.1f}%  Sharpe={m['sharpe']:.2f}  "
        f"maxDD={m['maxdd'] * 100:6.1f}%  x={m['wealth_end']:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/quant_cache/asym_warm_start.json.binance")
    ap.add_argument("--funding", default="binance")
    ap.add_argument("--rcgo-dir", type=int, default=1, choices=(1, -1))
    ap.add_argument("--w-rcgo", type=float, default=0.5)
    ap.add_argument(
        "--no-filter", action="store_true", help="ignore hard CGO q-filter (use continuous only)"
    )
    args = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(args.cache)
    symbols = _liquidity_mask(cw, vw)
    print(f"[data] {cw.shape} funding={args.funding}  tradable={len(symbols)}")

    fas = fas_scores(cw, aw, symbols)
    smb = smb_scores(vw, symbols)
    rcgo = rcgo_scores(dcl, dvl, aw, fas.index, symbols)

    from scripts.research_fas_clean import backtest, cgo_filter_daily

    # Baseline (proven): hard CGO filter dir=+1
    cgo_f = {} if args.no_filter else cgo_filter_daily(dcl, dvl, fas.index, symbols, d=1)
    base = backtest(cw, fas, smb, cgo_f, symbols)
    _show("BASELINE FAS+SMB + CGO filter(dir=+1)", base)

    # Invention: continuous RCGO blend, no hard filter
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        for d in (1, -1):
            m = backtest_continuous(cw, fas, smb, rcgo, symbols, w_rcgo=w, rcgo_dir=d)
            _show(f"INVENT RCGO_blend w={w:.2f} dir={d}", m)

    # Sanity: RCGO direction vs raw CGO direction
    m_pos = backtest_continuous(cw, fas, smb, rcgo, symbols, w_rcgo=args.w_rcgo, rcgo_dir=1)
    m_neg = backtest_continuous(cw, fas, smb, rcgo, symbols, w_rcgo=args.w_rcgo, rcgo_dir=-1)
    _show(f"INVENT RCGO dir=+1 w={args.w_rcgo}", m_pos)
    _show(f"INVENT RCGO dir=-1 w={args.w_rcgo}", m_neg)


if __name__ == "__main__":
    main()
