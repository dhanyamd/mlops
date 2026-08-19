"""Clean, research-grounded reference implementation of FAS_avg + SMB + CGO.

Methodology (faithful to OUR asym_signal.py + the papers we reasoned from):

  * Bianchi et al. "Order Flow and Cryptocurrency Returns": orthogonalize lagged
    order flow (here: funding accrual) on lagged returns; the RESIDUAL (the part
    uncorrelated with the price path) positively predicts FUTURE returns. Weekly
    L/S = +1.83%/wk, annualized Sharpe ~1.9. Our FAS is the derivatives analog:
    funding accrual residualized on [pr_h, |pr_h|], then a contrarian fade.

  * FAS_h = -z(price_ret_h) * z(residual of WEEK-w funding accrual on [pr_h, |pr_h|])
        -> short names whose funding accrual is anomalously high given their
           price path (crowded longs that already rallied); long the mirror.
  * SMB   = -z(log trailing 12w volume)                       (long small-caps)
  * CGO   = Griffin-Han capital-gains-overhang, keep LOW overhang (dir=-1),
            disposition-selling supply screen. Liu-Fang-Wang (Management Review
            2024): CGO MUST use DAILY closes+volume (weekly is INVALID), so we
            build it on daily bars exactly like asym_signal._cgo.
  * Book  = weekly quintile long/short of z(FAS_avg + SMB), 10bps turnover.

Data hygiene (NOT a strategy hack — mirrors asym_signal guards + the standard
illiquidity screen used in Grinblatt-Han / Zaremba): drop symbols that are not
continuously traded (near-zero volume over the window) or carry non-finite /
non-positive prices. A dead ticker (e.g. a 2,860% glitch on ~0-volume bars)
would otherwise pin the SMB rank and blow up the book — the exact failure mode
flagged in crypto L/S literature (Springer 2025: trimming one degenerate name
lifts Sharpe from negative to +1.51).

Run: python scripts/research_fas_clean.py [--cache PATH] [--funding binance|bybit]
                                      [--no-cgo] [--fas-sign {1,-1}]
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
_WEEK = 7 * 24 * 3600_000  # ms in a week (cache bars are hourly ms)


def _rank_z(series: pd.Series) -> pd.Series:
    """Cross-sectional rank-z (pct-rank*2-1), matching asym_signal._rank_z."""
    vals = series.dropna()
    if len(vals) < 5:
        return pd.Series(0.0, index=series.index)
    order = vals.sort_values()
    n = len(order)
    r = (order.rank() / n) - 0.5
    return (r * 2.0).reindex(series.index).fillna(0.0)


def load(cache_path: str, week_anchor: str = "W-MON"):
    """Weekly/daily frames from the hourly cache.

    ``week_anchor`` selects which weekday the weekly bars close on. It exists
    for the rebalance-timing-luck test (Hoffstein, Sibears & Faber 2018): a
    weekly strategy that only ever rebalances on Mondays has an unmeasured
    exposure to WHICH day it picked, and the only way to size that exposure is
    to rebuild the whole panel on each of the seven anchors. Default preserves
    the Monday grid every existing caller already uses.
    """
    c = json.load(open(cache_path))
    return build_frames(c["bars"], c.get("funding", {}), week_anchor)


def build_frames(bars: dict, fund: dict, week_anchor: str = "W-MON"):
    """Resample raw ``{symbol: [[ts_ms, close, vol], ...]}`` into the panel.

    Split out of ``load`` so the warehouse-backed source
    (``scripts.warehouse_panel``) reaches the same frames through the same
    arithmetic. If each source did its own resampling, "the warehouse path
    reproduces the file path" would be a coincidence maintained by hand; here
    it is the same function, and only the fetch differs.
    """
    hourly_close: dict[str, pd.Series] = {}
    hourly_vol: dict[str, pd.Series] = {}
    weekly_close: dict[str, pd.Series] = {}
    weekly_vol: dict[str, pd.Series] = {}
    weekly_accr: dict[str, pd.Series] = {}
    daily_close: dict[str, pd.Series] = {}
    daily_vol: dict[str, pd.Series] = {}
    for s in bars:
        rows = bars.get(s, [])
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts", "close", "vol"]).sort_values("ts")
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        cl = df["close"].astype(float)
        vl = df["vol"].astype(float)
        hourly_close[s] = cl
        hourly_vol[s] = vl
        weekly_close[s] = cl.resample(week_anchor).last()
        weekly_vol[s] = vl.resample(week_anchor).sum()
        daily_close[s] = cl.resample("D").last()
        daily_vol[s] = vl.resample("D").sum()
        frows = fund.get(s, [])
        if frows:
            fd = pd.DataFrame(frows, columns=["ts", "rate"]).sort_values("ts")
            fd.index = pd.to_datetime(fd["ts"], unit="ms", utc=True)
            weekly_accr[s] = fd["rate"].resample(week_anchor).sum()
    return (
        pd.DataFrame(weekly_close),
        pd.DataFrame(weekly_vol),
        pd.DataFrame(weekly_accr),
        pd.DataFrame(daily_close),
        pd.DataFrame(daily_vol),
    )


def _liquidity_mask(
    weekly_close: pd.DataFrame, weekly_vol: pd.DataFrame, min_active_frac: float = 0.99
) -> list[str]:
    """Continuously-traded, finite, positive-price screen (our logic + lit screen).

    A name must have positive volume in at least ``min_active_frac`` of weeks and
    finite positive closes everywhere. Mirrors asym_signal._weekly_close's
    `c <= 0 or not finite` guard and removes dead/glitchy tickers (e.g. a name
    with ~0-volume bars + a 2,860% price jump) that would otherwise corrupt the
    cross-sectional rank. This is a tradability filter, not an edge tweak.
    """
    keep = []
    for s in weekly_close.columns:
        wc = weekly_close[s].dropna()
        wv = weekly_vol[s].reindex(wc.index).fillna(0.0)
        if wc.shape[0] == 0 or (wc <= 0).any() or not np.isfinite(wc).all():
            continue
        active = float((wv > 0).mean())
        if active < min_active_frac:
            continue
        keep.append(s)
    return keep


def fas_scores(
    close_w: pd.DataFrame,
    accr_w: pd.DataFrame,
    symbols: list[str],
    horizons=(4, 8, 12, 26),
) -> pd.DataFrame:
    """FAS_avg z-scores per (week, symbol). No lookahead: at week w the score uses
    close/accrual strictly at w to predict w -> w+1 (the live rebalance holds 1wk).

    Accrual is the SINGLE week-w funding sum (asym_signal._fas_scores: accr[s]=fset[w]),
    regressed on [pr_h, |pr_h|]; we do NOT sum the h-week window.
    """
    close_w = close_w[symbols]
    accr_w = accr_w.reindex(index=close_w.index, columns=symbols)
    weeks = close_w.index
    n_w = len(weeks)
    fas_avg = pd.DataFrame(0.0, index=weeks, columns=symbols)
    for w_i in range(max(horizons) + 4, n_w - 1):  # leave 1 fwd week
        w = weeks[w_i]
        ac_all = accr_w.loc[w]  # THIS week's funding accrual only
        for h in horizons:
            pr = close_w.loc[w] / close_w.shift(h).loc[w] - 1.0  # price ret over h wks
            ac = ac_all
            df = pd.DataFrame({"pr": pr, "apr": pr.abs(), "acc": ac}).dropna()
            df = df[np.isfinite(df["pr"]) & np.isfinite(df["acc"])]
            if len(df) < 10:
                continue
            X = np.column_stack([np.ones(len(df)), df["pr"].values, df["apr"].values])
            y = df["acc"].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            pr_z = _rank_z(df["pr"])
            res_z = _rank_z(pd.Series(resid, index=df.index))
            fas_h = (-pr_z * res_z).reindex(symbols).fillna(0.0)
            fas_avg.loc[w] = fas_avg.loc[w] + fas_h / len(horizons)
    return fas_avg


def smb_scores(vol_w: pd.DataFrame, symbols: list[str], weeks_window=12) -> pd.DataFrame:
    """SMB = -z(log trailing 12w volume), long small-caps."""
    logv = np.log(vol_w[symbols].rolling(weeks_window).sum().clip(lower=1e-9))
    return -logv.apply(_rank_z)


def cgo_filter_daily(
    daily_close: pd.DataFrame,
    daily_vol: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    symbols: list[str],
    L=7,
    q=0.30,
    d=-1,
) -> dict:
    """Griffin-Han CGO on DAILY bars (Liu-Fang-Wang: weekly is invalid).

    For each weekly rebalance we take daily (close, volume) strictly before the
    week-end, volume-weighted, matching asym_signal._cgo. Keep LOW overhang
    (dir=-1) -> no profit-taking supply overhang. Returns {week: set(kept)}.
    """

    def _cgo(dbars_c, dbars_v):
        if len(dbars_c) <= L:
            return None
        P = np.asarray(dbars_c, float)
        V = np.asarray(dbars_v, float)
        Pt = P[-1]
        if Pt <= 0 or not np.isfinite(Pt):
            return None
        num = sum((Pt - P[-1 - s]) * V[-1 - s] for s in range(1, L + 1))
        den = Pt * sum(V[-1 - s] for s in range(1, L + 1))
        if den <= 0:
            return None
        return num / den

    out: dict = {}
    for wk in weekly_index:
        as_of = wk  # week-end timestamp; use daily bars strictly before
        cgo = {}
        for s in symbols:
            dc = daily_close[s].dropna()
            dv = daily_vol[s].reindex(dc.index).fillna(0.0)
            if dc.shape[0] <= L + 1:
                continue
            mask = dc.index < as_of
            sub_c = dc[mask]
            sub_v = dv[mask]
            if sub_c.shape[0] <= L + 1:
                continue
            val = _cgo(sub_c.values, sub_v.values)
            if val is not None and math.isfinite(val):
                cgo[s] = val
        if len(cgo) < 8:
            out[wk] = set(symbols)
            continue
        vals = sorted(cgo.values())
        n = len(vals)
        k = max(2, int(round(q * n)))
        thr = vals[n - k] if d == -1 else vals[k - 1]
        keep = {s for s, v in cgo.items() if (v <= thr if d == -1 else v >= thr)}
        out[wk] = keep
    return out


def backtest(
    close_w: pd.DataFrame,
    fas: pd.DataFrame,
    smb: pd.DataFrame,
    cgo: dict,
    symbols: list[str],
    quintile=0.20,
    cost_bps=10.0,
) -> dict:
    weeks = fas.index
    fwd = close_w[symbols].shift(-1) / close_w[symbols] - 1.0
    score = (fas[symbols] + smb[symbols]).apply(_rank_z)
    rets, ridx = [], []
    pos = None
    for w in weeks[:-1]:
        s = score.loc[w].dropna()
        kept = cgo.get(w, set(symbols))
        s = s[s.index.isin(kept)]
        if len(s) < 8:
            pos = None
            continue
        ranked = s.sort_values()
        n = max(2, int(round(quintile * len(ranked))))
        longs = ranked.index[-n:]
        shorts = ranked.index[:n]
        w_pos = pd.Series(0.0, index=symbols)
        w_pos[longs] = 1.0 / n
        w_pos[shorts] = -1.0 / n
        r = float((w_pos * fwd.loc[w]).reindex(symbols).sum(skipna=True))
        if pos is not None:
            turn = float((w_pos.reindex(pos.index).fillna(0) - pos).abs().sum())
            r -= cost_bps / 1e4 * turn
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
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + rets).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    return {
        "weeks": len(rets),
        "ann_ret": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "maxdd": dd,
        "wealth_end": float(wealth.iloc[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/quant_cache/asym_warm_start.json.binance")
    ap.add_argument("--funding", default="binance")
    ap.add_argument("--no-cgo", action="store_true")
    ap.add_argument("--fas-sign", type=int, default=1, choices=(1, -1))
    args = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(args.cache)
    print(
        f"[data] {cw.shape} weeks {cw.index.min().date()}..{cw.index.max().date()} "
        f"funding={args.funding}"
    )

    # tradability / sanity screen (our logic + literature illiquidity screen)
    symbols = _liquidity_mask(cw, vw)
    print(f"[universe] {len(symbols)} tradable symbols after liquidity screen")
    dropped = [s for s in cw.columns if s not in symbols]
    if dropped:
        print(f"[universe] dropped (dead/illiquid/glitch): {dropped}")

    fas = fas_scores(cw, aw, symbols) * args.fas_sign
    smb = smb_scores(vw, symbols)
    cgo = {} if args.no_cgo else cgo_filter_daily(dcl, dvl, fas.index, symbols)

    m = backtest(cw, fas, smb, cgo, symbols)
    tag = f"FAS_avg+SMB CGO={'OFF' if args.no_cgo else 'ON'} sign={args.fas_sign}"
    print(f"\n=== {tag} ===")
    print(
        f"  weeks={m['weeks']}  ann_ret={m['ann_ret'] * 100:6.2f}%  "
        f"ann_vol={m['ann_vol'] * 100:6.1f}%  Sharpe={m['sharpe']:.2f}  "
        f"maxDD={m['maxdd'] * 100:6.1f}%  wealthx={m['wealth_end']:.3f}"
    )


if __name__ == "__main__":
    main()
