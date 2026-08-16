"""Evaluate intraday-derived factors -- reproduced ones and this project's two.

METHOD GUARDRAILS (both learned the hard way today)
---------------------------------------------------
1. REPORT PEARSON IC, NOT JUST SPEARMAN. The salience factor showed Spearman
   IC +0.0506 at t=6.10 -- above the |IC|>0.05 bar the Chinese literature uses --
   while Pearson IC was only +0.0172. Spearman scores RANKS; P&L is paid in
   MAGNITUDES, and crypto returns are violently fat-tailed. A factor can rank
   the typical coin correctly and still be wrong about the handful of moves that
   dominate the book. Any factor whose Spearman/Pearson gap is large is ranking
   noise, not earning money.

2. REPORT DECILE MONOTONICITY. Salience also produced a decile table with no
   slope at all (decile 4 out-returned decile 10). A real cross-sectional factor
   has a gradient. `mono` here is the rank correlation between decile index and
   decile mean forward return; near +1 is a genuine sort, near 0 is noise that
   happens to have a positive average IC.

3. NO CONFIG SEARCH. Walk-forward already established on this book that adding
   candidate configurations DESTROYS out-of-sample value (9 candidates -> -0.10
   OOS; a single fixed factor -> +0.69). So every factor below is specified a
   priori from its source and reported side by side. Nothing is selected on the
   full sample, and the winner still has to survive walk-forward afterwards.

FACTORS
-------
Reproduced:
  cpv_mean/vol/trend  东吴证券 CPV. The report computes a daily corr(close,
                      volume) over intraday bars, then aggregates that series
                      over the month along three dimensions -- mean, volatility,
                      trend -- and combines. The three are reported separately
                      here AND as the equal-weight combination, so the
                      combination rule is visible rather than assumed.
  q                   方正/开源證券 smart money, Q = VWAP_smart / VWAP_all.
  rsj                 realised signed jump.
  ofi                 raw taker order-flow imbalance.

Invented here (see backfill_intraday_features.py for derivation):
  ifd                 Informed Flow Divergence -- the Chinese smart-money bar
                      selection, made sign-aware using the aggressor side that
                      only a crypto venue publishes.
  kyle                Directional Kyle Asymmetry -- price impact per unit BUY
                      volume vs per unit SELL volume.

SIGNS are not fitted. Each factor's expected direction comes from its source
(CPV negative per the report; smart-money Q negative -- informed buying below
the average price is bullish, so high Q is bearish). For the two invented
factors there is no published sign, so BOTH directions are shown and neither is
selected here; that choice has to be made out of sample.

Run:
    uv run python -m scripts.research_intraday --lookback 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_fas_clean import _liquidity_mask, _rank_z, load, smb_scores

FIELDS = ("cpv", "q", "ofi", "ifd", "kyle", "rsj", "rv", "avgtrd")


def load_intraday(d: str) -> dict[str, pd.DataFrame]:
    """Per-field DataFrames indexed by UTC day, columns = symbols."""
    per: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    for p in sorted(Path(d).glob("*.json")):
        recs = json.loads(p.read_text())
        if not recs:
            continue
        idx = pd.to_datetime([r["t"] for r in recs], unit="ms", utc=True)
        for f in FIELDS:
            s = pd.Series([r.get(f) for r in recs], index=idx, dtype=float)
            if s.notna().sum() > 50:
                per[f][p.stem] = s
    return {f: pd.DataFrame(v).sort_index() for f, v in per.items() if v}


def slope(a: np.ndarray) -> float:
    """OLS slope against time -- the CPV report's 'trend' dimension."""
    m = np.isfinite(a)
    if m.sum() < 5:
        return np.nan
    y = a[m]
    x = np.arange(len(a), dtype=float)[m]
    x = x - x.mean()
    d = (x * x).sum()
    return float((x * (y - y.mean())).sum() / d) if d > 0 else np.nan


def diagnostics(score: pd.DataFrame, fwd: pd.DataFrame) -> tuple[float, float, float]:
    """Spearman IC, Pearson IC, and decile monotonicity."""
    sp, pe, rows = [], [], []
    for w in score.index:
        if w not in fwd.index:
            continue
        a, b = score.loc[w], fwd.loc[w]
        m = a.notna() & b.notna()
        if m.sum() < 10:
            continue
        x, y = a[m], b[m]
        sp.append(x.corr(y, method="spearman"))
        pe.append(x.corr(y))
        if m.sum() >= 20:
            qd = pd.qcut(x.rank(method="first"), 10, labels=False)
            rows.append(y.groupby(qd).mean())
    if not sp:
        return (np.nan,) * 3
    mono = np.nan
    if rows:
        t = pd.DataFrame(rows).mean()
        if t.notna().sum() >= 5:
            mono = float(pd.Series(t.index, index=t.index).corr(t, method="spearman"))
    return float(pd.Series(sp).mean()), float(pd.Series(pe).mean()), mono


def book(score: pd.DataFrame, fwd: pd.DataFrame, sym, cost_bps: float, per_year: float,
         top: float = 0.20, cap: float = 1.0) -> pd.Series:
    rets, ridx, pos = [], [], None
    for w in score.index:
        if w not in fwd.index:
            continue
        s = score.loc[w].dropna()
        s = s[s != 0]
        if len(s) < 10:
            pos = None
            continue
        r = s.sort_values()
        n = max(2, int(round(top * len(r))))
        wp = pd.Series(0.0, index=sym)
        wp[list(r.index[-n:])] = 1.0 / n
        wp[list(r.index[:n])] = -1.0 / n
        ret = float((wp * fwd.loc[w].clip(upper=cap)).reindex(sym).sum(skipna=True))
        if pos is not None and cost_bps:
            ret -= cost_bps / 1e4 * float((wp - pos).abs().sum())
        rets.append(ret)
        ridx.append(w)
        pos = wp
    return pd.Series(rets, index=ridx)


def show(name, score, fwd, sym, cost_bps, per_year):
    r = book(score, fwd, sym, cost_bps, per_year)
    r = r[r != 0]
    if len(r) < 10:
        print(f"{name:<20} insufficient")
        return None
    sp, pe, mono = diagnostics(score, fwd)
    v = r.std() * math.sqrt(per_year)
    sh = r.mean() * per_year / v if v > 0 else 0.0
    t = (r.mean() / r.std()) * math.sqrt(len(r))
    wl = float((1 + r).cumprod().iloc[-1])
    print(f"{name:<20} {len(r):>5} {r.mean()*per_year:>7.1%} {v:>7.1%} {sh:>7.2f} "
          f"{t:>6.2f} {wl:>8.2f} {sp:>8.4f} {pe:>8.4f} {mono:>6.2f}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intraday", default="/tmp/quant_cache/intraday")
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--lookback", type=int, default=20, help="days aggregated, the report's 'month'")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--freq", default="weekly", choices=["weekly", "daily"])
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    fr = load_intraday(a.intraday)
    print(f"intraday fields: {sorted(fr)}   symbols {len(fr['cpv'].columns)}   "
          f"days {len(fr['cpv'])}")

    L = a.lookback
    raw: dict[str, pd.DataFrame] = {}
    raw["cpv_mean"] = fr["cpv"].rolling(L, min_periods=L // 2).mean()
    raw["cpv_vol"] = fr["cpv"].rolling(L, min_periods=L // 2).std()
    raw["cpv_trend"] = fr["cpv"].rolling(L, min_periods=L // 2).apply(slope, raw=True)
    for f in ("q", "ofi", "ifd", "kyle", "rsj"):
        raw[f] = fr[f].rolling(L, min_periods=L // 2).mean()

    if a.freq == "weekly":
        grid, per_year = cw.index, 52.0
        px = cw
    else:
        px = dcl
        grid, per_year = dcl.index, 365.0
    cols = [c for c in sym if c in fr["cpv"].columns]
    fwd = px[cols].shift(-1) / px[cols] - 1.0

    # point-in-time: value known at or before the rebalance stamp
    def at_grid(df):
        return df.reindex(df.index.union(grid)).ffill().reindex(grid)[cols]

    sc = {k: at_grid(v).apply(_rank_z) for k, v in raw.items()}
    smb = smb_scores(vw, sym)
    smb = at_grid(smb.reindex(columns=cols)).apply(_rank_z) if a.freq == "daily" \
        else smb[cols].apply(_rank_z)

    print(f"\nrebalance={a.freq}  lookback={L}d  cost={a.cost_bps}bps  "
          f"symbols={len(cols)}")
    print(f"{'factor':<20} {'n':>5} {'ann':>7} {'vol':>7} {'Sharpe':>7} {'t':>6} "
          f"{'wealth':>8} {'spIC':>8} {'peIC':>8} {'mono':>6}")
    print("-" * 100)
    show("SMB (baseline)", smb, fwd, cols, a.cost_bps, per_year)
    print("-- reproduced (sign from source) " + "-" * 66)
    show("CPV mean  [-]", -sc["cpv_mean"], fwd, cols, a.cost_bps, per_year)
    show("CPV vol   [-]", -sc["cpv_vol"], fwd, cols, a.cost_bps, per_year)
    show("CPV trend [-]", -sc["cpv_trend"], fwd, cols, a.cost_bps, per_year)
    comb = (-(sc["cpv_mean"] + sc["cpv_vol"] + sc["cpv_trend"])).apply(_rank_z)
    show("CPV combined [-]", comb, fwd, cols, a.cost_bps, per_year)
    show("SmartMoney Q [-]", -sc["q"], fwd, cols, a.cost_bps, per_year)
    show("RSJ [-]", -sc["rsj"], fwd, cols, a.cost_bps, per_year)
    show("OFI [+]", sc["ofi"], fwd, cols, a.cost_bps, per_year)
    print("-- invented here (both signs shown, none selected) " + "-" * 48)
    show("IFD [+]", sc["ifd"], fwd, cols, a.cost_bps, per_year)
    show("IFD [-]", -sc["ifd"], fwd, cols, a.cost_bps, per_year)
    show("KyleAsym [+]", sc["kyle"], fwd, cols, a.cost_bps, per_year)
    show("KyleAsym [-]", -sc["kyle"], fwd, cols, a.cost_bps, per_year)


if __name__ == "__main__":
    main()
