"""LOP -- Leveraged Overhang Pressure. A crypto-native mechanism, not a transfer.

THE REASONING, FROM FIRST PRINCIPLES
------------------------------------
Four positioning factors tested today all worked CONTRARIAN -- fade whoever just
crowded in (top-trader L/S change, whale-retail spread, price-OI quadrant, OI
change). Four factors agreeing is a hint that they are four crude proxies for
ONE underlying quantity. This file is an attempt to measure that quantity
directly.

What is mechanically unique about perpetual futures is that positions are
FORCIBLY closed by the exchange. Equities have margin calls and traditional
futures have them too, but neither has automated cascading auto-liquidation as a
dominant flow event. So perps contain a quantity that exists nowhere else:
LATENT FORCED-LIQUIDATION PRESSURE -- how much involuntary flow sits waiting
below (or above) the current price.

That reframes crowding-reversal. It is not investors CHOOSING to fade a crowded
trade (behavioural); it is the exchange FORCING exits (mechanical).

WHY THIS PROJECT'S RCGO FAILED, AND WHAT CHANGES
------------------------------------------------
RCGO implemented Grinblatt-Han: a turnover-weighted reference price, and capital
gain overhang relative to it. In spot equities the story is the disposition
effect -- holders sitting on losses are RELUCTANT to sell. It destroyed
performance here at every weight and sign.

The same arithmetic has a different meaning in a leveraged market, and the
weighting variable is wrong. What creates a liquidatable position is not volume
TRADED -- a coin can churn all day between the same two traders and create no
new exposure -- it is OPEN INTEREST CREATED. Turnover measures activity; DeltaOI
measures how much leveraged position now exists and at what price it was opened.
We had no OI series until the positioning backfill, so this substitution was not
previously testable.

    RCGO   : weight past prices by TURNOVER    -> behavioural reluctance
    LOP    : weight past prices by DeltaOI+    -> mechanical liquidation

CONSTRUCTION
------------
Following the 筹码分布 (chip distribution) formalism, which splits holdings into
获利盘 (in profit) and 套牢盘 (trapped), with the Grinblatt-Han survival weights
but OI in place of turnover:

    RP_w   = SUM_n [ dOI+(w-n) * PROD_{m<n}(1 - c(w-m)) ] * P(w-n)  / (normaliser)
    LGO_w  = (P_w - RP_w) / RP_w
    trapped_w = share of surviving OI opened ABOVE the current price

where c = closure rate = max(0, -dOI)/OI is the fraction of open positions that
went away that period -- the OI analogue of Grinblatt-Han's turnover survival term.

DIRECTION MATTERS, AND IT IS ASYMMETRIC
---------------------------------------
Trapped positions only matter if we know WHICH SIDE is trapped -- underwater
longs are forced sellers, underwater shorts are forced buyers. The long/short
ratio supplies that. And the asymmetry is real and documented: a study of BitMEX
perpetual data finds 3.51% of LONG positions versus 1.89% of SHORT positions
face forced liquidation daily, at ~60x average leverage among liquidated
positions. Longs liquidate at roughly twice the rate of shorts, so the pressure
term is scaled accordingly rather than treated as symmetric.

    LOP = trapped * (long_share - LONG_LIQ_ASYM * short_share)

SIGN IS NOT ASSUMED
-------------------
There are two defensible readings and they point opposite ways: high pressure
could predict NEGATIVE returns (the cascade is coming) or POSITIVE returns (the
cascade already happened and forced selling is exhausted). Both signs are
reported; neither is selected here.

Run:
    uv run python -m scripts.research_lop
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.factor_core as fc
from scripts.research_fas_clean import _liquidity_mask, load

LONG_LIQ_ASYM = 3.51 / 1.89   # BitMEX: longs forced out ~1.86x as often as shorts


def load_positioning(d: str, cols=None):
    P: dict[str, dict] = {}
    for p in sorted(Path(d).glob("*.json")):
        recs = json.loads(p.read_text())
        if len(recs) < 50:
            continue
        idx = pd.to_datetime([r["day"] for r in recs], utc=True)
        for f in ("sum_open_interest", "sum_toptrader_long_short_ratio_mean",
                  "count_long_short_ratio_mean"):
            s = pd.Series([r.get(f) for r in recs], index=idx, dtype=float)
            if s.notna().sum() > 40:
                P.setdefault(f, {})[p.stem] = s
    return {k: pd.DataFrame(v).sort_index() for k, v in P.items()}


def lop_components(px: pd.DataFrame, oi: pd.DataFrame, lookback: int = 26):
    """Reference price, gain overhang and trapped share, weighted by OI creation."""
    d_oi = oi.diff()
    opened = d_oi.clip(lower=0.0)                       # new leveraged positions
    closed = (-d_oi).clip(lower=0.0)
    closure = (closed / oi.shift(1).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    rp = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    trapped = pd.DataFrame(np.nan, index=px.index, columns=px.columns)

    surv_log = np.log1p(-closure.clip(upper=0.999))     # log survival per period
    for i in range(lookback, len(px)):
        w = px.index[i]
        sl = slice(i - lookback, i + 1)
        P = px.iloc[sl].to_numpy(dtype=float)           # (L+1, N)
        O = opened.iloc[sl].to_numpy(dtype=float)
        S = surv_log.iloc[sl].to_numpy(dtype=float)
        # survival of a cohort opened at row r through to the last row
        cum = np.cumsum(S[::-1], axis=0)[::-1]          # sum of log-survival AFTER r
        wgt = O * np.exp(np.nan_to_num(cum - S, nan=0.0))
        wgt = np.where(np.isfinite(wgt) & (wgt > 0), wgt, 0.0)
        tot = wgt.sum(axis=0)
        ok = tot > 0
        if not ok.any():
            continue
        rp_row = np.full(P.shape[1], np.nan)
        rp_row[ok] = (wgt[:, ok] * P[:, ok]).sum(axis=0) / tot[ok]
        rp.iloc[i] = rp_row
        cur = P[-1]
        above = np.where(P > cur[None, :], wgt, 0.0).sum(axis=0)
        tr = np.full(P.shape[1], np.nan)
        tr[ok] = above[ok] / tot[ok]
        trapped.iloc[i] = tr

    lgo = (px - rp) / rp
    return rp, lgo, trapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--positioning", default="/tmp/quant_cache/positioning")
    ap.add_argument("--lookback", type=int, default=26)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    P = load_positioning(a.positioning)
    cols = [c for c in sym if c in P["sum_open_interest"].columns]

    def G(df):
        return df[cols].reindex(df.index.union(cw.index)).ffill().reindex(cw.index)

    OI, TOP, ALL = (G(P["sum_open_interest"]),
                    G(P["sum_toptrader_long_short_ratio_mean"]),
                    G(P["count_long_short_ratio_mean"]))
    px = cw[cols]
    fwd = px.shift(-1) / px - 1.0
    print(f"symbols {len(cols)}   weeks {len(px)}   lookback {a.lookback}w")

    rp, lgo, trapped = lop_components(px, OI, a.lookback)
    print(f"LGO coverage: {int(lgo.notna().sum(axis=1).median())}/{len(cols)} symbols/week")
    print(f"LGO   mean {lgo.stack().mean():+.4f}  sd {lgo.stack().std():.4f}")
    print(f"trapped share  mean {trapped.stack().mean():.3f}  sd {trapped.stack().std():.3f}")

    # long share implied by the L/S ratio: r/(1+r)
    long_share = (TOP / (1.0 + TOP)).clip(0.0, 1.0)
    short_share = 1.0 - long_share
    direction = long_share - LONG_LIQ_ASYM * short_share
    lop = trapped * direction

    # the same construction with TURNOVER weights -- i.e. what RCGO effectively did
    _, lgo_vol, trapped_vol = lop_components(px, vw[cols].cumsum(), a.lookback)
    lop_vol = trapped_vol * direction

    def sh(r, p=52.0):
        r = r[r != 0]
        if len(r) < 20:
            return float("nan"), float("nan")
        v = r.std() * math.sqrt(p)
        return (r.mean() * p / v if v > 0 else 0.0), (r.mean() / r.std()) * math.sqrt(len(r))

    print(f"\n{'factor':<34} {'TS +':>14} {'TS -':>14} {'gross':>7}")
    print("-" * 74)
    for n, v in (("LGO (OI-weighted overhang)", lgo),
                 ("trapped share", trapped),
                 ("LOP = trapped x direction", lop),
                 ("-- RCGO-style, TURNOVER weights --", None),
                 ("LGO_vol (turnover-weighted)", lgo_vol),
                 ("LOP_vol (turnover-weighted)", lop_vol)):
        if v is None:
            print(f"{n:<34}")
            continue
        z = fc.ts_rank_pit(v)
        out = []
        vals = []
        for s in (1, -1):
            r = fc.backtest({"x": s * z}, fwd, cols, cost_bps=a.cost_bps)
            x, t = sh(r)
            out.append(f"{x:>7.2f}(t{t:>4.1f})")
            vals.append(x)
        gross = (vals[0] - vals[1]) / 2 if all(np.isfinite(vals)) else float("nan")
        print(f"{n:<34} " + " ".join(out) + f" {gross:>7.2f}")


if __name__ == "__main__":
    main()
