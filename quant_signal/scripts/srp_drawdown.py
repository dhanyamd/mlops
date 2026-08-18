"""Drawdown, tail statistics, and behaviour through the 2022 collapse.

The paper establishes statistical significance at length but never shows what the
equity curve does. That is the first thing a practitioner looks for and the first
omission a referee notices. Significance answers "is this real"; drawdown answers
"could anyone actually hold it".

Two questions are addressed.

DRAWDOWN AND TAILS. Maximum peak-to-trough loss, its duration and recovery,
worst single week and month, and the higher moments of the weekly return
distribution. A Sharpe ratio summarises the first two moments only; crypto
returns are severely non-normal, so the third and fourth are reported alongside.

THE 2022 COLLAPSE. The strategy claims market-neutrality, evidenced by a market
beta indistinguishable from zero. A regression coefficient is an average over the
sample; it does not by itself establish behaviour in the one period where
neutrality matters most. Cryptocurrency fell heavily through 2022. We therefore
report the strategy's return across that window next to the market's, which tests
the neutrality claim where it is most likely to fail rather than on average.

Run:
    uv run python -m scripts.srp_drawdown
"""

from __future__ import annotations

import argparse
import math

import pandas as pd
from scipy import stats

from scripts.srp_backtest import SRPConfig, SRPData, run


def drawdown_stats(r: pd.Series) -> dict:
    """Peak-to-trough statistics on the compounded equity curve."""
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    trough = dd.idxmin()
    peak_before = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    recovered = after[after >= eq.loc[peak_before]]
    rec_date = recovered.index[0] if len(recovered) else None
    return {
        "max_dd": float(dd.min()),
        "peak": peak_before,
        "trough": trough,
        "recovered": rec_date,
        "dd_weeks": int((eq.loc[peak_before:trough]).shape[0] - 1),
        "rec_weeks": (int(eq.loc[trough:rec_date].shape[0] - 1)
                      if rec_date is not None else None),
        "time_underwater_pct": float((dd < -1e-12).mean() * 100),
        "equity": eq,
        "dd": dd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crash-start", default="2021-11-01")
    ap.add_argument("--crash-end", default="2022-12-31")
    a = ap.parse_args()

    d = SRPData.load()
    r = run(d, SRPConfig()).returns
    mkt = d.fwd.mean(axis=1).reindex(r.index).dropna()
    r = r.reindex(mkt.index)

    st = drawdown_stats(r)
    print(f"EQUITY AND DRAWDOWN   ({len(r)} weeks, "
          f"{r.index[0].date()} to {r.index[-1].date()})\n")
    print(f"  cumulative return        : {(st['equity'].iloc[-1] - 1) * 100:+.1f}%")
    print(f"  annualised return        : {r.mean() * 52 * 100:+.2f}%")
    print(f"  annualised volatility    : {r.std() * math.sqrt(52) * 100:.2f}%")
    print(f"  Sharpe                   : {r.mean() / r.std() * math.sqrt(52):.3f}")
    print()
    print(f"  MAXIMUM DRAWDOWN         : {st['max_dd'] * 100:.2f}%")
    print(f"    peak                   : {st['peak'].date()}")
    print(f"    trough                 : {st['trough'].date()}  "
          f"({st['dd_weeks']} weeks of decline)")
    print(f"    recovered              : "
          f"{st['recovered'].date() if st['recovered'] is not None else 'not yet'}"
          + (f"  ({st['rec_weeks']} weeks)" if st['rec_weeks'] is not None else ""))
    print(f"    time under water       : {st['time_underwater_pct']:.0f}% of weeks")
    print()
    print(f"  worst week               : {r.min() * 100:+.2f}%")
    print(f"  best week                : {r.max() * 100:+.2f}%")
    m = (1 + r).resample("ME").prod() - 1
    print(f"  worst month              : {m.min() * 100:+.2f}%")
    print(f"  positive weeks           : {(r > 0).mean() * 100:.1f}%")
    print()
    print(f"  skewness                 : {stats.skew(r):+.3f}")
    print(f"  kurtosis (excess)        : {stats.kurtosis(r):+.3f}")
    print("  Sharpe assumes normality; these say how far from it the data sits.")

    # ---- the 2022 collapse ------------------------------------------------
    lo, hi = pd.Timestamp(a.crash_start, tz="UTC"), pd.Timestamp(a.crash_end, tz="UTC")
    w = (r.index >= lo) & (r.index <= hi)
    if w.sum() < 10:
        print("\n  (insufficient data in the stated crash window)")
        return
    rc, mc = r[w], mkt[w]
    print(f"\nTHE {lo.date()} TO {hi.date()} COLLAPSE   ({w.sum()} weeks)\n")
    print(f"  {'':<26}{'market':>12}{'strategy':>12}")
    print("  " + "-" * 50)
    print(f"  {'cumulative return':<26}{((1+mc).prod()-1)*100:>11.1f}%"
          f"{((1+rc).prod()-1)*100:>11.1f}%")
    print(f"  {'annualised':<26}{mc.mean()*52*100:>11.1f}%{rc.mean()*52*100:>11.1f}%")
    print(f"  {'annualised volatility':<26}{mc.std()*math.sqrt(52)*100:>11.1f}%"
          f"{rc.std()*math.sqrt(52)*100:>11.1f}%")
    print(f"  {'Sharpe':<26}{mc.mean()/mc.std()*math.sqrt(52):>12.2f}"
          f"{rc.mean()/rc.std()*math.sqrt(52):>12.2f}")
    print(f"  {'max drawdown':<26}{drawdown_stats(mc)['max_dd']*100:>11.1f}%"
          f"{drawdown_stats(rc)['max_dd']*100:>11.1f}%")
    print(f"  {'positive weeks':<26}{(mc>0).mean()*100:>11.1f}%{(rc>0).mean()*100:>11.1f}%")
    print()
    print(f"  correlation over the window: {rc.corr(mc):+.3f}")


if __name__ == "__main__":
    main()
