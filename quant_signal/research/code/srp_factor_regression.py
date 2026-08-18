"""Is it alpha, or exposure to known factors?

A Sharpe ratio answers "did this make money". It does not answer the question an
asset-pricing referee actually asks, which is whether the return survives
controlling for factors already known to be priced. \\citet{cong2026} construct a
four-factor cross-sectional model for digital assets; a strategy in this market
that does not report its loading on such factors has not established alpha, only
performance.

This regresses the strategy's weekly return on three controls built from the same
panel, all point-in-time:

    MKT   equal-weight return of the tradeable universe
    SIZE  long the smallest quintile by trailing dollar volume, short the largest
    MOM   long the top quintile by trailing 12-week return, short the bottom

    r_t = alpha + b_MKT MKT_t + b_SIZE SIZE_t + b_MOM MOM_t + e_t

The quantity of interest is alpha and its significance. A positive, significant
alpha means the return is not spanned by these factors.

STANDARD ERRORS
---------------
Weekly strategy returns are frequently autocorrelated, which biases ordinary
t-statistics upward. We therefore report Newey--West standard errors alongside
OLS ones, and quote the more conservative of the two. Reporting only the OLS
figure would overstate significance in exactly the direction that flatters the
strategy.

Run:
    uv run python -m scripts.srp_factor_regression
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.factor_core import quintile_weights, xs_rank
from scripts.srp_backtest import SRPConfig, SRPData, run


def _long_short(signal: pd.DataFrame, data: SRPData, top: float = 0.20) -> pd.Series:
    """Weekly return of a dollar-neutral quintile book formed on ``signal``."""
    out, idx = [], []
    for w in signal.index:
        if w not in data.fwd.index:
            continue
        s = signal.loc[w].dropna()
        if len(s) < 20:
            continue
        tgt = quintile_weights(s, data.cols, top=top)
        if tgt is None:
            continue
        out.append(float((tgt * data.fwd.loc[w]).reindex(data.cols).sum(skipna=True)))
        idx.append(w)
    return pd.Series(out, index=idx)


def newey_west_se(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC standard errors (Newey--West, Bartlett kernel)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = resid[L:, None] * X[L:]
        v = resid[:-L, None] * X[:-L]
        G = u.T @ v
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mom-window", type=int, default=12)
    ap.add_argument("--nw-lags", type=int, default=4)
    a = ap.parse_args()

    d = SRPData.load()
    cfg = SRPConfig()
    strat = run(d, cfg).returns

    # --- controls, all point-in-time -------------------------------------
    px, vol = d.cw, d.vw
    mkt = d.fwd.mean(axis=1)                                   # equal-weight market

    dollar = (vol * px).replace(0, np.nan)
    size_sig = xs_rank(-np.log(dollar.rolling(12).sum().clip(lower=1e-9)))
    size = _long_short(size_sig, d)                            # small minus big

    mom_sig = xs_rank(px / px.shift(a.mom_window) - 1.0)
    mom = _long_short(mom_sig, d)                              # winners minus losers

    J = pd.concat({"r": strat, "MKT": mkt, "SIZE": size, "MOM": mom}, axis=1).dropna()
    n = len(J)
    y = J["r"].to_numpy()
    X = np.column_stack([np.ones(n), J["MKT"], J["SIZE"], J["MOM"]])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - X.shape[1]
    s2 = resid @ resid / dof
    se_ols = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    se_nw = newey_west_se(X, resid, a.nw_lags)

    names = ["alpha", "MKT", "SIZE", "MOM"]
    print(f"SRP weekly return regressed on MKT, SIZE, MOM   (n = {n} weeks)\n")
    print(f"  {'':<7}{'coef':>10}{'t (OLS)':>10}{'t (NW)':>10}")
    print("  " + "-" * 37)
    for i, nm in enumerate(names):
        print(f"  {nm:<7}{beta[i]:>10.5f}{beta[i]/se_ols[i]:>10.2f}{beta[i]/se_nw[i]:>10.2f}")

    ann = beta[0] * 52
    t_ols, t_nw = beta[0] / se_ols[0], beta[0] / se_nw[0]
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    print()
    print(f"  annualised alpha        : {ann * 100:+.2f}%")
    print(f"  alpha t-stat (Newey-West, {a.nw_lags} lags): {t_nw:.2f}"
          f"   {'clears' if abs(t_nw) > 3 else 'BELOW'} the t>3.0 bar")
    print(f"  R-squared               : {r2:.4f}")
    print()
    print(f"  raw weekly mean return  : {y.mean() * 100:+.4f}%")
    print(f"  explained by factors    : {(y.mean() - beta[0]) * 100:+.4f}%")
    print(f"  UNEXPLAINED (alpha)     : {beta[0] * 100:+.4f}%"
          f"  = {100 * beta[0] / y.mean():.0f}% of the total")
    print()
    print(f"  autocorrelation of strategy returns (lag 1): {J['r'].autocorr(1):+.3f}")
    print("  Newey-West is the figure to quote; OLS is shown only for comparison.")


if __name__ == "__main__":
    main()
