"""Corrected factor primitives.

Three defects in research_fas_clean._rank_z silently shaped every result this
project has produced. They are fixed here rather than in place, so the live
signal path is not changed mid-session and the two can be compared directly.

DEFECT 1 -- LOOK-AHEAD (the serious one).
    `df.apply(_rank_z)` uses pandas' DEFAULT axis=0, so _rank_z receives each
    COLUMN -- one symbol's entire time series -- and ranks a given week against
    the symbol's whole history, INCLUDING WEEKS THAT HAVE NOT HAPPENED YET.
    Measured directly: deleting weeks 200+ changes the week-150 score for
    98 of 112 symbols (max change 0.99). Any backtest built on it is reading
    the future.

    Fixed two ways depending on what the factor means:
      * a CROSS-SECTIONAL factor ranks symbols against each other within one
        week -> xs_rank (axis=1). No time axis is touched, so no leak exists.
      * a TIME-SERIES factor ranks a symbol against its OWN PAST -> ts_rank_pit,
        a trailing rolling window that ends at the current bar.

DEFECT 2 -- UNCENTRED SCORE.
    `(rank()/n) - 0.5` with rank in 1..n spans [1/n - 0.5, 0.5]: its mean is
    1/n, not 0. Harmless for an additive rank-sorted book (a constant shift
    cannot reorder anything) but NOT harmless where scores are MULTIPLIED, and
    FAS is exactly that -- `-pr_z * res_z`. Correct form is (rank - 0.5)/n.

DEFECT 3 -- NaN BECOMES A MID-RANK.
    _rank_z ends with .fillna(0.0). Zero is the CENTRE of the score range, so a
    symbol with missing data is silently handed a median score and enters the
    book as a real candidate instead of being excluded. It also makes the
    function non-idempotent: calling it twice (which several call sites did,
    since smb_scores already ranks internally) converts those filled zeros into
    genuine ranked positions and changes the answer. NaN is preserved here and
    the book construction drops it.

Naming note: what this project has called "SMB" is NOT a size factor. Because
of defect 1 it ranked each coin's volume against its own history, which is
abnormal-volume / 异常换手率 -- a documented effect ("abnormal turnover
negatively predicts returns"), but a time-series one. The genuine
cross-sectional size factor is available here as xs_rank(-log volume).

MEASURED 2026-08-16: as a standalone book it earns ann Sharpe +0.621 at
t = 1.50 -- POSITIVE but statistically indistinguishable from zero. An earlier
version of this note claimed it earns a NEGATIVE Sharpe; that was never
computed and is wrong. The defensible statement is that cross-sectional size
carries no reliable signal on this universe, while the same input ranked
self-referentially (abnormal volume) earns +1.222. See
research/SRP_RESEARCH_LOG.md §5.1 for the full 11-factor comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _centred_rank(v: pd.Series) -> pd.Series:
    """(rank - 0.5)/n mapped to [-1, 1]. Centred, antisymmetric, NaN-preserving."""
    x = v.dropna()
    n = len(x)
    if n < 5:
        return pd.Series(np.nan, index=v.index)
    r = ((x.rank() - 0.5) / n - 0.5) * 2.0
    return r.reindex(v.index)


def xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """CROSS-SECTIONAL rank: within each row (date), across symbols.

    This is the correct primitive for any factor whose claim is "coin A is
    cheaper/smaller/more crowded THAN coin B". It cannot leak, because a row
    contains only one date.
    """
    return df.apply(_centred_rank, axis=1)


def ts_rank_pit(df: pd.DataFrame, window: int = 104, min_periods: int = 52) -> pd.DataFrame:
    """TIME-SERIES rank against the symbol's OWN TRAILING window.

    This is the correct primitive for "this coin's volume is unusual FOR IT".
    The window ends at the current bar, so no future observation can enter.
    """
    def _last(a: np.ndarray) -> float:
        s = pd.Series(a).dropna()
        if len(s) < 5:
            return np.nan
        return float(((s.rank().iloc[-1] - 0.5) / len(s) - 0.5) * 2.0)

    return df.rolling(window, min_periods=min_periods).apply(_last, raw=True)


def abnormal_volume(vol_w: pd.DataFrame, weeks: int = 12, window: int = 104) -> pd.DataFrame:
    """异常换手率: trailing volume ranked against the symbol's own past.

    Negated so the book is LONG coins whose activity is quiet relative to their
    own norm and SHORT the ones being traded abnormally hard, which is the sign
    the Chinese turnover literature reports.

    This is the factor this project has been calling SMB. It is not SMB.
    """
    lv = np.log(vol_w.rolling(weeks).sum().clip(lower=1e-9))
    return -ts_rank_pit(lv, window=window)


def size_xs(vol_w: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    """The genuine cross-sectional size tilt -- long small, short large."""
    lv = np.log(vol_w.rolling(weeks).sum().clip(lower=1e-9))
    return xs_rank(-lv)


def quintile_weights(score: pd.Series, sym, top: float = 0.20) -> pd.Series | None:
    """Long/short quintile book. NaN is EXCLUDED, never treated as mid-rank."""
    s = score.dropna()
    if len(s) < 10:
        return None
    r = s.sort_values()
    n = max(2, int(round(top * len(r))))
    w = pd.Series(0.0, index=sym)
    w[list(r.index[-n:])] = 1.0 / n
    w[list(r.index[:n])] = -1.0 / n
    return w


def backtest(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame, sym,
             cost_bps: float = 10.0, cap: float = 1.0, top: float = 0.20) -> pd.Series:
    """Equal-weight combination of already-ranked factors, quintile long/short."""
    grid = [w for w in next(iter(factors.values())).index if w in fwd.index]
    rets, ridx, prev = [], [], None
    for w in grid:
        F = pd.DataFrame({k: v.loc[w].reindex(sym) for k, v in factors.items()})
        combo = F.mean(axis=1, skipna=True)
        combo = combo[F.notna().any(axis=1)]
        tgt = quintile_weights(combo, sym, top)
        if tgt is None:
            prev = None
            continue
        f = fwd.loc[w]
        if cap is not None:
            f = f.clip(upper=cap)
        ret = float((tgt * f).reindex(sym).sum(skipna=True))
        if prev is not None and cost_bps:
            ret -= cost_bps / 1e4 * float((tgt - prev).abs().sum())
        rets.append(ret)
        ridx.append(w)
        prev = tgt
    r = pd.Series(rets, index=ridx)
    return r[r != 0]


def stats(r: pd.Series, per_year: float = 52.0) -> dict:
    import math
    if len(r) < 20:
        return {"n": len(r), "sharpe": float("nan"), "t": float("nan")}
    v = r.std() * math.sqrt(per_year)
    return {
        "n": len(r),
        "ann": r.mean() * per_year,
        "vol": v,
        "sharpe": r.mean() * per_year / v if v > 0 else 0.0,
        "t": (r.mean() / r.std()) * math.sqrt(len(r)),
        "wealth": float((1 + r).cumprod().iloc[-1]),
    }


def leak_test(factor_fn, df: pd.DataFrame, at: int = 150, truncate: int = 200) -> float:
    """Max score change at row `at` when rows past `truncate` are deleted.

    Any point-in-time factor must return ~0. Run this on every new factor.
    """
    a = factor_fn(df)
    b = factor_fn(df.iloc[:truncate])
    w = df.index[at]
    if w not in a.index or w not in b.index:
        return float("nan")
    return float((a.loc[w] - b.loc[w]).abs().max())
