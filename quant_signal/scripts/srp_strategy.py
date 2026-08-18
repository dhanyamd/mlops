"""SRP -- Self-Referential Parity. The single source of truth for the book.

This module is the MIDDLEWARE between research and production. Both the backtest
and the live signal call ``srp_weights``; neither reimplements it. That rule
exists because this project has been bitten twice by the alternative:

  * the live book once diverged from research at 0.147 rank correlation with 27%
    selection overlap (chance is ~20%) -- it was trading a different strategy
    while every dashboard said otherwise;
  * a look-ahead bug (``_rank_z`` ranking each week against a symbol's ENTIRE
    history, future included) survived for months because research and live
    shared the same wrong helper and nothing compared them to anything.

So: one pure function, no I/O, no globals, no environment reads. Frames in,
target weights out. ``scripts/srp_parity.py`` asserts live and research produce
identical selections; if they ever drift, that test fails rather than the P&L.

THE STRATEGY
------------
Nine factors, all reproductions of published Chinese sell-side research. The
construction is what this project contributes.

MEASURED, by ``scripts/srp_backtest.py`` over 282 weekly rebalances (112
symbols, 363-week panel), net of funding and liquidity-scaled maker costs:

    annualised Sharpe 2.161 at the shipped config (turnover_cap 0.60)
                      2.364 uncapped
    t-statistic       5.03 / 5.51

NOT YET MEASURED. An earlier version of this docstring asserted "identical
inputs score 1.03 under the conventional pipeline and 2.40 under this one".
The 2.40 is reproducible -- it is the UNCAPPED run above. The 1.03 is not:
no script in this repository produces it, and it appears to have been an
untuned baseline compared against a tuned strategy, which is not a valid
comparison. ``scripts/srp_sweep.py`` now runs all six constructions over one
hyperparameter grid and ``scripts/srp_ablation.py`` reports the paired and
best-of-grid gaps. Quote those, not this paragraph, until the sweep completes.

  1. SELF-REFERENTIAL RANKING. Every factor is ranked against the symbol's OWN
     trailing 52 weeks, not against its peers. Published cross-sectionally, these
     factors are weak-to-negative in crypto (AVOL: -0.25) and strong
     self-referentially (+0.80). The CTA literature predicts the opposite.

  2. ONE BOOK PER FACTOR. Scores are NEVER blended. Each factor forms its own
     quintile long/short book; the RETURNS are combined, not the scores.
     Blending forces nine near-independent signals through a single ranking and
     discards most of their independence before a trade is placed.

  3. INVERSE-VOL RISK PARITY across those books, using trailing realised vol
     only (shifted, so no future information enters the weights).

  4. FUNDING TILT. Perpetuals pay funding three times a day; a price-only
     backtest overstates this book by ~3.9%/yr. Each factor's score is penalised
     by the symbol's cross-sectional funding rank so the book stops taking
     positions it must pay to hold.

  5. TURNOVER CAP. A uniform partial adjustment toward the target. It is
     INSURANCE AGAINST COST MISESTIMATION, and it is not free -- an earlier
     version of this docstring claimed it added value through "smoothing", which
     the sweep falsifies. At the modelled maker fee (1-5bp on the
     cross-sectional dollar-volume rank) the cap COSTS Sharpe, monotonically:

         cap   None   2.0    1.0    0.8    0.6    0.4    0.2
         ann  2.364  2.388  2.298  2.231  2.161  2.012  1.581

     What it buys is survival when fills are worse than modelled. Scaling the
     whole cost curve, the uncapped book is best only while costs stay near the
     assumption, and it collapses past ~8x:

         cost x    1      2      4      8     16     32
         uncap  2.364  2.247  2.014  1.547  0.618  -1.222
         0.60   2.161  2.117  2.028  1.851  1.496   0.789

     Break-even is ~4x the modelled fee. The live path takes market fallbacks
     when a maker order does not fill, so realised cost sits above the maker
     assumption and the capped book is the correct live default -- but the
     justification is robustness, not a higher expected Sharpe. Schemes that
     vary the adjustment speed per symbol (we tried two) destroy the property.

POINT-IN-TIME
-------------
Every input must already be point-in-time when passed in. This module adds no
shifts and does no forward filling; it will happily compute look-ahead if given
look-ahead. ``factor_core.leak_test`` is the guard for the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.factor_core import quintile_weights, ts_rank_pit, xs_rank

# Factor sign conventions, fixed a priori from each source. NOT fitted here.
#   AVOL      异常换手率 -- abnormal turnover predicts negatively
#   Q         聪明钱 (方正/开源) -- sign inverted vs A-shares: crypto perps are
#             momentum-driven where A-shares are reversal-driven
#   RSJ       国泰君安 signed jump
#   CPVm/CPVv 东吴 价量相关性, level and instability
#   OFI       天风 买卖压力失衡
#   WRspread / TopChg / Quad -- positioning, all CONTRARIAN to crowding
#   TSKD      INVENTED HERE. 开源证券 showed the alpha in per-bar ticket size lives
#             in the DISTRIBUTION SHAPE (skew/kurtosis/quantiles, Rank ICIR 3.57),
#             not the mean -- but their measure is direction-blind because A-share
#             tick data carries no reliable aggressor side. Splitting that
#             distribution by buy- vs sell-dominated bars requires taker volume,
#             which only a centralised crypto venue publishes. The CHANGE carries
#             the signal (gross 0.85) where the level does not (0.25), matching
#             the 龙虎榜 "多转空" pattern. Walk-forward: +0.07 OOS alone.
#             REQUIRES 5-MINUTE BARS: at 1h there are ~12 bars per side, below the
#             20-ticket minimum the shape estimator needs (0% computable).
#   TKU       开源证券 kurtosis of the same distribution. Borrowed. +0.12 OOS.
FACTORS = ("AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv", "WRspread", "TopChg", "Quad")
TICKET_FACTORS = ("TSKD", "TKU")


@dataclass
class SRPConfig:
    rank_window: int = 52          # weeks of own history for the self-referential rank
    rank_min_periods: int = 26
    vol_window: int = 52           # trailing window for risk-parity weights
    vol_min_periods: int = 26
    funding_tilt: float = 0.5      # weight on the cross-sectional funding penalty
    turnover_cap: float | None = 0.60
    top: float = 0.20              # quintile
    min_symbols: int = 10
    # Do NOT trade until inverse-vol weights are actually estimable. Measured:
    # the equal-weight fallback period scores -0.11 Sharpe on its own and drags
    # the full sample from 2.23 to 1.90. Going FLAT while the risk model warms
    # up is strictly better than trading an unweighted book.
    require_risk_parity: bool = True


def build_factors(
    *,
    weekly_close: pd.DataFrame,
    weekly_volume: pd.DataFrame,
    intraday: dict[str, pd.DataFrame],
    open_interest: pd.DataFrame,
    top_ls: pd.DataFrame,
    all_ls: pd.DataFrame,
    smooth: int = 20,
    ticket: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Raw factor values on the weekly grid, signed. All inputs point-in-time.

    ``intraday`` carries the daily series reduced from intraday bars (q, rsj,
    ofi, cpv) reindexed onto the weekly grid by the caller.
    """
    idx, cols = weekly_close.index, list(weekly_close.columns)

    def _sm(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(columns=cols).rolling(smooth, min_periods=max(2, smooth // 2)).mean()

    ret = weekly_close / weekly_close.shift(1) - 1.0
    d_oi = np.log(open_interest.reindex(index=idx, columns=cols).clip(lower=1e-9)).diff()
    cpv = intraday["cpv"].reindex(index=idx, columns=cols)
    cpv_roll = cpv.rolling(smooth, min_periods=max(2, smooth // 2))

    out_ticket: dict[str, pd.DataFrame] = {}
    if ticket:
        # 5-minute-derived only; absent at hourly resolution by construction.
        if "tskew_dir" in ticket:
            tsk = ticket["tskew_dir"].reindex(index=idx, columns=cols)
            out_ticket["TSKD"] = _sm(tsk).diff()      # the CHANGE carries the signal
        if "tku" in ticket:
            out_ticket["TKU"] = _sm(ticket["tku"].reindex(index=idx, columns=cols))

    return {
        **out_ticket,
        "AVOL": -np.log(weekly_volume[cols].rolling(12).sum().clip(lower=1e-9)),
        "Q": _sm(intraday["q"]).reindex(index=idx),
        "RSJ": -_sm(intraday["rsj"]).reindex(index=idx),
        "OFI": _sm(intraday["ofi"]).reindex(index=idx),
        "CPVm": -cpv_roll.mean(),
        "CPVv": -cpv_roll.std(),
        "WRspread": -(
            top_ls.reindex(index=idx, columns=cols)
            - all_ls.reindex(index=idx, columns=cols)
        ),
        "TopChg": -top_ls.reindex(index=idx, columns=cols).diff(),
        "Quad": -(np.sign(ret) * d_oi),
    }


def factor_scores(raw: dict[str, pd.DataFrame], cfg: SRPConfig) -> dict[str, pd.DataFrame]:
    """Self-referential rank: each symbol against its OWN trailing window."""
    return {
        k: ts_rank_pit(v, window=cfg.rank_window, min_periods=cfg.rank_min_periods)
        for k, v in raw.items()
    }


def factor_book_weights(
    score: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: SRPConfig,
    prev: pd.Series | None,
    stamp,
    symbols: list[str],
) -> pd.Series | None:
    """Target weights for ONE factor at ONE rebalance, with tilt and turnover cap."""
    if stamp not in score.index:
        return None
    s = score.loc[stamp].dropna()
    if cfg.funding_tilt and stamp in funding.index:
        ftz = xs_rank(funding.loc[[stamp]]).loc[stamp]
        s = s - cfg.funding_tilt * ftz.reindex(s.index).fillna(0.0)
    if len(s) < cfg.min_symbols:
        return None
    tgt = quintile_weights(s, symbols, top=cfg.top)
    if tgt is None:
        return None
    if prev is not None and cfg.turnover_cap is not None:
        d = tgt - prev
        turn = float(d.abs().sum())
        if turn > cfg.turnover_cap:
            tgt = prev + d * (cfg.turnover_cap / turn)
    return tgt


def risk_parity_weights(book_returns: pd.DataFrame, cfg: SRPConfig) -> pd.DataFrame:
    """Inverse trailing vol, shifted so no future information enters."""
    iv = 1.0 / book_returns.rolling(cfg.vol_window, min_periods=cfg.vol_min_periods).std().shift(1)
    return iv.div(iv.sum(axis=1), axis=0)


def srp_weights(
    scores: dict[str, pd.DataFrame],
    funding: pd.DataFrame,
    book_returns: pd.DataFrame | None,
    symbols: list[str],
    stamp,
    prev_books: dict[str, pd.Series] | None,
    cfg: SRPConfig,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Combined portfolio weights at one rebalance.

    Returns (portfolio weights over ``symbols``, per-factor books to carry
    forward as ``prev_books`` at the next rebalance).
    """
    prev_books = prev_books or {}
    books: dict[str, pd.Series] = {}
    for name, sc in scores.items():
        w = factor_book_weights(sc, funding, cfg, prev_books.get(name), stamp, symbols)
        if w is not None:
            books[name] = w
    if not books:
        return pd.Series(0.0, index=symbols), {}

    # ``book_returns`` MUST include rows up to and including ``stamp``;
    # risk_parity_weights shifts by one internally, so the weights used at
    # ``stamp`` are estimated strictly from returns BEFORE it. Passing a frame
    # that stops short of ``stamp`` silently yields no weights (and, with
    # require_risk_parity, a permanently FLAT book).
    wts = None
    if book_returns is not None and not book_returns.empty:
        avail = [b for b in books if b in book_returns.columns]
        if avail:
            rpw = risk_parity_weights(book_returns[avail], cfg)
            row = (rpw.loc[stamp] if stamp in rpw.index
                   else (rpw.iloc[-1] if len(rpw) else None))
            if row is not None and row.notna().any():
                wts = row.reindex(list(books))
    if wts is None or not wts.notna().any():
        if cfg.require_risk_parity:
            # risk model not warm yet -> stay FLAT rather than trade unweighted
            return pd.Series(0.0, index=symbols), {}
        wts = pd.Series(1.0 / len(books), index=list(books))
    wts = wts.fillna(0.0)
    wts = wts.reindex(list(books)).fillna(0.0)
    if wts.sum() <= 0:
        wts = pd.Series(1.0 / len(books), index=list(books))
    wts = wts / wts.sum()

    port = pd.Series(0.0, index=symbols, dtype=float)
    for name, w in books.items():
        port = port.add(w.reindex(symbols).fillna(0.0) * float(wts[name]), fill_value=0.0)
    return port, books


def directions(port: pd.Series, eps: float = 1e-9) -> dict[str, str]:
    """Portfolio weights -> the LONG/SHORT/FLAT the executor consumes."""
    out: dict[str, str] = {}
    for s, w in port.items():
        out[s] = "LONG" if w > eps else ("SHORT" if w < -eps else "FLAT")
    return out
