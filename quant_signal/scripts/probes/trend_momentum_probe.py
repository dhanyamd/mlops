"""Short-Cycle Fused Momentum (SCFM) probe on REAL BRONZE CRYPTO_BARS minute
history (31-symbol universe, ~3 years).

Builds on the committed long-biased TSM design, but fixes the GAP the 2024-26
literature exposed: crypto momentum is SHORT-CYCLE and CRASH-PRONE, so equity-
style 4-week lookbacks with no crash control are the wrong config family.

  - Liu, Fang & Wang (中科院数学与系统科学研究院, 管理评论 36(6) 2024): crypto
    momentum lasts <= TWO WEEKS; the disposition effect <= half a month; weekly
    data invalid; capital-gains-overhang (CGO) + momentum beats either alone.
  - Li & Zhu (Res. Int. Bus. Finance 83, 2026, "Taming crypto anomalies"): the
    surviving crypto factors are MKT + two-week momentum (MOM2) + residual
    momentum; size died OOS and LEFT-TAIL RISK appeared OOS.
  - Fičura (VŠE, 2023): large liquid coins (BTC/ETH) show WEEKLY momentum and
    the distance-from-recent-high is a SUPERIOR predictor of returns.
  - Crypto factor zoo (.Zip), IRFA 2026: 7-day momentum + bid-ask survive
    transaction costs; momentum is the most cost-resilient factor.
  - Grobys et al. (FMPM 39, 2025): crypto momentum CRASHES severely;
    volatility-managing raises payoffs 200%+.
   - Hsieh, Huang & Liu (FRL 86, 2025): momentum works only in UP-UP regimes.
   - Grinblatt & Han (JFE 78, 2005): CGO is the driver behind momentum.
   - Huang, Sangiorgi & Urquhart (SSRN 4825389): volume-weighted TSMOM, Sharpe
     2.17; CTREND (JFQA 60, 2025): trend must fuse price AND volume.
   - Han, Kang & Ryu (SSRN 4675565): TSM strong, long-only beats long-short.
    - Sadaqat & Butt (JBEF 39, 2023): stop-loss momentum has the highest
      payoffs; 10-30% stops beat 40-50% -- realize losses sooner.
    - Grobys & Shahzad (IJFE 31(2), 2026): momentum variance is undefined
      (power law alpha<3) -> verdicts use median + winsorized stats too.
    - Bui & Nguyen (arXiv 2602.11708, "AdaptiveTrend"): 6-HOUR bars are the
      turnover x signal sweet spot (H1 Sharpe 1.54 -> H6 2.41 -> D1 1.63) and
      the ATR trailing stop is the SINGLE biggest payoff lift (ablation: +0.73
      Sharpe, -9.7pp MDD). OOS 2022-24, 150+ pairs, Sharpe 2.41, MDD -12.7%.
    - Bysik & Slepaczuk (arXiv 2606.00060): the cost-aware execution filter
      (lambda=2 x taker band: 20bps entry / 40bps flip) is the EDGE -- it turns
      naive sign-based ML from -64%/yr to +65.4%/yr, Sharpe 1.09.
    - Barroso-Santa-Clara / Moreira-Muir: risk-managed momentum.

The strategy (SCFM): long-biased, maker-executed, and the LONG entry requires
the AND of three published effects NO paper combines:
  (a) short-cycle trend  -- lookback in {2,3,5,7,14,28}d, LEARNED walk-forward;
  (b) proximity-to-high  -- close within {3,5}% of its rolling-lookback high
      (Fičura's superior continuation signal);
  (c) CGO > 0            -- aggregate unrealized gains positive (Grinblatt-Han
      disposition overlay), reference price = 1-week volume-weighted average.
Plus the crash layer: volatility-managed sizing (clip(baseline_rv/rv_24h,
0.5,2.0)), a trailing stop, and a crash guard (no entry into a -5%/-10% 24h
breakdown -- the left-tail-risk factor that now prices crypto).

Five sections are tested head-to-head so we can SEE whether each layer adds
value, not assume it:
   tsm     baseline long-biased TSM
   prox    + proximity-to-high filter
   cgo     + CGO>0 filter
   crash   + crash guard
   regime  + UP-UP market-state gate (A/B vs tsm)
   full    all layers combined (the bettable combo)

The ``atrend`` section A/B's the AdaptiveTrend exit layer against tsm on the
SAME real data: 6-hour bars (H6), LONG entries on fresh momentum up-flips
(MOM crosses from <= 0 to > 0 over L bars, fill at the next bar's open), and
the ATR(14) TRAILING stop S_t = max(S_{t-1}, C_t - atr_mult * ATR14_t) with a
30-day time fallback -- the exit mechanism AdaptiveTrend's ablation shows is
the biggest single payoff lift. atr_mult in {2.0, 2.5, 3.0} and the momentum
lookback in {7, 14, 28}d are LEARNED walk-forward, never hand-picked; the
crash and UP-UP regime gates remain available as entry filters.

The ``vreg`` section is the ANTI-OVERFIT answer to AdaptiveTrend: instead of
monthly grid-searching (L, alpha), it PRE-REGISTERS a pinned mapping from the
7-day realized-vol percentile (vs the prior 3 months) to (momentum lookback,
ATR multiplier): high vol -> L=7d/alpha=3.0, mid -> L=14d/alpha=2.5, low ->
L=28d/alpha=2.0. The mapping is NEVER re-fit, so there is nothing to overfit --
the vol regime IS the adaptation (AdaptiveTrend re-optimizes exactly this
state-dependent tradeoff, so vreg is its falsifiable, pinned twin). Entry
requires a fresh momentum up-flip over the regime-pinned L, a volume surge
(> 1.25x the prior 28-bar median, the CTREND price+volume fusion), the optional
UP-UP regime gate and crash guard; exit on the ATR(14) trailing stop, a 10%
hard stop (arXiv 2604.27150: tight stops beat wide), or the 30-day fallback.
Risk-managed sizing (clip(0.5, 2.0)) stays on as the most robust published
payoff lift (Barroso-Santa-Clara, Moreira-Muir, Grobys). The only LEARNED axes
are the crash/regime/vol-scale gates -- disclosed, per-symbol, walk-forward.

The ``xs_*`` sections are the CROSS-SECTIONAL answer the 2024-26 literature
converges on: the short-cycle / salience / anchor edges live in the CROSS-
SECTION of a wide universe, not in one asset's own time series (Fičura's PTH
and the MAX-fade both died on BTC/ETH alone). Over a ~31-name universe (top
liquid USDT pairs, 3y backfilled Binance 1Min history), each xs section ranks
the WHOLE market on ONE pre-registered signal at every WEEKLY UTC rebalance
(fill next open, exit next rebalance open, maker RT per leg) and goes LONG the
top quintile / SHORT the bottom quintile:
   xs_rel14  relative 2-week momentum (Keel MOM2 / Li-Zhu)     -- momentum
   xs_resm   residual momentum (Li & Zhu 2026): trailing-28-day BTC-beta
             residual summed over the lookback                    -- momentum
   xs_anchor anchor-based REVERSAL (SSRN 5001299): PTH vs the trailing-L
             max(HIGH); near the anchor high -> short             -- reversal
   xs_st     salience-theory ST (Bordalo; Cai & Zhao, JBF 159, 2024): high
             salience-weighted return -> overpriced -> short      -- reversal
The liquidity guard is PER-SYMBOL (Anomiq lesson): a name must still trade at
>= 0.5x its OWN trailing 28-bar median volume -- never a global gate. The crash
guard and UP-UP regime gate are MARKET-level (BTC = the market proxy). Config
selection is GLOBAL -- one config for the whole book, selected on the
in-sample POOLED legs, never per-symbol -- and the benchmark is the equal-
weight OOS basket of the same universe: a long-short book must beat plain
long-only Buy & Hold to pass.

Anti-overfit discipline (same as the committed design): NO config is hand-
picked. Within each section, each symbol's IN-SAMPLE half of events picks the
best config (max mean net-per-trade at maker costs); ONLY the OUT-OF-SAMPLE
half is scored. Every config's OOS result is disclosed. Signals are strictly
lagged (close of bar t decides, fill at open of t+1); lookbacks/RVs/CGO/prox/
crash are computed inside contiguous runs only. Every trade is scored at BOTH
our real costs: maker 4 bps RT and the taker gate 20 bps. The verdict compares
each section's OOS net-of-maker vs Buy & Hold over the SAME OOS window.

Run:  uv run python -m scripts.trend_momentum_probe [--out docs/trend_momentum_probe.json]
Smoke test (no-lookahead, no Snowflake):
      uv run python -m pytest tests/test_trend_momentum_probe.py -q
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from scripts.backfill_feature_windows import fetch_bars
from scripts.probes.intraday_30m_probe import _aggregate

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000

# Round trips in bps -- the live engine is maker-first (post-only near-side
# quote, Bybit 0.02% maker = 4 bps RT); the taker gate basis is 2 x 10 bps.
_MAKER_RT = 0.0004
_TAKER_RT = 0.0020
_GATE_TAKER_BPS = 2.0 * 10.0  # lambda=2 x 10 bps taker band

# Hourly lookbacks in hours: 2/3/5/7/14/28 days (short end per the crypto
# momentum literature; the long end is kept so the data can REJECT it).
LOOKBACKS_H = (48, 72, 120, 168, 336, 672)
# Max hold in hours (1/2/3/5 days); entries rebalance every ``hold`` hours on
# the UTC clock, so trades never overlap.
HOLDS_H = (24, 48, 72, 120)
# Trailing stop in percent (0.0 = none) -- exit at open of the first bar whose
# close is <= stop% below the running peak close.
STOPS = (0.0, 0.02, 0.04)
# Risk-managed momentum: position size = clip(baseline_rv / rv_24h, 0.5, 2.0).
VOL_SCALES = (False, True)

# Signal-fusion gates (all evaluated at the signal bar's close, known before
# the next-open fill). prox_tol: require close within tol% of the rolling
# lookback high. cgo: require capital-gains-overhang > 0 (1-week volume-
# weighted reference price). crash: skip entry if the 24h return is below -x.
PROX_TOLS = (0.05, 0.03)
CGO_REF_WINDOW_H = 168  # 1-week volume-weighted reference price
CRASH_LEVELS = (0.05, 0.10)
# UP-UP regime gate (Hsieh, Huang & Liu, FRL 86, 2025): crypto momentum works
# only in UP-UP transitions. Following Cooper et al. (2004), State[t] = 4-week
# close return >= 0, known at close of t-1. The market state is sampled WEEKLY,
# so UP-UP requires state[t] AND the immediately preceding weekly state (the
# Asem & Tian 2010 adjacent-period transition the paper builds on).
_REGIME_WINDOW_H = 672  # 4 weeks: state lookback window
_REGIME_STEP_H = 168  # 1 week: state sampled weekly, transition = adjacent week

# AdaptiveTrend-style H6 section (Bui & Nguyen, arXiv 2602.11708): aggregate to
# 6-hour bars; LONG entry on a fresh momentum up-flip; exit on the ATR(14)
# TRAILING stop S_t = max(S_{t-1}, C_t - atr_mult * ATR14_t) or the 30-day time
# fallback. atr_mult grid spans the paper's plateau (2.0-3.5); the momentum
# lookback spans 7/14/28 days at 6h (56 bars = 14d is the literature's short
# crypto-momentum horizon). Vol scaling uses the 24h realized vol on 6h bars
# (4 bars) against a 1-week baseline (28 bars), same semantics as the hourly
# sections.
_ATREND_BAR_MS = 6 * _HOUR_MS
_ATREND_ATR_WINDOW = 14
_ATREND_TIME_FALLBACK_BARS = 120  # 30 days at 6h
_ATREND_MULTIPLIERS = (2.0, 2.5, 3.0)
_ATREND_LOOKBACKS_H = (168, 336, 672)  # 7 / 14 / 28 days at 6h
_ATREND_VOL_WINDOW_BARS = 4  # 24h realized vol on 6h bars
_ATREND_VOL_BASELINE_BARS = 28  # 1-week baseline on 6h bars

# vreg (Vol-Regime-Adaptive Trend): the anti-overfit twin of AdaptiveTrend. On
# 6h bars, the 7-day realized-vol percentile vs the PRIOR 3 months PINS the
# (momentum lookback, ATR trailing multiplier) pair. The mapping below is
# pre-registered and NEVER re-fit -- the vol regime IS the adaptation.
_VREG_BAR_MS = 6 * _HOUR_MS
_VREG_ATR_WINDOW = 14
_VREG_RV_WINDOW_BARS = 28  # 7-day realized vol on 6h bars
_VREG_REGIME_WINDOW_BARS = 360  # 3-month regime percentile history (~90d)
_VREG_PCT_HIGH = 0.70
_VREG_PCT_LOW = 0.30
_VREG_TIME_FALLBACK_BARS = 120  # 30 days at 6h
_VREG_VOLUME_MEDIAN_BARS = 28
_VREG_VOLUME_SURGE = 1.25  # entry needs volume > 1.25x the prior 28-bar median
_VREG_HARD_STOP = 0.10  # 10% hard stop (arXiv 2604.27150: tight beats wide)
_VREG_LOOKBACK_BARS = (28, 56, 112)  # 7 / 14 / 28 days at 6h
_VREG_VOL_WINDOW_BARS = 4  # 24h realized vol on 6h bars
_VREG_VOL_BASELINE_BARS = 28  # 1-week baseline on 6h bars
# PINNED vol-regime -> (momentum lookback bars, ATR trailing multiplier).
# High vol: catch fast, stop wide. Low vol: chop is persistence, hold tight.
_VREG_MAP: dict[str, tuple[int, float]] = {
    "high": (28, 3.0),
    "mid": (56, 2.5),
    "low": (112, 2.0),
}

# tsmom (sign-based TSMOM, Moskowitz-Ooi-Pedersen JFE 2012 / AQR "Trends
# Everywhere"): on 6h bars, hold LONG while the trailing L-bar return is
# positive and SHORT while negative -- the SIGN construction AQR actually runs
# (not flip-entries with stops), sized by vol targeting w = clip(baseline_rv /
# rv_24h, 0.5, 2.0). Position flips only when the sign changes (MOP: the effect
# persists ~1y then partially reverses); the reversal overlay flattens early
# when the 3xL-bar return turns against the trade. L is LEARNED walk-forward
# from {7,14,28}d; crash/regime gates stay available as entry filters.
_TSMOM_BAR_MS = 6 * _HOUR_MS
_TSMOM_LOOKBACKS_BARS = (28, 56, 112)  # 7 / 14 / 28 days at 6h
_TSMOM_REVERSAL_MULT = 3  # reversal window = 3 x lookback bars (MOP reversal)
_TSMOM_TIME_FALLBACK_BARS = 120  # 30 days at 6h (reversal overlay cap)
_TSMOM_VOL_WINDOW_BARS = 4  # 24h realized vol on 6h bars
_TSMOM_VOL_BASELINE_BARS = 28  # 1-week baseline on 6h bars

# pth (Price-to-High, George & Hwang JFE 2004): position = sign(PTH_k - 0.5),
# where PTH_k = close / trailing k-bar max(HIGH). For large liquid coins,
# Fičura (2023) finds nearness-to-high POSITIVELY predicts returns (t = 4.93,
# monotonic across quintiles over 1-4W windows) and dominates raw momentum --
# the anchor-and-adjust effect. The 52-week anchor (Jia et al., JBF 182, 2026)
# is the 4th paper-confirmed lookback: Nearness52 earns ~130 bps/week long-short
# and survives as a 4th factor over Liu-Tsyvinski-Wu. Only k / vol / gates are
# learned; the sign construction and the 0.5 band are fixed by the paper. No
# reversal overlay: George-Hwang show PTH-driven returns do NOT reverse long-run.
_PTH_BAR_MS = 6 * _HOUR_MS
_PTH_LOOKBACKS_BARS = (28, 56, 112, 1456)  # 7 / 14 / 28 d / 52 wk at 6h
_PTH_BAND = 0.5  # upper half of the trailing range = long, lower half = short

# fade (salience/MAX contrarian): the Chinese retail-behavior literature
# (change in salience -- Zhang, Ma, Yang & Fan PBFJ 2024; factor MAX / lottery
# preference -- HKBU 2024) shows sharp salient up-moves are extrapolated by
# retail -> overpriced -> negative subsequent returns. Fade the extremes: short
# after a scaled trailing return > +tau sigma, long after < -tau, flat in
# between. tau is FIXED (pre-registered), never a config axis.
_FADE_BAR_MS = 6 * _HOUR_MS
_FADE_LOOKBACKS_BARS = (4, 8, 12, 28)  # 1 / 2 / 3 / 7 days at 6h
_FADE_TAU = 1.0  # sigma of the vol-scaled trailing return that triggers a fade
_FADE_VOL_WINDOW_BARS = 4  # 24h realized vol for the z-score

# xs (cross-sectional) sections: rank a wide universe on one pre-registered
# signal at each WEEKLY UTC rebalance, long the top quintile / short the
# bottom, per-symbol liquidity guard, market-level crash/regime gates. See the
# module docstring for the four signals and their literature anchors.
_XS_REBALANCE_H = 168  # weekly rebalance on the UTC clock
_XS_QUINTILE = 0.2  # top/bottom quintile of the ranked cross-section
_XS_MIN_SYMBOLS = 8  # a rebalance needs this many liquid names or it is skipped
_XS_VOLUME_FRAC = 0.5  # name must trade >= 0.5x its OWN trailing 28-bar median
_XS_VOLUME_MEDIAN_BARS = 28
_XS_BETA_WINDOW_BARS = 672  # residual-momentum market-beta window (28d hourly)
_XS_BETA_MIN_PERIODS = 168
_XS_LOOKBACKS_H = (168, 336, 672)  # 7 / 14 / 28 days (learned walk-forward)
_XS_DIRECTION: dict[str, str] = {
    "xs_rel14": "momentum",
    "xs_resm": "momentum",
    "xs_anchor": "reversal",
    "xs_st": "reversal",
}
_XS_TPY = 365.0 * 24.0 / _XS_REBALANCE_H

# Sections = gate combinations tested head-to-head (see module docstring).
# ``regime`` adds the UP-UP market-state gate (mandatory in the bettable
# ``full`` fusion; the standalone ``regime`` section A/B's it vs ``tsm``).
SECTIONS: dict[str, dict] = {
    "tsm": {"prox_tol": (None,), "cgo": (False,), "crash": (None,), "regime": (False,)},
    "prox": {"prox_tol": PROX_TOLS, "cgo": (False,), "crash": (None,), "regime": (False,)},
    "cgo": {"prox_tol": (None,), "cgo": (True,), "crash": (None,), "regime": (False,)},
    "crash": {"prox_tol": (None,), "cgo": (False,), "crash": CRASH_LEVELS, "regime": (False,)},
    "regime": {"prox_tol": (None,), "cgo": (False,), "crash": (None,), "regime": (True,)},
    "full": {"prox_tol": PROX_TOLS, "cgo": (True,), "crash": CRASH_LEVELS, "regime": (True,)},
    # AdaptiveTrend H6: its own config space (atr_mult x lookback x vol x crash
    # x regime) is expanded in probe_symbol -- the gates keys below only define
    # the crash / regime entry-filter grid for this section.
    "atrend": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    # vreg H6: pinned (L, alpha) mapping (NEVER a config axis -- see _VREG_MAP)
    # x vol x crash x regime. The grid only learns the gate/sizing axes.
    "vreg": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    # tsmom H6: AQR sign-based TSMOM (long/short by sign of the trailing
    # return), vol-targeted, optional reversal overlay. Grid = L x vol x
    # reversal x crash x regime; L and the overlay toggle are the ONLY learned
    # axes (the sign construction itself is fixed by the paper).
    "tsmom": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    # pth H7: Price-to-High (George & Hwang JFE 2004; Fičura 2023; Jia et al.
    # JBF 182, 2026). Position = sign(PTH_k - 0.5) with PTH_k = close /
    # trailing k-bar max(HIGH); the sign + 0.5 band are FIXED, the grid learns
    # only k (incl. the 52-week anchor) x vol x crash x regime.
    "pth": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    # fade H7: salience/MAX contrarian (Zhang, Ma, Yang & Fan PBFJ 2024;
    # Cai & Zhao JBF 159, 2024; Ethereum disposition 2026). Short after a
    # > +tau sigma vol-scaled trailing return, long after < -tau, flat in
    # between. tau is FIXED (pre-registered); grid = lookback x vol x crash x
    # regime.
    "fade": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    # xs_* cross-sectional sections: universe-level, so probe_symbol skips them
    # and probe_xs runs them across ALL symbols at once (see module docstring).
    "xs_rel14": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    "xs_resm": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    "xs_anchor": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
    "xs_st": {"crash": (None,) + CRASH_LEVELS, "regime": (False, True)},
}

XS_SECTIONS = ("xs_rel14", "xs_resm", "xs_anchor", "xs_st")

_WF_SPLIT = 0.6  # first 60% of events (by ts) learn, last 40% are OOS
_MIN_IS_TRADES = 10  # a config must have this many in-sample trades to be chosen
_MIN_OOS_TRADES = 15  # pooled-book OOS trades needed for a verdict
_MIN_OOS_TRADES_SYMBOL = 8  # per-symbol OOS trades for an informational verdict


def _contiguous_runs(hourly: pd.DataFrame, *, spacing_ms: int = _HOUR_MS) -> list[pd.DataFrame]:
    """Split one symbol's bars into contiguous runs (``spacing_ms`` spacing)."""
    starts = hourly["window_start_ms"].to_numpy()
    gaps = np.where(np.diff(starts) != spacing_ms)[0] + 1
    bounds = [0] + list(gaps) + [len(hourly)]
    return [
        hourly.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True)
        for i in range(len(bounds) - 1)
    ]


def _resample_ohlcv(hourly: pd.DataFrame, bucket_ms: int) -> pd.DataFrame:
    """Tumble a single symbol's OHLCV into ``bucket_ms`` buckets (open=first,
    high=max, low=min, close=last, volume=sum). In-progress trailing buckets
    are dropped by the contiguous-run split downstream."""
    df = hourly.copy()
    df["bucket"] = df["window_start_ms"] // bucket_ms
    agg = (
        df.groupby("bucket", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values("bucket")
        .reset_index(drop=True)
    )
    agg["window_start_ms"] = agg["bucket"] * bucket_ms
    return agg


def _up_up_mask(closes: np.ndarray, *, window_bars: int, step_bars: int) -> np.ndarray:
    """Weekly-sampled UP-UP state mask (Cooper et al. 2004 via Hsieh et al.
    2025): state[t] = ``window_bars``-bar close return >= 0, known at close of
    t-1; the gate requires the state at t AND at t - ``step_bars`` (the
    adjacent preceding period) to both be UP. Insufficient history fails
    closed (no entry)."""
    n = len(closes)
    up = np.zeros(n, dtype=bool)
    if n > window_bars + 1:
        ret = np.full(n, np.nan)
        ret[window_bars + 1 :] = closes[window_bars:-1] / closes[: -window_bars - 1] - 1.0
        up[window_bars + 1 :] = ret[window_bars + 1 :] >= 0.0
    up_up = up & np.roll(up, step_bars)
    up_up[: window_bars + 1] = False
    return up_up


def _trade_events(
    hourly: pd.DataFrame,
    *,
    lookback_h: int,
    hold_h: int,
    stop_pct: float,
    vol_scale: bool,
    prox_tol: float | None,
    cgo: bool,
    crash: float | None,
    regime: bool = False,
) -> list[dict]:
    """Non-overlapping scheduled rebalancing, strictly lagged, per run.

    Signal bar ``t`` (close known) fills at the OPEN of ``t+1``. Trades are
    scheduled every ``hold_h`` hours on the UTC clock so they never overlap.
    LONG only (long-biased; the crypto literature shows shorting erodes
    returns and creates momentum crashes). Entry requires trend up AND each
    active gate at ``t``. ``regime`` adds the UP-UP market-state gate (Hsieh,
    Huang & Liu 2025): only enter when the current and the immediately
    preceding weekly 4-week-return states are both UP. The trailing stop exits
    at the open after the first close ``stop_pct`` below the running peak
    close; otherwise the max-hold exit is at open[t+hold]. Net = w * (gross -
    round_trip), charged on the traded notional. No lookahead: every input is
    known at the signal close.
    """
    events: list[dict] = []
    for run in _contiguous_runs(hourly):
        opens = run["open"].to_numpy()
        closes = run["close"].to_numpy()
        ts = run["window_start_ms"].to_numpy()
        n = len(closes)
        if n < lookback_h + 2:
            continue
        log_ret = np.log(closes[1:] / closes[:-1])
        ret_lb = np.empty(n)
        ret_lb[:lookback_h] = np.nan
        ret_lb[lookback_h:] = closes[lookback_h:] / closes[:-lookback_h] - 1.0

        if vol_scale:
            lr = pd.Series(log_ret)
            rv24 = lr.shift(1).rolling(24).std().to_numpy()  # known at close of t
            base = pd.Series(rv24).rolling(168).mean().to_numpy()
            ratio = np.divide(
                base, rv24, out=np.ones_like(base), where=np.isfinite(rv24) & (rv24 > 0)
            )
            scale = np.clip(ratio, 0.5, 2.0)
        else:
            scale = np.ones(n)

        # Proximity to the rolling lookback high (Fičura continuation signal).
        if prox_tol is not None:
            hi = pd.Series(closes).rolling(lookback_h).max().to_numpy()
            prox = closes / np.where(np.isfinite(hi) & (hi > 0), hi, np.nan)
        else:
            prox = np.ones(n)

        # Capital gains overhang: (P - 1w volume-weighted ref price) / P.
        if cgo:
            vol = run["volume"].to_numpy()
            wv = pd.Series(closes * np.where(np.isfinite(vol), vol, 0.0))
            v = pd.Series(np.where(np.isfinite(vol), vol, 0.0))
            ref = wv.shift(1).rolling(CGO_REF_WINDOW_H).sum() / v.shift(1).rolling(
                CGO_REF_WINDOW_H
            ).sum().replace(0.0, np.nan)
            cgo_val = (closes - ref.to_numpy()) / closes
        else:
            cgo_val = np.ones(n)

        # Crash guard: 24h close return.
        if crash is not None:
            ret24 = np.full(n, np.nan)
            ret24[24:] = closes[24:] / closes[:-24] - 1.0
        else:
            ret24 = np.zeros(n)

        # UP-UP regime gate (Hsieh, Huang & Liu 2025): state[t] = 4-week close
        # return >= 0, known at close of t-1 (strictly lagged). The state is
        # sampled weekly; the gate requires the CURRENT and the immediately
        # preceding WEEKLY state to both be UP (adjacent-period transition).
        # Insufficient history fails closed.
        if regime:
            up_up = _up_up_mask(closes, window_bars=_REGIME_WINDOW_H, step_bars=_REGIME_STEP_H)
        else:
            up_up = np.ones(n, dtype=bool)

        hour_idx = ts // _HOUR_MS
        for t in np.flatnonzero((hour_idx % hold_h == 0) & np.isfinite(ret_lb) & (ret_lb != 0.0)):
            if t + 1 >= n:
                continue
            if ret_lb[t] <= 0.0:
                continue  # long-biased: no shorts
            if prox_tol is not None and not (np.isfinite(prox[t]) and prox[t] >= 1.0 - prox_tol):
                continue
            if cgo and not (np.isfinite(cgo_val[t]) and cgo_val[t] > 0.0):
                continue
            if crash is not None and np.isfinite(ret24[t]) and ret24[t] < -crash:
                continue
            if regime and not up_up[t]:
                continue
            entry = t + 1
            entry_price = opens[entry]
            w = float(scale[t])
            peak = closes[entry]
            exit_bar = min(entry + hold_h, n - 1)
            exit_price = opens[exit_bar]
            for j in range(entry + 1, exit_bar + 1):
                if closes[j] > peak:
                    peak = closes[j]
                if stop_pct > 0.0 and closes[j] <= peak * (1.0 - stop_pct):
                    exit_bar = j
                    exit_price = opens[min(j + 1, n - 1)] if j + 1 < n else closes[j]
                    break
            gross = exit_price / entry_price - 1.0
            events.append(
                {
                    "ts": int(ts[entry]),
                    "signal_ts": int(ts[t]),
                    "hold_h": hold_h,
                    "lookback_h": lookback_h,
                    "stop_pct": stop_pct,
                    "vol_scale": vol_scale,
                    "prox_tol": prox_tol,
                    "cgo": cgo,
                    "crash": crash,
                    "regime": regime,
                    "atr_mult": None,
                    "gross": round(float(gross), 9),
                    "w": round(w, 3),
                    "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                    "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                }
            )
    return events


def _atrend_events(
    hourly: pd.DataFrame,
    *,
    lookback_h: int,
    atr_mult: float,
    vol_scale: bool,
    crash: float | None,
    regime: bool = False,
) -> list[dict]:
    """AdaptiveTrend-style H6 event generator (Bui & Nguyen, arXiv 2602.11708).

    Aggregates the hourly series to 6-hour bars, then trades momentum FLIPS:
    LONG entry when the ``lookback_h//6``-bar momentum crosses from <= 0 to > 0
    (a fresh up-leg), filled at the NEXT bar's open -- never the signal close.
    Exit by the ATR(14) TRAILING stop ``S_t = max(S_{t-1}, C_t - atr_mult *
    ATR14_t)`` checked at each close and filled at the following open, or the
    30-day time fallback (120 bars). The trailing floor is initialized at the
    entry bar's close minus ``atr_mult * ATR14``; a position never re-enters
    until momentum turns down and back up (natural flip cooldown), and
    positions never overlap. The crash guard and UP-UP regime gate remain
    available as entry filters. No lookahead: every input is known at the
    signal/decision close.
    """
    bars6 = _resample_ohlcv(hourly, _ATREND_BAR_MS)
    L = lookback_h // 6
    events: list[dict] = []
    for run in _contiguous_runs(bars6, spacing_ms=_ATREND_BAR_MS):
        opens = run["open"].to_numpy()
        highs = run["high"].to_numpy()
        lows = run["low"].to_numpy()
        closes = run["close"].to_numpy()
        ts = run["window_start_ms"].to_numpy()
        n = len(closes)
        if n < L + 2:
            continue

        mom = np.full(n, np.nan)
        mom[L:] = closes[L:] / closes[:-L] - 1.0

        tr_full = np.full(n, np.nan)
        tr_full[1:] = np.maximum.reduce(
            [
                highs[1:] - lows[1:],
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ]
        )
        atr = pd.Series(tr_full).rolling(_ATREND_ATR_WINDOW).mean().to_numpy()

        if vol_scale:
            lr_full = np.empty(n)
            lr_full[0] = np.nan
            lr_full[1:] = np.log(closes[1:] / closes[:-1])
            rv = pd.Series(lr_full).shift(1).rolling(_ATREND_VOL_WINDOW_BARS).std().to_numpy()
            base = pd.Series(rv).rolling(_ATREND_VOL_BASELINE_BARS).mean().to_numpy()
            ratio = np.divide(base, rv, out=np.ones_like(base), where=np.isfinite(rv) & (rv > 0))
            scale = np.clip(ratio, 0.5, 2.0)
        else:
            scale = np.ones(n)

        if crash is not None:
            ret24 = np.full(n, np.nan)
            ret24[4:] = closes[4:] / closes[:-4] - 1.0
        else:
            ret24 = np.zeros(n)

        if regime:
            up_up = _up_up_mask(
                closes,
                window_bars=_REGIME_WINDOW_H // 6,
                step_bars=_REGIME_STEP_H // 6,
            )
        else:
            up_up = np.ones(n, dtype=bool)

        up = np.isfinite(mom) & (mom > 0.0)
        prev_up = np.zeros(n, dtype=bool)
        prev_up[1:] = up[:-1]
        flips = np.flatnonzero(up & ~prev_up)

        busy_until = -1
        for t in flips:
            if t + 1 >= n or t + 1 <= busy_until:
                continue
            if crash is not None and np.isfinite(ret24[t]) and ret24[t] < -crash:
                continue
            if regime and not up_up[t]:
                continue
            entry = t + 1
            if not np.isfinite(atr[entry]) or atr[entry] <= 0.0:
                continue
            entry_price = opens[entry]
            w = float(scale[t])
            stop = closes[entry] - atr_mult * atr[entry]
            fallback = min(entry + _ATREND_TIME_FALLBACK_BARS, n - 1)
            exit_bar = fallback
            exit_price = opens[exit_bar]
            for j in range(entry + 1, exit_bar + 1):
                floor = closes[j] - atr_mult * atr[j]
                if floor > stop:
                    stop = floor
                if closes[j] <= stop:
                    exit_bar = j
                    exit_price = opens[j + 1] if j + 1 < n else closes[j]
                    break
            gross = exit_price / entry_price - 1.0
            events.append(
                {
                    "ts": int(ts[entry]),
                    "signal_ts": int(ts[t]),
                    "hold_h": _ATREND_TIME_FALLBACK_BARS * 6,
                    "actual_hold_h": int((exit_bar - entry) * 6),
                    "lookback_h": lookback_h,
                    "stop_pct": 0.0,
                    "vol_scale": vol_scale,
                    "prox_tol": None,
                    "cgo": False,
                    "crash": crash,
                    "regime": regime,
                    "atr_mult": atr_mult,
                    "gross": round(float(gross), 9),
                    "w": round(w, 3),
                    "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                    "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                }
            )
            busy_until = exit_bar
    return events


def _vreg_params(pct: float) -> tuple[int, float]:
    """PINNED vol-regime -> (momentum lookback bars, ATR trailing multiplier).

    The mapping is pre-registered and never re-fit (the anti-overfit thesis of
    vreg vs AdaptiveTrend's monthly grid search). High realized vol wants a
    fast lookback with a wide stop; low vol means chop is persistence, so hold
    on with a tight stop.
    """
    if pct >= _VREG_PCT_HIGH:
        return _VREG_MAP["high"]
    if pct <= _VREG_PCT_LOW:
        return _VREG_MAP["low"]
    return _VREG_MAP["mid"]


def _vreg_events(
    hourly: pd.DataFrame,
    *,
    vol_scale: bool,
    crash: float | None,
    regime: bool = False,
) -> list[dict]:
    """Vol-Regime-Adaptive Trend event generator (H6 bars).

    At each bar close, the 7-day realized-vol percentile vs the PRIOR 3 months
    pins (L, alpha) from the pre-registered ``_VREG_MAP`` -- never re-fit.
    LONG entry requires a fresh momentum up-flip (MOM crosses <=0 -> >0 over
    the regime-pinned L bars), a volume surge (bar volume > 1.25x the prior
    28-bar median -- the CTREND price+volume fusion), and the optional crash
    guard / UP-UP regime gate; fill at the NEXT bar's open. Exit on the ATR(14)
    TRAILING stop ``S_t = max(S_{t-1}, C_t - alpha*ATR14_t)``, the 10% hard
    stop, or the 30-day time fallback, each filled at the following open.
    Risk-managed sizing ``clip(baseline_rv / rv_24h, 0.5, 2.0)``. No lookahead:
    every input is known at the signal/decision close.
    """
    bars6 = _resample_ohlcv(hourly, _VREG_BAR_MS)
    events: list[dict] = []
    for run in _contiguous_runs(bars6, spacing_ms=_VREG_BAR_MS):
        opens = run["open"].to_numpy()
        highs = run["high"].to_numpy()
        lows = run["low"].to_numpy()
        closes = run["close"].to_numpy()
        vols = run["volume"].to_numpy()
        ts = run["window_start_ms"].to_numpy()
        n = len(closes)
        if n < _VREG_REGIME_WINDOW_BARS + _VREG_RV_WINDOW_BARS + 2:
            continue

        lr = np.full(n, np.nan)
        lr[1:] = np.log(closes[1:] / closes[:-1])
        rv = pd.Series(lr).rolling(_VREG_RV_WINDOW_BARS).std().to_numpy()

        # Percentile of rv[t] vs the prior 3 months (strictly lagged window).
        pct = np.full(n, np.nan)
        for t in range(_VREG_REGIME_WINDOW_BARS, n):
            if not np.isfinite(rv[t]):
                continue
            hist = rv[t - _VREG_REGIME_WINDOW_BARS : t]
            fin = hist[np.isfinite(hist)]
            if fin.size < 2:
                continue
            pct[t] = float(np.mean(fin < rv[t]))

        moms: dict[int, np.ndarray] = {}
        for L in _VREG_LOOKBACK_BARS:
            m = np.full(n, np.nan)
            m[L:] = closes[L:] / closes[:-L] - 1.0
            moms[L] = m

        tr_full = np.full(n, np.nan)
        tr_full[1:] = np.maximum.reduce(
            [
                highs[1:] - lows[1:],
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ]
        )
        atr = pd.Series(tr_full).rolling(_VREG_ATR_WINDOW).mean().to_numpy()

        if vol_scale:
            rv24 = pd.Series(lr).shift(1).rolling(_VREG_VOL_WINDOW_BARS).std().to_numpy()
            base = pd.Series(rv24).rolling(_VREG_VOL_BASELINE_BARS).mean().to_numpy()
            ratio = np.divide(
                base, rv24, out=np.ones_like(base), where=np.isfinite(rv24) & (rv24 > 0)
            )
            scale = np.clip(ratio, 0.5, 2.0)
        else:
            scale = np.ones(n)

        med_vol = pd.Series(vols).shift(1).rolling(_VREG_VOLUME_MEDIAN_BARS).median().to_numpy()
        surge = vols > _VREG_VOLUME_SURGE * med_vol

        if crash is not None:
            ret24 = np.full(n, np.nan)
            ret24[4:] = closes[4:] / closes[:-4] - 1.0
        else:
            ret24 = np.zeros(n)

        if regime:
            up_up = _up_up_mask(
                closes,
                window_bars=_REGIME_WINDOW_H // 6,
                step_bars=_REGIME_STEP_H // 6,
            )
        else:
            up_up = np.ones(n, dtype=bool)

        busy_until = -1
        for t in range(_VREG_REGIME_WINDOW_BARS, n):
            if t + 1 >= n or t + 1 <= busy_until:
                continue
            if not np.isfinite(pct[t]):
                continue
            L, alpha = _vreg_params(pct[t])
            m = moms[L]
            if not (np.isfinite(m[t]) and m[t] > 0.0):
                continue
            if np.isfinite(m[t - 1]) and m[t - 1] > 0.0:
                continue  # not a fresh up-flip
            if not (np.isfinite(surge[t]) and surge[t]):
                continue
            if crash is not None and np.isfinite(ret24[t]) and ret24[t] < -crash:
                continue
            if regime and not up_up[t]:
                continue
            entry = t + 1
            if not np.isfinite(atr[entry]) or atr[entry] <= 0.0:
                continue
            entry_price = opens[entry]
            w = float(scale[t])
            stop = closes[entry] - alpha * atr[entry]
            fallback = min(entry + _VREG_TIME_FALLBACK_BARS, n - 1)
            exit_bar = fallback
            exit_price = opens[exit_bar]
            for j in range(entry + 1, fallback + 1):
                floor = closes[j] - alpha * atr[j]
                if floor > stop:
                    stop = floor
                if closes[j] <= stop or closes[j] <= entry_price * (1.0 - _VREG_HARD_STOP):
                    exit_bar = j
                    exit_price = opens[j + 1] if j + 1 < n else closes[j]
                    break
            gross = exit_price / entry_price - 1.0
            events.append(
                {
                    "ts": int(ts[entry]),
                    "signal_ts": int(ts[t]),
                    "vol_pct": round(float(pct[t]), 3),
                    "hold_h": _VREG_TIME_FALLBACK_BARS * 6,
                    "actual_hold_h": int((exit_bar - entry) * 6),
                    "lookback_h": L * 6,
                    "stop_pct": _VREG_HARD_STOP,
                    "vol_scale": vol_scale,
                    "prox_tol": None,
                    "cgo": False,
                    "crash": crash,
                    "regime": regime,
                    "atr_mult": alpha,
                    "vreg": True,
                    "gross": round(float(gross), 9),
                    "w": round(w, 3),
                    "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                    "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                }
            )
            busy_until = exit_bar
    return events


def _tsmom_events(
    hourly: pd.DataFrame,
    *,
    lookback_bars: int,
    vol_scale: bool,
    reversal: bool,
    crash: float | None,
    regime: bool = False,
) -> list[dict]:
    """AQR sign-based TSMOM event generator (H6 bars).

    Position is the SIGN of the trailing L-bar return -- long while positive,
    short while negative (Moskowitz, Ooi & Pedersen, JFE 2012; AQR "Trends
    Everywhere" JOIM) -- sized by vol targeting ``w = clip(baseline_rv /
    rv_24h, 0.5, 2.0)``. A trade opens when the sign FLIPS (fill at next open)
    and closes on the next flip (MOP: the effect persists ~1y then partially
    reverses). With ``reversal=True`` the position flattens EARLY when the
    3xL-bar return has turned against the trade, or after the 30-day fallback.
    Optional crash / UP-UP regime gates filter entries. No lookahead: every
    input is known at the signal close. Long trades score +ret, shorts -ret,
    both charged the maker round-trip on the traded notional.
    """
    bars6 = _resample_ohlcv(hourly, _TSMOM_BAR_MS)
    events: list[dict] = []
    for run in _contiguous_runs(bars6, spacing_ms=_TSMOM_BAR_MS):
        opens = run["open"].to_numpy()
        closes = run["close"].to_numpy()
        ts = run["window_start_ms"].to_numpy()
        n = len(closes)
        if n < lookback_bars + 2:
            continue

        mom = np.full(n, np.nan)
        mom[lookback_bars:] = closes[lookback_bars:] / closes[:-lookback_bars] - 1.0

        rev_bars = _TSMOM_REVERSAL_MULT * lookback_bars
        rev = np.full(n, np.nan)
        if rev_bars < n:
            rev[rev_bars:] = closes[rev_bars:] / closes[:-rev_bars] - 1.0

        if vol_scale:
            lr = pd.Series(np.full(n, np.nan))
            lr[1:] = np.log(closes[1:] / closes[:-1])
            rv24 = lr.shift(1).rolling(_TSMOM_VOL_WINDOW_BARS).std().to_numpy()
            base = pd.Series(rv24).rolling(_TSMOM_VOL_BASELINE_BARS).mean().to_numpy()
            ratio = np.divide(
                base, rv24, out=np.ones_like(base), where=np.isfinite(rv24) & (rv24 > 0)
            )
            scale = np.clip(ratio, 0.5, 2.0)
        else:
            scale = np.ones(n)

        if crash is not None:
            ret24 = np.full(n, np.nan)
            ret24[4:] = closes[4:] / closes[:-4] - 1.0
        else:
            ret24 = np.zeros(n)

        if regime:
            up_up = _up_up_mask(
                closes,
                window_bars=_REGIME_WINDOW_H // 6,
                step_bars=_REGIME_STEP_H // 6,
            )
        else:
            up_up = np.ones(n, dtype=bool)

        sign = np.zeros(n)
        fin = np.isfinite(mom)
        sign[fin & (mom > 0.0)] = 1.0
        sign[fin & (mom < 0.0)] = -1.0

        i = lookback_bars
        while i < n - 1:
            side = sign[i]
            if side == 0.0:
                i += 1
                continue
            if crash is not None and np.isfinite(ret24[i]) and ret24[i] < -crash:
                i += 1
                continue
            if regime and not up_up[i]:
                i += 1
                continue
            entry = i + 1
            entry_price = opens[entry]
            w = float(scale[i])
            fallback = min(entry + _TSMOM_TIME_FALLBACK_BARS, n - 1) if reversal else n - 1
            exit_bar = fallback
            exit_price = opens[exit_bar]
            j = entry + 1
            while j <= fallback:
                if sign[j] != side or (
                    reversal and np.isfinite(rev[j]) and np.sign(rev[j]) == -side
                ):
                    exit_bar = j
                    exit_price = opens[j + 1] if j + 1 < n else closes[j]
                    break
                j += 1
            gross = side * (exit_price / entry_price - 1.0)
            events.append(
                {
                    "ts": int(ts[entry]),
                    "signal_ts": int(ts[i]),
                    "hold_h": _TSMOM_TIME_FALLBACK_BARS * 6
                    if reversal
                    else int((fallback - entry) * 6),
                    "actual_hold_h": int((exit_bar - entry) * 6),
                    "lookback_h": lookback_bars * 6,
                    "stop_pct": 0.0,
                    "vol_scale": vol_scale,
                    "prox_tol": None,
                    "cgo": False,
                    "crash": crash,
                    "regime": regime,
                    "atr_mult": None,
                    "reversal": reversal,
                    "tsmom": True,
                    "gross": round(float(gross), 9),
                    "w": round(w, 3),
                    "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                    "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                }
            )
            i = exit_bar + 1
            if exit_bar == fallback or sign[exit_bar] == side:
                while i < n - 1 and sign[i] == side:
                    i += 1  # stand aside until the regime actually flips
    return events


def _pth_events(
    hourly: pd.DataFrame,
    *,
    lookback_bars: int,
    vol_scale: bool,
    crash: float | None,
    regime: bool = False,
) -> list[dict]:
    """George-Hwang / Fičura Price-to-High event generator (H6 bars).

    PTH_k = close / trailing k-bar max(HIGH) -- Fičura (2023) defines the
    nearness-to-high anchor with the INTRADAY high, not max(close). Position =
    sign(PTH_k - 0.5): long while the close sits in the upper half of the
    trailing range, short in the lower half (the 0.5 band is fixed by the
    paper). A trade opens when the band is crossed (fill at next open) and
    closes when it is crossed back; George-Hwang show PTH-driven returns do
    NOT reverse long-run, so there is no reversal overlay and no early
    flatten. The 52-week anchor (Jia et al., JBF 182, 2026) earns ~130 bps/wk
    long-short and is a paper-confirmed 4th factor. Optional crash / UP-UP
    regime gates filter entries; vol-targeted sizing w = clip(baseline_rv /
    rv_24h, 0.5, 2.0). No lookahead: every input is known at the signal close.
    """
    bars6 = _resample_ohlcv(hourly, _PTH_BAR_MS)
    events: list[dict] = []
    for run in _contiguous_runs(bars6, spacing_ms=_PTH_BAR_MS):
        opens = run["open"].to_numpy()
        highs = run["high"].to_numpy()
        closes = run["close"].to_numpy()
        ts = run["window_start_ms"].to_numpy()
        n = len(closes)
        if n < lookback_bars + 2:
            continue

        trailing = pd.Series(highs).rolling(lookback_bars, min_periods=lookback_bars).max()
        pth = np.full(n, np.nan)
        pth[lookback_bars - 1 :] = (
            closes[lookback_bars - 1 :] / trailing.to_numpy()[lookback_bars - 1 :]
        )

        if vol_scale:
            lr = pd.Series(np.full(n, np.nan))
            lr[1:] = np.log(closes[1:] / closes[:-1])
            rv24 = lr.shift(1).rolling(_TSMOM_VOL_WINDOW_BARS).std().to_numpy()
            base = pd.Series(rv24).rolling(_TSMOM_VOL_BASELINE_BARS).mean().to_numpy()
            ratio = np.divide(
                base, rv24, out=np.ones_like(base), where=np.isfinite(rv24) & (rv24 > 0)
            )
            scale = np.clip(ratio, 0.5, 2.0)
        else:
            scale = np.ones(n)

        if crash is not None:
            ret24 = np.full(n, np.nan)
            ret24[4:] = closes[4:] / closes[:-4] - 1.0
        else:
            ret24 = np.zeros(n)

        if regime:
            up_up = _up_up_mask(
                closes,
                window_bars=_REGIME_WINDOW_H // 6,
                step_bars=_REGIME_STEP_H // 6,
            )
        else:
            up_up = np.ones(n, dtype=bool)

        fin = np.isfinite(pth)
        sign = np.zeros(n)
        sign[fin & (pth > _PTH_BAND)] = 1.0
        sign[fin & (pth < _PTH_BAND)] = -1.0

        i = lookback_bars - 1
        while i < n - 1:
            side = sign[i]
            if side == 0.0:
                i += 1
                continue
            if crash is not None and np.isfinite(ret24[i]) and ret24[i] < -crash:
                i += 1
                continue
            if regime and not up_up[i]:
                i += 1
                continue
            entry = i + 1
            entry_price = opens[entry]
            w = float(scale[i])
            fallback = n - 1
            exit_bar = fallback
            exit_price = opens[exit_bar]
            j = entry + 1
            while j <= fallback:
                if sign[j] != side:
                    exit_bar = j
                    exit_price = opens[j + 1] if j + 1 < n else closes[j]
                    break
                j += 1
            gross = side * (exit_price / entry_price - 1.0)
            events.append(
                {
                    "ts": int(ts[entry]),
                    "signal_ts": int(ts[i]),
                    "hold_h": int((fallback - entry) * 6),
                    "actual_hold_h": int((exit_bar - entry) * 6),
                    "lookback_h": lookback_bars * 6,
                    "stop_pct": 0.0,
                    "vol_scale": vol_scale,
                    "prox_tol": None,
                    "cgo": False,
                    "crash": crash,
                    "regime": regime,
                    "atr_mult": None,
                    "pth": True,
                    "gross": round(float(gross), 9),
                    "w": round(w, 3),
                    "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                    "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                }
            )
            i = exit_bar + 1
    return events


def _fade_events(
    hourly: pd.DataFrame,
    *,
    lookback_bars: int,
    vol_scale: bool,
    crash: float | None,
    regime: bool = False,
) -> list[dict]:
    """Salience/MAX contrarian event generator (H6 bars).

    Retail extrapolates sharp salient up-moves (lottery preference, change-in-
    salience) -> overpriced -> negative subsequent returns; the mirror holds
    for salient crashes. z = trailing L-bar return / (24h realized vol * sqrt(L))
    -- the L-bar return normalized to its own horizon sigma. Fade the extremes:
    short when z > +tau, long when z < -tau, FLAT in between (tau = 1.0 is
    FIXED, pre-registered, never a config axis). A trade opens when z crosses
    out of +-tau (fill at next open) and closes when it crosses back inside.
    The Ethereum disposition-effect evidence (2026) is STRONGER in high-vol
    regimes -- the vol-scale axis stays available. Optional crash / UP-UP
    regime gates filter entries; vol-targeted sizing w = clip(baseline_rv /
    rv_24h, 0.5, 2.0). No lookahead: every input is known at the signal close.
    """
    bars6 = _resample_ohlcv(hourly, _FADE_BAR_MS)
    events: list[dict] = []
    for run in _contiguous_runs(bars6, spacing_ms=_FADE_BAR_MS):
        opens = run["open"].to_numpy()
        closes = run["close"].to_numpy()
        ts = run["window_start_ms"].to_numpy()
        n = len(closes)
        if n < lookback_bars + 2:
            continue

        lr = pd.Series(np.full(n, np.nan))
        lr[1:] = np.log(closes[1:] / closes[:-1])
        rv = lr.shift(1).rolling(_FADE_VOL_WINDOW_BARS).std().to_numpy()
        mom = np.full(n, np.nan)
        mom[lookback_bars:] = closes[lookback_bars:] / closes[:-lookback_bars] - 1.0
        z = np.full(n, np.nan)
        valid = np.isfinite(mom) & np.isfinite(rv) & (rv > 0)
        z[valid] = mom[valid] / (rv[valid] * math.sqrt(lookback_bars))
        z[~np.isfinite(z)] = 0.0  # flat market (zero vol) -> no salient move

        if vol_scale:
            lr2 = pd.Series(np.full(n, np.nan))
            lr2[1:] = np.log(closes[1:] / closes[:-1])
            rv24 = lr2.shift(1).rolling(_FADE_VOL_WINDOW_BARS).std().to_numpy()
            base = pd.Series(rv24).rolling(_FADE_LOOKBACKS_BARS[2]).mean().to_numpy()
            ratio = np.divide(
                base, rv24, out=np.ones_like(base), where=np.isfinite(rv24) & (rv24 > 0)
            )
            scale = np.clip(ratio, 0.5, 2.0)
        else:
            scale = np.ones(n)

        if crash is not None:
            ret24 = np.full(n, np.nan)
            ret24[4:] = closes[4:] / closes[:-4] - 1.0
        else:
            ret24 = np.zeros(n)

        if regime:
            up_up = _up_up_mask(
                closes,
                window_bars=_REGIME_WINDOW_H // 6,
                step_bars=_REGIME_STEP_H // 6,
            )
        else:
            up_up = np.ones(n, dtype=bool)

        sign = np.zeros(n)
        sign[z > _FADE_TAU] = -1.0  # salient spike -> fade (short)
        sign[z < -_FADE_TAU] = 1.0  # salient crash -> fade (long)

        i = lookback_bars
        while i < n - 1:
            side = sign[i]
            if side == 0.0:
                i += 1
                continue
            if crash is not None and np.isfinite(ret24[i]) and ret24[i] < -crash:
                i += 1
                continue
            if regime and not up_up[i]:
                i += 1
                continue
            entry = i + 1
            entry_price = opens[entry]
            w = float(scale[i])
            fallback = n - 1
            exit_bar = fallback
            exit_price = opens[exit_bar]
            j = entry + 1
            while j <= fallback:
                if sign[j] != side:
                    exit_bar = j
                    exit_price = opens[j + 1] if j + 1 < n else closes[j]
                    break
                j += 1
            gross = side * (exit_price / entry_price - 1.0)
            events.append(
                {
                    "ts": int(ts[entry]),
                    "signal_ts": int(ts[i]),
                    "hold_h": int((fallback - entry) * 6),
                    "actual_hold_h": int((exit_bar - entry) * 6),
                    "lookback_h": lookback_bars * 6,
                    "stop_pct": 0.0,
                    "vol_scale": vol_scale,
                    "prox_tol": None,
                    "cgo": False,
                    "crash": crash,
                    "regime": regime,
                    "atr_mult": None,
                    "fade": True,
                    "gross": round(float(gross), 9),
                    "w": round(w, 3),
                    "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                    "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                }
            )
            i = exit_bar + 1
    return events


def _market_symbol(symbols: list[str]) -> str:
    """BTC is the market proxy (the residual-momentum regressor, the crash and
    UP-UP gates); fall back to the first symbol if the universe lacks BTC."""
    return "BTCUSDT" if "BTCUSDT" in symbols else sorted(symbols)[0]


def _xs_precompute(
    hourly_by_symbol: dict[str, pd.DataFrame], *, signal: str, lookback_h: int
) -> dict:
    """Align the universe onto one absolute-hour grid and precompute, per
    symbol, the signal, vol-scale and volume-median arrays (NaN where a symbol
    has no bar). Signal lookbacks use min_periods=lookback so a gap or a late
    listing simply withholds that symbol until it has contiguous history --
    the cross-sectional analog of the single-asset contiguous-run discipline.
    The precomputed arrays are config-independent, so probe_xs computes them
    ONCE per (signal, lookback) instead of per config."""
    symbols = sorted(s for s, df in hourly_by_symbol.items() if df is not None and not df.empty)
    if not symbols:
        return {}
    market = _market_symbol(symbols)
    dfs: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = hourly_by_symbol[s].copy()
        df["hour"] = df["window_start_ms"] // _HOUR_MS
        dfs[s] = df.set_index("hour").sort_index()
    g0 = min(d.index.min() for d in dfs.values())
    g1 = max(d.index.max() for d in dfs.values())
    hours = np.arange(g0, g1 + 1)
    n = len(hours)

    def _aligned(d: pd.DataFrame, col: str) -> np.ndarray:
        arr = np.full(n, np.nan)
        arr[d.index.to_numpy() - g0] = d[col].to_numpy()
        return arr

    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}
    highs: dict[str, np.ndarray] = {}
    volumes: dict[str, np.ndarray] = {}
    for s, d in dfs.items():
        opens[s] = _aligned(d, "open")
        closes[s] = _aligned(d, "close")
        highs[s] = _aligned(d, "high")
        volumes[s] = _aligned(d, "volume")

    mkt_close = pd.Series(closes[market])
    mkt_lr = np.log(mkt_close / mkt_close.shift(1))

    sigs: dict[str, np.ndarray] = {}
    vscales: dict[str, np.ndarray] = {}
    vmeds: dict[str, np.ndarray] = {}
    for s in symbols:
        c = pd.Series(closes[s])
        lr = np.log(c / c.shift(1))
        if signal == "xs_rel14":
            sig = c / c.shift(lookback_h) - 1.0
        elif signal == "xs_anchor":
            hi = pd.Series(highs[s]).rolling(lookback_h, min_periods=lookback_h).max()
            sig = c / hi
        elif signal == "xs_resm":
            cov = lr.rolling(_XS_BETA_WINDOW_BARS, min_periods=_XS_BETA_MIN_PERIODS).cov(mkt_lr)
            var = mkt_lr.rolling(_XS_BETA_WINDOW_BARS, min_periods=_XS_BETA_MIN_PERIODS).var()
            beta = cov / var.replace(0.0, np.nan)
            resid = lr - beta.shift(1) * mkt_lr
            sig = resid.rolling(lookback_h, min_periods=lookback_h).sum()
        elif signal == "xs_st":
            mu = lr.rolling(lookback_h, min_periods=lookback_h).mean()
            den = lr.abs() + mu.abs()
            wgt = (lr - mu).abs() / den.replace(0.0, np.nan)
            num = (wgt * lr).rolling(lookback_h, min_periods=lookback_h).sum()
            den2 = wgt.rolling(lookback_h, min_periods=lookback_h).sum()
            sig = num / den2.replace(0.0, np.nan)
        else:
            raise ValueError(f"unknown xs signal: {signal}")
        sigs[s] = sig.to_numpy()

        rv24 = lr.shift(1).rolling(24).std()
        base_rv = rv24.rolling(_XS_LOOKBACKS_H[1]).mean()
        ratio = base_rv / rv24.replace(0.0, np.nan)
        vscales[s] = np.clip(ratio.to_numpy(), 0.5, 2.0)
        vmeds[s] = (
            (pd.Series(volumes[s]).shift(1).rolling(_XS_VOLUME_MEDIAN_BARS, min_periods=5))
            .median()
            .to_numpy()
        )

    return {
        "symbols": symbols,
        "market": market,
        "hours": hours,
        "opens": opens,
        "closes": closes,
        "highs": highs,
        "volumes": volumes,
        "sig": sigs,
        "vscale": vscales,
        "vmed": vmeds,
        "n": n,
    }


def _xs_events(
    hourly_by_symbol: dict[str, pd.DataFrame],
    *,
    signal: str,
    lookback_h: int,
    vol_scale: bool,
    crash: float | None,
    regime: bool = False,
    precomp: dict | None = None,
) -> list[dict]:
    """Cross-sectional portfolio event generator (WEEKLY UTC rebalance).

    At each rebalance bar ``t`` the pre-registered signal is ranked ACROSS the
    universe; the top ``_XS_QUINTILE`` is longed and the bottom shorted (or the
    mirror for the reversal signals), each leg filled at the OPEN of ``t+1`` and
    exited at the OPEN of ``t+_XS_REBALANCE_H+1`` (the next rebalance's fill),
    so every leg holds exactly one week and never overlaps. Sizing is equal
    weight per side, or inverse-vol normalized per side. The per-symbol
    liquidity guard (own trailing median) and the market-level crash / UP-UP
    gates are applied at the signal close. No lookahead: every input is known
    at the signal close.
    """
    p = (
        precomp
        if precomp is not None
        else _xs_precompute(hourly_by_symbol, signal=signal, lookback_h=lookback_h)
    )
    symbols = p["symbols"]
    if len(symbols) < _XS_MIN_SYMBOLS:
        return []
    n = p["n"]
    market = p["market"]
    mkt_closes = pd.Series(p["closes"][market])
    if regime:
        up_up = _up_up_mask(
            mkt_closes.to_numpy(),
            window_bars=_REGIME_WINDOW_H // 1,
            step_bars=_REGIME_STEP_H // 1,
        )
    else:
        up_up = None

    ts_arr = p["hours"] * _HOUR_MS
    events: list[dict] = []
    for t in np.flatnonzero(p["hours"] % _XS_REBALANCE_H == 0):
        if t + _XS_REBALANCE_H + 1 > n:
            continue  # need the entry bar t+1 and the exit bar t+169
        if crash is not None and t >= 24:
            r24 = mkt_closes.iloc[t] / mkt_closes.iloc[t - 24] - 1.0
            if np.isfinite(r24) and r24 < -crash:
                continue
        if regime and not up_up[t]:
            continue

        cand: list[tuple[str, float]] = []
        for s in symbols:
            sig = p["sig"][s][t]
            if not np.isfinite(sig):
                continue
            vmed = p["vmed"][s][t]
            vol = p["volumes"][s][t]
            if not (np.isfinite(vmed) and np.isfinite(vol) and vol >= _XS_VOLUME_FRAC * vmed):
                continue
            entry = p["opens"][s][t + 1]
            exit_ = p["opens"][s][t + _XS_REBALANCE_H + 1]
            if not (np.isfinite(entry) and np.isfinite(exit_)):
                continue
            cand.append((s, float(sig)))
        if len(cand) < _XS_MIN_SYMBOLS:
            continue

        cand.sort(key=lambda x: x[1])
        n_long = max(2, round(len(cand) * _XS_QUINTILE))
        if _XS_DIRECTION[signal] == "momentum":
            longs = cand[-n_long:]
            shorts = cand[:n_long]
        else:
            longs = cand[:n_long]
            shorts = cand[-n_long:]

        for side, group in ((1.0, longs), (-1.0, shorts)):
            if vol_scale:
                raw = {s: float(p["vscale"][s][t]) for s, _ in group}
                tot = sum(raw.values())
                ws = {s: raw[s] / tot if tot > 0 else 1.0 / len(group) for s, _ in group}
            else:
                ws = {s: 1.0 / len(group) for s, _ in group}
            for s, _ in group:
                w = ws[s]
                entry = p["opens"][s][t + 1]
                exit_ = p["opens"][s][t + _XS_REBALANCE_H + 1]
                gross = side * (exit_ / entry - 1.0)
                events.append(
                    {
                        "ts": int(ts_arr[t + 1]),
                        "signal_ts": int(ts_arr[t]),
                        "hold_h": _XS_REBALANCE_H,
                        "actual_hold_h": _XS_REBALANCE_H,
                        "lookback_h": lookback_h,
                        "stop_pct": 0.0,
                        "vol_scale": vol_scale,
                        "prox_tol": None,
                        "cgo": False,
                        "crash": crash,
                        "regime": regime,
                        "atr_mult": None,
                        "xs_signal": signal,
                        "symbol": s,
                        "side": side,
                        "gross": round(float(gross), 9),
                        "w": round(w, 4),
                        "net_maker": round(float(gross) * w - _MAKER_RT * w, 9),
                        "net_taker": round(float(gross) * w - _TAKER_RT * w, 9),
                    }
                )
    return events


def _config_key(params: dict, section: str = "") -> tuple:
    if section in XS_SECTIONS:
        # xs configs = (lookback, vol-scale, crash, regime) only; the signal
        # construction, quintile split and portfolio rules are fixed by the
        # papers and are never config axes.
        return (
            params["lookback_h"],
            params["vol_scale"],
            params["crash"],
            params["regime"],
        )
    if section == "vreg":
        # (L, alpha) are PINNED by the vol regime -- never config axes. Two
        # vreg trades of the same gate/sizing config must bucket together no
        # matter which regime each was in.
        return (params["vol_scale"], params["crash"], params["regime"])
    if section == "tsmom":
        # Learned axes: L (lookback), vol-targeting, the reversal overlay, and
        # the crash/regime entry filters. The sign construction is fixed.
        return (
            params["lookback_h"],
            params["vol_scale"],
            params["reversal"],
            params["crash"],
            params["regime"],
        )
    if section in ("pth", "fade"):
        # Learned axes: L (lookback), vol-targeting, and the crash/regime entry
        # filters. The signal construction (PTH-0.5 band / +-tau fade) is fixed.
        return (
            params["lookback_h"],
            params["vol_scale"],
            params["crash"],
            params["regime"],
        )
    return (
        params["lookback_h"],
        params["hold_h"],
        params["stop_pct"],
        params["vol_scale"],
        params["prox_tol"],
        params["cgo"],
        params["crash"],
        params["regime"],
        params.get("atr_mult"),
    )


def _stats(pnls: np.ndarray, *, trades_per_year: float) -> dict:
    if pnls.size == 0:
        return {"n": 0}
    m = float(np.mean(pnls))
    sd = float(np.std(pnls, ddof=1)) if pnls.size > 1 else 0.0
    # Grobys & Shahzad (Int. J. Fin. Econ. 31(2), 2026): crypto momentum's
    # variance is statistically UNDEFINED (power-law alpha<3), so mean/Sharpe
    # are unreliable -- always report the median + a winsorized mean too.
    lo, hi = np.percentile(pnls, [10, 90])
    win = float(np.mean(np.clip(pnls, lo, hi)))
    return {
        "n": int(pnls.size),
        "win_rate": round(float(np.mean(pnls > 0)), 3),
        "mean_net_bps": round(1e4 * m, 2),
        "median_net_bps": round(1e4 * float(np.median(pnls)), 2),
        "winsorized_mean_net_bps": round(1e4 * win, 2),
        "p10_net_bps": round(1e4 * float(np.percentile(pnls, 10)), 2),
        "p90_net_bps": round(1e4 * float(np.percentile(pnls, 90)), 2),
        "se_bps": round(1e4 * (sd / math.sqrt(pnls.size)) if pnls.size > 1 else 0.0, 2),
        "t_stat": round(m / (sd / math.sqrt(pnls.size)), 2) if pnls.size > 1 and sd > 0 else 0.0,
        "sharpe_ann": round(m / sd * math.sqrt(trades_per_year), 2) if sd > 0 else 0.0,
        "net_multiple": round(float(np.prod(1.0 + pnls)), 3),
    }


def probe_symbol(hourly: pd.DataFrame, symbol: str, *, wf_split: float) -> dict:
    """Run every section, walk-forward select + OOS-score each, disclose all."""
    hour_idx = hourly["window_start_ms"] // _HOUR_MS
    span_days = (
        int((hour_idx.max() - hour_idx.min() + 1) * _HOUR_MS // _DAY_MS) if len(hourly) else 0
    )

    all_events: list[dict] = []
    for section, gates in SECTIONS.items():
        if section in XS_SECTIONS:
            # xs sections are universe-level: probe_xs runs them across ALL
            # symbols at once (config selection must be global, not per-symbol).
            continue
        if section == "atrend":
            # AdaptiveTrend H6 grid: atr_mult x momentum lookback x vol x
            # crash x regime (see module docstring + _atrend_events).
            for lookback_h in _ATREND_LOOKBACKS_H:
                for atr_mult in _ATREND_MULTIPLIERS:
                    for vol_scale in VOL_SCALES:
                        for crash in gates["crash"]:
                            for regime in gates["regime"]:
                                for e in _atrend_events(
                                    hourly,
                                    lookback_h=lookback_h,
                                    atr_mult=atr_mult,
                                    vol_scale=vol_scale,
                                    crash=crash,
                                    regime=regime,
                                ):
                                    e["section"] = section
                                    all_events.append(e)
            continue
        if section == "vreg":
            # vreg H6 grid: the (L, alpha) mapping is PINNED by the vol regime
            # (never a config axis); only the gate/sizing axes are learned.
            for vol_scale in VOL_SCALES:
                for crash in gates["crash"]:
                    for regime in gates["regime"]:
                        for e in _vreg_events(
                            hourly,
                            vol_scale=vol_scale,
                            crash=crash,
                            regime=regime,
                        ):
                            e["section"] = section
                            all_events.append(e)
            continue
        if section == "tsmom":
            # tsmom H6 grid: L (lookback) x vol x reversal overlay x crash x
            # regime. Only L / vol / reversal / gates are learned; the sign
            # construction is fixed by the AQR papers.
            for lookback_bars in _TSMOM_LOOKBACKS_BARS:
                for vol_scale in VOL_SCALES:
                    for reversal in (False, True):
                        for crash in gates["crash"]:
                            for regime in gates["regime"]:
                                for e in _tsmom_events(
                                    hourly,
                                    lookback_bars=lookback_bars,
                                    vol_scale=vol_scale,
                                    reversal=reversal,
                                    crash=crash,
                                    regime=regime,
                                ):
                                    e["section"] = section
                                    all_events.append(e)
            continue
        if section == "pth":
            # pth H7 grid: k (lookback, incl. the 52-week anchor) x vol x crash
            # x regime. Only k / vol / gates are learned; the sign(PTH - 0.5)
            # construction and the 0.5 band are fixed by the papers.
            for lookback_bars in _PTH_LOOKBACKS_BARS:
                for vol_scale in VOL_SCALES:
                    for crash in gates["crash"]:
                        for regime in gates["regime"]:
                            for e in _pth_events(
                                hourly,
                                lookback_bars=lookback_bars,
                                vol_scale=vol_scale,
                                crash=crash,
                                regime=regime,
                            ):
                                e["section"] = section
                                all_events.append(e)
            continue
        if section == "fade":
            # fade H7 grid: lookback x vol x crash x regime. Only these axes
            # are learned; tau is FIXED at 1.0 (never a config axis).
            for lookback_bars in _FADE_LOOKBACKS_BARS:
                for vol_scale in VOL_SCALES:
                    for crash in gates["crash"]:
                        for regime in gates["regime"]:
                            for e in _fade_events(
                                hourly,
                                lookback_bars=lookback_bars,
                                vol_scale=vol_scale,
                                crash=crash,
                                regime=regime,
                            ):
                                e["section"] = section
                                all_events.append(e)
            continue
        for lookback_h in LOOKBACKS_H:
            for hold_h in HOLDS_H:
                for stop_pct in STOPS:
                    for vol_scale in VOL_SCALES:
                        for prox_tol in gates["prox_tol"]:
                            for cgo in gates["cgo"]:
                                for crash in gates["crash"]:
                                    for e in _trade_events(
                                        hourly,
                                        lookback_h=lookback_h,
                                        hold_h=hold_h,
                                        stop_pct=stop_pct,
                                        vol_scale=vol_scale,
                                        prox_tol=prox_tol,
                                        cgo=cgo,
                                        crash=crash,
                                        regime=gates["regime"][0],
                                    ):
                                        e["section"] = section
                                        all_events.append(e)
    if not all_events:
        return {
            "symbol": symbol,
            "span_days": span_days,
            "n_hours": int(len(hourly)),
            "error": "no tradable events (need ~28d of contiguous hourly history)",
        }

    # Global cut keeps the OOS calendar window IDENTICAL across sections so the
    # section comparison and B&H benchmark are apples-to-apples.
    ts = np.array([e["ts"] for e in all_events])
    cut = ts.min() + wf_split * (ts.max() - ts.min())

    # Buy & Hold over the fixed OOS calendar window [cut, end].
    bh = hourly[hourly["window_start_ms"] >= cut]
    bh_multiple = 1.0
    if len(bh) >= 2:
        bh_multiple = float(bh["close"].iloc[-1] / bh["open"].iloc[0])
    oos_span_days = max((int(bh["window_start_ms"].iloc[-1]) - cut) / _DAY_MS, 1.0)
    bh_ann = bh_multiple ** (365.0 / oos_span_days) - 1.0

    by_section: dict[str, list[dict]] = {s: [] for s in SECTIONS}
    for e in all_events:
        by_section[e["section"]].append(e)

    sections_out: dict[str, dict] = {}
    for section, evs in by_section.items():
        by_key: dict[tuple, list[dict]] = {}
        for e in evs:
            by_key.setdefault(_config_key(e, section), []).append(e)
        config_rows: list[dict] = []
        for key, sub in by_key.items():
            first = sub[0]
            hold_h = first["hold_h"]
            avg_hold_h = float(np.mean([e.get("actual_hold_h", hold_h) for e in sub]))
            tpy = 365.0 * 24.0 / avg_hold_h
            is_evs = [e for e in sub if e["ts"] < cut]
            oos_evs = [e for e in sub if e["ts"] >= cut]
            is_maker = np.array([e["net_maker"] for e in is_evs])
            oos_maker = np.array([e["net_maker"] for e in oos_evs])
            oos_taker = np.array([e["net_taker"] for e in oos_evs])
            if section == "vreg":
                base = {
                    "lookback_h": None,  # pinned by the vol regime, not a config axis
                    "hold_h": hold_h,
                    "avg_hold_h": round(avg_hold_h, 1),
                    "stop_pct": _VREG_HARD_STOP,
                    "vol_scale": key[0],
                    "prox_tol": None,
                    "cgo": False,
                    "crash": key[1],
                    "regime": key[2],
                    "atr_mult": None,
                    "vreg": True,
                }
            elif section == "tsmom":
                base = {
                    "lookback_h": key[0],
                    "hold_h": hold_h,
                    "avg_hold_h": round(avg_hold_h, 1),
                    "stop_pct": 0.0,
                    "vol_scale": key[1],
                    "prox_tol": None,
                    "cgo": False,
                    "crash": key[3],
                    "regime": key[4],
                    "atr_mult": None,
                    "reversal": key[2],
                    "tsmom": True,
                }
            elif section in ("pth", "fade"):
                base = {
                    "lookback_h": key[0],
                    "hold_h": hold_h,
                    "avg_hold_h": round(avg_hold_h, 1),
                    "stop_pct": 0.0,
                    "vol_scale": key[1],
                    "prox_tol": None,
                    "cgo": False,
                    "crash": key[2],
                    "regime": key[3],
                    "atr_mult": None,
                    "pth": section == "pth",
                    "fade": section == "fade",
                }
            else:
                base = {
                    "lookback_h": key[0],
                    "hold_h": hold_h,
                    "avg_hold_h": round(avg_hold_h, 1),
                    "stop_pct": key[2],
                    "vol_scale": key[3],
                    "prox_tol": key[4],
                    "cgo": key[5],
                    "crash": key[6],
                    "regime": key[7],
                    "atr_mult": key[8],
                    "vreg": False,
                }
            config_rows.append(
                {
                    **base,
                    "is_n": int(len(is_evs)),
                    "is_mean_net_maker_bps": round(1e4 * float(is_maker.mean()), 2)
                    if is_maker.size
                    else None,
                    "oos": _stats(oos_maker, trades_per_year=tpy),
                    "oos_taker": _stats(oos_taker, trades_per_year=tpy),
                    "oos_gross": _stats(
                        np.array([e["gross"] for e in oos_evs]), trades_per_year=tpy
                    ),
                    "oos_maker_pnls": [float(e["net_maker"]) for e in oos_evs],
                    "oos_taker_pnls": [float(e["net_taker"]) for e in oos_evs],
                }
            )
        eligible = [r for r in config_rows if (r["is_n"] or 0) >= _MIN_IS_TRADES]
        selected = max(eligible, key=lambda r: r["is_mean_net_maker_bps"]) if eligible else None
        sections_out[section] = {
            "n_configs": len(config_rows),
            "n_events": int(len(evs)),
            "selected": selected,
            "configs": config_rows,
        }

    # xs sections are universe-level -- never per-symbol -- so probe_xs owns
    # them entirely; drop any empty xs entries before returning.
    sections_out = {k: v for k, v in sections_out.items() if k not in XS_SECTIONS}

    return {
        "symbol": symbol,
        "span_days": span_days,
        "n_hours": int(len(hourly)),
        "n_events": len(all_events),
        "wf_split": wf_split,
        "cut_ms": int(cut),
        "bh_oos": {
            "start_ms": int(cut),
            "span_days": round(oos_span_days, 1),
            "multiple": round(bh_multiple, 4),
            "ann_return": round(bh_ann, 4),
        },
        "sections": sections_out,
    }


def _verdict(sel: dict | None, bh_mult: float, *, min_oos: int) -> dict:
    oos = (sel or {}).get("oos", {})
    n = oos.get("n", 0)
    if sel is None or n < min_oos:
        return {
            "passes": False,
            "reason": f"selected config has {n} OOS trades (< {min_oos})",
        }
    mean_net_maker_bps = oos["mean_net_bps"]
    winsorized_maker_bps = oos.get("winsorized_mean_net_bps", mean_net_maker_bps)
    strat_mult = oos["net_multiple"]
    positive = winsorized_maker_bps > 0.0
    beats_bh = strat_mult > bh_mult
    return {
        "passes": bool(positive and beats_bh),
        "positive": bool(positive),
        "beats_bh": bool(beats_bh),
        "clears_taker_gate": bool(mean_net_maker_bps >= _GATE_TAKER_BPS),
        "mean_net_maker_bps": mean_net_maker_bps,
        "winsorized_mean_net_maker_bps": winsorized_maker_bps,
        "oos_multiple": strat_mult,
        "bh_multiple": bh_mult,
        "reason": (
            "OOS net-of-maker positive (winsorized) and beats Buy & Hold"
            if (positive and beats_bh)
            else "OOS net-of-maker <= 0 (winsorized) OR below Buy & Hold -> kill"
        ),
    }


def probe_xs(hourly_by_symbol: dict[str, pd.DataFrame], *, wf_split: float) -> dict:
    """Cross-sectional sections across the WHOLE universe (see module docstring).

    Config selection is GLOBAL: every section's config grid is generated over
    all symbols at once, the in-sample POOLED legs pick the best config, and
    only the out-of-sample pooled legs are scored -- never per-symbol, because
    the rank/quantile construction has no single-asset meaning. The benchmark
    is the geometric-mean OOS multiple of the same universe's symbols (an
    equal-weight long-only basket): a long-short book must beat it to pass.
    """
    events_all: list[dict] = []
    for section in XS_SECTIONS:
        gates = SECTIONS[section]
        for lookback_h in _XS_LOOKBACKS_H:
            precomp = _xs_precompute(hourly_by_symbol, signal=section, lookback_h=lookback_h)
            if not precomp:
                continue
            for vol_scale in VOL_SCALES:
                for crash in gates["crash"]:
                    for regime in gates["regime"]:
                        for e in _xs_events(
                            hourly_by_symbol,
                            signal=section,
                            lookback_h=lookback_h,
                            vol_scale=vol_scale,
                            crash=crash,
                            regime=regime,
                            precomp=precomp,
                        ):
                            e["section"] = section
                            events_all.append(e)

    ts_all = np.array([e["ts"] for e in events_all])
    cut = int(ts_all.min() + wf_split * (ts_all.max() - ts_all.min())) if ts_all.size else 0

    mults: list[float] = []
    for s, df in hourly_by_symbol.items():
        if df is None or df.empty:
            continue
        bh = df[df["window_start_ms"] >= cut]
        if len(bh) >= 2:
            mults.append(float(bh["close"].iloc[-1] / bh["open"].iloc[0]))
    pooled_bh = float(np.exp(np.mean(np.log(mults)))) if mults else 1.0

    sections_out: dict[str, dict] = {}
    for section in XS_SECTIONS:
        evs = [e for e in events_all if e["section"] == section]
        by_key: dict[tuple, list[dict]] = {}
        for e in evs:
            by_key.setdefault(_config_key(e, section), []).append(e)
        config_rows: list[dict] = []
        for key, sub in by_key.items():
            is_evs = [e for e in sub if e["ts"] < cut]
            oos_evs = [e for e in sub if e["ts"] >= cut]
            is_maker = np.array([e["net_maker"] for e in is_evs])
            oos_maker = np.array([e["net_maker"] for e in oos_evs])
            oos_taker = np.array([e["net_taker"] for e in oos_evs])
            config_rows.append(
                {
                    "lookback_h": key[0],
                    "hold_h": _XS_REBALANCE_H,
                    "avg_hold_h": _XS_REBALANCE_H,
                    "stop_pct": 0.0,
                    "vol_scale": key[1],
                    "prox_tol": None,
                    "cgo": False,
                    "crash": key[2],
                    "regime": key[3],
                    "atr_mult": None,
                    "xs_signal": section,
                    "is_n": int(len(is_evs)),
                    "is_mean_net_maker_bps": round(1e4 * float(is_maker.mean()), 2)
                    if is_maker.size
                    else None,
                    "oos": _stats(oos_maker, trades_per_year=_XS_TPY),
                    "oos_taker": _stats(oos_taker, trades_per_year=_XS_TPY),
                    "oos_maker_pnls": [float(e["net_maker"]) for e in oos_evs],
                    "oos_taker_pnls": [float(e["net_taker"]) for e in oos_evs],
                }
            )
        eligible = [r for r in config_rows if (r["is_n"] or 0) >= _MIN_IS_TRADES]
        selected = max(eligible, key=lambda r: r["is_mean_net_maker_bps"]) if eligible else None
        if selected is not None and selected["oos"]["n"] >= _MIN_OOS_TRADES_SYMBOL:
            pooled = {
                "n_symbols": len(hourly_by_symbol),
                "n_trades": selected["oos"]["n"],
                "maker": selected["oos"],
                "taker": selected["oos_taker"],
            }
        else:
            pooled = {"n_symbols": len(hourly_by_symbol), "n_trades": 0}
        sections_out[section] = {
            "pooled": pooled,
            "per_symbol": {},
            "pooled_bh_multiple": round(pooled_bh, 4),
            "selected": selected,
            "n_configs": len(config_rows),
            "n_events": int(len(evs)),
        }
    return {"sections": sections_out}


def run_probe(hourly: dict[str, pd.DataFrame], *, wf_split: float) -> dict:
    symbols = sorted(hourly)
    per = [probe_symbol(hourly[s], s, wf_split=wf_split) for s in symbols]
    results: dict = {
        "symbols": per,
        "maker_round_trip_bps": 1e4 * _MAKER_RT,
        "taker_gate_bps": _GATE_TAKER_BPS,
        "wf_split": wf_split,
    }

    pooled_bh = 1.0
    for r in per:
        if not r.get("error"):
            pooled_bh *= r["bh_oos"]["multiple"]

    sections_out: dict[str, dict] = {}
    for section in SECTIONS:
        if section in XS_SECTIONS:
            continue  # universe-level: handled by probe_xs below
        pooled_maker: list[float] = []
        pooled_taker: list[float] = []
        holds: list[int] = []
        per_symbol: dict[str, dict] = {}
        for r in per:
            if r.get("error"):
                continue
            sec = r["sections"][section]
            sel = sec["selected"]
            if sel is not None and sel["oos"]["n"] >= _MIN_OOS_TRADES_SYMBOL:
                per_symbol[r["symbol"]] = _verdict(
                    sel, r["bh_oos"]["multiple"], min_oos=_MIN_OOS_TRADES_SYMBOL
                )
            if sel is not None and sel["oos"]["n"] >= _MIN_OOS_TRADES_SYMBOL:
                pooled_maker.extend(sel["oos_maker_pnls"])
                pooled_taker.extend(sel["oos_taker_pnls"])
                holds.append(sel["avg_hold_h"])
        if pooled_maker:
            maker = np.array(pooled_maker)
            taker = np.array(pooled_taker)
            avg_hold = float(np.mean(holds))
            tpy = 365.0 * 24.0 / avg_hold
            pooled_stats = {
                "n_symbols": len(per_symbol),
                "n_trades": int(maker.size),
                "maker": _stats(maker, trades_per_year=tpy),
                "taker": _stats(taker, trades_per_year=tpy),
            }
        else:
            pooled_stats = {"n_symbols": 0, "n_trades": 0}
        sections_out[section] = {
            "pooled": pooled_stats,
            "per_symbol": per_symbol,
            "pooled_bh_multiple": round(pooled_bh, 4),
        }
        if pooled_stats["n_trades"] >= _MIN_OOS_TRADES:
            sel_proxy = {"oos": pooled_stats["maker"]}
            sections_out[section]["verdict"] = _verdict(
                sel_proxy, pooled_bh, min_oos=_MIN_OOS_TRADES
            )
        else:
            sections_out[section]["verdict"] = _verdict(None, pooled_bh, min_oos=_MIN_OOS_TRADES)

    # Universe-level cross-sectional sections (global config selection).
    if len(symbols) >= 2:
        xs = probe_xs(hourly, wf_split=wf_split)
        for section in XS_SECTIONS:
            sec = xs["sections"][section]
            if sec["pooled"]["n_trades"] >= _MIN_OOS_TRADES:
                sec["verdict"] = _verdict(
                    {"oos": sec["pooled"]["maker"]},
                    sec["pooled_bh_multiple"],
                    min_oos=_MIN_OOS_TRADES,
                )
            else:
                sec["verdict"] = _verdict(None, sec["pooled_bh_multiple"], min_oos=_MIN_OOS_TRADES)
            sections_out[section] = sec
    else:
        for section in XS_SECTIONS:
            sections_out[section] = {
                "pooled": {"n_symbols": 0, "n_trades": 0},
                "per_symbol": {},
                "pooled_bh_multiple": 1.0,
                "selected": None,
                "n_configs": 0,
                "n_events": 0,
                "verdict": _verdict(None, 1.0, min_oos=_MIN_OOS_TRADES),
            }
    results["sections"] = sections_out
    return results


_DEFAULT_SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,TRXUSDT,LINKUSDT,NEARUSDT,"
    "ADAUSDT,SUIUSDT,UNIUSDT,AVAXUSDT,CRVUSDT,PEPEUSDT,LTCUSDT,ICPUSDT,AAVEUSDT,"
    "XLMUSDT,HBARUSDT,DOTUSDT,FILUSDT,ARBUSDT,LDOUSDT,BCHUSDT,OPUSDT,ATOMUSDT,"
    "ETCUSDT,RUNEUSDT,GRTUSDT,ZECUSDT"
)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbols (default: 31 top-volume USDT pairs with "
        "~3y of backfilled BRONZE history)",
    )
    parser.add_argument("--out", default=None, help="JSON output path (default: print only)")
    parser.add_argument(
        "--wf-split",
        type=float,
        default=_WF_SPLIT,
        help="fraction of events used in-sample (default 0.6)",
    )
    args = parser.parse_args()

    symbols = csv_list(args.symbols) if args.symbols else csv_list(_DEFAULT_SYMBOLS)
    settings = get_settings()
    bars = fetch_bars(settings, symbols)
    hourly = _aggregate(bars, _HOUR_MS)
    results = run_probe(
        {s.upper(): hourly[hourly["symbol"] == s.upper()] for s in symbols},
        wf_split=args.wf_split,
    )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("trend_momentum_probe_written", path=args.out)

    for r in results["symbols"]:
        n_ev = r.get("n_events", "?")
        print(f"\n=== {r['symbol']} — {r['span_days']} days, {n_ev} events ===")
        if r.get("error"):
            print(f"  {r['error']}")
            continue
        bh = r["bh_oos"]
        print(
            f"  B&H over OOS ({bh['span_days']}d): {bh['multiple']} "
            f"({100 * bh['ann_return']:+.1f}%/yr)"
        )
        for section, sec in r["sections"].items():
            sel = sec["selected"]
            if sel is None:
                print(f"  [{section}] no eligible config")
                continue
            oos = sel["oos"]
            if sel.get("vreg"):
                tail = (
                    f"| L=vreg(28/56/112d) alpha=vreg H={sel['avg_hold_h']:.0f}h "
                    f"hard_stop={sel['stop_pct']} vol={sel['vol_scale']} "
                    f"crash={sel['crash']} regime={sel['regime']}"
                )
            elif sel.get("tsmom"):
                tail = (
                    f"| L={sel['lookback_h']}h H={sel['avg_hold_h']:.0f}h "
                    f"vol={sel['vol_scale']} reversal={sel.get('reversal')} "
                    f"crash={sel['crash']} regime={sel['regime']}"
                )
            elif sel.get("pth") or sel.get("fade"):
                tail = (
                    f"| L={sel['lookback_h']}h H={sel['avg_hold_h']:.0f}h "
                    f"vol={sel['vol_scale']} crash={sel['crash']} regime={sel['regime']}"
                    + (" tau=1.0" if sel.get("fade") else "")
                )
            else:
                tail = (
                    f"| L={sel['lookback_h']}h H={sel['avg_hold_h']:.0f}h stop={sel['stop_pct']} "
                    f"vol={sel['vol_scale']} prox={sel['prox_tol']} cgo={sel['cgo']} "
                    f"crash={sel['crash']} regime={sel['regime']}"
                    + (f" ATRx{sel['atr_mult']}" if sel.get("atr_mult") else "")
                )
            print(
                f"  [{section}] n={oos['n']} win={oos['win_rate']} "
                f"net_maker={oos['mean_net_bps']:+.1f}bps(med {oos['median_net_bps']:+.1f},"
                f"win {oos['winsorized_mean_net_bps']:+.1f}, se {oos['se_bps']}) "
                f"net_taker={sel['oos_taker']['mean_net_bps']:+.1f}bps "
                f"sharpe={oos['sharpe_ann']} mult={oos['net_multiple']} {tail}"
            )

    print("\n=== POOLED BOOK (both symbols, selected configs) — verdict per section ===")
    for section, sec in results["sections"].items():
        v = sec["verdict"]
        p = sec["pooled"]
        status = "PASS" if v["passes"] else "KILL"
        if p["n_trades"]:
            m = p["maker"]
            print(
                f"  [{section}] {status}: n={m['n']} "
                f"net_maker={m['mean_net_bps']:+.1f}bps(med {m['median_net_bps']:+.1f},"
                f"win {m['winsorized_mean_net_bps']:+.1f}) win={m['win_rate']} "
                f"sharpe={m['sharpe_ann']} strat_mult={m['net_multiple']} "
                f"vs BH={sec['pooled_bh_multiple']}"
            )
        else:
            print(f"  [{section}] {status}: no trades — {v['reason']}")

    print("\n=== CROSS-SECTIONAL (universe-level, pooled legs, global config) ===")
    for section in XS_SECTIONS:
        sec = results["sections"][section]
        v = sec["verdict"]
        p = sec["pooled"]
        sel = sec.get("selected")
        status = "PASS" if v["passes"] else "KILL"
        if p["n_trades"]:
            m = p["maker"]
            tail = (
                f"| signal=L{sel['lookback_h']}h vol={sel['vol_scale']} "
                f"crash={sel['crash']} regime={sel['regime']}"
                if sel
                else ""
            )
            print(
                f"  [{section}] {status}: n={m['n']} "
                f"net_maker={m['mean_net_bps']:+.1f}bps(med {m['median_net_bps']:+.1f},"
                f"win {m['winsorized_mean_net_bps']:+.1f}) win={m['win_rate']} "
                f"sharpe={m['sharpe_ann']} strat_mult={m['net_multiple']} "
                f"vs BH={sec['pooled_bh_multiple']} {tail}"
            )
        else:
            print(f"  [{section}] {status}: no trades — {v['reason']}")


if __name__ == "__main__":
    main()
