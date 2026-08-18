"""Research: idiosyncratic tail-aware cross-sectional crypto momentum.

NO hardcoded magic numbers in the logic -- parameters live in CONFIG, and the
tail-filter threshold k is selected by WALK-FORWARD (calibrated on a rolling
train window, applied out-of-sample). Transaction costs are a config knob.

Pipeline:
  1. load long daily panel (Binance 2017-2026, crash regimes included)
  2. backtest NAIVE / BSC(vol-scaled) / TAIL_ALL(idiosyncratic short-leg filter)
     with transaction costs
  3. TAIL_ALL's k is chosen walk-forward (not hardcoded)
  4. tail metrics: Hill index both tails, skew, kurtosis, maxDD, CVaR, bootstrap CI

Run: uv run python scripts/research_xs_tail.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── parameters (NOT hardened into the logic) ───────────────────────────────────
CONFIG = {
    "formation_days": 14,  # momentum lookback (daily)
    "quintile": 0.20,  # long/short book size
    "vol_target": 0.55,  # BSC target annualized vol
    "vol_lookback_days": 126,  # realized-vol estimate window
    "cost_bps": 10.0,  # round-trip-ish taker+slippage per weekly rebalance (PARAM)
    "k_grid": [1.3, 1.5, 1.8, 2.2],  # candidate tail thresholds (searched, not hardcoded)
    "wf_train_weeks": 104,  # 2y rolling calibration window
    "wf_test_weeks": 13,  # 3m out-of-sample block
}
UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "ADAUSDT",
    "SUIUSDT",
    "UNIUSDT",
    "AVAXUSDT",
    "CRVUSDT",
    "PEPEUSDT",
    "LTCUSDT",
    "ICPUSDT",
    "AAVEUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "DOTUSDT",
    "FILUSDT",
    "ARBUSDT",
    "LDOUSDT",
    "BCHUSDT",
    "OPUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "RUNEUSDT",
    "GRTUSDT",
    "ZECUSDT",
]
LONG_CACHE = Path("/tmp/crypto_daily_long.csv")


# ── data ───────────────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    if LONG_CACHE.exists():
        p = pd.read_csv(LONG_CACHE, index_col=0, parse_dates=True)
        p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
        print(f"[data] long panel {p.shape} {p.index.min().date()}..{p.index.max().date()}")
        return p
    raise SystemExit("run scripts/pull_binance_daily.py first")


# ── weekly signals ─────────────────────────────────────────────────────────────
def weekly_frame(
    close: pd.DataFrame, formation: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fwd (next-week return), mom (formation-daily return), vol (annualized)."""
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    vol = (
        close.pct_change(fill_method=None).rolling(CONFIG["vol_lookback_days"]).std() * np.sqrt(252)
    ).reindex(w.index, method="ffill")
    return fwd.iloc[formation:], mom.iloc[formation:], vol.iloc[formation:]


def weights_at(date, mom, vol, variant: str, k):
    """Return target weights (long/short) for the NEXT week, or None if flat."""
    m = mom.loc[date].dropna()
    if m.shape[0] < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs = ranked.index[-n:]
    shorts = ranked.index[:n]
    w = pd.concat([pd.Series(1.0 / n, index=longs), pd.Series(-1.0 / n, index=shorts)])
    if variant in ("bsc", "tail_all"):
        rv = vol.loc[date].reindex(w.index).dropna()
        if rv.empty or rv.mean() <= 0:
            return None
        scale = float(np.clip(CONFIG["vol_target"] / rv.mean(), 0.0, 3.0))
        w = w * scale
    if variant == "tail_all" and k is not None:
        rv = vol.loc[date]
        med = rv.median()
        if not np.isfinite(med) or med <= 0:
            return None
        ratio = (rv / med).reindex(shorts).dropna()
        extreme = ratio[ratio > k]
        if not extreme.empty:
            worst = float(extreme.max())
            fade = float(np.clip(1.0 - (worst - k) / k, 0.0, 1.0))
            w = w * fade
    return w


def backtest(close: pd.DataFrame, variant: str, k_seq: list[float | None]) -> pd.Series:
    """k_seq: per-rebalance threshold (None => no tail filter). Applies costs."""
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    dates = fwd.index
    ret, prev = [], pd.Series(dtype=float)
    for i, date in enumerate(dates):
        k = k_seq[i] if i < len(k_seq) else k_seq[-1]
        w = weights_at(date, mom, vol, variant, k)
        if w is None:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev) and not prev.empty:
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


# ── walk-forward k selection (out-of-sample, not hardcoded) ─────────────────────
def walk_forward_k(close: pd.DataFrame, variant: str) -> list[float | None]:
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    dates = list(fwd.index)
    n = len(dates)
    k_seq: list[float | None] = [None] * n
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    for start in range(0, n - te, te):
        end = start + tr
        if end >= n - te:
            end = n - te
        # calibrate k on train block [start, end)
        best_k, best_s = None, -1e9
        for k in CONFIG["k_grid"]:
            ks = [k] * (end - start) + [None] * (n - end)
            block = backtest_on(close, variant, ks, start, end)
            s = block.mean() / (block.std() + 1e-9) * np.sqrt(52)
            if s > best_s:
                best_s, best_k = s, k
        for j in range(end, min(end + te, n)):
            k_seq[j] = best_k
    # any remaining tail uses last chosen k
    last = next((x for x in reversed(k_seq) if x is not None), CONFIG["k_grid"][1])
    k_seq = [x if x is not None else last for x in k_seq]
    return k_seq


def backtest_on(close, variant, k_seq, a, b) -> pd.Series:
    full = backtest(close, variant, k_seq)
    return full.iloc[a:b]


# ── metrics ─────────────────────────────────────────────────────────────────────
def hill(s: pd.Series, k: int = 15) -> float:
    s = s.dropna().sort_values(ascending=False).head(k)
    if len(s) < 5:
        return float("nan")
    th = float(s.min())
    return float("nan") if th <= 0 else float(1.0 / np.mean(np.log(s / th)))


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    n = len(ret)
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = np.random.default_rng(0)
    boot = [
        (rng.choice(ret.values, n, replace=True).mean() * 52)
        / (rng.choice(ret.values, n, replace=True).std() * np.sqrt(52))
        for _ in range(1000)
    ]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {
        "n": n,
        "ann_ret": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "ci": ci,
        "skew": float(ret.skew()),
        "exkurt": float(ret.kurt()),
        "maxdd": dd,
        "cvar05": ret.quantile(0.05),
        "left_hill": hill(-ret[ret < 0]),
        "right_hill": hill(ret[ret > 0]),
    }


def report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]"
    )
    print(
        f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%"
        f"  CVaR5%={m['cvar05'] * 100:6.2f}%"
    )
    print(f"  left-Hill={m['left_hill']:.2f}  right-Hill={m['right_hill']:.2f}  (higher=thinner)")


def main() -> None:
    close = load_panel()
    print(f"config: {CONFIG}")
    # NAIVE and BSC: no tail filter (k=None)
    report("NAIVE (costs, 2017-2026)", metrics(backtest(close, "naive", [None] * 5000)))
    report("BSC vol-scaled (costs)", metrics(backtest(close, "bsc", [None] * 5000)))
    # TAIL_ALL: walk-forward selected k (out-of-sample)
    k_wf = walk_forward_k(close, "tail_all")
    chosen = sorted({round(x, 2) for x in k_wf if x is not None})
    print(f"\n[walk-forward] k selected per block: {chosen}")
    report("TAIL_ALL (walk-forward k, costs)", metrics(backtest(close, "tail_all", k_wf)))


if __name__ == "__main__":
    main()
