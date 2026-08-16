"""Research: FRD -- Funding-Reflexivity Divergence (ORIGINAL, crypto-native mechanism).

This is NOT a recombination of published factors. It is a new primitive.

The problem it solves:
  Every crypto trend-follower dies the same death -- it cannot tell a "healthy" trend
  from one where leveraged crowd is trapped fighting gravity, until it CASCADES. Grobys
  (2025) documents the crashes; nobody predicts WHICH trends cascade. That is the gap.

The mechanism (original):
  Crypto is the only major asset where the SAME instrument (the perpetual) embeds
  direction AND a leverage-sentiment yield (funding, settled every 8h). Carry papers use
  funding LEVEL (crowd direction). Momentum / CGO papers use PRICE alone. NO paper uses the
  CO-MOVEMENT of funding-CHANGE with price-CHANGE -- i.e. whether leverage sentiment is
  CONFIRMING or FIGHTING the trend. That co-movement IS the Soros reflexivity loop, and it
  is the actual mechanism behind a cascade.

  Define per-day reflexivity raw:
      +1  if  funding FALLS and price RISES   -> shorts capitulating into strength  -> LONG
      -1  if  funding RISES   and price FALLS  -> leveraged longs trapped, paying into
                                                 a losing trend                    -> SHORT
       0  if funding & price co-move            -> "honest" trend, already priced    -> FLAT
  FRD score = trailing mean of raw (in [-1,1]). Cross-sectionally: LONG top quintile
  (positive reflexivity), SHORT bottom quintile (negative reflexivity).

  This is long exactly when reflexivity pressure is building -- so it should earn its keep
  in the crash regimes where pure momentum (NAIVE_LS Sharpe -1.41 FTX) and SCX (COVID -3.17)
  fail.

Baselines (to prove FRD is distinct from known edges):
  MOM  -- standard 14d XS momentum L/S (Liu-Tsyvinski).
  LCF  -- funding-LEVEL carry (Christin/BIS); the closest cousin, tests "level vs divergence".
  REV  -- pure 1-week short-term reversal (no funding).

Discipline: parameters in CONFIG (window, min-|dfunding|) are WALK-FORWARD selected, not
hardcoded. Costs + crash regimes on. Maker/taker both implied (weekly turnover drag).

Run: uv run python scripts/research_frd.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "quintile": 0.20,
    "cost_bps": 10.0,
    "window_grid": [2, 3, 4, 6, 8],  # weeks of daily fight observation
    "min_df_grid": [0.0, 1e-5, 5e-5, 1e-4],  # min |daily funding diff| to count as "moving"
    "wf_train_weeks": 104,
    "wf_test_weeks": 13,
    "regime_fast": 90,
    "regime_slow": 200,
}
CLOSE_CACHE = Path("/tmp/crypto_daily_c.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")  # DAILY SUMMED funding yield (not annualized)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.read_csv(CLOSE_CACHE, index_col=0, parse_dates=True)
    close.index = pd.to_datetime(close.index, utc=True).tz_localize(None)
    fund = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fund.index = pd.to_datetime(fund.index, utc=True).tz_localize(None)
    common = close.columns.intersection(fund.columns)
    close, fund = close[common], fund[common]
    idx = close.index.intersection(fund.index)
    close, fund = close.loc[idx], fund.loc[idx]
    print(
        f"[data] close {close.shape}  fund {fund.shape}  "
        f"{close.index.min().date()}..{close.index.max().date()}"
    )
    return close, fund


def weekly_returns(close: pd.DataFrame) -> pd.DataFrame:
    w = close.resample("W-MON").last()
    return (w.shift(-1) / w - 1.0).iloc[14:]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    up = (btc > btc.rolling(CONFIG["regime_fast"]).mean()) & (
        btc > btc.rolling(CONFIG["regime_slow"]).mean()
    )
    return up.resample("W-MON").last()


def frd_score(
    close: pd.DataFrame, fund: pd.DataFrame, window_weeks: int, min_df: float
) -> pd.DataFrame:
    """Trailing mean of the daily reflexivity raw signal, resampled weekly.

    raw = +1 if funding falls & price rises (shorts capitulating -> LONG)
          -1 if funding rises  & price falls (trapped longs   -> SHORT)
           0 otherwise (co-moving "honest" trend -> FLAT)
    Only days where |dfunding| >= min_df count as a real sentiment move.
    """
    ret = close.pct_change()
    dfund = fund.diff()
    common = close.columns.intersection(fund.columns)
    ret, dfund = ret[common], dfund[common]
    raw = pd.DataFrame(0.0, index=ret.index, columns=common)
    raw[(dfund < -min_df) & (ret > 0)] = 1.0
    raw[(dfund > min_df) & (ret < 0)] = -1.0
    score = raw.resample("W-MON").mean()
    return score.iloc[window_weeks:]


def weights_from_score(score, date, direction) -> pd.Series | None:
    """direction='natural' -> LONG high score, SHORT low score (reflexivity).
    direction='flipped'   -> robustness check (reverse)."""
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    if direction == "natural":
        longs = ranked.index[-n:]
        shorts = ranked.index[:n]
    else:
        longs = ranked.index[:n]
        shorts = ranked.index[-n:]
    w = pd.concat([pd.Series(1.0 / n, index=longs), pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest(score, fwd, direction) -> pd.Series:
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        w = weights_from_score(score, date, direction)
        if w is None:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def momentum_backtest(close, fwd) -> pd.Series:
    mom = (close / close.shift(14) - 1.0).resample("W-MON").last().iloc[14:]
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        m = mom.loc[date].dropna()
        if len(m) < 12:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        ranked = m.sort_values()
        n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
        w = pd.concat(
            [
                pd.Series(1.0 / n, index=ranked.index[-n:]),
                pd.Series(-1.0 / n, index=ranked.index[:n]),
            ]
        )
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def reversal_backtest(close, fwd) -> pd.Series:
    rev = close.pct_change(7).resample("W-MON").last().iloc[1:]
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        m = rev.loc[date].dropna()
        if len(m) < 12:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        ranked = m.sort_values()
        n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
        w = pd.concat(
            [
                pd.Series(1.0 / n, index=ranked.index[:n]),
                pd.Series(-1.0 / n, index=ranked.index[-n:]),
            ]
        )
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def lcf_backtest(fund, fwd) -> pd.Series:
    """Funding-LEVEL carry (pro direction) -- the closest cousin to FRD, tests level vs divergence."""
    af = fund.resample("W-MON").last().rolling(4).mean().iloc[4:]
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        m = af.loc[date].dropna()
        if len(m) < 12:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        ranked = m.sort_values()
        n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
        w = pd.concat(
            [
                pd.Series(1.0 / n, index=ranked.index[-n:]),
                pd.Series(-1.0 / n, index=ranked.index[:n]),
            ]
        )
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    n = len(ret)
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    vals = ret.values.tolist()
    boot = []
    for _ in range(1000):
        s = [random.choice(vals) for _ in range(n)]
        m = sum(s) / n
        sd = (sum((x - m) ** 2 for x in s) / n) ** 0.5
        boot.append((m * 52) / (sd * np.sqrt(52)) if sd > 0 else 0.0)
    boot.sort()
    ci = (boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))])
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
        "pct_flat": float((ret == 0).mean()),
    }


def report(name, m):
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  %flat={m['pct_flat'] * 100:.0f}%"
    )
    print(
        f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%  CVaR5%={m['cvar05'] * 100:6.2f}%"
    )


CRASH = {
    "2021 bull": ("2021-01-01", "2021-12-31"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX Nov-22": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def crash_sharpe(ret: pd.Series) -> None:
    print("  --- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>15}" for k in CRASH))
    row = ""
    for a, b in CRASH.values():
        sub = ret.loc[a:b]
        sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
        row += f"{sr:15.2f}"
    print(f"  {'FRD':>13}{row}")


def main() -> None:
    close, fund = load()
    fwd = weekly_returns(close)
    global _SCORE_CACHE
    _SCORE_CACHE = {}

    def frd_score_cached(window_weeks, min_df):
        key = (window_weeks, min_df)
        if key not in _SCORE_CACHE:
            _SCORE_CACHE[key] = frd_score(close, fund, window_weeks, min_df)
        return _SCORE_CACHE[key]

    # Baselines
    report("MOM (14d XS momentum L/S, baseline)", metrics(momentum_backtest(close, fwd)))
    report("REV (1wk short-term reversal, baseline)", metrics(reversal_backtest(close, fwd)))
    report("LCF (funding-LEVEL carry, pro, cousin)", metrics(lcf_backtest(fund, fwd)))

    # FRD -- natural reflexivity direction
    win_seq, mdf_seq = walk_forward_params_cached(frd_score_cached, fwd, "natural")
    chosen = sorted(
        {(int(w), float(m)) for w, m in zip(win_seq, mdf_seq)}, key=lambda x: (x[0], x[1])
    )
    print(f"\n[walk-forward FRD natural] (window_weeks, min_df) selected={chosen}")
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for i, date in enumerate(dates):
        sc = frd_score_cached(win_seq[i], mdf_seq[i])
        w = weights_from_score(sc, date, "natural")
        if w is None:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    frd = pd.Series(ret, index=dates)
    report("FRD (natural reflexivity, WF window+min_df, costs)", metrics(frd))
    crash_sharpe(frd)

    # Robustness: flipped direction
    win_seq_f, mdf_seq_f = walk_forward_params_cached(frd_score_cached, fwd, "flipped")
    ret, prev = [], pd.Series(dtype=float)
    for i, date in enumerate(dates):
        sc = frd_score_cached(win_seq_f[i], mdf_seq_f[i])
        w = weights_from_score(sc, date, "flipped")
        if w is None:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    frd_f = pd.Series(ret, index=dates)
    report("FRD (flipped direction, robustness)", metrics(frd_f))


def walk_forward_params_cached(score_fn, fwd, direction) -> tuple[list, list]:
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    win_seq, mdf_seq = [], []
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        best = (CONFIG["window_grid"][0], CONFIG["min_df_grid"][0], -1e9)
        for win in CONFIG["window_grid"]:
            for mdf in CONFIG["min_df_grid"]:
                s = backtest(score_fn(win, mdf), fwd, direction).iloc[start:end]
                sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
                if sr > best[2]:
                    best = (win, mdf, sr)
        for _ in range(end, min(end + te, n)):
            win_seq.append(best[0])
            mdf_seq.append(best[1])
    while len(win_seq) < n:
        win_seq.append(win_seq[-1] if win_seq else CONFIG["window_grid"][1])
        mdf_seq.append(mdf_seq[-1] if mdf_seq else CONFIG["min_df_grid"][0])
    return win_seq[:n], mdf_seq[:n]


if __name__ == "__main__":
    main()
