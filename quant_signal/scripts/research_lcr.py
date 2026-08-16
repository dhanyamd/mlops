"""Research: LCR -- Liquidation-Cascade Reversal (original, crypto-native signal).

Method (think like a researcher): survey what is DONE, find the GAP, fill it.
  DONE: cross-sectional momentum (Liu-Tsyvinski 2021), vol-scaling (BSC 2015),
        funding-rate carry, BTC trend regime. All TREND-following.
  GAP:  crypto's defining microstructure is pervasive retail LEVERAGE on perps.
        A coin that dumps on EXPANSIVE volume AND EXPANSIVE true range is
        overwhelmingly a leveraged-LONG liquidation cascade: forced, non-
        fundamental selling that exhausts and mean-reverts. The academic
        literature has NOT turned this into a cross-sectional, volume+range
        CONFIRMED contrarian factor. LCR fills that gap.

  cascade_score_t = sign(ret_t) * (vol_t/vol_ma) * (TR_t/TR_ma)
    positive = up/euphoria cascade, negative = down/panic cascade.
  Weekly: rank coins by sum of recent cascade_score, FADE extremes
    (long panic losers, short euphoria winners). Opposite of momentum.

Baselines prove the novelty earns its keep:
  MOM -- standard cross-sectional momentum L/S
  REV -- pure short-term reversal (NO volume/range confirmation)
Everything parameterized; costs + walk-forward + crash regimes.

Run: uv run python scripts/research_lcr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "cascade_lookback": 3,
    "vol_ma": 20,
    "range_ma": 20,
    "quintile": 0.20,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "lookback_grid": [2, 3, 5],
    "wf_train_weeks": 104,
    "wf_test_weeks": 13,
    "use_regime": False,
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
CACHE = {
    "close": Path("/tmp/crypto_daily_c.csv"),
    "high": Path("/tmp/crypto_daily_h.csv"),
    "low": Path("/tmp/crypto_daily_l.csv"),
    "vol": Path("/tmp/crypto_daily_v.csv"),
}


def load_panels() -> dict[str, pd.DataFrame]:
    out = {}
    for f, p in CACHE.items():
        if not p.exists():
            raise SystemExit("run scripts/pull_binance_daily.py first")
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        d.index = pd.to_datetime(d.index, utc=True).tz_localize(None)
        out[f] = d
    print(
        f"[data] {out['close'].shape} {out['close'].index.min().date()}..{out['close'].index.max().date()}"
    )
    return out


def true_range(high, low, close) -> pd.DataFrame:
    prev = close.shift(1)
    hl = (high - low).abs().values
    hc = (high - prev).abs().values
    lc = (low - prev).abs().values
    tr = pd.DataFrame(np.maximum(np.maximum(hl, hc), lc), index=high.index, columns=high.columns)
    return tr


def cascade_score(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close, high, low, vol = panels["close"], panels["high"], panels["low"], panels["vol"]
    ret = close.pct_change(fill_method=None)
    vol_ma = vol.rolling(CONFIG["vol_ma"]).mean()
    tr = true_range(high, low, close)
    tr_ma = tr.rolling(CONFIG["range_ma"]).mean()
    return np.sign(ret) * (vol / vol_ma) * (tr / tr_ma)


def weekly_returns(close: pd.DataFrame) -> pd.DataFrame:
    w = close.resample("W-MON").last()
    return (w.shift(-1) / w - 1.0).iloc[CONFIG["vol_ma"] :]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    up = (btc > btc.rolling(CONFIG["regime_fast"]).mean()) & (
        btc > btc.rolling(CONFIG["regime_slow"]).mean()
    )
    return up.resample("W-MON").last()


def lcr_weights(sig: pd.DataFrame, date, regime_flag, panic_only: bool):
    m = sig.loc[date].dropna()
    if len(m) < 12:
        return None
    if CONFIG["use_regime"] and not regime_flag:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs = ranked.index[:n]  # most-negative cascade = panic
    shorts = ranked.index[-n:]  # most-positive cascade = euphoria
    w = pd.Series(1.0 / n, index=longs)
    if not panic_only:
        w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest_lcr(panels, score, fwd, regime, panic_only: bool, lb: int) -> pd.Series:
    sig = score.rolling(lb).sum().resample("W-MON").last().iloc[CONFIG["vol_ma"] :]
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        w = lcr_weights(sig, date, bool(regime.loc[date]), panic_only)
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


def walk_forward_lb(panels, score, fwd, regime, panic_only: bool) -> list[int]:
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    lb_seq: list[int] = []
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        best_lb, best_s = CONFIG["lookback_grid"][0], -1e9
        for lb in CONFIG["lookback_grid"]:
            s = backtest_lcr(panels, score, fwd, regime, panic_only, lb).iloc[start:end]
            sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
            if sr > best_s:
                best_s, best_lb = sr, lb
        for _ in range(end, min(end + te, n)):
            lb_seq.append(best_lb)
    while len(lb_seq) < n:
        lb_seq.append(lb_seq[-1] if lb_seq else CONFIG["lookback_grid"][1])
    return lb_seq[:n]


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
    rev = close.pct_change(7).resample("W-MON").last().iloc[7:]
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
    "2018 bear": ("2018-01-01", "2018-12-31"),
    "COVID Mar-2020": ("2020-02-20", "2020-04-01"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX Nov-2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    panels = load_panels()
    close = panels["close"]
    fwd = weekly_returns(close)
    regime = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    score = cascade_score(panels)

    report("MOM (standard momentum L/S, baseline)", metrics(momentum_backtest(close, fwd)))
    report(
        "REV (pure 7d reversal, NO vol confirm, baseline)", metrics(reversal_backtest(close, fwd))
    )

    for panic_only in (False, True):
        tag = "LCR (cascade L/S)" if not panic_only else "LCR_panic (long panic only)"
        lb_seq = walk_forward_lb(panels, score, fwd, regime, panic_only)
        print(f"\n[walk-forward {tag}] lookback selected={sorted({int(x) for x in lb_seq})}")
        CONFIG["use_regime"] = False
        report(
            f"{tag} (WF lb, costs)",
            metrics(backtest_lcr(panels, score, fwd, regime, panic_only, 3)),
        )
        CONFIG["use_regime"] = True
        report(
            f"{tag} + BTC regime gate (WF lb, costs)",
            metrics(backtest_lcr(panels, score, fwd, regime, panic_only, 3)),
        )
        CONFIG["use_regime"] = False

    # crash sub-periods for LCR (no regime)
    lb_seq = walk_forward_lb(panels, score, fwd, regime, False)
    lcr = backtest_lcr(panels, score, fwd, regime, False, 3)
    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>15}" for k in CRASH))
    row = ""
    for a, b in CRASH.values():
        sub = lcr.loc[a:b]
        sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
        row += f"{sr:15.2f}"
    print(f"  {'LCR':>13}{row}")


if __name__ == "__main__":
    main()
