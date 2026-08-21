"""Research: AWR -- Attention-Weighted Reversal (original factor filling a literature gap).

Method: read Chinese quant labs/firms + journals, find the GAP, fill it.
  DONE (Western academia, Liu-Tsyvinski-Wu JoF 2022): crypto = CMKT + CSMB(size)
        + CMOM(momentum), 1-4wk. Equity factors explain nothing.
  DONE (Chinese practitioners, FMZ blog + Lucida/Falcon fund): a HIDDEN GEM the
        journals ignored -- long COLD/low-volume coins, short HOT/high-volume coins
        works ("热门币更倾向于下跌"); liquidity/attention factor is long-effective.
  GAP:  nobody turned this into a size-DETRENDED, cost-aware, walk-forward-validated
        CROSS-SECTIONAL attention factor. Raw volume is size-confounded (BTC is always
        highest volume), so the academic silence is partly a measurement artifact. We fix
        that: attention = ABNORMAL volume (vol / trailing mean), cross-sectionally
        comparable across caps. We add a disposition-overhang leg (Chinese paper: combine
        overhang + momentum) so we short HOT coins that are ALSO up (pumps) and long COLD
        coins that are ALSO down (washed out).

  attention_t = vol_t / vol_ma        (size-detrended; high = retail attention/pump)
  weekly: rank coins by recent attention; SHORT top quintile (hot), LONG bottom (cold).
  + overhang: among ranks, prefer longs that are down & shorts that are up.

Baselines: MOM (momentum), REV (pure reversal), VOLRAW (raw-volume rank, confounded).
Everything parameterized; costs + walk-forward + crash regimes.

Run: uv run python scripts/research_awr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "attn_lookback": 5,  # days of abnormal-volume averaged at rebalance
    "vol_ma": 20,  # trailing volume baseline
    "corr_win": 20,  # close-volume correlation window (FMZ factor)
    "quintile": 0.20,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "lookback_grid": [3, 5, 10],
    "wf_train_weeks": 104,
    "wf_test_weeks": 13,
    "use_regime": False,
    "use_overhang": False,
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


def weekly_returns(close: pd.DataFrame) -> pd.DataFrame:
    w = close.resample("W-MON").last()
    return (w.shift(-1) / w - 1.0).iloc[CONFIG["vol_ma"] :]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    up = (btc > btc.rolling(CONFIG["regime_fast"]).mean()) & (
        btc > btc.rolling(CONFIG["regime_slow"]).mean()
    )
    return up.resample("W-MON").last()


def attention_score(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Size-detrended attention = abnormal volume (vol / trailing mean). High = pump/attention."""
    vol = panels["vol"]
    return vol / vol.rolling(CONFIG["vol_ma"]).mean()


def raw_volume_rank(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Confounded baseline: raw volume level (size-laden)."""
    return panels["vol"]


def overhang(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Disposition overhang proxy: cumulative recent return (Chinese paper uses capital-gains overhang)."""
    close = panels["close"]
    return close / close.shift(10) - 1.0


def awr_weights(attn, date, overh, regime_flag, panic_long_only=False):
    a = attn.loc[date].dropna()
    if len(a) < 12:
        return None
    if CONFIG["use_regime"] and not regime_flag:
        return None
    ranked = a.sort_values()  # low attention = cold (long candidates first)
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs = ranked.index[:n]  # coldest coins
    shorts = ranked.index[-n:]  # hottest coins
    if CONFIG["use_overhang"] and overh is not None:
        # prefer longs that are DOWN (washed out) and shorts that are UP (pumps)
        oh = overh.loc[date].reindex(ranked.index).dropna()
        longs = [c for c in ranked.index[: 2 * n] if oh.get(c, 0) < 0][:n] or list(ranked.index[:n])
        shorts = [c for c in ranked.index[-2 * n :][::-1] if oh.get(c, 0) > 0][:n] or list(
            ranked.index[-n:]
        )
    w = pd.Series(1.0 / n, index=longs)
    w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest_awr(panels, score, overh, fwd, regime, lb: int) -> pd.Series:
    sig = score.rolling(lb).mean().resample("W-MON").last().iloc[CONFIG["vol_ma"] :]
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        w = awr_weights(sig, date, overh, bool(regime.loc[date]))
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


def walk_forward_lb(panels, score, overh, fwd, regime) -> list[int]:
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    lb_seq: list[int] = []
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        best_lb, best_s = CONFIG["lookback_grid"][0], -1e9
        for lb in CONFIG["lookback_grid"]:
            s = backtest_awr(panels, score, overh, fwd, regime, lb).iloc[start:end]
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
    attn = attention_score(panels)
    overh = overhang(panels).resample("W-MON").last()

    report("MOM (standard momentum L/S, baseline)", metrics(momentum_backtest(close, fwd)))
    report("REV (pure 7d reversal, baseline)", metrics(reversal_backtest(close, fwd)))

    # VOLRAW: raw-volume rank (size-confounded) -- shows why academia missed the factor
    CONFIG["use_regime"] = False
    CONFIG["use_overhang"] = False
    # reuse AWR machinery with raw volume as the score (high vol = short)
    lb_seq = walk_forward_lb(panels, raw_volume_rank(panels), overh, fwd, regime)
    print(f"\n[walk-forward VOLRAW] lookback selected={sorted({int(x) for x in lb_seq})}")
    report(
        "VOLRAW (raw-volume rank, SHORT hot, baseline/confounded)",
        metrics(backtest_awr(panels, raw_volume_rank(panels), overh, fwd, regime, 5)),
    )

    # AWR: size-detrended attention factor
    for over in (False, True):
        CONFIG["use_overhang"] = over
        tag = "AWR (attention contrarian)" + (" + disposition overhang" if over else "")
        lb_seq = walk_forward_lb(panels, attn, overh, fwd, regime)
        print(f"\n[walk-forward {tag}] lookback selected={sorted({int(x) for x in lb_seq})}")
        CONFIG["use_regime"] = False
        report(f"{tag} (WF lb, costs)", metrics(backtest_awr(panels, attn, overh, fwd, regime, 5)))
        CONFIG["use_regime"] = True
        report(
            f"{tag} + BTC regime gate (WF lb, costs)",
            metrics(backtest_awr(panels, attn, overh, fwd, regime, 5)),
        )
        CONFIG["use_regime"] = False

    # crash sub-periods for AWR (no regime, no overhang)
    CONFIG["use_overhang"] = False
    awr = backtest_awr(panels, attn, overh, fwd, regime, 5)
    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>15}" for k in CRASH))
    row = ""
    for a, b in CRASH.values():
        sub = awr.loc[a:b]
        sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
        row += f"{sr:15.2f}"
    print(f"  {'AWR':>13}{row}")


if __name__ == "__main__":
    main()
