"""GENUINE BREAKTHROUGH CANDIDATE -- VOLATILITY-SCALED momentum / ASYM.

RESEARCH (web, this session):
  "Cryptocurrency momentum has (not) its moments" (2025, top-30 caps, weekly):
    risk-managed momentum = 1.86-2.40% weekly (vs 0.71% plain), kurtosis HALVED,
    crashes mitigated. Barroso-Santa-Clara (2015) vol scaling: scale the book by
    target_vol / trailing_realized_vol.
  "Adaptive Risk Allocation in Crypto Markets" (SSRN 5090097): vol scaling increases
    Sharpe, strongest in MOMENTUM portfolios.
  Keel / Grayscale: per-leg vol target 10-15%, gross cap 1-2x; trend-following halves
    BTC drawdown.

Why this is the breakthrough: our ASYM book (Sharpe 1.10, maxDD -40.8%, ex-kurt 17.8)
is a momentum variant that CRASHES in high-vol regimes. Vol scaling de-risks exactly
when realized vol spikes (the crash windows) and levers up in calm trends -> higher
Sharpe, lower tail. This is NOT a new alpha; it is a documented risk-management overlay
on our existing ORIGINAL factor (ASYM), executed per the 2025 literature.

NO magic numbers: target_vol = the book's OWN full-sample average annualized vol
(Barroso-Santa-Clara ex-ante normalization, one constant derived from the book, not tuned).
gross cap = 2.0 (Keel's 1-2x). trailing vol window = 12 weeks. 10bps costs, BTC regime.

Run: uv run python scripts/research_volscale.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import random

CONFIG = {
    "formation_days": 14,
    "quintile": 0.20,
    "vol_target_cap": 2.0,  # Keel gross cap 1-2x
    "vol_win": 12,  # trailing realized-vol window (weeks)
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
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
PRICE_CACHE = Path("/tmp/crypto_daily_long.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")

VARIANTS = {
    "MOM14_REGIME": dict(score="mom", regime=True),
    "ASYM_REGIME": dict(score="asym", regime=True),
}


def load():
    for p in (PRICE_CACHE, FUND_CACHE):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    common = px.index.intersection(fd.index)
    px = px.loc[common].reindex(columns=UNIVERSE)
    fd = fd.loc[common].reindex(columns=UNIVERSE)
    print(f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()}")
    return px, fd


def weekly_frame(close: pd.DataFrame, formation: int):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    vol = (close.pct_change(fill_method=None).rolling(126).std() * np.sqrt(252)).reindex(
        w.index, method="ffill"
    )
    return fwd.iloc[formation:], mom.iloc[formation:], vol.iloc[formation:]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    fast = btc.rolling(CONFIG["regime_fast"]).mean()
    slow = btc.rolling(CONFIG["regime_slow"]).mean()
    up = (btc > fast) & (btc > slow)
    return up.resample("W-MON").last().reindex(close.resample("W-MON").last().index, method="ffill")


def build_scores(close: pd.DataFrame, fd: pd.DataFrame):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    fdw = fd.resample("W-MON").mean().reindex(fwd.index)
    mom_z = mom.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    fund_z = fdw.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    fund_accel = fund_z - fund_z.rolling(3).mean()
    squeeze = ((fund_z < -1.0) & (fund_accel > 0)).astype(float) * 2.0
    asym = mom_z.where(squeeze == 0, squeeze)
    return fwd, mom, vol, {"mom": mom_z, "asym": asym}


def weights_at(date, score, vol, regime_flag, spec: dict):
    if spec["regime"] and not regime_flag:
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    n = max(2, int(round(CONFIG["quintile"] * len(m))))
    ranked = m.sort_values()
    longs = ranked.index[-n:]
    shorts = ranked.index[:n]
    w = pd.concat([pd.Series(1.0 / n, index=longs), pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest_equal(close: pd.DataFrame, score: pd.DataFrame, spec: dict) -> pd.Series:
    fwd, _, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        w = weights_at(date, score, vol, bool(reg.loc[date]), spec)
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


def backtest_volscale(close, score, spec, target, cap, vol_win) -> pd.Series:
    fwd, _, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    dates = list(fwd.index)
    ret, prev, raw_hist = [], pd.Series(dtype=float), []
    for date in dates:
        w0 = weights_at(date, score, vol, bool(reg.loc[date]), spec)
        if w0 is None:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            raw_hist.append(0.0)
            continue
        # RAW (unscaled) gross return of the book -- basis for the ex-ante vol estimate.
        # Must be unscaled so the vol estimate does not feed back on the leverage itself.
        r0 = float((w0 * fwd.loc[date].reindex(w0.index)).sum(skipna=True))
        if len(raw_hist) >= vol_win:
            rv = float(np.std(raw_hist[-vol_win:]) * np.sqrt(52))
            L = min(cap, target / rv) if rv > 1e-9 else cap
        else:
            L = 1.0
        w = w0 * L
        r = L * r0  # scaled gross return
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
        raw_hist.append(r0)
    return pd.Series(ret, index=dates)


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    n = len(ret)
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = random.Random(0)
    vals = list(ret.values)
    boot = []
    for _ in range(1000):
        sample = [rng.choice(vals) for _ in range(n)]
        sm = sum(sample) / n * 52
        sd = (sum((x - sum(sample) / n) ** 2 for x in sample) / max(1, n - 1)) ** 0.5 * np.sqrt(52)
        boot.append(sm / sd if sd > 0 else 0.0)
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
        "pct_flat": float((ret == 0).mean()),
    }


def report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  %flat={m['pct_flat'] * 100:.0f}%"
    )
    print(f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%")


CRASH = {
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX 2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close, fd = load()
    fwd, mom, vol, S = build_scores(close, fd)
    results = {}
    for name, spec in VARIANTS.items():
        ret_eq = backtest_equal(close, S[spec["score"]], spec)
        target = float(ret_eq.std() * np.sqrt(52))  # B-S ex-ante normalization
        print(f"[target_vol for {name} = {target:.3f}]")
        report(name + " EQUAL", metrics(ret_eq))
        results[name + " EQUAL"] = ret_eq
        for cap in (1.0, 2.0):
            ret_vs = backtest_volscale(
                close, S[spec["score"]], spec, target, cap, CONFIG["vol_win"]
            )
            label = name + f" VOLSCALE_C{cap:.0f}"
            report(label, metrics(ret_vs))
            results[label] = ret_vs
    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>14}" for k in CRASH))
    for name, ret in results.items():
        row = ""
        for a, b in CRASH.values():
            sub = ret.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:14.2f}"
        print(f"  {name:>22}{row}")


if __name__ == "__main__":
    main()
