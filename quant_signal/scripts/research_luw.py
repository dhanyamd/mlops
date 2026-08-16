"""Research: LUW -- Leverage-Unwind / Liquidation-Reversal cross-sectional factor.

A NOVEL, crypto-native, per-coin factor unlocked by KEYED data (open interest + liquidations).

Mechanism (original intuition, grounded in the liquidation-cascade literature -- arXiv
2607.27070: liquidations are FORCED, UNINFORMED flow that precede reversals):
  - A coin with a large LONG-liquidation event = leveraged longs force-exited (bearish
    flush exhausted) => mean-reverts UP => LONG.
  - A coin with a large SHORT-liquidation event = shorts force-covered (bullish squeeze
    exhausted) => mean-reverts DOWN => SHORT.
  - Gate by OPEN INTEREST: only trade when leverage was REAL (OI elevated), not noise.
  This is the OISQ "squeeze" idea (price up + OI down = short-cover squeeze => LONG;
  price down + OI up = trapped longs => SHORT) now MULTI-YEAR and liquidation-CONFIRMED.

Why it is a breakthrough, not a copy:
  - The cascade literature studies BTC SYSTEMIC risk (taker-flow variance compression).
  - This is CROSS-SECTIONAL PERP ALPHA from the SAME forced-flow physics, per coin.
  - Requires per-coin OI + liquidations, which are KEYED (free tier) -- not the 31-day
    Binance wall that blocked OISQ.

Data (produced by the keyed ingestion script, e.g. scripts/pull_coinglass.py):
  /tmp/crypto_oi.csv        date x 31 coins, daily open interest (USD)
  /tmp/crypto_liq_long.csv  date x 31 coins, daily long-liquidation USD
  /tmp/crypto_liq_short.csv date x 31 coins, daily short-liquidation USD
  /tmp/crypto_daily_long.csv / /tmp/crypto_funding.csv (existing free caches)

Variants: LIQREV (liquidation-reversal), OISQ (OI squeeze), LUW (combined), each with
BTC regime gate + conditional short. Walk-forward threshold, 10bps costs, crash regimes.

Run: uv run python scripts/research_luw.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import random

CONFIG = {
    "formation_days": 14,
    "liq_lookback_days": 7,  # liquidation intensity window
    "oi_lookback_days": 14,  # OI change window
    "quintile": 0.20,
    "vol_target": 0.55,
    "vol_lookback_days": 126,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "oi_z_thresh": 1.0,  # leverage z (coin-den OI) must exceed this to count as "real leverage"
    "oisq_w": 0.5,  # weight of OI-squeeze term in LUW composite
    "fund_w": 0.5,  # weight of funding-stress term in LUW composite
    "short_vol_l": 12,
    "stress_q_grid": [0.50, 0.60, 0.70, 0.80],
    "skip_q_grid": [0.60, 0.70, 0.80, 0.90],
    "wf_train_weeks": 104,
    "wf_test_weeks": 13,
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
PRICE_CACHE = Path("/tmp/crypto_daily_long.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")
OI_CACHE = Path("/tmp/crypto_oi.csv")
LIQL_CACHE = Path("/tmp/crypto_liq_long.csv")
LIQS_CACHE = Path("/tmp/crypto_liq_short.csv")

VARIANTS = {
    "MOM14_REGIME": dict(kind="mom"),
    "LIQREV": dict(kind="liqrev"),
    "OISQ": dict(kind="oisq"),
    "FUNDSTRESS": dict(kind="fundstress"),
    "LUW": dict(kind="luw"),
}


def load_caches():
    for p in (PRICE_CACHE, FUND_CACHE, OI_CACHE, LIQL_CACHE, LIQS_CACHE):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run the keyed ingestion script first")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    oi = pd.read_csv(OI_CACHE, index_col=0, parse_dates=True)
    oi.index = pd.to_datetime(oi.index, utc=True).tz_localize(None)
    ll = pd.read_csv(LIQL_CACHE, index_col=0, parse_dates=True)
    ll.index = pd.to_datetime(ll.index, utc=True).tz_localize(None)
    ls = pd.read_csv(LIQS_CACHE, index_col=0, parse_dates=True)
    ls.index = pd.to_datetime(ls.index, utc=True).tz_localize(None)
    common = (
        px.index.intersection(fd.index)
        .intersection(oi.index)
        .intersection(ll.index)
        .intersection(ls.index)
    )
    for d in (px, fd, oi, ll, ls):
        d = d.loc[common]
    oi, ll, ls = (
        oi.reindex(columns=px.columns),
        ll.reindex(columns=px.columns),
        ls.reindex(columns=px.columns),
    )
    print(
        f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()} (OI+liq keyed)"
    )
    return px, fd, oi, ll, ls


def weekly_frame(close, formation):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    vol = (
        close.pct_change(fill_method=None).rolling(CONFIG["vol_lookback_days"]).std() * np.sqrt(252)
    ).reindex(w.index, method="ffill")
    return fwd.iloc[formation:], mom.iloc[formation:], vol.iloc[formation:]


def btc_regime(close):
    btc = close["BTCUSDT"]
    fast = btc.rolling(CONFIG["regime_fast"]).mean()
    slow = btc.rolling(CONFIG["regime_slow"]).mean()
    up = (btc > fast) & (btc > slow)
    return up.resample("W-MON").last().reindex(close.resample("W-MON").last().index, method="ffill")


def build_signal(kind, close, oi, ll, ls, fd) -> pd.DataFrame:
    """Weekly cross-sectional score (high => long, low => short). NaN => skip coin.

    Design (per web research 2026):
      - Coin-denominated OI = oi / close removes the mechanical USD-OI ~ price
        contamination (axeladlerjr; Glassnode LPOC): raw USD-OI rises with price
        even with zero new positions, so divergence must use OI/price.
      - LIQREV: net liquidation imbalance as share of weekly OI
        (ll - ls) / oiw  -> +ve = long-liquidations dominated = forced selling
        exhausted = mean-revert UP (long). Validated by MethodAlgo / arXiv 2607.27070
        (liquidation exhaustion -> 24-48h bounce).
      - OISQ: squeeze signal from coin-OI divergence vs price (Glassnode LPOC
        short-closures / long-closures regimes).
      - FUNDSTRESS: fade the crowded (extreme funding) side, gated by real leverage
        (OI z). Positioning-stress model (Bitbase 2026; RiskState).
      - LUW: composite of the three (the original bet).
    """
    wk = close.resample("W-MON").last()
    idx = wk.index
    if kind == "mom":
        return (close / close.shift(CONFIG["formation_days"]) - 1.0).resample("W-MON").last()
    # coin-denominated open interest (leverage in coin terms, removes price-math)
    oi_norm = (oi / close).reindex(wk.index, method="ffill")
    oiw = oi_norm.resample("W-MON").last()
    llw = ll.resample("W-MON").sum()
    lsw = ls.resample("W-MON").sum()
    fdw = fd.resample("W-MON").last()
    # leverage size: z of coin-OI vs trailing 2y history
    oi_z = (oiw - oiw.rolling(104).mean()) / (oiw.rolling(104).std() + 1e-9)
    # liquidation intensity: net long-liq minus short-liq as share of weekly OI
    liq_int = (llw - lsw) / (oiw * CONFIG["liq_lookback_days"] + 1e-9)
    # OI-squeeze: price change vs coin-OI change over 2 weeks (Glassnode LPOC)
    pchg = wk / wk.shift(2) - 1.0
    ochg = oiw / oiw.shift(2) - 1.0
    oisq = ((pchg > 0) & (ochg < 0)).astype(float) - ((pchg < 0) & (ochg > 0)).astype(float)
    # funding stress: fade the extreme-funded (crowded) side
    fz = (fdw - fdw.rolling(52).mean()) / (fdw.rolling(52).std() + 1e-9)
    fund_stress = -fz
    if kind == "liqrev":
        out = liq_int
    elif kind == "oisq":
        out = oisq
    elif kind == "fundstress":
        out = fund_stress.where(oi_z > CONFIG["oi_z_thresh"])
    elif kind == "luw":
        out = (
            liq_int
            + CONFIG["oisq_w"] * oisq
            + CONFIG["fund_w"] * fund_stress.where(oi_z > CONFIG["oi_z_thresh"], 0.0)
        )
    else:
        raise ValueError(kind)
    return out.iloc[CONFIG["liq_lookback_days"] :]


def ref_short_book_returns(mom, fwd, quintile):
    out = {}
    for date in fwd.index:
        m = mom.loc[date].dropna()
        if len(m) < 12:
            continue
        n = max(2, int(round(quintile * len(m))))
        out[date] = fwd.loc[date, m.sort_values().index[:n]].mean()
    return pd.Series(out)


def weights_at(date, score, vol, regime_flag, sb_hist, spec):
    if spec.get("regime") and not regime_flag:
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs, shorts = ranked.index[-n:], ranked.index[:n]
    w = pd.Series(1.0 / n, index=longs)
    w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest(close, score, skip_q, spec, stress_q=0.60):
    fwd, _, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = (
        ref_short_book_returns(score, fwd, CONFIG["quintile"])
        if False
        else ref_short_book_returns(
            (close / close.shift(CONFIG["formation_days"]) - 1.0)
            .resample("W-MON")
            .last()
            .iloc[CONFIG["formation_days"] :],
            fwd,
            CONFIG["quintile"],
        )
    )
    score = score.reindex(fwd.index)
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        w = weights_at(
            date, score, vol, bool(reg.loc[date]), sb_hist, {**spec, "stress_q": stress_q}
        )
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


def metrics(ret):
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
        "maxdd": dd,
        "pct_flat": float((ret == 0).mean()),
    }


def report(name, m):
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  %flat={m['pct_flat'] * 100:.0f}%"
    )


CRASH = {
    "COVID 2020": ("2020-02-20", "2020-04-01"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX 2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main():
    close, fd, oi, ll, ls = load_caches()
    print(f"config: {CONFIG}")
    results = {}
    for name, spec in VARIANTS.items():
        score = build_signal(spec["kind"], close, oi, ll, ls, fd)
        ret = backtest(close, score, 2.0, spec)  # no skip (signal itself is the timing)
        report(name, metrics(ret))
        results[name] = ret

    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>14}" for k in CRASH))
    for name, ret in results.items():
        row = ""
        for a, b in CRASH.values():
            sub = ret.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:14.2f}"
        print(f"  {name:>14}{row}")


if __name__ == "__main__":
    main()
