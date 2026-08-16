"""Research: CSCM -- Cross-Sectional Carry + Momentum composite (KEYLESS).

Two free-data edges, each validated in the literature, combined into one
market-neutral cross-sectional book:

  1. 14d cross-sectional MOMENTUM, BTC UP-UP regime-gated, SCX-style
     conditional-short crash protection.  (SCX family; WF Sharpe ~1.07-1.13)
  2. Cross-sectional FUNDING CARRY: short the names paying the richest
     funding (crowded longs), long the names paying negative funding.
     [Keel funding-carry Sharpe 1.69 net of costs; Unravel "Foundational"
     Momentum+Carry composite Sharpe 2+; Chi et al. 2023: basis/funding is
     the strongest XS predictor in crypto futures.]

Composite ranking score (PRE-SPECIFIED, equal weight, no in-sample fit):
    score = z(14d momentum) - z(trailing funding)
    High => winner AND cheap-to-hold (low funding) => LONG.
    Low  => loser AND crowded-long (high funding)  => SHORT.

KEYLESS: uses only /tmp/crypto_daily_long.csv + /tmp/crypto_funding.csv.
BTC UP-UP regime gate + vol targeting + SCX conditional-short crash protection.
WF OOS stress quantile, 10bps costs, crash-regime sub-periods, bootstrap CI.

Run: uv run python scripts/research_cscm.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import random

CONFIG = {
    "formation_days": 14,  # momentum lookback (daily)
    "fund_lookback_weeks": 4,  # trailing funding window for the carry z
    "quintile": 0.20,  # long/short book size
    "vol_target": 0.55,  # BTC target annualized vol
    "vol_lookback_days": 126,  # realized-vol estimate window
    "cost_bps": 10.0,  # per-side taker+slippage per weekly rebalance
    "regime_fast": 90,  # BTC UP-UP fast MA (daily)
    "regime_slow": 200,  # BTC UP-UP slow MA (daily)
    "short_vol_l": 12,  # trailing short-book vol window (weeks)
    "stress_q": 0.70,  # conditional-short stress quantile (pre-spec)
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

# score: "mom"=pure momentum, "carry"=funding carry, "comp"=composite
VARIANTS = {
    "MOM14_REGIME": dict(score="mom", regime=True, shorts=True, vol_scale=False, cond_short=False),
    "CARRY_REGIME": dict(
        score="carry", regime=True, shorts=True, vol_scale=False, cond_short=False
    ),
    "CSCM": dict(score="comp", regime=False, shorts=True, vol_scale=False, cond_short=False),
    "CSCM_REGIME": dict(score="comp", regime=True, shorts=True, vol_scale=False, cond_short=False),
    "CSCM_VOL": dict(score="comp", regime=True, shorts=True, vol_scale=True, cond_short=False),
    "CSCM_SCX": dict(score="comp", regime=True, shorts=True, vol_scale=False, cond_short=True),
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in (PRICE_CACHE, FUND_CACHE):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    common = px.index.intersection(fd.index)
    px, fd = px.loc[common].reindex(columns=UNIVERSE), fd.loc[common].reindex(columns=UNIVERSE)
    print(f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()}")
    return px, fd


def weekly_frame(close: pd.DataFrame, formation: int):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    vol = (
        close.pct_change(fill_method=None).rolling(CONFIG["vol_lookback_days"]).std() * np.sqrt(252)
    ).reindex(w.index, method="ffill")
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
    carry = -fund_z  # short high-funding, long low-funding
    comp = mom_z - fund_z  # equal-weight composite (pre-specified)
    return fwd, mom, vol, {"mom": mom_z, "carry": carry, "comp": comp}


def ref_short_book_returns(mom: pd.DataFrame, fwd: pd.DataFrame, quintile: float) -> pd.Series:
    out = {}
    for date in fwd.index:
        m = mom.loc[date].dropna()
        if len(m) < 12:
            continue
        n = max(2, int(round(quintile * len(m))))
        out[date] = fwd.loc[date, m.sort_values().index[:n]].mean()
    return pd.Series(out)


def weights_at(date, score, vol, regime_flag, sb_hist, spec: dict):
    if spec["regime"] and not regime_flag:
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs = ranked.index[-n:]
    shorts = ranked.index[:n]
    shorts_on = spec["shorts"]
    if spec["cond_short"] and shorts_on and len(sb_hist) >= CONFIG["short_vol_l"]:
        sbv = pd.Series(sb_hist).rolling(CONFIG["short_vol_l"]).std().dropna()
        if len(sbv) >= 4 and sbv.iloc[-1] > sbv.quantile(CONFIG["stress_q"]):
            shorts_on = False
    w = pd.Series(1.0 / n, index=longs)
    if shorts_on:
        w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    if spec["vol_scale"]:
        rv = vol.loc[date].reindex(w.index).dropna()
        if rv.empty or rv.mean() <= 0:
            return None
        w = w * float(np.clip(CONFIG["vol_target"] / rv.mean(), 0.0, 3.0))
    return w


def backtest(close: pd.DataFrame, score: pd.DataFrame, spec: dict) -> pd.Series:
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(mom, fwd, CONFIG["quintile"])
    dates = list(fwd.index)
    ret, prev, sb_hist = [], pd.Series(dtype=float), {}
    for date in dates:
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        w = weights_at(date, score, vol, bool(reg.loc[date]), sb_hist, spec)
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
        "cvar05": ret.quantile(0.05),
        "pct_flat": float((ret == 0).mean()),
    }


def report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  %flat={m['pct_flat'] * 100:.0f}%"
    )
    print(
        f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%"
        f"  CVaR5%={m['cvar05'] * 100:6.2f}%"
    )


CRASH = {
    "2018 bear": ("2018-01-01", "2018-12-31"),
    "COVID 2020": ("2020-02-20", "2020-04-01"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX 2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close, fd = load()
    print(f"config: {CONFIG}")
    fwd, mom, vol, S = build_scores(close, fd)
    results = {}
    for name, spec in VARIANTS.items():
        ret = backtest(close, S[spec["score"]], spec)
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
        print(f"  {name:>13}{row}")


if __name__ == "__main__":
    main()
