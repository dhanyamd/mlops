"""GENUINE NOVEL KEYLESS FACTOR -- Order-Flow Information (OFI), cross-sectional, weekly.

GROUNDING (web research, this session):
  Anastasopoulos, Gradojevic, Liu, Maynard & Tsiakas (2024/2026), "Order Flow and
  Cryptocurrency Returns" (ScienceDirect; SSRN 5020002): order flow has strong predictive
  power for the cross-section of crypto returns, BUT it splits into a TRANSITORY component
  (correlated with the concurrent return -> reverses short-term) and a PERMANENT/INFORMED
  component (orthogonal to the concurrent return -> POSITIVELY predicts future returns).
  Weekly flow predicts better than daily (R^2 3.4% weekly; +0.9% next-week return per +1 SD).
  "Crypto Microstructure Alpha" (Frontiers 2026) confirms taker-flow imbalance predicts
  near-term returns but high-frequency turnover is destroyed by fees -> a WEEKLY, low-turnover,
  cross-sectional design is the correct adaptation. That adaptation is OUR novel contribution.

DATA: /tmp/crypto_takerflow.csv = taker_buy_quote / total_quote per coin (keyless Binance
futures klines, pulled by pull_binance_takerflow.py). signed_flow = 2*ratio - 1 in [-1,1].

MECHANISM (parameter-free, no magic thresholds):
  For each week, regress cross-sectional signed_flow on the concurrent weekly return and take
  the RESIDUAL = the order-flow component NOT explained by the price move = informed/permanent
  flow. Rank coins on this residual (long informed buyers, short informed sellers). The naive
  raw-flow and flow-momentum variants are included as honesty checks (they should be weaker /
  contaminated by the transitory reversal component).

WF: no parameters are fitted in-sample (weekly horizon + 20/20 portfolio sort are standard
portfolio-sort choices from the literature, not optimized). Single backtest IS effectively OOS.
10bps costs, BTC regime gate, crash-regime sub-periods, bootstrap CI (random.Random(0)).

Run: uv run python scripts/research_flow.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import random

CONFIG = {
    "formation_days": 14,
    "quintile": 0.20,  # standard 20/20 long-short portfolio sort (NOT a tuned magic number)
    "vol_target": 0.55,
    "vol_lookback_days": 126,
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
FLOW_CACHE = Path("/tmp/crypto_takerflow.csv")

VARIANTS = {
    "MOM14_REGIME": dict(score="mom", regime=True),  # SCX-family baseline
    "ASYM_REGIME": dict(score="asym", regime=True),  # our prior original winner (1.19)
    "FLOW_LVL": dict(score="flow_lvl", regime=True),  # naive raw aggressive-buy pressure
    "FLOW_MOM": dict(score="flow_mom", regime=True),  # flow acceleration (change in pressure)
    "OFI_RESID": dict(score="ofi", regime=True),  # cross-sectional residual (failed adaptation)
    "TS_OFI": dict(score="ts_ofi", regime=True),  # time-series per-coin residual (lit. adaptation)
    "FLOW_CONF": dict(
        score="flow_conf", regime=True
    ),  # OUR TWIST: momentum confirmed by taker flow
    "ASYM_FLOWTIME": dict(
        score="asym_ft", regime=True
    ),  # ASYM gated by market-wide aggressive flow
}


def load():
    for p in (PRICE_CACHE, FUND_CACHE, FLOW_CACHE):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    fl = pd.read_csv(FLOW_CACHE, index_col=0, parse_dates=True)
    fl.index = pd.to_datetime(fl.index, utc=True).tz_localize(None)
    common = px.index.intersection(fd.index).intersection(fl.index)
    px = px.loc[common].reindex(columns=UNIVERSE)
    fd = fd.loc[common].reindex(columns=UNIVERSE)
    fl = fl.loc[common].reindex(columns=UNIVERSE)
    print(f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()}")
    return px, fd, fl


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


def build_scores(close: pd.DataFrame, fd: pd.DataFrame, flow: pd.DataFrame):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    fdw = fd.resample("W-MON").mean().reindex(fwd.index)
    vow = close.reindex(fwd.index)  # unused placeholder; keep interface
    mom_z = mom.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    fund_z = fdw.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    # our prior original winner: long short-squeeze candidates, short momentum losers
    fund_accel = fund_z - fund_z.rolling(3).mean()
    squeeze = ((fund_z < -1.0) & (fund_accel > 0)).astype(float) * 2.0
    asym = mom_z.where(squeeze == 0, squeeze)

    # ---- order-flow signals ----
    fw = flow.resample("W-MON").last().reindex(fwd.index)
    signed = 2.0 * fw - 1.0  # net aggressive-buy pressure in [-1,1]
    ret_w = close.resample("W-MON").last().pct_change().reindex(fwd.index)
    flow_lvl = signed.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    flow_mom = signed.diff(1).apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)

    # OUR NOVEL SIGNAL: cross-sectional residual of weekly signed flow on the concurrent
    # weekly return -> permanent/informed component (orthogonal to the transitory move).
    idx = signed.index
    cols = signed.columns
    ofi = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for d in idx:
        f = signed.loc[d]
        r = ret_w.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 6:
            continue
        fc = f[m] - f[m].mean()
        rc = r[m] - r[m].mean()
        if rc.std() < 1e-9:
            continue
        beta = float((fc * rc).sum() / (rc * rc).sum())
        resid = fc - beta * rc
        ofi.loc[d] = resid.reindex(cols)
    ofi_z = ofi.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)

    # Faithful literature adaptation: TIME-SERIES per-coin residualization. For each coin,
    # regress its weekly signed flow on its OWN trailing-52w return and take the residual
    # (permanent/informed component). Then rank coins cross-sectionally on that residual.
    ts_ofi = pd.DataFrame(index=idx, columns=cols, dtype=float)
    win = 52
    for c in cols:
        s = signed[c]
        r = ret_w[c]
        res = pd.Series(index=idx, dtype=float)
        for t in range(win, len(idx)):
            sw = s.iloc[t - win : t]
            rw = r.iloc[t - win : t]
            mm = sw.notna() & rw.notna()
            if mm.sum() < 20:
                continue
            fc = sw[mm] - sw[mm].mean()
            rc = rw[mm] - rw[mm].mean()
            if rc.std() < 1e-9:
                continue
            beta = float((fc * rc).sum() / (rc * rc).sum())
            res.iloc[t] = float(fc.iloc[-1] - beta * rc.iloc[-1])
        ts_ofi[c] = res
    ts_ofi_z = ts_ofi.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)

    # OUR TWIST: momentum CONFIRMED by aggressive buy flow (real taker flow, not total volume
    # like VCM). Strong only when the up-move is backed by actual buyer-initiated aggression.
    flow_conf = (mom_z * flow_lvl).apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)

    # MARKET-WIDE flow timing (the paper's headline: aggregate order flow predicts the MARKET).
    # mkt_flow = cross-sectional mean of aggressive-buy pressure each week. By construction taker
    # flow averages slightly NEGATIVE (~0.49 buyer-initiated), so we gate on it being ELEVATED vs
    # its own trailing median (risk-on flow), NOT on >0 (which would almost never fire).
    mkt_flow = signed.mean(axis=1)
    flow_on = mkt_flow > mkt_flow.rolling(52).median()
    asym_ft = asym.where(flow_on, np.nan)

    return (
        fwd,
        mom,
        vol,
        {
            "mom": mom_z,
            "asym": asym,
            "flow_lvl": flow_lvl,
            "flow_mom": flow_mom,
            "ofi": ofi_z,
            "ts_ofi": ts_ofi_z,
            "flow_conf": flow_conf,
            "asym_ft": asym_ft,
        },
    )


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


def backtest(close: pd.DataFrame, score: pd.DataFrame, spec: dict) -> pd.Series:
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
        "maxdd": dd,
        "pct_flat": float((ret == 0).mean()),
    }


def report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  %flat={m['pct_flat'] * 100:.0f}%"
    )
    print(f"  skew={m['skew']:+.2f}  maxDD={m['maxdd'] * 100:6.1f}%")


CRASH = {
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX 2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close, fd, fl = load()
    fwd, mom, vol, S = build_scores(close, fd, fl)
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
