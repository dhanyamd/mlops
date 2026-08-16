 """CORRECTED order-flow factor (Anastasopoulos et al. 2026, J. Financial Markets) + VPIN tail overlay.

WHY THE PRIOR FLOW TESTS FAILED:
  OFI_RESID regressed signed flow on the CONCURRENT weekly return -- that captures the
  TRANSITORY price-pressure component (which reverses), not the PERMANENT/informed flow.
  TS_OFI residualized each coin on its OWN 52w return (too noisy, weak). Both gave ~0.2-0.3.
  The paper's genuine signal: the component of order flow ORTHOGONAL to the LAGGED return
  -- the informed/permanent flow -- POSITIVELY predicts future returns. Weekly LS Sharpe 1.93,
  alpha 1.72%/wk vs the crypto 3-factor model.

FIXES (all keyless, taker-flow cache):
  OF_PERM  : CROSS-SECTIONAL sort on per-coin signed flow residualized on the LAGGED weekly
             return (pooled per week) = permanent/informed flow. Long informed-buy, short
             informed-sell.
  OF_WORLD : AGGREGATE (market) flow residualized on market LAGGED return -> the paper's
             headline market-timing signal. Used as a trade gate on ASYM.
  VPIN     : order-flow TOXICITY = |signed flow|; rolling mean. De-risk ASYM when toxicity
             spikes (Easley-O'Hara-Yang-Zhang 2024; Kitvanitphasu 2026: VPIN precedes jumps).
             Cuts the crash tail WITHOUT levering up calm periods (the vol-scaling mistake).

Run: uv run python scripts/research_flow2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import random

CONFIG = {
    "formation_days": 14,
    "quintile": 0.20,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "gate_win": 52,  # adaptive median window for flow/toxicity gates
    "vpin_win": 12,  # rolling toxicity window (weeks)
}

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "TRXUSDT",
    "LINKUSDT", "NEARUSDT", "ADAUSDT", "SUIUSDT", "UNIUSDT", "AVAXUSDT", "CRVUSDT",
    "LTCUSDT", "ICPUSDT", "AAVEUSDT", "XLMUSDT", "HBARUSDT", "DOTUSDT", "FILUSDT",
    "ARBUSDT", "LDOUSDT", "BCHUSDT", "OPUSDT", "ATOMUSDT", "ETCUSDT", "RUNEUSDT",
    "GRTUSDT", "ZECUSDT",
]
PRICE_CACHE = Path("/tmp/crypto_daily_long.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")
FLOW_CACHE = Path("/tmp/crypto_takerflow.csv")


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


def zs(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)


def build_scores(close: pd.DataFrame, fd: pd.DataFrame, flow: pd.DataFrame):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    fdw = fd.resample("W-MON").mean().reindex(fwd.index)
    mom_z = zs(mom)
    fund_z = zs(fdw)
    fund_accel = fund_z - fund_z.rolling(3).mean()
    squeeze = ((fund_z < -1.0) & (fund_accel > 0)).astype(float) * 2.0
    asym = mom_z.where(squeeze == 0, squeeze)

    # ---- order flow (taker-buy / total quote) -> signed aggressive-buy pressure ----
    signed = (2.0 * flow.resample("W-MON").last().reindex(fwd.index) - 1.0)
    ret_w = close.resample("W-MON").last().pct_change().reindex(fwd.index)
    ret_w_lag = ret_w.shift(1)

    # CORRECTED PERMANENT COMPONENT: residualize per-coin signed flow on the LAGGED return
    # (pooled across coins each week). This strips the transitory reversal, leaving informed flow.
    idx, cols = signed.index, signed.columns
    of_perm = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for d in idx:
        s = signed.loc[d]
        r = ret_w_lag.loc[d]
        m = s.notna() & r.notna()
        if m.sum() < 6:
            continue
        sc = s[m] - s[m].mean()
        rc = r[m] - r[m].mean()
        if rc.std() < 1e-9:
            continue
        beta = float((sc * rc).sum() / (rc * rc).sum())
        of_perm.loc[d] = (sc - beta * rc).reindex(cols)
    of_perm_z = zs(of_perm)

    # AGGREGATE / WORLD flow timing signal (paper headline): market flow residualized on the
    # market LAGGED return -> a single time-series; trade when permanent flow is elevated.
    market_signed = signed.mean(axis=1)
    market_ret = ret_w.mean(axis=1)
    mr_lag = market_ret.shift(1)
    mm = market_signed.notna() & mr_lag.notna()
    sc = market_signed[mm] - market_signed[mm].mean()
    rc = mr_lag[mm] - mr_lag[mm].mean()
    beta = float((sc * rc).sum() / (rc * rc).sum())
    of_world = market_signed - (market_signed.mean() + beta * (mr_lag - mr_lag.mean()))
    of_world_gate = (of_world > of_world.rolling(CONFIG["gate_win"]).median()).reindex(fwd.index)

    # VPIN toxicity overlay: |signed flow| rolling mean; de-risk when toxicity is elevated.
    tox = signed.abs().mean(axis=1)
    vpin = tox.rolling(CONFIG["vpin_win"]).mean()
    vpin_gate = (vpin <= vpin.rolling(CONFIG["gate_win"]).median()).reindex(fwd.index)

    scores = {
        "mom": mom_z,
        "asym": asym,
        "of_perm": of_perm_z,
    }
    gates = {
        "of_world": of_world_gate.fillna(False),
        "vpin": vpin_gate.fillna(True),
    }
    return fwd, mom, vol, scores, gates


def weights_at(date, score, regime_flag, gate_on, spec: dict):
    if spec.get("regime") and not regime_flag:
        return None
    if gate_on is not None and not bool(gate_on):
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    n = max(2, int(round(CONFIG["quintile"] * len(m))))
    ranked = m.sort_values()
    longs = ranked.index[-n:]
    shorts = ranked.index[:n]
    return pd.concat([pd.Series(1.0 / n, index=longs), pd.Series(-1.0 / n, index=shorts)])


def backtest(close: pd.DataFrame, score: pd.DataFrame, spec: dict, gate: pd.Series | None = None) -> pd.Series:
    fwd, _, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        gate_on = None if gate is None else bool(gate.loc[date]) if date in gate.index else None
        w = weights_at(date, score, bool(reg.loc[date]), gate_on, spec)
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
        "n": n, "ann_ret": ann, "ann_vol": vol, "sharpe": sharpe, "ci": ci,
        "skew": float(ret.skew()), "exkurt": float(ret.kurt()),
        "maxdd": dd, "pct_flat": float((ret == 0).mean()),
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
    close, fd, fl = load()
    fwd, mom, vol, S, G = build_scores(close, fd, fl)
    specs = {
        "MOM14_REGIME": dict(score="mom", regime=True),
        "ASYM_REGIME": dict(score="asym", regime=True),
        "OF_PERM_REGIME": dict(score="of_perm", regime=True),
        "ASYM+OF_WORLD": dict(score="asym", regime=True),
        "ASYM+VPIN": dict(score="asym", regime=True),
    }
    gates = {
        "ASYM+OF_WORLD": G["of_world"],
        "ASYM+VPIN": G["vpin"],
    }
    results = {}
    for name, spec in specs.items():
        g = gates.get(name)
        ret = backtest(close, S[spec["score"]], spec, gate=g)
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
