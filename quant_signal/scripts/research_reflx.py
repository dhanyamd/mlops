"""Research: REFLX -- Funding-Reflexivity (unified-leverage fragility) momentum timing.

A NOVEL factor derived from first principles (NOT a literature reproduction):

  Crypto perps are a 24/7 leveraged casino. Price is public but POSITIONING is hidden
  -- except it leaks through the funding rate, the only free, exchange-published,
  cross-sectional, multi-year positioning signal. Existing factors use funding two ways:
    - LCF uses funding LEVEL (carry) -- a known factor.
    - CoMOM (Lou & Polk 2022) uses RETURN COMOVEMENT -- an equity import.
  Both miss the TOPOLOGY.

  Original mechanism -- reflexivity / synchronized-unwind fragility:
    F_t = cross-sectional MEAN funding  => aggregate leverage DIRECTION of the casino.
    D_t = cross-sectional DISPERSION (std) of funding => how unified vs divergent the crowd is.
    FRAGILITY = |z(F_t)| - z(D_t).
      HIGH fragility = funding extreme in ONE direction AND dispersion COLLAPSED =>
        the whole cross-section is leveraged the SAME way (a "one-way market") => the
        exact fragile state where a tiny adverse move triggers SYNCHRONIZED liquidations
        => a cross-sectional momentum CRASH (the risk SCX defends heuristically).
      LOW fragility = funding extreme but DIVERGENT => no synchronized unwind => momentum survives.
  We TIME the momentum book: SKIP/flatten when fragility is high. This is the
  principled, positioning-based version of SCX's volatility-proxy conditional-short.

  Walk-forward selects the fragility SKIP threshold out-of-sample. Costs 10bps/side,
  BTC UP-UP regime gate + conditional-short mirror SCX for a fair comparison.

Run: uv run python scripts/research_reflx.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "formation_days": 14,
    "quintile": 0.20,
    "vol_target": 0.55,
    "vol_lookback_days": 126,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "z_window_w": 104,  # trailing weeks to z-score F_t / D_t (slow state vars)
    "short_vol_l": 12,
    "stress_q_grid": [0.50, 0.60, 0.70, 0.80],
    "skip_q_grid": [0.60, 0.70, 0.80, 0.90, 1.10, 1.30],  # fragility skip thresholds
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

VARIANTS = {
    "MOM14": dict(regime=False, shorts=True, cond_short=False),
    "MOM14_REGIME": dict(regime=True, shorts=True, cond_short=False),
    "REFLX_SKIP": dict(regime=True, shorts=True, cond_short=False),
    "REFLX_LONGONLY": dict(regime=True, shorts=True, cond_short=False, long_only=True),
    "REFLX_SCX": dict(regime=True, shorts=True, cond_short=True),
}


def load_panel():
    if not PRICE_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_daily.py first")
    if not FUND_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_funding.py first")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    common = px.index.intersection(fd.index)
    px, fd = px.loc[common], fd.loc[common]
    fd = fd.reindex(columns=px.columns)
    print(f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()}")
    return px, fd


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


def fragility_series(fund) -> pd.Series:
    """Original reflexivity state variable: |z(mean funding)| - z(dispersion of funding)."""
    fw = fund.resample("W-MON").last()
    F = fw.mean(axis=1)  # aggregate leverage direction
    D = fw.std(axis=1)  # cross-sectional dispersion (crowd unity)
    zF = (F - F.rolling(CONFIG["z_window_w"]).mean()) / F.rolling(CONFIG["z_window_w"]).std()
    zD = (D - D.rolling(CONFIG["z_window_w"]).mean()) / D.rolling(CONFIG["z_window_w"]).std()
    frag = zF.abs() - zD
    print(
        f"[reflx] fragility computed {frag.shape[0]} weeks; "
        f"mean={frag.mean():.2f} max={frag.max():.2f} min={frag.min():.2f}"
    )
    return frag


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
    if spec["regime"] and not regime_flag:
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs, shorts = ranked.index[-n:], ranked.index[:n]
    shorts_on = spec["shorts"] and not spec.get("long_only", False)
    if spec["cond_short"] and shorts_on and len(sb_hist) >= CONFIG["short_vol_l"]:
        sbv = pd.Series(sb_hist).rolling(CONFIG["short_vol_l"]).std().dropna()
        if len(sbv) >= 4 and sbv.iloc[-1] > sbv.quantile(spec.get("stress_q", 0.60)):
            shorts_on = False
    w = pd.Series(1.0 / n, index=longs)
    if shorts_on:
        w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest(close, frag, skip_q, spec, stress_q=0.60):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(mom, fwd, CONFIG["quintile"])
    frag = frag.reindex(fwd.index)
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        fv = frag.get(date, np.nan)
        if not np.isnan(fv) and fv >= skip_q:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        w = weights_at(date, mom, vol, bool(reg.loc[date]), sb_hist, {**spec, "stress_q": stress_q})
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


def walk_forward_q(close, frag, spec):
    fwd, _, _ = weekly_frame(close, CONFIG["formation_days"])
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    q_seq, best = [], CONFIG["skip_q_grid"][1]
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        bq, bs = CONFIG["skip_q_grid"][1], -1e9
        for q in CONFIG["skip_q_grid"]:
            s = backtest(close, frag, q, spec).iloc[start:end]
            sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
            if sr > bs:
                bs, bq = sr, q
        for _ in range(end, min(end + te, n)):
            q_seq.append(bq)
    while len(q_seq) < n:
        q_seq.append(best)
    return q_seq[:n]


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
    close, fund = load_panel()
    print(f"config: {CONFIG}")
    frag = fragility_series(fund)

    results = {}
    for name, spec in VARIANTS.items():
        if name in ("MOM14", "MOM14_REGIME"):
            ret = backtest(close, frag, 1e9, spec)
            report(name, metrics(ret))
            results[name] = ret
            continue
        q_seq = walk_forward_q(close, frag, spec)
        chosen = sorted({round(x, 2) for x in q_seq})
        # rebuild with WF-selected per-block skip threshold
        fwd, _, _ = weekly_frame(close, CONFIG["formation_days"])
        dates = list(fwd.index)
        reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
        mom_full = (
            (close / close.shift(CONFIG["formation_days"]) - 1.0)
            .resample("W-MON")
            .last()
            .iloc[CONFIG["formation_days"] :]
        )
        sb = ref_short_book_returns(mom_full, fwd, CONFIG["quintile"])
        ret, prev = [], pd.Series(dtype=float)
        for i, date in enumerate(dates):
            fv = frag.get(date, np.nan)
            if not np.isnan(fv) and fv >= q_seq[i]:
                ret.append(0.0)
                prev = pd.Series(dtype=float)
                continue
            sb_hist = {d: sb[d] for d in sb.index if d < date}
            w = weights_at(
                date, mom_full, _, bool(reg.loc[date]), sb_hist, {**spec, "stress_q": 0.60}
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
        ret = pd.Series(ret, index=dates)
        print(f"\n[walk-forward {name}] WF skip_q selected={chosen}")
        report(f"{name} (WF skip_q, costs)", metrics(ret))
        results[name] = ret

    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>14}" for k in CRASH))
    for name, ret in results.items():
        row = ""
        for a, b in CRASH.values():
            sub = ret.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:14.2f}"
        print(f"  {name:>16}{row}")


if __name__ == "__main__":
    main()
