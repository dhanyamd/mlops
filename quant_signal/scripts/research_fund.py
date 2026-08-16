"""Research: LCF -- Leverage-Crowding Factor (crypto-native, from the literature).

Method -- read the money-making papers, then build ONE mechanism (not a stack):

  Liu-Tsyvinski (NBER w24877): investor attention & momentum predict crypto returns;
     leverage-crowded trends persist.
  Christin et al. "Crypto Carry Trade" / BIS WP1087 / SSRN 3774118: the perpetual
     FUNDING RATE is crypto's native microstructure signal. "Carry in perpetual swaps
     POSITIVELY predicts returns in a cross-section of 51 cryptocurrencies" -- i.e.
     coins where leveraged longs pay the most (crowded-long, trend-chasing) earn the
     HIGHEST future returns. The crypto carry trade (short perp / collect funding) has
     in-sample Sharpe 7-10.
  BIS WP1087 (the gap): a HIGH carry predicts FUTURE CRASHES and forced sell-
     liquidations. So the crowding edge is real but FRAGILE at the extremes.

  => LCF mechanism: cross-sectionally LONG the highest-funding (crowded-long) coins,
     SHORT the lowest/negative-funding (under-owned) coins -- the documented crypto
     carry direction -- BUT cap exposure when funding is extreme (fragility cap), to
     avoid the cascade BIS documents. One signal, one economic story, crypto-native.

Baselines: MOM (standard XS momentum, Sharpe ~0.67). Costs + weekly walk-forward +
crash regimes. Pro -vs- contrarian direction both tested; fragility cap walk-forward
selected from a grid (no hardcoded magic numbers).

Run: uv run python scripts/research_fund.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "fund_ma_weeks": 4,
    "quintile": 0.20,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "lookback_grid": [2, 3, 4],
    "cap_grid": [None, 0.5, 1.0, 2.0],  # annualized funding cap (frac); None = no cap
    "wf_train_weeks": 104,
    "wf_test_weeks": 13,
    "use_regime": False,
}
CLOSE_CACHE = Path("/tmp/crypto_daily_c.csv")
FUND_CACHE = Path("/tmp/crypto_funding_ann.csv")  # daily summed funding * 365


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.read_csv(CLOSE_CACHE, index_col=0, parse_dates=True)
    close.index = pd.to_datetime(close.index, utc=True).tz_localize(None)
    fund = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fund.index = pd.to_datetime(fund.index, utc=True).tz_localize(None)
    fund = fund.dropna(axis=1, how="all")  # drop coins with no funding history (PEPE)
    common = close.columns.intersection(fund.columns)
    close, fund = close[common], fund[common]
    print(
        f"[data] close {close.shape}  fund {fund.shape}  "
        f"{close.index.min().date()}..{close.index.max().date()}"
    )
    return close, fund


def weekly_returns(close: pd.DataFrame) -> pd.DataFrame:
    w = close.resample("W-MON").last()
    return (w.shift(-1) / w - 1.0).iloc[CONFIG["fund_ma_weeks"] :]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    up = (btc > btc.rolling(CONFIG["regime_fast"]).mean()) & (
        btc > btc.rolling(CONFIG["regime_slow"]).mean()
    )
    return up.resample("W-MON").last()


def lcf_score(fund: pd.DataFrame) -> pd.DataFrame:
    """Trailing annualized funding, resampled weekly (last of week), smoothed."""
    af = fund.resample("W-MON").last()
    return af.rolling(CONFIG["fund_ma_weeks"]).mean().iloc[CONFIG["fund_ma_weeks"] :]


def weights_from_score(score, date, regime_flag, direction, cap, panic_only=False):
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    if CONFIG["use_regime"] and not regime_flag:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs = ranked.index[-n:] if direction == "pro" else ranked.index[:n]
    shorts = ranked.index[:n] if direction == "pro" else ranked.index[-n:]
    w = pd.Series(1.0 / n, index=longs)
    if not panic_only:
        w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    if cap is not None:
        # fragility cap: zero out positions on coins with |annualized funding| > cap
        ann = m.reindex(w.index)
        w = w.where(ann.abs() <= cap, 0.0)
        w = w[w != 0]
        if len(w) < 2:
            return None
        w = w / w.abs().sum() * np.sign(w)  # renormalize to unit gross (long/short balanced)
    return w


def backtest(score, fwd, regime, direction, cap) -> pd.Series:
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        w = weights_from_score(score, date, bool(regime.loc[date]), direction, cap)
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


def walk_forward_params(score, fwd, regime, direction) -> tuple[list, list]:
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    lb_seq, cap_seq = [], []
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        best = (CONFIG["lookback_grid"][0], CONFIG["cap_grid"][0], -1e9)
        for lb in CONFIG["lookback_grid"]:
            sc = score.rolling(lb).sum().shift(0)
            for cap in CONFIG["cap_grid"]:
                s = backtest(sc, fwd, regime, direction, cap).iloc[start:end]
                sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
                if sr > best[2]:
                    best = (lb, cap, sr)
        for _ in range(end, min(end + te, n)):
            lb_seq.append(best[0])
            cap_seq.append(best[1])
    while len(lb_seq) < n:
        lb_seq.append(lb_seq[-1] if lb_seq else CONFIG["lookback_grid"][1])
        cap_seq.append(cap_seq[-1] if cap_seq else CONFIG["cap_grid"][1])
    return lb_seq[:n], cap_seq[:n]


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


def vol_scaled(
    ret: pd.Series, target: float = 0.40, window: int = 12, max_lev: float = 2.0
) -> pd.Series:
    """Moreira-Muir / Barroso-Santa-Clara volatility targeting: scale gross exposure by
    inverse trailing realized vol (documented crypto-momentum crash remedy)."""
    vol = ret.rolling(window).std() * np.sqrt(52)
    lev = (target / vol).clip(upper=max_lev).fillna(0.0)
    return ret * lev.shift(1).fillna(0.0)


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
    "2021 bull": ("2021-01-01", "2021-12-31"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX Nov-22": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close, fund = load()
    fwd = weekly_returns(close).iloc[10:]  # align to momentum's 14wk warmup (4 already dropped)
    score = lcf_score(fund)
    fwd = fwd.reindex(score.index)  # restrict both to the funding-available window (2020-09+)
    regime = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)

    report("MOM (standard momentum L/S, baseline)", metrics(momentum_backtest(close, fwd)))

    for direction in ("pro", "contra"):
        tag = (
            "LCF (long high-funding)"
            if direction == "pro"
            else "LCF (short high-funding / contrarian)"
        )
        lb_seq, cap_seq = walk_forward_params(score, fwd, regime, direction)
        chosen = sorted(
            {(int(l), (round(c, 2) if c is not None else None)) for l, c in zip(lb_seq, cap_seq)},
            key=lambda x: (x[0], (x[1] if x[1] is not None else float("inf"))),
        )
        print(f"\n[walk-forward {tag}] (lookback, cap) selected={chosen}")
        # rebuild per-week weights using WF-selected (lb, cap)
        dates = list(fwd.index)
        ret, prev = [], pd.Series(dtype=float)
        for i, date in enumerate(dates):
            sc = score.rolling(lb_seq[i]).sum()
            w = weights_from_score(sc, date, bool(regime.loc[date]), direction, cap_seq[i])
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
        lcf = pd.Series(ret, index=dates)
        report(f"{tag} (WF lb+cap, costs)", metrics(lcf))

        # Volatility-scaled (Moreira-Muir) hardening -- the documented crypto crash remedy
        lcf_vs = vol_scaled(lcf)
        report(f"{tag} + VOL-SCALE (target 40%/yr, cap 2x)", metrics(lcf_vs))

        print("  --- crash-regime annualized Sharpe (raw | vol-scaled) ---")
        print("  " + "".join(f"{k:>15}" for k in CRASH))
        row, rowv = "", ""
        for a, b in CRASH.values():
            sub = lcf.loc[a:b]
            subv = lcf_vs.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            srv = subv.mean() / (subv.std() + 1e-9) * np.sqrt(52) if len(subv) > 2 else float("nan")
            row += f"{sr:15.2f}"
            rowv += f"{srv:15.2f}"
        print(f"  {'LCF':>13}{row}")
        print(f"  {'LCF+VS':>13}{rowv}")


if __name__ == "__main__":
    main()
