"""Research: SCX -- Skew-Convex, Regime-Gated Cross-Sectional Momentum.

Novel contribution (not a reproduction of published work):
  Crypto XS momentum's excess return concentrates in the LONG book's positive
  skew; the SHORT book is tail-risk + funding drag. SCX therefore:
    1. gates the whole book by a BTC UP-UP regime filter (flat in bears),
    2. always runs the long (winner) book in bull regime,
    3. makes SHORT exposure CONDITIONAL on the short side being "calm":
       when trailing short-book realized vol is in its stressed quantile,
       shorts are dropped and the book goes long-only. This is a dynamic,
       skew-aware net-exposure overlay -- not static vol-scaling.

Everything is parameterized in CONFIG / VARIANTS. Transaction costs are a knob.
Walk-forward selects the conditional-short stress quantile out-of-sample.
Crash-regime sub-periods are reported explicitly.

Run: uv run python scripts/research_scx.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── parameters (NOT hardened into the logic) ───────────────────────────────────
CONFIG = {
    "formation_days": 14,  # momentum lookback (daily)
    "quintile": 0.20,  # long/short book size
    "vol_target": 0.55,  # BSC target annualized vol
    "vol_lookback_days": 126,  # realized-vol estimate window
    "cost_bps": 10.0,  # per-side taker+slippage per weekly rebalance
    "regime_fast": 90,  # BTC UP-UP fast MA (daily)
    "regime_slow": 200,  # BTC UP-UP slow MA (daily)
    "short_vol_l": 12,  # trailing short-book vol window (weeks)
    "stress_q_grid": [0.50, 0.60, 0.70, 0.80],  # candidate conditional-short thresholds
    "wf_train_weeks": 104,  # 2y rolling calibration
    "wf_test_weeks": 13,  # 3m OOS block
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
LONG_CACHE = Path("/tmp/crypto_daily_long.csv")

# Variant specs: regime gate, shorts on/off, vol-scaling, conditional short.
VARIANTS = {
    "NAIVE_LS": dict(regime=False, shorts=True, vol_scale=False, cond_short=False, stress_q=0.60),
    "BSC_LS": dict(regime=False, shorts=True, vol_scale=True, cond_short=False, stress_q=0.60),
    "REGIME_LS": dict(regime=True, shorts=True, vol_scale=False, cond_short=False, stress_q=0.60),
    "REGIME_LONG": dict(
        regime=True, shorts=False, vol_scale=False, cond_short=False, stress_q=0.60
    ),
    "SCX": dict(regime=True, shorts=True, vol_scale=False, cond_short=True, stress_q=0.60),
    "SCX_VOL": dict(regime=True, shorts=True, vol_scale=True, cond_short=True, stress_q=0.60),
}


# ── data ───────────────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    if not LONG_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_daily.py first")
    p = pd.read_csv(LONG_CACHE, index_col=0, parse_dates=True)
    p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
    print(f"[data] panel {p.shape} {p.index.min().date()}..{p.index.max().date()}")
    return p


def weekly_frame(close: pd.DataFrame, formation: int):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    vol = (
        close.pct_change(fill_method=None).rolling(CONFIG["vol_lookback_days"]).std() * np.sqrt(252)
    ).reindex(w.index, method="ffill")
    return fwd.iloc[formation:], mom.iloc[formation:], vol.iloc[formation:]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    """UP-UP flag at each weekly boundary (BTC above fast & slow daily MAs)."""
    btc = close["BTCUSDT"]
    fast = btc.rolling(CONFIG["regime_fast"]).mean()
    slow = btc.rolling(CONFIG["regime_slow"]).mean()
    up = (btc > fast) & (btc > slow)
    return up.resample("W-MON").last().reindex(close.resample("W-MON").last().index, method="ffill")


def ref_short_book_returns(mom: pd.DataFrame, fwd: pd.DataFrame, quintile: float) -> pd.Series:
    """Strategy-independent weekly return of the equal-weight bottom-quintile book.
    Represents 'how dangerous is the short side recently' -- uses only past data
    at each rebalance because we iterate forward and read the *prior* week's fwd."""
    out = {}
    for date in fwd.index:
        m = mom.loc[date].dropna()
        if len(m) < 12:
            continue
        n = max(2, int(round(quintile * len(m))))
        shorts = m.sort_values().index[:n]
        out[date] = fwd.loc[date, shorts].mean()
    return pd.Series(out)


# ── core weight builder ─────────────────────────────────────────────────────────
def weights_at(date, mom, vol, regime_flag, sb_ret_hist, spec: dict):
    """Target weights for the NEXT week. None => flat (no exposure)."""
    if spec["regime"] and not regime_flag:
        return None
    m = mom.loc[date].dropna()
    if len(m) < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs = ranked.index[-n:]
    shorts = ranked.index[:n]

    # conditional-short decision: drop shorts when short book is stressed
    shorts_on = spec["shorts"]
    if spec["cond_short"] and shorts_on and len(sb_ret_hist) >= CONFIG["short_vol_l"]:
        sbv = pd.Series(sb_ret_hist).rolling(CONFIG["short_vol_l"]).std().dropna()
        if len(sbv) >= 4:
            thr = sbv.quantile(spec["stress_q"])
            if sbv.iloc[-1] > thr:
                shorts_on = False  # tail-active -> go long-only

    w = pd.Series(dtype=float)
    w = pd.concat([w, pd.Series(1.0 / n, index=longs)])
    if shorts_on:
        w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])

    if spec["vol_scale"]:
        rv = vol.loc[date].reindex(w.index).dropna()
        if rv.empty or rv.mean() <= 0:
            return None
        scale = float(np.clip(CONFIG["vol_target"] / rv.mean(), 0.0, 3.0))
        w = w * scale
    return w


def backtest(close: pd.DataFrame, spec: dict, stress_q: float) -> pd.Series:
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(mom, fwd, CONFIG["quintile"])
    s = dict(spec)
    s["stress_q"] = stress_q
    dates = list(fwd.index)
    ret, prev, sb_hist = [], pd.Series(dtype=float), {}
    for i, date in enumerate(dates):
        # short-book history available UP TO (not including) this rebalance
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        w = weights_at(date, mom, vol, bool(reg.loc[date]), sb_hist, s)
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


# ── walk-forward selection of the conditional-short stress quantile ─────────────
def walk_forward_q(close: pd.DataFrame, spec: dict) -> tuple[list[float], list[float]]:
    fwd, _, _ = weekly_frame(close, CONFIG["formation_days"])
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    q_seq: list[float] = []
    sh_seq: list[int] = []
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        best_q, best_sh, best_s = CONFIG["stress_q_grid"][1], 1, -1e9
        for q in CONFIG["stress_q_grid"]:
            s = backtest(close, spec, q).iloc[start:end]
            sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
            if sr > best_s:
                best_s, best_q, best_sh = sr, q, 1
        for j in range(end, min(end + te, n)):
            q_seq.append(best_q)
            sh_seq.append(best_sh)
    last_q = q_seq[-1] if q_seq else CONFIG["stress_q_grid"][1]
    while len(q_seq) < n:
        q_seq.append(last_q)
        sh_seq.append(1)
    return q_seq[:n], sh_seq[:n]


# ── metrics ─────────────────────────────────────────────────────────────────────
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


def report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]"
        f"  %flat={m['pct_flat'] * 100:.0f}%"
    )
    print(
        f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%"
        f"  CVaR5%={m['cvar05'] * 100:6.2f}%"
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
    close = load_panel()
    print(f"config: {CONFIG}")

    # baselines (fixed spec)
    for name, spec in VARIANTS.items():
        if name.startswith("SCX") or spec["cond_short"]:
            continue
        report(name, metrics(backtest(close, spec, spec["stress_q"])))

    # SCX variants: walk-forward the stress quantile (out-of-sample)
    for name, spec in VARIANTS.items():
        if not (name.startswith("SCX") or spec["cond_short"]):
            continue
        q_seq, sh_seq = walk_forward_q(close, spec)
        chosen = sorted({round(x, 2) for x in q_seq})
        avg_short = float(np.mean(sh_seq))
        print(f"\n[walk-forward {name}] q selected={chosen}  avg_short_on={avg_short * 100:.0f}%")
        # rebuild with WF q by re-running per-block is expensive; approximate:
        # recompute using the chosen per-block q via a dedicated backtest path
        ret = backtest_wf(close, spec, q_seq)
        report(f"{name} (WF q, costs)", metrics(ret))
        store[name] = ret

    # crash sub-periods for the key variants
    compare = {
        n: backtest(close, VARIANTS[n], VARIANTS[n]["stress_q"])
        for n in ["NAIVE_LS", "REGIME_LS", "REGIME_LONG", "SCX"]
        if n in VARIANTS
    }
    for n in ["NAIVE_LS", "REGIME_LS", "REGIME_LONG", "SCX"]:
        if n in store:
            compare[n] = store[n]
    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>15}" for k in CRASH))
    for name, ret in compare.items():
        row = ""
        for a, b in CRASH.values():
            sub = ret.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:15.2f}"
        print(f"  {name:>13}{row}")


def backtest_wf(close, spec, q_seq):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(mom, fwd, CONFIG["quintile"])
    dates = list(fwd.index)
    ret = []
    prev = pd.Series(dtype=float)
    for i, date in enumerate(dates):
        s = dict(spec)
        s["stress_q"] = q_seq[i]
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        w = weights_at(date, mom, vol, bool(reg.loc[date]), sb_hist, s)
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


store: dict[str, pd.Series] = {}


if __name__ == "__main__":
    main()
