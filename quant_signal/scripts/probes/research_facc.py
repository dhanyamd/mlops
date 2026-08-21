"""Research: FACC -- Funding-Acceleration x Cross-Sectional Momentum.

Novel contribution (not a reproduction; distinct from LCF/CFR which use funding
*level*):
  Funding *level* is a known contrarian signal (LCF validated, Sharpe 1.14). But
  the *acceleration* of funding -- the trailing change in the perpetual funding
  rate -- is a separate, far less studied dynamic. It measures leverage
  BUILDING vs UNWINDING in real time:
      - accelerating POSITIVE funding  => leveraged longs crowding in  => crowded,
        prone to squeeze / mean-reversion (fade it),
      - decelerating / turning NEGATIVE funding => shorts capitulating / covering
        => unwind exhaustion, prone to rebound (lean into it).
  FACC therefore blends relative momentum with a cross-sectional z-score of
  funding acceleration as a crowding overlay:
      score = z(14d momentum) - BETA * z(funding acceleration)
  so the book keeps momentum exposure but FADING names where funding is
  accelerating (crowded longs) and EMBRACING names where funding is decelerating
  (uncrowded / capitulating). BETA is walk-forward selected out-of-sample.

Crypto-native (funding is a perp primitive price cannot see), cross-sectional
(31 names), keyless (Binance daily funding cache). Costs 10bps/side, BTC UP-UP
regime gate + conditional-short skew overlay mirror SCX for a fair comparison.

Run: uv run python scripts/research_facc.py
"""

from __future__ import annotations

import random
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
    "accel_w": 2,  # funding-acceleration window (weeks): diff over 2 weeks
    "short_vol_l": 12,  # trailing short-book vol window (weeks)
    "stress_q_grid": [0.50, 0.60, 0.70, 0.80],
    "beta_grid": [0.0, 0.5, 1.0, 1.5, 2.0],  # crowding-overlay strength
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
PRICE_CACHE = Path("/tmp/crypto_daily_long.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")

# Variant specs: which score, regime gate, shorts, vol-scale, conditional short.
VARIANTS = {
    "MOM14": dict(
        kind="mom", regime=False, shorts=True, vol_scale=False, cond_short=False, beta=0.0
    ),
    "FACC_ONLY": dict(
        kind="facc", regime=False, shorts=True, vol_scale=False, cond_short=False, beta=1.0
    ),
    "FACC_MOM": dict(
        kind="blend", regime=False, shorts=True, vol_scale=False, cond_short=False, beta=1.0
    ),
    "FACC_MOM_REGIME": dict(
        kind="blend", regime=True, shorts=True, vol_scale=False, cond_short=False, beta=1.0
    ),
    "FACC_SCX": dict(
        kind="blend", regime=True, shorts=True, vol_scale=False, cond_short=True, beta=1.0
    ),
}


# ── data ───────────────────────────────────────────────────────────────────────
def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PRICE_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_daily.py first")
    if not FUND_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_funding.py first")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    # align on the intersection of dates (funding starts 2020-08)
    common = px.index.intersection(fd.index)
    px, fd = px.loc[common], fd.loc[common]
    fd = fd.reindex(columns=px.columns)
    print(
        f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()} "
        f"(funding aligned {fd.index.min().date()}..{fd.index.max().date()})"
    )
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


def funding_accel(fund: pd.DataFrame, accel_w: int) -> pd.DataFrame:
    """Cross-sectional z-score of the trailing funding-rate CHANGE (acceleration).

    Positive z => funding accelerating up (crowded longs building); negative z =>
    funding decelerating / turning negative (unwind / capitulation)."""
    fw = fund.resample("W-MON").last()
    acc = fw.diff(accel_w)  # change over the acceleration window
    # z-score across the universe at each weekly boundary
    mu, sd = acc.mean(axis=1), acc.std(axis=1)
    z = (acc.sub(mu, axis=0)).div(sd.replace(0, np.nan), axis=0)
    return z


def build_scores(mom: pd.DataFrame, facc: pd.DataFrame, spec: dict, beta: float) -> pd.DataFrame:
    """Composite weekly score (cross-sectionally comparable)."""
    if spec["kind"] == "mom":
        s = mom
    elif spec["kind"] == "facc":
        s = -facc  # pure contrarian on funding acceleration
    else:  # blend
        s = mom - beta * facc
    return s.reindex(index=mom.index, columns=mom.columns)


def ref_short_book_returns(mom: pd.DataFrame, fwd: pd.DataFrame, quintile: float) -> pd.Series:
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
def weights_at(date, score, vol, regime_flag, sb_ret_hist, spec: dict):
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
    if spec["cond_short"] and shorts_on and len(sb_ret_hist) >= CONFIG["short_vol_l"]:
        sbv = pd.Series(sb_ret_hist).rolling(CONFIG["short_vol_l"]).std().dropna()
        if len(sbv) >= 4:
            thr = sbv.quantile(spec["stress_q"])
            if sbv.iloc[-1] > thr:
                shorts_on = False

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


def backtest(close, score, spec, stress_q=0.60, beta=None) -> pd.Series:
    fwd, _, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(
        (close / close.shift(CONFIG["formation_days"]) - 1.0)
        .resample("W-MON")
        .last()
        .iloc[CONFIG["formation_days"] :],
        fwd,
        CONFIG["quintile"],
    )
    s = dict(spec)
    s["stress_q"] = stress_q
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        w = weights_at(date, score, vol, bool(reg.loc[date]), sb_hist, s)
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


# ── walk-forward: select BETA (crowding strength) and stress Q out-of-sample ────
def walk_forward(close, fund, spec) -> tuple[pd.Series, float, list[float]]:
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    facc = funding_accel(fund, CONFIG["accel_w"]).reindex(index=fwd.index, columns=fwd.columns)
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(mom, fwd, CONFIG["quintile"])
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    beta_seq: list[float] = []
    q_seq: list[float] = []
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        best_beta, best_q, best_sh, best_s = (
            CONFIG["beta_grid"][1],
            CONFIG["stress_q_grid"][1],
            1,
            -1e9,
        )
        for beta in CONFIG["beta_grid"]:
            score = build_scores(mom, facc, spec, beta)
            for q in CONFIG["stress_q_grid"]:
                s = backtest(close, score, spec, q, beta).iloc[start:end]
                sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
                if sr > best_s:
                    best_s, best_beta, best_q, best_sh = sr, beta, q, 1
        for _ in range(end, min(end + te, n)):
            beta_seq.append(best_beta)
            q_seq.append(best_q)
    while len(beta_seq) < n:
        beta_seq.append(best_beta)
        q_seq.append(best_q)
    # full-path rebuild with WF-selected per-block beta/q
    ret = []
    prev = pd.Series(dtype=float)
    for i, date in enumerate(dates):
        sb_hist = {d: sb[d] for d in sb.index if d < date}
        sc = build_scores(mom, facc, spec, beta_seq[i])
        s = dict(spec)
        s["stress_q"] = q_seq[i]
        w = weights_at(date, sc, vol, bool(reg.loc[date]), sb_hist, s)
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
    return pd.Series(ret, index=dates), best_beta, sorted({round(x, 2) for x in beta_seq})


# ── metrics ─────────────────────────────────────────────────────────────────────
def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    n = len(ret)
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = random.Random(0)
    boot = []
    vals = list(ret.values)
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
        "skew": float(pd.Series(ret).skew()),
        "exkurt": float(pd.Series(ret).kurt()),
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
        f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%"
        f"  CVaR5%={m['cvar05'] * 100:6.2f}%"
    )


CRASH = {
    "COVID 2020": ("2020-02-20", "2020-04-01"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX 2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close, fund = load_panel()
    print(f"config: {CONFIG}")

    results: dict[str, pd.Series] = {}
    # fixed baselines + FACC_MOM (beta fixed at 1.0)
    for name, spec in VARIANTS.items():
        if spec["kind"] == "blend" and name != "FACC_MOM":
            continue
        fwd, mom, _ = weekly_frame(close, CONFIG["formation_days"])
        facc = funding_accel(fund, CONFIG["accel_w"]).reindex(index=fwd.index, columns=fwd.columns)
        score = build_scores(mom, facc, spec, spec["beta"])
        ret = backtest(close, score, spec, spec.get("stress_q", 0.60), spec["beta"])
        report(name, metrics(ret))
        results[name] = ret

    # walk-forward the crowding overlay for the regime/SCX blends
    for name in ["FACC_MOM_REGIME", "FACC_SCX"]:
        spec = VARIANTS[name]
        ret, best_beta, betas = walk_forward(close, fund, spec)
        print(f"\n[walk-forward {name}] WF beta selected={betas} (last={best_beta})")
        report(f"{name} (WF beta, costs)", metrics(ret))
        results[name] = ret

    # crash-regime sub-period Sharpe
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
