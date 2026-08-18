"""Research: COMOM -- Crowded Momentum (cross-sectional crowding timing of momentum).

Novel contribution (a genuine gap, not a reproduction):
  Lou & Polk (2022, RFS, "Comomentum") show that the ABNORMAL RETURN CORRELATION
  among winner and loser names during the formation period ("comomentum", CoMOM)
  measures arbitrage-crowding in a momentum book: HIGH comomentum => the crowd is
  large => subsequent momentum CRASHES/reverts; LOW comomentum => momentum is
  stabilizing and profitable. EFMA 2024 ("Cross-Predictive Ability of Crowding")
  shows crowding in one strategy predicts crashes in another. This literature is
  100% EQUITY -- it has NEVER been applied to the crypto perpetual cross-section.

  We introduce the crypto-native extension:
    1. CoMOM from FREE weekly returns (residual correlations after controlling for
       the crypto market factor) => return-comovement crowding.
    2. FUNDING-DISPERSION crowding (crypto-native, FREE funding cache): cross-sectional
       std of funding rates => leveraged-positioning crowding (high = crowded longs).
    3. When the combined crowding state is high, SKIP the momentum rebalance
       (flatten / long-only). This is the PRINCIPLED version of SCX's heuristic
       short-book-vol conditional-short -- it times crashes with a literature-grounded
       crowding state variable instead of a volatility proxy.

  Walk-forward selects the crowding SKIP threshold out-of-sample. Costs 10bps/side,
  BTC UP-UP regime gate + conditional-short mirror SCX for a fair comparison.

Run: uv run python scripts/research_comom.py
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
    "vol_lookback_days": 126,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "comom_window_w": 52,  # trailing weeks of residuals for the crowding correlation
    "fund_window_w": 104,  # trailing weeks to percentile-rank funding dispersion
    "short_vol_l": 12,
    "stress_q_grid": [0.50, 0.60, 0.70, 0.80],
    "skip_q_grid": [0.60, 0.70, 0.80, 0.90],  # crowding skip thresholds (time-series pctile)
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
    "COMOM_SKIP": dict(regime=True, shorts=True, cond_short=False),
    "COMOM_FUND_SKIP": dict(regime=True, shorts=True, cond_short=False, use_fund=True),
    "COMOM_SCX": dict(regime=True, shorts=True, cond_short=True, use_fund=True),
}


# ── data ───────────────────────────────────────────────────────────────────────
def load_panel():
    if not PRICE_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_daily.py first")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True) if FUND_CACHE.exists() else None
    if fd is not None:
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


def comom_series(close) -> pd.Series:
    """Lou & Polk (2022) comomentum on the crypto cross-section.

    At each weekly boundary: rank by 14d momentum -> winner/loser quintiles; compute
    market-factor residual returns over the trailing comom_window_w weeks; CoMOM =
    0.5 * (avg pairwise corr of residuals within winners + within losers). High =>
    crowded momentum => subsequent crash risk.
    """
    rw = close.pct_change(fill_method=None).resample("W-MON").last()
    mom = (close / close.shift(CONFIG["formation_days"]) - 1.0).resample("W-MON").last()
    rw, mom = rw.iloc[CONFIG["formation_days"] :], mom.iloc[CONFIG["formation_days"] :]
    W = CONFIG["comom_window_w"]
    dates = list(rw.index)
    out = {}
    for k in range(W, len(rw)):
        win = rw.iloc[k - W : k].values  # (W, n)
        mkt = win.mean(axis=1)
        mc = mkt - mkt.mean()
        var_m = float((mc**2).sum())
        # per-coin beta to equal-weight market, then residuals
        resid = np.empty_like(win)
        for j in range(win.shape[1]):
            c = win[:, j] - win[:, j].mean()
            beta = float((c * mc).sum()) / var_m if var_m > 0 else 0.0
            resid[:, j] = win[:, j] - beta * mkt
        md = dates[k]
        m = mom.loc[md].dropna()
        if len(m) < 12:
            continue
        ranked = m.sort_values()
        n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
        win_idx = [rw.columns.get_loc(c) for c in ranked.index[-n:]]
        los_idx = [rw.columns.get_loc(c) for c in ranked.index[:n]]
        cw = float(np.corrcoef(resid[:, win_idx].T).mean()) if len(win_idx) > 1 else 0.0
        cl = float(np.corrcoef(resid[:, los_idx].T).mean()) if len(los_idx) > 1 else 0.0
        out[md] = 0.5 * (cw + cl)
    s = pd.Series(out)
    print(f"[comom] computed {s.shape[0]} weekly crowding observations")
    return s


def fund_disp_series(fund) -> pd.Series:
    if fund is None:
        return pd.Series(dtype=float)
    fw = fund.resample("W-MON").last()
    return fw.std(axis=1)


def crowd_state(comom: pd.Series, fund_disp: pd.Series, use_fund: bool) -> pd.Series:
    """Time-series percentile (0..1) of crowding; high => skip momentum.

    Combines CoMOM percentile with funding-dispersion percentile over a trailing
    window. Both are slow state variables (CoMOM is highly autocorrelated in Lou &
    Polk); percentile-ranking against the trailing 2y history avoids in-sample bias.
    """

    def pctile(s, win):
        return s.rolling(win).apply(lambda x: (x[-1] >= x).mean(), raw=True)

    c = pctile(comom, CONFIG["fund_window_w"])
    if use_fund and not fund_disp.empty:
        f = pctile(fund_disp.reindex(comom.index), CONFIG["fund_window_w"])
        c = 0.5 * (c + f.reindex(comom.index))
    return c


def ref_short_book_returns(mom, fwd, quintile):
    out = {}
    for date in fwd.index:
        m = mom.loc[date].dropna()
        if len(m) < 12:
            continue
        n = max(2, int(round(quintile * len(m))))
        out[date] = fwd.loc[date, m.sort_values().index[:n]].mean()
    return pd.Series(out)


# ── core weight builder ─────────────────────────────────────────────────────────
def weights_at(date, score, vol, regime_flag, sb_hist, spec):
    if spec["regime"] and not regime_flag:
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    ranked = m.sort_values()
    n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
    longs, shorts = ranked.index[-n:], ranked.index[:n]
    shorts_on = spec["shorts"]
    if spec["cond_short"] and shorts_on and len(sb_hist) >= CONFIG["short_vol_l"]:
        sbv = pd.Series(sb_hist).rolling(CONFIG["short_vol_l"]).std().dropna()
        if len(sbv) >= 4 and sbv.iloc[-1] > sbv.quantile(spec.get("stress_q", 0.60)):
            shorts_on = False
    w = pd.Series(1.0 / n, index=longs)
    if shorts_on:
        w = pd.concat([w, pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest(close, crowd, skip_q, spec, stress_q=0.60):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    sb = ref_short_book_returns(mom, fwd, CONFIG["quintile"])
    crowd = crowd.reindex(fwd.index)
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        if (
            crowd.get(date, 0.0) is not None
            and not np.isnan(crowd.get(date, np.nan))
            and crowd.get(date) >= skip_q
        ):
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


def walk_forward_q(close, crowd, spec):
    fwd, _, _ = weekly_frame(close, CONFIG["formation_days"])
    dates = list(fwd.index)
    n = len(dates)
    tr, te = CONFIG["wf_train_weeks"], CONFIG["wf_test_weeks"]
    q_seq, best = [], CONFIG["skip_q_grid"][1]
    for start in range(0, n - te, te):
        end = min(start + tr, n - te)
        bq, bs = CONFIG["skip_q_grid"][1], -1e9
        for q in CONFIG["skip_q_grid"]:
            s = backtest(close, crowd, q, spec).iloc[start:end]
            sr = s.mean() / (s.std() + 1e-9) * np.sqrt(52)
            if sr > bs:
                bs, bq = sr, q
        for _ in range(end, min(end + te, n)):
            q_seq.append(bq)
    while len(q_seq) < n:
        q_seq.append(best)
    return q_seq[:n]


# ── metrics ─────────────────────────────────────────────────────────────────────
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
    comom = comom_series(close)
    fund_disp = fund_disp_series(fund)

    results = {}
    for name, spec in VARIANTS.items():
        use_fund = spec.get("use_fund", False)
        crowd = crowd_state(comom, fund_disp, use_fund).reindex(comom.index)
        if name in ("MOM14", "MOM14_REGIME"):
            # baselines: never skip (skip_q = 2.0 sentinel)
            ret = backtest(close, crowd, 2.0, spec)
            report(name, metrics(ret))
            results[name] = ret
            continue
        q_seq = walk_forward_q(close, crowd, spec)
        chosen = sorted({round(x, 2) for x in q_seq})
        # rebuild with WF-selected per-block skip threshold
        fwd, _, _ = weekly_frame(close, CONFIG["formation_days"])
        dates = list(fwd.index)
        ret = []
        prev = pd.Series(dtype=float)
        reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
        sb = ref_short_book_returns(
            (close / close.shift(CONFIG["formation_days"]) - 1.0)
            .resample("W-MON")
            .last()
            .iloc[CONFIG["formation_days"] :],
            fwd,
            CONFIG["quintile"],
        )
        for i, date in enumerate(dates):
            if (
                crowd.get(date, 0.0) is not None
                and not np.isnan(crowd.get(date, np.nan))
                and crowd.get(date) >= q_seq[i]
            ):
                ret.append(0.0)
                prev = pd.Series(dtype=float)
                continue
            sb_hist = {d: sb[d] for d in sb.index if d < date}
            w = weights_at(
                date,
                (close / close.shift(CONFIG["formation_days"]) - 1.0)
                .resample("W-MON")
                .last()
                .iloc[CONFIG["formation_days"] :],
                _,
                bool(reg.loc[date]),
                sb_hist,
                {**spec, "stress_q": 0.60},
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
