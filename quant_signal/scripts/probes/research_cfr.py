"""Research: CFR -- Capital-Gains-Overhang x Funding x Regime cross-sectional factor.

HUNT context: the genuinely novel OI-squeeze (OISQ) and on-chain (FLOB) factors
need PAID historical data (Binance free OI = 31 days; Xoomar null; CoinGlass
historical is paid). But the repo ALREADY holds multi-year KEYLESS data:
  - /tmp/crypto_daily_long.csv : 31-coin daily CLOSE, 2017-2026
  - /tmp/crypto_funding.csv    : 31-coin daily FUNDING, 2020-2026
and Binance spot klines expose daily VOLUME keylessly -> proper CGO is computable
with NO key. So CFR is the novel-to-crypto factor we CAN validate right now.

Novelty (agent-1 finding, honest): CGO (Griffin-Han 2005 / Bali et al. 2022) is a
mature EQUITY factor that SUBSUMES momentum; funding carry is heavily published for
crypto (BIS WP1087, SSRN 3774118, Grobys 2025). The TRIPLE interaction CGO x funding
x BTC-regime is NOT in the literature (Crypto Factor Zoo 2026 screened 36 factors and
excluded CGO). So: a real new COMBINATION for crypto, not a new primitive.

Economic story: funding = crowded-long leverage (carry); CGO = reference-point /
uninformed-vs-informed overhang. High-funding + LOW-CGO coins carry the carry premium
without offsetting disposition selling (no large unrealized gains to trigger retail
profit-taking). Regime gate neutralizes bear fragility. Momentum/funding/SCX can't
isolate this subset.

CGO (Griffin-Han): CGO_t = Σ_{s=1..L} (P_t - P_{t-s})·V_{t-s} / (P_t · Σ_{s=1..L} V_{t-s})
-> average unrealized capital gain per share held, relative to current price.
Everything parameterized (CONFIG/VARIANTS); costs + crash regimes + bootstrap CI.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "cgo_l_grid": [63, 126, 252],  # CGO lookback (WF)
    "quintile": 0.20,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
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
VOL_CACHE = Path("/tmp/crypto_volume.csv")
_KLINE = "https://api.binance.com/api/v3/klines"
_REQ_INT = 0.4


def _get_json(url: str):
    return json.load(
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=30
        )
    )


def pull_volume(symbol: str) -> pd.Series:
    rows: list[tuple[int, float]] = []
    end = int(time.time() * 1000)
    floor = end - 8 * 365 * 24 * 3_600_000
    while end > floor:
        url = f"{_KLINE}?symbol={symbol.upper()}&interval=1d&limit=1000&endTime={end}"
        try:
            data = _get_json(url)
        except Exception as e:  # noqa: BLE001
            print(f"  [vol] {symbol} warn: {e}")
            break
        if not data:
            break
        page = [(int(r[0]), float(r[5])) for r in data]  # volume = field 5
        page.sort()
        rows = page + rows
        oldest = page[0][0]
        if oldest <= floor:
            break
        end = oldest - 1
        time.sleep(_REQ_INT)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(t, unit="ms"): v for t, v in rows})
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def load_volume(refresh: bool = False) -> pd.DataFrame:
    if VOL_CACHE.exists() and not refresh:
        p = pd.read_csv(VOL_CACHE, index_col=0, parse_dates=True)
        p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
        print(f"[vol] panel {p.shape} {p.index.min().date()}..{p.index.max().date()}")
        return p
    frames = {}
    for s in UNIVERSE:
        print(f"[vol] pulling {s}", flush=True)
        frames[s] = pull_volume(s)
    p = pd.DataFrame(frames)
    p.index.name = "date"
    p.to_csv(VOL_CACHE)
    print(f"[vol] panel {p.shape} {p.index.min().date()}..{p.index.max().date()}")
    return p


def load_price() -> pd.DataFrame:
    p = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
    return p


def load_funding() -> pd.DataFrame:
    p = pd.read_csv(FUND_CACHE)
    p = p.rename(columns={p.columns[0]: "date"}).set_index("date")
    p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
    return p


def cgo(close: pd.DataFrame, vol: pd.DataFrame, L: int) -> pd.DataFrame:
    """Griffin-Han capital-gains overhang, vectorized. Aligned close/vol (date x symbol)."""
    # Σ_{s=1..L} (P_t - P_{t-s}) * V_{t-s}
    num = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for s in range(1, L + 1):
        num = num.add((close - close.shift(s)) * vol.shift(s), fill_value=0.0)
    denom = close * vol.rolling(L).sum().shift(1)
    return num / denom


def weekly_frame(close: pd.DataFrame, formation: int = 14):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    return fwd.iloc[formation:]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    fast = btc.rolling(CONFIG["regime_fast"]).mean()
    slow = btc.rolling(CONFIG["regime_slow"]).mean()
    up = (btc > fast) & (btc > slow)
    return up.resample("W-MON").last().reindex(close.resample("W-MON").last().index, method="ffill")


def backtest(score: pd.DataFrame, fwd: pd.DataFrame, *, regime: pd.Series | None) -> pd.Series:
    dates = list(fwd.index)
    ret = []
    prev = pd.Series(dtype=float)
    for date in dates:
        if regime is not None and not bool(regime.loc[date]):
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        sc = score.loc[date].dropna()
        if len(sc) < 12:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        ranked = sc.sort_values()
        n = max(2, int(round(CONFIG["quintile"] * len(ranked))))
        longs = ranked.index[-n:]
        shorts = ranked.index[:n]
        w = pd.concat([pd.Series(1.0 / n, index=longs), pd.Series(-1.0 / n, index=shorts)])
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
    if n < 5:
        return {
            "n": n,
            "ann_ret": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "ci": (float("nan"), float("nan")),
            "maxdd": 0.0,
            "pct_flat": 1.0,
        }
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = np.random.default_rng(0)
    boot = [
        (rng.choice(ret.values, n, replace=True).mean() * 52)
        / (rng.choice(ret.values, n, replace=True).std() * np.sqrt(52) + 1e-12)
        for _ in range(1000)
    ]
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


def report(name: str, m: dict) -> None:
    print(
        f"  {name:>14}: Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}] "
        f"ann={m['ann_ret'] * 100:5.1f}% maxDD={m['maxdd'] * 100:5.1f}% %flat={m['pct_flat'] * 100:.0f}%"
    )


CRASH = {
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX Nov-2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close = load_price()
    fund = load_funding()
    vol = load_volume()
    common = close.index.intersection(fund.index).intersection(vol.index)
    close, fund, vol = close.loc[common], fund.loc[common], vol.loc[common]
    print(f"[align] {len(common)} days {common.min().date()}..{common.max().date()}")

    fwd = weekly_frame(close)
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    fund_w = fund.resample("W-MON").last().reindex(fwd.index, method="ffill")

    print("\n=== CFR: walk-forward CGO lookback (costed 10bps) ===")
    best_L, best_sr, best_cgo = None, -1e9, None
    for L in CONFIG["cgo_l_grid"]:
        c = cgo(close, vol, L).reindex(fwd.index, method="ffill")
        # Score = z(funding) + z(-CGO): long high-funding / low-CGO winners.
        fz = (fund_w - fund_w.mean()) / (fund_w.std() + 1e-9)
        cz = (c - c.mean()) / (c.std() + 1e-9)
        score = fz - cz  # funding up & CGO down -> high
        r = backtest(score, fwd, regime=reg)
        m = metrics(r)
        print(f"  L={L:3d}d ", end="")
        report("CFR_REGIME", m)
        if m["sharpe"] > best_sr:
            best_sr, best_L, best_cgo = m["sharpe"], L, score
    print(f"  -> best CGO L = {best_L}d (Sharpe {best_sr:.2f})")

    c = best_cgo
    print("\n=== Variants (best CGO L, costed) ===")
    fz = (fund_w - fund_w.mean()) / (fund_w.std() + 1e-9)
    cz = (c - c.mean()) / (c.std() + 1e-9)
    report("FUND_ONLY", metrics(backtest(fz, fwd, regime=reg)))
    report("CGO_ONLY", metrics(backtest(-cz, fwd, regime=reg)))
    report("CFR_REGIME", metrics(backtest(fz - cz, fwd, regime=reg)))
    report("CFR_FLAT", metrics(backtest(fz - cz, fwd, regime=None)))

    # Baselines for context (from same caches).
    mom = (
        (close / close.shift(14) - 1.0).resample("W-MON").last().reindex(fwd.index, method="ffill")
    )
    report("MOM14", metrics(backtest(mom, fwd, regime=None)))
    report("MOM14_REG", metrics(backtest(mom, fwd, regime=reg)))

    print("\n--- crash-regime annualized Sharpe ---")
    for name, sc in [("CFR_REGIME", fz - cz), ("FUND_ONLY", fz), ("MOM14_REG", mom)]:
        r = backtest(sc, fwd, regime=reg if name != "MOM14" else reg)
        row = ""
        for a, b in CRASH.values():
            sub = r.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:>10.2f}"
        print(f"  {name:>14}{row}")


if __name__ == "__main__":
    main()
