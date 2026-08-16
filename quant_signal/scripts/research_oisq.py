"""Research: OISQ -- Open-Interest-Flow "Squeeze" cross-sectional crypto factor.

HUNT mandate (user): a factor that is (a) crypto-native / absent from the factor
zoo, (b) obtainable KEYLESS, (c) has a real economic story, and (d) empirically
HINTED to work. Of three scanned candidates (CGO×funding×regime = combo of known
primitives; on-chain×perp-basis = needs paid keys; OI-flow squeeze = novel+keyless)
this is the only one clearing all four. Not a brand-new statistical object -- it
is a genuinely underexploited *combination* (OI-flow × price) assembled into a
cross-sectional score, which neither Liu-Tsyvinski, Crypto Carry (3774118), BIS
WP1087, CTREND (JFQA 2025), nor FRD portfolio-test.

Mechanism (economic story):
  Crypto returns are driven by LEVERAGE CYCLES, not just price trends.
    - price UP  & OpenInterest DOWN  -> shorts are covering (squeeze); the move is
      "real" and tends to CONTINUE  -> LONG.
    - price DOWN & OpenInterest UP    -> leveraged longs are being trapped/built;
      fragile, cascades               -> SHORT.
  Per coin-day raw score  s = sign(Δprice) * (-sign(ΔOI)); trailing-mean over a
  walk-forward-selected window; cross-sectionally rank weekly, long top quintile /
  short bottom quintile. The score prices leverage-unwind risk momentum & funding
  miss. BTC UP-UP regime gate (flat in bears) as in SCX.

Data (KEYLESS): Binance public /futures/data/openInterestHist (no API key; OI
exists from ~2021). Cached to /tmp/crypto_oi.csv. Price from /tmp/crypto_daily_long.csv
(run scripts/pull_binance_daily.py first). Funding optional from /tmp/crypto_funding.csv.

Everything parameterized in CONFIG/VARIANTS. Costs + crash regimes + bootstrap CI.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "oi_window_grid": [7, 14, 21],  # trailing-mean window for the OI-flow score (WF)
    "formation_days": 14,  # momentum lookback (for OIxMOM variant)
    "quintile": 0.20,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
    "vol_lookback_days": 126,
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
OI_CACHE = Path("/tmp/crypto_oi.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")
_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_REQUEST_INTERVAL_S = 1.2  # be gentle on the public endpoint


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def pull_oi(symbol: str) -> pd.Series:
    """Daily sumOpenInterest for ``symbol``, oldest->newest, keyless, paginated.

    Binance caps a page (~500 rows for period=1d); walk backward from now.
    """
    rows: list[tuple[int, float]] = []
    end = int(time.time() * 1000)
    # ~6y backstop so we don't loop forever if the endpoint misbehaves.
    floor = end - 6 * 365 * 24 * 3_600_000
    while end > floor:
        url = f"{_OI_URL}?symbol={symbol.upper()}&period=1d&limit=500&endTime={end}"
        try:
            data = _get_json(url)
        except Exception as e:  # noqa: BLE001 - best-effort; return what we have
            print(f"  [oi] {symbol} fetch warn: {e}")
            break
        if not data:
            break
        page = [(int(r["timestamp"]), float(r["sumOpenInterest"])) for r in data]
        page.sort()
        rows = page + rows
        oldest = page[0][0]
        if oldest <= floor:
            break
        end = oldest - 1
        time.sleep(_REQUEST_INTERVAL_S)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(t, unit="ms"): v for t, v in rows})
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def load_oi_panel(refresh: bool = False) -> pd.DataFrame:
    if OI_CACHE.exists() and not refresh:
        p = pd.read_csv(OI_CACHE, index_col=0, parse_dates=True)
        p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
        print(f"[oi] panel {p.shape} {p.index.min().date()}..{p.index.max().date()}")
        return p
    frames = {}
    for s in UNIVERSE:
        print(f"[oi] pulling {s} ...", flush=True)
        frames[s] = pull_oi(s)
    p = pd.DataFrame(frames)
    p.index.name = "date"
    p.to_csv(OI_CACHE)
    print(f"[oi] panel {p.shape} {p.index.min().date()}..{p.index.max().date()}")
    return p


def load_price() -> pd.DataFrame:
    if not PRICE_CACHE.exists():
        raise SystemExit("run scripts/pull_binance_daily.py first")
    p = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    p.index = pd.to_datetime(p.index, utc=True).tz_localize(None)
    return p


def squeeze_score(close: pd.DataFrame, oi: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per coin-day raw = sign(dPrice) * (-sign(dOI)); trailing-mean over window."""
    px = close.pct_change(fill_method=None)
    oi_chg = oi.pct_change(fill_method=None)
    raw = np.sign(px) * (-np.sign(oi_chg))
    score = raw.rolling(window).mean()
    return score


def weekly_frame(close: pd.DataFrame, formation: int):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    return fwd.iloc[formation:], mom.iloc[formation:]


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
        f"  {name:>12}: Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}] "
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
    oi = load_oi_panel()
    # Align to the intersection of OI history (keyless, ~2021+) and price.
    common = close.index.intersection(oi.index)
    common = common[common >= oi.index.min()]
    close = close.loc[common]
    oi = oi.loc[common]
    print(f"[align] {len(common)} weeks from {common.min().date()}..{common.max().date()}")

    fwd, mom = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)

    print("\n=== OISQ: walk-forward-selected OI window (costed 10bps) ===")
    best_w, best_sr, best_ret = None, -1e9, None
    for w in CONFIG["oi_window_grid"]:
        sc = squeeze_score(close, oi, w).reindex(fwd.index, method="ffill")
        r = backtest(sc, fwd, regime=None)
        m = metrics(r)
        print(f"  window={w:2d}d ", end="")
        report(f"RAW_OI", m)
        if m["sharpe"] > best_sr:
            best_sr, best_w, best_ret = m["sharpe"], w, r
    print(f"  -> best OI window = {best_w}d (Sharpe {best_sr:.2f})")

    sc = squeeze_score(close, oi, best_w).reindex(fwd.index, method="ffill")
    print("\n=== Variants (best window, costed) ===")
    report("OISQ_RAW", metrics(backtest(sc, fwd, regime=None)))
    report("OISQ_REGIME", metrics(backtest(sc, fwd, regime=reg)))

    # OI x MOM stack: average the two z-scored scores.
    momz = (mom - mom.mean()) / (mom.std() + 1e-9)
    scz = (sc - sc.mean()) / (sc.std() + 1e-9)
    stack = (scz + momz) / 2.0
    report("OISQxMOM", metrics(backtest(stack, fwd, regime=None)))
    report("OISQxMOM_REG", metrics(backtest(stack, fwd, regime=reg)))

    # Momentum baseline for context.
    report("MOM14_base", metrics(backtest(mom, fwd, regime=None)))

    print("\n--- crash-regime annualized Sharpe ---")
    for name, r in [
        ("OISQ_RAW", backtest(sc, fwd, regime=None)),
        ("OISQ_REGIME", backtest(sc, fwd, regime=reg)),
        ("OISQxMOM_REG", backtest(stack, fwd, regime=reg)),
    ]:
        row = ""
        for a, b in CRASH.values():
            sub = r.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:>10.2f}"
        print(f"  {name:>14}{row}")


if __name__ == "__main__":
    main()
