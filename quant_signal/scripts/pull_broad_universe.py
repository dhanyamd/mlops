"""KEYLESS BROAD-UNIVERSE PULL for the genuine factor-zoo breakthrough hunt.

WHY: Mercik/Zaremba/Demir (2026, "Crypto factor zoo", IRFA 113) show the sparse
priced crypto factors -- turnover volatility, salience theory value, new-address-to-price --
price the BROAD ~565-coin universe, NOT the liquid top-30 our current caches cover. Our
30-coin book cannot host a genuine microstructure breakthrough; a broad keyless universe can.

DATA (all keyless public REST, no API key):
  * Binance spot /api/v3/klines  -> daily close + quote volume (turnover proxy)
  * Binance perp /fapi/v1/fundingRate -> 8h funding rate (carry + ASYM positioning)
We deliberately DO NOT pull taker-flow (not keyless at scale) -- broad universe uses
price/funding/volume only, which is exactly what the zoo factors need.

RESUME-SAFE: writes one CSV per symbol under OUTDIR; re-running skips completed symbols,
so an interrupted/laptop-closed run can be continued. Final assembly builds wide panels.

Run: uv run python scripts/pull_broad_universe.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

SPOT = "https://api.binance.com/api/v3"
PERP = "https://fapi.binance.com/fapi/v1"
OUTDIR = Path("/tmp/broad_pull")
START_MS = 1577836800000  # 2020-01-01 UTC

STABLES = {
    "USDC",
    "BUSD",
    "TUSD",
    "FDUSD",
    "DAI",
    "USDP",
    "USDD",
    "TUSD",
    "USDT",
    "EUR",
    "BTC",
    "ETH",
}
BAD_SUFFIX = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S", "1L", "1S")


def get(url: str) -> list:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def spot_symbols() -> list[str]:
    info = get(f"{SPOT}/exchangeInfo")
    out = []
    for s in info["symbols"]:
        if s["quoteAsset"] != "USDT" or s["status"] != "TRADING":
            continue
        base = s["baseAsset"]
        if base in STABLES:
            continue
        if base.endswith(BAD_SUFFIX):
            continue
        out.append(s["symbol"])
    return sorted(out)


def pull_klines(symbol: str) -> pd.DataFrame:
    rows = []
    end = int(time.time() * 1000)
    start = START_MS
    while start < end:
        url = f"{SPOT}/klines?symbol={symbol}&interval=1d&startTime={start}&limit=1000"
        chunk = get(url)
        if not chunk:
            break
        rows.extend(chunk)
        start = chunk[-1][0] + 86_400_000
        if len(chunk) < 1000:
            break
        time.sleep(0.0)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"],
    )
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    df["close"] = df["c"].astype(float)
    df["qvol"] = df["qv"].astype(float)
    return df[["date", "close", "qvol"]].drop_duplicates("date").set_index("date").sort_index()


def pull_funding(symbol: str) -> pd.Series:
    # perp funding rate (8h cadence); resample to daily mean
    rows = []
    end = int(time.time() * 1000)
    start = START_MS
    while start < end:
        url = f"{PERP}/fundingRate?symbol={symbol}&startTime={start}&endTime={end}&limit=1000"
        chunk = get(url)
        if not chunk:
            break
        rows.extend(chunk)
        start = int(chunk[-1]["fundingTime"]) + 1
        if len(chunk) < 1000:
            break
        time.sleep(0.0)
    if not rows:
        return pd.Series(dtype=float)
    fr = pd.DataFrame(rows)
    fr["date"] = (
        pd.to_datetime(fr["fundingTime"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    )
    fr["rate"] = fr["fundingRate"].astype(float)
    daily = fr.groupby("date")["rate"].mean()
    return daily


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    done = {p.stem for p in OUTDIR.glob("*.csv")}
    syms = spot_symbols()
    print(f"[uni] {len(syms)} candidates, {len(done)} already pulled")
    n = 0
    for sym in syms:
        if sym in done:
            continue
        try:
            kl = pull_klines(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {sym} klines: {e}")
            kl = pd.DataFrame()
        if kl.empty or len(kl) < 60:
            continue
        try:
            fund = pull_funding(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {sym} funding: {e}")
            fund = pd.Series(dtype=float)
        out = kl.copy()
        out["funding"] = fund
        out.to_csv(OUTDIR / f"{sym}.csv")
        n += 1
        if n % 25 == 0:
            print(f"  pulled {n} symbols ({sym})...")
        time.sleep(0.0)
    print(f"[done] {n} new symbols; total {len(done) + n}")


if __name__ == "__main__":
    main()
