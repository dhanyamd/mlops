"""Pull long daily crypto history (2017+) from Binance via ccxt for research.

Caches to /tmp/crypto_daily_long.csv so the backtest need not re-pull.
Binance lists most of the live universe back to 2017-2021, covering the 2018
bear, 2020 COVID crash, and 2022 FTX collapse -- the crash regimes the
Bybit 2022-2026 window misses.
"""

from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

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
CACHE = Path("/tmp/crypto_daily_long.csv")
START_MS = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)


def pull() -> dict[str, pd.DataFrame]:
    """Return {field: wide daily panel} for field in close/high/low/vol."""
    ex = ccxt.binance({"enableRateLimit": True})
    fields = {"c": {}, "h": {}, "l": {}, "v": {}}
    for sym in UNIVERSE:
        pair = sym[:-4] + "/" + sym[-4:]  # BTCUSDT -> BTC/USDT
        try:
            bars, since = [], START_MS
            while since < int(time.time() * 1000):
                chunk = ex.fetch_ohlcv(pair, "1d", since, limit=1000)
                if not chunk:
                    break
                bars += chunk
                since = chunk[-1][0] + 1
                time.sleep(0.25)
            if not bars:
                print(f"[skip] {sym}: no data")
                continue
            df = pd.DataFrame(bars, columns=["ts", "o", "h", "l", "c", "v"])
            df = df.drop_duplicates("ts").set_index("ts")
            for f in fields:
                fields[f][sym] = df[f]
            print(f"[ok] {sym}: {len(df)} bars, {pd.to_datetime(df.index[0], unit='ms').date()}")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sym}: {e}")
    out = {}
    for f, d in fields.items():
        p = pd.DataFrame(d).sort_index()
        p.index = pd.to_datetime(p.index, unit="ms", utc=True).tz_localize(None)
        out[f] = p
        out[f].to_csv(CACHE.with_name(f"crypto_daily_{f}.csv"))
    return out


if __name__ == "__main__":
    p = pull()
    print(f"\ncached long panels -> {CACHE.with_name('crypto_daily_*.csv')}")
