"""Pull daily funding-rate history for the live 31-coin universe from Binance USD-M perps.

Why this exists (the research gap):
  The daily OHLCV panel only lets us rediscover what academia already knows
  (momentum wins, everything else fails). Crypto's native microstructure signal
  is the PERPETUAL FUNDING RATE -- the price leverage-crowded positions pay to
  hold. It is NOT in any daily close/volume panel, and it is the edge every
  market-structure study (Christin et al. "Crypto Carry Trade"; BIS WP1087;
  SSRN 3774118; Kbit "dislocation"; Lucida liquidity) points to. This script
  caches it so factors can use it.

Output: /tmp/crypto_funding.csv  (index=UTC date, cols=symbols,
        value = DAILY summed funding rate, i.e. that day's funding yield).
        A second file /tmp/crypto_funding_ann.csv holds the annualized figure
        (daily_yield * 365) for readability.

Run: uv run python scripts/pull_binance_funding.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

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
OUT = Path("/tmp/crypto_funding.csv")
OUT_ANN = Path("/tmp/crypto_funding_ann.csv")
START_MS = 1_598_688_000_000  # 2020-09-01, earliest funding we verified exists


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def pull_symbol(symbol: str) -> pd.Series:
    rows, end = [], START_MS
    while True:
        url = (
            f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}"
            f"&startTime={end}&limit=1000"
        )
        chunk = _get(url)
        if not chunk:
            break
        rows.extend(chunk)
        last = chunk[-1]["fundingTime"]
        if last <= end:
            break
        end = last + 1
        time.sleep(0.05)
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_localize(None)
    df["r"] = df["fundingRate"].astype(float)
    daily = df.groupby(df["ts"].dt.normalize())["r"].sum()
    daily.index.name = "date"
    print(
        f"  {symbol}: {len(daily)} days, last={daily.index.max().date()} "
        f"avg_ann={daily.mean() * 365 * 100:.2f}%"
    )
    return daily


def main() -> None:
    print(f"[funding] pulling {len(UNIVERSE)} symbols from 2020-09 ...")
    frames = {}
    for s in UNIVERSE:
        try:
            frames[s] = pull_symbol(s)
        except Exception as e:  # symbol may not have existed early / rate limited
            print(f"  {s}: SKIP ({e})")
            time.sleep(1)
    mat = pd.DataFrame(frames).sort_index()
    mat.index = pd.to_datetime(mat.index)
    mat.to_csv(OUT)
    (mat * 365.0).to_csv(OUT_ANN)
    print(f"[funding] wrote {mat.shape} -> {OUT}")
    print(f"[funding] coverage (non-null days) per symbol:\n{mat.notna().sum()}")


if __name__ == "__main__":
    main()
