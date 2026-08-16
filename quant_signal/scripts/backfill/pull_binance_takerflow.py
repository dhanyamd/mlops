"""Pull Binance USD-M futures DAILY klines (keyless, no API key) for the 31-coin universe
and derive the AGGRESSIVE TAKER-FLOW panel:

    taker_flow = taker_buy_quote_volume / quote_asset_volume   (col 10 / col 7)

taker_buy_quote_volume = volume executed by TAKER BUYS (aggressive market buyers);
quote_asset_volume = total quoted volume. So taker_flow > 0.5 => net aggressive BUYING
(smart/convicted money stepping in); < 0.5 => net aggressive SELLING / distribution.

This is a genuinely different, keyless, historical signal from our plain volume cache
(which only had total volume, no buy/sell split). Source: fapi.binance.com/fapi/v1/klines
(security NONE, weight ~10/call, 6000 weight/min per IP).

Writes:
    /tmp/crypto_takerflow.csv   (daily, 31 cols, taker_flow ratio in [0,1])
    /tmp/crypto_takerqvol.csv   (daily, 31 cols, total quote volume, for flow-momentum)

Run: uv run python scripts/pull_binance_takerflow.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

SYMS = [
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
START_MS = int(pd.Timestamp("2020-08-29").timestamp() * 1000)
OUT_FLOW = Path("/tmp/crypto_takerflow.csv")
OUT_QVOL = Path("/tmp/crypto_takerqvol.csv")
BASE = "https://fapi.binance.com/fapi/v1/klines"
LIMIT = 1000


def pull_symbol(sym: str) -> pd.DataFrame:
    frames = []
    start = START_MS
    while True:
        r = requests.get(
            BASE, params=dict(symbol=sym, interval="1d", limit=LIMIT, startTime=start), timeout=30
        )
        if r.status_code != 200:
            print(f"  [skip] {sym}: HTTP {r.status_code} {r.text[:120]}")
            return pd.DataFrame(columns=["flow", "qvol"])
        rows = r.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        last = int(rows[-1][0])
        if last <= start:
            break
        start = last + 1
        if len(frames) % 10 == 0:
            time.sleep(0.2)  # be gentle on the shared IP limit
        if last >= int(pd.Timestamp.utcnow().timestamp() * 1000) - 86_400_000:
            break
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=[0])
    idx = pd.to_datetime(raw[0].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    close_time = pd.to_datetime(raw[6].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    # col 7 = quote asset volume, col 10 = taker buy quote asset volume
    qvol = raw[7].astype(float).values
    tbuy = raw[10].astype(float).values
    flow = pd.Series(tbuy / qvol, index=idx).where(qvol > 0)
    qv = pd.Series(qvol, index=idx)
    return pd.DataFrame({"flow": flow, "qvol": qv})


def main() -> None:
    flow_panel, qvol_panel = [], []
    for i, sym in enumerate(SYMS, 1):
        df = pull_symbol(sym)
        flow_panel.append(df["flow"].rename(sym))
        qvol_panel.append(df["qvol"].rename(sym))
        print(
            f"  [{i:2d}/{len(SYMS)}] {sym}: {len(df)} bars  flow~{df['flow'].mean():.3f}",
            flush=True,
        )
    flow = pd.concat(flow_panel, axis=1).sort_index()
    qvol = pd.concat(qvol_panel, axis=1).sort_index()
    flow.to_csv(OUT_FLOW)
    qvol.to_csv(OUT_QVOL)
    print(f"[done] flow {flow.shape} -> {OUT_FLOW}")
    print(f"[done] qvol {qvol.shape} -> {OUT_QVOL}")


if __name__ == "__main__":
    main()
