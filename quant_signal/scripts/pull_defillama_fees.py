"""KEYLESS pull: DefiLlama protocol fees (dailyFees) -> per-coin panel.

DefiLlama free public API (api.llama.fi), NO auth. /summary/fees/{slug} returns
totalDataChart = [[unix_ts, dailyFees], ...] with deep daily history (verified:
aave 2078 pts from 2020-12, uniswap-v3 1925 pts from 2021-05).

Maps DefiLlama protocol slugs -> our USDT universe tickers for the DeFi names.
This is the blockchain-native, ORTHOGONAL signal the crypto factor-zoo (2026)
flags as priced and uncorrelated to momentum/microstructure. Saved to
/tmp/crypto_defillama_fees.csv (date x ticker, dailyFees USD).

Run: uv run python scripts/pull_defillama_fees.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

OUT = Path("/tmp/crypto_defillama_fees.csv")

# our USDT ticker -> candidate DefiLlama protocol slugs (first that returns data wins)
SLUG_MAP = {
    "UNIUSDT": ["uniswap-v3", "uniswap"],
    "AAVEUSDT": ["aave", "aave-v3"],
    "LDOUSDT": ["lido"],
    "CRVUSDT": ["curve-dex", "curve"],
    "ARBUSDT": ["arbitrum"],
    "GMXUSDT": ["gmx", "gmx-v2"],
    "COMPUSDT": ["compound", "compound-v3"],
    "CAKEUSDT": ["pancakeswap", "pancakeswap-amm"],
    "MKRUSDT": ["makerdao", "maker"],
    "SNXUSDT": ["synthetix", "synthix"],
}


def fetch(slug: str) -> list:
    url = f"https://api.llama.fi/summary/fees/{slug}"
    with urlopen(url, timeout=30) as r:
        d = json.load(r)
    return d.get("totalDataChart", [])


def main() -> None:
    frames = {}
    for ticker, slugs in SLUG_MAP.items():
        series = None
        for slug in slugs:
            try:
                chart = fetch(slug)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {slug}: {e}")
                continue
            if chart:
                idx = pd.to_datetime([c[0] for c in chart], unit="s", utc=True).tz_localize(None)
                series = pd.Series([c[1] for c in chart], index=idx, name=ticker)
                print(
                    f"  {ticker} <- {slug}: {len(series)} pts {series.index.min().date()}..{series.index.max().date()}"
                )
                break
            time.sleep(0.3)
        if series is not None:
            frames[ticker] = series
    if not frames:
        raise SystemExit("no Defillama fees pulled")
    panel = pd.DataFrame(frames).sort_index()
    panel = panel[panel.index >= "2020-08-29"]
    panel.to_csv(OUT)
    print(f"[saved] {OUT} shape={panel.shape}")


if __name__ == "__main__":
    main()
