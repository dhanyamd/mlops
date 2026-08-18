"""KEYLESS pull: DefiLlama protocol TVL (totalValueLocked) -> per-coin panel.

DefiLlama free public API (api.llama.fi), NO auth. /protocol/{slug} returns a
`tvl` array = [[unix_ts, totalLiquidityUSD], ...] with deep daily history
(verified: aave-v3 1614 pts from 2022-03). Maps protocol slugs -> our USDT
universe tickers for the DeFi names.

WHY: the "Magical Internet Money" paper (SSRN 4540433, To 2023) shows crypto
cashflow/valuation ratios are PRICED and NOT spanned by momentum/carry factor
models — a genuine orthogonal axis. CF Benchmarks (2026) "Value" factor =
Fees/TVL. Fidelity finds TVL is causal with price, so the RATIO (not raw TVL)
isolates cashflow productivity. We build Fees/TVL in research_novel.py.
Saved to /tmp/crypto_defillama_tvl.csv (date x ticker, daily TVL USD).

Run: uv run python scripts/pull_defillama_tvl.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

OUT = Path("/tmp/crypto_defillama_tvl.csv")
# our USDT ticker -> DefiLlama protocol slug
SLUG_MAP = {
    "UNIUSDT": "uniswap-v3",
    "AAVEUSDT": "aave-v3",
    "LDOUSDT": "lido",
    "CRVUSDT": "curve-dex",
    "ARBUSDT": "arbitrum",
    "GMXUSDT": "gmx",
    "COMPUSDT": "compound",
    "CAKEUSDT": "pancakeswap",
    "MKRUSDT": "makerdao",
    "SNXUSDT": "synthetix",
}


def fetch(slug: str) -> list:
    url = f"https://api.llama.fi/protocol/{slug}"
    with urlopen(url, timeout=30) as r:
        d = json.load(r)
    return d.get("tvl", [])


def main() -> None:
    frames = {}
    for ticker, slug in SLUG_MAP.items():
        try:
            chart = fetch(slug)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {slug}: {e}")
            continue
        if not chart:
            print(f"  skip {slug}: empty")
            continue
        idx = pd.to_datetime([c["date"] for c in chart], unit="s", utc=True).tz_localize(None)
        series = pd.Series([c["totalLiquidityUSD"] for c in chart], index=idx, name=ticker)
        print(
            f"  {ticker} <- {slug}: {len(series)} pts "
            f"{series.index.min().date()}..{series.index.max().date()}"
        )
        frames[ticker] = series
        time.sleep(0.3)
    if not frames:
        raise SystemExit("no Defillama TVL pulled")
    panel = pd.DataFrame(frames).sort_index()
    panel = panel[panel.index >= "2020-08-29"]
    panel.to_csv(OUT)
    print(f"[saved] {OUT} shape={panel.shape}")


if __name__ == "__main__":
    main()
