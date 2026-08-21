"""Pull KEYED derivatives microstructure for the live 31-coin universe from CoinGlass v4.

Why this exists:
  Free cross-sectional caches (price, funding, volume) are exhausted -- every novel
  XS factor collapses into a BTC-regime beta (REFLX confirmed this). The real,
  crypto-native microstructure signal is LEVERAGE: open interest (who is leveraged,
  how much) and LIQUIDATIONS (forced, uninformed flow that precedes reversals; see
  arXiv 2607.27070). Both are KEYED on CoinGlass. This script caches them so the
  LUW (Leverage-Unwind / Liquidation-Reversal) factor can run multi-year.

Endpoints (CoinGlass API v4, base https://open-api-v4.coinglass.com, header CG-API-KEY):
  OI  : /api/futures/open-interest/history
        params: exchange=Binance, symbol=<COIN>USDT, interval=1d, unit=usd
        -> data[].{time(ms), open, high, low, close}
        we keep `close` (OI in USD at interval end) as the daily figure.
  Liq: /api/futures/liquidation/aggregated-history
        params: exchange_list=Binance,Bybit,OKX, symbol=<COIN>, interval=1d
        -> data[].{time(ms), aggregated_long_liquidation_usd, aggregated_short_liquidation_usd}

Output (date-indexed, 31 columns, same schema as the free caches):
  /tmp/crypto_oi.csv        daily open interest (USD)
  /tmp/crypto_liq_long.csv  daily long-liquidation USD (aggregated across venues)
  /tmp/crypto_liq_short.csv daily short-liquidation USD (aggregated across venues)

Free (Hobbyist) plan covers daily interval all-time for both endpoints. Pagination
walks in 1000-day (limit=1000) windows; 429s back off. Set COINGLASS_API_KEY in env.

Run: uv run python scripts/pull_coinglass.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://open-api-v4.coinglass.com"
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
OUT_OI = Path("/tmp/crypto_oi.csv")
OUT_LL = Path("/tmp/crypto_liq_long.csv")
OUT_LS = Path("/tmp/crypto_liq_short.csv")
START_MS = 1_598_688_000_000  # 2020-09-01 (align with funding cache start)
DAY_MS = 86_400_000
LIMIT = 1000


def _get(url: str, api_key: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research", "CG-API-KEY": api_key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2**attempt * 5
                print(f"   429 rate limit, backoff {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"exhausted retries: {url}")


def _paginate(path: str, params: str, api_key: str):
    """Walk a CoinGlass endpoint in LIMIT-day windows from START_MS to now."""
    out, start = [], START_MS
    now = int(time.time() * 1000)
    while start < now:
        url = f"{BASE}{path}?{params}&start_time={start}&end_time={now}&limit={LIMIT}"
        js = _get(url, api_key)
        data = js.get("data") if isinstance(js, dict) else None
        if not data:
            break
        out.extend(data)
        last = data[-1]["time"]
        if len(data) < LIMIT or last <= start:
            break
        start = last + DAY_MS
        time.sleep(0.2)
    return out


def pull_oi(symbol: str, api_key: str) -> pd.Series:
    data = _paginate(
        "/api/futures/open-interest/history",
        f"exchange=Binance&symbol={symbol}&interval=1d&unit=usd",
        api_key,
    )
    if not data:
        return pd.Series(dtype=float)
    ts = pd.to_datetime([d["time"] for d in data], unit="ms", utc=True).tz_localize(None)
    oi = pd.Series([float(d.get("close", 0)) for d in data], index=ts, dtype=float)
    oi.index = oi.index.normalize()
    oi = oi[~oi.index.duplicated(keep="last")]
    oi.index.name = "date"
    print(
        f"  OI {symbol}: {len(oi)} days, last={oi.index.max().date()} "
        f"latest={oi.iloc[-1] / 1e9:.2f}B"
    )
    return oi


def pull_liq(coin: str, api_key: str) -> tuple[pd.Series, pd.Series]:
    data = _paginate(
        "/api/futures/liquidation/aggregated-history",
        f"exchange_list=Binance,Bybit,OKX&symbol={coin}&interval=1d",
        api_key,
    )
    if not data:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ts = pd.to_datetime([d["time"] for d in data], unit="ms", utc=True).tz_localize(None)
    ll = pd.Series(
        [float(d.get("aggregated_long_liquidation_usd", 0)) for d in data], index=ts, dtype=float
    )
    ls = pd.Series(
        [float(d.get("aggregated_short_liquidation_usd", 0)) for d in data], index=ts, dtype=float
    )
    for s in (ll, ls):
        s.index = s.index.normalize()
        s.index.name = "date"
    print(
        f"  LIQ {coin}: {len(ll)} days, "
        f"L={ll.iloc[-30:].mean() / 1e6:.1f}M/d S={ls.iloc[-30:].mean() / 1e6:.1f}M/d (30d)"
    )
    return ll, ls


def main() -> None:
    api_key = os.environ.get("COINGLASS_API_KEY")
    if not api_key:
        raise SystemExit("set COINGLASS_API_KEY in env (e.g. export COINGLASS_API_KEY=...)")
    print(f"[coinglass] pulling OI + liquidations for {len(UNIVERSE)} symbols from 2020-09 ...")
    oi_f, ll_f, ls_f = {}, {}, {}
    for s in UNIVERSE:
        coin = s.replace("USDT", "")
        try:
            oi_f[s] = pull_oi(s, api_key)
            ll, ls = pull_liq(coin, api_key)
            ll_f[s], ls_f[s] = ll, ls
        except Exception as e:
            print(f"  {s}: SKIP ({e})")
            time.sleep(1)
    oi = pd.DataFrame(oi_f).sort_index()
    ll = pd.DataFrame(ll_f).sort_index()
    ls = pd.DataFrame(ls_f).sort_index()
    for mat, out in ((oi, OUT_OI), (ll, OUT_LL), (ls, OUT_LS)):
        mat.index = pd.to_datetime(mat.index)
        mat.to_csv(out)
        print(f"[coinglass] wrote {mat.shape} -> {out}")
    print(f"[coinglass] OI non-null days/symbol:\n{oi.notna().sum()}")


if __name__ == "__main__":
    main()
