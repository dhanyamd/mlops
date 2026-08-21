"""Pull keyless CoinGecko market data to tag the broad universe by size / liquidity.

Gives us what Binance public REST cannot: MARKET CAP, rank, FDV, volume.
This powers the small-cap x high-range x low-vol family (agentic LLM paper, +2 Sharpe).

Outputs:
  /tmp/cg_markets.csv            current snapshot: id,symbol,name,market_cap,rank,volume,fdv,chg7d,chg30d
  /tmp/cg_mcap_history/<ID].csv  daily historical market_cap per coin (resumable)

Keyless: https://api.coingecko.com/api/v3  (no header, no key). ~30 req/min; backoff on 429.
Note: keyless historical daily caps at ~1y; a free Demo key lifts it to 2y. Fine for forward test.
"""

import glob
import os
import sys
import time

import pandas as pd
import requests

BASE = "https://api.coingecko.com/api/v3"
SNAP = "/tmp/cg_markets.csv"
HIST_DIR = "/tmp/cg_mcap_history"
BROAD_DIR = "/tmp/broad_pull"


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}


def _get(url, params, tries=5):
    for i in range(tries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(20 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {url}")


def pull_snapshot():
    if os.path.exists(SNAP) and "--refresh" not in sys.argv:
        print(f"[cg] snapshot exists: {SNAP} ({len(pd.read_csv(SNAP))} rows)")
        return pd.read_csv(SNAP)
    rows = []
    page = 1
    while True:
        data = _get(
            f"{BASE}/coins/markets",
            {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
            },
        )
        if not data:
            break
        for c in data:
            rows.append(
                {
                    "id": c["id"],
                    "symbol": (c.get("symbol") or "").upper(),
                    "name": c.get("name"),
                    "market_cap": c.get("market_cap"),
                    "rank": c.get("market_cap_rank"),
                    "volume": c.get("total_volume"),
                    "fdv": c.get("fully_diluted_valuation"),
                    "chg7d": c.get("price_change_percentage_7d_in_currency"),
                    "chg30d": c.get("price_change_percentage_30d_in_currency"),
                }
            )
        print(f"[cg] page {page}: {len(rows)} coins so far")
        if len(data) < 250:
            break
        page += 1
        time.sleep(2)
    df = pd.DataFrame(rows).dropna(subset=["market_cap"])
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    df.to_csv(SNAP, index=False)
    print(f"[cg] snapshot saved: {SNAP} ({len(df)} rows)")
    return df


def pull_history(snap, days=365):
    os.makedirs(HIST_DIR, exist_ok=True)
    # map broad-universe binance symbols -> coingecko id
    sym2id = {s: i for s, i in zip(snap["symbol"], snap["id"])}
    files = sorted(glob.glob(os.path.join(BROAD_DIR, "*.csv")))
    print(f"[cg] {len(files)} broad coins; history dir has {len(os.listdir(HIST_DIR))} done")
    done = 0
    for f in files:
        sym = os.path.basename(f)[:-4].replace("USDT", "")
        cid = sym2id.get(sym)
        if not cid:
            continue
        out = os.path.join(HIST_DIR, f"{cid}.csv")
        if os.path.exists(out):
            done += 1
            continue
        try:
            d = _get(f"{BASE}/coins/{cid}/market_chart", {"vs_currency": "usd", "days": days})
        except Exception as e:
            print(f"[cg] skip {sym}: {e}")
            continue
        mc = d.get("market_caps") or []
        if not mc:
            time.sleep(2)
            continue
        dd = pd.DataFrame(mc, columns=["ts", "market_cap"])
        dd["date"] = pd.to_datetime(dd["ts"], unit="ms").dt.strftime("%Y-%m-%d")
        dd[["date", "market_cap"]].to_csv(out, index=False)
        done += 1
        if done % 25 == 0:
            print(f"[cg] history {done}/{len(files)}")
        time.sleep(2.5)
    print(f"[cg] history done: {done} coins -> {HIST_DIR}")


if __name__ == "__main__":
    snap = pull_snapshot()
    if "--history" in sys.argv:
        pull_history(snap)
