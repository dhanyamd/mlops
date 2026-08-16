#!/usr/bin/env python3
"""Pull Hyperliquid perp data (keyless) for an independent OOS test of OUR FAS factor.

Endpoints (no API key):
  metaAndAssetCtxs -> universe + current openInterest/funding per coin
  fundingHistory   -> 8h funding (capped 500/call, paginate via startTime)
  candleSnapshot   -> daily candles (req wrapper; 5000-cap)

Saves to /tmp/hl_pull/{COIN}_funding.json  (list of {t, r})
                          {COIN}_candles.json  (list of daily {t,c,v})
                /tmp/hl_pull/oi_snapshot.json (current OI per coin)
                /tmp/hl_pull/manifest.json

Resumable: skips coins already fully pulled.
"""

import json, os, time, urllib.request

OUT = "/tmp/hl_pull"
os.makedirs(OUT, exist_ok=True)
URL = "https://api.hyperliquid.xyz/info"
HDR = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
START = 1714521600000  # 2024-05-01
TOP_N = 20  # most-liquid perps by 24h notional volume


def post(payload, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDR)
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f)


def main():
    m = post({"type": "metaAndAssetCtxs"})
    universe = m[0]["universe"]
    ctx = m[1]
    # ctx[i] parallels universe[i]; rank perps by 24h notional volume
    paired = [
        (universe[i]["name"], float(ctx[i].get("dayNtlVlm", 0))) for i in range(len(universe))
    ]
    paired.sort(key=lambda x: x[1], reverse=True)
    coins = [nm for nm, _ in paired[:TOP_N]]
    print(f"pulling {len(coins)} perps: {coins[:8]} ...")

    save(
        "oi_snapshot.json",
        {universe[i]["name"]: ctx[i].get("openInterest") for i in range(len(universe))},
    )

    manifest = {}
    for coin in coins:
        fpath = os.path.join(OUT, f"{coin}_funding.json")
        cpath = os.path.join(OUT, f"{coin}_candles.json")
        if os.path.exists(fpath) and os.path.exists(cpath):
            print(f"  skip {coin} (exists)")
            continue
        # --- funding (paginate) ---
        fund = []
        t = START
        while True:
            page = post({"type": "fundingHistory", "coin": coin, "startTime": t})
            if not page:
                break
            fund.extend([{"t": x["time"], "r": float(x["fundingRate"])} for x in page])
            if len(page) < 500:
                break
            t = page[-1]["time"] + 1
            time.sleep(0.5)
        # --- daily candles ---
        candles = post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": "1d",
                    "startTime": START,
                    "endTime": int(time.time() * 1000),
                },
            }
        )
        candles = [{"t": x["t"], "c": float(x["c"]), "v": float(x["v"])} for x in candles]
        if fund and candles:
            save(f"{coin}_funding.json", fund)
            save(f"{coin}_candles.json", candles)
            manifest[coin] = {"n_fund": len(fund), "n_cand": len(candles)}
            print(f"  {coin}: fund={len(fund)} candles={len(candles)}")
        else:
            print(f"  {coin}: EMPTY fund={len(fund)} candles={len(candles)}")
        time.sleep(0.4)
    save("manifest.json", manifest)
    print("DONE. manifest:", manifest)


if __name__ == "__main__":
    main()
