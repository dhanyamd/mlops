"""Daily OHLCV backfill -- the inputs Alpha191 needs and our cache lacks.

The existing warm cache stores only [close_time, close, volume]. 国泰君安's
Alpha191 library (《基于短周期价量特征的多因子选股体系》, 2017) is built entirely
from daily OPEN/HIGH/LOW/CLOSE/VOLUME plus VWAP, so most of its 191 formulas
cannot be evaluated against what we have. This pulls the full bar.

Also stored: quote_volume (to derive VWAP = quote_volume / volume, which many
Alpha191 formulas reference directly), trade count, and taker-buy volume.

Source is the same bulk archive used elsewhere (data.binance.vision monthly
klines), which is far faster than paginating REST and needs no key.

Run:
    uv run python -m scripts.backfill_ohlcv --symbols-from /tmp/quant_cache/fas_broad.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


def months(start: str, end: str) -> list[str]:
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch(sym: str, mo: str) -> list[list[str]] | None:
    url = f"{BASE}/{sym}/1d/{sym}-1d-{mo}.zip"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(0.8 * (attempt + 1))
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    else:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        rows = list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())))
    if rows and not rows[0][0].isdigit():
        rows = rows[1:]
    return rows


def do_symbol(sym: str, mos: list[str]) -> tuple[str, list[list]]:
    recs: dict[int, list] = {}
    for mo in mos:
        rows = fetch(sym, mo)
        if not rows:
            continue
        for r in rows:
            try:
                ct = int(r[6])
                o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                v, qv = float(r[5]), float(r[7])
                n, tb = float(r[8]), float(r[9])
            except (ValueError, IndexError):
                continue
            if v <= 0 or c <= 0:
                continue
            recs[ct] = [ct, o, h, l, c, v, qv, n, tb]
    return sym, [recs[k] for k in sorted(recs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-from", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--start", default="2019-09")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--out", default="/tmp/quant_cache/ohlcv_daily.json")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    syms = sorted(json.load(open(a.symbols_from))["bars"])
    mos = months(a.start, a.end)
    print(f"{len(syms)} symbols x {len(mos)} months -> {a.out}")
    out: dict[str, list] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (sym, rows) in enumerate(ex.map(lambda s: do_symbol(s, mos), syms), 1):
            if rows:
                out[sym] = rows
            print(f"  [{i:3d}/{len(syms)}] {sym:12} {len(rows):5d} days ({time.time()-t0:4.0f}s)",
                  flush=True)
    Path(a.out).write_text(json.dumps(
        {"ts": int(time.time() * 1000),
         "schema": ["close_time_ms", "open", "high", "low", "close",
                    "volume", "quote_volume", "trades", "taker_buy_volume"],
         "bars": out}))
    print(f"\nwrote {a.out}: {len(out)} symbols in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
