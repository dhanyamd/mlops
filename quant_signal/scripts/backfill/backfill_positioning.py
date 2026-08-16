"""Binance positioning metrics -- open interest and whale-vs-retail crowding.

WHY THIS DATA IS DIFFERENT FROM EVERYTHING ELSE WE HAVE
-------------------------------------------------------
Every "smart money" factor built in this project so far has to INFER who was
informed. 开源证券's 聪明钱 infers it from price impact per unit volume; our
IFD infers it from the aggressor side at high-impact bars. Inference is why
those factors are weak.

Binance publishes the answer directly. The metrics feed carries, at 5-minute
resolution since 2020-09:

    sum_open_interest                  total OI, base units
    sum_open_interest_value            total OI, USD
    sum_toptrader_long_short_ratio     TOP accounts, by POSITION SIZE
    count_toptrader_long_short_ratio   TOP accounts, by ACCOUNT COUNT
    count_long_short_ratio             ALL accounts, by account count
    sum_taker_long_short_vol_ratio     taker buy/sell volume ratio

The gap between what the LARGEST accounts hold and what ALL accounts hold is an
observed whale-vs-retail divergence -- no inference, no proxy. In commodities
this is COT data, published weekly with a multi-day lag, and the Chinese CTA
literature treats 大户持仓 as a core factor ("基于期货的期限结构、大户持仓这两个
因子", used as 过滤器或增强器). Here it is per-symbol and near-real-time.

THE CONSTRUCTION, AND THE WARNING THAT SHAPES IT
------------------------------------------------
The same literature is explicit that RAW open-interest change does not work:

    持仓量变化率测试结果不理想…因为期货市场是双边交易，单边持仓变化可能对应
    多空两种情况

Every contract has a long and a short, so rising OI alone says nothing about
direction. OI becomes informative only INTERACTED WITH PRICE -- the 四象限
(four-quadrant) framing:

    price up   + OI up    价涨仓增   new longs opening      -> continuation
    price up   + OI down  价涨仓减   shorts covering        -> weak rally
    price down + OI up    价跌仓增   new shorts opening     -> continuation
    price down + OI down  价跌仓减   longs liquidating      -> exhaustion

So the factor stored here is the raw series; the quadrant interaction and the
whale-retail spread are formed downstream where they stay visible.

SCOPE
-----
Only DAILY files exist (no monthly archive), so a full pull is 112 symbols x
~2170 days ~= 243k requests. This fetches ONE FILE PER WEEK (the Monday that
matches the research grid's W-MON label), which is ~35k requests and enough to
falsify the factor before committing to the full download. Weekly resolution is
sufficient because the quadrant construction compares price change and OI change
over the SAME interval, and the book rebalances weekly.

Run:
    uv run python -m scripts.backfill_positioning --symbols-from /tmp/quant_cache/fas_broad.json
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
COLS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def stamps(start: str, end: str, step_days: int = 7) -> list[str]:
    """Weekly (Monday-anchored) by default; step_days=1 for the full daily pull."""
    d = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    if step_days == 7:
        d += timedelta(days=(7 - d.weekday()) % 7)      # anchor to Monday
    last = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    out = []
    while d <= last:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=step_days)
    return out


def fetch_day(sym: str, day: str) -> dict | None:
    """Last observation of the day -- the point-in-time value at that stamp."""
    url = f"{BASE}/{sym}/{sym}-metrics-{day}.zip"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                              # not listed that day
            time.sleep(0.6 * (attempt + 1))
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    else:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            rows = list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode())))
    except Exception:
        return None
    if not rows:
        return None
    last = rows[-1]
    rec: dict = {"day": day}
    for c in COLS:
        try:
            v = float(last.get(c, ""))
            rec[c] = v if v == v else None               # NaN guard
        except (TypeError, ValueError):
            rec[c] = None
    # daily means too -- a single 23:55 print is noisy for ratio fields
    for c in ("sum_toptrader_long_short_ratio", "count_long_short_ratio"):
        vals = []
        for r_ in rows:
            try:
                vals.append(float(r_[c]))
            except (KeyError, TypeError, ValueError):
                pass
        rec[c + "_mean"] = sum(vals) / len(vals) if vals else None
    return rec


def do_symbol(sym: str, days: list[str], outdir: Path, day_workers: int = 1) -> tuple[str, int]:
    f = outdir / f"{sym}.json"
    if f.exists():
        try:
            return sym, len(json.loads(f.read_text()))
        except Exception:
            pass
    if day_workers > 1:
        # a full daily pull is ~2200 requests per symbol; fetching them serially
        # would take hours, and they are independent
        with ThreadPoolExecutor(max_workers=day_workers) as dx:
            got = list(dx.map(lambda d: fetch_day(sym, d), days))
        recs = [r for r in got if r]
    else:
        recs = [r for r in (fetch_day(sym, d) for d in days) if r]
    recs.sort(key=lambda r: r["day"])
    if recs:
        f.write_text(json.dumps(recs))
    return sym, len(recs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-from", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default="2020-09-01")
    ap.add_argument("--end", default="2026-08-10")
    ap.add_argument("--out", default="/tmp/quant_cache/positioning")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--day-workers", type=int, default=1,
                    help="parallel day fetches per symbol; raise for the full daily pull")
    ap.add_argument("--daily", action="store_true",
                    help="fetch EVERY day instead of one stamp per week")
    a = ap.parse_args()

    syms = ([s.strip().upper() for s in a.symbols.split(",") if s.strip()]
            if a.symbols else sorted(json.load(open(a.symbols_from))["bars"]))
    days = stamps(a.start, a.end, 1 if a.daily else 7)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(syms)} symbols x {len(days)} weekly stamps = {len(syms)*len(days):,} requests")
    print(f"-> {outdir}\n")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for sym, n in ex.map(lambda s: do_symbol(s, days, outdir, a.day_workers), syms):
            done += 1
            print(f"  [{done:3d}/{len(syms)}] {sym:12} {n:4d} weeks  ({time.time()-t0:5.0f}s)",
                  flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
