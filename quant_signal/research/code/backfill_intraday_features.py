"""Intraday-derived DAILY factors -- the 高频数据、低频因子 methodology.

WHY
---
Every factor this project has tested was computed from DAILY bars, and every one
of them landed between 0.2 and 0.95 Sharpe. That is not bad luck, it is a
ceiling. By the fundamental law of active management, IR ~ IC * sqrt(breadth);
published crypto cross-sectional ICs top out near 0.05, and crypto's effective
breadth is far below its name count because everything co-moves with BTC. A
weekly cross-sectional book built on daily bars therefore caps around IR ~ 1.

The Chinese sell-side quant literature raises that ceiling a specific way, and
states the principle outright: 高频数据、低频因子 -- compute the factor from
high-frequency bars, then REBALANCE SLOWLY. The information lives intraday; the
trading stays cheap. Reported results are far above what daily bars produce,
e.g. 东吴证券's CPV at ICIR -3.77 / IR 3.03 / 87% monthly win rate.

This script extracts that intraday information into daily per-symbol series.

WHAT IT COMPUTES, AND WHERE EACH COMES FROM
-------------------------------------------
  cpv     东吴证券 (高子剑/沈芷琦), 高频价量相关性 CPV.
          Daily corr(close, volume) across the day's intraday bars. The report
          then aggregates this daily series over a month along three dimensions
          -- mean, volatility, trend -- and combines them. We store the raw
          daily correlation here; the three aggregates are formed downstream so
          the aggregation choice stays visible rather than baked in.

  q       方正/开源证券, 聪明钱因子. Per bar S_t = |R_t| / sqrt(V_t); sort bars
          by S descending; accumulate until 20% of the day's volume is covered;
          those are the "smart money" bars. Q = VWAP_smart / VWAP_all. Q < 1
          means informed flow transacted BELOW the day's average price.

  ofi     Order-flow imbalance, (2*taker_buy - volume)/volume. The crypto
          analogue of 天风证券's 买卖压力失衡. Perp klines carry taker-buy
          volume directly, so signed flow needs no tick-rule inference -- this
          is the one input a centralised crypto venue gives away that equity
          researchers have to estimate.

  rv      Realised volatility, sqrt(sum of squared intraday returns).
  rsj     国泰君安 realised-skewness / signed-jump: (sum r+^2 - sum r-^2) / rv^2.
          Separates upside from downside realised variance.
  ntrd    Trade count, and
  avgtrd  mean volume per trade -- 开源证券's 分钟单笔金额 (per-trade size) is
          built on exactly this, as a retail-vs-institutional participation
          proxy.

BAR CHOICE
----------
5-minute bars. A-share minute factors are calibrated on a 4-hour session, ~240
bars/day. Crypto trades 1440 minutes/day, so 1-minute bars would give 6x the
sample these factors were designed around -- 5-minute bars give 288/day, the
closest match to the literature. It is also 1/5 the download.

Source is Binance's bulk archive (data.binance.vision), monthly ZIPs, which is
far faster than paginating the REST klines endpoint and needs no key. Raw bars
are DISCARDED after each month is reduced to daily values, so this stays small
and resumable: completed symbols are skipped on re-run.

Run:
    uv run python -m scripts.backfill_intraday_features --symbols-from /tmp/quant_cache/fas_broad.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
DAY_MS = 86_400_000


def month_list(start: str, end: str) -> list[str]:
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch_month(sym: str, mo: str, interval: str) -> list[list[str]] | None:
    url = f"{BASE}/{sym}/{interval}/{sym}-{interval}-{mo}.zip"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # symbol not listed that month; not an error
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    else:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode()
    rows = list(csv.reader(io.StringIO(raw)))
    if rows and not rows[0][0].isdigit():
        rows = rows[1:]              # newer archives carry a header row
    return rows


def daily_features(rows: list[list[str]]) -> dict[int, dict]:
    """Reduce intraday bars to one feature record per UTC day."""
    days: dict[int, list] = {}
    for r in rows:
        try:
            t = int(r[0])
            days.setdefault(t // DAY_MS, []).append(
                (float(r[4]), float(r[5]), float(r[8]), float(r[9]), t)  # close, vol, n, takerbuy, ts
            )
        except (ValueError, IndexError):
            continue

    out = {}
    for d, bars in days.items():
        if len(bars) < 20:           # a day too sparse to carry intraday structure
            continue
        cl = [b[0] for b in bars]
        vo = [b[1] for b in bars]
        nt = [b[2] for b in bars]
        tb = [b[3] for b in bars]
        tot_v = sum(vo)
        if tot_v <= 0 or any(c <= 0 for c in cl):
            continue

        # --- CPV: corr(close, volume) across the day's bars -----------------
        n = len(bars)
        mc, mv = sum(cl) / n, tot_v / n
        num = sum((c - mc) * (v - mv) for c, v in zip(cl, vo))
        dc = math.sqrt(sum((c - mc) ** 2 for c in cl))
        dv = math.sqrt(sum((v - mv) ** 2 for v in vo))
        cpv = num / (dc * dv) if dc > 0 and dv > 0 else None

        # --- smart money Q: S = |R|/sqrt(V), top 20% of volume --------------
        rets = [0.0] + [cl[i] / cl[i - 1] - 1.0 for i in range(1, n)]
        scored = [
            (abs(rets[i]) / math.sqrt(vo[i]), cl[i], vo[i], tb[i])
            for i in range(n)
            if vo[i] > 0
        ]
        q = ifd = None
        if scored:
            scored.sort(key=lambda x: -x[0])
            cum, pv, pw, sm_tb = 0.0, 0.0, 0.0, 0.0
            target = 0.20 * sum(s[2] for s in scored)
            for _, c, v, b in scored:
                pv += c * v
                pw += v
                sm_tb += b
                cum += v
                if cum >= target:
                    break
            vwap_all = sum(c * v for _, c, v, _b in scored) / sum(s[2] for s in scored)
            if pw > 0 and vwap_all > 0:
                q = (pv / pw) / vwap_all
            # INVENTED (this project): Informed Flow Divergence. 开源证券's smart
            # money rule selects WHICH bars were informed but is sign-blind,
            # because A-share data carries no trade direction. Perp klines give
            # the aggressor side outright, so we can ask whether the informed
            # bars were net BUYING or SELLING relative to the day's baseline.
            # Positive => informed flow accumulated; negative => distributed.
            if pw > 0 and tot_v > 0:
                ifd = (2.0 * sm_tb - pw) / pw - (2.0 * sum(tb) - tot_v) / tot_v

        # --- signed flow, realised vol, signed jump -------------------------
        ofi = (2.0 * sum(tb) - tot_v) / tot_v

        # INVENTED (this project): Directional Kyle Asymmetry. Kyle's lambda is
        # price impact per unit volume; splitting it BY SIDE needs signed flow,
        # so the equity literature cannot form it. Positive => it takes less
        # buying to lift the price than selling to push it down, i.e. a thin
        # ask side / upward fragility.
        up = [(rets[i], tb[i]) for i in range(1, n) if rets[i] > 0 and tb[i] > 0]
        dn = [(-rets[i], vo[i] - tb[i]) for i in range(1, n)
              if rets[i] < 0 and (vo[i] - tb[i]) > 0]
        kyle = None
        if len(up) >= 5 and len(dn) >= 5:
            lu = (sum(r for r, _ in up) / len(up)) / (sum(v for _, v in up) / len(up))
            ld = (sum(r for r, _ in dn) / len(dn)) / (sum(v for _, v in dn) / len(dn))
            if lu > 0 and ld > 0:
                kyle = math.log(lu / ld)
        rr = rets[1:]
        rv2 = sum(x * x for x in rr)
        rv = math.sqrt(rv2)
        pos = sum(x * x for x in rr if x > 0)
        neg = sum(x * x for x in rr if x < 0)
        rsj = (pos - neg) / rv2 if rv2 > 0 else None

        # --- session-split price-volume correlation ------------------------
        # 东吴证券 RPV (系列研究十一) splits corr(P,V) into 日内 and 隔夜 and finds
        # OPPOSITE signs -- intraday reverses, overnight (yesterday's volume vs
        # today's price, a deliberate 错配) trends -- so the two are DIFFERENCED,
        # not averaged. Our undifferenced CPV blend scored ~0 for exactly the
        # reason they describe: the components cancel.
        #
        # Crypto has no overnight gap, so the split is by GLOBAL SESSION
        # instead: Asia 00-08, Europe 08-16, US 16-24 UTC -- which is also the
        # 8h funding settlement clock. Same structural claim (correlation means
        # different things in different segments), adapted to a 24/7 market.
        def _corr(idx: list[int]) -> float | None:
            if len(idx) < 10:
                return None
            c = [cl[i] for i in idx]
            v = [vo[i] for i in idx]
            k = len(c)
            mc2, mv2 = sum(c) / k, sum(v) / k
            nu = sum((a - mc2) * (b - mv2) for a, b in zip(c, v))
            dc2 = math.sqrt(sum((a - mc2) ** 2 for a in c))
            dv2 = math.sqrt(sum((b - mv2) ** 2 for b in v))
            return nu / (dc2 * dv2) if dc2 > 0 and dv2 > 0 else None

        sess: dict[str, list[int]] = {"asia": [], "eur": [], "us": []}
        for i, b in enumerate(bars):
            h = (b[4] % DAY_MS) // 3_600_000      # true UTC hour of the bar
            sess["asia" if h < 8 else ("eur" if h < 16 else "us")].append(i)

        out[d] = {}
        for k2, idx in sess.items():
            out[d][f"cpv_{k2}"] = _corr(idx)
            if idx:
                sv = sum(vo[i] for i in idx)
                stb = sum(tb[i] for i in idx)
                out[d][f"ofi_{k2}"] = (2.0 * stb - sv) / sv if sv > 0 else None
                out[d][f"vshare_{k2}"] = sv / tot_v

        # --- ticket-size DISTRIBUTION (开源证券 分钟单笔金额) + our directional split --
        # 开源证券 finds the alpha is in the SHAPE of the per-bar ticket-size
        # distribution (分位数/标准差/偏度/峰度, Rank ICIR 3.57), NOT its mean --
        # "分布越集中，整体右偏程度越高，股价未来表现越好".
        #
        # Their measure is DIRECTION-BLIND: A-share tick data does not reliably
        # give the aggressor side, so a large buy ticket and a large sell ticket
        # are indistinguishable to them. Perp klines carry taker-buy volume, so
        # we can split the same distribution by whether the bar was buy- or
        # sell-dominated. tskew_dir is that difference -- directional
        # institutional attention, which their market cannot produce.
        tick = [vo[i] / nt[i] for i in range(n) if nt[i] > 0 and vo[i] > 0]
        def _shape(x: list[float]) -> tuple:
            if len(x) < 20:
                return (None, None, None)
            lg = sorted(math.log(t) for t in x)
            k = len(lg)
            m = sum(lg) / k
            sd = math.sqrt(sum((t - m) ** 2 for t in lg) / k)
            if sd <= 0:
                return (None, None, None)
            sk = sum(((t - m) / sd) ** 3 for t in lg) / k
            ku = sum(((t - m) / sd) ** 4 for t in lg) / k
            return (sd, sk, ku)
        tsd, tsk, tku = _shape(tick)
        buy_t = [vo[i] / nt[i] for i in range(n)
                 if nt[i] > 0 and vo[i] > 0 and tb[i] > 0.5 * vo[i]]
        sell_t = [vo[i] / nt[i] for i in range(n)
                  if nt[i] > 0 and vo[i] > 0 and tb[i] <= 0.5 * vo[i]]
        _, bsk, _ = _shape(buy_t)
        _, ssk, _ = _shape(sell_t)
        tskew_dir = (bsk - ssk) if (bsk is not None and ssk is not None) else None

        ntrd = sum(nt)
        out[d] |= {
            "tsd": tsd,
            "tsk": tsk,
            "tku": tku,
            "tskew_dir": tskew_dir,
            "t": d * DAY_MS,
            "close": cl[-1],
            "vol": tot_v,
            "cpv": cpv,
            "q": q,
            "ofi": ofi,
            "ifd": ifd,
            "kyle": kyle,
            "rv": rv,
            "rsj": rsj,
            "ntrd": ntrd,
            "avgtrd": (tot_v / ntrd) if ntrd > 0 else None,
        }
    return out


def do_symbol(sym: str, months: list[str], interval: str, outdir: Path) -> tuple[str, int]:
    f = outdir / f"{sym}.json"
    if f.exists():
        try:
            return sym, len(json.loads(f.read_text()))
        except Exception:
            pass                      # corrupt partial write; recompute
    recs: dict[int, dict] = {}
    for mo in months:
        rows = fetch_month(sym, mo, interval)
        if rows:
            recs.update(daily_features(rows))
    if not recs:
        return sym, 0
    f.write_text(json.dumps([recs[k] for k in sorted(recs)]))
    return sym, len(recs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-from", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--symbols", default="", help="CSV override")
    ap.add_argument("--start", default="2019-09")
    ap.add_argument("--end", default="2026-07")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--out", default="/tmp/quant_cache/intraday")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    if a.symbols:
        syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    else:
        syms = sorted(json.load(open(a.symbols_from))["bars"])
    months = month_list(a.start, a.end)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{len(syms)} symbols x {len(months)} months @ {a.interval} -> {outdir}")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for sym, n in ex.map(lambda s: do_symbol(s, months, a.interval, outdir), syms):
            done += 1
            print(f"  [{done:3d}/{len(syms)}] {sym:12} {n:5d} days   "
                  f"({time.time()-t0:5.0f}s)", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()
