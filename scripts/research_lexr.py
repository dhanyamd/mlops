#!/usr/bin/env python3
"""LEXR - Leverage-Exhaustion Reversal.

Novel factor invented from first principles + 2025-26 cascade-physics literature
(arxiv 2607/2608: liquidation cascades are SUBCRITICAL, branching lambda~0.1-0.2,
front-loaded 88% forced selling in 30min -> forced flow is a one-off EXHAUST, not a
self-reinforcing trend). Bitbase positioning model: "the most underused signal is the
liquidation imbalance." Mispricing/JFQA: reversal is the #1 weekly-alpha driver in
small/illiquid coins.

Mechanism: when leverage is FORCIBLY flushed (OI collapses) while the crowd was
crowded on the losing side, subcritical physics says the forced sellers are GONE and
price mean-reverts. Gate REVERSAL on the leverage-flush signature (OI/long-short
ratio) -- NOT on funding (funding ~= -momentum, which made ASYM/FVCS/REVX spanned).

Inputs are keyless Binance futures-data: openInterestHist (USD) + topLongShortAccountRatio.
NOTE: Binance caps these at 31 daily points (startTime rejected). This is a
PROOF-OF-ORTHOGONALITY test (cross-sectional rank-IC + spanning residual), not a
multi-year profitability claim. A real test needs the CoinGlass key (deferred).
"""

import json, math, os, sys, time, urllib.request, datetime as dt
import numpy as np
import pandas as pd

OUT = "/tmp/lexr_pull"
os.makedirs(OUT, exist_ok=True)
NHIST = 31  # Binance free ceiling for OI/ratio


def get(url, tries=4):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep(0.4)
    return None


def perp_universe(n=150):
    d = get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if not d:
        return []
    d = [x for x in d if x["symbol"].endswith("USDT")]
    d.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    return [x["symbol"] for x in d[:n]]


def pull(sym):
    p = f"{OUT}/{sym}.json"
    if os.path.exists(p):
        return
    oi = get(
        f"https://fapi.binance.com/futures/data/openInterestHist?symbol={sym}&period=1d&limit={NHIST}"
    )
    lsr = get(
        f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={sym}&period=1d&limit={NHIST}"
    )
    kl = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d&limit={NHIST + 7}")
    if not (oi and lsr and kl):
        return
    oi_v = [float(x["sumOpenInterestValue"]) for x in oi]
    oi_ts = [x["timestamp"] for x in oi]
    lsr_v = [float(x["longShortRatio"]) for x in lsr]
    lsr_ts = [x["timestamp"] for x in lsr]
    close = [float(k[4]) for k in kl]
    kl_ts = [k[6] for k in kl]
    rec = {
        "oi": oi_v,
        "oi_ts": oi_ts,
        "lsr": lsr_v,
        "lsr_ts": lsr_ts,
        "close": close,
        "kl_ts": kl_ts,
    }
    with open(p, "w") as f:
        json.dump(rec, f)


def winsor_rank_z(s: pd.Series) -> pd.Series:
    x = s.rank(pct=True)
    z = (x - 0.5) * 2
    return z.clip(-1, 1)


def build_panel():
    rows = {}
    for fn in os.listdir(OUT):
        if not fn.endswith(".json"):
            continue
        sym = fn[:-5]
        try:
            with open(f"{OUT}/{fn}") as f:
                r = json.load(f)
        except Exception:
            continue
        if len(r["close"]) < NHIST + 2:
            continue
        close = pd.Series(
            r["close"], index=pd.to_datetime(r["kl_ts"], unit="ms").floor("D")
        ).sort_index()
        oi = pd.Series(r["oi"], index=pd.to_datetime(r["oi_ts"], unit="ms").floor("D")).sort_index()
        lsr = pd.Series(
            r["lsr"], index=pd.to_datetime(r["lsr_ts"], unit="ms").floor("D")
        ).sort_index()
        df = pd.DataFrame({"close": close, "oi": oi, "lsr": lsr})
        # keep rows where we have OI + LSR (the leverage signal)
        df = df.dropna(subset=["oi", "lsr"]).dropna(subset=["close"])
        if len(df) < NHIST - 2:
            continue
        rows[sym] = df
    return rows


def panel_to_frame(rows):
    closes = pd.DataFrame({s: v["close"] for s, v in rows.items()})
    ois = pd.DataFrame({s: v["oi"] for s, v in rows.items()})
    lsrs = pd.DataFrame({s: v["lsr"] for s, v in rows.items()})
    return closes, ois, lsrs


def zscore_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: winsor_rank_z(c) if c.notna().sum() > 5 else c * 0)


def spearman_ic(sig: pd.DataFrame, fwd: pd.DataFrame) -> float:
    ics = []
    for d in sig.index:
        a = sig.loc[d].dropna()
        b = fwd.loc[d].reindex(a.index).dropna()
        a = a.reindex(b.index).dropna()
        if len(a) < 10:
            continue
        # rank corr
        ar = a.rank()
        br = b.rank()
        da = ar - ar.mean()
        db = br - br.mean()
        den = math.sqrt((da**2).sum() * (db**2).sum())
        if den <= 0:
            continue
        ics.append(float((da * db).sum() / den))
    return float(np.mean(ics)) if ics else float("nan")


def ortho_resid(target: pd.DataFrame, controls: list[pd.DataFrame]) -> pd.DataFrame:
    resid = target.copy()
    for d in target.index:
        y = target.loc[d].dropna()
        if len(y) < 10:
            continue
        X = []
        for c in controls:
            cc = c.loc[d].reindex(y.index)
            X.append(cc.values)
        X = np.array(X).T
        mask = ~np.isnan(X).any(axis=1)
        if mask.sum() < 10:
            continue
        yy = y.values[mask]
        XX = X[mask]
        try:
            beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
            pred = XX @ beta
            r = yy - pred
            resid.loc[d, y.index[mask]] = r
        except Exception:
            pass
    return resid


def main():
    print("== universe ==")
    syms = perp_universe(150)
    print(f"{len(syms)} perps")
    print("== pulling OI/long-short/price (keyless, resumable) ==")
    for i, s in enumerate(syms):
        pull(s)
        if i % 25 == 0:
            print(f"  {i}/{len(syms)}")
    rows = build_panel()
    print(f"== panel coins: {len(rows)} ==")
    if len(rows) < 20:
        print("TOO FEW COINS")
        sys.exit(1)
    closes, ois, lsrs = panel_to_frame(rows)
    rets = closes.pct_change()
    # forward 7d return
    fwd7 = closes.shift(-7).div(closes) - 1.0
    # formation 7d return (recent move)
    rec7 = closes.shift(1).div(closes.shift(8)) - 1.0
    # OI flush: negative delta log OI over 7d
    oi_flush = -(ois.shift(1).div(ois.shift(8)) - 1.0)
    # crowd: long/short ratio level (high = crowded long)
    crowd = lsrs
    # momentum (MHT-like): multi-horizon past return
    mom = (closes.shift(1).div(closes.shift(8)) - 1.0) * 0.5 + (
        closes.shift(1).div(closes.shift(15)) - 1.0
    ) * 0.5
    # plain reversal: -recent return
    rev_plain = -rec7

    # zscore cross-sectionally per date
    def z(df):
        return zscore_cols(df)

    oi_flush_z = z(oi_flush)
    crowd_z = z(crowd)
    rec7_z = z(rec7)
    mom_z = z(mom)
    rev_z = z(rev_plain)

    # LEXR = fade the forced move when leverage exhausted & crowd wrong-way
    # signal sign: bet opposite to recent move (-rec7_z) when oi_flush high AND crowd long
    lext = (-rec7_z) * oi_flush_z * crowd_z
    lext = z(lext)

    print("\n=== CROSS-SECTIONAL RANK-IC (signal vs fwd 7d return) ===")
    print(f"LEXR       IC = {spearman_ic(lext, fwd7):+.4f}")
    print(f"MOMENTUM   IC = {spearman_ic(mom_z, fwd7):+.4f}")
    print(f"REV_PLAIN  IC = {spearman_ic(rev_z, fwd7):+.4f}")

    print("\n=== SPANNING: LEXR residualized on [MOMENTUM, REV_PLAIN] ===")
    lexr_resid = ortho_resid(lext, [mom_z, rev_z])
    print(f"LEXR raw        IC = {spearman_ic(lext, fwd7):+.4f}")
    print(f"LEXR | MOM,REV  IC = {spearman_ic(lexr_resid, fwd7):+.4f}")

    # also test correlations
    print("\n=== rank-corr of LEXR with known factors (cross-sectional mean) ===")

    def mc(a, b):
        out = []
        for d in a.index:
            x = a.loc[d].dropna()
            y = b.loc[d].reindex(x.index).dropna()
            x = x.reindex(y.index).dropna()
            if len(x) < 10:
                continue
            xr = x.rank()
            yr = y.rank()
            dx = xr - xr.mean()
            dy = yr - yr.mean()
            den = math.sqrt((dx**2).sum() * (dy**2).sum())
            if den > 0:
                out.append(float((dx * dy).sum() / den))
        return float(np.mean(out)) if out else float("nan")

    print(f"LEXR ~ MOMENTUM = {mc(lext, mom_z):+.3f}")
    print(f"LEXR ~ REV_PLAIN= {mc(lext, rev_z):+.3f}")
    print(f"MOM  ~ REV      = {mc(mom_z, rev_z):+.3f}")


if __name__ == "__main__":
    main()
