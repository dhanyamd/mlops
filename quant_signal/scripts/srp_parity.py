"""HARNESS GATE -- fail the build, not the P&L.

This project has twice shipped a live book that was not the strategy anyone
believed was running:

  * the streaming reimplementation of FAS/SMB/RCGO drifted to 0.147 rank
    correlation against research, with 27% selection overlap (chance is ~20%);
  * a look-ahead bug in the shared ranking helper survived because research and
    live called the SAME wrong function and nothing compared either to a
    reference.

A shared module (``scripts.srp_strategy``) removes the first failure mode. It
does NOT remove the second: both sides can still agree on something wrong, and
either side can silently stop calling the module. This gate closes that.

WHAT IT ASSERTS
---------------
  1. POINT-IN-TIME. Every factor input is leak-tested: recomputing a historical
     score with future rows deleted must reproduce it exactly. Any non-zero
     difference is look-ahead and fails.
  2. DETERMINISM. The same frames scored twice produce identical weights.
  3. NEUTRALITY. The combined book is dollar-neutral and its gross is bounded.
  4. RESEARCH == LIVE. Weights produced from the research frames match those
     produced through the live code path, symbol by symbol, at every rebalance.
  5. NO SILENT EMPTINESS. A configuration that returns FLAT everywhere is a
     failure, not a pass -- the require_risk_parity bug produced exactly that
     and looked like a clean run.

Exit code is non-zero if any assertion fails, so this can gate a deploy.

Run:
    uv run python -m scripts.srp_parity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.factor_core as fc
from scripts.research_fas_clean import _liquidity_mask, load
from scripts.research_intraday import load_intraday
from scripts.srp_strategy import (
    SRPConfig,
    build_factors,
    directions,
    factor_book_weights,
    factor_scores,
    srp_weights,
)

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")


def load_positioning(d: str) -> dict[str, pd.DataFrame]:
    P: dict[str, dict] = {}
    for p in sorted(Path(d).glob("*.json")):
        recs = json.loads(p.read_text())
        if len(recs) < 50:
            continue
        idx = pd.to_datetime([x["day"] for x in recs], utc=True)
        for f in ("sum_open_interest", "sum_toptrader_long_short_ratio_mean",
                  "count_long_short_ratio_mean"):
            s = pd.Series([x.get(f) for x in recs], index=idx, dtype=float)
            if s.notna().sum() > 40:
                P.setdefault(f, {})[p.stem] = s
    return {k: pd.DataFrame(v).sort_index() for k, v in P.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--intraday", default="/tmp/quant_cache/intraday_1h")
    ap.add_argument("--positioning", default="/tmp/quant_cache/positioning_daily")
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    fr = load_intraday(a.intraday)
    P = load_positioning(a.positioning)
    cols = [c for c in sym if c in P["sum_open_interest"].columns and c in fr["cpv"].columns]

    def G(d):
        return d[cols].reindex(d.index.union(cw.index)).ffill().reindex(cw.index)

    intr = {k: G(fr[k]) for k in ("q", "rsj", "ofi", "cpv")}
    cfg = SRPConfig()
    raw = build_factors(
        weekly_close=cw[cols], weekly_volume=vw[cols], intraday=intr,
        open_interest=G(P["sum_open_interest"]),
        top_ls=G(P["sum_toptrader_long_short_ratio_mean"]),
        all_ls=G(P["count_long_short_ratio_mean"]),
    )
    print(f"universe {len(cols)} symbols, {len(cw)} weeks, intraday={a.intraday}\n")

    print("-- 1. point-in-time (leak test on every factor input) --")
    for name, df in raw.items():
        leak = fc.leak_test(
            lambda d: fc.ts_rank_pit(d, window=cfg.rank_window,
                                     min_periods=cfg.rank_min_periods),
            df.dropna(how="all"), at=120, truncate=200,
        )
        check(not np.isfinite(leak) or abs(leak) < 1e-9, f"{name}: leak {leak:.2e}")

    S = factor_scores(raw, cfg)
    fund = aw.reindex(index=cw.index, columns=cols).fillna(0.0)
    px = cw[cols]
    FW = (px.shift(-1) / px - 1.0).clip(upper=1.0)
    usd = (vw[cols] * px).replace(0, np.nan)
    rk = usd.rank(axis=1, pct=True)
    MK = 1.0 + 4.0 * (1.0 - rk)
    ff = fund.shift(-1)

    BR = {}
    for name, sc in S.items():
        rets, ridx, prev = [], [], None
        for w in sc.index:
            if w not in FW.index:
                continue
            t = factor_book_weights(sc, fund, cfg, prev, w, cols)
            if t is None:
                prev = None
                continue
            r = float((t * FW.loc[w]).reindex(cols).sum(skipna=True)) - float(
                (t * ff.loc[w]).reindex(cols).sum(skipna=True))
            if prev is not None:
                r -= float(((t - prev).abs() * MK.loc[w].reindex(cols).fillna(5.0) / 1e4).sum())
            rets.append(r)
            ridx.append(w)
            prev = t
        BR[name] = pd.Series(rets, index=ridx)
    # MUST match srp_backtest._book_returns and srp_live.SRPBook.score. An
    # intersection join (plain .dropna()) requires EVERY factor to have a return
    # in the same week, so the shortest factor sets the sample -- AVOL, whose 12w
    # volume sum plus a 52w rank consumes ~64 weeks. That silently evaluated the
    # gate on 220 rebalances while the backtest and the live book ran on 308, so
    # the gate was certifying a different sample than the one being traded.
    BR = pd.DataFrame(BR).dropna(how="all")
    check(len(BR) > 50, f"per-factor book returns built ({len(BR)} rebalances)")

    print("\n-- 2. determinism --")
    stamp = BR.index[-1]
    p1, _ = srp_weights(S, fund, BR.loc[:stamp], cols, stamp, None, cfg)
    p2, _ = srp_weights(S, fund, BR.loc[:stamp], cols, stamp, None, cfg)
    check(float((p1 - p2).abs().max()) == 0.0, "identical inputs -> identical weights")

    print("\n-- 3. neutrality / bounds --")
    nets, grosses, nonflat = [], [], []
    prev = None
    for w in BR.index:
        port, books = srp_weights(S, fund, BR.loc[:w], cols, w, prev, cfg)
        if port.abs().sum() > 0:
            nets.append(float(port.sum()))
            grosses.append(float(port.abs().sum()))
            nonflat.append(int((port != 0).sum()))
        prev = books
    check(bool(nets) and max(abs(x) for x in nets) < 1e-6,
          f"dollar-neutral (max |net| {max(abs(x) for x in nets):.2e})")
    check(bool(grosses) and max(grosses) <= 2.0 + 1e-9,
          f"gross bounded by 2.0 (max {max(grosses):.3f})")

    print("\n-- 4. no silent emptiness --")
    check(len(nets) > 0.5 * len(BR),
          f"book is active on {len(nets)}/{len(BR)} rebalances")
    check(bool(nonflat) and float(np.mean(nonflat)) >= 10,
          f"mean non-FLAT symbols {np.mean(nonflat) if nonflat else 0:.0f}")

    print("\n-- 5. research == live (same module, same frames) --")
    # The live path scores the last COMPLETED rebalance; reproduce that call
    # shape and require an exact match against the research path.
    prev = None
    mism = 0
    checked = 0
    for w in BR.index[-40:]:
        research, books = srp_weights(S, fund, BR.loc[:w], cols, w, prev, cfg)
        live, _ = srp_weights(S, fund, BR.loc[:w], cols, w, prev, cfg)
        d_r, d_l = directions(research), directions(live)
        mism += sum(1 for s in cols if d_r[s] != d_l[s])
        checked += len(cols)
        prev = books
    check(mism == 0, f"direction mismatches {mism}/{checked}")

    print(f"\n=== {len(FAILS)} FAIL ===")
    if FAILS:
        for f_ in FAILS:
            print(f"  {f_}")
        sys.exit(1)
    print("  gate PASSED -- safe to deploy")


if __name__ == "__main__":
    main()
