"""Walk-forward and rebalance-timing-luck for SRP -- the two tests that were claimed but never run.

Both of these existed only as prose. The "walk-forward 2.68-2.95" figure quoted
in earlier notes turned out to be a pair of Hill tail-index values copied from
an unrelated document (research/TAIL_AWARE_XS_MOMENTUM.md, whose Sharpes are
0.54-0.84), and the "timing luck +/-0.24 across 7 anchors" had no script, no
output and no artefact anywhere in the repository. This module produces both,
against the committed backtest, so the numbers describe the configuration that
is actually deployed.

WALK-FORWARD
------------
The in-sample Sharpe cannot separate skill from search: the configuration was
chosen knowing the whole sample. Walk-forward removes that by construction --
the config is selected on a TRAILING window and then scored on the NEXT block,
which had no influence on the choice:

    train [t-K, t)  -> pick the config with the best in-sample Sharpe
    test  [t, t+S)  -> record THAT config's returns; they are out of sample
    roll forward

Concatenating the test blocks gives a return series that is honest no matter how
much searching happened. Walk-Forward Efficiency (OOS Sharpe / in-sample Sharpe)
above ~50-60% is the usual bar for "this survives its own selection".

TIMING LUCK
-----------
Hoffstein, Sibears & Faber (2018): a weekly strategy that always rebalances on
the same weekday carries an unmeasured exposure to WHICH weekday it picked, and
the dispersion across anchors is often larger than the effect people are trying
to measure. The only way to size it is to rebuild the entire panel on each of
the seven anchors -- which is why ``SRPData.load`` takes ``week_anchor`` and why
this re-derives weekly closes, volumes and funding from the hourly source rather
than shifting one grid.

A strategy is anchor-robust if every anchor is profitable and the spread is
small relative to the mean. If one Monday-shaped result carries the paper, that
is luck, not alpha.

Run:
    uv run python -m scripts.srp_walkforward --mode walkforward
    uv run python -m scripts.srp_walkforward --mode timing
    uv run python -m scripts.srp_walkforward --mode both
"""

from __future__ import annotations

import argparse
import itertools
import math

import pandas as pd

from scripts.srp_backtest import SRPConfig, SRPData, run
from scripts.trial_registry import log_trial

# The candidate set walk-forward chooses among at each step. Deliberately small:
# every extra candidate is another thing the TRAINING window gets to pick from,
# and the point of this test is to be honest about selection, not to win.
CANDIDATES: list[dict] = [
    {"rank_window": rw, "funding_tilt": ft, "turnover_cap": tc, "top": tp}
    for rw, ft, tc, tp in itertools.product(
        [26, 52, 104], [0.0, 0.5], [None, 0.6], [0.20]
    )
]

ANCHORS = ["W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN"]


def _cfg(d: dict) -> SRPConfig:
    rw = d["rank_window"]
    return SRPConfig(
        rank_window=rw,
        rank_min_periods=max(4, rw // 2),
        funding_tilt=d["funding_tilt"],
        turnover_cap=d["turnover_cap"],
        top=d["top"],
    )


def _sharpe(r: pd.Series) -> float:
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std())


def walk_forward(data: SRPData, train: int, test: int,
                 log: bool = False) -> dict:
    """Rolling out-of-sample evaluation. Returns the concatenated OOS series."""
    # Each candidate is evaluated ONCE over the full history; the walk-forward
    # then slices those series by window. This is valid because ``run`` is
    # point-in-time -- a return at week w never depends on weeks after w -- so
    # restricting attention to a window is the same as having run only that
    # window, and it turns an O(candidates x folds) sweep into O(candidates).
    series: dict[int, pd.Series] = {}
    for i, c in enumerate(CANDIDATES):
        res = run(data, _cfg(c))
        if res.n_obs > 0:
            series[i] = res.returns
        if log:
            log_trial("srp_wf_candidate", {**c, "scope": "full"},
                      sharpe_weekly=res.sharpe_weekly, n_obs=res.n_obs,
                      note="walk-forward candidate, full-sample scoring")
    if not series:
        raise SystemExit("no candidate produced returns")

    grid = sorted(set().union(*(s.index for s in series.values())))
    grid = pd.DatetimeIndex(grid)
    oos: list[pd.Series] = []
    picks: list[tuple] = []

    start = train
    while start + test <= len(grid):
        tr = grid[start - train:start]
        te = grid[start:start + test]
        best_i, best_s = None, float("-inf")
        for i, s in series.items():
            w = s.reindex(tr).dropna()
            if len(w) < max(8, train // 3):
                continue
            sh = _sharpe(w)
            if math.isfinite(sh) and sh > best_s:
                best_i, best_s = i, sh
        if best_i is not None:
            blk = series[best_i].reindex(te).dropna()
            if len(blk):
                oos.append(blk)
                picks.append((te[0].date(), CANDIDATES[best_i], best_s, _sharpe(blk)))
        start += test

    if not oos:
        raise SystemExit("walk-forward produced no out-of-sample blocks")
    o = pd.concat(oos).sort_index()
    o = o[~o.index.duplicated()]

    # In-sample reference: the single best candidate over the WHOLE sample --
    # the number a researcher would quote without this test.
    best_full = max(series.values(), key=lambda s: _sharpe(s) if len(s) > 3 else -9)
    return {"oos": o, "picks": picks, "in_sample": _sharpe(best_full),
            "n_candidates": len(series)}


def timing_luck(cache: str, intraday: str, positioning: str, ticket: str,
                cfg: SRPConfig, log: bool = False) -> pd.DataFrame:
    """Rebuild the panel on each weekday anchor and score the same strategy."""
    rows = []
    for anc in ANCHORS:
        d = SRPData.load(cache, intraday, positioning, ticket, week_anchor=anc)
        res = run(d, cfg)
        rows.append({"anchor": anc, "sharpe_ann": res.sharpe_ann,
                     "n_obs": res.n_obs,
                     "t": res.sharpe_weekly * math.sqrt(res.n_obs)
                     if res.n_obs > 2 else float("nan"),
                     "turnover": res.turnover_mean})
        if log:
            log_trial("srp_timing", {"anchor": anc, **{k: getattr(cfg, k) for k in
                                                       ("rank_window", "funding_tilt",
                                                        "turnover_cap", "top")}},
                      sharpe_weekly=res.sharpe_weekly, n_obs=res.n_obs,
                      note=f"timing-luck anchor {anc}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["walkforward", "timing", "both"], default="both")
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--intraday", default="/tmp/quant_cache/intraday_1h")
    ap.add_argument("--positioning", default="/tmp/quant_cache/positioning_daily")
    ap.add_argument("--ticket", default="/tmp/quant_cache/intraday3")
    ap.add_argument("--train", type=int, default=104, help="training weeks")
    ap.add_argument("--test", type=int, default=26, help="test-block weeks")
    ap.add_argument("--log-trials", action="store_true",
                    help="write every candidate to the trial registry")
    a = ap.parse_args()

    if a.mode in ("walkforward", "both"):
        print("=== WALK-FORWARD (config chosen on trailing window only) ===")
        data = SRPData.load(a.cache, a.intraday, a.positioning, a.ticket)
        wf = walk_forward(data, a.train, a.test, log=a.log_trials)
        o = wf["oos"]
        sh_w = _sharpe(o)
        ann = sh_w * math.sqrt(52)
        ins = wf["in_sample"] * math.sqrt(52)
        print(f"  candidates per fold   : {wf['n_candidates']}")
        print(f"  train / test (weeks)  : {a.train} / {a.test}")
        print(f"  OOS observations      : {len(o)}")
        print(f"  OOS annualised Sharpe : {ann:.3f}")
        print(f"  OOS t-statistic       : {sh_w * math.sqrt(len(o)):.2f}")
        print(f"  in-sample best (ref)  : {ins:.3f}")
        wfe = 100.0 * ann / ins if ins else float("nan")
        print(f"  Walk-Forward Efficiency: {wfe:.0f}%  "
              f"({'ROBUST (>50%)' if wfe > 50 else 'WEAK'})")
        print(f"\n  {'test block':<12} {'chosen config':<58} {'train SR':>9} {'OOS SR':>8}")
        print("  " + "-" * 92)
        for d_, c, tr_, te_ in wf["picks"]:
            cs = (f"rw={c['rank_window']} ft={c['funding_tilt']} "
                  f"cap={c['turnover_cap']} top={c['top']}")
            print(f"  {str(d_):<12} {cs:<58} {tr_ * math.sqrt(52):>9.2f} "
                  f"{te_ * math.sqrt(52) if math.isfinite(te_) else float('nan'):>8.2f}")

    if a.mode in ("timing", "both"):
        print("\n=== REBALANCE TIMING LUCK (panel rebuilt on each anchor) ===")
        df = timing_luck(a.cache, a.intraday, a.positioning, a.ticket,
                         SRPConfig(), log=a.log_trials)
        print(df.to_string(index=False,
                           formatters={"sharpe_ann": "{:.3f}".format,
                                       "t": "{:.2f}".format,
                                       "turnover": "{:.3f}".format}))
        s = df["sharpe_ann"].astype(float)
        print(f"\n  mean {s.mean():.3f}   sd {s.std(ddof=1):.3f}   "
              f"min {s.min():.3f}   max {s.max():.3f}   spread {s.max() - s.min():.3f}")
        print(f"  anchors profitable    : {int((s > 0).sum())}/{len(s)}")
        print(f"  verdict               : "
              f"{'ANCHOR-ROBUST' if (s > 0).all() and s.std(ddof=1) < 0.5 * s.mean() else 'ANCHOR-SENSITIVE -- the Monday result is not the strategy'}")


if __name__ == "__main__":
    main()
