"""Diagnose the vreg KILL: bucket OOS trades by the vol regime that pinned them.

Re-runs ``_vreg_events`` for each symbol with the SELECTED vreg config (from the
probe JSON), splits at the probe's ``cut_ms``, and reports per-regime OOS stats.
Answers the falsifiability question: is the pinned map (long momentum in LOW vol
= L=28d, HIGH vol = L=7d) right or inverted on real data?

Usage::

    uv run python scripts/diagnose_vreg.py docs/trend_momentum_probe.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

from config.settings import get_settings
from scripts.backfill_feature_windows import fetch_bars
from scripts.probes.intraday_30m_probe import _aggregate
from scripts.probes.trend_momentum_probe import _vreg_events

REGIMES = [("low", 0.0, 0.3), ("mid", 0.3, 0.7), ("high", 0.7, 1.01)]
_LABEL = {
    "low": "LOW vol  (<=30pct -> L=28d a=2.0)",
    "mid": "MID vol  (30-70pct -> L=14d a=2.5)",
    "high": "HIGH vol (>70pct -> L=7d  a=3.0)",
}


def _bucket(pct: float) -> str:
    for name, lo, hi in REGIMES:
        if lo <= pct < hi:
            return name
    return "mid"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: uv run python scripts/diagnose_vreg.py <probe.json>")
    report = json.load(open(sys.argv[1]))
    settings = get_settings()

    for sym_rec in report["symbols"]:
        symbol = sym_rec["symbol"]
        cut = sym_rec["cut_ms"]
        sel = sym_rec["sections"]["vreg"]["selected"]
        if sel is None:
            print(f"\n=== {symbol}: no selected vreg config ===")
            continue
        bars = fetch_bars(settings, [symbol])
        hourly = _aggregate(bars, 3_600_000)
        hourly = hourly[hourly["symbol"] == symbol]

        events = _vreg_events(
            hourly,
            vol_scale=sel["vol_scale"],
            crash=sel["crash"],
            regime=sel["regime"],
        )
        oos = [e for e in events if e["ts"] >= cut]
        is_ = [e for e in events if e["ts"] < cut]
        print(
            f"\n=== {symbol} — selected vreg cfg vol_scale={sel['vol_scale']} "
            f"crash={sel['crash']} regime={sel['regime']}"
        )
        print(f"  events: {len(events)} total, {len(is_)} in-sample, {len(oos)} OOS")
        if not oos:
            continue
        ts = np.array([e["ts"] for e in events])
        cut_again = ts.min() + report["symbols"][0]["wf_split"] * (ts.max() - ts.min())
        print(f"  (probe cut {cut}; re-derived {int(cut_again)})")

        print(f"  {'regime':<42} {'n':>3} {'win':>6} {'mean':>8} {'med':>8} {'mult':>7}")
        for name, lo, hi in REGIMES:
            sub = [e for e in oos if _bucket(e["vol_pct"]) == name]
            if not sub:
                print(f"  {_LABEL[name]:<42} {0:>3} {'—':>6} {'—':>8} {'—':>8} {'—':>7}")
                continue
            pnls = np.array([e["net_maker"] for e in sub])
            mult = float(np.prod(1.0 + pnls))
            print(
                f"  {_LABEL[name]:<42} {len(sub):>3} {float(np.mean(pnls > 0)):>6.2f} "
                f"{1e4 * float(np.mean(pnls)):>8.1f} {1e4 * float(np.median(pnls)):>8.1f} "
                f"{mult:>7.3f}"
            )
        pnls_all = np.array([e["net_maker"] for e in oos])
        print(
            f"  {'ALL OOS':<42} {len(pnls_all):>3} {float(np.mean(pnls_all > 0)):>6.2f} "
            f"{1e4 * float(np.mean(pnls_all)):>8.1f} {1e4 * float(np.median(pnls_all)):>8.1f} "
            f"{float(np.prod(1.0 + pnls_all)):>7.3f}"
        )

        # regime mix in-sample vs OOS (is the OOS regime distribution different?)
        def _mix(evs: list[dict]) -> dict[str, int]:
            out = {"low": 0, "mid": 0, "high": 0}
            for e in evs:
                out[_bucket(e["vol_pct"])] += 1
            return out

        print(f"  regime mix IS: {_mix(is_)}  OOS: {_mix(oos)}")


if __name__ == "__main__":
    main()
