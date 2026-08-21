"""Backtest-vs-live PARITY ablation: find what destroys the validated edge.

The strategy is validated at Sharpe 1.93 by scripts/research_fas_invent.py, a
VECTORISED pandas portfolio backtest. The live book is a completely separate
EVENT-DRIVEN implementation (stream/asym_signal.py + stream/execution.py).
Two implementations of one strategy is the classic backtest-live divergence
trap: "the most expensive mistakes occur when a validated strategy gets
rewritten for live deployment and a difference slips into the translation"
(binaryfintech.com/en/blog/event-driven-backtesting, quantstart event-driven
backtesting series). The cure is a single code path exercised both ways --
which is exactly what scripts/replay_live_book.py does: it drives the REAL
live classes over historical data.

This harness runs that replay across a matrix of execution settings and
reports the Sharpe/PnL of each, so the contribution of every live-only
execution component is measured rather than guessed:

  * ATR trailing stop      (live-only; absent from the research book)
  * Barroso vol-scaling    (live-only; absent from the research book)
  * cost model             (live 2bps slip + 10bps taker BOTH legs = 24bps
                            round trip, vs research 10bps x turnover)
  * RCGO tilt weight       (live multiplies W x rank_z(resid); research does
                            rank_z(W x rcgo), where W cancels -- so live at
                            W=0.5 runs the invention at half the validated
                            strength)

Nothing here is hardcoded strategy logic: every knob is an existing env var
the production engine already reads, so a winning configuration is directly
deployable to the live daemon.

Run:  uv run python -m scripts.parity_ablation
      uv run python -m scripts.parity_ablation --quick   (fewer windows)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from config.settings import PROJECT_ROOT

REPLAY = PROJECT_ROOT / "scripts" / "replay_live_book.py"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# Research benchmark this must reproduce (scripts/research_fas_invent.py,
# --funding binance, 55 weeks): the number the live path has to inherit.
RESEARCH_SHARPE = 1.93
RESEARCH_ANN_RET = 0.8213

_SHARPE_H = re.compile(r"book Sharpe \(ann\., hourly Mark-to-Mkt\):\s*(-?[\d.]+)")
_SHARPE_W = re.compile(r"book Sharpe \(ann\., WEEKLY returns.*?\):\s*(-?[\d.]+)")
_TRADES = re.compile(r"total closed trades\s*:\s*(\d+)")
_WINS = re.compile(r"total wins\s*:\s*(\d+)\s+win_rate=([\d.]+)%")
_PNL = re.compile(r"total realized PnL\s*:\s*\$(-?[\d,.]+)")


def _f(m, idx=1, default=None):
    if not m:
        return default
    return float(m.group(idx).replace(",", ""))


class Variant:
    """One ablation cell: a label plus the env overrides that define it."""

    def __init__(self, label: str, why: str, env: dict[str, str], args: list[str]):
        self.label = label
        self.why = why
        self.env = env
        self.args = args


def build_matrix() -> list[Variant]:
    # Every variant holds the book to the VALIDATED horizon (weekly rebalance,
    # hold-until-signal-decay). What varies is live-only execution machinery.
    base_env = {"STREAM_EXECUTION_HOLD_UNTIL_DECAY": "True"}
    base_args = ["--rebalance-h", "168", "--rcgo-dir", "1", "--cgo-dir", "1"]

    def v(label, why, env=None, rcgo_w="1.0"):
        return Variant(
            label,
            why,
            {**base_env, **(env or {})},
            [*base_args, "--rcgo-w", rcgo_w],
        )

    return [
        v(
            "research-parity",
            "all live-only machinery OFF + research cost (10bps/leg-pair): "
            "should land nearest Sharpe 1.93",
            {
                "QUANT_TRAIL_OFF": "1",
                "QUANT_VOL_OFF": "1",
                "STREAM_EXECUTION_SLIPPAGE_BPS": "0.0",
                "STREAM_EXECUTION_TAKER_FEE_BPS": "5.0",
            },
        ),
        v(
            "+live costs",
            "adds the real 2bps slip + 10bps taker on both legs (24bps round trip)",
            {"QUANT_TRAIL_OFF": "1", "QUANT_VOL_OFF": "1"},
        ),
        v(
            "+vol scaling",
            "adds Barroso-Santa-Clara exposure scaling (live-only)",
            {"QUANT_TRAIL_OFF": "1"},
        ),
        v(
            "+trailing stop (full live)",
            "adds the ATR trailing stop -- this is the shipped live config",
            {},
        ),
        v(
            "research-parity, RCGO half",
            "parity but W=0.5: shows the live W-multiplier drift in isolation",
            {
                "QUANT_TRAIL_OFF": "1",
                "QUANT_VOL_OFF": "1",
                "STREAM_EXECUTION_SLIPPAGE_BPS": "0.0",
                "STREAM_EXECUTION_TAKER_FEE_BPS": "5.0",
            },
            rcgo_w="0.5",
        ),
    ]


def run_variant(var: Variant, max_windows: int) -> dict:
    env = {**os.environ, **var.env}
    args = list(var.args)
    if max_windows:
        args += ["--max-windows", str(max_windows)]
    proc = subprocess.run(
        [str(PYTHON), str(REPLAY), *args],
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    wins = _WINS.search(out)
    return {
        "label": var.label,
        "why": var.why,
        "sharpe_hourly": _f(_SHARPE_H.search(out)),
        "sharpe_weekly": _f(_SHARPE_W.search(out)),
        "trades": int(_f(_TRADES.search(out), default=0) or 0),
        "win_rate": float(wins.group(2)) if wins else None,
        "pnl": _f(_PNL.search(out)),
        "ok": proc.returncode == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-windows", type=int, default=0, help="cap replay windows (0=full history)"
    )
    ap.add_argument("--quick", action="store_true", help="shorthand for --max-windows 6000")
    ap.add_argument("--json-out", default="", help="also write results as JSON here")
    a = ap.parse_args()
    max_windows = 6000 if a.quick else a.max_windows

    variants = build_matrix()
    print("=== BACKTEST <-> LIVE PARITY ABLATION ===")
    print(f"research benchmark: Sharpe {RESEARCH_SHARPE} (ann {RESEARCH_ANN_RET:.1%}, 55 weeks)")
    print(f"variants: {len(variants)}   windows: {'full' if not max_windows else max_windows}\n")

    results = []
    for i, var in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] {var.label} ... ", end="", flush=True)
        r = run_variant(var, max_windows)
        results.append(r)
        print(
            f"Sharpe(h)={r['sharpe_hourly']}  trades={r['trades']}  "
            f"win={r['win_rate']}%  pnl=${r['pnl']}"
        )

    print(f"\n{'variant':<30} {'Sharpe(h)':>10} {'Sharpe(w)':>10} {'trades':>7} "
          f"{'win%':>6} {'net P&L':>11}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<30} {str(r['sharpe_hourly']):>10} {str(r['sharpe_weekly']):>10} "
            f"{r['trades']:>7} {str(r['win_rate']):>6} {str(r['pnl']):>11}"
        )

    print("\nwhat each row isolates:")
    for r in results:
        print(f"  {r['label']:<30} {r['why']}")

    # Attribution: how much Sharpe each added component costs, in order.
    print("\ncost of each live-only component (delta vs the row above):")
    prev = None
    for r in results[:4]:
        s = r["sharpe_hourly"]
        if prev is not None and s is not None:
            print(f"  {r['label']:<30} {s - prev:+.2f} Sharpe")
        prev = s

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    sys.exit(main())
