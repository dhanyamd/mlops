"""Read the durable live-fill ledger and report the running track record.

Redis (execution:crypto:1h:<SYMBOL>) is the fast path the dashboard reads,
but it resets on a daemon restart and is one FLUSHDB away from losing
everything. lake/live_ledger/fills.jsonl (stream/execution.py,
_append_durable_fill) is the append-only source of truth for "has this
actually worked over time" -- every closed fill, forever, written only by
the real daemon (RedisKV-backed), never by tests.

Usage:
    uv run python -m scripts.live_track_record
"""

from __future__ import annotations

import json
import math

from config.settings import PROJECT_ROOT, get_settings

LEDGER_PATH = PROJECT_ROOT / "lake" / "live_ledger" / "fills.jsonl"

# Sample-size thresholds for reading a live track record. Not invented:
# practitioner and backtest-validation guidance converges on ~30 trades as the
# floor for any rough read and 100+ before a result is statistically
# meaningful, with calendar time following from trade frequency rather than a
# fixed duration.
#   https://medium.com/@trading.dude/how-many-trades-are-enough-a-guide-to-statistical-significance-in-backtesting-093c2eac6f05
#   https://www.backtestbase.com/education/how-many-trades-for-backtest
#   https://blog.traderspost.io/article/paper-trading-strategy-development-guide
MIN_ROUGH = 30
MIN_SIGNIFICANT = 100


def load_fills() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    fills = []
    with LEDGER_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                fills.append(json.loads(line))
    return fills


def _is_orphan(fill: dict) -> bool:
    """Adopted-orphan cleanup, not a strategy decision.

    The engine tags these explicitly at close time (``adopted_orphan``).
    Records written before that flag existed are identified structurally, with
    no invented threshold: an adopted position has ``entry_window_end_ms == 0``,
    so the engine's own arithmetic

        bars_held = round((window_end - entry_window_end_ms) / window_ms)

    degenerates to ``window_end / window_ms`` — i.e. the holding period is the
    entire Unix epoch. Reconstructing that identity is exact, so it needs no
    magic cutoff: a real trade's entry window is never epoch 0.
    """
    if "adopted_orphan" in fill:
        return bool(fill["adopted_orphan"])
    bars_held = int(fill.get("bars_held") or 0)
    window_end = int(fill.get("window_end_ms") or 0)
    if bars_held <= 0 or window_end <= 0:
        return False
    # Compare against the engine's ACTUAL window size (config, not a literal).
    window_ms = get_settings().stream_window_ms
    epoch_bars = round(window_end / window_ms)
    return bars_held == epoch_bars


def main() -> None:
    all_fills = load_fills()
    orphans = [f for f in all_fills if _is_orphan(f)]
    fills = [f for f in all_fills if not _is_orphan(f)]
    n = len(fills)
    print(f"=== Live track record ({LEDGER_PATH.relative_to(PROJECT_ROOT)}) ===")
    print(f"strategy closed trades: {n}")
    if orphans:
        net = sum(f["net_pnl"] for f in orphans)
        print(
            f"(excluded: {len(orphans)} adopted-orphan cleanup close(s), net ${net:.2f} — "
            "positions inherited from the pre-fix period, not chosen by the strategy)"
        )

    if n == 0:
        print("No strategy trades closed yet.")
        return

    net = [f["net_pnl"] for f in fills]
    pct = [f["net_pnl_pct"] for f in fills]
    wins = sum(1 for x in net if x > 0)
    total_net = sum(net)
    win_rate = wins / n

    mean_pct = sum(pct) / n
    if n > 1:
        var = sum((p - mean_pct) ** 2 for p in pct) / (n - 1)
        std_pct = math.sqrt(var)
    else:
        std_pct = 0.0
    sharpe_per_trade = (mean_pct / std_pct) if std_pct > 0 else 0.0

    by_symbol: dict[str, list[dict]] = {}
    for f in fills:
        by_symbol.setdefault(f["symbol"], []).append(f)

    print(f"total net P&L: ${total_net:.2f}")
    print(f"win rate: {win_rate:.1%} ({wins}/{n})")
    print(f"mean return/trade: {mean_pct:.4%}  std: {std_pct:.4%}")
    print(f"per-trade Sharpe-like ratio (NOT annualized, tiny sample): {sharpe_per_trade:.3f}")
    print()
    print("Statistical read:")
    if n < MIN_ROUGH:
        print(
            f"  Below {MIN_ROUGH} trades — too few for even a rough read. "
            "This is pipeline-works proof, not edge proof."
        )
    elif n < MIN_SIGNIFICANT:
        print(
            f"  {n}/{MIN_SIGNIFICANT} toward a statistically meaningful sample. "
            "Directional signal only — don't trust the sign of the Sharpe yet."
        )
    else:
        print(
            f"  {n} trades — enough for a real (if still wide-CI) read. "
            "Compare against the backtest's Sharpe 1.82-1.93, not against zero."
        )
    print()
    print("By symbol:")
    for sym, fs in sorted(by_symbol.items(), key=lambda kv: -len(kv[1])):
        sym_net = sum(x["net_pnl"] for x in fs)
        sym_wins = sum(1 for x in fs if x["net_pnl"] > 0)
        print(f"  {sym:10} n={len(fs):3d}  win_rate={sym_wins/len(fs):.0%}  net=${sym_net:.2f}")


if __name__ == "__main__":
    main()
