"""SRP execution: trade the weekly factor book on its own clock.

The existing execution engine (``stream/execution.py``) is driven by the hourly
feature stream: it matches a stored signal to a bar by ``window_end_ms`` and
refuses to act when they differ. That check is the reason SRP could not simply
be pointed at it. SRP scores on a WEEKLY bar, so its records would never match
an hourly window, and making them match by echoing the hourly stamp would assert
that SRP scored on a window it never saw -- exactly the class of untruth the
parity gate exists to catch.

So SRP gets its own execution instance on its own clock, reusing the same
simulator and the same venue adapter:

    stream/srp_publisher.py  ->  srp:weights:<SYM>  (direction + weekly window)
    stream/srp_execution.py  ->  PaperExecutionSimulator(window_ms = 1 week)
                             ->  execution:srp:<SYM>

Three settings differ from the hourly engine, each for a reason:

  window_ms          One week. This is what makes the signal/bar match succeed
                     and what sizes the staleness budget correctly: a signal two
                     hours old is fresh for a weekly book and stale for an
                     hourly one.

  cost_filter_lambda 0. The hourly engine refuses to open unless the forecast
                     exceeds a multiple of the round-trip fee, because a
                     continuously-updating forecast will otherwise churn. SRP
                     has no forecast to threshold -- it has target weights, and
                     its churn is already bounded by the strategy's own turnover
                     cap. Applying a forecast filter to a weight would be
                     comparing quantities with different units.

  execution_prefix   Separate ledger. Two strategies sharing a position ledger
                     would each see the other's positions as their own and try
                     to close them.

Run:
    uv run python -m stream.srp_execution            # paper
    uv run python -m stream.srp_execution --venue bybit-demo
"""

from __future__ import annotations

import argparse
import time

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.execution import PaperExecutionSimulator, _build_venue, execution_key
from stream.kv import RedisKV
from stream.srp_publisher import BOOK_KEY, WEIGHTS_PREFIX, weight_key

logger = get_logger(__name__)

WEEK_MS = 7 * 24 * 60 * 60 * 1000
EXECUTION_PREFIX = "execution:srp"


def rebalance(simulator: PaperExecutionSimulator, kv, universe: list[str]) -> dict:
    """Drive one SRP rebalance through the execution engine.

    Reads the published book, and for each symbol hands the engine a message on
    the strategy's own weekly clock. Returns a summary of what moved.
    """
    book = kv.get_json(BOOK_KEY)
    if not book:
        logger.warning("SRP execution: no book published; nothing to do")
        return {"acted": 0, "reason": "no book"}
    if not book.get("scored"):
        # An unscored book is a real state, not an error: publishing FLAT is how
        # the publisher reports that it could not score. Acting on it would mean
        # closing the whole book on a data gap.
        logger.warning("SRP execution: book is unscored (%s); holding", book.get("reason"))
        return {"acted": 0, "reason": f"unscored: {book.get('reason')}"}

    window_end = book.get("window_end_ms")
    if not isinstance(window_end, (int, float)):
        logger.warning("SRP execution: book has no weekly window; holding")
        return {"acted": 0, "reason": "no window_end_ms"}
    window_end = int(window_end)

    # Prime the previous weekly bar so the first rebalance of a fresh process
    # is treated as a genuine week-on-week advance rather than an unknown one.
    # This is true by construction: the bar before a weekly close is one week
    # earlier. Without it the engine holds forever, because it never sees a
    # transition it can attribute to a new bar.
    for symbol in universe:
        simulator._last_window_end.setdefault(symbol, window_end - WEEK_MS)

    acted = 0
    results: dict[str, dict] = {}
    for symbol in universe:
        record = kv.get_json(weight_key(WEIGHTS_PREFIX, symbol))
        if not record:
            continue
        close = record.get("close") or _last_close(kv, symbol)
        if close is None:
            logger.debug("SRP execution: no mark for %s; skipping", symbol)
            continue
        out = simulator.handle(
            {"symbol": symbol, "close": float(close), "window_end_ms": window_end}
        )
        if out is not None:
            results[symbol] = out
            acted += 1

    logger.info(
        "SRP rebalance: window_end=%s, %d/%d symbols processed",
        window_end, acted, len(universe),
    )
    return {"acted": acted, "window_end_ms": window_end, "results": results}


def mark_to_market(simulator: PaperExecutionSimulator, kv, universe: list[str]) -> dict:
    """Re-price open positions against the latest live bar. No rebalancing.

    The book turns over weekly, but its P&L moves with every tick. This refreshes
    ``mark_price`` and unrealized P&L without advancing the strategy clock, so a
    weekly book can be watched in real time. Passing the CURRENT window end (not
    a new one) is what keeps it a re-mark rather than a rebalance: the engine
    only opens, closes or flips when it sees a window it has not seen before.
    """
    book = kv.get_json(BOOK_KEY) or {}
    window_end = book.get("window_end_ms")
    if not isinstance(window_end, (int, float)):
        return {"marked": 0, "reason": "no window_end_ms"}
    window_end = int(window_end)

    marked, total = 0, 0.0
    for symbol in universe:
        close = _last_close(kv, symbol)
        if close is None:
            continue
        simulator.handle({"symbol": symbol, "close": close, "window_end_ms": window_end})
        led = kv.get_json(execution_key(EXECUTION_PREFIX, symbol)) or {}
        position = led.get("position")
        if position:
            marked += 1
            total += float(position.get("unrealized_pnl") or 0.0)
    return {"marked": marked, "unrealized_pnl": round(total, 4),
            "window_end_ms": window_end}


def _last_close(kv, symbol: str) -> float | None:
    """Latest mark for ``symbol`` from the live bar the materializer keeps."""
    settings = get_settings()
    bar = kv.get_json(f"{settings.stream_redis_live_prefix}:{symbol.upper()}")
    if not bar:
        return None
    close = bar.get("close")
    return float(close) if isinstance(close, (int, float)) and close > 0 else None


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venue", choices=["paper", "bybit-demo"], default="paper",
                    help="paper simulates fills; bybit-demo places real orders "
                         "with virtual money on api-demo.bybit.com")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="after rebalancing, re-mark open positions against live "
                         "prices every SECONDS and print running P&L (0 = off)")
    a = ap.parse_args()

    settings = get_settings()
    universe = [s.upper() for s in csv_list(settings.ingest_default_crypto_symbols)]
    kv = RedisKV(settings.stream_redis_url)
    venue = _build_venue(settings) if a.venue == "bybit-demo" else None

    simulator = PaperExecutionSimulator(
        kv,
        execution_prefix=EXECUTION_PREFIX,
        prediction_prefix=WEIGHTS_PREFIX,
        notional_usd=settings.stream_execution_notional_usd,
        window_ms=WEEK_MS,
        venue=venue,
        cost_filter_lambda=0.0,
        signal_max_stale_windows=1,
    )
    summary = rebalance(simulator, kv, universe)
    logger.info("SRP execution finished: %s", {k: v for k, v in summary.items()
                                               if k != "results"})

    if a.watch > 0:
        logger.info("watching: re-marking every %.0fs (Ctrl-C to stop)", a.watch)
        try:
            while True:
                time.sleep(a.watch)
                snap = mark_to_market(simulator, kv, universe)
                logger.info(
                    "mark: %d open, unrealized $%+.2f",
                    snap.get("marked", 0), snap.get("unrealized_pnl", 0.0),
                )
        except KeyboardInterrupt:
            logger.info("watch stopped")


if __name__ == "__main__":
    main()
