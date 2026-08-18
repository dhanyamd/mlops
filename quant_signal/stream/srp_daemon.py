"""SRP on autopilot: re-score, rebalance and mark, forever.

The publisher and the execution engine are one-shot commands. This wraps them in
a loop so the book keeps itself current without anyone at a terminal, which is
what makes a live track record possible: closes only happen when the weekly bar
turns, and nobody is going to run a command by hand every Monday for a year.

WHAT IT DOES EACH TICK
  1. re-score the weekly book and publish target weights
  2. hand the book to the execution engine
  3. re-mark open positions against the latest live bar

WHY POLLING HOURLY FOR A WEEKLY STRATEGY IS NOT CHURN. The execution engine acts
only on a ``window_end_ms`` it has not seen before. While the weekly bar is
unchanged the engine re-marks and does nothing else, so an hourly tick costs
nothing and buys two things: live unrealized P&L between rebalances, and a
rebalance that fires within an hour of the week turning rather than whenever
someone remembers. The strategy stays weekly; only the observation is frequent.

A NOTE ON THE WEEKLY CACHE. The book is scored from the same panel research
reads (``fas_broad.json``). That file does not extend itself -- if it is not
backfilled, the last weekly bar stays put and no new rebalance will ever fire,
however long this runs. The daemon logs the bar it scored on every tick so a
stalled panel is visible rather than silent.

Run as a launchd agent alongside the other services:
    uv run python -m scripts.install_services install
Or directly:
    uv run python -m stream.srp_daemon --interval 3600
"""

from __future__ import annotations

import argparse
import time

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.execution import PaperExecutionSimulator, _build_venue
from stream.kv import RedisKV
from stream.srp_execution import EXECUTION_PREFIX, WEEK_MS, mark_to_market, rebalance
from stream.srp_publisher import WEIGHTS_PREFIX

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=3600.0,
                    help="seconds between ticks (default 1h; the strategy is "
                         "still weekly, this only sets how often it is checked)")
    ap.add_argument("--venue", choices=["paper", "bybit-demo"], default="paper")
    a = ap.parse_args()

    settings = get_settings()
    universe = [s.upper() for s in csv_list(settings.ingest_default_crypto_symbols)]
    kv = RedisKV(settings.stream_redis_url)
    venue = _build_venue(settings) if a.venue == "bybit-demo" else None

    # One simulator for the process lifetime: it carries the last window seen
    # per symbol, which is what distinguishes "new week, rebalance" from
    # "same week, just re-mark".
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

    logger.info("SRP daemon starting: interval %.0fs, venue=%s, %d symbols",
                a.interval, a.venue, len(universe))
    last_window: int | None = None

    while True:
        try:
            # Re-score in-process rather than shelling out, so a failure here is
            # logged and survivable instead of taking the loop down.
            from stream.srp_publisher import main as publish_book

            publish_book()
            out = rebalance(simulator, kv, universe)
            window = out.get("window_end_ms")

            if window != last_window and last_window is not None:
                logger.info("SRP REBALANCED: weekly bar advanced %s -> %s",
                            last_window, window)
            last_window = window

            snap = mark_to_market(simulator, kv, universe)
            logger.info(
                "SRP tick: week_end=%s open=%s unrealized=$%+.2f",
                window, snap.get("marked"), snap.get("unrealized_pnl", 0.0),
            )
        except Exception:  # noqa: BLE001 - a scheduler must outlive one bad tick
            logger.exception("SRP tick failed; continuing")

        time.sleep(a.interval)


if __name__ == "__main__":
    main()
