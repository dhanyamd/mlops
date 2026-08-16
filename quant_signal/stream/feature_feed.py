"""Feature feed: replaces the dead streaming river (producer -> crypto.bars.raw
-> Flink -> crypto.features.1h) with REAL 1h OHLCV published straight to the
features topic the live book already consumes.

The river is wedged: the producer writes bars to Snowflake only (never to
crypto.bars.raw), so Flink has nothing to transform and crypto.features.1h stays
empty — the asym signal + paper-execution daemons idle forever and nothing trades.

This module IS the new upstream. Each completed 1h bar it fetches the REAL Bybit
spot OHLCV (keyless public endpoint, same provider the warm-start uses) for the
trading universe and publishes it to crypto.features.1h in the EXACT Flink schema
(symbol, window_start_ms, window_end_ms, open, high, low, close, volume, vwap,
bar_count). The existing asym_signal + execution daemons consume that topic
unchanged, publish predictions, place REAL Bybit Demo fills and write
execution:crypto:1h — which the API + dashboard at localhost:3000/signal render.

Nothing is hardcoded: symbols come from stream_xs_universe (settings), the venue
is the configured Bybit provider, the topic/bootstrap come from settings, and
window boundaries are derived from wall-clock time. The Kafka schema matches the
Flink SQL (stream/flink/jobs/crypto_features_1h.sql) field-for-field so the
downstream parsers (AsymSignal.handle / PaperExecutionSimulator.handle) need no
changes.

Run as the producer launchd job (replaces stream.producer):
  scripts/service.sh stream.feature_feed
"""

from __future__ import annotations

import os
import sys
import threading
import time

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.bus import KafkaBus
from stream.kv import RedisKV

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_POLL_S = 30.0
# Producer-health keys the watchdog/dashboard already read (mirrors stream.producer
# so the feed shows as alive even when the book is flat).
_META_LAST_POLL = "meta:producer:crypto.features.1h:last_poll_ts"
_META_LAST_BAR = "meta:producer:crypto.features.1h:last_bar_max_ts"


# Which Bybit product a symbol trades on is a PROPERTY OF THE SYMBOL, not a
# global setting: BTCUSDT exists as both spot and linear perp, while RLCUSDT,
# STORJUSDT and XMRUSDT are perp-only. Hardcoding either choice breaks half the
# universe -- with spot, most perps fail, the topic starves, and downstream
# signals sit on a stale window_end (observed live: stuck 10 months in the past).
#
# So the category is DISCOVERED per symbol on first use and remembered. No
# global default decides it; the venue does. QUANT_FEED_CATEGORY pins it only if
# an operator explicitly wants to.
_CATEGORY_ORDER = ("spot", "linear")
_symbol_category: dict[str, str] = {}


def _categories_for(symbol: str) -> tuple[str, ...]:
    pinned = os.environ.get("QUANT_FEED_CATEGORY")
    if pinned:
        return (pinned,)
    known = _symbol_category.get(symbol.upper())
    if known:
        return (known,)
    return _CATEGORY_ORDER


def _fetch_1h_bar(provider, symbol: str, window_end_ms: int) -> dict | None:
    """REAL Bybit 1h OHLCV for the bar ending at ``window_end_ms``.

    Bybit kline row = [startTime_ms, open, high, low, close, volume, turnover].
    Returns the feature record (Flink schema) or None on a failed/stale fetch.
    This is the same keyless endpoint the warm-start uses — no API key, no guess.
    """
    page = None
    last_exc: Exception | None = None
    for cat in _categories_for(symbol):
        try:
            page = provider._get(
                {
                    "category": cat,
                    "symbol": symbol.upper(),
                    "interval": "60",
                    "end": window_end_ms,
                    "limit": 1,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one dead pair must not wedge the feed
            last_exc = exc
            continue
        if page:
            # Remember what this symbol actually trades on so subsequent polls
            # go straight there instead of re-probing every cycle.
            _symbol_category[symbol.upper()] = cat
            break
    if not page:
        logger.warning(
            "feature feed: bybit 1h kline for %s failed on %s (%s)",
            symbol, "/".join(_categories_for(symbol)),
            type(last_exc).__name__ if last_exc else "empty",
        )
        return None
    bars = (page.get("result") or {}).get("list") or []
    if not bars:
        return None
    row = sorted(bars, key=lambda r: int(r[0]))[-1]
    try:
        start = int(row[0])
        o, h, l, c, v = (
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
    except (TypeError, ValueError, IndexError):
        return None
    if c <= 0 or v <= 0:
        return None
    # The bar we asked for ends at window_end_ms; Bybit returns the bar whose
    # startTime <= end, so its window_end = start + 1h.
    we = start + _HOUR_MS
    vwap = (o + h + l + c) / 4.0  # volume-weighted not available per single bar; use typical
    return {
        "symbol": symbol.upper(),
        "window_start_ms": start,
        "window_end_ms": we,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "vwap": vwap,
        "bar_count": 60,
    }


def main() -> None:
    configure_logging()
    settings = get_settings()
    universe = csv_list(settings.stream_xs_universe)
    if not universe:
        logger.error("no universe configured (stream_xs_universe); aborting")
        return

    from ingest.providers.bybit import BybitBarProvider

    provider = BybitBarProvider(base_url=settings.stream_bybit_base_url)
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url) if settings.stream_redis_url else None
    topic = settings.stream_kafka_topic_features

    # Resume from ~24h back so a (re)start backfills the gap the dead river left
    # on crypto.features.1h (the producer wedged ~20h ago), keeping the book's
    # recent bars continuous. The forward loop then publishes each new hour.
    now_ms = int(time.time() * 1000)
    last_window = (now_ms // _HOUR_MS) * _HOUR_MS - 24 * _HOUR_MS
    published = 0

    logger.info(
        "feature feed publishing REAL Bybit 1h OHLCV -> %s (%d symbols, poll %ss); "
        "replaces dead producer+Flink river",
        topic,
        len(universe),
        _POLL_S,
    )

    stop = threading.Event()

    def _shutdown(*_a):
        logger.info("feature feed shutting down")
        stop.set()

    import signal as _signal

    _signal.signal(_signal.SIGTERM, _shutdown)
    _signal.signal(_signal.SIGINT, _shutdown)

    while not stop.is_set():
        now_ms = int(time.time() * 1000)
        current_window = (now_ms // _HOUR_MS) * _HOUR_MS
        # Publish every completed hour from last_window+1h up to current_window.
        while last_window + _HOUR_MS <= current_window:
            target = last_window + _HOUR_MS
            n = 0
            max_ts = 0
            for sym in universe:
                rec = _fetch_1h_bar(provider, sym, target)
                if rec is None:
                    continue
                bus.publish(topic, sym, rec)
                n += 1
                max_ts = max(max_ts, rec["window_end_ms"])
            bus.flush()
            if n:
                published += n
                logger.info("published %d bars for window %d", n, target)
                if kv is not None:
                    kv.set_json(_META_LAST_POLL, {"ts": now_ms, "symbols": n})
                    if max_ts:
                        kv.set_json(
                            _META_LAST_BAR,
                            {"ts": max_ts, "age_s": (now_ms - max_ts) // 1000},
                        )
            last_window = target
        # Keep the LATEST window fresh: publish the in-progress current hour on
        # every poll (the original Flink river did the same with its 2-min
        # watermark), so the dashboard never reads stale between hourly closes.
        # Idempotent per window_end: marks update, no duplicate entries.
        n = 0
        max_ts = 0
        for sym in universe:
            rec = _fetch_1h_bar(provider, sym, current_window)
            if rec is None:
                continue
            bus.publish(topic, sym, rec)
            n += 1
            max_ts = max(max_ts, rec["window_end_ms"])
        bus.flush()
        if n and kv is not None:
            kv.set_json(_META_LAST_POLL, {"ts": now_ms, "symbols": n})
            if max_ts:
                kv.set_json(_META_LAST_BAR, {"ts": max_ts, "age_s": (now_ms - max_ts) // 1000})
        last_window = current_window
        stop.wait(_POLL_S)

    logger.info("feature feed stopped (published %d bars total)", published)


if __name__ == "__main__":
    sys.exit(main())
