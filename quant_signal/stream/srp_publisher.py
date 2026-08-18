"""SRP → Redis: publish the paper strategy's live book to the online store.

The streaming stack already serves River predictions, Monte Carlo fan charts and
the asym signal, but SRP -- the strategy the paper is about -- had no presence in
the online store at all. ``srp_live.SRPBook`` could compute weights; nothing
wrote them anywhere the API could read.

This closes that. It is deliberately thin: all strategy logic lives in
``scripts.srp_strategy`` (which research and the parity gate also call), and this
module only turns a scored book into online-store records:

    srp:weights:<SYMBOL>   per-symbol target weight + direction   (SET)
    srp:book               the whole book plus metadata            (SET)

Two design choices worth stating.

WEIGHTS, NOT ORDERS. This publishes *target weights*. It does not size in
dollars, place orders or touch a venue. Turning weights into fills is a separate
concern with its own risk checks, and conflating them is how a research artefact
becomes an accidental trading system.

FLAT IS A RESULT. When the book cannot be scored -- missing positioning frames,
too few symbols, a warm-up window -- this publishes an explicit FLAT book with a
``reason``, rather than writing nothing. A stale record that looks live is worse
than an empty one that says why, and the dashboard can tell the difference.

Run with ``make stream-srp`` (or ``uv run python -m stream.srp_publisher``).
Pure logic lives in ``build_payload`` so tests drive it with a FakeKV.
"""

from __future__ import annotations

import time

import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

WEIGHTS_PREFIX = "srp:weights"
BOOK_KEY = "srp:book"


def weight_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def build_payload(
    directions: dict[str, str],
    weights: pd.Series | None,
    *,
    universe: list[str],
    stamp_ms: int,
    window_end_ms: int | None = None,
    reason: str | None = None,
) -> tuple[dict[str, dict], dict]:
    """Turn a scored book into ``(per-symbol records, book record)``.

    Pure: no clock, no I/O. ``weights`` is None when the book could not be
    scored, in which case every symbol is published FLAT at zero weight with
    ``reason`` explaining why.

    ``window_end_ms`` is the close of the WEEKLY bar this book was scored on --
    not wall-clock time. The execution engine matches a signal to a bar by this
    field, so publishing the hourly window here would assert that SRP scored on
    a window it never saw. It is the strategy's own clock or nothing.
    """
    per_symbol: dict[str, dict] = {}
    gross = 0.0
    net = 0.0
    n_long = n_short = 0

    for sym in universe:
        w = 0.0
        if weights is not None and sym in weights.index:
            raw = weights.get(sym)
            w = float(raw) if pd.notna(raw) else 0.0
        direction = directions.get(sym, "FLAT")
        gross += abs(w)
        net += w
        if w > 0:
            n_long += 1
        elif w < 0:
            n_short += 1
        per_symbol[sym] = {
            "symbol": sym,
            "weight": round(w, 8),
            "direction": direction,
            "stamp_ms": stamp_ms,
            "window_end_ms": window_end_ms,
        }

    book = {
        "stamp_ms": stamp_ms,
        "window_end_ms": window_end_ms,
        "n_symbols": len(universe),
        "n_long": n_long,
        "n_short": n_short,
        "gross": round(gross, 8),
        # Dollar-neutrality is asserted by the parity gate; publishing net lets
        # the dashboard show a drift rather than assume the invariant holds.
        "net": round(net, 8),
        "scored": weights is not None,
        "reason": reason,
        "weights": {s: per_symbol[s]["weight"] for s in universe},
    }
    return per_symbol, book


def publish(kv: KVStore, per_symbol: dict[str, dict], book: dict,
            *, prefix: str = WEIGHTS_PREFIX, book_key: str = BOOK_KEY) -> None:
    """Write the book to the online store. SET, not RPUSH: only the current
    target matters, and a stale history of targets invites reading the wrong one.
    """
    for sym, record in per_symbol.items():
        kv.set_json(weight_key(prefix, sym), record)
    kv.set_json(book_key, book)


def main() -> None:
    configure_logging()
    settings = get_settings()
    universe = [s.upper() for s in csv_list(settings.ingest_default_crypto_symbols)]
    kv = RedisKV(settings.stream_redis_url)

    # Imported here so the module stays importable (and testable) without the
    # research caches present.
    from stream.srp_live import SRPBook

    book_engine = SRPBook(universe)
    stamp_ms = int(time.time() * 1000)
    window_end_ms: int | None = None

    try:
        # Weekly close/volume/funding come from the same cache research reads,
        # which is what makes the parity gate's equality claim meaningful: both
        # paths score identical frames through identical code.
        from scripts.research_fas_clean import load as load_weekly

        cw, vw, aw, _dcl, _dvl = load_weekly(settings.srp_weekly_cache)
        # The last weekly bar in the panel IS the rebalance this book belongs to.
        window_end_ms = int(cw.index[-1].timestamp() * 1000)
        cols = [s for s in universe if s in cw.columns]
        if len(cols) < book_engine.min_symbols:
            raise ValueError(
                f"only {len(cols)} of {len(universe)} symbols in the weekly cache; "
                f"need {book_engine.min_symbols}"
            )
        directions, _books, weights = book_engine.score(
            cw[cols], vw[cols], aw.reindex(index=cw.index, columns=cols).fillna(0.0)
        )
        reason = None if weights is not None else "score returned FLAT"
    except Exception as exc:  # noqa: BLE001 - publishing FLAT beats publishing nothing
        logger.exception("SRP scoring failed; publishing FLAT")
        directions = {s: "FLAT" for s in universe}
        weights = None
        reason = f"scoring error: {type(exc).__name__}: {exc}"

    per_symbol, book = build_payload(
        directions, weights, universe=universe, stamp_ms=stamp_ms,
        window_end_ms=window_end_ms, reason=reason,
    )
    publish(kv, per_symbol, book)
    logger.info(
        "SRP published: %d symbols, gross %.4f, net %.4f, scored=%s, week_end=%s",
        book["n_symbols"], book["gross"], book["net"], book["scored"],
        book["window_end_ms"],
    )


if __name__ == "__main__":
    main()
