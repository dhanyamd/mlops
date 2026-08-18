"""SRP publisher: book -> online store. Hermetic (FakeKV, no Redis/caches)."""

from __future__ import annotations

import pandas as pd

from stream.kv import FakeKV
from stream.srp_publisher import (
    BOOK_KEY,
    WEIGHTS_PREFIX,
    build_payload,
    publish,
    weight_key,
)

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
STAMP = 1_787_000_000_000


def test_payload_carries_weights_and_directions() -> None:
    weights = pd.Series({"BTCUSDT": 0.25, "ETHUSDT": -0.25, "SOLUSDT": 0.0})
    directions = {"BTCUSDT": "LONG", "ETHUSDT": "SHORT", "SOLUSDT": "FLAT"}
    per_symbol, book = build_payload(
        directions, weights, universe=UNIVERSE, stamp_ms=STAMP
    )
    assert per_symbol["BTCUSDT"]["weight"] == 0.25
    assert per_symbol["ETHUSDT"]["direction"] == "SHORT"
    # A symbol absent from the weight series is published flat, not omitted:
    # a missing key is indistinguishable from a stale one to a reader.
    assert per_symbol["XRPUSDT"]["weight"] == 0.0
    assert per_symbol["XRPUSDT"]["direction"] == "FLAT"
    assert book["n_long"] == 1
    assert book["n_short"] == 1
    assert book["gross"] == 0.5
    assert book["scored"] is True


def test_book_reports_dollar_neutrality() -> None:
    """``net`` is published so a drift from neutrality is visible, not assumed."""
    weights = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": -0.5})
    _, book = build_payload({}, weights, universe=UNIVERSE, stamp_ms=STAMP)
    assert book["net"] == 0.0
    assert book["gross"] == 1.0

    lopsided = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": -0.3})
    _, skewed = build_payload({}, lopsided, universe=UNIVERSE, stamp_ms=STAMP)
    assert skewed["net"] == 0.2


def test_unscored_book_publishes_flat_with_a_reason() -> None:
    """An unscoreable book must still be published, with why.

    Writing nothing leaves the previous book in place, so a reader cannot tell a
    current target from a stale one. An explicit FLAT with a reason can.
    """
    per_symbol, book = build_payload(
        {s: "FLAT" for s in UNIVERSE},
        None,
        universe=UNIVERSE,
        stamp_ms=STAMP,
        reason="positioning frames missing",
    )
    assert book["scored"] is False
    assert book["reason"] == "positioning frames missing"
    assert book["gross"] == 0.0
    assert all(r["weight"] == 0.0 for r in per_symbol.values())
    assert len(per_symbol) == len(UNIVERSE)


def test_nan_weight_is_treated_as_flat() -> None:
    weights = pd.Series({"BTCUSDT": float("nan"), "ETHUSDT": 0.1})
    per_symbol, book = build_payload({}, weights, universe=UNIVERSE, stamp_ms=STAMP)
    assert per_symbol["BTCUSDT"]["weight"] == 0.0
    assert book["gross"] == 0.1


def test_publish_writes_every_symbol_and_the_book() -> None:
    kv = FakeKV()
    weights = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": -0.5})
    per_symbol, book = build_payload(
        {"BTCUSDT": "LONG", "ETHUSDT": "SHORT"},
        weights,
        universe=UNIVERSE,
        stamp_ms=STAMP,
    )
    publish(kv, per_symbol, book)

    for sym in UNIVERSE:
        stored = kv.get_json(weight_key(WEIGHTS_PREFIX, sym))
        assert stored is not None, f"{sym} missing from the online store"
        assert stored["stamp_ms"] == STAMP

    stored_book = kv.get_json(BOOK_KEY)
    assert stored_book["n_symbols"] == len(UNIVERSE)
    assert stored_book["weights"]["BTCUSDT"] == 0.5
