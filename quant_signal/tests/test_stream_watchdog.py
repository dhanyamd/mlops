"""Watchdog staleness detection is hermetic and clock-skew immune."""

from __future__ import annotations

from scripts.stream_watchdog import staleness_seconds
from stream.kv import KVStore


class _FakeKV(KVStore):
    """Minimal in-memory store: one live bar + one feature window."""

    def __init__(self, live_ts: int, feature_end_ms: int | None) -> None:
        self._live_ts = live_ts
        self._feature_end_ms = feature_end_ms

    def set_json(self, key: str, value: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_json(self, key: str) -> dict | None:
        if key.endswith("BTCUSDT"):
            return {"ts": self._live_ts}
        return None

    def list_json(
        self, key: str, *, reverse: bool = False, maxlen: int | None = None
    ) -> list[dict]:
        if self._feature_end_ms is None:
            return []
        return [{"window_end_ms": self._feature_end_ms}]


def test_staleness_healthy_window() -> None:
    # Feature window ended 90s before the latest raw bar: fresh.
    stale = staleness_seconds(
        _FakeKV(live_ts=1_800_000_000_000, feature_end_ms=1_799_999_910_000),
        live_prefix="live:crypto",
        feature_prefix="feature:crypto:5m",
        symbol="BTCUSDT",
    )
    assert stale == 90.0


def test_staleness_stalled_pipeline() -> None:
    # Feature window ended ~5.8 days before the latest raw bar: stale.
    stale = staleness_seconds(
        _FakeKV(live_ts=1_800_000_000_000, feature_end_ms=1_799_500_000_000),
        live_prefix="live:crypto",
        feature_prefix="feature:crypto:5m",
        symbol="BTCUSDT",
    )
    assert stale == 500_000.0


def test_staleness_none_without_data() -> None:
    # No feature windows yet is "warming up", not stale.
    stale = staleness_seconds(
        _FakeKV(live_ts=1_800_000_000_000, feature_end_ms=None),
        live_prefix="live:crypto",
        feature_prefix="feature:crypto:5m",
        symbol="BTCUSDT",
    )
    assert stale is None
