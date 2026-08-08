"""Pipeline health — per-stage freshness derived from event timestamps.

The host clock drifts, so every age is computed from event-time tags
(``window_end_ms`` on each artifact vs. the latest feature window), never
``datetime.now()``. Hermetic via FakeKV; the API endpoint follows the same
RedisKV-monkeypatch pattern as the other market endpoints.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from stream.kv import FakeKV
from stream.materializer import feature_key, live_key
from stream.pipeline_health import pipeline_summary
from stream.predictor import prediction_key, strategy_key
from stream.simulation import simulation_key

_TS = 1_800_000_000_000  # epoch ms reference for the latest feature window
_THRESHOLD = 900.0

_PREFIXES = dict(
    live_prefix="live:crypto",
    feature_prefix="feature:crypto:5m",
    prediction_prefix="prediction:crypto:5m",
    simulation_prefix="simulation:crypto:5m",
    strategy_prefix="strategy:crypto:5m",
)


def _populate(kv: FakeKV, *, feature_end: int = _TS, artifacts_end: int = _TS) -> None:
    """Healthy store: live bar, one feature window, every artifact tagged."""
    kv.set_json(live_key("live:crypto", "BTCUSDT"), {"ts": _TS, "provider": "binance"})
    kv.push_json(
        feature_key("feature:crypto:5m", "BTCUSDT"),
        {"symbol": "BTCUSDT", "window_end_ms": feature_end, "close": 100.0},
        maxlen=200,
    )
    for key_builder, prefix in (
        (prediction_key, "prediction:crypto:5m"),
        (simulation_key, "simulation:crypto:5m"),
        (strategy_key, "strategy:crypto:5m"),
    ):
        kv.set_json(
            key_builder(prefix, "BTCUSDT"), {"symbol": "BTCUSDT", "window_end_ms": artifacts_end}
        )


def _by_name(stages: list[dict], symbol: str) -> dict[str, dict]:
    return {s["name"]: s for s in stages if s["symbol"] == symbol}


def test_summary_all_stages_healthy() -> None:
    kv = FakeKV()
    _populate(kv)

    summary = pipeline_summary(kv, symbols=["BTCUSDT"], staleness_threshold=_THRESHOLD, **_PREFIXES)

    assert summary["healthy"] is True
    by_name = _by_name(summary["stages"], "BTCUSDT")
    assert set(by_name) == {"produce", "features", "predict", "simulate", "strategy"}
    assert all(by_name[n]["status"] == "healthy" for n in by_name)
    assert by_name["features"]["age_seconds"] == 0.0
    assert by_name["predict"]["age_seconds"] == 0.0
    assert by_name["produce"]["venue"] == "binance"
    assert by_name["produce"]["detail"].endswith("venue binance")


def test_summary_marks_stalled_features_stale() -> None:
    kv = FakeKV()
    _populate(kv, feature_end=_TS - 3_600_000)  # features an hour behind raw bars

    summary = pipeline_summary(kv, symbols=["BTCUSDT"], staleness_threshold=_THRESHOLD, **_PREFIXES)

    assert summary["healthy"] is False
    by_name = _by_name(summary["stages"], "BTCUSDT")
    assert by_name["features"]["status"] == "stale"
    assert by_name["features"]["age_seconds"] == 3_600.0


def test_summary_marks_stale_downstream_artifact() -> None:
    kv = FakeKV()
    _populate(kv, artifacts_end=_TS - 1_800_000)  # artifacts six 5m windows behind

    summary = pipeline_summary(kv, symbols=["BTCUSDT"], staleness_threshold=_THRESHOLD, **_PREFIXES)

    assert summary["healthy"] is False
    by_name = _by_name(summary["stages"], "BTCUSDT")
    assert by_name["features"]["status"] == "healthy"  # the pipeline itself is fine
    assert by_name["predict"]["status"] == "stale"
    assert by_name["simulate"]["status"] == "stale"
    assert by_name["strategy"]["status"] == "stale"
    assert by_name["predict"]["age_seconds"] == 1_800.0  # 30 minutes, in seconds


def test_summary_warming_without_data() -> None:
    summary = pipeline_summary(
        kv=FakeKV(), symbols=["BTCUSDT"], staleness_threshold=_THRESHOLD, **_PREFIXES
    )

    assert summary["healthy"] is True  # no data is warming up, not an outage
    by_name = _by_name(summary["stages"], "BTCUSDT")
    assert all(by_name[n]["status"] == "warming" for n in by_name)


def test_health_summary_endpoint_reads_redis(monkeypatch) -> None:
    monkeypatch.setattr("api.main.RedisKV", lambda url: FakeKV())
    with TestClient(app) as client:
        kv = client.app.state.kv
        assert kv is not None
        _populate(kv)
        resp = client.get("/api/market/health/summary")
    body = resp.json()
    assert body["enabled"] is True
    assert body["healthy"] is True
    assert body["threshold_seconds"] == _THRESHOLD
    btc = _by_name(body["stages"], "BTCUSDT")
    assert set(btc) == {"produce", "features", "predict", "simulate", "strategy"}


def test_health_summary_endpoint_disabled_without_stream(monkeypatch) -> None:
    from config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "stream_enabled", False)
    with TestClient(app) as client:
        resp = client.get("/api/market/health/summary")
    body = resp.json()
    assert body["enabled"] is False
    assert body["stages"] == []
