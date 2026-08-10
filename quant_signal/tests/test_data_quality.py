"""Data-quality / SLA tests — hermetic (no Kafka/Redis/Snowflake).

Scores the five-pillar observability model (Soda) / six Elementary dimensions
over synthetic window histories: freshness, volume, completeness, uniqueness,
ordering, gaps, validity, accuracy, plus the cross-symbol consistency
aggregate and the Settings-derived lineage manifest. Every assertion is
deterministic and every threshold comes from ``QualityPolicy`` (relative to the
cadence, never a literal).
"""

from __future__ import annotations

from config.settings import Settings
from stream.data_quality import (
    QualityPolicy,
    _aggregate,
    _ends,
    _gaps,
    _order_stats,
    lineage_manifest,
    quality_report,
    quality_summary,
)
from stream.kv import FakeKV
from stream.materializer import feature_key, live_key

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


def _settings(**overrides: object) -> Settings:
    return Settings(
        snowflake_account="a",
        snowflake_user="u",
        snowflake_password="p",
        _env_file=None,
        **overrides,
    )


def _policy(**overrides: object) -> QualityPolicy:
    base: dict[str, object] = dict(
        window_ms=HOUR_MS,
        freshness_max_cadences=2.0,
        volume_lookback_ms=DAY_MS,
        volume_warn_ratio=0.9,
        volume_min_ratio=0.8,
        max_duplicate_ratio=0.01,
        max_out_of_order=0,
        max_gaps_warn=1,
        required_fields=("open", "high", "low", "close", "vwap", "volume", "bar_count"),
        min_completeness=0.9,
        score_ok=0.9,
        score_warn=0.7,
    )
    base.update(overrides)
    return QualityPolicy(**base)  # type: ignore[arg-type]


def _feature_window(i: int, *, corrupt: bool = False, step_ms: int = HOUR_MS) -> dict:
    """Synthetic 1h window with plausible prices (positive, high≥low, ~0.1% move)."""
    return {
        "symbol": "BTCUSDT",
        "window_start_ms": i * step_ms,
        "window_end_ms": (i + 1) * step_ms,
        "open": 1000.0 + i,
        "high": 1000.5 + i,
        "low": 999.5 + i,
        "close": (-5.0 if corrupt else 1000.5 + i),
        "vwap": 1000.3 + i,
        "volume": 1000.0,
        "bar_count": 5,
    }


def _windows(n: int = 24, **kwargs: object) -> list[dict]:
    return [_feature_window(i, **kwargs) for i in range(n)]


# ── warming / healthy baselines ──────────────────────────────────────────────


def test_quality_report_warming_on_no_windows() -> None:
    report = quality_report([], symbol="BTCUSDT", policy=_policy())
    assert report["status"] == "warming"
    assert report["overall_score"] is None
    assert report["n_windows"] == 0
    assert all(d["status"] == "warming" for d in report["dimensions"].values())


def test_quality_report_healthy_on_contiguous_history() -> None:
    report = quality_report(_windows(24), symbol="BTCUSDT", policy=_policy(), freshness_seconds=0.0)
    assert report["status"] == "healthy"
    assert report["n_windows"] == 24
    assert report["overall_score"] == 1.0
    assert report["duplicates_count"] == 0
    assert report["out_of_order_count"] == 0
    assert report["gaps_count"] == 0
    assert report["dimensions"]["freshness"]["score"] == 1.0
    assert report["dimensions"]["volume"]["present_windows"] == 24
    assert report["dimensions"]["volume"]["expected_windows"] == 24
    assert report["dimensions"]["validity"]["score"] == 1.0
    assert report["dimensions"]["accuracy"]["score"] == 1.0


# ── uniqueness / ordering / gaps ─────────────────────────────────────────────


def test_quality_report_counts_duplicates() -> None:
    windows = [_feature_window(0), _feature_window(0), _feature_window(1)]
    report = quality_report(windows, symbol="BTCUSDT", policy=_policy())
    assert report["duplicates_count"] == 1
    assert report["dimensions"]["uniqueness"]["duplicates"] == 1
    assert report["dimensions"]["uniqueness"]["score"] == round(1 - 1 / 3, 4)


def test_quality_report_detects_out_of_order_in_arrival_order() -> None:
    # Arrival order is 6h → 4h → 5h: one non-increasing transition. Scoring
    # sorts, but ordering must read the raw stored sequence or it is blind.
    windows = [_feature_window(5), _feature_window(3), _feature_window(4)]
    report = quality_report(windows, symbol="BTCUSDT", policy=_policy())
    assert report["out_of_order_count"] == 1
    assert report["dimensions"]["ordering"]["out_of_order"] == 1
    assert report["dimensions"]["ordering"]["score"] == round(1 - 1 / 3, 4)


def test_order_stats_reads_raw_arrival_order() -> None:
    assert _order_stats([_feature_window(5), _feature_window(3), _feature_window(4)]) == (1, 3)
    assert _order_stats([_feature_window(0), _feature_window(1)]) == (0, 2)


def test_quality_report_counts_gaps() -> None:
    windows = [_feature_window(0), _feature_window(2), _feature_window(3)]
    report = quality_report(windows, symbol="BTCUSDT", policy=_policy())
    assert report["gaps_count"] == 1
    assert 2 * HOUR_MS in report["gaps_sample"]  # the missing 1h slot


def test_gaps_function_lists_missing_slots() -> None:
    pairs = _ends([_feature_window(0), _feature_window(3)])
    count, sample = _gaps(pairs, HOUR_MS)
    assert count == 2
    assert set(sample) == {2 * HOUR_MS, 3 * HOUR_MS}


# ── validity / accuracy ──────────────────────────────────────────────────────


def test_quality_report_validity_penalizes_missing_required_field() -> None:
    windows = _windows(3)
    windows[1]["close"] = None
    report = quality_report(windows, symbol="BTCUSDT", policy=_policy())
    dim = report["dimensions"]["validity"]
    assert dim["valid"] == 2
    assert dim["total"] == 3
    assert dim["score"] == round(2 / 3, 4)


def test_quality_report_accuracy_rejects_corrupt_bars() -> None:
    windows = _windows(2)
    windows[0]["close"] = -5.0  # negative close: corrupt
    report = quality_report(windows, symbol="BTCUSDT", policy=_policy())
    dim = report["dimensions"]["accuracy"]
    assert dim["plausible"] == 1
    assert dim["score"] == 0.5


# ── volume / completeness / freshness ────────────────────────────────────────


def test_quality_report_volume_uses_relative_baseline() -> None:
    # 12 of the expected 24 hourly windows within the trailing day → 0.5.
    report = quality_report(_windows(12), symbol="BTCUSDT", policy=_policy())
    assert report["dimensions"]["volume"]["present_windows"] == 12
    assert report["dimensions"]["volume"]["expected_windows"] == 24
    assert report["dimensions"]["volume"]["score"] == 0.5
    assert report["dimensions"]["completeness"]["score"] == 0.5


def test_quality_report_freshness_from_event_time_lag() -> None:
    # SLA = 2 cadences (7200s); a 1h lag → score 1 - 3600/14400 = 0.75 (warning).
    report = quality_report(
        _windows(24), symbol="BTCUSDT", policy=_policy(), freshness_seconds=3_600.0
    )
    dim = report["dimensions"]["freshness"]
    assert dim["score"] == 0.75
    assert dim["age_seconds"] == 3_600.0
    assert dim["status"] == "warning"


def test_quality_report_freshness_critical_when_beyond_sla() -> None:
    # 3h lag vs 2h SLA → 1 - 10800/14400 = 0.25 → the fleet's worst dimension.
    report = quality_report(
        _windows(24), symbol="BTCUSDT", policy=_policy(), freshness_seconds=3 * 3_600.0
    )
    assert report["dimensions"]["freshness"]["score"] == 0.25
    assert report["dimensions"]["freshness"]["status"] == "critical"
    assert report["status"] == "critical"


# ── cross-symbol aggregate ───────────────────────────────────────────────────


def test_aggregate_consistency_drops_behind_symbol() -> None:
    policy = _policy()
    a = quality_report(_windows(24), symbol="BTCUSDT", policy=policy, freshness_seconds=0.0)
    b = quality_report(
        _windows(23), symbol="ETHUSDT", policy=policy, freshness_seconds=0.0
    )  # one window behind
    agg = _aggregate([a, b], policy)
    assert agg["n_symbols"] == 2
    assert agg["consistency_score"] == 0.5
    assert agg["dimensions"]["consistency"]["score"] == 0.5


def test_aggregate_worst_dimension_attributes_symbol() -> None:
    policy = _policy()
    a_rows = _windows(24)
    a_rows[0]["close"] = -1.0  # corrupt → accuracy 23/24
    a = quality_report(a_rows, symbol="BTCUSDT", policy=policy, freshness_seconds=0.0)
    b = quality_report(_windows(24), symbol="ETHUSDT", policy=policy, freshness_seconds=0.0)
    agg = _aggregate([a, b], policy)
    assert agg["dimensions"]["accuracy"]["score"] == round(23 / 24, 4)
    assert agg["dimensions"]["accuracy"]["worst_symbol"] == "BTCUSDT"
    assert agg["overall_score"] == round(23 / 24, 4)  # worst dimension, not an average


# ── lineage + settings wiring ────────────────────────────────────────────────


def test_quality_policy_from_settings() -> None:
    policy = QualityPolicy.from_settings(_settings())
    assert policy.window_ms == HOUR_MS
    assert policy.volume_lookback_ms == DAY_MS
    assert policy.required_fields == (
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "bar_count",
    )


def test_lineage_manifest_stages_are_settings_derived() -> None:
    settings = _settings()
    manifest = lineage_manifest(settings)
    assert len(manifest) == 7
    assert manifest[0]["output"] == settings.stream_kafka_topic_raw
    assert manifest[1]["output"] == settings.stream_kafka_topic_features
    assert manifest[2]["output"] == settings.stream_redis_feature_prefix
    assert all(m["freshness_owner"] for m in manifest)


def test_quality_summary_over_fakekv() -> None:
    settings = _settings()
    kv = FakeKV()
    for i in range(24):
        kv.push_json(
            feature_key(settings.stream_redis_feature_prefix, "BTCUSDT"),
            _feature_window(i),
            maxlen=200,
        )
        kv.push_json(
            feature_key(settings.stream_redis_feature_prefix, "ETHUSDT"),
            _feature_window(i),
            maxlen=200,
        )
    # Live raw bar exactly at the latest window end → zero event-time staleness.
    kv.set_json(live_key(settings.stream_redis_live_prefix, "BTCUSDT"), {"ts": 24 * HOUR_MS})
    kv.set_json(live_key(settings.stream_redis_live_prefix, "ETHUSDT"), {"ts": 24 * HOUR_MS})

    summary = quality_summary(kv, settings)
    assert summary["computed_at_ms"] > 0
    assert summary["healthy"] is True
    assert summary["overall_score"] == 1.0
    assert summary["consistency"]["score"] == 1.0
    # All 4 default symbols reported, 2 with materialized history.
    assert len(summary["symbols"]) == 4
    assert len(summary["lineage"]) == 7
    assert summary["policy"]["window_ms"] == HOUR_MS
