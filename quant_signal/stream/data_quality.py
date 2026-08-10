"""Data-quality / SLA / visibility layer over the streaming window store.

The five-pillar data-observability model (freshness, volume, distribution,
schema, lineage — Soda's 5 pillars; Elementary's six quality dimensions:
completeness, uniqueness, freshness, validity, accuracy, consistency) applied
to the materialized feature-window history:

  - **freshness** — how far the latest window lags the venue's raw bars, in
    *event time* (same delta the watchdog alerts on; never the host clock);
  - **volume** — windows actually present in the trailing lookback vs the
    expected count for the cadence (24 windows/day at 1h), scored as a ratio
    against a relative baseline — Soda: "use relative thresholds rather than
    absolute ones" (~15–20% deviation is an anomaly);
  - **uniqueness** — duplicate ``window_end_ms`` (at-least-once delivery can
    deliver a window twice; dupes accumulate silently in downstream counts);
  - **ordering** — out-of-order windows break the chronology the online
    learner depends on (vexdata: sequence-number verification per entity);
  - **gaps** — missing window slots within the held history (a dropped Kafka
    message leaves a hole; holes are surfaced, never silently absorbed);
  - **validity** — every required field present with a usable type;
  - **accuracy** — values within plausible bounds (positive close, sane
    high≥low, ratio features within the predictor's ±50% sanity clamp);
  - **consistency** — symbols advance together (share at the same latest
    window); computed at the aggregate, cross-symbol level.

Per-symbol dimension scores are 0..1; the overall score is the WORST
dimension (Elementary-style health where one broken dimension means the
asset is not healthy — a weighted average can be gamed). Status bands
(healthy / warning / critical) come from Settings, never literals.

The lineage manifest maps the pipeline stages (venue → topic → Flink →
online store → consumers) from Settings keys, so blast radius and freshness
owners are queryable, not a stale diagram (Soda: "use lineage to scope
incident response, not just to debug after the fact").

Pure per-window scoring lives here so tests are hermetic; the summary wrapper
reads the online store (like ``pipeline_health.pipeline_summary``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from scripts.stream_watchdog import staleness_seconds
from stream.kv import KVStore
from stream.materializer import feature_key


def _ends(windows: Sequence[Mapping]) -> list[tuple[int, dict]]:
    """(window_end_ms, window) pairs with a valid integer end, sorted ascending."""
    pairs: list[tuple[int, dict]] = []
    for w in windows:
        end = w.get("window_end_ms")
        if isinstance(end, (int, float)) and end == end:
            pairs.append((int(end), dict(w)))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _order_stats(windows: Sequence[Mapping]) -> tuple[int, int]:
    """(out_of_order, total) in the STORED arrival order.

    The online store appends windows in arrival order (RPUSH+LTRIM), so any
    non-increasing ``window_end_ms`` transition is an out-of-order delivery —
    the chronology the online learner depends on (vexdata: verify sequence
    numbers per entity). Scoring sorts for determinism, but ordering must read
    the raw order or the check is blind.
    """
    ends: list[int] = []
    for w in windows:
        end = w.get("window_end_ms")
        if isinstance(end, (int, float)) and end == end:
            ends.append(int(end))
    out_of_order = sum(1 for a, b in zip(ends, ends[1:]) if b <= a)
    return out_of_order, len(ends)


@dataclass(frozen=True)
class QualityPolicy:
    """Thresholds for every quality dimension — built from Settings (never
    literals in the scoring code). All timing is RELATIVE to the cadence."""

    window_ms: int
    freshness_max_cadences: float
    volume_lookback_ms: int
    volume_warn_ratio: float
    volume_min_ratio: float
    max_duplicate_ratio: float
    max_out_of_order: int
    max_gaps_warn: int
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    min_completeness: float = 0.9
    score_ok: float = 0.9
    score_warn: float = 0.7

    @classmethod
    def from_settings(cls, settings: Any) -> "QualityPolicy":
        from config.settings import csv_list

        return cls(
            window_ms=int(settings.stream_window_ms),
            freshness_max_cadences=float(settings.quality_freshness_max_cadences),
            volume_lookback_ms=int(settings.quality_volume_lookback_hours * 3_600_000),
            volume_warn_ratio=float(settings.quality_volume_warn_ratio),
            volume_min_ratio=float(settings.quality_volume_min_ratio),
            max_duplicate_ratio=float(settings.quality_max_duplicate_ratio),
            max_out_of_order=int(settings.quality_max_out_of_order),
            max_gaps_warn=int(settings.quality_max_gaps_warn),
            required_fields=tuple(csv_list(settings.quality_required_fields)),
            min_completeness=float(settings.quality_min_completeness),
            score_ok=float(settings.quality_score_ok),
            score_warn=float(settings.quality_score_warn),
        )


def _status(score: float, policy: QualityPolicy) -> str:
    if score >= policy.score_ok:
        return "healthy"
    if score >= policy.score_warn:
        return "warning"
    return "critical"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _trailing(
    pairs: list[tuple[int, dict]], lookback_ms: int, now_ms: int
) -> list[tuple[int, dict]]:
    """Windows whose end falls within the trailing lookback ending at now_ms."""
    cutoff = now_ms - lookback_ms
    return [p for p in pairs if p[0] >= cutoff]


def _gaps(pairs: list[tuple[int, dict]], window_ms: int, sample: int = 10) -> tuple[int, list[int]]:
    """Missing window slots within the held history.

    Expected ends form an arithmetic series (window_ms cadence) from the
    oldest to the newest held window; any slot absent from the set is a gap
    (a dropped message or an unwritten window). Returns (count, sample ends).
    """
    if len(pairs) < 2:
        return 0, []
    first = pairs[0][0]
    last = pairs[-1][0]
    present = {p[0] for p in pairs}
    missing: list[int] = []
    for end in range(first, last + 1, window_ms):
        if end not in present:
            missing.append(end)
    return len(missing), missing[:sample]


def _validity(pairs: list[tuple[int, dict]], required: tuple[str, ...]) -> tuple[int, int]:
    """(passed, total) windows carrying every required field with a usable type."""
    ok = 0
    for _end, w in pairs:
        if all(isinstance(w.get(f), (int, float)) and w.get(f) == w.get(f) for f in required):
            ok += 1
    return ok, len(pairs)


def _accuracy(pairs: list[tuple[int, dict]]) -> tuple[int, int]:
    """(passed, total) windows whose values are plausible (predictor clamps).

    Mirrors the sanity bounds the predictor enforces (``_MAX_FEATURE_RATIO``
    = ±50% on ratio features): a positive close, high ≥ low, a positive bar
    count, and ratio features within the half — anything else is a corrupt
    bar that would poison the online scalers.
    """
    ok = 0
    for _end, w in pairs:
        close = w.get("close")
        if not isinstance(close, (int, float)) or close <= 0:
            continue
        if isinstance(w.get("high"), (int, float)) and isinstance(w.get("low"), (int, float)):
            if w["high"] < w["low"]:
                continue
        ratios: list[float] = []
        if isinstance(w.get("open"), (int, float)) and w["open"]:
            ratios.append(abs(close / w["open"] - 1.0))
        if isinstance(w.get("high"), (int, float)) and isinstance(w.get("low"), (int, float)):
            ratios.append(abs((w["high"] - w["low"]) / close))
        if isinstance(w.get("vwap"), (int, float)) and w["vwap"]:
            ratios.append(abs((w["vwap"] - close) / close))
        if any(r > 0.5 for r in ratios):
            continue
        bar_count = w.get("bar_count")
        if isinstance(bar_count, (int, float)) and bar_count <= 0:
            continue
        ok += 1
    return ok, len(pairs)


def quality_report(
    windows: Sequence[Mapping],
    *,
    symbol: str,
    policy: QualityPolicy,
    freshness_seconds: float | None = None,
    now_ms: int | None = None,
) -> dict:
    """Per-symbol quality dimensions over the held window history.

    ``freshness_seconds`` is the event-time lag of the latest window behind
    the venue's raw bars (computed by the caller via the watchdog's
    ``staleness_seconds``); when None the symbol is still warming up and
    freshness is scored neutral rather than critical. ``now_ms`` anchors the
    volume/completeness lookback (defaults to the latest window's end so the
    report is reproducible without a wall clock).
    """
    pairs = _ends(windows)
    total = len(pairs)
    if total == 0:
        return {
            "symbol": symbol,
            "n_windows": 0,
            "status": "warming",
            "dimensions": {
                "freshness": {"score": None, "status": "warming"},
                "volume": {"score": None, "status": "warming"},
                "completeness": {"score": None, "status": "warming"},
                "uniqueness": {"score": None, "status": "warming"},
                "ordering": {"score": None, "status": "warming"},
                "validity": {"score": None, "status": "warming"},
                "accuracy": {"score": None, "status": "warming"},
                "consistency": {"score": None, "status": "warming"},
            },
            "overall_score": None,
            "gaps_count": 0,
            "duplicates_count": 0,
            "out_of_order_count": 0,
            "latest_window_end_ms": None,
        }

    now = now_ms if now_ms is not None else pairs[-1][0]

    # Freshness — event-time lag vs the venue's latest raw bar.
    sla_seconds = policy.freshness_max_cadences * policy.window_ms / 1000.0
    if freshness_seconds is None:
        freshness_score = None
        freshness_status = "warming"
    else:
        freshness_score = _clamp01(1.0 - freshness_seconds / (2.0 * sla_seconds))
        freshness_status = _status(freshness_score, policy)

    # Volume + completeness over the trailing lookback (relative baseline).
    trailing = _trailing(pairs, policy.volume_lookback_ms, now)
    expected = max(1, policy.volume_lookback_ms // policy.window_ms)
    present = len(trailing)
    completeness = _clamp01(present / expected)

    # Uniqueness over the held history; ordering over the raw stored sequence.
    ends = [p[0] for p in pairs]
    duplicates = len(ends) - len(set(ends))
    uniqueness = _clamp01(1.0 - (duplicates / total if total else 0.0))
    out_of_order, order_total = _order_stats(windows)
    ordering_score = _clamp01(1.0 - out_of_order / (order_total if order_total else 1.0))

    gaps_count, gaps_sample = _gaps(pairs, policy.window_ms)

    valid_ok, valid_total = _validity(pairs, policy.required_fields)
    validity = _clamp01(valid_ok / valid_total) if valid_total else 0.0

    acc_ok, acc_total = _accuracy(pairs)
    accuracy = _clamp01(acc_ok / acc_total) if acc_total else 0.0

    dimensions = {
        "freshness": {
            "score": round(freshness_score, 4) if freshness_score is not None else None,
            "status": freshness_status,
            "age_seconds": round(freshness_seconds, 1) if freshness_seconds is not None else None,
        },
        "volume": {
            "score": round(completeness, 4),
            "status": _status(completeness, policy),
            "present_windows": present,
            "expected_windows": expected,
        },
        "completeness": {
            "score": round(completeness, 4),
            "status": _status(completeness, policy),
            "present": present,
            "expected": expected,
        },
        "uniqueness": {
            "score": round(uniqueness, 4),
            "status": _status(uniqueness, policy),
            "duplicates": duplicates,
            "total": total,
        },
        "ordering": {
            "score": round(ordering_score, 4),
            "status": _status(ordering_score, policy),
            "out_of_order": out_of_order,
        },
        "validity": {
            "score": round(validity, 4),
            "status": _status(validity, policy),
            "valid": valid_ok,
            "total": valid_total,
        },
        "accuracy": {
            "score": round(accuracy, 4),
            "status": _status(accuracy, policy),
            "plausible": acc_ok,
            "total": acc_total,
        },
        "consistency": {"score": None, "status": "warming", "detail": "computed across symbols"},
    }

    scored = [d["score"] for d in dimensions.values() if d["score"] is not None]
    overall = min(scored) if scored else None

    return {
        "symbol": symbol,
        "n_windows": total,
        "status": _status(overall, policy) if overall is not None else "warming",
        "dimensions": dimensions,
        "overall_score": round(overall, 4) if overall is not None else None,
        "gaps_count": gaps_count,
        "gaps_sample": gaps_sample,
        "duplicates_count": duplicates,
        "out_of_order_count": out_of_order,
        "latest_window_end_ms": ends[-1],
        "policy": {
            "window_ms": policy.window_ms,
            "freshness_max_cadences": policy.freshness_max_cadences,
            "volume_lookback_ms": policy.volume_lookback_ms,
            "volume_min_ratio": policy.volume_min_ratio,
        },
    }


def _aggregate(symbol_reports: Sequence[dict], policy: QualityPolicy) -> dict:
    """Cross-symbol view: worst-per-dimension + consistency alignment.

    Consistency = the share of symbols whose latest window matches the most
    recent common window end (the pipeline advances together — one stale
    symbol drags the fleet).
    """
    reports = [r for r in symbol_reports if r["n_windows"] > 0]
    if not reports:
        return {"n_symbols": 0, "consistency_score": None, "consistency_status": "warming"}

    latest_ends = [r["latest_window_end_ms"] for r in reports]
    newest = max(latest_ends)
    aligned = sum(1 for end in latest_ends if end == newest)
    consistency = _clamp01(aligned / len(reports))

    dim_names = ("freshness", "volume", "uniqueness", "ordering", "validity", "accuracy")
    dimensions: dict[str, dict[str, object]] = {}
    for name in dim_names:
        scores = [
            r["dimensions"][name]["score"]
            for r in reports
            if r["dimensions"][name]["score"] is not None
        ]
        score = min(scores) if scores else None
        dimensions[name] = {
            "score": round(score, 4) if score is not None else None,
            "status": _status(score, policy) if score is not None else "warming",
            "worst_symbol": next(
                (r["symbol"] for r in reports if r["dimensions"][name]["score"] == score),
                None,
            ),
        }
    dimensions["consistency"] = {
        "score": round(consistency, 4),
        "status": _status(consistency, policy),
        "detail": f"{aligned}/{len(reports)} symbols at the latest window",
    }

    scored = [d["score"] for d in dimensions.values() if d["score"] is not None]
    overall = min(scored) if scored else None
    return {
        "n_symbols": len(reports),
        "consistency_score": round(consistency, 4),
        "consistency_status": _status(consistency, policy),
        "dimensions": dimensions,
        "overall_score": round(overall, 4) if overall is not None else None,
    }


def lineage_manifest(settings: Any) -> list[dict]:
    """Stage-by-stage data-flow manifest derived from Settings.

    Every hop is a real store/topic/key from config — nothing hardcoded — so
    blast radius is queryable: when freshness breaks upstream, the manifest
    answers "what depends on this window?" (Soda: lineage scopes incident
    response instead of debugging after the fact).
    """
    return [
        {
            "stage": "produce",
            "input": "venue (Binance/Bybit minute bars)",
            "output": settings.stream_kafka_topic_raw,
            "artifact": "raw bar (live:crypto:<SYMBOL>)",
            "freshness_owner": "producer agent",
        },
        {
            "stage": "window",
            "input": settings.stream_kafka_topic_raw,
            "output": settings.stream_kafka_topic_features,
            "artifact": f"Flink window features ({settings.stream_kafka_topic_features})",
            "freshness_owner": "Flink job",
        },
        {
            "stage": "materialize",
            "input": settings.stream_kafka_topic_features,
            "output": settings.stream_redis_feature_prefix,
            "artifact": "feature:crypto:1h:<SYMBOL> (bounded list)",
            "freshness_owner": "materializer agent",
        },
        {
            "stage": "predict",
            "input": settings.stream_redis_feature_prefix,
            "output": settings.stream_redis_prediction_prefix,
            "artifact": "prediction:crypto:1h:<SYMBOL>",
            "freshness_owner": "predictor agent",
        },
        {
            "stage": "simulate",
            "input": settings.stream_redis_prediction_prefix,
            "output": settings.stream_redis_simulation_prefix,
            "artifact": "simulation:crypto:1h:<SYMBOL>",
            "freshness_owner": "simulation agent",
        },
        {
            "stage": "execute",
            "input": settings.stream_redis_prediction_prefix,
            "output": settings.stream_redis_execution_prefix,
            "artifact": "execution:crypto:1h:<SYMBOL>",
            "freshness_owner": "execution agent",
        },
        {
            "stage": "serve",
            "input": settings.stream_redis_feature_prefix,
            "output": "dashboard API /ws/market",
            "artifact": "read-only API reads",
            "freshness_owner": "api agent",
        },
    ]


def quality_summary(kv: KVStore, settings: Any) -> dict:
    """Full pipeline quality: per-symbol reports + cross-symbol aggregate +
    lineage, sourced entirely from the online store (no Kafka/Snowflake).

    Freshness uses the watchdog's event-time delta (latest raw bar vs latest
    window); all other dimensions come from the held window history.
    """
    from config.settings import csv_list

    symbols = csv_list(settings.ingest_default_crypto_symbols)
    policy = QualityPolicy.from_settings(settings)

    reports: list[dict] = []
    for symbol in symbols:
        rows = kv.list_json(
            feature_key(settings.stream_redis_feature_prefix, symbol),
            reverse=False,
            maxlen=settings.stream_redis_feature_maxlen,
        )
        freshness = staleness_seconds(
            kv,
            live_prefix=settings.stream_redis_live_prefix,
            feature_prefix=settings.stream_redis_feature_prefix,
            symbol=symbol,
        )
        reports.append(
            quality_report(
                rows,
                symbol=symbol.upper(),
                policy=policy,
                freshness_seconds=freshness,
            )
        )

    aggregate = _aggregate(reports, policy)
    return {
        "computed_at_ms": int(datetime.now(UTC).timestamp() * 1000),
        "healthy": (
            aggregate.get("overall_score") is not None
            and aggregate["overall_score"] >= policy.score_ok
        ),
        "overall_score": aggregate.get("overall_score"),
        "consistency": {
            "score": aggregate.get("consistency_score"),
            "status": aggregate.get("consistency_status"),
        },
        "dimensions": aggregate.get("dimensions", {}),
        "symbols": reports,
        "lineage": lineage_manifest(settings),
        "policy": {
            "window_ms": policy.window_ms,
            "freshness_max_cadences": policy.freshness_max_cadences,
            "volume_lookback_ms": policy.volume_lookback_ms,
            "volume_min_ratio": policy.volume_min_ratio,
            "max_duplicate_ratio": policy.max_duplicate_ratio,
            "max_out_of_order": policy.max_out_of_order,
        },
    }
