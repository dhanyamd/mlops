"""Pipeline health summary — per-stage freshness in *event time*.

Every downstream artifact tags the feature window it was derived from
(``window_end_ms``), and raw bars carry the venue's event timestamp (``ts``).
Comparing those event timestamps against the latest feature window gives the
age of each stage without ever trusting the host clock (which drifts):

    features  age = latest raw bar ts − latest feature window_end_ms
    predict   age = prediction window_end_ms − latest feature window_end_ms
    simulate  age = simulation window_end_ms − latest feature window_end_ms
    strategy  age = strategy window_end_ms    − latest feature window_end_ms

A stage whose tag equals the latest window is fresh (age 0); one or more
windows behind has a positive age; a missing artifact reports ``None``
("warming up", not an error). All prefixes come from Settings — the health
layer never hardcodes a store key.
"""

from __future__ import annotations

from scripts.stream_watchdog import staleness_seconds
from stream.kv import KVStore
from stream.materializer import feature_key, live_key
from stream.predictor import prediction_key, strategy_key
from stream.simulation import simulation_key


def _latest_window_end(kv: KVStore, feature_prefix: str, symbol: str) -> int | None:
    rows = kv.list_json(feature_key(feature_prefix, symbol), reverse=True, maxlen=1)
    if not rows:
        return None
    end = rows[0].get("window_end_ms")
    return int(end) if isinstance(end, (int, float)) else None


def _artifact_age(artifact: dict | None, latest_end: int | None) -> float | None:
    """Seconds the artifact's tagged window lags the latest feature window.

    None when there is no artifact or it carries no event-time tag. A negative
    value (artifact ahead of the stored window) is clamped to fresh (0).
    """
    if artifact is None or latest_end is None:
        return None
    tag = artifact.get("window_end_ms")
    if not isinstance(tag, (int, float)):
        return None
    return max(0.0, float(latest_end - tag) / 1000.0)


def _status(age: float | None, threshold: float) -> str:
    if age is None:
        return "warming"
    return "healthy" if age <= threshold else "stale"


def pipeline_summary(
    kv: KVStore,
    *,
    symbols: list[str],
    live_prefix: str,
    feature_prefix: str,
    prediction_prefix: str,
    simulation_prefix: str,
    strategy_prefix: str,
    staleness_threshold: float,
) -> dict:
    """Per-symbol stage freshness, derived entirely from event timestamps."""
    stages: list[dict] = []
    any_stale = False

    for symbol in symbols:
        latest_end = _latest_window_end(kv, feature_prefix, symbol)

        live = kv.get_json(live_key(live_prefix, symbol))
        features_age = staleness_seconds(
            kv,
            live_prefix=live_prefix,
            feature_prefix=feature_prefix,
            symbol=symbol,
        )
        prediction = kv.get_json(prediction_key(prediction_prefix, symbol))
        simulation = kv.get_json(simulation_key(simulation_prefix, symbol))
        strategy = kv.get_json(strategy_key(strategy_prefix, symbol))

        feature_stage = {
            "name": "features",
            "symbol": symbol,
            "status": _status(features_age, staleness_threshold),
            "age_seconds": features_age,
            "detail": (
                f"Flink windows end {int(features_age)}s behind raw bars"
                if features_age is not None
                else "no feature windows yet"
            ),
        }
        stages.append(
            {
                "name": "produce",
                "symbol": symbol,
                "status": "healthy" if live else "warming",
                "age_seconds": None,
                "venue": live.get("provider") if live else None,
                "detail": (
                    f"live bar {live.get('ts')} · venue {live.get('provider') or 'unknown'}"
                    if live
                    else "no live bar yet"
                ),
            }
        )
        stages.append(feature_stage)

        for name, artifact in (
            ("predict", prediction),
            ("simulate", simulation),
            ("strategy", strategy),
        ):
            age = _artifact_age(artifact, latest_end)
            status = _status(age, staleness_threshold)
            stages.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "status": status,
                    "age_seconds": age,
                    "detail": (
                        f"{name} window {int(age)}s behind latest feature window"
                        if age is not None
                        else "warming up"
                    ),
                }
            )
        any_stale = (
            any_stale
            or feature_stage["status"] == "stale"
            or any(s["status"] == "stale" for s in stages[-3:])
        )

    return {
        "healthy": not any_stale,
        "threshold_seconds": staleness_threshold,
        "stages": stages,
    }
