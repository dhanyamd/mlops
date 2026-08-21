"""Autonomous research harness CLI: sweep the predictor over real stored data.

Loads the materialized feature-window history per symbol from the online
store (the SAME windows the live gate replays — no synthetic data), builds
the candidate grid from Settings, replays the exact progressive-validation
loop over every configuration, ranks the trials, deflates the winner for the
total trial count, and persists each symbol's leaderboard to
``research:harness:leaderboard:<SYMBOL>``. Optionally logs the sweep to
MLflow Tracking (``track_research_sweep``) when enabled.

Run with ``make stream-research`` (or ``python -m scripts.run_research``).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

from config.logging import configure_logging, get_logger
from config.settings import csv_floats, csv_list, get_settings
from stream.kv import KVStore, RedisKV
from stream.materializer import feature_key
from stream.mlflow_tracking import track_research_sweep
from stream.research_harness import build_candidates, run_sweep, summarize_sweep

log = get_logger(__name__)


def collect_windows(
    kv: KVStore,
    symbols: Sequence[str],
    prefix: str,
    maxlen: int,
) -> dict[str, list[dict]]:
    """All stored windows per symbol (oldest first), for the replay.

    Pulls only what already exists in the online store — the harness is an
    offline sweep over the live pipeline's materialized history, never a
    separate feed.
    """
    windows: dict[str, list[dict]] = {}
    for symbol in symbols:
        rows = kv.list_json(feature_key(prefix, symbol), reverse=False, maxlen=maxlen)
        if rows:
            windows[symbol.upper()] = [dict(r) for r in rows]
    return windows


def main() -> None:
    configure_logging()
    settings = get_settings()

    kv = RedisKV(settings.stream_redis_url)
    symbols = csv_list(settings.ingest_default_crypto_symbols)
    windows = collect_windows(
        kv,
        symbols,
        settings.stream_redis_feature_prefix,
        settings.stream_redis_feature_maxlen,
    )
    if not windows:
        log.error(
            "research_harness_no_data",
            prefix=settings.stream_redis_feature_prefix,
            symbols=symbols,
        )
        sys.exit(1)

    candidates = build_candidates(
        lambda_values=csv_floats(settings.research_lambda_values),
        taker_cost_values=csv_floats(settings.research_taker_cost_values),
        feature_modes=csv_list(settings.research_feature_modes),
        method=settings.research_search_method,
        budget=settings.research_budget,
        seed=settings.research_seed,
    )
    log.info(
        "research_harness_start",
        symbols=list(windows),
        candidates=len(candidates),
        method=settings.research_search_method,
    )

    for symbol, rows in windows.items():
        cross_windows: Mapping[str, Sequence[Mapping]] = {
            s: w for s, w in windows.items() if s != symbol
        }
        trials = run_sweep(
            rows,
            candidates=candidates,
            cross_windows=cross_windows,
            alpha=settings.stream_prediction_alpha,
            gamma=settings.stream_prediction_gamma,
            residual_window=settings.stream_prediction_residual_window,
            n_blocks=settings.research_n_blocks,
        )
        summary = summarize_sweep(trials, settings=settings, n_trials=len(trials))
        key = f"{settings.research_leaderboard_prefix}:{symbol}"
        kv.set_json(key, summary)
        if settings.mlflow_tracking_enabled:
            track_research_sweep(
                symbol,
                summary,
                n_trials=len(trials),
                method=settings.research_search_method,
            )
        winner = summary.get("winner") or {}
        log.info(
            "research_harness_complete",
            symbol=symbol,
            trials=len(trials),
            eligible=summary.get("trials_eligible"),
            winner_params=winner.get("params"),
            winner_passes=winner.get("passes"),
            expected_max_sharpe_noise=summary.get("expected_max_sharpe_noise"),
            leaderboard_key=key,
        )


if __name__ == "__main__":
    main()
