"""Stream watchdog: detect and heal a stalled Flink feature pipeline.

The Flink 5m-window job has (twice) silently frozen: its Kafka source lost the
broker connection, the consumer-group offsets/checkpoints went stale, and the
features topic stopped advancing for many hours while every downstream
consumer kept serving the last window. Nothing surfaced — the API still
returned 200s, so the dashboard just looked "same" until someone noticed.

This watchdog measures staleness from *venue event timestamps* (never the
host clock, which drifts on this machine):

    staleness = latest raw bar ts  −  latest feature window_end_ms

Both live in the Redis online store, so no Kafka client or Snowflake is
needed to check health. When staleness exceeds a threshold it logs a clear
alert and, with ``--fix``, restarts the Flink job from a clean state (delete
the consumer group, wipe checkpoints, restart the cluster, resubmit the SQL
job) — the exact remediation that recovered the pipeline manually before.

Run as a loop:  uv run python -m scripts.stream_watchdog --interval 60 --fix
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.kv import KVStore, RedisKV
from stream.materializer import feature_key, live_key

logger = get_logger(__name__)


def staleness_seconds(
    kv: KVStore,
    *,
    live_prefix: str,
    feature_prefix: str,
    symbol: str,
) -> float | None:
    """Seconds between the latest raw bar and the latest feature window.

    None when either side is missing entirely (no data yet — not stale).
    """
    live = kv.get_json(live_key(live_prefix, symbol))
    features = kv.list_json(feature_key(feature_prefix, symbol), reverse=True, maxlen=1)
    if not live or not features:
        return None
    raw_ts = live.get("ts")
    feature_end = features[0].get("window_end_ms")
    if not isinstance(raw_ts, (int, float)) or not isinstance(feature_end, (int, float)):
        return None
    return float(raw_ts - feature_end) / 1000.0


def _subprocess(args: list[str]) -> None:
    logger.warning("running: %s", " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True)
    if result.stdout:
        logger.info("  out: %s", result.stdout.strip())
    if result.stderr:
        logger.info("  err: %s", result.stderr.strip())
    if result.returncode != 0:
        logger.error("  exit %d", result.returncode)


def heal_flink() -> None:
    """Restart Flink from a clean state (the manual remediation, automated)."""
    settings = get_settings()
    jm = settings.stream_flink_jobmanager_container
    tm = settings.stream_flink_taskmanager_container
    rp = settings.stream_redpanda_container
    group = settings.stream_flink_consumer_group
    sql_path = settings.stream_flink_sql_path

    # Cancel any running SQL job (best-effort; the job may already be gone).
    _subprocess(
        [
            "docker",
            "exec",
            jm,
            "bash",
            "-c",
            "for j in $(curl -s http://localhost:8081/jobs/overview | python3 -c "
            '\'import json,sys;print(" ".join(j["jid"] for j in json.load(sys.stdin)["jobs"] if j["state"]=="RUNNING"))\' '  # noqa: E501
            "); do curl -s -X PATCH http://localhost:8081/jobs/$j?mode=cancel; done",
        ]
    )
    _subprocess(["docker", "exec", rp, "rpk", "group", "delete", group])
    _subprocess(["docker", "restart", jm, tm])
    time.sleep(20)
    _subprocess(
        [
            "docker",
            "exec",
            jm,
            "bash",
            "-c",
            f"sql-client.sh -f {sql_path} -d",
        ]
    )


def run_once(kv: KVStore, *, threshold: float, fix: bool) -> bool:
    """Check every symbol; alert (and optionally fix) if any is stale."""
    settings = get_settings()
    from config.settings import csv_list

    symbols = csv_list(settings.ingest_default_crypto_symbols)
    stale_symbols: list[tuple[str, float]] = []
    for symbol in symbols:
        stale = staleness_seconds(
            kv,
            live_prefix=settings.stream_redis_live_prefix,
            feature_prefix=settings.stream_redis_feature_prefix,
            symbol=symbol,
        )
        if stale is not None and stale > threshold:
            stale_symbols.append((symbol, stale))
    if not stale_symbols:
        return False
    for symbol, stale in stale_symbols:
        logger.error("STREAM STALE for %s: features %d s behind raw bars", symbol, int(stale))
    if fix:
        logger.warning("healing Flink pipeline…")
        heal_flink()
    return True


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60, help="check loop interval (s)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="staleness alert (s); defaults to stream_watchdog_staleness_threshold_seconds",
    )
    parser.add_argument("--fix", action="store_true", help="auto-heal Flink when stale")
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    args = parser.parse_args()

    settings = get_settings()
    threshold = (
        args.threshold
        if args.threshold is not None
        else settings.stream_watchdog_staleness_threshold_seconds
    )
    kv = RedisKV(settings.stream_redis_url)
    logger.info(
        "watchdog checking %s staleness >%ss (fix=%s)",
        settings.ingest_default_crypto_symbols,
        threshold,
        args.fix,
    )
    if args.once:
        stale = run_once(kv, threshold=threshold, fix=args.fix)
        sys.exit(1 if stale else 0)
    while True:
        run_once(kv, threshold=threshold, fix=args.fix)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
