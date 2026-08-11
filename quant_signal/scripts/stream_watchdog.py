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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.kv import KVStore, RedisKV
from stream.materializer import feature_key, live_key

logger = get_logger(__name__)

# Last-heal timestamp persisted across process restarts (launchd keeps the
# daemon alive, and restarting must not reset the cooldown or it would re-heal
# the just-started jobs immediately).
HEAL_STATE_FILE = Path("/tmp/stream_watchdog_last_heal")


def _docker() -> str:
    """Resolve the docker CLI for launchd-spawned processes.

    launchd runs jobs with a minimal PATH (no /usr/local/bin), so a subprocess
    call to bare ``docker`` raises FileNotFoundError — which is exactly how the
    watchdog silently stopped healing the stalled Flink pipeline for hours while
    every staleness check looked like a heal attempt. Augment PATH with the
    common macOS docker locations before resolving.
    """
    candidates = [os.path.expanduser("~/.local/bin"), "/usr/local/bin", "/opt/homebrew/bin"]
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = (
        os.pathsep.join([p for p in candidates if p not in existing]) + os.pathsep + existing
    )
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise RuntimeError("docker not found on PATH — cannot heal the Flink pipeline")
    return docker_bin


def _last_heal_ts() -> float | None:
    try:
        return float(HEAL_STATE_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _mark_healed() -> None:
    HEAL_STATE_FILE.write_text(f"{time.time()}\n")


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
    """Restart Flink from a clean state (the manual remediation, automated).

    Both window jobs are resubmitted: the 1h live trading clock and the 5m
    benchmark/Alpha-Autopsy book. Each has its own consumer group so deleting
    offsets is per-job.
    """
    settings = get_settings()
    docker_bin = _docker()
    jm = settings.stream_flink_jobmanager_container
    tm = settings.stream_flink_taskmanager_container
    rp = settings.stream_redpanda_container
    jobs = [
        (settings.stream_flink_consumer_group, settings.stream_flink_sql_path),
        (settings.stream_flink_consumer_group_5m, settings.stream_flink_sql_path_5m),
    ]

    # Cancel any running SQL jobs (best-effort; the jobs may already be gone).
    # The jobmanager image ships only curl/sed/grep (no python3/jq), and this
    # Flink reports jids as "jid" in the overview, so parse the JSON with
    # sed/grep rather than a python one-liner (which silently no-oped before).
    _subprocess(
        [
            docker_bin,
            "exec",
            jm,
            "bash",
            "-c",
            "for j in $(curl -s http://localhost:8081/jobs/overview | "
            "sed 's/}/}\\n/g' | grep '\"state\":\"RUNNING\"' | "
            'grep -oE \'"[a-z]*id":"[0-9a-f]*"\' | '
            'sed -E \'s/"[a-z]*id":"([0-9a-f]*)"/\\1/\'); do '
            'curl -s -X PATCH "http://localhost:8081/jobs/$j?mode=cancel"; done',
        ]
    )
    for group, _ in jobs:
        _subprocess([docker_bin, "exec", rp, "rpk", "group", "delete", group])
    _subprocess([docker_bin, "restart", jm, tm])
    time.sleep(20)
    for _, sql_path in jobs:
        _subprocess(
            [
                docker_bin,
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
        cooldown = settings.stream_watchdog_heal_cooldown_seconds
        last_heal = _last_heal_ts()
        if last_heal is not None and time.time() - last_heal < cooldown:
            logger.warning(
                "heal skipped (cooldown): last heal %.0fs ago < %ds — waiting for "
                "the restarted windows to emit before considering another heal",
                time.time() - last_heal,
                cooldown,
            )
        else:
            logger.warning("healing Flink pipeline…")
            heal_flink()
            _mark_healed()
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
