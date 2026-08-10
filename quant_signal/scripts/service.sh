#!/usr/bin/env bash
# Launchd wrapper for the quant_signal stream services.
#
# launchd launches this script with a bare environment, so we source the
# project .env first (API keys, venue selection) and then replace the shell
# with the venv python running the requested module. `exec` is required:
# launchd tracks the job's lifetime by the top process it spawned, and it
# must not daemonize or fork-and-exit (launchd would think the job died and
# respawn it in a loop).
#
# Usage:
#   scripts/service.sh stream.predictor
#   scripts/service.sh scripts.stream_watchdog --interval 60 --fix
#   scripts/service.sh uvicorn api.main:app --host 0.0.0.0 --port 8000
set -a
. "$(dirname "$0")/../.env"
set +a

DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/.venv/bin/python" -u -m "$@"
