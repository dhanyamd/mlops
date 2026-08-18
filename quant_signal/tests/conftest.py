"""Shared pytest fixtures.

Wraps every test in Prefect's documented ``prefect_test_harness`` so flows/tasks
run with a proper (in-memory) run context and no ephemeral server subprocess
leaks into test output or CI logs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

# Tests call flows via .fn() (no flow-run context); without this, Prefect's API
# log handler warns on every log line. Documented in Prefect's test docs.
os.environ.setdefault("PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW", "ignore")

# prefect_test_harness() backs the ephemeral server with a temp SQLite DB; with
# analytics left on, Prefect's background telemetry heartbeat writes to that DB
# at teardown and can trip "database is locked" noise in the summary. Off =
# no telemetry, no lock contention, hermetic runs. (Prefect's own documented
# toggle for send_telemetry_heartbeat.)
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

# Settings requires snowflake_account/user with no defaults, and four test
# modules call get_settings() at import time -- so without credentials the
# suite dies at COLLECTION, not in a test. That made the suite pass locally
# (a developer .env is present) and fail in CI (none is), which is exactly
# backwards: tests must not depend on a machine's leftover credentials.
#
# setdefault, so a real .env or a CI secret still wins. Nothing here connects
# to Snowflake; these only satisfy construction.
os.environ.setdefault("SNOWFLAKE_ACCOUNT", "test_account")
os.environ.setdefault("SNOWFLAKE_USER", "test_user")
os.environ.setdefault("SNOWFLAKE_PASSWORD", "test_password")

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def _prefect_test_harness() -> Iterator[None]:
    with prefect_test_harness():
        yield


@pytest.fixture(autouse=True)
def _disable_live_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: never spawn the real Binance poller thread.

    ``api.main.start_stream`` is called by the FastAPI lifespan that
    ``TestClient`` triggers; returning None means no network thread and no
    Snowflake writes. Stream tests inject a fake provider themselves.
    """
    monkeypatch.setattr("api.main.start_stream", lambda: None)
