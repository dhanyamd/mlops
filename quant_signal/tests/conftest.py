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

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def _prefect_test_harness() -> Iterator[None]:
    with prefect_test_harness():
        yield
