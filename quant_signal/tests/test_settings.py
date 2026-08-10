"""Config behavior tests — run without needing a Snowflake account.

Secrets discipline and fail-fast validation are contract, not convenience:
these tests lock that in.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_settings_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
    monkeypatch.setenv("SNOWFLAKE_USER", "devuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")
    monkeypatch.delenv("SNOWFLAKE_PRIVATE_KEY_FILE", raising=False)

    s = Settings(_env_file=None)

    assert s.snowflake_account == "acct123"
    assert s.snowflake_user == "devuser"
    assert s.snowflake_password == "hunter2"
    assert s.uses_key_pair_auth is False


def test_defaults_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
    monkeypatch.setenv("SNOWFLAKE_USER", "devuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
    s = Settings(_env_file=None)

    assert s.snowflake_database == "QUANT"
    assert s.snowflake_warehouse == "QUANT_WH"
    assert s.snowflake_role == "ACCOUNTADMIN"


def test_stream_venue_and_watchdog_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
    monkeypatch.setenv("SNOWFLAKE_USER", "devuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
    s = Settings(_env_file=None)

    assert s.stream_venue == "binance"
    assert s.stream_watchdog_staleness_threshold_seconds == 7200.0
    assert s.stream_flink_consumer_group == "flink-crypto-features-1h"
    assert s.stream_flink_sql_path == "/opt/flink/jobs/crypto_features_1h.sql"
    assert s.stream_flink_consumer_group_5m == "flink-crypto-features"
    assert s.stream_flink_sql_path_5m == "/opt/flink/jobs/crypto_features.sql"
    assert s.stream_execution_cost_filter_lambda == 2.0
    assert s.stream_execution_hold_until_decay is True


def test_key_pair_auth_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
    monkeypatch.setenv("SNOWFLAKE_USER", "devuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "")
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_FILE", "/tmp/rsa_key.p8")

    s = Settings(_env_file=None)

    assert s.uses_key_pair_auth is True


def test_empty_private_key_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty env values must NOT flip us into key-pair auth (Path("") == Path(".")).
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct123")
    monkeypatch.setenv("SNOWFLAKE_USER", "devuser")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "pw")
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_FILE", "")
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")

    s = Settings(_env_file=None)

    assert s.snowflake_private_key_file is None
    assert s.uses_key_pair_auth is False
    assert s.snowflake_password == "pw"


def test_password_never_appears_in_repr() -> None:
    s = Settings(
        snowflake_account="a",
        snowflake_user="u",
        snowflake_password="super-secret-pw",
        _env_file=None,
    )
    assert "super-secret-pw" not in repr(s)


def test_missing_auth_fails_fast() -> None:
    with pytest.raises(ValidationError):
        Settings(
            snowflake_account="a",
            snowflake_user="u",
            snowflake_password="",
            snowflake_private_key_file=None,
            _env_file=None,
        )
