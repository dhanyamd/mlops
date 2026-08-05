"""SnowflakeClient connection-parameter tests — no live Snowflake needed.

We test the exact dict passed to ``snowflake.connector.connect`` so auth
branching, secrets hygiene, and query tagging are locked down.
"""

from __future__ import annotations

import pandas as pd
import pytest
import snowflake.connector as sf
from pydantic import ValidationError

import db.snowflake as sfmod
from config.settings import Settings
from db.snowflake import SnowflakeClient, _validate_identifier


def _settings(**overrides) -> Settings:
    base = {
        "snowflake_account": "GULXCKK-PI01025",
        "snowflake_user": "devuser",
        "snowflake_password": "pw",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_password_auth_params() -> None:
    params = SnowflakeClient(_settings())._connect_params()
    assert params["account"] == "GULXCKK-PI01025"
    assert params["password"] == "pw"
    assert "authenticator" not in params
    assert "private_key_file" not in params
    assert params["query_tag"] == "quant_signal"


def test_key_pair_auth_params() -> None:
    params = SnowflakeClient(
        _settings(
            snowflake_password="",
            snowflake_private_key_file="/tmp/rsa_key.p8",
            snowflake_private_key_passphrase="pass",
        )
    )._connect_params()
    assert params["authenticator"] == "SNOWFLAKE_JWT"
    assert params["private_key_file"] == "/tmp/rsa_key.p8"
    assert params["private_key_file_pwd"] == "pass"
    assert "password" not in params


def test_missing_auth_fails_at_settings() -> None:
    # Settings validation rejects no-auth configs before a client can exist.
    with pytest.raises(ValidationError):
        _settings(snowflake_password="", snowflake_private_key_file=None)


def test_mfa_auth_params() -> None:
    params = SnowflakeClient(_settings(snowflake_use_mfa=True))._connect_params()
    assert params["authenticator"] == "username_password_mfa"
    assert params["password"] == "pw"
    assert params["client_request_mfa_token"] is True


def test_mfa_passcode_from_env() -> None:
    params = SnowflakeClient(
        _settings(snowflake_use_mfa=True, snowflake_mfa_passcode="123456")
    )._connect_params()
    assert params["passcode"] == "123456"


def test_mfa_auth_without_caching() -> None:
    params = SnowflakeClient(
        _settings(snowflake_use_mfa=True, snowflake_mfa_token_caching=False)
    )._connect_params()
    assert params["authenticator"] == "username_password_mfa"
    assert "client_request_mfa_token" not in params


def test_bootstrap_connects_without_warehouse() -> None:
    client = SnowflakeClient(_settings(), use_warehouse=False)
    params = client._connect_params()
    assert "warehouse" not in params


class _FakeCursor:
    """Mimics a Snowflake cursor whose SHOW queries fail Arrow fetch."""

    def __init__(self, rows: list[tuple], columns: list[str]) -> None:
        self._rows = rows
        self.description = [(c, None, None, None, None, None, None) for c in columns]

    def execute(self, sql: str, params: tuple | None = None) -> "_FakeCursor":
        return self

    def fetch_pandas_all(self) -> pd.DataFrame:
        raise sf.errors.NotSupportedError

    def fetchall(self) -> list[tuple]:
        return self._rows


def test_query_df_falls_back_for_show_statements(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cursor = _FakeCursor(
        rows=[("QUANT", "BRONZE"), ("QUANT", "GOLD")], columns=["created_on", "name"]
    )

    class _FakeConn:
        def cursor(self) -> _FakeCursor:
            return fake_cursor

        def close(self) -> None:
            return None

    monkeypatch.setattr(sf, "connect", lambda **kwargs: _FakeConn())

    df = SnowflakeClient(_settings()).query_df("SHOW SCHEMAS")
    assert list(df.columns) == ["created_on", "name"]
    assert df.shape == (2, 2)


# ── CRUD safety guards (no live Snowflake) ──────────────────────────────────


class _RecordingCursor:
    """Captures every statement + params, mimics rowcount/fetchone."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params_list: list[object] = []
        self.rowcount = 1
        self._fetchone = (1,)

    def execute(self, sql: str, params: object = None) -> "_RecordingCursor":
        self.statements.append(sql)
        self.params_list.append(params)
        return self

    def fetchone(self) -> tuple:
        return self._fetchone

    def close(self) -> None:
        return None


class _RecordingConnection:
    def __init__(self) -> None:
        self.cur = _RecordingCursor()

    def cursor(self) -> _RecordingCursor:
        return self.cur

    def close(self) -> None:
        return None


def _record(monkeypatch: pytest.MonkeyPatch) -> _RecordingConnection:
    conn = _RecordingConnection()
    monkeypatch.setattr(sf, "connect", lambda **kwargs: conn)
    return conn


@pytest.mark.parametrize(
    "bad",
    [
        "symbol; DROP TABLE x",
        "two words",
        "1leading_digit",
        'has"quote',
        "sym-bol",
    ],
)
def test_validate_identifier_rejects_unsafe_names(bad: str) -> None:
    with pytest.raises(ValueError):
        _validate_identifier(bad)


def test_validate_identifier_accepts_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _validate_identifier(" equity_bars ") == "equity_bars"
    assert _validate_identifier("_meta$") == "_meta$"


def test_delete_requires_where() -> None:
    with pytest.raises(ValueError):
        SnowflakeClient(_settings()).delete("equity_bars")


def test_delete_full_table_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _record(monkeypatch)
    SnowflakeClient(_settings()).delete("equity_bars", allow_full_table=True)
    sql = conn.cur.statements[0]
    assert sql.startswith("DELETE FROM")
    assert " WHERE " not in sql
    assert '"EQUITY_BARS"' in sql


def test_update_requires_where() -> None:
    with pytest.raises(ValueError):
        SnowflakeClient(_settings()).update("equity_bars", {"close": 1.0}, where="")


def test_update_binds_params_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _record(monkeypatch)
    affected = SnowflakeClient(_settings()).update(
        "equity_bars",
        {"close": 100.5},
        "symbol = %(sym)s",
        where_params={"sym": "AAPL"},
    )
    sql = conn.cur.statements[0]
    assert 'UPDATE "QUANT"."BRONZE"."EQUITY_BARS" SET "CLOSE" = %(__v0)s' in sql
    assert "symbol = %(sym)s" in sql
    assert conn.cur.params_list[0] == {"__v0": 100.5, "sym": "AAPL"}
    assert affected == 1


def test_upsert_merge_keys_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, str] = {}

    def fake_write_pandas(conn, df, **kwargs):  # type: ignore[no-untyped-def]
        calls["table"] = kwargs["table_name"]
        return (True, 1, len(df), [])

    monkeypatch.setattr(sfmod, "write_pandas", fake_write_pandas)
    conn = _record(monkeypatch)

    df = pd.DataFrame({"symbol": ["AAPL"], "ts": [pd.Timestamp("2026-01-01")], "close": [150.0]})
    SnowflakeClient(_settings()).upsert_df(df, "equity_bars", merge_keys=["symbol", "ts"])

    assert calls["table"].startswith("TMP_")
    merged = " ".join(conn.cur.statements)
    assert '"EQUITY_BARS"' in merged
    assert "ON (" in merged and 't."SYMBOL"' in merged


def test_upsert_duplicate_columns_after_case_fold() -> None:
    df = pd.DataFrame({"symbol": ["a"], "SYMBOL": ["b"]})
    with pytest.raises(ValueError):
        SnowflakeClient(_settings()).upsert_df(df, "t", merge_keys=["symbol"])


def test_upsert_empty_df_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_write_pandas(conn, df, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(True)
        return (True, 1, 0, [])

    monkeypatch.setattr(sfmod, "write_pandas", fake_write_pandas)
    assert (
        SnowflakeClient(_settings()).upsert_df(pd.DataFrame({"a": []}), "t", merge_keys=["a"]) == 0
    )
    assert calls == []


def test_insert_uppercases_names_for_dbt_interop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_write_pandas(conn, df, **kwargs):  # type: ignore[no-untyped-def]
        calls["table"] = kwargs["table_name"]
        calls["cols"] = list(df.columns)
        return (True, 1, len(df), [])

    monkeypatch.setattr(sfmod, "write_pandas", fake_write_pandas)
    _record(monkeypatch)
    df = pd.DataFrame({"symbol": ["AAPL"], "close": [150.0]})
    SnowflakeClient(_settings()).insert_df(df, "equity_bars")
    assert calls["table"] == "EQUITY_BARS"
    assert calls["cols"] == ["SYMBOL", "CLOSE"]


def test_insert_empty_df_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_write_pandas(conn, df, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(True)
        return (True, 1, 0, [])

    monkeypatch.setattr(sfmod, "write_pandas", fake_write_pandas)
    assert SnowflakeClient(_settings()).insert_df(pd.DataFrame({"a": []}), "t") == 0
    assert calls == []


def test_table_exists_scopes_information_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _record(monkeypatch)
    assert SnowflakeClient(_settings()).table_exists("equity_bars") is True
    sql = conn.cur.statements[0]
    assert '"QUANT".INFORMATION_SCHEMA.TABLES' in sql
    assert "TABLE_CATALOG" not in sql


def test_table_exists_validates_database_identifier() -> None:
    with pytest.raises(ValueError):
        SnowflakeClient(_settings()).table_exists("t", database="bad.db")


# ── Bootstrap SQL splitter (regression: semicolons inside comments) ─────────


def test_split_statements_ignores_semicolons_inside_comments() -> None:
    from db.bootstrap import _split_statements

    sql = (
        "-- ON ALL TABLES only covers tables that already exist; future tables need this.\n"
        "CREATE DATABASE IF NOT EXISTS QUANT;\n"
        "-- another ; tricky comment\n"
        "CREATE WAREHOUSE IF NOT EXISTS QUANT_WH\n"
        "    WITH WAREHOUSE_SIZE = 'XSMALL';\n"
    )
    statements = _split_statements(sql)
    assert statements == [
        "CREATE DATABASE IF NOT EXISTS QUANT",
        "CREATE WAREHOUSE IF NOT EXISTS QUANT_WH\n    WITH WAREHOUSE_SIZE = 'XSMALL'",
    ]
