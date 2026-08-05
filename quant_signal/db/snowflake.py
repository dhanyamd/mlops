"""Snowflake client — production-grade wrapper around the Python connector.

Design principles (prod engineer):
- **No hardcoded values.** Everything from ``Settings`` (env-driven).
- **Secrets never logged.** ``Field(repr=False)`` + we never log params.
- **SQL injection safe.** Every identifier passed through ``_validate_identifier``
  and is quoted; values go through bound parameters, never f-strings.
- **Fails loudly, not silently.** DELETE/UPDATE require an explicit WHERE;
  empty inputs short-circuit; bad identifiers raise before touching Snowflake.
- **Observable.** Every call logs JSON with ``elapsed_ms`` and ``query_tag``.
- **Idempotent helpers.** ``upsert_df`` (MERGE) for incremental pipelines,
  ``insert_df`` append for raw landing.

Supports password, password+MFA (TOTP with token caching), and key-pair (JWT)
authentication.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from config.logging import get_logger
from config.settings import Settings, get_settings

log = get_logger(__name__)

# Unquoted Snowflake identifiers fold to UPPERCASE, so we normalize to upper.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _validate_identifier(name: str) -> str:
    """Reject anything that could break out of an identifier or is unsafe."""
    value = str(name).strip()
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid Snowflake identifier: {name!r}")
    return value


def _q(name: str) -> str:
    """Quote a validated, uppercased identifier."""
    return f'"{_validate_identifier(name).upper()}"'


class SnowflakeClient:
    """Full CRUD + table-management wrapper. Always closes sessions."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_warehouse: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._use_warehouse = use_warehouse

    # ── Connection ───────────────────────────────────────────────────────────

    def _connect_params(self) -> dict[str, Any]:
        s = self._settings
        params: dict[str, Any] = {
            "account": s.snowflake_account,
            "user": s.snowflake_user,
            "database": s.snowflake_database,
            "schema": s.snowflake_schema,
            "role": s.snowflake_role,
            "query_tag": s.snowflake_query_tag,
        }
        # bootstrap runs DDL before the warehouse exists, so the warehouse is
        # optional at connect time (metadata ops need no compute).
        if self._use_warehouse:
            params["warehouse"] = s.snowflake_warehouse
        if s.uses_key_pair_auth:
            params.update(
                authenticator="SNOWFLAKE_JWT",
                private_key_file=str(s.snowflake_private_key_file),
                private_key_file_pwd=s.snowflake_private_key_passphrase,
            )
        elif s.snowflake_use_mfa:
            params.update(
                authenticator="username_password_mfa",
                password=s.snowflake_password,
            )
            if s.snowflake_mfa_token_caching:
                params["client_request_mfa_token"] = True
            # TOTP passcode: prefer env var; otherwise prompt once (only when
            # stdin is a terminal). CI/Spark should use key-pair auth instead.
            passcode = s.snowflake_mfa_passcode
            if passcode is None and sys.stdin.isatty():
                passcode = getpass.getpass("MFA TOTP passcode: ")
            if passcode:
                params["passcode"] = passcode
        elif s.snowflake_password:
            params["password"] = s.snowflake_password
        else:
            # Settings validation should already prevent this; fail loudly anyway.
            raise ValueError("Snowflake auth requires a password or a private key file")
        return params

    @contextmanager
    def connection(self) -> Iterator[snowflake.connector.SnowflakeConnection]:
        """Yield a connection, guaranteed to be closed (also on error)."""
        conn = snowflake.connector.connect(**self._connect_params())
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[snowflake.connector.SnowflakeConnection]:
        """Atomic multi-statement block: commit on success, rollback on error."""
        with self.connection() as conn:
            conn.autocommit(False)
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    # ── Generic execution ────────────────────────────────────────────────────

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> None:
        """Run DDL/DML. Use ``query_df`` for reads."""
        start = time.perf_counter()
        with self.connection() as conn:
            conn.cursor().execute(sql, params)
        log.info(
            "snowflake_execute",
            statement=sql[:300],
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )

    def ping(self) -> bool:
        """True if the warehouse answers ``SELECT 1``."""
        start = time.perf_counter()
        with self.connection() as conn:
            ok = conn.cursor().execute("SELECT 1").fetchone()[0] == 1
        log.info(
            "snowflake_ping",
            ok=ok,
            database=self._settings.snowflake_database,
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )
        return ok

    # ── Reads ────────────────────────────────────────────────────────────────

    def query_df(self, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
        """Run a read query and return a pandas DataFrame.

        ``SHOW``/``DESC`` statements don't return Arrow result sets, so falls
        back to ``fetchall`` for those (still returns a DataFrame).
        """
        start = time.perf_counter()
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            try:
                df = cur.fetch_pandas_all()
            except snowflake.connector.errors.NotSupportedError:
                columns = [col[0] for col in cur.description] if cur.description else []
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=columns)
        log.info(
            "snowflake_query",
            rows=len(df),
            columns=list(df.columns),
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )
        return df

    # ── Writes ───────────────────────────────────────────────────────────────

    def insert_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        *,
        database: str | None = None,
        schema: str | None = None,
        overwrite: bool = False,
        bulk_upload_chunks: bool = False,
    ) -> int:
        """Append a DataFrame into ``{database}.{schema}.{table_name}``.

        Uses ``write_pandas`` (staged upload → COPY INTO). Empty input is a
        no-op. Returns rows written; raises if Snowflake reports failure.
        """
        if df.empty:
            return 0
        # Same quoting discipline as upsert_df: write_pandas quotes identifiers,
        # so normalize names to UPPERCASE for dbt/query interop.
        df = df.copy()
        df.columns = [str(c).upper() for c in df.columns]
        if len(df.columns) != len(set(df.columns)):
            raise ValueError("duplicate columns after case-folding; rename before insert")
        for col in df.columns:
            _validate_identifier(col)
        table = _validate_identifier(table_name).upper()
        db = _validate_identifier(database or self._settings.snowflake_database).upper()
        schema = _validate_identifier(schema or self._settings.snowflake_schema).upper()
        start = time.perf_counter()
        with self.connection() as conn:
            success, _nchunks, nrows, _output = write_pandas(
                conn,
                df,
                table_name=table,
                database=db,
                schema=schema,
                auto_create_table=True,
                overwrite=overwrite,
                bulk_upload_chunks=bulk_upload_chunks,
                # Without this the connector binds pandas datetime64[ns] as
                # epoch-nanoseconds and auto-creates NUMBER columns for them
                # (silently corrupting TIMESTAMP_NTZ types). With it, datetimes
                # bind correctly AND auto-created columns become TIMESTAMP_NTZ.
                use_logical_type=True,
            )
        if not success:
            raise RuntimeError(f"write_pandas failed for {db}.{schema}.{table}")
        log.info(
            "snowflake_insert",
            table=f"{db}.{schema}.{table}",
            rows=int(nrows),
            overwrite=overwrite,
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )
        return int(nrows)

    def upsert_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        merge_keys: list[str],
        *,
        database: str | None = None,
        schema: str | None = None,
        temp_table: str | None = None,
    ) -> int:
        """Idempotent incremental load: MERGE into an existing table.

        - Table is created on first run (schema inferred from the DataFrame).
        - ``merge_keys`` decide MATCHED (UPDATE) vs NOT MATCHED (INSERT).
        - Loads into a per-process temp table, merges, drops temp — no data
          touches the target until the merge succeeds (atomic).
        Returns the number of rows merged.
        """
        df = df.copy()
        df.columns = [str(c).upper() for c in df.columns]
        if df.empty:
            return 0
        if len(df.columns) != len(set(df.columns)):
            raise ValueError("duplicate columns after case-folding; rename before upsert")
        for col in df.columns:
            _validate_identifier(col)
        # merge_keys may be lowercase ("symbol"); normalize to the uppercased columns.
        keys = [_validate_identifier(k).upper() for k in merge_keys]
        missing = [k for k in keys if k not in set(df.columns)]
        if missing:
            raise ValueError(f"merge_keys not present in DataFrame: {missing}")

        # write_pandas quotes identifiers, so names MUST be UPPERCASE or the temp
        # table ("TMP_EQUITY_BARS") won't match _q() references and dbt's unquoted
        # (uppercase-folding) refs won't see the target either.
        table = _validate_identifier(table_name).upper()
        temp = _validate_identifier(temp_table or f"TMP_{table}_{os.getpid()}").upper()
        db = _validate_identifier(database or self._settings.snowflake_database).upper()
        schema = _validate_identifier(schema or self._settings.snowflake_schema).upper()
        target = _q(db) + "." + _q(schema) + "." + _q(table)
        tmp = _q(db) + "." + _q(schema) + "." + _q(temp)

        cols = df.columns.tolist()
        set_clause = ", ".join(f"t.{_q(c)} = s.{_q(c)}" for c in cols)
        insert_cols = ", ".join(_q(c) for c in cols)
        insert_vals = ", ".join(f"s.{_q(c)}" for c in cols)
        on_clause = " AND ".join(f"t.{_q(k)} = s.{_q(k)}" for k in keys)

        start = time.perf_counter()
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {tmp}")
            write_pandas(
                conn,
                df,
                table_name=temp,
                database=db,
                schema=schema,
                auto_create_table=True,
                # Same timestamp-binding fix as insert_df (see above).
                use_logical_type=True,
            )
            # First run: create the target with the same schema (empty).
            cur.execute(f"CREATE TABLE IF NOT EXISTS {target} LIKE {tmp}")
            cur.execute(
                f"MERGE INTO {target} t "
                f"USING {tmp} s ON ({on_clause}) "
                f"WHEN MATCHED THEN UPDATE SET {set_clause} "
                f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
            )
            merged = cur.rowcount if cur.rowcount is not None else len(df)
            cur.execute(f"DROP TABLE IF EXISTS {tmp}")
        log.info(
            "snowflake_upsert",
            table=f"{db}.{schema}.{table}",
            merge_keys=keys,
            rows=merged,
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )
        return int(merged)

    # ── Mutations (safety-guarded) ───────────────────────────────────────────

    def update(
        self,
        table_name: str,
        set_values: dict[str, Any],
        where: str,
        *,
        where_params: dict[str, Any] | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> int:
        """``UPDATE ... SET ... WHERE ...``.

        ``where`` is REQUIRED (never silently update a full table). Values are
        bound via generated ``__v{i}`` placeholders; the WHERE clause may use
        your own named params via ``where_params`` (avoid ``__v`` prefix).
        """
        if not where or not where.strip():
            raise ValueError("update() requires a non-empty WHERE clause")
        qualified = self._qualify(table_name, database, schema)
        params: dict[str, Any] = {}
        set_clause = []
        for i, (col, value) in enumerate(set_values.items()):
            col = _validate_identifier(col)
            key = f"__v{i}"
            params[key] = value
            set_clause.append(f"{_q(col)} = %({key})s")
        if where_params:
            params.update(where_params)
        start = time.perf_counter()
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE {qualified} SET {', '.join(set_clause)} WHERE {where}",
                params,
            )
            affected = cur.rowcount or 0
        log.info(
            "snowflake_update",
            table=qualified,
            rows=affected,
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )
        return int(affected)

    def delete(
        self,
        table_name: str,
        where: str | None = None,
        *,
        where_params: dict[str, Any] | None = None,
        allow_full_table: bool = False,
        database: str | None = None,
        schema: str | None = None,
    ) -> int:
        """``DELETE FROM ...``. WHERE required unless ``allow_full_table``."""
        if not where and not allow_full_table:
            raise ValueError(
                "delete() requires a WHERE clause; pass allow_full_table=True "
                "to intentionally delete all rows (prefer truncate())"
            )
        qualified = self._qualify(table_name, database, schema)
        sql = f"DELETE FROM {qualified}"
        if where:
            sql += f" WHERE {where}"
        start = time.perf_counter()
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, where_params)
            affected = cur.rowcount or 0
        log.info(
            "snowflake_delete",
            table=qualified,
            rows=affected,
            full_table=not where,
            elapsed_ms=round((time.perf_counter() - start) * 1000),
        )
        return int(affected)

    def truncate(
        self, table_name: str, *, database: str | None = None, schema: str | None = None
    ) -> None:
        """Deliberate full wipe (faster than DELETE and never returns rows)."""
        qualified = self._qualify(table_name, database, schema)
        self.execute(f"TRUNCATE TABLE {qualified}")

    def drop_table(
        self, table_name: str, *, database: str | None = None, schema: str | None = None
    ) -> None:
        """``DROP TABLE IF EXISTS``."""
        qualified = self._qualify(table_name, database, schema)
        self.execute(f"DROP TABLE IF EXISTS {qualified}")

    # ── Table management ─────────────────────────────────────────────────────

    def table_exists(
        self, table_name: str, *, database: str | None = None, schema: str | None = None
    ) -> bool:
        """True if the table exists (via INFORMATION_SCHEMA, no warehouse needed)."""
        table = _validate_identifier(table_name).upper()
        db = _validate_identifier(database or self._settings.snowflake_database).upper()
        schema = _validate_identifier(schema or self._settings.snowflake_schema).upper()
        # INFORMATION_SCHEMA is scoped to the CURRENT database, so qualify the FROM
        # with the target database explicitly instead of filtering TABLE_CATALOG.
        sql = (
            f"SELECT COUNT(*) FROM {_q(db)}.INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'"
        )
        with self.connection() as conn:
            count = conn.cursor().execute(sql).fetchone()[0]
        return int(count) == 1

    def list_tables(self, *, database: str | None = None, schema: str | None = None) -> list[str]:
        """Names of tables in a schema (SHOW TABLES)."""
        db = _validate_identifier(database or self._settings.snowflake_database).upper()
        schema = _validate_identifier(schema or self._settings.snowflake_schema).upper()
        df = self.query_df(f"SHOW TABLES IN SCHEMA {_q(db)}.{_q(schema)}")
        return sorted(df["name"].astype(str).tolist())

    def describe_table(
        self, table_name: str, *, database: str | None = None, schema: str | None = None
    ) -> pd.DataFrame:
        """Column metadata (name, type, nullable) from INFORMATION_SCHEMA."""
        table = _validate_identifier(table_name).upper()
        db = _validate_identifier(database or self._settings.snowflake_database).upper()
        schema = _validate_identifier(schema or self._settings.snowflake_schema).upper()
        return self.query_df(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION "
            f"FROM {_q(db)}.INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}' "
            "ORDER BY ORDINAL_POSITION"
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _qualify(self, table_name: str, database: str | None, schema: str | None) -> str:
        table = _validate_identifier(table_name)
        db = database or self._settings.snowflake_database
        schema = schema or self._settings.snowflake_schema
        return f"{_q(db)}.{_q(schema)}.{_q(table)}"
