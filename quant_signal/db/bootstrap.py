"""Idempotent bootstrap of the QUANT warehouse objects.

Creates database, warehouse, bronze/silver/gold schemas and least-privilege
roles, then grants the roles to the current user. Safe to re-run.
"""

from __future__ import annotations

from pathlib import Path

from config.logging import get_logger
from config.settings import get_settings
from db.snowflake import SnowflakeClient

log = get_logger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"
ROLES = ("QUANT_INGEST", "QUANT_TRANSFORMER", "QUANT_READER")


def _split_statements(sql: str) -> list[str]:
    """Split a SQL script into statements, respecting ``--`` comment lines.

    A naive ``sql.split(";")`` breaks when a comment contains a semicolon.
    Bootstrap SQL is written one statement per line ending in ``;``, so we
    split on lines instead: skip comment/blank lines, and cut a statement when
    its last line ends with ``;``.
    """
    statements: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current).rstrip().rstrip(";"))
            current = []
    if current:  # statement without a trailing semicolon
        statements.append("\n".join(current))
    return [s.strip() for s in statements if s.strip()]


def bootstrap() -> None:
    settings = get_settings()
    # Bootstrap connects WITHOUT a warehouse: it may need to create QUANT_WH
    # itself, and database/schema/role DDL needs no compute.
    client = SnowflakeClient(settings, use_warehouse=False)
    sql = (SQL_DIR / "bootstrap.sql").read_text(encoding="utf-8")

    with client.connection() as conn:
        cur = conn.cursor()
        for statement in _split_statements(sql):
            cur.execute(statement)
        cur.execute("SELECT CURRENT_USER()")
        user = cur.fetchone()[0]
        for role in ROLES:
            cur.execute(f'GRANT ROLE {role} TO USER "{user}"')
        log.info("bootstrap_complete", database=settings.snowflake_database, user=user)


if __name__ == "__main__":
    bootstrap()
