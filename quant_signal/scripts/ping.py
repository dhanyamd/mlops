"""Connection smoke test: ``python scripts/ping.py`` (or ``make ping``)."""

from __future__ import annotations

from config.logging import configure_logging, get_logger
from db.snowflake import SnowflakeClient

configure_logging()
log = get_logger("ping")


def main() -> None:
    client = SnowflakeClient()
    if client.ping():
        log.info("connection_ok")
    else:
        raise SystemExit("Snowflake connection failed")


if __name__ == "__main__":
    main()
