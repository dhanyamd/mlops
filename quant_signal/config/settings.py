"""Application configuration — every value comes from the environment.

No credentials, hosts, or connection strings are hardcoded anywhere in the
codebase. Secrets are masked from logs/repr via ``Field(repr=False)``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def csv_list(value: str) -> list[str]:
    """Split an env-provided CSV string into non-empty, stripped parts.

    Env vars are strings; this is the single convention for list-valued
    settings (``INGEST_DEFAULT_SYMBOLS=AAPL,MSFT``), kept out of every flow.
    """
    return [part.strip() for part in value.split(",") if part.strip()]


class Settings(BaseSettings):
    """Typed, validated settings loaded from env vars (and optional ``.env``)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Snowflake connection ────────────────────────────────────────────────
    snowflake_account: str
    snowflake_user: str
    # Use *either* password or key-pair auth (see model validator below).
    snowflake_password: str | None = Field(default=None, repr=False)
    snowflake_role: str = "ACCOUNTADMIN"
    snowflake_database: str = "QUANT"
    snowflake_schema: str = "BRONZE"
    snowflake_warehouse: str = "QUANT_WH"
    # Rows that fail the ingestion contract are never dropped: they land here.
    snowflake_quarantine_schema: str = "QUARANTINE"
    # Attached to every query for cost/credit attribution per pipeline.
    snowflake_query_tag: str = "quant_signal"

    # Modern Snowflake accounts require MFA for password logins. When enabled,
    # the connector uses authenticator="username_password_mfa" (Duo push or
    # TOTP) and caches the MFA token (valid ~4h) so automation only prompts
    # on first connect. Requires MFA enrollment in Snowsight (Duo/authenticator
    # app, NOT passkey) and ALLOW_CLIENT_MFA_CACHING=TRUE (done in bootstrap).
    snowflake_use_mfa: bool = False
    snowflake_mfa_token_caching: bool = True
    # Optional TOTP passcode (SNOWFLAKE_MFA_PASSCODE). Leave blank to be
    # prompted interactively on the first MFA connection.
    snowflake_mfa_passcode: str | None = Field(default=None, repr=False)

    # ── Key-pair auth (alternative to password) ─────────────────────────────
    snowflake_private_key_file: Path | None = None
    snowflake_private_key_passphrase: str | None = Field(default=None, repr=False)

    # ── Data ingestion ──────────────────────────────────────────────────────
    # SEC EDGAR requires a descriptive User-Agent ("Name you@example.com").
    # Leave blank to use a clearly non-production default.
    edgar_user_agent: str | None = None

    # Default instruments/series for the real ingestion flows. Every flow ALSO
    # accepts explicit parameters; these only supply the defaults so nothing is
    # hardcoded in code. Comma-separated in the environment.
    ingest_default_provider: str = "yahoo"
    ingest_default_days: int = 365
    ingest_default_symbols: str = "AAPL,MSFT,NVDA"
    ingest_default_crypto_symbols: str = "BTCUSDT,ETHUSDT"
    ingest_default_macro_series: str = "VIXCLS,CPIAUCSL,DGS10,DGS2,FEDFUNDS,UNRATE"
    ingest_default_tickers: str = "AAPL,MSFT"
    # Fundamental metrics extracted from SEC EDGAR XBRL company facts.
    ingest_default_metrics: str = "Revenues,NetIncomeLoss,Assets"
    # Disk cache location for the Yahoo provider (None → ~/.cache/quant_signal).
    yahoo_cache_dir: Path | None = None

    # ── Runtime ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Dashboard API ───────────────────────────────────────────────────────
    # Allowed browser origins for the read-only dashboard API (CSV). The
    # Next.js dev server proxies /api → :8000 via rewrites, so this only
    # matters if a browser hits the API directly from another origin.
    api_cors_origins: str = "http://localhost:3000"
    # Recompute the PEAD event study at most this often per parameter set
    # (a full run is a ~10s Snowflake query, far slower than the other reads).
    api_pead_cache_ttl_seconds: int = 60

    # ── Live market stream (near-real-time showcase) ────────────────────────
    # Background poller in the API process: fetches recent Binance minute bars,
    # persists them to BRONZE.CRYPTO_BARS (best-effort), and broadcasts deltas
    # to /ws/market subscribers. Disable for a pure query API.
    stream_enabled: bool = True
    stream_poll_seconds: int = 15
    # Ring-buffer depth kept per symbol for WebSocket snapshots.
    stream_history_minutes: int = 180

    @field_validator("snowflake_account")
    @classmethod
    def _validate_account(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("SNOWFLAKE_ACCOUNT must not be empty")
        if "snowflakecomputing.com" in value.lower() or "." in value:
            raise ValueError(
                "SNOWFLAKE_ACCOUNT must be the bare account identifier "
                "(e.g. 'GULXCKK-PI01025'), without '.snowflakecomputing.com'"
            )
        return value

    @field_validator("snowflake_private_key_file", mode="before")
    @classmethod
    def _empty_key_file_is_none(cls, value: object) -> object:
        # An empty .env value becomes Path("") == Path(".") which is truthy
        # and would silently switch us into key-pair auth. Treat blank as None.
        if value is None:
            return None
        as_str = str(value)
        if not as_str or as_str == ".":
            return None
        return value

    @model_validator(mode="after")
    def _validate_auth(self) -> "Settings":
        if not self.snowflake_password and not self.snowflake_private_key_file:
            raise ValueError(
                "Snowflake auth requires either SNOWFLAKE_PASSWORD or "
                "SNOWFLAKE_PRIVATE_KEY_FILE to be set"
            )
        return self

    @property
    def uses_key_pair_auth(self) -> bool:
        return self.snowflake_private_key_file is not None


@lru_cache
def get_settings() -> Settings:
    """Cached settings factory — the single entry point for config.

    Cached so every module reads one validated instance; the cache can be
    cleared in tests via ``get_settings.cache_clear()``.
    """
    return Settings()
