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
    # Consumer-ready analytics marts (dbt GOLD + Spark feature batch output).
    snowflake_gold_schema: str = "GOLD"
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
    # Alpaca Market Data credentials (free IEX feed — the official US-equity
    # upgrade over Yahoo, ~2.5% of consolidated volume). Unlike the keyless
    # providers this needs a free API key pair; blank → the provider is
    # unavailable and selecting it raises a clear error.
    ingest_provider_alpaca_api_key: str | None = Field(default=None, repr=False)
    ingest_provider_alpaca_secret_key: str | None = Field(default=None, repr=False)

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
    # Ingestion runs in a standalone producer (``stream/producer.py``) that
    # publishes Binance minute bars to the Kafka bus; the API consumes the raw
    # topic and fans deltas out over /ws/market. Disable for a pure query API.
    stream_enabled: bool = True
    stream_poll_seconds: int = 15
    stream_poll_timeout_seconds: float = 45
    # Ring-buffer depth kept per symbol for WebSocket snapshots.
    stream_history_minutes: int = 180

    # ── Streaming stack (M3): Kafka ingestion bus + Redis online store ──────
    # Bootstrap servers for the message bus (comma-separated for a cluster).
    stream_kafka_bootstrap_servers: str = "localhost:9092"
    # Raw Binance minute bars (producer → Flink → materializer). Keyed by symbol.
    stream_kafka_topic_raw: str = "crypto.bars.raw"
    # 5m window features computed by the Flink SQL job (crypto_features.sql).
    stream_kafka_topic_features: str = "crypto.features.5m"
    # Online store where the materializer lands live bars + window features.
    # 6380: the docker-compose Redis is remapped off 6379 so it never
    # collides with a host Redis (the API/producer connect from the host).
    stream_redis_url: str = "redis://localhost:6380"
    stream_redis_live_prefix: str = "live:crypto"
    stream_redis_feature_prefix: str = "feature:crypto:5m"
    # Windows of Flink features kept per symbol (RPUSH + LTRIM bound).
    stream_redis_feature_maxlen: int = 200
    # Staleness gate for the watchdog and the API's pipeline-status endpoint:
    # "healthy" means the latest feature window ended no more than this many
    # seconds before the latest raw bar (event-time delta, so the host clock
    # drifting is irrelevant). Sized to > one 5m window + watermark + slack.
    stream_watchdog_staleness_threshold_seconds: float = 900.0
    # Venue (source exchange) for the live stream: the provider built by name
    # from the registry and stamped on every raw bar's ``provider`` field.
    stream_venue: str = "binance"
    # Flink infrastructure the watchdog heals by name — the daemon never
    # hardcodes container ids, consumer groups, or job paths.
    stream_flink_jobmanager_container: str = "quant_signal-flink-jobmanager-1"
    stream_flink_taskmanager_container: str = "quant_signal-flink-taskmanager-1"
    stream_redpanda_container: str = "quant_signal-redpanda-1"
    stream_flink_consumer_group: str = "flink-crypto-features"
    stream_flink_sql_path: str = "/opt/flink/jobs/crypto_features.sql"

    # ── Prediction layer (M3.5): River + conformal intervals + Monte Carlo ───
    # Online store prefixes for the predictor and simulator outputs.
    stream_redis_prediction_prefix: str = "prediction:crypto:5m"
    stream_redis_simulation_prefix: str = "simulation:crypto:5m"
    # Live strategy P&L curve (compounded equity vs buy-and-hold) derived from
    # the predictor's realized directions, for the Signal Terminal P&L strip.
    stream_redis_strategy_prefix: str = "strategy:crypto:5m"
    # Windows kept in the per-symbol strategy equity curve.
    stream_strategy_maxlen: int = 500
    # Adaptive Conformal Inference (Gibbs & Candès 2021): target miscoverage
    # and step size; the nominal level adapts so long-run coverage tracks
    # (1 - alpha) even under drift.
    stream_prediction_alpha: float = 0.1
    stream_prediction_gamma: float = 0.005
    # Residuals kept for the conformal interval quantile.
    stream_prediction_residual_window: int = 200
    # Monte Carlo engine: paths (all simulated in one vectorized numpy call),
    # forward horizon (5m steps), trailing 5m windows used to estimate the
    # per-step log-return volatility.
    stream_simulation_paths: int = 10_000
    stream_simulation_horizon_steps: int = 12
    stream_simulation_vol_windows: int = 40
    # Raw paths shipped to the UI for the all-paths fan (thin canvas lines);
    # percentile statistics always use *all* paths — this is purely how many
    # strokes the browser renders per window.
    stream_simulation_sample_paths: int = 1000
    # Strategy validation Monte Carlo (QuantPad-style pass probability):
    # bootstrap the realized signal returns into simulated futures and score
    # them against prop-firm-style rules (max drawdown breach + profit target,
    # e.g. Topstep 50K ≈ 6% target / FTMO ≈ 8-10% target, trailing DD).
    stream_validation_sims: int = 10_000
    stream_validation_max_drawdown: float = 0.08
    # Profit target: a future only "passes" if it reaches this return from
    # starting equity without breaching max drawdown first.
    stream_validation_target: float = 0.06
    stream_validation_seed: int = 42

    # ── MLflow experiment tracking (offline validation runs) ────────────────
    # The served models are online learners (River) + MC, so we use MLflow
    # *tracking* (params/metrics/artifacts per validation run), not the model
    # registry. Optional `mlflow` extra; without it, tracking is a no-op.
    # Tracking is explicit (?track=true) so the 15s UI poll never spams runs.
    mlflow_tracking_enabled: bool = False
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "quant_signal"

    # ── Prediction promotion gate (progressive validation) ──────────────────
    # The live predictor may learn but must not trade until it clears the
    # gate: enough scored windows, positive skill vs both naive baselines, IC
    # and direction accuracy above floor, conformal coverage near nominal,
    # strategy return clearing buy-and-hold after taker costs, and a Deflated
    # Sharpe (multiple-testing-corrected) above the significance floor.
    stream_gate_min_windows: int = 100
    stream_gate_min_skill: float = 0.0
    stream_gate_min_ic: float = 0.0
    stream_gate_min_direction_accuracy: float = 0.5
    stream_gate_coverage_tol: float = 0.05
    stream_gate_min_dsr: float = 0.95
    # Trials the model has gone through — the multiple-testing charge DSR is
    # deflated by. Default 1 = "claim this is the first attempt"; any search
    # that tried N configurations must disclose N here or the deflation is
    # meaningless (Harvey / Bailey & López de Prado).
    stream_gate_n_trials: int = 1
    # Taker cost charged per position flip against the strategy (5bps default).
    stream_gate_taker_cost: float = 0.0005

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
