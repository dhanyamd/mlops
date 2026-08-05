# quant_signal

Production-grade quant signal platform: **Snowflake-backed pipelines** with
**Spark**, data quality, and full observability. Built from research on how
quant research houses (Two Sigma / Man AHL patterns) actually run data
infrastructure.

> Status: **M0 — foundations**. Config, Snowflake client, structured logging,
> idempotent warehouse bootstrap, and the dbt project skeleton with enforced
> contracts. No code is hardcoded: every credential and connection value comes
> from the environment.

## Non-negotiables

- **Nothing hardcoded.** All config via `pydantic-settings` from env vars
  (`config/settings.py`). Secrets use `Field(repr=False)` (never logged),
  dbt's `DBT_ENV_SECRET_*` prefix (auto-scrubbed from logs), and every query
  carries a `query_tag` for credit/cost attribution.
- **Fail fast.** A missing auth method or a malformed account identifier
  raises `ValidationError` at startup — misconfig never silently half-runs.
- **Reproducible infra.** `make bootstrap` idempotently creates the database,
  warehouse, schemas, and least-privilege roles (`QUANT_INGEST` /
  `QUANT_TRANSFORMER` / `QUANT_READER`).
- **Quality as contract.** Every silver/gold dbt model must declare a full
  column contract (`contract: enforced: true`) or the build fails.
- **Structured logs.** JSON lines via `structlog`; secrets never logged.

## Stack (this folder)

| Concern | Tool |
|---|---|
| Config | `pydantic-settings` |
| Warehouse | Snowflake (`snowflake-connector-python`) |
| Transform + quality | `dbt-core` + `dbt-snowflake` (contracts, tests) |
| Statistical tests | `dbt_expectations` |
| Data observability | `elementary` (anomaly/schema-drift detection) |
| Orchestration (later) | Prefect |
| Distributed compute (later) | Spark + `spark-snowflake` connector |

## Layout

```
quant_signal/
├── config/
│   ├── settings.py         # pydantic-settings: env-driven, secrets masked
│   └── logging.py          # structlog JSON logging (level from env)
├── db/
│   ├── snowflake.py        # SnowflakeClient: query/insert/ping, query_tag
│   ├── bootstrap.py        # idempotent infra bootstrap (make bootstrap)
│   └── sql/bootstrap.sql   # DB, warehouse, schemas, least-privilege roles
├── dbt/
│   ├── profiles.yml        # creds ONLY via env_var(...), never plaintext
│   ├── dbt_project.yml     # silver/gold: contracts enforced at project level
│   ├── packages.yml        # dbt_expectations + elementary
│   └── models/             # sources.yml + silver/gold models with contracts
├── scripts/ping.py         # connection smoke test (make ping)
├── tests/                  # config + connection-param tests (no live DB)
└── Makefile                # setup / lint / test / bootstrap / dbt targets
```

## Setup (needs a Snowflake trial account)

1. Sign up at `signup.snowflake.com` (free trial, $400 credits, no card).
2. In Snowsight, create a warehouse `QUANT_WH` (XS, auto-suspend 60s) under
   **Admin → Warehouses** (or note an existing one like `COMPUTE_WH`).
3. Copy `.env.example` → `.env` and fill in your Snowflake values (both the
   `SNOWFLAKE_*` and matching `DBT_*` variables). Your password is the one you
   set at signup — Snowflake never shows it back.
4. Install, verify, bootstrap:

```bash
cd quant_signal
make setup        # uv sync (runtime + dev deps)
make dbt-setup    # uv sync (adds dbt-core + adapter)
make check        # ruff + pytest — no Snowflake needed
make bootstrap    # creates QUANT DB, QUANT_WH, schemas, roles (idempotent)
make ping         # verifies the live connection (SELECT 1)
make dbt-debug    # verifies the dbt connection
make dbt-deps     # installs dbt_expectations + elementary
make dbt-run      # builds silver/gold with contracts + tests
```

## Using the client

```python
from db.snowflake import SnowflakeClient

client = SnowflakeClient()
print(client.ping())                       # True once connected
df = client.query_df("SELECT 1 AS one")
client.insert_df(df, table_name="my_table")  # appends into QUANT.BRONZE
```

## Notes

- **No API key.** Snowflake authenticates with account + user + password (or
  RSA key-pair via `SNOWFLAKE_PRIVATE_KEY_FILE`). Snowflake is SaaS-only; your
  machine talks to your cloud trial account.
- **MFA.** New Snowflake accounts require MFA for password logins. Enroll in
  Snowsight (user menu → Settings → Authentication → Duo or an authenticator
  app — **not** a passkey, which can't be used programmatically), set
  `SNOWFLAKE_USE_MFA=true`, and the connector uses `username_password_mfa`
  with Duo push. The first connection prompts once; `ALLOW_CLIENT_MFA_CACHING`
  (set by `make bootstrap`) caches the token ~4h. For fully non-interactive
  automation later, use key-pair auth instead (MFA doesn't apply to it).
- **Spark → Snowflake** (next milestones): the `spark-snowflake` connector has
  no native stream sink — you write via `foreachBatch`. Streaming holds
  sessions/stages open and burns credits; keep the warehouse small +
  auto-suspend.
