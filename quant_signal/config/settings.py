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


def csv_floats(value: str) -> list[float]:
    """Split an env-provided CSV string into floats (research-grid axes).

    The harness sweep axes (λ multiples, taker-cost assumptions) are lists of
    floats in the environment — never literals in code.
    """
    return [float(part) for part in value.split(",") if part.strip()]


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
    # Crypto universe: the two majors are the *signals* (they move first), the
    # follower coins are the *targets* (they react a few minutes later — the
    # lead-lag edge documented by Jia et al. 2023 and Guo et al. 2024).
    ingest_default_crypto_symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT"
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
    # publishes venue minute bars (bybit by default) to the Kafka bus; the API
    # consumes the raw topic and fans deltas out over /ws/market. Disable for a
    # pure query API.
    stream_enabled: bool = True
    stream_poll_seconds: int = 15
    stream_poll_timeout_seconds: float = 45
    # Ring-buffer depth kept per symbol for WebSocket snapshots.
    stream_history_minutes: int = 180

    # ── Streaming stack (M3): Kafka ingestion bus + Redis online store ──────
    # Bootstrap servers for the message bus (comma-separated for a cluster).
    stream_kafka_bootstrap_servers: str = "localhost:9092"
    # Raw venue minute bars (producer → Flink → materializer). Keyed by symbol.
    stream_kafka_topic_raw: str = "crypto.bars.raw"
    # 1h window features computed by the Flink SQL job (crypto_features_1h.sql)
    # — the TRADING cadence. Research: hourly horizons turn over far less than
    # 5m, so taker fees stop dominating (Bysik & Ślepaczuk, "Machine Learning-
    # Based Bitcoin Trading Under Transaction Costs", arXiv:2606.00060, 2026;
    # the AdaptiveTrend paper reports Sharpe 2.41 at 6h vs 1.54 at 1h — longer
    # horizons monotonically cut the fee bleed). The seesaw/lead-lag edge the
    # literature documents at 5m (Jia et al. 2023, Guo et al. 2024) has NOT
    # been tested at 1h — that is our novel extension, cost-filtered.
    stream_kafka_topic_features: str = "crypto.features.1h"
    # The 5m feature pipeline stays RUNNING as the benchmark/data feed (the
    # "Alpha Autopsy" backtest source), even though trading moved to 1h. The
    # materializer keeps both routes; everything else reads the 1h topic/prefix.
    stream_kafka_topic_features_5m: str = "crypto.features.5m"
    # Online store where the materializer lands live bars + window features.
    # 6380: the docker-compose Redis is remapped off 6379 so it never
    # collides with a host Redis (the API/producer connect from the host).
    stream_redis_url: str = "redis://localhost:6380"
    stream_redis_live_prefix: str = "live:crypto"
    stream_redis_feature_prefix: str = "feature:crypto:1h"
    # The 5m feature prefix, kept alive for the benchmark feed.
    stream_redis_feature_prefix_5m: str = "feature:crypto:5m"
    # Windows of Flink features kept per symbol (RPUSH + LTRIM bound).
    stream_redis_feature_maxlen: int = 200
    # Staleness gate for the watchdog and the API's pipeline-status endpoint:
    # "healthy" means the latest feature window ended no more than this many
    # seconds before the latest raw bar (event-time delta, so the host clock
    # drifting is irrelevant). Sized to > one 1h window + watermark + slack
    # (a healthy 1h pipeline sits 0–60min behind the in-progress raw bar, so
    # the threshold must clear a full hour plus margin).
    stream_watchdog_staleness_threshold_seconds: float = 7200.0
    # Minimum gap between automatic Flink heals. After a heal the jobs restart
    # from ``latest-offset``, so the first 1h window cannot fire until the next
    # hour boundary (up to ~10 min with the watermark). Without this cooldown the
    # watchdog re-heals every interval, killing the just-started jobs before they
    # emit anything — a destructive feedback loop that keeps the pipeline down.
    stream_watchdog_heal_cooldown_seconds: float = 900.0
    # Venue (source exchange) for the live stream: the provider built by name
    # from the registry and stamped on every raw bar's ``provider`` field.
    stream_venue: str = "binance"
    # Bybit mainnet host for the keyless kline endpoint. Region mirrors exist
    # (api.bybit.id for Indonesia, api.bybit.eu for EEA, ...); the global host
    # works unless the machine runs from a Bybit-excluded jurisdiction.
    stream_bybit_base_url: str = "https://api.bybit.com"
    # Flink infrastructure the watchdog heals by name — the daemon never
    # hardcodes container ids, consumer groups, or job paths.
    stream_flink_jobmanager_container: str = "quant_signal-flink-jobmanager-1"
    stream_flink_taskmanager_container: str = "quant_signal-flink-taskmanager-1"
    stream_redpanda_container: str = "quant_signal-redpanda-1"
    # The 1h trading job (crypto_features_1h.sql) — the one the watchdog guards
    # as the critical pipeline. The 5m benchmark job stays running too and is
    # healed from its own group/SQL pair (below).
    stream_flink_consumer_group: str = "flink-crypto-features-1h"
    stream_flink_sql_path: str = "/opt/flink/jobs/crypto_features_1h.sql"
    # The 5m benchmark feed (crypto_features.sql) — kept alive as the Alpha
    # Autopsy data source, but it is NOT the trading cadence anymore.
    stream_flink_consumer_group_5m: str = "flink-crypto-features"
    stream_flink_sql_path_5m: str = "/opt/flink/jobs/crypto_features.sql"

    # ── Prediction layer (M3.5): River + conformal intervals + Monte Carlo ───
    # Online store prefixes for the predictor and simulator outputs. The
    # prediction/strategy/simulation/execution layers all run on the 1h cadence
    # (the TRADING clock); only the raw+features 5m feed stays for the benchmark.
    stream_redis_prediction_prefix: str = "prediction:crypto:1h"
    stream_redis_simulation_prefix: str = "simulation:crypto:1h"
    # Live strategy P&L curve (compounded equity vs buy-and-hold) derived from
    # the predictor's realized directions, for the Signal Terminal P&L strip.
    stream_redis_strategy_prefix: str = "strategy:crypto:1h"
    # Windows kept in the per-symbol strategy equity curve.
    stream_strategy_maxlen: int = 500
    # Adaptive Conformal Inference (Gibbs & Candès 2021): target miscoverage
    # and step size; the nominal level adapts so long-run coverage tracks
    # (1 - alpha) even under drift.
    stream_prediction_alpha: float = 0.1
    stream_prediction_gamma: float = 0.005
    # Residuals kept for the conformal interval quantile.
    stream_prediction_residual_window: int = 200
    # Trading rule: a LONG/SHORT is only placed when the predicted next-1h
    # return clears ±this. 0.002 = 20 bps = λ·c with λ=2 and c=10 bps round-trip
    # cost — the cost-aware filter of Bysik & Ślepaczuk ("Machine Learning-Based
    # Bitcoin Trading Under Transaction Costs", arXiv:2606.00060, 2026): their
    # eq. (5) opens a position only when |r̂| > λ·c·|pos*−pos_prev|. On hourly
    # BTC with λ=2 the filter flips XGBoost long-only from −64.0% ARC (10 619
    # trades, all the fee bleed) to +65.4% ARC (251 trades), and long-short from
    # −92.6% to +44.4% (60 trades). 20 bps also clears Bybit's 11 bps taker
    # round trip (5.5 bps/leg, v5 fee schedule) with margin. The banding
    # intuition — a higher hurdle to enter than to maintain — is the single most
    # effective cost-mitigation technique (Novy-Marx & Velikov, "A Taxonomy of
    # Anomalies and Their Trading Costs", RFS 29(1), 2016); the engine HOLDs
    # through signals weaker than the entry band (Gârleanu & Pedersen, "Dynamic
    # Trading with Predictable Returns and Transaction Costs", JoF 68(6), 2013:
    # costs create a no-trade region — hold on decay, don't churn).
    stream_prediction_direction_threshold: float = 0.002
    # Cross-coin lead-lag features: lagged returns of these symbols are fed as
    # features into EVERY symbol's model. The majors (BTC, ETH) move first and
    # the followers react a few minutes later — the "seesaw effect" (Jia, Wu,
    # Yan & Liu, J. Empirical Finance 74, 2023) and cross-crypto return
    # predictability (Guo, Sang, Tu & Wang, JEDC 163, 2024): large coins lead
    # small coins but not vice-versa, and a LASSO long-short earns 5-6 bps per
    # 5m gross / 0.07-0.08 bps net (73-84% annualized) after realistic costs.
    # Lagged leaders for SOL/XRP are the documented tradeable edge at 5m.
    stream_prediction_cross_symbols: str = "BTCUSDT,ETHUSDT"
    # Vol-transmitter symbols (Jinan SA-Log-HAR, Fu/Zhu/Liu arXiv:2507.22409;
    # TVTP-MS-HAR, Mathematics 13(15), 2025): XRP/XLM/LTC are NET vol
    # transmitters whose spillovers AMPLIFY at both tails — their trailing
    # 24h realized vol and vol-shock feed every model as the state-dependent
    # early-warning channel, and their shock state drives the regime gate.
    # NOTE: the transmitter hierarchy is SAMPLE-dependent (FinInnov 2025:
    # BTC/ETH transmit) — so the coefficients are learned online, never
    # hardcoded; this list only names who is ALLOWED to act as a vol source.
    stream_prediction_vol_symbols: str = "XRPUSDT,SOLUSDT"
    # Monte Carlo engine: paths (a power of two so the Sobol low-discrepancy
    # net keeps its (0,m,s)-net stratification property — Niederreiter 1992),
    # forward horizon (5m steps), trailing 5m windows used to estimate the
    # per-step log-return volatility.
    stream_simulation_paths: int = 16_384
    stream_simulation_horizon_steps: int = 12
    stream_simulation_vol_windows: int = 40
    # Sampler for the path shocks. Crude pseudo-random MC converges at the
    # O(N^-1/2) rate (Staum, WSC 2003), so doubling accuracy costs 4x paths.
    # Scrambled Sobol quasi-MC (RQMC) replaces the pseudo-random uniforms with
    # a low-discrepancy point set whose integration error is ~O((log N)^s / N)
    # (Sobol 1967; Bratley & Fox 1988; Joe & Kuo 2008), far tighter at the
    # tails (VaR/ES, tail odds, Kelly). Scrambling (Owen 1997/2003) keeps the
    # estimate unbiased so the reality-check monitor stays meaningful, with the
    # seed drawn from the window so each 5m forecast stays reproducible. The
    # inverse-CDF transform (norm.ppf / t.ppf) preserves the low-discrepancy
    # ordering (Glasserman 2003, ch.5); a Box-Muller transform would destroy
    # it. "crude" keeps the old numpy default_rng path as a fallback.
    stream_simulation_sampler: str = "sobol-rqmc"
    # Volatility model: RiskMetrics EWMA. sigma_t² = λ·sigma_{t−1}² +
    # (1−λ)·r_t² decays past shocks geometrically, so vol responds fast to
    # bursts and decays slowly — volatility clusters (Mandelbrot 1963; Fama
    # 1965) and the equal-weight trailing stdev smears the cluster into the
    # forecast. J.P. Morgan's RiskMetrics (1994/2006) and Hendricks (NY Fed
    # 1996) fix λ=0.94.
    stream_simulation_ewma_lambda: float = 0.94
    # Innovation family: Student-t with the df estimated from the standardized
    # residuals (clamped to [min, max]). Fat tails are the empirical norm, and
    # Normal-innovation GARCH materially under-covers tail risk while
    # Student-t captures the tail shape (Bollerslev 1987; Horváth & Šopov
    # 2016). df→∞ recovers the Normal model.
    stream_simulation_t_df_min: float = 4.0
    stream_simulation_t_df_max: float = 30.0
    # MLE drift toggle: when true, mu = mean(log returns) + sigma^2/2 (MLE for
    # GBM per-5m log returns), so the median fan path and P(up) track the real
    # trailing trend instead of hovering at 50% under a driftless martingale.
    stream_simulation_drift: bool = True
    # Decision-layer thresholds on the simulated distribution (the "what to
    # bet" box). Kelly is the growth-optimal fraction f* = E[log r]/Var(log r)
    # of capital for a position with continuous log-returns (Kelly 1956;
    # Breiman 1961; Thorp 2006) — reported capped, because firms run
    # FRACTIONAL Kelly in practice: full-Kelly is only optimal if you know the
    # true edge, and estimation error / drawdown risk argues for half-Kelly or
    # less (MacLean, Thorp & Ziemba 2010; Stanford risk-constrained Kelly,
    # Boyd et al.; Thorp, "Kelly Simulations"). A position side is recommended
    # only when the simulated edge clears a floor expressed in units of
    # terminal volatility (edge_min_sigma × σ·√horizon): below that the honest
    # call is FLAT, because a near-zero edge estimate is noise, not signal.
    stream_simulation_kelly_cap: float = 0.25
    stream_simulation_edge_min_sigma: float = 0.05
    # Raw paths shipped to the UI for the all-paths fan (thin canvas lines);
    # percentile statistics always use *all* paths — this is purely how many
    # strokes the browser renders per window.
    stream_simulation_sample_paths: int = 1000
    # Forecast calibration monitor ("reality check"): the nominal coverage of
    # the MC fan's central 10–90 band at the 1-step-ahead horizon (0.8), and
    # the anytime-valid e-value alarm level (Ville's inequality gives
    # P(sup M_t >= 1/alpha) <= alpha under calibration, so 0.005 ≈ 1 alarm in
    # 200 windows by chance while monitoring continuously).
    stream_reality_nominal_coverage: float = 0.8
    stream_reality_evalue_alpha: float = 0.005
    # Strategy validation Monte Carlo (QuantPad-style pass probability):
    # bootstrap the realized signal returns into simulated futures and score
    # them against pass/fail rules. By default the rules are RISK-SCALED to
    # the strategy's own realized terminal volatility (target = target_sigma ×
    # σ_T, max DD = max_drawdown_sigma × σ_T, where σ_T = per-window σ·√n):
    # a fixed human-scaled 6%/8% contract is structurally unreachable for a
    # 5m signal (per-window returns of a few bps), so the pass gauge would pin
    # at 0%/100%/0% forever. Risk-scaling restores a well-posed, moving
    # question about realized edge relative to risk (research: the target-to-
    # drawdown ratio, not the absolute percentages, decides challenge
    # difficulty — OneTradeJournal/CrossTrade/PropFlux). The explicit
    # fixed-contract rules below (QuantPad/FTMO-style defaults) are what the
    # what-if scenarios override with when set.
    stream_validation_sims: int = 10_000
    stream_validation_target_sigma: float = 1.0
    stream_validation_max_drawdown_sigma: float = 1.5
    # Profit target / max drawdown for the *explicit fixed-contract* mode
    # (QuantPad/FTMO-style, e.g. Topstep 50K ≈ 6% target / FTMO ≈ 8-10%).
    # Passed through when a scenario or caller supplies them; the live default
    # risk-scales instead.
    stream_validation_target: float = 0.06
    stream_validation_max_drawdown: float = 0.08
    stream_validation_seed: int = 42
    # What-if scenario library for the Signal Terminal's stress preview. Each
    # scenario re-runs the *real* engines (MC fan / bootstrap validation) with
    # the listed knobs and is echoed into the response so the UI labels it:
    #   sigma_scale   → volatility multiplier for the price MC fan/surface
    #   target        → explicit profit-target contract (validation)
    #   max_drawdown  → explicit drawdown rule override (validation), so a
    #                   tighter stop busts more futures
    # A scenario is a what-if on real calibrated inputs, never a fake feed.
    stream_scenarios: dict[str, dict[str, float]] = {
        "stress": {"sigma_scale": 4.0, "target": 0.06, "max_drawdown": 0.03},
    }

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

    # ── Autonomous research harness (offline parameter sweep) ────────────────
    # The harness ("Autonomous AI Researcher") sweeps the predictor's config
    # grid offline, replays the EXACT progressive-validation loop over every
    # configuration, ranks the trials, and deflates the winner for the total
    # number of trials — the multiple-testing disclosure the literature
    # demands: a Sharpe picked as the best of N trials is the maximum of N
    # draws, and the maximum of pure noise is large (Bailey & López de Prado,
    # "The Deflated Sharpe Ratio", JPM 40(5) 94–107, 2014; qbx-research:
    # K=50 typical trials → spurious max Sharpe ~0.4–0.8). The sweep axis is
    # λ — the cost-aware filter multiple of Bysik & Ślepaczuk (arXiv:
    # 2606.00060, eq. 5); the entry hurdle is derived as direction_threshold
    # = λ × round-trip taker cost (λ=2 @ 10bps = the live 20bps band).
    research_lambda_values: str = "0.0,0.5,1.0,1.5,2.0,2.5,3.0"
    research_taker_cost_values: str = "0.0005,0.001"
    # Feature-set variants (ablation chain toward the live set):
    #   "single"  = current-window own features only (baseline)
    #   "history" = + own multi-scale realized vol (HAR-RV: lag_ret, rv_1h/4h/24h,
    #               vol_shock — Corsi 2009; the Jinan vol-spillover line extends it)
    #   "vol"     = + cross-vol spillover from the transmitters (lag_<x>_rv24h,
    #               lag_<x>_vol_shock, stress_max, stress_count — SA-Log-HAR)
    #   "cross"   = + lagged cross-coin RETURNS too (the full LIVE feature set;
    #               sign learned online, never hardcoded — the spillover
    #               direction is sample-dependent across the literature).
    research_feature_modes: str = "single,history,vol,cross"
    # Search method: "grid" = exhaustive Cartesian product (small spaces);
    # "random" = seeded budget-constrained draws (large spaces) — same
    # coverage-per-sample argument as qbx-research's latin_hypercube.
    research_search_method: str = "grid"
    # 0 → full grid; >0 → cap trials when method is "random".
    research_budget: int = 0
    # Pinned so a sweep is byte-identical across reruns (reproducibility is a
    # first-class property of research infrastructure — Hasan Javed harness).
    research_seed: int = 42
    # Ranking objective + hard constraint: a trial must clear its own assumed
    # taker cost (non-negative excess return) to be a candidate for winner.
    research_objective: str = "annual_sharpe_strategy"
    research_min_excess_return: float = 0.0
    research_top_k: int = 10
    # Stability gate: the winner's per-block IC must not vary wildly across
    # time (coefficient of variation < this) or the edge is one lucky period
    # (offline-pixel WFV guide: CV > 0.2 → unstable, reject). Plateau over
    # peak — an isolated optimum is a "cliff" (AlgoXpert, arXiv:2603.09219).
    research_max_block_ic_cv: float = 0.2
    # Number of contiguous blocks the replay splits history into for the
    # per-block stability check.
    research_n_blocks: int = 4
    # Where each symbol's sweep leaderboard lands in the online store.
    research_leaderboard_prefix: str = "research:harness:leaderboard"

    # ── Data quality / SLA / visibility ──────────────────────────────────────
    # The five-pillar observability model (freshness, volume, distribution,
    # schema, lineage — Soda/Elementary) applied to the streaming window
    # store: every dimension's threshold is RELATIVE to the pipeline cadence
    # (window_ms), never a fixed wall-clock number, because an hourly pipeline
    # should alert on a 2h stall, not on "24h" (Soda: "set freshness
    # thresholds relative to your pipeline's expected cadence, not a fixed
    # window").
    # Freshness SLA: the feature window may lag the venue's latest raw bar by
    # at most this many cadences before it's a stall.
    quality_freshness_max_cadences: float = 2.0
    # Volume: expected windows in the trailing lookback (lookback_ms / window_ms
    # for a 24/7 crypto calendar); a deviation beyond these ratios is an
    # anomaly (Soda: alert on ~15-20% relative deviation, not absolute counts).
    quality_volume_lookback_hours: float = 24.0
    quality_volume_warn_ratio: float = 0.9
    quality_volume_min_ratio: float = 0.8
    # Uniqueness/ordering/gaps: at-least-once Kafka delivery can duplicate a
    # window; a dropped message leaves a gap; out-of-order windows break the
    # chronology the online model depends on. Surfaced, never silently
    # absorbed (vexdata: duplicates and gaps accumulate silently otherwise).
    quality_max_duplicate_ratio: float = 0.01
    quality_max_out_of_order: int = 0
    quality_max_gaps_warn: int = 1
    # Validity/accuracy: required numeric fields per window, and value bounds
    # mirroring the predictor's own sanity clamps (a positive close, sane
    # high≥low, ratio features within ±50%).
    quality_required_fields: str = "open,high,low,close,vwap,volume,bar_count"
    quality_min_completeness: float = 0.9
    # Health-score bands for the Elementary-style quality dimensions; the
    # overall score is the WORST dimension (one broken dimension = not
    # healthy), so the dashboard can't be gamed by weighting.
    quality_score_ok: float = 0.9
    quality_score_warn: float = 0.7
    # Where the per-symbol quality summaries land in the online store.
    quality_prefix: str = "data:quality"

    # ── Paper execution layer (M3.6): simulated fills on real signals ────────
    # The execution simulator turns the predictor's realized directions into
    # filled trades: market order at the *next* window's close (no lookahead),
    # fixed-bps slippage on both legs, taker fee on entry and exit, and a fill
    # ledger. Deterministic by construction — fills are next-close + fixed bps,
    # so there is no PRNG to seed; ``window_end_ms`` is echoed for audit.
    stream_redis_execution_prefix: str = "execution:crypto:1h"
    # 1h window cadence in ms — the fill/exit rhythm of the paper book (must
    # mirror the Flink 1h feature windows; configurable so nothing is baked in).
    # 3 600 000 ms = one hour: the TRADING cadence after the 5m rebuild (the
    # 5m book was structurally negative at taker costs — 9/9 maker-first
    # fallbacks; the paper: "the main obstacle in hourly cryptocurrency trading
    # is not only weak predictability, but also the way forecasts are converted
    # into trades", Bysik & Ślepaczuk, arXiv:2606.00060, 2026).
    stream_window_ms: int = 3_600_000
    # Notional deployed per trade (USD) — a fixed-size paper book.
    stream_execution_notional_usd: float = 1000.0
    # Pessimistic market-fill slippage in bps (pay the spread's adverse side).
    stream_execution_slippage_bps: float = 2.0
    # Taker fee in bps (Binance-style), charged on both entry and exit.
    stream_execution_taker_fee_bps: float = 10.0
    # Cost-aware execution filter (Bysik & Ślepaczuk 2026, eq. 5): trade only
    # when |r̂| > λ·c·|pos*−pos_prev|, c = round-trip taker fee. λ=2 → entry at
    # 20 bps, reversal at 40 bps. ``hold_until_decay`` keeps the position
    # through signals weaker than the entry band (banding / no-trade region)
    # instead of rolling every window — the mechanism that turned −64.0% ARC
    # (XGBoost long-only, 10 619 trades) into +65.4% ARC (251 trades) in the
    # paper. Set False to keep the legacy one-window roll for the 5m benchmark.
    stream_execution_cost_filter_lambda: float = 2.0
    stream_execution_hold_until_decay: bool = True
    # Prefer maker fills on the demo venue: a post-only limit at the near-side
    # quote (bid for LONG entries / LONG-exit sells, ask for SHORT) rests on
    # the book and pays Bybit's 0.02% maker fee instead of 0.055% taker — an
    # 11 → 4 bps round trip (Bybit v5 fee schedule). The order rests briefly
    # (poll ~15s); if unfilled it is cancelled and the engine falls back to a
    # market order, so signals still participate. Maker pricing also fills at
    # the better side of the spread, not the adverse side.
    stream_execution_maker_first: bool = True
    # Fill-ledger cap kept per symbol (older fills are trimmed for the UI).
    stream_execution_ledger_maxlen: int = 50
    # Max closed trades per symbol before new entries halt (book keeps marking
    # to market and closing the open position — no more new risk).
    stream_execution_max_trades: int = 100
    # Execution venue. "paper" fills at the next window's close with a fixed
    # adverse-side slippage + taker fee model (no real orders). "bybit-demo"
    # places REAL market orders on Bybit's free Demo Trading account
    # (api-demo.bybit.com, virtual USDT — no deposit, no KYC), so the book sees
    # actual exchange fills, latency and rejections. Honest behavior: when the
    # demo keys are absent the venue falls back to paper at config time; once a
    # demo venue is active, a failed order is SKIPPED (counted), never faked
    # with a paper fill. Research: Bybit v5 Open API Demo docs — demo keys only
    # work against api-demo.bybit.com and WS Trade is unsupported on demo, so
    # fills are read back over REST.
    stream_execution_venue: str = "paper"
    # Bybit Demo credentials (gitignored .env). Created under the Demo Trading
    # account (bybit.com → profile → Demo Trading → API), NOT mainnet; mainnet
    # keys fail on the demo host with ErrCode 10003. repr=False: never logged.
    bybit_demo_api_key: str | None = Field(default=None, repr=False)
    bybit_demo_api_secret: str | None = Field(default=None, repr=False)
    # Accepted aliases for the same credentials: API_KEY_BYBIT / API_SECRET_BYBIT
    # (the names some earlier Bybit bots export). The canonical BYBIT_DEMO_*
    # names win when both are set; either pair enables the demo venue.
    api_key_bybit: str | None = Field(default=None, repr=False)
    api_secret_bybit: str | None = Field(default=None, repr=False)
    # Server-time tolerance for signed requests (recv_window, Bybit v5).
    bybit_demo_recv_window_ms: int = 5000

    # ── Batch warehouse sink (streaming online store → ClickHouse BRONZE) ─────
    # The materialize_clickhouse flow copies the live Redis artifacts (features,
    # predictions, strategy, execution) into ClickHouse for batch analytics +
    # Grafana. ClickHouse is the OLAP warehouse (Snowflake-class) in the root
    # docker-compose; "default" user has full access on the local dev instance.
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    # Mirrors infra/clickhouse/users.d/default.xml — the dev container's
    # deterministic default-user password (24.3 image ships a random one).
    clickhouse_password: str = "mlops"
    clickhouse_database: str = "quant"

    # ── Data lake (Iceberg on S3-compatible object storage) ─────────────────
    # Open-table-format tier: the Snowflake mart (GOLD.FEATURES, ...) is
    # versioned to an Apache Iceberg table on MinIO/S3. Iceberg is storage-
    # agnostic, so the exact same tables run on real S3 later by swapping the
    # endpoint — the AWS Terraform's storage module already provisions the
    # bucket (infra/terraform/modules/storage). Catalog backend is a SQLAlchemy
    # URI: SQLite for local dev, a JDBC/Postgres URI for a shared catalog.
    lake_enabled: bool = False
    # S3-compatible endpoint (HTTP URL — MinIO/S3 path-style addressing is
    # derived from the host; never prefix with s3://).
    lake_endpoint: str = "http://localhost:9000"
    # Credentials for the object store (MinIO root user in dev; IAM keys in AWS).
    lake_access_key: str | None = Field(default=None, repr=False)
    lake_secret_key: str | None = Field(default=None, repr=False)
    lake_region: str = "us-east-1"
    # Bucket that holds the Iceberg warehouse (created idempotently on export).
    lake_bucket: str = "quant-lake"
    # Iceberg warehouse URI = the bucket root; tables land at
    # s3://{lake_bucket}/{namespace}/{table}.
    lake_namespace: str = "gold"
    lake_table_features: str = "features"
    # Snapshot retention (hours). Repeated exports keep old snapshots (time
    # travel), but unbounded snapshot history is the #1 Iceberg metadata trap
    # at scale — manifest bookkeeping becomes the latency floor. Expire
    # snapshots older than this on every export (7 days default).
    lake_snapshot_retention_hours: float = 168.0
    # Coarse identity partition column. Iceberg's scaling rule: partition on
    # the coarsest grain that prunes your dominant query pattern and stay well
    # under a few thousand partitions; finer grains make metadata planning the
    # bottleneck. SYMBOL is 32 low-cardinality values — right-sized for
    # per-symbol scans and time-series backtests.
    lake_partition_by: str = "SYMBOL"
    # Catalog database file (SQLite). Absolute under the project so it works
    # from any CWD; the whole lake/ dir is gitignored as local state.
    lake_catalog_uri: str = Field(
        default_factory=lambda: f"sqlite:///{PROJECT_ROOT}/lake/catalog.db"
    )

    @property
    def has_bybit_demo_credentials(self) -> bool:
        return bool(
            (self.bybit_demo_api_key or self.api_key_bybit)
            and (self.bybit_demo_api_secret or self.api_secret_bybit)
        )

    @property
    def demo_api_key(self) -> str | None:
        return self.bybit_demo_api_key or self.api_key_bybit

    @property
    def demo_api_secret(self) -> str | None:
        return self.bybit_demo_api_secret or self.api_secret_bybit

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
        if self.lake_enabled and (not self.lake_access_key or not self.lake_secret_key):
            raise ValueError(
                "LAKE_ENABLED=true requires LAKE_ACCESS_KEY and LAKE_SECRET_KEY "
                "to be set (the object-store credentials)"
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
