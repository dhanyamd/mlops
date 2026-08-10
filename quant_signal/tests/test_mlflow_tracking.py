"""MLflow tracking tests — hermetic (no MLflow server, no network).

The served models are online learners (River) + Monte Carlo, so only MLflow
*tracking* is used (params/metrics/artifacts per validation run), not the model
registry. Covers: tracking disabled -> no-op; mlflow absent -> no-op; a fake
mlflow module receiving the right params/metrics/artifacts; and the endpoint's
explicit ?track=true opt-in.
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

from api.main import app
from config.settings import get_settings
from stream.kv import FakeKV
from stream.mlflow_tracking import track_gate_report, track_validation
from stream.predictive_eval import evaluate_predictor, passes_gate
from tests.test_strategy_mc import _PROFITABLE


def _feature_windows(n: int = 150) -> list[dict]:
    """Synthetic 5m feature windows (clean uptrend) for the gate replay.

    Mirrors the fixture in test_predictive_eval: real prices, a monotonic
    trend the model can actually learn, and the Flink schema (symbol +
    window_end_ms + OHLCV) the materializer lands in Redis.
    """
    return [
        {
            "symbol": "BTCUSDT",
            "window_start_ms": i * 300_000,
            "window_end_ms": (i + 1) * 300_000,
            "open": 1000.0 + i,
            "high": 1000.5 + i,
            "low": 999.5 + i,
            "close": 1000.5 + i,
            "vwap": 1000.3 + i,
            "volume": 1000.0,
            "bar_count": 5,
        }
        for i in range(n)
    ]


def _sample_validation() -> dict:
    from stream.strategy_mc import StrategyMonteCarlo

    mc = StrategyMonteCarlo(n_sims=100, target=0.06, max_drawdown=0.08, seed=42)
    return mc.validate(_PROFITABLE)


def _fake_mlflow() -> types.ModuleType:
    """A recording fake mlflow module (context-manager start_run)."""
    mod = types.ModuleType("mlflow")
    calls = {"params": {}, "metrics": {}, "dicts": {}, "run_id": None}

    def set_tracking_uri(uri):  # type: ignore[no-untyped-def]
        calls["uri"] = uri

    def set_experiment(name):  # type: ignore[no-untyped-def]
        calls["experiment"] = name

    class _Run:
        info = types.SimpleNamespace(run_id="run-123")

    class _RunCtx:
        def __init__(self, **kw):  # type: ignore[no-untyped-def]
            self._run = _Run()

        def __enter__(self) -> _Run:
            return self._run

        def __exit__(self, *exc):  # type: ignore[no-untyped-def]
            return None

    def start_run(run_name=None):  # type: ignore[no-untyped-def]
        calls["run_name"] = run_name
        return _RunCtx()

    def log_params(params):  # type: ignore[no-untyped-def]
        calls["params"].update(params)

    def log_metrics(metrics):  # type: ignore[no-untyped-def]
        calls["metrics"].update(metrics)

    def log_dict(d, artifact_file):  # type: ignore[no-untyped-def]
        calls["dicts"][artifact_file] = d

    def active_run() -> _Run:
        return _Run()

    mod.set_tracking_uri = set_tracking_uri
    mod.set_experiment = set_experiment
    mod.start_run = start_run
    mod.log_params = log_params
    mod.log_metrics = log_metrics
    mod.log_dict = log_dict
    mod.active_run = active_run
    mod._calls = calls  # type: ignore[attr-defined]
    return mod


def test_tracking_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_tracking_enabled", False)
    validation = _sample_validation()
    assert (
        track_validation("BTCUSDT", validation, target=0.06, max_drawdown=0.08, seed=42, n_sims=100)
        is None
    )


def test_tracking_without_mlflow_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_tracking_enabled", True)
    monkeypatch.setitem(sys.modules, "mlflow", None)  # import mlflow fails
    validation = _sample_validation()
    assert (
        track_validation("BTCUSDT", validation, target=0.06, max_drawdown=0.08, seed=42, n_sims=100)
        is None
    )


def test_tracking_logs_params_metrics_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_tracking_enabled", True)
    fake = _fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    validation = _sample_validation()

    run_id = track_validation(
        "BTCUSDT", validation, target=0.06, max_drawdown=0.08, seed=42, n_sims=100
    )

    calls = fake._calls  # type: ignore[attr-defined]
    assert run_id == "run-123"
    assert calls["run_name"] == "strategy-validation-btcusdt"
    assert calls["uri"] == get_settings().mlflow_tracking_uri
    assert calls["experiment"] == "quant_signal"
    assert calls["params"]["symbol"] == "BTCUSDT"
    assert calls["params"]["target"] == 0.06
    assert calls["params"]["max_drawdown"] == 0.08
    assert calls["params"]["seed"] == 42
    assert calls["metrics"]["pass_probability"] == validation["pass_probability"]
    assert calls["metrics"]["p95_max_drawdown"] == validation["p95_max_drawdown"]
    assert calls["metrics"]["n_sims"] == 100
    assert calls["dicts"]["terminal_histogram.json"] == validation["terminal_histogram"]
    assert calls["dicts"]["drawdown_histogram.json"] == validation["drawdown_histogram"]


def test_tracking_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_tracking_enabled", True)

    class _Boom:
        def set_tracking_uri(self, uri):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "mlflow", _Boom())
    validation = _sample_validation()
    assert (
        track_validation("BTCUSDT", validation, target=0.06, max_drawdown=0.08, seed=42, n_sims=100)
        is None
    )


def test_endpoint_tracks_only_with_query_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.main.RedisKV", lambda url: FakeKV())
    calls: list[dict] = []

    def fake_track(symbol, validation, **kw):  # type: ignore[no-untyped-def]
        calls.append({"symbol": symbol, "validation": validation})
        return "run-1"

    monkeypatch.setattr("api.main.track_validation", fake_track)

    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_tracking_enabled", True)

    with TestClient(app) as client:
        kv = client.app.state.kv
        kv.set_json(
            "strategy:crypto:1h:BTCUSDT",
            {
                "symbol": "BTCUSDT",
                "n_windows": 30,
                "strategy_equity": [1.0 + 0.002 * i for i in range(30)],
                "buyhold_equity": [1.0] * 30,
            },
        )
        # Without the flag: no tracking run.
        resp = client.get("/api/market/validation/btcusdt")
        assert resp.json()["validation"] is not None
        assert calls == []
        # With the flag: exactly one run.
        resp = client.get("/api/market/validation/btcusdt?track=true")
        assert resp.json()["validation"] is not None
        assert len(calls) == 1
        assert calls[0]["symbol"] == "BTCUSDT"
        assert "pass_probability" in calls[0]["validation"]


# ── Promotion gate: MLflow tracking + API wiring ────────────────────────────


def _gated_report() -> tuple[dict, list[str], bool]:
    """Progressive-validation report for the synthetic uptrend, gated once.

    The gate is deliberately strict, so this fixture may or may not pass — the
    tests assert the *tracking* records whatever verdict was computed, not
    that the fixture clears the gate.
    """
    report = evaluate_predictor(_feature_windows(200), taker_cost=0.0005)
    passed, failures = passes_gate(report, get_settings(), n_trials=1)
    return report, failures, passed


def test_gate_tracking_logs_params_metrics_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "mlflow_tracking_enabled", True)
    fake = _fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    report, failures, passed = _gated_report()
    run_id = track_gate_report(
        "BTCUSDT",
        report,
        failures,
        n_trials=1,
        taker_cost=0.0005,
        alpha=0.1,
        gamma=0.005,
        residual_window=200,
    )

    calls = fake._calls  # type: ignore[attr-defined]
    assert run_id == "run-123"
    assert calls["run_name"] == "gate-eval-btcusdt"
    assert calls["uri"] == settings.mlflow_tracking_uri
    assert calls["experiment"] == "quant_signal"
    assert calls["params"]["symbol"] == "BTCUSDT"
    assert calls["params"]["n_trials"] == 1
    assert calls["params"]["taker_cost"] == 0.0005
    # Flat metrics land as floats; the verdict is 0/1 so runs are filterable.
    assert calls["metrics"]["passes"] == (1.0 if passed else 0.0)
    assert calls["metrics"]["n_failures"] == len(failures)
    assert calls["metrics"]["deflated_sharpe"] == report["deflated_sharpe"]
    # Artifact carries the report + reasons, minus the raw per-window returns.
    artifact = calls["dicts"]["gate_report.json"]
    assert artifact["failures"] == failures
    assert artifact["passes"] is passed
    assert "_strat_rets" not in artifact


def test_gate_endpoint_returns_verdict_and_tracks_only_with_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api.main.RedisKV", lambda url: FakeKV())
    calls: list[dict] = []

    def fake_track(symbol, report, failures, **kw):  # type: ignore[no-untyped-def]
        calls.append({"symbol": symbol, "failures": failures})
        return "run-1"

    monkeypatch.setattr("api.main.track_gate_report", fake_track)

    with TestClient(app) as client:
        kv = client.app.state.kv
        for window in _feature_windows(150):
            kv.push_json("feature:crypto:1h:BTCUSDT", window, maxlen=200)

        # Without the flag: verdict served, no tracking run.
        resp = client.get("/api/market/gate/btcusdt")
        assert resp.status_code == 200
        gate = resp.json()["gate"]
        assert gate is not None
        assert isinstance(gate["passes"], bool)
        assert isinstance(gate["failures"], list)
        assert "ic" in gate and "deflated_sharpe" in gate
        assert "_strat_rets" not in gate
        assert calls == []

        # With the flag: exactly one tracked run with the same verdict.
        resp = client.get("/api/market/gate/btcusdt?track=true")
        assert resp.json()["gate"]["passes"] == gate["passes"]
        assert len(calls) == 1
        assert calls[0]["symbol"] == "BTCUSDT"


def test_gate_endpoint_disabled_without_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.main.RedisKV", lambda url: None)
    with TestClient(app) as client:
        resp = client.get("/api/market/gate/btcusdt")
    assert resp.json() == {"symbol": "BTCUSDT", "enabled": False, "gate": None}
