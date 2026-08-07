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
from stream.mlflow_tracking import track_validation
from tests.test_strategy_mc import _PROFITABLE


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
            "strategy:crypto:5m:BTCUSDT",
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
