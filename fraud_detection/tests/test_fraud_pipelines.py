"""Fraud pipeline unit tests."""

import pandas as pd

from fraud_detection.data.ingest import transaction_vector
from shared.monitoring.drift import detect_univariate_drift, should_retrain


def test_transaction_vector_shape() -> None:
    row = pd.Series({f"V{i}": 0.1 for i in range(1, 29)} | {"Amount": 100.0, "Time": 3600.0})
    vec = transaction_vector(row)
    assert len(vec) == 30


def test_drift_triggers_retrain() -> None:
    ref = pd.Series(list(range(100)), name="x")
    cur = pd.Series(list(range(100, 200)), name="x")
    report = detect_univariate_drift(ref, cur, threshold=0.05)
    assert bool(report.drifted) is True
    assert should_retrain([report], min_drifted=1) is True
