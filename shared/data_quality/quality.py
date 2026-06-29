"""Data quality gates using Great Expectations.

Why Great Expectations in production:
  Raw data has bugs. Sensors fail. Pipelines emit NULLs. CSVs have wrong
  delimiters. If you train a model on garbage data, you get a garbage model.
  A quality gate stops the pipeline BEFORE the bad data touches your model.

Architecture:
  ETL extract → QUALITY GATE (this module) → transform → load
  If quality gate fails:
    1. Prometheus metric mlops_data_quality_gate set to 0
    2. Structured log with severity=error
    3. Pipeline halts (prevents bad data from reaching training)
    4. Alert sent via webhook (Pillar 14)

Usage:
    from shared.data_quality.quality import validate_fraud_transactions

    df = pd.read_parquet("raw_transactions.parquet")
    result = validate_fraud_transactions(df)
    if not result.passed:
        raise DataQualityError(result.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from shared.observability.logging import get_logger
from shared.observability.metrics import DATA_QUALITY_GATE

log = get_logger(__name__)


@dataclass
class QualityResult:
    """Result of a data quality gate check."""

    checkpoint_name: str
    passed: bool
    total_expectations: int
    passed_expectations: int
    failed_expectations: int
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASSED ✅" if self.passed else "FAILED ❌"
        lines = [
            f"Quality Gate: {self.checkpoint_name} — {status}",
            f"  Expectations: {self.passed_expectations}/{self.total_expectations} passed",
        ]
        for f in self.failures:
            lines.append(f"  ❌ {f['expectation']}: {f['message']}")
        return "\n".join(lines)


class DataQualityError(Exception):
    """Raised when a data quality gate fails and the pipeline must halt."""


def _run_expectations(
    df: pd.DataFrame,
    expectations: list[dict],
    checkpoint_name: str,
    project: str,
) -> QualityResult:
    """Run a list of expectations against a DataFrame.

    Each expectation is a dict:
        {"check": "no_nulls", "column": "amount"}
        {"check": "range", "column": "amount", "min": 0, "max": 100000}
        {"check": "min_rows", "count": 100}
        {"check": "unique", "column": "transaction_id"}
        {"check": "values_in", "column": "is_fraud", "values": [0, 1]}
    """
    failures = []
    passed_count = 0

    for exp in expectations:
        check = exp["check"]
        col = exp.get("column")

        try:
            if check == "no_nulls":
                null_count = int(df[col].isnull().sum())
                if null_count > 0:
                    failures.append({
                        "expectation": f"no_nulls({col})",
                        "message": f"{null_count} null values found",
                    })
                else:
                    passed_count += 1

            elif check == "range":
                out_of_range = df[
                    (df[col] < exp["min"]) | (df[col] > exp["max"])
                ]
                if len(out_of_range) > 0:
                    failures.append({
                        "expectation": f"range({col}, {exp['min']}-{exp['max']})",
                        "message": f"{len(out_of_range)} values out of range",
                    })
                else:
                    passed_count += 1

            elif check == "min_rows":
                if len(df) < exp["count"]:
                    failures.append({
                        "expectation": f"min_rows({exp['count']})",
                        "message": f"Only {len(df)} rows (need {exp['count']})",
                    })
                else:
                    passed_count += 1

            elif check == "unique":
                dupes = int(df[col].duplicated().sum())
                if dupes > 0:
                    failures.append({
                        "expectation": f"unique({col})",
                        "message": f"{dupes} duplicate values",
                    })
                else:
                    passed_count += 1

            elif check == "values_in":
                invalid = df[~df[col].isin(exp["values"])]
                if len(invalid) > 0:
                    failures.append({
                        "expectation": f"values_in({col}, {exp['values']})",
                        "message": f"{len(invalid)} invalid values",
                    })
                else:
                    passed_count += 1

            elif check == "no_negative":
                negatives = df[df[col] < 0]
                if len(negatives) > 0:
                    failures.append({
                        "expectation": f"no_negative({col})",
                        "message": f"{len(negatives)} negative values",
                    })
                else:
                    passed_count += 1

            else:
                log.warning("unknown_quality_check", check=check)

        except Exception as exc:
            failures.append({
                "expectation": f"{check}({col})",
                "message": f"Check error: {str(exc)}",
            })

    passed = len(failures) == 0
    total = passed_count + len(failures)

    # Prometheus metric
    DATA_QUALITY_GATE.labels(project=project, checkpoint=checkpoint_name).set(
        1 if passed else 0
    )

    result = QualityResult(
        checkpoint_name=checkpoint_name,
        passed=passed,
        total_expectations=total,
        passed_expectations=passed_count,
        failed_expectations=len(failures),
        failures=failures,
    )

    if passed:
        log.info("quality_gate_passed", checkpoint=checkpoint_name, expectations=total)
    else:
        log.error("quality_gate_failed", checkpoint=checkpoint_name, summary=result.summary)

    return result


# ── Fraud Detection Quality Gates ──────────────────────────────────────────────

FRAUD_RAW_EXPECTATIONS = [
    {"check": "min_rows", "count": 1000},
    {"check": "no_nulls", "column": "transaction_id"},
    {"check": "no_nulls", "column": "user_id"},
    {"check": "no_nulls", "column": "amount"},
    {"check": "no_nulls", "column": "is_fraud"},
    {"check": "unique", "column": "transaction_id"},
    {"check": "range", "column": "amount", "min": 0.01, "max": 100000},
    {"check": "values_in", "column": "is_fraud", "values": [0, 1]},
]

FRAUD_FEATURES_EXPECTATIONS = [
    {"check": "min_rows", "count": 1000},
    {"check": "no_nulls", "column": "amount"},
    {"check": "no_nulls", "column": "txn_count_1h"},
    {"check": "no_nulls", "column": "velocity_ratio"},
    {"check": "no_negative", "column": "txn_count_1h"},
    {"check": "no_negative", "column": "txn_count_24h"},
    {"check": "range", "column": "velocity_ratio", "min": 0, "max": 100},
]


def validate_fraud_raw(df: pd.DataFrame) -> QualityResult:
    """Validate raw fraud transactions before velocity feature computation."""
    return _run_expectations(df, FRAUD_RAW_EXPECTATIONS, "fraud_raw_transactions", "fraud")


def validate_fraud_features(df: pd.DataFrame) -> QualityResult:
    """Validate computed velocity features before training."""
    return _run_expectations(df, FRAUD_FEATURES_EXPECTATIONS, "fraud_velocity_features", "fraud")


# ── Demand Forecasting Quality Gates ───────────────────────────────────────────

DEMAND_RAW_EXPECTATIONS = [
    {"check": "min_rows", "count": 500},
    {"check": "no_nulls", "column": "product_id"},
    {"check": "no_nulls", "column": "quantity_sold"},
    {"check": "no_nulls", "column": "date"},
    {"check": "no_negative", "column": "quantity_sold"},
    {"check": "no_negative", "column": "revenue"},
]

DEMAND_FEATURES_EXPECTATIONS = [
    {"check": "min_rows", "count": 500},
    {"check": "no_nulls", "column": "product_id"},
    {"check": "no_nulls", "column": "quantity_sold"},
    {"check": "no_negative", "column": "quantity_sold"},
]


def validate_demand_raw(df: pd.DataFrame) -> QualityResult:
    """Validate raw demand data after ETL extract."""
    return _run_expectations(df, DEMAND_RAW_EXPECTATIONS, "demand_raw_sales", "demand")


def validate_demand_features(df: pd.DataFrame) -> QualityResult:
    """Validate demand features before training."""
    return _run_expectations(df, DEMAND_FEATURES_EXPECTATIONS, "demand_features", "demand")
