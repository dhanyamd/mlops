"""Data drift and model monitoring utilities (Day 5 concepts)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftReport:
    """Summary of drift detection results."""

    feature: str
    statistic: float
    p_value: float
    drifted: bool
    method: str


def detect_univariate_drift(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.05,
    method: str = "ks",
) -> DriftReport:
    """
    Detect univariate drift using KS test or PSI.

    Methods from ML Academy Day 5:
    - ks: Kolmogorov-Smirnov test
    - psi: Population Stability Index
    """
    ref = reference.dropna().values
    cur = current.dropna().values

    if method == "ks":
        statistic, p_value = stats.ks_2samp(ref, cur)
        drifted = p_value < threshold
        return DriftReport(
            feature=reference.name or "unknown",
            statistic=float(statistic),
            p_value=float(p_value),
            drifted=drifted,
            method="kolmogorov_smirnov",
        )

    # PSI (Population Stability Index)
    bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=10)
    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)
    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    drifted = psi > 0.2  # common PSI threshold
    return DriftReport(
        feature=reference.name or "unknown",
        statistic=psi,
        p_value=0.0,
        drifted=drifted,
        method="population_stability_index",
    )


def detect_dataset_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    threshold: float = 0.05,
) -> list[DriftReport]:
    """Run univariate drift detection across all numeric features."""
    cols = numeric_columns or list(reference.select_dtypes(include="number").columns)
    reports = []
    for col in cols:
        if col not in current.columns:
            continue
        ref = reference[col].copy()
        ref.name = col
        cur = current[col].copy()
        cur.name = col
        reports.append(detect_univariate_drift(ref, cur, threshold=threshold))
    return reports


def should_retrain(reports: list[DriftReport], min_drifted: int = 2) -> bool:
    """Drift-based retraining trigger (orchestration decision logic)."""
    drifted_count = sum(1 for r in reports if r.drifted)
    return drifted_count >= min_drifted
