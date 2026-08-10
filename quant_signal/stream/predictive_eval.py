"""Progressive validation + promotion gate for the online predictor.

Why this exists
---------------
The research literature is blunt about the alternative. A single-asset,
next-window-return prediction with a handful of technical features is the
*weakest* return-prediction setting — Gu, Kelly & Xiu (2020) and follow-ups show
short-horizon ML return forecasts collapse to ~zero net of transaction costs;
López de Prado calls naive backtesting "the most dangerous tool in finance";
multiple-testing work (Bailey & López de Prado; TrustedQuant 2026) shows a
56k-trial search produces max Sharpe ~4-11 from pure noise. A model that isn't
evaluated this way is not a model — it's a guess wearing a number.

So before the live ``OnlinePredictor`` is allowed to emit LONG/SHORT, we replay
its exact predict-before-learn loop over the historical feature stream and
score it honestly:

  - **progressive validation** (test-then-train, River's canonical online
    protocol) — every prediction is scored before its label is learned;
  - **transaction-cost-adjusted P&L** (taker cost per position flip) vs
    buy-and-hold — the "does it survive real fills" check;
  - **Information Coefficient (IC)** between predicted and realized returns;
  - **naive baselines**: predict-zero and persistence, reported as MASE — a
    model that can't beat "predict nothing" has no edge;
  - **conformal coverage** vs nominal alpha (the ACI contract);
  - **stability across contiguous blocks** (signal must not be one lucky
    period — Man Group / NordVarg gate);
  - **Deflated Sharpe Ratio (DSR)** charged for the number of trials the model
    has gone through — the multiple-testing correction.

``passes_gate()`` is the promotion predicate. Until the live predictor clears
it, it may learn but must not trade.

This module is pure (no I/O): the replay loop and the gate take/return plain
dicts so the suite is hermetic and the API/predictor call it directly.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from scipy import stats

from config.settings import Settings
from stream.predictor import ConformalInterval, _direction, _features, _model


def _windows_per_year(sorted_windows: Sequence[Mapping]) -> float:
    """Annualization factor derived from the data's own cadence.

    The 5m-era constant (365×24×12) silently mis-annualized the Sharpe once
    the pipeline moved to 1h windows. Instead of hardcoding a cadence, read
    it from the median spacing between consecutive window closes: hourly data
    → ~8 760 windows/year, 5m → ~105 120. The median (not mean) is robust to
    a single long gap (downtime, a dropped window); unknown cadence defaults
    to hourly.
    """
    ends = [int(w["window_end_ms"]) for w in sorted_windows if w.get("window_end_ms") is not None]
    deltas = [b - a for a, b in zip(ends, ends[1:]) if b > a]
    if not deltas:
        return 365.0 * 24.0
    median_ms = statistics.median(deltas)
    if median_ms <= 0:
        return 365.0 * 24.0
    return 365.0 * 24.0 * 3_600_000.0 / median_ms


def _cross_returns(
    cross_windows: Mapping[str, Sequence[Mapping]],
    symbol: str,
    window_end: int,
) -> dict[str, float]:
    """Lagged returns of each cross symbol from windows ending strictly before
    ``window_end`` — the same rule the live predictor applies online.

    For each cross symbol take its last two closes whose windows ended before
    this one → a realized return the model would actually know at decision
    time (no lookahead). Warm-up pairs are omitted; the sign is deliberately
    NOT hardcoded (the seesaw vs positive-spillover literature disagrees —
    see ``stream/predictor.py``).
    """
    returns: dict[str, float] = {}
    for cross, windows in cross_windows.items():
        if cross == symbol:
            continue
        closes = [
            (int(w["window_end_ms"]), float(w["close"]))
            for w in windows
            if w.get("window_end_ms") is not None and isinstance(w.get("close"), (int, float))
        ]
        closes = [h for h in closes if h[0] < window_end]
        if len(closes) < 2:
            continue
        older, newer = closes[-2], closes[-1]
        if older[0] >= newer[0] or newer[1] == 0:
            continue
        returns[cross] = newer[1] / older[1] - 1.0
    return returns


def _spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, as a plain float.

    ``scipy.stats.spearmanr`` is wrapped in a dynamic decorator that type
    checkers cannot resolve (its return type collapses to the opaque ``_``),
    but the documented API is the ``SignificanceResult.statistic`` attribute
    (scipy >= 1.9). Reach it through an explicit cast so the intent stays
    visible instead of burying a type-ignore on every call site.
    """
    res = stats.spearmanr(x, y)
    return float(cast(Any, res).statistic)


def _mase_skill(model_mae: float, baseline_mae: float) -> float | None:
    """1 - MAE_model/MAE_baseline: 1.0 = perfect, 0 = no edge, <0 = worse."""
    if not baseline_mae or math.isclose(baseline_mae, 0.0):
        return None
    return 1.0 - model_mae / baseline_mae


def _dsr(strat_returns: list[float], n_trials: int) -> float | None:
    """Deflated Sharpe (Bailey & López de Prado) over per-window returns.

    Charges the null with the expected best Sharpe of ``n_trials`` skill-less
    strategies before declaring the observed Sharpe significant.
    """
    n = len(strat_returns)
    if n < 3 or n_trials < 1:
        return None
    arr = np.asarray(strat_returns, dtype=float)
    std = float(np.std(arr))
    # Zero/near-zero variance makes the Sharpe and the moment-based variance
    # degenerate (scipy returns NaN on near-identical data); treat as unknown
    # rather than letting NaN comparisons silently pass the gate.
    if not math.isfinite(std) or std <= 1e-12:
        return None
    sr = float(np.mean(arr)) / std
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))
    se2 = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    if not math.isfinite(se2) or se2 <= 0:
        return None
    se = math.sqrt(se2)
    # Expected max of N iid standard normals (Gumbel approximation).
    if n_trials > 1:
        euler = 0.5772156649015329
        sr0 = (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials) + euler * stats.norm.ppf(
            1 - 1.0 / (n_trials * math.e)
        )
        sr0 *= se
    else:
        sr0 = 0.0
    return float(stats.norm.cdf((sr - sr0) / (se + 1e-12)))


def evaluate_predictor(
    windows: Sequence[Mapping],
    *,
    alpha: float = 0.1,
    gamma: float = 0.005,
    residual_window: int = 200,
    taker_cost: float = 0.0005,
    n_blocks: int = 4,
    direction_threshold: float = 0.0005,
    cross_windows: Mapping[str, Sequence[Mapping]] | None = None,
) -> dict:
    """Progressive validation of the live predictor over historical windows.

    ``windows`` are the same feature-window dicts the predictor consumes (each
    with ``symbol, close, open, high, low, vwap, volume, bar_count,
    window_end_ms``), oldest-first. ``cross_windows`` maps each cross symbol to
    its own window history; when given, the replay feeds each window the lagged
    cross-coin returns the live predictor uses, so the offline gate tests the
    exact feature set the online model sees. Returns a plain report dict.
    """
    sorted_windows = sorted(
        (w for w in windows if w.get("window_end_ms") is not None),
        key=lambda w: w["window_end_ms"],
    )
    if not sorted_windows:
        return {"n_windows": 0}

    model = _model()
    conformal = ConformalInterval(alpha=alpha, gamma=gamma, residual_window=residual_window)

    preds: list[float] = []
    realized_list: list[float] = []
    covers: list[bool] = []
    directions: list[str] = []
    strat_rets: list[float] = []
    buyhold_rets: list[float] = []

    last_features: dict | None = None
    last_close: float | None = None
    last_y_hat: float | None = None
    last_interval: tuple[float, float] | None = None
    last_direction: str | None = None
    position: str = "FLAT"

    for msg in sorted_windows:
        cross = (
            _cross_returns(
                cross_windows,
                str(msg.get("symbol") or "").upper(),
                int(msg["window_end_ms"]),
            )
            if cross_windows
            else None
        )
        features = _features(msg, cross)
        if features is None:
            continue
        close = float(msg["close"])
        if last_features is not None and last_close and close != 0 and last_y_hat is not None:
            realized = close / last_close - 1.0
            model.learn_one(last_features, realized)
            if last_interval is not None:
                conformal.update(realized, last_y_hat, last_interval)
                covers.append(last_interval[0] <= realized <= last_interval[1])
            preds.append(last_y_hat)
            realized_list.append(realized)
            directions.append(last_direction or "FLAT")
            buyhold_rets.append(realized)
            ret = 0.0
            if last_direction == "LONG":
                ret = realized
            elif last_direction == "SHORT":
                ret = -realized
            if last_direction in ("LONG", "SHORT") and last_direction != position:
                ret -= taker_cost  # only on a position open/flip, not per window held
            strat_rets.append(ret)
            position = last_direction if last_direction in ("LONG", "SHORT") else "FLAT"

        y_hat = float(model.predict_one(features))
        interval = conformal.predict(y_hat)
        direction = _direction(y_hat, direction_threshold)
        last_features, last_close = features, close
        last_y_hat, last_interval, last_direction = y_hat, interval, direction

    if len(preds) < 3:
        return {
            "n_windows": len(sorted_windows),
            "n_scored": len(preds),
        }

    errs = [abs(y - y_hat) for y, y_hat in zip(realized_list, preds)]
    mae = statistics.fmean(errs)
    rmse = math.sqrt(statistics.fmean(e * e for e in errs))
    mae_zero = statistics.fmean(abs(y) for y in realized_list)
    # Persistence baseline: predict the previous realized return.
    pers_mae = (
        statistics.fmean(abs(y - prev) for prev, y in zip(realized_list, realized_list[1:]))
        if len(realized_list) > 2
        else mae_zero
    )

    ic = _spearmanr(preds, realized_list)
    directional = sum(
        1
        for d, y in zip(directions, realized_list)
        if (d == "LONG" and y > 0) or (d == "SHORT" and y < 0)
    )
    dir_acc = directional / len(directions) if directions else 0.0

    eq_buyhold = float(np.prod([1.0 + r for r in buyhold_rets]))
    eq_strategy = float(np.prod([1.0 + r for r in strat_rets]))

    # Contiguous-block stability: per-block MAE and IC must not be one lucky
    # period. Blocks may be tiny — use at least one full block.
    block_size = max(1, len(preds) // max(1, n_blocks))
    blocks = []
    for i in range(0, len(preds), block_size):
        bp = preds[i : i + block_size]
        br = realized_list[i : i + block_size]
        if len(bp) < 2:
            continue
        b_mae = statistics.fmean(abs(a - b) for a, b in zip(br, bp))
        blocks.append(
            {
                "start": i,
                "end": i + len(bp) - 1,
                "mae": round(b_mae, 6),
                "ic": round(_spearmanr(bp, br), 4),
            }
        )

    coverage = statistics.fmean(1.0 if c else 0.0 for c in covers) if covers else None

    strat_sharpe = (
        statistics.fmean(strat_rets)
        / (statistics.pstdev(strat_rets) + 1e-12)
        * math.sqrt(_windows_per_year(sorted_windows))
        if strat_rets
        else None
    )

    return {
        "n_windows": len(sorted_windows),
        "n_scored": len(preds),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mae_zero": round(mae_zero, 6),
        "mae_persistence": round(pers_mae, 6),
        "skill_vs_zero": _mase_skill(mae, mae_zero),
        "skill_vs_persistence": _mase_skill(mae, pers_mae),
        "ic": round(ic, 4),
        "direction_accuracy": round(dir_acc, 4),
        "coverage": round(coverage, 4) if coverage is not None else None,
        "nominal_alpha": alpha,
        "total_return_strategy": round(eq_strategy - 1.0, 6),
        "total_return_buyhold": round(eq_buyhold - 1.0, 6),
        "excess_return": round(eq_strategy - eq_buyhold, 6),
        "annual_sharpe_strategy": round(strat_sharpe, 4) if strat_sharpe else None,
        "taker_cost": taker_cost,
        "blocks": blocks,
        # Raw per-window strategy returns — consumed by passes_gate() for the
        # Deflated Sharpe. Kept on the report so the gate needs no re-derivation.
        "_strat_rets": strat_rets,
        "passes": None,  # set by passes_gate()
    }


def passes_gate(report: dict, settings: Settings, *, n_trials: int = 1) -> tuple[bool, list[str]]:
    """Promotion predicate: is this model cleared to emit directions?

    Every check must pass (the "validation gate" the literature requires):
      - enough scored windows,
      - model beats both naive baselines (positive skill),
      - IC above the floor and direction accuracy above 50%,
      - conformal coverage near nominal,
      - strategy clears buy-and-hold after transaction costs,
      - DSR (charged for n_trials) above the significance floor.

    Also records the verdict and Deflated Sharpe back onto the report
    (``report["passes"]`` / ``report["deflated_sharpe"]``).
    """
    failures: list[str] = []

    n = report.get("n_scored", 0)
    if n < settings.stream_gate_min_windows:
        failures.append(f"n_scored {n} < min {settings.stream_gate_min_windows} (warm-up)")
    for label, key in (
        ("vs zero", "skill_vs_zero"),
        ("vs persistence", "skill_vs_persistence"),
    ):
        skill = report.get(key)
        if skill is None or skill <= settings.stream_gate_min_skill:
            failures.append(f"no skill {label} (skill={skill})")
    if report.get("ic", 0.0) <= settings.stream_gate_min_ic:
        failures.append(f"ic {report.get('ic')} <= min {settings.stream_gate_min_ic}")
    if report.get("direction_accuracy", 0.0) <= settings.stream_gate_min_direction_accuracy:
        failures.append(
            f"direction_accuracy {report.get('direction_accuracy')} <= "
            f"{settings.stream_gate_min_direction_accuracy}"
        )
    cov = report.get("coverage")
    if (
        cov is None
        or abs(cov - (1.0 - report.get("nominal_alpha", 0.1))) > settings.stream_gate_coverage_tol
    ):
        failures.append(f"coverage {cov} off nominal (tol {settings.stream_gate_coverage_tol})")
    if report.get("excess_return", 0.0) <= 0.0:
        failures.append(f"excess_return {report.get('excess_return')} <= 0 after costs")

    strat_rets = report.get("_strat_rets")
    if strat_rets is None or len(strat_rets) < 3:
        failures.append("deflated_sharpe not computable (missing strategy returns)")
        report["deflated_sharpe"] = None
    else:
        dsr = _dsr(strat_rets, n_trials)
        report["deflated_sharpe"] = dsr
        if dsr is None or dsr <= settings.stream_gate_min_dsr:
            failures.append(f"deflated_sharpe {dsr} <= {settings.stream_gate_min_dsr}")

    report["passes"] = not failures
    return (not failures, failures)
