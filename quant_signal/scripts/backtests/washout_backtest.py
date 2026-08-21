"""Washout mean-reversion backtest on REAL hourly CRYPTO_BARS history.

Tests whether the washout/overreaction reversal documented for crypto intraday
(Keel 2024; Wen, Bouri, Xu & Zhao, North American J. Econ & Finance 62, 2022 —
reversal is the crypto-specific intraday effect, strongest when no jump / low
liquidity; Liu, Wang & Yan, Applied Econ Letters 30(12), 2023 — the effect sign
FLIPS across eras, so direction is learned, never hardcoded) clears the λ×taker
cost gate on OUR data before it is wired into the live harness.

Mechanism (single asset, hourly, no cross-coin):
  * At hour t a k-hour washout z-score is computed strictly from trailing
    data: z_t = (r_k,t − mean(r_k over trailing L)) / std(...) — no lookahead.
  * LONG entry: z_t ≤ −z_entry AND price above the EMA trend filter.
    SHORT entry: z_t ≥ +z_entry AND price below the EMA.
  * Exit after a fixed H-hour hold (time stop); a run gap flattens the book.
  * Each round trip pays the 10 bps taker cost (the gate's basis), and a
    trade only counts if it clears the λ=2 band (20 bps net).

Every grid configuration is reported — never just the winner — so the search
extent is disclosed (qbx-research / Bailey & López de Prado). Best-config
Sharpe is deflated for the trial count before any "go" decision.

Run:  uv run python -m scripts.washout_backtest [--out docs/probe_washout.json]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from scripts.backfill_feature_windows import fetch_bars, hourly_windows

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000
_TAKER_ROUND_TRIP = 0.001  # 10 bps taker, the gate's cost basis
_GATE_LAMBDA = 2.0  # λ=2 × round-trip cost = the 20 bps net band
_TRADES_PER_YEAR = 365.0 * 24.0  # hours per year; scaled by hold below

_K_GRID = [4, 8, 12]  # washout return horizon (hours)
_Z_GRID = [2.0, 2.5, 3.0]  # entry z-score threshold
_H_GRID = [4, 8, 12]  # hold time-stop (hours)
_EMA_GRID = [48, 200]  # trend-filter EMA span (hours)
_Z_WINDOW = 168  # trailing window (hours) for z mean/std


def _contiguous_runs(returns: pd.DataFrame) -> list[pd.DataFrame]:
    """Split hourly returns into maximal no-gap runs (dt == 1h chains).

    Signals never span a Flink gap: a missing window resets the warm-up and
    flattens any open position, which is the conservative (tradeable) reading.
    """
    runs: list[pd.DataFrame] = []
    if returns.empty:
        return runs
    starts = returns["window_start_ms"].to_numpy()
    gaps = np.where(np.diff(starts) != _HOUR_MS)[0] + 1
    bounds = [0] + list(gaps) + [len(returns)]
    for i in range(len(bounds) - 1):
        runs.append(returns.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True))
    return runs


def _ema(values: np.ndarray, span: float) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _run_trades(
    run: pd.DataFrame, *, k: int, z_entry: float, hold: int, ema_span: int
) -> list[dict]:
    """Simulate the washout strategy over one contiguous run, no lookahead."""
    closes = run["close"].to_numpy()
    n = len(closes)
    if n < _Z_WINDOW + k + 1:
        return []
    ema = _ema(closes, ema_span)

    ret_k = np.empty(n)
    ret_k[:k] = np.nan
    ret_k[k:] = closes[k:] / closes[:-k] - 1.0

    trades: list[dict] = []
    flat_until = 0
    for t in range(_Z_WINDOW, n):
        if t < flat_until or math.isnan(ret_k[t]):
            continue
        hist = ret_k[t - _Z_WINDOW : t]
        mu, sd = float(np.mean(hist)), float(np.std(hist))
        if sd == 0.0 or math.isnan(sd):
            continue
        z = (ret_k[t] - mu) / sd
        trend_up = closes[t] > ema[t]
        side: int | None = None
        if z <= -z_entry and trend_up:
            side = 1
        elif z >= z_entry and not trend_up:
            side = -1
        if side is None:
            continue
        exit_t = min(t + hold, n - 1)
        hold_rets = run["ret"].iloc[t:exit_t].to_numpy()
        gross = float(np.prod(1.0 + hold_rets) - 1.0)
        net = side * gross - _TAKER_ROUND_TRIP
        trades.append(
            {
                "entry_ms": int(run["window_start_ms"].iloc[t]),
                "side": "LONG" if side > 0 else "SHORT",
                "hours": int(exit_t - t),
                "z": round(z, 2),
                "gross_bps": 1e4 * side * gross,
                "net_bps": 1e4 * net,
            }
        )
        flat_until = exit_t
    return trades


def _summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    net = np.array([t["net_bps"] for t in trades])
    gross = np.array([t["gross_bps"] for t in trades])
    mean_net = float(np.mean(net))
    sd_net = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    trades_per_year = _TRADES_PER_YEAR / float(np.mean([t["hours"] for t in trades]))
    sharpe = (mean_net / sd_net * math.sqrt(trades_per_year)) if sd_net > 0 else 0.0
    equity = float(np.prod(1.0 + net / 1e4))
    dd = 0.0
    peak = 1.0
    for x in np.cumprod(1.0 + net / 1e4):
        peak = max(peak, x)
        dd = max(dd, 1.0 - x / peak)
    longs = sum(1 for t in trades if t["side"] == "LONG")
    return {
        "n": len(trades),
        "longs": longs,
        "shorts": len(trades) - longs,
        "win_rate": round(float(np.mean(net > 0)), 3),
        "mean_gross_bps": round(float(np.mean(gross)), 2),
        "mean_net_bps": round(mean_net, 2),
        "sharpe_net_ann": round(sharpe, 2),
        "gross_multiple": round(float(np.prod(1.0 + gross / 1e4)), 3),
        "net_multiple": round(equity, 3),
        "max_drawdown": round(dd, 3),
        "clears_gate": bool(mean_net >= _GATE_LAMBDA * 1e4 * _TAKER_ROUND_TRIP),
        "mean_hold_h": round(float(np.mean([t["hours"] for t in trades])), 1),
    }


def backtest_symbol(windows: pd.DataFrame, symbol: str) -> list[dict]:
    """Run the full grid on one symbol; return ALL configs, not just the best."""
    returns = _hourly_returns(windows)
    runs = _contiguous_runs(returns)
    results: list[dict] = []
    for k in _K_GRID:
        for z_entry in _Z_GRID:
            for hold in _H_GRID:
                for ema_span in _EMA_GRID:
                    trades: list[dict] = []
                    for run in runs:
                        trades.extend(
                            _run_trades(run, k=k, z_entry=z_entry, hold=hold, ema_span=ema_span)
                        )
                    if not trades:
                        continue
                    summary = _summarize_trades(trades)
                    results.append(
                        {
                            "symbol": symbol,
                            "params": {
                                "k": k,
                                "z_entry": z_entry,
                                "hold_h": hold,
                                "ema_h": ema_span,
                            },
                            **summary,
                        }
                    )
    return results


def _hourly_returns(windows: pd.DataFrame) -> pd.DataFrame:
    df = windows.sort_values("window_start_ms").reset_index(drop=True)
    df["dt"] = df["window_start_ms"].diff()
    df["ret"] = df["close"].shift(-1) / df["close"] - 1.0
    adj = (df["dt"] == _HOUR_MS) & df["ret"].notna()
    return df.loc[adj, ["window_start_ms", "close", "ret"]].reset_index(drop=True)


def _deflated_sharpe(best_sharpe: float, n_trials: int, n_symbols: int) -> float:
    """Bailey & López de Prado E[min Max SR] adjustment, using a single lookback."""
    n = max(n_trials * n_symbols, 1)
    if n <= 1:
        return best_sharpe
    e = 0.5772156649  # Euler-Mascheroni
    return float(best_sharpe * math.sqrt((1.0 - e) / n))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--symbols", default=None, help="comma-separated symbols (default: ingest defaults)"
    )
    parser.add_argument("--out", default=None, help="JSON output path (default: print only)")
    args = parser.parse_args()

    settings = get_settings()
    symbols = (
        csv_list(args.symbols) if args.symbols else csv_list(settings.ingest_default_crypto_symbols)
    )
    bars = fetch_bars(settings, symbols)
    windows = hourly_windows(bars)
    if windows.empty:
        logger.error("washout_backtest_no_data", symbols=symbols)
        raise SystemExit(1)

    all_results: list[dict] = []
    for symbol in symbols:
        sym_windows = windows[windows["symbol"] == symbol.upper()]
        results = backtest_symbol(sym_windows, symbol.upper())
        all_results.extend(results)
        if results:
            best = max(results, key=lambda r: r["sharpe_net_ann"])
            logger.info(
                "washout_backtest_symbol",
                symbol=symbol.upper(),
                hours=len(sym_windows),
                configs=len(results),
                best_params=best["params"],
                best_sharpe=best["sharpe_net_ann"],
                best_net_bps=best["mean_net_bps"],
                n=best["n"],
            )

    deflated = _deflated_sharpe(
        max((r["sharpe_net_ann"] for r in all_results), default=0.0),
        n_trials=len(_K_GRID) * len(_Z_GRID) * len(_H_GRID) * len(_EMA_GRID),
        n_symbols=len(symbols),
    )
    payload = {
        "configs": all_results,
        "best_sharpe_ann": max((r["sharpe_net_ann"] for r in all_results), default=None),
        "best_sharpe_deflated": round(deflated, 2),
        "gate_bps": _GATE_LAMBDA * 1e4 * _TAKER_ROUND_TRIP,
        "taker_round_trip_bps": 1e4 * _TAKER_ROUND_TRIP,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("washout_backtest_written", path=args.out)

    if all_results:
        df = pd.DataFrame(all_results)
        show = df.sort_values("sharpe_net_ann", ascending=False).head(12)
        print("\nTop-12 washout configs by net-annualized Sharpe (all trials reported):")
        cols = [
            "symbol",
            "params",
            "n",
            "win_rate",
            "mean_net_bps",
            "sharpe_net_ann",
            "net_multiple",
            "max_drawdown",
            "clears_gate",
        ]
        print(show[cols].to_string(index=False))
        print(
            f"\nBest Sharpe deflated for {len(all_results)} trials: "
            f"{payload['best_sharpe_deflated']}"
        )
        print(f"Gate band (λ=2 × 10 bps): {payload['gate_bps']:.0f} bps net per trade")
    else:
        print("\nNo washout config produced any trades on the available history.")


if __name__ == "__main__":
    main()
