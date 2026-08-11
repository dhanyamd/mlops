"""30-minute intraday-timing probe on REAL BRONZE CRYPTO_BARS minute history.

Tests two single-asset mechanisms at 30m resolution where the samples are real
(185 days x 48 half-hours ~ 8.9k bars per symbol, vs the 2-12-trade 1h washout):

A) Shen/Urquhart/Wang (Financial Review 57(2) 2022, Tianjin Univ.) timing
   structure: the FIRST half-hour (ONFH) and the SECOND-TO-LAST half-hour (SLH)
   predict the LAST half-hour of a session. Paper: Sharpe 1.72, 16.7%/yr on
   SPX half-hours, mechanism = liquidity provision + disposition effect.
   Crypto has no single institutional "session", so we let the DATA pick the
   close slot: for each of the 48 UTC half-hour slots s we define a 24h session
   ending at s and test Last_s = a + b1*First_s + b2*SLH_s across days. Every
   slot is reported (the search extent is disclosed, never just the winner).

B) Washout mean-reversion at 30m (the 1h version traded 2-12 times in 6
   months — underpowered; 30m quadruples the sample). Reversal is the
   crypto-specific intraday effect (Wen, Bouri, Xu & Zhao, N. Am. J. Econ.
   & Finance 62, 2022). Direction is learned, never hardcoded (Liu, Wang &
   Yan, Applied Econ Letters 30(12), 2023: the sign FLIPS across eras).

Both are net of the 10 bps taker round trip; a config only counts as passing
if mean net per trade clears the lambda=2 x 10 bps = 20 bps band.

Run:  uv run python -m scripts.intraday_30m_probe [--out docs/probe_30m.json]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from scripts.backfill_feature_windows import fetch_bars

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_30M_MS = 1_800_000
_DAY_MS = 86_400_000
_SLOTS_PER_DAY = 48
_TAKER_ROUND_TRIP = 0.001  # 10 bps taker, the gate's cost basis
_GATE_LAMBDA = 2.0  # x round-trip cost = 20 bps net band

# Washout grid at 30m: return horizon k (bars), entry z, hold H (bars), EMA span (bars).
_K_GRID = [2, 4, 8]  # 1h, 2h, 4h
_Z_GRID = [2.0, 2.5, 3.0]
_H_GRID = [2, 4, 8]
_EMA_GRID = [96, 400]  # 48h, ~8.3 days (Keel's 200h daily analog)

# Regime-Gated Washout (RGW) grid + regime window. The own-asset vol regime
# (24h RV vs its 7d baseline) decides fade-vs-ride on the SAME washout signal.
_RGW_K = [2, 4]
_RGW_Z = [2.0, 2.5]
_RGW_H = [2, 4]
_RGW_EMA = [96, 400]
_WF_SPLIT = 0.6  # walk-forward: first 60% of events learn, last 40% are OOS
_RV_WIN = 48  # 24h of 30m bars
_RV_BASELINE = 336  # 7d of 30m bars


def _aggregate(bars: pd.DataFrame, bucket_ms: int) -> pd.DataFrame:
    """TUMBLE OHLCV over minute bars per symbol; drops in-progress buckets."""
    df = bars.copy()
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["bucket"] = df["ts_ms"] // bucket_ms
    df["wv"] = df["close"] * df["volume"]
    agg = df.groupby(["symbol", "bucket"], sort=True, as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg["window_start_ms"] = agg["bucket"] * bucket_ms
    agg["window_end_ms"] = agg["window_start_ms"] + bucket_ms
    vwap = df.groupby(["symbol", "bucket"])["wv"].sum()
    vol = df.groupby(["symbol", "bucket"])["volume"].sum()
    agg["vwap"] = (vwap / vol).astype(float).values
    return agg.sort_values(["symbol", "window_start_ms"]).reset_index(drop=True)


def _slot_returns(bars30m: pd.DataFrame) -> pd.DataFrame:
    """Per-slot returns across adjacent 30m bars only (no gap leakage)."""
    df = bars30m.sort_values("window_start_ms").reset_index(drop=True)
    df["slot"] = (df["window_start_ms"] % _DAY_MS) // _30M_MS
    df["dt"] = df["window_start_ms"].diff()
    df["ret"] = df["close"].shift(-1) / df["close"] - 1.0
    df["next_slot"] = df["slot"].shift(-1)
    adj = (df["dt"] == _30M_MS) & df["ret"].notna()
    return df.loc[adj, ["window_start_ms", "slot", "close", "ret"]].reset_index(drop=True)


def _shen_slot(results: pd.DataFrame, slot: int) -> dict:
    """Shen structure for one session boundary: First/SLH predict Last.

    Session = the 48 slots ending at ``slot``. First = slot (s-47), SLH =
    slot (s-1), Last = slot s. Alignment uses the ABSOLUTE 30m slot index
    (ts // 30m) so a session that wraps a UTC day boundary lines up correctly
    and a missing bar (gap) simply drops that day. Pooled across all sessions.
    """
    df = results.copy()
    df["abs_slot"] = df["window_start_ms"] // _30M_MS
    rets_s = df.set_index("abs_slot")["ret"]
    last = df.loc[df["abs_slot"] % _SLOTS_PER_DAY == slot].set_index("abs_slot")["ret"]
    if last.empty:
        return {"slot": slot, "n_days": 0}
    first = rets_s.reindex(last.index - (_SLOTS_PER_DAY - 1)).to_numpy()
    slh = rets_s.reindex(last.index - 1).to_numpy()
    joined = pd.DataFrame(
        {"first": first, "slh": slh, "last": last.to_numpy()}, index=last.index
    ).dropna()
    if len(joined) < 30:
        return {"slot": slot, "n_days": int(len(joined))}
    first_arr = joined["first"].to_numpy()
    y = joined["last"].to_numpy()
    n = len(y)
    x = np.column_stack((np.ones(n), first_arr, joined["slh"].to_numpy()))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sigma2 = float(resid @ resid) / (n - 3)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    b1, b2 = float(beta[1]), float(beta[2])
    r2 = 1.0 - float(resid.var()) / float(y.var())
    # Timing rule (paper): long if First>0 & SLH<0, short if First<0 & SLH>0, else flat.
    rule = np.where(
        (joined["first"] > 0) & (joined["slh"] < 0),
        1.0,
        np.where((joined["first"] < 0) & (joined["slh"] > 0), -1.0, 0.0),
    )
    raw = rule * y
    gross = float(np.mean(raw))
    se_rule = float(np.std(raw, ddof=1) / math.sqrt(len(raw)))
    sharpe = (gross / se_rule * math.sqrt(_SLOTS_PER_DAY * 365.0)) if se_rule > 0 else 0.0
    return {
        "slot": slot,
        "n_days": int(len(joined)),
        "b1_first": round(b1, 4),
        "b2_slh": round(b2, 4),
        "se_b1": round(float(np.sqrt(cov[1, 1])), 4),
        "r2": round(r2, 4),
        "rule_gross_bps": round(1e4 * gross, 2),
        "rule_net_bps": round(1e4 * (gross - _TAKER_ROUND_TRIP), 2),
        "rule_sharpe_ann": round(sharpe, 2),
        "rule_activity": float(np.mean(rule != 0)),
    }


def _contiguous_runs(returns: pd.DataFrame) -> list[pd.DataFrame]:
    starts = returns["window_start_ms"].to_numpy()
    gaps = np.where(np.diff(starts) != _30M_MS)[0] + 1
    bounds = [0] + list(gaps) + [len(returns)]
    runs: list[pd.DataFrame] = []
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


def _washout_config(
    results: pd.DataFrame, *, k: int, z_entry: float, hold: int, ema_span: int
) -> dict:
    trades: list[dict] = []
    for run in _contiguous_runs(results):
        closes = run["close"].to_numpy()
        n = len(closes)
        if n < 336 + k + 1:
            continue
        ema = _ema(closes, ema_span)
        ret_k = np.empty(n)
        ret_k[:k] = np.nan
        ret_k[k:] = closes[k:] / closes[:-k] - 1.0
        flat_until = 0
        for t in range(336, n):
            if t < flat_until or math.isnan(ret_k[t]):
                continue
            hist = ret_k[t - 336 : t]
            mu, sd = float(np.mean(hist)), float(np.std(hist))
            if sd == 0.0 or math.isnan(sd):
                continue
            z = (ret_k[t] - mu) / sd
            trend_up = closes[t] > ema[t]
            side = (
                1
                if (z <= -z_entry and trend_up)
                else (-1 if (z >= z_entry and not trend_up) else None)
            )
            if side is None:
                continue
            exit_t = min(t + hold, n - 1)
            hold_rets = run["ret"].iloc[t:exit_t].to_numpy()
            gross = float(np.prod(1.0 + hold_rets) - 1.0)
            trades.append({"gross": side * gross, "net": side * gross - _TAKER_ROUND_TRIP})
            flat_until = exit_t
    if not trades:
        return {"k": k, "z_entry": z_entry, "hold_h": hold, "ema_h": ema_span, "n": 0}
    net = np.array([t["net"] for t in trades])
    gross = np.array([t["gross"] for t in trades])
    mean_net = float(np.mean(net))
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    trades_per_year = 365.0 * _SLOTS_PER_DAY / hold
    sharpe = (mean_net / sd * math.sqrt(trades_per_year)) if sd > 0 else 0.0
    return {
        "k": k,
        "z_entry": z_entry,
        "hold_h": hold,
        "ema_h": ema_span,
        "n": len(trades),
        "win_rate": round(float(np.mean(net > 0)), 3),
        "mean_gross_bps": round(1e4 * float(np.mean(gross)), 2),
        "mean_net_bps": round(1e4 * mean_net, 2),
        "sharpe_net_ann": round(sharpe, 2),
        "net_multiple": round(float(np.prod(1.0 + net)), 3),
        "clears_gate": bool(mean_net >= _GATE_LAMBDA * _TAKER_ROUND_TRIP),
    }


def _run_regime_trades(
    run: pd.DataFrame, *, k: int, z_entry: float, hold: int, ema_span: int
) -> list[dict]:
    """RGW over one contiguous run. Calm regime fades the washout (reversal /
    liquidity provision); stress regime rides it (momentum continuation). The
    regime is the own-asset 24h realized vol vs its rolling 7d baseline — no
    cross-coin, no lookahead. Direction is learned per regime, never hardcoded.
    """
    closes = run["close"].to_numpy()
    n = len(closes)
    if n < _RV_BASELINE + k + 1:
        return []
    ema = _ema(closes, ema_span)
    log_ret = np.log(closes[1:] / closes[:-1])
    ret_k = np.empty(n)
    ret_k[:k] = np.nan
    ret_k[k:] = closes[k:] / closes[:-k] - 1.0

    # Vol-shock regime, vectorized once: 24h RV known at bar t divided by its
    # trailing 7d mean. RV at t uses only log-rets < t (shift(1) closes the
    # window before t) so there is no lookahead.
    lr = pd.Series(log_ret)
    rv = lr.shift(1).rolling(_RV_WIN).std()
    baseline = rv.shift(1).rolling(_RV_BASELINE).mean()
    shock = (rv / baseline).to_numpy()  # NaN wherever not enough history
    shock = np.where(np.isfinite(shock), shock, 1.0)

    trades: list[dict] = []
    flat_until = 0
    for t in range(_RV_BASELINE, n):
        if t < flat_until or math.isnan(ret_k[t]):
            continue
        hist = ret_k[t - _RV_BASELINE : t]
        mu, sd = float(np.mean(hist)), float(np.std(hist))
        if sd == 0.0 or math.isnan(sd):
            continue
        z = (ret_k[t] - mu) / sd
        if abs(z) < z_entry:
            continue
        trend_up = closes[t] > ema[t]
        calm = shock[t] <= 1.0
        if calm:
            side = (
                1
                if (z <= -z_entry and trend_up)
                else (-1 if (z >= z_entry and not trend_up) else None)
            )
        else:
            side = (
                1
                if (z >= z_entry and trend_up)
                else (-1 if (z <= -z_entry and not trend_up) else None)
            )
        if side is None:
            continue
        exit_t = min(t + hold, n - 1)
        hold_rets = run["ret"].iloc[t:exit_t].to_numpy()
        gross = float(np.prod(1.0 + hold_rets) - 1.0)
        trades.append(
            {
                "regime": "calm" if calm else "stress",
                "side": "LONG" if side > 0 else "SHORT",
                "gross": side * gross,
                "net": side * gross - _TAKER_ROUND_TRIP,
            }
        )
        flat_until = exit_t
    return trades


def _regime_gated_config(
    results: pd.DataFrame, *, k: int, z_entry: float, hold: int, ema_span: int
) -> dict:
    trades: list[dict] = []
    for run in _contiguous_runs(results):
        trades.extend(_run_regime_trades(run, k=k, z_entry=z_entry, hold=hold, ema_span=ema_span))
    if not trades:
        return {"k": k, "z_entry": z_entry, "hold_h": hold, "ema_h": ema_span, "n": 0}
    net = np.array([t["net"] for t in trades])
    mean_net = float(np.mean(net))
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    trades_per_year = 365.0 * _SLOTS_PER_DAY / hold
    sharpe = (mean_net / sd * math.sqrt(trades_per_year)) if sd > 0 else 0.0

    def _leg(regime: str) -> dict:
        leg = np.array([t["net"] for t in trades if t["regime"] == regime])
        if leg.size == 0:
            return {"n": 0}
        m = float(np.mean(leg))
        s = float(np.std(leg, ddof=1)) if leg.size > 1 else 0.0
        return {
            "n": int(leg.size),
            "win_rate": round(float(np.mean(leg > 0)), 3),
            "mean_net_bps": round(1e4 * m, 2),
            "sharpe_net_ann": round(m / s * math.sqrt(trades_per_year), 2) if s > 0 else 0.0,
        }

    return {
        "k": k,
        "z_entry": z_entry,
        "hold_h": hold,
        "ema_h": ema_span,
        "n": len(trades),
        "win_rate": round(float(np.mean(net > 0)), 3),
        "mean_net_bps": round(1e4 * mean_net, 2),
        "sharpe_net_ann": round(sharpe, 2),
        "net_multiple": round(float(np.prod(1.0 + net)), 3),
        "calm_fade": _leg("calm"),
        "stress_ride": _leg("stress"),
        "clears_gate": bool(mean_net >= _GATE_LAMBDA * _TAKER_ROUND_TRIP),
    }


def _event_pnls(
    run: pd.DataFrame, *, k: int, z_entry: float, hold: int, ema_span: int
) -> list[dict]:
    """Washout events with BOTH candidate styles' net pnl + regime + ts.

    For each |z| >= z_entry event, pnl_fade is the trend-consistent reversal
    trade (buy the dip in an uptrend, short the rip in a downtrend) and
    pnl_ride the trend-consistent momentum trade (buy the continuation up,
    short the breakdown down); each is None when the trend disagrees with it.
    Everything is known strictly from trailing data.
    """
    closes = run["close"].to_numpy()
    n = len(closes)
    if n < _RV_BASELINE + k + 1:
        return []
    ema = _ema(closes, ema_span)
    log_ret = np.log(closes[1:] / closes[:-1])
    ret_k = np.empty(n)
    ret_k[:k] = np.nan
    ret_k[k:] = closes[k:] / closes[:-k] - 1.0

    lr = pd.Series(log_ret)
    rv = lr.shift(1).rolling(_RV_WIN).std()
    baseline = rv.shift(1).rolling(_RV_BASELINE).mean()
    shock = (rv / baseline).to_numpy()
    shock = np.where(np.isfinite(shock), shock, 1.0)

    events: list[dict] = []
    for t in range(_RV_BASELINE, n):
        if math.isnan(ret_k[t]):
            continue
        hist = ret_k[t - _RV_BASELINE : t]
        mu, sd = float(np.mean(hist)), float(np.std(hist))
        if sd == 0.0 or math.isnan(sd):
            continue
        z = (ret_k[t] - mu) / sd
        if abs(z) < z_entry:
            continue
        exit_t = min(t + hold, n - 1)
        r = float(np.prod(1.0 + run["ret"].iloc[t:exit_t].to_numpy()) - 1.0)
        trend_up = closes[t] > ema[t]
        up_wash = z > 0
        fade = (
            (r - _TAKER_ROUND_TRIP)
            if (not up_wash and trend_up)
            else ((-r - _TAKER_ROUND_TRIP) if (up_wash and not trend_up) else None)
        )
        ride = (
            (r - _TAKER_ROUND_TRIP)
            if (up_wash and trend_up)
            else ((-r - _TAKER_ROUND_TRIP) if (not up_wash and not trend_up) else None)
        )
        events.append(
            {
                "ts": int(run["window_start_ms"].iloc[t]),
                "regime": "calm" if shock[t] <= 1.0 else "stress",
                "fade": fade,
                "ride": ride,
            }
        )
    return events


def _walk_forward_config(
    results: pd.DataFrame, *, k: int, z_entry: float, hold: int, ema_span: int, split: float
) -> dict:
    """Learn fade-vs-ride per regime in-sample, trade the choice out-of-sample.

    The regime-direction is NOT hardcoded: the in-sample leg of each regime
    picks the style with the higher mean net pnl (or flat if both are
    negative), and only the out-of-sample trades are scored. Every OOS trade
    is disclosed and the per-regime choice is reported.
    """
    events: list[dict] = []
    for run in _contiguous_runs(results):
        events.extend(_event_pnls(run, k=k, z_entry=z_entry, hold=hold, ema_span=ema_span))
    if not events:
        return {
            "k": k,
            "z_entry": z_entry,
            "hold_h": hold,
            "ema_h": ema_span,
            "n": 0,
            "is_events": 0,
            "oos_events": 0,
            "choices": {"calm": "flat", "stress": "flat"},
            "win_rate": 0.0,
            "mean_net_bps": 0.0,
            "sharpe_net_ann": 0.0,
            "net_multiple": 1.0,
            "by_regime": {},
            "clears_gate": False,
        }
    ts = np.array([e["ts"] for e in events])
    cut = ts.min() + split * (ts.max() - ts.min())
    is_events = [e for e in events if e["ts"] < cut]
    oos_events = [e for e in events if e["ts"] >= cut]

    def _mean(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    choices: dict[str, str | None] = {}
    for regime in ("calm", "stress"):
        sub = [e for e in is_events if e["regime"] == regime]
        m_fade, m_ride = _mean([e["fade"] for e in sub]), _mean([e["ride"] for e in sub])
        if m_fade is None and m_ride is None:
            choices[regime] = None
        elif m_fade is None:
            choices[regime] = "ride" if m_ride and m_ride > 0 else None
        elif m_ride is None:
            choices[regime] = "fade" if m_fade > 0 else None
        else:
            best = "fade" if m_fade >= m_ride else "ride"
            choices[regime] = best if max(m_fade, m_ride) > 0 else None

    oos_pnls: list[tuple[str, float]] = []
    for e in oos_events:
        style = choices.get(e["regime"])
        if style is None:
            continue
        pnl = e[style]
        if pnl is None:
            continue
        oos_pnls.append((e["regime"], pnl))
    if not oos_pnls:
        return {
            "k": k,
            "z_entry": z_entry,
            "hold_h": hold,
            "ema_h": ema_span,
            "n": 0,
            "is_events": len(is_events),
            "oos_events": len(oos_events),
            "choices": {r: (choices[r] or "flat") for r in ("calm", "stress")},
            "win_rate": 0.0,
            "mean_net_bps": 0.0,
            "sharpe_net_ann": 0.0,
            "net_multiple": 1.0,
            "by_regime": {},
            "clears_gate": False,
        }
    pnls = np.array([p for _, p in oos_pnls])
    mean_net = float(np.mean(pnls))
    sd = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    trades_per_year = 365.0 * _SLOTS_PER_DAY / hold
    sharpe = (mean_net / sd * math.sqrt(trades_per_year)) if sd > 0 else 0.0
    by_regime: dict[str, dict] = {}
    for regime in ("calm", "stress"):
        leg = np.array([p for r, p in oos_pnls if r == regime])
        by_regime[regime] = {
            "n": int(leg.size),
            "mean_net_bps": round(1e4 * float(np.mean(leg)), 2) if leg.size else None,
        }
    return {
        "k": k,
        "z_entry": z_entry,
        "hold_h": hold,
        "ema_h": ema_span,
        "n": int(len(pnls)),
        "is_events": len(is_events),
        "oos_events": len(oos_events),
        "choices": {r: (choices[r] or "flat") for r in ("calm", "stress")},
        "win_rate": round(float(np.mean(pnls > 0)), 3),
        "mean_net_bps": round(1e4 * mean_net, 2),
        "sharpe_net_ann": round(sharpe, 2),
        "net_multiple": round(float(np.prod(1.0 + pnls)), 3),
        "by_regime": by_regime,
        "clears_gate": bool(mean_net >= _GATE_LAMBDA * _TAKER_ROUND_TRIP),
    }


def _funding_clock(results: pd.DataFrame) -> dict:
    """Volatility + return by UTC 30m slot, highlighting the 8h funding marks
    (00:00 / 08:00 / 16:00 UTC — Binance perpetual funding times).

    Hansen & Kim (Duke/Yonsei 2026): volatility/volume burst around these
    marks from periodic algorithmic participation, with return predictability
    concentrated there. This checks whether the bursts exist in OUR data.
    """
    if results.empty:
        return {}
    per_slot: dict[int, list[float]] = {}
    for _, row in results.iterrows():
        per_slot.setdefault(int(row["slot"]), []).append(float(row["ret"]))
    slots = []
    for slot in range(_SLOTS_PER_DAY):
        rets = np.array(per_slot.get(slot, []))
        if rets.size < 20:
            continue
        slots.append(
            {
                "utc_min": slot * 30,
                "is_funding": bool(slot % 16 == 0),  # 00:00 / 08:00 / 16:00 UTC marks
                "n": int(rets.size),
                "mean_bps": round(1e4 * float(rets.mean()), 2),
                "vol_bps": round(1e4 * float(rets.std(ddof=1)), 2),
                "t": round(float(rets.mean() / (rets.std(ddof=1) / math.sqrt(rets.size))), 2),
            }
        )
    funding = [s for s in slots if s["is_funding"]]
    other = [s for s in slots if not s["is_funding"]]
    f_vol = float(np.mean([s["vol_bps"] for s in funding]))
    o_vol = float(np.mean([s["vol_bps"] for s in other]))
    return {
        "slots": slots,
        "funding_mean_vol_bps": round(f_vol, 2),
        "other_mean_vol_bps": round(o_vol, 2),
        "funding_vol_ratio": round(f_vol / o_vol, 3) if o_vol > 0 else None,
        "funding_mean_ret_bps": round(float(np.mean([s["mean_bps"] for s in funding])), 2),
        "other_mean_ret_bps": round(float(np.mean([s["mean_bps"] for s in other])), 2),
    }


def probe_symbol(bars30m: pd.DataFrame, symbol: str) -> dict:
    rets = _slot_returns(bars30m)
    out: dict = {
        "symbol": symbol,
        "n_bars": int(len(bars30m)),
        "n_returns": int(len(rets)),
    }
    if rets.empty:
        out["error"] = "no adjacent 30m returns"
        return out

    shen_rows = [_shen_slot(rets, s) for s in range(_SLOTS_PER_DAY)]
    out["shen_slots"] = shen_rows
    active = [r for r in shen_rows if r.get("n_days", 0) >= 30]
    best = max(active, key=lambda r: r["rule_sharpe_ann"]) if active else None
    out["shen_best"] = best

    out["washout"] = [
        _washout_config(rets, k=k, z_entry=z, hold=h, ema_span=e)
        for k in _K_GRID
        for z in _Z_GRID
        for h in _H_GRID
        for e in _EMA_GRID
    ]

    out["regime_gated"] = [
        _regime_gated_config(rets, k=k, z_entry=z, hold=h, ema_span=e)
        for k in _RGW_K
        for z in _RGW_Z
        for h in _RGW_H
        for e in _RGW_EMA
    ]

    out["walk_forward"] = [
        _walk_forward_config(rets, k=k, z_entry=z, hold=h, ema_span=e, split=_WF_SPLIT)
        for k in _RGW_K
        for z in _RGW_Z
        for h in _RGW_H
        for e in _RGW_EMA
    ]

    out["funding_clock"] = _funding_clock(rets)
    return out


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in (
        "win_rate",
        "mean_gross_bps",
        "mean_net_bps",
        "sharpe_net_ann",
        "net_multiple",
        "clears_gate",
    ):
        if col not in df.columns:
            df[col] = 0
    return df


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
    bars30m = _aggregate(bars, _30M_MS)
    logger.info(
        "intraday_30m_aggregated", bars=len(bars30m), symbols=sorted(bars30m["symbol"].unique())
    )

    results: list[dict] = []
    for symbol in symbols:
        sym = bars30m[bars30m["symbol"] == symbol.upper()]
        results.append(probe_symbol(sym, symbol.upper()))

    payload = {
        "symbols": results,
        "taker_round_trip_bps": 1e4 * _TAKER_ROUND_TRIP,
        "gate_bps": _GATE_LAMBDA * 1e4 * _TAKER_ROUND_TRIP,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("intraday_30m_written", path=args.out)

    for r in results:
        if "error" in r:
            print(f"\n{r['symbol']}: {r['error']}")
            continue
        print(f"\n=== {r['symbol']} — {r['n_returns']} 30m returns ===")
        best = r["shen_best"]
        print("\n[Shen timing structure — best session-boundary slot]")
        if best:
            print(
                f"  close slot {best['slot']}: First b1={best['b1_first']}, "
                f"SLH b2={best['b2_slh']}, R2={best['r2']}, n_days={best['n_days']}"
            )
            print(
                f"  rule: {best['rule_gross_bps']:+.1f} bps/day gross, "
                f"{best['rule_net_bps']:+.1f} bps/day net, "
                f"Sharpe {best['rule_sharpe_ann']}, "
                f"activity {100 * best['rule_activity']:.0f}% of days"
            )
        else:
            print("  insufficient data in any slot")
        w = _df(r["washout"]).sort_values("sharpe_net_ann", ascending=False)
        print("\n[Washout at 30m — top configs by net Sharpe (all reported)]")
        print(w.head(8).to_string(index=False))
        rg = _df(r["regime_gated"]).sort_values("sharpe_net_ann", ascending=False)
        print("\n[RGW — Regime-Gated Washout (my design): calm→fade, stress→ride]")
        print(rg.head(8).to_string(index=False))
        wf = _df(r["walk_forward"]).sort_values("sharpe_net_ann", ascending=False)
        print("\n[RGW walk-forward — direction LEARNED in-sample, scored OOS (60/40 split)]")
        cols = [
            "k",
            "z_entry",
            "hold_h",
            "ema_h",
            "n",
            "is_events",
            "oos_events",
            "choices",
            "win_rate",
            "mean_net_bps",
            "sharpe_net_ann",
            "net_multiple",
            "clears_gate",
        ]
        print(wf.head(8)[cols].to_string(index=False))
        fc = r["funding_clock"]
        if fc:
            print("\n[Funding-clock (Hansen & Kim 2026) — 30m vol/return at 00/08/16 UTC marks]")
            print(
                f"  funding-mark vol {fc['funding_mean_vol_bps']} bps vs other "
                f"{fc['other_mean_vol_bps']} bps (ratio {fc['funding_vol_ratio']}) | "
                f"ret {fc['funding_mean_ret_bps']:+.1f} vs {fc['other_mean_ret_bps']:+.1f} bps"
            )


if __name__ == "__main__":
    main()
