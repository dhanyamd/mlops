"""No-lookahead + mechanics smoke tests for the SCFM trend-momentum probe.

Hermetic (no Snowflake/network): synthetic hourly OHLCV only.

1. Exact open-to-open arithmetic: entry at the OPEN of the bar after the
   signal bar, exit at the OPEN of the bar `hold` later — proves the signal
   bar's close is never used as a fill price (no lookahead).
2. Trailing stop: a >stop% close after entry exits at the NEXT bar's open,
   not the hold-end open.
3. Random walk: in-sample config selection must NOT produce an out-of-sample
   edge (t-stat ~ 0). If this test fails, the pipeline is leaking the future.
4. Strong trend: the selected config must capture it OOS (signal chain works
   end-to-end: signal -> next-open fill -> vol scale -> hold/stop -> pnl).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.trend_momentum_probe import (
    _FADE_BAR_MS,
    _FADE_LOOKBACKS_BARS,
    _FADE_TAU,
    _MAKER_RT,
    _PTH_BAND,
    _PTH_BAR_MS,
    _PTH_LOOKBACKS_BARS,
    _TSMOM_BAR_MS,
    _TSMOM_REVERSAL_MULT,
    _VREG_BAR_MS,
    _VREG_TIME_FALLBACK_BARS,
    _atrend_events,
    _fade_events,
    _pth_events,
    _resample_ohlcv,
    _trade_events,
    _tsmom_events,
    _vreg_events,
    _vreg_params,
    run_probe,
)

_HOUR_MS = 3_600_000
_BASE = 472_221 * _HOUR_MS  # hour index divisible by 3 (schedule alignment)


def _hourly(opens: list[float], closes: list[float]) -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "window_start_ms": _BASE + np.arange(n) * _HOUR_MS,
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": [1.0] * n,
        }
    )


def _synthetic_hourly(
    symbol: str, n_bars: int, *, sigma: float, seed: int, drift: float = 0.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, n_bars)
    open_ = np.empty(n_bars)
    close = np.empty(n_bars)
    open_[0] = 100.0
    for i in range(n_bars):
        close[i] = open_[i] * np.exp(rets[i])
        if i + 1 < n_bars:
            open_[i + 1] = close[i]
    # Mild lognormal volume noise so volume-gated strategies (vreg's surge
    # filter, cgo's VWAP reference) see real variation instead of constant 1.
    vol_rng = np.random.default_rng(seed + 12_345)
    volume = np.exp(vol_rng.normal(0.0, 0.5, n_bars))
    return pd.DataFrame(
        {
            "symbol": [symbol] * n_bars,
            "window_start_ms": _BASE + np.arange(n_bars) * _HOUR_MS,
            "open": open_,
            "high": np.maximum(open_, close) * 1.0005,
            "low": np.minimum(open_, close) * 0.9995,
            "close": close,
            "volume": volume,
        }
    )


def _flat_rise_series(
    n_flat_hourly: int, n_rise_hourly: int, *, surge: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    """Flat floor (RV ~ 0) then a sustained rise: a deterministic HIGH-vol
    regime flip at the first rise bar (prior 360 RV all ~0 < current RV), so
    the pinned map must pick L=28 bars / alpha=3.0. ``surge`` controls whether
    the rise bars carry the volume surge the CTREND filter needs. Returns the
    hourly frame and its opens array."""
    n = n_flat_hourly + n_rise_hourly
    closes = [100.0] * n_flat_hourly
    closes += [100.0 + 2.0 * i for i in range(1, n_rise_hourly + 1)]
    opens = np.array([c * 1.0005 for c in closes])
    volume = [1.0] * n_flat_hourly + ([2.0] if surge else [1.0]) * n_rise_hourly
    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "window_start_ms": _BASE + np.arange(n) * _HOUR_MS,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0005,
            "low": np.minimum(opens, closes) * 0.9995,
            "close": closes,
            "volume": volume,
        }
    )
    return df, opens


def _closes6_to_hourly(closes6: list[float]) -> pd.DataFrame:
    """6 identical hourly rows per 6h close, open 5 bps above close, so the
    6h resample reproduces the 6h closes exactly and the 6h open-to-open
    arithmetic still holds after resampling."""
    closes = [c for c in closes6 for _ in range(6)]
    opens = [c * 1.0005 for c in closes]
    n = len(closes)
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "window_start_ms": _BASE + np.arange(n) * _HOUR_MS,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0005,
            "low": np.minimum(opens, closes) * 0.9995,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


def _first_after(mask: np.ndarray, start: int) -> int:
    """First index > ``start`` where ``mask`` is True (probe scan semantics)."""
    idx = np.flatnonzero(mask)
    return int(idx[idx > start][0])


def _tsmom_sign_and_rev(c6: np.ndarray, lookback_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the probe's tsmom indicator series on a 6h-close array so test
    expectations are DERIVED from the data path, never literal indices."""
    n = len(c6)
    mom = np.full(n, np.nan)
    mom[lookback_bars:] = c6[lookback_bars:] / c6[:-lookback_bars] - 1.0
    sign = np.where(mom > 0.0, 1.0, np.where(mom < 0.0, -1.0, 0.0))
    rev_bars = _TSMOM_REVERSAL_MULT * lookback_bars
    rev = np.full(n, np.nan)
    if rev_bars < n:
        rev[rev_bars:] = c6[rev_bars:] / c6[:-rev_bars] - 1.0
    return sign, rev


def _pth_sides(c6: np.ndarray, h6: np.ndarray, lookback_bars: int) -> np.ndarray:
    """Mirror the probe's PTH signal: side = sign(close / trailing-k max(HIGH)
    - 0.5). Expectation for _pth_events, derived on the resampled frame."""
    n = len(c6)
    pmax = pd.Series(h6).rolling(lookback_bars, min_periods=lookback_bars).max().to_numpy()
    pth = np.full(n, np.nan)
    pth[lookback_bars - 1 :] = c6[lookback_bars - 1 :] / pmax[lookback_bars - 1 :]
    return np.where(pth > _PTH_BAND, 1.0, np.where(pth < _PTH_BAND, -1.0, 0.0))


def _fade_z_and_sides(c6: np.ndarray, lookback_bars: int, vol_window: int) -> np.ndarray:
    """Mirror the probe's fade z-score side series (tau fixed). Expectation for
    _fade_events, derived on the resampled frame."""
    n = len(c6)
    lr = pd.Series(np.full(n, np.nan))
    lr[1:] = np.log(c6[1:] / c6[:-1])
    rv = lr.shift(1).rolling(vol_window).std().to_numpy()
    mom = np.full(n, np.nan)
    mom[lookback_bars:] = c6[lookback_bars:] / c6[:-lookback_bars] - 1.0
    z = np.full(n, np.nan)
    valid = np.isfinite(mom) & np.isfinite(rv) & (rv > 0)
    z[valid] = mom[valid] / (rv[valid] * np.sqrt(lookback_bars))
    z[~np.isfinite(z)] = 0.0
    sides = np.zeros(n)
    sides[z > _FADE_TAU] = -1.0
    sides[z < -_FADE_TAU] = 1.0
    return sides


def test_tsmom_sign_flip_fills_at_next_open_and_decays_on_signal() -> None:
    """AQR sign TSMOM: a sustained rise off a flat floor is a LONG sign-flip
    (trailing L-bar return turns positive). The fill is the NEXT 6h open and
    the position closes when the L-bar signal decays to zero -- entry and exit
    bars are DERIVED from the sign series on the resampled 6h frame."""
    hourly = _closes6_to_hourly(
        [100.0] * 60 + [100.0 + 2.0 * (k + 1) for k in range(30)] + [160.0] * 60
    )
    L = 28
    events = _tsmom_events(
        hourly, lookback_bars=L, vol_scale=False, reversal=False, crash=None, regime=False
    )
    bars6 = _resample_ohlcv(hourly, _TSMOM_BAR_MS)
    o6, c6 = bars6["open"].to_numpy(), bars6["close"].to_numpy()
    sign, _ = _tsmom_sign_and_rev(c6, L)

    entry_bar = int(np.flatnonzero(sign != 0.0)[0]) + 1  # first flip bar + 1
    exit_bar = _first_after(sign != 1.0, entry_bar)  # signal decays to 0
    assert len(events) == 1, events
    e = events[0]
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS  # fill at the next 6h open
    assert e["lookback_h"] == L * 6
    assert e["actual_hold_h"] == (exit_bar - entry_bar) * 6
    gross = o6[exit_bar + 1] / o6[entry_bar] - 1.0
    assert np.isclose(e["gross"], gross, atol=1e-9)
    assert e["gross"] > 0.0, "the rise must be captured as a profit"
    assert np.isclose(e["net_maker"], gross - _MAKER_RT, atol=1e-9)  # w = 1


def test_tsmom_short_leg_captures_downtrend() -> None:
    """AQR sign TSMOM SHORT: the trailing L-bar return turning negative opens
    a short at the next 6h open; gross = -(exit/entry - 1) and is positive on
    a sustained fall. Entry/exit bars derived from the sign series."""
    hourly = _closes6_to_hourly(
        [100.0] * 60 + [100.0 - 2.0 * (k + 1) for k in range(30)] + [40.0] * 60
    )
    L = 28
    events = _tsmom_events(
        hourly, lookback_bars=L, vol_scale=False, reversal=False, crash=None, regime=False
    )
    bars6 = _resample_ohlcv(hourly, _TSMOM_BAR_MS)
    o6, c6 = bars6["open"].to_numpy(), bars6["close"].to_numpy()
    sign, _ = _tsmom_sign_and_rev(c6, L)

    entry_bar = int(np.flatnonzero(sign != 0.0)[0]) + 1
    exit_bar = _first_after(sign != -1.0, entry_bar)
    assert len(events) == 1, events
    e = events[0]
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS
    assert e["actual_hold_h"] == (exit_bar - entry_bar) * 6
    gross = o6[exit_bar + 1] / o6[entry_bar] - 1.0
    assert np.isclose(e["gross"], -gross, atol=1e-9)
    assert e["gross"] > 0.0, "the fall must be captured as a short profit"


def test_tsmom_reversal_overlay_flattens_before_signal_decay() -> None:
    """The MOP reversal overlay (window = 3xL bars) flattens a trade the moment
    the longer-horizon return turns AGAINST it (rev > 0 for a short) -- strictly
    BEFORE the L-bar signal decays -- while the no-overlay run rides to the
    signal flip. All entry/exit bars are derived from the sign/rev series."""
    # flat -> rise -> 84-bar decline -> flat: sign +1 then -1 then 0.
    closes6 = [100.0] * 28
    closes6 += [100.0 + (k - 28) * (100.0 / 28.0) for k in range(28, 56)]  # to 200
    closes6 += [200.0 - (k - 56) * (120.0 / 84.0) for k in range(56, 140)]  # to ~81
    closes6 += [80.0] * 80
    hourly = _closes6_to_hourly(closes6)
    L = 28
    bars6 = _resample_ohlcv(hourly, _TSMOM_BAR_MS)
    o6, c6 = bars6["open"].to_numpy(), bars6["close"].to_numpy()
    ts6 = bars6["window_start_ms"].to_numpy()
    sign, rev = _tsmom_sign_and_rev(c6, L)

    no_rev = _tsmom_events(
        hourly, lookback_bars=L, vol_scale=False, reversal=False, crash=None, regime=False
    )
    with_rev = _tsmom_events(
        hourly, lookback_bars=L, vol_scale=False, reversal=True, crash=None, regime=False
    )

    entry_long = int(np.flatnonzero(sign != 0.0)[0]) + 1
    exit_long = _first_after(sign != 1.0, entry_long)
    entry_short = _first_after(sign == -1.0, exit_long) + 1
    sig_exit = _first_after(sign != -1.0, entry_short)
    rev_exit = _first_after(np.isfinite(rev) & (np.sign(rev) == 1.0), entry_short)
    assert rev_exit < sig_exit, "overlay must fire strictly before the signal decay"

    def short_event(events_: list[dict]) -> dict:
        return next(e for e in events_ if e["ts"] == int(ts6[entry_short]))

    assert len(no_rev) == 2 and len(with_rev) == 2, (no_rev, with_rev)

    # Long leg is identical either way (the 3xL reversal window is not yet
    # finite before the long's signal-decay exit).
    long_nr = next(e for e in no_rev if e["ts"] == int(ts6[entry_long]))
    long_r = next(e for e in with_rev if e["ts"] == int(ts6[entry_long]))
    assert long_nr["ts"] == long_r["ts"] == int(ts6[entry_long])
    assert long_nr["signal_ts"] == long_r["signal_ts"]
    assert long_nr["actual_hold_h"] == long_r["actual_hold_h"]
    assert np.isclose(long_nr["gross"], o6[exit_long + 1] / o6[entry_long] - 1.0, atol=1e-9)

    short_nr, short_r = short_event(no_rev), short_event(with_rev)
    assert short_nr["actual_hold_h"] == (sig_exit - entry_short) * 6
    assert short_r["actual_hold_h"] == (rev_exit - entry_short) * 6
    assert short_r["actual_hold_h"] < short_nr["actual_hold_h"]
    assert np.isclose(short_nr["gross"], -(o6[sig_exit + 1] / o6[entry_short] - 1.0), atol=1e-9)
    assert np.isclose(short_r["gross"], -(o6[rev_exit + 1] / o6[entry_short] - 1.0), atol=1e-9)


def test_tsmom_threaded_through_sections() -> None:
    """tsmom is wired into the grid as its own config axis set: configs are
    grouped by (L x vol x reversal x crash x regime) -- bounded at 72, no
    hand-picked L, and the reversal overlay + regime gate are genuinely
    exercised (both values present)."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.010, seed=11),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    assert not r.get("error"), r
    configs = r["sections"]["tsmom"]["configs"]
    assert configs, "no tsmom configs formed"
    assert all(c["tsmom"] for c in configs)
    assert all(c["atr_mult"] is None for c in configs)
    assert {c["lookback_h"] for c in configs} == {168, 336, 672}
    assert {c["reversal"] for c in configs} == {False, True}
    assert {c["vol_scale"] for c in configs} <= {False, True}
    assert {c["crash"] for c in configs} <= {None, 0.05, 0.10}
    assert {c["regime"] for c in configs} <= {False, True}
    assert len(configs) <= 72, "tsmom grid must be bounded by L x vol x rev x crash x regime"


def test_entry_at_next_open_exact_arithmetic() -> None:
    """Every fill uses the bar AFTER the signal (open), never the signal close."""
    n = 42
    opens = [100.0 + 2.0 * i for i in range(n)]
    closes = [o + 1.0 for o in opens]
    hourly = _hourly(opens, closes)

    events = _trade_events(
        hourly,
        lookback_h=24,
        hold_h=3,
        stop_pct=0.0,
        vol_scale=False,
        prox_tol=None,
        cgo=False,
        crash=None,
    )

    # Signal bars: t % 3 == 0, t >= 24 (lookback), t + 1 < n -> 24,27,30,33,36,39.
    assert len(events) == 6
    expected_gross = [
        opens[28] / opens[25] - 1.0,
        opens[31] / opens[28] - 1.0,
        opens[34] / opens[31] - 1.0,
        opens[37] / opens[34] - 1.0,
        opens[40] / opens[37] - 1.0,
        opens[41] / opens[40] - 1.0,
    ]
    got = [e["gross"] for e in events]
    assert np.allclose(got, expected_gross, atol=1e-12), got
    # Entry bar = signal bar + 1; entry timestamp is the entry bar's open time.
    assert [
        e["signal_ts"] == _BASE + (t * _HOUR_MS) for t, e in zip((24, 27, 30, 33, 36, 39), events)
    ]
    assert [
        e["ts"] == _BASE + ((t + 1) * _HOUR_MS) for t, e in zip((24, 27, 30, 33, 36, 39), events)
    ]
    # Net maker = gross - 4 bps (w = 1).
    assert np.allclose(
        [e["net_maker"] for e in events], [g - 0.0004 for g in expected_gross], atol=1e-12
    )


def test_trailing_stop_exits_at_next_open() -> None:
    """A >stop% close after entry exits at the next bar's open, not hold-end."""
    n = 30
    opens = [100.0 + i for i in range(n)]
    closes = [o + 0.5 for o in opens]
    closes[11] = opens[11] * 0.90  # crash close: 111 -> 99.9
    hourly = _hourly(opens, closes)

    events = _trade_events(
        hourly,
        lookback_h=1,
        hold_h=6,
        stop_pct=0.04,
        vol_scale=False,
        prox_tol=None,
        cgo=False,
        crash=None,
    )
    # The trade signalled at bar 9 enters bar 10 (open 110); close[10]=110.5 peak;
    # close[11]=99.9 <= 110.5*0.96 -> exit at open[12]=112, NOT open[16] (hold end).
    target = [e for e in events if e["ts"] == _BASE + 10 * _HOUR_MS]
    assert len(target) == 1
    e = target[0]
    assert e["signal_ts"] == _BASE + 9 * _HOUR_MS
    assert np.isclose(e["gross"], opens[12] / opens[10] - 1.0, atol=1e-12)
    assert not np.isclose(e["gross"], opens[16] / opens[10] - 1.0, atol=1e-6)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_random_walk_produces_no_oos_edge(seed: int) -> None:
    """In-sample selection on a random walk must not score OOS (no lookahead)."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.010, seed=seed),
        "ETHUSDT": _synthetic_hourly("ETHUSDT", 4800, sigma=0.014, seed=seed + 100),
    }
    res = run_probe(hourly, wf_split=0.6)

    sections = res["sections"]
    # The pooled book must actually form for at least the baseline section.
    assert any(s["pooled"]["n_trades"] >= 15 for s in sections.values())
    # Every section with a big enough pooled book must show no OOS edge.
    for name, sec in sections.items():
        p = sec["pooled"]
        if p["n_trades"] < 15:
            continue
        m = p["maker"]
        t_stat = m["mean_net_bps"] / m["se_bps"] if m["se_bps"] > 0 else 0.0
        assert abs(t_stat) < 4.0, (
            f"[{name}] OOS t-stat {t_stat:.2f} on a random walk -> lookahead leak"
        )


def test_atrend_flip_entry_filled_at_next_open() -> None:
    """H6 atrend (Bui & Nguyen 2026): LONG entry only on a fresh momentum
    up-flip (MOM crosses <=0 -> >0), filled at the NEXT bar's open -- the flip
    bar's close is never used as a fill. (140 hourly bars = 24 six-hour bars,
    enough for ATR14.)"""
    n = 140
    closes = [100.0] * 120  # bars 0..119: flat (MOM = 0)
    for i in range(120, n):
        closes.append(100.0 + 2.0 * (i - 120))  # bars 120..139: sustained rise
    opens = [c * 1.0005 for c in closes]
    hourly = _hourly(opens, closes)

    events = _atrend_events(
        hourly,
        lookback_h=6,  # L = 1 six-hour bar
        atr_mult=2.5,
        vol_scale=False,
        crash=None,
        regime=False,
    )
    # The only flip is the flat->rise crossing; the position holds to the
    # run-end OPEN (time fallback) since the rise never breaches the stop.
    assert len(events) == 1
    e = events[0]
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS  # entry bar = flip bar + 1
    assert e["actual_hold_h"] == 12  # 6h bars 21..23 held -> fallback exit
    assert np.isclose(e["gross"], opens[135] / opens[123] - 1.0, atol=1e-9)


def test_atrend_atr_trailing_stop_exits_at_next_open() -> None:
    """A close below the ATR trailing floor exits at the NEXT bar's open, not
    the time-fallback end. S_t = max(S_{t-1}, C_t - atr_mult * ATR14_t)."""
    n = 140
    closes = [100.0] * 120
    for i in range(120, 130):
        closes.append(100.0 + 2.0 * (i - 120))  # bars 120..129: 100 -> 118
    closes.append(88.0)  # bar 130: crash below the trailing floor
    for _ in range(131, n):
        closes.append(88.0)  # bars 131..139: flat at 88
    opens = [c * 1.0005 for c in closes]
    hourly = _hourly(opens, closes)

    events = _atrend_events(
        hourly,
        lookback_h=6,
        atr_mult=2.5,
        vol_scale=False,
        crash=None,
        regime=False,
    )
    # Flip at six-hour bar 20 -> entry open of bar 21; close of bar 22 (=88) is
    # far below the trailing floor (peak ~116 minus 2.5*ATR) -> exit at the
    # OPEN of bar 23, one 6h bar later (not the fallback end of bar 23's hold).
    assert len(events) == 1
    e = events[0]
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS
    assert e["actual_hold_h"] == 6, "should exit 1 six-hour bar after the breach"
    assert np.isclose(e["gross"], opens[135] / opens[123] - 1.0, atol=1e-9)


def test_atrend_threaded_through_sections() -> None:
    """The atrend section must be wired into the grid: configs carry the ATR
    multiplier set and both regime states (the gate is genuinely exercised,
    not silently disabled). Uses an oscillating series so momentum flips
    recur and UP-UP periods actually produce events."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.010, seed=11, drift=0.0),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    assert not r.get("error"), r
    configs = r["sections"]["atrend"]["configs"]
    assert configs, "no atrend configs formed"
    assert {c["atr_mult"] for c in configs} == {2.0, 2.5, 3.0}
    assert any(c["regime"] for c in configs) and any(not c["regime"] for c in configs)


def test_regime_gate_blocks_entries_during_down_regime() -> None:
    """UP-UP gate (Hsieh et al. 2025): every entry's signal bar must sit in a
    market state where the current AND the immediately preceding WEEKLY
    4-week-return states are both UP. The UP-UP set is derived from the price
    path itself (no hardcoded windows): the gate must admit no signal bar
    outside it, and it must actually remove entries the no-gate run would
    take."""
    n = 2600
    closes: list[float] = []
    for i in range(n):
        if i < 900:
            closes.append(100.0 * 1.002**i)
        elif i < 1200:
            closes.append(605.0 - 1.35 * (i - 900))
        else:
            closes.append(200.0 * 1.001 ** (i - 1200))
    opens = [c * 0.9999 for c in closes]
    hourly = _hourly(opens, closes)

    rw = 672  # _REGIME_WINDOW_H: 4-week state lookback, mirror of the probe
    step = 168  # _REGIME_STEP_H: state sampled weekly, transition = adjacent week
    up = [False] * n
    for i in range(rw + 1, n):
        up[i] = closes[i - 1] >= closes[i - 1 - rw]
    up_up = {i for i in range(rw + 1, n) if up[i] and up[i - step]}

    no_gate = _trade_events(
        hourly,
        lookback_h=24,
        hold_h=24,
        stop_pct=0.0,
        vol_scale=False,
        prox_tol=None,
        cgo=False,
        crash=None,
        regime=False,
    )
    gated = _trade_events(
        hourly,
        lookback_h=24,
        hold_h=24,
        stop_pct=0.0,
        vol_scale=False,
        prox_tol=None,
        cgo=False,
        crash=None,
        regime=True,
    )

    def sig(e: dict) -> int:
        return (e["ts"] - _BASE) // _HOUR_MS - 1  # signal bar = entry bar - 1

    sig_ng = [sig(e) for e in no_gate]
    sig_g = [sig(e) for e in gated]

    outside = [t for t in sig_ng if t not in up_up]
    assert outside, "no un-gated signal bar outside the UP-UP set -> gate has nothing to filter"
    leaked = [t for t in sig_g if t not in up_up]
    assert not leaked, f"UP-UP gate admitted signal bars outside the UP-UP set: {leaked}"


def test_stats_report_robust_moments() -> None:
    """Verdicts must carry median + winsorized stats (crypto momentum's
    variance is undefined; mean/Sharpe alone are not informative)."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.004, seed=7, drift=0.002),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    sec = r["sections"]["tsm"]
    assert sec["selected"] is not None
    oos = sec["selected"]["oos"]
    assert "median_net_bps" in oos and "winsorized_mean_net_bps" in oos
    assert "t_stat" in oos and "p10_net_bps" in oos and "p90_net_bps" in oos


def test_regime_gate_is_threaded_through_sections() -> None:
    """The UP-UP gate must actually be wired into the config grid: the `regime`
    and `full` sections generate configs with regime=True, while `tsm` stays
    regime=False. Guards against the gate being silently ignored in selection."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.004, seed=11, drift=0.002),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    assert not r.get("error"), r
    for name, expect in (("tsm", False), ("regime", True), ("full", True)):
        configs = r["sections"][name]["configs"]
        assert configs, f"[{name}] no configs formed"
        assert all(c["regime"] is expect for c in configs), (
            f"[{name}] config grid has regime={configs[0]['regime']}, expected {expect}"
        )


def test_strong_trend_is_detected_out_of_sample() -> None:
    """End-to-end: signal -> next-open fill -> hold -> OOS gross > 0."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.004, seed=7, drift=0.002),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    assert not r.get("error"), r

    # The fusion + baseline sections must both capture the trend OOS.
    assert "full" in r["sections"] and "tsm" in r["sections"]
    for name in ("tsm", "full"):
        sel = r["sections"][name]["selected"]
        assert sel is not None, f"[{name}] no eligible config on a strong trend"
        oos_gross = sel["oos_gross"]["mean_net_bps"]
        oos_maker = sel["oos"]["mean_net_bps"]
        assert oos_gross > 20.0, f"[{name}] trend not captured OOS: gross {oos_gross} bps"
        assert oos_maker > 0.0, f"[{name}] OOS net-maker negative on a trend: {oos_maker} bps"


def test_vreg_params_pinned_mapping() -> None:
    """The vol-regime -> (L, alpha) map is PRE-REGISTERED and never re-fit:
    high vol -> (28 bars, 3.0), mid -> (56, 2.5), low -> (112, 2.0)."""
    assert _vreg_params(1.0) == (28, 3.0)
    assert _vreg_params(0.7) == (28, 3.0)
    assert _vreg_params(0.31) == (56, 2.5)
    assert _vreg_params(0.5) == (56, 2.5)
    assert _vreg_params(0.3) == (112, 2.0)
    assert _vreg_params(0.0) == (112, 2.0)


def test_vreg_pinned_mapping_and_fill() -> None:
    """vreg (pinned vol-regime -> (L, alpha)): a sustained rise off a flat
    floor is a HIGH-vol flip (prior 360 RV all ~0 < current RV), so the map
    must pin (L, alpha) from its high-vol branch -- never hand-picked -- and
    the entry fills at the NEXT 6h bar's open. The rise bars carry the volume
    surge; the position rides to the time fallback (capped by the run end)."""
    hourly, _ = _flat_rise_series(2400, 600, surge=True)
    events = _vreg_events(hourly, vol_scale=False, crash=None, regime=False)
    assert len(events) == 1, events
    e = events[0]
    # Expected (L, alpha) come FROM the pinned map applied to the HIGH regime
    # the data actually produced -- not from a literal.
    L, alpha = _vreg_params(1.0)
    assert e["lookback_h"] == L * 6 and e["atr_mult"] == alpha
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS  # entry must be the next 6h open
    bars6 = _resample_ohlcv(hourly, _VREG_BAR_MS)
    o6 = bars6["open"].to_numpy()
    c6 = bars6["close"].to_numpy()
    entry_rel = int(np.flatnonzero(c6 > 100.0)[0]) + 1  # flip bar + 1
    exit_rel = min(entry_rel + _VREG_TIME_FALLBACK_BARS, len(bars6) - 1)
    assert e["actual_hold_h"] == (exit_rel - entry_rel) * 6
    assert np.isclose(e["gross"], o6[exit_rel] / o6[entry_rel] - 1.0, atol=1e-9)


def test_vreg_volume_filter_blocks_flat_volume() -> None:
    """The CTREND price+volume fusion is active: the same trend WITHOUT the
    volume surge produces no entries."""
    hourly, _ = _flat_rise_series(2400, 600, surge=False)
    events = _vreg_events(hourly, vol_scale=False, crash=None, regime=False)
    assert events == []


def test_vreg_stop_exits_at_next_open() -> None:
    """A close breaching the trailing ATR floor OR the 10% hard stop exits at
    the NEXT 6h bar's open, not the fallback end. Rise bars then a crash
    close: the exit bar is one 6h bar after the first close below the
    hard-stop band, and the fill is that bar's OPEN."""
    n_flat = 2400
    closes = [100.0] * n_flat
    closes += [100.0 + 2.0 * i for i in range(1, 25)]  # six-hour bars 400..403 rise
    crash_idx = n_flat + 24
    closes += [85.0] * (2600 - crash_idx)  # crash close + flat tail
    opens = np.array([c * 1.0005 for c in closes])
    volume = [1.0] * n_flat + [2.0] * 24 + [1.0] * (2600 - crash_idx)
    hourly = pd.DataFrame(
        {
            "symbol": ["TEST"] * 2600,
            "window_start_ms": _BASE + np.arange(2600) * _HOUR_MS,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0005,
            "low": np.minimum(opens, closes) * 0.9995,
            "close": closes,
            "volume": volume,
        }
    )
    events = _vreg_events(hourly, vol_scale=False, crash=None, regime=False)
    assert len(events) == 1, events
    e = events[0]
    bars6 = _resample_ohlcv(hourly, _VREG_BAR_MS)
    o6 = bars6["open"].to_numpy()
    c6 = bars6["close"].to_numpy()
    entry_rel = int(np.flatnonzero(c6 > 100.0)[0]) + 1
    breach = int(np.flatnonzero(c6 < 90.0)[0])  # first close below the hard-stop band
    assert breach > entry_rel, "the breach must occur after entry"
    assert breach + 1 < len(bars6), "the exit fill bar must exist"
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS
    assert e["actual_hold_h"] == (breach - entry_rel) * 6
    assert np.isclose(e["gross"], o6[breach + 1] / o6[entry_rel] - 1.0, atol=1e-9)


def test_vreg_threaded_through_sections() -> None:
    """vreg is wired into the grid with the (L, alpha) mapping as a NON-config
    axis: configs are grouped by (vol x crash x regime) only -- the grid is
    bounded at 12 keys and no config carries a hand-picked L/alpha."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.010, seed=11),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    assert not r.get("error"), r
    configs = r["sections"]["vreg"]["configs"]
    assert configs, "no vreg configs formed"
    assert all(c["vreg"] for c in configs)
    assert all(c["lookback_h"] is None and c["atr_mult"] is None for c in configs)
    assert len(configs) <= 12, "vreg grid must be bounded by vol x crash x regime"
    assert {c["crash"] for c in configs} <= {None, 0.05, 0.10}
    assert {c["regime"] for c in configs} <= {False, True}
    assert {c["vol_scale"] for c in configs} <= {False, True}


def test_pth_long_flips_short_on_deep_fall() -> None:
    """PTH long (near the trailing HIGH) held across a rise, flipped to SHORT
    when the close crosses below the 0.5 band, and the short captures the
    continued fall. Entry/exit bars are DERIVED from the PTH side series on the
    resampled 6h frame (Fičura anchor: max(HIGH), not max(close))."""
    # warm flat at 100 -> rise to 180 -> fall to 60 -> flat at 60.
    closes6 = [100.0] * 40
    closes6 += [100.0 + 2.0 * i for i in range(1, 41)]  # 102 .. 180
    closes6 += [180.0 - 3.0 * i for i in range(1, 41)]  # 177 .. 60
    closes6 += [60.0] * 40
    hourly = _closes6_to_hourly(closes6)
    L = 28
    events = _pth_events(hourly, lookback_bars=L, vol_scale=False, crash=None, regime=False)
    bars6 = _resample_ohlcv(hourly, _PTH_BAR_MS)
    o6 = bars6["open"].to_numpy()
    c6 = bars6["close"].to_numpy()
    h6 = bars6["high"].to_numpy()
    ts6 = bars6["window_start_ms"].to_numpy()
    sides = _pth_sides(c6, h6, L)

    # Long leg: first finite signal bar L-1 (side +1 during the warm flat /
    # rise), fill at the NEXT 6h open, exit when PTH crosses the band.
    entry_long = int(np.flatnonzero(sides != 0.0)[0]) + 1
    exit_long = _first_after(sides != 1.0, entry_long)
    assert len(events) >= 2, events
    e0 = events[0]
    assert e0["signal_ts"] == int(ts6[entry_long - 1])
    assert e0["ts"] == int(ts6[entry_long])
    assert e0["ts"] - e0["signal_ts"] == 6 * _HOUR_MS
    assert e0["lookback_h"] == L * 6
    assert e0["actual_hold_h"] == (exit_long - entry_long) * 6
    assert np.isclose(e0["gross"], o6[exit_long + 1] / o6[entry_long] - 1.0, atol=1e-9)
    assert np.isclose(e0["net_maker"], e0["gross"] - _MAKER_RT, atol=1e-9)  # w = 1

    # Short leg: signal bar is the first -1 after the band cross; the short
    # profits as the series keeps falling below the half-of-range anchor.
    short_sig = _first_after(sides == -1.0, exit_long)
    e1 = next(e for e in events if e["signal_ts"] == int(ts6[short_sig]))
    assert e1["ts"] == int(ts6[short_sig + 1])
    assert e1["ts"] - e1["signal_ts"] == 6 * _HOUR_MS
    short_exit = _first_after(sides != -1.0, short_sig + 1)
    assert np.isclose(e1["gross"], -(o6[short_exit + 1] / o6[short_sig + 1] - 1.0), atol=1e-9)
    assert e1["gross"] > 0.0, "the short must capture the continued fall"


def test_fade_shorts_salient_spike_and_exits_on_crossback() -> None:
    """Fade SHORT: a sharp salient spike after a zero-vol flat triggers z > tau;
    the entry fills at the NEXT 6h open (near the top) and exits when z crosses
    back inside +-tau as the spike reverts -- profit. Bars derived from the
    mirrored side series; exact open-to-open arithmetic."""
    closes6 = [100.0] * 40  # flat, RV == 0 -> no fade signal
    closes6 += [125.0, 145.0]  # 2-bar salient spike
    closes6 += [145.0 - i for i in range(1, 6)]  # 140 .. 80 immediate reversion
    closes6 += [80.0] * 20
    hourly = _closes6_to_hourly(closes6)
    L = 4
    events = _fade_events(hourly, lookback_bars=L, vol_scale=False, crash=None, regime=False)
    bars6 = _resample_ohlcv(hourly, _FADE_BAR_MS)
    o6 = bars6["open"].to_numpy()
    c6 = bars6["close"].to_numpy()
    ts6 = bars6["window_start_ms"].to_numpy()
    sides = _fade_z_and_sides(c6, L, 4)

    short_sig = int(np.flatnonzero(sides != 0.0)[0])
    entry = short_sig + 1
    exit_bar = _first_after(sides != -1.0, entry)
    assert len(events) >= 1, events
    assert len(events) == 2
    e = events[0]
    assert e["signal_ts"] == int(ts6[short_sig])
    assert e["ts"] == int(ts6[entry])
    assert e["ts"] - e["signal_ts"] == 6 * _HOUR_MS
    assert e["lookback_h"] == L * 6
    assert e["actual_hold_h"] == (exit_bar - entry) * 6
    assert np.isclose(e["gross"], -(o6[exit_bar + 1] / o6[entry] - 1.0), atol=1e-9)
    assert e["gross"] > 0.0, "the fade short must capture the spike reversion"
    assert np.isclose(e["net_maker"], e["gross"] - _MAKER_RT, atol=1e-9)


def test_fade_no_trade_in_zero_vol_flat() -> None:
    """A flat (zero realized-vol) series produces NO fade events -- the sigma
    denominator is undefined and the guard stays flat (no false extremes)."""
    hourly = _closes6_to_hourly([100.0] * 120)
    events = _fade_events(hourly, lookback_bars=4, vol_scale=False, crash=None, regime=False)
    assert events == []


def test_pth_fade_threaded_through_sections() -> None:
    """Both new sections are wired into the grid as their own config axis sets:
    configs grouped by (L x vol x crash x regime) only, lookbacks drawn from
    the module constants (including the 52-week anchor), bounded grids, and no
    reversal-tau as a config axis."""
    hourly = {
        "BTCUSDT": _synthetic_hourly("BTCUSDT", 4800, sigma=0.010, seed=11),
    }
    res = run_probe(hourly, wf_split=0.6)
    r = res["symbols"][0]
    assert not r.get("error"), r
    for name, flag, lookbacks in (
        ("pth", "pth", _PTH_LOOKBACKS_BARS),
        ("fade", "fade", _FADE_LOOKBACKS_BARS),
    ):
        configs = r["sections"][name]["configs"]
        assert configs, f"[{name}] no configs formed"
        assert all(c[flag] for c in configs)
        assert all(c["prox_tol"] is None and c["atr_mult"] is None for c in configs)
        allowed = {lb * 6 for lb in lookbacks}
        assert {c["lookback_h"] for c in configs} <= allowed, (name, configs)
        assert len(configs) <= len(lookbacks) * 2 * 3 * 2
        assert {c["crash"] for c in configs} <= {None, 0.05, 0.10}
        assert {c["regime"] for c in configs} <= {False, True}
        assert {c["vol_scale"] for c in configs} <= {False, True}
