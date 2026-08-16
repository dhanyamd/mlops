"""Execution engine: fills on the predictor's real signals.

Turns the predictor's realized directions into filled trades the way a real
execution stack would (FLOX SimulatedExecutor / ordersim / Sequence pattern):

  - A signal is filled at the close of the bar it was made on (the paper's
    eq. 4 timing: act at t on the forecast of r_{t+1}), using the freshest
    forecast available — the current bar's prediction, or the previous bar's
    if the online store has not caught up (no lookahead, no starvation).
  - Paper venue (default): the fill pays the adverse side of a fixed-bps spread
    (``slippage_bps``) and a taker fee (``taker_fee_bps``) on both entry and
    exit, deterministically (no PRNG to seed).
  - Bybit Demo venue (``venue=BybitDemoVenue``): the same state machine drives
    REAL market orders on Bybit's free Demo account (virtual funds). Fill
    prices, quantities and fees come from the venue, so the book sees actual
    exchange fills, latency and rejections. A rejected/pending order is
    SKIPPED and counted (``orders_rejected``) — never faked with a paper fill.
  - Cost-aware position management (default, ``hold_until_decay=True``): a
    position is opened only when the forecast clears the entry band and is then
    HELD through signals weaker than it — "banding", i.e. a higher hurdle to
    enter than to maintain (Novy-Marx & Velikov, "A Taxonomy of Anomalies and
    Their Trading Costs", RFS 29(1), 2016), the no-trade region of Gârleanu &
    Pedersen ("Dynamic Trading with Predictable Returns and Transaction
    Costs", JoF 68(6), 2013). The exact rule is Bysik & Ślepaczuk's
    (arXiv:2606.00060, 2026) eq. (5): trade only when |r̂| > λ·c·|pos*−pos_prev|
    with λ = ``cost_filter_lambda`` and c = round-trip taker fee — entry
    (turnover 1) at |r̂| > λ·c = 20 bps, reversal (turnover 2) at |r̂| > 2·λ·c =
    40 bps. A position persists until its signal reverses past the flip band
    (or, with ``hold_until_decay=False``, rolls after exactly one window).
  - A fill ledger and per-symbol P&L (realized / unrealized, compounded equity,
    win rate) land in the online store as ``execution:crypto:1h:<SYMBOL>``.

``window_end_ms`` is echoed in every payload as the event-time audit tag,
matching the rest of the streaming layer.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from config.logging import configure_logging, get_logger
from config.settings import PROJECT_ROOT, csv_list, get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV
from stream.predictor import prediction_key

# Durable, append-only record of every closed fill, independent of Redis.
# Redis (execution:crypto:1h:<SYMBOL>) is the fast path the dashboard reads,
# but it is one `FLUSHDB` away from losing the entire live track record with
# no recovery. This file is the source of truth for "did the strategy
# actually work over days/weeks" -- append-only, never truncated, never
# touched by a cache flush or a daemon restart.
_DURABLE_LEDGER_PATH = PROJECT_ROOT / "lake" / "live_ledger" / "fills.jsonl"


def _append_durable_fill(symbol: str, fill: dict) -> None:
    try:
        _DURABLE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DURABLE_LEDGER_PATH.open("a") as f:
            f.write(json.dumps({"symbol": symbol, **fill}) + "\n")
    except OSError:
        get_logger(__name__).exception("durable fill log write failed for %s", symbol)

logger = get_logger(__name__)


class ExecutionVenue(Protocol):
    """Structural interface for a live-fill venue (BybitDemoVenue or test fake).

    Returns the venue's actual fill (price/qty/fees) or None when the order did
    not fill — the engine never fakes a fill on None.
    """

    def open_market(self, symbol: str, side: str, notional_usd: float) -> dict | None: ...

    def close_market(self, symbol: str, side: str, qty: float) -> dict | None: ...


# 1h window length in ms — the fill/exit cadence for the paper book (the
# TRADING clock after the 5m rebuild; mirrored by Flink's 1h feature windows).
_WINDOW_MS = 3_600_000

# Equity-curve cap kept per symbol (older points trimmed for the UI).
_EQUITY_MAXLEN = 200


def execution_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _mark_pnl(position: dict, mark: float) -> float:
    """Unrealized P&L ($) of the open position marked to ``mark`` (pre-fees)."""
    qty = position["qty"]
    if position["side"] == "LONG":
        return qty * (mark - position["entry_price"])
    return qty * (position["entry_price"] - mark)


class PaperExecutionSimulator:
    """Per-symbol execution book fed by the 1h feature stream.

    State machine per symbol, driven one bar at a time:

      hold    with ``hold_until_decay`` (default): a position persists while
              the forecast stays on its side or within the no-trade band; it
              is closed only when the signal reverses past the 2·λ·c flip band
              (eq. 5 — entry |r̂|>λ·c, flip |r̂|>2·λ·c, else no trade). With
              ``hold_until_decay=False`` the book instead closes exactly one
              window after entry (the legacy 5m benchmark cadence).
      entry   no open position + a fresh forecast (current bar, falling back
              to the previous bar's) clearing the λ·c band → market order at
              the current bar's close (paper eq. 4 fill timing), slippage +
              entry fee (or a real market order on the demo venue)
      roll    the current bar's signal is stored for the next bar's fill

    The signal for a bar is read from the predictor's online-store payload
    matched by that bar's ``window_end_ms``; if the store has not caught up, no
    entry happens that bar (a real book would skip a stale signal too).

    ``venue`` is optional. When set (a ``BybitDemoVenue``), open/close fills go
    to the venue and the returned fill prices/fees/qty are used verbatim; a
    failed order is skipped (``orders_rejected``) rather than paper-filled.
    Without a venue the book uses the deterministic next-close + fixed-bps
    paper model.
    """

    def __init__(
        self,
        kv: KVStore,
        *,
        execution_prefix: str,
        prediction_prefix: str,
        notional_usd: float = 1000.0,
        slippage_bps: float = 2.0,
        taker_fee_bps: float = 10.0,
        ledger_maxlen: int = 50,
        max_trades: int = 100,
        window_ms: int = _WINDOW_MS,
        venue: ExecutionVenue | None = None,
        cost_filter_lambda: float = 2.0,
        hold_until_decay: bool = True,
        # A position must not outlive the forecast that opened it. The signal
        # is re-formed every ``rebalance_h`` hours, so a position still open
        # after that is riding a stale forecast. ``hold_until_decay`` only
        # exits on a *sign reversal past the flip band*, which in practice
        # held names for 7.7 weeks on average (measured: 336h/504h/672h/1176h
        # holds) -- far beyond the horizon the edge was validated on.
        #
        # Liu/Fang/Wang 2024 (管理评论 36(6), docs/liu_fang_wang_2024_*.txt)
        # measures this decay directly in crypto: "仅存在两周的动量效应"
        # (momentum lasts only two weeks) and "处置效应对收益率的影响不超过
        # 一个月" (disposition effect <= one month). Holding ~7.7 weeks trades
        # a signal whose predictive power has already decayed to nothing.
        #
        # 0 => derive from the signal's own rebalance horizon (the correct
        # default); a positive value caps the hold explicitly in hours.
        max_hold_h: int = 0,
        # Windows of predictor/engine drift tolerated before a forecast is
        # treated as unusable. See _freshest.
        signal_max_stale_windows: int = 2,
        # Append closed fills to the durable live ledger. OFF for replays and
        # simulations so historical fills never enter the live track record.
        durable_log: bool = True,
        trail_atr_mult: float = 2.5,
        trail_atr_bars: int = 56,
        trail_act_mult: float = 1.5,
        trail_min: float = 0.03,
        trail_max: float = 0.25,
        # Volatility-scaling (Barroso & Santa-Clara 2015; "Crypto momentum has
        # (not) its moments" 2025): scale gross exposure by target_vol/realized_vol
        # so the book shrinks when volatility spikes (crash protection) and expands
        # when calm. Plain crypto momentum is NEGATIVE; vol-scaled is +1.86-2.40%/wk.
        vol_scale: bool = True,
        vol_proxy: str = "BTCUSDT",
        vol_target: float = 0.12,
        vol_floor: float = 0.25,
        vol_cap: float = 2.0,
        vol_lookback: int = 336,
    ) -> None:
        self._kv = kv
        self._execution_prefix = execution_prefix
        self._prediction_prefix = prediction_prefix
        self._notional = notional_usd
        self._slippage = slippage_bps / 10_000.0
        self._taker_fee = taker_fee_bps / 10_000.0
        self._ledger_maxlen = ledger_maxlen
        self._max_trades = max_trades
        self._window_ms = window_ms
        self._max_hold_ms = max_hold_h * 3_600_000 if max_hold_h > 0 else 0
        # How stale a stored forecast may be and still be tradeable when the
        # exact-window match misses (see _freshest), in whole windows.
        self._signal_max_stale_ms = signal_max_stale_windows * window_ms
        self._durable_log = durable_log
        self._venue = venue
        # Cost-aware filter (Bysik & Ślepaczuk 2026, eq. 5): trade only when
        # |r̂| > λ·c·|pos*−pos_prev|. λ=2 with c = round-trip taker fee (10 bps)
        # gives a 20 bps entry band and a 40 bps reversal band. When
        # ``hold_until_decay`` is False the engine keeps the legacy behaviour
        # (close exactly one window after entry) for the 5m benchmark book.
        self._cost_lambda = cost_filter_lambda
        self._hold_until_decay = hold_until_decay
        self._entry_threshold = cost_filter_lambda * self._taker_fee
        self._flip_threshold = 2.0 * self._entry_threshold
        # Trailing-stop exit (Ekström–Lindberg "Optimal Closing of a Momentum
        # Trade", 2013 + Leung "Optimal trading with a trailing stop", 2017):
        # the optimal liquidation under an unobservable momentum-decay change
        # point is a stop on the running extremum since entry. Research
        # (StratBase, Blofin, Leung) shows ATR-based stops (2-3x ATR) vastly
        # beat fixed-% in crypto's 60-70% chop time: they widen in vol expansions
        # instead of chopping on noise. We use a 4H-equivalent ATR (mean |4h
        # close-to-close return| over trail_atr_bars) scaled by trail_atr_mult,
        # activated only once the trade is in profit by >= trail_act_mult x ATR.
        # Long closes when close <= peak·(1-α); short when close >= trough·(1+α).
        self._trail_atr_mult = trail_atr_mult
        self._trail_atr_bars = trail_atr_bars
        self._trail_act_mult = trail_act_mult
        self._trail_min = trail_min
        self._trail_max = trail_max
        # Env overrides for A/B tuning the stop. Research (StratBase, Blofin,
        # Leung "Optimal trading with a trailing stop"): ATR-based stops
        # (2-3x ATR) vastly beat fixed-% in crypto's 60-70% chop time; activate
        # only after the trade is in profit by >=1.5x ATR. QUANT_TRAIL_OFF=1
        # disables the stop (pure hold-to-rebalance close, Jegadeesh-Titman).
        # NOTE: trail_max = 0.0 does NOT disable the stop -- the exit test is
        # ``drawdown >= alpha``, and a drawdown is >= 0 by construction, so an
        # alpha of 0 fires on EVERY bar. That turned the "off" switch into the
        # tightest possible stop (measured: 2800 trades, every one held a
        # single 1h bar). Disabling must be an explicit flag.
        self._trail_enabled = os.environ.get("QUANT_TRAIL_OFF") != "1"
        if not self._trail_enabled:
            self._trail_max = 0.0
        else:
            self._trail_atr_mult = float(
                os.environ.get("QUANT_TRAIL_ATR_MULT", self._trail_atr_mult)
            )
            self._trail_atr_bars = int(os.environ.get("QUANT_TRAIL_ATR_BARS", self._trail_atr_bars))
            self._trail_act_mult = float(os.environ.get("QUANT_TRAIL_ACT", self._trail_act_mult))
            self._trail_min = float(os.environ.get("QUANT_TRAIL_MIN", self._trail_min))
            self._trail_max = float(os.environ.get("QUANT_TRAIL_MAX", self._trail_max))

        self._vol_scale_on = vol_scale and os.environ.get("QUANT_VOL_OFF") != "1"
        self._vol_proxy = vol_proxy
        self._vol_target = float(os.environ.get("QUANT_VOL_TARGET", vol_target))
        self._vol_floor = float(os.environ.get("QUANT_VOL_FLOOR", vol_floor))
        self._vol_cap = float(os.environ.get("QUANT_VOL_CAP", vol_cap))
        self._vol_lookback = int(os.environ.get("QUANT_VOL_LOOKBACK", vol_lookback))

        # Per-symbol state.
        self._last_window_end: dict[str, int | None] = {}
        self._position: dict[str, dict | None] = {}
        self._ledger: dict[str, deque] = {}
        self._equity: dict[str, list[float]] = {}
        self._realized_pnl: dict[str, float] = {}
        self._gross_pnl: dict[str, float] = {}
        self._gross_volume: dict[str, float] = {}
        self._total_fees: dict[str, float] = {}
        self._n_trades: dict[str, int] = {}
        self._n_wins: dict[str, int] = {}
        self._signals_skipped: dict[str, int] = {}
        self._orders_rejected: dict[str, int] = {}
        self._closes: dict[str, deque] = {}
        # Funding-rate series per symbol (end_ms, rate) for accruing the funding
        # CASHFLOW that IS the edge of crypto funding-carry strategies. Our paper
        # venue fills at the close (price P&L only), so without this the carry edge
        # is invisible in backtest — the live Bybit Demo book does pay it. Keel /
        # BIS show funding carry at Sharpe 1.7-2.1; we must model the drip to test it.
        self._funding_series: dict[str, list[tuple[int, float]]] = {}

    # ── signal resolution ───────────────────────────────────────────────────

    def _prediction_for(self, symbol: str, window_end: int | None) -> Mapping | None:
        """The predictor's stored payload for ``window_end``, or None if unknown."""
        if window_end is None:
            return None
        prediction = self._kv.get_json(prediction_key(self._prediction_prefix, symbol))
        if not prediction:
            return None
        tag = prediction.get("window_end_ms")
        if not isinstance(tag, (int, float)) or int(tag) != window_end:
            return None
        return prediction

    def _signal_for(self, symbol: str, window_end: int | None) -> str | None:
        """Direction the predictor held for ``window_end``, or None if unknown."""
        prediction = self._prediction_for(symbol, window_end)
        if prediction is None:
            return None
        direction = prediction.get("direction")
        return direction if isinstance(direction, str) else None

    # ── fills ───────────────────────────────────────────────────────────────

    def _fill_price(self, side: str, close: float) -> float:
        """Adverse-side fill price: buyers pay more, sellers receive less."""
        if side == "LONG":
            return close * (1.0 + self._slippage)
        return close * (1.0 - self._slippage)

    # ── ATR-based trailing-stop exit (Ekström–Lindberg / Leung + crypto research) ──

    def _trail_atr(self, symbol: str) -> float | None:
        """4H-equivalent ATR: mean |4h close-to-close return| over ``trail_atr_bars``.

        Bars arrive hourly, so a 4h step (step=4) resamples to the 4H timeframe the
        crypto literature tunes trailing stops on (StratBase/Blofin: 4H trails 4-6%
        BTC, 6-10% alts = ~2-3x ATR). Returns fractional ATR or None if insufficient history.
        """
        buf = self._closes.get(symbol)
        step = 4
        need = step * 4 + 1
        if buf is None or len(buf) < need:
            return None
        trs = [
            abs(buf[i] - buf[i - step]) / buf[i - step]
            for i in range(step, len(buf), step)
            if buf[i - step] > 0
        ]
        if len(trs) < 4:
            return None
        return statistics.fmean(trs)

    def _trail_alpha(self, symbol: str) -> float:
        """ATR-scaled drawdown band α (clamped to [trail_min, trail_max])."""
        atr = self._trail_atr(symbol)
        if atr is None:
            return self._trail_max
        return min(self._trail_max, max(self._trail_min, self._trail_atr_mult * atr))

    @staticmethod
    def _trail_drawdown(position: dict, close: float) -> float | None:
        """Drawdown of ``close`` from the running extremum since entry."""
        if position["side"] == "LONG":
            peak = position.get("peak") or close
            return (peak - close) / peak if peak > 0 else None
        trough = position.get("trough") or close
        return (close - trough) / trough if trough > 0 else None

    def _trail_stop_hit(self, symbol: str, position: dict, close: float) -> bool:
        """True when price has retraced >= α from the running extremum AND the trade
        has first moved into profit by >= trail_act_mult x ATR (activation: don't
        chop a position that hasn't earned its stop yet — StratBase/Leung)."""
        if not self._trail_enabled:
            return False
        atr = self._trail_atr(symbol)
        if atr is None:
            return False
        # The position dict stores "entry_price" -- there is no "entry" key, so
        # this lookup silently returned None on every call and skipped the
        # activation guard below entirely. The stop then fired from the very
        # first bar on any retracement, including on positions that never got
        # into profit, which is the opposite of the Leung/StratBase design it
        # cites (activate only once the trade is ahead by act_mult x ATR).
        entry = position.get("entry_price")
        if entry and entry > 0:
            move = (
                (close - entry) / entry if position["side"] == "LONG" else (entry - close) / entry
            )
            if move < self._trail_act_mult * atr:
                return False
        dd = self._trail_drawdown(position, close)
        if dd is None:
            return False
        alpha = min(self._trail_max, max(self._trail_min, self._trail_atr_mult * atr))
        return dd >= alpha

    def _vol_scale(self) -> float:
        """Barroso-Santa-Clara exposure scaler: target_vol / realized_vol.

        Realized vol is the proxy symbol's (BTC) rolling 1h log-return stdev,
        annualized. When vol is high the factor is in crash regime, so we shrink
        gross exposure; when calm we expand. Clamped to [floor, cap]. Scale-
        invariant for Sharpe, so the target choice only sets average exposure.
        """
        if not self._vol_scale_on:
            return 1.0
        buf = self._closes.get(self._vol_proxy)
        if buf is None or len(buf) < self._vol_lookback + 1:
            return 1.0
        rets = [
            math.log(buf[i] / buf[i - 1])
            for i in range(len(buf) - self._vol_lookback, len(buf))
            if buf[i - 1] > 0
        ]
        if len(rets) < 10:
            return 1.0
        ann_vol = statistics.pstdev(rets) * math.sqrt(24 * 365)
        if ann_vol <= 1e-6:
            return 1.0
        return min(self._vol_cap, max(self._vol_floor, self._vol_target / ann_vol))

    def _open(self, symbol: str, side: str, close: float, window_end: int) -> None:
        eff = self._notional * self._vol_scale()
        if self._venue is not None:
            fill = self._venue.open_market(symbol, side, eff)
            if fill is None:
                # The venue rejected / did not fill the order this bar. Honest
                # behavior: skip the entry and count it — never fake a fill.
                logger.warning(
                    "bybit demo: entry skipped for %s (%s) — order not filled", symbol, side
                )
                self._orders_rejected[symbol] = self._orders_rejected.get(symbol, 0) + 1
                return
            self._position[symbol] = {
                "side": side,
                "entry_price": _round(fill["fill_price"]),
                "qty": _round(fill["qty"], 8),
                "entry_fees": _round(fill["fees"], 4),
                "entry_window_end_ms": window_end,
                "peak": _round(fill["fill_price"]),
                "trough": _round(fill["fill_price"]),
            }
            return
        entry_price = self._fill_price(side, close)
        qty = eff / entry_price
        entry_fee = eff * self._taker_fee
        self._position[symbol] = {
            "side": side,
            "entry_price": _round(entry_price),
            "qty": _round(qty, 8),
            "entry_fees": _round(entry_fee, 4),
            "entry_window_end_ms": window_end,
            "peak": _round(entry_price),
            "trough": _round(entry_price),
        }

    def _close(self, symbol: str, position: dict, close: float, window_end: int) -> None:
        side = position["side"]
        if self._venue is not None:
            fill = self._venue.close_market(symbol, side, position["qty"])
            if fill is not None and fill.get("desync"):
                # The venue says this position is already flat -- our book was
                # stale (manual intervention, ADL, a prior desync). Retrying
                # would fail the same way forever, so clear it now rather than
                # leave it "open" indefinitely with unrealized P&L that can
                # never realize. No fill happened, so no P&L is recorded --
                # never fabricate a trade that didn't occur.
                logger.warning(
                    "execution: clearing desynced position %s %s @ %.6f (venue reports flat)",
                    symbol,
                    side,
                    position["entry_price"],
                )
                self._position[symbol] = None
                return
            if fill is None:
                # The close order did not fill this bar: keep the position open
                # so the next window retries the close. Count it for honesty.
                logger.warning("bybit demo: close order not filled for %s (%s)", symbol, side)
                self._orders_rejected[symbol] = self._orders_rejected.get(symbol, 0) + 1
                return
            exit_price = fill["fill_price"]
            qty = fill["qty"]
            fees = position["entry_fees"] + fill["fees"]
        else:
            # Closing is the opposite leg: a LONG sells (receives less), a SHORT
            # buys back (pays more), so the exit fill pays the adverse side too.
            exit_side = "SHORT" if side == "LONG" else "LONG"
            exit_price = self._fill_price(exit_side, close)
            qty = position["qty"]
            fees = position["entry_fees"] + qty * exit_price * self._taker_fee
        gross = (
            qty * (exit_price - position["entry_price"])
            if side == "LONG"
            else qty * (position["entry_price"] - exit_price)
        )
        net = gross - fees
        pnl_pct = net / self._notional
        # ADOPTED ORPHANS carry entry_window_end_ms == 0 (a position found on the
        # venue that this engine did not open, so its entry time is unknown).
        # Differencing against 0 makes the arithmetic degenerate to
        # window_end / window_ms -- the age of the Unix epoch, ~496,000 hourly
        # bars, which was being written into the fill record and rendered in the
        # dashboard as a 56-year holding period.
        #
        # The holding period is genuinely UNKNOWN for these, so report 0 rather
        # than fabricating a number. 0 keeps the field an int for the UI and for
        # scripts/live_track_record.py, which already excludes orphans via the
        # explicit ``adopted_orphan`` flag emitted below.
        entry_w = position.get("entry_window_end_ms") or 0
        bars_held = (
            max(1, round((window_end - entry_w) / self._window_ms)) if entry_w else 0
        )

        self._position[symbol] = None
        self._realized_pnl[symbol] = self._realized_pnl.get(symbol, 0.0) + net
        self._gross_pnl[symbol] = self._gross_pnl.get(symbol, 0.0) + gross
        self._gross_volume[symbol] = (
            self._gross_volume.get(symbol, 0.0) + position["entry_price"] * qty + exit_price * qty
        )
        self._total_fees[symbol] = self._total_fees.get(symbol, 0.0) + fees
        self._n_trades[symbol] = self._n_trades.get(symbol, 0) + 1
        if net > 0.0:
            self._n_wins[symbol] = self._n_wins.get(symbol, 0) + 1

        equity = self._equity.setdefault(symbol, [1.0])
        equity.append(equity[-1] * (1.0 + pnl_pct))
        if len(equity) > _EQUITY_MAXLEN:
            del equity[:-_EQUITY_MAXLEN]

        fill_record = {
            "window_end_ms": window_end,
            "side": side,
            "entry_price": position["entry_price"],
            "exit_price": _round(exit_price),
            "qty": position["qty"],
            "fees": _round(fees, 4),
            "gross_pnl": _round(gross, 4),
            "net_pnl": _round(net, 4),
            "net_pnl_pct": _round(pnl_pct, 6),
            "bars_held": bars_held,
            # Adopted orphans carry entry_window_end_ms == 0: this engine never
            # chose the entry, it inherited an already-open venue position.
            # Their P&L reflects whatever the price did while the book was
            # broken, NOT a strategy decision, so the track record must not
            # score them as strategy trades.
            "adopted_orphan": position.get("entry_window_end_ms", 0) == 0,
        }
        fills = self._ledger.setdefault(symbol, deque())
        fills.appendleft(fill_record)
        while len(fills) > self._ledger_maxlen:
            fills.pop()
        # Guard on the real Redis backing so hermetic tests (FakeKV) never
        # touch the production durable log -- and on _durable_log, so a
        # historical REPLAY driving the live Redis cannot append years-old
        # fills to the genuine live track record (it did: 159 replay fills
        # landed in fills.jsonl alongside 21 real ones).
        if self._durable_log and isinstance(self._kv, RedisKV):
            _append_durable_fill(symbol, fill_record)

    def _freshest(
        self, symbol: str, window_end: int | None, prev_end: int | None
    ) -> tuple[str | None, float | None]:
        """Freshest stored forecast when bar ``window_end`` closes.

        The paper (Bysik & Ślepaczuk eq. 4) fills at the forecast bar's close —
        act at ``t`` with the forecast of ``r_{t+1}``. We prefer the prediction
        made at this bar's close (``window_end``) and fall back to the previous
        bar's forecast (``prev_end``) only if the online store has not caught
        up, so a predictor→engine write race never starves the book. Returns
        ``(direction, predicted_return)`` or ``(None, None)``.

        Matching used to require the stored tag to EQUAL ``window_end`` or
        ``prev_end`` exactly. The signal daemon and this engine consume the
        same feature topic in SEPARATE consumer groups at independent offsets,
        and the signal daemon rewrites every symbol's prediction stamped with
        whatever window it is on. Any drift beyond one window -- which a
        restart of either process causes -- left neither candidate matching,
        the signal read as None, and the entry was skipped. Measured live:
        7 of 12 selected symbols never opened a position at all
        (signals_skipped 3-4, n_trades 0), leaving the book deployed at a
        fraction of its intended size.

        So: accept the freshest forecast stamped AT OR BEFORE this bar, within
        ``_signal_max_stale_ms``. Never after -- a forecast stamped in the
        future is look-ahead and is still rejected.
        """
        for candidate in (window_end, prev_end):
            prediction = self._prediction_for(symbol, candidate)
            if prediction is None:
                continue
            direction = prediction.get("direction")
            yhat = prediction.get("predicted_return")
            return (
                direction if isinstance(direction, str) else None,
                float(yhat) if isinstance(yhat, (int, float)) else None,
            )

        # Exact match failed: fall back to the stored forecast if it is in the
        # past relative to this bar and not staler than the bound.
        if window_end is not None:
            prediction = self._kv.get_json(prediction_key(self._prediction_prefix, symbol))
            tag = (prediction or {}).get("window_end_ms")
            if prediction and isinstance(tag, (int, float)):
                age = window_end - int(tag)
                if 0 <= age <= self._signal_max_stale_ms:
                    direction = prediction.get("direction")
                    yhat = prediction.get("predicted_return")
                    return (
                        direction if isinstance(direction, str) else None,
                        float(yhat) if isinstance(yhat, (int, float)) else None,
                    )
        return None, None

    def _advance(self, symbol: str, close: float, window_end: int, prev_end: int | None) -> None:
        """One bar of the state machine (manage position → record signal bar)."""
        if self._hold_until_decay:
            # Cost-aware management (Bysik & Ślepaczuk 2026, eqs. 5-7): trade
            # only when |r̂| > λ·c·|pos*−pos_prev|. Same-side or weak signals
            # (turnover 0 / within the no-trade region) keep the position —
            # banding holds winners until they reverse (Novy-Marx & Velikov
            # 2016; Gârleanu & Pedersen 2013). A flip (turnover 2) requires
            # |r̂| past the 2·λ·c band; anything weaker is ignored.
            signal, yhat = self._freshest(symbol, window_end, prev_end)
            position = self._position.get(symbol)

            # Forecast expiry: close a position that has outlived the horizon
            # its signal was formed on, BEFORE any hold/banding logic can keep
            # it alive. Banding is designed to hold a position through weak
            # signals, which is right within the forecast horizon and wrong
            # past it -- past it, the book is holding on a forecast that no
            # longer exists rather than on conviction.
            if position is not None and self._max_hold_ms > 0:
                held_ms = window_end - position.get("entry_window_end_ms", window_end)
                if held_ms >= self._max_hold_ms:
                    if signal in ("LONG", "SHORT") and signal == position["side"]:
                        # ROLL, don't round-trip. The forecast expiring means
                        # "re-evaluate", not "must trade". When the refreshed
                        # forecast still agrees with the position, closing and
                        # immediately re-opening the SAME position pays a full
                        # round trip to end up exactly where we started -- pure
                        # cost, zero change in exposure. The research book this
                        # is validated against charges only |w_new - w_old|, so
                        # an unchanged weight costs it nothing. Gârleanu &
                        # Pedersen (2013): under transaction costs the optimum
                        # is to trade toward the target, never to churn through
                        # it. Reset the clock and hold.
                        position["entry_window_end_ms"] = window_end
                    else:
                        self._close(symbol, position, close, window_end)
                        position = self._position.get(symbol)

            # Running close history for the vol-scaled trailing stop, and the
            # running-extremum exit (Ekström–Lindberg / Leung). The trailing
            # stop is price-based, so it is checked before any signal logic.
            # Buffer must cover BOTH the ATR stop (short) and vol-scaling
            # (long) lookbacks, else vol_scale silently returns 1.0.
            buf_max = max(self._trail_atr_bars, self._vol_lookback + 2)
            self._closes.setdefault(symbol, deque(maxlen=buf_max)).append(close)
            if position is not None:
                if position["side"] == "LONG":
                    position["peak"] = max(position.get("peak", close), close)
                else:
                    position["trough"] = min(position.get("trough", close), close)
                if self._trail_stop_hit(symbol, position, close):
                    self._close(symbol, position, close, window_end)
                elif signal in ("LONG", "SHORT") and signal != position["side"]:
                    # Opposite-side signal: flip only past the 2·λ·c band; a
                    # weaker opposite signal stays in the no-trade region.
                    if yhat is not None and abs(yhat) > self._flip_threshold:
                        self._close(symbol, position, close, window_end)
                        if self._n_trades.get(symbol, 0) < self._max_trades:
                            self._open(symbol, signal, close, window_end)
                elif signal not in ("LONG", "SHORT"):
                    # Symbol left the selected long/short set (FLAT / no
                    # forecast) → exit the position so the book rotates and
                    # realizes P&L at each rebalance instead of holding forever.
                    self._close(symbol, position, close, window_end)
            elif (
                signal in ("LONG", "SHORT")
                and prev_end is not None
                and window_end >= prev_end + self._window_ms
                and self._n_trades.get(symbol, 0) < self._max_trades
                and yhat is not None
                and abs(yhat) > self._entry_threshold
            ):
                # Entry gate: the fresh forecast must clear the λ·c band in
                # magnitude (turnover 1), independent of the predictor's own
                # threshold. Fills at the signal bar's close (paper eq. 4).
                self._open(symbol, signal, close, window_end)
            elif (
                signal is None and prev_end is not None and window_end >= prev_end + self._window_ms
            ):
                # No prediction for the signal bar (online store lagged) → the
                # entry is skipped, exactly as a live book would skip a stale
                # signal. Counted for observability, not treated as an error.
                self._signals_skipped[symbol] = self._signals_skipped.get(symbol, 0) + 1
        else:
            # Legacy cadence (5m benchmark book): close exactly one window
            # after entry, re-enter on the previous window's signal.
            position = self._position.get(symbol)
            if (
                position is not None
                and window_end >= position["entry_window_end_ms"] + self._window_ms
            ):
                self._close(symbol, position, close, window_end)
                position = None
            if self._position.get(symbol) is None:
                signal = self._signal_for(symbol, prev_end)
                if (
                    signal in ("LONG", "SHORT")
                    and prev_end is not None
                    and window_end >= prev_end + self._window_ms
                    and self._n_trades.get(symbol, 0) < self._max_trades
                ):
                    self._open(symbol, signal, close, window_end)
                elif (
                    signal is None
                    and prev_end is not None
                    and window_end >= prev_end + self._window_ms
                ):
                    self._signals_skipped[symbol] = self._signals_skipped.get(symbol, 0) + 1

        self._last_window_end[symbol] = window_end

    # ── payload ─────────────────────────────────────────────────────────────

    def _payload(self, symbol: str, window_end: int | None) -> dict:
        position = self._position.get(symbol)
        equity = self._equity.get(symbol, [1.0])
        n_trades = self._n_trades.get(symbol, 0)
        n_wins = self._n_wins.get(symbol, 0)
        unrealized = 0.0
        position_view = None
        if position is not None:
            unrealized = float(position.get("unrealized_pnl") or 0.0)
            position_view = {
                "side": position["side"],
                "entry_price": position["entry_price"],
                "qty": position["qty"],
                "entry_fees": position["entry_fees"],
                "entry_window_end_ms": position["entry_window_end_ms"],
                "peak": position.get("peak"),
                "trough": position.get("trough"),
                "trail_alpha": round(self._trail_alpha(symbol), 4),
                "mark_price": position.get("mark_price"),
                "unrealized_pnl": position.get("unrealized_pnl"),
                "unrealized_pnl_pct": position.get("unrealized_pnl_pct"),
            }
        realized = self._realized_pnl.get(symbol, 0.0)
        gross = self._gross_pnl.get(symbol, 0.0)
        fees = self._total_fees.get(symbol, 0.0)
        demo = self._venue is not None
        filter_assumptions = {
            "cost_filter_lambda": self._cost_lambda,
            "entry_threshold_bps": round(self._entry_threshold * 10_000, 2),
            "flip_threshold_bps": round(self._flip_threshold * 10_000, 2),
            "hold_until_decay": self._hold_until_decay,
            "filter_rule": (
                "open only when |predicted_return| > lambda * round-trip taker fee "
                "(20 bps at lambda=2); flip only past 2 * lambda * fee (40 bps); "
                "otherwise HOLD — Bysik & Slepaczuk (2026) eq. (5)"
            ),
        }
        assumptions = (
            {
                "fill_timing": (
                    "freshest forecast at window t, real market order at window t close "
                    "(paper eq. 4: act at t), reduce-only market close on reversal "
                    "past the flip band"
                ),
                "slippage_bps": 0.0,
                "taker_fee_bps": 0.0,
                "cost_model": (
                    "actual Bybit Demo fills (virtual funds) — fill prices, qty and "
                    "fees from the venue's order response; no fixed-bps model"
                ),
                "venue": "bybit-demo",
                "funding": "virtual USDT — no real money, no KYC; results are demo only",
                "not_modeled": "margin, funding, partial fills, queue position, and market impact",
                **filter_assumptions,
            }
            if demo
            else {
                "fill_timing": (
                    "freshest forecast at window t, market fill at window t close "
                    "(paper eq. 4: act at t), exit on reversal past the flip band "
                    "(or after one window when hold_until_decay is off)"
                ),
                "slippage_bps": round(self._slippage * 10_000, 2),
                "taker_fee_bps": round(self._taker_fee * 10_000, 2),
                "cost_model": (
                    "adverse-side fill (pay spread on buy, give spread on sell) "
                    "+ taker fee both legs"
                ),
                "deterministic": "fills are next-close + fixed bps; no PRNG, no hidden randomness",
                "not_modeled": "margin, funding, partial fills, queue position, and market impact",
                **filter_assumptions,
            }
        )
        return {
            "symbol": symbol,
            "window_end_ms": window_end,
            "venue": "bybit-demo" if demo else "paper",
            "notional_usd": self._notional,
            "slippage_bps": assumptions["slippage_bps"],
            "taker_fee_bps": assumptions["taker_fee_bps"],
            "cost_filter_lambda": self._cost_lambda,
            "entry_threshold_bps": assumptions["entry_threshold_bps"],
            "flip_threshold_bps": assumptions["flip_threshold_bps"],
            "hold_until_decay": self._hold_until_decay,
            "n_trades": n_trades,
            "n_wins": n_wins,
            "win_rate": (round(n_wins / n_trades, 4) if n_trades else None),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "net_pnl": round(realized + unrealized, 2),
            "gross_pnl": round(gross, 2),
            "gross_volume": round(self._gross_volume.get(symbol, 0.0), 2),
            "total_fees": round(fees, 2),
            "fees_pct_of_gross_pnl": (round(fees / abs(gross) * 100.0, 2) if gross else None),
            "signals_skipped": self._signals_skipped.get(symbol, 0),
            "orders_rejected": self._orders_rejected.get(symbol, 0),
            "total_return": round(equity[-1] - 1.0, 6),
            "equity": [round(e, 6) for e in equity],
            "position": position_view,
            "fills": list(self._ledger.get(symbol, deque())),
            "assumptions": assumptions,
            "updated_at": _now_iso(),
        }

    def handle(self, msg: dict) -> dict | None:
        """One feature window: roll the book forward and refresh the store."""
        symbol = str(msg.get("symbol") or "").upper()
        if not symbol:
            return None
        close = msg.get("close")
        if not isinstance(close, (int, float)) or close != close or close == 0:
            return None
        window_end = msg.get("window_end_ms")
        window_end = int(window_end) if isinstance(window_end, (int, float)) else None
        if window_end is None:
            return None

        prev_end = self._last_window_end.get(symbol)
        self._advance(symbol, float(close), window_end, prev_end)

        # Mark the open position to the latest close (pre-exit-bar windows):
        # unrealized P&L is part of the book even when nothing closes this bar.
        position = self._position.get(symbol)
        if position is not None:
            position["mark_price"] = close
            unrealized = _mark_pnl(position, close)
            position["unrealized_pnl"] = _round(unrealized, 4)
            position["unrealized_pnl_pct"] = _round(unrealized / self._notional, 6)

        payload = self._payload(symbol, window_end)
        self._kv.set_json(execution_key(self._execution_prefix, symbol), payload)
        return payload

    def run_forever(
        self,
        bus: MessageBus,
        features_topic: str,
        group_id: str,
        stop: threading.Event | None = None,
    ) -> None:
        for _topic, msg in bus.iter_consume(features_topic, group_id, stop=stop):
            self.handle(msg)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _build_venue(settings, venue_cls=None) -> ExecutionVenue | None:
    """Resolve the execution venue, degrading honestly to paper.

    Returns the venue when demo trading is actually reachable, else None (paper
    fills). Never fabricates fills: a demo account that exists but rejects the
    keys (ErrCode 10003 = key/domain mismatch) logs loudly and falls back.
    """
    if settings.stream_execution_venue == "paper":
        # Explicit opt-out: previously STREAM_EXECUTION_VENUE was purely
        # cosmetic here -- if demo keys were present in the environment the
        # Bybit venue was built regardless, so "paper" mode could not
        # actually be selected. Real venue reconciliation surfaced fills the
        # book recorded as closed that were still open on the exchange
        # (fill-confirmation race, not yet root-caused); paper mode must be a
        # genuine, deterministic escape hatch from that class of bug.
        return None
    if not settings.has_bybit_demo_credentials:
        if settings.stream_execution_venue == "bybit-demo":
            logger.warning(
                "STREAM_EXECUTION_VENUE=bybit-demo but no demo keys found "
                "(BYBIT_DEMO_API_KEY/SECRET or API_KEY_BYBIT/API_SECRET_BYBIT) "
                "→ falling back to paper fills"
            )
        elif settings.stream_execution_venue != "paper":
            logger.warning(
                "unknown STREAM_EXECUTION_VENUE=%s → paper fills",
                settings.stream_execution_venue,
            )
        return None
    from stream.bybit_demo import BybitDemoVenue

    if venue_cls is None:
        venue_cls = BybitDemoVenue
    try:
        venue = venue_cls(
            settings.demo_api_key or "",
            settings.demo_api_secret or "",
            recv_window_ms=settings.bybit_demo_recv_window_ms,
            maker_first=settings.stream_execution_maker_first,
            maker_first_exit=settings.stream_execution_maker_first_exit,
        )
        equity = venue.balance()
    except Exception:
        logger.exception(
            "bybit demo venue could not be initialized (network/credential "
            "problem) → falling back to paper fills"
        )
        return None
    if equity is None:
        logger.warning(
            "demo keys present but api-demo.bybit.com rejected the balance "
            "read (ErrCode 10003 = key/domain mismatch; keys must be created "
            "under Demo Trading, not mainnet) → falling back to paper fills"
        )
        return None
    logger.info(
        "execution venue = bybit-demo (keys found in .env; virtual equity $%s)",
        equity,
    )
    return venue


def main() -> None:
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    venue = _build_venue(settings)
    # Trade the prediction prefix of the LIVE strategy. When STREAM_STRATEGY=asym
    # the signal writes to prediction:crypto:asym:1h; execution must read that
    # prefix to fill the FAS book. The execution ledger stays on the shared
    # execution:crypto:1h key so the dashboard view is unchanged.
    prediction_prefix = (
        settings.stream_asym_prediction_prefix
        if settings.stream_strategy == "asym"
        else settings.stream_redis_prediction_prefix
    )
    simulator = PaperExecutionSimulator(
        kv,
        execution_prefix=settings.stream_redis_execution_prefix,
        prediction_prefix=prediction_prefix,
        notional_usd=settings.stream_execution_notional_usd,
        slippage_bps=settings.stream_execution_slippage_bps,
        taker_fee_bps=settings.stream_execution_taker_fee_bps,
        ledger_maxlen=settings.stream_execution_ledger_maxlen,
        max_trades=settings.stream_execution_max_trades,
        window_ms=settings.stream_window_ms,
        venue=venue,
        cost_filter_lambda=settings.stream_execution_cost_filter_lambda,
        hold_until_decay=settings.stream_execution_hold_until_decay,
        # 0 => expire with the signal's own rebalance horizon.
        max_hold_h=(
            settings.stream_execution_max_hold_h or settings.stream_xs_rebalance_h
        ),
        signal_max_stale_windows=settings.stream_execution_signal_max_stale_windows,
    )

    # Reconcile in-memory positions with the ledger so positions opened by a
    # prior run (e.g. the one-shot live_book_runner) are managed live instead
    # of orphaned. Without this a restarted engine has no memory of open Bybit
    # Demo positions and would open duplicates on the next signal.
    universe = csv_list(settings.stream_xs_universe)
    for sym in universe:
        ledger_pos = kv.get_json(execution_key(settings.stream_redis_execution_prefix, sym))
        # Restore the last-seen window so the entry gate (window_end >= prev_end
        # + window_ms) can fire on the very next window instead of waiting ~2
        # windows for in-memory state to rebuild after a restart. Without this a
        # restart silently knocks every FLAT symbol out of the book for 1-2
        # windows (it can never re-open until prev_end is seeded), leaving half
        # the intended positions dark after every deploy/restart.
        if ledger_pos and isinstance(ledger_pos.get("window_end_ms"), int):
            simulator._last_window_end[sym] = ledger_pos["window_end_ms"]
        pos = ledger_pos.get("position") if ledger_pos else None
        if pos and pos.get("side") in ("LONG", "SHORT"):
            simulator._position[sym] = {
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "qty": pos["qty"],
                "entry_fees": pos.get("entry_fees", 0.0),
                "entry_window_end_ms": pos.get("entry_window_end_ms", 0),
            }
            logger.info(
                "execution reconciled open position %s %s @ %.6f",
                sym,
                pos["side"],
                pos["entry_price"],
            )

        # Restore cumulative track-record stats too. Without this a restart
        # (crash, deploy, reboot -- not just an open position) silently zeroes
        # n_trades/realized_pnl/fills even though nothing about the book
        # actually changed; the only thing previously reconciled was the open
        # position, so every restart looked like a fresh book with no history.
        if ledger_pos:
            n_trades = ledger_pos.get("n_trades")
            if isinstance(n_trades, int) and n_trades > 0:
                simulator._n_trades[sym] = n_trades
                simulator._n_wins[sym] = ledger_pos.get("n_wins", 0) or 0
                simulator._realized_pnl[sym] = float(ledger_pos.get("realized_pnl") or 0.0)
                simulator._gross_pnl[sym] = float(ledger_pos.get("gross_pnl") or 0.0)
                simulator._gross_volume[sym] = float(ledger_pos.get("gross_volume") or 0.0)
                simulator._total_fees[sym] = float(ledger_pos.get("total_fees") or 0.0)
                equity = ledger_pos.get("equity")
                if isinstance(equity, list) and equity:
                    simulator._equity[sym] = [float(e) for e in equity]
                fills = ledger_pos.get("fills")
                if isinstance(fills, list) and fills:
                    simulator._ledger[sym] = deque(fills, maxlen=None)
                simulator._signals_skipped[sym] = ledger_pos.get("signals_skipped", 0) or 0
                simulator._orders_rejected[sym] = ledger_pos.get("orders_rejected", 0) or 0
                logger.info(
                    "execution reconciled track record %s: n_trades=%d realized_pnl=%.2f",
                    sym,
                    n_trades,
                    simulator._realized_pnl[sym],
                )

    # Reconcile the OTHER direction: venue → book. The loop above only walks
    # what the online store already knows about, so any position the store
    # lost (Redis flush, fresh deploy, a one-shot live_book_runner run) stays
    # open on the exchange, unmanaged and invisible, accumulating duplicate
    # exposure -- the mess close_orphans.py exists to sweep up by hand.
    # Adopting them puts them back under the state machine so they are marked
    # to market and closed normally, and their exit lands in the ledger.
    if venue is not None and hasattr(venue, "open_positions"):
        for vpos in venue.open_positions():
            sym = str(vpos.get("symbol") or "").upper()
            if not sym or simulator._position.get(sym) is not None:
                continue  # already tracked -- the store's view wins
            entry_price = float(vpos.get("entry_price") or 0.0)
            if entry_price <= 0.0:
                logger.warning("execution: cannot adopt %s (no venue entry price)", sym)
                continue
            simulator._position[sym] = {
                "side": vpos["side"],
                "entry_price": entry_price,
                "qty": float(vpos["qty"]),
                # Entry fees were paid on a fill this process never saw; 0.0
                # understates cost slightly but never invents a number, and
                # the exit leg's real fee is still charged on close.
                "entry_fees": 0.0,
                # Unknown entry window ⇒ 0, so the one-window exit rule fires
                # on the next bar rather than holding an orphan indefinitely.
                "entry_window_end_ms": 0,
            }
            logger.warning(
                "execution adopted ORPHAN venue position %s %s qty=%s @ %.6f "
                "(open on the exchange but absent from the book) — will be managed and closed",
                sym,
                vpos["side"],
                vpos["qty"],
                entry_price,
            )

    logger.info(
        "execution consuming %s → %s (strategy=%s, prediction_prefix=%s, venue=%s, "
        "notional=$%.0f, slip=%sbps, taker=%sbps, lambda=%s, hold_until_decay=%s)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_execution_prefix,
        settings.stream_strategy,
        prediction_prefix,
        "bybit-demo" if venue is not None else "paper",
        settings.stream_execution_notional_usd,
        settings.stream_execution_slippage_bps,
        settings.stream_execution_taker_fee_bps,
        settings.stream_execution_cost_filter_lambda,
        settings.stream_execution_hold_until_decay,
    )
    try:
        simulator.run_forever(
            bus,
            settings.stream_kafka_topic_features,
            group_id="paper-execution",
        )
    except KeyboardInterrupt:
        logger.info("execution simulator stopped")


if __name__ == "__main__":
    main()
