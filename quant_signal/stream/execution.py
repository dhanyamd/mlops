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

import threading
from collections import deque
from collections.abc import Mapping
from typing import Protocol

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV
from stream.predictor import prediction_key

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

    def _open(self, symbol: str, side: str, close: float, window_end: int) -> None:
        if self._venue is not None:
            fill = self._venue.open_market(symbol, side, self._notional)
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
            }
            return
        entry_price = self._fill_price(side, close)
        qty = self._notional / entry_price
        entry_fee = self._notional * self._taker_fee
        self._position[symbol] = {
            "side": side,
            "entry_price": _round(entry_price),
            "qty": _round(qty, 8),
            "entry_fees": _round(entry_fee, 4),
            "entry_window_end_ms": window_end,
        }

    def _close(self, symbol: str, position: dict, close: float, window_end: int) -> None:
        side = position["side"]
        if self._venue is not None:
            fill = self._venue.close_market(symbol, side, position["qty"])
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
        bars_held = max(1, round((window_end - position["entry_window_end_ms"]) / self._window_ms))

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

        fills = self._ledger.setdefault(symbol, deque())
        fills.appendleft(
            {
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
            }
        )
        while len(fills) > self._ledger_maxlen:
            fills.pop()

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

            if position is not None:
                flip_side = None
                if signal == "SHORT" and position["side"] == "LONG":
                    flip_side = "SHORT"
                elif signal == "LONG" and position["side"] == "SHORT":
                    flip_side = "LONG"
                if flip_side is not None and yhat is not None and abs(yhat) > self._flip_threshold:
                    self._close(symbol, position, close, window_end)
                    if self._n_trades.get(symbol, 0) < self._max_trades:
                        self._open(symbol, flip_side, close, window_end)
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
    simulator = PaperExecutionSimulator(
        kv,
        execution_prefix=settings.stream_redis_execution_prefix,
        prediction_prefix=settings.stream_redis_prediction_prefix,
        notional_usd=settings.stream_execution_notional_usd,
        slippage_bps=settings.stream_execution_slippage_bps,
        taker_fee_bps=settings.stream_execution_taker_fee_bps,
        ledger_maxlen=settings.stream_execution_ledger_maxlen,
        max_trades=settings.stream_execution_max_trades,
        window_ms=settings.stream_window_ms,
        venue=venue,
        cost_filter_lambda=settings.stream_execution_cost_filter_lambda,
        hold_until_decay=settings.stream_execution_hold_until_decay,
    )
    logger.info(
        "execution consuming %s → %s (venue=%s, notional=$%.0f, slip=%sbps, taker=%sbps, "
        "lambda=%s, hold_until_decay=%s)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_execution_prefix,
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
