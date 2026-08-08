"""Paper execution engine: simulated fills on the predictor's real signals.

Turns the predictor's realized directions into filled trades the way a real
execution stack would (FLOX SimulatedExecutor / ordersim / Sequence pattern):

  - A signal made at window ``t`` (the predictor's stored direction) is filled
    as a market order at the *next* window's close price — never the signal
    bar — so the simulation is honest about latency (no lookahead).
  - The fill pays the adverse side of a fixed-bps spread (``slippage_bps``) and
    a taker fee (``taker_fee_bps``) on both entry and exit.
  - Positions hold one window and roll continuously: each bar closes the
    position opened a window earlier, then opens the new one if a fresh signal
    is available.
  - A fill ledger and per-symbol P&L (realized / unrealized, compounded equity,
    win rate) land in the online store as ``execution:crypto:5m:<SYMBOL>``.

Fills are deterministic by construction (next-close price + fixed bps), so there
is no PRNG to seed; ``window_end_ms`` is echoed in every payload as the
event-time audit tag, matching the rest of the streaming layer.
"""

from __future__ import annotations

import threading
from collections import deque

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV
from stream.predictor import prediction_key

logger = get_logger(__name__)

# 5m window length in ms — the fill/exit cadence for the paper book.
_WINDOW_MS = 300_000

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
    """Per-symbol simulated fills fed by the 5m feature stream.

    State machine per symbol, driven one bar at a time:

      exit   position entered a window ago (``entry_window_end_ms + WINDOW_MS``)
             closes at the current bar's close, paying slippage + exit fee
      entry  no open position + a tradeable signal from the *previous* bar →
             market order at the current bar's close (next-bar fill), slippage
             + entry fee
      roll   the current bar's signal is stored for the next bar's fill

    The signal for a bar is read from the predictor's online-store payload
    matched by that bar's ``window_end_ms``; if the store has not caught up, no
    entry happens that bar (a real book would skip a stale signal too).
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

    # ── signal resolution ───────────────────────────────────────────────────

    def _signal_for(self, symbol: str, window_end: int | None) -> str | None:
        """Direction the predictor held for ``window_end``, or None if unknown."""
        if window_end is None:
            return None
        prediction = self._kv.get_json(prediction_key(self._prediction_prefix, symbol))
        if not prediction:
            return None
        tag = prediction.get("window_end_ms")
        if not isinstance(tag, (int, float)) or int(tag) != window_end:
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
        # Closing is the opposite leg: a LONG sells (receives less), a SHORT
        # buys back (pays more), so the exit fill pays the adverse side too.
        exit_side = "SHORT" if side == "LONG" else "LONG"
        exit_price = self._fill_price(exit_side, close)
        qty = position["qty"]
        exit_fee = qty * exit_price * self._taker_fee
        fees = position["entry_fees"] + exit_fee
        gross = (
            qty * (exit_price - position["entry_price"])
            if side == "LONG"
            else qty * (position["entry_price"] - exit_price)
        )
        net = gross - fees
        pnl_pct = net / self._notional

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
                "bars_held": 1,
            }
        )
        while len(fills) > self._ledger_maxlen:
            fills.pop()

    def _advance(self, symbol: str, close: float, window_end: int, prev_end: int | None) -> None:
        """One bar of the state machine (exit → enter → record signal bar)."""
        position = self._position.get(symbol)

        if position is not None and window_end >= position["entry_window_end_ms"] + self._window_ms:
            self._close(symbol, position, close, window_end)
            position = None

        if position is None:
            signal = self._signal_for(symbol, prev_end)
            if (
                signal in ("LONG", "SHORT")
                and prev_end is not None
                and window_end >= prev_end + self._window_ms
                and self._n_trades.get(symbol, 0) < self._max_trades
            ):
                self._open(symbol, signal, close, window_end)
            elif (
                signal is None and prev_end is not None and window_end >= prev_end + self._window_ms
            ):
                # No prediction for the signal bar (online store lagged) → the
                # entry is skipped, exactly as a live book would skip a stale
                # signal. Counted for observability, not treated as an error.
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
        return {
            "symbol": symbol,
            "window_end_ms": window_end,
            "notional_usd": self._notional,
            "slippage_bps": round(self._slippage * 10_000, 2),
            "taker_fee_bps": round(self._taker_fee * 10_000, 2),
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
            "total_return": round(equity[-1] - 1.0, 6),
            "equity": [round(e, 6) for e in equity],
            "position": position_view,
            "fills": list(self._ledger.get(symbol, deque())),
            "assumptions": {
                "fill_timing": (
                    "signal at window t, market fill at window t+1 close, exit at window t+2 close"
                ),
                "slippage_bps": round(self._slippage * 10_000, 2),
                "taker_fee_bps": round(self._taker_fee * 10_000, 2),
                "cost_model": (
                    "adverse-side fill (pay spread on buy, give spread on sell) "
                    "+ taker fee both legs"
                ),
                "deterministic": "fills are next-close + fixed bps; no PRNG, no hidden randomness",
                "not_modeled": "margin, funding, partial fills, queue position, and market impact",
            },
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


def main() -> None:
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
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
    )
    logger.info(
        "execution consuming %s → %s (notional=$%.0f, slip=%sbps, taker=%sbps)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_execution_prefix,
        settings.stream_execution_notional_usd,
        settings.stream_execution_slippage_bps,
        settings.stream_execution_taker_fee_bps,
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
