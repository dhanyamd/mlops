"""Bybit Demo execution venue: real market orders on virtual funds.

Wraps pybit's unified-trading HTTP client pointed at Bybit's free Demo Trading
environment (``api-demo.bybit.com``). A Demo account is auto-credited virtual
USDT/BTC on signup — no deposit, no KYC — and fills market orders against the
live order book, so the execution book sees real fill prices, fees, latency and
rejections instead of a fixed-bps model.

Research-backed constraints (Bybit v5 Open API Demo docs; pybit issue #203):

  - ``pybit HTTP(demo=True)`` routes to ``api-demo.bybit.com`` and expects keys
    created under the *Demo* account; mainnet keys fail with ErrCode 10003.
  - WebSocket Trade is NOT supported on the demo venue, so fills are read back
    over REST: ``place_order`` echoes only the ``orderId``, so each fill is
    confirmed by polling ``get_order_history`` (``avgPrice`` / ``cumExecQty`` /
    ``cumExecFee``/``cumFeeDetail``) and reconciled against the open-position
    list each window.
  - The HTTP client is bounded (3s timeout, no retries, retry codes disabled):
    a rate-limited demo API must fail fast — never block the consumer thread —
    or the Kafka consumer exceeds its max-poll interval and the engine dies.
  - USDT perpetuals (category "linear", one-way mode) support both LONG and
    SHORT — matching the paper engine's position model — with a lot-size step
    read from the instrument info (``lotSizeFilter.qtyStep``), never hardcoded.

Every method is defensive: it returns ``None`` (or raises a clear error) rather
than fabricating a fill, so the caller can skip the bar honestly.
"""

from __future__ import annotations

import logging
import time

from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# Linear perpetuals step the order size by 0.001 (BTCUSDT). Read from venue at
# runtime via instruments-info; this is only the floor used if that call fails.
_FALLBACK_QTY_STEP = 0.001
_FALLBACK_MIN_QTY = 0.001

# Fill confirmation: a demo ``place_order`` echoes only the ``orderId``, so
# fills are read back from order history (Bybit v5 order creation is
# asynchronous). Empirically the demo order-history record lags the fill by up
# to ~4s (measured on api-demo), so poll for ~8s; if it is still not Filled we
# return None and the engine skips the bar honestly.
_FILL_POLLS = 16
_FILL_POLL_INTERVAL_S = 0.5

# Maker fills rest on the book as post-only limits, so "Filled" legitimately
# takes longer than a market order. Poll for ~15s: long enough for the price to
# trade at our level in liquid BTC/ETH, short enough to keep the consumer
# thread inside its Kafka max-poll budget. Unfilled → cancel + market fallback.
_MAKER_FILL_POLLS = 30


def _fill_fee(row: dict) -> float:
    """Maker/taker fee on a linear order history record.

    ``cumFeeDetail`` supersedes ``cumExecFee`` for linear/spot (Bybit v5 docs);
    sum its coin values and fall back to ``cumExecFee`` when it is absent.
    """
    detail = row.get("cumFeeDetail")
    if isinstance(detail, dict):
        try:
            return sum(float(v) for v in detail.values() if v)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(row.get("cumExecFee") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class BybitDemoVenue:
    """REST client for Bybit's free Demo Trading environment (virtual funds)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        recv_window_ms: int = 5000,
        maker_first: bool = True,
    ) -> None:
        # Bounded HTTP client: pybit's default 10s timeout + its rate-limit
        # retry sleep (ErrCode 10006 waits until the limit resets — minutes!)
        # stalled the consumer thread on a slow demo API and crashed the engine
        # with a Kafka max-poll-exceeded error. So: 3s per request, no retries,
        # and rate-limit/retry codes disabled (retry_codes={-1} is a non-empty
        # sentinel — pybit replaces an empty set with its defaults). Failures
        # surface immediately and the fill-poll loop retries at its own pace.
        self._http = HTTP(
            demo=True,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=recv_window_ms,
            timeout=3,
            max_retries=1,
            retry_codes={-1},
        )
        self._maker_first = maker_first

    # ── market data helpers ─────────────────────────────────────────────────

    def _best_ba(self, symbol: str) -> tuple[float, float] | None:
        """(best bid, best ask) from a depth-1 orderbook snapshot, or None."""
        try:
            result = self._http.get_orderbook(category="linear", symbol=symbol, limit=1)
            if not result.get("retCode") == 0:
                return None
            rows = result.get("result", {}) or {}
            bids = rows.get("b") or []
            asks = rows.get("a") or []
            if not bids or not asks:
                return None
            return float(bids[0][0]), float(asks[0][0])
        except Exception:  # pragma: no cover - network tolerance
            logger.exception("bybit demo: get_orderbook failed")
            return None

    def _qty_step(self, symbol: str) -> tuple[float, float]:
        """(min_order_qty, qty_step) for a linear perp, from the instrument
        contract — the exchange's own precision, not a hardcoded value."""
        try:
            result = self._http.get_instruments_info(category="linear", symbol=symbol)
            if not result.get("retCode") == 0:
                return _FALLBACK_MIN_QTY, _FALLBACK_QTY_STEP
            rows = result.get("result", {}).get("list", [])
            if not rows:
                return _FALLBACK_MIN_QTY, _FALLBACK_QTY_STEP
            lot = rows[0].get("lotSizeFilter", {})
            return (
                float(lot.get("minOrderQty") or _FALLBACK_MIN_QTY),
                float(lot.get("qtyStep") or _FALLBACK_QTY_STEP),
            )
        except Exception:  # pragma: no cover - network tolerance
            logger.warning("bybit demo: instruments-info failed; using fallback qty step")
            return _FALLBACK_MIN_QTY, _FALLBACK_QTY_STEP

    def _qty_for_notional(self, symbol: str, notional_usd: float) -> float | None:
        """Contract-sized order quantity for ``notional_usd``, floored to the
        venue's qtyStep. None when the notional can't buy the minimum lot."""
        price = self.last_price(symbol)
        if price is None or price <= 0.0:
            return None
        _min_qty, step = self._qty_step(symbol)
        qty = float(int(notional_usd / price / step) * step)
        if qty < _min_qty:
            logger.warning(
                "bybit demo: notional $%.0f < one lot of %s (min %s)",
                notional_usd,
                symbol,
                _min_qty,
            )
            return None
        return qty

    def last_price(self, symbol: str) -> float | None:
        try:
            result = self._http.get_tickers(category="linear", symbol=symbol)
            if not result.get("retCode") == 0:
                return None
            rows = result.get("result", {}).get("list", [])
            if not rows:
                return None
            return float(rows[0]["lastPrice"])
        except Exception:  # pragma: no cover - network tolerance
            logger.exception("bybit demo: get_tickers failed")
            return None

    # ── fills ───────────────────────────────────────────────────────────────

    def open_market(self, symbol: str, side: str, notional_usd: float) -> dict | None:
        """Open a position, preferring a maker fill when enabled (default).

        With ``maker_first`` the engine first rests a post-only limit at the
        near-side quote (bid for a LONG) and only falls back to a market order
        if the book never trades at that price within the maker poll window.
        Returns the actual fill (price/qty/fees) or None when no order filled
        (rejected/pending) — the caller skips the bar honestly.
        """
        if self._maker_first:
            fill = self._open_maker(symbol, side, notional_usd)
            if fill is not None:
                return fill
        return self._open_market_impl(symbol, side, notional_usd)

    def _open_market_impl(self, symbol: str, side: str, notional_usd: float) -> dict | None:
        """Market order to open a position."""
        qty = self._qty_for_notional(symbol, notional_usd)
        if qty is None:
            return None
        side_bb = "Buy" if side == "LONG" else "Sell"
        try:
            result = self._http.place_order(
                category="linear",
                symbol=symbol,
                side=side_bb,
                orderType="Market",
                qty=str(qty),
            )
        except Exception:
            logger.exception("bybit demo: open market order failed for %s %s", symbol, side)
            return None
        if result.get("retCode") != 0:
            logger.warning("bybit demo: open order rejected: %s", result.get("retMsg"))
            return None
        order_id = (result.get("result") or {}).get("orderId")
        if not order_id:
            logger.warning("bybit demo: open order returned no orderId; fill unconfirmable")
            return None
        return self._fill_from_order(symbol, order_id, qty)

    def _open_maker(self, symbol: str, side: str, notional_usd: float) -> dict | None:
        """Post-only limit at the near side of the book; market fallback.

        A resting limit pays Bybit's 0.02% maker fee (vs 0.055% taker) and
        fills at the better side of the spread — an 11 → 4 bps round trip
        (Bybit v5 fee schedule). PostOnly guarantees the order cannot cross,
        so it can never slip to a taker fee unintentionally. The order rests
        for the maker poll window; if the price never trades at our level the
        order is cancelled and a market order takes over, so momentum signals
        still participate.
        """
        qty = self._qty_for_notional(symbol, notional_usd)
        if qty is None:
            return None
        ba = self._best_ba(symbol)
        if ba is None:
            logger.warning("bybit demo: no orderbook for %s — using market fill", symbol)
            return None
        bid, ask = ba
        price = bid if side == "LONG" else ask
        side_bb = "Buy" if side == "LONG" else "Sell"
        order_id = self._place_limit(symbol, side_bb, qty, price, reduce_only=False)
        if order_id:
            fill = self._fill_from_order(symbol, order_id, qty, polls=_MAKER_FILL_POLLS, quiet=True)
            if fill is not None:
                return fill
            self._cancel(symbol, order_id)
        logger.warning(
            "bybit demo: maker entry not filled for %s (%s @ %s) → market fallback",
            symbol,
            side,
            price,
        )
        return None

    def close_market(self, symbol: str, side: str, qty: float) -> dict | None:
        """Close an open position, preferring a maker fill when enabled.

        None when nothing filled, so the caller keeps the position and retries
        next window.
        """
        if self._maker_first:
            fill = self._close_maker(symbol, side, qty)
            if fill is not None:
                return fill
        return self._close_market_impl(symbol, side, qty)

    def _close_market_impl(self, symbol: str, side: str, qty: float) -> dict | None:
        """Market reduce-only order to close an open position."""
        side_bb = "Sell" if side == "LONG" else "Buy"
        try:
            result = self._http.place_order(
                category="linear",
                symbol=symbol,
                side=side_bb,
                orderType="Market",
                qty=str(qty),
                reduceOnly=True,
            )
        except Exception:
            logger.exception("bybit demo: close market order failed for %s %s", symbol, side)
            return None
        if result.get("retCode") != 0:
            logger.warning("bybit demo: close order rejected: %s", result.get("retMsg"))
            return None
        order_id = (result.get("result") or {}).get("orderId")
        if not order_id:
            logger.warning("bybit demo: close order returned no orderId; fill unconfirmable")
            return None
        return self._fill_from_order(symbol, order_id, qty)

    def _close_maker(self, symbol: str, side: str, qty: float) -> dict | None:
        """Reduce-only post-only limit at the near side; market fallback."""
        ba = self._best_ba(symbol)
        if ba is None:
            logger.warning("bybit demo: no orderbook for %s — using market fill", symbol)
            return None
        bid, ask = ba
        price = bid if side == "LONG" else ask
        side_bb = "Sell" if side == "LONG" else "Buy"
        order_id = self._place_limit(symbol, side_bb, qty, price, reduce_only=True)
        if order_id:
            fill = self._fill_from_order(symbol, order_id, qty, polls=_MAKER_FILL_POLLS, quiet=True)
            if fill is not None:
                return fill
            self._cancel(symbol, order_id)
        logger.warning(
            "bybit demo: maker exit not filled for %s (%s @ %s) → market fallback",
            symbol,
            side,
            price,
        )
        return None

    def _place_limit(
        self, symbol: str, side_bb: str, qty: float, price: float, *, reduce_only: bool
    ) -> str | None:
        """Post-only limit order. Returns the orderId or None (rejected/errored).

        timeInForce=PostOnly (Bybit v5): if the limit would execute on arrival
        it is cancelled instead — guaranteeing a maker fill, never an
        accidental taker. The near-side bid/ask prices come from the orderbook
        so they always satisfy the venue's tickSize precision.
        """
        try:
            result = self._http.place_order(
                category="linear",
                symbol=symbol,
                side=side_bb,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="PostOnly",
                reduceOnly=reduce_only,
            )
        except Exception:
            logger.exception("bybit demo: post-only limit failed for %s %s", symbol, side_bb)
            return None
        if result.get("retCode") != 0:
            logger.warning(
                "bybit demo: post-only limit rejected (%s) → market fallback",
                result.get("retMsg"),
            )
            return None
        return (result.get("result") or {}).get("orderId")

    def _cancel(self, symbol: str, order_id: str) -> None:
        """Best-effort cancel of an unfilled resting order."""
        try:
            self._http.cancel_order(category="linear", symbol=symbol, orderId=order_id)
        except Exception:
            logger.warning("bybit demo: cancel failed for order %s", order_id)

    def _fill_from_order(
        self,
        symbol: str,
        order_id: str,
        qty: float,
        *,
        polls: int = _FILL_POLLS,
        quiet: bool = False,
    ) -> dict | None:
        """Fill for a placed order, read back from order history.

        ``place_order`` only echoes the ``orderId``; the executed average price,
        filled qty and fees arrive on the order-history record once the order
        fills (order creation is asynchronous on Bybit v5, so poll briefly).
        Returns None when the order is not confirmed Filled — the caller skips
        the bar honestly rather than guessing a price. ``quiet`` suppresses the
        per-poll warning for resting maker orders (still-New is the expected
        state there, not an error).
        """
        for attempt in range(polls):
            try:
                history = self._http.get_order_history(
                    category="linear", symbol=symbol, orderId=order_id
                )
            except Exception:
                logger.warning(
                    "bybit demo: order-history read failed (attempt %s/%s)",
                    attempt + 1,
                    polls,
                )
                time.sleep(_FILL_POLL_INTERVAL_S)
                continue
            if history.get("retCode") != 0:
                logger.warning("bybit demo: order-history rejected: %s", history.get("retMsg"))
                return None
            rows = history.get("result", {}).get("list", [])
            if not rows:
                time.sleep(_FILL_POLL_INTERVAL_S)
                continue
            row = rows[0]
            if row.get("orderStatus") != "Filled":
                if not quiet:
                    logger.warning(
                        "bybit demo: order not filled yet (status=%s)", row.get("orderStatus")
                    )
                time.sleep(_FILL_POLL_INTERVAL_S)
                continue
            try:
                fill_price = float(row.get("avgPrice") or 0.0)
                fill_qty = float(row.get("cumExecQty") or qty)
                fees = _fill_fee(row)
            except (TypeError, ValueError):
                return None
            if fill_price <= 0.0:
                return None
            return {"fill_price": fill_price, "qty": fill_qty, "fees": fees}
        logger.warning("bybit demo: order %s not confirmed filled after %s polls", order_id, polls)
        return None

    # ── account / position reconciliation ───────────────────────────────────

    def balance(self) -> float | None:
        """Total demo-account equity in USDT (virtual funds), or None."""
        try:
            result = self._http.get_wallet_balance(accountType="UNIFIED")
            if not result.get("retCode") == 0:
                return None
            wallets = result.get("result", {}).get("list", [])
            if not wallets:
                return None
            return float(wallets[0].get("totalEquity") or 0.0)
        except Exception:  # pragma: no cover - network tolerance
            logger.exception("bybit demo: get_wallet_balance failed")
            return None

    def position(self, symbol: str) -> dict | None:
        """Open linear position for ``symbol`` (one-way mode) or None."""
        try:
            result = self._http.get_positions(category="linear", symbol=symbol)
            if not result.get("retCode") == 0:
                return None
            rows = result.get("result", {}).get("list", [])
            for row in rows:
                size = float(row.get("size") or 0.0)
                if size == 0.0:
                    continue
                return {
                    "side": "LONG" if row.get("side") == "Buy" else "SHORT",
                    "qty": abs(size),
                    "entry_price": float(row.get("avgPrice") or 0.0),
                    "unrealized_pnl": float(row.get("unrealisedPnl") or 0.0),
                }
        except Exception:  # pragma: no cover - network tolerance
            logger.exception("bybit demo: get_positions failed")
            return None
        return None
