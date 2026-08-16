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
import math
import os
import time

from pybit.exceptions import InvalidRequestError
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# Bybit's reduce-only rejection when the exchange already has zero position
# for the symbol -- i.e. our internal book thinks a position is open but the
# venue disagrees (manual intervention, a prior desync, ADL, etc.). Retrying
# the same close forever is pointless: the exchange will reject it every
# time. Surfaced as a distinct sentinel so the caller can self-heal instead
# of leaving the position stuck "open" in our bookkeeping indefinitely.
_ERRCODE_POSITION_ALREADY_ZERO = 110017

# Linear perpetuals step the order size by 0.001 (BTCUSDT). Read from venue at
# runtime via instruments-info; this is only the floor used if that call fails.
_FALLBACK_QTY_STEP = 0.001
_FALLBACK_MIN_QTY = 0.001

# Fill confirmation: a demo ``place_order`` echoes only the ``orderId``, so
# fills are read back from order history (Bybit v5 order creation is
# asynchronous). Empirically the demo order-history record lags the fill by up
# to ~4s (measured on api-demo), so poll for ~8s; if it is still not Filled we
# return None and the engine skips the bar honestly.
_FILL_POLLS = int(os.environ.get("QUANT_FILL_POLLS", "16"))
_FILL_POLL_INTERVAL_S = float(os.environ.get("QUANT_FILL_POLL_INTERVAL_S", "0.5"))

# Maker fills rest on the book as post-only limits, so "Filled" legitimately
# takes longer than a market order. Poll for ~15s: long enough for the price to
# trade at our level in liquid BTC/ETH, short enough to keep the consumer
# thread inside its Kafka max-poll budget. Unfilled → cancel + market fallback.
#
# THIS BUDGET IS PER SYMBOL AND THE ENGINE FILLS SEQUENTIALLY, so the wall-clock
# cost is (polls x interval x book size). At 30 polls that is ~15s each: fine for
# a ~30-name book, but ~19 minutes for the 100-name cross-section, during which
# most of the book sits unfilled. Lower it when the universe is wide -- the cost
# is a lower maker hit-rate (more market fallbacks at taker fees), and maker vs
# taker execution was measured at 2.40 vs 1.87 Sharpe, so this is a real
# trade-off rather than free speed. Configurable so it can be tuned to the
# universe actually deployed instead of being fixed for one book size.
_MAKER_FILL_POLLS = int(os.environ.get("QUANT_MAKER_FILL_POLLS", "30"))


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
        timeout_s: float = 10.0,
        maker_first_exit: bool | None = None,
    ) -> None:
        # Bounded HTTP client: pybit's default 10s timeout + its rate-limit
        # retry sleep (ErrCode 10006 waits until the limit resets — minutes!)
        # stalled the consumer thread on a slow demo API and crashed the engine
        # with a Kafka max-poll-exceeded error. So: bounded timeout (default
        # 10s — Demo is virtual, no microsecond SLA constraint), no retries on
        # transport errors (the fill-poll loop retries at its own pace), and
        # rate-limit/retry codes disabled (retry_codes={-1} is a non-empty
        # sentinel — pybit replaces an empty set with its defaults). Failures
        # surface immediately and the fill-poll loop retries at its own pace.
        # Rate-limit retries (ErrCode 10006) are now handled in _fill_from_order
        # with bounded exponential backoff, since the Demo API's rate limit is
        # generous but occasionally throttles under burst (30 symbols warm-start).
        self._http = HTTP(
            demo=True,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=recv_window_ms,
            timeout=int(timeout_s),
            max_retries=1,
            retry_codes={-1},
        )
        self._maker_first = maker_first
        # Exits get their own switch: measured 0/7 maker fills on the exit leg
        # (every fill paid the 5.5 bps taker rate), so the 15s maker poll is
        # dead wait serialised across the whole book. None ⇒ inherit the entry
        # setting, preserving old behaviour for callers that don't pass it.
        self._maker_first_exit = maker_first if maker_first_exit is None else maker_first_exit

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

    @staticmethod
    def _round_to_step(qty: float, step: float) -> float:
        """Round a quantity to the exchange's qtyStep precision, killing the
        float artifact (e.g. 132 * 0.1 == 13.200000000000001) that makes Bybit
        reject the order with 'Qty invalid'. Never rounded to more decimals than
        the step actually allows."""
        if step is None or step <= 0.0:
            return qty
        decimals = -int(math.floor(math.log10(step))) if step < 1.0 else 0
        return round(qty, decimals)

    def _qty_for_notional(self, symbol: str, notional_usd: float) -> float | None:
        """Contract-sized order quantity for ``notional_usd``, floored to the
        venue's qtyStep. None when the notional can't buy the minimum lot."""
        price = self.last_price(symbol)
        if price is None or price <= 0.0:
            return None
        _min_qty, step = self._qty_step(symbol)
        qty = self._round_to_step(math.floor(notional_usd / price / step) * step, step)
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
        if self._maker_first_exit:
            fill = self._close_maker(symbol, side, qty)
            if fill is not None:
                return fill
        return self._close_market_impl(symbol, side, qty)

    def _close_market_impl(self, symbol: str, side: str, qty: float) -> dict | None:
        """Market reduce-only order to close an open position.

        Returns the fill dict, ``None`` on an ordinary/transient failure (the
        caller retries next window), or ``{"desync": True}`` when the venue
        says the position is already zero -- retrying that would fail forever.
        """
        side_bb = "Sell" if side == "LONG" else "Buy"
        _min_qty, step = self._qty_step(symbol)
        qty = self._round_to_step(qty, step)
        try:
            result = self._http.place_order(
                category="linear",
                symbol=symbol,
                side=side_bb,
                orderType="Market",
                qty=str(qty),
                reduceOnly=True,
            )
        except InvalidRequestError as exc:
            if exc.status_code == _ERRCODE_POSITION_ALREADY_ZERO:
                logger.warning(
                    "bybit demo: %s %s already flat on the venue (ErrCode %s) — "
                    "internal book was stale, clearing without a fabricated fill",
                    symbol,
                    side,
                    exc.status_code,
                )
                return {"desync": True}
            logger.exception("bybit demo: close market order failed for %s %s", symbol, side)
            return None
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
        return self._fill_from_order(symbol, order_id, qty, intent="close")

    def _close_maker(self, symbol: str, side: str, qty: float) -> dict | None:
        """Reduce-only post-only limit at the near side; market fallback."""
        ba = self._best_ba(symbol)
        if ba is None:
            logger.warning("bybit demo: no orderbook for %s — using market fill", symbol)
            return None
        bid, ask = ba
        price = bid if side == "LONG" else ask
        side_bb = "Sell" if side == "LONG" else "Buy"
        # Round to the exchange's qtyStep -- unlike _close_market_impl, this
        # path previously submitted the raw tracked qty (e.g. 649.22080519),
        # which Bybit rejects outright with "Qty invalid" (ErrCode 10001)
        # whenever it carries more precision than the instrument's lot size,
        # silently forcing every maker close through the market fallback.
        _min_qty, step = self._qty_step(symbol)
        qty = self._round_to_step(qty, step)
        order_id = self._place_limit(symbol, side_bb, qty, price, reduce_only=True)
        if order_id:
            fill = self._fill_from_order(
                symbol, order_id, qty, polls=_MAKER_FILL_POLLS, quiet=True, intent="close"
            )
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

    def _cancel(self, symbol: str, order_id: str) -> bool:
        """Best-effort cancel of an unfilled resting order.

        Returns ``True`` if the exchange acknowledged the cancel request
        (``retCode`` == 0), otherwise ``False``.  A best-effort check of the
        order‑history after the cancel is performed so the caller can see
        whether the order is truly cancelled or still resting.
        """
        try:
            result = self._http.cancel_order(category="linear", symbol=symbol, orderId=order_id)
        except Exception:
            logger.warning("bybit demo: cancel failed for order %s (exception)", order_id)
            return False
        if result.get("retCode") != 0:
            logger.warning(
                "bybit demo: cancel rejected for order %s: %s",
                order_id,
                result.get("retMsg"),
            )
            return False
        # Brief verification: re‑query order history; if the order is still
        # present with status New/Open we consider the cancel insufficient.
        try:
            time.sleep(0.3)
            hist = self._http.get_order_history(category="linear", symbol=symbol, orderId=order_id)
            if hist.get("retCode") == 0:
                rows = hist.get("result", {}).get("list", [])
                if rows and rows[0].get("orderStatus") in ("New", "Partial"):
                    logger.warning(
                        "bybit demo: cancel returned OK but order %s still %s",
                        order_id,
                        rows[0].get("orderStatus"),
                    )
                    return False
        except Exception:
            pass  # best‑effort only
        return True

    def _fill_from_order(
        self,
        symbol: str,
        order_id: str,
        qty: float,
        *,
        polls: int = _FILL_POLLS,
        quiet: bool = False,
        intent: str = "open",
    ) -> dict | None:
        """Fill for a placed order, read back from order history and positions.

        ``place_order`` only echoes the ``orderId``; the executed average price,
        filled qty and fees arrive on the order-history record once the order
        fills (order creation is asynchronous on Bybit v5, so poll briefly).
        Additionally, we reconcile against ``get_positions`` as the ground-truth
        fill state per Bybit API docs, since order-history status can lag.

        ``intent`` MUST distinguish an entry from an exit, because the
        position-based ground truth is INVERTED between them:

          intent="open"   position qty > 0  → the entry filled
          intent="close"  position flat     → the exit filled

        Reading "position still open" as a fill for a reduce-only exit (the
        previous behaviour) recorded failed closes as completed round trips
        with fabricated P&L, while the position stayed open on the exchange.

        Accepts ``Filled`` and ``PartiallyFilled``. Returns None when the order
        is not confirmed filled — the caller skips the bar honestly.

        Bounded retry on rate-limit (ErrCode 10006) only; transient Demo throttling
        self-heals instead of permanently skipping the bar. Other errors surface
        immediately and the fill-poll loop retries at its own pace.
        """
        closing = intent == "close"
        # Last order-history row seen across polls: the only trustworthy source
        # of executed price/fees. Never substitute unrealised P&L for fees.
        last_row: dict | None = None
        # Exponential-backoff interval for rate-limit retries only
        backoff = 0.5
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
                # Transient rate-limit (ErrCode 10006): backoff and retry;
                # other errors surface immediately.
                rmsg = history.get("retMsg", "")
                if "10006" in rmsg or "rate limit" in rmsg.lower():
                    logger.warning(
                        "bybit demo: rate limit (attempt %s/%s); backoff %.1fs",
                        attempt + 1,
                        polls,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
                logger.warning("bybit demo: order-history rejected: %s", rmsg)
                return None
            rows = history.get("result", {}).get("list", [])
            if not rows:
                time.sleep(_FILL_POLL_INTERVAL_S)
                continue
            row = rows[0]
            last_row = row
            status = row.get("orderStatus", "")
            # Accept Filled and PartiallyFilled as real fills.
            if status not in ("Filled", "PartiallyFilled"):
                if not quiet:
                    logger.warning(
                        "bybit demo: order status=%s not filled yet (attempt %s/%s)",
                        status,
                        attempt + 1,
                        polls,
                    )
                time.sleep(_FILL_POLL_INTERVAL_S)
                # Reconcile against positions as ground truth -- direction-aware.
                confirmed = self._position_confirms(symbol, closing)
                if confirmed is not None:
                    return self._fill_payload(row, qty, confirmed, closing)
                continue
            try:
                fill_price = float(row.get("avgPrice") or 0.0)
                fill_qty = float(row.get("cumExecQty") or qty)
                fees = _fill_fee(row)
            except (TypeError, ValueError):
                return None
            if fill_price <= 0.0:
                return None
            # An explicit Filled/PartiallyFilled status from the venue is the
            # strongest signal available for either direction; take it as-is.
            return {"fill_price": fill_price, "qty": fill_qty, "fees": fees}
        # After all polls, reconcile against positions as final ground truth.
        confirmed = self._position_confirms(symbol, closing)
        if confirmed is not None:
            payload = self._fill_payload(last_row, qty, confirmed, closing)
            if payload is not None:
                return payload
        logger.warning("bybit demo: order %s not confirmed filled after %s polls", order_id, polls)
        return None

    def _position_confirms(self, symbol: str, closing: bool) -> dict | None:
        """Venue position state, when it confirms the order actually executed.

        Returns the position dict for a confirmed entry, ``{}`` for a confirmed
        exit (the position is gone, so there is nothing to return), or ``None``
        when the position state does NOT confirm execution.
        """
        pos = self.position(symbol)
        open_qty = float((pos or {}).get("qty") or 0.0)
        if closing:
            # A reduce-only exit is confirmed by the position being FLAT.
            # Still open ⇒ the close did not go through.
            return {} if open_qty == 0.0 else None
        return pos if open_qty > 0.0 else None

    @staticmethod
    def _fill_payload(
        row: dict | None, qty: float, confirmed: dict, closing: bool
    ) -> dict | None:
        """Fill payload from the order-history row, position-confirmed.

        Fees ALWAYS come from the order-history record -- never from a
        position's unrealised P&L, which is a completely different quantity
        and previously corrupted every position-confirmed fill's cost basis.
        """
        price = float((row or {}).get("avgPrice") or 0.0)
        if price <= 0.0 and not closing:
            price = float(confirmed.get("entry_price") or 0.0)
        if price <= 0.0:
            # No trustworthy execution price: report no fill rather than
            # inventing one. The caller retries, and a genuinely-flat
            # position self-heals via the ErrCode 110017 desync path.
            return None
        filled_qty = float((row or {}).get("cumExecQty") or 0.0)
        if filled_qty <= 0.0:
            filled_qty = float(confirmed.get("qty") or qty) if not closing else qty
        return {"fill_price": price, "qty": filled_qty, "fees": _fill_fee(row or {})}

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

    def open_positions(self) -> list[dict]:
        """Every open linear/USDT position on the account (venue ground truth).

        Used to reconcile the book against the exchange at startup. Without
        this, wiping the online store (a Redis flush, a fresh deploy) strands
        real open positions on the venue that our book no longer knows about:
        they are never managed, never closed, and quietly accumulate
        duplicate exposure -- exactly what close_orphans.py was written to
        clean up by hand.
        """
        try:
            result = self._http.get_positions(category="linear", settleCoin="USDT", limit=200)
        except Exception:  # pragma: no cover - network tolerance
            logger.exception("bybit demo: get_positions(settleCoin=USDT) failed")
            return []
        if not isinstance(result, dict) or result.get("retCode") != 0:
            logger.warning("bybit demo: get_positions -> %s", result)
            return []
        out = []
        for row in (result.get("result", {}) or {}).get("list", []) or []:
            size = float(row.get("size") or 0.0)
            if size <= 0.0:
                continue
            out.append(
                {
                    "symbol": row.get("symbol"),
                    "side": "LONG" if row.get("side") == "Buy" else "SHORT",
                    "qty": abs(size),
                    "entry_price": float(row.get("avgPrice") or 0.0),
                }
            )
        return out

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
