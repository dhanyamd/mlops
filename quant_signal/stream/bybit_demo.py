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
    over REST from the order response (``avgPrice`` / ``cumExecQty`` /
    ``cumExecFee``) and reconciled against the open-position list each window.
  - USDT perpetuals (category "linear", one-way mode) support both LONG and
    SHORT — matching the paper engine's position model — with a lot-size step
    read from the instrument info (``lotSizeFilter.qtyStep``), never hardcoded.

Every method is defensive: it returns ``None`` (or raises a clear error) rather
than fabricating a fill, so the caller can skip the bar honestly.
"""

from __future__ import annotations

import logging

from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# Linear perpetuals step the order size by 0.001 (BTCUSDT). Read from venue at
# runtime via instruments-info; this is only the floor used if that call fails.
_FALLBACK_QTY_STEP = 0.001
_FALLBACK_MIN_QTY = 0.001


class BybitDemoVenue:
    """REST client for Bybit's free Demo Trading environment (virtual funds)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        recv_window_ms: int = 5000,
    ) -> None:
        self._http = HTTP(
            demo=True,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=recv_window_ms,
        )

    # ── market data helpers ─────────────────────────────────────────────────

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
        """Market order to open a position. Returns the fill or None if the
        order did not fill immediately (rejected/pending) — the caller skips."""
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
        return self._fill_from_order(result, qty)

    def close_market(self, symbol: str, side: str, qty: float) -> dict | None:
        """Market reduce-only order to close an open position. None when not
        filled, so the caller keeps the position and retries next window."""
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
        return self._fill_from_order(result, qty)

    @staticmethod
    def _fill_from_order(result: dict, qty: float) -> dict | None:
        """Fill from a place-order response, or None if the market order did
        not fully execute immediately (it may be resting/pending)."""
        if not result.get("retCode") == 0:
            logger.warning("bybit demo: order rejected: %s", result.get("retMsg"))
            return None
        row = result.get("result", {})
        if row.get("orderStatus") != "Filled":
            logger.warning("bybit demo: order not filled yet (status=%s)", row.get("orderStatus"))
            return None
        try:
            fill_price = float(row.get("avgPrice") or 0.0)
            fill_qty = float(row.get("cumExecQty") or qty)
            fees = float(row.get("cumExecFee") or 0.0)
        except (TypeError, ValueError):
            return None
        if fill_price <= 0.0:
            return None
        return {"fill_price": fill_price, "qty": fill_qty, "fees": fees}

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
