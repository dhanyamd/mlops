"""Live research executor — rebalances the Bybit Demo book to the TARGET PORTFOLIO
emitted by stream.research_signal.ResearchSignal (the validated research_novel
ENS_MCD_SLOW book). This is the live counterpart to research_novel.backtest():
weekly long/short weight rebalancing with turnover costs, instead of the
threshold-entry/hold-until-decay logic in PaperExecutionSimulator.

Faithful to the research backtest:
  * target weights come straight from research_novel.weights_at (equal-weight
    top/bottom quintile, long + / short -).
  * rebalance every week; close positions that left the book, open new ones.
  * turnover cost = taker fee on each open AND close (research uses cost_bps).

State is mirrored to the same `execution:crypto:1h:SYM` KV schema the dashboard
(LiveBookPnL / ExecutionPanel) already reads, so the web shows real research PnL.
"""

from __future__ import annotations

import math

from config.logging import configure_logging, get_logger  # noqa: F401
from stream.kv import KVStore
from stream.research_signal import target_key

logger = get_logger("research_executor")

_WEEK_MS = 7 * 24 * 3_600_000


class ResearchExecutor:
    def __init__(
        self,
        kv: KVStore,
        prefix: str = "research",
        universe: list[str] | None = None,
        total_capital: float = 8000.0,
        taker_fee_bps: float = 5.0,
        venue=None,  # BybitDemoVenue for live; None = paper (fill at mark)
        window_ms: int = 3_600_000,
    ) -> None:
        self._kv = kv
        self._prefix = prefix
        self._universe = list(universe or [])
        self._capital = total_capital
        self._fee = taker_fee_bps / 10_000.0
        self._venue = venue
        self._window_ms = window_ms
        self._pos: dict[str, dict] = {}
        self._realized: dict[str, float] = {s: 0.0 for s in self._universe}
        self._gross: dict[str, float] = {s: 0.0 for s in self._universe}
        self._fees: dict[str, float] = {s: 0.0 for s in self._universe}
        self._n_trades: dict[str, int] = {s: 0 for s in self._universe}
        self._n_wins: dict[str, int] = {s: 0 for s in self._universe}
        self._ledger: dict[str, list[dict]] = {s: [] for s in self._universe}
        self._marks: dict[str, float] = {}
        self._last_week: int | None = None

    # ── live marks ────────────────────────────────────────────────────────────

    def handle(self, msg: dict) -> None:
        symbol = str(msg.get("symbol") or "").upper()
        if symbol not in self._marks:
            return
        close = msg.get("close")
        if not isinstance(close, (int, float)) or not math.isfinite(close) or close == 0:
            return
        w_end = int(msg.get("window_end_ms"))
        self._marks[symbol] = float(close)
        week = w_end // _WEEK_MS
        if self._last_week is None or week != self._last_week:
            self._last_week = week
            targets = self._kv.get_json(target_key(self._prefix))
            if targets:
                self.rebalance(targets.get("targets", {}), w_end)

    # ── rebalancing ───────────────────────────────────────────────────────────

    def rebalance(self, targets: dict[str, float], window_end: int) -> None:
        price = self._marks
        for s in self._universe:
            w = float(targets.get(s, 0.0))
            desired = w * self._capital  # signed notional (long +, short -)
            side = "Buy" if desired > 0 else ("Sell" if desired < 0 else None)
            cur = self._pos.get(s)
            if side is None:
                if cur is not None:
                    self._close(s, price[s], window_end)
                continue
            if cur is not None and cur["side"] != side:
                self._close(s, price[s], window_end)
                cur = None
            if cur is None and abs(desired) > 0:
                self._open(s, side, abs(desired), price[s], window_end)

    def _open(self, sym: str, side: str, notional: float, price: float, window_end: int) -> None:
        if not price or not math.isfinite(price) or price <= 0:
            return
        qty = notional / price
        fee = notional * self._fee
        self._pos[sym] = {
            "side": side,
            "qty": qty,
            "entry": price,
            "fees": fee,
            "window_end_ms": window_end,
        }
        self._fees[sym] += fee
        self._n_trades[sym] += 1
        self._ledger[sym].append(
            {
                "side": side,
                "entry_price": price,
                "notional": notional,
                "fee": fee,
                "window_end_ms": window_end,
            }
        )

    def _close(self, sym: str, price: float, window_end: int) -> None:
        cur = self._pos.pop(sym, None)
        if cur is None or not price or not math.isfinite(price):
            return
        sign = 1.0 if cur["side"] == "Buy" else -1.0
        close_fee = (cur["qty"] * price) * self._fee
        gross = (price - cur["entry"]) * cur["qty"] * sign
        net = gross - cur["fees"] - close_fee
        self._gross[sym] += gross
        self._fees[sym] += close_fee
        self._realized[sym] += net
        if net > 0:
            self._n_wins[sym] += 1
        if self._ledger[sym]:
            self._ledger[sym][-1].update(
                {"exit_price": price, "net_pnl": net, "exit_window_end_ms": window_end}
            )

    # ── state for dashboard ───────────────────────────────────────────────────

    def equity(self) -> float:
        unreal = 0.0
        for s, p in self._pos.items():
            if not p:
                continue
            m = self._marks.get(s)
            if not m:
                continue
            sign = 1.0 if p["side"] == "Buy" else -1.0
            unreal += (m - p["entry"]) * p["qty"] * sign
        return sum(self._realized.values()) + unreal

    def payload(self, symbol: str, window_end: int | None = None) -> dict:
        p = self._pos.get(symbol)
        position_view = None
        unreal = 0.0
        if p is not None:
            m = self._marks.get(symbol)
            sign = 1.0 if p["side"] == "Buy" else -1.0
            unreal = (m - p["entry"]) * p["qty"] * sign if m else 0.0
            position_view = {
                "side": p["side"],
                "qty": p["qty"],
                "entry_price": p["entry"],
                "mark_price": m,
                "unrealized_pnl": round(unreal, 2),
            }
        return {
            "symbol": symbol,
            "window_end_ms": window_end,
            "venue": "bybit-demo" if self._venue is not None else "paper",
            "notional_usd": self._capital,
            "n_trades": self._n_trades.get(symbol, 0),
            "n_wins": self._n_wins.get(symbol, 0),
            "realized_pnl": round(self._realized.get(symbol, 0.0), 2),
            "unrealized_pnl": round(unreal, 2),
            "net_pnl": round(self._realized.get(symbol, 0.0) + unreal, 2),
            "position": position_view,
        }


if __name__ == "__main__":
    configure_logging()
    print("ResearchExecutor ready (wire after ResearchSignal)")
