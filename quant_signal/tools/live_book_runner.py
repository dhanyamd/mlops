"""Direct live-book runner: validated FAS+SMB+FACC book -> REAL Bybit Demo fills.

The streaming pipeline (producer -> Redpanda -> Flink -> crypto.features.1h) is
currently wedged (producer `poll exceeded 45s deadline`, raw bars not flowing,
so the feature topic is empty and every signal is idle). To let the user SEE
real trades on the ledger NOW, this script drives the SAME production classes
the daemons use -- AsymSignal (FAS_avg + SMB + FACC) + PaperExecutionSimulator +
BybitDemoVenue -- but feeds them REAL keyless data pulled directly, instead of
waiting on the dead Kafka feature topic.

Nothing is hardcoded:
  * history is the real warm-start (Bybit 1h bars + keyless Binance funding),
  * windows / quintile / selection are the live settings,
  * fills are REAL Bybit Demo market orders (virtual USDT, no real money),
  * the ledger lands on the shared execution:crypto:1h:<SYMBOL> key.

Regime is set to OBSERVATION mode (regime=False) on purpose: BTC is in a bear
regime right now (close < 52w MA) which would otherwise flatten the book. On the
Demo account that only costs virtual USDT, so we watch real fills through the
bear. Re-enable the gate (stream_asym_regime=true) before any real-money run.
"""

from __future__ import annotations

import time

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.asym_signal import _HOUR_MS, AsymSignal, prediction_key
from stream.execution import PaperExecutionSimulator, _build_venue
from stream.kv import RedisKV

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()
    kv = RedisKV(settings.stream_redis_url)
    universe = csv_list(settings.stream_xs_universe)

    sig = AsymSignal(
        kv,
        prediction_prefix=settings.stream_asym_prediction_prefix,
        universe=universe,
        rebalance_h=settings.stream_xs_rebalance_h,
        quintile=settings.stream_asym_quintile,
        min_symbols=settings.stream_asym_min_symbols,
        regime=False,  # OBSERVATION MODE for demo (bear regime would flatten)
        regime_slow_days=settings.stream_asym_regime_slow_days,
        market_symbol=settings.stream_asym_market_symbol,
        horizons=settings.stream_asym_horizons,
        accrual_weeks=settings.stream_asym_accrual_weeks,
        smb_weeks=settings.stream_asym_smb_weeks,
        use_facc=True,
        use_rev=False,
    )

    logger.info("warm-starting asym book with REAL keyless data (%d symbols)...", len(universe))
    t0 = time.time()
    sig.warm_start(settings)
    logger.info("warm-start done in %.1fs", time.time() - t0)

    # Rebalance at the current weekly boundary (same cadence as the live daemon).
    now_ms = int(time.time() * 1000)
    window_end = (now_ms // (_HOUR_MS * sig._rebalance_h)) * (_HOUR_MS * sig._rebalance_h)
    sel = sig._selection(window_end)
    longs = [s for s, (d, _y) in sel.items() if d == "LONG"]
    shorts = [s for s, (d, _y) in sel.items() if d == "SHORT"]
    logger.info(
        "selection @%d : %d LONG / %d SHORT / %d FLAT",
        window_end,
        len(longs),
        len(shorts),
        sum(1 for _d, _y in sel.values() if _d == "FLAT"),
    )

    # Publish the predictions so the production execution engine can read them.
    for s, (direction, yhat) in sel.items():
        kv.set_json(
            prediction_key(settings.stream_asym_prediction_prefix, s),
            {
                "symbol": s,
                "window_end_ms": window_end,
                "predicted_return": round(float(yhat), 6),
                "direction": direction,
                "signal": "asym",
                "updated_at": sig._kv_now(),
            },
        )

    venue = _build_venue(settings)
    sim = PaperExecutionSimulator(
        kv,
        execution_prefix=settings.stream_redis_execution_prefix,
        prediction_prefix=settings.stream_asym_prediction_prefix,
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

    # Drive one rebalance window through the real execution engine so it places
    # REAL Bybit Demo market orders for the LONG/SHORT quintiles. Feed a prior
    # window first (entry gate needs a previous bar) then the rebalance window.
    prev_end = window_end - _HOUR_MS
    for s in universe:
        hist = sig._closes.get(s)
        last_close = hist[-1][1] if hist else 0.0
        sim.handle({"symbol": s, "close": last_close, "window_end_ms": prev_end})
    for s in universe:
        hist = sig._closes.get(s)
        last_close = hist[-1][1] if hist else 0.0
        sim.handle({"symbol": s, "close": last_close, "window_end_ms": window_end})

    # Report real fills from the ledger.
    opens = 0
    for s in sorted(longs + shorts):
        payload = kv.get_json(f"{settings.stream_redis_execution_prefix}:{s}")
        if not payload:
            continue
        pos = payload.get("position")
        if pos:
            opens += 1
            print(
                f"  {s:10s} {pos['side']:5s} @ {pos['entry_price']:.6f} "
                f"qty={pos['qty']:.4f} notional=${settings.stream_execution_notional_usd:.0f} "
                f"(venue={'bybit-demo' if venue else 'paper'})"
            )
    print(f"\nREAL positions opened on ledger: {opens}")
    print(f"LONG: {longs}")
    print(f"SHORT: {shorts}")
    print("Ledger keys: execution:crypto:1h:<SYMBOL>  (watch on the dashboard)")


if __name__ == "__main__":
    main()
