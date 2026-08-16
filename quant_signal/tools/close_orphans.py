"""One-off cleanup: close every open position still sitting on the Bybit Demo
account after the execution ledger was wiped on restart.

These are REAL demo orders still open on api-demo.bybit.com (virtual USDT, no
real money). We read the LIVE open-position list (no hardcoded symbol list —
the account currently holds 16 positions, several of which were not placed by
our book) and close each at market so the live asym daemon can rebuild a clean
book without duplicate exposure.

Run with:
  cd /Users/dhanyamd/Projects/mlops/quant_signal && /Users/dhanyamd/.local/bin/uv run python close_orphans.py
"""

from __future__ import annotations

import sys

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bybit_demo import BybitDemoVenue

logger = get_logger(__name__)


def main() -> int:
    configure_logging()
    settings = get_settings()
    venue = BybitDemoVenue(
        settings.demo_api_key or "",
        settings.demo_api_secret or "",
        maker_first=False,  # market close: fast, reliable
    )

    # Read EVERY open linear/USDT position straight from the exchange. No
    # hardcoded symbol list — the venue returns whatever the account actually
    # holds (16 positions, incl. ones our book never placed).
    try:
        resp = venue._http.get_positions(category="linear", settleCoin="USDT", limit=200)
    except Exception:
        logger.exception("bybit demo: get_positions(settleCoin=USDT) failed")
        return 2
    if not isinstance(resp, dict) or resp.get("retCode") != 0:
        logger.warning("bybit demo: get_positions -> %s", resp)
        return 2

    open_positions = [
        p
        for p in (resp.get("result", {}) or {}).get("list", []) or []
        if float(p.get("size") or 0.0) > 0.0
    ]
    print(f"Found {len(open_positions)} open demo positions")

    closed = 0
    failed = 0
    for pos in open_positions:
        sym = pos.get("symbol")
        side = pos.get("side")  # "Buy" (LONG) or "Sell" (SHORT)
        size = float(pos.get("size") or 0.0)
        if not sym or size <= 0.0 or side not in ("Buy", "Sell"):
            logger.warning("bybit demo: skipping malformed position %s", pos)
            continue
        close_side = "LONG" if side == "Buy" else "SHORT"
        fill = venue.close_market(sym, close_side, size)
        if fill is not None:
            closed += 1
            print(f"CLOSED {sym} {side} size={size} @ {fill.get('fill_price')}")
        else:
            failed += 1
            print(f"FAILED {sym} {side} size={size} (no fill)")

    print(f"\nOrphaned positions closed: {closed}  failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
