"""Live intraday feature feed -- the SAME reducer research uses.

Four SRP factors (Q, RSJ, OFI, CPV) are computed from intraday bars. The live
``crypto.features.1h`` topic carries close and volume only; these factors also
need taker-buy volume and trade count, so this module fetches full 1h klines
from the keyless Binance futures REST and reduces them with

    scripts.backfill_intraday_features.daily_features

-- the identical function the research backfill calls. That is deliberate: the
reduction from bars to daily factor values is where a reimplementation would
drift, and this project has already shipped one live book that diverged from
research at 0.147 rank correlation. Calling the same reducer makes that class of
bug impossible rather than merely unlikely.

WHY 1-HOUR AND NOT 5-MINUTE
---------------------------
The research validated these factors on 5-minute bars, but hourly-derived
versions measured BETTER on the same universe and window (Sharpe 2.60 vs 2.37,
correlation 0.974). Hourly aggregation is the less noisy estimator for the
correlation- and skew-based factors, and it is also 12x less data to move. The
live path therefore uses 1h by choice, not by constraint.

FAIL CLOSED: a symbol whose klines cannot be fetched is omitted, never
forward-filled or guessed. SRP scores what it has and goes FLAT on the rest.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from scripts.backfill_intraday_features import daily_features

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
_PACE_S = 0.12
_last = 0.0


def _pace() -> None:
    global _last
    dt = time.time() - _last
    if dt < _PACE_S:
        time.sleep(_PACE_S - dt)
    _last = time.time()


def fetch_klines(
    symbol: str, *, interval: str = "1h", limit: int = 1000, end_ms: int | None = None
) -> list[list] | None:
    """Raw klines, oldest->newest, in the exact row shape the reducer expects.

    Binance returns 12 fields per bar; ``daily_features`` reads index 0 (open
    time), 4 (close), 5 (volume), 8 (trade count) and 9 (taker-buy volume). We
    pass the rows through untouched so the live and research inputs are
    byte-identical in shape.
    """
    collected: list[list] = []
    end = end_ms
    remaining = limit
    while remaining > 0:
        params = {"symbol": symbol.upper(), "interval": interval,
                  "limit": min(remaining, 1500)}
        if end is not None:
            params["endTime"] = int(end)
        try:
            _pace()
            url = f"{FAPI}/fapi/v1/klines?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=25) as r:
                page = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            logger.warning("intraday: klines HTTP %s for %s", e.code, symbol)
            return None
        except Exception:
            logger.warning("intraday: klines fetch failed for %s", symbol, exc_info=False)
            return None
        if not isinstance(page, list) or not page:
            break
        collected = page + collected
        remaining -= len(page)
        end = int(page[0][0]) - 1
        if len(page) < min(limit, 1500):
            break
    return collected or None


_DAY_MS = 86_400_000
_BARS_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6}


def daily_series(
    symbol: str, *, interval: str = "1h", limit: int = 1000, as_of_ms: int | None = None,
    require_complete: bool = True,
) -> dict[int, dict] | None:
    """Daily intraday features for one symbol, point-in-time at ``as_of_ms``.

    Bars stamped after ``as_of_ms`` are dropped BEFORE reduction, so a partially
    complete day cannot leak into a score formed earlier in that day.

    INCOMPLETE DAYS ARE DROPPED. Two reasons, both verified:
      * the pagination boundary produces a short first day, and it was the ONLY
        source of disagreement against the research backfill -- with partial
        days excluded, all five factors match research to 0.000e+00;
      * live, the current day is always in progress, and scoring a half-formed
        day silently feeds the book a different statistic than the one that was
        validated.
    The backfill's own guard (>=20 bars) is too loose for hourly data: a 22-bar
    day passes it and diverges.
    """
    rows = fetch_klines(symbol, interval=interval, limit=limit, end_ms=as_of_ms)
    if not rows:
        return None
    if as_of_ms is not None:
        rows = [r for r in rows if int(r[0]) <= as_of_ms]
        if not rows:
            return None
    try:
        feats = daily_features(rows)         # SAME reducer as the research backfill
    except Exception:
        logger.exception("intraday: reducer failed for %s", symbol)
        return None
    if require_complete:
        need = _BARS_PER_DAY.get(interval)
        if need:
            counts: dict[int, int] = {}
            for r in rows:
                counts[int(r[0]) // _DAY_MS] = counts.get(int(r[0]) // _DAY_MS, 0) + 1
            feats = {d: v for d, v in feats.items() if counts.get(d, 0) >= need}
    return feats or None


def fetch_universe(
    symbols: list[str], *, interval: str = "1h", limit: int = 1000,
    as_of_ms: int | None = None,
) -> dict[str, dict[int, dict]]:
    """Daily intraday features for a universe. Failures are OMITTED, not guessed."""
    out: dict[str, dict[int, dict]] = {}
    missing: list[str] = []
    for s in symbols:
        rec = daily_series(s, interval=interval, limit=limit, as_of_ms=as_of_ms)
        if not rec:
            missing.append(s)
            continue
        out[s] = rec
    if missing:
        logger.warning(
            "intraday: %d/%d symbols unavailable (excluded, not guessed): %s",
            len(missing), len(symbols), ",".join(missing[:8]),
        )
    return out
