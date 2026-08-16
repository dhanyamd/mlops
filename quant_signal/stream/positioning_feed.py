"""Live positioning feed -- open interest and whale-vs-retail, keyless Binance.

Three of the nine SRP factors (WRspread, TopChg, Quad) need exchange positioning
data, which the 1h feature topic does not carry. This fetches it from the same
keyless public REST surface the funding fetcher already uses.

    /futures/data/openInterestHist          -> sum_open_interest
    /futures/data/topLongShortPositionRatio -> top traders, by POSITION SIZE
    /futures/data/globalLongShortAccountRatio -> ALL accounts, by account count

These endpoints cap at ~30 days of history, which is why the RESEARCH backfill
had to come from data.binance.vision. For LIVE it is ample: the factors need the
current value and a one-week change, and the rolling registry accumulates the
rest as the process runs.

DESIGN RULES (inherited from the funding fetcher, deliberately):
  * keyless, politely paced -- no credentials in the signal path;
  * FAIL CLOSED -- on a fetch failure the caller gets nothing and the book goes
    FLAT for those factors rather than trading a stale or guessed value. A wrong
    position is more expensive than a missed one;
  * point-in-time -- only observations stamped at or before the query instant
    are returned, so a live score can never see a value it could not have had.

Sanity: the ratios are LONG/SHORT ratios, not shares. A value of 2.0 means twice
as much long as short. They are stored raw; SRP's factors take differences and
changes of them, so no normalisation happens here.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
_PACE_S = 0.20
_last_call = 0.0


def _pace() -> None:
    global _last_call
    dt = time.time() - _last_call
    if dt < _PACE_S:
        time.sleep(_PACE_S - dt)
    _last_call = time.time()


def _get(path: str, params: dict) -> list | None:
    url = f"{FAPI}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            _pace()
            with urllib.request.urlopen(url, timeout=20) as r:
                out = json.loads(r.read())
            return out if isinstance(out, list) else None
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):          # symbol not offered on this endpoint
                return None
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    logger.warning("positioning: fetch failed after retries: %s", path)
    return None


_PERIOD_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000,
              "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
              "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000}


def _series(
    rows: list | None, key: str, as_of_ms: int | None, period: str = "1d"
) -> list[tuple[int, float]]:
    """[(timestamp_ms, value)] sorted, point-in-time, ALIGNED to the archive clock.

    Binance stamps these REST rows at the END of the interval: a row labelled
    2026-08-11 00:00 with period=1d is the state observed through 2026-08-10.
    The bulk archive (data.binance.vision), which seeds our history, labels the
    same observation with the day it happened in. Verified directly: archive
    day 2026-08-09 closes at 23:55 with OI 107,032.4, and REST reports that
    value stamped 2026-08-10.

    Merging the two without this shift would make every live observation one
    period stale relative to history -- invisible in tests, wrong in production.
    We subtract one period so both sources label an observation by when it
    OCCURRED, never by when it was published.
    """
    step = _PERIOD_MS.get(period, 86_400_000)
    out: list[tuple[int, float]] = []
    for r in rows or []:
        try:
            ts = int(r.get("timestamp")) - step      # -> observation instant
            v = float(r.get(key))
        except (TypeError, ValueError, AttributeError):
            continue
        if v != v:                              # NaN guard
            continue
        if as_of_ms is not None and ts > as_of_ms:
            continue
        out.append((ts, v))
    out.sort()
    return out


def fetch_positioning(
    symbol: str, *, period: str = "1d", limit: int = 30, as_of_ms: int | None = None
) -> dict[str, list[tuple[int, float]]] | None:
    """Open interest and the two long/short ratios for one symbol.

    Returns None if ANY of the three is unavailable -- a partial record would
    make WRspread (top minus all) silently wrong rather than absent.
    """
    p = {"symbol": symbol, "period": period, "limit": limit}
    oi = _series(_get("/futures/data/openInterestHist", p), "sumOpenInterest", as_of_ms, period)
    top = _series(_get("/futures/data/topLongShortPositionRatio", p),
                  "longShortRatio", as_of_ms, period)
    allc = _series(_get("/futures/data/globalLongShortAccountRatio", p),
                   "longShortRatio", as_of_ms, period)
    if not oi or not top or not allc:
        return None
    return {"open_interest": oi, "top_ls": top, "all_ls": allc}


def fetch_universe(
    symbols: list[str], *, period: str = "1d", limit: int = 30, as_of_ms: int | None = None
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Positioning for a universe. Symbols that fail are OMITTED, never guessed."""
    out: dict[str, dict] = {}
    missing: list[str] = []
    for s in symbols:
        rec = fetch_positioning(s, period=period, limit=limit, as_of_ms=as_of_ms)
        if rec is None:
            missing.append(s)
            continue
        out[s] = rec
    if missing:
        logger.warning(
            "positioning: %d/%d symbols unavailable (they will be excluded, not guessed): %s",
            len(missing), len(symbols), ",".join(missing[:8]),
        )
    return out
