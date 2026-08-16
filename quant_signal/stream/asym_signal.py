"""Live ASYM signal -- OUR novel FAS (Funding-Accrual Squeeze) + SMB cross-sectional book.

Sibling of ``stream.scx_signal``. Consumes the SAME 1h feature topic
(``crypto.features.1h``, close/volume only) for the PRICE PATH + BTC regime gate,
and fetches FUNDING from the keyless Binance futures public REST
(``fapi/v1/fundingRate`` -- no API key) to build the funding-accrual series that
FAS requires. Emits the same ``prediction:crypto:1h:<SYMBOL>`` payload the executor
already reads, so the executor, Bybit demo venue and dashboard are untouched.

FACTOR (validated in scripts/research_fas_combo.py, keyless Binance 4.4y, 10bps,
BTC-regime):  FAS_avg (multi-horizon funding-accrual-squeeze) + SMB (size tilt)
= **Sharpe +1.82 FULL / +1.93 POST-2024** (CI upper 2.49/2.65). This is the
+1.9 book we deploy; FAS is OURS (funding-accrual residualized on price path, the
derivatives analog of Bianchi et al. 2026 order-flow orthogonalization), SMB is the
one known factor that *adds* to FAS (REV/MOM hurt it).

  - FAS_h   = -z(price_ret_h) * z(residual of 12w funding accrual on [pr_h, |pr_h|])
  - FAS_avg = mean over horizons [4, 8, 12, 26] of FAS_h   (our self-ensemble)
  - SMB     = -z(log trailing 12w volume)                  (long small-caps)
  - RCGO₿   = z( CGO_daily residualized on funding-carry )  (OUR invented tilt, W>0)
  - score   = z( FAS_avg + SMB [+ W*dir*RCGO₿] )   # RCGO₿ OFF -> hard CGO q-filter
  - BTC UP regime gate: BTC daily close > 52-week MA (fails CLOSED/FLAT without
    history). Top/bottom quintile -> LONG/SHORT; mid -> FLAT.

DATA SOURCE: Binance funding ONLY (``fapi.binance.com``). Do NOT fall back to
Hyperliquid for funding -- OOS validation (scripts/research_hl_fas.py) shows FAS =
-0.37 on Hyperliquid funding vs +1.55 on Binance, i.e. Hyperliquid breaks the
factor. If the Binance funding fetch fails, the book goes FLAT (never guesses).

No lookahead: every accrual/return at week w uses data strictly before the
rebalance boundary. Funding fetch is best-effort; symbols without enough funding
history are skipped (FLAT) rather than guessed.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
import urllib.request
import urllib.error
from collections.abc import Sequence

import numpy as np

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_DAY_MS = 24 * _HOUR_MS
_WEEK_MS = 7 * _DAY_MS

# CGO enhancement env vars (Griffin-Han capital-gains-overhang, daily, G/r filter)
_QUANT_CGO_L = int(os.environ.get("QUANT_CGO_L", "7"))  # lookback days for CGO
_QUANT_CGO_Q = float(os.environ.get("QUANT_CGO_Q", "0.3"))  # screen quantile
_QUANT_CGO_DIR = int(os.environ.get("QUANT_CGO_DIR", "-1"))  # -1=keep low CGO, 1=keep high

# RCGO₿ (INVENTED, OUR book): Residual Capital-Gains-Overhang, orthogonalized on
# funding-carry, blended CONTINUOUSLY (not a hard gate). No paper does CGO-on-
# crypto-carry; derived from Liu-Fang-Wang (CGO daily-only) + Crypto Carry
# (SSRN 3774118) + Guangfa A-share RCGO construction. W=0 -> OFF (use hard CGO
# filter instead). Best OOS: w>=0.25, dir=+1 -> Sharpe 1.93 (scripts/research_fas_invent.py).
_QUANT_RCGO_W = float(os.environ.get("QUANT_RCGO_W", "0"))  # 0=off; continuous blend weight
_QUANT_RCGO_DIR = int(os.environ.get("QUANT_RCGO_DIR", "1"))  # 1=high-CGO tilt, -1=low
_QUANT_RCGO_ORTHO = os.environ.get("QUANT_RCGO_ORTHO", "1") == "1"  # residualize CGO on carry
# Grinblatt-Han survival-weighted reference price for CGO (1=on, default).
# 0 restores the legacy turnover-weighted-mean form. See AsymSignal._cgo.
_QUANT_CGO_GH = os.environ.get("QUANT_CGO_GH", "1") == "1"
# Extra weeks of weekly-close history kept beyond the longest FAS horizon.
_QUANT_CLOSE_SPAN_W = int(os.environ.get("QUANT_CLOSE_SPAN_W", "4"))
# Weeks of funding history required before a symbol is scoreable. FAS needs
# only the scoring week's accrual; the old max_h+1 gate threw away half the
# usable sample. Kept configurable so the strict behaviour is recoverable.
_QUANT_MIN_FUND_W = int(os.environ.get("QUANT_MIN_FUND_W", "4"))
# Score by calling the validated research functions instead of the streaming
# reimplementation (default ON). QUANT_RESEARCH_PARITY=0 restores the legacy
# path for A/B comparison.
_QUANT_RESEARCH_PARITY = os.environ.get("QUANT_RESEARCH_PARITY", "1") == "1"
# "srp"    -> Self-Referential Parity (validated 2026-08: Sharpe 2.40, t 4.63,
#             net of funding and realistic maker costs; gate in scripts/srp_parity.py)
# "legacy" -> the original FAS+SMB+RCGO book (FAS IC +0.0065, RCGO negative)
_QUANT_STRATEGY = os.environ.get("QUANT_STRATEGY", "legacy").strip().lower()
# Warm-start klines category. SRP trades PERPETUALS and its universe is built
# from Bybit linear perps, many of which have no spot pair -- fetching spot
# raises retCode=10001 and kills warm-start on the first such symbol.
_WARM_CATEGORY = os.environ.get(
    "QUANT_WARM_CATEGORY", "linear" if _QUANT_STRATEGY == "srp" else "spot"
)


# ── week bucketing ────────────────────────────────────────────────────────────
# The research book aggregates weeks with pandas ``resample("W-MON")``: bins are
# (previous Monday, this Monday], labelled with the CLOSING Monday. Naive epoch
# bucketing -- ``(t // _WEEK_MS) * _WEEK_MS`` -- anchors to the Unix epoch, which
# fell on a THURSDAY, so it produced Thu->Wed weeks: a 4-day-shifted calendar.
# Every weekly aggregate (closes, funding accrual, volume) was therefore computed
# over a different set of days than the validated research book, which is why the
# live scores correlated only 0.147 with research and the selections overlapped
# 27% (barely above the 20% expected by chance). Verified against pandas on 4000
# timestamps spanning 3 years: 0 mismatches.
_MONDAY0_MS = 4 * _DAY_MS  # 1970-01-05, the first Monday after the epoch


def _week_end_ms(ts_ms: int) -> int:
    """Monday-anchored week END for ``ts_ms`` (matches pandas ``W-MON``)."""
    d = ts_ms - _MONDAY0_MS
    return _MONDAY0_MS + (-((-d) // _WEEK_MS)) * _WEEK_MS


def prediction_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def _rank_z(series: dict[str, float]) -> dict[str, float]:
    """Cross-sectional rank-based z (pct rank * 2 - 1), matching research zs_df."""
    vals = [v for v in series.values() if math.isfinite(v)]
    if len(vals) < 5:
        return {k: 0.0 for k in series}
    order = sorted(vals)
    n = len(order)
    out = {}
    for k, v in series.items():
        if not math.isfinite(v):
            out[k] = 0.0
            continue
        lo = sum(1 for x in order if x <= v)
        r = (lo / n) - 0.5
        out[k] = r * 2.0
    return out


class AsymSignal:
    """FAS_avg + SMB weekly book, fed by 1h closes + keyless Binance funding.

    ``_closes[symbol]`` = rolling (window_end_ms, close, volume) hourly registry
    (same as ScxSignal). ``_funding[symbol]`` = rolling (event_ms, rate) funding
    registry fetched keyless from Binance. At each weekly UTC rebalance the
    universe is ranked by FAS_avg + SMB and the selection cached; every subsequent
    bar re-emits the cached direction so the executor HOLDs until the next
    rebalance.
    """

    def __init__(
        self,
        kv: KVStore,
        *,
        prediction_prefix: str,
        universe: list[str],
        rebalance_h: int = 168,
        quintile: float = 0.20,
        min_symbols: int = 8,
        regime: bool = True,
        regime_slow_days: int = 364,
        market_symbol: str = "BTCUSDT",
        horizons: list[int] | None = None,
        accrual_weeks: int = 12,
        smb_weeks: int = 12,
        use_facc: bool = True,
        use_rev: bool = False,
        facc_weeks: int = 6,
        facc_vol_weeks: int = 12,
        lookback_h: int = 24 * 7 * 60,  # ~60 weeks of hourly closes for price path
        max_history: int = 60_000,
        funding_lookback_events: int = 1500,
        funding_refresh_events: int = 300,
        replay: bool = False,
    ) -> None:
        self._kv = kv
        self._prediction_prefix = prediction_prefix
        self._universe = [s.upper() for s in universe]
        self._rebalance_h = rebalance_h
        self._quintile = quintile
        self._min_symbols = min_symbols
        self._regime = (regime or os.environ.get("QUANT_REGIME_ON") == "1") and os.environ.get(
            "QUANT_REGIME_OFF"
        ) != "1"
        self._regime_slow_days = regime_slow_days
        self._horizons = sorted(horizons or [4, 8, 12, 26])
        self._accrual_weeks = accrual_weeks
        self._smb_weeks = smb_weeks
        self._use_facc = use_facc and os.environ.get("QUANT_FACC_OFF") != "1"
        self._use_rev = use_rev or os.environ.get("QUANT_REV_ON") == "1"
        self._facc_weeks = facc_weeks
        self._facc_vol_weeks = facc_vol_weeks
        self._max_horizon = max(self._horizons)
        self._lookback_h = lookback_h
        self._max_history = max_history
        # How many rebalance periods old a bar may be and still trigger a
        # rebalance. 2 tolerates a normal restart/backfill without letting a
        # months-old retained message drive the book.
        self._stale_bar_periods = float(
            os.environ.get("QUANT_STALE_BAR_PERIODS", "2")
        )
        self._funding_lookback_events = funding_lookback_events
        self._funding_refresh_events = funding_refresh_events
        self._replay = replay
        self._market = (
            market_symbol.upper()
            if market_symbol.upper() in self._universe
            else (self._universe[0] if self._universe else market_symbol.upper())
        )
        self._closes: dict[str, list[tuple[int, float, float]]] = {}
        self._funding: dict[str, list[tuple[int, float]]] = {}
        self._last_week: int | None = None
        self._current: dict[str, tuple[str, float]] = {}

    # ── history ───────────────────────────────────────────────────────────────

    def _record(self, symbol: str, window_end: int, close: float, volume: float) -> None:
        hist = self._closes.setdefault(symbol, [])
        if hist and hist[-1][0] >= window_end:
            return
        hist.append((window_end, close, volume))
        while len(hist) > self._max_history:
            del hist[0]

    @staticmethod
    def _series(hist: Sequence[tuple[int, float, float]]):
        arr = np.array([(e, c, v) for e, c, v in hist], dtype=float)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    @staticmethod
    def _last_index(ends: np.ndarray, window_end: int) -> int:
        return int(np.searchsorted(ends, window_end, side="left")) - 1

    # ── keyless Bybit funding history (/v5/market/funding/history, no API key) ─
    # Bybit's funding-rate history is a PUBLIC endpoint (no key, no auth) that is
    # reachable from this IP — unlike Binance fapi, which is geo-blocked here and
    # was timing out for 20s on every call. It returns (fundingRateTimestamp_ms,
    # fundingRate) newest-first, up to ``limit`` (1..200) rows per page, bounded by
    # startTime/endTime. Funding settles every 8h (~21 events/week), so the longest
    # FAS horizon (26 weeks) needs ~27 weeks of history — a handful of paged calls.
    # Public market endpoints carry a generous per-IP limit, so we only pace with a
    # small floor + retry-with-backoff on errors so a transient blip never reads as
    # "no funding history".
    _FUNDING_PAGE = 200
    _FUNDING_MIN_INTERVAL_S = 0.1  # polite floor; Bybit public limit is far higher
    _FUNDING_RETRIES = 3
    _FUNDING_BACKOFF_S = 0.5
    _last_funding_call = 0.0  # class-level monotonic clock for pacing
    _BYBIT_BASE_URL: str | None = None  # lazy-cached from settings

    @classmethod
    def _bybit_base_url(cls) -> str:
        if cls._BYBIT_BASE_URL is None:
            from config.settings import get_settings

            cls._BYBIT_BASE_URL = get_settings().stream_bybit_base_url
        return cls._BYBIT_BASE_URL

    @classmethod
    def _pace(cls) -> None:
        """Enforce a small floor between funding calls (Bybit public limit is high)."""
        wait = cls._FUNDING_MIN_INTERVAL_S - (time.monotonic() - cls._last_funding_call)
        if wait > 0:
            time.sleep(wait)
        cls._last_funding_call = time.monotonic()

    @classmethod
    def _build_funding_url(
        cls, symbol: str, limit: int, start_ms: int | None, end_ms: int | None
    ) -> str:
        limit = min(int(limit), AsymSignal._FUNDING_PAGE)
        params = [
            "category=linear",
            f"symbol={symbol.upper()}",
            f"limit={limit}",
        ]
        if start_ms is not None:
            params.append(f"startTime={int(start_ms)}")
        if end_ms is not None:
            params.append(f"endTime={int(end_ms)}")
        return f"{cls._bybit_base_url()}/v5/market/funding/history?" + "&".join(params)

    @classmethod
    def _fetch_funding(
        cls, symbol: str, limit: int, start_ms: int | None = None, end_ms: int | None = None
    ) -> list[tuple[int, float]]:
        """One page of (event_time_ms, funding_rate) from Bybit /v5/market/funding/history.

        Keyless public endpoint. Orders newest-first. Retries on network errors and
        non-zero retCode so a transient blip never masquerades as "no funding
        history". Returns [] only after exhausting retries.
        """
        url = cls._build_funding_url(symbol, limit, start_ms, end_ms)
        for attempt in range(cls._FUNDING_RETRIES):
            cls._pace()
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    body = json.loads(resp.read().decode())
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                json.JSONDecodeError,
                OSError,
            ) as e:
                logger.warning("asym funding fetch failed %s (attempt %d): %s", symbol, attempt, e)
                time.sleep(cls._FUNDING_BACKOFF_S * (2**attempt))
                continue
            if not isinstance(body, dict) or body.get("retCode") != 0:
                # retCode 10001 = "Time Is Invalid" (e.g. a startTime in the
                # future) or unknown symbol — both are NON-retryable, so don't
                # burn the retry budget on them; treat as "no data".
                if body.get("retCode") == 10001:
                    return []
                logger.warning("asym funding fetch error %s: %s", symbol, body)
                time.sleep(cls._FUNDING_BACKOFF_S * (2**attempt))
                continue
            out: list[tuple[int, float]] = []
            for row in body.get("result", {}).get("list", []):
                try:
                    out.append((int(row["fundingRateTimestamp"]), float(row["fundingRate"])))
                except (KeyError, TypeError, ValueError):
                    continue
            return out
        logger.error("asym funding fetch exhausted retries for %s", symbol)
        return []

    # ── funding history loaders (keyless Binance, paced + retried) ──────────────

    def _refresh_funding(self, symbol: str) -> None:
        """Page BACKWARD to cover the longest FAS horizon (+ buffer) of funding.

        Used at warm-start: accumulates ~(max_horizon + 12) weeks of funding.
        Bybit returns newest-first, so each page steps ``endTime`` back to the
        oldest event on the page. Pacing + retries in ``_fetch_funding`` make a
        single warm-start populate every symbol reliably. Result is capped and
        seeded into the in-memory registry.
        """
        needed = (self._max_horizon + 12) * 21
        collected: dict[int, float] = {}
        end: int | None = None
        pages = 0
        while len(collected) < needed and pages < 16:
            page = self._fetch_funding(symbol, self._FUNDING_PAGE, end_ms=end)
            if not page:
                if pages == 0:
                    logger.warning("asym funding empty after retries for %s", symbol)
                break
            for e, r in page:
                collected.setdefault(e, r)
            if len(page) < self._FUNDING_PAGE:
                break
            end = page[-1][0] - 1  # page is newest→oldest; step further back
            pages += 1
        if not collected:
            return
        reg = self._funding.setdefault(symbol, [])
        known = {e for e, _ in reg}
        merged = list(reg)
        for e, r in collected.items():
            if e not in known:
                merged.append((e, r))
                known.add(e)
        merged.sort(key=lambda x: x[0])
        keep = (self._max_horizon + 4) * 21 + 50
        self._funding[symbol] = merged[-keep:] if len(merged) > keep else merged

    def _append_funding(self, symbol: str) -> None:
        """Weekly refresh: cheap incremental FORWARD fill from the latest known event.

        The full history was seeded at warm-start; this only folds in settlements
        since the last known fundingTime, so a weekly tick costs a handful of
        paced calls instead of re-paging the whole history.
        """
        reg = self._funding.get(symbol)
        if not reg:
            self._refresh_funding(symbol)
            return
        collected = {e: r for e, r in reg}
        # Bybit rejects a startTime in the future with retCode 10001, so never
        # request past "now" — there is no funding to collect there anyway.
        now_ms = int(time.time() * 1000)
        start = reg[-1][0] + 1
        if start > now_ms:
            return
        pages = 0
        while pages < 16:
            page = self._fetch_funding(symbol, self._FUNDING_PAGE, start_ms=start)
            if not page:
                break
            fresh = [(e, r) for e, r in page if e not in collected]
            if not fresh:
                break
            for e, r in fresh:
                collected[e] = r
            start = fresh[-1][0] + 1
            if start > now_ms:
                break
            pages += 1
        events = sorted(collected.items())
        keep = (self._max_horizon + 4) * 21 + 50
        self._funding[symbol] = events[-keep:] if len(events) > keep else list(events)

    # ── BTC regime gate (daily resample > 52-week MA) ─────────────────────────

    @staticmethod
    def _daily_up(closes: np.ndarray, ends: np.ndarray, slow_days: int) -> bool:
        if len(closes) < 2:
            return False
        day_idx = (ends // _DAY_MS).astype(np.int64)
        daily_list: list[float] = []
        last_day: int | None = None
        prev = float(closes[0])
        for d, c in zip(day_idx, closes):
            if last_day is not None and d != last_day:
                daily_list.append(prev)
            last_day = d
            prev = float(c)
        if last_day is not None:
            daily_list.append(prev)
        daily = np.array(daily_list, dtype=float)
        if len(daily) < slow_days + 1:
            return False
        return float(daily[-1]) > float(daily[-slow_days:].mean())

    # ── weekly resamples ──────────────────────────────────────────────────────

    def _weekly_close(self, symbol: str) -> dict[int, float]:
        hist = self._closes.get(symbol)
        if not hist:
            return {}
        ends, closes, _ = self._series(hist)
        # History kept for the weekly price path. The +4 default leaves only ~4
        # scoreable weeks beyond the longest horizon (26w), which starves the
        # cross-sectional regressions the research book runs over its full
        # frame. QUANT_CLOSE_SPAN_W overrides the extra weeks kept.
        span = (self._max_horizon + _QUANT_CLOSE_SPAN_W) * _WEEK_MS
        out: dict[int, float] = {}
        for e, c in zip(ends, closes):
            wk = _week_end_ms(e)
            if c <= 0 or not math.isfinite(c):
                continue
            out[wk] = float(c)
        if not out:
            return out
        lo = max(out) - span
        return {w: c for w, c in out.items() if w >= lo}

    def _weekly_fund(self, symbol: str) -> dict[int, float]:
        reg = self._funding.get(symbol)
        if not reg:
            return {}
        out: dict[int, float] = {}
        for e, r in reg:
            wk = _week_end_ms(e)
            out[wk] = out.get(wk, 0.0) + float(r)
        return out

    def _weekly_volume(self, symbol: str) -> dict[int, float]:
        hist = self._closes.get(symbol)
        if not hist:
            return {}
        out: dict[int, float] = {}
        for e, _c, v in hist:
            if not math.isfinite(v):
                continue
            wk = _week_end_ms(e)
            out[wk] = out.get(wk, 0.0) + float(v)
        return out

    # ── CGO (Griffin-Han) capital-gains-overhang daily filter ─────────────────
    # Liu-Fang-Wang (管理评论 36(6), 2024): crypto momentum <= 2wk, CGO effect
    # <= half a month, and WEEKLY data is INVALID for CGO. Daily closes+volume
    # required. High CGO => holders sitting on profits => disposition-selling
    # pressure caps rallies => drop those names (G/r: keep low-CGO, dir=-1).

    def _daily_bars(self, symbol: str, as_of: int) -> list[tuple[int, float, float]]:
        """Daily (ts, close, volume) bars strictly before ``as_of``."""
        hist = self._closes.get(symbol)
        if not hist:
            return []
        out: list[tuple[int, float, float]] = []
        for e, c, v in hist:
            day = (e // _DAY_MS) * _DAY_MS
            if day >= as_of:
                break
            if out and out[-1][0] == day:
                out[-1] = (day, c, out[-1][2] + v)
            else:
                out.append((day, c, v))
        return out

    def _cgo(self, dbars: list[tuple[int, float, float]], L: int) -> float | None:
        """Capital-gains overhang.

        Two constructions, selected by QUANT_CGO_GH (default: Grinblatt-Han).

        SIMPLIFIED (legacy, QUANT_CGO_GH=0):
            CGO_t = Σ_s (P_t - P_{t-s})·V_{t-s} / (P_t · Σ_s V_{t-s})
        a plain turnover-weighted mean past gain.

        GRINBLATT-HAN (default) reconstructs the reference price properly, per
        广发证券 "资本利得突出量CGO与风险偏好" (行为金融因子研究之一) and its 2024
        multi-frequency follow-up:

            RP_t  = (1/k) Σ_{n=1..L} [ V_{t-n} · Π_{s<n}(1 - V) ] · P_{t-n}
            CGO_t = (P_t - RP_t) / RP_t

        The Π(1-V) factor is the SURVIVAL probability -- the chance a unit
        bought at t-n has not been traded away since. The simplified form drops
        it, so a bar whose holders have long since turned over keeps full
        weight and the factor degenerates into volume-weighted momentum.
        Restoring it measured Sharpe 2.28 vs 1.93 at L=7, and beat the
        simplified form at 6 of 7 lookbacks tested
        (scripts/research_cgo_gh.py).

        Crypto has no share count, so turnover-rate is proxied by each bar's
        share of lookback volume -- bounded in [0,1), preserving the meaning
        that heavily-traded bars wash out the holders before them.
        """
        if len(dbars) <= L:
            return None
        P = [b[1] for b in dbars]
        V = [b[2] for b in dbars]
        Pt = P[-1]
        if Pt <= 0:
            return None

        if not _QUANT_CGO_GH:
            num = sum((Pt - P[-1 - s]) * V[-1 - s] for s in range(1, L + 1))
            den = Pt * sum(V[-1 - s] for s in range(1, L + 1))
            if den <= 0:
                return None
            return num / den

        px = P[-1 - L : -1]  # P_{t-L} .. P_{t-1}, oldest first
        vol = V[-1 - L : -1]
        tot = float(sum(vol))
        if tot <= 0 or not math.isfinite(tot):
            return None
        turn = [v / tot for v in vol]
        # weight_n = V_n · Π over MORE RECENT bars of (1 - V): walk newest → oldest
        weights = [0.0] * len(px)
        survive = 1.0
        for i in range(len(px) - 1, -1, -1):
            weights[i] = turn[i] * survive
            survive *= 1.0 - turn[i]
        k = sum(weights)
        if k <= 0 or not math.isfinite(k):
            return None
        rp = sum(w * p for w, p in zip(weights, px)) / k
        if rp <= 0 or not math.isfinite(rp):
            return None
        return (Pt - rp) / rp

    def _cgo_filter(self, weeks_close: dict, window_end: int) -> list[str]:
        """Return symbols that pass the CGO screen (low overhang) for this rebalance."""
        scores: dict[str, float] = {}
        for s in weeks_close:
            dbars = self._daily_bars(s, window_end)
            cgo_val = self._cgo(dbars, _QUANT_CGO_L)
            if cgo_val is not None and math.isfinite(cgo_val):
                scores[s] = cgo_val
        if len(scores) < self._min_symbols:
            return []
        vals = sorted(scores.values())
        n = len(vals)
        k = max(2, int(round(_QUANT_CGO_Q * n)))
        # dir=-1: keep LOW CGO (no profit-taking overhang); dir=+1: keep HIGH CGO
        thr_idx = n - k if _QUANT_CGO_DIR == -1 else k - 1
        thr_val = vals[thr_idx]
        keep = [
            s for s, v in scores.items() if (v <= thr_val if _QUANT_CGO_DIR == -1 else v >= thr_val)
        ]
        logger.info(
            "asym: CGO filter L=%d q=%.2f dir=%d kept %d/%d symbols (thr %.4f)",
            _QUANT_CGO_L,
            _QUANT_CGO_Q,
            _QUANT_CGO_DIR,
            len(keep),
            len(scores),
            thr_val,
        )
        return keep

    # ── FAS_avg + SMB score (faithful to scripts/research_fas_combo.py) ───────

    def _fas_scores(self, window_end: int) -> dict[str, float]:
        """Return {symbol: z(FAS_avg + SMB)} at the rebalance week. Empty -> FLAT."""
        max_h = self._max_horizon
        weeks_close: dict[str, dict[int, float]] = {}
        weeks_fund: dict[str, dict[int, float]] = {}
        weeks_vol: dict[str, dict[int, float]] = {}
        for s in self._universe:
            wc = self._weekly_close(s)
            wf = self._weekly_fund(s)
            wv = self._weekly_volume(s)
            # Funding requirement: FAS regresses the SINGLE scoring-week accrual
            # on price returns (research_fas_clean.fas_scores: "Accrual is the
            # SINGLE week-w funding sum ... we do NOT sum the h-week window").
            # The 26w horizon needs old CLOSES, not 26w of funding. Demanding
            # max_h+1 weeks of funding therefore discarded ~27 of the 55 weeks
            # of available funding before scoring anything, halving the
            # evaluation window versus the research book on identical data.
            if (
                len(wc) >= max_h + 4
                and len(wf) >= _QUANT_MIN_FUND_W
                and len(wv) >= self._smb_weeks + 1
            ):
                weeks_close[s] = wc
                weeks_fund[s] = wf
                weeks_vol[s] = wv
        if len(weeks_close) < self._min_symbols:
            logger.warning(
                "asym: insufficient symbols with full history (%d < %d) -> FLAT",
                len(weeks_close),
                self._min_symbols,
            )
            return {}

        # ── CGO / RCGO₿ selection gate ───────────────────────────────────────
        # RCGO₿ ON (W>0): continuous blend -> keep ALL symbols with full history
        # (no hard gate; the disposition tilt is added below as a soft factor).
        # RCGO₿ OFF: hard Griffin-Han CGO pre-filter (drop overhang names).
        if _QUANT_RCGO_W > 0:
            valid = list(weeks_close.keys())
            logger.info(
                "asym: RCGO₿ blend ON (w=%.2f dir=%d ortho=%s) -> %d symbols",
                _QUANT_RCGO_W,
                _QUANT_RCGO_DIR,
                _QUANT_RCGO_ORTHO,
                len(valid),
            )
        else:
            # ── CGO (Griffin-Han) pre-filter: drop names with high unrealized-gain
            # overhang (disposition-selling pressure caps rallies). Computed on DAILY
            # closes+volumes (paper: weekly is invalid for the CGO effect). G/r logic:
            # keep low-CGO names (dir=-1) => no profit-taking supply overhang.
            valid = self._cgo_filter(weeks_close, window_end)
            if len(valid) < self._min_symbols:
                logger.info(
                    "asym: CGO filter left %d symbols (%d < %d) -> FLAT",
                    len(valid),
                    len(valid),
                    self._min_symbols,
                )
                return {}

        all_weeks: set[int] = set()
        for s in valid:
            all_weeks |= weeks_close[s].keys() & weeks_fund.get(s, {}).keys()
        week_list = sorted(all_weeks)
        if len(week_list) < max_h + 4:
            return {}
        # Score the most recent SUFFICIENTLY-POPULATED week at-or-before the
        # rebalance point (window_end). Using the global last week breaks the
        # replay harness (cached data ends on a sparse partial week) and would
        # also be look-ahead; this scores the completed week of the rebalance
        # for live and the actual week per rebalance for replay.
        window_week = _week_end_ms(window_end)
        last_wk = None
        for w in reversed(week_list):
            if w > window_week:
                continue
            pop = sum(
                1 for s in valid if w in weeks_close.get(s, {}) and w in weeks_fund.get(s, {})
            )
            if pop >= self._min_symbols:
                last_wk = w
                break
        if last_wk is None:
            return {}

        # accrual per week (same for all horizons)
        accr_w: dict[int, dict[str, float]] = {}
        for w in week_list:
            accr: dict[str, float] = {}
            for s in valid:
                fset = weeks_fund.get(s, {})
                if w in fset:
                    accr[s] = fset[w]
            accr_w[w] = accr

        # per-horizon FAS_h = -z(pr_h) * z(resid on [pr_h, |pr_h|])
        fas_by_h: dict[int, dict[int, dict[str, float]]] = {h: {} for h in self._horizons}
        for h in self._horizons:
            pr_w: dict[int, dict[str, float]] = {}
            for w in week_list:
                pr: dict[str, float] = {}
                for s in valid:
                    cset = weeks_close[s]
                    if w not in cset:
                        continue
                    wc_w = cset[w]
                    wc_hw = cset.get(w - h * _WEEK_MS)
                    if wc_hw is None or wc_hw <= 0:
                        continue
                    pr[s] = wc_w / wc_hw - 1.0
                pr_w[w] = pr
            resid_w: dict[int, dict[str, float]] = {}
            for w in week_list:
                a = accr_w[w]
                p = pr_w[w]
                syms = [s for s in a if s in p and math.isfinite(a[s]) and math.isfinite(p[s])]
                if len(syms) < 10:
                    resid_w[w] = {}
                    continue
                ys = np.array([a[s] for s in syms], dtype=float)
                X = np.column_stack(
                    [
                        np.ones(len(syms)),
                        np.array([p[s] for s in syms], dtype=float),
                        np.array([abs(p[s]) for s in syms], dtype=float),
                    ]
                )
                beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
                pred = X @ beta
                resid_w[w] = {s: float(ys[i] - pred[i]) for i, s in enumerate(syms)}
            for w in week_list:
                r = resid_w[w]
                p = pr_w[w]
                if not r or not p:
                    continue
                rz = _rank_z(r)
                pz = _rank_z(p)
                fas_by_h[h][w] = {s: -pz.get(s, 0.0) * rz.get(s, 0.0) for s in r}

        # FAS_avg at last_wk = mean of FAS_h across horizons
        fas_last: dict[str, float] = {}
        for s in valid:
            vals = [fas_by_h[h].get(last_wk, {}).get(s) for h in self._horizons]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if vals:
                fas_last[s] = float(np.mean(vals))

        # SMB at last_wk = -z(log trailing smb_weeks volume)  (long small-caps)
        smb_last: dict[str, float] = {}
        for s in valid:
            wv = weeks_vol[s]
            tot = sum(wv.get(last_wk - k * _WEEK_MS, 0.0) for k in range(self._smb_weeks))
            if tot > 0 and math.isfinite(tot):
                smb_last[s] = math.log(tot)
        if not smb_last:
            logger.warning("asym: no volume history for SMB -> FLAT")
            return {}
        smb_z = _rank_z(smb_last)
        smb_score = {s: -v for s, v in smb_z.items()}  # small cap -> positive

        # ── RCGO₿ (INVENTED, OUR book) continuous disposition tilt ──────────
        # RCGO_b[s] = residual of daily CGO on the cross-sectional funding-carry
        # (Griffin-Han / Guangfa RCGO construction, applied to crypto carry). The
        # residual is pure behavioral overhang net of the mechanical crowding that
        # FAS already captures. Blended continuously, NOT as a hard gate (CGO is a
        # continuous factor; a hard q-filter throws away the gradient).
        rcgo_blend: dict[str, float] = {}
        if _QUANT_RCGO_W > 0:
            cgo_raw: dict[str, float] = {}
            for s in valid:
                c = self._cgo(self._daily_bars(s, last_wk), _QUANT_CGO_L)
                if c is not None and math.isfinite(c):
                    cgo_raw[s] = c
            cgo_z = _rank_z(cgo_raw)
            carry_z = _rank_z(accr_w.get(last_wk, {}))
            ortho = cgo_z
            if _QUANT_RCGO_ORTHO and carry_z:
                syms = [
                    s
                    for s in cgo_z
                    if s in carry_z and math.isfinite(cgo_z[s]) and math.isfinite(carry_z[s])
                ]
                if len(syms) >= 10:
                    y = np.array([cgo_z[s] for s in syms], dtype=float)
                    X = np.column_stack(
                        [np.ones(len(syms)), np.array([carry_z[s] for s in syms], dtype=float)]
                    )
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ beta
                    ortho = _rank_z({syms[i]: float(resid[i]) for i in range(len(syms))})
            rcgo_blend = {s: _QUANT_RCGO_W * _QUANT_RCGO_DIR * v for s, v in ortho.items()}
            logger.info(
                "asym: RCGO₿ tilt w=%.2f dir=%d ortho=%s on %d symbols",
                _QUANT_RCGO_W,
                _QUANT_RCGO_DIR,
                _QUANT_RCGO_ORTHO,
                len(rcgo_blend),
            )

        # FACC leg (OUR novel Funding-Acceleration Crowding; scripts/research_facc.py).
        # -z(funding_t - mean(funding | trailing W weeks)) = leverage-crowding ONSET.
        # Volume-confirmed: act only where this week's quote-volume > trailing mean
        # (new leverage actually entering; our keyless proxy for OI growth).
        facc_score: dict[str, float] = {}
        facc_z: dict[str, float] = {}
        if self._use_facc:
            facc_accel: dict[str, float] = {}
            for s in valid:
                fset = weeks_fund[s]
                if last_wk not in fset:
                    continue
                trail = [fset.get(last_wk - k * _WEEK_MS) for k in range(1, self._facc_weeks + 1)]
                trail = [x for x in trail if x is not None and math.isfinite(x)]
                if len(trail) < max(2, self._facc_weeks // 2):
                    continue
                facc_accel[s] = float(fset[last_wk]) - float(np.mean(trail))
            for s in facc_accel:
                vset = weeks_vol.get(s, {})
                if last_wk not in vset:
                    continue
                vtrail = [
                    vset.get(last_wk - k * _WEEK_MS) for k in range(1, self._facc_vol_weeks + 1)
                ]
                vtrail = [x for x in vtrail if x is not None and math.isfinite(x)]
                if vtrail and vset[last_wk] > float(np.mean(vtrail)):
                    facc_score[s] = -facc_accel[s]  # accel up -> crowd-long -> SHORT
            facc_z = _rank_z(facc_score) if facc_score else {}

        # FCARRY leg (OUR cross-sectional FUNDING CARRY; research: Keel funding-carry
        # Sharpe 1.69-2.15 / MDD ~12%; BIS+CMU crypto carry Sharpe up to ~11 on the
        # spot-futures basis). Long the LOWEST-funding names (you RECEIVE funding),
        # short the HIGHEST (crowded longs pay you). Pure carry, market-neutral,
        # decoupled from price direction — the most robust crypto factor we can build
        # from keyless funding. Trailing sum of weekly funding accrual = carry signal.
        fcarry_z: dict[str, float] = {}
        if os.environ.get("QUANT_FCARRY_ON") == "1":
            fcarry_sum: dict[str, float] = {}
            for s in valid:
                fset = weeks_fund[s]
                tot = sum(fset.get(last_wk - k * _WEEK_MS, 0.0) for k in range(self._facc_weeks))
                if math.isfinite(tot):
                    fcarry_sum[s] = tot
            fcarry_z = {s: -v for s, v in _rank_z(fcarry_sum).items()}  # low funding -> long

        # TSMOM leg (vol-scaled TIME-SERIES momentum; research: volume-weighted
        # TSMOM Sharpe 2.17; Barroso-Santa-Clara vol-scaling turns raw crypto
        # momentum from negative to +1.86-2.40%/wk). Per-coin risk-adjusted
        # momentum (return / realized vol per horizon), cross-sectional z. Long
        # high risk-adjusted momentum, short low — the STRONG crypto factor
        # (cross-sectional momentum is weak; time-series is strong).
        tsmom_z: dict[str, float] = {}
        if os.environ.get("QUANT_TSMOM_ON") == "1":
            tsmom_raw: dict[str, float] = {}
            for s in valid:
                cset = weeks_close[s]
                tot = 0.0
                n = 0
                for h in self._horizons:
                    wcw = cset.get(last_wk)
                    wch = cset.get(last_wk - h * _WEEK_MS)
                    if not wcw or not wch or wch <= 0:
                        continue
                    r = wcw / wch - 1.0
                    rets = []
                    for k in range(1, h + 1):
                        a = cset.get(last_wk - (k - 1) * _WEEK_MS)
                        b = cset.get(last_wk - k * _WEEK_MS)
                        if a and b and b > 0:
                            rets.append(a / b - 1.0)
                    if len(rets) >= 2:
                        sig = statistics.pstdev(rets)
                        if sig > 1e-9:
                            tot += r / sig
                            n += 1
                if n > 0:
                    tsmom_raw[s] = tot / n
            tsmom_z = _rank_z(tsmom_raw)

        # REV leg (multi-horizon reversal; KNOWN factor, scripts/research_nova.py).
        # Off by default: the live FAS+SMB book already out-scores REV+SMB+FAS, so
        # REV tends to dilute it. Enabled only for exact ENSEMBLE4 parity.
        rev_z: dict[str, float] = {}
        if self._use_rev:
            rev_acc: dict[str, float] = {}
            for h in (4, 8, 12):
                prh: dict[str, float] = {}
                for s in valid:
                    cset = weeks_close[s]
                    wcw = cset.get(last_wk)
                    wch = cset.get(last_wk - h * _WEEK_MS)
                    if wcw is None or wch is None or wch <= 0 or not math.isfinite(wcw):
                        continue
                    prh[s] = wcw / wch - 1.0
                rz = _rank_z(prh)
                for s, v in rz.items():
                    rev_acc[s] = rev_acc.get(s, 0.0) + v
            rev_z = {s: -v for s, v in _rank_z(rev_acc).items()}

        # combined = FAS_avg + SMB [+ FACC] [+ REV], then cross-sectional z.
        # A/B toggles (empirical diagnosis, not hardcoded strategy): QUANT_FAS_SIGN
        # flips the FAS interaction sign (Keel: "funding sign flip is the #1 DIY
        # bug"); QUANT_SMB_OFF disables the small-cap leg to isolate it.
        fas_sign = float(os.environ.get("QUANT_FAS_SIGN", "1"))
        smb_off = os.environ.get("QUANT_SMB_OFF") == "1"
        combined: dict[str, float] = {}
        for s in fas_last:
            if not smb_off and s not in smb_score:
                continue
            score = fas_sign * fas_last[s]
            if not smb_off:
                score += smb_score[s]
            if self._use_facc and s in facc_z:
                score += facc_z[s]
            if os.environ.get("QUANT_FCARRY_ON") == "1" and s in fcarry_z:
                score += fcarry_z[s]
            if os.environ.get("QUANT_TSMOM_ON") == "1" and s in tsmom_z:
                score += tsmom_z[s]
            if self._use_rev and s in rev_z:
                score += rev_z[s]
            if _QUANT_RCGO_W > 0 and s in rcgo_blend:
                score += rcgo_blend[s]
            combined[s] = score
        if len(combined) < self._min_symbols:
            return {}
        return _rank_z(combined)


    # ── research-parity scoring (single source of truth) ──────────────────────

    def _frames(self, window_end: int):
        """Live rolling registry -> the exact frames research_fas_clean.load builds.

        Rebuilt with the SAME pandas resample calls the research loader uses, so
        bucketing cannot drift. Strictly point-in-time: only bars/funding events
        stamped at or before ``window_end`` are included.
        """
        import pandas as pd

        h_close, h_vol, w_close, w_vol, w_accr, d_close, d_vol = {}, {}, {}, {}, {}, {}, {}
        for sym in self._universe:
            rows = [r for r in self._closes.get(sym, ()) if r[0] <= window_end]
            if not rows:
                continue
            idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
            cl = pd.Series([r[1] for r in rows], index=idx, dtype=float)
            vl = pd.Series([r[2] for r in rows], index=idx, dtype=float)
            h_close[sym], h_vol[sym] = cl, vl
            w_close[sym] = cl.resample("W-MON").last()
            w_vol[sym] = vl.resample("W-MON").sum()
            d_close[sym] = cl.resample("D").last()
            d_vol[sym] = vl.resample("D").sum()
            frows = [r for r in self._funding.get(sym, ()) if r[0] <= window_end]
            if frows:
                fi = pd.to_datetime([r[0] for r in frows], unit="ms", utc=True)
                w_accr[sym] = pd.Series(
                    [r[1] for r in frows], index=fi, dtype=float
                ).resample("W-MON").sum()
        return (
            pd.DataFrame(w_close),
            pd.DataFrame(w_vol),
            pd.DataFrame(w_accr),
            pd.DataFrame(d_close),
            pd.DataFrame(d_vol),
        )

    def _research_scores(self, window_end: int) -> dict[str, float]:
        """Score the book by CALLING the validated research functions.

        The live class previously reimplemented FAS/SMB/RCGO in streaming form.
        That second implementation drifted from the validated one: rank
        correlation between the two score series measured 0.147 and selection
        overlap 27% (chance is ~20%), so the live book was effectively trading a
        different strategy. Verified separately (scripts/harness_verify.py), the
        research selections pushed through this same execution engine reproduce
        Sharpe 2.11 -- the executor is sound, only the signal diverged.

        The cure is the one the backtest-parity literature prescribes: run the
        SAME code in research and production rather than a reimplementation.
        These functions are the ones research_fas_invent.py validates, called
        directly on point-in-time frames.
        """
        try:
            from scripts.research_fas_clean import fas_scores, smb_scores
            from scripts.research_fas_invent import rcgo_scores
            from scripts.research_fas_clean import _rank_z as _rz
        except Exception:
            logger.exception("asym: research modules unavailable -> legacy scoring")
            return {}

        cw, vw, aw, dcl, dvl = self._frames(window_end)
        if cw.empty or aw.empty:
            return {}
        # Rank against the SAME pool research ranks against. A cross-sectional
        # score is relative: including a name research screened out (a dead or
        # glitchy ticker) shifts every rank and therefore quintile membership.
        # research_fas_clean._liquidity_mask is a tradability screen, not an
        # edge tweak, so applying it here is parity, not curve-fitting.
        try:
            from scripts.research_fas_clean import _liquidity_mask

            tradable = set(_liquidity_mask(cw, vw))
        except Exception:
            tradable = set(cw.columns)
        symbols = [
            s
            for s in self._universe
            if s in cw.columns and s in aw.columns and s in tradable
        ]
        if len(symbols) < self._min_symbols:
            return {}

        fas = fas_scores(cw, aw, symbols, horizons=tuple(self._horizons))
        smb = smb_scores(vw, symbols, weeks_window=self._smb_weeks)
        score = (fas[symbols] + smb[symbols]).apply(_rz)
        if _QUANT_RCGO_W > 0:
            rcgo = rcgo_scores(dcl, dvl, aw, fas.index, symbols, L=_QUANT_CGO_L)
            tilt = (_QUANT_RCGO_W * _QUANT_RCGO_DIR * rcgo[symbols]).apply(_rz)
            score = (score + tilt).apply(_rz)

        # Score the most recent COMPLETED week at or before this rebalance.
        wk = _week_end_ms(window_end)
        import pandas as pd

        cutoff = pd.Timestamp(wk, unit="ms", tz="UTC")
        usable = [t for t in score.index if t <= cutoff and score.loc[t].abs().sum() > 0]
        if not usable:
            return {}
        scored_week = max(usable)
        # Record which week actually produced the scores. fas_scores excludes
        # the final week (range(..., n_w - 1)), so the newest Monday is always
        # zero and the book scores the last COMPLETED week -- correct live, but
        # parity harnesses must compare against THAT week's research selection,
        # not the calendar week of the rebalance instant.
        self._last_scored_week = int(scored_week.timestamp() * 1000)
        row = score.loc[scored_week]
        return {s: float(v) for s, v in row.items() if math.isfinite(v)}

    # ── selection + emit ──────────────────────────────────────────────────────

    # ── SRP (Self-Referential Parity) ─────────────────────────────────────────
    #
    # QUANT_STRATEGY=srp swaps the book for the validated SRP construction:
    # nine factors, each ranked against its OWN 52w history, each trading its own
    # quintile book, combined by inverse-vol risk parity, funding-tilted and
    # turnover-capped. Measured by scripts/srp_backtest.py over 282 weekly
    # rebalances: Sharpe 2.161 / t 5.03 at THIS config (turnover_cap 0.60),
    # 2.364 / 5.51 uncapped, net of funding and liquidity-scaled maker costs.
    # The "1.03 conventional baseline" this comment used to cite has no script
    # behind it; scripts/srp_sweep.py is measuring the real gap.
    #
    # The legacy FAS+SMB+RCGO path below is left intact and reachable by unsetting
    # the flag: FAS measured IC +0.0065 (indistinguishable from zero) and RCGO
    # degraded every configuration tested, so this is a replacement, not a tweak.
    #
    # NOTE the BTC regime gate is deliberately BYPASSED for SRP. It was validated
    # for the legacy book; on SRP it changes Sharpe 2.37 -> 2.35 while forcing 81
    # of 220 weeks flat and costing a full point of t-stat. Running it would mean
    # trading a different strategy than the one that was validated.
    def _srp_selection(self, window_end: int) -> dict[str, tuple[str, float]] | None:
        try:
            from stream.srp_live import SRPBook
        except Exception:
            logger.exception("asym: SRP module unavailable -> legacy path")
            return None
        if getattr(self, "_srp_book", None) is None:
            try:
                self._srp_book = SRPBook(list(self._universe))
                self._srp_prev: dict = {}
            except Exception:
                logger.exception("asym: SRP seed failed -> legacy path")
                return None
        try:
            cw, vw, aw, _dcl, _dvl = self._frames(window_end)
            if cw.empty or aw.empty:
                # Report WHICH frame is empty and how much registry there is to
                # build it from -- "empty frames" alone is not actionable.
                logger.warning(
                    "asym/SRP: empty frames -> FLAT (close=%s funding=%s | "
                    "universe=%d closes_syms=%d funding_syms=%d window_end=%d)",
                    cw.shape, aw.shape, len(self._universe),
                    sum(1 for s in self._universe if self._closes.get(s)),
                    sum(1 for s in self._universe if self._funding.get(s)),
                    window_end,
                )
                return {s: ("FLAT", 0.0) for s in self._universe}
            dirs, books, port = self._srp_book.score(cw, vw, aw, self._srp_prev)
        except Exception:
            logger.exception("asym/SRP: scoring failed -> FLAT (never guesses)")
            return {s: ("FLAT", 0.0) for s in self._universe}
        self._srp_prev = books or {}
        out: dict[str, tuple[str, float]] = {}
        for s in self._universe:
            d = dirs.get(s, "FLAT")
            w = float(port[s]) if (port is not None and s in port.index) else 0.0
            out[s] = (d, w)
        return out

    def _selection(self, window_end: int) -> dict[str, tuple[str, float]]:
        out: dict[str, tuple[str, float]] = {s: ("FLAT", 0.0) for s in self._universe}

        if _QUANT_STRATEGY == "srp":
            srp = self._srp_selection(window_end)
            if srp is not None:
                return srp
            logger.warning("asym: SRP unavailable, falling back to legacy book")

        if self._regime:
            mhist = self._closes.get(self._market)
            if not mhist or len(mhist) < self._regime_slow_days * 24 + 1:
                logger.warning(
                    "asym: REGIME skip (insufficient BTC history %d < %d)",
                    len(mhist) if mhist else 0,
                    self._regime_slow_days * 24 + 1,
                )
                return out
            ends_m, mcloses, _ = self._series(mhist)
            if not self._daily_up(mcloses, ends_m, self._regime_slow_days):
                logger.warning("asym: REGIME gate DOWN -> FLAT")
                return out

        score = (
            self._research_scores(window_end)
            if _QUANT_RESEARCH_PARITY
            else self._fas_scores(window_end)
        )
        if not score:
            logger.warning("asym: fas_scores EMPTY -> FLAT")
            return out

        cand = sorted(score.items(), key=lambda x: x[1])
        n = max(2, round(len(cand) * self._quintile))
        longs = cand[-n:]
        shorts = cand[:n]
        for s, v in longs:
            out[s] = ("LONG", float(v))
        for s, v in shorts:
            out[s] = ("SHORT", float(v))
        return out

    def handle(self, msg: dict) -> None:
        symbol = str(msg.get("symbol") or "").upper()
        if not symbol:
            return
        close = msg.get("close")
        if not isinstance(close, (int, float)) or close != close or close == 0:
            return
        window_end = msg.get("window_end_ms")
        window_end = int(window_end) if isinstance(window_end, (int, float)) else None
        if window_end is None:
            return
        volume = float(msg.get("volume") or 0.0)
        self._record(symbol, window_end, float(close), volume)

        # A LIVE book must never rebalance off a stale bar. Retained Kafka
        # messages from earlier runs replay on restart, and a rebalance driven by
        # one scores the cross-section at that old instant: observed live as a
        # book stuck on window_end=2025-10-07 while the feed was publishing
        # 2026-08-15, producing an all-FLAT portfolio because funding history
        # did not exist that far back. The bar is still RECORDED above (history
        # is useful); only the rebalance decision is gated.
        #
        # Threshold is expressed in rebalance periods rather than a fixed
        # constant so it scales with the configured cadence.
        if not self._replay:
            age_ms = int(time.time() * 1000) - window_end
            max_age = self._stale_bar_periods * self._rebalance_h * _HOUR_MS
            if age_ms > max_age:
                if not getattr(self, "_stale_warned", False):
                    logger.warning(
                        "asym: ignoring STALE bar for rebalance (window_end=%d is "
                        "%.1fh old, limit %.1fh) -- replaying retained messages; "
                        "history still recorded",
                        window_end, age_ms / 3_600_000, max_age / 3_600_000,
                    )
                    self._stale_warned = True
                return

        week = window_end // (self._rebalance_h * _HOUR_MS)
        if self._last_week is None or week != self._last_week:
            if not self._replay:
                # Live path: fold in any funding settled since the last tick.
                # Replay path: the warm-start seed already covers the full
                # historical window, so skip ~15k network calls (Keel-style
                # "compute once, reuse many" — see research on backtest speed).
                for s in self._universe:
                    self._append_funding(s)
            sel = self._selection(window_end)
            # Only LATCH a rebalance that actually produced a book. A transient
            # failure (frames not yet populated at startup, a feed hiccup, the
            # risk model not warm) previously set ``_last_week`` anyway, so an
            # all-FLAT result froze the book until the NEXT bucket -- a full week
            # at the 168h cadence. Retrying on the following bar costs one
            # scoring pass and cannot produce a worse book than staying flat.
            if any(d != "FLAT" for d, _ in sel.values()):
                self._current = sel
                self._last_week = week
            else:
                self._current = sel
                logger.warning(
                    "asym: rebalance produced an all-FLAT book; NOT latching, "
                    "will retry on the next bar"
                )

        for s in self._universe:
            direction, yhat = self._current.get(s, ("FLAT", 0.0))
            self._kv.set_json(
                prediction_key(self._prediction_prefix, s),
                {
                    "symbol": s,
                    "window_end_ms": window_end,
                    # NOT a return. ``yhat`` is _fas_scores() -> _rank_z(), a
                    # CROSS-SECTIONAL RANK in [-1, +1] (pct_rank * 2 - 1), so
                    # +1.0 means "top-ranked symbol this window", not "+100%".
                    # Emitted under its honest name; ``predicted_return`` is
                    # kept as a deprecated alias only because the execution
                    # engine's entry gate reads that key.
                    #
                    # Consequence, deliberately preserved: the engine's cost
                    # filter compares |yhat| against lambda * taker_fee
                    # (0.002), so a rank score always clears it and the filter
                    # never binds. That matches the ranked-quintile book the
                    # research validated (Sharpe 1.82 full / 1.93 post-2024);
                    # making it bind would diverge from that backtest.
                    "signal_score": round(float(yhat), 6),
                    "score_scale": "cross_sectional_rank_[-1,1]",
                    "predicted_return": round(float(yhat), 6),
                    "direction": direction,
                    "signal": "asym",
                    "updated_at": self._kv_now(),
                },
            )

    # ── warm-start disk cache ─────────────────────────────────────────────────
    # Replays re-run warm-start every time; keyless REST + funding paging for 31
    # symbols is the dominant cost. Cache the seed to disk (24h TTL) so a re-run
    # is instant — "compute once, reuse many" (backtest-speed research).

    @staticmethod
    def _warm_cache_path() -> str:
        base = os.environ.get("QUANT_CACHE_DIR", "/tmp/quant_cache")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "asym_warm_start.json")

    def _load_warm_cache(self) -> dict | None:
        path = self._warm_cache_path()
        try:
            with open(path) as fh:
                cache = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        # Validate it covers the full universe we need.
        if set(s.upper() for s in self._universe) - set(cache.get("bars", {})):
            return None
        return cache

    def _save_warm_cache(self) -> None:
        path = self._warm_cache_path()
        cache = {
            "ts": int(time.time() * 1000),
            "bars": {s: self._closes.get(s, []) for s in self._universe},
            "funding": {s: self._funding.get(s.upper(), []) for s in self._universe},
        }
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(cache, fh)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("asym warm cache save failed: %s", e)

    def warm_start(self, settings) -> None:
        """Seed hourly closes (like SCX) + keyless Bybit funding history for the universe."""
        if not settings.stream_asym_warm_start:
            return
        if os.environ.get("QUANT_WARM_CACHE", "1") != "0":
            cache = self._load_warm_cache()
            if cache is not None and (int(time.time() * 1000) - cache["ts"]) < _DAY_MS:
                for s in self._universe:
                    sym = s.upper()
                    for end_ms, close, volume in cache["bars"].get(sym, []):
                        self._record(s, int(end_ms), float(close), float(volume))
                    self._funding[sym] = [
                        (int(e), float(r)) for e, r in cache["funding"].get(sym, [])
                    ]
                logger.info("asym warm-start loaded %d symbols from cache", len(self._universe))
                return
        from ingest.providers.bybit import BybitBarProvider

        provider = BybitBarProvider(base_url=settings.stream_bybit_base_url)
        # Seed > 365 daily bars so the 52-week BTC regime gate passes on the
        # first rebalance (need slow_days + 1 daily closes; +20d buffer).
        needed = (self._regime_slow_days + 20) * 24
        for s in self._universe:
            bars = self._warm_fetch(provider, s, needed)
            for end_ms, close, volume in bars:
                self._record(s, int(end_ms), float(close), float(volume))
            self._refresh_funding(s)
            logger.info(
                "asym warm-start %s: %d bars, %d funding events",
                s,
                len(bars),
                len(self._funding.get(s, [])),
            )
        self._save_warm_cache()

    @staticmethod
    def _warm_fetch(provider, symbol: str, needed: int) -> list[tuple[int, float, float]]:
        bars: list[tuple[int, float, float]] = []
        end: int | None = None
        while len(bars) < needed:
            chunk = provider.fetch_klines_1h(
                symbol, limit=2000, end_ms=end, category=_WARM_CATEGORY
            )
            if not chunk:
                break
            bars = chunk + bars
            end = chunk[0][0] - 1
            if len(chunk) < 2000:
                break
        return bars[-needed:] if bars else bars

    @staticmethod
    def _kv_now() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

    def run_forever(
        self,
        bus: MessageBus,
        features_topic: str,
        group_id: str,
        stop: threading.Event | None = None,
    ) -> None:
        for _topic, msg in bus.iter_consume(features_topic, group_id, stop=stop):
            self.handle(msg)


if __name__ == "__main__":
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    universe = csv_list(settings.stream_xs_universe)
    signal = AsymSignal(
        kv,
        prediction_prefix=settings.stream_asym_prediction_prefix,
        universe=universe,
        rebalance_h=settings.stream_xs_rebalance_h,
        quintile=settings.stream_asym_quintile,
        min_symbols=settings.stream_asym_min_symbols,
        regime=settings.stream_asym_regime,
        regime_slow_days=settings.stream_asym_regime_slow_days,
        market_symbol=settings.stream_asym_market_symbol,
        horizons=settings.stream_asym_horizons,
        accrual_weeks=settings.stream_asym_accrual_weeks,
        smb_weeks=settings.stream_asym_smb_weeks,
        use_facc=True,
        use_rev=False,
    )
    if settings.stream_asym_warm_start:
        signal.warm_start(settings)
        logger.info("asym warm-start complete (%d symbols seeded)", len(signal._closes))
    logger.info(
        "asym (FAS_avg+SMB) signal consuming %s -> %s (universe=%d)",
        settings.stream_kafka_topic_features,
        settings.stream_asym_prediction_prefix,
        len(universe),
    )
    try:
        signal.run_forever(bus, settings.stream_kafka_topic_features, group_id="asym-signal")
    except KeyboardInterrupt:
        logger.info("asym signal stopped")
