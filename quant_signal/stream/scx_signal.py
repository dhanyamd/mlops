"""Live SCX signal -- Skew-Convex, Regime-Gated Cross-Sectional Momentum.

Sibling of ``stream.xs_signal`` (and ``stream.lcf_signal``). Consumes the SAME
1h feature topic (``crypto.features.1h``) and, instead of the online regressor or
the plain momentum book, runs the VALIDATED **SCX** factor and writes its
directions into the online store as ``prediction:crypto:1h:<SYMBOL>`` -- the
exact payload ``stream.execution.PaperExecutionSimulator`` already reads, so the
executor, the Bybit demo venue and the dashboard are untouched.

SCX (validated in ``scripts/research_scx.py``, WF OOS 2017-2026, 456 weeks, 31
coins, 10bps, Binance daily): Sharpe 1.13 (REGIME_LONG 1.18, SCX_VOL 0.96) vs
momentum 0.76 and funding-LEVEL -1.79. The novel edge is NOT static vol-scaling
(TAIL proved BSC vol-scaling hurts crypto XS momentum) but a **skew-aware
net-exposure overlay**:

  - BTC UP-UP regime gate (flat in bears): BTC above both its 90-day and 200-day
    MAs (on a daily resample of the 1h stream). Without history the gate fails
    CLOSED (FLAT) -- never trade blind.
  - always run the LONG (winner) book in bull regime,
  - SHORT exposure is CONDITIONAL on the short side being "calm": when trailing
    short-book realized vol (12-week window) is in its stressed quantile
    (walk-forward-selected stress_q=0.60), shorts are dropped and the book goes
    long-only. The short-book return history is built lookahead-free: each week's
    short-book realized return is only recorded at the NEXT rebalance, once the
    week has actually closed.

Live translation of the research daily panel (faithful, no new magic numbers):
  - 14-day momentum = 336 hourly bars  (close / close.shift(336) - 1)
  - regime 90/200-DAY BTC MAs, computed on the daily resample of BTC hourly closes
  - short-book vol = rolling 12-week std of the per-week bottom-quintile book return

The execution engine keys off the per-window ``direction`` + ``predicted_return``
it already consumes, HOLDs through weaker same-side signals and flips only past
the 2*lambda*cost band -- so a weekly-stable LONG/SHORT (FLAT mid-quintile) falls
out as a clean weekly rebalance. No lookahead: every input uses closes strictly
before the rebalance bar.

Run with ``STREAM_STRATEGY=scx`` (the launchd service boots ``stream.xs_signal``
and dispatches to ``ScxSignal``). Pure logic lives in ``handle`` so tests drive it
directly with ``FakeBus``/``FakeKV``; warm-start seeds BTC to 200+ days (so the
regime gate is live on the first bar) and the rest of the universe to 336h.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

import numpy as np

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_DAY_MS = 24 * _HOUR_MS


def prediction_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


class ScxSignal:
    """Skew-convex, regime-gated cross-sectional weekly book, fed by the 1h stream.

    ``_closes[symbol]`` is a rolling registry of ``(window_end_ms, close,
    volume)`` strictly in time order (Kafka may deliver symbols out of order, so
    every read sorts). At each weekly UTC rebalance boundary the universe is
    ranked and the selection cached; every subsequent bar in the week re-emits the
    cached direction so the executor HOLDs until the next rebalance flips it.
    Best-effort warm-start seeds close history from venue 1h klines.
    """

    def __init__(
        self,
        kv: KVStore,
        *,
        prediction_prefix: str,
        universe: list[str],
        lookback_h: int = 336,
        quintile: float = 0.20,
        min_symbols: int = 8,
        rebalance_h: int = 168,
        regime: bool = True,
        regime_fast_days: int = 90,
        regime_slow_days: int = 200,
        shorts: bool = True,
        cond_short: bool = True,
        short_vol_l: int = 12,
        stress_q: float = 0.60,
        market_symbol: str = "BTCUSDT",
        max_history: int = 5300,
    ) -> None:
        self._kv = kv
        self._prediction_prefix = prediction_prefix
        self._universe = [s.upper() for s in universe]
        self._lookback_h = lookback_h
        self._quintile = quintile
        self._min_symbols = min_symbols
        self._rebalance_h = rebalance_h
        self._regime = regime
        self._regime_fast_days = regime_fast_days
        self._regime_slow_days = regime_slow_days
        self._shorts = shorts
        self._cond_short = cond_short
        self._short_vol_l = short_vol_l
        self._stress_q = stress_q
        self._market = (
            market_symbol.upper()
            if market_symbol.upper() in self._universe
            else (self._universe[0] if self._universe else market_symbol.upper())
        )
        self._max_history = max_history
        self._closes: dict[str, list[tuple[int, float, float]]] = {}
        self._last_week: int | None = None
        self._current: dict[str, tuple[str, float]] = {}
        # Lookahead-free short-book realized-return history + prior selection.
        self._sb_hist: list[float] = []
        self._last_shorts: list[str] = []
        self._last_shorts_entry: dict[str, float] = {}

    # ── history ───────────────────────────────────────────────────────────────

    def _record(self, symbol: str, window_end: int, close: float, volume: float) -> None:
        hist = self._closes.setdefault(symbol, [])
        if hist and hist[-1][0] >= window_end:
            return
        hist.append((window_end, close, volume))
        while len(hist) > self._max_history:
            del hist[0]

    @staticmethod
    def _series(
        hist: Sequence[tuple[int, float, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.array([(e, c, v) for e, c, v in hist], dtype=float)
        ends = arr[:, 0]
        closes = arr[:, 1]
        vols = arr[:, 2]
        return ends, closes, vols

    @staticmethod
    def _last_index(ends: np.ndarray, window_end: int) -> int:
        """Index of the latest close strictly before ``window_end`` (no lookahead)."""
        return int(np.searchsorted(ends, window_end, side="left")) - 1

    # ── BTC UP-UP regime gate (90/200-DAY MA on the daily resample) ───────────

    @staticmethod
    def _daily_up(closes: np.ndarray, ends: np.ndarray, fast_days: int, slow_days: int) -> bool:
        """True when BTC's last daily close is above its fast & slow daily MAs.

        Resamples the hourly BTC closes to daily (last close per UTC day) so the
        gate matches the research daily panel (BTC > 90d & 200d MA). Without
        ``slow_days`` of daily history the gate fails closed (FLAT).
        """
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
        last = daily[-1]
        fast = float(daily[-fast_days:].mean())
        slow = float(daily[-slow_days:].mean())
        return last > fast and last > slow

    # ── conditional-short: is the short side currently stressed? ─────────────

    def _short_stressed(self) -> bool:
        """True when trailing short-book vol is in its stressed quantile.

        Mirrors research_scx.weights_at: rolling ``short_vol_l``-week std of the
        short-book return history; if the latest rolling std exceeds the
        ``stress_q`` quantile of that series, the short side is stressed. Returns
        False (shorts stay on) until enough history accrues -- fails OPEN.
        """
        if not (self._cond_short and self._shorts):
            return False
        if len(self._sb_hist) < self._short_vol_l + 3:
            return False
        arr = np.array(self._sb_hist, dtype=float)
        roller = np.array(
            [arr[i : i + self._short_vol_l].std() for i in range(len(arr) - self._short_vol_l + 1)]
        )
        if len(roller) < 4:
            return False
        # Defensive: a degenerate (effectively zero-vol) short book has no tail to
        # speak of; floating-point noise in std() must not trip the stress flag.
        if float(roller.max()) < 1e-9:
            return False
        thr = float(np.quantile(roller, self._stress_q))
        return bool(roller[-1] > thr)

    # ── signal math (faithful to scripts/research_scx.py) ─────────────────────

    def _selection(self, window_end: int) -> dict[str, tuple[str, float]]:
        """Rank the universe at ``window_end``; return symbol -> (direction, sig)."""
        out: dict[str, tuple[str, float]] = {s: ("FLAT", 0.0) for s in self._universe}

        cand: list[tuple[str, float]] = []
        sigs: dict[str, float] = {}
        entry: dict[str, float] = {}
        for s in self._universe:
            hist = self._closes.get(s)
            if not hist or len(hist) < self._lookback_h + 1:
                continue
            ends, closes, _ = self._series(hist)
            idx = self._last_index(ends, window_end)
            if idx < self._lookback_h:
                continue
            close_now = closes[idx]
            close_lb = closes[idx - self._lookback_h]
            if close_lb <= 0 or not math.isfinite(close_lb) or not math.isfinite(close_now):
                continue
            sig = close_now / close_lb - 1.0
            if not math.isfinite(sig):
                continue
            sigs[s] = sig
            entry[s] = float(close_now)
            cand.append((s, sig))

        if len(cand) < self._min_symbols:
            self._last_shorts = []
            self._last_shorts_entry = {}
            return out

        # Market-level regime gate (whole book FLAT if it fails).
        if self._regime:
            mhist = self._closes.get(self._market)
            if not mhist or len(mhist) < self._regime_slow_days * 24 + 1:
                self._last_shorts = []
                self._last_shorts_entry = {}
                return out
            ends_m, mcloses, _ = self._series(mhist)
            if not self._daily_up(mcloses, ends_m, self._regime_fast_days, self._regime_slow_days):
                self._last_shorts = []
                self._last_shorts_entry = {}
                return out

        cand.sort(key=lambda x: x[1])
        n = max(2, round(len(cand) * self._quintile))
        longs = cand[-n:]
        shorts = cand[:n]

        shorts_on = self._shorts and not self._short_stressed()

        for s, _ in longs:
            out[s] = ("LONG", float(sigs[s]))
        if shorts_on:
            for s, _ in shorts:
                out[s] = ("SHORT", float(sigs[s]))

        # Remember this week's shorts (and their entry closes) for the lookahead-
        # free short-book return bookkeeping at the NEXT rebalance.
        if shorts_on:
            self._last_shorts = [s for s, _ in shorts]
            self._last_shorts_entry = {s: entry[s] for s, _ in shorts}
        else:
            self._last_shorts = []
            self._last_shorts_entry = {}
        return out

    # ── short-book return bookkeeping (lookahead-free) ────────────────────────

    def _record_short_book_return(self, window_end: int) -> None:
        """Append last week's realized short-book return to ``_sb_hist``.

        Called at the START of a new weekly rebalance, BEFORE this week's
        selection. Uses the shorts selected at the PREVIOUS rebalance and their
        realized return from the previous boundary to the current one -- so the
        outcome is fully known (no lookahead), exactly as research_scx builds
        ``ref_short_book_returns`` from strictly-prior weeks.
        """
        if not self._last_shorts:
            return
        rets: list[float] = []
        for s in self._last_shorts:
            hist = self._closes.get(s)
            if not hist:
                continue
            ends, closes, _ = self._series(hist)
            idx = self._last_index(ends, window_end)
            if idx < 0:
                continue
            entry = self._last_shorts_entry.get(s)
            exit_close = float(closes[idx])
            if entry and entry > 0 and exit_close > 0:
                rets.append(exit_close / entry - 1.0)
        if rets:
            self._sb_hist.append(float(np.mean(rets)))
            # Bound the history so it stays a rolling window.
            while len(self._sb_hist) > self._short_vol_l * 6:
                del self._sb_hist[0]

    # ── rebalance + emit ─────────────────────────────────────────────────────

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

        week = window_end // (self._rebalance_h * _HOUR_MS)
        if self._last_week is None or week != self._last_week:
            self._record_short_book_return(window_end)
            self._current = self._selection(window_end)
            self._last_week = week

        # Emit the (weekly-stable) selection for every universe symbol at this
        # bar so the executor reads a fresh prediction per symbol, matched by
        # window_end_ms. Mid-quintile / gated names are FLAT (no trade).
        for s in self._universe:
            direction, yhat = self._current.get(s, ("FLAT", 0.0))
            self._kv.set_json(
                prediction_key(self._prediction_prefix, s),
                {
                    "symbol": s,
                    "window_end_ms": window_end,
                    "predicted_return": round(float(yhat), 6),
                    "direction": direction,
                    "signal": "scx",
                    "updated_at": self._kv_now(),
                },
            )

    def warm_start(self, settings) -> None:
        """Seed close history from the venue's 1h klines (best-effort).

        BTC is pulled to ~200+ days so the 90/200-day regime gate is satisfied
        on the first live bar (no ~200-day dead wait). The rest of the universe
        is pulled to just past the 336h momentum lookback. Bybit caps a page at
        1000 and ``fetch_klines_1h`` at 2000; paginate backward until enough or
        exhausted. Any symbol that fails to fetch simply accumulates live.
        """
        if not settings.stream_scx_warm_start:
            return
        from ingest.providers.bybit import BybitBarProvider

        provider = BybitBarProvider(base_url=settings.stream_bybit_base_url)
        btc_needed = (self._regime_slow_days + 10) * 24
        per_sym_needed = self._lookback_h + 10

        for s in self._universe:
            needed = btc_needed if s == self._market else per_sym_needed
            bars = self._warm_fetch(provider, s, needed)
            for end_ms, close, volume in bars:
                self._record(s, int(end_ms), float(close), float(volume))
            logger.info("scx warm-start %s: %d hourly bars", s, len(bars))

    @staticmethod
    def _warm_fetch(provider, symbol: str, needed: int) -> list[tuple[int, float, float]]:
        bars: list[tuple[int, float, float]] = []
        end: int | None = None
        while len(bars) < needed:
            chunk = provider.fetch_klines_1h(symbol, limit=2000, end_ms=end)
            if not chunk:
                break
            bars = chunk + bars  # older pages prepended -> oldest->newest overall
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
    # The launchd service boots ``stream.xs_signal``; that module's ``main()`` is
    # the single dispatcher and routes ``STREAM_STRATEGY=scx`` here. This block
    # lets ``python -m stream.scx_signal`` run standalone for local smoke tests.
    from config.logging import configure_logging, get_logger
    from config.settings import get_settings
    from stream.bus import KafkaBus
    from stream.kv import RedisKV
    from config.settings import csv_list

    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    universe = csv_list(settings.stream_xs_universe)
    signal = ScxSignal(
        kv,
        prediction_prefix=settings.stream_redis_prediction_prefix,
        universe=universe,
        lookback_h=settings.stream_scx_lookback_h,
        quintile=settings.stream_scx_quintile,
        min_symbols=settings.stream_scx_min_symbols,
        rebalance_h=settings.stream_xs_rebalance_h,
        regime=settings.stream_scx_regime,
        regime_fast_days=settings.stream_scx_regime_fast_days,
        regime_slow_days=settings.stream_scx_regime_slow_days,
        shorts=settings.stream_scx_shorts,
        cond_short=settings.stream_scx_cond_short,
        short_vol_l=settings.stream_scx_short_vol_l,
        stress_q=settings.stream_scx_stress_q,
        market_symbol=settings.stream_scx_market_symbol,
        max_history=settings.stream_scx_max_history,
    )
    if settings.stream_scx_warm_start:
        signal.warm_start(settings)
        logger.info("scx warm-start complete (%d symbols seeded)", len(signal._closes))
    logger.info(
        "scx signal consuming %s -> %s (universe=%d)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_prediction_prefix,
        len(universe),
    )
    try:
        signal.run_forever(bus, settings.stream_kafka_topic_features, group_id="scx-signal")
    except KeyboardInterrupt:
        logger.info("scx signal stopped")
