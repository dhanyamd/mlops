"""Live cross-sectional momentum signal (xs_rel14) — drop-in River replacement.

Consumes the same Flink 1h feature topic the legacy River ``predictor`` did
(``crypto.features.1h``) and, instead of an online regression, runs the
VALIDATED cross-sectional 14-day relative-momentum signal and writes its
directions into the online store as ``prediction:crypto:1h:<SYMBOL>`` — the
exact payload ``stream.execution.PaperExecutionSimulator`` already reads, so
the executor, the Bybit demo venue and the dashboard are untouched. This is the
single probe section that passed the live verdict gate:

    [xs_rel14] PASS: net_maker=+137.9bps (median +116.2, win 0.583)
                     sharpe=1.67  strat_mult=1.336 vs BH=1.3028
                     config: L336h, vol=True, crash=None, regime=True

Signal (mirrors ``scripts/trend_momentum_probe.py`` ``_xs_precompute`` /
``_xs_events`` for ``xs_rel14`` so the live book IS the research, not a rewrite):

  - sig[s] = close / close.shift(lookback_h=336) - 1  (14-day relative momentum)
  - vol-scale = clip( rolling28d-mean(rv24) / rv24 , 0.5, 2.0 )  (inverse-vol)
  - per-symbol liquidity gate: volume >= 0.5 x own trailing 28-bar median
  - market UP-UP regime gate (Cooper/Hsieh): both the current and the prior
    4-week BTC returns >= 0, else the whole book stays FLAT that week
  - weekly UTC rebalance: top ``quintile`` longed, bottom ``quintile`` shorted
    (momentum direction); min ``min_symbols`` liquid names or the week is skipped

The execution engine keys off the per-window ``direction`` + ``predicted_return``
it already consumes, HOLDs through weaker same-side signals and flips only past
the 2*lambda*cost band — so a weekly-stable LONG/SHORT (FLAT mid-quintile) falls
out as a clean weekly rebalance with cost-aware holds. No lookahead: every input
uses closes strictly before the rebalance bar.

Run with ``make stream-predictor`` (the launchd service boots ``stream.xs_signal``
when ``STREAM_STRATEGY=xs_rel14``). Pure logic lives in ``handle`` so tests
drive it directly with ``FakeBus``/``FakeKV``; warm-start pulls venue 1h klines.
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


def prediction_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def _up_up_at(closes: np.ndarray, idx: int, window_bars: int) -> bool:
    """UP-UP regime state at ``idx`` (Cooper et al. 2004 via Hsieh et al. 2025).

    True only when the current 4-week return AND the immediately preceding
    4-week return are both >= 0. Needs history back ``2*window_bars+1`` bars;
    without it the gate fails closed (FLAT) — conservative, never trades blind.
    """
    if idx < 2 * window_bars + 1:
        return False
    ret_now = closes[idx - 1] / closes[idx - 1 - window_bars] - 1.0
    ret_prev = closes[idx - 1 - window_bars] / closes[idx - 1 - 2 * window_bars] - 1.0
    return ret_now >= 0.0 and ret_prev >= 0.0


class XsSignal:
    """Cross-sectional weekly momentum book, fed by the 1h feature stream.

    ``_closes[symbol]`` is a rolling registry of ``(window_end_ms, close,
    volume)`` strictly in time order (Kafka may deliver symbols out of order, so
    every read sorts). At each weekly UTC rebalance boundary the signal is ranked
    across the universe and the selection is cached; every subsequent bar in the
    week re-emits the cached direction so the executor HOLDs until the next
    rebalance flips it. Best-effort warm-start seeds ``_closes`` from venue 1h
    klines so the 336h lookback is satisfied on the first live bar.
    """

    def __init__(
        self,
        kv: KVStore,
        *,
        prediction_prefix: str,
        universe: list[str],
        lookback_h: int = 336,
        quintile: float = 0.2,
        min_symbols: int = 8,
        volume_frac: float = 0.5,
        volume_median_bars: int = 28,
        rebalance_h: int = 168,
        vol_scale: bool = True,
        crash: float | None = None,
        regime: bool = True,
        market_symbol: str = "BTCUSDT",
        max_history: int = 1500,
    ) -> None:
        self._kv = kv
        self._prediction_prefix = prediction_prefix
        self._universe = [s.upper() for s in universe]
        self._lookback_h = lookback_h
        self._quintile = quintile
        self._min_symbols = min_symbols
        self._volume_frac = volume_frac
        self._volume_median_bars = volume_median_bars
        self._rebalance_h = rebalance_h
        self._vol_scale = vol_scale
        self._crash = crash
        self._regime = regime
        self._market = (
            market_symbol.upper()
            if market_symbol.upper() in self._universe
            else (self._universe[0] if self._universe else market_symbol.upper())
        )
        self._max_history = max_history
        self._closes: dict[str, list[tuple[int, float, float]]] = {}
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
        """Index of the latest close strictly before ``window_end`` (no lookahead).

        The weekly recompute may fire on the first symbol to arrive that hour,
        before the other symbols' current-hour bars land; using closes strictly
        before ``window_end`` makes every symbol's snapshot consistent (the probe
        ranks at bar ``t`` using the close known at ``t``, never a future bar).
        """
        return int(np.searchsorted(ends, window_end, side="left")) - 1

    # ── signal math (faithful to the probe) ──────────────────────────────────

    def _selection(self, window_end: int) -> dict[str, tuple[str, float]]:
        """Rank the universe at ``window_end``; return symbol -> (direction, sig)."""
        out: dict[str, tuple[str, float]] = {s: ("FLAT", 0.0) for s in self._universe}

        cand: list[tuple[str, float]] = []
        sigs: dict[str, float] = {}
        for s in self._universe:
            hist = self._closes.get(s)
            if not hist or len(hist) < self._lookback_h + 1:
                continue
            ends, closes, vols = self._series(hist)
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
            # Volume liquidity gate (own trailing median, strictly prior).
            lo = max(0, idx - self._volume_median_bars)
            med_window = vols[lo:idx]
            if len(med_window) < min(5, self._volume_median_bars):
                continue
            vmed = float(np.median(med_window))
            ok_vol = math.isfinite(vmed) and math.isfinite(vols[idx])
            ok_vol = ok_vol and vols[idx] >= self._volume_frac * vmed
            if not ok_vol:
                continue
            sigs[s] = sig
            cand.append((s, sig))

        if len(cand) < self._min_symbols:
            return out

        # Market-level gates (whole book FLAT if they fail).
        mhist = self._closes.get(self._market)
        if self._regime:
            if not mhist or len(mhist) < 2 * 672 + 2:
                return out
            _, mcloses, _ = self._series(mhist)
            midx = len(mcloses) - 1
            if not _up_up_at(mcloses, midx, 672):
                return out
        if self._crash is not None and mhist and len(mhist) >= 24:
            _, mcloses, _ = self._series(mhist)
            midx = len(mcloses) - 1
            if midx >= 24:
                r24 = mcloses[midx] / mcloses[midx - 24] - 1.0
                if math.isfinite(r24) and r24 < -self._crash:
                    return out

        cand.sort(key=lambda x: x[1])
        n_long = max(2, round(len(cand) * self._quintile))
        longs = cand[-n_long:]
        shorts = cand[:n_long]

        for side, group in (("LONG", longs), ("SHORT", shorts)):
            for s, _ in group:
                out[s] = (side, float(sigs[s]))
        return out

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
                    "signal": "xs_rel14",
                    "updated_at": self._kv_now(),
                },
            )

    def warm_start(self, settings) -> None:
        """Seed close history from the venue's 1h klines (best-effort).

        Pulls up to ``max_history`` hourly bars per universe symbol so the
        336h lookback is satisfied immediately — no 14-day dead wait before the
        book trades. Any symbol that fails to fetch simply accumulates live.
        """
        if not settings.stream_xs_warm_start:
            return
        from ingest.providers.bybit import BybitBarProvider

        provider = BybitBarProvider(base_url=settings.stream_bybit_base_url)
        for s in self._universe:
            try:
                bars = provider.fetch_klines_1h(s, limit=self._max_history)
            except Exception:  # noqa: BLE001 - best-effort; live stream fills the gap
                logger.warning("xs warm-start skip %s (venue fetch failed)", s)
                continue
            if not bars:
                continue
            for end_ms, close, volume in bars:
                self._record(s, int(end_ms), float(close), float(volume))
            logger.info("xs warm-start %s: %d hourly bars", s, len(bars))

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


def main() -> None:
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    universe = csv_list(settings.stream_xs_universe)
    strat = settings.stream_strategy

    if strat == "lcf":
        # Leverage-Crowding Factor (crypto-native funding carry). Shares the
        # same 1h feature topic for weekly rebalance timing but ranks by the
        # trailing perpetual funding rate pulled live from Binance. Validated in
        # scripts/research_fund.py: Sharpe 1.14 vs momentum 0.64, maxDD -41%.
        from stream.lcf_signal import LcfSignal

        signal = LcfSignal(
            kv,
            prediction_prefix=settings.stream_redis_prediction_prefix,
            universe=universe,
            fund_lookback_days=settings.stream_lcf_fund_lookback_days,
            quintile=settings.stream_lcf_quintile,
            min_symbols=settings.stream_lcf_min_symbols,
            cap_ann=settings.stream_lcf_cap_ann,
            direction=settings.stream_lcf_direction,
            rebalance_h=settings.stream_xs_rebalance_h,
        )
        group_id = "lcf-signal"
        tag = "lcf"
    elif strat == "scx":
        # SCX -- Skew-Convex, Regime-Gated Cross-Sectional Momentum. The
        # highest-value bettable factor (scripts/research_scx.py, WF Sharpe 1.13
        # vs momentum 0.76 / funding-LEVEL -1.79): BTC UP-UP regime gate (flat in
        # bears) + always-on winner book + CONDITIONAL short that drops the short
        # book when trailing short-book vol is stressed (skew-aware overlay, not
        # static vol-scaling). Recommended replacement for xs_rel14.
        from stream.scx_signal import ScxSignal

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
        group_id = "scx-signal"
        tag = "scx"
        if settings.stream_scx_warm_start:
            signal.warm_start(settings)
            logger.info("scx warm-start complete (%d symbols seeded)", len(signal._closes))
    elif strat == "asym":
        # ASYM -- OUR novel FAS_avg (multi-horizon Funding-Accrual Squeeze) + SMB
        # cross-sectional book. Validated in scripts/research_fas_combo.py
        # (keyless Binance 4.4y, 10bps, BTC-regime): Sharpe +1.82 FULL /
        # +1.93 POST-2024 (CI upper 2.49/2.65). FAS is OURS (funding accrual
        # residualized on price path, the derivatives analog of Bianchi et al.
        # 2026 order-flow orthogonalization); SMB is the one known factor that
        # adds to FAS (REV/MOM hurt it). Consumes the 1h topic for price path +
        # BTC regime gate, fetches funding keyless from Binance fapi/v1/
        # fundingRate, writes to its OWN prefix (prediction:crypto:asym:1h). The
        # executor reads this prefix when STREAM_STRATEGY=asym, so the FAS book
        # trades live on Bybit demo.
        from stream.asym_signal import AsymSignal

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
        )
        group_id = "asym-signal"
        tag = "asym"
        if settings.stream_asym_warm_start:
            signal.warm_start(settings)
            logger.info("asym warm-start complete (%d symbols seeded)", len(signal._closes))
    else:
        # Default: cross-sectional 14-day relative momentum (xs_rel14). The
        # legacy "river" online regressor is selectable the same way; it is the
        # stream.predictor module and not dispatched here.
        signal = XsSignal(
            kv,
            prediction_prefix=settings.stream_redis_prediction_prefix,
            universe=universe,
            lookback_h=settings.stream_xs_lookback_h,
            quintile=settings.stream_xs_quintile,
            min_symbols=settings.stream_xs_min_symbols,
            volume_frac=settings.stream_xs_volume_frac,
            volume_median_bars=settings.stream_xs_volume_median_bars,
            rebalance_h=settings.stream_xs_rebalance_h,
            vol_scale=settings.stream_xs_vol_scale,
            crash=settings.stream_xs_crash,
            regime=settings.stream_xs_regime,
            market_symbol=settings.stream_xs_market_symbol,
            max_history=settings.stream_xs_max_history,
        )
        group_id = "xs-rel14-signal"
        tag = "xs_rel14"
        if settings.stream_xs_warm_start:
            signal.warm_start(settings)
            logger.info("xs_rel14 warm-start complete (%d symbols seeded)", len(signal._closes))

    logger.info(
        "%s signal consuming %s → %s (universe=%d)",
        tag,
        settings.stream_kafka_topic_features,
        settings.stream_redis_prediction_prefix,
        len(universe),
    )
    try:
        signal.run_forever(
            bus,
            settings.stream_kafka_topic_features,
            group_id=group_id,
        )
    except KeyboardInterrupt:
        logger.info("%s signal stopped", tag)


if __name__ == "__main__":
    main()
