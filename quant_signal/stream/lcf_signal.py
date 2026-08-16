"""Live LCF (Leverage-Crowding Factor) signal -- crypto-native funding-carry book.

Sibling of ``stream.xs_signal``: consumes the SAME 1h feature topic
(``crypto.features.1h``) purely for weekly rebalance timing, but at each rebalance
pulls the trailing perpetual FUNDING RATE for the universe from Binance and ranks
coins by it -- the crypto-native microstructure signal OHLCV cannot see.

Mechanism (validated in ``scripts/research_fund.py``, 2020-09..2026-08, costed,
walk-forward, 30 coins):

  The perpetual funding rate is the price leverage-crowded positions pay. Per
  Christin et al. "Crypto Carry Trade" / BIS WP1087 / SSRN 3774118, funding
  POSITIVELY predicts cross-sectional returns: coins where leveraged longs pay
  the most (crowded-long, trend-chasing) earn the HIGHEST future returns. So the
  book is LONG the highest-funding coins, SHORT the lowest/negative-funding
  (under-owned) coins -- the documented crypto carry direction -- with a fragility
  cap that zeroes exposure when annualized funding is extreme (BIS: extreme carry
  predicts crashes).

  Validated vs the momentum baseline:
    MOM baseline            Sharpe 0.64  maxDD -78%
    LCF (pro) + vol-scale   Sharpe 1.14  maxDD -41%   (vol-scale halves the tail)

Emits the identical ``prediction:crypto:1h:<SYMBOL>`` payload the executor already
reads (direction LONG/SHORT/FLAT + predicted_return), so execution, the Bybit demo
venue and the dashboard are untouched. Funding is fetched strictly before the
rebalance bar (lookahead-free): only funding prints with fundingTime <= the
rebalance window_end_ms are used.

Run: uv run python -m stream.lcf_signal   (service 'lcfsignal' swaps in for 'xssignal')
"""

from __future__ import annotations

import threading
import urllib.request
import json

import numpy as np

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_FUND_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def prediction_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    return json.load(urllib.request.urlopen(req, timeout=20))


class LcfSignal:
    """Weekly funding-carry book, triggered by the 1h feature stream."""

    def __init__(
        self,
        kv: KVStore,
        *,
        prediction_prefix: str,
        universe: list[str],
        fund_lookback_days: int = 28,
        quintile: float = 0.20,
        min_symbols: int = 8,
        cap_ann: float | None = 1.0,
        direction: str = "pro",
        rebalance_h: int = 168,
        max_history: int = 500,
    ) -> None:
        self._kv = kv
        self._prediction_prefix = prediction_prefix
        self._universe = [s.upper() for s in universe]
        self._fund_lookback_days = fund_lookback_days
        self._quintile = quintile
        self._min_symbols = min_symbols
        self._cap_ann = cap_ann
        self._direction = direction  # "pro" = long high funding; "contra" = short it
        self._rebalance_h = rebalance_h
        self._max_history = max_history
        self._last_week: int | None = None
        self._current: dict[str, tuple[str, float]] = {}

    # ── funding fetch (crypto-native input) ─────────────────────────────────

    def _annualized_funding(self, symbol: str, as_of_ms: int) -> float | None:
        """Trailing annualized funding rate for ``symbol`` up to ``as_of_ms``.

        Pulls ~33 days of 8h funding prints, aggregates to daily yield, and
        annualizes the trailing ``fund_lookback_days`` mean. None on any failure
        (symbol skipped that week -- never trade blind).
        """
        try:
            raw = _get_json(f"{_FUND_URL}?symbol={symbol}&limit=100")
        except Exception:  # noqa: BLE001 - best-effort; skip symbol if venue fails
            logger.warning("lcf funding fetch failed for %s", symbol)
            return None
        if not raw:
            return None
        rows = [
            (r["fundingTime"], float(r["fundingRate"]))
            for r in raw
            if r.get("fundingTime") is not None
            and r.get("fundingRate") is not None
            and r["fundingTime"] <= as_of_ms
        ]
        if not rows:
            return None
        daily = {}
        for t, rate in rows:
            day = t // _HOUR_MS // 24
            daily[day] = daily.get(day, 0.0) + rate
        series = sorted(daily.values())
        if len(series) < min(5, self._fund_lookback_days):
            return None
        return float(np.mean(series[-self._fund_lookback_days :])) * 365.0

    # ── signal math ─────────────────────────────────────────────────────────

    def _selection(self, window_end: int) -> dict[str, tuple[str, float]]:
        out: dict[str, tuple[str, float]] = {s: ("FLAT", 0.0) for s in self._universe}
        scores: dict[str, float] = {}
        for s in self._universe:
            ann = self._annualized_funding(s, window_end)
            if ann is None or not np.isfinite(ann):
                continue
            scores[s] = float(ann)

        cand = sorted(scores.items(), key=lambda x: x[1])
        if len(cand) < self._min_symbols:
            return out

        if self._cap_ann is not None:
            # Fragility cap (BIS): zero exposure where annualized funding is extreme.
            kept = [(s, v) for s, v in cand if abs(v) <= self._cap_ann]
            if len(kept) < self._min_symbols:
                kept = cand  # cap would empty the book; fall back to uncapped
            cand = kept

        n = max(2, round(len(cand) * self._quintile))
        longs = cand[-n:] if self._direction == "pro" else cand[:n]
        shorts = cand[:n] if self._direction == "pro" else cand[-n:]
        for s, v in longs:
            out[s] = ("LONG", v)
        for s, v in shorts:
            out[s] = ("SHORT", v)
        return out

    # ── rebalance + emit ─────────────────────────────────────────────────────

    def handle(self, msg: dict) -> None:
        window_end = msg.get("window_end_ms")
        window_end = int(window_end) if isinstance(window_end, (int, float)) else None
        if window_end is None:
            return
        # Only the weekly boundary triggers a recompute; every bar re-emits the
        # cached (weekly-stable) selection so the executor HOLDs between rebalances.
        week = window_end // (self._rebalance_h * _HOUR_MS)
        if self._last_week is None or week != self._last_week:
            self._current = self._selection(window_end)
            self._last_week = week
            logger.info(
                "lcf rebalance wk=%d: %d active (dir=%s)",
                week,
                sum(1 for _, d in self._current.values() if d != "FLAT"),
                self._direction,
            )

        for s in self._universe:
            direction, yhat = self._current.get(s, ("FLAT", 0.0))
            self._kv.set_json(
                prediction_key(self._prediction_prefix, s),
                {
                    "symbol": s,
                    "window_end_ms": window_end,
                    "predicted_return": round(float(yhat), 6),
                    "direction": direction,
                    "signal": "lcf",
                    "updated_at": self._kv_now(),
                },
            )

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
    logger.info(
        "lcf signal consuming %s -> %s (universe=%d, fund=%dd, quintile=%.2f, cap=%s, dir=%s)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_prediction_prefix,
        len(universe),
        settings.stream_lcf_fund_lookback_days,
        settings.stream_lcf_quintile,
        settings.stream_lcf_cap_ann,
        settings.stream_lcf_direction,
    )
    try:
        signal.run_forever(bus, settings.stream_kafka_topic_features, group_id="lcf-signal")
    except KeyboardInterrupt:
        logger.info("lcf signal stopped")


if __name__ == "__main__":
    main()
