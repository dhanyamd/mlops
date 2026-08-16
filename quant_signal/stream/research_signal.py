"""Live research strategy signal — wires the REAL validated research_novel book
into the live pipeline. NO re-implementation: it calls research_novel.build_scores /
weights_at / btc_regime directly, so the live book == the backtest that produced
Sharpe 1.62 / 21x (ENS_MCD_SLOW: ens_mcd = mom_z - fund_z + dbeta_z, weekly
long-short weights, BTC slow-regime gate).

Unlike AsymSignal (threshold entries + hold-until-decay), this emits a TARGET
PORTFOLIO (symbol -> weight, long + / short -) every week. The executor
rebalances live Bybit Demo positions to match those targets.

Consumed by the signal daemon in place of AsymSignal; reads the same hourly
feature windows and emits `portfolio:targets:<prefix>` to KV.
"""

from __future__ import annotations

import os
import math
import pandas as pd
from config.logging import configure_logging, get_logger  # noqa: F401
from stream.kv import KVStore

# The REAL research backtester — we call its functions, we do not re-derive them.
import scripts.research_novel as rn

logger = get_logger("research_signal")

_WEEK = "W-MON"


def target_key(prefix: str) -> str:
    return f"portfolio:targets:{prefix}"


class ResearchSignal:
    """Weekly cross-sectional research book (research_novel.ens_mcd) -> target weights.

    The ``spec`` is literally a research_novel spec dict (e.g. the ENS_MCD_SLOW
    entry from its main()), so the live strategy is identical to the validated one.
    """

    def __init__(
        self,
        kv: KVStore,
        prefix: str = "research",
        universe: list[str] | None = None,
        spec: dict | None = None,
        notional_usd: float = 8000.0,
    ) -> None:
        self._kv = kv
        self._prefix = prefix
        self._universe = list(universe or [])
        # REAL validated spec (research_novel.main: ENS_MCD_SLOW). Not magic numbers.
        self._spec = spec or dict(score="ens_mcd", regime=True, regime_mode="slow")
        self._notional = notional_usd
        self._close: dict[str, list[tuple[int, float]]] = {s: [] for s in self._universe}
        self._fund: dict[str, list[tuple[int, float]]] = {s: [] for s in self._universe}
        self._last_week: int | None = None

    # ── live ingestion ────────────────────────────────────────────────────────

    def seed_funding(self, symbol: str, events: list[tuple[int, float]]) -> None:
        """Bootstrap funding history (warm cache / REST) before live trading."""
        self._fund.setdefault(symbol, []).extend(events)

    def handle(self, msg: dict) -> None:
        symbol = str(msg.get("symbol") or "").upper()
        if symbol not in self._close:
            return
        close = msg.get("close")
        if not isinstance(close, (int, float)) or not math.isfinite(close) or close == 0:
            return
        w_end = msg.get("window_end_ms")
        if not isinstance(w_end, (int, float)):
            return
        w_end = int(w_end)
        self._close[symbol].append((w_end, float(close)))
        week = w_end // (7 * 24 * 3_600_000)
        if self._last_week is None or week != self._last_week:
            self._last_week = week
            self._emit(w_end)

    # ── weekly target computation (REAL research_novel code) ──────────────────

    def _close_df(self) -> pd.DataFrame:
        series = {}
        for s in self._universe:
            evs = self._close.get(s) or []
            if evs:
                series[s] = pd.Series({pd.Timestamp(int(ts), unit="ms"): float(c) for ts, c in evs})
            else:
                # Empty close history -> explicit empty DatetimeIndex so the
                # union with other symbols stays a DatetimeIndex (a RangeIndex
                # from an empty default series would poison it and break
                # resample() inside research_novel.weekly_frame).
                series[s] = pd.Series(index=pd.DatetimeIndex([]), dtype=float)
        return pd.DataFrame(series).sort_index()

    def _fund_df(self) -> pd.DataFrame:
        series = {}
        for s in self._universe:
            evs = self._fund.get(s) or []
            if evs:
                series[s] = pd.Series({pd.Timestamp(int(ts), unit="ms"): float(r) for ts, r in evs})
            else:
                # Empty funding -> explicit empty DatetimeIndex so the union
                # with other symbols stays a DatetimeIndex (RangeIndex would
                # poison it and break resample()).
                series[s] = pd.Series(index=pd.DatetimeIndex([]), dtype=float)
        return pd.DataFrame(series).sort_index()

    def compute_targets(self) -> dict[str, float]:
        """Return {symbol: weight} long/short (sum-zero) for the latest week, or {} if flat.

        Composes the REAL research_novel `ens_mcd` score (research_novel line 519:
        ``ens_mcd = (mom_z - fund_z) + dbeta_z``) from research_novel's own
        primitives (weekly_frame, zs, CONFIG, btc_regime). We do NOT call the
        monolithic build_scores because it also computes an unused `rot` score that
        crashes on gap-filled hourly data (bool->float64 resample coercion). The
        alpha is identical to the validated backtester.
        """
        df = self._close_df()
        # BTCUSDT is the benchmark (regime gate + downside-beta). Symbols in the
        # universe can have mismatched history (e.g. ZECUSDT carries an extra year
        # of data), which would stretch the resample range and poison weekly_frame.
        # Restrict to rows where BTCUSDT actually has a price — equivalent to the
        # backtester's intersection of data sources.
        if "BTCUSDT" not in df or df["BTCUSDT"].isna().all():
            return {}
        df = df[df["BTCUSDT"].notna()]
        # df is HOURLY, so df.shape[0] counts hours, not weeks. The weekly
        # features (formation_days momentum + dbeta_win downside-beta regression)
        # need enough WEEKS after resample — not raw rows. Count the benchmark's
        # (BTCUSDT) weeks: requiring all 31 symbols present per week is too strict
        # (minor per-symbol gaps would drop every week), and the score itself
        # already enforces the >=12-symbol coverage floor downstream.
        n_weeks = df[["BTCUSDT"]].resample(_WEEK).last().dropna(how="any").shape[0]
        min_weeks = (
            max(rn.CONFIG["formation_days"], rn.CONFIG["dbeta_win"]) + rn.CONFIG["formation_days"]
        )
        if n_weeks < min_weeks:
            return {}
        fund = self._fund_df()

        formation = rn.CONFIG["formation_days"]  # 14 (research_novel CONFIG)
        fwd, mom, _vol = rn.weekly_frame(df, formation)
        mom_z = rn.zs(mom)
        fund_w = fund.resample(_WEEK).mean().reindex(mom_z.index)
        fund_z = rn.zs(fund_w)

        # Downside-beta vs BTC (research_novel build_scores, exact): 26w rolling
        # regression of each coin's weekly return on BTC's, using ONLY BTC-down weeks.
        ret_w = df.resample(_WEEK).last().pct_change().reindex(mom_z.index)
        btc_w = ret_w["BTCUSDT"]
        win = 26
        dbeta = pd.DataFrame(index=mom_z.index, columns=self._universe, dtype=float)
        for d in mom_z.index:
            i = ret_w.index.get_loc(d)
            if i < win + 1:
                continue
            rm = btc_w.iloc[i - win : i]
            for c in self._universe:
                if c == "BTCUSDT":
                    continue
                ri = ret_w[c].iloc[i - win : i]
                m = rm < 0
                if m.sum() < 8:
                    continue
                x = rm[m].values
                y = ri[m].values
                xm = x - x.mean()
                vx = float((xm * xm).sum())
                if vx < 1e-12:
                    continue
                dbeta.loc[d, c] = -float(((y - y.mean()) * xm).sum()) / vx
        dbeta_z = rn.zs(dbeta)

        # research_novel line 519: ens_mcd = (mom_z - fund_z) + dbeta_z  (DataFrame: dates x symbols)
        score = mom_z - fund_z + dbeta_z
        if score.empty or len(score.iloc[-1].dropna()) < 12:
            return {}

        if self._spec.get("regime", True):
            reg = rn.btc_regime(df)  # float64 here; truthiness (0/1) is correct
            mode = self._spec.get("regime_mode", "up")
            if mode not in reg.columns:
                mode = "up"
            if not bool(reg[mode].iloc[-1]):
                return {}  # risk-off -> flat (research_novel weights_at behaviour)

        ivol = fwd.rolling(12).std()  # for inv_vol sizing (unused by ENS_MCD_SLOW)
        date = score.index[-1]
        w = rn.weights_at(date, score, ivol, True, None, self._spec)
        return w.to_dict() if w is not None else {}

    def _emit(self, window_end: int) -> None:
        targets = self.compute_targets()
        payload = {
            "signal": f"research_{self._spec.get('score')}",
            "window_end_ms": window_end,
            "spec": self._spec,
            "targets": targets,
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        }
        self._kv.set_json(target_key(self._prefix), payload)
        logger.info(
            "research: emitted %d targets (longs=%d shorts=%d)",
            len(targets),
            sum(1 for v in targets.values() if v > 0),
            sum(1 for v in targets.values() if v < 0),
        )


if __name__ == "__main__":
    configure_logging()
    from config.settings import csv_list, get_settings

    settings = get_settings()
    kv = KVStore()
    universe = csv_list(settings.stream_xs_universe)
    sig = ResearchSignal(kv, universe=universe)
    print(f"ResearchSignal (real research_novel) ready for {len(universe)} symbols")
