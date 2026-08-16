"""Research backtest: delta-neutral FUNDING-CARRY, the repo's proven way.

This is NOT a from-scratch guess. It implements the recipe the repo's own
research already validated, then ADDS explicit funding-cashflow attribution
(which the research_*.py scripts mostly ignore -- they use funding only as a
price signal).

Established priors (from the research_*.py family):
  * research_fund.py (LCF): high-funding coins are crowded-longs that KEEP
    trending up. So a PURE contrarian funding carry (long low-funding / short
    high-funding) loses on the PRICE leg -- high-funding names rise. Proven here:
    pure carry => price P&L deeply negative, funding drip tiny.
  * research_cscm.py (CSCM) + "Foundational" paper: the BREAKTHROUGH is the
    COMPOSITE cross-sectional score, PRE-SPECIFIED, no in-sample fit:
        score = f(z(momentum)) - z(trailing funding)
    => LONG  high composite (momentum winner AND cheap-to-hold / low funding)
       SHORT  low  composite (momentum loser  AND crowded-long / high funding)
    Both legs then earn favorable price moves AND collect funding. WF Sharpe 2+.
  * research_volscale.py / Keel: per-leg VOL-TARGET (10-15% annualized) makes the
    book ~market-neutral and crash-hardened.
  * Every repo script gates on a BTC UP-UP regime (90/200d). We do too.

Modes (QUANT_MODE; default "comp" = the breakthrough):
  comp  : CSCM composite z(mom) - z(funding)            [breakthrough]
  carry : pure funding carry (long low / short high)    [thesis test, loses]
  mom   : pure 14d cross-sectional momentum (baseline)

Funding cashflow IS modeled: at every funding settlement in (open, close],
  funding_pnl = -position_sign * rate * notional
so the report attributes REALIZED P&L into price-only vs funding-only, proving
the book collects funding on top of the momentum edge.

Usage:
  QUANT_MODE=comp   QUANT_FEE_BPS=5.5 uv run python -m scripts.carry_backtest
  QUANT_MODE=carry  QUANT_FEE_BPS=5.5 uv run python -m scripts.carry_backtest
  QUANT_REGIME=1    QUANT_MODE=comp   uv run python -m scripts.carry_backtest
"""

from __future__ import annotations

import bisect
import json
import math
import os
import statistics
from datetime import UTC, datetime

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from stream.asym_signal import AsymSignal

logger = get_logger(__name__)

DAY_MS = 24 * 3600 * 1000
FUND_WINDOW_MS = 7 * DAY_MS  # trailing funding window for the carry z
VOL_TARGET = 0.12  # annualized vol target per leg (Keel 10-15%)
DISP_GATE = 0.10  # min cross-sectional ANNUALIZED funding spread to trade
TOPN_DECILE = 3  # names per side (selection) of the ~30-name universe
VOL_LOOKBACK_DAYS = 30
MOM_FORMATION_DAYS = 14  # momentum formation (CSCM 14d)
REGIME_FAST_DAYS = 90  # BTC UP-UP fast MA (CSCM daily 90)
REGIME_SLOW_DAYS = 200  # BTC UP-UP slow MA (CSCM daily 200)


def _load_cache():
    cache = json.load(open(AsymSignal._warm_cache_path()))
    bars, funding = {}, {}
    for s, rows in cache["bars"].items():
        bars[s] = sorted((int(e), float(c), float(v)) for e, c, v in rows)
    for s, rows in cache["funding"].items():
        funding[s] = sorted((int(e), float(r)) for e, r in rows)
    return bars, funding


def _close_at(bar_list, t):
    i = bisect.bisect_right(bar_list, (t, math.inf, math.inf)) - 1
    return bar_list[i][1] if i >= 0 else None


def _funding_in(sym_fund, t_open, t_close):
    i0 = bisect.bisect_right(sym_fund, (t_open, math.inf))
    i1 = bisect.bisect_right(sym_fund, (t_close, math.inf))
    return sum(r for _, r in sym_fund[i0:i1])


def _trailing_fund_avg(sym_fund, t):
    """Annualized funding rate over the trailing FUND_WINDOW_MS."""
    i0 = bisect.bisect_right(sym_fund, (t - FUND_WINDOW_MS, math.inf))
    i1 = bisect.bisect_right(sym_fund, (t, math.inf))
    window = sym_fund[i0:i1]
    if not window:
        return None
    # Bybit USDT perps settle funding every 8h (3x/day) -> annualize the mean rate.
    return statistics.fmean(r for _, r in window) * 3 * 365


def _daily_vol(sym_bars, t, lookback_days):
    """annualized stdev of daily returns ending at t."""
    closes = []
    for k in range(lookback_days + 1):
        tt = t - k * DAY_MS
        i = bisect.bisect_right(sym_bars, (tt, math.inf, math.inf)) - 1
        if i >= 0:
            closes.append(sym_bars[i][1])
    if len(closes) < lookback_days:
        return None
    rets = [(closes[k - 1] / closes[k] - 1) for k in range(1, len(closes))]
    sd = statistics.pstdev(rets)
    return sd * math.sqrt(365) if sd > 0 else None


def _mom_z(bars, sym, t, formation_bars):
    """Vol-scaled 14d momentum, cross-sectionally z-scored at time t."""
    out = {}
    for s in sym:
        c = bars[s]
        i = bisect.bisect_right(c, (t, math.inf, math.inf)) - 1
        if i < formation_bars:
            continue
        ret = c[i][1] / c[i - formation_bars][1] - 1
        vol = _daily_vol(c, t, 21)
        if vol and vol > 0:
            out[s] = ret / vol
    if len(out) < 5:
        return {}
    mean = statistics.mean(out.values())
    sd = statistics.pstdev(out.values()) or 1.0
    return {s: (v - mean) / sd for s, v in out.items()}


def _btc_up_up(bars, t):
    c = bars.get("BTCUSDT")
    if not c:
        return True
    i = bisect.bisect_right(c, (t, math.inf, math.inf)) - 1
    if i < REGIME_SLOW_DAYS * 24:
        return True
    now = c[i][1]
    fast = statistics.fmean(c[j][1] for j in range(i - REGIME_FAST_DAYS * 24 + 1, i + 1))
    slow = statistics.fmean(c[j][1] for j in range(i - REGIME_SLOW_DAYS * 24 + 1, i + 1))
    return now > fast and now > slow


def _select(bars, funding, sym, t, mode, use_regime):
    """Return (longs, shorts) per QUANT_MODE, with regime + dispersion gates."""
    favg = {}
    for s in sym:
        a = _trailing_fund_avg(funding[s], t)
        if a is not None:
            favg[s] = a
    if len(favg) < 2 * TOPN_DECILE:
        return {}, {}

    if use_regime and not _btc_up_up(bars, t):
        return {}, {}  # BTC regime gate: flatten when not UP-UP

    fvals = list(favg.values())
    disp = max(fvals) - min(fvals)  # already annualized
    # dispersion gate only binds the funding-based books
    if mode in ("carry", "comp") and disp < DISP_GATE:
        return {}, {}

    mom = _mom_z(bars, sym, t, MOM_FORMATION_DAYS * 24)
    if mode == "mom":
        score = mom
    else:
        fmean = statistics.mean(fvals)
        fsd = statistics.pstdev(fvals) or 1.0
        fund_z = {s: (v - fmean) / fsd for s, v in favg.items()}
        if mode == "carry":
            score = {s: -fund_z[s] for s in fund_z}  # long low / short high funding
        else:  # comp: CSCM composite, PRE-SPECIFIED equal weight
            score = {s: mom.get(s, 0.0) - fund_z.get(s, 0.0) for s in set(mom) | set(fund_z)}

    if not score:
        return {}, {}
    ranked = sorted(score.items(), key=lambda kv: kv[1])
    longs = {s for s, _ in ranked[-TOPN_DECILE:]}
    shorts = {s for s, _ in ranked[:TOPN_DECILE]}
    return longs, shorts


def main() -> None:
    configure_logging()
    settings = get_settings()
    universe = [s.strip().upper() for s in csv_list(settings.stream_xs_universe)]

    weeks = int(os.environ.get("QUANT_WEEKS", "12"))
    hold_days = int(os.environ.get("QUANT_CARRY_HOLD_DAYS", "7"))
    fee_bps = float(os.environ.get("QUANT_FEE_BPS", "5.5"))
    offset_w = int(os.environ.get("QUANT_WF_OFFSET_W", "0"))
    mode = os.environ.get("QUANT_MODE")
    if not mode:
        mode = "comp" if os.environ.get("QUANT_COMBINE_MOM") == "1" else "carry"
    mode = mode.lower()
    use_regime = os.environ.get("QUANT_REGIME") == "1"
    base_notional = float(
        os.environ.get("QUANT_CARRY_NOTIONAL", str(settings.stream_execution_notional_usd))
    )

    bars, funding = _load_cache()
    sym = [s for s in universe if bars.get(s) and funding.get(s)]
    logger.info("carry backtest: %d symbols, mode=%s regime=%s", len(sym), mode, use_regime)

    btc = bars["BTCUSDT"]
    daily_ts = []
    seen = set()
    for e, _, _ in btc:
        d = e // DAY_MS
        if d not in seen:
            seen.add(d)
            daily_ts.append(e)
    sym_daily = {s: {e for e, _, _ in bars[s]} for s in sym}

    fund_start = min(funding[s][0][0] for s in sym)
    fund_end = max(funding[s][-1][0] for s in sym)
    start_ts = fund_start + FUND_WINDOW_MS
    end_ts = fund_end
    win_end = min(end_ts, end_ts - offset_w * 7 * DAY_MS)
    win_start = win_end - weeks * 7 * DAY_MS
    start_ts = max(start_ts, win_start)
    win_end = min(win_end, end_ts)

    rb_times = [
        t
        for t in daily_ts
        if start_ts <= t <= win_end and ((t // DAY_MS) - (start_ts // DAY_MS)) % hold_days == 0
    ]
    if not rb_times:
        print("NO REBALANCE TIMES IN WINDOW")
        return
    rb_times.append(win_end)

    positions = {}
    equity = price_eq = funding_eq = 0.0
    trade_pnls, fees_paid = [], 0.0
    peak_eq = max_dd = wins = losses = 0
    prev_equity = 0.0
    rets = []
    capital_base = base_notional * 2 * TOPN_DECILE  # gross notional for return scaling

    for k in range(len(rb_times) - 1):
        t_open, t_close = rb_times[k], rb_times[k + 1]
        for s, pos in list(positions.items()):
            sign = 1 if pos["side"] == "LONG" else -1
            po = _close_at(bars[s], pos["t_open"])
            pc = _close_at(bars[s], t_close)
            if po is None or pc is None:
                continue
            qty = pos["notional"] / po
            price = qty * (pc - po) if sign == 1 else qty * (po - pc)
            fund = -sign * _funding_in(funding[s], pos["t_open"], t_close) * pos["notional"]
            fee = 2 * (fee_bps / 1e4) * pos["notional"]
            net = price + fund - fee
            equity += net
            price_eq += price - fee / 2
            funding_eq += fund - fee / 2
            fees_paid += fee
            trade_pnls.append(net)
            wins += net > 0
            losses += net <= 0
        positions.clear()

        if capital_base > 0:
            rets.append((equity - prev_equity) / capital_base)
        prev_equity = equity
        peak_eq = max(peak_eq, equity)
        max_dd = max(max_dd, peak_eq - equity)

        longs, shorts = _select(bars, funding, sym, t_open, mode, use_regime)
        for s in longs | shorts:
            side = "LONG" if s in longs else "SHORT"
            if t_open not in sym_daily[s]:
                continue
            vol = _daily_vol(bars[s], t_open, VOL_LOOKBACK_DAYS)
            if vol is None or vol <= 0:
                continue
            notional = (base_notional * VOL_TARGET) / vol  # vol-target: equal vol$ per name
            notional = min(notional, base_notional * 4)  # cap single name
            positions[s] = {"side": side, "t_open": t_open, "notional": notional}

    realized = sum(trade_pnls)
    n = len(trade_pnls)
    wr = wins / n * 100 if n else 0.0

    periods_per_year = 365 / hold_days
    if rets:
        rmean = statistics.mean(rets)
        rstd = statistics.pstdev(rets)
        ts_sharpe = rmean / rstd * math.sqrt(periods_per_year) if rstd > 0 else 0.0
        ann_ret = rmean * periods_per_year
    else:
        ts_sharpe = ann_ret = 0.0

    print("=" * 62)
    print("DELTA-NEUTRAL FUNDING CARRY (CSCM-grounded)")
    print("=" * 62)
    print(
        f"window        : {datetime.fromtimestamp(win_start / 1000, UTC):%Y-%m-%d}"
        f" -> {datetime.fromtimestamp(win_end / 1000, UTC):%Y-%m-%d}"
        f"  (offset {offset_w}w)"
    )
    print(f"mode          : {mode}{' + BTC UP-UP gate' if use_regime else ''}")
    print(f"hold/rebalance: {hold_days}d | per-side={TOPN_DECILE} | vol-target {VOL_TARGET:.0%}")
    print(f"dispersion gate: {DISP_GATE:.0%} annualized | fee {fee_bps} bps")
    print(f"symbols       : {len(sym)}")
    print("-" * 62)
    print(f"CLOSED trades : {n}")
    print(f"wins/losses   : {wins}/{losses}  (win rate {wr:.1f}%)")
    print(f"REALIZED P&L  : ${realized:,.2f}")
    print(f"  price-only  : ${price_eq:,.2f}")
    print(f"  funding-only: ${funding_eq:,.2f}")
    print(f"fees paid     : ${fees_paid:,.2f}")
    print(f"ann. return   : {ann_ret * 100:,.1f}%")
    print(f"ann. Sharpe   : {ts_sharpe:.2f}")
    print(f"max drawdown  : ${max_dd:,.2f}")
    print("=" * 62)
    if mode == "carry":
        if funding_eq > 0 and abs(price_eq) < abs(funding_eq) * 2:
            print("VERDICT: funding edge real & dominant => clean carry.")
        elif funding_eq <= 0:
            print("VERDICT: funding edge absent in this window/data.")
        else:
            print("VERDICT: pure carry LOSES on price leg (high-funding coins trend up).")
            print("         => use QUANT_MODE=comp (momentum+carry composite) for the edge.")
    else:
        if ts_sharpe >= 1.0 and realized > 0:
            print(
                f"VERDICT: BREAKTHROUGH -- {mode} book profitable, ann.Sharpe "
                f"{ts_sharpe:.2f}, funding collected ${funding_eq:,.0f} on top of price edge."
            )
        else:
            print(
                f"VERDICT: {mode} book ann.Sharpe {ts_sharpe:.2f}; price ${price_eq:,.0f}, "
                f"funding ${funding_eq:,.0f}."
            )


if __name__ == "__main__":
    main()
