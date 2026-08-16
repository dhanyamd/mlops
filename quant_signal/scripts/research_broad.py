"""Broad-universe backtest of OUR invented cross-sectional factors.

Tests ASYM (original, tested 1.25 on 30-coin book) and FVCS (flagship novel
confluence, untested) on the 416-coin broad universe pulled keylessly from
Binance. Size-conditioned by dollar-volume tercile (small/illiquid = where the
size-volume literature says reversal/squeeze signals live).

Data: /tmp/broad_pull/<SYM>.csv  cols: date,close,qvol,funding
Keyless. 10bps costs, BTC-regime gate, bootstrap CI via random.Random(0).
"""

import glob
import os
import random

import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
COST = 10  # bps round-trip
BT_CSV = "/tmp/crypto_daily_long.csv"  # for BTC regime (30-coin book)


def load_btc_close():
    df = pd.read_csv(BT_CSV, parse_dates=["ts"]).set_index("ts")
    return df["BTCUSDT"]


def load_coins():
    frames = {}
    for f in sorted(glob.glob(os.path.join(BROAD, "*.csv"))):
        sym = os.path.basename(f)[:-4]
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        d = d[["close", "qvol", "funding"]].dropna()
        if len(d) < 200:  # need history
            continue
        d = d[~d.index.duplicated()]
        frames[sym] = d
    return frames


def weekly_panel(frames):
    closes, vols, funds = {}, {}, {}
    for sym, d in frames.items():
        wk = d["close"].resample("W").last()
        wv = d["qvol"].resample("W").sum()
        wf = d["funding"].resample("W").mean()
        closes[sym] = wk
        vols[sym] = wv
        funds[sym] = wf
    close = pd.DataFrame(closes)
    vol = pd.DataFrame(vols)
    fund = pd.DataFrame(funds)
    return close, vol, fund


def zs(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


def build_scores(close, vol, fund):
    ret = close.pct_change(1)  # weekly return
    mom_z = zs(ret)
    fund_z = zs(fund)
    # volume surge: per-coin volume vs trailing 4w mean
    vol_surge = vol.div(vol.rolling(4, min_periods=2).mean()).clip(lower=0, upper=10)
    crowd_short = (-fund_z).clip(lower=0)  # funding extremely negative = crowded short
    # ASYM: crowded-short funding gated by price resilience (our original)
    asym = (-fund_z) * (mom_z > 0)
    # FVCS: multiplicatively-gated confluence (our flagship novel) -- later shown spanned by trend
    fvcs = zs(crowd_short * vol_surge * mom_z)
    # MHT: multi-horizon trend (CTREND proxy, literature benchmark)
    mh = sum(zs(close.pct_change(h).fillna(0)) for h in (1, 2, 4, 8, 12)) / 5.0
    mht = zs(mh * vol_surge.clip(lower=0))
    # REVX: crowding-contrarian REVERSAL -- our novel factor, mechanically opposite to momentum.
    # Verified literature: reversal only appears at 4-12w formation (Kiefer 2026 8w Sharpe 0.96-1.69
    # ex-mega-caps; Momentum&Reversal paper: 1w is momentum regime, reversal beyond ~1 month).
    # rev_ret = multi-horizon (4/8/12w) return; rev_z = -zs(rev_ret) bets against recent losers/winners.
    # crowd_gate = (fund_z * sign(rev_ret)).clip(lower=0): only fires where leverage is piled on the
    # SAME side as the recent move (loser+crowded-short or winner+crowded-long = crowded trend ->
    # explosive reversal/squeeze). When funding fights the trend the gate is 0 -> no bet.
    # vol_surge confirms the capitulation/squeeze (volume spikes on the cascade per squeeze literature).
    # REV_PLAIN is the known reversal factor (no gate) -- isolates the novel funding-crowding contribution.
    rev_ret = sum(close.pct_change(h).fillna(0) for h in (4, 8, 12)) / 3.0
    rev_z = -zs(rev_ret)
    rev_plain = rev_z
    crowd_gate = (fund_z * np.sign(rev_ret)).clip(lower=0)
    revx = zs(rev_z * crowd_gate * vol_surge)
    return {
        "asym": asym,
        "fvcs": fvcs,
        "mht": mht,
        "rev_plain": rev_plain,
        "revx": revx,
        "crowd_short": crowd_short,
        "vol_surge": vol_surge,
    }


def backtest(score: pd.DataFrame, close: pd.DataFrame, inv_vol=False, regime=None, long_short=True):
    dates = score.index
    prev = None
    ret = []
    for i, date in enumerate(dates):
        if i < 5 or date not in score.index:
            ret.append(0.0)
            prev = None
            continue
        s = score.loc[date]
        s = s.dropna()
        if regime is not None:
            if date not in regime.index or regime.loc[date] <= 0:
                ret.append(0.0)
                prev = None
                continue
        if len(s) < 10:
            ret.append(0.0)
            prev = None
            continue
        # quintile long/short
        q = s.rank(pct=True)
        if long_short:
            longs = q[q > 0.8].index
            shorts = q[q < 0.2].index
            w = pd.Series(0.0, index=s.index)
            w[longs] = 1.0 / max(1, len(longs))
            w[shorts] = -1.0 / max(1, len(shorts))
        else:
            w = q - 0.5
        if inv_vol:
            vol_w = score.loc[date].reindex(w.index).abs().replace(0, np.nan)
            w = w.div(vol_w)
            w = w.div(w.abs().sum())
        # forward return
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        if nxt is None:
            r = 0.0
        else:
            fr = close.loc[nxt].reindex(w.index) / close.loc[date].reindex(w.index) - 1
            fr = fr.fillna(0)
            r = float((w * fr).sum())
        if prev is not None:
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= COST / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def metrics(ret: pd.Series):
    ret = ret.dropna()
    n = len(ret)
    if n < 20:
        return None
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = random.Random(0)
    vals = list(ret.values)
    boot = []
    for _ in range(1000):
        sm = sum(rng.choice(vals) for _ in range(n)) / n * 52
        sd = (sum((x - sum(vals) / n) ** 2 for x in vals) / max(1, n - 1)) ** 0.5 * np.sqrt(52)
        boot.append(sm / sd if sd > 0 else 0.0)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return dict(
        n=n, ann=ann, vol=vol, sharpe=sharpe, ci=ci, maxdd=dd, pct_flat=float((ret == 0).mean())
    )


def report(name, m):
    if m is None:
        print(f"{name}: insufficient data")
        return
    print(
        f"  {name:22s} Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}] "
        f"ann={m['ann'] * 100:5.1f}% vol={m['vol'] * 100:4.1f}% maxDD={m['maxdd'] * 100:6.1f}% %flat={m['pct_flat'] * 100:.0f}%"
    )


def orthogonalize(target: pd.DataFrame, controls: list) -> pd.DataFrame:
    """Residualize target on control signals cross-sectionally each date (spanning reg).
    Returns the part of `target` unexplained by `controls` (re-z-scored). Standard
    factor-distinctness test (Fama 1998; Barillas-Shanken 2018)."""
    resid = target.copy()
    for date in target.index:
        y = target.loc[date]
        df = pd.concat([y] + [c.loc[date] for c in controls], axis=1).dropna()
        if len(df) < 20:
            resid.loc[date] = np.nan
            continue
        yy = df.iloc[:, 0].to_numpy(float)
        XX = df.iloc[:, 1:].to_numpy(float)
        A = np.column_stack([np.ones(len(XX)), XX])
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        pred = A @ beta
        resid.loc[date, df.index] = yy - pred
    return zs(resid)


def avg_rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    corrs = []
    for date in a.index:
        ra = a.loc[date].dropna()
        rb = b.loc[date].reindex(ra.index).dropna()
        if len(rb) > 20:
            corrs.append(ra.corr(rb, method="spearman"))
    return float(np.nanmean(corrs)) if corrs else 0.0


def main():
    print("[broad] loading coins ...")
    frames = load_coins()
    print(f"[broad] {len(frames)} coins")
    close, vol, fund = weekly_panel(frames)
    scores = build_scores(close, vol, fund)
    # BTC regime gate
    btc = load_btc_close().resample("W").last().reindex(close.index).ffill()
    regime = (btc > btc.rolling(52, min_periods=13).mean()).astype(int)

    print("\n=== FULL BROAD UNIVERSE (416 coins) ===")
    for key in ("asym", "fvcs", "mht", "rev_plain", "revx"):
        report(f"{key}_LS", metrics(backtest(scores[key], close, regime=regime)))
        report(f"{key}_LS_IVW", metrics(backtest(scores[key], close, inv_vol=True, regime=regime)))

    # size-conditioned: dollar-volume tercile split (small/illiquid = bottom tercile)
    avg_vol = vol.mean()
    terc = avg_vol.rank(pct=True)
    small = terc[terc < 0.34].index
    large = terc[terc > 0.66].index
    print(
        f"\n=== SMALL / ILLIQUID ({len(small)} coins, bottom vol tercile) — where reversal lives ==="
    )
    for key in ("asym", "fvcs", "mht", "rev_plain", "revx"):
        sc = scores[key].reindex(columns=small)
        report(f"{key}_SMALL_LS", metrics(backtest(sc, close, regime=regime.reindex(sc.index))))
    print(f"\n=== LARGE / LIQUID ({len(large)} coins, top vol tercile) ===")
    for key in ("asym", "fvcs", "mht", "rev_plain", "revx"):
        sc = scores[key].reindex(columns=large)
        report(f"{key}_LARGE_LS", metrics(backtest(sc, close, regime=regime.reindex(sc.index))))

    # ORTHOGONALITY: is FVCS distinct from trend (MHT) and funding (ASYM), or trend in disguise?
    print("\n=== ORTHOGONALITY (spanning test) ===")
    print(f"  rank-corr FVCS~MHT : {avg_rank_corr(scores['fvcs'], scores['mht']):.2f}")
    print(f"  rank-corr FVCS~ASYM : {avg_rank_corr(scores['fvcs'], scores['asym']):.2f}")
    print(f"  rank-corr MHT~ASYM  : {avg_rank_corr(scores['mht'], scores['asym']):.2f}")
    fvcs_orth = orthogonalize(scores["fvcs"], [scores["mht"], scores["asym"]])
    print("  FVCS residualized on [MHT, ASYM] (the distinct part):")
    report("fvcs_orth_LS", metrics(backtest(fvcs_orth, close, regime=regime)))
    sc = fvcs_orth.reindex(columns=large)
    report("fvcs_orth_LARGE_LS", metrics(backtest(sc, close, regime=regime.reindex(sc.index))))

    # REVX NOVELTY: does the funding-crowding gate add value beyond the KNOWN plain reversal factor?
    print("\n=== REVX NOVELTY (gate vs plain reversal) ===")
    print(f"  rank-corr REVX~MHT       : {avg_rank_corr(scores['revx'], scores['mht']):.2f}")
    print(f"  rank-corr REVX~REV_PLAIN : {avg_rank_corr(scores['revx'], scores['rev_plain']):.2f}")
    print(f"  rank-corr REVX~ASYM      : {avg_rank_corr(scores['revx'], scores['asym']):.2f}")
    print(f"  rank-corr REV_PLAIN~MHT  : {avg_rank_corr(scores['rev_plain'], scores['mht']):.2f}")
    revx_orth = orthogonalize(scores["revx"], [scores["rev_plain"], scores["mht"], scores["asym"]])
    print("  REVX residualized on [REV_PLAIN, MHT, ASYM] (the novel funding-gated part):")
    report("revx_orth_LS", metrics(backtest(revx_orth, close, regime=regime)))
    sc = revx_orth.reindex(columns=small)
    report("revx_orth_SMALL_LS", metrics(backtest(sc, close, regime=regime.reindex(sc.index))))
    sc = revx_orth.reindex(columns=large)
    report("revx_orth_LARGE_LS", metrics(backtest(sc, close, regime=regime.reindex(sc.index))))


if __name__ == "__main__":
    main()
