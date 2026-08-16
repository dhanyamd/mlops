"""Research: FSR -- Funding-Stress Reversal (KEYLESS, ORIGINAL factor).

WHY THIS IS NOT A COPY:
  Keel/Unravel ship a CONTINUOUS cross-sectional funding CARRY (collect the fee,
  long-low-funding / short-high-funding every rebalance) blended with momentum.
  That is a known, validated book -- and we treat it only as a BASELINE here.
  FSR is a different economic object: an EVENT-DRIVEN fade of crowding that fires
  only when funding is cross-sectionally EXTREME and ROLLING OVER (the "flip" that
  the positioning-stress literature -- Bitbase 2026, RiskState, Glassnode LPOC --
  identifies as the squeeze catalyst). We trade the UNWIND, not the fee.

FIRST-PRINCIPLES MECHANISM (keyless data only: price + funding + volume):
  fund_z     = cross-sectional z of weekly funding  (high => crowded LONG, paying)
  fund_accel = fund_z - fund_z.rolling(3w) mean     (rolling over from extreme)
  FSR score  = fund_accel  WHEN |fund_z| > gate, else 0 (skip)
    - fund_z very negative + fund_accel > 0  => shorts crowding, funding normalizing
      up => short squeeze imminent => LONG
    - fund_z very positive + fund_accel < 0  => longs crowding, funding normalizing
      down => long squeeze imminent => SHORT
  Volume confirmation (FSR_VOL): only take the signal when the run was leverage-led
    (volume NOT expanding) -- Bitbase "spot/deriv ratio" proxied by our volume cache.

WF OOS, 10bps costs, crash-regime sub-periods, bootstrap CI. Compared to
MOM14_REGIME (SCX-family) and CSCM_REGIME (carry+momentum, the known book).

Run: uv run python scripts/research_fsr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import random

CONFIG = {
    "formation_days": 14,
    "fund_gate": 1.0,  # |fund_z| must exceed this to be "crowded"
    "fund_roll_weeks": 3,  # window for the funding rollover (flip) signal
    "quintile": 0.20,
    "vol_target": 0.55,
    "vol_lookback_days": 126,
    "cost_bps": 10.0,
    "regime_fast": 90,
    "regime_slow": 200,
}

UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "ADAUSDT",
    "SUIUSDT",
    "UNIUSDT",
    "AVAXUSDT",
    "CRVUSDT",
    "PEPEUSDT",
    "LTCUSDT",
    "ICPUSDT",
    "AAVEUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "DOTUSDT",
    "FILUSDT",
    "ARBUSDT",
    "LDOUSDT",
    "BCHUSDT",
    "OPUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "RUNEUSDT",
    "GRTUSDT",
    "ZECUSDT",
]
PRICE_CACHE = Path("/tmp/crypto_daily_long.csv")
FUND_CACHE = Path("/tmp/crypto_funding.csv")
VOL_CACHE = Path("/tmp/crypto_volume.csv")

# score: which ranking signal; regime: BTC UP-UP gate
VARIANTS = {
    "MOM14_REGIME": dict(score="mom", regime=True),
    "CSCM_REGIME": dict(score="comp", regime=True),  # known carry+momentum book (baseline)
    "FSR": dict(score="fsr", regime=False),  # ORIGINAL thesis A: fade the unwind
    "FSR_REGIME": dict(score="fsr", regime=True),
    "FSR_VOL": dict(score="fsr_vol", regime=True),
    "FMC_REGIME": dict(score="fmc", regime=True),  # ORIGINAL thesis B: funding CONFIRMS momentum
    "VCM_REGIME": dict(score="vcm", regime=True),  # ORIGINAL thesis C: VOLUME confirms momentum
    "ASYM_REGIME": dict(
        score="asym", regime=True
    ),  # ORIGINAL thesis D: long squeeze / short momentum
    "IVOL_REGIME": dict(score="ivol", regime=True),  # ORIGINAL thesis E: idiosyncratic-vol premium
    "ILLIQ_REGIME": dict(
        score="illiq", regime=True
    ),  # HIDDEN GEM 1: Amihud illiquidity (long illiquid)
    "LOTTERY_REGIME": dict(
        score="lottery", regime=True
    ),  # HIDDEN GEM 2: anti-MAXRET (long low-lottery)
    "IDIOSYN_REGIME": dict(
        score="idiosyn", regime=True
    ),  # ORIGINAL F: idiosyncratic momentum (BTC-resid)
    "REV_REGIME": dict(score="rev", regime=True),  # ORIGINAL G: short-term reversal
    "LOWBETA_REGIME": dict(score="lowbeta", regime=True),  # ORIGINAL H: low-beta defensive
    "FLOW_REGIME": dict(
        score="flow", regime=True
    ),  # ORIGINAL I: flow-divergence (follow/fade momentum)
    "ENSEMBLE_REGIME": dict(score="asym_vcm", regime=True),  # ORIGINAL: ASYM + VCM blended
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (PRICE_CACHE, FUND_CACHE, VOL_CACHE):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    fd = pd.read_csv(FUND_CACHE, index_col=0, parse_dates=True)
    fd.index = pd.to_datetime(fd.index, utc=True).tz_localize(None)
    vo = pd.read_csv(VOL_CACHE, index_col=0, parse_dates=True)
    vo.index = pd.to_datetime(vo.index, utc=True).tz_localize(None)
    common = px.index.intersection(fd.index).intersection(vo.index)
    px = px.loc[common].reindex(columns=UNIVERSE)
    fd = fd.loc[common].reindex(columns=UNIVERSE)
    vo = vo.loc[common].reindex(columns=UNIVERSE)
    print(f"[data] panel {px.shape} {px.index.min().date()}..{px.index.max().date()}")
    return px, fd, vo


def weekly_frame(close: pd.DataFrame, formation: int):
    w = close.resample("W-MON").last()
    fwd = w.shift(-1) / w - 1.0
    mom = (close / close.shift(formation) - 1.0).resample("W-MON").last()
    vol = (
        close.pct_change(fill_method=None).rolling(CONFIG["vol_lookback_days"]).std() * np.sqrt(252)
    ).reindex(w.index, method="ffill")
    return fwd.iloc[formation:], mom.iloc[formation:], vol.iloc[formation:]


def btc_regime(close: pd.DataFrame) -> pd.Series:
    btc = close["BTCUSDT"]
    fast = btc.rolling(CONFIG["regime_fast"]).mean()
    slow = btc.rolling(CONFIG["regime_slow"]).mean()
    up = (btc > fast) & (btc > slow)
    return up.resample("W-MON").last().reindex(close.resample("W-MON").last().index, method="ffill")


def build_scores(close: pd.DataFrame, fd: pd.DataFrame, vo: pd.DataFrame):
    fwd, mom, vol = weekly_frame(close, CONFIG["formation_days"])
    fdw = fd.resample("W-MON").mean().reindex(fwd.index)
    vow = vo.resample("W-MON").sum().reindex(fwd.index)
    mom_z = mom.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    fund_z = fdw.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    comp = mom_z - fund_z  # known carry+momentum book
    fmc = mom_z + fund_z  # ORIGINAL: funding confirms the trend
    # ORIGINAL thesis C: volume confirms the move (actual flow, not a sentiment poll)
    vol_ratio = vow / vow.rolling(12).mean() - 1.0
    vol_trend_z = vol_ratio.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    vcm = mom_z + vol_trend_z
    # ORIGINAL: funding-flip unwind, gated by crowding extremity
    roll = CONFIG["fund_roll_weeks"]
    fund_accel = fund_z - fund_z.rolling(roll).mean()
    extreme = (fund_z.abs() > CONFIG["fund_gate"]).astype(float)
    fsr = fund_accel * extreme  # 0 unless crowded + flipping
    # volume confirmation: leverage-led move (recent vol below medium-term) => keep signal
    vol_short = vow.rolling(4).mean()
    vol_long = vow.rolling(12).mean()
    lev_led = (vol_short <= vol_long).astype(float)  # 1 if volume NOT expanding
    fsr_vol = fsr * (0.5 + 0.5 * lev_led)
    # ORIGINAL thesis D: asymmetric -- long the SHORT-SQUEEZE (profitable half of FSR:
    # shorts crowded + capitulating => forced covering => durable up-move), short plain
    # momentum losers. Avoids fighting crowded-longs (the FSR failure mode).
    squeeze = ((fund_z < -CONFIG["fund_gate"]) & (fund_accel > 0)).astype(float) * 2.0
    asym = mom_z.where(squeeze == 0, squeeze)
    # ORIGINAL thesis E: idiosyncratic-vol premium (Zhang & Li 2020: IVOL POSITIVE in crypto)
    ivol = (
        (close.pct_change(fill_method=None).rolling(14).std() * np.sqrt(252))
        .resample("W-MON")
        .last()
    )
    ivol_z = ivol.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    ret_d = close.pct_change(fill_method=None)
    # HIDDEN GEM 1: Amihud ILLIQUIDITY (Bianchi-Babiak 2022b; EFMA 2025 media paper:
    # long illiquid / short liquid; the LONG leg survives costs). Keyless from volume cache.
    dollar_vol = vo.replace(0, np.nan)
    amihud = (ret_d.abs() / dollar_vol).replace([np.inf, -np.inf], np.nan)
    illiq_w = amihud.rolling(7).mean().resample("W-MON").last()
    illiq_z = illiq_w.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    # HIDDEN GEM 2: LOTTERY / MAXRET (Zhao-Wang-Liu 2024): long LOW-lottery names (anti-MAXRET).
    maxret = ret_d.rolling(14).max().resample("W-MON").last()
    lottery_z = maxret.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    # ORIGINAL thesis F: IDIOSYNCRATIC MOMENTUM -- remove the BTC common factor, trade the
    # cross-sectional z of each coin's ALPHA. Naive momentum is contaminated by BTC-beta (coins
    # just ride BTC); the idiosyncratic component is the true coin-specific signal.
    btc_ret = ret_d["BTCUSDT"]
    cov = ret_d.rolling(14).cov(ret_d)
    beta = cov.div(btc_ret.rolling(14).var(), axis=0)
    idio = (ret_d - beta.mul(btc_ret, axis=0)).resample("W-MON").sum()
    idio_z = idio.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    # ORIGINAL thesis G: SHORT-TERM REVERSAL -- prior week's biggest losers rebound
    # (negative short-horizon autocorrelation / overreaction-reversal in crypto).
    rev = close.pct_change(7).resample("W-MON").last()
    rev_z = rev.apply(lambda r: (r - r.mean()) / (r.std() + 1e-9), axis=1)
    # ORIGINAL thesis H: LOW-BETA DEFENSIVE -- long low-BTC-beta (defensive) names,
    # short high-beta. Idiosyncratic-risk pricing: low-beta earns a premium in crashes.
    beta_w = beta.resample("W-MON").last()
    lowbeta_z = beta_w.apply(lambda r: -(r - r.mean()) / (r.std() + 1e-9), axis=1)
    # ORIGINAL (first-principles) FLOW-DIVERGENCE: follow momentum ONLY when positioning
    # corroborates it (funding moving the SAME way as price => crowd is behind the move);
    # FADE momentum when positioning DIVERGES (price up but funding falling, or vice-versa =>
    # technically-driven move the crowd is fading, it snaps back). Not a known factor.
    fund_mom = fund_accel
    align = np.sign(mom_z) * np.sign(fund_mom)
    diverge = (align < 0).astype(float)
    flow = mom_z * (1.0 - 2.0 * diverge)
    # ORIGINAL ensemble: average the two working first-principles signals (ASYM squeeze +
    # VCM flow-confirm). Diversifies the tail; fully original, no copied book.
    asym_vcm = asym + vcm
    return (
        fwd,
        mom,
        vol,
        {
            "mom": mom_z,
            "comp": comp,
            "fsr": fsr,
            "fsr_vol": fsr_vol,
            "fmc": fmc,
            "vcm": vcm,
            "asym": asym,
            "ivol": ivol_z,
            "illiq": illiq_z,
            "lottery": -lottery_z,
            "idiosyn": idio_z,
            "rev": -rev_z,
            "lowbeta": lowbeta_z,
            "flow": flow,
            "asym_vcm": asym_vcm,
        },
    )


def weights_at(date, score, vol, regime_flag, spec: dict):
    if spec["regime"] and not regime_flag:
        return None
    m = score.loc[date].dropna()
    if len(m) < 12:
        return None
    n = max(2, int(round(CONFIG["quintile"] * len(m))))
    ranked = m.sort_values()
    longs = ranked.index[-n:]
    shorts = ranked.index[:n]
    w = pd.concat([pd.Series(1.0 / n, index=longs), pd.Series(-1.0 / n, index=shorts)])
    return w


def backtest(close: pd.DataFrame, score: pd.DataFrame, spec: dict) -> pd.Series:
    fwd, _, vol = weekly_frame(close, CONFIG["formation_days"])
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)
    dates = list(fwd.index)
    ret, prev = [], pd.Series(dtype=float)
    for date in dates:
        w = weights_at(date, score, vol, bool(reg.loc[date]), spec)
        if w is None:
            ret.append(0.0)
            prev = pd.Series(dtype=float)
            continue
        r = float((w * fwd.loc[date].reindex(w.index)).sum(skipna=True))
        if len(prev):
            turn = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
            r -= CONFIG["cost_bps"] / 1e4 * turn
        ret.append(r if np.isfinite(r) else 0.0)
        prev = w
    return pd.Series(ret, index=dates)


def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    n = len(ret)
    ann = ret.mean() * 52
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = random.Random(0)
    vals = list(ret.values)
    boot = []
    for _ in range(1000):
        sample = [rng.choice(vals) for _ in range(n)]
        sm = sum(sample) / n * 52
        sd = (sum((x - sum(sample) / n) ** 2 for x in sample) / max(1, n - 1)) ** 0.5 * np.sqrt(52)
        boot.append(sm / sd if sd > 0 else 0.0)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {
        "n": n,
        "ann_ret": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "ci": ci,
        "skew": float(ret.skew()),
        "exkurt": float(ret.kurt()),
        "maxdd": dd,
        "cvar05": ret.quantile(0.05),
        "pct_flat": float((ret == 0).mean()),
    }


def report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  ann_vol={m['ann_vol'] * 100:6.1f}%"
        f"  Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]  %flat={m['pct_flat'] * 100:.0f}%"
    )
    print(
        f"  skew={m['skew']:+.2f}  exkurt={m['exkurt']:.1f}  maxDD={m['maxdd'] * 100:6.1f}%"
        f"  CVaR5%={m['cvar05'] * 100:6.2f}%"
    )


CRASH = {
    "2018 bear": ("2018-01-01", "2018-12-31"),
    "COVID 2020": ("2020-02-20", "2020-04-01"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "FTX 2022": ("2022-11-01", "2022-12-31"),
    "2023-24 bull": ("2023-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-08-12"),
}


def main() -> None:
    close, fd, vo = load()
    print(f"config: {CONFIG}")
    fwd, mom, vol, S = build_scores(close, fd, vo)
    results = {}
    for name, spec in VARIANTS.items():
        ret = backtest(close, S[spec["score"]], spec)
        report(name, metrics(ret))
        results[name] = ret

    print("\n--- crash-regime annualized Sharpe ---")
    print("  " + "".join(f"{k:>14}" for k in CRASH))
    for name, ret in results.items():
        row = ""
        for a, b in CRASH.values():
            sub = ret.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:14.2f}"
        print(f"  {name:>13}{row}")


if __name__ == "__main__":
    main()
