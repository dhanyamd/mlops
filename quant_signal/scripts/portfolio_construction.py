"""Portfolio construction -- the layer this project never built.

Every backtest run in this project so far combined factors by equal-weighting
their z-scores and then sorted into a top-20%/bottom-20% quintile book. The
Chinese sell-side literature treats that construction as a DIAGNOSTIC for
whether a factor works at all, never as the portfolio you actually run. Three
techniques separate the two, and none of them require a new factor:

1. 因子正交化 -- ORTHOGONALISATION (天风金工, 因子正交全攻略).
   Correlated factors double-count exposure: adding two 0.4-correlated scores
   silently bets twice on their shared component. 对称正交化 (symmetric
   orthogonalisation) rotates the factor matrix so the columns are mutually
   orthogonal while staying as close as possible to the originals:

       F_orth = F (F'F)^(-1/2)

   Unlike Schmidt/Gram-Schmidt it has no ordering dependence -- no factor is
   privileged as "first" -- and it needs no return data, so it cannot leak.
   The report's claim for this step alone is IR 1.7 -> 2.6+.

2. 因子合成 -- WEIGHTING (华泰多因子系列之十, 因子合成方法实证分析).
   Equal weight is the most STABLE but not the best; IC-weighting and
   maximised-ICIR do better. Max-ICIR is Markowitz with IC in place of return:

       w ∝ Σ_IC^{-1} · IC_mean

   Σ_IC is ill-conditioned at these dimensions, so it is shrunk (Ledoit-Wolf).
   All weights here are estimated on a TRAILING window only.

3. 换手率约束 -- TURNOVER CONSTRAINT.
   The same literature is blunt that rebalancing faster does NOT harvest more
   alpha -- "Alpha信号的半衰期通常远大于1天，频繁换手不会多赚Alpha，只会多交
   手续费" -- and that daily rebalancing at 10bps two-way burns ~25%/yr in
   friction. Their fix is not frequency but a CAP: 单次调仓换手率上限30%. When
   the target book implies more turnover than the cap, we move only partway
   toward it, which keeps most of the signal and bounds the cost exactly.

Nothing here is fitted on the evaluation sample: orthogonalisation is a
per-cross-section rotation using no returns, and the ICIR weights use a
trailing window that ends before the return being predicted.

Run (isolate the construction effect on the factors we already have):
    uv run python -m scripts.portfolio_construction
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd


def symmetric_orthogonalise(F: pd.DataFrame) -> pd.DataFrame:
    """对称正交化: F (F'F)^{-1/2}, computed on one cross-section.

    Rows are assets, columns factors. Rows with any missing exposure are left
    untouched (returned as NaN) rather than imputed, so a coin that is missing
    one factor never gets a fabricated exposure to it.
    """
    ok = F.notna().all(axis=1)
    if ok.sum() < len(F.columns) + 2:
        return F
    X = F[ok].to_numpy(dtype=float)
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    X = X / sd
    M = X.T @ X
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    S = V @ np.diag(w**-0.5) @ V.T          # M^{-1/2}, symmetric by construction
    out = F.copy() * np.nan
    out.loc[ok, :] = X @ S
    return out


def ledoit_wolf_shrink(S: np.ndarray, n: int) -> np.ndarray:
    """Shrink a covariance matrix toward a scaled identity."""
    k = S.shape[0]
    mu = np.trace(S) / k
    target = mu * np.eye(k)
    # shrinkage intensity grows as the sample shrinks relative to dimension
    alpha = min(1.0, max(0.0, k / max(n, 1)))
    return (1 - alpha) * S + alpha * target


def icir_weights(ic_hist: pd.DataFrame, mode: str) -> pd.Series:
    """Factor weights from trailing IC history. No future information."""
    cols = list(ic_hist.columns)
    m = ic_hist.mean()
    if mode == "equal":
        return pd.Series(1.0 / len(cols), index=cols)
    if mode == "ic":
        w = m
    elif mode == "icir":
        s = ic_hist.std().replace(0, np.nan)
        w = m / s
    elif mode == "max_icir":
        S = ic_hist.cov().to_numpy(dtype=float)
        S = ledoit_wolf_shrink(S, len(ic_hist))
        try:
            w = pd.Series(np.linalg.solve(S, m.to_numpy(dtype=float)), index=cols)
        except np.linalg.LinAlgError:
            w = m
    else:
        raise ValueError(mode)
    w = w.fillna(0.0)
    a = w.abs().sum()
    return w / a if a > 0 else pd.Series(1.0 / len(cols), index=cols)


def apply_turnover_cap(target: pd.Series, prev: pd.Series | None, cap: float) -> pd.Series:
    """单次调仓换手率上限: move only partway toward the target if it costs too much."""
    if prev is None or cap is None:
        return target
    turn = float((target - prev).abs().sum())
    if turn <= cap or turn == 0:
        return target
    lam = cap / turn
    return prev + (target - prev) * lam


def apply_turnover_priority(target: pd.Series, prev: pd.Series | None, cap: float) -> pd.Series:
    """INVENTED (this project): spend the turnover budget by conviction, not evenly.

    The published cap scales EVERY pending trade by the same lambda, so a
    position whose signal barely moved consumes budget at the same rate as one
    that flipped outright. Alpha is not distributed evenly across trades, so
    neither should the spending be. Here trades are ranked by |target - prev|
    and funded in full, largest first, until the budget is exhausted; the rest
    are simply not made this period.

    Same cost ceiling, but concentrated on the positions where the signal
    actually changed.
    """
    if prev is None or cap is None:
        return target
    delta = (target - prev)
    turn = float(delta.abs().sum())
    if turn <= cap or turn == 0:
        return target
    out = prev.copy()
    budget = cap
    for name in delta.abs().sort_values(ascending=False).index:
        d = float(delta[name])
        if budget <= 0:
            break
        step = math.copysign(min(abs(d), budget), d)
        out[name] = prev[name] + step
        budget -= abs(step)
    return out


def latent_sector_neutralise(F: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """INVENTED (this project): 行业中性化 without industry labels.

    A-share multi-factor books neutralise factor exposures against INDUSTRY
    dummies, which removes the dominant shared component and is a large part of
    why their information ratios survive out of sample. Crypto has no usable
    industry classification -- and needs the treatment more, since average
    pairwise correlation here runs ~0.8 against ~0.3 in equities. That shared
    mode is exactly what collapses effective breadth, and breadth enters the
    information ratio as a square root.

    So the basis is derived from the data instead of from labels: principal
    components of the TRAILING return correlation matrix stand in for latent
    sectors, and every factor exposure is residualised against them.

    How many components are real? Rather than fixing k, we use the
    Marchenko-Pastur upper edge from random matrix theory,

        lambda_max = (1 + sqrt(N/T))^2,

    which is the largest eigenvalue a PURE NOISE correlation matrix of this
    shape would produce. Eigenvalues above it carry structure that sampling
    noise cannot explain; those below it are noise and are left alone. k is
    therefore set by the shape of the data, not chosen.
    """
    R = rets.dropna(axis=1, how="any")
    if R.shape[0] < 20 or R.shape[1] < 10:
        return F
    T, N = R.shape
    C = np.corrcoef(R.to_numpy(dtype=float).T)
    if not np.all(np.isfinite(C)):
        return F
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    lam_max = (1.0 + math.sqrt(N / T)) ** 2       # Marchenko-Pastur noise edge
    k = int((vals > lam_max).sum())
    if k < 1:
        return F
    B = pd.DataFrame(vecs[:, :k], index=R.columns)   # latent sector loadings

    out = F.copy()
    for col in F.columns:
        y = F[col]
        idx = y.index.intersection(B.index)
        m = y.reindex(idx).notna()
        names = idx[m]
        if len(names) < k + 5:
            continue
        X = B.loc[names].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(names)), X])
        yy = y.reindex(names).to_numpy(dtype=float)
        try:
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out.loc[names, col] = yy - X @ beta          # residual = sector-neutral alpha
    return out


def quintile_weights(score: pd.Series, sym, top: float = 0.20) -> pd.Series | None:
    s = score.dropna()
    s = s[s != 0]
    if len(s) < 10:
        return None
    r = s.sort_values()
    n = max(2, int(round(top * len(r))))
    w = pd.Series(0.0, index=sym)
    w[list(r.index[-n:])] = 1.0 / n
    w[list(r.index[:n])] = -1.0 / n
    return w


def backtest(
    factors: dict[str, pd.DataFrame],
    fwd: pd.DataFrame,
    sym,
    *,
    orthogonalise: bool,
    weight_mode: str,
    turnover_cap: float | None,
    cost_bps: float,
    per_year: float,
    ic_window: int = 52,
    neutralise: bool = False,
    turnover_mode: str = "scale",
    px: pd.DataFrame | None = None,
    lsn_window: int = 52,
) -> pd.Series:
    """One book. Every switch is explicit so effects can be isolated."""
    grid = [w for w in next(iter(factors.values())).index if w in fwd.index]
    names = list(factors)
    ic_hist: list[dict] = []
    rets, ridx, prev = [], [], None
    hist_rets = px[sym].pct_change() if (neutralise and px is not None) else None

    for w in grid:
        F = pd.DataFrame({k: factors[k].loc[w].reindex(sym) for k in names})
        if neutralise and hist_rets is not None:
            # trailing returns strictly up to this rebalance -- no look-ahead
            past = hist_rets.loc[hist_rets.index <= w].iloc[-lsn_window:]
            if len(past) >= 20:
                F = latent_sector_neutralise(F, past)
        if orthogonalise:
            F = symmetric_orthogonalise(F)

        # trailing IC only -- computed from rows strictly before this rebalance
        if len(ic_hist) >= max(12, ic_window // 4):
            hist = pd.DataFrame(ic_hist[-ic_window:])
            wt = icir_weights(hist, weight_mode)
        else:
            wt = pd.Series(1.0 / len(names), index=names)

        combo = (F * wt).sum(axis=1, min_count=1)
        tgt = quintile_weights(combo, sym)
        if tgt is None:
            prev = None
            continue
        if turnover_mode == "priority":
            tgt = apply_turnover_priority(tgt, prev, turnover_cap)
        else:
            tgt = apply_turnover_cap(tgt, prev, turnover_cap)

        f = fwd.loc[w].clip(upper=1.0)
        ret = float((tgt * f).reindex(sym).sum(skipna=True))
        if prev is not None and cost_bps:
            ret -= cost_bps / 1e4 * float((tgt - prev).abs().sum())
        rets.append(ret)
        ridx.append(w)
        prev = tgt

        # record this period's realised IC for the NEXT period's weights
        row = {}
        for k in names:
            a, b = F[k], f
            m = a.notna() & b.notna()
            row[k] = a[m].corr(b[m], method="spearman") if m.sum() >= 10 else np.nan
        ic_hist.append(row)

    r = pd.Series(rets, index=ridx)
    return r[r != 0]


def stats(r: pd.Series, per_year: float) -> tuple[float, float, float, float]:
    v = r.std() * math.sqrt(per_year)
    sh = r.mean() * per_year / v if v > 0 else 0.0
    t = (r.mean() / r.std()) * math.sqrt(len(r)) if len(r) > 2 else float("nan")
    return r.mean() * per_year, v, sh, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--turnover-cap", type=float, default=0.30)
    a = ap.parse_args()

    from scripts.research_fas_clean import _liquidity_mask, _rank_z, load, smb_scores
    from scripts.research_salience import salience_and_turnover

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    st, tv = salience_and_turnover(dcl, dvl, cw.index, sym, 28)
    mom = (cw[sym] / cw[sym].shift(2) - 1.0)

    factors = {
        "SMB": smb_scores(vw, sym)[sym].apply(_rank_z),
        "ST": (-st).apply(_rank_z),
        "TVOL": (-tv).apply(_rank_z),
        "MOM2": mom.apply(_rank_z),
    }
    fwd = cw[sym].shift(-1) / cw[sym] - 1.0
    print(f"factors {list(factors)}   symbols {len(sym)}   weeks {len(cw)}")
    print("raw factor correlation (why orthogonalisation should matter):")
    flat = pd.DataFrame({k: v.stack() for k, v in factors.items()})
    print(flat.corr().round(2).to_string())

    cap = a.turnover_cap
    print(f"\ncost={a.cost_bps}bps   turnover cap={cap}")
    print(f"{'construction (one change at a time)':<46} {'n':>4} {'ann':>7} "
          f"{'vol':>7} {'Sharpe':>7} {'t':>6}")
    print("-" * 82)

    ladder = [
        ("baseline: equal-weight + quintile (what we ran all day)",
         dict(orthogonalise=False, weight_mode="equal", turnover_cap=None)),
        ("+ 因子正交化 (symmetric orthogonalisation)",
         dict(orthogonalise=True, weight_mode="equal", turnover_cap=None)),
        ("+ max-ICIR weighting (华泰)",
         dict(orthogonalise=True, weight_mode="max_icir", turnover_cap=None)),
        ("+ 换手率约束 (uniform scale cap)",
         dict(orthogonalise=True, weight_mode="max_icir", turnover_cap=cap)),
        ("+ INVENTED priority turnover allocation",
         dict(orthogonalise=True, weight_mode="max_icir", turnover_cap=cap,
              turnover_mode="priority")),
        ("+ INVENTED latent sector neutralisation (RMT-k)",
         dict(orthogonalise=True, weight_mode="max_icir", turnover_cap=cap,
              turnover_mode="priority", neutralise=True)),
    ]
    for label, kw in ladder:
        r = backtest(factors, fwd, sym, cost_bps=a.cost_bps, per_year=52.0,
                     px=cw, **kw)
        if len(r) < 20:
            print(f"{label:<46} insufficient")
            continue
        ann, vol, sh, t = stats(r, 52.0)
        print(f"{label:<46} {len(r):>4} {ann:>7.1%} {vol:>7.1%} {sh:>7.2f} {t:>6.2f}")

    print("\nisolating the two inventions against the same stack:")
    for label, kw in [
        ("LSN only (no orth, equal weight, no cap)",
         dict(orthogonalise=False, weight_mode="equal", turnover_cap=None,
              neutralise=True)),
        ("priority cap only (no orth, equal weight)",
         dict(orthogonalise=False, weight_mode="equal", turnover_cap=cap,
              turnover_mode="priority")),
        ("uniform cap only (no orth, equal weight)",
         dict(orthogonalise=False, weight_mode="equal", turnover_cap=cap)),
    ]:
        r = backtest(factors, fwd, sym, cost_bps=a.cost_bps, per_year=52.0,
                     px=cw, **kw)
        if len(r) < 20:
            continue
        ann, vol, sh, t = stats(r, 52.0)
        print(f"{label:<46} {len(r):>4} {ann:>7.1%} {vol:>7.1%} {sh:>7.2f} {t:>6.2f}")


if __name__ == "__main__":
    main()
