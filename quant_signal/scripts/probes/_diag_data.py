import numpy as np, pandas as pd
import scripts.research_fas_clean as m

cw, vw, aw, dcl, dvl = m.load("/tmp/quant_cache/asym_warm_start.json.binance")
syms = m._liquidity_mask(cw, vw)
fas = m.fas_scores(cw, aw, syms)
smb = m.smb_scores(vw, syms)
score = (fas[syms] + smb[syms]).apply(m._rank_z)
fwd = cw[syms].shift(-1) / cw[syms] - 1.0

# tradeable weeks: >=20 symbols have a close
tk = cw[syms].notna().sum(axis=1) >= 20
weeks = cw[syms].index[tk]
weeks = weeks[:-1]  # need a forward week
print(f"cross-verifying on {len(weeks)} tradeable weeks\n")


def _cgo_arr(c, v, L=7):
    if len(c) <= L:
        return np.nan
    P = np.asarray(c, float)
    V = np.asarray(v, float)
    Pt = P[-1]
    if Pt <= 0 or not np.isfinite(Pt):
        return np.nan
    num = sum((Pt - P[-1 - s]) * V[-1 - s] for s in range(1, L + 1))
    den = Pt * sum(V[-1 - s] for s in range(1, L + 1))
    return num / den if den > 0 else np.nan


def cgo_at(wk):
    out = {}
    for s in syms:
        dc = dcl[s].dropna()
        dv = dvl[s].reindex(dc.index).fillna(0.0)
        msk = dc.index < wk
        sc, sv = dc[msk], dv[msk]
        if sc.shape[0] <= 8:
            continue
        val = _cgo_arr(sc.values, sv.values)
        if np.isfinite(val):
            out[s] = val
    return out


def quintile_spread(series_dict, fwd_row, q=0.2):
    """mean fwd ret of top-q vs bottom-q ranked by series_dict."""
    items = [
        (s, series_dict[s])
        for s in series_dict
        if np.isfinite(series_dict[s]) and np.isfinite(fwd_row.get(s, np.nan))
    ]
    if len(items) < 8:
        return np.nan, np.nan, np.nan
    items.sort(key=lambda x: x[1])
    n = max(2, int(round(q * len(items))))
    bot = np.mean([fwd_row[s] for s, _ in items[:n]])
    top = np.mean([fwd_row[s] for s, _ in items[-n:]])
    return bot, top, top - bot


# accumulate spreads
fas_top, fas_bot, fas_spr = [], [], []
cgo_top, cgo_bot, cgo_spr = [], [], []
all_cgo, all_fwd = [], []
for wk in weeks:
    fr = fwd.loc[wk]
    sc = score.loc[wk].dropna()
    cg = cgo_at(wk)
    b, t, s = quintile_spread({s: sc[s] for s in sc.index}, fr)
    if np.isfinite(s):
        fas_top.append(t)
        fas_bot.append(b)
        fas_spr.append(s)
    b2, t2, s2 = quintile_spread(cg, fr)
    if np.isfinite(s2):
        cgo_top.append(t2)
        cgo_bot.append(b2)
        cgo_spr.append(s2)
    for s, v in cg.items():
        if np.isfinite(fr.get(s, np.nan)):
            all_cgo.append(v)
            all_fwd.append(fr[s])

fas_spr = np.array(fas_spr)
cgo_spr = np.array(cgo_spr)
all_cgo = np.array(all_cgo)
all_fwd = np.array(all_fwd)


def stat(x):
    x = np.asarray(x, float)
    return x.mean() * 52, x.mean() / x.std() * np.sqrt(52) if x.std() > 0 else 0, len(x)


print("=== FAS+SMB book (raw data cross-check) ===")
print(f"  top-quintile fwd ann : {stat(fas_top)[0] * 100:6.1f}%")
print(f"  bot-quintile fwd ann : {stat(fas_bot)[0] * 100:6.1f}%")
print(
    f"  L/S spread ann_ret   : {stat(fas_spr)[0] * 100:6.1f}%  Sharpe={stat(fas_spr)[2]:.2f}  (n={len(fas_spr)})"
)
print("\n=== CGO -> forward return (raw data, Griffin-Han direction test) ===")
print(f"  HIGH-CGO quintile fwd ann : {stat(cgo_top)[0] * 100:6.1f}%")
print(f"  LOW-CGO  quintile fwd ann : {stat(cgo_bot)[0] * 100:6.1f}%")
print(
    f"  HIGH-LOW spread ann_ret   : {stat(cgo_spr)[0] * 100:6.1f}%  Sharpe={stat(cgo_spr)[2]:.2f}  (n={len(cgo_spr)})"
)
corr = np.corrcoef(all_cgo, all_fwd)[0, 1]
print(f"  corr(CGO, fwd_ret) = {corr:.3f}  (positive => high CGO predicts UP, i.e. dir=+1 correct)")
