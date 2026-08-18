import scripts.research_fas_clean as m

cw, vw, aw, dcl, dvl = m.load("/tmp/quant_cache/asym_warm_start.json.binance")
syms = m._liquidity_mask(cw, vw)

# 1) data coverage: first valid weekly close per symbol
print("=== per-symbol first valid weekly close ===")
fc = cw[syms].apply(lambda s: s.dropna().index.min())
for s in sorted(syms, key=lambda x: fc[x]):
    print(f"  {s:10s} {fc[s].date()}")
print(
    "common start (max of firsts):",
    fc.max().date(),
    "| cw span:",
    cw.index.min().date(),
    "..",
    cw.index.max().date(),
)

# 2) weeks where >=20 symbols have data (tradeable weeks)
tradeable = cw[syms].notna().sum(axis=1) >= 20
print("tradeable weeks (>=20 syms):", int(tradeable.sum()), "of", len(cw))

# 3) CGO direction test
fas = m.fas_scores(cw, aw, syms)
smb = m.smb_scores(vw, syms)
score = (fas[syms] + smb[syms]).apply(m._rank_z)


def run(cgo, tag):
    r = m.backtest(cw, fas, smb, cgo, syms)
    print(
        f"  {tag:24s} Sharpe={r['sharpe']:.2f} ann_ret={r['ann_ret'] * 100:6.1f}% vol={r['ann_vol'] * 100:5.1f}% weeks={r['weeks']} maxDD={r['maxdd'] * 100:6.1f}%"
    )


print("=== CGO direction / off ===")
run({}, "CGO OFF")
run(m.cgo_filter_daily(dcl, dvl, score.index, syms, d=-1), "CGO dir=-1 (keep LOW)")
run(m.cgo_filter_daily(dcl, dvl, score.index, syms, d=1), "CGO dir=+1 (keep HIGH)")
run(m.cgo_filter_daily(dcl, dvl, score.index, syms, L=14, d=-1), "CGO L=14 dir=-1")
run(m.cgo_filter_daily(dcl, dvl, score.index, syms, q=0.5, d=-1), "CGO q=0.5 dir=-1")
