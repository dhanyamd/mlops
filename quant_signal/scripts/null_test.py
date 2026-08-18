"""Null test: replace the real signal with a RANDOM one, same everything else.

If the backtest's profit survives randomised selections, the money is coming
from a mechanical artefact (fill timing, cost model, mark-to-market) rather
than from the factor. A real edge must collapse toward zero here.
"""
import json
import os
import random
import sys

os.environ.update({"QUANT_CGO_DIR":"1","QUANT_CGO_L":"7","QUANT_REGIME_OFF":"1","QUANT_SMB_OFF":"0",
  "QUANT_FACC_OFF":"1","QUANT_RCGO_W":"1.0","QUANT_RESEARCH_PARITY":"1","QUANT_CGO_GH":"1",
  "QUANT_TRAIL_OFF":"1"})
from config.settings import csv_list, get_settings
from stream.execution import PaperExecutionSimulator
from stream.kv import FakeKV
from stream.predictor import prediction_key

SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
s=get_settings(); uni=csv_list(s.stream_xs_universe)
cache=json.load(open("/tmp/quant_cache/asym_warm_start.json.binance"))
ci={x:{int(r[0]):(float(r[1]),float(r[2] or 0.0)) for r in cache["bars"].get(x,[])} for x in uni}
wins=sorted({w for x in uni for w in ci[x]})
WEEK=168*3_600_000
kv=FakeKV()
sim=PaperExecutionSimulator(kv,execution_prefix="execution:null",prediction_prefix="prediction:null",
  notional_usd=1000.0,slippage_bps=0.0,taker_fee_bps=2.75,window_ms=s.stream_window_ms,
  venue=None,hold_until_decay=True,max_hold_h=168,durable_log=False)
rnd=random.Random(SEED)
cur={}; last_wk=None
for w in wins:
    wk=w//WEEK
    if wk!=last_wk:                      # random reselection each week
        pool=list(uni); rnd.shuffle(pool)
        cur={x:("LONG" if i<6 else "SHORT" if i<12 else "FLAT") for i,x in enumerate(pool)}
        last_wk=wk
    for x in uni:
        kv.set_json(prediction_key("prediction:null",x),
          {"symbol":x,"window_end_ms":w,"direction":cur.get(x,"FLAT"),
           "predicted_return":0.5 if cur.get(x)=="LONG" else -0.5 if cur.get(x)=="SHORT" else 0.0,
           "updated_at":"2026-01-01T00:00:00+00:00"})
    for x in uni:
        cv=ci[x].get(w)
        if cv: sim.handle({"symbol":x,"close":cv[0],"volume":cv[1],"window_end_ms":w})
t=sum(sim._n_trades.values()); wn=sum(sim._n_wins.values()); pnl=sum(sim._realized_pnl.values())
print(f"{SEED},{t},{wn},{pnl:.4f}")
