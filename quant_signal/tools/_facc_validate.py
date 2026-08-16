import sys, math, random

sys.path.insert(0, ".")
from stream.asym_signal import AsymSignal
import numpy as np

HOUR_MS = 3_600_000
WEEK_MS = 7 * 24 * HOUR_MS

syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BCHUSDT", "ADAUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT"]
n_weeks = 45
hours = n_weeks * 7 * 24

# synthetic hourly timestamps (UTC-aligned weeks)
t0 = n_weeks * WEEK_MS  # arbitrary large epoch; only relative spacing matters
ends = [t0 + h * HOUR_MS for h in range(hours)]

sig = AsymSignal(kv=None, prediction_prefix="pred", universe=syms, min_symbols=4, regime=False)

rng = random.Random(7)
for s in syms:
    price = 100.0 * (1 + rng.random())
    closes, vols, fenv = [], [], []
    fp = rng.uniform(-0.0003, 0.0003)
    for h, e in enumerate(ends):
        # random walk price
        price *= 1 + rng.gauss(0, 0.01)
        vol = abs(rng.gauss(1e6, 3e5)) + 1e5
        closes.append((e, price, vol))
        # funding every 8h
        if h % 8 == 0:
            # inject a clear funding-acceleration spike on SOLUSDT in the last weeks
            if s == "SOLUSDT" and h > hours - 200:
                fp = 0.001  # rising funding -> accel up
            elif s == "ADAUSDT" and h > hours - 200:
                fp = -0.001  # falling funding -> accel down
            else:
                fp = rng.uniform(-0.0004, 0.0004)
            fenv.append((e, fp))
    for c in closes:
        sig._record(s, c[0], c[1], c[2])
    sig._funding[s] = [(e, r) for e, r in fenv]

last_week = (ends[-1] // WEEK_MS) * WEEK_MS

s0 = syms[0]
print("DEBUG s0 weekly_close weeks:", len(sig._weekly_close(s0)))
print("DEBUG s0 weekly_fund weeks:", len(sig._weekly_fund(s0)))
print("DEBUG s0 weekly_volume weeks:", len(sig._weekly_volume(s0)))
print("DEBUG max_horizon:", sig._max_horizon, "min_symbols:", sig._min_symbols)

out_on = sig._fas_scores(last_week)
out_off = AsymSignal(
    kv=None, prediction_prefix="p", universe=syms, min_symbols=4, regime=False, use_facc=False
)._fas_scores(last_week)
# rebuild off with same data
sig_off = AsymSignal(
    kv=None, prediction_prefix="p", universe=syms, min_symbols=4, regime=False, use_facc=False
)
for s in syms:
    for c in sig._closes[s]:
        sig_off._record(s, c[0], c[1], c[2])
    sig_off._funding[s] = sig._funding[s]
out_off = sig_off._fas_scores(last_week)

print("FACC ON  n=", len(out_on), "sample:", {k: round(v, 3) for k, v in list(out_on.items())[:4]})
print("FACC OFF n=", len(out_off))
diff = {
    s: round(out_on.get(s, 0) - out_off.get(s, 0), 4)
    for s in out_on
    if abs(out_on.get(s, 0) - out_off.get(s, 0)) > 1e-9
}
print("symbols whose score changed by FACC:", list(diff.keys()))
print(
    "SOLUSDT score ON/OFF:", round(out_on.get("SOLUSDT", 0), 3), round(out_off.get("SOLUSDT", 0), 3)
)
print(
    "ADAUSDT score ON/OFF:", round(out_on.get("ADAUSDT", 0), 3), round(out_off.get("ADAUSDT", 0), 3)
)
vals = list(out_on.values())
print(
    "rank-z bounds ok:",
    min(vals) >= -1.01 and max(vals) <= 1.01,
    "min/max=",
    round(min(vals), 3),
    round(max(vals), 3),
)
print("VALIDATION:", "PASS" if len(out_on) >= 4 and diff else "FAIL")
