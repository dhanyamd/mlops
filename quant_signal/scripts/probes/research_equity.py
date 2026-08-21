"""Show the VALIDATED research strategy's equity curve (NAV) as a sparkline.

This is the strategy that actually makes money (research_novel backtester) —
distinct from the live trading engine. Run: uv run python scripts/research_equity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import scripts.research_novel as rn

_BARS = " ▁▂▃▄▅▆▇█"


def nav_spark(wealth: np.ndarray, width: int = 70) -> str:
    lo, hi = float(wealth.min()), float(wealth.max())
    if hi <= lo:
        return "─" * width
    idx = np.linspace(0, len(wealth) - 1, width).astype(int)
    out = []
    for i in idx:
        v = float(wealth[i])
        out.append(_BARS[min(len(_BARS) - 1, int((v - lo) / (hi - lo) * (len(_BARS) - 1)))])
    return "".join(out)


def main() -> None:
    close, fd, fl, vl, fees, tvl = rn.load()
    fwd, mom, vol, S, G = rn.build_scores(close, fd, fl, vl, fees, tvl)
    picks = {
        "ENS_MCD_SLOW": dict(score="ens_mcd", regime=True, regime_mode="slow"),
        "ASYM_SLOW": dict(score="asym", regime=True, regime_mode="slow"),
        "ASYM_CARRY_REGIME": dict(score="asym_carry", regime=True),
    }
    for name, spec in picks.items():
        ret = rn.backtest(close, S[spec["score"]], spec)
        m = rn.metrics(ret)
        wealth = (1 + ret.dropna()).cumprod().values
        mult = float(wealth[-1]) if len(wealth) else 1.0
        print(f"\n=== {name}  (validated research strategy) ===")
        print(
            f"  weeks={m['n']}  ann_ret={m['ann_ret'] * 100:6.2f}%  "
            f"Sharpe={m['sharpe']:.2f}  maxDD={m['maxdd'] * 100:6.1f}%  %flat={m['pct_flat'] * 100:.0f}%"
        )
        print(f"  NAV growth (start=1.0 -> end={mult:.2f}x):")
        print(f"  {nav_spark(wealth)}")
        print(f"  min={wealth.min():.2f}  max={wealth.max():.2f}  end={wealth[-1]:.2f}")


if __name__ == "__main__":
    main()
