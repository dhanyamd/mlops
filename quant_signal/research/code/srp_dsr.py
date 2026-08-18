"""Deflated Sharpe Ratio for SRP -- with N and sd MEASURED, not asserted.

Bailey & Lopez de Prado (2014) deflate an observed Sharpe by the expected
maximum Sharpe of the best of N zero-skill trials:

    SR* = sd(SR_trials) * [ (1-gamma) Z^-1(1 - 1/N) + gamma Z^-1(1 - 1/(N e)) ]
    DSR = PSR(SR*) = Z[ (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) ]

Both deflation inputs describe the SEARCH, not the strategy, so neither can be
recovered from the winning backtest. The previous implementation therefore took
them from the researcher -- a hardcoded trial count and a permutation-based
stand-in for the spread. That is circular: the whole premise of DSR is that a
researcher's own recollection of how many things they tried is the least
reliable number in the study, because the undercount is unconscious.

Here both come from ``scripts.trial_registry``, which is written by the code
that runs each backtest (``scripts.srp_sweep``). Nothing on this page is typed
by a human.

ON THE CHOICE OF N
------------------
Trials are not independent -- they share factors, a universe and a price panel --
so the number of EFFECTIVELY independent trials is below the raw count. We
report the raw count anyway. Raising N raises SR* and therefore LOWERS the DSR,
so the raw count is the conservative choice: it charges the strategy for more
independent searching than actually occurred. If DSR passes at the raw N it
passes at any defensible effective N, and the sensitivity table shows by how
much.

Run:
    uv run python -m scripts.srp_sweep      # populate the registry
    uv run python -m scripts.srp_dsr        # deflate against it
"""

from __future__ import annotations

import argparse
import math

from scipy import stats

from scripts.deflated_sharpe import expected_max_sr, psr
from scripts.trial_registry import DEFAULT_PATH, load_trials, trial_stats

FAMILY = "srp"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default=FAMILY)
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    ap.add_argument("--config-hash", default=None,
                    help="deflate this trial; default = the best one (the selected strategy)")
    a = ap.parse_args()

    st = trial_stats(a.family, a.path)
    if st["n_trials"] == 0:
        raise SystemExit(
            f"registry {a.path} has no '{a.family}' trials.\n"
            "Run:  uv run python -m scripts.srp_sweep"
        )
    if st["n_finite"] < 2:
        raise SystemExit(
            f"only {st['n_finite']} trial(s) produced a finite Sharpe; "
            "sd across trials is undefined. Widen the sweep."
        )

    trials = load_trials(a.family, a.path)
    if a.config_hash:
        sel = [t for t in trials if t.config_hash == a.config_hash]
        if not sel:
            raise SystemExit(f"no trial with hash {a.config_hash}")
        chosen = sel[0]
    else:
        chosen = max((t for t in trials if math.isfinite(t.sharpe_weekly)),
                     key=lambda t: t.sharpe_weekly)

    N = st["n_trials"]
    sd = st["sd_weekly"]
    sr_w = chosen.sharpe_weekly
    T = chosen.n_obs

    # Higher moments of the SELECTED strategy's own return series. Recomputed
    # from the backtest rather than stored, so this cannot drift from the config.
    from scripts.srp_backtest import SRPConfig, SRPData, run

    cfgd = dict(chosen.config)
    smooth = int(cfgd.pop("smooth", 20))
    factors = cfgd.pop("factors", None)
    costs = bool(cfgd.pop("costs", True))
    ranking = cfgd.pop("ranking", "self")
    combine = cfgd.pop("combine", "books")
    weighting = cfgd.pop("weighting", "riskparity")
    data = SRPData.load()
    res = run(data, SRPConfig(**cfgd), smooth=smooth,
              factors=tuple(factors) if factors else None, costs=costs,
              ranking=ranking, combine=combine, weighting=weighting)
    r = res.returns
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    if abs(res.sharpe_weekly - sr_w) > 1e-9:
        print(f"  WARNING: re-run Sharpe {res.sharpe_weekly:.6f} != logged "
              f"{sr_w:.6f} -- registry is stale for {chosen.config_hash}")
        sr_w, T = res.sharpe_weekly, res.n_obs

    print("=== selected strategy ===")
    print(f"  config hash           : {chosen.config_hash}  (git {chosen.git_sha})")
    print(f"  weekly observations T : {T}")
    print(f"  weekly Sharpe         : {sr_w:.4f}")
    print(f"  annualised Sharpe     : {sr_w * math.sqrt(52):.3f}")
    print(f"  skew                  : {skew:+.3f}")
    print(f"  kurtosis (non-excess) : {kurt:.3f}")
    print(f"  t-statistic           : {sr_w * math.sqrt(T):.2f}")
    print(f"  PSR(SR*=0)            : {psr(sr_w, 0.0, T, skew, kurt):.4f}")

    print("\n=== search, as measured from the registry ===")
    print(f"  registry              : {a.path}")
    print(f"  distinct configs run  : {N}")
    print(f"  with finite Sharpe    : {st['n_finite']}")
    print(f"  mean trial Sharpe (wk): {st['mean_weekly']:+.4f}")
    print(f"  sd  trial Sharpe (wk) : {sd:.4f}   <- DSR deflation input")
    print(f"  best trial (wk)       : {st['best_weekly']:.4f}")

    sr_star = expected_max_sr(sd, N)
    dsr = psr(sr_w, sr_star, T, skew, kurt)
    print(f"\n=== deflated Sharpe (N measured = {N}) ===")
    print(f"  SR* (expected max of {N} zero-skill trials): {sr_star:.4f} weekly")
    print(f"  DSR                                        : {dsr:.4f}   "
          f"{'PASS (>0.95)' if dsr > 0.95 else 'FAIL'}")

    # Effective N is BELOW the raw count (trials share data and factors), and a
    # lower N is a lower hurdle. Show that the verdict does not hinge on it.
    print("\n  sensitivity -- raising N only makes the hurdle harder:")
    print(f"    {'N':>8} {'SR* (wk)':>10} {'DSR':>9}")
    print("    " + "-" * 30)
    for n in sorted({max(2, N // 4), max(2, N // 2), N, N * 2, N * 4, 1000}):
        s_ = expected_max_sr(sd, n)
        print(f"    {n:>8} {s_:>10.4f} {psr(sr_w, s_, T, skew, kurt):>9.4f}"
              f"{'  <- measured' if n == N else ''}")


if __name__ == "__main__":
    main()
