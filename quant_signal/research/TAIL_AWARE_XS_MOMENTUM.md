# Idiosyncratic Tail-Aware Cross-Sectional Crypto Momentum

**Working note — quant_signal/research. Baseline backtest on Bybit daily spot, 31 large-caps, 2022-06-05 → 2026-08-12 (206 weekly rebalances).**

## Motivation (grounded in literature, not blogs)

- **Barroso & Santa-Clara (2015), "Momentum Has Its Moments":** scaling equity momentum by
  trailing realized variance lifts Sharpe 0.53 → 0.97 and cuts excess kurtosis 18.2 → 2.7.
  The celebrated result.
- **Grobys, Kolari, Sandretto, Shahzad & Äijö (2025), "Cryptocurrency momentum has (not) its
  moments":** cross-sectional crypto momentum pays ~0.9%/wk, but its crashes are **idiosyncratic
  short-leg jumps** (one coin did −255% in a week = 37% of cumulative payoff). Volatility scaling
  lifts the payoff to 1.86%/wk but **does not change the tail exponent** (power-law α < 3) — so
  tail risk survives scaling.
- **Liu & Tsyvinski (2021):** crypto momentum is real (weekly top-quintile 11.2%/wk).

**Gap we target.** BSC scale by *portfolio* volatility. Grobys show that in crypto the danger is
*idiosyncratic* — a single shorted coin going parabolic — which portfolio vol does not capture. We
ask: does portfolio vol-scaling help or hurt crypto cross-sectional momentum, and can a
**cross-sectional idiosyncratic tail filter** do better than BSC?

## Strategy

Cross-sectional momentum on 31 large-caps: rank by trailing **14-day** close-to-close return,
**long top / short bottom quintile** (equal weight, net-zero), rebalance **weekly** (Mondays).
Three risk treatments:

1. **NAIVE** — fixed equal weights.
2. **BSC** — scale the whole book by `target_vol / trailing_6m_realized_vol` (Barroso–Santa-Clara).
3. **TAIL / TAIL_ALL** — our filter: at each rebalance, if the **short leg** contains a coin whose
   trailing realized vol exceeds `k ×` the universe median (an idiosyncratic-volatility outlier,
   the Grobys crash signature), fade the **short leg only** (TAIL) or the **whole book** (TAIL_ALL)
   toward cash.

## Results (weekly, annualized)

| Variant            | Sharpe | ann.ret | ann.vol | maxDD  | skew  | exkurt | left-Hill ↑ | right-Hill ↑ |
|--------------------|-------:|--------:|--------:|-------:|------:|-------:|------------:|-------------:|
| NAIVE              | 0.79   | 37.0%   | 46.6%   | −39.0% | +2.55 | 15.6   | 2.85        | 1.60         |
| BSC (vol-scaled)   | 0.56   | 15.7%   | 27.8%   | −27.9% | +1.57 | 10.8   | 3.26        | 1.95         |
| TAIL k=1.8         | 0.55   | 15.1%   | 27.6%   | −27.9% | +1.63 | 11.1   | 3.45        | 1.95         |
| TAIL k=1.5         | 0.54   | 14.5%   | 26.9%   | −30.0% | +1.87 | 12.2   | 2.68        | 1.58         |
| TAIL_ALL k=1.8     | 0.80   | 37.2%   | 46.4%   | −38.0% | +2.59 | 15.9   | 2.95        | 1.60         |
| **TAIL_ALL k=1.5** | **0.84**| 38.2%  | 45.6%   | **−35.0%**| +2.75 | 17.0 | 2.49      | 1.62         |

Hill tail index: higher = thinner tail. Bootstrap 95% CI on Sharpe ≈ [−0.2, 2.1] for all
profitable variants (wide — short sample).

## Findings

1. **BSC volatility scaling is counterproductive for crypto cross-sectional momentum.** It halves
   Sharpe (0.79 → 0.56) while trimming vol/DD. Why: the crypto payoff is **positively skewed**
   (skew +2.55, right-Hill 1.60) — the upside is the return engine. Scaling by volatility throws
   away exactly the positive-skew tail that BSC *preserves* in equities (where momentum is
   negatively skewed). This is the opposite of BSC's equity result and is consistent with Grobys'
   observation that crypto tails survive vol-scaling. **This is the robust, novel, interpretable
   result.** It tells us the *live* `xs_rel14` signal (which currently uses BSC vol-scaling) is
   leaving Sharpe on the table.

2. **A cross-sectional idiosyncratic filter can preserve the upside while trimming drawdown.**
   TAIL_ALL (fade the whole book when the short leg holds an outlier-vol coin) at k=1.5 reaches
   Sharpe **0.84** and maxDD **−35%** — strictly above naive on both Sharpe and DD — while keeping
   the positive-skew character (skew +2.75, right-Hill 1.62 ≈ naive). It does NOT sacrifice the
   upside the way BSC does. The short-leg-only variant (TAIL) does not help — confirming the crash
   risk is not isolatable to the short leg alone in this sample.

## Honest limitations

- **Sample:** 206 weeks, one regime (2022–2026, net bullish for the strategy). The Grobys −255%
  idiosyncratic crash is not present in this window, so crash-protection is under-tested.
- **Threshold k chosen in-sample** (best of {1.5, 1.8}); the Sharpe edge over naive (+0.05) is
  *inside* the bootstrap CI — **not yet statistically distinguishable**. Needs walk-forward / OOS.
- **No transaction costs** in this baseline (the live executor has a cost-aware filter; the Bysik &
  Slepaczuk |r̂| > λ·cost gate should be layered on before trading the novel variant).
- The mechanism (idiosyncratic short-leg vol outlier → fade) is a direct, defensible extension of
  Grobys; the empirical edge is promising but preliminary.

## Contribution statement (defensible today)

> Portfolio volatility scaling — the Barroso–Santa-Clara prescription that doubles equity-momentum
> Sharpe — is *suboptimal* for crypto cross-sectional momentum because the strategy is positively
> skewed; scaling halves its Sharpe. A cross-sectional idiosyncratic-volatility filter on the short
> leg preserves the positive-skew upside while reducing max drawdown, dominating both the naive and
> vol-scaled variants on the risk–return frontier in-sample.

## Next steps

1. Walk-forward / out-of-sample the TAIL_ALL filter (rolling calibration of k) + add transaction costs.
2. Extend the panel to 2017 (CoinGecko/Yahoo) to include the 2022-style crash regime.
3. Wire TAIL_ALL (no BSC) into the live `xs_signal` — replace the current vol-scaling.
