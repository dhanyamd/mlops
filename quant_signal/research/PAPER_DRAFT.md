# Alpha Without a Peer Group: Self-Referential Factor Investing in Perpetual Futures

*Working paper — draft for Management Science. Prepared for double-anonymous review;
author identifiers removed.*

---

## Abstract

*(≤250 words per Management Science guidelines; 100-word plain-language version for FAJ in Appendix D)*

Cross-sectional factor investing rests on an assumption so basic it is rarely
stated: that assets can meaningfully be ranked against one another. Equities
supply the anchor that makes this reasonable — shared accounting, industry
structure, and cash flows. Cryptocurrency perpetual futures supply none of it. We
ask what happens to factor construction when the comparability anchor is absent.
Transporting ten factors from the published equity literature to 112 perpetual
contracts over a 363-week panel, we find that ranking each contract against its
own trailing history dominates ranking it against its peers in nine of eleven
cases, improving mean annualised Sharpe from 1.01 to 1.45. We decompose the
effect: approximately 41% is attributable to scale normalisation and the
remainder to distributional robustness, since a rank transform is invariant to
distributional shape while standardisation assumes approximate normality. We
further show that preserving factor independence through portfolio construction —
forming one book per factor and combining returns rather than blending scores —
adds 0.665 Sharpe (t = 5.82) when both constructions are tuned over an identical
grid. We introduce a factor exploiting aggressor-side information that centralised
crypto venues publish exactly and equity research must infer, and which ranks third
of eleven by leave-one-out contribution. Results
survive walk-forward selection across seven rebalance anchors (mean out-of-sample
Sharpe 2.068, all profitable), a held-out universe test, transaction costs at four
times the modelled level, and a deflated Sharpe correction whose trial count is
read from an executed experiment log rather than recalled.

**Keywords:** factor investing, cross-sectional ranking, cryptocurrency,
perpetual futures, portfolio construction, backtest overfitting

**JEL:** G11, G12, G14, C58

---

## 1. Introduction

Every cross-sectional factor model begins with a comparison. We say a stock is
cheap, or small, or heavily traded, and each of those words carries an implicit
completion: *relative to other stocks*. The comparison feels natural because
equity markets supply the scaffolding that makes it meaningful. Two firms report
earnings under the same accounting standards, operate in classifiable industries,
and are valued against cash flows that exist. When we rank them, we are ranking
like against like.

Cryptocurrency perpetual futures dissolve that scaffolding entirely. There are no
earnings. There is no industry classification. There is no accounting identity
tying price to anything. Asking whether one token's turnover is high *relative to
another token's* is a question the data cannot really answer, because the two
quantities were never commensurate to begin with. And yet the factor literature,
when it has turned to this market, has almost universally imported the
cross-sectional apparatus wholesale.

This paper asks what that assumption costs, and what replaces it.

Our approach is deliberately conservative. Rather than proposing new signals, we
take ten factors already established in the published equity literature and
transport them unchanged to 112 perpetual contracts across a 363-week panel. The
factors keep their original definitions and their original sign conventions; we
fit nothing. The only thing we vary is the frame of comparison. In one
specification each contract is ranked against its peers in the same week, as the
literature does. In the other, each contract is ranked against its own trailing
history — a self-referential frame that asks not "is this asset unusual compared
to others" but "is this asset unusual *for itself*."

The self-referential frame wins in nine of eleven cases, raising mean annualised
Sharpe from 1.01 to 1.45. Importantly, it does not merely win; we can say why. A
controlled intermediate specification — standardising each asset against its own
history before ranking cross-sectionally — recovers 41% of the gap, identifying
scale heterogeneity as one mechanism. The remaining 59% survives standardisation,
and we attribute it to distributional robustness: a rank transform is invariant to
the shape of the distribution it is applied to, while a z-score assumes
approximate normality, an assumption crypto factor distributions violate severely.

A second finding concerns what happens after the ranking. Standard practice
blends factor scores into a composite and forms a single portfolio from it. This
is a lossy operation: it forces several near-independent signals through one
ordering and discards most of their independence before a position is taken. When
we instead form one book per factor and combine the *returns*, holding every
hyperparameter fixed, performance improves by 0.665 Sharpe (t = 5.82), with the
self-referential construction winning in 15 of 17 matched comparisons. Both
constructions were tuned over an identical grid; comparing a tuned method against
an untuned baseline is the most common way this literature manufactures a result,
and we take care not to.

Third, and most directly connected to our thesis, we construct a factor that
**cannot exist in equity data**. Research on Chinese A-shares has established that
predictive information lives in the *shape* of the per-bar trade-size distribution
rather than its mean. That measure is necessarily direction-blind: A-share tick
data does not reliably identify the aggressor, so a large buy and a large sell are
indistinguishable. Perpetual futures publish taker-side volume on every bar. This
permits the same distribution to be split by initiating side and its asymmetry
measured — an observation an entire literature was structurally unable to make.
The factor is modest in isolation, but its significance is conceptual: it
demonstrates that what differs across these markets is not only the assets but the
*information environment*, and that the environment is what makes the
methodological question live.

Finally, this paper is unusual in what it reports about itself. During
preparation, an audit of every headline claim against the code that supposedly
produced it found four figures with no executable artifact — including a pair of
statistics that, on inspection, proved to be tail-index values transcribed from a
document about a different strategy. We report this in Section 8, along with the
infrastructure now preventing its recurrence: an append-only experiment registry
from which the multiple-testing correction reads its inputs, and a parity gate
asserting that the deployed system computes the same positions as the research
code. The relationship between a reported backtest and the code that generated it
is, we argue, an empirical question that ought to be tested rather than assumed.

---

## 2. The comparability assumption

Let $f_{i,t}$ denote a raw factor observation for asset $i$ at time $t$. A
cross-sectional factor model transforms this into a score by ranking within a date:

$$s^{\text{XS}}_{i,t} = \text{rank}\big(f_{i,t} \mid \{f_{j,t}\}_{j=1}^{N}\big)$$

This transformation is meaningful only if the $f_{j,t}$ are commensurate — if the
comparison of asset $i$'s value against asset $j$'s carries information about
their relative future returns. In equities this holds approximately, sustained by
common exposure to observable fundamentals (Fama and French 1993) and by
characteristics with shared economic interpretation (Daniel and Titman 1997).

The self-referential alternative ranks within an asset instead:

$$s^{\text{TS}}_{i,t} = \text{rank}\big(f_{i,t} \mid \{f_{i,\tau}\}_{\tau = t-w}^{t-1}\big)$$

This transformation makes no cross-asset comparison at all. Its closest antecedent
is time-series momentum (Moskowitz, Ooi, and Pedersen 2012), which demonstrated
that an asset's own past return predicts its future return independently of any
cross-sectional ordering. Our contribution is not the primitive but its systematic
application: we apply it to eleven factors spanning price, flow, positioning and
trade-size distribution, and we identify the mechanism that makes it work.

The distinction matters because these two transformations degrade differently. As
the cross-section becomes more heterogeneous, $s^{\text{XS}}$ increasingly ranks
*asset identity* rather than *signal*: a contract whose order flow is
systematically larger than another's will occupy the same tail of the
cross-sectional distribution week after week, regardless of whether anything
informative has happened to it. $s^{\text{TS}}$ is immune to this by construction,
because the comparison set contains only the asset itself.

We should be explicit about what our evidence does and does not support. We find
**no sign reversal** — cross-sectional ranking remains profitable in this market
(mean annualised Sharpe 1.01 across eleven single-factor books). The claim is one
of degradation, not failure. A stronger reading, that the cross-sectional
framework does not transfer to markets without a comparability anchor, is not
supported by these data and we do not make it.

---

## 3. Data and setting

**Universe.** 112 USDT-margined perpetual futures contracts, screened for
continuous trading and finite positive prices. The panel spans 363 weeks; after
factor warm-up the strategy is evaluable on 308 weekly rebalances and active on
282 of them.

**Sources.** Price and volume are constructed from hourly bars drawn from a public
bulk archive and resampled to weekly frequency. Intraday factor inputs are reduced
from 1-hour and 5-minute klines to one record per UTC day. Positioning data — open
interest, top-trader and all-account long/short ratios — comes from a keyless
exchange metrics endpoint at daily frequency. No proprietary or paid data is used;
every input is publicly reproducible.

**Why perpetual futures.** Two properties of this market are load-bearing.

First, **aggressor side is published.** Every bar carries taker-buy volume.
Equity microstructure research must *infer* trade direction, typically via the
tick rule (Lee and Ready 1991), and A-share tick data does not support reliable
inference at all. This distinction is what makes Section 4.3 possible.

Second, **funding is observable and material.** Perpetual futures pay a funding
rate several times daily. A price-only backtest of a long/short book in this
market overstates returns by approximately 3.9% annually. We subtract realised
funding at every rebalance rather than approximating it, and we tilt the book away
from positions it must pay to hold.

**A resolution constraint discovered by failure.** The trade-size distribution
factors cannot be computed at hourly resolution: a day divides into roughly twelve
bars per side, below the twenty-observation minimum the shape estimator requires.
Measured computability is 0% at 1-hour and 100% at 5-minute. The remaining
intraday factors, however, measured *better* at hourly resolution. The pipeline
therefore runs at two resolutions simultaneously — not as an optimisation, but as a
constraint the data imposed.

---

## 4. Factors

### 4.1 Design discipline

Every factor is specified *a priori* from its source, including its sign. Nothing
is fitted. This is worth stating plainly because it determines what can be
overfit: with no estimated parameters anywhere in the model, the entire
overfitting surface reduces to a handful of discrete hyperparameters, and Section
7.3 holds all of them out.

Each raw factor is smoothed with a twenty-period rolling mean and then ranked
under whichever frame the specification calls for.

### 4.2 The ten transported factors

Each is a reproduction of published equity research. Table 1 gives definitions and
sign rationale. Briefly: abnormal turnover (negative), smart-money VWAP ratio,
realised signed jump, order-flow imbalance, price-volume correlation in level and
in instability, three positioning factors capturing crowding among large accounts,
and the kurtosis of the log trade-size distribution.

One sign convention is inverted relative to its source. The smart-money factor is
documented as a reversal signal in A-shares; in perpetual futures it is
momentum-consistent. We report this as an empirical finding rather than a fitted
choice, and note it as a limitation: a single inverted sign, chosen with knowledge
of the data, is a channel through which information can leak.

### 4.3 A factor that equity data cannot produce

Research on Chinese A-shares established that the predictive content of per-bar
trade size lies in the *shape* of its distribution — quantiles, dispersion,
skewness, kurtosis — rather than its mean, with the reported interpretation that a
more concentrated, more right-skewed distribution precedes stronger returns.

That measure carries an unavoidable limitation. A-share tick data does not
reliably identify which side initiated a trade, so an unusually large buy and an
unusually large sell are indistinguishable within the distribution. The literature
can therefore observe *that* unusually sized orders are being placed but not *who
is placing them*.

Perpetual futures remove the limitation. With taker-buy volume published on every
bar, bars can be partitioned by initiating side and the distribution measured
separately within each partition. For each day, letting $v_n$, $n_n$ and $b_n$
denote volume, trade count and taker-buy volume in bar $n$:

$$\text{ticket}_n = v_n / n_n, \qquad
\mathcal{B} = \{n : b_n > \tfrac{1}{2}v_n\}, \qquad
\mathcal{S} = \{n : b_n \le \tfrac{1}{2}v_n\}$$

$$\text{TSKD}_t = \Delta\Big[\text{skew}\big(\log \text{ticket}_{n \in \mathcal{B}}\big)
- \text{skew}\big(\log \text{ticket}_{n \in \mathcal{S}}\big)\Big]$$

Two properties are worth drawing out.

**The change carries the signal, not the level.** Measured as a level, this
asymmetry is weakly informative; measured as a first difference it is materially
stronger. The interpretation is that what predicts is not who is currently
positioned but who has *just changed* — consistent with the direction-reversal
patterns documented in disclosure-based Chinese market research.

**It earns its place in the portfolio.** Measured by leave-one-out on a matched
282-week sample — removing the factor from the full book and asking what the book
loses — TSKD ranks **third of eleven**:

| factor | book without it | contribution |
|---|---|---|
| Q | 2.033 | +0.128 |
| TKU | 2.084 | +0.078 |
| **TSKD** | **2.115** | **+0.046** |
| WRspread | 2.123 | +0.038 |
| AVOL | 2.135 | +0.026 |
| OFI | 2.144 | +0.018 |
| TopChg | 2.150 | +0.011 |
| CPVv | 2.170 | −0.008 |
| RSJ | 2.179 | −0.017 |
| Quad | 2.180 | −0.019 |
| CPVm | 2.189 | −0.028 |

Standalone, under the same configuration the strategy deploys, TSKD returns
annualised **1.442 (t = 3.35)** over 281 weeks. The contribution is present in
**both halves** of the sample (+0.030 and +0.079), so it is structural rather
than a single-period artefact.

Two honest qualifications. First, a **quintile sort on TSKD is not monotonic**
(rank correlation between quintile index and forward return: +0.296); the traded
extremes carry the signal while the interior quintiles are noisy, and we report
this because the diagnostic is standard. Second, TSKD is **not** unusually
orthogonal — mean absolute correlation with the other ten books is 0.374 against
a factor-average of 0.301. It earns its contribution through predictive content,
not through diversification.

**Note also that four factors carry negative leave-one-out contributions.** The
book is marginally better without CPVm, Quad, RSJ and CPVv. We report the
eleven-factor specification because it was fixed a priori, but a referee is
entitled to ask why factors that subtract value are retained, and the honest
answer is that removing them post hoc would be a selection decision made with
knowledge of the outcome.

---

## 5. Construction

### 5.1 One book per factor

Standard practice forms a composite score $\bar{s}_{i,t} = \frac{1}{K}\sum_k
s^{(k)}_{i,t}$ and builds one portfolio from the ordering it induces. The
operation is lossy in a specific way: several signals that are close to
independent are compressed into a single ranking, and the independence is
discarded before any position is taken.

We instead form a dollar-neutral quintile book *per factor* — long the top 20%,
short the bottom 20%, weights $\pm 1/n$ — and combine the resulting **return
series**. Section 6.3 quantifies what this preserves.

### 5.2 Inverse-volatility combination

Book returns are combined by inverse trailing volatility,

$$w^{(k)}_t \;\propto\; 1 \big/ \hat\sigma^{(k)}_{t-1}, \qquad
\textstyle\sum_k w^{(k)}_t = 1$$

with the volatility estimate lagged so that no contemporaneous information enters
the weights. Where the risk model is not yet estimable the book is held flat
rather than traded equally weighted; trading an unweighted book during warm-up
measurably degrades full-sample performance.

### 5.3 Carry is a factor, not an adjustment

Each score is penalised by the contract's cross-sectional funding rank. Note that
this is the one place a cross-sectional comparison is appropriate: "which contract
costs more to hold than the others" is a genuine peer comparison, since funding is
denominated identically across contracts.

**This component is far more consequential than the word "tilt" suggests, and we
report it as a finding rather than as housekeeping.** Removing it from
single-factor books:

| factor | with carry | without carry | carry contributes |
|---|---|---|---|
| TSKD | 1.442 | 0.394 | +1.048 |
| TKU | 1.494 | 0.674 | +0.820 |
| OFI | 1.241 | 0.480 | +0.762 |
| AVOL | 1.222 | 0.663 | +0.559 |
| Quad | 1.896 | 1.356 | +0.540 |

Carry is a **first-order signal in perpetual futures**, contributing 0.5–1.0
Sharpe to every book. Traded alone on the matched sample it returns 1.250
(t = 2.91). Every single-factor figure elsewhere in this paper includes it, and
we state so explicitly rather than presenting those numbers as the ranking effect
in isolation.

Critically, **the central finding does not depend on it** — §6.1 reports the
ranking comparison both with carry and with carry entirely removed.

### 5.4 Turnover limitation as cost insurance

A uniform partial adjustment toward the target caps per-rebalance turnover. We
report a finding that contradicts our own earlier interpretation. At the modelled
maker fee the cap **costs** performance monotonically (Table 4). What it purchases
is robustness to cost misestimation: scaling the entire cost curve, the uncapped
book is superior only while realised costs remain near the assumption and
deteriorates sharply beyond roughly four times it, while the capped book remains
viable to eight times. Break-even is at approximately 4×.

The deployed configuration therefore uses a cap that is *not* the
Sharpe-maximising choice. We consider this the correct decision — the live
execution path takes market-order fallbacks when passive orders do not fill, so
realised costs sit above the maker assumption — but it is a deliberate divergence
and we state it rather than let it be discovered.

---

## 6. Results

### 6.1 The comparison frame

Table 2 reports eleven single-factor books under both ranking frames, holding
data, universe, costs and construction fixed. Because §5.3 establishes that carry
contributes 0.5–1.0 Sharpe to every book, we report the comparison **both as the
strategy trades it and with carry removed entirely**, so the ranking effect can be
seen in isolation.

| | cross-sectional | self-referential | gap | self wins |
|---|---|---|---|---|
| **as traded** (with carry) | 1.011 | **1.451** | **+0.440** | **9 of 11** |
| **carry removed** | 0.383 | **0.648** | **+0.265** | **8 of 11** |
| sign reversals | — | — | — | **0 of 11** |

**The ranking effect is independent of carry.** With the funding tilt fully
removed, self-referential ranking still wins in 8 of 11 factors with a gap of
+0.265. Carry amplifies the effect; it does not create it.

The improvement is systematic rather than driven by outliers, and no factor
changes sign under either specification. Cross-sectional ranking works in this
market; it simply works less well. We note that one factor (TopChg) reverses the
sign of its ranking preference when carry is removed; at this noise level
individual reversals should not be interpreted, and only the aggregate is
meaningful.

### 6.2 Mechanism

To identify *why*, we introduce an intermediate specification: standardise each
asset against its own trailing window, then rank cross-sectionally. If scale
heterogeneity drives the effect, this should recover it.

| specification | mean annualised Sharpe |
|---|---|
| cross-sectional, raw | 1.011 |
| cross-sectional, per-asset standardised | 1.192 |
| self-referential | 1.451 |

Standardisation recovers **41%** of the gap. Scale heterogeneity is therefore one
mechanism but not the dominant one. We attribute the residual to distributional
robustness: standardisation normalises only the first two moments, whereas a rank
against an asset's own history is invariant to the entire distributional shape.
Crypto factor distributions are severely non-normal and regime-dependent, and a
z-score's implicit normality assumption is correspondingly costly. Tests that
would separate the fat-tail component from the higher-moment component are left
to future work.

### 6.3 Construction

Both constructions were swept over an identical hyperparameter grid, so a tuned
method is compared against a tuned baseline (1,286 executed configurations).

**Paired** — identical hyperparameters, construction varied:

| | |
|---|---|
| matched cells | 17 |
| self-referential, per-factor books | 1.657 |
| conventional (cross-sectional, blended, equal-weight) | 0.992 |
| mean difference | **+0.665**\*\* |
| *t*-statistic of the difference | **5.82** |
| cells favouring the proposed construction | 15 of 17 |

Best-of-grid across all six construction variants places the proposed
construction first on both maximum (2.571) and mean (1.537), and the conventional
construction last (1.971 and 0.995).

### 6.4 Full strategy

| | |
|---|---|
| annualised Sharpe (deployed configuration) | 2.161 |
| *t*-statistic | 5.03\*\* |
| weekly rebalances | 282 |
| mean gross exposure | 1.011 |
| mean turnover | 0.369 |

---

## 7. Validation architecture

A single backtest number is weak evidence. We therefore report a structured set of
tests, each designed to break the result along a different axis.

### 7.1 Walk-forward with rebalance-anchor variation

Configuration is selected on a trailing 104-week window and evaluated on the
following 26 weeks, rolling forward. Because a weekly strategy also carries an
unmeasured exposure to *which weekday* it rebalances (Hoffstein, Sibears, and
Faber 2018), the entire panel is rebuilt from hourly bars on each of seven
weekday anchors and the procedure repeated.

| anchor | Mon | Sat | Sun | Tue | Wed | Thu | Fri |
|---|---|---|---|---|---|---|---|
| OOS Sharpe | 2.713 | 2.496 | 2.423 | 1.855 | 1.753 | 1.656 | 1.579 |
| *t* | 5.08 | 4.67 | 4.53 | 3.47 | 3.28 | 3.10 | 2.95 |

**Mean 2.068, all seven anchors profitable, 182 out-of-sample weeks per anchor.**
We report the mean rather than the Monday figure, and note explicitly that Monday
— the schedule actually traded — is the most favourable of the seven. Six of seven
anchors clear the *t* > 3.0 threshold proposed by Harvey, Liu, and Zhu (2016);
Friday, at 2.95, does not.

Results are additionally stable across fold geometry: seven train/test
configurations from 78/13 to 156/26 all produce profitable out-of-sample series
with *t* ranging from 3.70 to 5.55. The geometry used for the headline is
mid-range rather than maximal.

### 7.2 Held-out universe

Time-series out-of-sample testing does not establish that hyperparameters
generalise across *assets*. We therefore hold out symbols: configuration is
selected on 78 contracts and evaluated on 34 that had no influence on the choice,
across eight random splits, with the cross-section rebuilt within each side so
that ranks, quintile boundaries and the cost model are computed only among the
symbols present.

A naive comparison here is misleading, and we flag it because we initially made
it. A 34-contract book is mechanically inferior to a 78-contract one, since
information ratio scales with the square root of breadth (Grinold 1989).
Comparing them attributes a diversification effect to overfitting. We therefore
add a matched-breadth control: the same selected configuration evaluated on a
random 34 of the *training* contracts.

| comparison | ratio | interpretation |
|---|---|---|
| held-out / train (78) | 51% | conflates two effects — not interpretable |
| control (seen, 34) / train (78) | 62% | breadth, mechanical |
| **held-out / control** | **83%** | **overfitting, at matched breadth** |

All eight splits are profitable out of universe. More informative than the ratio
is the stability of the *selection itself*: all eight splits chose identical
funding tilt, turnover treatment and quintile width, and five of eight chose the
same ranking window. Fitted noise would not converge this way.

We note the limitation: only one of eight held-out splits reaches *t* > 3.0, a
consequence of reduced breadth rather than of the effect's absence, and the
held-out and control distributions overlap substantially. The defensible claim is
that retention is materially above 50%, not that it is precisely 83%.

### 7.3 Multiple-testing correction with a measured trial count

The deflated Sharpe ratio (Bailey and López de Prado 2014) requires the number of
trials and their dispersion. Both describe the *search* rather than the strategy
and cannot be recovered from the winning backtest, so in practice they are
supplied from the researcher's recollection — precisely the quantity the
correction exists to discipline.

We instead log every evaluated configuration to an append-only registry, written
by the code performing the evaluation, and read both inputs from it.

| trial set | *N* | DSR | |
|---|---|---|---|
| all logged configurations | 1,286 | 0.9475 | below threshold |
| strategy-search configurations | 211 | **0.9885** | passes |

We report both. The defensible choice is *N* = 211, since roughly 1,075 of the
logged cells constitute the construction ablation executed *after* the strategy
was specified, to measure the baseline fairly rather than to select among
candidates. But the conservative reading falls below 0.95 and we state it. Its
instability is itself the argument against it: the value fell from 0.9507 to
0.9475 purely because an ablation continued running. A statistic that penalises
conducting additional controlled experiments is not measuring selection bias.

### 7.4 Transaction-cost robustness

Costs are modelled as liquidity-scaled maker fees, from 1 basis point for the most
liquid quintile to 5 for the least, interpolated on cross-sectional dollar-volume
rank. Scaling the entire curve:

| cost multiple | 1× | 2× | 4× | 8× | 16× | 32× |
|---|---|---|---|---|---|---|
| uncapped | 2.364 | 2.247 | 2.014 | 1.547 | 0.618 | −1.222 |
| deployed (capped) | 2.161 | 2.117 | 2.028 | 1.851 | 1.496 | 0.789 |

The deployed configuration remains viable at eight times the modelled cost.

### 7.5 Research–production equivalence

The relationship between a published backtest and the system that trades it is
normally asserted. We test it. Research and live paths call a single shared
implementation, and an automated gate asserts point-in-time integrity,
determinism, dollar-neutrality, non-emptiness, and identical position directions.
The gate exits non-zero, so it can block deployment.

```
point-in-time     look-ahead leak 0.00e+00 on every factor
determinism       identical inputs → identical weights
neutrality        max |net| 9.02e-17
non-emptiness     308 rebalances built, active on 282
equivalence       0 direction mismatches / 4,480
```

The point-in-time assertion recomputes a historical score with future observations
deleted and requires exact reproduction. This test exists because it previously
*failed*: a ranking helper applied along the wrong axis ranked each observation
against the asset's entire history, future included, altering historical scores for
98 of 112 contracts. Research and live code agreed perfectly throughout, because
both called the same incorrect function.

---

## 8. What did not survive

### 8.1 An audit of our own claims

Before submission we searched the repository for the executable artifact behind
every headline figure. Four had none.

| claim | finding |
|---|---|
| walk-forward Sharpe "2.68–2.95" | tail-index statistics from a document about an unrelated strategy — not Sharpe ratios |
| timing-luck dispersion "±0.24" | no script, no output, no artifact |
| deflated Sharpe "0.9966 / 0.9816" | no artifact |
| conventional-construction baseline "1.03" | no artifact; approximately the conventional construction's *mean* (0.992), compared against the proposed construction's *maximum* |

A separate claim — that one factor reverses sign between ranking frames — was
tested directly and **failed**: zero of eleven factors reverse.

We report this for two reasons. First, the corrected figures are in this paper and
the discarded ones are not. Second, the episode motivates the infrastructure in
Sections 7.3 and 7.5: the reliability of a reported number is an empirical
property of the pipeline that produced it.

### 8.2 Negative results

**A falsified hypothesis of our own.** We conjectured that crypto factors are
structurally more orthogonal than equity factors. Measured pairwise correlation
among Alpha101-style books is **+0.2173** in this market against a reported 0.159
in equities. Factors here are *more* correlated, not less; the orthogonality that
makes the per-factor construction valuable derives from data-source diversity —
price, flow, positioning, trade-size distribution — not from the asset class.

**Rank-based significance without magnitude significance.** One candidate factor
produced a Spearman information coefficient of +0.0506 (*t* = 6.10), comfortably
above conventional thresholds, alongside a Pearson coefficient of only +0.0172 and
a decile table with no monotonic gradient. Returns here are severely fat-tailed;
a factor can order the typical asset correctly while being wrong about the few
observations that dominate realised profit.

**Nineteen rejected constructions.** These are enumerated in Appendix C. They are
logged rather than discarded because a search branch that failed is still a search
branch, and omitting it produces exactly the trial undercount Section 7.3 guards
against.

---

## 9. Practical implications

**Reconsider the comparison frame before reconsidering the signal.** The
improvement documented here required no new data and no new factor. It came from
changing what each observation is ranked against. Any market with heterogeneous
constituents and no fundamental anchor — crypto, commodities, and thinly-linked
alternative assets — is a candidate for the same treatment.

**Do not blend scores.** Compressing several signals into one ordering discards
independence that is expensive to acquire. Forming separate books and combining
their returns preserves it, and in this setting is worth more than any individual
factor in the model.

**Model funding explicitly.** In perpetual futures, funding accrual is a
first-order component of a long/short book's return. Price-only evaluation
overstates performance by approximately 3.9% annually.

**Choose the cost-robust configuration, not the Sharpe-maximising one.** Our
deployed parameters are deliberately not the best-performing ones in-sample.
Given that realised execution costs exceed modelled ones whenever passive orders
fail to fill, the configuration that survives an eight-fold cost error is worth
more than the one that maximises a backtest.

**Test the equivalence of research and production.** It is normally assumed. In
this project it twice proved false, and the failure was silent in both cases.

**Limitations.** Our evidence covers one venue's data over roughly seven years.
Out-of-sample testing is conducted in time and across the cross-section, but not
across venues. Capacity is not estimated in currency terms; the maker-fee
assumption will not survive at institutional scale in less liquid contracts, and
Section 7.4 should be read as bounding, not resolving, that concern. One factor's
sign convention was inverted with knowledge of the data. The held-out universe
result rests on eight splits and carries substantial uncertainty.

---

## 10. Conclusion

Factor investing imported into cryptocurrency perpetual futures inherits an
assumption that this market does not support: that its constituents are
comparable. We find the assumption is costly but not fatal — cross-sectional
ranking remains profitable, while ranking each contract against its own history
performs materially better in nine of eleven cases, for reasons we can partly
decompose. We further find that how factor signals are combined after ranking
matters more than any individual signal, and we exhibit a factor whose very
construction depends on information this market publishes and equity markets do
not.

What we would emphasise as much as any individual result is the structure of the
evidence: out-of-sample in time and across assets, replicated over seven rebalance
schedules and seven fold geometries, corrected for multiple testing using a trial
count read from an executed log, stress-tested to eight times modelled costs, and
verified to be computed identically by the deployed system. Each of these tests
was capable of overturning the result. We also report the four claims that did not
survive our own audit, because the credibility of a backtest is inseparable from
the reproducibility of the pipeline that produced it.

---

## References

Asness CS, Moskowitz TJ, Pedersen LH (2013) Value and momentum everywhere. *Journal of Finance* 68(3):929–985.

Bailey DH, López de Prado M (2014) The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management* 40(5):94–107.

Black F, Litterman R (1992) Global portfolio optimization. *Financial Analysts Journal* 48(5):28–43.

Daniel K, Titman S (1997) Evidence on the characteristics of cross sectional variation in stock returns. *Journal of Finance* 52(1):1–33.

Fama EF, French KR (1993) Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics* 33(1):3–56.

Grinold RC (1989) The fundamental law of active management. *Journal of Portfolio Management* 15(3):30–37.

Gu S, Kelly B, Xiu D (2020) Empirical asset pricing via machine learning. *Review of Financial Studies* 33(5):2223–2273.

Harvey CR, Liu Y, Zhu H (2016) …and the cross-section of expected returns. *Review of Financial Studies* 29(1):5–68.

Hoffstein C, Sibears J, Faber N (2018) Rebalance timing luck: The difference between hired and fired. *Journal of Index Investing*.

Kyle AS (1985) Continuous auctions and insider trading. *Econometrica* 53(6):1315–1335.

Lee CMC, Ready MJ (1991) Inferring trade direction from intraday data. *Journal of Finance* 46(2):733–746.

Ledoit O, Wolf M (2004) A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2):365–411.

Moskowitz TJ, Ooi YH, Pedersen LH (2012) Time series momentum. *Journal of Financial Economics* 104(2):228–250.

*[TO ADD: the Management Science crypto factor-pricing paper — closest prior work, must be cited and differentiated. See Section 11 checklist.]*

---

## 11. Pre-submission checklist

- [ ] **Cite and differentiate the MS crypto factor-pricing paper.** Closest prior work; builds a *cross-sectional* four-factor model. Our Section 6.1 is a direct empirical response.
- [ ] **Capacity estimate in currency terms** (Task #20).
- [ ] **AI disclosure**, on the PDF beside the abstract: *"AI tools were used for code implementation and manuscript preparation. All results were verified by re-execution."*
- [ ] **Data and code availability statement** — MS requires disclosure; nine committed scripts reproduce every number.
- [ ] Anonymise: no author name, affiliation, or acknowledgements; self-citation in third person; strip PDF metadata.
- [ ] Format: 11-point, 1.5 or double spacing, 1-inch margins, abstract ≤250 words.
- [ ] Tables 1–5 assembled (definitions; ranking frames; construction ablation; turnover/cost; validation summary).
- [ ] Appendix C: nineteen rejected constructions, enumerated.
- [ ] Appendix D: 100-word plain-language abstract for the FAJ version.
- [ ] Verify every number against `research/SRP_RESEARCH_LOG.md`; cite no figure labelled `[UNVERIFIED]`.

---

*All results are reproducible from the accompanying code. Numerical claims are
labelled and traceable to the named script that produced them in the research log.*
