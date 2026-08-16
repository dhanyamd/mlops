# SRP — Self-Referential Parity

## A complete research record: what was built, what was invented, what failed, and what survived audit

**Status of every number in this document is labelled.**
`[VERIFIED]` — reproduced by committed code, re-runnable today.
`[PENDING]` — computation in flight.
`[UNVERIFIED]` — quoted in earlier notes, **no artifact found in the repository**. Do not cite.

---

## 0. What this is in one paragraph

A market-neutral weekly long/short book over ~112 USDT-margined crypto perpetuals. Eleven factors, ten reproduced from published Chinese sell-side research and one invented here. The contribution is not the factors — anyone can read the same reports. It is the **construction**: how eleven weak, correlated signals are turned into a portfolio, and the engineering that proves the live system trades the same strategy the backtest tested.

---

## 1. The central research claim

> **The industry optimises the wrong thing.** Everyone hunts for better signals. We show that how you assemble signals matters more than which signals you pick — and we isolate the effect by holding the signals, the data, the universe and the cost model fixed while changing only the construction.

Three ideas underneath it:

1. **Rank each asset against its own past, not against its peers.** In equities, "Apple is expensive relative to Microsoft" is meaningful — shared economy, shared accounting, comparable businesses. In crypto it is not. Bitcoin and a small token share almost nothing. Yet the entire factor literature ranks cross-sectionally, because that is how equity factors work. Ranking each coin against its **own trailing history** inverts the sign of published factors in this market.

2. **Never blend scores. Trade each factor separately and combine the returns.** Standard practice averages *n* factor scores into one composite and builds one portfolio from it. That destroys exactly what made the factors independent — like averaging ten doctors' diagnoses into a single number before choosing treatment. We build eleven portfolios and combine what each earns.

3. **Prove the live system runs the backtested strategy.** Everyone asserts this. Almost nobody demonstrates it. We have a gate that compares live and research decisions and fails the *build* — not the P&L — if a single one disagrees.

---

## 2. Data infrastructure

| Layer | Source | Notes |
|---|---|---|
| Weekly closes / volumes / funding | Binance archive (`data.binance.vision`) + keyless REST, hourly bars resampled `W-MON` | 363-week panel, 112 symbols |
| Intraday features | 1h klines reduced to one record per UTC day | `cpv, q, ofi, rsj, rv` |
| Ticket-shape features | **5-minute** klines, same reducer | `tsd, tsk, tku, tskew_dir` — see §4 |
| Positioning | Binance futures metrics REST (`openInterestHist`, `topLongShortAccountRatio`, `globalLongShortAccountRatio`) | daily |
| Execution | Bybit demo, linear perps | maker-first with market fallback |
| Transport | Kafka/Redpanda → Redis (6380) → FastAPI (8000) → Next.js (3000) | launchd daemons |

**Why crypto perps at all.** Two data properties exist here and nowhere else, and both are load-bearing:

- **Taker side is published.** Every kline carries `taker_buy_volume`. Equity tick data requires the Lee-Ready tick rule to *infer* aggressor side, and A-share data does not reliably give it at all. This is what makes the invented factor (§4) possible.
- **Funding is observable and large.** Perpetuals pay funding three times a day. A price-only backtest overstates this book by roughly 3.9%/yr. Funding is subtracted explicitly at every rebalance, never approximated.

**A resolution finding that cost real time.** The ticket-shape factors *cannot* be computed at hourly resolution. At 1h a day splits into ~12 bars per side — below the 20-observation minimum the shape estimator requires. Measured: **0% computable at 1h, 100% at 5m**. But the other four intraday factors measured *better* at 1h (2.60 vs 2.37). So the pipeline deliberately runs **two resolutions**: 1h for `q/rsj/ofi/cpv`, 5m for the ticket factors. This is not an optimisation, it is a constraint discovered by failing.

---

## 3. The ten reproduced factors, in detail

All factors are defined **a priori from their source**, including sign. No sign was fitted. Every factor is smoothed with a 20-period rolling mean (`_sm`) unless noted, then ranked point-in-time (§5).

### 3.1 The daily intraday reduction

Each UTC day of intraday bars is reduced to a handful of scalars. Given a day's bars with close `cₙ`, volume `vₙ`, trade count `nₙ`, taker-buy volume `bₙ`, and bar returns `rₙ = cₙ/cₙ₋₁ − 1`:

**`cpv`** — Pearson correlation between close and volume across the day's bars.

```
cpv = Σ(cᵢ − c̄)(vᵢ − v̄) / [ √Σ(cᵢ − c̄)²  ·  √Σ(vᵢ − v̄)² ]
```

**`q`** — 聪明钱 (smart money), 方正/开源证券. Bars are scored by `S = |r| / √v`, sorted descending, and the top 20% *of volume* is designated "smart". Then:

```
q = VWAP(smart bars) / VWAP(all bars)
```

The logic: informed traders move price on relatively little volume. If their VWAP sits *below* the day's VWAP they accumulated cheaply — bullish. So **high q is bearish**.

**`ofi`** — 买卖压力失衡 (order-flow imbalance), 天风证券. Directly computable here because taker side is published:

```
ofi = (2·Σbᵢ − Σvᵢ) / Σvᵢ        ∈ [−1, +1]
```

**`rsj`** — realised signed jump, 国泰君安. Decomposes realised variance into upside and downside halves:

```
RV² = Σrᵢ²,   pos = Σ_{rᵢ>0} rᵢ²,   neg = Σ_{rᵢ<0} rᵢ²
rsj = (pos − neg) / RV²
```

### 3.2 The nine weekly factors

| # | Factor | Source | Definition | Sign rationale |
|---|---|---|---|---|
| 1 | **AVOL** | 异常换手率 | `−log( Σ₁₂ᵂ weekly volume )` | Abnormal turnover predicts negatively. Long the quietly-traded, short the frantically-traded. |
| 2 | **Q** | 方正/开源 聪明钱 | `_sm(q)` | **Sign inverted vs A-shares.** Crypto perps are momentum-driven where A-shares are reversal-driven. This inversion was a finding, not an assumption. |
| 3 | **RSJ** | 国泰君安 signed jump | `−_sm(rsj)` | Upside-jump-dominated names underperform. |
| 4 | **OFI** | 天风 买卖压力 | `_sm(ofi)` | Net taker buying persists at weekly horizon. |
| 5 | **CPVm** | 东吴 价量相关性 | `−rolling₂₀mean(cpv)` | High price-volume correlation = crowded; fades. |
| 6 | **CPVv** | 东吴, instability | `−rolling₂₀std(cpv)` | Unstable price-volume relation = unstable regime. |
| 7 | **WRspread** | positioning | `−( top_ls − all_ls )` | Gap between *large-account* and *all-account* long/short ratio = crowding proxy. **Contrarian.** |
| 8 | **TopChg** | 大户持仓变动 | `−diff( top_ls )` | Large accounts *increasing* long exposure is a fade signal. |
| 9 | **Quad** | CTA 四象限 | `−sign(weekly return) · diff(log OI)` | Price up + open interest up = new longs crowding in → fade. The four sign combinations are the "four quadrants". |

**10. TKU** — 开源证券, kurtosis of the log ticket-size distribution: `_sm(tku)`. Borrowed, not invented. Reported OOS contribution `[UNVERIFIED]` +0.12.

---

## 4. The invention: TSKD (Directional Ticket-Skew Change)

This is the one factor that is genuinely ours, and the reasoning matters more than the result.

### 4.1 What the source says

开源证券 studied the **per-bar ticket size** distribution — the average trade size within each minute bar — and found something counterintuitive: the alpha is not in the *mean* ticket size. It is in the **shape** of the distribution (分位数, 标准差, 偏度, 峰度), with a reported Rank ICIR of **3.57**. Their summary: *"分布越集中，整体右偏程度越高，股价未来表现越好"* — the more concentrated the distribution and the more right-skewed, the better the future return.

### 4.2 The gap in their measure

**Their measure is direction-blind, and it has to be.** A-share tick data does not reliably carry the aggressor side. A large *buy* ticket and a large *sell* ticket are indistinguishable to them. So they can say "unusual-sized orders are being placed" but not **who is placing them**.

That is a data limitation, not a modelling choice — and it is exactly the limitation a centralised crypto venue removes.

### 4.3 What we built

Every perpetual kline carries `taker_buy_volume`. So the same distribution can be **split by side**. For each day:

```
ticketᵢ    = vᵢ / nᵢ                                  per-bar average trade size
shape(x)   = (sd, skew, kurt) of log(x),  requires |x| ≥ 20

buy-dominated  bars:  bᵢ >  0.5·vᵢ
sell-dominated bars:  bᵢ ≤  0.5·vᵢ

tskew_dir  = skew( log buy-dominated tickets )
           − skew( log sell-dominated tickets )
```

Then, weekly:

```
TSKD = diff( _sm( tskew_dir ) )        <- the CHANGE, not the level
```

### 4.4 The two findings inside it

**Finding 1 — the level is nearly worthless; the change carries the signal.** Standalone gross Sharpe: level **0.25**, change **0.85**. This mirrors the 龙虎榜 **多转空** ("long flips to short") pattern in the Chinese literature: what predicts is not who is positioned, but who just *changed*.

**Finding 2 — it requires 5-minute bars.** At 1h resolution the per-side sample falls below the 20-ticket minimum and the factor is **0% computable**. This is why the pipeline carries two resolutions.

### 4.5 Honest sizing

**Re-measured 2026-08-16 against the committed backtest** `[VERIFIED]`. Incremental contribution, adding factors to the nine-factor book:

| factor set | ann Sharpe | t | Δ |
|---|---|---|---|
| 9 factors (no ticket) | 2.011 | 4.68 | — |
| 9 + TKU | 2.115 | 4.93 | **+0.105** |
| 9 + **TSKD** | 2.084 | 4.85 | **+0.073** |
| 11 (both) | 2.161 | 5.03 | +0.151 |

**TSKD's claimed +0.07 verifies at +0.073.** TKU's claimed +0.12 comes in at +0.105. *(The claimed "+0.27 paired with TKU" does NOT hold — both together add +0.151.)*

This is the one earlier claim that survived audit intact. TSKD is real, measured, and small. **Do not lead a paper with it; lead with the construction.** Its interest is conceptual — it is a factor that A-share researchers could not have built, because their data lacks the aggressor side.

Standalone single-factor book: **1.442** self-referentially, 1.240 cross-sectionally (§5.1). The earlier "gross 0.85 / net 0.76" figures remain `[UNVERIFIED]` and were measured under a different construction.

---

## 5. The construction — the actual contribution

### 5.1 Self-referential ranking

```python
ts_rank_pit(df, window=52, min_periods=26)
# each value ranked against the SAME SYMBOL's trailing 52 weeks
# score = ((rank − 0.5)/n − 0.5) · 2   ∈ [−1, +1]
```

versus the conventional

```python
xs_rank(df)     # each value ranked against OTHER SYMBOLS that week
```

**The claimed sign flip does not exist.** `[VERIFIED — measured 2026-08-16]` Earlier notes asserted AVOL scores −0.25 cross-sectionally and +0.80 self-referentially. Single-factor books, same data, same costs, only the ranking changed:

| factor | cross-sectional | self-referential | Δ |
|---|---|---|---|
| AVOL | 0.621 | 1.222 | +0.602 |
| Q | 0.728 | 0.572 | −0.156 |
| RSJ | 1.178 | 1.398 | +0.220 |
| OFI | 0.741 | 1.241 | +0.501 |
| CPVm | 1.004 | 0.938 | −0.066 |
| CPVv | 0.512 | 1.396 | +0.884 |
| WRspread | 0.797 | 1.994 | +1.197 |
| TopChg | 1.590 | 2.370 | +0.780 |
| Quad | 1.659 | 1.896 | +0.237 |
| TSKD | 1.240 | 1.442 | +0.202 |
| TKU | 1.050 | 1.494 | +0.445 |
| **mean** | **+1.011** | **+1.451** | **+0.440** |

```
self beats cross : 9/11 factors
SIGN FLIPS       : 0/11
```

AVOL cross-sectionally is **+0.621**, not −0.25. Cross-sectional ranking works perfectly well in crypto — it is simply **weaker**. The correct claim is a systematic improvement (9/11, +0.44 mean), **not** a paradigm failure. Any framing built on "the cross-sectional framework does not transfer to crypto" is unsupported by this data and must not be written.

*(Note: mean cross-sectional +1.011 is suspiciously close to the phantom "1.03 conventional baseline" of §9. Possibly its origin. Speculation, not evidence.)*

### 5.1b The funding tilt is a FACTOR, not an adjustment `[VERIFIED 2026-08-17]`

§5.4 describes the funding tilt as a penalty that stops the book holding positions
it must pay for. **That materially understates it.** Removing the tilt from
single-factor books:

| factor | tilt 0.5 | tilt 0.0 | the tilt is worth |
|---|---|---|---|
| TSKD | 1.442 | 0.394 | +1.048 |
| TKU | 1.494 | 0.674 | +0.820 |
| OFI | 1.241 | 0.480 | +0.762 |
| AVOL | 1.222 | 0.663 | +0.559 |
| Quad | 1.896 | 1.356 | +0.540 |

Carry is a **first-order signal** in perpetual futures, contributing 0.5–1.0
Sharpe to every book. Every single-factor number reported elsewhere in this log
includes it, and the paper must say so rather than presenting those as the
ranking effect alone.

**TSKD without the tilt is essentially zero** (−0.005 with neither tilt nor cap;
0.394 with the cap). The previously reported "standalone 1.442" was mostly carry.

**Critical test — does the cross-vs-self finding survive without the tilt?** Yes.

```
                    as traded (tilt 0.5)      tilt REMOVED
  mean cross-sectional      +1.011               +0.383
  mean self-referential     +1.451               +0.648
  gap                       +0.440               +0.265
  self-referential wins      9/11                 8/11
```

The ranking effect is **independent of carry**: 8/11 factors and a +0.265 gap with
the tilt fully removed. The tilt amplifies the effect but does not create it.
**Report both columns.** The first is what the strategy earns; the second isolates
what the ranking choice contributes.

> Note: TopChg reverses sign on the ranking choice when the tilt is removed
> (+0.780 → −0.513). Individual factor reversals should not be interpreted at
> this noise level; only the aggregate is meaningful.

### 5.1a Mechanism: why does self-referential win? `[VERIFIED]`

Hypothesis: cross-sectional ranking compares raw factor values across assets whose natural scales differ by orders of magnitude (BTC's order flow vs a small alt's), so part of what it ranks is **asset identity rather than signal**. Test: standardise each asset against its own trailing window *first*, then rank cross-sectionally (`ranking="cross_z"`).

```
mean cross    +1.011
mean cross_z  +1.192
mean self     +1.451

gap closed by standardisation alone: 41%
```

**Scale heterogeneity explains ~41% of the advantage. The remaining ~59% does not come from scale.**

The natural reading of the residual: z-scoring normalises only the **first two moments**. Ranking against one's own history normalises the **entire distribution**. Crypto factor distributions are severely non-normal and regime-shifting; a rank transform is invariant to distributional shape, a z-score assumes approximate normality. Further tests that would separate this (rank-vs-z on an identical window; median/MAD normalisation to isolate the fat-tail component) are **not yet run**.

> ⚠ The per-factor recovery percentages ranged from −87% to +456% because they divide by small individual gaps. **Only the aggregate 41% is defensible.** Do not publish the per-factor column without a substantially larger factor set.

### 5.2 One book per factor

Each factor independently forms a quintile long/short book — top 20% long at `+1/n`, bottom 20% short at `−1/n`, dollar-neutral by construction. Scores are **never** summed into a composite. The *returns* of the eleven books are what get combined.

### 5.3 Inverse-volatility risk parity

```python
iv = 1 / book_returns.rolling(52, min_periods=26).std().shift(1)
weights = iv / iv.sum(axis=1)
```

The `.shift(1)` is load-bearing — weights used at week *w* are estimated strictly from returns before *w*.

**`require_risk_parity=True`**: if the risk model is not yet warm, the book goes **FLAT** rather than trading equal-weighted. Measured: the equal-weight warm-up period scores −0.11 on its own and dragged the full sample from 2.23 to 1.90 `[UNVERIFIED]`.

### 5.4 Funding tilt

```python
score = score − 0.5 · xs_rank(funding)
```

Note this is the one place a **cross-sectional** rank is correct: "which coin costs more to hold than the others" is a genuine peer comparison.

### 5.5 Turnover cap — **a claim that was falsified today**

The cap moves only partway toward the target when turnover exceeds the budget:

```python
if turnover > cap:  target = prev + (target − prev)·(cap/turnover)
```

The docstring used to claim its value was "smoothing". **That is false.** `[VERIFIED]`:

| cap | None | 2.0 | 1.0 | 0.8 | **0.6** | 0.4 | 0.2 |
|---|---|---|---|---|---|---|---|
| ann Sharpe | 2.364 | 2.388 | 2.298 | 2.231 | **2.161** | 2.012 | 1.581 |

**The cap monotonically costs Sharpe.** What it actually buys is **robustness to cost misestimation** `[VERIFIED]`:

| cost multiple | 1× | 2× | 4× | 8× | 16× | 32× |
|---|---|---|---|---|---|---|
| uncapped | 2.364 | 2.247 | 2.014 | 1.547 | 0.618 | **−1.222** |
| cap 0.60 | 2.161 | 2.117 | **2.028** | **1.851** | 1.496 | 0.789 |

Break-even ≈ **4× the modelled maker fee**. The live executor falls back to *market* orders when maker doesn't fill, so realised cost plausibly sits in that zone — the capped book is the correct live default, **but for robustness, not for expected Sharpe.**

---

## 6. Methodological failures found and fixed

This section is the one a referee will respect most.

### 6.1 The look-ahead bug (the serious one)

`research_fas_clean._rank_z` was called as `df.apply(_rank_z)`. **pandas defaults to `axis=0`**, so each *column* — one symbol's entire time series — was passed in, and every week was ranked against the symbol's **whole history, including weeks that had not happened yet.**

Measured with `factor_core.leak_test`: deleting rows past 200 changed the week-150 score for **98 of 112 symbols**, max change **0.9884**.

It survived for months because research and live called the **same wrong helper**, so every consistency check passed. This is the origin of the parity gate (§7): agreement between two implementations proves nothing if both are wrong.

**Consequence:** what the project called "SMB" was never a size factor. Ranking volume against a symbol's own history is 异常换手率. The genuine cross-sectional size factor earns **−0.25** on this universe.

### 6.2 Uncentred rank score

`(rank()/n) − 0.5` with rank ∈ 1..n has mean `1/n`, not 0. Harmless for an additive rank-sorted book — a constant shift cannot reorder anything — but **not** harmless where scores are *multiplied*, and the old FAS factor was exactly `−pr_z · res_z`. Corrected to `(rank − 0.5)/n`.

### 6.3 NaN silently became a mid-rank

`_rank_z` ended with `.fillna(0.0)`. Zero is the **centre** of the score range, so a symbol with missing data received a median score and entered the book as a real candidate. It also made the function non-idempotent — several call sites ranked twice, converting filled zeros into genuine ranked positions. `factor_core` preserves NaN and the book construction excludes it.

### 6.4 Quote vs base volume

Klines index `[7]` (quote volume) was used where `[5]` (base volume) was meant.

### 6.5 The `dropna` intersection collapse

Per-factor book returns were joined with `dropna()` — requiring *all* factors to have a return in the same week. The intersection is set by the **shortest** factor. AVOL is shortest by construction (12-week volume sum, then a 52-week rank, consuming ~64 weeks). On the live registry's shorter history this collapsed 87 usable weeks to **20** — below the 26 the risk model needs — leaving the book **permanently FLAT**. Research never hit it because 363 weeks hid it. Fixed to `dropna(how="all")`; each factor's inverse-vol weight is estimated from its own column anyway.

**A second instance of the same defect, found and fixed 2026-08-16.** `scripts/srp_parity.py` still carried the intersection join after the backtest and live book had been corrected, so **the gate was certifying 220 rebalances while the traded book ran on 308** — it was validating a different sample than the one in production. Fixed; the gate now reports 308 built / 282 active, matching `srp_backtest.py` exactly, and all five assertions still pass.

The general lesson: fixing a defect in two of three call sites is not fixing it. This is precisely why the gate exists, and it is also why the gate itself needs auditing.

### 6.6 Live-path defects

| Defect | Symptom | Fix |
|---|---|---|
| `STREAM_ASYM_REBALANCE_H=24` | live rebalanced daily; SRP validated weekly | → 168 |
| REST/archive day offset | positioning one day stale | REST stamps at interval *end*; subtract one period |
| Partial-day bars | only source of live/research divergence | `require_complete` guard on bar counts |
| spot-vs-linear hardcode | `retCode=10001` for perp-only names; 1 of 6 symbols fetched | per-symbol category **discovery**, not a default |
| Rebalance latching on failure | one transient all-FLAT froze the book for a week | only latch a **non-FLAT** selection |
| Stale-bar rebalance | scored off a `window_end` 312 days old | staleness guard |
| Maker fill throughput | 30 polls × 0.5s × 76 symbols ≈ 19 min | budget made configurable → 5.1 min |

---

## 7. The harness gate

`scripts/srp_parity.py` asserts, and exits non-zero on failure `[VERIFIED]`:

```
universe 112 symbols, 363 weeks
  1. point-in-time    leak 0.00e+00 on all 9 factors
  2. determinism      identical inputs -> identical weights
  3. neutrality       max |net| 9.02e-17 ; gross bounded by 2.0 (max 1.462)
  4. no silent empty  308 rebalances built, active on 282 ; mean 98 non-FLAT
  5. research == live 0 mismatches / 4480
=== 0 FAIL ===
```

308 built / 282 active now matches `srp_backtest.py` exactly — see §6.5 for the join defect that previously made the gate evaluate 220.

Assertion 5 exists because a previous streaming reimplementation drifted to **0.147 rank correlation** against research with **27% selection overlap** — chance is ~20%. It was trading a different strategy while every dashboard said otherwise.

Assertion 4 exists because the `require_risk_parity` bug produced an all-FLAT book that *looked like a clean run*.

---

## 8. Results

### 8.1 In-sample `[VERIFIED]`

```
universe 112 symbols, 363-week panel, 282 active weekly rebalances
net of funding and liquidity-scaled maker costs (1–5bp on dollar-volume rank)

  turnover_cap 0.60 (shipped, live)   ann 2.161   t 5.03
  uncapped                            ann 2.364   t 5.51
  mean gross 1.011   mean turnover 0.369
```

### 8.2 Rebalance timing luck `[VERIFIED]`

Panel **rebuilt from hourly bars** on each of seven weekday anchors (Hoffstein, Sibears & Faber 2018) — not a shifted view of one grid:

| anchor | MON | TUE | WED | THU | FRI | SAT | SUN |
|---|---|---|---|---|---|---|---|
| ann Sharpe | **2.161** | 1.617 | 2.089 | 1.604 | 1.577 | 1.578 | 1.747 |
| t | 5.03 | 3.77 | 4.86 | 3.73 | 3.67 | 3.67 | 4.07 |

```
mean 1.768   sd 0.252   range 1.577–2.161   spread 0.584
7/7 anchors profitable, all t > 3.6
```

> **The honest headline is 1.768, not 2.161.** Monday — the anchor we trade and previously reported — is the **best of seven**. Reporting it alone reports the luckiest draw.

### 8.3 Walk-forward `[VERIFIED, Monday anchor only]`

Configuration chosen on a trailing 104-week window, scored on the next 26 weeks it had never seen:

```
OOS observations        182        (3.5 years fully out of sample)
OOS annualised Sharpe   2.713
OOS t-statistic         5.08
in-sample best (ref)    2.383
Walk-Forward Efficiency 114%       (ROBUST, >50%)
7/7 test blocks profitable (0.33 … 6.03)
```

**Caveats, stated plainly.** WFE > 100% is unusual — the walk-forward *adapts* its config every 26 weeks while the reference is one frozen config, and with only 7 folds this is noisy. The candidate set is deliberately small (12). And it is out-of-sample **in time, not in universe** — same 112 symbols throughout. **Critically, this ran on the Monday anchor, which §8.2 shows is the lucky one, so 2.713 is flattered by an unknown amount.**

### 8.4 Walk-forward across all seven anchors — **THE HEADLINE** `[VERIFIED]`

Out-of-sample *and* timing-luck-corrected simultaneously. Configuration selected on a trailing 104-week window, scored on the next 26 weeks, repeated on a panel rebuilt at each of the seven weekday anchors.

| anchor | MON | SAT | SUN | TUE | WED | THU | FRI |
|---|---|---|---|---|---|---|---|
| OOS Sharpe | 2.713 | 2.496 | 2.423 | 1.855 | 1.753 | 1.656 | 1.579 |
| OOS t | 5.08 | 4.67 | 4.53 | 3.47 | 3.28 | 3.10 | **2.95** |

```
mean 2.068   sd 0.462   min 1.579   max 2.713
profitable anchors : 7/7
WFE on the mean    : 103%
n_oos per anchor   : 182 weeks (3.5 years)
```

**Quote 2.068.** Six of seven anchors clear Harvey, Liu & Zhu's t > 3.0; **Friday at 2.95 does not, and that should be stated rather than rounded away.**

Note the OOS mean (2.068) *exceeds* the fixed-config in-sample anchor mean (1.768) of §8.2. This is not an anomaly: the walk-forward re-selects configuration every 26 weeks and repeatedly chose `turnover_cap=None` in later folds, whereas §8.2 is frozen at the shipped `cap=0.60`. Adaptivity is worth roughly +0.30, and WFE of 103% indicates the selection procedure is not overfitting.

### 8.4a Split sensitivity — is 2.068 an artefact of the fold geometry? `[VERIFIED]`

Monday anchor, identical candidate set, only train/test geometry varied:

| train | test | folds | n_oos | OOS ann | t |
|---|---|---|---|---|---|
| 78 | 13 | 16 | 208 | 2.257 | 4.51 |
| 78 | 26 | 8 | 208 | 1.850 | 3.70 |
| 104 | 13 | 14 | 182 | 2.968 | 5.55 |
| **104** | **26** | **7** | **182** | **2.713** | **5.08** |
| 104 | 52 | 3 | 156 | 2.859 | 4.95 |
| 130 | 26 | 6 | 156 | 3.007 | 5.21 |
| 156 | 26 | 5 | 130 | 2.985 | 4.72 |

```
mean 2.663   range 1.850–3.007   all seven t > 3.0 (min 3.70)
```

**Not an artefact.** Every geometry is profitable and clears Harvey. The geometry used for the headline (104/26) is **mid-range, not the maximum** — 130/26 scores higher at 3.007 — so the reported figure is if anything conservative. Longer training windows perform better, which is coherent (more data to select on) rather than noise.

*Caveat: this ran on the Monday anchor, which §8.2 shows is the best of seven, so absolute levels here are anchor-flattered. The headline remains the anchor-averaged **2.068**; what this establishes is stability.*

### 8.5 Deflated Sharpe Ratio — N and sd MEASURED, not asserted `[VERIFIED]`

Registry frozen at **1,286 executed configurations**. Every input below was produced by code (`scripts/trial_registry.py`); none was typed.

```
selected (best) trial   3840c1966bc338af
  rank_window 104, vol_window 26, funding_tilt 0.25, turnover_cap None, top 0.20, smooth 20
  T = 269 weekly observations,  ann 2.571,  t = 5.85
  skew +0.145,  kurtosis (non-excess) 4.391
  PSR(SR* = 0) = 1.0000
```

| trial set | N | sd (wk) | SR* (wk) | DSR | |
|---|---|---|---|---|---|
| ALL logged configs (most conservative) | 1286 | 0.0766 | 0.2547 | **0.9475** | **FAIL** |
| SRP construction only (the strategy search) | 211 | 0.0768 | 0.2139 | **0.9885** | PASS |
| SRP construction + full 11-factor set | 75 | 0.0625 | 0.1517 | 0.9994 | PASS |

**Both numbers must be reported.** The defensible choice is N = 211, because Bailey & López de Prado's *N* counts trials in the search **for the winning strategy**. Roughly 1,075 of the 1,286 cells are the six-construction ablation and factor-set arms, executed *after* the strategy existed and solely to measure the conventional baseline fairly (§8.6). They were never selection candidates; SRP's five construction choices were fixed a priori from the source literature.

The conservative reading is nonetheless **a fail at 0.9475, and hiding that would be indefensible.** Its instability is itself the argument against it: DSR fell from 0.9507 (N=1205) to 0.9475 (N=1286) purely because a background ablation kept running. A statistic that punishes running more controlled experiments is not measuring selection bias.

> **⚠ The DSR deflates the in-sample MAXIMUM (2.571), which is NOT the deployed configuration** (`52/52/0.5/0.60`, ann 2.161). Do not let the paper imply otherwise. The recommended structure: **walk-forward (§8.4) is the primary evidence** — it is immune to selection bias by construction and needs no deflation — with DSR as a secondary check on the in-sample maximum, and §5.5 explaining why the deployed config is deliberately *not* the best-scoring one.

### 8.5a Held-out UNIVERSE — out-of-sample in the cross-section `[VERIFIED]`

Every other test here is out-of-sample in TIME. This one holds out **symbols**:
hyperparameters are selected on 78 coins and the winning configuration is scored
on 34 coins that had no say in choosing it. Nothing is fitted anywhere in SRP —
factors and signs are a priori — so hyperparameter selection is the *entire*
overfitting surface, and this test covers all of it. `scripts/srp_holdout.py`,
8 random splits.

```
                  train78  held-out34    seen34
  mean              2.204       1.127     1.366
  median            2.169       1.194     1.407
  sd                0.212       0.286     0.281

  held-out splits profitable : 8/8
  held-out t > 3.0           : 1/8
```

**The matched-breadth control is essential and was initially missed.** A 34-symbol
book is mechanically worse than a 78-symbol one — IR ≈ IC·√breadth — so comparing
them attributes a diversification effect to overfitting. The `seen34` column
scores the same train-selected config on a random 34 of the **training** symbols:
same book size, coins that *were* seen. Decomposing:

| comparison | ratio | meaning |
|---|---|---|
| held-out34 / train78 | 51% | **naive — conflates both effects, do not cite** |
| seen34 / train78 | 62% | breadth alone, mechanical |
| **held-out34 / seen34** | **83%** | **overfitting, at matched breadth** |

**Configuration selection is highly stable**, which is the stronger result:

```
8/8 splits selected  funding_tilt=0.5, turnover_cap=None, top=0.20
5/8 selected rank_window=26,  3/8 selected 52
```

Eight independent 78-coin subsets converged on essentially the same settings. Had
these been fitted noise, the selections would scatter.

> **Caveats to state, not bury.** Only **1 of 8** held-out splits reaches t > 3.0
> — a power limitation of a 34-symbol book, not evidence of failure. The held-out
> and control distributions overlap heavily (sd 0.286 vs 0.281; split 3 scored
> *higher* held-out than seen). With 8 splits the defensible claim is "materially
> above 50%, consistent with modest overfitting", **not** a precise 83%.
>
> Note also that all 8 splits chose `turnover_cap=None`, while the deployed book
> runs `cap=0.60` for the cost-robustness reason in §5.5. That divergence is
> deliberate and must be stated explicitly rather than left for a referee to find.

### 8.6 The construction gap, tuned vs tuned `[VERIFIED — 1,191 executed trials]`

This replaces the unverifiable "1.03 vs 2.40". Both constructions were swept over the **same** hyperparameter grid, so a tuned strategy is compared against a tuned baseline.

**Paired** — identical hyperparameters, only the construction differs:

```
paired cells          17
SRP mean              1.657
conventional mean     0.992
mean gap             +0.665   (median +0.694)
sd of gap             0.471
t-stat of gap         5.82
SRP wins              15/17  (88%)
```

**Best-of-grid**, all six constructions:

| construction | n | best | median | mean |
|---|---|---|---|---|
| **self / books / riskparity (SRP)** | 197 | **2.571** | 1.622 | **1.537** |
| self / blend / equal | 213 | 2.468 | 1.317 | 1.288 |
| self / books / equal | 200 | 2.296 | 1.594 | 1.510 |
| cross / books / equal | 165 | 2.200 | 1.175 | 1.152 |
| cross / books / riskparity | 202 | 2.113 | 1.233 | 1.129 |
| **cross / blend / equal (conventional)** | 214 | 1.971 | 1.066 | 0.995 |

**The defensible claim is a construction gap of +0.60 to +0.67 Sharpe (paired t = 5.82), not the +1.37 the old figure implied.** SRP ranks first on both best-of-grid and mean.

**Where "1.03" came from.** The conventional construction's *mean* is **0.992**. The discarded claim compared that mean against the SRP *best* (2.40, uncapped) — mean versus best, which is exactly the untuned-baseline error §9 suspected. Both numbers were roughly real; the comparison was not.

---

## 9. Claims that did NOT survive audit

An audit on 2026-08-16 searched the repository for the artifact behind every headline number. Four had none:

| Claim | What was actually found |
|---|---|
| "walk-forward 2.68–2.95" | **Hill tail-index values** from `research/TAIL_AWARE_XS_MOMENTUM.md` — a different strategy, whose Sharpes are 0.54–0.84. Not Sharpes at all. |
| "timing luck ±0.24 across 7 anchors, all profitable 1.75–2.53" | No script, no output, no file. *(Measured sd today: 0.252 — close. Measured range: 1.577–2.161 — not close.)* |
| "DSR PASS N=50 (0.9966), N=400 (0.9816)" | No file contains these numbers. |
| "conventional pipeline 1.03 vs ours 2.40" | 2.40 **is** reproducible — it is the *uncapped* run (2.364). **1.03 has no script behind it** and appears to be an untuned baseline compared against a tuned strategy. |

**The lesson, and it is the paper's methodology section:** the hardcoded DSR trial count was not an isolated defect. It was a symptom. Numbers were being written into docstrings as established fact with no executable artifact. The fix is structural — `scripts/trial_registry.py` logs every evaluated configuration (config hash, Sharpe, observation count, git SHA), and DSR reads N and sd from that log. **No human types either number.**

---

## 10. Failed inventions (19)

Every one was built, measured, and rejected. Kept because a search branch that died is still a search branch — omitting them is exactly the undercount the Deflated Sharpe Ratio exists to catch.

**Signal inventions:** FAS (funding-accrual residual — measured IC +0.0065, indistinguishable from zero) · RCGO₿ (crypto capital-gains overhang — degraded every configuration tested) · Salience/ST (Cosemans & Frehen: Spearman IC +0.0506 at t=6.10 but **Pearson only +0.0172** and **no decile monotonicity** — ranking noise, not earning money) · SAT · **IFD** (Informed Flow Divergence — the smart-money bar selection made sign-aware) · **Kyle Asymmetry** (price impact per unit *buy* volume vs per unit *sell* volume, only formable with signed flow) · session-RPV (东吴 RPV's overnight/intraday split remapped to Asia/Europe/US sessions, since crypto has no overnight gap) · LOP · WCC · FSD · TCN−ACN · forward funding cost.

**Construction inventions:** **LSN** (latent sector neutralisation — industry-dummy neutralisation without industry labels, using principal components of the trailing return correlation matrix, with *k* set by the **Marchenko-Pastur noise edge** `λ_max = (1+√(N/T))²` rather than chosen) · **priority turnover** (spend the turnover budget by conviction — fund the largest signal changes in full rather than scaling every trade equally) · cost-aware turnover · clientele conditioning · leg specialisation · MOD-correction transfer · orthogonalisation transfer.

**The most instructive failure:** both turnover-allocation inventions *destroyed* the cap's benefit. That is what revealed the cap is a uniform-adjustment property, not a cost-optimisation one — and eventually (§5.5) that it isn't a benefit at all, but insurance.

---

## 11. Negative results worth publishing

**The orthogonality thesis was falsified by our own data.** The hypothesis: crypto factors are structurally more orthogonal than equity factors, so combining them should earn more. Measured Alpha101 book correlation in crypto: **+0.2173**, versus WorldQuant's reported **0.159** in equities. Crypto factors are *more* correlated, not less. Orthogonality comes from **data-source diversity** — price, flow, positioning, ticket shape — not from the asset class.

**Spearman IC without Pearson IC is a trap.** Salience ranked the typical coin correctly (Spearman +0.0506, t=6.10 — above the |IC|>0.05 bar the Chinese literature uses) while being wrong about the handful of moves that dominate a fat-tailed book (Pearson +0.0172). **P&L is paid in magnitudes, not ranks.**

**Decile monotonicity is a necessary check.** Salience produced a decile table with no slope — decile 4 out-returned decile 10. A real cross-sectional factor has a gradient.

---

## 12. Open items before submission

1. ~~Reconcile the `dropna` inconsistency between the gate and the backtest.~~ **DONE 2026-08-16 — gate now reports 308/282, matching the backtest; 0 FAIL. See §6.5.**
2. **Anchor-averaged walk-forward** — the publishable OOS figure. `[PENDING]`
3. **DSR from the executed registry.** `[PENDING]`
4. **Tuned-vs-tuned baseline gap** — replaces the unverifiable 1.03. `[PENDING]`
5. **Re-measure TSKD's contribution** against the committed backtest. Currently `[UNVERIFIED]`.
6. ~~Re-measure the cross-sectional vs self-referential sign flip.~~ **DONE 2026-08-16 — no flip exists (0/11). Replaced with the systematic-improvement claim and a partial mechanism, §5.1/§5.1a.** Follow-up still open: separate the distributional component from the scale component (rank-vs-z on an identical window; median/MAD normalisation).
7. ~~Held-out universe test.~~ **DONE 2026-08-16 — 8/8 splits profitable, 83% retention at matched breadth, configuration selection stable across all 8 splits. See §8.5a.**
8. **Capacity analysis.** The 1–5bp maker assumption breaks at size on illiquid alts. §5.5 shows results survive 4× and the capped book survives 8×, which is a strong answer — but a dollar capacity estimate should be stated, not implied.

---

## 13. Reproducibility

```
scripts/factor_core.py       corrected primitives + leak_test
scripts/srp_strategy.py      the strategy; shared by research AND live
scripts/srp_backtest.py      committed evaluator (was an uncommitted scratch file)
scripts/srp_parity.py        the gate — exits non-zero, fails the build
scripts/srp_sweep.py         executes the search space, logs every cell
scripts/trial_registry.py    append-only JSONL: config hash, Sharpe, n, git SHA
scripts/srp_dsr.py           DSR with N and sd READ from the registry
scripts/srp_ablation.py      paired / best-of-grid / main-effects decomposition
scripts/srp_walkforward.py   walk-forward + 7-anchor timing luck
stream/srp_live.py           live book — data plumbing only, no strategy logic
```

The rule that makes this hold: **`srp_strategy.py` is the single source of truth.** Both the backtest and the live signal call `srp_weights`; neither reimplements it. `srp_parity.py` asserts they agree. If they ever drift, the test fails — not the P&L.

---

*Record compiled 2026-08-16. Every `[VERIFIED]` number in this document was reproduced on that date by the scripts named above.*
