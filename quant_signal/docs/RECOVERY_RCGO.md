# FAS + SMB + RCGO₿ — Provenance, Results, and Live Status

> Working notes so the thread state survives.
> **Updated 2026-08-15** with the Grinblatt-Han CGO fix (1.93 → 2.28), the
> permutation test (p = 0.002), the research/live parity refactor, and an
> honest publication-readiness assessment.

---

## 0. PROVENANCE — what is OURS vs what is BORROWED

This section exists because the distinction matters for any writeup. Claiming a
standard primitive as an invention destroys credibility; failing to claim a real
one gives away the contribution.

### OUR INVENTIONS (no paper does these)

**FAS — Funding-Accrual Squeeze.** The core signal.

    FAS_h[s,w] = -z(pr_h[s,w]) * z( resid_w[s] )
    resid_w    = residual of a CROSS-SECTIONAL OLS of week-w funding accrual on
                 [1, pr_h, |pr_h|]
    FAS_avg    = mean over horizons h in {4, 8, 12, 26} weeks

*Reasoning chain that produced it (not copied):*
1. Funding accrual is mechanically correlated with the price path — coins that
   ran up carry crowded longs paying funding. Raw funding is therefore
   contaminated by momentum and cannot be a clean signal.
2. Regressing accrual on BOTH `pr_h` and `|pr_h|` strips the directional AND the
   magnitude component, leaving accrual that is *unexplained* by the price path
   — crowding that the move does not justify.
3. Interacting that residual with `-z(pr_h)` targets the squeeze: unexplained
   crowding AGAINST the recent move is where forced unwinds happen.
4. Averaging over four horizons is a self-ensemble — no single lookback is
   privileged, so the signal is not a lookback-tuning artefact.

*Intellectual ancestry (inspiration, not derivation):* Bianchi et al. orthogonalize
ORDER FLOW on returns and trade the residual. FAS is the derivatives analog —
same orthogonalization logic applied to funding accrual, an input no equity
market has. The construction, the `|pr_h|` term, the sign convention and the
multi-horizon ensemble are ours.

**RCGO₿ — Residual Capital-Gains-Overhang on crypto funding carry.**

    RCGO_b[s,w] = z( CGO_daily[s,w] - beta_w * CARRY_xs_z[s,w] )
    CARRY_xs_z  = cross-sectional z of weekly funding accrual (Crypto Carry)
    beta_w      = per-week cross-sectional OLS of CGO on carry

*Reasoning chain:*
1. CGO measures behavioural overhang (unrealised gains held → disposition-driven
   selling pressure). In crypto it is contaminated by *mechanical* crowding:
   coins with big unrealised gains also carry crowded funding.
2. Residualizing CGO on funding carry separates the behavioural component from
   the mechanical one. What is left is overhang NET of crowding.
3. Blended CONTINUOUSLY, not as a hard quantile gate — CGO is a continuous
   factor and a q-filter throws away the gradient.
4. `dir = +1` (HIGH residual overhang wins) is **data-driven and crypto-specific**.
   It is the OPPOSITE of the A-share result (low-CGO wins). We did not import
   the equity prior; the sign was measured. `dir = -1` collapses to Sharpe 0.27,
   which confirms the direction is real and not a coin flip.

**No published work combines capital-gains-overhang with perpetual-futures
funding carry.** That combination is the novel contribution, and it is what takes
the book from 1.64 → 2.28.

### BORROWED / STANDARD (cite, never claim)

| component | source | role |
|---|---|---|
| **SMB** (size) | Fama-French | Standard size tilt, `-z(log 12w volume)`. NOT ours. Included because it is the one known factor that *adds* to FAS (REV/MOM hurt it). |
| **CGO + reference price** | Grinblatt & Han (2005) | The overhang primitive. Standard. |
| **GH survival-weighted RP** | 广发证券 "资本利得突出量CGO与风险偏好" + 2024 multi-frequency follow-up | `RP_t = (1/k)Σ[V_{t-n}·Π(1-V)]·P_{t-n}`. A correct implementation of a known formula — **not** an invention. |
| **Crypto Carry** | SSRN 3774118 | Funding/carry predicts XS crypto returns. Used as the control in RCGO₿. |
| **Daily-only CGO in crypto** | Liu/Fang/Wang, 管理评论 2024, 36(6):94-106 | "仅存在两周的动量效应"; weekly CGO data INVALID. Drove both the daily-CGO choice and the holding-horizon fix. |
| Cost band (λ·c entry, 2λ·c flip) | Bysik & Ślepaczuk 2026 eq.5; Novy-Marx & Velikov 2016; Gârleanu & Pedersen 2013 | Execution banding / no-trade region. |
| Vol scaling | Barroso & Santa-Clara | Exposure scaler, bounded 0.25×–2.0×. |
| Permutation testing | Monte-Carlo permutation literature | Significance methodology. |
| t > 3.0 bar | Harvey et al. (2016) | Publication standard for new factors. |

**Honest framing for a paper:** *"We apply the Grinblatt-Han reference price
(2005), following the multi-frequency construction of 广发证券 (2024), as an input
to a novel funding-carry-orthogonalized overhang factor (RCGO₿), combined with an
original funding-accrual-squeeze signal (FAS)."* Citing the primitives strengthens
the claim — it shows the novelty is in the construction.

---

## 1. RESULTS (reproduced 2026-08-15)

    QUANT_CGO_GH=1 uv run python -m scripts.research_fas_invent --funding binance

| Book | ann_ret | vol | **Sharpe** | maxDD | wealth |
|---|---|---|---|---|---|
| Baseline FAS+SMB + hard CGO filter (dir=+1) | 73.55% | 44.7% | **1.64** | -17.6% | 1.97× |
| **INVENTION RCGO₿ blend w=0.50 dir=+1 (GH CGO)** | **101.45%** | 44.4% | **2.28** | **-15.8%** | **2.636×** |

Return UP and drawdown DOWN — the gain is not leverage.

Robustness: `dir=-1` → 0.27 (direction confirmed). `w ≥ 0.25` all land in the same
region (not a knife-edge weight). GH beats the simplified CGO at 6 of 7 lookbacks.

### Significance — `scripts/perm_test_research.py`

    observed Sharpe            2.283
    null (500 permutations)    mean -0.896   sd 0.737   max 1.236   p95 0.279
    p-value                    0.0020        SIGNIFICANT at 1%

Signal-permutation test: returns kept in original order, WHICH symbols get
selected is permuted. **0 of 500 random selections beat the strategy.** Observed
sits >4σ above the null mean.

*Methodological note:* an earlier null comparing **total P&L** showed nothing
(p=0.11) because crypto P&L is dominated by whichever coin 10×'d — null sd was
$15,579 against a $1,507 result. The **risk-adjusted (Sharpe)** statistic is the
valid one. Do not repeat the P&L version.

---

## 2. THE BIG BUG — research/live divergence (fixed 2026-08-15)

The strategy was **implemented twice**: `research_fas_invent.py` (vectorised,
validated) and a ~900-line streaming reimplementation in `stream/asym_signal.py`.
They had drifted until they agreed almost not at all.

    score rank-correlation research vs live:  0.147   (chance ≈ 0)
    selection overlap:                        27%     (chance ≈ 20%)

Six separate root-cause hypotheses were tested and **all were wrong** (epoch-vs-
Monday week grids, funding-history gate, close-span truncation, hold caps,
liquidity-mask look-ahead, signal lag). The fix was not finding every difference —
it was **deleting the second implementation**:

`AsymSignal._research_scores()` now CALLS `fas_scores()`, `smb_scores()`,
`rcgo_scores()` directly, on frames rebuilt from the streaming registry with the
SAME pandas resample calls. Frames verified byte-identical (0 of 3472 cells
differ). Result: **correlation 0.147 → 0.9927**.

This is the failure mode the backtest-parity literature names: *"reimplementation
is where subtle bugs creep in causing live performance to diverge from backtested
expectations."* Parity is now structural, not chased. `QUANT_RESEARCH_PARITY=0`
restores the legacy path for A/B.

**Verification that the executor is sound** (`scripts/ops/harness_verify.py`):
research selections pushed through the LIVE execution engine reproduce
**Sharpe 2.11, +$5,863 realized on $12,000** over 53 weeks. So the executor and
the metric are fine; only the signal had diverged.

---

## 3. EXECUTION BUGS FIXED (2026-08-14/15)

| bug | effect |
|---|---|
| Trailing stop read `position["entry"]` — a key that never exists | activation guard dead; stop fired from bar 1 on any retracement, including on losers |
| `QUANT_TRAIL_OFF=1` set `alpha=0`, and `drawdown >= 0` is always true | the "disable" switch was the TIGHTEST possible stop (2800 trades, all 1h holds) |
| RCGO weight applied as `W × rank_z(resid)` | research does `rank_z(W × rcgo)` where W CANCELS — live ran the invention at half strength |
| Positions outlived their forecast (7.7 weeks avg) | Liu/Fang/Wang: crypto momentum decays in 2 weeks — trading a dead signal |
| Fill confirmation treated "position still open" as a filled EXIT | failed closes recorded as completed round trips with fabricated P&L (OPUSDT booked +$22.60; real was +$0.12) |
| Unrounded qty on maker close | Bybit rejected with "Qty invalid" (ErrCode 10001) |
| Exact window-stamp matching for predictions | predictor/engine offset drift starved 7 of 12 symbols — book ran at 17% of intended size |
| Close-and-reopen at forecast expiry | paid a full round trip to re-establish an IDENTICAL position; roll-don't-churn nearly doubled weekly Sharpe |
| Replay wrote historical fills to the live durable ledger | 159 backtest fills contaminated 21 genuine live ones; `durable_log=False` guard added |
| Broken weekly-return formula in the replay | divided weekly ΔP&L by CUMULATIVE P&L, not capital — every replay Sharpe quoted before this was meaningless |

---

## 4. HOLDING HORIZON — measured, not assumed

Retested on the CORRECTED signal (earlier "daily = -3.14" came from the broken one):

| rebalance | trades | win rate | realized | weekly Sharpe |
|---|---|---|---|---|
| 24h | 1961 | 46.5% | +$797 | 0.87 |
| 48h | 1221 | 46.8% | -$341 | -0.16 |
| 72h | 880 | 48.3% | +$499 | 0.66 |
| **168h** | **411** | **48.2%** | **+$1,507** | **1.44** |

Weekly wins. Daily is still POSITIVE (0.87) — not the disaster the old notes
claimed — but roughly half the money for ~5× the turnover.

Fees are NOT the binding constraint: at ZERO cost the live path reaches 1.50, vs
2.28 for the research portfolio. The residual gap is the difference between an
idealised portfolio return series and discrete order-level execution.

**Minutes/seconds is impossible for THIS strategy** — funding settles every 1-8h,
so a minute-level funding signal does not exist to be computed. Hyperliquid/dYdX
settle hourly (8× finer than Binance) and remain UNTESTED; the note claiming
"Hyperliquid breaks FAS (-0.37)" cites `scripts/research_hl_fas.py`, **which does
not exist in this repo** — treat that claim as unverified.

---

## 5. PUBLICATION READINESS — honest assessment

Harvey et al. (2016): new factors need **t > 3.0**, not 2.0, given field-wide
data mining.

    Sharpe 2.283, 55 weekly observations  ->  t = 2.35    BELOW the bar
    weeks needed for t > 3.0 at this Sharpe: 90

| requirement | status |
|---|---|
| t-stat > 3.0 | ❌ 2.35 — **need more data, this is the binding constraint** |
| Permutation significance | ✅ p = 0.002 |
| Walk-forward validation | ❌ not done |
| Deflated Sharpe (multiple-testing correction) | ❌ not done |
| Multiple regimes (bull/bear/chop) | ❌ one recent stretch only |
| Novel contribution | ✅ FAS + RCGO₿ |
| Live implementation, parity-verified | ✅ |

**Cache holds 55 weeks of funding; Binance has ~6 years.** Backfilling is the
single highest-value next step — it fixes the t-stat, the out-of-sample gap and
the regime coverage simultaneously. `scripts/backfill_binance_history.py` exists.

Expect shrinkage: McLean & Pontiff (2016) find published predictors decline 26%
out-of-sample and 58% post-publication.

---

## 6. LIVE STATUS

Deployed on Bybit demo (virtual USDT), 12 positions (6L/6S), corrected signal.
Genuine live ledger: 14 strategy closes, net -$24.06 — **all from the pre-fix
churn window (1h holds)**, not a verdict on the strategy. First closes on the
corrected config land ~2026-08-21.

Live config: SMB ON, RCGO₿ w=1.0 dir=+1 ortho, GH CGO ON, FACC/REGIME/FCARRY/
TSMOM/REV OFF, trailing stop OFF, vol-scaling ON, research-parity ON.

**The BTC regime gate is OFF** — the only downside-protection leg. The backtest
period contains no sustained bear market, so its absence is untested against a
crash. Consider before any real capital.

---

## 7. COMMANDS

    source .venv/bin/activate
    QUANT_CGO_GH=1 python -m scripts.research_fas_invent --funding binance   # 2.28
    python -m scripts.perm_test_research --n 500                             # p-value
    python -m scripts.backtests.selection_diff --weeks 8                               # live↔research
    python -m scripts.ops.harness_verify                                         # executor sanity
    python -m scripts.backtests.parity_ablation                                        # cost attribution
    python -m scripts.probes.research_cgo_gh --lookbacks 5,7,14                     # GH vs simplified
    python -m scripts.ops.live_track_record                                      # live ledger
    python scripts/ops/replay_live_book.py --rebalance-h 168 --rcgo-w 1.0 --rcgo-dir 1 --cgo-dir 1
