# Continuation Prompt — Crypto Cross-Sectional Quant Research (`quant_signal`)

> Paste this into a fresh session to resume. It is self-contained: project state, literature map
> (Chinese + global), what is proven/falsified, and the open research agenda.

---

## 0. Who we are / what this repo is

We are building **`quant_signal`** — a production-grade, real-data quant platform whose
research arm hunts **cross-sectional (XS) crypto factors** by the discipline:
*read the literature → find the gap → encode ONE defensible economic mechanism →
validate walk-forward (WF) out-of-sample with transaction costs and crash regimes →
promote the survivor into the live stack.*

Repo root: `/Users/dhanyamd/Projects/mlops`. Active package: `quant_signal/`.
Hard rule repeated in every research script: **no hardcoded magic numbers** — parameters live
in `CONFIG`/`VARIANTS`; thresholds (k, stress quantile, fragility cap) are chosen **walk-forward**,
applied OOS. Transaction costs are a config knob. Honesty/limitations are mandatory in every write-up.

### Architecture (do not reinvent — extend)
```
providers (yahoo/binance/fred/sec_edgar)  [keyless except Alpaca]
   → contract gate (pydantic) → BRONZE (Snowflake MERGE upsert)
   → dbt SILVER (typed/contract) → GOLD (gold_daily_bars, gold_1h, etc.)
   → research harness (walk-forward XS backtests)  +  live stack (Kafka→Redis→ClickHouse→
     Grafana, Prefect, MLflow, Bybit Demo execution venue)
```
- Real keyless sources verified live; **no synthetic data in production**.
- Data lake tier (Iceberg→MinIO/S3) + scale-aware coarse SYMBOL partitioning added recently.
- Live execution is on **Bybit Demo** (honest fallback: poll fills from order history, ~8s lag,
  3s HTTP timeout, no retries, bounded jitter). `xs_signal` / `lcf_signal` are the streaming signals.
- Deployed production XS signal is **`xs_rel14`** = BSC vol-scaled full L/S 14-day momentum.
  We have strong evidence it is the *weakest* variant (see §3) and should be replaced by SCX.

---

## 1. RESEARCH DEEP DIVE — literature map (Chinese + global, 2024-2026)

### 1a. Chinese-language sources (practitioner + academic) — HIGH VALUE, under-used in Western academia
1. **Liu, Fang & Wang (刘帅/房勇/汪寿阳), 管理评论 36(6) 2024** — *基于处置效应及动量效应的加密货币市场投资策略*
   (PDF+TXT already in `quant_signal/docs/`). Authoritative (CAS Academy of Math & Systems Science).
   - Crypto momentum lasts **≤ 2 weeks**; disposition effect (capital-gains-overhang, CGO) ≤ 1 month.
   - **Weekly data is invalid** for crypto; must use daily+ frequency.
   - CGO (η_t = (P_t − R_t)/P_t, reference price R_t via turnover-weighted Grinblatt–Han 2005) **+ momentum
     beats either alone**; best combo is **G/r** (filter by CGO first, then momentum). Long-only only (no shorting in China).
   - 7-day CGO is the significant window. This is the licence for our short-cycle + disposition-overhang factors.
2. **NCCU thesis (政大)** — *加密貨幣永續合約資金費率與美股指數期貨關聯性分析* (2023-10 → 2025-12).
   Funding rate is right-skewed, fat-tailed; **weak full-sample linear link to US equity futures, but
   strong ASYMMETRIC tail linkage** (regime-dependent). Treat funding rate as a *state/leverage-risk indicator*, not a stable predictor.
3. **juejin.cn 掘金 (2025-03)** — ML funding-rate prediction (BTCUSDT). Features: prev_funding, price_diff,
   funding_ma3, hour, dow. **R² 0.613, direction acc 76.4%, MSE 1.87e-10**. Linear model captures direction;
   ~1-2 period lag at turning points. Confirms funding is *persistent + mean-reverting* → tradable.
4. **heth.ink (2024-12)** — 数字货币永续合约资金费率套利策略. **Coin-margined (反向) perp arb**: use spot as
   margin for the short → 100% capital efficiency, no liquidation at 1x. Long-run funding ~15% APR (bull 30-50%).
   Adds **staking yield** (SOL ~11%) as extra carry. RL-style reward = discounted Σ future funding − cost.
   → Our `research_fund.py` (LCF) should borrow the *coin-margined* + *staking* economics, not just USD-margined.
5. **winterresearch.com/fundskew** — BTC funding APR × 25Δ option skew 共振/分歧 (2.7y). Both z-scored on 3y sample.
   - 9 long-resonances → 60d all up; 7 short-resonances → 6/7 up in 30d.
   - Real **divergence** (funding ↗ / skew ↘) = top warning: 30d median −4.0%, down 75%.
   - Sample too short (n=2-10/class), BTC-only (ETH partial-day noise 26%). **Action:** port to our panel as a 3rd dimension for LCF gating.
6. **FMZ (fmz.com) / Lucida / Falcon (Chinese quant funds)** — practitioner lore: **long COLD/low-volume coins,
   short HOT/high-volume coins** ("热门币更倾向于下跌"). Liquidity/attention factor is long-effective. This is the
   seed for **AWR** (Attention-Weighted Reversal) — but the journals missed it because raw volume is size-confounded.

### 1b. Global academic / practitioner
7. **Crypto Carry — SSRN 3774118 (Christin et al.)**: perpetual **funding rate positively predicts returns in the
   time-series of BTC AND a cross-section of 51 cryptos**. Crypto carry (short perp/collect funding) in-sample Sharpe 7-10.
   → This is the empirical backbone of **LCF**.
8. **BIS WP1087**: a HIGH carry predicts FUTURE CRASHES + forced liquidations. → the **fragility cap** on LCF.
9. **Crypto Factor Zoo (.Zip), IRFA 2026**: 36 factors, 565 cryptos, 2018-2024. Only **3 factors beyond MKT** needed:
   turnover-volatility, salience value, new-address/price (EW); bid-ask + **7-day momentum** (VW). **Momentum is the
   most cost-resilient factor.** Factor premia are *evolutionary, not stable* (only bid-ask survives both subperiods).
10. **Liu, Tsyvinski & Wu (JoF 2022)** — Common Risk Factors: **CMKT + CSMB (size) + CMOM (momentum)**; 1-4wk momentum.
11. **CTREND — JFQA 60(7) 2025**: ML trend factor fusing **price AND volume** across horizons; robust, survives costs.
12. **Grobys et al. (FMPM 39, 2025)** — crypto momentum **crashes via idiosyncratic short-leg jumps** (one coin −255%/wk =
    37% of cumulative payoff); vol-scaling lifts payoff but **does not change the tail exponent** (power-law α<3).
    → Backbone of **TAIL** and **SCX** conditional-short.
13. **Barroso & Santa-Clara (2015)** — equity momentum vol-scaling: Sharpe 0.53→0.97. **We showed this is SUBOPTIMAL
    in crypto** (positively-skewed payoff; scaling throws away the upside). See §3.
14. **Order Flow and Crypto Returns (ScienceDirect 2026)**: world order flow predicts XS returns (daily +0.2%, weekly +0.9%,
    controlling for lagged returns); permanent component dominates weekly. → candidate feature for AWR / LCR confirmation.
15. **The Two-Tiered Structure of Funding Rate Markets (MDPI 2026)**: 26 exchanges, 35.7M 1-min obs. **CEX dominate price
    discovery (CEX-CEX corr > DEX-DEX by 61%, zero DEX→CEX causality)**; 17% of obs show ≥20bps cross-exchange arb
    spreads but only 40% profitable after costs. → cross-exchange funding arb is real but frictions-bound (our `research_fund` is single-venue; multi-venue is a stretch goal).
16. **Bui & Nguyen — AdaptiveTrend (arXiv 2602.11708)**: **6-HOUR bars** are the turnover×signal sweet spot
    (H1 Sharpe 1.54 → H6 2.41 → D1 1.63); **ATR trailing stop is the single biggest lift** (+0.73 Sharpe, −9.7pp MDD).
    Asymmetric 70/30 long/short (crypto positive drift). OOS 2022-24, 150+ pairs, Sharpe 2.41, MDD −12.7%.
    → Directly relevant to the **SCFM 5m/30m probe** (§4): we should test H6 / ATR-trailing, not just 5m/30m.
17. **Bysik & Slepaczuk (arXiv 2606.00060)**: cost-aware execution filter (λ=2×taker band: 20bps entry / 40bps flip)
    turns naive sign-ML from −64%/yr to **+65.4%/yr, Sharpe 1.09**. → the **live execution gate** we should port to `xs_signal`.
18. **Sadaqat & Butt (JBEF 39, 2023)**: stop-loss momentum highest payoff; **10-30% stops beat 40-50%** (realize sooner).
19. **Fičura (VŠE 2023)**: large liquid coins show WEEKLY momentum; **distance-from-recent-high** superior predictor.
20. **Huang, Sangiorgi & Urquhart (SSRN 4825389)**: volume-weighted TSMOM Sharpe 2.17. **Han, Kang & Ryu (SSRN 4675565)**:
    TSM strong, **long-only beats long-short**. **Hsieh, Huang & Liu (FRL 86, 2025)**: momentum works ONLY in UP-UP regimes.
21. **Sparkline crypto factors**: 4-factor (MKT, SMB, MOM, IHML-intangible-value); crypto momentum strongest 1-4wk, large-caps.
22. **starkiller.capital / mbrenndoerfer.com**: practitioner XS momentum (30d/7d) + market-structure/market-making notes.
23. **Kling Capital / KCE Capital**: AI-native quant shops (KAI platform, autonomous researcher agents, $178M AUM) — *context*,
    not method. Confirms the "autonomous agent runs multi-hour experiments" workflow we are emulating.

### 1c. Actionable gaps still open
- **Gap A (size-detrended attention):** AWR exists as proto; need WF OOS + cost validation vs MOM/REV/VOLRAW.
- **Gap B (liquidation-cascade contrarian):** LCR exists as proto; confirm volume+range cascade score beats pure REV.
- **Gap C (funding carry with fragility cap + coin-margined + staking):** LCF proto; add BIS fragility cap + heth.ink economics + winterresearch 3rd dimension.
- **Gap D (idiosyncratic tail filter):** TAIL in-sample promising (Sharpe 0.84); needs WF + crash-regime sample (2017/2022).
- **Gap E (regime-gated conditional-short):** SCX VALIDATED WF OOS (Sharpe 1.13-1.18) — **promote to production, replace xs_rel14**.
- **Gap F (short-cycle fused momentum H6 + ATR trailing):** SCFM probe done at 5m/30m; extend to H6 + ATR per Bui & Nguyen.

---

## 2. Research methodology (the contract for any new factor)

Every `scripts/research_*.py` follows:
1. **Docstring states the literature reviewed, the GAP, and the ONE mechanism** (not a stack).
2. **Parameters in `CONFIG`/`VARIANTS`** — never hardcoded into logic.
3. **Baselines**: MOM (XS momentum), REV (pure short-term reversal), and a confounded naive (e.g. VOLRAW) to prove novelty.
4. **Walk-forward**: rolling train→calibrate threshold→OOS apply. Report **per-config OOS** (n, win_rate, mean/median net bps,
   winsorized mean, p10/p90, SE, t-stat, ann Sharpe, net multiple) for **maker AND taker** sides.
5. **Crash regimes** reported explicitly (2018, COVID-2020, 2022, FTX-2022, bull, 2025-26).
6. **Tail metrics**: Hill index both tails, skew, exkurt, maxDD, CVaR5%, bootstrap 95% CI on Sharpe.
7. **Honest limitations** section mandatory (sample length, in-vs-OOS threshold choice, costs, single history).

Write-ups go to `quant_signal/research/*.md`; probes to `quant_signal/docs/*.json`.

---

## 3. WHAT IS DONE / PROVEN (promote or keep)

- **SCX — Skew-Convex Regime-Gated XS Momentum** (`research_scx.py`, `research/SCX_STRATEGY.md`)
  VALIDATED WF OOS 2017-2026, 456 weeks, 31 coins, 10bps, Binance daily.
  - REGIME_LONG (bull-gated long-only winners): **Sharpe 1.18**, MDD −60%.
  - **SCX (gate + conditional short)**: **Sharpe 1.13**, MDD −55% — best risk-adjusted *with* crash insurance.
  - SCX_VOL (above + vol-scale): Sharpe 0.96, MDD −43% (capital-preservation choice).
  - Regime gate flattens 2018/2022/FTX (works). **One failure: COVID −3.17** (slow MA lags fast crash; shorts are the only offset → keep SCX over long-only).
  - **RECOMMENDATION: deploy SCX; replace live `xs_rel14` (BSC L/S, ~0.72).**
- **TAIL_AWARE_XS_MOMENTUM** (`research/research_xs_tail.py`, `TAIL_AWARE_XS_MOMENTUM.md`)
  In-sample 2022-2026, 31 large-caps, 206 weekly rebalances. **Key novel result: BSC vol-scaling is
  COUNTERPRODUCTIVE for crypto XS momentum (Sharpe 0.79→0.56) because payoff is positively skewed**;
  scaling throws away the upside. TAIL_ALL k=1.5: Sharpe **0.84**, MDD −35%. Edge vs naive (+0.05) is
  *inside* bootstrap CI → needs WF + crash sample before promotion.
- **LCF / LCR / AWR** (`research_fund.py`, `research_lcr.py`, `research_awr.py`): protos written, docstrings
  encode the mechanism + baselines, but **not yet WF-OOS-validated**. Next to validate.

## 4. WHAT IS FALSIFIED (do NOT re-litigate without new data)

From `git log` + probe JSONs (`quant_signal/docs/probe_5m.json`, `probe_30m.json`, `probe_calendar.json`, `probe_washout.json`):
- **Calendar / time-of-day (overnight) probe** — FALSIFIED on 185d real history.
- **Washout mean-reversion (1h)** — underpowered (2-12 trades / 185d); edge weak.
- **30m RGW + funding-clock probe** — washout edge weak, regime-gating adds nothing, funding vol burst ABSENT.
- **5m RGW reversal** — REAL but **sub-gate** (below Bysik & Slepaczuk cost filter); funding-mark vol burst found but not tradable at 5m.
- Implication: short-cycle edges live at **H6 + ATR trailing** (Bui & Nguyen), not 5m/30m raw. Point SCFM there.

---

## 5. OPEN NEXT STEPS (priority order)

1. **PROMOTE SCX to production**: wire SCX (no BSC) into `xs_signal`; retire `xs_rel14`. Keep conditional short as insurance.
2. **Port Bysik & Slepaczuk cost-aware gate** (20bps entry / 40bps flip, λ=2×taker) into the live executor.
3. **WF-validate the three protos**: LCF (add BIS fragility cap + coin-margined/staking economics + winterresearch funding×skew 3rd dim), LCR (volume+range cascade vs REV), AWR (size-detrended attention vs VOLRAW).
4. **Extend TAIL_ALL** to a 2017-2026 panel (include 2022-style crash) + walk-forward k; confirm DD reduction survives OOS.
5. **SCFM v2**: test **6-hour bars + ATR trailing stop** (Bui & Nguyen), short-cycle lookback WF-learned in {2,3,5,7,14}d, maker-executed, long-biased fused momentum (distance-from-high + volume-weighted TSMOM + CGO+momentum per Liu-Fang-Wang).
6. **New factor candidates from the dive**: (a) cross-exchange funding Arb (CEX-DEX, MDPI 2026) — frictions-bound, stretch;
   (b) world order-flow XS predictor (ScienceDirect 2026) as AWR/LCR confirmation; (c) disposition-overhang+2wk-momentum G/r long-only (Liu-Fang-Wang) as a dedicated long book.
7. **Live validation**: forward-test SCX on Bybit Demo with honest fills before sizing up; track %flat (capital idle in bears is a feature, not a bug).

---

## 6. HARD RULES (carry into every turn)
- Real data only. No synthetic in production. Verify endpoints live.
- No hardcoded magic numbers; WF-calibrate thresholds; costs + crash regimes always on.
- Report maker AND taker; bootstrap CI on Sharpe; Hill tails; honest limitations.
- A factor must beat its naive baseline (MOM/REV/confounded) AND survive OOS before promotion.
- One economic mechanism per factor; cite the paper that licences it.
- Keep research write-ups (`research/*.md`) and probes (`docs/*.json`) in sync with code.

## 7. Repo quick-reference (files that matter)
- `quant_signal/scripts/research_*.py` — factor protos (awr, fund, lcr, scx, xs_tail).
- `quant_signal/scripts/trend_momentum_probe.py` + `pull_binance_*.py`, `backfill_binance_history.py`, `diagnose_vreg.py`.
- `quant_signal/stream/{xs_signal,lcf_signal,bybit_demo}.py` — live signals + venue.
- `quant_signal/research/{SCX_STRATEGY,TAIL_AWARE_XS_MOMENTUM}.md` — validated write-ups.
- `quant_signal/docs/{architecture,system_design,snowpipe_streaming}.md` — platform.
- `quant_signal/docs/liu_fang_wang_2024_*.{pdf,txt}` — Chinese disposition+momentum paper (primary source).
- `quant_signal/docs/probe_{5m,30m,calendar,washout,trend_momentum}.json` — falsification evidence.

**GOAL for the new session:** take the validated SCX to production, WF-validate LCF/LCR/AWR, and run SCFM v2 at H6+ATR — all against real Binance/Bybit data, costs on, crash regimes reported, honest.
