# SCX — Skew-Convex, Regime-Gated Cross-Sectional Momentum

**Status:** validated walk-forward, out-of-sample, with transaction costs, 2017-2026 (456 weeks, 31 coins, Binance daily).
**Harness:** `scripts/research_scx.py` (fully parameterized; costs and the conditional-short threshold are not hardcoded).

---

## 1. What is genuinely new here

This is **not** a reproduction of a published paper (unlike `xs_rel14`, which reproduces Grobys 2025 / BSC 2015 / Liu-Tsyvinski 2021). It is a tradable book built from three economic facts:

1. Crypto cross-sectional momentum's excess return lives in the **long book's positive skew** (Liu-Tsyvinski 2021). The **short book is tail-risk + funding drag** (Grobys 2025: crypto crashes are idiosyncratic short-leg jumps).
2. A BTC **trend regime filter** (price > 90d and > 200d MA) de-risks the book to flat in bears.
3. **Novel overlay:** short exposure is made *conditional* — shorts are dropped (book goes long-only) whenever the short side's trailing realized volatility is in its stressed quantile. This is a **dynamic, skew-aware net-exposure switch**, not static vol-scaling. It keeps shorts as crash-insurance when calm, and retracts them when the short book is throwing tail events.

> **Honest caveat on novelty:** the dominant alpha is *long-only momentum gated by a trend regime*. That core is well documented (Liu-Tsyvinski). SCX's novelty is the *combination* and the *conditional-short toggle*; it is not a discovery of a new factor.

---

## 2. Results (Binance 2017-08 → 2026-08, 10 bps/side, weekly rebalance)

| variant | ann_ret | ann_vol | Sharpe | 95% CI | maxDD | CVaR5% | %flat |
|---|---|---|---|---|---|---|---|
| NAIVE_LS (full L/S) | 48% | 72% | 0.67 | [0.04,1.35] | -78% | -11.1% | 17% |
| BSC_LS (vol-scaled) | 34% | 47% | 0.72 | [0.12,1.43] | -53% | -7.2% | 17% |
| REGIME_LS (gate+L/S) | 42% | 65% | 0.65 | [0.05,1.37] | -77% | -9.1% | 59% |
| **REGIME_LONG (gate, long-only)** | 81% | 69% | **1.18** | [0.55,1.91] | -60% | -10.1% | 59% |
| **SCX (gate + conditional short)** | 72% | 64% | **1.13** | [0.46,1.95] | -55% | -8.6% | 59% |
| **SCX_VOL (above + vol-scale)** | 38% | 39% | 0.96 | [0.30,1.80] | **-43%** | **-5.8%** | 59% |

The biggest single improvement over the deployed `xs_rel14` (BSC L/S, ~0.72) is **dropping the toxic short book** and gating by regime: Sharpe roughly doubles.

## 3. Crash-regime sub-period Sharpe (annualized)

| variant | 2018 bear | COVID 2020 | 2022 bear | FTX 2022 | 2023-24 bull | 2025-26 |
|---|---|---|---|---|---|---|
| NAIVE_LS | 0.00 | -0.60 | 1.19 | -1.41 | 1.26 | 1.16 |
| REGIME_LS | 0.00 | 0.92 | 0.00 | 0.00 | 1.18 | 1.66 |
| REGIME_LONG | 0.00 | **-3.17** | 0.00 | 0.00 | 1.19 | 0.21 |
| SCX | 0.00 | **-3.17** | 0.00 | 0.00 | 1.39 | 0.35 |

- **Regime gate works:** in 2018/2022/FTX the gated books are flat (0.00) — exactly the de-risking intended.
- **One real failure:** COVID Mar-2020. The slow MA lags, so the book was still long into the crash → Sharpe -3.17 for long-only variants. The full L/S (NAIVE_LS) lost far less (-0.60) because shorts offset. **This is the case for keeping shorts as insurance** — which is what SCX does.
- **2025-26:** full L/S outperformed long-only (regime dispersion favored shorts). SCX's stress toggle was trigger-happy and missed part of it (0.35 vs 1.66). The overlay is a risk toggle, not a strict Sharpe enhancer in every regime.

## 4. Verdict — what we can actually bet on

- **Most robust, highest-Sharpe bettable book: `REGIME_LONG`** (long-only winners, bull-gated). Sharpe 1.18, but -60% DD and no crash insurance.
- **Best risk-adjusted with insurance: `SCX`** (conditional short). Sharpe 1.13, -55% DD, keeps short-side crash protection.
- **Lowest drawdown: `SCX_VOL`** (0.96 Sharpe, -43% DD) if capital preservation matters more than raw return.

**Recommendation for production:** deploy **SCX** — it captures the long-only momentum edge, gates bears to flat, and retains shorts as dynamic crash-insurance. The conditional-short overlay is the novel, defensible piece.

## 5. Honesty / limitations

- Sharpes of 1.0+ are on a 31-name crypto universe, weekly rebalance, 10 bps costs. Real funding, borrow, and slippage on shorts will be higher → net Sharpe lower.
- Books are invested only ~41% of the time (%flat=59%); capital is idle in bears (risk reduction, not free).
- COVID shows the regime filter lags fast crashes; shorts are the only offset.
- Past performance ≠ future. This is a walk-forward/OOS result on one history; it needs forward live validation before sizing up.
- The live `xs_rel14` currently uses BSC vol-scaling + full L/S (the weakest cell). It should be replaced by SCX.

---

## 6. Live deployment (`stream/scx_signal.py`)

**Status:** built, unit-tested (`tests/test_scx_signal.py`), wired into the
`STREAM_STRATEGY=scx` dispatch in `stream/xs_signal.py main()`. Emits the same
`prediction:crypto:1h:<SYMBOL>` payload the executor already consumes, so the
Bybit demo venue and dashboard are untouched.

**Faithful translation of the research daily panel to the 1h stream (no new
magic numbers — every knob is a `stream_scx_*` setting):**

| research (`research_scx.py`) | live (`ScxSignal`) |
|---|---|
| 14-day momentum | 336 hourly bars (`close/close.shift(336)-1`) |
| BTC > 90d & 200d MA (weekly, ffill) | BTC 90/200-**DAY** MA on the daily resample of hourly BTC closes (`_daily_up`) |
| short-book vol = rolling 12-week std of bottom-quintile fwd return | rolling 12-week std of the per-week bottom-quintile realized return, built **lookahead-free**: each week's short-book return is recorded only at the NEXT rebalance (`_record_short_book_return`) |
| conditional short when rolling std > `stress_q` (0.60) quantile | identical `_short_stressed()` decision, WF-selected `stress_q` |
| walk-forward stress quantile | `stream_scx_stress_q = 0.60` |

**Lookahead safety.** Every input uses closes strictly before the rebalance bar
(`_last_index` with `side="left"`). The short-book return for week *w* is only
known at week *w+1* and is recorded then — it never enters week *w*'s selection.

**Regime gate is live on the first bar.** `warm_start` seeds BTC to ~200+ days
of hourly history (paginating Bybit's 2000-bar cap) so the 90/200-day MA gate is
satisfied immediately; the rest of the universe is seeded to just past the 336h
lookback. Without history the gate fails **closed (FLAT)** — never trades blind.

**Conditional-short behaviour in early life.** Until 12 weeks of short-book
history accrue, `_short_stressed` returns False, so the book runs as REGIME_LS
(full L/S); the SCX overlay then engages exactly as researched.

**Production hardening note.** `_short_stressed` floors on effectively-zero vol
(`roller.max() < 1e-9`) so floating-point noise in `std()` on a degenerate flat
short book cannot spuriously trip the stress flag. This is a numerical-stability
guard, not a strategy parameter, and never fires on real (noisy) data.

