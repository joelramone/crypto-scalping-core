# HYP-SHORT-COMPRESSION-BREAKDOWN-001: Short Compression Breakdown Baseline

## Registration

- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Purpose:** `short_compression_breakdown_discovery_baseline`
- **Strategy family:** `STRATEGY_FAMILY_7_SHORT_COMPRESSION_BREAKDOWN`
- **Symbol:** `BTCUSDT`
- **Timeframe:** `15m`
- **Direction:** **SHORT ONLY**
- **Verdict policy:** `authoritative_baseline_gates`

This document pre-registers one Family #7 discovery baseline. It freezes the
research design before any implementation, market-data inspection, signal count,
or outcome measurement.

## Research question

Can a deterministic short setup based on prior price compression followed by a
confirmed downside breakdown produce fewer, more selective trades with materially
higher gross edge per trade and lower fee drag than the rejected Family #5 and
Family #6 baselines?

## Design-only candidates considered

The candidates below are ranked solely by causal clarity, parsimony, and tuning
risk. No market outcomes or signal counts informed the ranking.

| Rank | Conceptual candidate | Independence | Causality | New thresholds | Expected sample / turnover | Complexity | Hidden tuning risk | Interpretability |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **B: one strict inside bar, then close below its low** | Fully independent of regime-transition and EMA-pullback logic | Uses only completed candles `t-2`, `t-1`, and `t` | None; only structural inequalities | A single compression relationship should be less restrictive than nested or multi-window alternatives; selective turnover with the best design-only prospect of reaching the sample gate | Low | Low | High |
| 2 | **A: consecutive narrow-range candles, then break of their low** | Fully independent | Causal when all compression candles precede `t` | Requires freezing how many contractions define compression | Likely lower turnover than B, with a greater insufficient-sample risk | Medium | Medium because contraction count is a tunable degree of freedom | High |
| 3 | **C: short rolling high-low range below a long rolling range, then boundary break** | Fully independent | Causal when both windows end at `t-1` | Requires at least short- and long-window choices and precise range aggregation semantics | Unknown without prohibited measurement; overlapping rolling windows may create clustered signals | Medium | High because window pairs invite optimization | Medium |

Candidate B is selected because strict containment is a self-scaling definition of
range contraction, needs no volatility cutoff or rolling-window parameter, and
uses the smallest interpretable sequence that separates compression from a later
confirmed breakdown. Sample sufficiency is only a design expectation; it was not
estimated from data and is not guaranteed.

## Structural independence

This hypothesis tests the causal sequence **inside-range compression -> downside
boundary break**. Its entry rule has no `base_regime`, `TREND_DOWN`, regime
transition, EMA level, EMA alignment, pullback, or continuation predicate.

- It is not `HYP-SHORT-DIRECTIONAL-001` plus a compression filter: a signal can
  occur regardless of the current or prior regime, and no transition into
  `TREND_DOWN` is required.
- It is not `HYP-SHORT-PULLBACK-CONTINUATION-001` plus a compression filter: no
  bearish EMA structure, bullish pullback, EMA touch, or pullback failure is
  required.

The compression and its boundary are the complete setup, not filters applied to
either earlier family.

## Exact frozen entry rule

Candles are causally resampled, completed `15m` candles. For a candidate signal
candle `t`:

- `t-2` is the completed **mother bar**.
- `t-1` is the completed **compression bar**.
- `t` is the completed **breakdown confirmation bar**.

Create a raw short-entry signal if and only if all three strict inequalities are
true:

```text
high[t-1] < high[t-2]
AND low[t-1] > low[t-2]
AND close[t] < low[t-1]
```

The first two inequalities make `t-1` a strict inside bar: its entire high-low
range is contained strictly within the mother bar. The frozen compression
boundary is `low[t-1]`. The breakdown compares `close[t]`, not merely `low[t]`,
with that boundary. There is no separate candle-color condition; a completed
close strictly below the frozen boundary is the bearish confirmation.

The compression definition and boundary exclude candle `t`. Equality never
qualifies. No tolerance, alternative boundary, nested-bar variant, gap exception,
or additional filter is allowed. At least three completed resampled candles are
required; the first two rows cannot signal, and any row lacking a required OHLC
value cannot signal.

## Signal timing

Evaluate the rule only after candle `t` is complete and enter short at
`close[t]`. Candles `t-2` and `t-1` are already complete, and no candle after `t`
participates. This preregistration does not authorize implementation of the rule.

## Frozen execution policy

The future authorized baseline must use the same execution assumptions frozen for
Families #5 and #6:

| Item | Frozen value |
| --- | --- |
| Direction | short only |
| Timeframe | `15m` |
| Entry | `close[t]` after completed signal candle |
| `take_profit_pct` | `0.012` |
| `stop_loss_pct` | `0.008` |
| `max_holding_candles` | `24` |
| Strategy exits | `false` |
| Position concurrency | one open position at a time; raw signals during an open position are suppressed |
| Fixed notional | `100 USDT` |
| Fee rate | `0.0004` on entry notional and `0.0004` on exit notional |
| Leverage | none |
| Slippage model | none |
| Spread model | none |
| Compounding | none |

Exit evaluation begins at `t+1`; the entry candle cannot exit. For a short, take
profit is `entry_price * (1 - 0.012)` and stop loss is
`entry_price * (1 + 0.008)`. If both are touched in one candle, stop loss takes
precedence. If neither is touched and strategy exits remain false, exit at the
close of `t+24` or the final available discovery candle, whichever comes first.

## Data boundary

The sole discovery source for one separately authorized future baseline is
`data/BTCUSDT_1m.csv`, restricted to calendar year `2025` and causally resampled
to `15m` according to existing repository behavior. Dataset selection must occur
before feature calculation. No external validation allocation is made here.

- `2025`: **DISCOVERY ONLY** for the future single baseline; not accessed by this
  documentation task.
- `2026-01-01` through `2026-08-05`, inclusive: **NOT AUTHORIZED / DO NOT ACCESS**.
- Data strictly after `2026-08-05`: **RESERVED / DO NOT ACCESS**.

Neither 2026 segment may be opened, profiled, counted, or used to influence this
hypothesis without separate human authorization.

## Mandatory standard diagnostics

The future baseline must persist the standard `TradeDiagnostics` output and
report all of the following:

- **Sample:** `raw_entry_signals`, `completed_trades`, `suppressed_signals`,
  `suppression_rate`, and `raw_signals_per_opened_trade`.
- **Gross / net:** `gross_pnl_before_fees`, `total_fees`, `net_pnl`,
  `gross_expectancy`, `fee_expectancy`, `net_expectancy`,
  `gross_profit_factor`, and `net_profit_factor`.
- **Payoff:** `wins`, `losses`, `flats`, `win_rate`, `avg_winner`, `avg_loser`,
  `payoff_ratio`, `break_even_win_rate`, and
  `actual_minus_break_even_win_rate`.
- **Exits:** counts and PnL for `TP`, `SL`, `max_holding`, and `strategy exit`.
- **Holding:** average, median, P25, P75, and P95 holding duration.
- **Monthly:** positive months, negative months, profitable-month percentage,
  best month, worst month, and top-two positive-PnL concentration.

Also report the observational metric:

```text
gross_edge_to_fee_ratio = gross_expectancy / fee_expectancy
```

If `fee_expectancy` is zero, record this ratio as undefined rather than inventing
a substitute value. The ratio is descriptive and is not a verdict gate.

## Observational expectations, not acceptance gates

Family #7 tests whether structural compression-breakdown selection improves trade
economics at lower turnover. It descriptively expects fewer raw signals than
broad regime-transition approaches; a lower suppression rate is plausible; higher
gross expectancy per trade is desirable; and `gross_edge_to_fee_ratio > 1` would
be economically encouraging. Fewer trades are acceptable provided
`completed_trades >= 100`.

These statements are expectations only. They cannot override, supplement, or be
converted into the deterministic verdict gates, and they do not authorize a
comparison-driven modification.

## Deterministic verdict policy

Apply `authoritative_baseline_gates` in this exact order after the single future
2025 baseline:

1. **`INSUFFICIENT_SAMPLE`** if `completed_trades < 100`. Stop evaluation.
2. If `completed_trades >= 100`, return **`BASELINE_REJECT`** if any condition is
   true: `gross_expectancy <= 0`, `net_profit_factor <= 1`,
   `net_expectancy <= 0`, `net_pnl <= 0`, or
   `positive_pnl_concentration_top_2_months > 0.80`.
3. Return **`BASELINE_CANDIDATE`** only if all conditions pass:
   `completed_trades >= 100`, `gross_expectancy > 0`,
   `net_profit_factor > 1`, `net_expectancy > 0`, `net_pnl > 0`, and
   `positive_pnl_concentration_top_2_months <= 0.80`.

No observational metric may alter the verdict.

## Anti-tuning termination rule and prohibited modifications

Exactly one 2025 baseline is permitted after separate human authorization. If its
verdict is `INSUFFICIENT_SAMPLE` or `BASELINE_REJECT`, this exact Family #7
hypothesis closes. It must not be rescued by changing the compression definition,
lookback length, breakout boundary, close-versus-low semantics, timeframe, TP, SL,
maximum holding period, strategy exits, volume conditions, ATR conditions, EMA
conditions, RSI conditions, cooldown, candle-body filters, wick filters, or regime
filters. Near-neighbor tests, alternate containment semantics, additional
confirmations, sensitivity runs, and optimization are also prohibited.

A materially changed idea requires Family #8, a new hypothesis ID, and a new
preregistration. If the verdict is `BASELINE_CANDIDATE`, it does not authorize
2026 access or external validation; both require separate human authorization.

## Authorization boundary

**THIS PREREGISTRATION AUTHORIZES DOCUMENTATION ONLY.**

It does **not** authorize:

- strategy implementation;
- strategy registry changes;
- synthetic tests;
- configuration creation;
- market-data loading;
- grid search or other optimization;
- backtesting;
- signal counting;
- creation or execution of `EXP-000023`;
- Research Memory changes;
- leaderboard creation;
- journal creation;
- performance report creation; or
- access to any 2026 data.

Every later stage requires separate human authorization. Human review is required,
and this preregistration must not be treated as permission to run the baseline.
