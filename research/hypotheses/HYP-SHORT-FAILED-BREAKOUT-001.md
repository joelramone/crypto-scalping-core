# HYP-SHORT-FAILED-BREAKOUT-001: Short Failed Breakout Baseline

## Registration

- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Purpose:** `short_failed_breakout_discovery_baseline`
- **Strategy family:** `STRATEGY_FAMILY_8_SHORT_FAILED_BREAKOUT`
- **Symbol:** `BTCUSDT`
- **Direction:** **SHORT ONLY**
- **Timeframe:** `15m`
- **Verdict policy:** `authoritative_baseline_gates`
- **Future experiment:** `EXP-000024` is `RESERVED_FOR_FUTURE_AUTHORIZED_EXECUTION`

This document pre-registers exactly one Family #8 discovery baseline. It freezes
the research design before implementation, market-data access, signal counting,
or outcome measurement. It does not create or authorize `EXP-000024`.

## Research thesis and independent mechanism

A failed bullish breakout may contain short directional information when price
first exceeds a previously observable local high and subsequently rejects that
breakout by closing back below the breakout level.

```text
existing observable resistance
        ↓
bullish breakout attempt
        ↓
failure to maintain acceptance above resistance
        ↓
close back below resistance
        ↓
SHORT
```

This is an independent hypothesis family testing breakout rejection. It is not
regime-transition trading, momentum pullback continuation, compression
breakdown, Donchian optimization, or a rescue of Family #5, Family #6, or Family
#7. The failed-breakout sequence is the complete setup, not a filter or
reinterpretation applied to a prior family.

## Exact frozen entry rule

Use completed, causally resampled `15m` candles. Define the resistance frozen
before the breakout candle as:

```text
resistance[t-2] = max(high[t-3], high[t-4], high[t-5])
```

After candle `t` is complete, create a raw short-entry signal at `close[t]` if
and only if all three strict inequalities are true:

```text
high[t-1] > resistance[t-2]
AND close[t-1] > resistance[t-2]
AND close[t] < resistance[t-2]
```

Candles `t-5`, `t-4`, and `t-3` establish the local resistance. Candle `t-1`
breaks that resistance and closes above it. Candle `t` confirms rejection by
closing back below the same frozen resistance. Equality never qualifies. There
is no alternative resistance definition, tolerance, candle-color condition, or
additional entry predicate. At least six completed resampled candles are
required to evaluate the rule, and any row lacking required OHLC values cannot
signal.

## Causality and lookahead safety

The rule is causal and lookahead-safe at the stated signal time. The resistance
uses only completed candles through `t-3`, so it is completely observable and
frozen before breakout candle `t-1`. The breakout predicates use completed candle
`t-1`; the rejection predicate uses completed signal candle `t`. Evaluation and
short entry occur only after `t` closes, and no candle after `t` participates.
The rule must not be altered based on market results.

## Exactly one frozen future configuration

The future baseline must contain exactly one configuration. The execution values
below are controls, not parameters to optimize:

| Item | Frozen value |
| --- | --- |
| Direction | short only |
| Timeframe | `15m` |
| Entry | `close[t]` after completed signal candle |
| `take_profit_pct` | `0.012` |
| `stop_loss_pct` | `0.008` |
| `max_holding_candles` | `24` |
| Strategy-specific exits | none (`generate_exits = false`) |
| Position concurrency | one open position at a time; raw signals during an open position are suppressed |
| Fixed notional | `100 USDT` |
| Fee rate | `0.0004` on entry notional and `0.0004` on exit notional |
| Leverage | none |
| Slippage model | none |
| Spread model | none |
| Compounding | none |

The future implementation must rely only on the frozen simulator take profit,
stop loss, and maximum holding period. Exit evaluation begins at `t+1`; the
entry candle cannot exit. For a short, take profit is
`entry_price * (1 - 0.012)` and stop loss is
`entry_price * (1 + 0.008)`. If both are touched in one candle, stop loss takes
precedence. Otherwise, close at the close of `t+24` or the final available
discovery candle, whichever comes first. No adaptive or strategy-specific exit
is allowed.

There is no parameter grid, sweep, sensitivity analysis, or second
configuration.

## Dataset boundary

The sole discovery source for one separately authorized future baseline is
`data/BTCUSDT_1m.csv`, restricted to `2025-01-01` through `2025-12-31`, and
causally resampled to `15m` according to existing repository behavior. Dataset
selection must occur before feature calculation.

- `2025-01-01` through `2025-12-31`: **DISCOVERY ONLY** for the future single
  baseline; not accessed by this documentation task.
- `2026-01-01` through `2026-08-05`, inclusive: **NOT AUTHORIZED FOR THIS
  BASELINE**.
- Data after `2026-08-05`: **RESERVED / NOT AUTHORIZED**.

No prohibited or reserved interval may be opened, profiled, counted, or used to
influence this hypothesis without separate human authorization.

## Required permanent diagnostics

The future baseline must persist and report all of the following standard
permanent diagnostics:

- **Sample:** `raw_entry_signals`, `completed_trades`, `suppressed_signals`,
  `suppression_rate`, `raw_signals_per_opened_trade`.
- **Gross:** `gross_pnl_before_fees`, `gross_expectancy`,
  `gross_profit_factor`.
- **Costs:** `total_fees`, `fee_expectancy`, `gross_edge_to_fee_ratio`.
- **Net:** `net_pnl`, `net_expectancy`, `net_profit_factor`, `max_drawdown`.
- **Payoff:** `win_rate`, `average_winner`, `median_winner`, `average_loser`,
  `median_loser`, `payoff_ratio`, `break_even_win_rate`,
  `actual_minus_break_even_win_rate`.
- **Holding:** `average_holding_candles`, `median_holding`, `holding_p25`,
  `holding_p75`, `holding_p95`.
- **Exit structure:** `take_profit_exits`, `stop_loss_exits`,
  `max_holding_exits`, `strategy_exit_exits`, plus percentages and net PnL for
  each exit type.
- **Temporal stability:** monthly diagnostics, `positive_months`,
  `negative_months`, `profitable_month_percentage`, `best_month`, `worst_month`,
  and `positive_pnl_concentration_top_2_months`.

`gross_edge_to_fee_ratio` is descriptive, not a verdict gate. If its denominator
is zero, record the ratio as undefined rather than inventing a substitute value.

## Deterministic verdict policy

The future single baseline must apply the existing
`authoritative_baseline_gates` policy, without adding or reinterpreting gates, in
this exact order:

1. **`INSUFFICIENT_SAMPLE`** if `completed_trades < 100`. Stop evaluation.
2. If `completed_trades >= 100`, return **`BASELINE_REJECT`** if any condition is
   true: `gross_expectancy <= 0`, `net_profit_factor <= 1`,
   `net_expectancy <= 0`, `net_pnl <= 0`, or
   `positive_pnl_concentration_top_2_months > 0.80`.
3. Return **`BASELINE_CANDIDATE`** only if all conditions pass:
   `completed_trades >= 100`, `gross_expectancy > 0`,
   `net_profit_factor > 1`, `net_expectancy > 0`, `net_pnl > 0`, and
   `positive_pnl_concentration_top_2_months <= 0.80`.

No diagnostic or descriptive comparison may override the deterministic verdict.

## Anti-tuning termination rule

If the baseline is rejected or has an insufficient sample, this hypothesis
closes. There is no rescue, parameter tuning, nearby lookback, alternative
resistance window, alternate timeframe, additional confirmation filter, volume
filter, RSI filter, EMA filter, ATR filter, regime filter, alternative exit, or
second configuration. Do not change an inequality, the close-based confirmation,
TP, SL, maximum holding period, fee or sizing assumptions, or data boundary. Do
not run a parameter grid, sweep, sensitivity analysis, optimization, or
near-neighbor variant.

A materially different idea requires a new hypothesis family, a new hypothesis
ID, and new human approval.

## Family independence and immutability

Family #5 remains immutable. Family #6 remains immutable. Family #7 remains
immutable. Family #8 must not modify or reinterpret their designs, artifacts,
results, or verdicts. Descriptive comparisons after a separately authorized
Family #8 execution may be allowed, but they cannot change any previous verdict.

## Experiment reservation and authorization boundary

`EXP-000024` is `RESERVED_FOR_FUTURE_AUTHORIZED_EXECUTION`. It is not created by
this preregistration. The reservation does not authorize a journal, Research
Memory row, leaderboard, configuration, implementation, or execution artifact.

**THIS PREREGISTRATION AUTHORIZES DOCUMENTATION ONLY.** It does not authorize:

- strategy implementation or registry changes;
- simulator or governance changes;
- synthetic or market-data strategy tests;
- configuration creation;
- opening or processing market data;
- signal counting or sample-size estimation;
- resampling;
- backtesting;
- `grid_search` or any optimization;
- creation or execution of `EXP-000024`;
- modification of `research/memory/index.csv`;
- journal, leaderboard, or performance-report creation; or
- access to any 2026 data.

Every later stage requires separate human authorization and human review.
