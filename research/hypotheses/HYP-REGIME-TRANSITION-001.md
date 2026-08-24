# HYP-REGIME-TRANSITION-001: Bullish Regime Transition Baseline

## Registration

- **Hypothesis ID:** `HYP-REGIME-TRANSITION-001`
- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Purpose:** `discovery_baseline`
- **Strategy family:** `STRATEGY_FAMILY_4_REGIME_TRANSITION_DIRECTIONAL_STATE_CHANGE`
- **Symbol:** `BTCUSDT`
- **Timeframe:** `15m`
- **Direction:** Long only

This document pre-registers one minimal, deterministic hypothesis: a transition from a non-directional market state into a newly established bullish directional state may contain more forward displacement than entry during an already mature trend. It registers a baseline experiment only; it does not claim profitability and does not authorize strategy implementation, backtesting, optimization, or inspection of outcomes during specification work.

## Data boundaries

The sole discovery source is `data/BTCUSDT_1m.csv`, resampled causally to `15m`, and restricted to calendar year `2025`. Its classification is `DISCOVERY_USED`, and it is permitted for one initial baseline experiment only.

The interval `2026-01-01` through `2026-08-05`, inclusive, is prohibited during baseline design and execution because it has already been used for validation or research elsewhere. Data after `2026-08-05` must not be inspected because it remains reserved by `HYP-DONCHIAN-HIGHVOL-001`.

## Existing regime infrastructure is frozen

The baseline must use the existing causal regime implementation and its base-regime definitions exactly, without modification. That infrastructure exposes `TREND_UP`, `TREND_DOWN`, `RANGE`, and `NEUTRAL`, plus the separate `is_high_volatility` overlay. Its existing causal features include EMA20, EMA50, EMA200, EMA slopes, EMA separations, ADX14, realized volatility, `return_4`, `return_12`, and `return_24`.

No regime threshold, ADX threshold, EMA period, volatility percentile, feature definition, or regime classification may be optimized, recalibrated, or replaced. `is_high_volatility` is not part of the signal.

## Frozen hypothesis and exact transition rule

For each completed candle `t`, define a raw long-entry signal exactly as:

```text
base_regime[t-1] != TREND_UP
AND
base_regime[t] == TREND_UP
```

The transition itself is the entire signal hypothesis. `TREND_UP -> TREND_UP` must not create another signal. The baseline is therefore not a generic persistent-state `TREND_UP` entry, Donchian breakout, EMA pullback, or Bollinger re-entry.

The signal is evaluated only after candle `t` is complete. Entry occurs through the official simulator at `close[t]`, with no future information. No breakout condition, RSI, volume filter, close-location filter, Bollinger Band, ATR gate, high-volatility condition, return threshold, candle confirmation, quality score, cooldown threshold, machine learning, or feature weighting may be added.

## Frozen exits

The initial baseline must use the following neutral `15m` research exits without optimization:

| Parameter | Frozen value |
| --- | ---: |
| `take_profit_pct` | `0.012` |
| `stop_loss_pct` | `0.008` |
| `max_holding_candles` | `24` |
| `generate_exits` | `false` |

## Mandatory TradeDiagnostics report

The permanent `TradeDiagnostics` output is mandatory. The future baseline report must contain:

- **Gross/net:** `gross_pnl_before_fees`, `total_fees`, `net_pnl`, `gross_expectancy`, `fee_expectancy`, `net_expectancy`, `gross_profit_factor`, and `net_profit_factor`.
- **Payoff:** wins, losses, flats, win rate, average winner, median winner, average loser, median loser, payoff ratio, break-even win rate, and actual-minus-break-even win rate.
- **Exits:** count, percentage, and PnL for TP, SL, and max-holding exits; plus average, median, P25, P75, and P95 holding duration.
- **Overlap:** `raw_entry_signals`, `completed_trades`, `suppressed_signals`, `suppression_rate`, and `raw_signals_per_opened_trade`.
- **Monthly:** trades, profit factor, expectancy, and net PnL by month; positive-month count; negative-month count; and positive PnL concentration.

Because this hypothesis is transition-based, raw signal frequency is expected to be materially lower than persistent `TREND_UP`-state signal frequency. This is descriptive only and has no numeric pass threshold. The report must expose at least `raw_entry_signals`, `completed_trades`, and `suppression_rate` for this signal-independence diagnostic.

## Pre-registered baseline verdict

Verdicts are applied in the following order:

1. **`INSUFFICIENT_SAMPLE`:** `completed_trades < 100`.
2. **`BASELINE_REJECT`:** when `completed_trades >= 100` and any of `net_profit_factor <= 1`, `net_expectancy <= 0`, or `net_pnl <= 0`; also reject when `gross_expectancy <= 0`, or when `positive_pnl_concentration_top_2_months > 0.80`.
3. **`BASELINE_CANDIDATE`:** only when all of the following hold: `completed_trades >= 100`, `gross_expectancy > 0`, `net_profit_factor > 1`, `net_expectancy > 0`, `net_pnl > 0`, and `positive_pnl_concentration_top_2_months <= 0.80`.

No optimization is allowed after `BASELINE_REJECT`.

## Anti-tuning termination rule

After observing the 2025 baseline, it is prohibited to change the ADX threshold, EMA periods, regime definitions, transition type, TP, SL, or holding period; add `HIGH_VOLATILITY` conditions, return filters, volume filters, or candle filters; or otherwise alter the frozen specification to rescue a rejected baseline.

If the baseline fails, this exact hypothesis ends.
