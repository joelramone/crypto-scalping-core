# HYP-SHORT-DIRECTIONAL-001: Short Directional Baseline

## Registration

- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Purpose:** `directional_asymmetry_discovery_baseline`
- **Direction:** **SHORT ONLY**
- **Timeframe:** `15m`

This document pre-registers Family #5 without implementing its strategy or authorizing a market backtest.

## Frozen hypothesis

Using the existing causal `base_regime` implementation unchanged, generate a signal only when the following exact condition is true:

```text
base_regime[t-1] != TREND_DOWN
AND
base_regime[t] == TREND_DOWN
```

Evaluate the signal after completed candle `t` and enter short at `close[t]`. No future information is allowed.

## Dataset boundary

The sole discovery dataset is `data/BTCUSDT_1m.csv`, restricted to calendar year 2025, causally resampled to `15m`, and classified `DISCOVERY_USED`.

The interval `2026-01-01` through `2026-08-05`, inclusive, is **PROHIBITED**. Data after `2026-08-05` is **RESERVED / DO NOT ACCESS**.

## No filters

The transition is the complete signal. Do not add Donchian, Bollinger, RSI, volume, an ATR filter, `HIGH_VOLATILITY`, a return threshold, candle confirmation, a quality score, close-location, cooldown, machine learning, or any other filter.

## Frozen exits

| Parameter | Frozen value |
| --- | ---: |
| `take_profit_pct` | `0.012` |
| `stop_loss_pct` | `0.008` |
| `max_holding_candles` | `24` |
| `generate_exits` | `false` |

## Pre-registered verdict

Apply verdicts in this order:

1. **`INSUFFICIENT_SAMPLE`** when `completed_trades < 100`.
2. With at least 100 completed trades, **`BASELINE_REJECT`** if any of `gross_expectancy <= 0`, `net_profit_factor <= 1`, `net_expectancy <= 0`, or `net_pnl <= 0` is true. Also reject when `positive_pnl_concentration_top_2_months > 0.80`.
3. **`BASELINE_CANDIDATE`** only if all conditions pass: at least 100 completed trades, positive gross expectancy, net profit factor above 1, positive net expectancy, positive net PnL, and top-two-month positive-PnL concentration no greater than 0.80.

## Anti-tuning termination rule

Once 2025 results are observed, do not rescue this hypothesis through regime-threshold changes, exit changes, filters, timeframe changes, transition changes, volatility conditions, candle confirmation, or multiple variants. If rejected, this exact hypothesis ends.

No Family #5 strategy, configuration, grid search, or market-data experiment is authorized by this registration.
