# HYP-MEAN-REVERSION-BASELINE-001: Volatility Exhaustion Baseline

## Registration

- **Hypothesis ID:** `HYP-MEAN-REVERSION-BASELINE-001`
- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Research classification:** `DISCOVERY_BASELINE`
- **Strategy:** `volatility_exhaustion`
- **Direction:** long only
- **Timeframe:** `15m`

This protocol is frozen before the executable strategy is implemented or its results are inspected. It defines an independent mean-reversion research family and does not authorize changes to the frozen Donchian branch.

## Discovery data boundary

The only dataset permitted for this baseline is `data/BTCUSDT_1m.csv`, resampled through the official pipeline to 15-minute candles. Its 2025 observations are already discovery-contaminated and may be reused only for discovery.

- **2025:** DISCOVERY DATA
- **2026-01-01 through 2026-08-05:** NOT USED IN THIS EXPERIMENT
- **Post-2026-08-05:** RESERVED / NOT USED

No 2026 market data may be read during this baseline evaluation. Donchian reserved future-confirmation data is prohibited.

## Frozen entry rule

A long entry exists at candle `t` if and only if both conditions hold:

1. `close[t-1] < bb_lower[t-1]`
2. `close[t] >= bb_lower[t]`

The official Bollinger feature-pipeline conventions remain unchanged: `bb_mid` is the rolling 20-close mean, `bb_std` is the rolling standard deviation, and `bb_upper`/`bb_lower` are the existing pipeline bands. Bollinger lookback and multiplier are not strategy parameters. Entry uses the official simulator's normal entry-at-close behavior.

No RSI, volume-ratio, ATR, trend, EMA, ADX, regime, candle-score, close-location, z-score, VWAP, machine-learning, or short-entry condition is permitted.

## Frozen exits and simulation

| Parameter | Frozen value |
| --- | ---: |
| `take_profit_pct` | `0.003` |
| `stop_loss_pct` | `0.002` |
| `max_holding_candles` | `25` |

These values reuse the existing `BollingerReversionStrategy` exit defaults rather than being selected after observing baseline results. `generate_exits()` is always false. Only the official simulator controls sizing, fees, take profit, stop loss, maximum holding time, and trade accounting.

## Required baseline reporting

The single frozen run must report total candles, feature rows after warm-up, trades, wins, losses, win rate, gross profit, gross loss, fees, profit factor, expectancy, net PnL, max drawdown, average holding candles, exit-reason distribution, and monthly trade count, profit factor, expectancy, and net PnL.

## Deterministic stop rules

- **INSUFFICIENT_SAMPLE:** fewer than 100 trades.
- **BASELINE_REJECT:** at least 100 trades and any of profit factor `<= 1`, expectancy `<= 0`, or net PnL `<= 0`.
- **BASELINE_CANDIDATE:** at least 100 trades; profit factor `> 1`; expectancy `> 0`; net PnL `> 0`; and performance is not almost entirely concentrated in one month.

For determinism, “almost entirely concentrated in one month” means that one calendar month contributes more than 80% of positive monthly net PnL. If there is no positive monthly net PnL, the positive economic gates already prevent candidate status.

No optimization follows a `BASELINE_REJECT`. Failure must not be reinterpreted as authorization to add filters, change parameters, or implement Stage 3.
