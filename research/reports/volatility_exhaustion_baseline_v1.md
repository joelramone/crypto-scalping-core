# Volatility Exhaustion Baseline v1

## Execution status

**NOT EXECUTED — REQUIRED DISCOVERY DATASET UNAVAILABLE**

The frozen baseline could not be evaluated because `data/BTCUSDT_1m.csv` is not present in this checkout. No substitute, synthetic, 2026, post-2026-08-05, or Donchian reserved future-confirmation data was read. An attempt to obtain the permitted 2025 public source was rejected by the execution environment's network proxy with HTTP 403, so no market results are fabricated in this report.

## Frozen specification

- Strategy: `volatility_exhaustion`
- Timeframe: `15m`, produced by the official resampling pipeline
- Direction: long only
- Entry at candle `t`: `close[t-1] < bb_lower[t-1]` and `close[t] >= bb_lower[t]`
- Entry execution: official simulator entry at close
- Take profit: `0.003`
- Stop loss: `0.002`
- Maximum holding: `25` candles
- Strategy exits: always false
- Sizing, fees, TP, SL, maximum holding, and accounting: official simulator only
- Configurations: exactly one; no optimization

## Data contamination boundary

- **2025:** DISCOVERY DATA
- **2026-01-01 through 2026-08-05:** NOT USED IN THIS EXPERIMENT
- **Post-2026-08-05:** RESERVED / NOT USED

## Required aggregate metrics

| Metric | Result |
| --- | ---: |
| Total candles | Not available — dataset missing |
| Feature rows after warm-up | Not available — dataset missing |
| Total trades | Not available — baseline not executed |
| Wins | Not available |
| Losses | Not available |
| Win rate | Not available |
| Gross profit | Not available |
| Gross loss | Not available |
| Fees | Not available |
| Profit Factor | Not available |
| Expectancy | Not available |
| Net PnL | Not available |
| Max drawdown | Not available |
| Average holding candles | Not available |

## Exit-reason distribution

Not available because the baseline was not executed.

## Monthly stability

Monthly trade counts, Profit Factor, expectancy, and net PnL are not available because the baseline was not executed. Concentration in one month therefore cannot be assessed.

## Pre-registered stop rules

- `INSUFFICIENT_SAMPLE` applies only to an executed baseline with fewer than 100 trades.
- `BASELINE_REJECT` applies with at least 100 trades when Profit Factor is at most 1, expectancy is at most 0, or net PnL is at most 0.
- `BASELINE_CANDIDATE` requires at least 100 trades, Profit Factor above 1, positive expectancy, positive net PnL, and no single month contributing more than 80% of positive monthly net PnL.

## Deterministic verdict

**NOT EVALUATED.** No pre-registered performance verdict can be assigned without executing the permitted dataset. This is an environment/data-availability state, not `INSUFFICIENT_SAMPLE`, `BASELINE_REJECT`, or `BASELINE_CANDIDATE`. No optimization, filter addition, parameter proposal, or Stage 3 work was performed.
