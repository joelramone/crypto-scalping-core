# HYP-MOMENTUM-PULLBACK-BASELINE-001

## Hypothesis

On BTCUSDT 15-minute candles, a bullish continuation through the prior high after a bearish pullback to EMA20, while EMA20 > EMA50 > EMA200, has positive expectancy after the official simulator's fees. This is a long-only Stage 1 baseline, not an optimization.

## Frozen strategy rules

For completed candle `t`, trend requires `ema20[t] > ema50[t]` and `ema50[t] > ema200[t]`. Candle `t-1` must satisfy `low[t-1] <= ema20[t-1]`, `close[t-1] > ema50[t-1]`, and `close[t-1] < open[t-1]`. Candle `t` must satisfy `close[t] > open[t]` and `close[t] > high[t-1]`, with trend alignment still valid at `t`. The entry signal occurs at `t`; no future candle participates. There are no tolerances or additional filters.

`generate_exits()` is always false. Frozen simulator exits are take profit `0.012`, stop loss `0.008`, and maximum holding period `24` candles. No alternatives may be tested.

## Experiment boundary

- Symbol: `BTCUSDT`
- Timeframe: `15m`
- Source: `data/BTCUSDT_1m.csv`
- `2025 = DISCOVERY_USED`
- `2026-01-01 through 2026-08-05 = VALIDATION_USED / NOT USED IN BASELINE`
- `post-2026-08-05 = RESERVED / NOT TOUCHED`
- The separate raw 2026 file is excluded from this baseline.

Only the single frozen configuration is permitted. Validation, walk-forward, Monte Carlo, parameter grids, tuning, and Stage 2 are outside this experiment. Results must not be used to change this baseline.

## Simulator and accounting assumptions

`app/research/simulation.py` is the unchanged official simulator. It enters at `close[t]`; applies long-only TP, SL, then maximum-holding semantics; uses fixed notional of 100 USDT; and charges the configured fee rate of `0.0004` on entry and exit notionals. Position sizing, execution, fees, and accounting come exclusively from that simulator. Slippage is absent and therefore remains a known limitation.

## Data contamination boundaries and stop rules

The baseline may consume 2025 discovery data only. The 2026 validation interval must not be used in discovery, and reserved post-2026-08-05 data must not be inspected or accessed.

Verdict rules are deterministic: fewer than 100 completed trades is `INSUFFICIENT_SAMPLE`. Otherwise, Profit Factor <= 1.0, expectancy <= 0, or net PnL <= 0 is `BASELINE_REJECT`. If those checks pass but one calendar month contributes more than 80% of total positive monthly PnL, the verdict is `BASELINE_REJECT`; otherwise it is `BASELINE_CANDIDATE`. No filters may be proposed or added after rejection.
