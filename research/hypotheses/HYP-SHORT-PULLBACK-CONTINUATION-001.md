# HYP-SHORT-PULLBACK-CONTINUATION-001: Short Pullback Continuation Baseline

## Registration

- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Purpose:** `short_pullback_continuation_discovery_baseline`
- **Strategy family:** `STRATEGY_FAMILY_6_SHORT_PULLBACK_CONTINUATION`
- **Symbol:** `BTCUSDT`
- **Timeframe:** `15m`
- **Direction:** **SHORT ONLY**

This document pre-registers Family #6. It freezes one baseline specification without implementing the strategy, reading market outcomes, or authorizing `EXP-000022`.

## Frozen hypothesis and exact entry rule

On completed candle `t`, create a raw short-entry signal if and only if every condition below is true:

```text
ema20[t] < ema50[t]
AND ema50[t] < ema200[t]
AND high[t-1] >= ema20[t-1]
AND close[t-1] < ema50[t-1]
AND close[t-1] > open[t-1]
AND close[t] < open[t]
AND close[t] < low[t-1]
```

The EMA values are the existing causal EMA20, EMA50, and EMA200 features. Candle `t-1` is the bullish pullback: it touches or exceeds EMA20, closes below EMA50, and closes above its open. Candle `t` is the bearish continuation confirmation: it closes below its open and strictly below the prior candle's low while bearish EMA alignment remains valid.

All comparisons are exact. There are no tolerances, cross-within-candle interpretations, alternative patterns, or additional entry filters. The signal is evaluated only after candle `t` is complete; no future candle participates.

## Frozen execution

Execution must use `app/research/simulation.py` unchanged:

- Enter short at `close[t]` with fixed notional `100 USDT`.
- Permit at most one open position. Raw signals occurring while a position is open are suppressed by the simulator.
- Begin exit evaluation at candle `t+1`; the entry candle cannot trigger an exit.
- Use short take-profit at `entry_price * (1 - 0.012)` and short stop-loss at `entry_price * (1 + 0.008)`.
- If stop-loss and take-profit are both touched in one candle, apply stop-loss first.
- If neither price exit triggers, close at the close of `t+24` or the final available candle, whichever comes first.
- `generate_exits()` must always return false, so no strategy exit is allowed.
- Charge fee rate `0.0004` on entry and exit notionals exactly as the official simulator does.
- Do not add slippage, spread, leverage, compounding, alternate position sizing, or execution assumptions.

| Parameter | Frozen value |
| --- | ---: |
| `take_profit_pct` | `0.012` |
| `stop_loss_pct` | `0.008` |
| `max_holding_candles` | `24` |
| `generate_exits` | `false` |
| `fixed_notional_usdt` | `100` |
| `fee_rate` | `0.0004` |

## Data boundary

The sole discovery source is `data/BTCUSDT_1m.csv`, restricted to calendar year `2025` and causally resampled to `15m`. It is classified `DISCOVERY_USED` and may be consumed by one future baseline run only after implementation, tests, and human review are complete.

The interval `2026-01-01` through `2026-08-05`, inclusive, is **PROHIBITED** for this baseline. Data strictly after `2026-08-05` is **RESERVED / DO NOT ACCESS**. The separate raw 2026 file, if present, must not be opened, profiled, or used. Dataset selection must occur before feature calculation, and no metric from a prohibited or reserved interval may influence this hypothesis.

## Mandatory diagnostics

The future baseline report must persist the permanent `TradeDiagnostics` output: gross and net PnL, fees, gross and net expectancy, gross and net profit factor, wins/losses/flats, win rate, payoff statistics, break-even win rate, exit counts and PnL, holding-duration statistics, raw and suppressed signal counts, suppression rate, and monthly trade/PnL diagnostics including top-two positive-PnL concentration.

## Pre-registered verdict policy

Apply exactly one verdict in this order:

1. **`INSUFFICIENT_SAMPLE`** when `completed_trades < 100`. Stop evaluation; this is not a candidate and does not authorize a rescue variant.
2. With at least 100 completed trades, **`BASELINE_REJECT`** if any of `gross_expectancy <= 0`, `net_profit_factor <= 1`, `net_expectancy <= 0`, `net_pnl <= 0`, or `positive_pnl_concentration_top_2_months > 0.80` is true.
3. **`BASELINE_CANDIDATE`** only when all conditions pass: `completed_trades >= 100`, `gross_expectancy > 0`, `net_profit_factor > 1`, `net_expectancy > 0`, `net_pnl > 0`, and `positive_pnl_concentration_top_2_months <= 0.80`.

No other metric may override the deterministic verdict.

## Anti-tuning termination rule

Only one parameter combination and one 2025 baseline run are permitted. After any 2025 result is observed, do not change the entry inequalities, EMA periods, direction, timeframe, TP, SL, holding period, fee or sizing assumptions, or data boundary. Do not add trend-strength, volatility, volume, RSI, ATR, Donchian, Bollinger, return, candle-shape, close-location, quality-score, cooldown, regime, machine-learning, or other filters. Do not run near-neighbor values, sensitivity grids, alternative confirmations, or rescue variants.

If the verdict is `BASELINE_REJECT` or `INSUFFICIENT_SAMPLE`, this exact hypothesis ends. A materially different idea requires a new hypothesis ID and new human approval; it may not be represented as a continuation of this baseline.

## Authorization boundary

This preregistration authorizes documentation only. It does **not** authorize strategy implementation, configuration creation, tests against market data, optimization, backtesting, report generation, or creation/execution of `EXP-000022`. Those stages require separate human authorization in sequence.
