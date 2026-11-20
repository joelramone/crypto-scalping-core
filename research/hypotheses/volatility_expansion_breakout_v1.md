# HYP-VOLATILITY-EXPANSION-BREAKOUT-001: Volatility Expansion Breakout V1

## Registration

- **Hypothesis ID:** `HYP-VOLATILITY-EXPANSION-BREAKOUT-001`
- **Protocol file:** `research/hypotheses/volatility_expansion_breakout_v1.md` (machine-readable twin: `volatility_expansion_breakout_v1.yaml`)
- **Status:** `BASELINE_REJECT` (Dataset B run 2026-11-20, result: `research/backtests/volatility_expansion_breakout_v1_dataset_b.json`). Hypothesis ended per the anti-tuning rule.
- **Purpose:** `discovery_baseline`
- **Symbol:** `BTCUSDT` (Binance USD-M perpetual)
- **Signal timeframe:** `15m`, resampled causally from `1m`
- **Direction:** two separately evaluated single-direction legs: `LONG` and `SHORT`
- **Registered:** 2026-11-20, before Dataset B was downloaded. No outcome on any part of Dataset B had been seen.
- **Note:** the original volatility-expansion work (commit `320ac53`) was lost and its specification is unknown. This is a new design, not a reconstruction.

This document freezes one deterministic test: does a single candle whose true range is unusually large relative to the previous day, and which closes near its extreme, start a move in that direction that survives an execution delay and the full fee burden? It does not claim profitability and does not authorize optimization.

## Mechanism

A sudden range expansion that closes at its extreme can reflect new information, or forced flow such as liquidations and stop cascades, that the market has not fully priced by the close. If price discovery takes longer than one candle, the move should continue after the candle closes. It could fail because the expansion is the whole move, or because forced flow reverts once it is exhausted.

## Independence from rejected hypotheses

- **Not Donchian breakout (EXP-000013 to 000015):** there is no channel, and no prior high or low has to be broken.
- **Not compression breakdown (EXP-000023):** no prior squeeze or low-volatility state is required.
- **Not volatility exhaustion:** it bets on continuation of the expansion, not reversion to a band.
- **Not regime transition (EXP-000021):** there is no EMA, ADX or regime classifier.

## Data boundaries

| Role | Path | Period (UTC) | Use |
| --- | --- | --- | --- |
| **Dataset B (discovery)** | `data/binance_usdm_BTCUSDT_1m_20230415_20240101.csv` | `2023-04-15T00:00:00Z` to `2024-01-01T00:00:00Z`, end exclusive | Authorized by the owner as TASK-033B. 375,840 1m candles expected. One baseline run per leg. |

- The SHA-256 of Dataset B is recorded in the YAML twin right after download and validation, before any strategy code touches the data. The run must verify it and stop on a mismatch.
- Prohibited for this hypothesis:
  - Dataset A, `2023-01-01` to `2023-04-15` (used by `HYP-AGGRESSOR-FLOW-PERSISTENCE-001`)
  - `2024-10-01` to `2024-10-08` (TASK-019B technical validation only)
  - calendar year 2025 and everything from `2026-01-01` on (used or reserved elsewhere)
- Data from `2024-01-01` onward that is not listed above stays reserved for a possible future confirmation, with separate authorization and registration.

## Frozen feature definitions

A `15m` candle `t` aggregates the fifteen `1m` candles with open times `[T, T+15m)`, labeled by `T`. Only complete candles (15 one-minute rows) are kept.

```text
TR[t]        = max(high[t] - low[t], |high[t] - close[t-1]|, |low[t] - close[t-1]|)
ATR_prev[t]  = mean(TR[t-96 .. t-1])         # previous 24 hours, current candle excluded
expansion[t] = TR[t] >= 2.0 * ATR_prev[t]
loc[t]       = (close[t] - low[t]) / (high[t] - low[t])
```

`ATR_prev` requires 96 prior candles, so the first 97 candles of Dataset B cannot signal.

## Frozen signal rule

Fixed constants, chosen before any data was downloaded and never tuned: `ATR_LOOKBACK = 96`, `EXPANSION_MULT = 2.0`, `LOC_HIGH = 0.75`, `LOC_LOW = 0.25`.

```text
LONG  signal at t: expansion[t] AND loc[t] >= 0.75 AND close[t] > open[t]
SHORT signal at t: expansion[t] AND loc[t] <= 0.25 AND close[t] < open[t]
```

- The signal is evaluated after candle `t` closes, with no information from `t+1` or later.
- No channel, EMA, RSI, Bollinger, volume, taker-flow, time-of-day, regime, cooldown or machine-learning component may be added.

## Frozen execution and exits

| Parameter | Frozen value |
| --- | ---: |
| `execution_mode` | `next_bar_open` (entry at `open[t+1]`) |
| `take_profit_pct` | `0.010` |
| `stop_loss_pct` | `0.006` |
| `max_holding_candles` | `32` (8 hours on `15m`) |
| `generate_exits` | `false` |
| `fee_rate_per_side` | `0.0004` (repository default, unchanged) |
| Position sizing | Official simulator: fixed 100 USDT notional, one position at a time |

The exits are wider than in the rejected 15m baselines on purpose: the 0.08% round-trip fee is under 10% of the TP distance.

## Mandatory diagnostics

The permanent `TradeDiagnostics` output is mandatory **for each leg**, with the same blocks as `HYP-REGIME-TRANSITION-001`: gross/net, payoff, exits, overlap and monthly, including `positive_pnl_concentration_top_2_months`.

Additional descriptive diagnostics (no pass threshold):

- **Signal-candle move vs. post-signal move:** for each raw signal, the signed return `open[t] -> close[t]` and the signed return `open[t+1] -> close[t+32]`. Report mean, median and fraction positive.
- **Expansion frequency:** the share of eligible `15m` candles with `expansion[t]` true. This is reported only and must not be used to change any constant.

## Pre-registered verdict

Applied to each leg separately, in this order:

1. **`INSUFFICIENT_SAMPLE`:** `completed_trades < 100`.
2. **`BASELINE_REJECT`:** `completed_trades >= 100` and any of `net_profit_factor <= 1`, `net_expectancy <= 0`, `net_pnl <= 0`, `gross_expectancy <= 0`, or `positive_pnl_concentration_top_2_months > 0.80`.
3. **`BASELINE_CANDIDATE`:** all of `completed_trades >= 100`, `gross_expectancy > 0`, `net_profit_factor > 1`, `net_expectancy > 0`, `net_pnl > 0` and `positive_pnl_concentration_top_2_months <= 0.80`.

Hypothesis-level verdict:

- **`BASELINE_CANDIDATE`** only if at least one leg is `BASELINE_CANDIDATE` **and** the opposite leg has `gross_expectancy > 0`. A one-sided result where the mirror leg loses before fees is treated as market drift during the sample.
- **`INSUFFICIENT_SAMPLE`** if neither leg reaches 100 completed trades.
- **`BASELINE_REJECT`** otherwise.

Both legs must always be reported.

## Anti-tuning termination rule

After any outcome on Dataset B has been seen, it is prohibited to:

- change `ATR_LOOKBACK`, `EXPANSION_MULT`, `LOC_HIGH`, `LOC_LOW`, the `15m` timeframe, the TR formula or the direction rule;
- change TP, SL, holding period, execution mode or fees;
- add any filter;
- trim or switch the period;
- re-run under a new name with cosmetic changes.

If the hypothesis-level verdict is `BASELINE_REJECT`, this hypothesis ends. A `BASELINE_CANDIDATE` only permits requesting authorization for an untouched confirmation dataset, under these same frozen rules.
