# HYP-AGGRESSOR-FLOW-PERSISTENCE-001: Aggressor Flow Persistence V1 (kline proxy)

## Registration

- **Hypothesis ID:** `HYP-AGGRESSOR-FLOW-PERSISTENCE-001`
- **Protocol file:** `research/hypotheses/aggressor_flow_persistence_v1.md` (machine-readable twin: `aggressor_flow_persistence_v1.yaml`)
- **Status:** `BASELINE_REJECT` (Dataset A run 2026-11-20, result: `research/backtests/aggressor_flow_persistence_v1_dataset_a.json`). Hypothesis ended per the anti-tuning rule.
- **Purpose:** `discovery_baseline`
- **Origin:** Candidate A ("Aggressive-flow information relative to available liquidity") in `research/reports/scalping_new_edge_discovery_v1.md`
- **Symbol:** `BTCUSDT` (Binance USD-M perpetual)
- **Signal timeframe:** `5m`, resampled causally from `1m`
- **Direction:** two separately evaluated single-direction legs: `LONG` and `SHORT`
- **Registered:** 2026-11-20, before any outcome on Dataset A was computed or inspected

This document freezes one deterministic test: does **persistent one-sided taker (aggressor) flow**, observed only after it has happened, predict a signed price continuation that survives an execution delay and the full fee burden? It does not claim profitability and does not authorize optimization.

## Scope limitation: this is a kline proxy, not the full Candidate A

Candidate A needs individual trade prints, sequenced L2 depth and receipt timestamps. None of those are available. This V1 uses only the per-candle aggregate `taker_buy_base_volume` from Binance klines. That proxy:

- keeps the total taker-buy and taker-sell volume per candle;
- loses event order inside the candle, depth, replenishment and queue information;
- can therefore only test the **coarse** form of the mechanism.

A `BASELINE_REJECT` here rejects the kline-proxy form only. It does not by itself reject a future print/L2 study. A `BASELINE_CANDIDATE` here does not validate execution at size, because the simulator has no spread, slippage, impact or partial fills.

## Data boundaries

| Role | Path | Period (UTC) | Use |
| --- | --- | --- | --- |
| **Dataset A (discovery)** | `data/binance_usdm_BTCUSDT_1m_20230101_20230415.csv` | `2023-01-01T00:00:00Z` to `2023-04-15T00:00:00Z`, end exclusive | Authorized by TASK-032B. 149,760 1m candles. SHA-256 `2a49857054bf14663fd2a91a6980254883be6d3d84e5147ff5750155a40607af`. One baseline run per leg. |

- The run must verify the SHA-256 above before computing anything. If it does not match, the run stops.
- Prohibited for this hypothesis: calendar year 2025 (`DISCOVERY_USED` elsewhere), `2026-01-01` onward (used or reserved elsewhere), and `2024-10-01` to `2024-10-08` (TASK-019B technical validation only).
- Data after `2023-04-15T00:00:00Z` must not be downloaded or inspected for this hypothesis. Any confirmation dataset needs a separate owner authorization and a separate registration, and may only be chosen after this verdict.

## Frozen feature definitions

All features are computed on completed `5m` candles. A `5m` candle `t` is built from the five `1m` candles with open times `[T, T+5m)`, labeled by its open time `T`:

- `open` = first `1m` open; `high` = max; `low` = min; `close` = last `1m` close
- `volume` = sum of `1m` `volume`; `taker_buy` = sum of `1m` `taker_buy_base_volume`

Aggressor imbalance, in `[-1, 1]`:

```text
imb[t] = (2 * taker_buy[t] - volume[t]) / volume[t]
```

If `volume[t] == 0`, then `imb[t]` is undefined and no signal can use candle `t`.

## Frozen signal rule

Fixed constants, chosen before data inspection and never tuned: `THETA = 0.10`, `K = 3`.

```text
long_state[t]  = imb[t] >= +THETA AND imb[t-1] >= +THETA AND imb[t-2] >= +THETA
short_state[t] = imb[t] <= -THETA AND imb[t-1] <= -THETA AND imb[t-2] <= -THETA

LONG  signal at t:  long_state[t]  AND NOT long_state[t-1]
SHORT signal at t:  short_state[t] AND NOT short_state[t-1]
```

- Only the **onset** of persistence is a signal. A continuing state does not create new signals.
- The signal is evaluated after candle `t` is complete. No information from `t+1` or later may be used.
- The first `K` candles of Dataset A cannot signal (warm-up).
- No price-pattern, breakout, EMA, RSI, Bollinger, ATR, volatility, volume-level, trade-count, time-of-day or regime condition may be added. Adding any of them breaks the independence requirement of Candidate A.

## Frozen execution and exits

| Parameter | Frozen value |
| --- | ---: |
| `execution_mode` | `next_bar_open` (entry at `open[t+1]`, one-bar observation delay) |
| `take_profit_pct` | `0.006` |
| `stop_loss_pct` | `0.004` |
| `max_holding_candles` | `12` (one hour on `5m`) |
| `generate_exits` | `false` |
| `fee_rate_per_side` | `0.0004` (repository default, unchanged) |
| Position sizing | Official simulator: fixed 100 USDT notional, one position at a time |

Each leg runs as its own single-direction strategy through the official simulator. Signals that arrive while a position is open are suppressed and counted.

## Mandatory diagnostics

The permanent `TradeDiagnostics` output is mandatory **for each leg**. It must contain the same blocks as `HYP-REGIME-TRANSITION-001`:

- **Gross/net:** `gross_pnl_before_fees`, `total_fees`, `net_pnl`, `gross_expectancy`, `fee_expectancy`, `net_expectancy`, `gross_profit_factor`, `net_profit_factor`
- **Payoff:** wins, losses, flats, win rate, average and median winner, average and median loser, payoff ratio, break-even win rate, actual-minus-break-even win rate
- **Exits:** count, percentage and PnL for TP, SL and max-holding; holding duration average, median, P25, P75, P95
- **Overlap:** `raw_entry_signals`, `completed_trades`, `suppressed_signals`, `suppression_rate`, `raw_signals_per_opened_trade`
- **Monthly:** trades, profit factor, expectancy and net PnL by month; positive and negative month counts; `positive_pnl_concentration_top_2_months`

Additional descriptive diagnostics (no pass threshold):

- **Contemporaneous vs. post-observation move:** for each raw signal, the signed return from `open[t-2]` to `close[t]` (the move during the observation window) and the signed return from `open[t+1]` to `close[t+12]` (the post-observation move). Report mean, median and the fraction of signals with a positive post-observation move. If the edge sits only in the first number, the mechanism is just measuring the move that already happened.
- **Imbalance distribution:** the share of `5m` candles with `|imb| >= THETA`, so readers can see how selective the threshold is. This is reported only and must not be used to change `THETA`.

## Pre-registered verdict

Applied to each leg separately, in this order:

1. **`INSUFFICIENT_SAMPLE`:** `completed_trades < 100`.
2. **`BASELINE_REJECT`:** `completed_trades >= 100` and any of `net_profit_factor <= 1`, `net_expectancy <= 0`, `net_pnl <= 0`, `gross_expectancy <= 0`, or `positive_pnl_concentration_top_2_months > 0.80`.
3. **`BASELINE_CANDIDATE`:** all of `completed_trades >= 100`, `gross_expectancy > 0`, `net_profit_factor > 1`, `net_expectancy > 0`, `net_pnl > 0` and `positive_pnl_concentration_top_2_months <= 0.80`.

Hypothesis-level verdict:

- **`BASELINE_CANDIDATE`** only if at least one leg is `BASELINE_CANDIDATE` **and** the opposite leg has `gross_expectancy > 0`. The mechanism is symmetric. A one-sided result where the mirror leg loses before fees is treated as likely market drift during the sample, not flow persistence.
- **`INSUFFICIENT_SAMPLE`** if neither leg reaches 100 completed trades.
- **`BASELINE_REJECT`** otherwise.

Both legs must always be reported. Reporting only the better leg is prohibited.

## Anti-tuning termination rule

After any outcome on Dataset A has been seen, it is prohibited to:

- change `THETA`, `K`, the `5m` timeframe, the imbalance formula or the onset rule;
- change TP, SL, holding period, execution mode or fees;
- add any filter listed under the signal rule;
- switch to a different period inside or outside Dataset A, or drop months, weeks or days;
- re-run under a new name with cosmetic changes.

If the hypothesis-level verdict is `BASELINE_REJECT`, this hypothesis ends. A `BASELINE_CANDIDATE` only permits requesting authorization for an untouched confirmation dataset, under these same frozen rules.
