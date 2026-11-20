# Candle-Signal Failures: EXP-000025 and EXP-000026

## Summary

Two new preregistered mechanisms, each tested once on its own previously untouched dataset, were rejected for the same reason: **the signal measures a move that has already happened, and nothing tradable is left after a one-bar delay.** Both experiments are `CLOSED_REJECTED` with `NO_GROSS_EDGE`. Lower fees cannot rescue either one.

| Experiment | Hypothesis | Dataset | Timeframe | Leg | Trades | Gross PF | Net PF | Gross Exp. (USDT) | Verdict |
|---|---|---|---|---|---:|---:|---:|---:|---|
| [EXP-000025](../journal/EXP-000025.md) | [Aggressor Flow Persistence V1](../hypotheses/aggressor_flow_persistence_v1.md) | A: 2023-01-01 to 2023-04-15 | 5m | LONG | 557 | 0.962 | 0.525 | -0.0050 | BASELINE_REJECT |
| | | | | SHORT | 648 | 0.818 | 0.432 | -0.0240 | BASELINE_REJECT |
| [EXP-000026](../journal/EXP-000026.md) | [Volatility Expansion Breakout V1](../hypotheses/volatility_expansion_breakout_v1.md) | B: 2023-04-15 to 2024-01-01 | 15m | LONG | 375 | 0.832 | 0.659 | -0.0607 | BASELINE_REJECT |
| | | | | SHORT | 327 | 1.003 | 0.785 | +0.0011 | BASELINE_REJECT |

Sources: `research/backtests/aggressor_flow_persistence_v1_dataset_a.json` and `research/backtests/volatility_expansion_breakout_v1_dataset_b.json`.

## The common pattern

Both runners record the signed move **during** the signal window and the signed move **after** entry at the next open:

| Experiment | Leg | Move during signal window | Positive during | Move after entry | Positive after |
|---|---|---:|---:|---:|---:|
| EXP-000025 | LONG | +0.268% | 99.4% | -0.016% | 46.1% |
| EXP-000025 | SHORT | +0.224% | 98.4% | -0.033% | 43.2% |
| EXP-000026 | LONG | +0.462% | 100% | +0.001% | 48.3% |
| EXP-000026 | SHORT | +0.466% | 100% | -0.125% | 47.0% |

The round-trip fee is 0.08%. In every leg, the move after entry is close to zero or slightly adverse, and positive in fewer than half the signals. Taker imbalance and range expansion are two different signals, but both end up selecting candles that already moved, not candles about to move.

## Relation to earlier failures

[EXP-000021 to 000024](failure_patterns_exp_000021_000024.md) were mostly `FEE_DOMINATED`: a small positive gross edge, erased by fees. EXP-000025 and 000026 are worse: no gross edge at all, with independent data, both directions and wider exits (EXP-000026 TP is 1.0%, more than 12 times the round-trip fee). Fee drag is therefore not the only problem. These continuation signals built from candles carry no information about the next move on BTCUSDT.

## Data budget after these experiments (BTCUSDT 1m)

| Period | Status |
|---|---|
| before 2023-01-01 | Untouched. Not downloaded. |
| 2023-01-01 to 2023-04-15 | Used: Dataset A, EXP-000025. |
| 2023-04-15 to 2024-01-01 | Used: Dataset B, EXP-000026. |
| 2024-01-01 to 2024-10-01 | No registered experiment, but present locally in `data/binance_klines.csv` (2024-01-01 to 2026-02-19), the default real-data CSV of `app/main.py`. Exposure is unknown; treat as **possibly seen**, not untouched. |
| 2024-10-01 to 2024-10-08 | TASK-019B technical validation only. |
| 2024-10-08 to 2025-01-01 | Same status as 2024-01-01 to 2024-10-01. |
| 2025 | Discovery-used (EXP-000001 to 000024). |
| 2026-01-01 onward | Used or reserved (`HYP-DONCHIAN-HIGHVOL-001`). |

The only period with a clean claim to be untouched is **before 2023-01-01**. Any new candle hypothesis should be frozen before choosing a period from there, and each period should be used for one hypothesis only.

## Implications for prioritization

- Another continuation rule built from OHLCV candles has a low prior of success. Both mechanisms that looked most independent of the earlier failures still showed no edge after the observation delay.
- The descriptive data suggests a mild reversion after strong candles. Testing the inverted signals on the same datasets is **prohibited** post-hoc tuning. A reversion hypothesis would need its own mechanism, preregistration and an untouched period.
- The [microstructure](microstructure_data_execution_feasibility_v1.md) and [funding carry](funding_carry_decision_gate_v1.md) tracks remain blocked on external data and documentation, not on research effort.

## Bookkeeping

- `research/memory/index.csv` gains rows for EXP-000025 and EXP-000026. It also gains the previously missing EXP-000024 row, copied from its [journal](../journal/EXP-000024.md). Without that row, `generate_next_experiment_id` returned `EXP-000024` again and would have reused an existing ID. The next allocated ID is now `EXP-000027`.
- No leaderboard CSV is written: both experiments used one-shot runners, not the grid-search optimizer, and their full output is in the JSON result files above. The index rows record the better leg by net PF, and `best_max_drawdown` is left blank because the runners do not report it.
