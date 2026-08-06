# Donchian Breakout 15m Close-Location Monte Carlo v1

## Execution status

The implementation is complete, but this checkout does not contain the required ignored
source dataset, `data/BTCUSDT_1m.csv`. No trade returns or simulation rows have been
fabricated from summary statistics. Run the documented module command in a checkout that
contains the source dataset to reconcile the official 112 filtered OOS `BacktestTrade`
records and replace this status report and the header-only CSV with deterministic results.

## 1. Objective

Pending source data.

## 2. Exact source trades

Pending source data; only filtered OOS records will be accepted.

## 3. Reconciliation

Not executed; the command fails before simulation when source data is absent or mismatched.

## 4. Simulation methods

Permutation and bootstrap are implemented with 10,000 simulations each by default.

## 5. Assumptions

Starting capital is 100 USDT and ruin equity is 80 USDT by default. PnL is not resized.

## 6. Permutation results

Pending source data.

## 7. Bootstrap results

Pending source data.

## 8. Drawdown distribution

Pending source data.

## 9. Losing-streak distribution

Pending source data.

## 10. Tail-risk analysis

Pending source data.

## 11. Sample-size warning

The expected 112 OOS trades are a small sample, especially for tail estimates.

## 12. Deterministic verdict

Not emitted without reconciled source records.

## 13. Recommended next validation

Run the fixed command with the original source data, then collect additional untouched OOS trades.
