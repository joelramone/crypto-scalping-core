# Donchian HIGH_VOLATILITY Attribution v1

## 1. Objective
Determine whether Donchian entry trades in the existing HIGH_VOLATILITY overlay degrade economics.

## 2. Frozen methodology
The executable experiment is implemented in `app/research/regimes/high_volatility_attribution.py`. It runs the official simulator once per period and frozen variant, then partitions the returned `BacktestTrade` objects solely by the entry candle's Phase 1 label. No threshold or trading rule is changed.

## 3. 2025 results
Unavailable in this checkout: `data/BTCUSDT_1m.csv` is gitignored and absent. No metrics were fabricated.

## 4. 2026 results
Unavailable in this checkout: `data/BTCUSDT_1m_2026-01-01_through_2026-08-05_binance_usdm_raw.csv` is gitignored and absent. No metrics were fabricated.

## 5. Frozen 0.94 candidate attribution
Unavailable until the same two source files are supplied. The executable uses only the already-frozen 0.94 candidate parameters.

## 6. Cross-period comparison
Unavailable until both independent periods can be executed.

## 7. Loss concentration analysis
Unavailable. The generated report will distinguish losing-trade share from the share of absolute losing net PnL.

## 8. Evidence for/against HIGH_VOLATILITY avoidance
Not evaluated without the official source records.

## 9. Limitations
The required gitignored market datasets are not present. The existing summary statistics are insufficient to reconstruct official trades, fees, holding periods, drawdown, or candidate partitions.

## 10. Deterministic verdict
**NOT_EVALUATED_DATA_UNAVAILABLE**. This is an execution status, not a fourth evidentiary verdict.

## 11. Recommended next experiment
First run this fixed attribution with the two exact source datasets. Only after a deterministic verdict should an unchanged avoidance rule be pre-registered on untouched data; this attribution does not authorize a production filter.
