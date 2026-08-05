# Donchian Breakout 15m Close-Location Walk-Forward v1

## 1. Methodology

This validation fixes `donchian_breakout` at lookback 3, volume ratio 0.4, take profit 0.012, stop loss 0.008, maximum holding 24 candles, and minimum quality score 0. It compares only close-location filters 0.00 and 0.94. Training is informational; no fitting or optimization occurs.

Execution is pending because this checkout does not contain the ignored source file `data/BTCUSDT_1m.csv`. No result is fabricated from the prior in-sample experiment.

## 2. Exact window boundaries

Intervals are half-open. Based on the repository's documented one-year 2025 dataset coverage, the complete schedule is:

| Window | Train start | Train end | Test start | Test end |
|---:|:---|:---|:---|:---|
| 1 | 2025-01-01T00:00:00+00:00 | 2025-07-01T00:00:00+00:00 | 2025-07-01T00:00:00+00:00 | 2025-10-01T00:00:00+00:00 |
| 2 | 2025-04-01T00:00:00+00:00 | 2025-10-01T00:00:00+00:00 | 2025-10-01T00:00:00+00:00 | 2026-01-01T00:00:00+00:00 |

## 3. Leakage controls

The official loader, resampler, feature pipeline, strategy, simulator, trade model, and metric calculator are used. Each simulation receives historical warm-up candles, but entries are masked to `[test_start, test_end)`. No pre-test position can consume simulator state. The simulator receives no candle at or after `test_end`; an open position is therefore closed at the last in-window close and recorded as `max_holding`. Test results never affect parameters.

## 4. Window-level baseline results

Pending source data. The companion CSV contains the stable output schema but no fabricated rows.

## 5. Window-level filtered results

Pending source data. Exactly the 0.94 filter will be evaluated.

## 6. Aggregate OOS comparison

Pending source data. The implementation concatenates independent OOS trades and recomputes gross profit, gross loss, fees, net PnL, Profit Factor, expectancy, and drawdown from the aggregate equity sequence. It does not average window Profit Factors or drawdowns.

## 7. Trade-retention analysis

Pending source data. Aggregate retention is filtered executed trades divided by baseline executed trades. The filtered entry signal predicate is tested as a strict subset of baseline signals.

## 8. Sample-size limitations

One year supplies only two complete OOS windows after a six-month initial training span. This is a severe regime-coverage and statistical-power limitation even before the final trade count is known.

## 9. Deterministic verdict

**NOT_EVALUATED_SOURCE_DATA_MISSING**

The implementation emits `REJECT`, `PROMISING_BUT_UNCONFIRMED`, or `CANDIDATE_FOR_MONTE_CARLO` only after real OOS trades exist. Emitting a requested verdict without the source data would fabricate evidence.

## 10. Recommended next step

Place the original `data/BTCUSDT_1m.csv` in this checkout and run the documented command unchanged. If the result remains limited to two windows, collect additional unseen years before Monte Carlo validation.
