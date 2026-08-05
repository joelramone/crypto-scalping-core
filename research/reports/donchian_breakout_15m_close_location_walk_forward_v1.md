# Donchian Breakout 15m Close-Location Walk-Forward v1

## Execution status

The fixed walk-forward implementation is complete, but this checkout does not contain the
required ignored source dataset, `data/BTCUSDT_1m.csv`. The environment also rejects access
to the Binance Futures API, so no OOS metric has been fabricated or copied from an in-sample
report. Run the command below in a checkout containing the source dataset to replace this
status report with deterministic results from the official simulator and metric functions.

```bash
python -m app.research.walk_forward.close_location_walk_forward
```

## Fixed validation setup

- Symbol: `BTCUSDT`
- Source: `data/BTCUSDT_1m.csv`
- Timeframe: `15m`
- Strategy: `lookback=3`, `volume_ratio=0.4`, `take_profit_pct=0.012`,
  `stop_loss_pct=0.008`, `max_holding_candles=24`, `min_quality_score=0`
- Variants: baseline `min_close_location_filter=0.0`; filtered
  `min_close_location_filter=0.94`
- Schedule: rolling 6-month train, 3-month test, 3-month step
- No fitting, optimization, or alternate-threshold testing is performed.
- All intervals are half-open: start inclusive, end exclusive.

## Window boundaries

| Window | Train start | Train end | OOS start | OOS end |
|---:|:---|:---|:---|:---|
| 1 | 2025-01-01T00:00:00+00:00 | 2025-07-01T00:00:00+00:00 | 2025-07-01T00:00:00+00:00 | 2025-10-01T00:00:00+00:00 |
| 2 | 2025-04-01T00:00:00+00:00 | 2025-10-01T00:00:00+00:00 | 2025-10-01T00:00:00+00:00 | 2026-01-01T00:00:00+00:00 |

## OOS results

Not executed because the required source dataset is unavailable in this checkout. The report
writer recomputes aggregate OOS metrics from concatenated chronological OOS trades; it never
averages window Profit Factors or drawdowns.

## Verdict

Not emitted without OOS results.
