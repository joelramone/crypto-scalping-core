# Market Regime Analysis v1

## 1. Executive summary

The executable Phase 1 analysis is implemented, but numerical research results could not be generated in this checkout because both requested, gitignored source datasets are absent. No substitute dataset was used and no metrics were fabricated. Run the documented command with the two requested files to replace this status artifact with the complete deterministic report.

## 2. Regime methodology

The generator classifies exclusive `TREND_UP`, `TREND_DOWN`, `RANGE`, and `NEUTRAL` base states. `HIGH_VOLATILITY` is a separate overlay so volatility can be analyzed without discarding directional state.

## 3. Feature definitions

The implementation calculates ATR14, ATR percentage, trailing 20-candle realized volatility, EMA20/50/200, percentage EMA20/50 slopes, EMA20/50 and EMA50/200 separations, ADX14, and trailing 4/12/24-candle returns.

## 4. Thresholds used

Trend ADX is 25; range ADX is 20; maximum absolute range EMA separation is 0.25%; maximum absolute range slope is 0.05%. High volatility is above the trailing 90th percentile over 500 candles after 200 historical observations, using a one-candle-shifted threshold. Defaults were selected without strategy PnL.

## 5. 2025 regime distribution

Unavailable: `data/BTCUSDT_1m.csv` is absent from this checkout.

## 6. 2026 regime distribution

Unavailable: `data/BTCUSDT_1m_2026-01-01_through_2026-08-05_binance_usdm_raw.csv` is absent from this checkout.

## 7. Regime transition analysis

Unavailable until the requested datasets are supplied.

## 8. Strategy × regime matrix

Unavailable until the requested datasets are supplied. The generator uses unchanged defaults and the official research simulator.

## 9. Cross-year comparison

Unavailable until both periods can be independently labeled and evaluated.

## 10. Strongest observations

No numerical observation is reported without the specified source data.

## 11. Weak or contradictory evidence

The absence of local source data is an execution limitation, not market evidence.

## 12. Candidate hypotheses for next experiments

The implemented report will assess trend-conditioned Donchian performance, RANGE-conditioned mean reversion, loss concentration in high volatility, cross-year distribution shifts, and whether consistent regime avoidance warrants a future pre-registered experiment. It does not create filters.

## 13. Warning against same-data optimization

**Do not optimize regime thresholds, strategy parameters, or filters on the 2025/2026 descriptive results.** Freeze any next-experiment rule before testing untouched data.
