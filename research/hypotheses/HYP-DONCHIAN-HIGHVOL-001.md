# HYP-DONCHIAN-HIGHVOL-001: Donchian HIGH_VOLATILITY Replication

## Registration

- **Hypothesis ID:** `HYP-DONCHIAN-HIGHVOL-001`
- **Status:** `PRE_REGISTERED_NOT_TESTED`
- **Purpose:** Prospectively replicate directional winner-versus-loser associations for the frozen Donchian candidate in the existing `HIGH_VOLATILITY` overlay. This registration is a research protocol, not a trading filter.

## Discovery basis and separation from confirmation

The discovery analysis attributed 55 `HIGH_VOLATILITY` candidate trades across 2025 and the partial 2026 dataset. The 2025 and 2026 data are discovery data and cannot be used for confirmation, threshold selection, feature weighting, model selection, or any other confirmatory decision under this registration.

The candidate, regime definition, entry features, simulator behavior, exits, fees, and sizing must remain unchanged for the replication. This document does not authorize a strategy, simulator, or regime implementation change.

## Frozen strategy

| Parameter | Frozen value |
| --- | ---: |
| Timeframe | `15m` |
| `lookback` | `3` |
| `volume_ratio` | `0.4` |
| `take_profit_pct` | `0.012` |
| `stop_loss_pct` | `0.008` |
| `max_holding_candles` | `24` |
| `min_quality_score` | `0` |
| `min_close_location_filter` | `0.94` |

## Frozen `HIGH_VOLATILITY` definition

The replication must use exactly the existing regime implementation at the time of registration:

1. Use `realized_volatility_20` as the observed volatility series.
2. For each candle, calculate `high_volatility_threshold` as the rolling 90th percentile (`high_volatility_percentile=0.90`) over 500 observations (`volatility_lookback=500`), requiring at least 200 observations (`volatility_min_history=200`).
3. Shift that rolling threshold by one candle, so the current observation is not used to construct its own threshold.
4. Set `is_high_volatility` to true only when `realized_volatility_20` is strictly greater than the shifted threshold. Missing comparisons are filled as false.
5. Treat `HIGH_VOLATILITY` as an overlay rather than an exclusive base regime, and attribute a candidate trade using only `is_high_volatility` on its entry candle.

No part of this definition may be recalibrated on the confirmation dataset.

## Pre-registered directional hypotheses

For each feature, “winner” and “loser” retain the existing attribution definitions, and the comparison is made among the new `HIGH_VOLATILITY` candidate trades at their entry candles. Only direction is registered; magnitude is not.

| Feature | Expected winner-versus-loser direction |
| --- | --- |
| `volatility_threshold_ratio` | Higher values are expected to associate with better outcomes. |
| `range_expansion_ratio` | Lower values are expected to associate with better outcomes. |
| `body_to_range` | Higher values are expected to associate with better outcomes. |
| `volume_ratio` | Lower values are expected to associate with better outcomes. |
| `ema_alignment_strength` | Lower values are expected to associate with better outcomes. |
| `ema20_ema50_separation` | Lower values are expected to associate with better outcomes. |
| `ema50_ema200_separation` | Lower values are expected to associate with better outcomes. |
| `adx14` | Lower values are expected to associate with better outcomes. |

Contradictory discovery features, including RSI, ATR, EMA slope, and breakout distance, are excluded from the confirmatory directional count.

## Decisions frozen before confirmation

- No numeric feature thresholds have been selected.
- No feature weights have been selected.
- No machine-learning model has been selected.
- No trading filter is specified or authorized.
- Feature directions will be evaluated as registered; they must not be converted into post hoc cutoffs.

## Confirmation dataset and evaluation gate

The confirmation dataset is BTCUSDT market data strictly later than the current partial 2026 dataset end of **2026-08-05**. It must not contain any candle at or before that date and must be untouched by discovery analysis or tuning.

Evaluation may begin only after the unchanged pipeline produces **at least 50 new `HIGH_VOLATILITY` candidate trades** from that confirmation dataset. Results must not be inspected, scored, or used to revise this protocol before the minimum sample is complete.

## Success criteria

Confirmation succeeds only if all of the following hold on the eligible confirmation sample:

1. At least 6 of the 8 registered winner-versus-loser directions reproduce.
2. Candidate profit factor is greater than `1`.
3. Candidate expectancy is greater than `0`.
4. Candidate net PnL is greater than `0`.
5. There is no material max-drawdown deterioration relative to the frozen candidate's discovery benchmark. The drawdown comparison must be reported explicitly; this registration does not introduce a post hoc numeric tolerance.

## Failure criteria

Confirmation fails if any of the following hold on the eligible confirmation sample:

1. Fewer than 4 of the 8 registered directions reproduce.
2. Candidate profit factor is less than or equal to `1`.
3. Candidate expectancy is less than or equal to `0`.

An outcome in which 4 or 5 directions reproduce, while none of the economic failure criteria apply, is inconclusive rather than successful. Material drawdown deterioration prevents a success designation even when it does not independently meet the failure definition above.

## Anti-tuning warning

**Do not tune thresholds after observing the confirmation dataset.** Do not select feature cutoffs, weights, transformations, models, exclusions, or alternate regime/strategy parameters in response to confirmation results and then describe those results as confirmation of this hypothesis. Any such work is a new discovery exercise and requires a separately pre-registered, later untouched dataset.
