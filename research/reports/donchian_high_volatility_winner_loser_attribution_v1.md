# Donchian HIGH_VOLATILITY Winner-vs-Loser Attribution v1

## Scope and frozen methodology

This is descriptive research attribution, not a HIGH_VOLATILITY-only strategy or a production-readiness claim. The executable analysis is `app/research/regimes/high_volatility_winner_loser_attribution.py`. It independently processes each dataset through the official feature and regime pipeline, runs the official simulator once per period with the frozen Donchian 15m candidate (`min_close_location_filter=0.94`; all other parameters unchanged), and selects the original `BacktestTrade` objects only when their entry candle has the existing `is_high_volatility` label. It derives or tests no threshold.

## Execution status

The two required, gitignored source datasets are absent from this checkout. Therefore entry-level feature statistics cannot be computed here without fabricating observations. The companion CSV contains the stable output schema but no fabricated rows. Running the module when both specified files are present deterministically replaces this status report and CSV with the 2025, 2026, and combined feature tables.

## 2025 results

The supplied prior attribution establishes 40 HIGH_VOLATILITY trades, profit factor 1.110841, expectancy +0.041844, and net PnL +1.673758. Winner/loser counts and entry feature records are unavailable from aggregate metrics alone.

## 2026 external OOS results

The supplied prior attribution establishes 15 HIGH_VOLATILITY trades, profit factor 1.293581, expectancy +0.103998, and net PnL +1.559976. Winner/loser counts and entry feature records are unavailable from aggregate metrics alone.

## Combined results

The supplied counts establish 55 HIGH_VOLATILITY trades. Winner/loser feature summaries and correlations cannot be reconstructed from the supplied period aggregates.

## Cross-period assessment

Same-direction features: **not evaluated because entry-level records are unavailable**.

Contradictory features: **not evaluated because entry-level records are unavailable**.

The executable marks a row as sample-dominated whenever either outcome group has fewer than 10 valid feature observations.

## Limitations

Only 55 HIGH_VOLATILITY trades were reported across the periods, and splitting winners from losers makes each comparison smaller. The hypothesis and periods are not discovery-independent. Correlation is univariate, sensitive to outliers and feature collinearity, and is not causal. The 2026 dataset is partial through 2026-08-05. Pooling periods can conceal distribution shifts. Multiple descriptive comparisons increase the chance of incidental patterns. No uncertainty interval or multiplicity-adjusted inference is claimed.

## Recommended next research step

Supply the two exact source datasets and execute this fixed attribution. After a human validates its persisted outputs, pre-register replication of any directional hypotheses on untouched later data while retaining the candidate, regime definition, features, simulator, exits, fees, and sizing unchanged. Do not translate this report into thresholds. This change does not implement the replication.
