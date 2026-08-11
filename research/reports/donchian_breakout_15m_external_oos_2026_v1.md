# Donchian Breakout 15m External OOS 2026 v1

## Frozen setup and data boundary

- Symbol: `BTCUSDT`; timeframe: `15m`.
- Source: `data/BTCUSDT_1m_2026-01-01_through_2026-08-05_binance_usdm_raw.csv` only (312480 one-minute rows).
- Source interval: `2026-01-01T00:00:00+00:00` through `2026-08-05T23:59:00+00:00`.
- Fixed strategy: lookback=3, volume_ratio=0.4, take_profit_pct=0.012, stop_loss_pct=0.008, max_holding_candles=24, min_quality_score=0.
- Variants: baseline filter=0.00 and candidate filter=0.94. No optimization or alternate threshold was run.
- Warm-up: the official feature pipeline runs on the full 2026 source after official 15m resampling; rows with incomplete indicators are then dropped. Thus only earlier 2026 candles warm features, and no 2025 candle or trade is used.
- Trades are produced by the official `simulate_strategy` simulator as `BacktestTrade` records; all metrics below are recomputed from those records.
- Monthly partitions use the UTC month of each trade's opening timestamp. August ends with the source on 2026-08-05.

## Aggregate 2026 results

| Variant   | Filter | Trades | Wins | Losses | Win rate | Gross profit | Gross loss |      Fees |       PF | Expectancy |    Net PnL |    Max DD | Avg holding candles | Retention |
| :-------- | -----: | -----: | ---: | -----: | -------: | -----------: | ---------: | --------: | -------: | ---------: | ---------: | --------: | ------------------: | --------: |
| baseline  |   0.00 |    343 |  134 |    209 |   39.07% |   109.129724 | 142.732163 | 27.437534 | 0.764577 |  -0.097966 | -33.602439 | 36.026243 |               14.05 |   100.00% |
| candidate |   0.94 |     97 |   38 |     59 |   39.18% |    30.935890 |  39.401768 |  7.759718 | 0.785140 |  -0.087277 |  -8.465878 | 10.980821 |               14.61 |    28.28% |

## Monthly results (UTC entry month)

| Month   | Variant   | Filter | Trades | Win rate |       PF | Expectancy |    Net PnL |    Max DD |
| :------ | :-------- | -----: | -----: | -------: | -------: | ---------: | ---------: | --------: |
| 2026-01 | baseline  |   0.00 |     34 |   35.29% | 0.688839 |  -0.127454 |  -4.333438 |  6.195571 |
| 2026-01 | candidate |   0.94 |     13 |   15.38% | 0.227539 |  -0.448109 |  -5.825422 |  5.825422 |
| 2026-02 | baseline  |   0.00 |     41 |   34.15% | 0.723265 |  -0.133038 |  -5.454571 |  9.501715 |
| 2026-02 | candidate |   0.94 |     11 |   45.45% | 1.578566 |   0.159477 |   1.754250 |  1.844150 |
| 2026-03 | baseline  |   0.00 |     60 |   45.00% | 1.067565 |   0.028835 |   1.730088 |  5.631334 |
| 2026-03 | candidate |   0.94 |     14 |   64.29% | 2.069084 |   0.335876 |   4.702258 |  1.759360 |
| 2026-04 | baseline  |   0.00 |     58 |   50.00% | 1.127437 |   0.045153 |   2.618890 |  3.362248 |
| 2026-04 | candidate |   0.94 |     17 |   35.29% | 0.549791 |  -0.201336 |  -3.422704 |  3.502117 |
| 2026-05 | baseline  |   0.00 |     43 |   37.21% | 0.534714 |  -0.173327 |  -7.453074 |  9.087183 |
| 2026-05 | candidate |   0.94 |     11 |   27.27% | 0.211344 |  -0.280028 |  -3.080305 |  3.327341 |
| 2026-06 | baseline  |   0.00 |     41 |   29.27% | 0.383369 |  -0.328374 | -13.463333 | 13.472962 |
| 2026-06 | candidate |   0.94 |      7 |   14.29% | 0.212107 |  -0.594080 |  -4.158560 |  4.398400 |
| 2026-07 | baseline  |   0.00 |     59 |   35.59% | 0.671518 |  -0.133774 |  -7.892658 | 10.874363 |
| 2026-07 | candidate |   0.94 |     21 |   47.62% | 1.247188 |   0.079617 |   1.671963 |  3.021980 |
| 2026-08 | baseline  |   0.00 |      7 |   42.86% | 1.609934 |   0.092237 |   0.645658 |  0.451826 |
| 2026-08 | candidate |   0.94 |      3 |   66.67% | 0.877958 |  -0.035786 |  -0.107358 |  0.879680 |

## Prior 2025 walk-forward OOS reference (not combined)

| Trades |       PF | Expectancy |  Net PnL |   Max DD |
| -----: | -------: | ---------: | -------: | -------: |
|    112 | 1.080648 |   0.027449 | 3.074296 | 6.548168 |

The 2025 values are a reference only; every 2026 value uses exclusively 2026 trade records.

## Deterministic assessment

- PF > 1: **no**
- Expectancy > 0: **no**
- Net PnL > 0: **no**
- Improves PF or expectancy versus 2026 baseline: **yes**
- Drawdown not materially worse (<=125% of baseline): **yes**
- Trade retention >=25%: **yes**
- Candidate sample size: **preliminary** (97 trades).
- Material thresholds: underperformance requires both PF and expectancy below 80% of baseline; drawdown worsens materially above 125% of baseline; high concentration means the top two positive months contribute at least 80% of positive monthly PnL.

## Verdict

**REJECT**

This external validation does not authorize real-money trading.
