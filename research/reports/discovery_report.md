# Quant Research Lab Discovery Report

## Executive Summary

- Total experiments: 567
- Strategies tested: donchian_breakout
- Best strategy by average PF: donchian_breakout
- Best timeframe by average PF: 5m
- Average Profit Factor: 0.5767
- Average Expectancy: -0.0814
- Average Drawdown: 181.9486

## Best Experiments

| strategy | timeframe | profit_factor | expectancy | max_drawdown | total_trades | source_file |
| --- | --- | --- | --- | --- | --- | --- |
| donchian_breakout | 5m | 0.7755 | -0.0583 | 68.3540 | 1057 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7737 | -0.0587 | 64.5162 | 1007 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7693 | -0.0547 | 66.1222 | 1115 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7625 | -0.0587 | 62.2127 | 1013 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7602 | -0.0574 | 64.5665 | 1059 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7598 | -0.0631 | 66.2488 | 940 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7586 | -0.0657 | 65.7788 | 960 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7569 | -0.0637 | 58.0523 | 865 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7540 | -0.0605 | 71.3513 | 1099 | donchian_breakout_5m_smoke_v1.csv |
| donchian_breakout | 5m | 0.7528 | -0.0652 | 59.8588 | 894 | donchian_breakout_5m_smoke_v1.csv |

## Parameter Importance

| parameter | corr_profit_factor | corr_expectancy | importance_score |
| --- | --- | --- | --- |
| stop_loss_pct | 0.7865 | 0.1439 | 0.7865 |
| take_profit_pct | 0.7468 | -0.0283 | 0.7468 |
| lookback | 0.1020 | -0.6145 | 0.6145 |
| volume_ratio | -0.5043 | -0.1008 | 0.5043 |
| max_holding_candles | -0.1524 | -0.2966 | 0.2966 |

## Pattern Discovery

- Top profit-factor threshold: 0.7139
- Experiments in top bucket: 57
- Most common timeframe: 5m
- Most common TP: 0.0060
- Most common SL: 0.0050
- Most common holding: 36.0000
- Most common RSI: n/a
- Most common volume: 1.0000
- Most common lookback: 10.0000

## Recommendations

- Prioritize 5m experiments first because that timeframe currently has the strongest average profit factor.
- Start new sweeps from the strongest strategy family so far: donchian_breakout.
- Bias new grids toward TP=0.006, SL=0.005, and holding=36, since those values recur most often in the top profit-factor bucket.
- Keep RSI=n/a, volume ratio=1.0, and lookback=10 near the current winning cluster before exploring wider ranges.
- Focus tuning attention on stop_loss_pct; it shows the strongest linear relationship with profit factor or expectancy in the current dataset.
