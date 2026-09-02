# HYP-SHORT-DIRECTIONAL-001 Baseline Closure

## Research identity and final verdict

- **Hypothesis:** `HYP-SHORT-DIRECTIONAL-001`
- **Experiment:** `EXP-000021`
- **Strategy:** `short_regime_transition`
- **Timeframe:** `15m`
- **Parameter combinations:** `1` (one frozen baseline configuration)
- **Verdict:** `BASELINE_REJECT`
- **Status:** `CLOSED_REJECTED`
- **Primary failure classification:** `FEE_DOMINATED`
- **Research action:** `NO_RESCUE_AUTHORIZED`

The 331 completed trades clear the minimum-sample gate, but net profit factor,
net expectancy, and net PnL fail the preregistered baseline gates. This exact
hypothesis is closed and must not be tuned or rescued.

## Authoritative diagnostics

These values are persisted from the locally observed stored artifacts supplied
for closure. They were not recomputed from market data.

| Category | Metric | Value |
| --- | --- | ---: |
| Trades | completed / wins / losses / flats | 331 / 143 / 188 / 0 |
| Trades | win rate | 0.432024 |
| Gross | PnL before fees | 22.95316 |
| Gross | expectancy | +0.069345 |
| Gross | profit / loss before fees | 132.501422 / 109.548262 |
| Gross | profit factor | 1.209526 |
| Costs | total fees | 26.470819 |
| Costs | fee expectancy | 0.079972 |
| Net | PnL | -3.517658 |
| Net | expectancy | -0.010627 |
| Net | profit factor | 0.971714 |
| Risk | max drawdown | 28.0196 |
| Payoff | average / median winner | 0.845043 / 1.12048 |
| Payoff | average / median loser | -0.661483 / -0.88032 |
| Payoff | payoff ratio | 1.277498 |
| Payoff | break-even win rate | 0.439078 |
| Payoff | actual minus break-even win rate | -0.007054 |
| Holding | average / median candles | 14.407855 / 13 |
| Holding | P25 / P75 / P95 | 6 / 24 / 24 |
| Overlap | raw signals / suppressed signals | 464 / 133 |
| Overlap | suppression rate | 0.286638 |
| Overlap | raw signals per opened trade | 1.401813 |

## Exit diagnostics

| Exit | Count | Percentage | Net PnL |
| --- | ---: | ---: | ---: |
| take_profit | 95 | 0.287009 | 106.4456 |
| stop_loss | 117 | 0.353474 | -102.99744 |
| max_holding | 119 | 0.359517 | -6.965818 |
| strategy_exit | 0 | 0 | 0 |

## Monthly diagnostics

| Month | Trades | PF | Expectancy | Net PnL | Profitable |
| --- | ---: | ---: | ---: | ---: | :---: |
| 2025-01 | 24 | 1.1080167185 | 0.0475446388 | 1.1410713311 | true |
| 2025-02 | 38 | 1.5552651241 | 0.1669816638 | 6.3453032232 | true |
| 2025-03 | 25 | 0.6703554808 | -0.1615508792 | -4.0387719809 | false |
| 2025-04 | 17 | 0.5468118670 | -0.2037771966 | -3.4642123421 | false |
| 2025-05 | 29 | 1.2708575697 | 0.0696874859 | 2.0209370901 | true |
| 2025-06 | 26 | 0.4092951806 | -0.3251180241 | -8.4530686272 | false |
| 2025-07 | 21 | 0.3734479932 | -0.2493174879 | -5.2356672457 | false |
| 2025-08 | 31 | 1.0433049809 | 0.0144110254 | 0.4467417862 | true |
| 2025-09 | 20 | 0.2499331738 | -0.2785668176 | -5.5713363520 | false |
| 2025-10 | 31 | 1.4198196025 | 0.1489477890 | 4.6173814586 | true |
| 2025-11 | 37 | 1.8322988164 | 0.2598249116 | 9.6135217305 | true |
| 2025-12 | 32 | 0.9220328464 | -0.0293612035 | -0.9395585107 | false |

There were 6 positive and 6 negative months (profitable-month percentage
`0.5`). The best month was `2025-11`, the worst was `2025-06`, and top-two
positive-PnL concentration was `0.659866`.

## Primary failure classification

**`FEE_DOMINATED`.** Gross expectancy was positive at `+0.069345`, but fee
expectancy was higher at `0.079972`, producing net expectancy of `-0.010627`.
Likewise, gross profit factor was `1.209526` while net profit factor fell to
`0.971714`. The gross edge is positive but insufficient to overcome the frozen
transaction-cost model.

This is not `NO_RAW_EDGE`; positive gross expectancy and gross PF above one
show a raw gross edge. `POOR_PAYOFF` is not the primary failure: the payoff
ratio was `1.277498`. The raw win rate alone does not support `LOW_HIT_RATE`;
the relevant actual-minus-break-even deficit was only `-0.007054`.

Secondary observations are moderate follow-through weakness and continuing
temporal instability.

## Directional asymmetry — discovery knowledge only

Relative to the frozen bullish transition baseline (`EXP-000020`), the 2025
short baseline exhibited measurable directional asymmetry: higher gross PF,
higher gross expectancy, higher payoff ratio, lower break-even win rate, a
smaller actual-versus-break-even win-rate deficit, higher net PF, less-negative
net expectancy and PnL, higher take-profit percentage, lower max-holding
percentage, shorter average and median holding, lower signal suppression, more
profitable months, and lower top-two positive-PnL concentration.

This finding is discovery knowledge only. It is not strategy approval,
validation, permission to tune Family #5, or permission to use 2026 or reserved
data. Both directional baselines have verdict `BASELINE_REJECT`.

## Data governance and closure

- **2025:** `DISCOVERY_USED`
- **2026-01-01 through 2026-08-05:** `NOT USED`
- **post-2026-08-05:** `RESERVED / NOT ACCESSED`

Experiment closed under preregistered anti-tuning rules. No rescue or parameter
optimization is authorized. No market dataset was opened to produce this
closure artifact.
