# HYP-SHORT-DIRECTIONAL-001 Baseline Closure

## Research identity

- **Hypothesis:** `HYP-SHORT-DIRECTIONAL-001`
- **Experiment:** `EXP-000021`
- **Strategy:** `short_regime_transition`
- **Configuration:** `research/optimization/grid_search/short_regime_transition_baseline.yaml`
- **Parameter combinations:** `1`
- **Verdict:** `BASELINE_REJECT`
- **Status:** `CLOSED_REJECTED`

## Deterministic verdict

The observed optimizer summary records 331 completed trades, so the
`completed_trades < 100` insufficient-sample rule does not apply. The baseline
is rejected independently by all three available net-performance gates:

- `net_profit_factor = 0.9717 <= 1`
- `net_expectancy = -0.0106 <= 0`
- `net_pnl = -3.5177 <= 0`

The authoritative preregistration requires `BASELINE_REJECT` when any one of
those conditions is true. Relative performance against another strategy does
not alter this verdict.

## Artifact integrity notice

At finalization time, the repository does not contain the referenced
`research/journal/EXP-000021.md` or
`research/leaderboards/short_regime_transition_baseline_v1.csv`, and
`research/memory/index.csv` has no `EXP-000021` row. No stored trade ledger or
serialized `TradeDiagnostics` for `EXP-000021` is present elsewhere under
`research/`.

Consequently, only the human-observed optimizer summary supplied for this
closure is recorded below. Missing diagnostics are deliberately marked
`NOT AVAILABLE IN STORED ARTIFACTS`; they are not estimated, reconstructed, or
silently invented. Reproducing them would require another market-data run,
which this closure explicitly prohibits.

## Gross edge

| Metric | EXP-000021 |
| --- | ---: |
| gross_pnl_before_fees | NOT AVAILABLE IN STORED ARTIFACTS |
| gross_expectancy | NOT AVAILABLE IN STORED ARTIFACTS |
| gross_profit_before_fees | NOT AVAILABLE IN STORED ARTIFACTS |
| gross_loss_before_fees | NOT AVAILABLE IN STORED ARTIFACTS |
| gross_profit_factor | NOT AVAILABLE IN STORED ARTIFACTS |

No conclusion about the presence or absence of raw gross edge is supportable
from the retained values.

## Cost burden

| Metric | EXP-000021 |
| --- | ---: |
| total_fees | NOT AVAILABLE IN STORED ARTIFACTS |
| fee_expectancy | NOT AVAILABLE IN STORED ARTIFACTS |

Fee domination cannot be established from the retained values.

## Net performance

| Metric | EXP-000021 |
| --- | ---: |
| completed_trades | 331 |
| win_rate | 43.20% |
| net_profit_factor | 0.9717 |
| net_expectancy | -0.0106 |
| net_pnl | -3.5177 |

The baseline loses after costs on both a per-trade and aggregate basis, and its
net profit factor is below one.

## Payoff structure

Wins, losses, flats, average and median winner, average and median loser,
payoff ratio, break-even win rate, and actual-minus-break-even win rate are
`NOT AVAILABLE IN STORED ARTIFACTS`. The rounded top-level win rate is not used
to reverse-engineer permanent diagnostics.

## Exit and follow-through structure

Counts, percentages, and net PnL for `take_profit`, `stop_loss`, `max_holding`,
and `strategy_exit` are `NOT AVAILABLE IN STORED ARTIFACTS`. Average, median,
P25, P75, and P95 holding periods are also unavailable. Follow-through and
max-holding dependency therefore cannot be classified responsibly.

## Signal overlap

Raw entry signals, suppressed signals, suppression rate, and raw signals per
opened trade are `NOT AVAILABLE IN STORED ARTIFACTS`. The completed-trade count
is 331. Signal overlap cannot be classified from this count alone.

## Monthly stability

Per-month trades, profit factor, expectancy, net PnL, and profitability for the
twelve 2025 months are `NOT AVAILABLE IN STORED ARTIFACTS`. Positive months,
negative months, profitable-month percentage, best month, worst month, and
positive-PnL concentration in the top two months are likewise unavailable.
Temporal stability and the concentration rejection gate cannot be evaluated;
the three independent net-performance rejection gates already determine the
verdict.

## Long versus short asymmetry

The long values below are the frozen `EXP-000020` metrics supplied for this
closure. Comparisons are limited to short metrics that were retained.

| Metric | Long EXP-000020 | Short EXP-000021 | Measured asymmetry |
| --- | ---: | ---: | --- |
| completed trades | 309 | 331 | Short higher by 22 |
| gross profit factor | 1.1673959163 | NOT AVAILABLE | Not measurable |
| gross expectancy | 0.0542618354 | NOT AVAILABLE | Not measurable |
| total fees | 24.7267067629 | NOT AVAILABLE | Not measurable |
| fee expectancy | 0.0800217047 | NOT AVAILABLE | Not measurable |
| net profit factor | 0.9297643119 | 0.9717 | Short higher by 0.0419356881; both below 1 |
| net expectancy | -0.0257598693 | -0.0106 | Short less negative by 0.0151598693; both negative |
| net PnL | -7.9597996163 | -3.5177 | Short less negative by 4.4420996163; both negative |
| win rate | 45.95469256% | 43.20% | Short lower by 2.75469256 percentage points |
| payoff ratio | 1.0934552119 | NOT AVAILABLE | Not measurable |
| actual minus break-even WR | -1.81322692 pp | NOT AVAILABLE | Not measurable |
| max-holding dependency | 42.07119741% | NOT AVAILABLE | Not measurable |
| average holding | 16.4336569579 | NOT AVAILABLE | Not measurable |
| median holding | 20 | NOT AVAILABLE | Not measurable |
| holding P75 | 24 | NOT AVAILABLE | Not measurable |
| holding P95 | 24 | NOT AVAILABLE | Not measurable |
| suppression rate | 36.28865979% | NOT AVAILABLE | Not measurable |
| raw signals/opened trade | 1.5695792880 | NOT AVAILABLE | Not measurable |
| positive months | 5 | NOT AVAILABLE | Not measurable |
| negative months | 7 | NOT AVAILABLE | Not measurable |
| profitable-month percentage | 41.66666667% | NOT AVAILABLE | Not measurable |
| top-two positive-PnL concentration | 84.9006% | NOT AVAILABLE | Not measurable |
| verdict | BASELINE_REJECT | BASELINE_REJECT | Same deterministic outcome |

The short baseline's less negative net metrics are relative differences only;
they are not strategy success. Gross edge, fee burden, payoff quality,
actual-minus-break-even win rate, max-holding dependency, holding duration,
signal suppression, and monthly stability cannot be compared without the
missing permanent diagnostics.

## Failure classification

**Classification: `MIXED` (net-negative performance and lower hit rate versus
the frozen long baseline).** The directly supported failures are a net profit
factor of 0.9717, net expectancy of -0.0106, and net PnL of -3.5177 across 331
trades. The 43.20% win rate is 2.75469256 percentage points below the long
baseline, but `LOW_HIT_RATE` cannot be isolated without the short break-even
win rate. `NO_RAW_EDGE`, `FEE_DOMINATED`, `POOR_PAYOFF`, `NO_FOLLOW_THROUGH`,
`TEMPORAL_INSTABILITY`, and `SIGNAL_OVERLAP` are not assigned because their
required permanent diagnostics are unavailable.

## Research closure

`HYP-SHORT-DIRECTIONAL-001` is closed as `CLOSED_REJECTED` under experiment
`EXP-000021` with verdict `BASELINE_REJECT`.

This exact hypothesis must not be rescued through parameter tuning, filters,
exit changes, regime threshold changes, timeframe changes, volatility
conditions, or alternate transition definitions.

No new experiment was run. No parameter was changed. No strategy code was
modified. No grid search, Monte Carlo, walk-forward, external validation,
alternate short rule, or Family #5 optimization was executed.

## Data boundaries

- **2025:** `DISCOVERY_USED`
- **2026-01-01 through 2026-08-05:** `NOT USED`
- **post-2026-08-05:** `RESERVED / NOT ACCESSED`

No 2026 data was accessed during finalization, including reserved data after
2026-08-05.
