# Historical Failure Patterns — EXP-000021 through EXP-000024

## Scope and reproduction

- Source inventory verified in GitHub `main` at `f0e2f128816549187828e29bf0f5287a93598430`.
- Verifiable completed experiments included: 4 / 4.
- PR #177 is merged (merge commit is the inspected main HEAD). This report extends its analysis-only implementation.
- Only stored leaderboard summaries and journals were read. No market dataset, reserved data, optimizer, historical backtest, experiment-ID allocator, or Research Memory writer was used.
- Reproduce from the repository root: `venv/bin/python -m app.research.automation.historical_report`.
- Outputs are deterministic for identical source bytes. Numbers below are stored CSV values rounded to six decimal places; no missing metric is reconstructed. PnL and expectancy retain the source USDT units and frozen costs/fills.
- `FEE_DOMINATED`, `NO_GROSS_EDGE`, and `POSITIVE_NET_EDGE` are descriptive triage classes, not governance verdicts. Journal verdicts are quoted independently.

## Persistence and source inventory

| Experiment | Strategy | Leaderboard | Journal | Existing governance verdict |
|---|---|---|---|---|
| EXP-000021 | short_regime_transition | [CSV](../leaderboards/short_regime_transition_baseline_v1.csv) | [Journal](../journal/EXP-000021.md) | BASELINE_REJECT |
| EXP-000022 | short_momentum_pullback_continuation | [CSV](../leaderboards/short_momentum_pullback_continuation_baseline_v1.csv) | [Journal](../journal/EXP-000022.md) | BASELINE_REJECT |
| EXP-000023 | short_compression_breakdown | [CSV](../leaderboards/short_compression_breakdown_baseline_v1.csv) | [Journal](../journal/EXP-000023.md) | BASELINE_REJECT |
| EXP-000024 | short_failed_breakout | [CSV](../leaderboards/short_failed_breakout_baseline_v1.csv) | [Journal](../journal/EXP-000024.md) | BASELINE_REJECT |

### Source SHA-256 checksums

- `research/journal/EXP-000021.md`: `d45da3954e80ed0857fcd9bcba65a753016b4da2b0f3c1c86562cba0ee8cb8aa`
- `research/leaderboards/short_regime_transition_baseline_v1.csv`: `8ea8dcbbe9942ed9cc705392d4548357491a1103a551cc4c636ca2e6065b4cf0`
- `research/journal/EXP-000022.md`: `cc245aeafbff5134b216e2a2173e66af766637a995b95b2240d156f479db3787`
- `research/leaderboards/short_momentum_pullback_continuation_baseline_v1.csv`: `22e4b8de08d8ec660f5377c3c8dde07080107584eea34189e01ad004b787eed7`
- `research/journal/EXP-000023.md`: `03e5f352b2bb6f82f2615c4903d103a4796f37b689a6dfc34ed2d973492c8910`
- `research/leaderboards/short_compression_breakdown_baseline_v1.csv`: `78bb4e3d4afae0f3bb8766e9d30edf9319d5f625f9b0e354e8b4016cfafeff26`
- `research/journal/EXP-000024.md`: `591c2d790b4adc353fb967641abb374cefbffde18bfeaeef05cb94e38063a4ee`
- `research/leaderboards/short_failed_breakout_baseline_v1.csv`: `1e694debf909f7378f580bcd49af5f303cc7756a2ad27558909771e009d11e65`

### Missing artifacts and provenance limitations

- No journal or leaderboard is missing for these four experiments. EXP-000024 is persisted in main; it is not assumed from a local-only result.
- EXP-000024 journal leaves its failure classification blank; the report supplies a separate triage classification from its stored economics, without changing that journal.
- The EXP-000024 preregistration still says RESERVED_FOR_FUTURE_AUTHORIZED_EXECUTION, while the later persisted journal says CLOSED_REJECTED. This is a documentation chronology conflict, not proof of execution authorization; no new execution is authorized here.
- A separate closure report exists for EXP-000021 (`research/reports/short_regime_transition_baseline_v1.md`); no corresponding standalone closure reports were found for EXP-000022–024. Their CSVs and journals supply the reported evidence.
- Stored artifacts are evidence of recorded outcomes, not independent verification of the original market runs. Journals for EXP-000022–024 list a “passing configuration” even though the authoritative verdict is BASELINE_REJECT; optimizer eligibility is not governance approval.

## Economics and primary triage

| Strategy | TF | Trades | Gross PnL | Fees | Net PnL | Gross PF | Net PF | Gross Exp. | Net Exp. | Profitable Months | Suppression | Triage Class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| short_regime_transition | 15m | 331 | 22.953160 | 26.470819 | -3.517658 | 1.209526 | 0.971714 | 0.069345 | -0.010627 | 50.0% | 28.7% | FEE_DOMINATED |
| short_momentum_pullback_continuation | 15m | 206 | 4.267975 | 16.478293 | -12.210317 | 1.061365 | 0.845917 | 0.020718 | -0.059273 | 33.3% | 37.8% | FEE_DOMINATED |
| short_compression_breakdown | 15m | 925 | 15.224734 | 73.993910 | -58.769176 | 1.050758 | 0.827943 | 0.016459 | -0.063534 | 33.3% | 40.4% | FEE_DOMINATED |
| short_failed_breakout | 15m | 985 | -5.221940 | 78.802089 | -84.024029 | 0.983832 | 0.771355 | -0.005301 | -0.085304 | 16.7% | 43.5% | NO_GROSS_EDGE |

## Stored payoff, timing, and overlap diagnostics

| Experiment | Win rate | Break-even win rate | Actual minus break-even | Payoff ratio | Suppression rate | Profitable / unprofitable months | Top-two positive-PnL concentration |
|---|---:|---:|---:|---:|---:|---|---:|
| EXP-000021 | 0.432024 | 0.439078 | -0.007054 | 1.277498 | 0.286638 | 6 / 6 | 0.659866 |
| EXP-000022 | 0.398058 | 0.438752 | -0.040694 | 1.279192 | 0.377644 | 4 / 8 | 0.676778 |
| EXP-000023 | 0.417297 | 0.463797 | -0.046499 | 1.156117 | 0.403995 | 4 / 8 | 0.883464 |
| EXP-000024 | 0.418274 | 0.482443 | -0.064169 | 1.072783 | 0.434882 | 2 / 10 | 1.000000 |

### Months with stored positive and negative net PnL

- EXP-000021: profitable 2025-01, 2025-02, 2025-05, 2025-08, 2025-10, 2025-11; unprofitable 2025-03, 2025-04, 2025-06, 2025-07, 2025-09, 2025-12. Zero-PnL months, if any, belong to neither group.
- EXP-000022: profitable 2025-02, 2025-03, 2025-05, 2025-10; unprofitable 2025-01, 2025-04, 2025-06, 2025-07, 2025-08, 2025-09, 2025-11, 2025-12. Zero-PnL months, if any, belong to neither group.
- EXP-000023: profitable 2025-02, 2025-08, 2025-11, 2025-12; unprofitable 2025-01, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-09, 2025-10. Zero-PnL months, if any, belong to neither group.
- EXP-000024: profitable 2025-01, 2025-09; unprofitable 2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08, 2025-10, 2025-11, 2025-12. Zero-PnL months, if any, belong to neither group.

## Recurring mechanisms and interpretation

The observations below describe the included persisted baselines; they are not causal estimates or new market experiments.

1. **Absence of gross trading edge:** 1 included baseline(s) have NO_GROSS_EDGE. Negative or zero stored gross PnL already fails before costs; fees amplify this failure rather than explain it away.
2. **Positive gross edge consumed by fees:** 3 included baseline(s) are FEE_DOMINATED. Their positive gross PnL becomes nonpositive net PnL. This is an observed discovery-period gross edge, not an established out-of-sample edge.
   - EXP-000021: FEE_DOMINATED; stored gross expectancy 0.069345, fee expectancy 0.079972, net expectancy -0.010627 USDT/trade.
   - EXP-000022: FEE_DOMINATED; stored gross expectancy 0.020718, fee expectancy 0.079992, net expectancy -0.059273 USDT/trade.
   - EXP-000023: FEE_DOMINATED; stored gross expectancy 0.016459, fee expectancy 0.079993, net expectancy -0.063534 USDT/trade.
   - EXP-000024: NO_GROSS_EDGE; stored gross expectancy -0.005301, fee expectancy 0.080002, net expectancy -0.085304 USDT/trade.
3. **Insufficient payoff or win rate:** inspect the stored actual-minus-break-even gaps alongside payoff ratios above. A negative gap records insufficient win rate for the observed payoff after costs; a win rate below 50% alone does not diagnose failure. No missing payoff or break-even metric is inferred.
4. **Temporal instability:** the monthly sign and concentration summaries above distinguish broad losses from profits concentrated in a few months. A stored top-two positive-PnL concentration above 0.80 independently fails the existing governance concentration gate. These descriptive summaries do not establish why instability occurs.
5. **Signal overlap or suppression:** the stored suppression rates record entries blocked while a position was already open. They do not show those missed signals would have been profitable or quantify a recoverable edge.

## Next research direction and tactical recommendation

Keep all four BASELINE_REJECT / CLOSED_REJECTED outcomes intact. Do not tune, rescue, change costs or fills, reuse reserved data, or rerun these frozen configurations.
The next tactical step is a documentation-only review of cost-adjusted edge requirements and signal independence before preregistering any new hypothesis. A future distinct hypothesis should explain why expected gross displacement can exceed the unchanged cost model and why its evidence should be temporally stable. Human authorization is required before selecting a new strategic hypothesis or running a market experiment.
