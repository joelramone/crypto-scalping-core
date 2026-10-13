"""Reproduce TASK-002 from stored leaderboards and journals, never market data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from app.research.automation.analyze_existing import analyze_existing, render_failure_pattern_report

EXPERIMENTS = (
    ('EXP-000021', 'short_regime_transition'),
    ('EXP-000022', 'short_momentum_pullback_continuation'),
    ('EXP-000023', 'short_compression_breakdown'),
    ('EXP-000024', 'short_failed_breakout'),
)
REPORT_NAME = 'failure_patterns_exp_000021_000024.md'
# Verified origin/main at inspection. All eight source artifacts exist at this commit.
SOURCE_COMMIT = 'f0e2f128816549187828e29bf0f5287a93598430'


def render_historical_report(research: Path) -> str:
    """Read the fixed completed-artifact inventory and retain source provenance."""
    records = []
    missing = []
    for experiment, strategy in EXPERIMENTS:
        journal = research / 'journal' / f'{experiment}.md'
        leaderboard = research / 'leaderboards' / f'{strategy}_baseline_v1.csv'
        absent = [str(p.relative_to(research)) for p in (journal, leaderboard) if not p.is_file()]
        if absent:
            missing.append(f"- {experiment}: excluded; missing {', '.join(absent)}")
            continue
        journal_text = journal.read_text(encoding='utf-8')
        expected = (f'- Experiment ID: {experiment}', f'- Strategy: {strategy}',
                    f'- Leaderboard File: research/leaderboards/{leaderboard.name}')
        if not all(line in journal_text.splitlines() for line in expected):
            raise ValueError(f'Journal identity/source mismatch: {journal}')
        verdict = re.search(r'^- Verdict: (\S+)\s*$', journal_text, re.MULTILINE)
        if not verdict:
            raise ValueError(f'Missing persisted governance verdict: {journal}')
        with leaderboard.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0]['strategy'] != strategy:
            raise ValueError(f'Expected one frozen baseline row: {leaderboard}')
        diagnostic = analyze_existing([leaderboard])[0]
        # Relative sources make the rendered report independent of checkout location.
        diagnostic['source'] = f'research/leaderboards/{leaderboard.name}'
        records.append((experiment, rows[0], diagnostic, journal, leaderboard, verdict.group(1)))

    lines = [
        '# Historical Failure Patterns — EXP-000021 through EXP-000024', '',
        '## Scope and reproduction', '',
        f'- Source inventory verified in GitHub `main` at `{SOURCE_COMMIT}`.',
        f'- Verifiable completed experiments included: {len(records)} / 4.',
        '- PR #177 is merged (merge commit is the inspected main HEAD). This report extends its analysis-only implementation.',
        '- Only stored leaderboard summaries and journals were read. No market dataset, reserved data, optimizer, historical backtest, experiment-ID allocator, or Research Memory writer was used.',
        '- Reproduce from the repository root: `venv/bin/python -m app.research.automation.historical_report`.',
        '- Outputs are deterministic for identical source bytes. Numbers below are stored CSV values rounded to six decimal places; no missing metric is reconstructed. PnL and expectancy retain the source USDT units and frozen costs/fills.',
        '- `FEE_DOMINATED`, `NO_GROSS_EDGE`, and `POSITIVE_NET_EDGE` are descriptive triage classes, not governance verdicts. Journal verdicts are quoted independently.',
        '', '## Persistence and source inventory', '',
        '| Experiment | Strategy | Leaderboard | Journal | Existing governance verdict |',
        '|---|---|---|---|---|',
    ]
    for experiment, row, diagnostic, journal, leaderboard, verdict in records:
        lines.append(f'| {experiment} | {row["strategy"]} | [CSV](../leaderboards/{leaderboard.name}) | [Journal](../journal/{journal.name}) | {verdict} |')
    lines += ['', '### Source SHA-256 checksums', '']
    for _, _, _, journal, leaderboard, _ in records:
        for path in (journal, leaderboard):
            lines.append(f'- `research/{path.relative_to(research).as_posix()}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`')
    lines += ['', '### Missing artifacts and provenance limitations', '']
    lines += missing or ['- No journal or leaderboard is missing for these four experiments. EXP-000024 is persisted in main; it is not assumed from a local-only result.']
    lines += [
        '- EXP-000024 journal leaves its failure classification blank; the report supplies a separate triage classification from its stored economics, without changing that journal.',
        '- The EXP-000024 preregistration still says RESERVED_FOR_FUTURE_AUTHORIZED_EXECUTION, while the later persisted journal says CLOSED_REJECTED. This is a documentation chronology conflict, not proof of execution authorization; no new execution is authorized here.',
        '- A separate closure report exists for EXP-000021 (`research/reports/short_regime_transition_baseline_v1.md`); no corresponding standalone closure reports were found for EXP-000022–024. Their CSVs and journals supply the reported evidence.',
        '- Stored artifacts are evidence of recorded outcomes, not independent verification of the original market runs. Journals for EXP-000022–024 list a “passing configuration” even though the authoritative verdict is BASELINE_REJECT; optimizer eligibility is not governance approval.',
        '', '## Economics and primary triage', '',
    ]
    generic = render_failure_pattern_report([r[2] for r in records]) if records else 'No verifiable baselines available.\n'
    # Retain only the comparison table; detailed provenance and governance appear here.
    if records:
        generic = generic.split('## Cross-Family Comparison\n\n', 1)[1].split('\n\n## Sources', 1)[0]
    lines.append(generic)
    lines += ['', '## Stored payoff, timing, and overlap diagnostics', '',
              '| Experiment | Win rate | Break-even win rate | Actual minus break-even | Payoff ratio | Suppression rate | Profitable / unprofitable months | Top-two positive-PnL concentration |',
              '|---|---:|---:|---:|---:|---:|---|---:|']

    def value(row: dict[str, str], key: str) -> str:
        raw = row.get(key, '')
        return f'{float(raw):.6f}' if raw else 'N/A (not stored)'

    for experiment, row, *_ in records:
        counts = f"{row.get('positive_months') or 'N/A'} / {row.get('negative_months') or 'N/A'}"
        metrics = [value(row, key) for key in ('win_rate', 'break_even_win_rate',
                   'actual_minus_break_even_win_rate', 'payoff_ratio', 'suppression_rate')]
        lines.append(f'| {experiment} | ' + ' | '.join(metrics) + f' | {counts} | {value(row, "positive_pnl_concentration_top_2_months")} |')
    lines += ['', '### Months with stored positive and negative net PnL', '']
    for experiment, row, *_ in records:
        months = json.loads(row['monthly_diagnostics']) if row.get('monthly_diagnostics') else None
        if months is None:
            lines.append(f'- {experiment}: N/A (monthly diagnostics not stored).')
        else:
            positive = ', '.join(m['month'] for m in months if m['net_pnl'] > 0) or 'none'
            negative = ', '.join(m['month'] for m in months if m['net_pnl'] < 0) or 'none'
            lines.append(f'- {experiment}: profitable {positive}; unprofitable {negative}. Zero-PnL months, if any, belong to neither group.')

    classes = [r[2]['failure_mode'] for r in records]
    lines += ['', '## Recurring mechanisms and interpretation', '',
        'The observations below describe the included persisted baselines; they are not causal estimates or new market experiments.', '',
        f"1. **Absence of gross trading edge:** {classes.count('NO_GROSS_EDGE')} included baseline(s) have NO_GROSS_EDGE. Negative or zero stored gross PnL already fails before costs; fees amplify this failure rather than explain it away.",
        f"2. **Positive gross edge consumed by fees:** {classes.count('FEE_DOMINATED')} included baseline(s) are FEE_DOMINATED. Their positive gross PnL becomes nonpositive net PnL. This is an observed discovery-period gross edge, not an established out-of-sample edge.",
    ]
    for experiment, row, diagnostic, *_ in records:
        lines.append(f"   - {experiment}: {diagnostic['failure_mode']}; stored gross expectancy {value(row, 'gross_expectancy')}, fee expectancy {value(row, 'fee_expectancy')}, net expectancy {value(row, 'net_expectancy')} USDT/trade.")
    lines += [
        '3. **Insufficient payoff or win rate:** inspect the stored actual-minus-break-even gaps alongside payoff ratios above. A negative gap records insufficient win rate for the observed payoff after costs; a win rate below 50% alone does not diagnose failure. No missing payoff or break-even metric is inferred.',
        '4. **Temporal instability:** the monthly sign and concentration summaries above distinguish broad losses from profits concentrated in a few months. A stored top-two positive-PnL concentration above 0.80 independently fails the existing governance concentration gate. These descriptive summaries do not establish why instability occurs.',
        '5. **Signal overlap or suppression:** the stored suppression rates record entries blocked while a position was already open. They do not show those missed signals would have been profitable or quantify a recoverable edge.',
        '', '## Next research direction and tactical recommendation', '',
        'Keep all four BASELINE_REJECT / CLOSED_REJECTED outcomes intact. Do not tune, rescue, change costs or fills, reuse reserved data, or rerun these frozen configurations.',
        'The next tactical step is a documentation-only review of cost-adjusted edge requirements and signal independence before preregistering any new hypothesis. A future distinct hypothesis should explain why expected gross displacement can exceed the unchanged cost model and why its evidence should be temporally stable. Human authorization is required before selecting a new strategic hypothesis or running a market experiment.', '',
    ]
    return '\n'.join(lines)


def main() -> None:
    research = Path(__file__).resolve().parents[3] / 'research'
    report = render_historical_report(research)
    destination = research / 'reports' / REPORT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding='utf-8')
    print(f'Wrote {destination}')


if __name__ == '__main__':
    main()
