"""Synthetic artifact tests; no historical experiments or market data are run."""

from pathlib import Path

import pandas as pd
import pytest

from app.research.automation.historical_report import (
    EXPERIMENTS, REPORT_NAME, render_historical_report,
)


def _artifacts(research: Path, count: int = 4) -> None:
    (research / 'journal').mkdir(parents=True)
    (research / 'leaderboards').mkdir()
    for index, (experiment, strategy) in enumerate(EXPERIMENTS[:count]):
        name = f'{strategy}_baseline_v1.csv'
        (research / 'journal' / f'{experiment}.md').write_text(
            f'- Experiment ID: {experiment}\n- Strategy: {strategy}\n'
            f'- Leaderboard File: research/leaderboards/{name}\n'
            '- Verdict: BASELINE_REJECT\n')
        pd.DataFrame([{
            'strategy': strategy, 'timeframe': '15m', 'total_trades': 100,
            'gross_pnl_before_fees': -1 if index == 3 else 2,
            'total_fees': 5, 'net_pnl': -6 if index == 3 else -3,
            'gross_profit_factor': 0.9 if index == 3 else 1.1,
            'net_profit_factor': 0.8, 'gross_expectancy': -0.01 if index == 3 else 0.02,
            'net_expectancy': -0.06 if index == 3 else -0.03,
            'suppression_rate': 0.2, 'profitable_month_percentage': 0.5,
        }]).to_csv(research / 'leaderboards' / name, index=False)


def test_historical_report_is_deterministic_and_preserves_artifacts(tmp_path):
    _artifacts(tmp_path)
    paths = sorted(p for p in tmp_path.rglob('*') if p.is_file())
    before = {p: p.read_bytes() for p in paths}
    first = render_historical_report(tmp_path)
    assert first == render_historical_report(tmp_path)
    assert before == {p: p.read_bytes() for p in paths}
    assert 'included: 4 / 4' in first
    assert '3 included baseline(s) are FEE_DOMINATED' in first
    assert '1 included baseline(s) have NO_GROSS_EDGE' in first
    assert 'N/A (not stored)' in first
    assert 'BASELINE_REJECT' in first


@pytest.mark.parametrize('folder', ['journal', 'leaderboards'])
def test_missing_source_excludes_experiment(tmp_path, folder):
    _artifacts(tmp_path)
    experiment, strategy = EXPERIMENTS[-1]
    file = f'{experiment}.md' if folder == 'journal' else f'{strategy}_baseline_v1.csv'
    (tmp_path / folder / file).unlink()
    report = render_historical_report(tmp_path)
    assert 'included: 3 / 4' in report
    assert 'EXP-000024: excluded; missing' in report
    assert '| EXP-000024 |' not in report
    assert '| short_failed_breakout |' not in report


def test_mismatched_journal_is_rejected(tmp_path):
    _artifacts(tmp_path)
    (tmp_path / 'journal' / 'EXP-000021.md').write_text('wrong identity')
    with pytest.raises(ValueError, match='identity/source mismatch'):
        render_historical_report(tmp_path)


def test_missing_verdict_is_rejected(tmp_path):
    _artifacts(tmp_path)
    path = tmp_path / 'journal' / 'EXP-000021.md'
    path.write_text(path.read_text().replace('- Verdict: BASELINE_REJECT\n', ''))
    with pytest.raises(ValueError, match='Missing persisted governance verdict'):
        render_historical_report(tmp_path)


def test_multi_configuration_artifact_is_not_presented_as_frozen_baseline(tmp_path):
    _artifacts(tmp_path)
    path = tmp_path / 'leaderboards' / 'short_regime_transition_baseline_v1.csv'
    frame = pd.read_csv(path)
    pd.concat([frame, frame]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='one frozen baseline row'):
        render_historical_report(tmp_path)


def test_committed_report_matches_persisted_artifacts():
    # Stored summaries and journals only; no dataset or experiment executor.
    research = Path(__file__).resolve().parents[2] / 'research'
    assert render_historical_report(research) == (research / 'reports' / REPORT_NAME).read_text()
