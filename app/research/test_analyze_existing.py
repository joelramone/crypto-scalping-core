from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from app.research.automation.analyze_existing import analyze_existing, render_failure_pattern_report


def _leaderboard(path: Path, strategy: str, gross: float, fees: float, net: float) -> Path:
    pd.DataFrame([{
        "strategy": strategy,
        "timeframe": "15m",
        "total_trades": 100,
        "gross_pnl_before_fees": gross,
        "total_fees": fees,
        "net_pnl": net,
        "gross_profit_factor": 1.1,
        "net_profit_factor": 0.9,
        "gross_expectancy": gross / 100,
        "net_expectancy": net / 100,
        "suppression_rate": 0.2,
        "profitable_month_percentage": 0.5,
    }]).to_csv(path, index=False)
    return path


def test_analyze_existing_classifies_failure_modes(tmp_path: Path) -> None:
    paths = [
        _leaderboard(tmp_path / "no_edge.csv", "no_edge", -1.0, 5.0, -6.0),
        _leaderboard(tmp_path / "fees.csv", "fees", 2.0, 5.0, -3.0),
        _leaderboard(tmp_path / "positive.csv", "positive", 8.0, 2.0, 6.0),
    ]
    result = analyze_existing(paths)
    assert [item["failure_mode"] for item in result] == [
        "NO_GROSS_EDGE",
        "FEE_DOMINATED",
        "POSITIVE_NET_EDGE",
    ]


def test_report_contains_cross_family_summary(tmp_path: Path) -> None:
    paths = [
        _leaderboard(tmp_path / "a.csv", "family_a", -1.0, 4.0, -5.0),
        _leaderboard(tmp_path / "b.csv", "family_b", 1.0, 4.0, -3.0),
    ]
    report = render_failure_pattern_report(analyze_existing(paths))
    assert "Baselines analyzed: 2" in report
    assert "NO_GROSS_EDGE: 1" in report
    assert "FEE_DOMINATED: 1" in report
    assert "family_a" in report
    assert "family_b" in report
    assert "No optimizer execution" in report

from app.research.automation.analyze_existing import write_failure_pattern_report


def test_missing_leaderboard(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_existing([tmp_path / 'missing.csv'])


@pytest.mark.parametrize('contents', ['', 'strategy,timeframe,total_trades\n'])
def test_empty_leaderboard(tmp_path, contents):
    path = tmp_path / 'empty.csv'
    path.write_text(contents)
    with pytest.raises(ValueError, match='empty'):
        analyze_existing([path])


def test_missing_columns(tmp_path):
    path = tmp_path / 'missing_columns.csv'
    pd.DataFrame([{'strategy': 'baseline'}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing required columns.*gross_expectancy'):
        analyze_existing([path])


@pytest.mark.parametrize(('field', 'value'), [
    ('net_pnl', float('nan')), ('gross_pnl_before_fees', float('inf')),
    ('gross_expectancy', 'invalid'), ('total_fees', -1),
    ('total_trades', 1.5), ('total_trades', -1), ('total_trades', float('inf')),
    ('suppression_rate', 1.1), ('profitable_month_percentage', -0.1),
    ('net_profit_factor', -1), ('gross_profit_factor', float('-inf')),
    ('strategy', ''), ('timeframe', ''),
])
def test_invalid_required_metrics(tmp_path, field, value):
    path = _leaderboard(tmp_path / 'invalid.csv', 'baseline', 2, 5, -3)
    frame = pd.read_csv(path)
    frame[field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=f'invalid {field}'):
        analyze_existing([path])


def test_positive_infinite_pf_is_supported(tmp_path):
    path = _leaderboard(tmp_path / 'no_losses.csv', 'baseline', 8, 2, 6)
    frame = pd.read_csv(path)
    frame['gross_profit_factor'] = float('inf')
    frame.to_csv(path, index=False)
    assert analyze_existing([path])[0]['gross_profit_factor'] == float('inf')


def test_empty_input():
    with pytest.raises(ValueError, match='At least one'):
        analyze_existing([])


def test_report_is_deterministic_and_inputs_unchanged(tmp_path):
    paths = [_leaderboard(tmp_path / 'a.csv', 'a', -1, 2, -3),
             _leaderboard(tmp_path / 'b.csv', 'b', 8, 2, 6)]
    originals = [p.read_bytes() for p in paths]
    output = tmp_path / 'report.md'
    write_failure_pattern_report(paths, output)
    first = output.read_bytes()
    write_failure_pattern_report(paths, output)
    assert output.read_bytes() == first
    assert [p.read_bytes() for p in paths] == originals
    assert b'not authoritative governance verdicts' in first
    assert b'Suppression' in first


def test_first_stored_row_is_not_reranked(tmp_path):
    path = _leaderboard(tmp_path / 'a.csv', 'first', -1, 2, -3)
    frame = pd.read_csv(path)
    second = frame.iloc[0].copy()
    second['strategy'] = 'second'
    second['net_pnl'] = 100
    pd.concat([frame, pd.DataFrame([second])]).to_csv(path, index=False)
    assert analyze_existing([path])[0]['strategy'] == 'first'


def test_output_cannot_overwrite_input_or_memory(tmp_path):
    path = _leaderboard(tmp_path / 'a.csv', 'a', -1, 2, -3)
    original = path.read_bytes()
    research = Path(__file__).resolve().parents[2] / 'research'
    for output in [path, research / 'memory' / 'index.csv', research / 'journal' / 'EXP-000021.md']:
        with pytest.raises(ValueError, match='cannot overwrite'):
            write_failure_pattern_report([path], output)
    assert path.read_bytes() == original


def test_invalid_input_does_not_touch_existing_report(tmp_path):
    output = tmp_path / 'existing.md'
    output.write_text('preserve me')
    with pytest.raises(FileNotFoundError):
        write_failure_pattern_report([tmp_path / 'missing.csv'], output)
    assert output.read_text() == 'preserve me'


def test_analysis_only_cli_has_no_execution_or_memory_imports(tmp_path):
    path = _leaderboard(tmp_path / 'a.csv', 'a', 2, 5, -3)
    output = tmp_path / 'report.md'
    # Fresh interpreter: block dangerous module imports before importing the CLI.
    # Audit hooks also forbid processes and any read outside this CSV after import.
    script = '''
import sys
from pathlib import Path
class BlockResearchExecution:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('app.research.optimizer', 'app.research.memory',
                   'app.research.backtester', 'app.research.download_data',
                   'app.research.automation.runner', 'app.research.simulation')
        if fullname.startswith(blocked):
            raise AssertionError('Forbidden research import: ' + fullname)
sys.meta_path.insert(0, BlockResearchExecution())
from app.research.automation.analyze_existing import main
source, output = sys.argv[1:]
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect'):
        raise AssertionError('Forbidden execution/network operation: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        target = Path(args[0]).resolve()
        if target not in {Path(source).resolve(), Path(output).resolve()}:
            raise AssertionError('Unexpected file access: ' + str(target))
sys.addaudithook(audit)
sys.argv = ['analyze_existing', source, '--output', output]
main()
'''
    result = subprocess.run([sys.executable, '-c', script, str(path), str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert output.exists()
