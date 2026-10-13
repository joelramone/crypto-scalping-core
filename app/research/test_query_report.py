"""Query retrieval tests use report text only, never market datasets."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.research.query_report import REPORT_PATH, SOURCE, main, query_report

SYNTHETIC = '''# Synthetic historical report

## Inventory

| Experiment | Strategy | Verdict |
|---|---|---|
| EXP-000021 | alpha | BASELINE_REJECT |
| EXP-000024 | beta | BASELINE_REJECT |

## Economics

| Strategy | Gross PnL | Triage |
|---|---:|---|
| alpha | 2.000000 | FEE_DOMINATED |
| beta | -1.000000 | NO_GROSS_EDGE |

## Limitations

EXP-000024 has a missing optional metric. No value is inferred.

### Governance

Triage is not a governance verdict.
'''


@pytest.fixture
def report(tmp_path):
    path = tmp_path / 'report.md'
    path.write_text(SYNTHETIC, encoding='utf-8')
    return path


def test_experiment_query_uses_report_inventory_and_retains_headers(report):
    result = query_report(report, experiment='exp-000024')
    assert len(result['matches']) == 3
    assert all('alpha' not in match['text'] for match in result['matches'])
    row = next(m for m in result['matches'] if '| beta | -1' in m['text'])
    assert row['table_header']['text'] == '| Strategy | Gross PnL | Triage |'
    assert 'BASELINE_REJECT' in result['matches'][0]['text']
    assert 'NO_GROSS_EDGE' in row['text']
    assert 'No value is inferred' in result['matches'][-1]['text']


def test_filters_intersect_and_search_is_literal(report):
    result = query_report(report, experiment='EXP-000024', section='economics', query='no_gross_edge')
    assert len(result['matches']) == 1
    assert query_report(report, query='.*')['matches'] == []
    assert query_report(report, experiment='EXP-000021', query='NO_GROSS_EDGE')['matches'] == []


def test_determinism_exact_line_citations_and_no_mutation(report):
    before = report.read_bytes()
    first = query_report(report)
    assert first == query_report(report)
    assert first['source_sha256'] == hashlib.sha256(before).hexdigest()
    for match in first['matches']:
        assert match['text'] == SYNTHETIC.splitlines()[match['line'] - 1]
        if 'table_header' in match:
            header = match['table_header']
            assert header['text'] == SYNTHETIC.splitlines()[header['line'] - 1]
    assert report.read_bytes() == before


@pytest.mark.parametrize('filters', [
    {'experiment': 'EXP-999999'}, {'section': 'missing'},
    {'experiment': ''}, {'section': ' '}, {'query': ''},
])
def test_invalid_filters_fail_explicitly(report, filters):
    with pytest.raises(ValueError):
        query_report(report, **filters)


def test_missing_and_empty_report(tmp_path):
    path = tmp_path / 'missing.md'
    with pytest.raises(FileNotFoundError):
        query_report(path)
    path.write_text(' \n')
    with pytest.raises(ValueError, match='empty'):
        query_report(path)


def test_cli_json_and_section_listing(monkeypatch, capsys, report):
    monkeypatch.setattr('app.research.query_report.query_report',
                        lambda **kwargs: query_report(report, **kwargs))
    main(['--experiment', 'EXP-000024', '--json'])
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'local'
    assert len(result['matches']) == 3
    main(['--list-sections'])
    assert 'Economics' in capsys.readouterr().out
    main(['--query', 'unavailable'])
    assert 'No matching report excerpts' in capsys.readouterr().out


def test_cli_conflicting_options_and_errors(monkeypatch, capsys, report):
    monkeypatch.setattr('app.research.query_report.query_report',
                        lambda **kwargs: query_report(report, **kwargs))
    for args in [['--list-sections', '--query', 'fees'], ['--experiment', 'EXP-999999']]:
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2
    assert 'not found' in capsys.readouterr().err


def test_committed_report_supports_all_persisted_experiments():
    before = REPORT_PATH.read_bytes()
    for experiment in ('EXP-000021', 'EXP-000022', 'EXP-000023', 'EXP-000024'):
        result = query_report(experiment=experiment)
        assert result['source'] == SOURCE
        assert result['matches']
        assert any('BASELINE_REJECT' in m['text'] for m in result['matches'])
    result = query_report(experiment='EXP-000024', section='Economics and primary triage')
    assert len(result['matches']) == 1
    assert 'NO_GROSS_EDGE' in result['matches'][0]['text']
    assert REPORT_PATH.read_bytes() == before


def test_cli_cannot_load_other_artifacts_or_call_network_even_with_api_credentials(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-placeholder')
    script = '''
import sys
from pathlib import Path
class BlockExecution:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('openai', 'requests', 'app.research.automation',
                   'app.research.optimizer', 'app.research.memory',
                   'app.research.backtester', 'app.research.simulation')
        if fullname.startswith(blocked):
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockExecution())
from app.research.query_report import main, REPORT_PATH
import contextlib
import io
# Warm up argparse/locale imports before restricting artifact reads.
with contextlib.redirect_stdout(io.StringIO()):
    main(['--list-sections'])
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect'):
        raise AssertionError('Forbidden operation: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        if Path(args[0]).resolve() != REPORT_PATH.resolve() or args[1] != 'r':
            raise AssertionError('Unexpected artifact access or write: ' + str(args))
sys.addaudithook(audit)
main(['--experiment', 'EXP-000024', '--json'])
'''
    completed = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['mode'] == 'local'
