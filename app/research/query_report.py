"""Deterministic, local retrieval from the committed failure-pattern report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SOURCE = 'research/reports/failure_patterns_exp_000021_000024.md'
REPORT_PATH = Path(__file__).resolve().parents[2] / SOURCE


def query_report(
    report: Path = REPORT_PATH,
    *,
    experiment: str | None = None,
    section: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Return exact lines with citations; filters intersect, without inference.

    Experiment-to-strategy identities come from the report's inventory table.
    The report is the only file read; linked artifacts are never followed.
    """
    for label, value in (('experiment', experiment), ('section', section), ('query', query)):
        if value is not None and not value.strip():
            raise ValueError(f'{label} must not be blank')
    content = report.read_bytes()
    lines = content.decode('utf-8').splitlines()
    if not any(line.strip() for line in lines):
        raise ValueError('Historical report is empty')

    headings = [line.lstrip('#').strip() for line in lines if line.startswith(('## ', '### '))]
    if section is not None and section.casefold() not in {s.casefold() for s in headings}:
        raise ValueError(f'Section not found in report: {section}')
    identities = dict(re.findall(r'^\| (EXP-\d{6}) \| ([A-Za-z_][A-Za-z0-9_]*) \|', '\n'.join(lines), re.MULTILINE))
    identity = None
    if experiment is not None:
        experiment = experiment.upper()
        if experiment not in identities:
            raise ValueError(f'Experiment not found in report inventory: {experiment}')
        identity = re.compile(r'(?<![\w])(?:' + re.escape(experiment) + '|'
                              + re.escape(identities[experiment]) + r')(?![\w])')

    matches = []
    current_section = ''
    table_header = None
    for number, line in enumerate(lines, 1):
        if line.startswith('#'):
            current_section = line.lstrip('#').strip()
        if line.startswith('|'):
            if table_header is None:
                table_header = {'line': number, 'text': line}
        else:
            table_header = None
        if not line.strip():
            continue
        if section is not None and current_section.casefold() != section.casefold():
            continue
        if query is not None and query.casefold() not in line.casefold():
            continue
        if identity is not None and not identity.search(line):
            continue
        match = {'section': current_section, 'line': number, 'text': line}
        if table_header is not None and number != table_header['line']:
            match['table_header'] = dict(table_header)
        matches.append(match)

    return {
        'mode': 'local',
        'source': report.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
        if report.resolve().is_relative_to(Path(__file__).resolve().parents[2]) else str(report),
        'source_sha256': hashlib.sha256(content).hexdigest(),
        'sections': headings,
        'matches': matches,
        'notice': 'Exact report excerpts. Triage classifications do not replace journal governance verdicts.',
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Query the committed historical report locally; no API calls.')
    parser.add_argument('--experiment', help='Experiment ID, e.g. EXP-000024')
    parser.add_argument('--section', help='Exact heading (case-insensitive); use --list-sections')
    parser.add_argument('--query', help='Literal case-insensitive substring, not a natural-language question')
    parser.add_argument('--list-sections', action='store_true')
    parser.add_argument('--json', action='store_true', help='Stable JSON with source hash and one-based line citations')
    args = parser.parse_args(argv)
    if args.list_sections and any((args.experiment, args.section, args.query)):
        parser.error('--list-sections cannot be combined with query filters')
    try:
        result = query_report(experiment=args.experiment, section=args.section, query=args.query)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.list_sections:
        result['matches'] = []
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"Source: {result['source']}\nSHA-256: {result['source_sha256']}")
    print(result['notice'])
    if args.list_sections:
        print('\n'.join(result['sections']))
    elif not result['matches']:
        print('No matching report excerpts.')
    else:
        for match in result['matches']:
            print(f"\n[{match['section']}, line {match['line']}]")
            if 'table_header' in match:
                header = match['table_header']
                print(f"[table header, line {header['line']}] {header['text']}")
            print(match['text'])


if __name__ == '__main__':
    main()
