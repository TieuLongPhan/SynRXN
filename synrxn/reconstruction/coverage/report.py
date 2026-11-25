"""Inspect task/table coverage; export optional reports to a separate directory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..paths import REPO, historical_file, physical_path

__source_file__ = historical_file('kdd/coverage/report.py')


def build(repo=REPO):
    """Read current contracts and rights decisions without modifying metadata."""
    registry = json.loads(physical_path('kdd/task_registry.json', repo).read_text(encoding='utf8'))
    audit = json.loads(physical_path('kdd/coverage/release_audit.json', repo).read_text(encoding='utf8'))
    releases = {(table['namespace'], table['key']) for table in audit['release_tables']}
    rights = {(table['namespace'], table['key']): table
              for table in audit['release_tables'] + audit['held_tables']}
    rows = []
    for task in registry['tasks']:
        for binding in task['bindings']:
            key = binding['namespace'], binding['key']
            terms = rights.get(key, {})
            rows.append({
                'family_id': task['family_id'], 'task_id': task['id'],
                'namespace': binding['namespace'], 'dataset_view': binding['key'],
                'source_id': binding.get('source_id', binding['key']),
                'source_rows': binding['rows'], 'counting_unit': binding['counting_unit'],
                'role': binding.get('role', 'primary'), 'release_eligible': key in releases,
                'release_license_expression': terms.get('license_expression'),
                'partition': terms.get('partition'), 'commercial_use_allowed': terms.get('commercial_use_allowed'),
                'reference_evidence': binding.get('reference_evidence', task['qualification']),
                'qualification': binding.get('qualification', task['qualification']),
            })
    families = []
    for family in registry['families']:
        tasks = [task for task in registry['tasks'] if task['family_id'] == family['id']]
        families.append({'family_id': family['id'], 'title': family['title'], 'local_tasks': len(tasks),
                         'release_eligible_tasks': sum(any((binding['namespace'], binding['key']) in releases
                                                           for binding in task['bindings']) for task in tasks)})
    return {'summary': {'families': len(families), 'local_definitions': len(registry['tasks']),
                        'release_eligible_definitions': sum(family['release_eligible_tasks'] for family in families),
                        'binding_rows': len(rows)},
            'families': families, 'bindings': rows}


def export(report: dict, output: Path, *, reports=False) -> None:
    """Write deliberately requested exports outside authoritative metadata."""
    output = Path(output).resolve()
    protected = (REPO / 'Data', REPO / 'archive', REPO / 'synrxn')
    if any(output == root or output.is_relative_to(root) for root in protected):
        raise ValueError('Choose a report directory outside source, data and historical evidence trees')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty report directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'coverage.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    if not reports:
        return
    with (output / 'task_dataset_counts.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['bindings'][0]))
        writer.writeheader()
        writer.writerows(report['bindings'])
    lines = ['# Task coverage', '', '| Family | Local tasks | Tasks with admitted data |', '| --- | ---: | ---: |']
    for family in report['families']:
        lines.append(f"| {family['title']} | {family['local_tasks']} | {family['release_eligible_tasks']} |")
    lines += ['', 'Rows and reference qualifications are retained in coverage.json. Task totals overlap; source units differ.',
              'This report describes available references, not completed predictive evaluations.', '']
    (output / 'coverage_report.md').write_text('\n'.join(lines), encoding='utf8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Export coverage JSON to a new directory; default prints counts only')
    parser.add_argument('--reports', action='store_true', help='Also export optional CSV and Markdown summaries')
    args = parser.parse_args(argv)
    if args.reports and args.output is None:
        parser.error('--reports requires an explicit --output directory')
    report = build()
    if args.output is not None:
        export(report, args.output, reports=args.reports)
    print(json.dumps(report['summary'], sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
