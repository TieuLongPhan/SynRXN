# Direct implementation; historical source identity is explicit below.
"""Stream every catalogued table and describe its content without changing it.

Run ``python kdd/coverage/quality.py --output reports/content-quality.json``.
Checksums and logical row counts are independently checked. Missingness,
identifier duplication and logical JSON issues are findings for review, not
automatic scientific exclusions or assertions about physical truth.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/quality.py')
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import sys
REPO = Path(__source_file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from synrxn.reconstruction.families.suite import dataset_inventory, local_path
IDENTIFIERS = frozenset({'sample_id', 'record_id', 'source_record_id', 'r_id'})
JSON_FIELDS = frozenset({'inputs_json', 'targets_json'})
POTENTIAL_MISSING = frozenset({'nan', 'null', 'none', 'na', 'n/a', '<na>', 'nat'})
MAX_EXAMPLES = 5

def sha256(path: Path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def missing_kind(value):
    """Zero/false/empty containers are valid values; strings are not coerced."""
    if value is None:
        return 'none'
    if isinstance(value, str) and (not value or value.isspace()):
        return 'empty_string'
    if isinstance(value, float) and (not math.isfinite(value)):
        return 'nonfinite_float'
    return None

def potential_missing_string(value):
    return isinstance(value, str) and len(value) <= 8 and (value.strip().casefold() in POTENTIAL_MISSING)

def new_column():
    return {'present_rows': 0, 'explicit_missing_count': 0, 'missing_none': 0, 'missing_empty_string': 0, 'missing_nonfinite_float': 0, 'potential_missing_string_count': 0, 'empty_container_count': 0, 'types': Counter()}

def add_value(stats, value):
    stats['present_rows'] += 1
    stats['types'][type(value).__name__] += 1
    missing = missing_kind(value)
    if missing:
        stats['explicit_missing_count'] += 1
        stats['missing_' + missing] += 1
    elif potential_missing_string(value):
        stats['potential_missing_string_count'] += 1
    if isinstance(value, (list, dict, tuple)) and (not value):
        stats['empty_container_count'] += 1

def finish_column(stats, denominator):
    result = dict(stats)
    result['types'] = dict(sorted(stats['types'].items()))
    result['absent_rows'] = denominator - result['present_rows']
    result['missing_count'] = result['explicit_missing_count'] + result['absent_rows']
    return result

def new_identifier():
    return {'present_rows': 0, 'nonempty_rows': 0, 'potential_missing_string_count': 0, 'non_scalar_rows': 0, 'duplicate_occurrences': 0, 'duplicate_examples': [], 'seen': set(), 'duplicates': set()}

def add_identifier(stats, value):
    stats['present_rows'] += 1
    if missing_kind(value):
        return
    stats['nonempty_rows'] += 1
    stats['potential_missing_string_count'] += int(potential_missing_string(value))
    if isinstance(value, (dict, list, tuple)):
        stats['non_scalar_rows'] += 1
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    else:
        encoded = value
    token = (type(value).__name__, encoded)
    if token in stats['seen']:
        stats['duplicate_occurrences'] += 1
        if token not in stats['duplicates'] and len(stats['duplicate_examples']) < MAX_EXAMPLES:
            stats['duplicate_examples'].append(str(value)[:160])
        stats['duplicates'].add(token)
    else:
        stats['seen'].add(token)

def finish_identifier(stats, denominator):
    return {key: value for key, value in stats.items() if key not in {'seen', 'duplicates'}} | {'missing_or_absent_rows': denominator - stats['nonempty_rows'], 'distinct_nonempty_values': len(stats['seen']), 'duplicate_distinct_values': len(stats['duplicates'])}

def new_json_profile():
    return {'present_rows': 0, 'missing_payload_count': 0, 'potential_missing_string_count': 0, 'parsed_object_rows': 0, 'parse_errors': 0, 'nonobject_payloads': 0, 'parse_error_examples': [], 'logical_keys': {}}

def add_json_profile(stats, value, index):
    stats['present_rows'] += 1
    if missing_kind(value):
        stats['missing_payload_count'] += 1
        return
    stats['potential_missing_string_count'] += int(potential_missing_string(value))
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError) as error:
        stats['parse_errors'] += 1
        if len(stats['parse_error_examples']) < MAX_EXAMPLES:
            stats['parse_error_examples'].append({'row_index_0based': index, 'error': type(error).__name__})
        return
    if not isinstance(payload, dict):
        stats['nonobject_payloads'] += 1
        return
    stats['parsed_object_rows'] += 1
    for key, item in payload.items():
        name = str(key)
        column = stats['logical_keys'].get(name)
        if column is None:
            column = stats['logical_keys'][name] = new_column()
        add_value(column, item)

def finish_json_profile(stats, denominator):
    return {key: value for key, value in stats.items() if key != 'logical_keys'} | {'absent_payload_rows': denominator - stats['present_rows'], 'logical_keys': {key: finish_column(column, stats['parsed_object_rows']) for key, column in sorted(stats['logical_keys'].items())}, 'interpretation': 'Logical-key missing counts use parsed object rows as denominator; optional and empty reference collections may be valid.'}

def _malformed(state, index, reason, **detail):
    state['malformed_rows'] += 1
    if len(state['malformed_row_examples']) < MAX_EXAMPLES:
        state['malformed_row_examples'].append({'row_index_0based': index, 'reason': reason, **detail})

def iter_rows(path: Path, format_: str, state):
    """Yield logical records; quoted CSV newlines never create extra records."""
    if format_ in {'csv', 'csv.gz'}:
        csv.field_size_limit(128 * 1024 * 1024)
        opener = gzip.open if format_ == 'csv.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError('CSV container has no header')
            state['declared_columns'] = header
            state['duplicate_header_names'] = sorted((key for key, value in Counter(header).items() if value > 1))
            index = 0
            for values in reader:
                if not values:
                    state['blank_records_skipped'] += 1
                    continue
                if len(values) != len(header):
                    _malformed(state, index, 'CSV field count differs from header', fields=len(values), expected_fields=len(header))
                yield {key: value for key, value in zip(header, values)}
                index += 1
    elif format_ == 'parquet':
        import pyarrow.parquet as pq
        container = pq.ParquetFile(path)
        state['declared_columns'] = container.schema_arrow.names
        for batch in container.iter_batches(batch_size=4096):
            yield from batch.to_pylist()
    elif format_ in {'jsonl', 'jsonl.gz'}:
        opener = gzip.open if format_ == 'jsonl.gz' else open
        with opener(path, 'rt', encoding='utf-8') as handle:
            index = 0
            for physical_line, line in enumerate(handle, 1):
                if not line.strip():
                    state['blank_records_skipped'] += 1
                    continue
                try:
                    row = json.loads(line)
                    decoded = True
                except ValueError as error:
                    _malformed(state, index, 'JSONL decode failure', physical_line=physical_line)
                    state['container_errors'].append(f'JSONL syntax error at line {physical_line}: {type(error).__name__}')
                    row = None
                    decoded = False
                if decoded and (not isinstance(row, dict)):
                    _malformed(state, index, 'JSONL record is not an object', physical_line=physical_line)
                    row = None
                yield row
                index += 1
    else:
        raise ValueError('Unsupported table container: ' + str(format_))

def audit_table(record: dict, repo: Path=REPO):
    result = {**record, 'status': 'failed', 'observed_rows': 0, 'profiled_object_rows': 0, 'observed_bytes': None, 'observed_sha256': None, 'columns': {}, 'candidate_identifiers': {}, 'logical_json': {}, 'malformed_rows': 0, 'malformed_row_examples': [], 'blank_records_skipped': 0, 'declared_columns': [], 'duplicate_header_names': [], 'container_errors': [], 'fatal_errors': [], 'review_findings': []}
    columns, identifiers, logical = ({}, {}, {})
    try:
        path = local_path(repo, record['path'])
        before = path.stat()
        result['observed_bytes'] = before.st_size
        result['observed_sha256'] = sha256(path)
        expected_hash = record.get('expected_sha256', record.get('sha256'))
        if expected_hash and result['observed_sha256'] != expected_hash:
            result['fatal_errors'].append('Checksum mismatch against the catalog pin')
        expected_bytes = record.get('expected_bytes', record.get('bytes'))
        if expected_bytes is not None and before.st_size != expected_bytes:
            result['review_findings'].append({'kind': 'byte_size_metadata_mismatch', 'expected': expected_bytes, 'observed': before.st_size})
        format_ = record.get('format')
        if not format_:
            format_ = 'csv.gz' if path.name.endswith('.csv.gz') else 'jsonl.gz' if path.name.endswith('.jsonl.gz') else path.suffix.lstrip('.')
        for row in iter_rows(path, format_, result):
            index = result['observed_rows']
            result['observed_rows'] += 1
            if row is None:
                continue
            result['profiled_object_rows'] += 1
            for key, value in row.items():
                column = columns.get(key)
                if column is None:
                    column = columns[key] = new_column()
                add_value(column, value)
                if key in IDENTIFIERS:
                    identifier = identifiers.get(key)
                    if identifier is None:
                        identifier = identifiers[key] = new_identifier()
                    add_identifier(identifier, value)
                if key in JSON_FIELDS:
                    profile = logical.get(key)
                    if profile is None:
                        profile = logical[key] = new_json_profile()
                    add_json_profile(profile, value, index)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            result['observed_after_sha256'] = sha256(path)
            if result['observed_after_sha256'] != result['observed_sha256']:
                result['fatal_errors'].append('Checksum changed while content was being audited')
            else:
                result['review_findings'].append({'kind': 'file_metadata_changed_but_bytes_identical'})
    except Exception as error:
        result['container_errors'].append(type(error).__name__ + ': ' + str(error))
    if result['container_errors']:
        result['fatal_errors'].append('Malformed or unreadable table container')
    expected_rows = record.get('expected_rows', record.get('rows'))
    if expected_rows is not None and result['observed_rows'] != expected_rows:
        result['fatal_errors'].append(f"Row count mismatch: {result['observed_rows']} != {expected_rows}")
    total = result['observed_rows']
    for key in result['declared_columns']:
        columns.setdefault(key, new_column())
        if key in IDENTIFIERS:
            identifiers.setdefault(key, new_identifier())
        if key in JSON_FIELDS:
            logical.setdefault(key, new_json_profile())
    result['columns'] = {key: finish_column(stats, total) for key, stats in sorted(columns.items())}
    result['candidate_identifiers'] = {key: finish_identifier(stats, total) for key, stats in sorted(identifiers.items())}
    result['logical_json'] = {key: finish_json_profile(stats, total) for key, stats in sorted(logical.items())}
    if not identifiers:
        result['review_findings'].append({'kind': 'no_candidate_row_identifier', 'candidate_names': sorted(IDENTIFIERS)})
    for key, stats in result['candidate_identifiers'].items():
        for field in ['duplicate_occurrences', 'missing_or_absent_rows', 'non_scalar_rows', 'potential_missing_string_count']:
            if stats[field]:
                result['review_findings'].append({'kind': 'identifier_' + field, 'column': key, 'count': stats[field]})
    for key, stats in result['columns'].items():
        for field in ['missing_count', 'potential_missing_string_count']:
            if stats[field]:
                result['review_findings'].append({'kind': field, 'column': key, 'count': stats[field]})
    for key, stats in result['logical_json'].items():
        for field in ['parse_errors', 'nonobject_payloads', 'missing_payload_count', 'absent_payload_rows']:
            if stats[field]:
                result['review_findings'].append({'kind': 'logical_json_' + field, 'column': key, 'count': stats[field]})
        for logical_key, intrinsic in stats['logical_keys'].items():
            for field in ['missing_count', 'potential_missing_string_count']:
                if intrinsic[field]:
                    result['review_findings'].append({'kind': 'logical_key_' + field, 'column': key, 'logical_key': logical_key, 'count': intrinsic[field]})
    if result['malformed_rows']:
        result['review_findings'].append({'kind': 'malformed_rows', 'count': result['malformed_rows']})
    if result['duplicate_header_names']:
        result['review_findings'].append({'kind': 'duplicate_header_names', 'columns': result['duplicate_header_names']})
    result['fatal_errors'] = sorted(set(result['fatal_errors']))
    result['status'] = 'failed' if result['fatal_errors'] else 'profiled'
    return result

def build(repo=REPO, inventory=None, progress=None):
    inventory = dataset_inventory(repo) if inventory is None else inventory
    tables = []
    for index, record in enumerate(inventory, 1):
        result = audit_table(record, repo)
        tables.append(result)
        if progress:
            progress(index, len(inventory), result)
    fatal = [table for table in tables if table['fatal_errors']]
    return {'schema': 'synrxn.content-quality-audit.v1', 'status': 'failed' if fatal else 'passed', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'auditor': {'path': Path(__source_file__).resolve().relative_to(REPO).as_posix(), 'sha256': sha256(Path(__source_file__))}, 'interpretation': 'Descriptive source-content audit. Missingness, duplicate candidate IDs and logical payload issues require source/task review; they do not automatically determine eligibility, independent observations or physical truth. Source tables are not modified.', 'missing_policy': 'None, empty/whitespace strings and nonfinite floating values are explicit missing. Absent fields are counted separately and included in missing_count. Strings nan/null/none/na/n/a/<na>/nat are potential missing only. Zero, false and empty containers are retained as values.', 'identifier_policy': 'Only exact candidate column names sample_id/record_id/source_record_id/r_id are checked, with native types preserved. Repeated reactions, groups and source memberships are never treated as row duplicates.', 'tables': tables, 'summary': {'tables': len(tables), 'tables_by_namespace': dict(Counter((t['namespace'] for t in tables))), 'observed_rows': sum((t['observed_rows'] for t in tables)), 'profiled_object_rows': sum((t['profiled_object_rows'] for t in tables)), 'fatal_tables': len(fatal), 'tables_with_review_findings': sum((bool(t['review_findings']) for t in tables)), 'malformed_rows': sum((t['malformed_rows'] for t in tables)), 'candidate_identifier_duplicate_occurrences': sum((s['duplicate_occurrences'] for t in tables for s in t['candidate_identifiers'].values()))}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'reports/content-quality.json')
    args = parser.parse_args()

    def progress(index, total, table):
        print(json.dumps({'table': index, 'of': total, 'key': table['namespace'] + ':' + table['key'], 'rows': table['observed_rows'], 'status': table['status'], 'review_findings': len(table['review_findings'])}), flush=True)
    result = build(progress=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': result['status'], 'summary': result['summary'], 'output': str(args.output)}))
    return int(result['status'] != 'passed')
__configure_runtime(globals(), 'kdd/coverage/quality.py')
if __name__ == '__main__':
    raise SystemExit(main())
