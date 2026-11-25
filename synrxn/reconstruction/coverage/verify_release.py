# Direct implementation; historical source identity is explicit below.
"""Verify catalogued data/dependencies and produce an explicit redistribution list.

Offline. Permissions are evidence decisions, never inferred from a filename or
from a compiler's software license. Unresolved and restricted assets remain
local and are never selected by the release manifest. Does not publish data.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/audit.py')
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import re
import sys
REPO = Path(__source_file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from synrxn.reconstruction.families.suite import audit_file, dataset_inventory, local_path
from synrxn.reconstruction.coverage.rights import source_has_evidence, table_has_evidence
from synrxn.reconstruction.coverage.license_policy import permitted, partition_for, commercial_allowed, obligations

def read(path):
    return json.loads(path.read_text())

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def branches(repo=REPO):
    return [(path.parent.name, path, read(path)) for path in sorted((repo / 'Data/coverage_expansion').glob('*/catalog.json'))]

def pinned_objects(value):
    """Find exact local evidence/native-asset objects, including nested rights."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and value.get('sha256'):
            yield value
        if isinstance(value.get('grant_snapshot'), str) and value.get('grant_snapshot_sha256'):
            yield {'path': value['grant_snapshot'], 'sha256': value['grant_snapshot_sha256']}
        if isinstance(value.get('license_evidence'), str) and value.get('license_evidence_sha256'):
            yield {'path': value['license_evidence'], 'sha256': value['license_evidence_sha256']}
        for item in value.values():
            yield from pinned_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from pinned_objects(item)

def release_decision(table, source, catalog, repo=REPO):
    license_ = table.get('license_expression', table.get('license', source.get('license')))
    reasons = []
    if not permitted(license_):
        reasons.append('Unsupported, restricted or unresolved data license')
    if table.get('release_eligible') is False or source.get('redistribution_allowed') is False:
        reasons.append('Explicitly excluded by upstream-rights audit')
    if not table_has_evidence(table, source, catalog, repo):
        reasons.append('Missing pinned source-specific redistribution evidence')
    return {'release_eligible': not reasons, 'license_expression': license_, 'reasons': reasons, 'partition': partition_for(license_), 'commercial_use_allowed': commercial_allowed(license_), 'license_obligations': obligations(license_) if permitted(license_) else []}

def rows_of(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()
    elif path.name.endswith('.jsonl.gz'):
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                yield json.loads(line)
    elif path.suffix == '.csv':
        csv.field_size_limit(32 * 1024 * 1024)
        with path.open(newline='') as handle:
            yield from csv.DictReader(handle)
    else:
        raise ValueError('Unsupported coverage table format: ' + str(path))

def verify_contract(table, repo=REPO):
    """Check actual rows, identifiers, input/target separation and references."""
    errors, columns, count, identifiers = ([], set(), 0, set())
    path = local_path(repo, table['path'])
    contracts = table.get('task_fields', {})
    if set(contracts) != set(table['task_ids']):
        errors.append('Task IDs and contracts disagree')
    for task, contract in contracts.items():
        if not contract.get('inputs') or not contract.get('targets') or (not contract.get('group_fields')):
            errors.append(task + ': missing input/target/group contract')
        if set(contract.get('inputs', [])) & set(contract.get('targets', [])):
            errors.append(task + ': a target column is also an input')
        if set(contract.get('input_keys', [])) & set(contract.get('target_keys', [])):
            errors.append(task + ': a logical target is also an input')
    for row in rows_of(path):
        count += 1
        columns.update(row)
        identifier = row.get('sample_id', row.get('record_id', row.get('source_record_id')))
        if identifier is None or str(identifier) == '':
            errors.append('Missing row provenance identifier')
            break
        if identifier in identifiers:
            errors.append('Duplicate exported row identifier: ' + str(identifier))
            break
        identifiers.add(identifier)
        for task, contract in contracts.items():
            for key in contract.get('targets', []) + contract.get('group_fields', []):
                if key not in row or row[key] is None or (isinstance(row[key], str) and (not row[key])):
                    errors.append(task + ': missing target/group value ' + key)
        if table.get('storage_layout') == 'json_objects':
            inputs, targets = (json.loads(row['inputs_json']), json.loads(row['targets_json']))
            provenance = json.loads(row['provenance_json'])
            if not provenance.get('source_locator') or not provenance.get('source_sha256'):
                errors.append('Missing native source locator/checksum')
                break
            for task, contract in contracts.items():
                if not set(contract.get('target_keys', [])) <= set(targets):
                    errors.append(task + ': missing logical target')
                    break
                if not set(inputs) <= set(contract.get('input_keys', [])):
                    errors.append(task + ': undeclared logical input')
                    break
    if count != table['rows']:
        errors.append(f"Row count mismatch: {count} != {table['rows']}")
    for task, contract in contracts.items():
        requested = set(contract.get('inputs', []) + contract.get('targets', []) + contract.get('group_fields', []))
        if not requested <= columns:
            errors.append(task + ': missing declared columns: ' + ','.join(sorted(requested - columns)))
    return {'path': table['path'], 'observed_rows': count, 'status': 'failed' if errors else 'verified', 'errors': sorted(set(errors))}

def build(repo=REPO, *, check_rows=True, check_assets=True):
    legacy = read(repo / 'kdd/coverage/legacy_rights.json')
    legacy_by_key = {(t['namespace'], t['key']): t for t in legacy['tables']}
    release, held, checks, contracts = ([], [], [], [])
    assets = {}
    for item in pinned_objects(legacy):
        assets[item['path']] = item
    for item in dataset_inventory(repo):
        if item['namespace'] == 'coverage':
            continue
        checks.append(audit_file(item, repo, check_rows))
        rights = legacy_by_key[item['namespace'], item['key']]
        entry = {**item, 'license_expression': rights.get('release_license_expression', rights.get('verified_upstream_license_spdx', rights['declared_license_spdx'])), 'partition': rights['partition'], 'rights_status': rights['status'], 'evidence_ids': rights['evidence_ids'], 'qualification': rights['reason']}
        entry['partition'] = partition_for(entry['license_expression'])
        entry['commercial_use_allowed'] = commercial_allowed(entry['license_expression'])
        entry['license_obligations'] = obligations(entry['license_expression']) if permitted(entry['license_expression']) else []
        if rights['release_eligible'] and permitted(entry['license_expression']):
            release.append(entry)
        else:
            held.append({**entry, 'reason': rights['reason'], 'remaining_qualification': rights['remaining_qualification']})
    raw_release = {}
    notices = {}
    catalogs = []
    for branch, path, catalog in branches(repo):
        catalogs.append({'path': path.relative_to(repo).as_posix(), 'sha256': digest(path)})
        sources = {s['id']: s for s in catalog['sources']}
        for item in pinned_objects(catalog):
            previous = assets.get(item['path'])
            if previous and previous['sha256'] != item['sha256']:
                raise ValueError('Conflicting dependency checksum: ' + item['path'])
            assets[item['path']] = item
        for field, hashfield in [('adapter', 'adapter_sha256'), ('builder', 'builder_sha256'), ('source_lock', 'source_lock_sha256')]:
            if catalog.get(field) and catalog.get(hashfield):
                assets[catalog[field]] = {'path': catalog[field], 'sha256': catalog[hashfield]}
        if catalog.get('rights_audit_path'):
            for item in pinned_objects(read(repo / catalog['rights_audit_path'])):
                assets[item['path']] = item
        for table in catalog['tables']:
            source = sources[table['source_id']]
            decision = release_decision(table, source, catalog, repo)
            fmt = 'parquet' if table['path'].endswith('.parquet') else 'jsonl.gz' if table['path'].endswith('.jsonl.gz') else 'csv'
            entry = {'namespace': 'coverage', 'key': branch + '/' + table['id'], 'path': table['path'], 'format': fmt, 'expected_sha256': table['sha256'], 'expected_bytes': table.get('bytes'), 'expected_rows': table['rows'], 'source_id': table['source_id'], 'source_url': table['source_url'], 'task_ids': table['task_ids'], 'counting_unit': table['counting_unit'], 'reference_evidence': table['reference_evidence'], 'qualification': table['qualification'], **decision}
            checks.append(audit_file(entry, repo, False))
            if check_rows:
                contracts.append(verify_contract(table, repo))
            (release if decision['release_eligible'] else held).append(entry)
            if not decision['release_eligible']:
                continue
            for item in source.get('assets', source.get('files', [])) + table.get('source_assets', []):
                if not isinstance(item, dict) or not item.get('path') or (not item.get('sha256')):
                    continue
                allowed = item.get('redistribution_allowed', source.get('redistribution_allowed', False))
                if 'redistribution_allowed' not in item and item.get('license') and source_has_evidence(source, catalog, repo):
                    allowed = permitted(item['license'])
                if allowed:
                    raw_release[item['path']] = {**item, 'source_id': source['id'], 'license_expression': decision['license_expression']}
        for table in catalog.get('quarantined_tables', []):
            held.append({'namespace': 'coverage_quarantine', 'key': branch + '/' + table['id'], 'path': table['path'], 'reason': table.get('redistribution_status', 'Explicit source-rights quarantine')})
            if (repo / table['path']).exists() and table.get('sha256'):
                checks.append(audit_file({'namespace': 'coverage_quarantine', 'key': branch + '/' + table['id'], 'path': table['path'], 'expected_sha256': table['sha256']}, repo))
        for candidate in sorted(path.parent.rglob('*')):
            if candidate.is_file() and 'quarantine' not in candidate.parts and any((word in candidate.name.upper() for word in ['NOTICE', 'LICENSE', 'ATTRIBUTION'])):
                relative = candidate.relative_to(repo).as_posix()
                notices[relative] = {'path': relative, 'sha256': digest(candidate), 'bytes': candidate.stat().st_size}
    for item in sorted(assets.values(), key=lambda a: a['path']):
        if check_assets:
            checks.append(audit_file({'namespace': 'dependency', 'key': item['path'], 'path': item['path'], 'expected_sha256': item['sha256'], 'expected_bytes': item.get('bytes')}, repo))
    errors = [r for r in checks + contracts if r['status'] != 'verified']
    return {'schema': 'synrxn.coverage-release-audit.v1', 'status': 'failed' if errors else 'passed', 'policy': 'Verified dataset-specific grants govern each table. CC BY/CC0 and permissive grants admitted with attribution; ShareAlike and GNU copyleft conditions retained. User-authorized CC BY-NC and BY-NC-SA data are admitted in explicit noncommercial partitions. No unrestricted commercial-use license is inferred. ND adaptations, internal-use-only terms, unresolved grants and unresolved license conflicts remain excluded. No publication performed.', 'row_audit_requested': check_rows, 'dependency_audit_requested': check_assets, 'catalogs': catalogs, 'checks': checks, 'contracts': contracts, 'errors': errors, 'release_tables': release, 'held_tables': held, 'release_native_assets': list(raw_release.values()), 'release_notices': list(notices.values()), 'summary': {'catalogued_tables': len(release) + sum((t['namespace'] != 'coverage_quarantine' for t in held)), 'quarantined_tables': sum((t['namespace'] == 'coverage_quarantine' for t in held)), 'release_tables': len(release), 'held_tables': len(held), 'release_partition_counts': dict(Counter((t['partition'] for t in release))), 'verified_files': sum((c['status'] == 'verified' for c in checks)), 'verified_new_table_contracts': sum((c['status'] == 'verified' for c in contracts)), 'failed_checks': len(errors)}, 'limits': ['Rows and task views overlap; never sum them as independent experiments.', 'This audit qualifies source/field contracts and redistributability; predictive benchmarks are pending.', 'The manifest selects explicit paths. It never authorizes bundling Data/raw or copying quarantined carriers.']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'kdd/coverage/release_audit.json')
    parser.add_argument('--no-rows', action='store_true')
    parser.add_argument('--no-assets', action='store_true', help='Skip native asset checksum verification')
    args = parser.parse_args()
    result = build(check_rows=not args.no_rows, check_assets=not args.no_assets)
    from synrxn.reconstruction.coverage.publication_notices import build as build_notices
    result['release_notices'] += build_notices(REPO, result['release_tables'], read(REPO / 'kdd/coverage/legacy_rights.json'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    manifest = {k: result[k] for k in ['schema', 'status', 'row_audit_requested', 'dependency_audit_requested', 'policy', 'summary', 'catalogs', 'release_tables', 'held_tables', 'release_native_assets', 'release_notices', 'limits']}
    manifest['release_verification'] = 'complete' if result['status'] == 'passed' and result['row_audit_requested'] and result['dependency_audit_requested'] else 'incomplete; do not use for release packaging'
    args.output.with_name('release_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': result['status'], 'summary': result['summary'], 'errors': result['errors'][:10]}))
    return int(result['status'] != 'passed')
__configure_runtime(globals(), 'kdd/coverage/audit.py')
audit_contract = verify_contract
if __name__ == '__main__':
    raise SystemExit(main())
