# Direct implementation; historical source identity is explicit below.
"""Reproduce interpretation of the two observed partition-local ID findings.

This checks source-record identity. It does not establish chemical separation
between splits and does not change records or override native split roles.
"""
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/identifier_namespaces.py')
import argparse
import csv
import gzip
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import sys
REPO = Path(__source_file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from synrxn.reconstruction.families.suite import local_path
DECLARATIONS = {('canonical', 'classification/ecreact'): ('r_id', 'split'), ('canonical', 'yield/uspto_yield'): ('source_record_id', 'upstream_split')}

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def build(quality_path, repo=REPO):
    quality = json.loads(quality_path.read_text())
    results = []
    for table in quality['tables']:
        pair = DECLARATIONS.get((table['namespace'], table['key']))
        if not pair:
            continue
        id_field, namespace_field = pair
        path = local_path(repo, table['path'])
        if sha(path) != table['observed_sha256']:
            raise ValueError('Source changed since content audit: ' + table['path'])
        seen, bare, duplicates, missing, count = (set(), set(), 0, 0, 0)
        with gzip.open(path, 'rt', newline='') as handle:
            for row in csv.DictReader(handle):
                count += 1
                identifier, namespace = (row[id_field], row[namespace_field])
                if not identifier or not namespace:
                    missing += 1
                key = (namespace, identifier)
                duplicates += int(key in seen)
                seen.add(key)
                bare.add(identifier)
        if count != table['observed_rows']:
            raise ValueError('Source row count changed')
        results.append({'namespace': table['namespace'], 'key': table['key'], 'path': table['path'], 'source_sha256': table['observed_sha256'], 'rows': count, 'identifier_field': id_field, 'native_namespace_field': namespace_field, 'bare_identifier_repetitions': count - len(bare), 'within_namespace_duplicate_occurrences': duplicates, 'missing_identity_components': missing, 'interpretation': 'Native identifier is partition-local; use the native partition plus identifier as the source-record key. This is not evidence of chemical train/test independence.'})
    return {'schema': 'synrxn.identifier-namespace-audit.v1', 'source_quality_report_sha256': sha(quality_path), 'auditor_sha256': sha(Path(__source_file__)), 'tables': results, 'status': 'verified' if len(results) == len(DECLARATIONS) and all((not r['within_namespace_duplicate_occurrences'] and (not r['missing_identity_components']) for r in results)) else 'review_required'}
__configure_runtime(globals(), 'kdd/coverage/identifier_namespaces.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-report', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=REPO / 'kdd/coverage/identifier_namespace_audit.json')
    args = parser.parse_args()
    result = build(args.quality_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
