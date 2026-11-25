# Direct implementation; historical source identity is explicit below.
"""Stage SynEPD references and audit strict replay with at most two CPU cores.

This prepares data and its evaluation contract. It runs no predictor, installs
no packages, edits no upstream repository and changes no canonical release.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/prepare_synepd_task.py')
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import shutil
import sqlite3
import subprocess
import sys
ROOT = Path(__source_file__).resolve().parents[1]

def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def packed(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synepd-dir', type=Path, default=ROOT.parent / 'SynEPD')
    parser.add_argument('--synkit-dir', type=Path, help='Use an explicitly selected SynKit development checkout without installing it.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'ncs/artifacts').resolve()) or output.exists():
        raise ValueError('choose a fresh output directory below ncs/artifacts')
    if args.synkit_dir:
        sys.path.insert(0, str(args.synkit_dir.resolve()))
    sys.path.insert(0, str(ROOT))
    import synkit
    from synkit.Mechanism import MechanismRecord, MechanismReplayer, mechanism_from_legacy_epd
    from synrxn.metrics.mechanism import METRIC_CONTRACT, evaluate_mechanism_predictions
    evaluate_mechanism_predictions({}, {}, top_k=(1,))
    source_root = args.synepd_dir.resolve()
    source = source_root / 'data'
    manifest = json.loads((source / 'release-manifest.json').read_text())
    snapshots = {'release-manifest.json': digest(source / 'release-manifest.json')}
    for entry in [manifest['database'], *manifest['sources']]:
        path = source / entry['name']
        actual = digest(path)
        if actual != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError('source release pin mismatch: ' + path.name)
        snapshots[path.name] = actual
    payload = json.loads((source / 'polar.json').read_text())
    records = payload['records']
    if len(records) != payload['count'] or len({row['id'] for row in records}) != len(records):
        raise ValueError('source count or source IDs are inconsistent')
    connection = sqlite3.connect((source / 'epdb.sqlite').as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('source SQLite integrity check failed')
    if connection.execute('PRAGMA foreign_key_check').fetchall():
        raise ValueError('source SQLite foreign key check failed')
    actual_counts = {}
    for table, expected in manifest['database']['counts'].items():
        observed = connection.execute('SELECT count(*) FROM "' + table + '"').fetchone()[0]
        if observed != expected:
            raise ValueError('source SQLite count differs: ' + table)
        actual_counts[table] = observed
    db_rows = {row['case_id']: dict(row) for row in connection.execute('SELECT * FROM reaction')}
    arrows = {}
    for row in connection.execute('SELECT * FROM epd_arrow ORDER BY reaction_id, arrow_index'):
        arrows.setdefault(row['reaction_id'], []).append([row['arrow_type_code'], json.loads(row['source_atoms']), json.loads(row['target_atoms'])])
    centers = {row['reaction_id']: dict(row) for row in connection.execute('SELECT i.reaction_id,r.wlhash AS rc_hash,m.wlhash AS mc_hash FROM its i JOIN reaction_center r ON r.id=i.rc_id JOIN mechanistic_center m ON m.id=i.mc_id')}

    def source_case(row):
        return row.get('case_id') or f"{row.get('family', 'polar')}_{row['id']:06d}"
    if set(db_rows) != {source_case(row) for row in records}:
        raise ValueError('JSON/SQLite reaction membership differs')
    output.mkdir(parents=True)
    evidence = output / 'evidence'
    evidence.mkdir()
    for name in snapshots:
        shutil.copyfile(source / name, evidence / name)
    for name in ('README.md', 'LICENSE', 'pyproject.toml'):
        shutil.copyfile(source_root / name, evidence / name)
    code = output / 'code'
    code.mkdir()
    for path in (Path(__source_file__), ROOT / 'synrxn/metrics/mechanism.py'):
        destination = code / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    backend_root = Path(synkit.__file__).parent
    backend_code = {p.relative_to(backend_root).as_posix(): digest(p) for p in backend_root.rglob('*.py')}
    shutil.copytree(backend_root / 'Mechanism', code / 'backend/synkit/Mechanism', ignore=shutil.ignore_patterns('__pycache__'))
    if args.synkit_dir:
        patch = subprocess.check_output(['git', 'diff', '--', 'synkit/Mechanism/equivalence.py'], cwd=args.synkit_dir, text=True)
        (evidence / 'mapped-hydrogen-comparison.patch').write_text(patch)
    replayer = MechanismReplayer(validation='strict', repair=False, verify_stereo='stepwise')
    statuses, levels, modes, issue_codes, families, reaction_groups = (Counter() for _ in range(6))
    eligible = primary_tier = 0
    fields = ['sample_id', 'source_record_id', 'source_case_id', 'source_sqlite_id', 'reaction_smiles', 'electron_path_json', 'mechanism_record_json', 'representation_json', 'taxonomy_json', 'reaction_name', 'overall_reaction_group', 'rc_group', 'mc_group', 'reference_replay_status', 'verification_level', 'source_version', 'license_spdx']
    with (output / 'synepd-mechanisms.csv.gz').open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed, io.TextIOWrapper(compressed, encoding='utf8', newline='') as table, (output / 'model-inputs.jsonl').open('w') as inputs, (output / 'reference-audit.jsonl').open('w') as audits:
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(records, start=1):
            identifier = row['id']
            source_row = db_rows[source_case(row)]
            sqlite_id = source_row['id']
            if row['rsmi'] != source_row['aam_key'] or row['epd'] != arrows.get(sqlite_id):
                raise ValueError(f'JSON/SQLite reaction or arrows differ for {identifier}')
            policy = row.get('epd_representation') or {'mode': 'exact'}
            mode = policy['mode']
            if mode not in {'exact', 'closed_shell_pair'}:
                raise ValueError(f'unsupported source representation policy: {mode}')
            case_id = 'synepd-' + hashlib.sha256(f"{manifest['dataset_release']['version']}:{identifier}".encode()).hexdigest()[:20]
            serialized, certificate = (None, None)
            audit = {'sample_id': case_id, 'source_record_id': identifier, 'source_case_id': source_case(row), 'source_sqlite_id': sqlite_id, 'representation': policy}
            try:
                record = mechanism_from_legacy_epd(row['rsmi'], row['epd'], provenance={'dataset': 'SynEPD', 'source_record_id': identifier, 'source_version': manifest['dataset_release']['version']})
                if mode == 'closed_shell_pair':
                    record = replace(record, metadata={'endpoint_resource_policy': mode, 'closed_shell_atom_maps': policy['closed_shell_atom_maps']})
                serialized = record.to_dict()
                if MechanismRecord.from_dict(serialized).to_dict() != serialized:
                    raise ValueError('mechanism serialization does not round-trip')
                certificate = replayer.replay(record).certificate.to_dict()
                audit.update(status=certificate['status'], certificate=certificate)
            except Exception as exc:
                audit.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
            statuses[audit['status']] += 1
            level = certificate['verification_level'] if certificate else 'UNAVAILABLE'
            levels[level] += 1
            modes[mode] += 1
            if certificate:
                issue_codes.update((issue['code'] for issue in certificate['issues']))
            eligible += audit['status'] == 'VALID'
            primary_tier += audit['status'] == 'VALID' and level in {'EXACT', 'NORMALIZED'}
            group = 'rxn-' + hashlib.sha256(source_row['canonical_rsmi'].encode()).hexdigest()
            reaction_groups[group] += 1
            codes = row.get('tax_codes') or [row['tax_code']]
            families.update({'.'.join(value.split('.')[:2]) for value in codes})
            center = centers[sqlite_id]
            writer.writerow(dict(zip(fields, [case_id, identifier, source_case(row), sqlite_id, row['rsmi'], packed(row['epd']), packed(serialized), packed(policy), packed(codes), row['reaction_name'], group, center['rc_hash'], center['mc_hash'], audit['status'], level, manifest['dataset_release']['version'], 'CC-BY-4.0'])))
            inputs.write(packed({'sample_id': case_id, 'mapped_overall_reaction': row['rsmi']}) + '\n')
            audits.write(packed(audit) + '\n')
            if index % 100 == 0:
                print(f'Audited {index}/{len(records)} reference annotations: {dict(statuses)}', flush=True)
    connection.close()
    for name, before in snapshots.items():
        if digest(source / name) != before:
            raise ValueError('source changed during preparation: ' + name)
    report = {'schema': 'synrxn.synepd-task-preparation.v1', 'recorded_at': datetime.now(timezone.utc).isoformat(), 'cpu_limit': 2, 'source_repository': 'https://github.com/TieuLongPhan/SynEPD', 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(), 'source_release': manifest['dataset_release'], 'source_files_sha256': snapshots, 'source_database_counts': actual_counts, 'staged_rows': len(records), 'arrows': sum(map(len, arrows.values())), 'source_join': 'Exact case_id membership, mapped reaction strings and ordered arrow lists; JSON source IDs and SQLite primary keys are distinct.', 'strict_reference_replay': dict(statuses), 'reference_rows_passing_replay': eligible, 'primary_verification_levels': ['EXACT', 'NORMALIZED'], 'reference_rows_passing_primary_tier_screen': primary_tier, 'verification_levels': dict(levels), 'representation_modes': dict(modes), 'issue_codes': dict(issue_codes), 'distinct_overall_reaction_groups': len(reaction_groups), 'repeated_overall_reaction_groups': sum((count > 1 for count in reaction_groups.values())), 'families': dict(families), 'metric_contract': METRIC_CONTRACT, 'backend': {'version': synkit.__version__, 'source_root': str(backend_root), 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.synkit_dir, text=True).strip() if args.synkit_dir else None, 'python_source_sha256': backend_code, 'working_tree_changes': subprocess.check_output(['git', 'status', '--short'], cwd=args.synkit_dir, text=True).splitlines() if args.synkit_dir else None, 'mapped_hydrogen_equivalence_capability': 'passed'}, 'input_contract': 'Reaction-conditioned arrow generation: mapped reactants AND products are supplied.', 'primary_metric': 'Strict-replay-valid event recovery at rank k, against accepted reference alternatives.', 'secondary_metrics': ['Ordered trajectory recovery', 'Strict replay validity', 'Prediction coverage'], 'diagnostic': 'Net equivalence is endpoint consistency, not mechanism prediction success.', 'release_membership': 'Staged evidence only; not included in canonical candidate manifests.', 'benchmark_eligible_count': None, 'limits': ['Curated annotations are reference electron flow, not experimentally proven unique mechanisms.', 'Upstream arrow order is preserved; inferred steps do not assert physical temporal order.', '11 closed-shell-pair source policies must remain distinguishable from exact serialization.', 'Repeated overall reactions need grouping and explicit alternative-target review.', 'Taxonomy, names, RC/MC hashes and electron paths are excluded from model-input files.', 'Group-held-out memberships and prediction experiments have not been produced.']}
    (output / 'preparation-report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    files = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob('*') if path.is_file()}
    (output / 'artifact-manifest.json').write_text(json.dumps({'files': files}, indent=2, sort_keys=True) + '\n')
    pointer = {'artifact': output.relative_to(ROOT).as_posix(), 'report_sha256': digest(output / 'preparation-report.json'), 'manifest_sha256': digest(output / 'artifact-manifest.json')}
    (ROOT / 'ncs/audit/mechanistic-task.json').write_text(json.dumps(pointer, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['staged_rows', 'arrows', 'strict_reference_replay', 'verification_levels', 'distinct_overall_reaction_groups', 'repeated_overall_reaction_groups']}, indent=2))
__configure_runtime(globals(), 'script/prepare_synepd_task.py')
if __name__ == '__main__':
    main()
