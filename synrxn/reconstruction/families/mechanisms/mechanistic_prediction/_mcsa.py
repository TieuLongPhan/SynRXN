# Direct implementation; historical source identity is explicit below.
"""Adapt a verified M-CSA v5 native rebuild to separate SynRXN mechanism views.

Native drawing groups remain atomic groups, and proposal/chain step boundaries
remain explicit. No predictor or benchmark is run. Invalid references remain in
the canonical extension and audit, outside the paired primary task view.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/prepare_mcsa_task.py')
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import html
import io
import gzip
from importlib.metadata import version
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import shutil
import sys
ROOT = Path(__source_file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / 'ncs/sources/mcsa-v5'
VIEWS = ('complete_reactions_observed', 'complete_reactions', 'processed', 'steps')

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

@contextmanager
def compressed(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding='utf8', newline='') as stream:
            yield stream

def typed_record(row, link):
    """Represent each native drawing transition as one simultaneous move group."""
    from synkit.Mechanism import MechanismRecord, MechanisticStep
    from synkit.Mechanism.adapters import group_from_legacy_epd
    arrows = row['epd_lw']
    offsets = link.get('arrow_offsets', [0, len(arrows)])
    if not offsets or offsets[0] != 0 or offsets[-1] != len(arrows) or any((type(value) is not int for value in offsets)) or any((a > b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError('Invalid native step boundaries')
    steps = tuple((MechanisticStep(step_id=f's{number}', groups=(group_from_legacy_epd(arrows[start:end], group_id=f'g{number}'),) if end > start else ()) for number, (start, end) in enumerate(zip(offsets, offsets[1:]), 1)))
    return MechanismRecord(mapped_reaction=row['rsmi'], steps=steps, provenance={'dataset': 'M-CSA', 'conversion': 'SynDART v5_2026-10-15', 'ordering': 'Native transitions ordered; arrows within each drawing grouped together'})

def verify_build(build, lock):
    manifest = json.loads((build / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest['recipe_sha256'] != lock['recipe_sha256']:
        raise ValueError('Native build is incomplete or uses a different recipe')
    for asset in manifest['outputs']:
        name = Path(asset['path'])
        if name.is_absolute() or '..' in name.parts or (not (build / name).resolve().is_relative_to(build)):
            raise ValueError('Unsafe native build asset')
        if digest(build / name) != asset['sha256']:
            raise ValueError('Changed native build asset: ' + str(name))
    for view, expected in lock['portable_outputs'].items():
        for name, sha in expected['files'].items():
            if digest(build / view / name) != sha:
                raise ValueError('Native portable release differs: ' + view + '/' + name)
    for name in ('validation', 'reproducibility'):
        if json.loads((build / f'audit/{name}.json').read_text())['status'] != 'pass':
            raise ValueError('Native build validation failed')
    return manifest

def checked_sources(row, link, records, metadata):
    ids = link.get('source_ids', [link.get('source_id')])
    selected = [records[identifier] for identifier in ids]
    contexts = [metadata[identifier] for identifier in ids]
    first = contexts[0]
    if any(((item['entry_id'], item['mechanism_id']) != (first['entry_id'], first['mechanism_id']) for item in contexts)):
        raise ValueError('A portable row joins different native proposals')
    if link.get('entry_id', first['entry_id']) != first['entry_id']:
        raise ValueError('Native entry identity mismatch')
    if link.get('mechanism_id', first['mechanism_id']) != first['mechanism_id']:
        raise ValueError('Native proposal identity mismatch')
    expected_reaction = selected[0]['rsmi'].split('>>')[0] + '>>' + selected[-1]['rsmi'].split('>>')[1]
    if row['rsmi'] != expected_reaction or row['epd_lw'] != [arrow for item in selected for arrow in item['epd']]:
        raise ValueError('Portable endpoints or arrows differ from native step identities')
    for a, b in zip(selected, selected[1:]):
        if a['rsmi'].split('>>')[1] != b['rsmi'].split('>>')[0]:
            raise ValueError('A continuous chain has a missing intermediate')
    return (ids, contexts)

def prepare(build, output):
    from synkit.Mechanism import MechanismRecord, MechanismReplayer
    import synkit
    from synrxn.metrics.mechanism import METRIC_CONTRACT, evaluate_mechanism_predictions
    evaluate_mechanism_predictions({}, {}, top_k=(1,))
    lock = json.loads((BUNDLE / 'bundle-lock.json').read_text())
    native_manifest = verify_build(build, lock)
    if output.exists():
        raise ValueError('Choose a fresh extension output directory')
    output.mkdir(parents=True)
    clean = build / 'clean'
    payload = json.loads((clean / 'mcsa.json').read_text())
    records = {row['id']: row for row in payload['records']}
    metadata = {row['id']: row['mcsa'] for row in jsonl(clean / 'metadata.jsonl')}
    if len(records) != payload['count'] or records.keys() != metadata.keys():
        raise ValueError('Native accepted/metadata identities differ')
    mechanisms = {(row['entry_id'], row['mechanism_id']): row for row in jsonl(clean / 'mechanisms.jsonl')}
    exposure = json.loads((build / 'development_exposure.json').read_text())
    exposed = set(exposure['conservative_known_entry_ids'])
    certificates = {}
    for row in jsonl(clean / 'steps.jsonl'):
        if row['conversion']['status'] == 'pass':
            certificates[row['id']] = row['conversion']['certificate']
    if records.keys() != certificates.keys():
        raise ValueError('Missing native conversion certificate')
    evidence = output / 'evidence'
    evidence.mkdir()
    code = output / 'code'
    code.mkdir()
    for name in ('prepare_mcsa_task.py', 'rebuild_mcsa.py'):
        shutil.copyfile(ROOT / 'script' / name, code / name)
    for name in ('development_exposure.json', 'recipe.json', 'audit/validation.json', 'audit/reproducibility.json', 'audit/scope.json', 'clean/sequence_corrections.jsonl', 'clean/rejected.jsonl'):
        dest = evidence / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / name, dest)
    for view in VIEWS:
        for name in ('mcsa.json', 'index.jsonl'):
            dest = evidence / 'portable' / view / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(build / view / name, dest)
    shutil.copyfile(BUNDLE / 'bundle-lock.json', evidence / 'bundle-lock.json')
    data = output / 'Data'
    data.mkdir()
    (data / 'LICENSE').write_text('M-CSA data: Creative Commons Attribution 4.0 International (CC-BY-4.0).\nhttps://creativecommons.org/licenses/by/4.0/\nSource: M-CSA, EMBL-EBI, Thornton Group. https://www.ebi.ac.uk/thornton-srv/m-csa/\nRibeiro et al. Nucleic Acids Research 46(D1), D618–D623 (2018). doi:10.1093/nar/gkx1012\nChanges: SynDART v5 drawing conversion; SynRXN CSV, native joins, typed grouped records and replay audits.\nConverter code is separately Apache-2.0; its license and third-party notices accompany the source bundle.\n')
    catalog = {'schema_version': '1.0', 'column_definitions': {}, 'datasets': {}}
    fields = ['sample_id', 'source_record_id', 'source_ids_json', 'reaction_smiles', 'electron_path_json', 'mechanism_record_json', 'entry_id', 'mechanism_id', 'enzyme_group', 'native_index_json', 'source_context_json', 'initial_participant_context', 'development_exposed', 'source_development_exposed', 'reference_replay_status', 'verification_level', 'source_version', 'license_spdx']
    for field in fields:
        role = 'reaction' if field == 'reaction_smiles' else 'target' if field in {'electron_path_json', 'mechanism_record_json'} else 'identifier' if field == 'sample_id' else 'reference_metadata'
        catalog['column_definitions'][field] = {'role': role, 'description': {'source_context_json': 'Joined native step metadata, certificates and proposal; reference-only context, withheld from inputs.', 'development_exposed': 'Entry appears in the full converter-development exposure ledger; false does not establish blindness.', 'source_development_exposed': 'Original historical pilot flag, preserved separately from the expanded exposure ledger.', 'native_index_json': 'Unchanged native identity index including step offsets, transient pairs and participant context where recorded.', 'mechanism_record_json': 'Typed reference with atomic native drawing groups and ordered source transitions.'}.get(field, field.replace('_', ' ') + '; source/reference data, not a permitted model input except reaction_smiles.')}
    primary_levels = {'EXACT', 'NORMALIZED'}
    reports, native_memberships = ({}, {})
    replayer = MechanismReplayer(validation='strict', repair=False, verify_stereo='stepwise')
    for view in VIEWS:
        rows = json.loads((build / view / 'mcsa.json').read_text())
        links = list(jsonl(build / view / 'index.jsonl'))
        if len(rows) != len(links) or len(rows) != lock['portable_outputs'][view]['count']:
            raise ValueError('Portable/native index lengths differ')
        key = f'mechanism/mcsa_{view}.csv.gz'
        view_dir = output / 'task-views' / view
        tiers, statuses, partitions, groups, context_counts = (Counter() for _ in range(5))
        primary, arrows_total, zero_arrows = (0, 0, 0)
        with ExitStack() as stack:
            table = stack.enter_context(compressed(data / key))
            writer = csv.DictWriter(table, fieldnames=fields)
            writer.writeheader()
            streams = {name: stack.enter_context(compressed(view_dir / (name + '.jsonl.gz'))) for name in ('inputs', 'targets', 'membership', 'exclusions', 'reference-audit')}
            for position, (row, link) in enumerate(zip(rows, links)):
                if row['index'] != position or link['index'] != position or row['number_arrow'] != len(row['epd_lw']):
                    raise ValueError('Portable indices or arrow count differ')
                ids, contexts = checked_sources(row, link, records, metadata)
                first = contexts[0]
                entry_id, proposal_id = (first['entry_id'], first['mechanism_id'])
                group = f'MCSA.{entry_id:06d}'
                source_id = f'{view}:' + ','.join(map(str, ids))
                sample_id = 'mcsa-' + hashlib.sha256((lock['recipe_sha256'] + ':' + source_id).encode()).hexdigest()[:24]
                record = typed_record(row, link)
                serialized = record.to_dict()
                if MechanismRecord.from_dict(serialized).to_dict() != serialized:
                    raise ValueError('Mechanism serialization changed on round trip')
                certificate = replayer.replay(record).certificate.to_dict()
                status, level = (certificate['status'], certificate['verification_level'])
                eligible = status == 'VALID' and level in primary_levels
                context = certificates[ids[0]].get('initial_participant_context', 'unknown')
                full_context = {'source_steps': contexts, 'source_certificates': [certificates[i] for i in ids], 'proposal': mechanisms[entry_id, proposal_id]}
                if view == 'complete_reactions_observed' and context != 'native_initial_source_observation':
                    raise ValueError('Observed-start view includes reconstructed participants')
                writer.writerow(dict(zip(fields, [sample_id, source_id, packed(ids), row['rsmi'], packed(row['epd_lw']), packed(serialized), entry_id, proposal_id, group, packed(link), packed(full_context), context, str(entry_id in exposed).lower(), str(bool(first.get('development_exposed'))).lower(), status, level, 'v5_2026-10-15', 'CC-BY-4.0'])))
                instance_id = 'mechanistic_prediction:' + sample_id
                native_memberships[sample_id] = ids
                audit = {'sample_id': sample_id, 'source_ids': ids, 'certificate': certificate, 'primary_eligible': eligible, 'source_view': view}
                streams['reference-audit'].write(packed(audit) + '\n')
                membership = {'instance_id': instance_id, 'group': group, 'proposed_split': None, 'source_split': None, 'development_exposed': entry_id in exposed, 'source_ids': ids, 'source_view': view}
                if eligible:
                    streams['inputs'].write(packed({'instance_id': instance_id, 'inputs': {'mapped_overall_reaction': row['rsmi']}}) + '\n')
                    streams['targets'].write(packed({'instance_id': instance_id, 'targets': {'electron_path_json': row['epd_lw'], 'mechanism_record_json': serialized}, 'target_mask': {'electron_path_json': True, 'mechanism_record_json': True}, 'group': group, 'source_record_id': source_id, 'evidence_tier': level}) + '\n')
                    streams['membership'].write(packed(membership) + '\n')
                    primary += 1
                else:
                    streams['exclusions'].write(packed({**membership, 'reason': 'outside_primary_replay_tier', 'reference_replay_status': status, 'verification_level': level}) + '\n')
                statuses[status] += 1
                tiers[level] += 1
                partitions['development_exposed' if entry_id in exposed else 'exposure_not_established'] += 1
                groups[group] += 1
                context_counts[context] += 1
                arrows_total += row['number_arrow']
                zero_arrows += row['number_arrow'] == 0
                if (position + 1) % 100 == 0:
                    print(f'M-CSA {view}: checked {position + 1}/{len(rows)} references', flush=True)
        reports[view] = {'dataset': key, 'records': len(rows), 'arrows': arrows_total, 'zero_arrow_records': zero_arrows, 'primary_eligible': primary, 'primary_excluded': len(rows) - primary, 'reference_replay_status': dict(statuses), 'verification_levels': dict(tiers), 'enzyme_entry_groups': len(groups), 'participant_context': dict(context_counts), 'development_exposure': dict(partitions), 'csv_sha256': digest(data / key)}
        catalog['datasets'][key] = {'title': 'M-CSA ' + view.replace('_', ' '), 'description': 'Overlapping M-CSA enzyme-mechanism view; native groups and context retained.', 'benchmark_role': 'Reference annotations for reaction-conditioned electron-flow recovery; no model results.', 'license': 'CC BY 4.0', 'license_spdx': 'CC-BY-4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/', 'redistribution_partition': 'main', 'citations': ['ribeiro2018mcsa'], 'targets': ['electron_path_json', 'mechanism_record_json'], 'row_identifier': 'sample_id', 'row_identifier_unique': True, 'split_column': 'split', 'split_values': [], 'split_status': 'none'}
        write_json(data / 'sources/mechanism' / ('mcsa_' + view + '.json'), {'schema': 'synrxn.mcsa-source-lock.v1', 'dataset': key, 'sha256': digest(data / key), 'license_spdx': 'CC-BY-4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/', 'source_url': 'https://www.ebi.ac.uk/thornton-srv/m-csa/', 'doi': '10.1093/nar/gkx1012', 'source_snapshot_manifest_sha256': native_manifest['source']['manifest_sha256'], 'converter_recipe_sha256': lock['recipe_sha256'], 'native_portable': lock['portable_outputs'][view], 'changes': 'SynDART v5 conversion, native-identity joins, grouped typed records and strict SynRXN replay audit.', 'source_scope': 'Frozen 2026-10-15 snapshot; mutable current M-CSA API is not a replacement for these bytes.'})
        write_json(view_dir / 'summary.json', reports[view])
    write_json(data / 'metadata.yaml', catalog)
    from synrxn.data.build_manifest import build_manifest
    from synrxn.data.validate import validate_catalog
    csv.field_size_limit(16 * 1024 * 1024)
    manifest_path = output / 'manifest.json'
    write_json(manifest_path, build_manifest(data, {'title': 'SynRXN M-CSA mechanism extension', 'version': 'mcsa-v5_2026-10-15'}, include_code_manifest=False, repo_root_hint=ROOT, candidate=True))
    catalog_check = validate_catalog(data, data / 'metadata.yaml', manifest_path)
    write_json(output / 'catalog-validation.json', catalog_check)
    if catalog_check['errors']:
        raise ValueError('Extension catalog validation failed: ' + str(catalog_check['errors']))
    protocol = {'schema': 'synrxn.mcsa-task-extension.v1', 'task': 'mechanistic_prediction', 'metric_contract': METRIC_CONTRACT, 'primary_verification_levels': sorted(primary_levels), 'inputs': ['mapped_overall_reaction'], 'primary_eligibility_scope': 'Representation/replay consistency only; benchmark eligibility and participant identifiability are not established.', 'targets': ['electron_path_json', 'mechanism_record_json'], 'ordering': 'Source transitions ordered; each drawing is one atomic group, not an inferred arrow chronology.', 'grouping': 'Enzyme entry across proposals, steps, chains and all overlapping views.', 'splits': 'Unassigned. Preserve the full converter-development exposure ledger; nonmembership does not establish blindness.', 'withheld': ['arrows', 'step boundaries', 'transient pairs', 'intermediate structures', 'native certificates', 'enzyme name', 'EC taxonomy', 'mechanism prose', 'source identifiers', 'exposure flags'], 'context': 'Endpoint structures include source enzyme/cofactor fragments and sometimes reconstructed participants.', 'prediction_metrics': ['strict-valid event recovery top-k', 'ordered/grouped trajectory recovery top-k', 'strict validity', 'prediction coverage'], 'diagnostic': 'Net equivalence checks supplied endpoints, not mechanism prediction success.', 'limitations': ['Four views overlap and must not be summed as independent reactions.', 'Complete mechanisms with observed starts are the conservative whole-route view.', 'Native scope includes unsupported fishhooks, radicals and metal involvement; rejected native cases remain in evidence.', 'Spectator coordination projection, capped residues and derived participant context are representation choices.', 'Stereo is not independently established by the native converter; replay does not prove physical mechanism truth.', 'No predictive experiment, held-out split or independence claim is produced.']}
    write_json(output / 'task-contract.json', protocol)
    write_json(output / 'native-memberships.json', native_memberships)
    report = {'schema': 'synrxn.mcsa-task-preparation.v1', 'status': 'complete', 'cpu_limit': 2, 'benchmark_execution': False, 'release_membership': 'Separate extension; frozen 49-table candidate unchanged', 'native_manifest_sha256': digest(build / 'manifest.json'), 'source_snapshot': native_manifest['source'], 'bundle_lock_sha256': digest(BUNDLE / 'bundle-lock.json'), 'adapter_sha256': digest(Path(__source_file__)), 'backend_version': synkit.__version__, 'environment': {'python': sys.version.split()[0], 'packages': {name: version(name) for name in ('synkit', 'rdkit', 'numpy', 'pandas', 'networkx', 'PyYAML')}}, 'backend_mechanism_files_sha256': {p.name: digest(p) for p in sorted((Path(synkit.__file__).parent / 'Mechanism').glob('*.py'))}, 'views': reports, 'split_assignments': 'unassigned', 'metric_contract': METRIC_CONTRACT, 'benchmark_eligible_count': None, 'known_converter_development_entries': len(exposed), 'limitations': protocol['limitations']}
    write_json(output / 'preparation-report.json', report)
    rows_html = ''.join(('<tr><td>' + html.escape(view.replace('_', ' ')) + '</td>' + ''.join((f'<td>{reports[view][key]:,}</td>' for key in ('records', 'arrows', 'primary_eligible', 'primary_excluded'))) + '</tr>' for view in VIEWS))
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>M-CSA mechanism extension</title><style>body{font:17px system-ui;max-width:1100px;margin:3em auto;padding:0 1em;line-height:1.6}table{border-collapse:collapse;width:100%}th,td{padding:.7em;border-bottom:1px solid #aaa;text-align:left}aside{background:#eef3fa;padding:1em}</style><h1>M-CSA mechanism extension</h1><aside>Built from frozen native drawings. Two CPUs maximum. No benchmarks run. Separate from the 49-table paper candidate.</aside><div style="overflow-x:auto"><table><thead><tr><th>Overlapping view</th><th>Records</th><th>Arrows</th><th>Primary replay tier</th><th>Diagnostic</th></tr></thead><tbody>' + rows_html + '</tbody></table></div><p>Views overlap. Primary eligibility requires strict VALID replay at EXACT or NORMALIZED level. Complete routes with observed starting participants are the conservative whole-route view. Reconstructed context remains marked.</p><p>Train/test assignments remain unassigned. Enzyme entry IDs group all steps, proposals and views; the complete converter-development exposure ledger is retained. Primary counts describe representation consistency, not final benchmark eligibility. The native audit records 2,024 ambiguous correspondences and 543 complete successor observations; all certificates remain available.</p><ul>' + ''.join(('<li>' + html.escape(item) + '</li>' for item in protocol['limitations'])) + '</ul><p><a href="preparation-report.json">Audit report</a> · <a href="task-contract.json">Task and metrics contract</a> · <a href="Data/metadata.yaml">Dataset catalog</a> · <a href="evidence/development_exposure.json">Exposure ledger</a></p><p>Source: <a href="https://www.ebi.ac.uk/thornton-srv/m-csa/">M-CSA, EMBL-EBI</a>, CC BY 4.0. <a href="https://doi.org/10.1093/nar/gkx1012">Ribeiro et al. (2018)</a>.</p></html>')
    write_json(output / 'artifact-manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'views': reports, 'benchmark_execution': False}, indent=2), flush=True)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-backend', action='store_true')
    args = parser.parse_args()
    if args.check_backend:
        from synrxn.metrics.mechanism import evaluate_mechanism_predictions
        evaluate_mechanism_predictions({}, {}, top_k=(1,))
        print('SynRXN mechanism backend capability: pass')
        return
    if not args.build_dir or not args.output_dir:
        parser.error('--build-dir and --output-dir are required')
    prepare(args.build_dir.resolve(), args.output_dir.resolve())
__configure_runtime(globals(), 'script/prepare_mcsa_task.py')
if __name__ == '__main__':
    main()
