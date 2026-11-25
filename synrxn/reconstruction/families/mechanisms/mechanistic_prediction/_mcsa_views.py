# Direct implementation; historical source identity is explicit below.
"""Register frozen M-CSA enzyme mechanisms as four overlapping task views.

Offline: ``python kdd/coverage/adapters/mcsa.py``. This adapter verifies the
existing SynRXN/SynDART v5 companion, preserves every reference, and does not
rerun native conversion, chemistry, training, predictions or split selection.
The preferred whole-route view has 278 observed-initial records, of which
192 satisfy the frozen strict replay tier. The other views are diagnostics.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file, source_path as __synrxn_source_path
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/mcsa.py')
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import pandas as pd
ROOT = Path(__source_file__).resolve().parents[3]
LOCK = Path(__source_file__).with_name('mcsa_sources.json')
OUT = ROOT / 'Data/coverage_expansion/mechanisms'
COMPANION = 'ncs/artifacts/mcsa-001'
SOURCE_URL = 'https://www.ebi.ac.uk/thornton-srv/m-csa/'
VIEWS = {'complete_reactions_observed': ('mcsa_observed_mechanisms', 278, 192, 'complete mechanism proposal with native initial participants', True), 'complete_reactions': ('mcsa_full_mechanisms', 479, 312, 'complete mechanism proposal', False), 'processed': ('mcsa_continuous_chains', 934, 650, 'continuous complete or partial native mechanism chain', False), 'steps': ('mcsa_elementary_steps', 2393, 1657, 'accepted native drawing transition', False)}
INPUT_FIELDS = ['reaction_smiles']
TARGET_FIELDS = ['electron_path_json', 'mechanism_record_json']
GROUP_FIELDS = ['enzyme_group', 'mechanism_group', 'native_transition_groups_json']

def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def pin_for(lock, path):
    return next((asset for asset in lock['assets'] if asset['path'] == path))

def verify_assets(lock, repo=ROOT):
    """Fail closed on changed companion, native bundle, evidence or converter."""
    for asset in lock['assets']:
        relative = Path(asset['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe M-CSA source-lock path')
        path = __synrxn_source_path(relative, repo=repo)
        if not path.is_file() or path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
            raise ValueError('Missing/changed frozen M-CSA dependency: ' + str(relative))

def qualified_row(row, view, source_sha):
    """Add grouping/eligibility without exposing reference metadata as inputs."""
    row = dict(row)
    source_ids = json.loads(row['source_ids_json'])
    native_index = json.loads(row['native_index_json'])
    record = json.loads(row['mechanism_record_json'])
    arrows = json.loads(row['electron_path_json'])
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError('Missing or repeated native transition identity')
    if view == 'steps':
        if source_ids != [native_index['source_id']]:
            raise ValueError('Native step identity differs from index')
    elif source_ids != native_index['source_ids']:
        raise ValueError('Native route identities differ from index')
    if not isinstance(record, dict) or not isinstance(arrows, list):
        raise ValueError('Invalid typed mechanism/electron-path reference')
    if row['reaction_smiles'].count('>>') != 1:
        raise ValueError('Invalid mapped overall reaction')
    if row['reference_replay_status'] != 'VALID' or row['verification_level'] not in {'EXACT', 'DELTA_CONSISTENT'}:
        raise ValueError('Unexpected frozen M-CSA replay outcome')
    context = row['initial_participant_context']
    if context not in {'native_initial_source_observation', 'mechanism_derived_component_preconditions'}:
        raise ValueError('Unknown initial participant evidence')
    if view == 'complete_reactions_observed' and context != 'native_initial_source_observation':
        raise ValueError('Reconstructed participants in observed-only view')
    for key in ['development_exposed', 'source_development_exposed']:
        if row[key] not in {'true', 'false'}:
            raise ValueError('Missing native development-exposure flag')
        row[key] = row[key] == 'true'
    row['entry_id'], row['mechanism_id'] = (int(row['entry_id']), int(row['mechanism_id']))
    mechanism = f"MCSA:{row['entry_id']}:{row['mechanism_id']}"
    row.update({'mechanism_group': mechanism, 'native_transition_groups_json': packed([mechanism + ':transition:' + str(identifier) for identifier in source_ids]), 'source_sha256': source_sha, 'source_locator': f"{COMPANION}/Data/mechanism/mcsa_{view}.csv.gz#{row['source_record_id']}", 'source_family': 'MCSA_native_v5_2026-10-15', 'native_split': 'unassigned', 'split_role': 'no held-out membership assigned; converter-development exposure retained', 'primary_replay_eligible': row['verification_level'] == 'EXACT', 'preferred_primary_reference': view == 'complete_reactions_observed' and row['verification_level'] == 'EXACT', 'view_role': 'preferred_observed_whole_route' if view == 'complete_reactions_observed' else 'overlapping_diagnostic_view', 'reference_kind': 'curated enzyme mechanism proposal; grouped native transitions'})
    return row

def build():
    lock = json.loads(LOCK.read_text())
    verify_assets(lock)
    report = json.loads((ROOT / COMPANION / 'preparation-report.json').read_text())
    if report['status'] != 'complete' or report['split_assignments'] != 'unassigned':
        raise ValueError('Unexpected companion completion/split contract')
    OUT.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(100000000)
    tables = []
    for view, (identifier, expected_rows, exact_rows, unit, preferred) in VIEWS.items():
        source_path = f'{COMPANION}/Data/mechanism/mcsa_{view}.csv.gz'
        asset = pin_for(lock, source_path)
        source_lock_path = f'{COMPANION}/Data/sources/mechanism/mcsa_{view}.json'
        upstream_lock = json.loads((ROOT / source_lock_path).read_text())
        if upstream_lock['sha256'] != asset['sha256']:
            raise ValueError('Companion CSV does not match its native source lock')
        with gzip.open(ROOT / source_path, 'rt', newline='') as handle:
            rows = [qualified_row(row, view, asset['sha256']) for row in csv.DictReader(handle)]
        frame = pd.DataFrame(rows)
        if len(frame) != expected_rows or frame['sample_id'].duplicated().any() or frame['source_record_id'].duplicated().any():
            raise ValueError('M-CSA view count/identity mismatch')
        if int(frame['primary_replay_eligible'].sum()) != exact_rows or report['views'][view]['primary_eligible'] != exact_rows:
            raise ValueError('Frozen replay-tier count changed')
        path = OUT / (identifier + '.parquet')
        frame.to_parquet(path, index=False, compression='zstd')
        context_counts = dict(Counter(frame['initial_participant_context']))
        qualification = f'Frozen M-CSA v5 source references: {expected_rows} stored records; {exact_rows} EXACT strict replay references, {expected_rows - exact_rows} DELTA_CONSISTENT references retained for diagnostics. ' + ('Preferred whole-route view uses observed initial participants; only the 192 EXACT records are preferred primary references. ' if preferred else 'Overlapping supporting/diagnostic view of the same native enzyme proposals; not independent data or extra tasks. ') + 'Reconstructed initial components, when flagged, use mechanism-derived preconditions and are not independently observed endpoint inputs. Source transitions are ordered; arrows within a native drawing are one atomic group, not physical chronology. Native splits remain unassigned and development-exposure flags are preserved; replay consistency does not establish benchmark blindness or physical truth. Unsupported fishhooks, radical and covalent-metal cases remain outside the accepted conversion. All source identities, certificates, context and arrows are reference-only metadata, withheld from model inputs.'
        task = {'inputs': INPUT_FIELDS, 'targets': TARGET_FIELDS, 'group_fields': GROUP_FIELDS, 'metrics': ['strict_valid_event_recovery_top_k', 'ordered_grouped_trajectory_recovery_top_k', 'strict_validity', 'prediction_coverage'], 'qualification': qualification, 'eligibility_field': 'preferred_primary_reference' if preferred else 'primary_replay_eligible', 'split_status': 'unassigned; no independent evaluation cohort claimed', 'withheld_reference_columns': [column for column in frame.columns if column not in INPUT_FIELDS]}
        tables.append({'id': identifier, 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path), 'bytes': path.stat().st_size, 'rows': expected_rows, 'counting_unit': unit, 'task_ids': ['mechanistic_prediction'], 'task_fields': {'mechanistic_prediction': task}, 'columns': list(frame.columns), 'source_id': 'MCSA', 'source_ids': ['MCSA'], 'source_url': SOURCE_URL, 'source_sha256': asset['sha256'], 'source_asset_path': source_path, 'source_assets': [asset, pin_for(lock, source_lock_path)], 'license': 'CC-BY-4.0', 'license_expression': 'CC-BY-4.0', 'release_eligible': True, 'redistribution_status': 'redistributable_with_attribution_and_preserved_notices', 'reference_evidence': 'Direct native M-CSA MRV electron-flow proposals; frozen SynDART v5 conversion and SynRXN typed replay tiers. No EasIFA labels or predictor-generated mechanisms.', 'qualification': qualification, 'view_role': 'preferred_observed_whole_route' if preferred else 'overlapping_diagnostic_view', 'role': 'primary' if preferred else 'overlapping_view', 'overlap_family': 'MCSA_native_v5_2026-10-15', 'independent_dataset': False, 'strict_replay_eligible_rows': exact_rows, 'preferred_primary_rows': 192 if preferred else 0, 'participant_context_counts': context_counts, 'native_split_status': 'unassigned', 'development_exposure_policy': 'True is recorded converter-development exposure; false does not establish blindness.', 'source_relationship': 'The observed subset is contained in complete proposals, which overlap continuous chains and elementary steps; active-site rows share native M-CSA entry provenance. Never sum as independent reactions.'})
    notice = OUT / 'MCSA.NOTICE.txt'
    notice.write_text('M-CSA enzyme mechanism reference views\nNative data: M-CSA, EMBL-EBI, Thornton Group. CC BY 4.0.\nhttps://www.ebi.ac.uk/thornton-srv/m-csa/\nhttps://creativecommons.org/licenses/by/4.0/\nRibeiro AJM et al. Mechanism and Catalytic Site Atlas (M-CSA): a database of enzyme reaction mechanisms and active sites. Nucleic Acids Research46(D1), D618-D623 (2018). https://doi.org/10.1093/nar/gkx1012\nChanges: frozen SynDART v5 conversion and SynRXN native-identity joins, grouped typed references, replay-tier qualification; coverage adapter adds grouping and explicit unassigned split/eligibility fields.\nConverter code retains Apache-2.0 terms and adapted SynKit MIT notices in the frozen source kit. Native data are CC BY4.0; code terms do not replace the data grant.\nFour views overlap. Preferred observed-start full mechanisms:278stored/192EXACT references. Other full mechanisms479, continuous chains934 and transitions2393 are supporting/diagnostic views. No independent held-out cohort or predictive result is established. No endorsement implied.\n')
    rights = pin_for(lock, 'Data/raw/coverage/bio/mcsa-rights.html')
    source = {'id': 'MCSA', 'name': 'M-CSA native enzyme mechanisms, frozen15October2026', 'url': SOURCE_URL, 'paper_url': 'https://doi.org/10.1093/nar/gkx1012', 'license': 'CC-BY-4.0', 'data_license_spdx': 'CC-BY-4.0', 'redistribution_allowed': True, 'rights': {'scope': 'native M-CSA records and MRV drawings, adapted with attributed frozen conversion', 'grant_url': SOURCE_URL, 'grant_snapshot': rights, 'notice': {'path': notice.relative_to(ROOT).as_posix(), 'sha256': sha256(notice)}}, 'assets': lock['assets'], 'source_snapshot': report['source_snapshot']}
    catalog = {'schema': 'synrxn.coverage-expansion.mechanisms.v1', 'sources': [source], 'tables': tables, 'source_lock': LOCK.relative_to(ROOT).as_posix(), 'source_lock_sha256': sha256(LOCK), 'builder': Path(__source_file__).relative_to(ROOT).as_posix(), 'builder_sha256': sha256(__source_file__), 'adapter': Path(__source_file__).relative_to(ROOT).as_posix(), 'adapter_sha256': sha256(__source_file__), 'task_count_policy': 'Four variants of existing mechanistic_prediction; no new scientific tasks.', 'preferred_primary_table': 'mcsa_observed_mechanisms', 'preferred_primary_rows': 192, 'overlap_policy': '4084 view records; not4084independent mechanisms. All views and native M-CSA active-site data must share entry grouping.', 'split_policy': 'Unassigned. Native enzyme entries, proposals and transitions remain linked. Converter-development exposures remain visible.', 'reproduction': lock['reproduction']}
    (OUT / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps({'tables': len(tables), 'stored_view_records': sum((t['rows'] for t in tables)), 'preferred_primary_rows': 192}, indent=2))
    return catalog
__configure_runtime(globals(), 'kdd/coverage/adapters/mcsa.py')
if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
