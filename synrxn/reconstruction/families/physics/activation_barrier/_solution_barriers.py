# Direct implementation; historical source identity is explicit below.
"""Acquire and reproduce native multi-solvent activation Gibbs free energies.

python kdd/coverage/adapters/solution_barriers.py --fetch
python kdd/coverage/adapters/solution_barriers.py  # offline from pinned workbook

Only finite cached cells of the seven methods in the LiPRED paper are exported.
No formula recalculation, inferred SMILES, gas-barrier conversion, or experimental
label duplication is performed. The published 4513 count differs from the 4514
finite cells in the pinned native workbook; this discrepancy is retained.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/solution_barriers.py')
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import shutil
import zipfile
import openpyxl
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/solution_barriers/LiPRED2026'
OUT = ROOT / 'Data/coverage_expansion/solution_barriers'
LOCK_PATH = ROOT / 'kdd/coverage/solution_barriers_sources.json'
METHODS = {8: ('G_TC', 'TC'), 9: ('G_TC_VibCorr', 'TC-vib'), 10: ('G_TC_AVGof6', 'TC-SMD'), 12: ('G_TC_high', 'TC-high'), 13: ('G_Direct', 'Direct'), 14: ('quasi_G_TC', 'TC-quasi'), 15: ('quasi_G_dir', 'Direct-quasi')}

def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        while (chunk := handle.read(4 * 1024 * 1024)):
            result.update(chunk)
    return result.hexdigest()

def rel(path):
    return Path(path).relative_to(ROOT).as_posix()

def check(path, expected):
    if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        raise ValueError('Native source differs from pinned release: ' + str(path))

def fetch():
    import requests
    lock = json.loads(LOCK_PATH.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW / 'metadata.json'
    if not metadata_path.exists():
        response = requests.get(lock['record_api'], timeout=60)
        response.raise_for_status()
        metadata = response.json()
        if metadata['id'] != lock['record_id'] or metadata['metadata']['license']['id'] != 'cc-by-4.0':
            raise ValueError('Native dataset identity or CC BY4 grant changed')
        serialized = (json.dumps(metadata, indent=2) + '\n').encode()
        if hashlib.sha256(serialized).hexdigest() != lock['metadata_snapshot_sha256']:
            raise ValueError('Zenodo metadata changed; review and repin before acquisition')
        metadata_path.write_bytes(serialized)
    if digest(metadata_path) != lock['metadata_snapshot_sha256']:
        raise ValueError('Pinned metadata snapshot differs')
    metadata = json.loads(metadata_path.read_text())
    for item in (lock['archive'], lock['readme']):
        path = RAW / item['name']
        if not path.exists():
            response = requests.get(item['url'], timeout=180)
            response.raise_for_status()
            path.write_bytes(response.content)
        check(path, item)
        native = next((asset for asset in metadata['files'] if asset['key'] == item['name']))
        if native['size'] != item['bytes'] or ('md5' in item and native['checksum'] != 'md5:' + item['md5']):
            raise ValueError('Native record asset identity differs')
    with zipfile.ZipFile(RAW / lock['archive']['name']) as archive:
        (RAW / lock['workbook']['name']).write_bytes(archive.read(lock['archive']['member']))
    check(RAW / lock['workbook']['name'], lock['workbook'])

def group_key(reaction_class, substrate_1, substrate_2):
    substrates = sorted((value for value in (substrate_1, substrate_2) if value))
    return hashlib.sha256(json.dumps([reaction_class, substrates], ensure_ascii=False).encode()).hexdigest()

def build():
    lock = json.loads(LOCK_PATH.read_text())
    native_path, metadata_path = (RAW / lock['workbook']['name'], RAW / 'metadata.json')
    check(native_path, lock['workbook'])
    check(RAW / lock['readme']['name'], lock['readme'])
    if digest(metadata_path) != lock['metadata_snapshot_sha256']:
        raise ValueError('Metadata snapshot pin differs')
    metadata = json.loads(metadata_path.read_text())
    if metadata['id'] != lock['record_id'] or metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Not the verified native dataset license')
    workbook = openpyxl.load_workbook(native_path, data_only=True, read_only=True)
    native_data = list(workbook['data'].values)
    native_predictions = list(workbook['pred vs exp'].values)
    if len(native_predictions) - 2 != lock['native_workbook_context_rows']:
        raise ValueError('Unexpected native context row count')
    for column, (native_name, _) in METHODS.items():
        if native_predictions[1][column] != native_name or native_data[1][column + 42] != native_name:
            raise ValueError('Native column contract changed')
    contexts = defaultdict(list)
    for row in native_predictions[2:]:
        key = (group_key(row[1], row[2], row[3]), row[4], row[5], row[6], row[7])
        contexts[key].append(int(row[0]))
    duplicate_entries = {entry for entries in contexts.values() if len(entries) > 1 for entry in entries}
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ['source_record_id', 'source_entry', 'source_sheet', 'source_excel_row', 'source_excel_column', 'reaction_class', 'substrate_1_name', 'substrate_2_name', 'reaction_group', 'solvent_name', 'hypothetical_solvent', 'solvent_identity_qualified', 'duplicate_context', 'temperature_K', 'low_level_native', 'high_level_native', 'method', 'native_method_column', 'phase', 'solvation_model', 'reference_kind', 'target_quantity', 'standard_state', 'reference_state_correction_kcal_per_mol', 'activation_gibbs_kcal_per_mol']
    output_path = OUT / 'lipred_solution_activation_gibbs.csv'
    counts, groups, solvents, hypotheticals, duplicates, types = (Counter(), set(), set(), 0, 0, Counter())
    records, companion_reactant_order_swaps = (0, [])
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for excel_row, (row, raw_row) in enumerate(zip(native_predictions[2:], native_data[2:], strict=True), 3):
            if row[:2] != raw_row[:2] or row[4:8] != raw_row[4:8] or sorted((x for x in row[2:4] if x)) != sorted((x for x in raw_row[2:4] if x)):
                raise ValueError('Companion native sheets have different context identities')
            if row[2:4] != raw_row[2:4]:
                companion_reactant_order_swaps.append(int(row[0]))
            group = group_key(row[1], row[2], row[3])
            hypothetical = str(row[4]).startswith('hypo')
            for column, (native_name, method) in METHODS.items():
                value = row[column]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                if value != raw_row[column + 42]:
                    raise ValueError('Companion cached absolute barrier values differ')
                record = {'source_record_id': f'LiPRED2026:entry:{int(row[0])}:{native_name}', 'source_entry': int(row[0]), 'source_sheet': 'pred vs exp', 'source_excel_row': excel_row, 'source_excel_column': openpyxl.utils.get_column_letter(column + 1), 'reaction_class': row[1], 'substrate_1_name': row[2], 'substrate_2_name': row[3] or '', 'reaction_group': group, 'solvent_name': row[4], 'hypothetical_solvent': hypothetical, 'solvent_identity_qualified': not hypothetical, 'duplicate_context': int(row[0]) in duplicate_entries, 'temperature_K': row[5], 'low_level_native': row[6], 'high_level_native': row[7], 'method': method, 'native_method_column': native_name, 'phase': 'solution', 'solvation_model': 'SMD', 'reference_kind': 'computed_native_cached_solution_activation_gibbs', 'target_quantity': 'Delta_G_dagger', 'standard_state': 'solution 1 mol/L; gas reference 1 atm (paper eq 1)', 'reference_state_correction_kcal_per_mol': raw_row[40], 'activation_gibbs_kcal_per_mol': value}
                writer.writerow(record)
                records += 1
                counts[method] += 1
                groups.add(group)
                solvents.add(row[4])
                hypotheticals += hypothetical
                duplicates += record['duplicate_context']
                types[row[1]] += 1
    if records != lock['seven_method_native_finite_values']:
        raise ValueError('Incomplete finite native method-cell extraction')
    qualification = 'Native computed solution-phase activation Gibbs free energies, not Arrhenius Ea or electronic barriers. 4514 method-specific references come from 970 workbook context rows, 30 unordered native-name/channel groups and 542 native solvent tokens. These are not 4514 independent reactions. Paper states 4513 values/28 reactions; the pinned broader thesis workbook includes two Fmoc groups and differs in count. No exact paper-subset claim. Seven published methods are selected; two additional native AVGof6 variants are omitted. No reaction or solvent SMILES are supplied in this workbook; retain native names and codes without inference. Hypothetical hypo1-9 tokens lack descriptors here and require exclusion from physical-solvent evaluation. Native entries 408 and 531 repeat the same named Menschutkin context with different cached values and are flagged. Group all methods/solvents by reaction_group; no train/test split has been qualified. Source cached formulas are read without recalculation; original standard-state corrections are already in targets.'
    qualification_path = OUT / 'qualification.json'
    qualification_path.write_text(json.dumps({'schema': 'synrxn.solution-barrier-qualification.v1', 'native_context_rows': 970, 'unique_named_contexts': len(contexts), 'reaction_groups': len(groups), 'native_solvent_tokens': len(solvents), 'computed_method_reference_rows': records, 'method_counts': dict(counts), 'reaction_class_counts': dict(types), 'hypothetical_solvent_reference_rows': hypotheticals, 'physical_named_solvent_reference_rows': records - hypotheticals, 'duplicate_context_entries': sorted(duplicate_entries), 'duplicate_context_reference_rows': duplicates, 'companion_sheet_reactant_order_swaps': companion_reactant_order_swaps, 'paper_reported_computed_values': 4513, 'paper_reported_reactions': 28, 'count_discrepancy_resolved': False, 'qualification': qualification, 'target_verification': 'Every target exactly matches cached cells in both native workbook sheets; no formula recomputation.'}, indent=2) + '\n')
    notice_path = OUT / 'LiPRED2026.NOTICE.txt'
    notice_path.write_text("LiPRED-associated chapter5 QM dataset from Lingfeng Gui's thesis supplementary material v3.\nAuthor: Lingfeng Gui. Dataset DOI: https://doi.org/10.5281/zenodo.16358189\nPaper: Gui L, Armstrong A, Adjiman CS, Sayyed FB, Galindo A. Generation and benchmarking of a diverse reaction database of quantum mechanical liquid-phase activation Gibbs free energies. PCCP 2026,28,13007-13020. https://doi.org/10.1039/D6CP00088F\nDataset license: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/\nChanges: extracted seven named method columns into long CSV, added native cell identifiers, reaction grouping, hypothetical-solvent and duplicate-context flags. Native computed targets retained. See qualification.json for the difference between paper counts and this native workbook export. No endorsement implied.\n")
    license_path = OUT / 'CC-BY-4.0.txt'
    shutil.copyfile(ROOT / 'Data/raw/coverage/physics/CC-BY-4.0.txt', license_path)
    source_id = 'LiPRED2026'
    source = {'id': source_id, 'name': 'LiPRED-associated native multi-solvent activation Gibbs free energies', 'url': lock['data_doi'], 'paper_url': lock['paper_url'], 'license': 'CC-BY-4.0', 'data_license_spdx': 'CC-BY-4.0', 'redistribution_allowed': True, 'release_status': 'redistributable_with_attribution_and_source_notices', 'source_sha256': digest(native_path), 'files': [{'path': rel(path), 'sha256': digest(path), 'bytes': path.stat().st_size} for path in [native_path, RAW / 'README.md', metadata_path]], 'rights': {'scope': 'actual native chapter5 workbook in the CC BY4 Zenodo dataset deposit', 'grant_url': lock['record_api'], 'grant_snapshot': {'path': rel(metadata_path), 'sha256': digest(metadata_path)}, 'license_terms_url': 'https://creativecommons.org/licenses/by/4.0/', 'license_snapshot': {'path': rel(license_path), 'sha256': digest(license_path)}, 'notice': {'path': rel(notice_path), 'sha256': digest(notice_path)}}, 'upstream_assets': [lock['archive'], lock['readme']]}
    table = {'id': 'lipred_solution_activation_gibbs', 'path': rel(output_path), 'rows': records, 'role': 'primary', 'task_ids': ['activation_barrier'], 'counting_unit': 'computed method-specific solution activation Gibbs reference', 'source_id': source_id, 'source_url': source['url'], 'source_sha256': source['source_sha256'], 'license': source['license'], 'sha256': digest(output_path), 'columns': columns, 'release_status': source['release_status'], 'reference_evidence': 'Exact native cached absolute Delta G dagger cells in both workbook sheets.', 'parent_groups': len(groups), 'native_context_rows': 970, 'native_solvent_tokens': len(solvents), 'target_units': {'activation_gibbs_kcal_per_mol': 'kcal/mol'}, 'qualification': qualification, 'qualification_report': {'path': rel(qualification_path), 'sha256': digest(qualification_path)}, 'task_fields': {'activation_barrier': {'inputs': ['reaction_class', 'substrate_1_name', 'substrate_2_name', 'solvent_name', 'temperature_K', 'low_level_native', 'high_level_native', 'method', 'solvation_model'], 'targets': ['activation_gibbs_kcal_per_mol'], 'group_fields': ['reaction_group', 'source_entry'], 'metrics': ['MAE_kcal_per_mol_by_method', 'RMSE_kcal_per_mol_by_method', 'coverage'], 'qualification': 'Names-only native identities; no invented SMILES. Exclude hypothetical_solvent=True for physical-solvent evaluation. Group all method/solvent variants of each named reaction/channel. Report unresolved native duplicate contexts separately.'}}}
    catalog = {'schema': 'synrxn.coverage-solution-barriers.v1', 'sources': [source], 'tables': [table], 'adapter': rel(__source_file__), 'adapter_sha256': digest(__source_file__), 'source_lock': rel(LOCK_PATH), 'source_lock_sha256': digest(LOCK_PATH), 'reproduction': {'offline_command': 'python kdd/coverage/adapters/solution_barriers.py', 'acquisition_command': 'python kdd/coverage/adapters/solution_barriers.py --fetch'}}
    (OUT / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps({'rows': records, 'reaction_groups': len(groups), 'native_solvent_tokens': len(solvents), 'hypothetical_solvent_rows': hypotheticals, 'method_counts': dict(counts)}, indent=2))
    return catalog
__configure_runtime(globals(), 'kdd/coverage/adapters/solution_barriers.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch()
    build()
