# Direct implementation; historical source identity is explicit below.
"""Prepare the full CC-BY4 QuantumPioneer166158-reaction kinetics release.

Run with --fetch for checksum-pinned acquisition, or offline with no arguments.
Original NC legacy lograte files are never read, changed or relicensed here.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/quantumpioneer.py')
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from collections import Counter
from synrxn.reconstruction.paths import HistoricalPath as Path
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/physics/QuantumPioneer2026'
OUT = ROOT / 'Data/coverage_expansion/physics_rates'
ASSETS = {'quantumpioneer_kinetics_dataset.csv': {'bytes': 39105280, 'md5': 'a9f4df281b82ab8ee9980bad0b8012e2'}, 'SCHEMA.md': {'bytes': 12318, 'md5': 'd0e7f364ac456c5fadf062636d6bbbcb'}}

def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        while (chunk := handle.read(4 * 1024 * 1024)):
            result.update(chunk)
    return result.hexdigest()

def rel(path):
    return Path(path).relative_to(ROOT).as_posix()

def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    response = requests.get('https://zenodo.org/api/records/20060606', timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('actual QuantumPioneer dataset grant differs from pinnedCCBY4')
    (RAW / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from synrxn.reconstruction.expansion import fetch as module
    for name, expected in ASSETS.items():
        native = next((item for item in metadata['files'] if item['key'] == name))
        if native['size'] != expected['bytes'] or native['checksum'] != 'md5:' + expected['md5']:
            raise ValueError('source release changed: ' + name)
        path = RAW / name
        if not path.exists() or path.stat().st_size != expected['bytes'] or digest(path, 'md5') != expected['md5']:
            if expected['bytes'] > 2 * 1024 * 1024:
                module.ranged_download({'name': name, 'bytes': expected['bytes'], 'url': native['links']['self'], 'checksum': native['checksum']}, path, workers=10)
            else:
                response = requests.get(native['links']['self'], timeout=30)
                response.raise_for_status()
                path.write_bytes(response.content)
        if digest(path, 'md5') != expected['md5']:
            raise ValueError('source checksum mismatch: ' + name)

def parent_key(reaction):
    return hashlib.sha256('.'.join(sorted(reaction.split('>>')[0].split('.'))).encode()).hexdigest()

def propose_split(parent):
    value = int(hashlib.sha256(('synrxn2027:' + parent).encode()).hexdigest()[:16], 16) % 100
    return 'train' if value < 80 else 'validation' if value < 90 else 'test'

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('not an allowed native dataset grant')
    for name, expected in ASSETS.items():
        path = RAW / name
        if path.stat().st_size != expected['bytes'] or digest(path, 'md5') != expected['md5']:
            raise ValueError('incomplete/truncated original QuantumPioneer asset: ' + name)
    columns = ['source_record_id', 'source_row_index', 'reaction_smiles', 'reactant_pair_group', 'proposed_split', 'temperature_K', 'reference_kind', 'theory', 'rate_unit', 'energy_unit', 'k_298', 'log10_k_298', 'A_low', 'Ea_low', 'A_high', 'Ea_high', 'barrier', 'deltaHrxn', 'deltaHrxn298', 'deltaGrxn298', 'P2M']
    native_path = RAW / 'quantumpioneer_kinetics_dataset.csv'
    output_path = OUT / 'quantumpioneer_kinetics.csv'
    n, parents, reaction_ids, split_counts = (0, set(), set(), Counter())
    with native_path.open() as source, output_path.open('w', newline='') as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(reader):
            reaction = row['rxn_smi']
            if reaction.count('>>') != 1 or not all(reaction.split('>>')):
                raise ValueError('invalid native reaction delimiter')
            numbers = {key: float(value) for key, value in row.items() if key != 'rxn_smi'}
            if not all((math.isfinite(value) for value in numbers.values())) or numbers['k_298'] <= 0:
                raise ValueError('invalid native kinetic reference')
            group = parent_key(reaction)
            split = propose_split(group)
            record = {'source_record_id': f'QuantumPioneer2026:kinetics:{index}', 'source_row_index': index, 'reaction_smiles': reaction, 'reactant_pair_group': group, 'proposed_split': split, 'temperature_K': 298.0, 'reference_kind': 'computed_transition_state_theory', 'theory': 'DLPNO-CCSD(T)-F12d/cc-pVTZ-F12 energies; wB97X-D/def2-SVP geometries; scaledDFTZPE; sourceTST', 'rate_unit': 'm^3/(mol*s)', 'energy_unit': 'kcal/mol', 'log10_k_298': math.log10(numbers['k_298']), **numbers}
            writer.writerow(record)
            n += 1
            parents.add(group)
            reaction_ids.add(reaction)
            split_counts[split] += 1
    if n != 166158 or len(reaction_ids) != n:
        raise ValueError('expected complete166158 unique native kinetics records')
    notice = OUT / 'QuantumPioneer2026.NOTICE.txt'
    creators = [item['name'] for item in metadata['metadata']['creators']]
    notice.write_text('QuantumPioneer Dataset v1\nCreators: ' + '; '.join(creators) + '\nData DOI: https://doi.org/10.5281/zenodo.20060606\nPaper: https://doi.org/10.1021/jacs.6c10371\n' + 'License: CC BY4.0 https://creativecommons.org/licenses/by/4.0/\n' + 'Changes: added stable source rowIDs, exact-recorded-reactant grouping and proposed group splits; added base10log of native k298. Native physical targets retained without unit conversion. Computed references, not measured rate constants. No endorsement implied.\n')
    license_path = OUT / 'CC-BY-4.0.txt'
    shutil.copyfile(ROOT / 'Data/raw/coverage/physics/CC-BY-4.0.txt', license_path)
    source_id = 'QuantumPioneer2026'
    source = {'id': source_id, 'name': 'QuantumPioneer native HAT kinetics', 'url': 'https://doi.org/10.5281/zenodo.20060606', 'paper_url': 'https://doi.org/10.1021/jacs.6c10371', 'creators': creators, 'license': 'CC-BY-4.0', 'data_license_spdx': 'CC-BY-4.0', 'redistribution_allowed': True, 'release_status': 'redistributable_with_attribution_and_source_notices', 'source_sha256': digest(native_path), 'files': [{'path': rel(p), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in [native_path, RAW / 'SCHEMA.md', metadata_path]], 'reference_evidence': 'Computed native transition-state-theory rate coefficients and source Arrhenius fits; DLPNO+scaledDFTZPE barrier and reaction thermo references. Not experimental measurements.', 'rights': {'scope': 'actual native Zenodo dataset assets', 'grant_url': 'https://zenodo.org/api/records/20060606', 'grant_snapshot': {'path': rel(metadata_path), 'sha256': digest(metadata_path)}, 'license_terms_url': 'https://creativecommons.org/licenses/by/4.0/', 'license_snapshot': {'path': rel(license_path), 'sha256': digest(license_path)}, 'notice': {'path': rel(notice), 'sha256': digest(notice)}}, 'upstream_assets': [{'name': name, **asset, 'url': f'https://zenodo.org/api/records/20060606/files/{name}/content'} for name, asset in ASSETS.items()]}
    inputs = ['reaction_smiles', 'temperature_K', 'theory']
    group_fields = ['reactant_pair_group', 'source_record_id', 'proposed_split']
    task_fields = {'reaction_rates': {'inputs': inputs, 'targets': ['k_298', 'log10_k_298'], 'group_fields': group_fields, 'metrics': ['MAE_log10_k298', 'RMSE_log10_k298', 'coverage'], 'qualification': 'Primary target is bimoleculark298; logarithm is the same target in transformedscale, not an independent label. ArrheniusA/Ea must never be input features for k298prediction.'}, 'activation_barrier': {'inputs': ['reaction_smiles', 'theory'], 'targets': ['barrier'], 'group_fields': group_fields, 'metrics': ['MAE_kcal_per_mol', 'RMSE_kcal_per_mol'], 'qualification': 'DLPNO+scaledDFTZPE forward potential-energy barrier; notEa_low/Ea_high and notGibbsactivationfreeenergy.'}, 'reaction_energetics': {'inputs': inputs, 'targets': ['deltaHrxn', 'deltaHrxn298', 'deltaGrxn298'], 'group_fields': group_fields, 'metrics': ['separate_quantity_MAE_kcal_per_mol'], 'qualification': 'Separate0Kreactionenthalpy,298Kenthalpy,298KGibbsreactionenergy variants; never collapse into a single scalarphysicalquantity. NativeCSVdeltaHrxn corresponds to SCHEMAHrxn naming.'}}
    table = {'id': 'quantumpioneer_hAT_kinetics', 'path': rel(output_path), 'task_ids': list(task_fields), 'rows': n, 'counting_unit': 'native computed HAT reaction with k298 and thermo-kinetic references', 'source_id': source_id, 'source_url': source['url'], 'source_sha256': source['source_sha256'], 'license': source['license'], 'columns': columns, 'sha256': digest(output_path), 'task_fields': task_fields, 'reference_evidence': source['reference_evidence'], 'release_status': source['release_status'], 'target_units': {'k_298': 'm^3/(mol*s)', 'log10_k_298': 'log10 of rate expressed in m^3/(mol*s)', 'A_low': 'm^3/(mol*s)', 'Ea_low': 'kcal/mol;300-1000Kfit', 'A_high': 'm^3/(mol*s)', 'Ea_high': 'kcal/mol;1000-2000Kfit', 'barrier': 'kcal/mol;DLPNO+scaledDFTZPE', 'deltaHrxn': 'kcal/mol;DLPNO+scaledDFTZPE', 'deltaHrxn298': 'kcal/mol;298K', 'deltaGrxn298': 'kcal/mol;298K', 'P2M': 'kcal/mol;sourcePetersson-to-Melius298K'}, 'parent_groups': len(parents), 'proposed_split_counts': dict(split_counts), 'qualification': 'Full166158 native positive finitek298 references. Native rates/TST are computed; no conversion to lograte legacy sourceunits. Exactrecorded-reactant groups keep candidateH-abstractionsites of same source reactants together; proposed split is new and does not establish scaffold independence. KineticCSVhasgasreferenceTST/thermo; no per-solvent correction inferred from separate solvationassets. No independent physical recomputation.'}
    result = {'schema': 'synrxn.coverage-physics-rates.v1', 'sources': [source], 'tables': [table], 'adapter': {'path': rel(__source_file__), 'sha256': digest(__source_file__), 'offline_command': 'python kdd/coverage/adapters/quantumpioneer.py', 'acquisition_command': 'python kdd/coverage/adapters/quantumpioneer.py --fetch'}}
    (OUT / 'catalog.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': n, 'parent_groups': len(parents), 'tasks': list(task_fields), 'proposed_split_counts': dict(split_counts)}, indent=2))
    return result
__configure_runtime(globals(), 'kdd/coverage/adapters/quantumpioneer.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch', action='store_true')
    args = p.parse_args()
    if args.fetch:
        fetch()
    build()
