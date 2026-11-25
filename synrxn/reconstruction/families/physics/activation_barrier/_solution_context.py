# Direct implementation; historical source identity is explicit below.
"""Rebuild solvent-aware overlapping views without changing canonical tables.

``python kdd/coverage/adapters/solution_context.py`` is offline. ``--fetch``
restores missing pinned native assets, using verified local copies first.
SNAr context is independently checked against the original 518-entry supplement.
Cycloaddition context is joined against the complete native 5,269-row dataset.
These views add condition inputs, not new independent reaction observations.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/solution_context.py')
import argparse
from collections import Counter, defaultdict, deque
import csv
import gzip
import hashlib
import io
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import shutil
import zipfile
from rdkit import Chem
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/solution_context'
OUT = ROOT / 'Data/coverage_expansion/solution_context'
LOCK = ROOT / 'kdd/coverage/adapters/solution_context_sources.json'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def relative(path):
    return Path(path).relative_to(ROOT).as_posix()

def pin(path):
    return {'path': relative(path), 'sha256': digest(path), 'bytes': Path(path).stat().st_size}

def read_rows(path, encoding='utf8'):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding=encoding, newline='') as handle:
        return list(csv.DictReader(handle))

def eligible_snar():
    result = []
    for index, row in enumerate(read_rows(RAW / 'snar.native.csv', 'cp1252')):
        try:
            valid = bool(row['Reaction SMILES']) and math.isfinite(float(row['Activation Free Energy (kcalmol-1)']))
        except ValueError:
            valid = False
        if valid:
            result.append((index, row))
    if len(result) != 503:
        raise ValueError('SNAr native eligibility changed')
    return result

def reconstructed_snar_context():
    canonical = read_rows(RAW / 'snar.canonical.csv.gz')
    native = eligible_snar()
    if len(canonical) != len(native):
        raise ValueError('SNAr membership differs')
    rows = []
    for current, (index, original) in zip(canonical, native, strict=True):
        if current['rxn'] != original['Reaction SMILES'] or not math.isclose(float(current['ea']), float(original['Activation Free Energy (kcalmol-1)']), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('SNAr canonical/native reaction or target differs')
        rows.append({'r_id': current['r_id'], 'native_record_id': original['Entry'], 'native_row': str(index), 'reported_temperature_kelvin': original['Temp (K)'], 'reported_solvent': original['Solvent'], 'reported_reference': original['Reference'], 'reported_doi': original['DOI'], 'reported_duplicate_entry_ids': original['Entry IDs of Duplicates']})
    return rows

def fetch():
    """Restore exact locked bytes; refuse changed upstream snapshots."""
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    assets = json.loads(LOCK.read_text())['assets']
    for asset in assets:
        path = ROOT / asset['path']
        if path.exists() and digest(path) == asset['sha256']:
            continue
        if path.name == 'snar.context.csv':
            continue
        original = ROOT / asset.get('bootstrap_path', '__missing__')
        if original.is_file() and digest(original) == asset['sha256']:
            shutil.copyfile(original, path)
        else:
            response = requests.get(asset['url'], timeout=90)
            response.raise_for_status()
            content = response.content
            for member in asset.get('archive_members', []):
                content = zipfile.ZipFile(io.BytesIO(content)).read(member)
            if hashlib.sha256(content).hexdigest() != asset['sha256']:
                raise ValueError('Retrieved bytes differ from source lock: ' + path.name)
            path.write_bytes(content)
    context = RAW / 'snar.context.csv'
    if not context.exists():
        rows = reconstructed_snar_context()
        with context.open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    verify_assets()

def verify_assets():
    for asset in json.loads(LOCK.read_text())['assets']:
        path = ROOT / asset['path']
        if not path.is_file() or path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
            raise ValueError('Missing or changed pinned solution-context dependency: ' + asset['path'])

def reaction_group(reaction):
    """Group identical chemical endpoints across solvents, temperatures and maps.

    RDKit canonicalization preserves stereochemistry; tautomer/resonance/scaffold
    equivalence is not claimed. No energy or condition enters this grouping.
    """
    sides = reaction.split('>>')
    if len(sides) != 2:
        raise ValueError('Expected two reaction endpoints')
    canonical = []
    for side in sides:
        molecules = []
        for component in side.split('.'):
            mol = Chem.MolFromSmiles(component)
            if mol is None:
                raise ValueError('Unparseable reaction endpoint')
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            molecules.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        canonical.append('.'.join(sorted(molecules)))
    return hashlib.sha256(('solution-reaction-v1:' + '>>'.join(canonical)).encode()).hexdigest()

def proposed_split(group):
    value = int(hashlib.sha256(('synrxn2027:' + group).encode()).hexdigest()[:16], 16) % 100
    return 'train' if value < 80 else 'validation' if value < 90 else 'test'

def source_record(identifier, name, paper, license_, paths, grant, grant_url, creators, evidence):
    terms = RAW / (license_ + '.txt')
    notice = OUT / (identifier + '.NOTICE.txt')
    notice.write_text('\n'.join([name, 'Creators: ' + '; '.join(creators), 'Paper: ' + paper, 'License: ' + license_, 'Terms: https://creativecommons.org/licenses/by/' + license_.split('-')[-1] + '/', 'Changes: joined original solvent and temperature to unchanged SynRXN targets; added canonical parent references and reaction-group splits.', 'These are overlapping views of existing canonical observations. No independent data or physical revalidation is claimed.', 'No endorsement by original creators is implied.', '']))
    return {'id': identifier, 'name': name, 'url': paper, 'paper_url': paper, 'creators': creators, 'license': license_, 'data_license_spdx': license_, 'redistribution_allowed': True, 'release_status': 'redistributable_with_attribution_and_source_notices', 'source_sha256': digest(paths[0]), 'files': [pin(path) for path in paths], 'reference_evidence': evidence, 'rights': {'scope': 'source dataset factual reaction and condition tables and their already released SynRXN curation', 'grant_url': grant_url, 'grant_snapshot': pin(grant), 'license_terms_url': 'https://creativecommons.org/licenses/by/' + license_.split('-')[-1] + '/', 'license_snapshot': pin(terms), 'notice': pin(notice)}}

def save_table(identifier, rows, source, parent, task_fields, qualification):
    path = OUT / (identifier + '.csv')
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return {'id': identifier, 'path': relative(path), 'rows': len(rows), 'columns': list(rows[0]), 'sha256': digest(path), 'task_ids': list(task_fields), 'task_fields': task_fields, 'source_id': source['id'], 'source_url': source['url'], 'source_sha256': source['source_sha256'], 'license': source['license'], 'release_status': source['release_status'], 'reference_evidence': source['reference_evidence'], 'counting_unit': 'existing canonical reaction-condition observation', 'role': 'overlapping_view', 'overlapping_view': True, 'overlap_parent': parent, 'derived_view_of': parent, 'independent_observations_added': 0, 'builder': 'kdd/coverage/adapters/solution_context.py', 'adapter_sha256': digest(__source_file__), 'source_lock': relative(LOCK), 'source_lock_sha256': digest(LOCK), 'solvent_strings': len({r['solvent'] for r in rows}), 'temperature_values': len({r['temperature_K'] for r in rows}), 'reaction_groups': len({r['reaction_group'] for r in rows}), 'proposed_split_counts': dict(Counter((r['proposed_split'] for r in rows))), 'target_units': {target: 'kcal/mol' for fields in task_fields.values() for target in fields['targets']}, 'qualification': qualification}

def fields(targets, qualification):
    return {'inputs': ['reaction_smiles', 'solvent', 'temperature_K', 'reference_kind'], 'targets': targets, 'group_fields': ['reaction_group', 'parent_dataset', 'parent_record_id', 'proposed_split'], 'metrics': ['MAE_kcal_per_mol', 'RMSE_kcal_per_mol', 'coverage'], 'qualification': qualification}

def build():
    verify_assets()
    OUT.mkdir(parents=True, exist_ok=True)
    context = read_rows(RAW / 'snar.context.csv')
    if context != reconstructed_snar_context():
        raise ValueError('Pinned SNAr context does not reproduce the original supplement')
    by_id = {row['r_id']: row for row in context}
    rows = []
    for current in read_rows(RAW / 'snar.canonical.csv.gz'):
        original = by_id[current['r_id']]
        group = reaction_group(current['rxn'])
        rows.append({'source_record_id': 'SNArSolution2021:' + original['native_record_id'], 'parent_dataset': 'property/snar', 'parent_record_id': 'property/snar:' + current['r_id'], 'native_record_id': original['native_record_id'], 'native_row': original['native_row'], 'reaction_smiles': current['rxn'], 'solvent': original['reported_solvent'], 'temperature_K': original['reported_temperature_kelvin'], 'reaction_group': group, 'proposed_split': proposed_split(group), 'source_split': 'not assigned in original 518-entry compilation', 'reference_kind': 'experimental_rate_derived_Eyring_activation_free_energy', 'activation_free_energy_kcal_mol': current['ea'], 'literature_reference': original['reported_reference'], 'literature_doi': original['reported_doi'], 'duplicate_entry_ids': original['reported_duplicate_entry_ids']})
    source = source_record('SNArSolution2021', 'SNAr experimental activation free energies with reported solution conditions', 'https://doi.org/10.1039/D0SC04896H', 'CC-BY-3.0', [RAW / name for name in ['snar.native.csv', 'snar.canonical.csv.gz', 'snar.context.csv', 'snar.grant.xml', 'snar.crossref.json']], RAW / 'snar.crossref.json', 'https://api.crossref.org/works/10.1039/D0SC04896H', ['Kjell Jorner', 'Tore Brinck', 'Per-Ola Norrby', 'David Buttar'], 'All503 canonical reactions and activation free energies match source-ordered eligible entries of original518-entry supplement. Solvent/temperature/reference/duplicate fields checked independently against that table; no remeasurement.')
    source['rights']['scope'] = 'publisher_bundled_supplementary_data'
    source['rights']['publication_doi'] = '10.1039/D0SC04896H'
    source['rights']['scope_qualification'] = "Original publisher-bundled SNAr supplementary reaction table and its verified SynRXN curation/context join. The publisher's article license is not treated as a grant for independently hosted repository data."
    tables = [save_table('snar_solution_context', rows, source, 'property/snar', {'activation_barrier': fields(['activation_free_energy_kcal_mol'], 'Eyring-derived activation FREE energy at reported solvent and temperature; not Arrhenius Ea or gas-phase electronic barrier.')}, '503 unchanged observations;41 literal solvent strings (including mixtures/aliases) and13 temperatures. Exact normalized reaction endpoints remain grouped across conditions. Proposed splits are new group splits, not source evaluation splits or scaffold/solvent extrapolation guarantees. Parent canonical rows must never be used as independent train/test data. Preserve raw solvent labels;41 strings does not mean41 chemically distinct pure solvents.')]
    sources = [source]
    native_rows = read_rows(RAW / 'cycloadd.native.csv')
    lookup = defaultdict(deque)
    for index, row in enumerate(native_rows):
        lookup[row['rxn_smiles'], round(float(row['G_act']), 8), round(float(row['G_r']), 8)].append((index, row))
    rows = []
    for current in read_rows(RAW / 'cycloadd.canonical.csv.gz'):
        key = (current['aam'], round(float(current['G_act']), 8), round(float(current['G_r']), 8))
        if not lookup[key]:
            raise ValueError('Cycloaddition canonical reaction/targets absent from native data')
        index, original = lookup[key].popleft()
        if not all((math.isclose(float(current[field]), float(original[field]), rel_tol=1e-12, abs_tol=1e-12) for field in ['G_act', 'G_r'])):
            raise ValueError('Cycloaddition target differs from native source')
        group = reaction_group(current['aam'])
        rows.append({'source_record_id': 'CycloaddSolution2023:' + original['rxn_id'], 'parent_dataset': 'property/cycloadd', 'parent_record_id': 'property/cycloadd:' + current['r_id'], 'native_record_id': original['rxn_id'], 'native_row': index, 'reaction_smiles': current['aam'], 'solvent': original['solvent'], 'temperature_K': original['temp'], 'reaction_group': group, 'proposed_split': proposed_split(group), 'source_split': current['split'], 'reference_kind': 'computed_B3LYP_D3BJ_SMD_water_activation_and_reaction_free_energy', 'activation_free_energy_kcal_mol': current['G_act'], 'reaction_free_energy_kcal_mol': current['G_r']})
    if len(rows) != 5269 or any(lookup.values()):
        raise ValueError('Cycloaddition join is not a complete native bijection')
    metadata = json.loads((RAW / 'cycloadd.metadata.json').read_text())
    source = source_record('CycloaddSolution2023', 'Dipolar cycloaddition computed aqueous reaction profiles', 'https://doi.org/10.6084/m9.figshare.21707888.v5', 'CC-BY-4.0', [RAW / name for name in ['cycloadd.native.csv', 'cycloadd.canonical.csv.gz', 'cycloadd.metadata.json']], RAW / 'cycloadd.metadata.json', 'https://api.figshare.com/v2/articles/21707888/versions/5', [author['full_name'] for author in metadata['authors']], 'Complete5269-reaction native CSV bijection verifies both free-energy targets, reaction strings and source solvent/temperature. Source paper uses aqueous SMD continuum at B3LYP-D3(BJ)/def2-TZVP//B3LYP-D3(BJ)/def2-SVP; not gas phase or explicit-solvent trajectories.')
    source['paper_url'] = 'https://doi.org/10.1038/s41597-023-01977-8'
    sources.append(source)
    tables.append(save_table('cycloadd_solution_context', rows, source, 'property/cycloadd', {'activation_barrier': fields(['activation_free_energy_kcal_mol'], 'Computed aqueous SMD activation Gibbs free energy; not gas-phase electronic barrier.'), 'reaction_energetics': fields(['reaction_free_energy_kcal_mol'], 'Computed aqueous SMD reaction Gibbs free energy at298.15K; separate from activation target.')}, '5269 unchanged observations, all water/298.15K. Solvent and temperature are constant context: this table cannot establish solvent-effect or temperature generalization. Original Chemprop source splits retained as metadata; use proposed reaction-group split for this overlapping view and apply same groups to canonical parent rows. Not physically recalculated.'))
    result = {'schema': 'synrxn.coverage-solution-context.v1', 'sources': sources, 'tables': tables, 'source_lock': relative(LOCK), 'source_lock_sha256': digest(LOCK), 'overlap_policy': 'Both tables are overlapping solvent-aware views of canonical parents, adding zero independent observations. Parent rows and all condition variants must share evaluation groups; never concatenate as independent examples.', 'adapter': relative(__source_file__), 'adapter_sha256': digest(__source_file__), 'reproduction': {'offline_command': 'python kdd/coverage/adapters/solution_context.py', 'acquisition_command': 'python kdd/coverage/adapters/solution_context.py --fetch'}}
    (OUT / 'catalog.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'tables': [{k: t[k] for k in ['id', 'rows', 'solvent_strings', 'temperature_values', 'reaction_groups']} for t in tables], 'independent_observations_added': 0}))
    return result
__configure_runtime(globals(), 'kdd/coverage/adapters/solution_context.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch()
    build()
