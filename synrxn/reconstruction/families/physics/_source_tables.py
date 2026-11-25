# Direct implementation; historical source identity is explicit below.
"""Build provenance-preserving physical reaction task references.

Run ``python kdd/coverage/adapters/physics.py --fetch`` to acquire pinned
upstream files and build; omit --fetch for an entirely offline rebuild.
Transition1x configurations remain in their native HDF5 file.  Its CSV is an
index of complete reaction trajectories, not a CSV pretending each trajectory
is one molecular configuration.  ChemSINDy observations are explicitly derived
ODE simulations of published example mechanisms, never experimental data.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/physics.py')
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import time
import zipfile
from synrxn.reconstruction.paths import HistoricalPath as Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/physics'
OUT = ROOT / 'Data/coverage_expansion/physics'
CHEM_COMMIT = '9aa3f8adb997348bd1e85c1b61fa8a835a91c759'
CHEM_REPO = 'https://github.com/nla-group/ChemSINDy'
T1X = {'name': 'Transition1x.h5', 'size': 6623929000, 'url': 'https://ndownloader.figshare.com/files/36035789', 'md5': 'faab0413cf6b042b6d50e6272dea9a8c'}
ORG_FILES = [{'name': 'OrgReact_xyz.zip', 'size': 14928834, 'url': 'https://ndownloader.figshare.com/files/64758867', 'md5': '47783af56ffe750041df2ba95e8a934d'}, {'name': 'energy_and_bonds_changes_files.zip', 'size': 112204, 'url': 'https://ndownloader.figshare.com/files/64783677', 'md5': '5db5b52442732408e8769091d4d95034'}]
CHEM_SHA256 = {'mechanisms.py': '0035b000a877aded91b34f0ad02f6d1fbe56dca89e4d37599ffdf514fa428999', 'M1_graph-recovery.ipynb': 'b2af4464ed7daf549dd4e6dd1db12fc52e77c3b331823f4695734285759af4ef', 'M20_VdV_graph-recovery.ipynb': '37041dfb175129b1d8705b0ca3285bcc26fcdfd05b894f90e35267ccf6dd3893', 'LICENSE': '81e3b6b345a00a81e369bfcb707fa33d3b7bf51244c33497e6460eb6e7c453ab'}

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        while (chunk := handle.read(4 * 1024 * 1024)):
            h.update(chunk)
    return h.hexdigest()

def relative(path):
    return Path(path).relative_to(ROOT).as_posix()

def fetch_asset(asset, workers=24):
    """Fetch exact HTTP ranges, verify byte lengths AND publisher MD5.

    Incomplete .part files never become admitted native data.  Cached ranges
    carry their original offsets and are only assembled in numeric order.
    """
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / asset['name']
    if path.exists() and path.stat().st_size == asset['size'] and (digest(path, 'md5') == asset['md5']):
        return path
    width = 4 * 1024 * 1024
    identity = hashlib.sha256(json.dumps(asset, sort_keys=True).encode()).hexdigest()[:16]
    directory = RAW / '.ranges' / identity
    directory.mkdir(parents=True, exist_ok=True)

    def get(start):
        end = min(asset['size'], start + width) - 1
        part = directory / f'{start:012d}.bin'
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(5):
            try:
                response = requests.get(asset['url'] + f'?synrxn={start}-{time.time_ns()}', headers={'Range': f'bytes={start}-{end}'}, timeout=60)
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                    raise ValueError('upstream did not honor exact requested range')
                if len(response.content) != end - start + 1:
                    raise ValueError('truncated HTTP range')
                part.write_bytes(response.content)
                return part
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(get, range(0, asset['size'], width)))
    temporary = path.with_name(path.name + '.part')
    with temporary.open('wb') as handle:
        for part in parts:
            handle.write(part.read_bytes())
    if temporary.stat().st_size != asset['size'] or digest(temporary, 'md5') != asset['md5']:
        raise ValueError(f"publisher checksum mismatch: {asset['name']}")
    temporary.replace(path)
    return path

def acquire():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for name, article, version in [('Transition1x', 19614657, 4), ('OrgReact', 31668682, 3)]:
        response = requests.get(f'https://api.figshare.com/v2/articles/{article}/versions/{version}', timeout=30)
        response.raise_for_status()
        (RAW / f'{name}.metadata.json').write_text(json.dumps(response.json(), indent=2) + '\n')
    for asset in ORG_FILES + [T1X]:
        print('Verify/acquire', asset['name'], flush=True)
        fetch_asset(asset)
    folder = RAW / 'ChemSINDy'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['LICENSE', 'README.md', 'mechanisms.py', 'M1_graph-recovery.ipynb', 'M20_VdV_graph-recovery.ipynb']:
        url = f'https://raw.githubusercontent.com/nla-group/ChemSINDy/{CHEM_COMMIT}/{name}'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        (folder / name).write_bytes(response.content)
    (folder / 'revision.json').write_text(json.dumps({'repository': 'nla-group/ChemSINDy', 'commit': CHEM_COMMIT}, indent=2) + '\n')
    for name, url in [('MIT-license.html', 'https://opensource.org/licenses/MIT'), ('CC-BY-4.0.txt', 'https://creativecommons.org/licenses/by/4.0/legalcode.txt')]:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        (RAW / name).write_bytes(response.content)

def encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)

def save_table(identifier, rows, source, task_fields, qualification, counting_unit):
    if not rows:
        raise ValueError(f'empty reference table {identifier}')
    path = OUT / f'{identifier}.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {'id': identifier, 'path': relative(path), 'task_ids': list(task_fields), 'rows': len(rows), 'counting_unit': counting_unit, 'source_id': source['id'], 'source_url': source['url'], 'source_sha256': source['source_sha256'], 'license': source['license'], 'columns': list(rows[0]), 'qualification': qualification, 'reference_evidence': source['reference_evidence'], 'task_fields': task_fields, 'sha256': digest(path), 'release_status': source.get('release_status', 'license_identified')}

def source_record(identifier, name, url, license_, paths, evidence, **kwargs):
    files = [{'path': relative(path), 'sha256': digest(path), 'bytes': Path(path).stat().st_size} for path in paths]
    return {'id': identifier, 'name': name, 'url': url, 'license': license_, 'files': files, 'source_sha256': files[0]['sha256'], 'reference_evidence': evidence, **kwargs}

def attach_redistribution_evidence(source, declaration_path, declaration_url, license_path, license_url, changes):
    """Retain the actual source grant and attribution needed for redistribution."""
    directory = OUT / 'notices'
    directory.mkdir(parents=True, exist_ok=True)
    notice = directory / f"{source['id']}.NOTICE.txt"
    notice.write_text('\n'.join([source['name'], 'Source: ' + source['url'], 'Creators: ' + '; '.join(source['creators']), 'License: ' + source['license'], 'Source license declaration: ' + declaration_url, 'License terms: ' + license_url, 'Changes in this distribution: ' + changes, 'No endorsement by the original authors is implied.', '']))
    copied_license = directory / f"{source['id']}.LICENSE{license_path.suffix}"
    copied_license.write_bytes(license_path.read_bytes())
    source.update(data_license_spdx=source['license'], redistribution_allowed=True, release_status='redistributable_with_attribution_and_source_notices', rights={'scope': 'native dataset assets' if source['id'] != 'ChemSINDy2026' else 'MIT source code/notebooks and explicitly regenerated simulated references; no third-party measured data copied', 'grant_url': declaration_url, 'grant_snapshot': {'path': relative(declaration_path), 'sha256': digest(declaration_path)}, 'license_terms_url': license_url, 'license_snapshot': {'path': relative(license_path), 'sha256': digest(license_path)}, 'notice': {'path': relative(notice), 'sha256': digest(notice)}, 'distributed_license_notice': {'path': relative(copied_license), 'sha256': digest(copied_license)}})

def parse_xyz(data):
    lines = data.decode().strip().splitlines()
    count = int(lines[0])
    atoms, positions = ([], [])
    for line in lines[2:]:
        fields = line.split()
        atoms.append(fields[0])
        positions.append([float(x) for x in fields[1:4]])
    if len(atoms) != count or np.asarray(positions).shape != (count, 3) or (not np.isfinite(positions).all()):
        raise ValueError('invalid native XYZ')
    return (atoms, positions)

def build_orgreact():
    for asset in ORG_FILES:
        path = RAW / asset['name']
        if path.stat().st_size != asset['size'] or digest(path, 'md5') != asset['md5']:
            raise ValueError('OrgReact source differs from pinned release')
    source = source_record('OrgReact2026', 'OrgReact native endpoint and transition-state structures', 'https://doi.org/10.6084/m9.figshare.31668682.v3', 'CC-BY-4.0', [RAW / x['name'] for x in ORG_FILES] + [RAW / 'OrgReact.metadata.json'], 'Computed: source paper reports wB97X-D/def2-TZVP optimization, imaginary-frequency and IRC validation; no independent rerun here.', paper_url='https://www.nature.com/articles/s41597-026-07771-6', license_evidence='Figshare article31668682 metadata license CC BY4.0; metadata stored alongside raw files.', overlap_policy='Related to Transition1x/Grambow reactants; source reports944 shared reactions. Never assume independent cross-dataset holdout.')
    source['upstream_assets'] = ORG_FILES
    source['creators'] = [author['full_name'] for author in json.loads((RAW / 'OrgReact.metadata.json').read_text())['authors']]
    if json.loads((RAW / 'OrgReact.metadata.json').read_text())['license']['name'] != 'CC BY 4.0':
        raise ValueError('native OrgReact dataset license declaration differs')
    attach_redistribution_evidence(source, RAW / 'OrgReact.metadata.json', 'https://api.figshare.com/v2/articles/31668682/versions/3', RAW / 'CC-BY-4.0.txt', 'https://creativecommons.org/licenses/by/4.0/', 'Parsed XYZ endpoint/TS structures and joined source energetic/bond-change tables; native files remain unchanged. Added row IDs, explicit missing charge/spin status, task contracts and provenance.')
    z = zipfile.ZipFile(RAW / 'OrgReact_xyz.zip')
    metadata = zipfile.ZipFile(RAW / 'energy_and_bonds_changes_files.zip')
    energies = {x['Reaction']: float(x['Reaction energy barrier(kcal/mol)']) for x in csv.DictReader(io.StringIO(metadata.read('energy_and_bonds_changes_files/OrgReact_energy.csv').decode()))}
    changes = {x['Reaction']: x['Types of chemical bond changes'] for x in csv.DictReader(io.StringIO(metadata.read('energy_and_bonds_changes_files/OrgReact_bond_changes.csv').decode()))}
    names = sorted({name.split('/')[1] for name in z.namelist() if name.endswith('/TS.xyz')}, key=lambda x: int(x[3:]))
    rows = []
    for name in names:
        structures = [parse_xyz(z.read(f'OrgReact_xyz/{name}/{role}.xyz')) for role in ['RS', 'PS', 'TS']]
        if not structures[0][0] == structures[1][0] == structures[2][0]:
            raise ValueError(f'native endpoint atom order differs: {name}')
        atoms = structures[0][0]
        endpoint_group = hashlib.sha256(encode([atoms, structures[0][1]]).encode()).hexdigest()
        rows.append({'source_record_id': f'OrgReact2026:{name}', 'reaction_id': name, 'endpoint_group': endpoint_group, 'source_members': encode([f'OrgReact_xyz/{name}/{s}.xyz' for s in ['RS', 'PS', 'TS']]), 'elements': encode(atoms), 'reactant_positions_A': encode(structures[0][1]), 'product_positions_A': encode(structures[1][1]), 'ts_positions_A': encode(structures[2][1]), 'reaction_barrier_kcal_mol': energies.get(name, ''), 'bond_change_class': changes.get(name, ''), 'theory': 'wB97X-D/def2-TZVP', 'charge': '', 'multiplicity': '', 'electronic_context_status': 'charge/spin not explicit in released XYZ; not inferred', 'source_split': 'not released', 'reference_kind': 'computed_source_reported_IRC_and_frequency_validation'})
    fields = {'ts_geometry': {'inputs': ['elements', 'reactant_positions_A', 'product_positions_A', 'theory'], 'targets': ['ts_positions_A'], 'group_fields': ['endpoint_group', 'reaction_id'], 'metrics': ['symmetry-aware aligned RMSD_A', 'coverage', 'optional independently verified saddle recovery']}}
    table = save_table('orgreact_ts_geometry', rows, source, fields, 'Complete published XYZ endpoint/TS reference set; retain native atom order. Geometry-only target qualified; charge/spin and per-row IRC logs are absent from these released assets. No reverified saddle claim, no ready-made independent split.', 'reaction endpoint/transition-state triplet')
    table['missing_barrier_rows'] = sum((x['reaction_barrier_kcal_mol'] == '' for x in rows))
    return (source, [table])

def _native_array(group, key):
    value = group[key][()]
    return value.tolist()

def build_transition1x():
    import h5py
    path = RAW / 'Transition1x.h5'
    if path.stat().st_size != T1X['size'] or digest(path, 'md5') != T1X['md5']:
        raise ValueError('Transition1x incomplete or publisher checksum mismatch')
    source = source_record('Transition1x2022', 'Transition1x full native reactive trajectory release', 'https://doi.org/10.6084/m9.figshare.19614657.v4', 'MIT', [path, RAW / 'Transition1x.metadata.json'], 'Computed: native wB97x/6-31G(d) NEB-region energies in eV, forces in eV/A, positions in A; trajectories and final endpoint/TS labels are source generated.', paper_url='https://www.nature.com/articles/s41597-022-01870-w', license_evidence='Figshare article19614657 metadata grants MIT for native HDF5; publisher MD5 faab0413cf6b042b6d50e6272dea9a8c verified.', selection='Entire upstream HDF5 admitted; no truncated file or arbitrary sample counted as full dataset.', overlap_policy='Derived from Grambow reactions; related to legacy b97xd3 and OrgReact. Use native formula split for within-source evaluation and audit any cross-source holdout.')
    source['upstream_assets'] = [T1X]
    source['creators'] = [author['full_name'] for author in json.loads((RAW / 'Transition1x.metadata.json').read_text())['authors']]
    if json.loads((RAW / 'Transition1x.metadata.json').read_text())['license']['name'] != 'MIT':
        raise ValueError('native Transition1x dataset license declaration differs')
    attach_redistribution_evidence(source, RAW / 'Transition1x.metadata.json', 'https://api.figshare.com/v2/articles/19614657/versions/4', RAW / 'MIT-license.html', 'https://opensource.org/licenses/MIT', 'Native HDF5 is unmodified. Added complete trajectory index, preserved native splits, extracted final endpoint/TS coordinates, explicit missing charge/spin status and task contracts.')
    rows, final_rows = ([], [])
    final_errors = []
    with h5py.File(path, 'r') as h:
        data = h['data']
        splits = {formula: split for split in ['train', 'val', 'test'] if split in h for formula in h[split]}
        for formula in sorted(data):
            for reaction in sorted(data[formula]):
                group = data[formula][reaction]
                if not isinstance(group, h5py.Group) or 'positions' not in group:
                    continue
                prefix = group.name
                numbers = _native_array(group, 'atomic_numbers')
                positions = group['positions']
                energy_key, force_key = ('wB97x_6-31G(d).energy', 'wB97x_6-31G(d).forces')
                if positions.shape[-2:] != (len(numbers), 3) or group[force_key].shape != positions.shape:
                    raise ValueError(f'native force/position shape mismatch: {prefix}')
                n = positions.shape[0]
                if group[energy_key].shape[0] != n:
                    raise ValueError(f'native energy/position count mismatch: {prefix}')
                if not all((np.isfinite(group[key][()]).all() for key in ['positions', energy_key, force_key])):
                    raise ValueError(f'non-finite native physical reference: {prefix}')
                common = {'source_record_id': f'Transition1x2022:{formula}/{reaction}', 'trajectory_id': f'{formula}/{reaction}', 'formula': formula, 'native_split': splits.get(formula, 'unassigned'), 'native_file': relative(path), 'native_group': prefix, 'atomic_numbers': encode(numbers), 'theory': 'wB97x/6-31G(d)', 'charge': '', 'multiplicity': '', 'electronic_context_status': 'not explicit per row in native release; not inferred'}
                rows.append({**common, 'configuration_count': n, 'positions_path': prefix + '/positions', 'energies_path': prefix + '/' + energy_key, 'forces_path': prefix + '/' + force_key, 'positions_unit': 'angstrom', 'energies_unit': 'eV', 'forces_unit': 'eV/angstrom'})
                if all((role in group for role in ['reactant', 'product', 'transition_state'])):
                    finals = [group[role] for role in ['reactant', 'product', 'transition_state']]
                    try:
                        xyz = [np.asarray(_native_array(g, 'positions')).reshape(-1, 3).tolist() for g in finals]
                        energies = [float(np.asarray(g[energy_key][()]).reshape(-1)[0]) for g in finals]
                        if any((len(x) != len(numbers) for x in xyz)):
                            raise ValueError('final atom counts differ')
                        final_rows.append({**common, 'reactant_positions_A': encode(xyz[0]), 'product_positions_A': encode(xyz[1]), 'ts_positions_A': encode(xyz[2]), 'reactant_energy_eV': energies[0], 'product_energy_eV': energies[1], 'ts_energy_eV': energies[2], 'reference_kind': 'native_final_NEB_stationary_points'})
                    except (ValueError, KeyError) as error:
                        final_errors.append({'native_group': prefix, 'reason': str(error)})
                else:
                    final_errors.append({'native_group': prefix, 'reason': 'missing final endpoint/TS subgroup'})
    fields = {'reactive_potential': {'inputs': ['atomic_numbers', 'positions_path', 'native_file', 'theory'], 'targets': ['energies_path', 'forces_path'], 'group_fields': ['trajectory_id', 'formula', 'native_split'], 'metrics': ['energy_MAE_eV', 'force_MAE_eV_per_A', 'per-trajectory error']}}
    table = save_table('transition1x_reactive_trajectories', rows, source, fields, 'CSV indexes full native trajectory tensors: rows count reaction trajectories, configuration_count sums molecular configurations. Native formula train/val/test retained; adjacent frames stay together. Per-row charge/spin are not supplied in native HDF5.', 'complete native reaction trajectory; configuration_count gives individual atomic configurations')
    table['atomic_configurations'] = sum((x['configuration_count'] for x in rows))
    table['native_split_trajectories'] = {s: sum((x['native_split'] == s for x in rows)) for s in ['train', 'val', 'test', 'unassigned']}
    ts_fields = {'ts_geometry': {'inputs': ['atomic_numbers', 'reactant_positions_A', 'product_positions_A', 'theory'], 'targets': ['ts_positions_A'], 'group_fields': ['trajectory_id', 'formula', 'native_split'], 'metrics': ['symmetry-aware aligned RMSD_A', 'coverage']}}
    final_table = save_table('transition1x_final_ts_geometry', final_rows, source, ts_fields, 'Source final NEB structures only; no intermediate-frame maximum substituted for TS. Missing final groups explicitly excluded; source-reported computational reference, independently recomputed saddle validation pending.', 'native reaction endpoint/transition-state triplet')
    final_table['excluded_final_groups'] = final_errors
    return (source, [table, final_table])

def notebook_text(path):
    notebook = json.loads(path.read_text())
    return '\n'.join((''.join(output.get('text', [])) for cell in notebook['cells'] for output in cell.get('outputs', [])))

def mechanism(model):
    """Reviewed stoichiometry transcribed from pinned authors' mechanisms.py.

    Reactant/product arrays have one row per directed elementary reaction.
    No source code is dynamically executed.
    """
    labels = ['A', 'P', 'cat', 'catA'] + (['catI', 'catAI'] if model == 'M20' else [])
    m = len(labels)
    reactants = np.zeros((4 if model == 'M1' else 6, m), dtype=int)
    products = np.zeros_like(reactants)
    reactants[0, [0, 2]], products[0, 3] = (1, 1)
    reactants[1, 3], products[1, [0, 2]] = (1, 1)
    reactants[2, 3], products[2, [1, 2]] = (1, 1)
    reactants[3, [1, 2]], products[3, 3] = (1, 1)
    if model == 'M20':
        reactants[4, 2], products[4, 4] = (1, 1)
        reactants[5, 3], products[5, 5] = (1, 1)
    return (labels, reactants, products)

def rates_from_state(state, reactants):
    return np.prod(np.asarray(state)[None, :] ** reactants, axis=1)

def build_chemsindy():
    folder = RAW / 'ChemSINDy'
    paths = [folder / n for n in ['mechanisms.py', 'M1_graph-recovery.ipynb', 'M20_VdV_graph-recovery.ipynb', 'LICENSE']]
    for path in paths:
        if CHEM_SHA256[path.name] and digest(path) != CHEM_SHA256[path.name]:
            raise ValueError(f'pinned ChemSINDy source differs: {path.name}')
    source = source_record('ChemSINDy2026', 'ChemSINDy published elementary-network simulation examples', CHEM_REPO + '/tree/' + CHEM_COMMIT, 'MIT', paths, 'Derived simulation: published M1/M20 mass-action mechanisms and saved notebook example parameters; solve_ivp integration is performed by this adapter. These are not acquired experimental trajectories.', paper_url='https://arxiv.org/abs/2602.11849', publication_status='2026 preprint', license_evidence='Pinned repository LICENSE is MIT; raw source code/notebooks are retained with SHA256.', commit=CHEM_COMMIT, simulation_policy='M1 exact RNG seed9006 recreates original full-precision draws; M20 uses the explicitly printed 8-decimal constants and 3-decimal initial conditions because notebook RNG was not seeded. This is a documented rounded-example reconstruction, not a bitwise reproduction of original hidden draws.')
    source['creators'] = ['Abraham Reyes-Velazquez', 'Stefan Güttel', 'Igor Larrosa', 'Jonas Latz']
    attach_redistribution_evidence(source, folder / 'LICENSE', f'https://github.com/nla-group/ChemSINDy/blob/{CHEM_COMMIT}/LICENSE', folder / 'LICENSE', 'https://opensource.org/licenses/MIT', 'Reviewed and transcribed M1/M20 stoichiometry; regenerated explicit ODE simulations using source notebook example settings. M1 uses exact native seed9006; M20 reconstructs printed rounded constants and initial values. These are derived simulated references, not experimental observations or a bitwise reproduction of hidden M20 random draws.')
    rows, points = ([], [])
    for model, notebook_name, experiments, npts in [('M1', 'M1_graph-recovery.ipynb', 6, 30), ('M20', 'M20_VdV_graph-recovery.ipynb', 8, 50)]:
        text = notebook_text(folder / notebook_name)
        match = re.search('Run kinetic rates:\\s*\\[([^]]+)\\].*?Run initial values:\\s*(\\[\\[.*?\\]\\])', text, re.S)
        if not match:
            raise ValueError('native notebook lacks saved reference constants/initial conditions')
        printed_k = np.fromstring(match.group(1), sep=' ')
        initial = np.fromstring(match.group(2).replace('[', ' ').replace(']', ' '), sep=' ').reshape(experiments, -1)
        if model == 'M1':
            rng = np.random.default_rng(9006)
            constants = rng.uniform(0.05, 1.0, size=4)
            exact_initial = rng.uniform(0, 1, size=(6, 4))
            if not np.allclose(constants, printed_k, atol=5e-09) or not np.array_equal(np.round(exact_initial, 3), initial):
                raise ValueError('published M1 seed no longer matches saved native references')
            initial = exact_initial
            reconstruction = 'native_seed9006_exact_draws'
        else:
            constants = printed_k
            reconstruction = 'native_printed_constants8dp_initial3dp_reconstruction'
        labels, reactants, products = mechanism(model)
        stoich = products - reactants
        times = np.linspace(0, 20, npts)
        design = []
        for i, y0 in enumerate(initial):
            fun = lambda t, x: stoich.T @ (constants * rates_from_state(x, reactants))
            result = solve_ivp(fun, (0, 20), y0=y0, t_eval=times, rtol=1e-10, atol=1e-10)
            if not result.success or not np.isfinite(result.y).all() or np.min(result.y) < -1e-08:
                raise ValueError('invalid mass-action reference integration')
            for state in result.y.T:
                design.append(stoich.T * rates_from_state(state, reactants)[None, :])
            trajectory = f'ChemSINDy2026:{model}/experiment{i}'
            rows.append({'source_record_id': trajectory, 'trajectory_id': trajectory, 'mechanism_id': model, 'source_notebook': relative(folder / notebook_name), 'source_reference_kind': 'derived_simulation', 'reconstruction': reconstruction, 'species': encode(labels), 'initial_concentrations': encode(y0.tolist()), 'time_values': encode(times.tolist()), 'concentration_trajectories': encode(result.y.T.tolist()), 'reactant_stoichiometry': encode(reactants.tolist()), 'product_stoichiometry': encode(products.tolist()), 'rate_constants': encode(constants.tolist()), 'parameter_names': encode(['k1', 'k_minus1', 'k2', 'k_minus2'] + (['kI', 'kAI'] if model == 'M20' else [])), 'units': 'source arbitrary concentration/time units; not assigned experimental molarity or seconds', 'solver': 'scipy.solve_ivp RK45 rtol=1e-10 atol=1e-10; no added noise', 'source_split': 'none; only2 mechanism families; do not split adjacent samples'})
            for j, (t, values) in enumerate(zip(times, result.y.T)):
                points.append({'source_record_id': f'{trajectory}:point{j}', 'trajectory_id': trajectory, 'mechanism_id': model, 'time': float(t), 'species': encode(labels), 'concentrations': encode(values.tolist()), 'reference_kind': 'derived_simulation'})
        rank = np.linalg.matrix_rank(np.concatenate(design, axis=0))
        if rank != len(constants):
            raise ValueError(f'supplied known-network parameter experiment design is rank deficient: {model}')
    fields = {'kinetic_parameter_inference': {'inputs': ['species', 'initial_concentrations', 'time_values', 'concentration_trajectories', 'reactant_stoichiometry', 'product_stoichiometry', 'units'], 'targets': ['rate_constants'], 'group_fields': ['mechanism_id', 'trajectory_id'], 'metrics': ['relative_rate_parameter_error', 'held_out_trajectory_RMSE', 'reference_parameter_design_rank']}, 'reaction_network_modeling': {'inputs': ['species', 'initial_concentrations', 'time_values', 'concentration_trajectories', 'units'], 'targets': ['reactant_stoichiometry', 'product_stoichiometry', 'rate_constants'], 'group_fields': ['mechanism_id', 'trajectory_id'], 'metrics': ['directed_stoichiometric_edge_precision_recall', 'dynamical_equivalence', 'held_out_species_time_RMSE']}}
    table = save_table('chemsindy_reference_trajectories', rows, source, fields, '14 derived simulated trajectories from two published native example mechanisms; 580 observations. Parameter inference conditional on declared known stoichiometry has full-rank sensitivity design. Network graph recovery can remain nonunique; score dynamical equivalence separately. No experimental-identification or broad out-of-family generalization claim.', 'complete derived simulated trajectory with known native-example parameters')
    table['time_points'] = len(points)
    table['distinct_mechanism_families'] = 2
    point_path = OUT / 'chemsindy_reference_timepoints.csv'
    with point_path.open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)
    table['observation_table'] = {'path': relative(point_path), 'rows': len(points), 'counting_unit': 'simulated observation timepoint', 'sha256': digest(point_path)}
    return (source, [table])

def build(skip_transition1x=False):
    OUT.mkdir(parents=True, exist_ok=True)
    sources, tables = ([], [])
    for builder in [build_orgreact, build_chemsindy] + ([] if skip_transition1x else [build_transition1x]):
        source, current = builder()
        sources.append(source)
        tables.extend(current)
    result = {'schema': 'synrxn.coverage-physics.v1', 'sources': sources, 'tables': tables, 'adapter': {'path': relative(__source_file__), 'sha256': digest(__source_file__), 'offline_command': 'python kdd/coverage/adapters/physics.py', 'acquisition_command': 'python kdd/coverage/adapters/physics.py --fetch', 'dependencies': ['numpy', 'scipy', 'h5py', 'requests (acquisition only)']}, 'scope': 'Elementary physical reaction networks; multi-step target-to-stock synthesis planning excluded.', 'reference_policy': 'Observed, computed and locally derived simulated references are distinct. Native data hashes and source member IDs are retained; source-reported physical validation is not claimed independently reproduced.'}
    (OUT / 'catalog.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'tables': [{'id': t['id'], 'rows': t['rows'], 'task_ids': t['task_ids'], 'atomic_configurations': t.get('atomic_configurations')} for t in tables]}, indent=2))
    return result

def load_reactive_configuration(row, configuration_index, include_targets=False):
    """Read one indexed native configuration under an explicit input whitelist.

    Targets are absent by default. Native HDF5 paths are infrastructure fields;
    models receive positions, species and theory through this loader only.
    """
    import h5py
    count = int(row['configuration_count'])
    if not 0 <= configuration_index < count:
        raise IndexError('configuration index outside trajectory')
    with h5py.File(ROOT / row['native_file'], 'r') as handle:
        inputs = {'atomic_numbers': json.loads(row['atomic_numbers']), 'positions_A': handle[row['positions_path']][configuration_index].tolist(), 'theory': row['theory']}
        if not include_targets:
            return inputs
        targets = {'energy_eV': float(handle[row['energies_path']][configuration_index]), 'forces_eV_per_A': handle[row['forces_path']][configuration_index].tolist()}
        return (inputs, targets)
__configure_runtime(globals(), 'kdd/coverage/adapters/physics.py')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch', action='store_true', help='Acquire exact pinned public upstream assets before rebuilding')
    p.add_argument('--skip-transition1x', action='store_true', help='Explicit partial build; never claim reactive-potential coverage')
    args = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        acquire()
    build(skip_transition1x=args.skip_transition1x)
