# Direct implementation; historical source identity is explicit below.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Acquire, preserve and qualify analytical/process reference data.

Run ``python kdd/coverage/adapters/process.py --fetch`` to retrieve pinned
upstream assets, or omit --fetch to rebuild entirely from the local cache.
No original SynRXN tables are changed. Native files, licences, identities and
reference provenance remain separate from generated task views.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/process.py')
import argparse
import gzip
import hashlib
import io
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import urllib.parse
import zipfile
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from synrxn.reconstruction.families.analysis.analytical_interpretation import nmr_pmg_284_spectra
from synrxn.reconstruction.families.analysis.analytical_interpretation.nmr_pmg_284_spectra import (
    integrate_spectrum,
)
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/process'
OUT = ROOT / 'Data/coverage_expansion/process'
PINS = {'scale_up': ('chenzhengyuAI/complex-reaction-system-scale-up', '0080d57a1650ecb349eea2c3dbd5afd89b61cde5'), 'reac_discovery': ('catm542-ai/Reac-Discovery', 'ce917a718c17bcf388a36744aad03b535a408949'), 'magnetstein_reactions': ('BDomzal/magnetstein_x_chemical_reactions', 'a764ab5f3050ec99d8141b8e83f74cf51728f60a'), 'do_mpc': ('do-mpc/do-mpc', '9d5a8df0e4d6acac899bf0a65ffd4c30a6e2e4cc')}
SOURCES = {'FCCSCALEUP2025': {'title': 'Scale-up of complex molecular reaction system by hybrid mechanistic modeling and deep transfer learning', 'url': 'https://doi.org/10.1038/s41467-025-63982-2', 'repository': 'https://github.com/chenzhengyuAI/complex-reaction-system-scale-up', 'archive': 'https://doi.org/10.5281/zenodo.16810762', 'license': 'MIT', 'license_evidence': 'Data/raw/coverage/process/scale_up/LICENSE', 'authors': ['Zhengyu Chen', 'Yongqing Xie', 'Chunming Xu', 'Linzhou Zhang'], 'rights_note': 'Use the separately MIT-licensed author repository; publisher article licence is CC-BY-NC-ND.'}, 'REACDISCOVERY2025': {'title': 'Reac-Discovery: an artificial intelligence–driven platform for continuous-flow catalytic reactor discovery and optimization', 'url': 'https://doi.org/10.1038/s41467-025-64127-1', 'repository': 'https://github.com/catm542-ai/Reac-Discovery', 'archive': 'https://doi.org/10.5281/zenodo.16966777', 'license': 'MIT', 'license_evidence': 'Data/raw/coverage/process/reac_discovery/LICENSE', 'authors': ['Cristopher Tinajero', 'Marcileia Zanatta', 'Julián E. Sánchez-Velandia', 'Eduardo García-Verdugo', 'Victor Sans'], 'rights_note': 'Use the separately MIT-licensed author repository; publisher article licence is CC-BY-NC-ND.'}, 'MAGNETSTEIN2025': {'title': 'NMR Reaction Monitoring Robust to Spectral Distortions', 'url': 'https://doi.org/10.1021/acs.analchem.5c00800', 'repository': 'https://github.com/BDomzal/magnetstein_x_chemical_reactions', 'archive': 'https://doi.org/10.5281/zenodo.14814657', 'license': 'CC-BY-4.0', 'license_evidence': 'Data/raw/coverage/process/magnetstein_reactions/zenodo_record.json', 'rights_note': 'Released views use only CC-BY-4.0 Zenodo spectra and independently derived integration references. Unlicensed author-repository Mnova comparison CSVs are quarantined from released targets/assets; integration-window boundaries are cited factual calibration specifications.'}, 'LOWFIELDNMR2026': {'title': 'Real-time chemical reaction monitoring with 25 MHz low-field NMR', 'url': 'https://doi.org/10.26434/chemrxiv.15005188/v1', 'archive': 'https://zenodo.org/records/20595287', 'license': 'CC-BY-4.0', 'license_evidence': 'Data/raw/coverage/process/lowfield_nmr2026/record.json', 'rights_note': 'Preprint-associated June 2026 experimental data, not a claimed peer-reviewed result.'}, 'DOMPCCSTR': {'title': 'do-mpc nonlinear CSTR control example', 'url': 'https://github.com/do-mpc/do-mpc/tree/9d5a8df0e4d6acac899bf0a65ffd4c30a6e2e4cc/examples/CSTR', 'repository': 'https://github.com/do-mpc/do-mpc', 'license': 'LGPL-3.0-or-later', 'license_evidence': 'Data/raw/coverage/process/do_mpc/LICENSE.txt', 'authors': ['Sergio Lucia', 'Alexandru Tatulea-Codrean', 'TU Dortmund', 'do-mpc contributors'], 'rights_note': 'Source model and generated reference remain an explicit LGPL partition. Derived simulation, not experimental plant data or an optimal-controller demonstration.'}}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def download(url: str, path: Path) -> None:
    import requests
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)

class RemoteZipFile(io.RawIOBase):
    """Read immutable Zenodo ZIP members using HTTP ranges, verifying ZIP CRCs."""

    def __init__(self, url: str, size: int):
        import requests
        self.url, self.size, self.position = (url, size, 0)
        self.session = requests.Session()

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else size
        end = min(self.position + size, self.size) - 1
        if end < self.position:
            return b''
        response = self.session.get(self.url, headers={'Range': f'bytes={self.position}-{end}'}, timeout=120)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - self.position + 1:
            raise ValueError('Remote source did not honour the exact HTTP byte range')
        self.position += len(response.content)
        return response.content

def fetch() -> None:
    """Download exact revision assets and immutable Zenodo record metadata."""
    scale_files = ['LICENSE', 'README.md', 'lab scale model/FeedMoleculeContent_lab.csv', 'lab scale model/ProductMoleculeContent_lab.csv', 'pilot scale model/Bulk_prop.py', 'pilot scale model/FeedMoleculeContent_pilot.csv', 'pilot scale model/ProductBulkProperty_pilot_prop.csv']
    scale_files += [f'pilot scale model/{kind}_{split}_pilot.csv' for kind in ('input', 'targets') for split in ('train', 'validation', 'test')]
    reac_files = ['LICENSE', 'README.md'] + [f'data/Training_Validation_data_model_{model}_{case}.xlsx' for model in ('M1', 'M2') for case in ('CS1', 'CS2')]
    magnet_files = ['README.md', 'config.json', 'src/utils.py']
    for key, files in (('scale_up', scale_files), ('reac_discovery', reac_files), ('magnetstein_reactions', magnet_files), ('do_mpc', ['LICENSE.txt', 'examples/CSTR/template_model.py', 'examples/CSTR/template_mpc.py'])):
        repo, commit = PINS[key]
        for name in files:
            download(f'https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(name)}', RAW / key / name)
    for recid, directory, metadata_name in (('14814657', 'magnetstein_reactions', 'zenodo_record.json'), ('20595287', 'lowfield_nmr2026', 'record.json')):
        path = RAW / directory / metadata_name
        download(f'https://zenodo.org/api/records/{recid}', path)
        record = json.loads(path.read_text())
        if recid == '20595287':
            wanted = {'Results_Reactions_1-4.xlsx', 'Yield_calculations.m', 'PLOTTIME.m'}
            for file in record['files']:
                if file['key'] in wanted:
                    download(file['links']['self'], RAW / directory / file['key'])
        else:
            file = next((f for f in record['files'] if f['key'] == 'data.zip'))
            with zipfile.ZipFile(RemoteZipFile(file['links']['self'], file['size'])) as archive:
                members = []
                for member in archive.infolist():
                    if '_full_mixture_spectrum' in member.filename or not ('preprocessed' in member.filename or 'time_axis' in member.filename):
                        continue
                    target = RAW / directory / member.filename
                    if not target.resolve().is_relative_to((RAW / directory).resolve()):
                        raise ValueError('Unsafe archive member path')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(archive.read(member))
                    members.append({'path': relative(target), 'native_member': member.filename, 'crc32': f'{member.CRC:08x}', 'size': member.file_size, 'sha256': sha(target)})
                write_json(RAW / directory / 'selected_archive_members.json', {'archive_url': file['links']['self'], 'archive_checksum': file['checksum'], 'archive_size': file['size'], 'selection': 'Three region-cropped native trajectories and calibration templates; full-spectrum duplicate views omitted', 'members': members})
    impurity = impurity_adapter()
    if impurity is not None:
        impurity.fetch()

def impurity_adapter():
    """Load the separate impurity source adapter without changing core datasets."""
    import importlib.util
    path = Path(__source_file__).with_name('impurity.py')
    if not path.exists():
        return None
    specification = importlib.util.spec_from_file_location('synrxn_coverage_impurity', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

def artifact(identifier: str, records: list[dict], source_id: str, source_files: list[Path], task_fields: dict, qualification: str, evidence: str, *, status='source_backed_task_view', license=None, independent_groups=None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (identifier + '.jsonl.gz')
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as f:
        for record in records:
            f.write((json.dumps(clean(record), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    assets = [{'path': relative(p), 'sha256': sha(p), 'bytes': p.stat().st_size, 'url': asset_url(p, source_id), 'license': license or SOURCES[source_id]['license'], 'redistribution_allowed': True} for p in source_files]
    return {'id': identifier, 'path': relative(path), 'task_ids': list(task_fields), 'rows': len(records), 'counting_unit': 'reference record; task views can overlap native experiments', 'source_id': source_id, 'source_url': SOURCES[source_id]['url'], 'source_sha256': hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(), 'source_assets': assets, 'license': license or SOURCES[source_id]['license'], 'columns': sorted(set().union(*(r.keys() for r in records))), 'task_fields': task_fields, 'qualification': qualification, 'reference_evidence': evidence, 'readiness': status, 'independent_groups': independent_groups, 'sha256': sha(path)}

def asset_url(path: Path, source_id: str) -> str:
    if source_id in ('FCCSCALEUP2025', 'REACDISCOVERY2025', 'DOMPCCSTR'):
        key = {'FCCSCALEUP2025': 'scale_up', 'REACDISCOVERY2025': 'reac_discovery', 'DOMPCCSTR': 'do_mpc'}[source_id]
        repo, commit = PINS[key]
        return f'https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path.relative_to(RAW / key).as_posix())}'
    if source_id == 'MAGNETSTEIN2025':
        if path.name == 'zenodo_record.json':
            return 'https://zenodo.org/api/records/14814657'
        member = path.relative_to(RAW / 'magnetstein_reactions').as_posix()
        return 'https://zenodo.org/api/records/14814657/files/data.zip/content#member=' + urllib.parse.quote(member)
    if path.name == 'record.json':
        return 'https://zenodo.org/api/records/20595287'
    record = json.loads((RAW / 'lowfield_nmr2026/record.json').read_text())
    return next((file['links']['self'] for file in record['files'] if file['key'] == path.name))

def contract(inputs, targets, groups, metrics, **extra):
    return {'inputs': inputs, 'targets': targets, 'group_fields': groups, 'metrics': metrics, **extra}

def reactor_tables() -> list[dict]:
    tables = []
    for case in ('CS1', 'CS2'):
        path = RAW / 'reac_discovery/data' / f'Training_Validation_data_model_M2_{case}.xlsx'
        frame = pd.read_excel(path, sheet_name='Hoja1').rename(columns={'YIELD': 'Yield'})
        required = ['SA', 'PP', 'FV', 't', 'Qg', 'Ql', 'C', 'T', 'Yield', 'STY']
        assert list(frame.columns) == required or set(frame.columns) == set(required)
        records = []
        for i, row in frame.iterrows():
            assert np.isfinite(row[required].astype(float)).all()
            records.append({'record_id': f'{case}:M2:Hoja1:{i + 2}', 'reaction_system': case, 'native_workbook': path.name, 'native_sheet': 'Hoja1', 'native_excel_row': i + 2, 'reactor_geometry_id': ':'.join((f'{float(row[x]):.12g}' for x in ('SA', 'PP', 'FV', 't'))), **{str(k): float(v) for k, v in row.items()}})
        tables.append(artifact(f'reac_discovery_{case.lower()}', records, 'REACDISCOVERY2025', [path, RAW / 'reac_discovery/LICENSE'], {'reactor_scale_modeling': contract(['reaction_system', 'SA', 'PP', 'FV', 't', 'Qg', 'Ql', 'C', 'T'], ['Yield', 'STY'], ['reaction_system', 'reactor_geometry_id'], ['MAE per endpoint', 'held-out-geometry MAE'])}, 'Experimental reactor geometry and operating conditions predict NMR-derived yield and reported space–time yield. These are fixed reactor-scale experiments, not matched laboratory/pilot scale-up pairs. M1 is an overlapping representation of the same campaign and is deliberately not counted again. Source geometry symbols/units retained; source methods required before cross-source unit conversion.', 'Author repository M2 workbook Hoja1; article Figures4–6 describe experimental yield/STY measurements and M1/M2 geometry views.', independent_groups=len({r['reactor_geometry_id'] for r in records})))
    return tables

def fcc_tables() -> list[dict]:
    tables = []
    base = RAW / 'scale_up'
    for scale, split, xname, yname, label_kind in [('lab', 'source_full', 'lab scale model/FeedMoleculeContent_lab.csv', 'lab scale model/ProductMoleculeContent_lab.csv', 'native_mechanistic_simulation'), *[('pilot', split, f'pilot scale model/input_{split}_pilot.csv', f'pilot scale model/targets_{split}_pilot.csv', 'measured_pilot_experiment' if split == 'test' else 'native_interpolation_of_pilot_experiments') for split in ('train', 'validation', 'test')]]:
        xp, yp = (base / xname, base / yname)
        x, y = (pd.read_csv(xp, header=None).to_numpy(float), pd.read_csv(yp, header=None).to_numpy(float))
        assert len(x) == len(y) and x.shape[1] == 133 and (y.shape[1] == (129 if scale == 'lab' else 10))
        assert np.isfinite(x).all() and np.isfinite(y).all()
        rows = []
        for i, (features, targets) in enumerate(zip(x, y)):
            record = {'record_id': f'{scale}:{split}:native_row_{i + 1}', 'scale': scale, 'native_split': split, 'native_row': i + 1, 'reference_kind': label_kind, 'feedstock_family': 'naphtha1_generated' if scale == 'lab' else 'naphtha2', 'feed_temperature_C': features[0], 'reaction_temperature_C': features[1], 'WHSV_per_hour': features[2], 'catalyst_to_oil_ratio': features[3], 'feed_molecular_composition': features[4:]}
            if scale == 'lab':
                record['product_molecular_composition'] = targets
            else:
                record['product_bulk_properties'] = targets
                record['coke_mass_percent'] = targets[4]
            rows.append(record)
        inputs = ['scale', 'feed_temperature_C', 'reaction_temperature_C', 'WHSV_per_hour', 'catalyst_to_oil_ratio', 'feed_molecular_composition']
        tasks = {'reactor_scale_modeling': contract(inputs, ['product_molecular_composition' if scale == 'lab' else 'product_bulk_properties'], ['feedstock_family', 'scale'], ['MAE per source endpoint'], split_policy='Do not treat interpolation descendants of the same fifteen pilot experiments as independent held-out experimental evidence.')}
        if scale == 'pilot':
            tasks['process_impact_assessment'] = contract(inputs, ['coke_mass_percent'], ['feedstock_family', 'scale'], ['MAE in coke mass percentage points'], scope='Narrow process deposition/material-loss endpoint; not life-cycle impact, cost, energy consumption or hazard assessment.')
        tables.append(artifact(f'fcc_{scale}_{split}', rows, 'FCCSCALEUP2025', [xp, yp, base / 'LICENSE', base / 'README.md', base / 'pilot scale model/Bulk_prop.py'], tasks, f'Reference kind is {label_kind}. All native CSVs are headerless. Lab has10000 simulated compositions; pilot test has15 physical experiments, while train/validation have48/12 interpolated rows. Repeated pilot measurements underpin interpolation and cannot establish independence. Native source-full pilot files have1016/982 rows and are excluded from joining. Lab/pilot target spaces differ (129 molecular components versus10 bulk properties); no artificial matched cross-scale pairing.', 'Author README defines first4 process columns and129 feed components. Native Bulk_prop.py returns dry gas,LPG,gasoline,diesel,coke in that order, then5 gasoline PIONA endpoints. Article methods identify infrared carbon-sulfur analysis of coke; original15 pilot experiments reserved for testing and interpolation used for training/validation.', independent_groups=15 if scale == 'pilot' else None))
    return tables

def load_spectrum(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read only the current spectrum; no other time columns enter task inputs."""
    data = np.loadtxt(ROOT / row['spectrum_path'], usecols=(0, row['spectrum_column']), ndmin=2)
    finite = np.isfinite(data).all(axis=1)
    return (data[finite, 0], data[finite, 1])

def analytical_tables() -> list[dict]:
    base = RAW / 'magnetstein_reactions'
    config = json.loads((base / 'config.json').read_text())
    definitions = [('saccharose_hydrolysis', 'Saccharose hydrolysis', 'preprocessed_mixture.csv'), ('PMG_284', 'PMG 284 monitoring', 'preprocessed_mixture.csv'), ('PMG_287', 'PMG 287 monitoring', 'preprocessed_mixture_cut_to_regions.csv')]
    tables = []
    for native, name, filename in definitions:
        spectrum_path = base / 'data' / native / filename
        if native == 'PMG_284':
            rows = nmr_pmg_284_spectra.build_rows(ROOT)
        else:
            with spectrum_path.open() as f:
                column_count = len(f.readline().split())
            spectrum_count = column_count - 1
            if native == 'saccharose_hydrolysis':
                times = np.loadtxt(base / 'data' / native / 'time_axis_sucrose_hydrolysis.txt')
                assert len(times) == spectrum_count == 1023
            else:
                assert spectrum_count == 1000
                times = np.arange(spectrum_count) * 5.0
            matrix = np.loadtxt(spectrum_path)
            windows = config['integration_intervals'][name]
            components = config['substances_names'][name]
            component_windows = [[w] for w in windows] if native != 'PMG_287' else [windows[:5], windows[5:8], windows[8:]]
            refs = np.empty((spectrum_count, len(components)))
            for column in range(1, spectrum_count + 1):
                refs[column - 1] = integrate_spectrum(matrix[:, 0], matrix[:, column], component_windows)
            assert np.isfinite(refs).all() and np.isfinite(times).all()
            spectrum_hash = sha(spectrum_path)
            rows = []
            for i, (target, time) in enumerate(zip(refs, times)):
                clipped = np.maximum(target, 0)
                if clipped.sum() <= 0:
                    raise ValueError('Native reference has no positive analytical signal')
                rows.append({'record_id': f'{native}:spectrum_column_{i + 1}', 'trajectory_id': native, 'spectrum_path': relative(spectrum_path), 'spectrum_sha256': spectrum_hash, 'spectrum_column': i + 1, 'elapsed_seconds': float(time), 'component_names': components, 'reference_integrals': target, 'reference_area_fractions': clipped / clipped.sum(), 'native_spectrum_column': i + 1, 'missing_signal_bins': int(np.isnan(matrix[:, i + 1]).sum()), 'reference_kind': 'derived_trapezoid_integrals_from_measured_CCBY_spectra; clipped-and-normalised areas are analytical estimates, not absolute concentrations', 'integration_windows_ppm': component_windows})
        tasks = {'analytical_interpretation': contract(['spectrum_path', 'spectrum_column', 'component_names', 'integration_windows_ppm'], ['reference_area_fractions'], ['trajectory_id'], ['MAE relative area per component', 'compositional L1 error'], input_loader='kdd.coverage.adapters.process.load_spectrum; only the specified current native column may be read', scope='Known-component relative analytical signal quantification, not independent absolute concentration or open-world structure elucidation.')}
        assets = [spectrum_path, base / 'zenodo_record.json']
        if native == 'saccharose_hydrolysis':
            assets.append(base / 'data' / native / 'time_axis_sucrose_hydrolysis.txt')
        tables.append(artifact(f'nmr_{native.lower()}_snapshots', rows, 'MAGNETSTEIN2025', assets, tasks, 'Measured NMR signal columns paired with independently computed trapezoidal component-window integrals. Targets are explicitly DERIVED analytical estimates from the same spectra, not independent concentration ground truth. Negative component integrals clipped only for derived area fractions; signed integrals preserved. NaNs are excluded within each integration window, never filled as measured zeros. Future-extracted component templates are forbidden inputs. SourcePMG284 contains999 native snapshots versus1000 nativeMnova rows, and noMnova labels are used. Three underlying measured trajectories; no random-snapshot independence claim. Numerical integration is the primary deterministic baseline; this task view is a reproducible analytical-processing contract, not evidence of a difficult concentration-estimation benchmark.', 'CC-BY4.0 Zenodo14814657 native cropped spectral matrices. Factual component/window calibration specifications are documented in official author config.json;5-second hydrosilylation sampling in src/utils.py and measured sucrose timestamps in the CC-BYtime-axis asset. Independently implemented numpy.trapezoid integration; no copied reference estimates or unpublished calibration.', status='derived_analytical_reference_from_licensed_measurements', independent_groups=1))
        forecasts = []
        for current, following in zip(rows[:-1], rows[1:]):
            forecasts.append({'record_id': current['record_id'] + ':next', 'trajectory_id': native, **{k: current[k] for k in ('spectrum_path', 'spectrum_column', 'elapsed_seconds', 'component_names', 'integration_windows_ppm')}, 'target_elapsed_seconds': following['elapsed_seconds'], 'next_reference_area_fractions': following['reference_area_fractions'], 'target_native_spectrum_column': following['spectrum_column']})
        tables.append(artifact(f'nmr_{native.lower()}_progress', forecasts, 'MAGNETSTEIN2025', assets, {'reaction_progress_monitoring': contract(['spectrum_path', 'spectrum_column', 'elapsed_seconds', 'target_elapsed_seconds', 'component_names', 'integration_windows_ppm'], ['next_reference_area_fractions'], ['trajectory_id'], ['next-step MAE relative area', 'error versus current-state persistence'], temporal_policy='Input loader exposes only current spectrum. Target spectrum, next references, full trajectories and future-derived component templates are forbidden inputs.')}, 'Forecasting view of the same native measured trajectories; overlaps analytical snapshot records and is not an additional physical experiment. Native time ordering preserved. Target is next measured-spectrum DERIVED relative integrated area, not a new independent chemical label.', 'Current native spectrum-column index paired with subsequent independently derived spectral area reference; source mapping and time evidence identical to the snapshot table.', status='derived_analytical_reference_from_licensed_measurements', independent_groups=1))
    return tables

def lowfield_tables() -> list[dict]:
    base = RAW / 'lowfield_nmr2026'
    path = base / 'Results_Reactions_1-4.xlsx'
    frame = pd.read_excel(path, sheet_name='Reactions', header=None)
    data = frame.iloc[2:, :5].astype(float).to_numpy()
    assert data.shape == (60, 5) and np.isfinite(data).all()
    rows = []
    for reaction in range(1, 5):
        for i in range(59):
            rows.append({'record_id': f'reaction{reaction}:minute{int(data[i, 0])}:next', 'trajectory_id': f'reaction{reaction}', 'native_sheet': 'Reactions', 'native_excel_row': i + 3, 'elapsed_minutes': data[i, 0], 'observed_yield_percent': data[i, reaction], 'target_elapsed_minutes': data[i + 1, 0], 'next_yield_percent': data[i + 1, reaction], 'reference_kind': 'reported_lowfield_NMR_yield_time_profile'})
    return [artifact('lowfield_nmr_yield_progress', rows, 'LOWFIELDNMR2026', [path, base / 'Yield_calculations.m', base / 'record.json'], {'reaction_progress_monitoring': contract(['trajectory_id', 'elapsed_minutes', 'observed_yield_percent', 'target_elapsed_minutes'], ['next_yield_percent'], ['trajectory_id'], ['next-step MAE yield percentage points', 'MAE versus persistence'], temporal_policy='Only the current measured yield and earlier measurements are allowed. This view does not require raw FIDs or claim raw-spectrum interpretation.')}, 'Four measured yield profiles,60 observations per reaction,236 adjacent forecast pairs. Measurements from the same reaction remain one group. Processed yield observations, not raw-FID-to-yield calibration. Native MATLAB plotting script labels x-axis Time[min] and y-axis Yield[%]. Raw FIDs76GB are indexed upstream but were not downloaded because no verified trajectory-specific rawFID association was required for this narrower view.', 'Source workbookReactions:A3:E62 and authorYield_calculations.m. Overview worksheet separately provides terminal lowfield/1H-Bruker/13C-Bruker comparisons for four reactions.', independent_groups=4)]

def cstr_rhs(_time, state, action, alpha=1.0, beta=1.0):
    """NumPy transcription of the pinned do-mpc CSTR equations (LGPL3+).

    Copyright2014–2019SergioLucia,AlexandruTatulea-Codrean,TUDortmund.
    Adapted2026forSciPy integration; original code/licence retained in rawcache.
    """
    ca, cb, tr, tk = state
    flow, heat = action
    k1 = beta * 1287000000000.0 * np.exp(-9758.3 / (tr + 273.15))
    k2 = 1287000000000.0 * np.exp(-9758.3 / (tr + 273.15))
    k3 = 9043000000.0 * np.exp(-alpha * 8560.0 / (tr + 273.15))
    delta = tr - tk
    return np.array([flow * (5.1 - ca) - k1 * ca - k3 * ca ** 2, -flow * cb + k1 * ca - k2 * cb, -(k1 * ca * 4.2 + k2 * cb * -11.0 + k3 * ca ** 2 * -41.85) / (0.9342 * 3.01) + flow * (130.0 - tr) - 4032.0 * 0.215 * delta / (0.9342 * 3.01 * 10.01), (heat + 4032.0 * 0.215 * delta) / (5.0 * 2.0)])

def cstr_step(state, action, duration_hours=0.005, alpha=1.0, beta=1.0):
    action = np.asarray(action, float)
    if action.shape != (2,) or not 5 <= action[0] <= 100 or (not -8500 <= action[1] <= 0):
        raise ValueError('Action outside native CSTR bounds')
    solution = solve_ivp(lambda t, x: cstr_rhs(t, x, action, alpha, beta), [0, duration_hours], state, method='DOP853', rtol=1e-10, atol=1e-12)
    if not solution.success or not np.isfinite(solution.y[:, -1]).all():
        raise ValueError('CSTR integration failed')
    return solution.y[:, -1]

def control_table() -> list[dict]:
    """Produce action-conditioned transitions; evaluation uses cstr_step oracle."""
    rng = np.random.default_rng(20261020)
    rows = []
    for episode in range(12):
        alpha, beta = [(1.0, 1.0), (0.95, 0.9), (1.05, 1.1)][episode % 3]
        state = np.array([0.8, 0.5, 130.0, 125.0]) + rng.uniform([-0.1, -0.05, -3, -3], [0.1, 0.05, 3, 3])
        for index in range(50):
            action = np.array([rng.uniform(10, 80), rng.uniform(-8000, -500)])
            next_state = cstr_step(state, action, alpha=alpha, beta=beta)
            rows.append({'record_id': f'episode{episode}:step{index}', 'trajectory_id': f'episode{episode}', 'step_index': index, 'time_hours': index * 0.005, 'duration_hours': 0.005, 'state': state, 'action': action, 'alpha': alpha, 'beta': beta, 'next_state': next_state, 'setpoint_C_b_mol_per_L': 0.6, 'tracking_squared_error': float((next_state[1] - 0.6) ** 2), 'reactor_temperature_excess_C': max(float(next_state[2]) - 140, 0), 'reference_kind': 'derived_ODE_simulation_from_pinned_published_CSTR_model', 'seed': 20261020})
            state = next_state
    base = RAW / 'do_mpc'
    return [artifact('cstr_control_transitions', rows, 'DOMPCCSTR', [base / 'examples/CSTR/template_model.py', base / 'examples/CSTR/template_mpc.py', base / 'LICENSE.txt'], {'process_control': contract(['state', 'duration_hours', 'alpha', 'beta', 'setpoint_C_b_mol_per_L'], ['action'], ['trajectory_id'], ['closed-loop integrated squared C_b tracking error', 'temperature-bound violation', 'control effort and action-bound violations'], action_space={'F_per_hour': [5, 100], 'Q_dot_kJ_per_hour': [-8500, 0]}, evaluation_oracle='kdd.coverage.adapters.process.cstr_step', policy='A policy chooses action from current state; cstr_step produces next state. Recorded excitation action is not an optimal label; supervised action imitation MAE is not a valid control-performance score.', auxiliary_system_identification={'inputs': ['state', 'action', 'duration_hours', 'alpha', 'beta'], 'targets': ['next_state']})}, 'Derived simulation:12 seeded excitation episodes,50 transitions each,not600 experimental reactor runs. Native model usesA→B→C plus competing2A→D. Source equations, kinetics, heat-transfer constants, action bounds,0.005h sampling andC_b=0.6 objective retained. Integrator changed to deterministicSciPyDOP853; dynamic control evaluates a proposed policy with the explicit transition oracle. No claim of matching native robustMPC or proving plant safety.', 'Pinned do-mpc examples/CSTR/template_model.py andtemplate_mpc.py; original LGPL notice retained. Script seed20261020, alpha/beta scenarios1/1,0.95/0.9,1.05/1.1.', status='derived_simulator_task_contract', independent_groups=12)]

def build() -> dict:
    tables = reactor_tables() + fcc_tables() + analytical_tables() + lowfield_tables() + control_table()
    sources = []
    for source_id, source in SOURCES.items():
        entry = {'id': source_id, **source}
        entry['used_by_tables'] = [t['id'] for t in tables if t['source_id'] == source_id]
        for key, (repo, commit) in PINS.items():
            if source.get('repository') == 'https://github.com/' + repo:
                entry['commit'] = commit
        evidence = ROOT / source['license_evidence']
        if evidence.suffix == '.json':
            metadata = json.loads(evidence.read_text())
            entry['authors'] = [creator['name'] for creator in metadata.get('metadata', {}).get('creators', [])]
        entry['redistribution_allowed'] = True
        entry['license_evidence_sha256'] = sha(evidence)
        entry['license_evidence_url'] = asset_url(evidence, source_id)
        entry['license_url'] = 'https://creativecommons.org/licenses/by/4.0/' if source['license'] == 'CC-BY-4.0' else 'https://opensource.org/license/mit' if source['license'] == 'MIT' else 'https://www.gnu.org/licenses/lgpl-3.0.html'
        receipt = RAW / 'magnetstein_reactions/selected_archive_members.json'
        if source_id == 'MAGNETSTEIN2025' and receipt.exists():
            entry['acquisition_receipt'] = relative(receipt)
            entry['acquisition_receipt_sha256'] = sha(receipt)
        sources.append(entry)
    catalog = {'schema': 'synrxn.coverage-process.v1', 'adapter': relative(Path(__source_file__).resolve()), 'adapter_sha256': sha(Path(__source_file__).resolve()), 'tables': tables, 'sources': sources, 'task_ids': sorted(set().union(*(t['task_ids'] for t in tables))), 'qualification_notes': ['Task views do not establish independent benchmark splits or completed model results.', 'NMR snapshot/progress views overlap the same3 native measured trajectories; their labels are independently derived analytical integrals, not independent concentration ground truth.', 'Pilot interpolation descendants overlap the15 original experiments; cannot split them as independent experimental evidence.', 'Process impact currently covers measured coke deposition only; full energy,cost,life-cycle and safety coverage remains unfilled.', 'Control evidence is a published-model simulator and declared derived transitions; policy evaluation remains future model work.'], 'excluded_assets': [{'source_id': 'FCCSCALEUP2025', 'assets': ['pilot scale model/FeedMoleculeContent_pilot.csv', 'pilot scale model/ProductBulkProperty_pilot_prop.csv'], 'reason': 'Nativefullinput1016rows versus target982rows; noassumed positionaljoin.'}, {'source_id': 'REACDISCOVERY2025', 'assets': ['M1workbooks'], 'reason': 'Overlapping representations ofM2physicalcampaigns; not additionalindependentexperiments.'}, {'source_id': 'MAGNETSTEIN2025', 'assets': ['future-extractedcomponenttemplates', 'full-spectrumduplicateviews'], 'reason': 'Avoidfuture-template leakage and doublecounting.'}]}
    catalog['excluded_assets'].append({'source_id': 'MAGNETSTEIN2025', 'assets': ['author-repositoryMnovaCSVreferences'], 'reason': 'Reference redistribution grant unspecified; quarantine unlicensed native comparison estimates. Released task labels are independently derived from explicitlyCCBYspectra.'})
    catalog['excluded_assets'].append({'source_id': 'DLRN2025', 'assets': ['Data/raw/coverage/process/dlrn'], 'reason': 'Academic-only customlicence restricts redistribution; not used in any task table or released sourceasset.'})
    impurity = impurity_adapter()
    if impurity is not None and (impurity.RAW / 'SS_rxns_80806.json.gz').exists() and (impurity.RAW / 'github-code-repository.zip').exists():
        impurity.build()
    notice = OUT / 'ATTRIBUTION.txt'
    lines = ['SynRXN analytical/process coverage: source attribution and changes', '', 'Original source assets and their licence notices remain preserved in the paths listed below.', 'No DLRN academic-only assets, unlicensed Mnova comparison estimates or publisher CC-BY-NC-ND files are redistributed by these task views.', '']
    for source in sources:
        lines.extend([f"[{source['id']}] {source['title']}", 'Creators/authors: ' + '; '.join(source.get('authors', [])), 'Publication/source: ' + source['url'], 'Data archive: ' + source.get('archive', source.get('repository', source['url'])), 'Licence: ' + source['license'] + ' — ' + source['license_url'], 'Licence evidence: ' + source['license_evidence'] + ' (SHA-256 ' + source['license_evidence_sha256'] + ')', 'Changes: converted native tables to JSONL; retained native IDs, units and reference kinds; supplied task input/group/metric contracts.'])
        if source['id'] == 'MAGNETSTEIN2025':
            lines.append('Additional changes: independent trapezoidal integration gives explicitly derived relative analytical-area references; measured NaNs are omitted, not replaced as measured zeros; no independent concentration truth is claimed.')
        if source['id'] == 'LOWFIELDNMR2026':
            lines.append('Additional changes: adjacent next-yield forecast pairs use current measured yield only; four original reaction profiles remain distinct groups.')
        if source['id'] == 'FCCSCALEUP2025':
            lines.append('Additional changes: headerless CSVs read without discarding row1; physical pilot experiments, native simulations and interpolation descendants are separate; no mismatched source-full pilot join.')
        if source['id'] == 'DOMPCCSTR':
            lines.append('LGPL code changes: the pinned CSTR equations are transcribed in cstr_rhs with NumPy and integrated with SciPy DOP853; seeded excitation trajectories are derived simulations. Original model source, LGPL notice and attribution remain retained. The adapter carries SPDX LGPL-3.0-or-later.')
        evidence = ROOT / source['license_evidence']
        if evidence.suffix != '.json':
            lines.extend(['Preserved native licence/copyright notice:', evidence.read_text().rstrip()])
        lines.append('')
    notice.write_text('\n'.join(lines) + '\n')
    catalog['attribution_path'] = relative(notice)
    catalog['attribution_sha256'] = sha(notice)
    write_json(OUT / 'catalog.json', catalog)
    return catalog
__configure_runtime(globals(), 'kdd/coverage/adapters/process.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Retrieve exact pinned source assets before rebuilding')
    args = parser.parse_args()
    if args.fetch:
        fetch()
    result = build()
    print(json.dumps({'tables': len(result['tables']), 'rows': sum((t['rows'] for t in result['tables'])), 'tasks': result['task_ids']}, indent=2))
