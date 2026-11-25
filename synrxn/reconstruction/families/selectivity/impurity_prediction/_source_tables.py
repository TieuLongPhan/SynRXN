# Direct implementation; historical source identity is explicit below.
"""Acquire licensed observed impurity references and build narrow task variants.

No inferred stoichiometric by-products or generated impurity candidates are used.
The organic variant predicts the profile of a source-identified side product;
the inorganic variant predicts explicitly reported impurity-phase formulas.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/impurity.py')
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import zipfile
import numpy as np
import pandas as pd
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/impurity'
OUT = ROOT / 'Data/coverage_expansion/impurity'
PBA_MEMBER = 'robowski-maps-main/summary_of_reaction_yield_data/PBA/Dataset_PBA_composition.csv'
PBA_README = 'robowski-maps-main/summary_of_reaction_yield_data/README.md'
PBA_LICENSE = 'robowski-maps-main/LICENSE'
FIGSHARE = 'https://api.figshare.com/v2/articles/30423274/versions/1'
ZENODO = 'https://zenodo.org/api/records/14880579'
SOURCES = {'ROBOWSKI_IMPURITY2025': {'id': 'ROBOWSKI_IMPURITY2025', 'title': 'Robot-assisted mapping of chemical reaction hyperspaces and networks', 'url': 'https://doi.org/10.1038/s41586-025-09490-1', 'archive': 'https://doi.org/10.5281/zenodo.14880579', 'license': 'MIT', 'license_url': 'https://opensource.org/license/mit', 'license_evidence': 'Data/raw/coverage/impurity/14880579_metadata.json', 'license_evidence_url': ZENODO, 'qualification': 'Separately MIT-licensed author data deposit; publisher article CC-BY-NC-ND assets are not redistributed. Organic assay references are source spectral-unmixing estimates from measured UV–Vis spectra.'}, 'SOLIDSTATE_IMPURITY2025': {'id': 'SOLIDSTATE_IMPURITY2025', 'title': 'Text-mined dataset of solid-state syntheses with impurity phases using Large Language Model', 'url': 'https://doi.org/10.1038/s41597-025-06222-y', 'archive': 'https://doi.org/10.6084/m9.figshare.30423274.v1', 'license': 'CC-BY-4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/', 'license_evidence': 'Data/raw/coverage/impurity/30423274_metadata.json', 'license_evidence_url': FIGSHARE, 'qualification': 'Literature-reported experimental impurity phases extracted with an LLM and parsed by source authors. Labels are extracted observations, not new experiments or predicted thermodynamic phases. Per-record original publication DOI retained.'}}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def rel(path):
    return Path(path).relative_to(ROOT).as_posix()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value

def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for url, name in [(FIGSHARE, '30423274_metadata.json'), (ZENODO, '14880579_metadata.json')]:
        path = RAW / name
        if not path.exists():
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
    for name, url in [('SS_rxns_80806.json.gz', 'https://ndownloader.figshare.com/files/58973674'), ('github-code-repository.zip', 'https://zenodo.org/api/records/14880579/files/github-code-repository.zip/content')]:
        path = RAW / name
        if not path.exists():
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            path.write_bytes(response.content)
    extract()

def extract():
    with zipfile.ZipFile(RAW / 'github-code-repository.zip') as archive:
        for name in (PBA_MEMBER, PBA_README, PBA_LICENSE):
            target = RAW / 'extracted' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError('Cached ZIP member differs from pinned archive: ' + name)
            else:
                target.write_bytes(data)

def source_assets(source_id, files):
    assets = []
    for path in files:
        if path.name.endswith('_metadata.json'):
            url = SOURCES[source_id]['license_evidence_url']
        elif path.name == 'SS_rxns_80806.json.gz':
            url = 'https://ndownloader.figshare.com/files/58973674'
        else:
            import urllib.parse
            member = path.relative_to(RAW / 'extracted').as_posix()
            url = 'https://zenodo.org/api/records/14880579/files/github-code-repository.zip/content#member=' + urllib.parse.quote(member)
        assets.append({'path': rel(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'url': url, 'license': SOURCES[source_id]['license'], 'redistribution_allowed': True})
    return assets

def table(identifier, rows, source_id, files, task_contract, qualification, evidence):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (identifier + '.jsonl.gz')
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as file:
        for row in rows:
            file.write((json.dumps(clean(row), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    assets = source_assets(source_id, files)
    return {'id': identifier, 'path': rel(path), 'rows': len(rows), 'task_ids': ['impurity_prediction'], 'source_id': source_id, 'source_url': SOURCES[source_id]['url'], 'license': SOURCES[source_id]['license'], 'source_assets': assets, 'source_sha256': hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(), 'sha256': sha(path), 'columns': sorted(set().union(*(row.keys() for row in rows))), 'task_fields': {'impurity_prediction': task_contract}, 'counting_unit': 'experimental reaction condition' if source_id == 'ROBOWSKI_IMPURITY2025' else 'literature-extracted synthesis recipe with explicit impurity phase identity', 'qualification': qualification, 'reference_evidence': evidence, 'readiness': 'source_backed_task_view'}

def organic():
    path = RAW / 'extracted' / PBA_MEMBER
    frame = pd.read_csv(path)
    metals = ['Mn', 'Fe', 'Co', 'Ni', 'Cu', 'M1-Fe', 'M1-Co']
    assert len(frame) == 756 and frame['Index'].is_unique
    np.testing.assert_allclose(frame[metals[:5]].sum(axis=1), 1, atol=1e-08)
    np.testing.assert_allclose(frame[metals[5:]].sum(axis=1), 1, atol=1e-08)
    rows = []
    for _, row in frame.iterrows():
        assert np.isfinite(row[metals + ['Yield_of_benzaldehyde (%)']].to_numpy(float)).all()
        fractions = {metal: float(row[metal]) for metal in metals}
        rows.append({'record_id': f"PBA:native_index_{int(row['Index'])}", 'native_index': int(row['Index']), 'reaction_system': 'PBA_catalysed_styrene_epoxidation', 'reactant_smiles': 'C=Cc1ccccc1', 'intended_product_smiles': 'c1ccc(C2CO2)cc1', 'candidate_side_product_smiles': ['O=Cc1ccccc1'], 'source_product_codes': {'reactant': '20a', 'intended_product': '20c', 'side_product': '20b'}, 'catalyst_metal_fractions': fractions, 'catalyst_composition_group': ';'.join((f'{metal}={fractions[metal]:.10g}' for metal in metals)), 'temperature_C': 72.0, 'duration_hours': 6.0, 'fixed_context': 'Styrene/tBuOOH in acetonitrile with CTAB aqueous phase and aqueous PBA suspension; fixed protocol specified by source Methods.', 'side_product_yield_percent': {'O=Cc1ccccc1': float(row['Yield_of_benzaldehyde (%)'])}, 'reference_kind': 'native_UVVis_spectral_unmixing_of_experimental_crude_mixture', 'reference_units': 'percent yield relative to least abundant reactant, as supplied'})
    return table('pba_benzaldehyde_side_product', rows, 'ROBOWSKI_IMPURITY2025', [path, RAW / 'extracted' / PBA_README, RAW / 'extracted' / PBA_LICENSE, RAW / '14880579_metadata.json'], {'inputs': ['reactant_smiles', 'intended_product_smiles', 'candidate_side_product_smiles', 'catalyst_metal_fractions', 'temperature_C', 'duration_hours', 'fixed_context'], 'targets': ['side_product_yield_percent'], 'group_fields': ['reaction_system', 'catalyst_composition_group'], 'metrics': ['MAE benzaldehyde yield percentage points', 'RMSE benzaldehyde yield percentage points'], 'variant': 'closed-candidate experimental side-product profile', 'scope': 'One source-identified benzaldehyde side product; not an open-world impurity structure-discovery benchmark.'}, '756 native experimental conditions; benzaldehyde is explicitly identified by the source as the main undesired by-product of styrene epoxidation. The task predicts its measured-spectrum-derived yield for a known candidate. Native fitted zero values are preserved as quantitative estimates and are not relabelled absence/not-detected. Desired-product yield and selectivity columns are excluded from model inputs. No generated candidates or interpolated viewer tables are used. One chemical transformation with a catalyst-composition panel; grouping must prevent repeated compositions crossing splits.', "Primary paper Figure6c–d and the 'Five-dimensional compositional space' section explicitly distinguish desired styrene oxide from benzaldehyde20b, the main by-product. Methods specify72°C/6h and UV–Vis spectral unmixing. Native summary README defines units and source CSV contains756 indexed catalyst-composition assays.")

def concrete_formula(entity):
    if not isinstance(entity, dict) or entity.get('amounts_vars') or entity.get('elements_vars'):
        return None
    formula = entity.get('material_formula')
    if not isinstance(formula, str) or not formula or formula.lower() in {'unknown', 'none', 'null', 'impure', 'pure'}:
        return None
    composition = entity.get('composition')
    if not isinstance(composition, list) or not composition or (not all((c.get('elements') for c in composition))):
        return None
    return formula

def conditions(operations):
    """Whitelist numeric operations; omit free-text subjects/results and MP fields."""
    result = []
    for operation in operations or []:
        if not isinstance(operation, dict):
            continue
        value = {'act_type': operation.get('act_type'), 'temperature': [], 'duration': []}
        for native, target in [('temp_values', 'temperature'), ('time_values', 'duration')]:
            for measurement in operation.get(native, []):
                if isinstance(measurement, dict):
                    value[target].append({k: measurement[k] for k in ('min', 'max', 'values', 'units') if k in measurement})
        if value['temperature'] or value['duration']:
            result.append(value)
    return result

def inorganic():
    path = RAW / 'SS_rxns_80806.json.gz'
    with gzip.open(path, 'rt') as file:
        native = json.load(file)
    assert len(native) == 80806
    audit = Counter(total_native_records=len(native))
    rows = []
    for index, record in enumerate(native):
        phases = record.get('impurity_phase', [])
        if not phases:
            audit['no_reported_impurity_phase_excluded_not_negative'] += 1
            continue
        audit['native_reported_impurity_phase_records'] += 1
        known_phases = [concrete_formula(p) for p in phases]
        identified = sorted(set((p for p in known_phases if p)))
        if not identified:
            audit['no_concrete_impurity_identity_excluded'] += 1
            continue
        if not record.get('target_reaction'):
            audit['unbalanced_or_missing_target_reaction_excluded'] += 1
            continue
        target = [concrete_formula(v) for v in record.get('target', [])]
        precursors = [concrete_formula(v) for v in record.get('precursors', [])]
        if not target or not all(target) or (not precursors) or (not all(precursors)):
            audit['unresolved_target_or_precursor_identity_excluded'] += 1
            continue
        if set(target) & set(identified):
            audit['target_labelled_as_impurity_excluded'] += 1
            continue
        doi = record.get('DOI')
        if not isinstance(doi, str) or not doi.startswith('10.'):
            audit['missing_original_doi_excluded'] += 1
            continue
        ops = conditions(record.get('conditions_forDOI'))
        audit['eligible_records'] += 1
        audit['eligible_with_numeric_condition_context'] += bool(ops)
        partial = len(identified) < len(phases)
        audit['eligible_partial_identity_sets'] += partial
        signature = '|'.join(sorted(precursors)) + '>>' + '|'.join(sorted(target))
        rows.append({'record_id': f'SS80806:native_index_{index}', 'native_index': index, 'source_publication_doi': doi, 'precursor_formulas': sorted(precursors), 'intended_target_formulas': sorted(target), 'synthesis_condition_context': ops, 'condition_linkage': 'DOI-level extracted context; not a guaranteed unique recipe-to-operation assignment', 'reaction_identity_group': hashlib.sha256(signature.encode()).hexdigest(), 'reported_impurity_formulas': identified, 'label_completeness': 'partial_known_subset' if partial else 'all_native_extracted_identities_concrete; not an exhaustive chemical absence claim', 'reference_kind': 'literature_reported_experimental_impurity_phases; LLM_extracted_and_source_parsed', 'domain': 'inorganic_solid_state', 'target_identity_kind': 'phase_composition_formula; not crystal structure or molecular SMILES'})
    audit['eligible_unique_original_dois'] = len({r['source_publication_doi'] for r in rows})
    audit['eligible_unique_impurity_formulas'] = len({f for r in rows for f in r['reported_impurity_formulas']})
    json_write(OUT / 'solid_state_qualification.json', dict(audit))
    result = table('solid_state_reported_impurity_phases', rows, 'SOLIDSTATE_IMPURITY2025', [path, RAW / '30423274_metadata.json'], {'inputs': ['precursor_formulas', 'intended_target_formulas', 'synthesis_condition_context'], 'targets': ['reported_impurity_formulas'], 'group_fields': ['source_publication_doi', 'reaction_identity_group'], 'metrics': ['recall@k of reported concrete impurity formulas', 'multilabel recall over observed identities', 'exact reported-set match for complete-native-identity subset'], 'variant': 'literature-extracted inorganic impurity-phase identity prediction', 'scope': 'Positive reported-phase identities; no unreported impurity is treated as absent. Formula labels do not establish measured polymorph/crystal structure or organic impurity identity.', 'forbidden_inputs': ['impurity_phase', 'impurity_reaction', 'reported_impurity_formulas', 'mp_id', 'mp_eabovehull', 'free-text result subjects']}, 'Selects source records with an explicitly identified impurity phase, balanced target reaction, concrete precursor and target compositions and original DOI. Native unknown/variable identities are excluded; partial known subsets are flagged. The61937 no-impurity-report cohort is never used as negative evidence. Source labels are extracted literature observations, not computationally generated candidate phases. Materials Project most-stable-entry assignments/energies are omitted from all features and references. Native synthesis operations are DOI-level context and may not map uniquely to a particular recipe, so exact condition-specific causal claims are unsupported. Original publication and reaction identity must be grouped in eventual benchmark splits.', 'Source Data Records explicitly distinguish intended target from impurity_phase entities and retain original publication DOI. Primary Technical Validation describes source extraction, balancing and thermodynamic validation. Figsharev1 grantsCC-BY4.0 for the native80806-record structured data file.')
    result['qualification_audit'] = rel(OUT / 'solid_state_qualification.json')
    result['qualification_audit_sha256'] = sha(OUT / 'solid_state_qualification.json')
    return result

def build():
    extract()
    tables = [organic(), inorganic()]
    sources = []
    for source in SOURCES.values():
        value = dict(source)
        value['license_evidence_sha256'] = sha(ROOT / value['license_evidence'])
        value['used_by_tables'] = [t['id'] for t in tables if t['source_id'] == source['id']]
        metadata = json.loads((ROOT / value['license_evidence']).read_text())
        value['authors'] = [a['full_name'] for a in metadata['authors']] if source['id'] == 'SOLIDSTATE_IMPURITY2025' else [a['name'] for a in metadata['metadata']['creators']]
        value['required_attribution'] = [value['url'], value['archive'], value['license_url']]
        value['redistribution_allowed'] = True
        if source['id'] == 'ROBOWSKI_IMPURITY2025':
            value['license_notice'] = rel(RAW / 'extracted' / PBA_LICENSE)
            value['license_notice_sha256'] = sha(RAW / 'extracted' / PBA_LICENSE)
        sources.append(value)
    catalog = {'schema': 'synrxn.coverage-impurity.v1', 'adapter': rel(Path(__source_file__).resolve()), 'adapter_sha256': sha(Path(__source_file__).resolve()), 'tables': tables, 'sources': sources, 'task_ids': ['impurity_prediction'], 'scope_note': 'Two real-source narrow variants: known organic side-product profile and reported inorganic impurity-phase identity. They do not establish broad open-world organic impurity discovery or completed benchmark results.'}
    notice = OUT / 'ATTRIBUTION.txt'
    lines = ['SynRXN impurity coverage: original attribution and changes', '']
    for source in sources:
        lines.extend([f"[{source['id']}] {source['title']}", 'Original creators: ' + '; '.join(source['authors']), 'Publication: ' + source['url'], 'Data: ' + source['archive'], 'Licence: ' + source['license'] + ' — ' + source['license_url'], 'Licence grant evidence: ' + source['license_evidence'] + ' (SHA-256 ' + source['license_evidence_sha256'] + ')'])
        if source['id'] == 'ROBOWSKI_IMPURITY2025':
            lines.append('Changes: selected756 native PBA assays; preserved source benzaldehyde side-product yield estimates; supplied stated chemical identities and fixed protocol context; excluded desired-product outcome columns from model inputs. This is a closed-candidate profile task, not generated impurity truth.')
            lines.extend(['Preserved original MIT notice:', (ROOT / source['license_notice']).read_text().rstrip()])
        else:
            lines.append('Changes: selected7261 native records with explicit concrete impurity identities, balanced target reactions and resolved precursors/targets; excluded unknown/no-report identities as negatives; retained original experimental publication DOIs and partial-label flags; exposed only whitelisted numerical operation context; omitted Materials Project assignments and free-text results. Labels remain source literature extractions, not new independently validated measurements.')
        lines.append('')
    notice.write_text('\n'.join(lines) + '\n')
    catalog['attribution_path'] = rel(notice)
    catalog['attribution_sha256'] = sha(notice)
    json_write(OUT / 'catalog.json', catalog)
    return catalog
__configure_runtime(globals(), 'kdd/coverage/adapters/impurity.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch()
    result = build()
    print(json.dumps({'tables': [{'id': t['id'], 'rows': t['rows']} for t in result['tables']], 'task_ids': result['task_ids']}, indent=2))
