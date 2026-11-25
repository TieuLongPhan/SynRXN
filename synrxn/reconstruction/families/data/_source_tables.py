# Direct implementation; historical source identity is explicit below.
"""Rebuild literature, curation, retrieval, inverse-component and scope views.

No predictions are relabelled as observations. Controlled annotation deletions
are explicitly generated corruptions of independent expert references. ORD
component design is a retrospective, measured finite-oracle variant. Minot
scope discovery retains source hit labels, not global chemical novelty claims.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/knowledge.py')
import argparse
from collections import defaultdict
from functools import lru_cache
import csv
import hashlib
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import sys
import tarfile
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__source_file__).resolve().parents[3]))
from synrxn.reconstruction.coverage.common import REPO, asset, contract, emit, encoded, identity, record, sha

def infons(element):
    return {item.attrib['key']: item.text or '' for item in element.findall('infon')}

def parse_document(document):
    passages, entities, relations = ([], [], [])
    for passage in document.findall('passage'):
        passages.append({'offset': int(passage.findtext('offset', '0')), 'type': infons(passage).get('type'), 'text': passage.findtext('text', '')})
        for annotation in passage.findall('annotation'):
            info = infons(annotation)
            entities.append({'id': annotation.attrib['id'], 'type': info.get('type'), 'identifier': info.get('identifier'), 'text': annotation.findtext('text', ''), 'locations': [{'offset': int(item.attrib['offset']), 'length': int(item.attrib['length'])} for item in annotation.findall('location')]})
        for relation in passage.findall('relation'):
            relations.append({'id': relation.attrib['id'], 'type': infons(relation).get('type'), 'nodes': [{'refid': node.attrib['refid'], 'role': node.attrib.get('role', '')} for node in relation.findall('node')]})
    if len({item['id'] for item in entities}) != len(entities):
        raise ValueError('Repeated annotation identifiers within document')
    return {'document_id': document.findtext('id'), 'passages': passages, 'entities': entities, 'relations': relations}

def literature_tables(output: Path, repo: Path):
    metadata = json.loads((repo / 'kdd/integration/sources/C24.json').read_text())
    declared = next((item for item in metadata['files'] if item['name'] == 'EnzChemRED.tar.gz'))
    raw = asset(repo / 'Data/raw/synrxn2/C24/EnzChemRED.tar.gz', declared['url'], repo=repo, expected_bytes=declared['bytes'], checksum=declared['checksum'])
    grant_path = repo / 'kdd/integration/sources/C24.json'
    source = {'id': 'EnzChemRED2024', 'title': 'EnzChemRED expert-annotated enzyme chemistry corpus', 'url': 'https://doi.org/10.5281/zenodo.11067998', 'paper_url': metadata['paper_urls'][0], 'license': 'CC-BY-4.0', 'assets': [raw], 'redistribution_allowed': True, 'rights': {'grant_url': 'https://zenodo.org/records/11067998', 'grant_snapshot': grant_path.relative_to(repo).as_posix(), 'grant_snapshot_sha256': sha(grant_path), 'evidence': 'Official data record declares cc-by-4.0 for the annotated dataset; retain PMID attribution and native text provenance.'}, 'changes': 'Parse BioC XML; preserve native spans and relations; generated deletion controls and query/gallery links are explicitly derived.'}
    rows, curation, documents = ([], [], [])
    pairs = defaultdict(lambda: {'documents': set(), 'names': set(), 'locators': []})
    with tarfile.open(repo / raw['path'], 'r:gz') as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.name.endswith('.xml'):
                continue
            for doc in ET.fromstring(archive.extractfile(member).read()).findall('document'):
                parsed = parse_document(doc)
                doc_id = parsed['document_id']
                documents.append(doc_id)
                provenance = {'source_sha256': raw['sha256'], 'source_locator': member.name, 'document_id': doc_id, 'source_url': raw['url']}
                inputs = {'passages': parsed['passages']}
                gold = {'entities': parsed['entities'], 'relations': parsed['relations']}
                rows.append(record(source['id'], 'documents', doc_id, 'PMID:' + doc_id, inputs=inputs, targets=gold, provenance=provenance))
                by_id = {item['id']: item for item in parsed['entities']}
                for relation in parsed['relations']:
                    if relation['type'] != 'Conversion':
                        continue
                    chemicals = [by_id[node['refid']] for node in relation['nodes'] if node['refid'] in by_id and by_id[node['refid']]['type'] == 'Chemical' and (by_id[node['refid']].get('identifier', '') not in {'', '-'})]
                    chemical_ids = tuple(sorted({item['identifier'] for item in chemicals}))
                    if len(chemical_ids) != 2:
                        continue
                    pairs[chemical_ids]['documents'].add(doc_id)
                    pairs[chemical_ids]['names'].update((item['text'] for item in chemicals))
                    pairs[chemical_ids]['locators'].append({'member': member.name, 'relation_id': relation['id']})
                for kind in ['clean', 'controlled_relation_deletion']:
                    corrupted = json.loads(encoded(gold))
                    if kind != 'clean':
                        if not corrupted['relations']:
                            continue
                        index = int(identity(doc_id)[:8], 16) % len(corrupted['relations'])
                        corrupted['relations'].pop(index)
                    curation.append(record(source['id'], 'curation', doc_id + ':' + kind, 'PMID:' + doc_id, inputs={'passages': parsed['passages'], 'candidate_record': corrupted}, targets={'needs_correction': corrupted != gold, 'corrected_record': gold}, provenance=provenance, context={'corruption': kind, 'corruption_evidence': 'controlled generated deletion, not observed curator error'}))
    if len(documents) != len(set(documents)):
        raise ValueError('Duplicate source documents')
    tables = [emit(output, 'documents', rows, source=source, repo=repo, unit='expert-annotated source document', tasks={'literature_extraction': contract(['passages'], ['entities', 'relations'], ['entity_F1', 'relation_F1', 'document_coverage'])}, qualification='Native expert BioC entities/conversions from title/abstract text. Not full-paper multimodal extraction; offsets retain native passage coordinates.', evidence='expert_curated_text_annotations'), emit(output, 'curation', curation, source=source, repo=repo, unit='controlled record-curation case', tasks={'reaction_record_curation': contract(['passages', 'candidate_record'], ['needs_correction', 'corrected_record'], ['correction_F1', 'false_edit_rate', 'coverage'])}, qualification='Clean controls and deterministic single-relation deletions reuse independent expert annotations. This is a controlled corruption variant, not a dataset of naturally observed database errors. Keep both cases of each document together.', evidence='expert_reference_with_generated_corruption')]
    retrieval = []
    for chemical_ids, item in sorted(pairs.items()):
        retrieval.append(record(source['id'], 'retrieval', encoded(chemical_ids), 'CHEBI_PAIR:' + encoded(chemical_ids), inputs={'chemical_identifiers': list(chemical_ids), 'chemical_names': sorted(item['names'])}, targets={'relevant_document_ids': sorted(item['documents'])}, provenance={'source_sha256': raw['sha256'], 'source_url': raw['url'], 'source_locator': item['locators']}, context={'gallery_table': source['id'] + '/documents', 'gallery_inputs': ['document_id', 'passages'], 'relevance': 'expert-annotated conversion evidence for this unordered chemical-ID pair'}))
    tables.append(emit(output, 'retrieval', retrieval, source=source, repo=repo, unit='annotated chemical-conversion precedent query', tasks={'reaction_retrieval': contract(['chemical_identifiers', 'chemical_names'], ['relevant_document_ids'], ['recall_at_k', 'nDCG', 'query_coverage'])}, qualification='Query-to-literature-precedent retrieval with expert linked conversion references, not graded mechanistic similarity. Gallery exposes only document IDs/text; gold entities and relation labels are withheld. Unannotated documents are unknown relevance, not confirmed negatives. Document text can be shared across query splits; no unseen-paper claim.', evidence='expert_annotated_conversion_document_links'))
    return (source, tables)

def component_design(output: Path, repo: Path):
    import rdkit.Chem as Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    raw_manifest = json.loads((repo / 'kdd/coverage/adapters/knowledge_sources.json').read_text())['catdrx_manifest']
    item = next((item for item in raw_manifest['assets'] if item['path'].endswith('dataset/ord.csv')))
    path = repo / item['path']
    if sha(path) != item['sha256']:
        raise ValueError('CatDRX ORD source changed')
    raw = asset(path, item['url'], repo=repo, expected_bytes=item['bytes'])
    ord_audit = json.loads((repo / 'kdd/integration/sources/C03.json').read_text())
    license_path = repo / 'Data/raw/synrxn2/notes/C03/LICENSE'
    source = {'id': 'CatDRX_ORD2025', 'title': "Reaction-conditioned inverse component references from CatDRX's ORD derivative", 'url': 'https://github.com/ohuelab/CatDRX/tree/' + raw_manifest['commit'], 'paper_url': 'https://doi.org/10.1038/s42004-025-01732-7', 'license': 'CC-BY-SA-4.0', 'assets': [raw], 'redistribution_allowed': True, 'rights': {'grant_url': 'https://github.com/open-reaction-database/ord-data/blob/' + ord_audit['github']['commit'] + '/LICENSE', 'grant_snapshot': license_path.relative_to(repo).as_posix(), 'grant_snapshot_sha256': sha(license_path), 'evidence': 'Retain upstream ORD data CC BY-SA4.0, distinct from CatDRX MIT software license. Only the ORD-derived CSV is admitted; other third-party task CSVs are excluded pending their dataset grants.'}, 'changes': 'Generate target-isolated inverse component views; withhold catalyst structure, mass, identifiers and target-derived fields; retain source ORD IDs and measured responses.'}
    rows, excluded = ([], defaultdict(int))

    @lru_cache(maxsize=200000)
    def canonical(text):
        mol = Chem.MolFromSmiles(text)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None
    with path.open(newline='', encoding='utf8') as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            catalyst = canonical(row['Catalyst_SMILES_new'])
            try:
                response = float(row['Yield'])
            except (ValueError, TypeError):
                excluded['missing_response'] += 1
                continue
            if catalyst is None or not math.isfinite(response) or (not 0 <= response <= 100):
                excluded['invalid_structure_or_response'] += 1
                continue
            catalyst_components = set(catalyst.split('.'))

            def strip_target(text):
                normalized = canonical(text or '')
                if normalized is None:
                    return None
                return '.'.join((part for part in normalized.split('.') if part not in catalyst_components))
            reactants, product, reagents = [strip_target(row[key]) for key in ['Reactant_SMILES_new', 'Product_SMILES_new', 'Reagent_SMILES_new']]
            if not reactants or not product or reagents is None:
                excluded['unqualified_endpoint_or_context'] += 1
                continue
            inputs = {'reactants': reactants, 'products': product, 'reagents': reagents, 'desired_response_pct_interval': [min(90, int(response // 10) * 10), min(100, int(response // 10) * 10 + 10)]}
            try:
                reaction_time = float(row['Time_h'])
                if math.isfinite(reaction_time):
                    inputs['time_h'] = reaction_time
            except (ValueError, TypeError):
                pass
            rows.append(record(source['id'], 'inverse_components', str(ordinal), 'CATALYST:' + catalyst, inputs=inputs, targets={'catalyst_smiles': catalyst, 'observed_response_pct': response}, provenance={'source_sha256': raw['sha256'], 'source_url': raw['url'], 'source_locator': 'dataset/ord.csv:row:' + str(ordinal), 'ord_reaction_id': row['index']}, context={'oracle_scope': 'measured source candidates only; unmeasured designs have no outcome label', 'native_dataset': 'ORD derivative in CatDRX', 'source_row': ordinal}))
    table = emit(output, 'inverse_components', rows, source=source, repo=repo, unit='recorded component-performance reference', tasks={'reaction_component_design': contract(['reactants', 'products', 'reagents', 'desired_response_pct_interval', 'time_h'], ['catalyst_smiles', 'observed_response_pct'], ['component_recovery_top_k', 'structure_validity', 'training_relative_novelty', 'observed_oracle_response'])}, qualification='Retrospective reaction-conditioned component generation/recovery with catalyst-identity holdout. Input does not list candidates; evaluation has a finite measured lookup oracle. No prospective de novo performance is inferred for off-dataset structures. Yield measurements, native candidate identities and upstream ORD reuse remain explicit.', evidence='source_reported_measurement_and_recorded_component_structure')
    table['excluded_rows'] = dict(excluded)
    return (source, [table])

def scope_discovery(output: Path, repo: Path):
    import pyarrow.parquet as pq
    audit = json.loads((repo / 'kdd/integration/sources/C02.json').read_text())
    expansion = json.loads((repo / 'Data/expansion/catalog.json').read_text())
    tables = [item for item in expansion['tables'] if item['source_id'] == 'C02' and 'reaction_success' in item.get('task_fields', {})]
    raw_assets = {item['name']: item for table in tables for item in table['source_assets']}
    source = {'id': 'MinotScope2026', 'title': 'Native adaptive borylation substrate-scope campaigns', 'url': 'https://doi.org/10.5281/zenodo.20773622', 'paper_url': audit['paper_urls'][0], 'license': 'CC-BY-4.0', 'redistribution_allowed': True, 'rights': {'grant_url': 'https://zenodo.org/records/20773622', 'grant_snapshot': 'kdd/integration/sources/C02.json', 'grant_snapshot_sha256': sha(repo / 'kdd/integration/sources/C02.json'), 'evidence': 'Official data-record CC BY4.0 grant applies to the native measured campaign CSVs.'}, 'assets': [asset(repo / 'Data/raw/synrxn2/C02' / item['name'], item['url'], repo=repo, expected_bytes=item['bytes_observed']) for item in raw_assets.values()], 'changes': 'Reuse measured source campaigns with explicit dates/rounds as retrospective substrate-scope discovery; no new experiments or chemical novelty labels.'}
    rows = []
    for table in tables:
        path = repo / 'Data/expansion' / table['path']
        if sha(path) != table['sha256']:
            raise ValueError('Captured Minot adapter changed')
        for native in pq.read_table(path).to_pylist():
            fields, targets = (json.loads(native['inputs_json']), json.loads(native['targets_json']))
            original = json.loads(native['native_json'])
            if targets.get('binary') not in [0, 1] or 'sum_mono_bo_conv' not in targets:
                continue
            rows.append(record(source['id'], 'scope_campaigns', native['record_id'], native['parent_id'], inputs=fields, targets={'native_hit_label': targets['binary'], 'conversion_fraction': targets['sum_mono_bo_conv']}, provenance={'source_sha256': table['source_assets'][0]['sha256'], 'source_locator': native['record_id'], 'source_url': table['source_assets'][0]['url'], 'native_experiment_id': original.get('rxn_id')}, context={'campaign': table['key'], 'recorded_date': original.get('rxn_date'), 'discovery_scope': 'source-specific substrate-scope hits; not globally novel transformations', 'source_binary_threshold': 'native label, not inferred'}, native_split=native['split']))
    table = emit(output, 'scope_campaigns', rows, source=source, repo=repo, unit='attempted substrate-scope experiment', tasks={'new_reactivity_discovery': contract(sorted({key for row in rows for key in json.loads(row['inputs_json'])}), ['native_hit_label', 'conversion_fraction'], ['source_hit_recovery_vs_budget', 'substrate_scope_coverage', 'failure_accounting'])}, qualification='Bounded retrospective substrate-scope discovery variant from native adaptive rounds, preserving failed attempts and dates. Source success labels and conversion are references; no claim of new transformations, exhaustive product characterization or prospective discovery. These same observations already support other Minot tasks and are not additional independent experiments.', evidence='reported_experimental_hit_labels_and_conversion')
    return (source, [table])

def build(output: Path, repo: Path=REPO):
    output.mkdir(parents=True, exist_ok=True)
    sources, tables = ([], [])
    for builder in [literature_tables, component_design, scope_discovery]:
        source, additions = builder(output, repo)
        sources.append(source)
        tables += additions
    for source in sources:
        notice = output / (source['id'] + '-NOTICE.txt')
        notice.write_text(source['title'] + '\n' + source['url'] + '\nData license: ' + source['license'] + '\nChanges: ' + source['changes'] + '\n' + source['rights']['evidence'] + '\n')
        source['notice'] = {'path': notice.relative_to(repo).as_posix(), 'sha256': sha(notice)}
    for table in tables:
        table['release_eligible'] = True
    catalog = {'schema': 'synrxn.coverage-branch.v1', 'adapter': 'kdd/coverage/adapters/knowledge.py', 'adapter_sha256': sha(Path(__source_file__)), 'source_lock': 'kdd/coverage/adapters/knowledge_sources.json', 'source_lock_sha256': sha(repo / 'kdd/coverage/adapters/knowledge_sources.json'), 'sources': sources, 'tables': tables, 'excluded_sources': [{'source': 'SEISMiQ simulated_impurities.zip', 'reason': '2,541 prior generated candidates for five queries are not experimentally observed impurities; retained as reference-only acquisition, not a primary impurity benchmark.'}, {'source': 'SEISMiQ aggregated spectral test sets', 'reason': 'Underlying aggregated spectral source terms differ; not admitted based solely on wrapper deposit license.'}, {'source': 'ElectroDiscovery2022 ACS supplement', 'reason': "Dataset record CC-BY-NC4.0 is outside the user's unrestricted redistribution release policy."}, {'source': 'CatDRX non-ORD task CSVs', 'reason': 'MIT software license alone is insufficient to clear all third-party dataset grants; only ORD-derived data retain their verified upstream ShareAlike license.'}]}
    (output / 'catalog.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'tables': len(tables), 'rows': {item['id']: item['rows'] for item in tables}}))
    return catalog

def fetch(repo: Path=REPO):
    """Fetch only pinned inputs; refuse changed bytes rather than silently repin."""
    from urllib.request import urlopen
    import shutil
    lock = json.loads((repo / 'kdd/coverage/adapters/knowledge_sources.json').read_text())
    for item in lock['assets']:
        path = repo / item['path']
        if path.exists() and path.stat().st_size == item['bytes'] and (sha(path) == item['sha256']):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        with urlopen(item['url'], timeout=180) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response, out)
        if temporary.stat().st_size != item['bytes'] or sha(temporary) != item['sha256']:
            raise ValueError('Upstream bytes changed: ' + item['url'])
        temporary.replace(path)
__configure_runtime(globals(), 'kdd/coverage/adapters/knowledge.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'Data/coverage_expansion/knowledge')
    parser.add_argument('--fetch', action='store_true', help='Fetch missing exact pinned native assets before rebuilding')
    args = parser.parse_args()
    if args.fetch:
        fetch()
    build(args.output)
