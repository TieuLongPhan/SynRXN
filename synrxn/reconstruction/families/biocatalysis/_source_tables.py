# Direct implementation; historical source identity is explicit below.
"""Build provenance-preserving biocatalysis views from pinned public assets.

Run ``python kdd/coverage/adapters/bio.py --fetch`` on a new checkout, or omit
``--fetch`` to use verified local assets. No model training or synthetic negative
label generation occurs. The upstream pickle contains only builtin containers;
the restricted loader refuses every class/global rather than executing code.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/adapters/bio.py')
import argparse
import ast
from collections import defaultdict
import hashlib
import io
import json
import math
from synrxn.reconstruction.paths import HistoricalPath as Path
import pickle
import re
import tarfile
import urllib.request
import zipfile
import numpy as np
import pandas as pd
ROOT = Path(__source_file__).resolve().parents[3]
RAW = ROOT / 'Data/raw/coverage/bio'
OUT = ROOT / 'Data/coverage_expansion/bio'
LOCK = Path(__source_file__).with_name('bio_sources.json')
AA = re.compile('^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def split_group(key: str) -> str:
    """Deterministic *derived* 80/10/10 grouping, never an upstream split claim."""
    bucket = int(digest(key)[:8], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'

def close_split_components(frame: pd.DataFrame, fields: list[str], namespace: str):
    """Keep every exact identity in the declared grouping fields in one split."""
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    keys = []
    for values in zip(*(frame[field].astype(str) for field in fields)):
        tokens = [field + ':' + value for field, value in zip(fields, values)]
        keys.append(tokens[0])
        representative = root(tokens[0])
        for token in tokens[1:]:
            other = root(token)
            if other != representative:
                lo, hi = sorted([representative, other])
                parent[hi] = lo
                representative = lo
    members = defaultdict(list)
    for key in parent:
        members[root(key)].append(key)
    signatures = {key: digest('\n'.join(sorted(values))) for key, values in members.items()}
    frame['split_component'] = [signatures[root(key)] for key in keys]
    frame['derived_split'] = frame['split_component'].map(lambda value: split_group(namespace + ':' + value))
    return frame

def clean(value) -> str:
    value = str(value).strip()
    return '' if value.lower() in {'nan', 'none', '?', 'na', 'n/a', ''} else value

def sequence(value) -> str:
    s = re.sub('\\s+', '', clean(value)).upper().rstrip('*')
    return s if s and AA.fullmatch(s) else ''

def fasta_sequences(text: str):
    current, lines = (None, [])
    for line in text.splitlines():
        if line.startswith('>'):
            if current is not None:
                yield (current, sequence(''.join(lines)))
            current, lines = (line[1:].split('|')[1], [])
        elif line.strip():
            lines.append(line.strip())
    if current is not None:
        yield (current, sequence(''.join(lines)))

class BuiltinsOnlyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Class loading refused: {module}.{name}')

def read_native_pairs(path: Path):
    with zipfile.ZipFile(path) as outer:
        if outer.testzip() is not None:
            raise ValueError('ReactZyme ZIP CRC failure')
        for name in outer.namelist():
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                payload = [n for n in inner.namelist() if n.endswith('/data.pkl')]
                if len(payload) != 1:
                    raise ValueError('Unexpected native PyTorch container')
                pairs = BuiltinsOnlyUnpickler(io.BytesIO(inner.read(payload[0]))).load()
            split = 'test' if 'positive_test_' in name else 'train_validation'
            for native_id, pair in pairs.items():
                if not isinstance(pair, tuple) or len(pair) != 2 or (not all((isinstance(x, str) for x in pair))):
                    raise ValueError('Unexpected ReactZyme record')
                yield (name, native_id, split, pair[0], pair[1])

def decode_sites(site_labels: str, site_types: str, length: int):
    """Retain native 1-based inclusive intervals and create explicit residue labels.

    The type mapping and assignment order reproduce the upstream dataloader:
    0=unannotated, 1=binding, 2=catalytic, 3=other annotated site. Unannotated
    residues are not guaranteed experimentally established negative sites.
    """
    labels, types = (ast.literal_eval(site_labels), ast.literal_eval(site_types))
    if len(labels) != len(types):
        raise ValueError('Site/type length mismatch')
    output = [0] * length
    for interval, kind in zip(labels, types):
        if kind not in {0, 1, 2} or len(interval) not in {1, 2}:
            raise ValueError('Unexpected native site annotation')
        lo, hi = (interval[0], interval[-1])
        if not 1 <= lo <= hi <= length:
            raise ValueError('Site index outside sequence')
        output[lo - 1:hi] = [kind + 1] * (hi - lo + 1)
    return output

def verify_assets(lock: dict, fetch: bool=False):
    for asset in lock['assets']:
        path = ROOT / asset['path']
        valid = path.is_file() and path.stat().st_size == asset['bytes'] and (sha256(path) == asset['sha256'])
        if not valid and fetch:
            if asset.get('local_frozen_bundle'):
                raise ValueError(f"Restore the project frozen M-CSA bundle pinned by {asset['source_lock']}; the live API is not byte-identical to {path}")
            if asset.get('metadata_snapshot'):
                record = next((v for v in lock['zenodo_records'].values() if str(v['id']) in asset['url']))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + '.part')
                with urllib.request.urlopen(asset['url'], timeout=180) as response, tmp.open('wb') as output:
                    while (block := response.read(1 << 20)):
                        output.write(block)
                tmp.replace(path)
            valid = path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256']
        if not valid:
            raise ValueError(f'Missing/changed pinned source: {path}; use --fetch')

def asset_entry(lock: dict, basename: str):
    return next((a for a in lock['assets'] if Path(a['path']).name == basename))

def contract(inputs, targets, groups, metrics):
    return {'inputs': inputs, 'targets': targets, 'group_fields': groups, 'metrics': metrics}

def write_table(frame: pd.DataFrame, key: str, tasks: dict, source_id: str, source_asset: dict, license_note: str, qualification: str, evidence: str, release_eligible: bool=True, raw_assets_redistribution_allowed: bool=False, **extra):
    if frame.empty or frame['source_record_id'].duplicated().any():
        raise ValueError(f'Empty table or duplicate source IDs: {key}')
    if 'source_sha256' not in frame:
        frame['source_sha256'] = source_asset['sha256']
    for task in tasks.values():
        fields = task['inputs'] + task['targets'] + task['group_fields']
        if set(fields) - set(frame.columns):
            raise ValueError(f'Missing task fields in {key}: {set(fields) - set(frame.columns)}')
        if set(task['inputs']) & set(task['targets']):
            raise ValueError('Target leakage in task contract')
    directory = OUT if release_eligible else OUT / 'quarantine'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{key}.parquet'
    frame.to_parquet(path, index=False, compression='zstd')
    qualified_source_id = source_id
    source_ids = source_id.split('+') if '+' in source_id else [source_id.split(':', 1)[0]]
    return {'id': key, 'path': str(path.relative_to(ROOT)), 'task_ids': list(tasks), 'rows': len(frame), 'counting_unit': 'source-derived qualified task row', 'source_id': source_ids[0], 'source_ids': source_ids, 'qualified_source_id': qualified_source_id, 'source_url': source_asset['url'], 'source_sha256': source_asset['sha256'], 'license': license_note, 'columns': list(frame.columns), 'task_fields': tasks, 'qualification': qualification, 'reference_evidence': evidence, 'sha256': sha256(path), 'release_eligible': release_eligible, 'source_asset_path': source_asset['path'], 'raw_assets_redistribution_allowed': raw_assets_redistribution_allowed, 'redistribution_status': 'redistributable_with_attribution_and_preserved_notices' if release_eligible else 'quarantined_not_for_redistribution', **extra}

def build_retrieval(lock: dict):
    up = pd.read_csv(RAW / 'C14--cleaned_uniprot_rhea.tsv', sep='\t', dtype=str, keep_default_na=False)
    rhea = pd.read_csv(RAW / 'C14--rhea_molecules.tsv', sep='\t', dtype=str, keep_default_na=False)
    seq_ids = defaultdict(set)
    for row in up.to_dict('records'):
        seq_ids[sequence(row['Sequence'])].add(row['Entry'])
    records = []
    for member, native_id, native_split, molecules, seq in read_native_pairs(RAW / 'C14--enzyme_smi_split.zip'):
        seq = sequence(seq)
        if not seq or not molecules:
            raise ValueError('Invalid native positive pair')
        records.append({'source_record_id': f'C14:{member}:{native_id}', 'source_member': member, 'native_record_id': str(native_id), 'enzyme_sequence': seq, 'uniprot_ids_json': json.dumps(sorted(seq_ids.get(seq, []))), 'reaction_molecule_smiles': molecules, 'positive_reference': 1, 'sequence_group': digest(seq), 'reaction_group': digest(molecules), 'native_split': native_split, 'split_role': 'upstream_sequence-smi_partition'})
    df = pd.DataFrame(records)
    c = contract(['enzyme_sequence'], ['reaction_molecule_smiles'], ['sequence_group', 'reaction_group', 'native_split'], ['Recall@k', 'MRR', 'query-macro enrichment'])
    table = write_table(df, 'reactzyme_retrieval', {'enzyme_retrieval': c}, 'C14', asset_entry(lock, 'C14--enzyme_smi_split.zip'), 'MIT AND CC-BY-4.0; ReactZyme dataset MIT with UniProt/Rhea attribution retained', 'Native positive pairs only; no manufactured negatives. Native reaction input concatenates substrate/product molecule SMILES without a reaction arrow; preserve this upstream representation. Original train+validation pool is not silently relabeled as training. No new homology audit or model evaluation claimed.', 'Curated UniProt/Rhea positive enzyme-reaction associations; native sequence-smi split from ReactZyme release.', raw_assets_redistribution_allowed=True, native_split_counts=df['native_split'].value_counts().to_dict(), unique_sequence_groups=df['sequence_group'].nunique(), unique_reaction_groups=df['reaction_group'].nunique(), license_expression='MIT AND CC-BY-4.0', license_chain=[{'component': 'ReactZyme dataset', 'license': 'MIT', 'evidence': 'https://zenodo.org/api/records/13635807'}, {'component': 'UniProt sequences and associations', 'license': 'CC-BY-4.0', 'evidence': 'https://rest.uniprot.org/help/license'}, {'component': 'Rhea reactions', 'license': 'CC-BY-4.0', 'evidence': 'https://sparql.rhea-db.org/.well-known/void'}])
    return (table, up, rhea)

def build_active_sites(lock: dict, up: pd.DataFrame, rhea: pd.DataFrame):
    source = asset_entry(lock, 'C23--dataset--ec_site_dataset--uniprot_ecreact_cluster_split_merge_dataset_limit_100--test_dataset_with_similarity_idx.csv')
    sites = pd.read_csv(ROOT / source['path'], dtype=str, keep_default_na=False)
    up_by_id = {r['Entry']: (i, r) for i, r in enumerate(up.to_dict('records'))}
    rhea_by_id = {r['Rhea ID']: (i, r) for i, r in enumerate(rhea.to_dict('records'))}
    rows, rejected = ([], defaultdict(int))
    for i, row in enumerate(sites.to_dict('records')):
        identifier, seq = (row['alphafolddb-id'], sequence(row['aa_sequence']))
        match = up_by_id.get(identifier)
        if not match or sequence(match[1]['Sequence']) != seq:
            rejected['missing_or_nonidentical_uniprot_sequence'] += 1
            continue
        if row['is_valid'].lower() != 'true' or sequence(row['aa_sequence_calculated']) != seq:
            rejected['invalid_or_structure_sequence_mismatch'] += 1
            continue
        try:
            target = decode_sites(row['site_labels'], row['site_types'], len(seq))
        except ValueError:
            rejected['invalid_site_indices'] += 1
            continue
        for rid in sorted(set(match[1]['Rhea ID'].split(';'))):
            reaction = rhea_by_id.get(rid)
            if not reaction:
                continue
            rindex, rrow = reaction
            rows.append({'source_record_id': f'C23:test:{i}:C14:{rid}', 'native_record_id': str(i), 'uniprot_id': identifier, 'enzyme_sequence': seq, 'reaction_smiles': rrow['substrate'] + '>>' + rrow['product'], 'rhea_id': rid, 'site_intervals_native_json': row['site_labels'], 'site_types_native_json': row['site_types'], 'residue_labels_json': json.dumps(target, separators=(',', ':')), 'sequence_group': digest(seq), 'native_cluster_40': row['cluster_ther_0.4'], 'native_cluster_80': row['cluster_ther_0.8'], 'reaction_group': rid, 'native_split': 'test', 'split_role': 'EasIFA_native_test_only_joined_Rhea_reference', 'pairing_evidence': 'exact_UniProt_accession_and_sequence_Rhea_annotation', 'site_evidence': 'native_EasIFA_UniProt_site_reference; not reaction-specific experimental labeling', 'uniprot_source_record_id': f'C14:cleaned_uniprot_rhea.tsv:{match[0]}', 'rhea_source_record_id': f'C14:rhea_molecules.tsv:{rindex}', 'uniprot_asset_sha256': asset_entry(lock, 'C14--cleaned_uniprot_rhea.tsv')['sha256'], 'rhea_asset_sha256': asset_entry(lock, 'C14--rhea_molecules.tsv')['sha256']})
    df = pd.DataFrame(rows)
    return write_table(df, 'easifa_rhea_active_sites', {'active_sites': contract(['enzyme_sequence', 'reaction_smiles'], ['residue_labels_json'], ['sequence_group', 'native_cluster_40', 'native_cluster_80', 'reaction_group', 'native_split'], ['protein-macro residue AUPRC', 'F1', 'MCC', 'type-specific recall'])}, 'C23+C14', source, 'EasIFA code MIT; source-data redistribution grant unresolved. UniProt/Rhea attribution retained. Local analysis only until data terms are cleared.', 'Qualified exact-ID/exact-sequence join. Only native EasIFA test proteins available here; training and structure coordinates not acquired. Repeated reactions share the same protein-level site reference and must remain grouped. Reference zeros mean unannotated, not verified nonfunctional residues. Binding/catalytic/other site types stay distinct; this is reference recovery, not newly measured reaction-specific sites.', 'Native EasIFA UniProt site labels plus separate curated UniProt-Rhea positive association; 1-based inclusive intervals converted using the pinned upstream dataloader.', release_eligible=False, native_source_rows=len(sites), unique_proteins=df['uniprot_id'].nunique(), labeled_residues_unique_proteins=int(sum((sum((v != 0 for v in json.loads(x))) for x in df.drop_duplicates('uniprot_id')['residue_labels_json']))), rejection_counts={**dict(rejected), 'no_qualified_rhea_reference': int(len(sites) - sum(rejected.values()) - df['native_record_id'].nunique())}, additional_source_assets=[asset_entry(lock, 'C14--cleaned_uniprot_rhea.tsv'), asset_entry(lock, 'C14--rhea_molecules.tsv')])

def build_kinetics(lock: dict):
    tables = []
    for parameter, target in [('kcat', 'log10kcat_max'), ('km', 'log10km_mean'), ('ki', 'log10ki_mean')]:
        source = asset_entry(lock, f'C15--datasets--splits--{parameter}-random_trainvaltest.csv')
        native = pd.read_csv(ROOT / source['path'], dtype=str, keep_default_na=False)
        df = pd.DataFrame({'source_record_id': [f'C15:{parameter}:csvrow:{i}' for i in range(len(native))], 'native_record_id': native['Unnamed: 0'], 'native_inner_record_id': native['Unnamed: 0.1'], 'enzyme_sequence': native['sequence'].map(sequence), 'uniprot_id': native['uniprot'], 'sequence_annotation_source': native['sequence_source'], 'substrate_smiles': native['reactant_smiles' if parameter == 'kcat' else 'substrate_smiles'], 'parameter': parameter, 'target_log10_native_aggregate': pd.to_numeric(native[target], errors='coerce'), 'native_target_column': target, 'native_measurement_value': pd.to_numeric(native['value'], errors='coerce'), 'native_measurement_log10': pd.to_numeric(native['log10_value'], errors='coerce'), 'temperature_native': pd.to_numeric(native['temperature'], errors='coerce'), 'ph': pd.to_numeric(native['ph'], errors='coerce'), 'ec': native['ec'], 'taxonomy_id': native['taxonomy_id'], 'native_pair_group': native['group'], 'native_cluster_40': native['sequence_40cluster'], 'native_cluster_80': native['sequence_80cluster'], 'sequence_group': native['sequence'].map(digest), 'native_split': 'combined_train_validation_test_release', 'split_role': 'derived_sequence40_cluster_partition; native_combined_pool_preserved'})
        df['derived_split'] = df['native_cluster_40'].map(lambda x: split_group(parameter + ':cluster40:' + x))
        mask = df['enzyme_sequence'].ne('') & df['substrate_smiles'].ne('') & np.isfinite(df['target_log10_native_aggregate'])
        df = df.loc[mask].copy()
        df['unit_qualification'] = 'native log10 aggregate; intended kcat s^-1 / Km,Ki mM; upstream SABIO rate conversion concern unresolved; no conversion applied here'
        tables.append(write_table(df, f'catpred_{parameter}', {'enzyme_kinetic_parameters': contract(['enzyme_sequence', 'substrate_smiles', 'parameter'], ['target_log10_native_aggregate'], ['sequence_group', 'native_pair_group', 'native_cluster_40', 'native_cluster_80', 'derived_split'], ['parameter-specific log10 MAE', 'RMSE', 'R2', 'uncertainty calibration'])}, 'C15', source, 'Zenodo14775076 dataset MIT; pinned repository code MIT. BRENDA/SABIO-RK upstream rights require separate redistribution review; local analysis retained.', 'kcat, Km and Ki are separate endpoint tables. Native max/mean aggregate target retained alongside the nonaggregated row value. pH/temperature describe individual source measurements and are metadata, not inputs to a pooled target. Native combined pool has a deterministic derived 40%-cluster split, not a reproduction of original random split. Homology clusters are native annotations and have not been independently rerun. Intended source units are kcat s^-1 and Km/Ki mM; pinned SABIO processing code multiplies minute^-1/hour^-1 values by60/3600, raising an upstream normalization concern. Per-measurement unit lineage is absent from released tables, so no correction or physical-unit accuracy claim is made here.', 'Published CatPred-DB aggregate kinetic targets compiled from BRENDA/SABIO-RK; sequence-source annotations, native row IDs and homology clusters retained.', release_eligible=False, native_source_rows=len(native), excluded_rows=int((~mask).sum()), unit_audit_evidence=asset_entry(lock, 'C15--scripts--processing--process_sabio.py'), unique_pair_groups=df['native_pair_group'].nunique(), native_target_column=target, derived_split_counts=df['derived_split'].value_counts().to_dict()))
    return tables

def build_brenda_kinetics(lock: dict):
    """Release only an identifiable BRENDA record, never a cross-source aggregate.

    The pinned BRENDA processor assigns sequence_source='brenda'; SABIO assigns
    only 'sabio' or 'uniprot_search'. The merge preserves the first source row's
    value/log10_value while adding max/mean columns potentially mixing SABIO.
    Consequently filtering the retained BRENDA record is valid for its original
    value; filtering alone would NOT clear the aggregate max/mean target.
    """
    tables = []
    for parameter in ['kcat', 'km', 'ki']:
        source = asset_entry(lock, f'C15--datasets--splits--{parameter}-random_trainvaltest.csv')
        native = pd.read_csv(ROOT / source['path'], dtype=str, keep_default_na=False)
        records = []
        for i, row in enumerate(native.to_dict('records')):
            seq = sequence(row['sequence'])
            if row['sequence_source'] != 'brenda' or not seq:
                continue
            try:
                value, target = (float(row['value']), float(row['log10_value']))
            except ValueError:
                continue
            if not value > 0 or not math.isfinite(target) or (not math.isclose(math.log10(value), target, abs_tol=1e-09)):
                continue
            substrate = clean(row['reactant_smiles'] if parameter == 'kcat' else row['substrate_smiles'])
            if not substrate:
                continue
            records.append({'source_record_id': f'C15:{parameter}:csvrow:{i}', 'native_record_id': row['Unnamed: 0'], 'native_inner_record_id': row['Unnamed: 0.1'], 'enzyme_sequence': seq, 'uniprot_id': row['uniprot'], 'substrate_smiles': substrate, 'parameter': parameter, 'target_log10_brenda_record': target, 'native_brenda_record_value': value, 'intended_native_unit': 's^-1' if parameter == 'kcat' else 'mM', 'temperature_native': clean(row['temperature']), 'ph': clean(row['ph']), 'ec': row['ec'], 'kinetic_source': 'BRENDA', 'source_origin_evidence': 'pinned processor assigns brenda only to BRENDA records; merge retains first row value/log10_value', 'sequence_group': digest(seq), 'native_pair_group': row['group'], 'native_cluster_40': row['sequence_40cluster'], 'native_cluster_80': row['sequence_80cluster'], 'native_split': 'combined_train_validation_test_release', 'derived_split': split_group(parameter + ':cluster40:' + row['sequence_40cluster']), 'split_role': 'derived_native40cluster_partition_of_BRENDA_record_subset'})
        df = pd.DataFrame(records)
        tables.append(write_table(df, f'brenda_record_{parameter}', {'enzyme_kinetic_parameters': contract(['enzyme_sequence', 'substrate_smiles', 'parameter', 'temperature_native', 'ph'], ['target_log10_brenda_record'], ['sequence_group', 'native_pair_group', 'native_cluster_40', 'native_cluster_80', 'derived_split'], ['parameter-specific log10 MAE', 'RMSE', 'R2'])}, 'C15:BRENDA_only', source, 'MIT AND CC-BY-4.0; CatPred contribution MIT; underlying BRENDA/UniProt data CC-BY-4.0', "Release subset contains only native rows explicitly labeled brenda, using each retained row's value/log10_value. All pooled max/mean target columns, sabio rows and ambiguous uniprot_search origins are excluded. This changes the prediction target from CatPred's aggregate benchmark to a retained BRENDA-record response. Source preprocessing capped extreme values; no claim of uncensored raw laboratory measurements. Intended source units retained, no independent experimental unit audit. Derived 40%-cluster split is not the original random split. Native combined source assets must not be redistributed as part of this cleared subset.", 'Identifiable BRENDA retained record, established by pinned source-specific processors and merge.py; mathematical log10/value consistency checked.', license_expression='MIT AND CC-BY-4.0', license_chain=[{'component': 'CatPred dataset/contribution', 'license': 'MIT', 'evidence': 'https://zenodo.org/api/records/14775076'}, {'component': 'BRENDA data', 'license': 'CC-BY-4.0', 'evidence': 'https://www.brenda-enzymes.org/download.php'}, {'component': 'UniProt sequences', 'license': 'CC-BY-4.0', 'evidence': 'https://rest.uniprot.org/help/license'}], origin_audit_assets=[asset_entry(lock, 'C15--scripts--processing--process_brenda.py'), asset_entry(lock, 'C15--scripts--processing--process_sabio.py'), asset_entry(lock, 'C15--scripts--processing--merge.py')], native_source_rows=len(native), excluded_rows=len(native) - len(df), native_target_column='log10_value; pooled log10kcat_max/log10km_mean/log10ki_mean deliberately excluded'))
    return tables

def build_mcsa_sites(lock: dict):
    amino = dict(zip(['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val'], 'ARNDCQEGHILKMFPSTWYV'))
    sequences = {}
    for asset in lock['assets']:
        if Path(asset['path']).name.startswith('MCSA--uniprot-reference-'):
            for accession, seq in fasta_sequences((ROOT / asset['path']).read_text()):
                sequences[accession] = (seq, asset)
    source = next((a for a in lock['assets'] if a.get('local_frozen_bundle')))
    with tarfile.open(ROOT / source['path']) as archive:
        entries = [json.loads(line) for line in archive.extractfile('entries.jsonl')]
    rows, rejections = ([], defaultdict(int))
    for index, entry in enumerate(entries):
        ids = {x for x in re.split('[,;\\s]+', entry.get('reference_uniprot_id', '')) if x}
        compounds = [{k: c.get(k) for k in ['chebi_id', 'count', 'name', 'type']} for c in entry.get('reaction', {}).get('compounds', [])]
        if not any((c['type'] == 'reactant' for c in compounds)) or not any((c['type'] == 'product' for c in compounds)):
            rejections['missing_reaction_sides'] += 1
            continue
        for identifier in sorted(ids):
            if identifier not in sequences:
                rejections['missing_reference_sequence'] += 1
                continue
            seq, seq_asset = sequences[identifier]
            residue_refs = []
            invalid = False
            for residue in entry.get('residues', []):
                for position in residue.get('residue_sequences', []):
                    if position.get('uniprot_id') != identifier or not position.get('is_reference'):
                        continue
                    p = position['resid']
                    code = amino.get(position['code'])
                    if not isinstance(p, int) or not 1 <= p <= len(seq) or (not code) or (seq[p - 1] != code):
                        invalid = True
                        break
                    residue_refs.append({'position_1based': p, 'amino_acid': code, 'roles': residue.get('roles', []), 'roles_summary': residue.get('roles_summary', '')})
            if invalid:
                rejections['reference_residue_sequence_mismatch'] += 1
                continue
            if not residue_refs:
                rejections['no_reference_catalytic_residues'] += 1
                continue
            labels = [0] * len(seq)
            for residue in residue_refs:
                labels[residue['position_1based'] - 1] = 1
            group = f"MCSA:{entry['mcsa_id']}"
            rows.append({'source_record_id': f'MCSA:entries.jsonl:{index}:{identifier}', 'native_record_id': str(entry['mcsa_id']), 'mcsa_id': entry['mcsa_id'], 'uniprot_id': identifier, 'enzyme_sequence': seq, 'reaction_compounds_json': json.dumps(compounds, sort_keys=True), 'enzyme_name': entry.get('enzyme_name', ''), 'residue_labels_json': json.dumps(labels, separators=(',', ':')), 'catalytic_residue_evidence_json': json.dumps(residue_refs, sort_keys=True), 'sequence_group': digest(seq), 'mechanism_group': group, 'native_split': 'not_provided', 'derived_split': split_group(group), 'split_role': 'derived_native_MCSA_entry_partition; development_exposed_source', 'sequence_asset_path': seq_asset['path'], 'sequence_source_sha256': seq_asset['sha256'], 'native_entry_url': entry.get('url', ''), 'label_evidence': 'native_MCSA_reference_UniProt_catalytic_residues; all recorded positions/residue identities checked against frozen sequence'})
    df = pd.DataFrame(rows)
    df = close_split_components(df, ['sequence_group', 'mechanism_group'], 'MCSA')
    df['split_role'] = 'derived_exact_sequence_and_native_entry_connected_components; development_exposed_source'
    return write_table(df, 'mcsa_reference_active_sites', {'active_sites': contract(['enzyme_sequence', 'reaction_compounds_json'], ['residue_labels_json'], ['sequence_group', 'mechanism_group', 'split_component', 'derived_split'], ['protein-macro residue AUPRC', 'F1', 'MCC'])}, 'MCSA+UniProt', source, 'CC-BY-4.0; native M-CSA annotations and UniProt sequences', 'Native M-CSA reaction-associated reference catalytic residues, joined to frozen UniProt FASTA and checked for exact amino-acid identity at every labeled position. Reaction representation retains ChEBI/name/stoichiometry participants, not invented SMILES. Binary zeros are unannotated residues, not experimentally proven nonfunctional sites. Only reference annotations retained; homology-transferred labels excluded. Derived partitions close native-entry and exact-sequence connected components; they remain development-exposed, not a new independent homology holdout. Previously inspected M-CSA data overlap the existing mechanism companion.', 'Direct M-CSA reference catalytic-residue annotations and separately licensed UniProt sequence; no EasIFA annotations or code required.', raw_assets_redistribution_allowed=True, license_expression='CC-BY-4.0', license_chain=[{'component': 'M-CSA native entries/residues/reactions', 'license': 'CC-BY-4.0', 'evidence': 'https://www.ebi.ac.uk/thornton-srv/m-csa/'}, {'component': 'UniProt sequence', 'license': 'CC-BY-4.0', 'evidence': 'https://rest.uniprot.org/help/license'}], unique_proteins=df['uniprot_id'].nunique(), native_source_entries=len(entries), rejection_counts=dict(rejections))

def build_mutations(lock: dict):
    source = asset_entry(lock, 'C13--Data.zip')
    records, exclusions = ([], defaultdict(int))
    raw_rows = 0
    with zipfile.ZipFile(ROOT / source['path']) as archive:
        if archive.testzip() is not None:
            raise ValueError('EnzEngDB ZIP CRC failure')
        meta = pd.read_excel(archive.open('Data/meta_data.xlsx'), dtype=str, keep_default_na=False)
        metadata = {clean(r['experiment_id']): r for r in meta.to_dict('records')}
        members = sorted((n for n in archive.namelist() if n.startswith('Data/experiments/') and n.endswith('.csv')))
        for member in members:
            frame = pd.read_csv(archive.open(member), dtype=str, keep_default_na=False)
            raw_rows += len(frame)
            campaign = Path(member).stem
            detail = metadata.get(campaign, {})
            doi = clean(detail.get('doi', ''))
            assay = clean(detail.get('assay_technique', ''))
            contexts = []
            native_rows = frame.to_dict('records')
            for r in native_rows:
                reaction = clean(r.get('reaction_smiles', r.get('smiles_reaction', '')))
                if not reaction:
                    s, p = (clean(detail.get('substrate_smiles', '')), clean(detail.get('product_smiles', '')))
                    reaction = s + '>>' + p if s and p else ''
                kind = clean(r.get('fitness_type', '')) or 'native_fitness_value'
                context = json.dumps({'reaction': reaction, 'fitness_type': kind, 'enzyme_form': clean(r.get('enzyme_form', '')), 'substrate_concentration': clean(r.get('substrate_concentration', '')), 'reaction_condition': clean(r.get('reaction_condition', '')), 'metadata_conditions': clean(detail.get('additional_information', '')), 'assay_technique': assay}, sort_keys=True)
                contexts.append((reaction, kind, context))
            values = pd.to_numeric(frame['fitness_value'], errors='coerce').to_numpy()
            seqs = frame['aa_sequence'].map(sequence).tolist()
            parents = defaultdict(list)
            for i, (reaction, kind, context) in enumerate(contexts):
                if native_rows[i]['amino_acid_substitutions'] == '#PARENT#' and seqs[i] and np.isfinite(values[i]):
                    parents[context].append(i)
            for i, row in enumerate(native_rows):
                if row['amino_acid_substitutions'] == '#PARENT#':
                    exclusions['parent_reference_rows'] += 1
                    continue
                reaction, kind, context = contexts[i]
                if not seqs[i] or not np.isfinite(values[i]) or reaction.count('>>') != 1 or (not all(reaction.split('>>'))):
                    exclusions['missing_sequence_fitness_or_reaction'] += 1
                    continue
                choices = parents.get(context, [])
                if not choices:
                    exclusions['no_compatible_parent_context'] += 1
                    continue
                parent_sequences = {seqs[j] for j in choices}
                if len(parent_sequences) != 1:
                    exclusions['ambiguous_parent_sequence'] += 1
                    continue
                if not doi or not assay:
                    exclusions['missing_paper_or_assay_metadata'] += 1
                    continue
                parent_seq = next(iter(parent_sequences))
                if seqs[i] == parent_seq:
                    exclusions['variant_identical_to_parent'] += 1
                    continue
                baseline = float(np.median([values[j] for j in choices]))
                record_id = f'C13:{member}:csvrow:{i}'
                prefix = clean(detail.get('prefix', ''))
                records.append({'source_record_id': record_id, 'native_record_id': row['id'], 'source_member': member, 'native_csv_row_index': i, 'campaign_id': campaign, 'reference_doi': doi, 'reference_kind': 'data_deposit' if 'zenodo' in doi.lower() else 'literature_doi_as_recorded', 'source_collection': prefix, 'curation_evidence': 'LLM_extracted_literature; not_independently_verified' if prefix == 'LLMDB' else 'source_curated_experimental_collection; not_independently_remeasured', 'assay_technique': assay, 'fitness_type': kind, 'reaction_smiles': reaction, 'parent_sequence': parent_seq, 'variant_sequence': seqs[i], 'mutations_native': row['amino_acid_substitutions'], 'fitness_value_native': float(values[i]), 'parent_fitness_median_native': baseline, 'fitness_delta_native': float(values[i]) - baseline, 'parent_reference_record_ids_json': json.dumps([f'C13:{member}:csvrow:{j}' for j in choices]), 'parent_reference_count': len(choices), 'assay_context_json': context, 'parent_sequence_group': digest(parent_seq), 'reaction_group': digest(reaction), 'campaign_assay_group': campaign + ':' + digest(context), 'native_split': 'not_provided', 'derived_split': split_group('C13:paper:' + doi), 'split_role': 'derived_source_reference_DOI_partition', 'target_evidence': 'within-campaign native measured fitness minus compatible parent median; not standardized across assays'})
    df = pd.DataFrame(records)
    df = close_split_components(df, ['reference_doi', 'parent_sequence_group'], 'C13')
    df['split_role'] = 'derived_reference_DOI_and_exact_parent_sequence_connected_components'
    return write_table(df, 'enzeng_mutation_effects', {'mutation_effects': contract(['parent_sequence', 'variant_sequence', 'reaction_smiles', 'fitness_type'], ['fitness_delta_native'], ['reference_doi', 'campaign_id', 'campaign_assay_group', 'parent_sequence_group', 'reaction_group', 'split_component', 'derived_split'], ['within-campaign Spearman', 'within-assay native delta MAE', 'direction accuracy'])}, 'C13', source, 'CC-BY-4.0 on Zenodo17310823 Data.zip; preserve per-paper provenance and attribution', 'Only explicit native parent rows (#PARENT#), identical recorded assay context, nonidentical valid parent/variant sequences, finite native fitness and resolved campaign reference/assay metadata. Parent replicates aggregate by median with every reference ID retained. Native fitness and derived change are assay-specific, not universal kcat/yield. Source collection/evidence separates LLM-extracted literature from the other experimental collections. DOI fields can reference a compiled data deposit, not an individual primary paper. Mutation strings preserved without claiming independently validated residue numbering. Derived partitions close reference-DOI and exact-parent-sequence components; this is not an independent homology holdout. Missing assay conditions remain unknown; within-source comparisons require audit before a frozen benchmark.', 'Source EnzEngDB release experiments and metadata; no fabricated fitness labels. Delta is explicitly derived from native variant measurement and compatible parent reference(s).', raw_assets_redistribution_allowed=True, native_experiment_files=len(members), native_source_rows=raw_rows, exclusion_counts=dict(exclusions), unique_reference_dois=df['reference_doi'].nunique(), unique_campaigns=df['campaign_id'].nunique(), source_collection_counts=df['source_collection'].value_counts().to_dict(), derived_split_counts=df['derived_split'].value_counts().to_dict(), license_expression='CC-BY-4.0', license_chain=[{'component': 'EnzEngDB data release', 'license': 'CC-BY-4.0', 'evidence': 'https://zenodo.org/api/records/17310823'}])

def build_inverse_biocatalyst_design(lock: dict):
    mutation = pd.read_parquet(OUT / 'enzeng_mutation_effects.parquet')
    curated = mutation.loc[mutation['source_collection'].eq('ARNLD')]
    source = asset_entry(lock, 'C13--Data.zip')
    rows = []
    with zipfile.ZipFile(ROOT / source['path']) as archive:
        for member, group in curated.groupby('source_member', sort=True):
            native = pd.read_csv(archive.open(member), dtype=str, keep_default_na=False)
            for row in group.to_dict('records'):
                original = native.iloc[row['native_csv_row_index']]
                for native_column, endpoint in [('TTN (if applicable)', 'total_turnover_number'), ('activity_for_reaction_% (if applicable)', 'reaction_activity_percent_as_reported')]:
                    try:
                        response = float(original.get(native_column, ''))
                    except ValueError:
                        continue
                    if not math.isfinite(response) or response < 0:
                        continue
                    rows.append({'source_record_id': row['source_record_id'] + ':endpoint:' + endpoint, 'native_record_id': row['native_record_id'], 'source_member': member, 'native_csv_row_index': row['native_csv_row_index'], 'parent_sequence': row['parent_sequence'], 'variant_sequence': row['variant_sequence'], 'reaction_smiles': row['reaction_smiles'], 'recorded_assay_context_json': row['assay_context_json'], 'desired_endpoint_kind': endpoint, 'desired_native_response': response, 'measured_response_native': response, 'native_endpoint_column': native_column, 'reference_doi': row['reference_doi'], 'campaign_id': row['campaign_id'], 'parent_sequence_group': row['parent_sequence_group'], 'reaction_group': row['reaction_group'], 'campaign_assay_group': row['campaign_assay_group'], 'derived_split': row['derived_split'], 'split_component': row['split_component'], 'native_split': 'not_provided', 'split_role': 'inherited_reference_DOI_and_exact_parent_sequence_components', 'component_role': 'biocatalyst_sequence', 'source_collection': 'ARNLD', 'oracle_scope': 'observed_variant_sequence_and_recorded_context_only; no extrapolated fitness', 'target_evidence': 'native curated explicit catalytic TTN or reaction-activity endpoint; DMS/proxy and LLMDB rows excluded'})
    df = pd.DataFrame(rows)
    return write_table(df, 'enzeng_inverse_biocatalyst_design', {'reaction_component_design': contract(['parent_sequence', 'reaction_smiles', 'recorded_assay_context_json', 'desired_endpoint_kind', 'desired_native_response'], ['variant_sequence'], ['reference_doi', 'campaign_id', 'campaign_assay_group', 'parent_sequence_group', 'reaction_group', 'split_component', 'derived_split'], ['sequence validity', 'mutation constraint satisfaction', 'native measured-oracle goal attainment for exact observed sequences only'])}, 'C13:curated_catalytic_endpoints', source, 'CC-BY-4.0; EnzEngDB data release with explicit record/paper attribution', 'Inverse biocatalyst-sequence design subtype, not general small-molecule catalyst generation. Only ARNLD curated campaigns with explicit finite nonnegative TTN/reaction-activity-percent labels and qualified parent/variant sequence context. All DMSDB and LLMDB rows excluded. Desired response is an explicit design goal, not an outcome-prediction input. Each native variant is a demonstrated solution rather than the only correct output. Oracle values exist only for exact observed variant/context/endpoint tuples; unseen generated sequences require experiments and receive no fabricated response. Activity-percent is retained as reported, not automatically equated with isolated yield. Views overlap the mutation-effect source rows.', 'Native explicit catalytic endpoint from the CC-BY-4.0 EnzEngDB experiment member, with source row ID and parent reference inherited from qualified mutation view.', raw_assets_redistribution_allowed=True, license_expression='CC-BY-4.0', license_chain=[{'component': 'EnzEngDB Data.zip', 'license': 'CC-BY-4.0', 'evidence': 'https://zenodo.org/api/records/17310823'}], primary_or_overlapping='overlapping_endpoint_view_of_enzeng_mutation_effects', unique_native_measurement_rows=df[['source_member', 'native_csv_row_index']].drop_duplicates().shape[0], unique_campaigns=df['campaign_id'].nunique(), endpoint_counts=df['desired_endpoint_kind'].value_counts().to_dict())

def build(fetch: bool=False):
    lock = json.loads(LOCK.read_text())
    verify_assets(lock, fetch)
    OUT.mkdir(parents=True, exist_ok=True)
    retrieval, up, rhea = build_retrieval(lock)
    quarantine_dir = OUT / 'quarantine'
    quarantine_dir.mkdir(exist_ok=True)
    for name in ['easifa_rhea_active_sites', 'catpred_kcat', 'catpred_km', 'catpred_ki']:
        old = OUT / f'{name}.parquet'
        if old.exists():
            old.replace(quarantine_dir / old.name)
    quarantined = [build_active_sites(lock, up, rhea), *build_kinetics(lock)]
    tables = [retrieval, build_mcsa_sites(lock), *build_brenda_kinetics(lock), build_mutations(lock), build_inverse_biocatalyst_design(lock)]
    sources = []
    for sid, paper in [('C13', 'https://doi.org/10.1093/nar/gkaf1142'), ('C14', 'https://arxiv.org/abs/2408.13659'), ('C15', 'https://doi.org/10.1038/s41467-025-57215-9'), ('C23', 'https://doi.org/10.1038/s41467-024-51511-6')]:
        record = lock['zenodo_records'].get(sid)
        source_rights = {}
        if record:
            primary = asset_entry(lock, sid + '-zenodo.json')
            source_rights = {'scope': 'dataset deposit/contribution; upstream component grants remain separate', 'grant_url': primary['url'], 'grant_snapshot': {'path': primary['path'], 'sha256': primary['sha256']}}
        sources.append({'id': sid, 'paper_url': paper, 'repository': lock['commits'].get(sid), 'zenodo_record': lock['zenodo_records'].get(sid, {}).get('id'), 'license_record': lock['zenodo_records'].get(sid, {}).get('metadata', {}).get('license'), 'license': 'CC-BY-4.0' if sid == 'C13' else 'MIT' if sid in {'C14', 'C15'} else 'unresolved_data_grant', 'rights': source_rights, 'assets': [a for a in lock['assets'] if Path(a['path']).name.startswith(sid + '--')], 'raw_assets_redistribution_allowed': all((a.get('redistribution_allowed', False) for a in lock['assets'] if Path(a['path']).name.startswith(sid + '--'))), 'raw_assets_policy': 'Only individually flagged assets may be bundled; source-level status is not a blanket grant for a raw directory.', 'redistribution_status': 'mixed_source; only explicitly cleared table subsets releaseable' if sid in {'C15', 'C23'} else 'redistributable_with_attribution_and_preserved_notices'})
    mcsa_grant, uniprot_grant = (asset_entry(lock, 'mcsa-rights.html'), asset_entry(lock, 'uniprot-rights.json'))
    sources.extend([{'id': 'MCSA', 'paper_url': 'https://doi.org/10.1093/nar/gkx1012', 'license': 'CC-BY-4.0', 'rights': {'grant_url': mcsa_grant['url'], 'grant_snapshot': {'path': mcsa_grant['path'], 'sha256': mcsa_grant['sha256']}}, 'raw_assets_redistribution_allowed': True, 'assets': [a for a in lock['assets'] if a.get('local_frozen_bundle')]}, {'id': 'UniProt', 'license': 'CC-BY-4.0', 'rights_url': 'https://rest.uniprot.org/help/license', 'rights': {'grant_url': uniprot_grant['url'], 'grant_snapshot': {'path': uniprot_grant['path'], 'sha256': uniprot_grant['sha256']}}, 'raw_assets_redistribution_allowed': True, 'assets': [a for a in lock['assets'] if Path(a['path']).name.startswith('MCSA--uniprot-reference-')]}])
    rights_audit = {'schema': 'synrxn.coverage-bio-rights-audit.v1', 'audit_date': '2026-10-20', 'policy': 'Only attribution/permissive redistribution grants accepted; code license alone does not clear data; contradictory upstream restrictions override a broader compiler claim for the release set.', 'primary_rights_evidence': [a for a in lock['assets'] if a.get('rights_evidence')], 'release_table_ids': [t['id'] for t in tables], 'quarantined_table_ids': [t['id'] for t in quarantined], 'decisions': [{'source': 'EnzEngDB', 'decision': 'release_CC-BY-4.0', 'evidence': 'https://zenodo.org/api/records/17310823'}, {'source': 'ReactZyme', 'decision': 'release_MIT_and_upstream_CC-BY-4.0', 'evidence': 'https://zenodo.org/api/records/13635807'}, {'source': 'CatPred pooled aggregate endpoints', 'decision': 'quarantine_due_to_SABIO_internal_noncommercial_terms', 'evidence': 'https://sabiork.h-its.org/layouts/content/termscondition.gsp', 'current_app_evidence': 'Data/raw/coverage/bio/sabio-current-app.js'}, {'source': 'CatPred retained BRENDA records', 'decision': 'release_only_explicit_brenda_rows_and_nonaggregate_value; no_SABIO_or_ambiguous_origin_rows', 'evidence': 'https://www.brenda-enzymes.org/download.php'}, {'source': 'EasIFA annotations', 'decision': 'quarantine_unresolved_data_grant; replace_with_direct_MCSA', 'evidence': 'https://github.com/wangxr0526/EasIFA'}, {'source': 'M-CSA catalytic references with UniProt sequences', 'decision': 'release_CC-BY-4.0', 'evidence': 'https://www.ebi.ac.uk/thornton-srv/m-csa/'}]}
    (OUT / 'rights_audit.json').write_text(json.dumps(rights_audit, indent=2) + '\n')
    notices = OUT / 'licenses'
    notices.mkdir(exist_ok=True)
    (notices / 'CatPred-MIT.txt').write_bytes((RAW / 'C15--LICENSE').read_bytes())
    attribution = 'Biocatalysis data attribution and redistribution notices\n\nEnzEngDB / Enzyme Engineering Database authors and contributors.\nData deposit: https://zenodo.org/records/17310823 (CC-BY-4.0).\nPublication: https://doi.org/10.1093/nar/gkaf1142\nChanges: source qualification, native-parent median differences, and inverse\nbiocatalyst-design views. Original campaign/reference DOI and extraction tier\nremain in each row. Inverse design excludes DMSDB and LLMDB observations.\n\nReactZyme: Chenqing Hua and coauthors.\nData deposit: https://zenodo.org/records/13635807 (MIT dataset grant).\nPublication: https://arxiv.org/abs/2408.13659\nChanges: positive-pair normalization, stable row IDs and native split retention.\nThe deposit license metadata is retained; no new negative labels were invented.\n\nCatPred-DB: Veda Sheersh Boorla and Costas D. Maranas.\nData deposit: https://zenodo.org/records/14775076 (MIT contribution grant).\nPublication: https://doi.org/10.1038/s41467-025-57215-9\nOriginal MIT notice: licenses/CatPred-MIT.txt.\nChanges: retained BRENDA-record endpoints only. Mixed SABIO-containing pooled\naggregate targets and ambiguous source origins are excluded from release.\n\nBRENDA Enzyme Database, original curators and current DSMZ contributors.\nhttps://www.brenda-enzymes.org/download.php (CC-BY-4.0).\nOnly identifiable retained BRENDA records are included, as mapped by the\npinned CatPred processing/merge source; source clipping and units are recorded.\n\nM-CSA: A. J. M. Ribeiro, G. L. Holliday, N. Furnham, J. D. Tyzack, K. Ferris,\nJ. M. Thornton, EMBL-EBI and contributors.\nhttps://www.ebi.ac.uk/thornton-srv/m-csa/ (CC-BY-4.0).\nPublication: https://doi.org/10.1093/nar/gkx1012\nChanges: native reference catalytic residues joined to frozen UniProt sequences,\nposition/amino-acid checks, residue labels, and explicit grouping identifiers.\n\nUniProt Consortium: https://www.uniprot.org/ (CC-BY-4.0).\nRights: https://rest.uniprot.org/help/license\nRhea/SIB contributors: https://www.rhea-db.org/ (CC-BY-4.0).\nRights: https://sparql.rhea-db.org/.well-known/void\nSequences, reactions and associations retain source identifiers and attribution.\n\nCC-BY-4.0 terms: https://creativecommons.org/licenses/by/4.0/\nMIT terms: https://opensource.org/license/mit\nKeep these notices, license metadata and row provenance with redistributed data.\nThe quarantine directory is outside the release set. Mixed CatPred raw assets,\nEasIFA labels and raw website/application evidence are not cleared by this file.\nOnly raw assets individually marked redistribution_allowed=true may be bundled.\n'
    (OUT / 'ATTRIBUTION.txt').write_text(attribution)
    catalog = {'schema': 'synrxn.coverage-expansion.bio.v1', 'source_lock': str(LOCK.relative_to(ROOT)), 'source_lock_sha256': sha256(LOCK), 'builder': str(Path(__source_file__).resolve().relative_to(ROOT)), 'builder_sha256': sha256(Path(__source_file__)), 'sources': sources, 'tables': tables, 'quarantined_tables': quarantined, 'rights_audit_path': str((OUT / 'rights_audit.json').relative_to(ROOT)), 'rights_audit_sha256': sha256(OUT / 'rights_audit.json'), 'summary': {'tables': len(tables), 'task_ids': sorted({t for v in tables for t in v['task_ids']}), 'rows_by_table': {v['id']: v['rows'] for v in tables}}, 'qualification': 'Release table list has explicit permissive/attribution source-license chains. Quarantined views are excluded from redistribution. Source-derived views do not imply trained model results, an independent homology audit, or universal physical-unit validity.'}
    (OUT / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps(catalog['summary'], indent=2))
    return catalog
__configure_runtime(globals(), 'kdd/coverage/adapters/bio.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Fetch missing pinned assets, verifying full size and SHA256')
    build(parser.parse_args().fetch)
