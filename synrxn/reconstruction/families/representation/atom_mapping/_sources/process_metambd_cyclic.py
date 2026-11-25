#!/usr/bin/env python
"""Build one standard AAM table from the cyclic train and test source cases.

Train keeps the first candidate. Test deduplicates all candidates by exact full
ITS isomorphism within each source row. Raw candidates and memberships remain
in a separate provenance file. Both splits use the same record schema.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import islice
from multiprocessing import get_context
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from rdkit import Chem, RDLogger
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.IO.chem_converter import rsmi_to_its

from synrxn.reconstruction.paths import REPO, repo_path
NAME = 'metambd_cyclic'
FIELDS = ['r_id', 'ground_truth', 'rxn', 'original_id', 'split', 'ground_truth_variants']
NODE_LABELS = ['element', 'charge', 'radical', 'aromatic', 'hcount', 'lone_pairs', 'isotope', 'present']


def mapped_molecules(value):
    """Validate heavy-atom correspondences without assigning proton mappings."""
    if value.count('>>') != 1:
        raise ValueError('not_a_reaction')
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecules = [Chem.MolFromSmiles(side, parameters) for side in value.split('>>')]
    if any(molecule is None for molecule in molecules):
        raise ValueError('invalid_or_truncated_smiles')
    heavy = []
    hydrogen_maps = []
    for molecule in molecules:
        atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
        indices = [atom.GetAtomMapNum() for atom in atoms]
        if not atoms or not all(indices) or len(indices) != len(set(indices)):
            raise ValueError('missing_or_duplicate_heavy_atom_maps')
        heavy.append({atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in atoms})
        hydrogen_maps.append({atom.GetAtomMapNum() for atom in molecule.GetAtoms()
                              if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum()})
    if heavy[0] != heavy[1]:
        raise ValueError('heavy_atom_mapping_not_conserved')
    mismatch = hydrogen_maps[0] != hydrogen_maps[1]
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 1:
                atom.SetAtomMapNum(0)
    return molecules, mismatch


def reaction_views(value):
    """Keep exact tautomers and proton participants; remove mapped spectators."""
    molecules, mismatch = mapped_molecules(value)
    fragments = []
    for molecule in molecules:
        fragments.append(Counter(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                                 for fragment in Chem.GetMolFrags(molecule, asMols=True)))
    shared = fragments[0] & fragments[1]
    active = [side - shared for side in fragments]
    if not all(active):
        raise ValueError('no_active_reaction_after_spectator_removal')
    ground_truth = '>>'.join('.'.join(sorted(side.elements())) for side in active)
    unmapped, its_sides = [], []
    for side in active:
        molecule = Chem.MolFromSmiles('.'.join(sorted(side.elements())))
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        unmapped.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        its_components = []
        for component in side.elements():
            fragment = Chem.MolFromSmiles(component)
            if any(atom.GetAtomicNum() != 1 for atom in fragment.GetAtoms()):
                its_components.append(component)
        its_sides.append('.'.join(sorted(its_components)))
    return {'ground_truth': ground_truth, 'rxn': '>>'.join(unmapped),
            'its_ground_truth': '>>'.join(its_sides),
            'hydrogen_map_mismatch': mismatch, 'spectator_pairs_removed': sum(shared.values())}


def full_its(projection):
    """Retain the whole heavy-atom ITS and stereo registry, ignoring map IDs in matching."""
    return rsmi_to_its(projection, core=False, format='tuple',
                       node_attrs=NODE_LABELS[:-1] + ['atom_map'], edge_attrs=['order'])


def hydrogen_only_key(mapped):
    """Keep unmapped proton participants distinct even outside the heavy-atom ITS."""
    sides = []
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    for side in mapped.split('>>'):
        components = []
        molecule = Chem.MolFromSmiles(side, parameters)
        for fragment in Chem.GetMolFrags(molecule, asMols=True):
            if all(atom.GetAtomicNum() == 1 for atom in fragment.GetAtoms()):
                for atom in fragment.GetAtoms():
                    atom.SetAtomMapNum(0)
                components.append(Chem.MolToSmiles(fragment))
        sides.append(tuple(sorted(components)))
    return tuple(sides)


class ITSClasses:
    """Bucket full graphs with WL, then resolve exact stereo-aware isomorphism."""

    def __init__(self):
        self.hasher = WLHash(node=NODE_LABELS, edge='order', iterations=3, digest_size=16)
        self.matcher = GraphCluster(node_label_names=NODE_LABELS,
                                    node_label_default=[None] * len(NODE_LABELS), edge_attribute='order')
        self.buckets = defaultdict(list)
        self.graphs = []

    def assign(self, graph, proton_key):
        signature = self.hasher.weisfeiler_lehman_graph_hash(graph)
        bucket = (proton_key, graph.number_of_nodes(), graph.number_of_edges(), signature)
        for index in self.buckets[bucket]:
            pair = [{'its': self.graphs[index]}, {'its': graph}]
            self.matcher.fit(pair, rule_key='its', attribute_key=None)
            if pair[0]['class'] == pair[1]['class']:
                return index
        index = len(self.graphs)
        self.graphs.append(graph)
        self.buckets[bucket].append(index)
        return index


def initialize_worker():
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_cases(archive):
    """Read every original CSV row as one case, with its native split label."""
    with zipfile.ZipFile(archive) as source:
        members = {}
        for member in source.namelist():
            for split in ('train', 'test'):
                if Path(member).name == f'{split}_ringreactions.csv':
                    if split in members:
                        raise ValueError(f'Duplicate archive member for {split}')
                    members[split] = member
        if set(members) != {'train', 'test'}:
            raise ValueError('Archive must contain train_ringreactions.csv and test_ringreactions.csv')
        for split in ('train', 'test'):
            with source.open(members[split]) as handle:
                reader = csv.reader(io.TextIOWrapper(handle, encoding='utf-8-sig', newline=''))
                for source_row, candidates in enumerate(reader, 1):
                    yield {'split': split, 'source_row': source_row, 'archive_member': members[split],
                           'candidate_strings': candidates}


def process_case(case):
    """Use one pipeline for both splits, preserving case and candidate provenance."""
    split, source_row = case['split'], case['source_row']
    classes, seen = ITSClasses(), {}
    selected, memberships, failures = [], [], []
    stats = Counter(source_rows=1, source_cells=len(case['candidate_strings']))
    first = None
    for column, string in enumerate(case['candidate_strings'], 1):
        membership = {'source_column': column, 'representative_source_column': None}
        try:
            result = reaction_views(string)
            if column == 1:
                first = result
            if split == 'train' and column > 1:
                membership['status'] = 'not_selected'
            else:
                mapped = result['ground_truth']
                if mapped not in seen:
                    seen[mapped] = classes.assign(full_its(result['its_ground_truth']), hydrogen_only_key(mapped))
                index = seen[mapped]
                if index == len(selected):
                    selected.append((column, result))
                membership.update(status='valid', its_class=index,
                                  representative_source_column=selected[index][0])
        except Exception as error:
            reason = type(error).__name__ + ': ' + str(error)[:200]
            if column == 1:
                raise ValueError(f'{split} row {source_row}: invalid first candidate: {reason}') from error
            membership.update(status='invalid_source', reason=reason)
            failures.append({'split': split, 'source_row': source_row, 'source_column': column, 'reason': reason})
            stats['rejected_cells'] += 1
        else:
            stats['accepted_cells'] += 1
            stats['hydrogen_map_mismatch_cells'] += result['hydrogen_map_mismatch']
            stats['spectator_component_pairs_removed'] += result['spectator_pairs_removed']
        memberships.append(membership)
    if first is None:
        raise ValueError(f'{split} row {source_row}: no first candidate')
    variants = [result['ground_truth'] for _, result in selected]
    record = {'r_id': f'{NAME}_{split}_{source_row}', 'ground_truth': first['ground_truth'],
              'rxn': first['rxn'], 'original_id': f'{split}:r{source_row}', 'split': split,
              'ground_truth_variants': json.dumps(variants)}
    provenance = {**case, 'r_id': record['r_id'], 'candidate_memberships': memberships,
                  'representative_source_columns': [column for column, _ in selected],
                  'its_ground_truth_variants': [result['its_ground_truth'] for _, result in selected]}
    stats['mapping_references'] = len(variants)
    return record, provenance, dict(stats), failures


def write_bytes(path, value):
    """Replace changed artifacts atomically; identical rebuilds preserve their bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == value:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2) + '\n').encode())


def save_outputs(output, provenance_path, records, provenance):
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(records)
    write_bytes(output, gzip.compress(table.getvalue().encode(), mtime=0))
    lines = ''.join(json.dumps(row) + '\n' for row in provenance)
    write_bytes(provenance_path, gzip.compress(lines.encode(), mtime=0))


def source_record(archive, output, summary):
    """Pin local source identity without inferring labels, rights or validation."""
    return {'schema_version': '1.0', 'dataset_key': f'aam/{NAME}.csv.gz',
            'lock_status': 'pinned_local_user_supplied_archive', 'source_file': str(archive),
            'source_sha256': summary['archive_sha256'], 'source_members': summary['source_members'],
            'source_url': '', 'source_doi': '', 'source_commit_or_version': 'local archive received 2026-10-26',
            'retrieved_at': '2026-10-26', 'upstream_license': 'not supplied in archive',
            'license_spdx': 'LicenseRef-Unknown', 'license_url': '', 'redistribution_partition': 'excluded',
            'citations': [], 'required_attribution': [],
            'outstanding_fields': ['upstream_source_identity', 'upstream_license',
                                   'independent_mapping_validation', 'source_candidate_order_semantics'],
            'qualification': 'Source candidate heavy-atom mappings; selected first targets are not verified gold.',
            'canonical_table': str(output), 'canonical_rows': summary['canonical_rows'],
            'canonical_sha256': summary['output_sha256'][str(output)],
            'processing_script': 'synrxn/reconstruction/families/representation/atom_mapping/_sources/process_metambd_cyclic.py',
            'processing_script_sha256': summary['processing_script_sha256'],
            'processing_report': summary['report'], 'candidate_provenance': summary['candidate_provenance']}


def build(archive, output, provenance_path, report, workers=2, batch_size=16, source_path=None):
    """Build the combined table, candidate sidecar and reproducibility report."""
    initialize_worker()
    records, provenance, failures = [], [], []
    counts = defaultdict(Counter)
    jobs = archive_cases(archive)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             mp_context=get_context('spawn')) as executor:
        while batch := list(islice(jobs, batch_size)):
            for record, case, stats, rejected in executor.map(process_case, batch, chunksize=1):
                records.append(record)
                provenance.append(case)
                counts[record['split']].update(stats)
                failures.extend(rejected)
    save_outputs(output, provenance_path, records, provenance)
    with zipfile.ZipFile(archive) as zipped:
        members = {case['split']: case['archive_member'] for case in provenance}
        payloads = {split: {'member': member, 'bytes': zipped.getinfo(member).file_size,
                            'sha256': hashlib.sha256(zipped.read(member)).hexdigest()}
                    for split, member in members.items()}
    inputs = {split: {row['rxn'] for row in records if row['split'] == split} for split in counts}
    summary = {'schema': 'synrxn.metambd-cyclic-processing.v3', 'dataset': NAME, 'task': 'aam',
               'source_archive': str(archive), 'archive_sha256': sha(archive), 'source_members': payloads,
               'canonical_aam_table': str(output), 'canonical_rows': len(records), 'fields': FIELDS,
               'counts': {split: {**dict(stats), 'distinct_inputs': len(inputs[split])}
                          for split, stats in counts.items()},
               'train_policy': 'first candidate per original source row; one selected reference',
               'test_policy': 'one case per source row; full ITS deduplication within each case',
               'rxn_policy': 'first candidate only; spectator removal; clear maps; canonical isomeric SMILES',
               'mapping_scope': 'heavy atoms; preserve proton participants; clear unmatched H map labels',
               'its_policy': 'full heavy-atom ITS; absolute paired attributes, isotopes and stereo; exact iso after WL',
               'proton_policy': 'hydrogen-only component signatures supplement heavy-atom ITS comparison',
               'reference_tier': 'source candidates; no positive/negative correctness labels inferred',
               'raw_candidates': 'all original strings and order preserved in candidate provenance',
               'candidate_provenance': str(provenance_path), 'report': str(report), 'rejected_cells': failures,
               'exact_first_input_overlap': len(inputs.get('train', set()) & inputs.get('test', set())),
               'source_license': 'not supplied in archive', 'redistribution_partition': 'excluded',
               'runtime_versions': {name: version(name) for name in ('rdkit', 'synkit', 'networkx')},
               'processing_script_sha256': sha(__file__),
               'output_sha256': {str(path): sha(path) for path in (output, provenance_path)}}
    write_json(report, summary)
    if source_path is not None:
        write_json(source_path, source_record(archive, output, summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=repo_path(f'Data/raw/{NAME}/ringreactions.zip'))
    parser.add_argument('--output', type=Path, default=repo_path(f'Data/aam/{NAME}.csv.gz'))
    parser.add_argument('--provenance', type=Path,
                        default=REPO / f'Data/metadata/reconstruction/{NAME}/candidates.jsonl.gz')
    parser.add_argument('--report', type=Path,
                        default=REPO / f'Data/metadata/reconstruction/{NAME}/processing-summary.json')
    parser.add_argument('--source-record', type=Path, default=REPO / f'Data/metadata/sources/aam/{NAME}.json')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error('--workers and --batch-size must be positive')
    summary = build(args.archive, args.output, args.provenance, args.report,
                    args.workers, args.batch_size, args.source_record)
    print(json.dumps({'canonical_rows': summary['canonical_rows'], 'counts': summary['counts']}, indent=2))


if __name__ == '__main__':
    main()
