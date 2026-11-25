"""Align FlowER centers globally while preserving frozen reactions and witnesses."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import io
from itertools import combinations, islice
import json
from multiprocessing import get_context
from pathlib import Path
import zipfile

from synrxn.reconstruction.paths import REPO, repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, require, write_json

FIELDS = ('r_id', 'center_cluster_id', 'center_type', 'source_group', 'source_parent_group',
          'source_lines', 'source_seed_id', 'source_member', 'source_split')


def write_csv(path, rows, fields):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf8', newline='') as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)


def initialize_worker():
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_pipeline import initialize_worker as initialize
    initialize()


def single_step_center(reaction):
    """Express an elementary ITS as a two-state MC with the same label schema."""
    import networkx as nx
    from synkit.IO.chem_converter import rsmi_to_its
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import NODE_FIELDS, mechanistic_center

    graph = rsmi_to_its(reaction, format='tuple', core=False,
                        node_attrs=['element', 'atom_map', *(f for f in NODE_FIELDS if f != 'present')],
                        edge_attrs=['order'])
    timeline = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        require(attrs['present'] == (True, True), 'Elementary atom pool is open')
        require(attrs['element'][0] == attrs['element'][1], 'Elementary mapped element changes')
        timeline.add_node(node, element=attrs['element'][0], **{f: attrs[f] for f in NODE_FIELDS})
    for left, right, attrs in graph.edges(data=True):
        timeline.add_edge(left, right, order=attrs['order'])
    return mechanistic_center(timeline)[0]


def project(task):
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import mechanistic_center, pathway_mtg
    row, reactions = task
    graph = (single_step_center(reactions[0]) if len(reactions) == 1
             else mechanistic_center(pathway_mtg(reactions))[0])
    require(bool(graph), f"Empty center: {row['r_id']}")
    return row, graph


def witnesses(path):
    with gzip.open(path, 'rt') as handle:
        values = [json.loads(line) for line in handle]
    rows = []
    for value in sorted(values, key=lambda value: int(value['r_id'])):
        source = value.get('source_membership', value.get('source_representative'))
        require(source is not None, 'Missing native source witness')
        lines = source['source_lines']
        lines = json.loads(lines) if isinstance(lines, str) else lines
        require(len(lines) == int(value['steps']), 'Witness length differs from step count')
        role = value['role']
        rows.append({'r_id': str(value['r_id']), 'role': role, 'steps': int(value['steps']),
                     'center_type': 'RC' if int(value['steps']) == 1 else 'MC',
                     'source_group': str(source['source_group']),
                     'source_parent_group': str(source['source_group']).split('::', 1)[0],
                     'source_lines': json.dumps(lines, separators=(',', ':')),
                     'source_seed_id': source.get('seed_id', ''),
                     'source_member': {'train': 'inputs/train.txt.gz', 'val': 'inputs/valid.txt.gz',
                                       'test': 'inputs/test.txt.gz'}[role],
                     'source_split': source.get('source_split', 'validation' if role == 'val' else role)})
    return rows


def source_lines(dump, rows):
    required = defaultdict(set)
    for row in rows:
        required[row['source_member']].update(json.loads(row['source_lines']))
    result = {}
    with zipfile.ZipFile(dump) as archive:
        for member, wanted in required.items():
            values = {}
            last = max(wanted)
            with archive.open(member) as binary, gzip.GzipFile(fileobj=binary) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf8') as text:
                    for number, line in enumerate(text, 1):
                        if number in wanted:
                            reaction, group = line.rstrip('\r\n').rsplit('|', 1)
                            reaction = '>>'.join('.'.join(sorted(side.split('.'))) for side in reaction.split('>>'))
                            values[number] = (reaction, group)
                        if number == last:
                            break
            require(len(values) == len(wanted), f'Missing source lines: {member}')
            result[member] = values
    return result


def overlap(rows):
    sets = defaultdict(set)
    for row in rows:
        sets[row['role']].add(row['center_cluster_id'])
    return {'classes_by_role': {role: len(values) for role, values in sets.items()},
            'pairwise': {a + '_' + b: len(sets[a] & sets[b])
                         for a, b in combinations(('train', 'val', 'test'), 2)},
            'all_three': len(sets['train'] & sets['val'] & sets['test'])}


def build(dump, provenance, output, workers=4, batch_size=128):
    """Reconstruct selected source centers only, and assign deterministic global IDs."""
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import ExactClusters
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_store import pack_graph

    initialize_worker()
    rows = witnesses(provenance)
    selected = source_lines(dump, rows)
    tasks = []
    for row in rows:
        lines = [selected[row['source_member']][line] for line in json.loads(row['source_lines'])]
        require(all(group == row['source_group'] for _, group in lines), 'Witness mechanism group mismatch')
        tasks.append((row, [reaction for reaction, _ in lines]))
    clusters, annotations = ExactClusters(), []
    jobs = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             mp_context=get_context('spawn')) as executor:
        while batch := list(islice(jobs, batch_size)):
            for row, graph in executor.map(project, batch, chunksize=4):
                row['center_cluster_id'] = f'center-{clusters.assign(graph):06d}'
                annotations.append(row)
            print(f'Grouped {len(annotations):,}/{len(rows):,} source centers', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / 'center_annotations.csv.gz'
    write_csv(csv_path, [{field: row[field] for field in FIELDS} for row in annotations], FIELDS)
    templates = output / 'center_templates.jsonl.gz'
    with templates.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            for index, graph in enumerate(clusters.graphs):
                record = {'center_cluster_id': f'center-{index:06d}', 'graph': json.loads(pack_graph(graph))}
                zipped.write((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
    report = {'schema': 'synrxn.flower-global-centers.v1', 'rows': len(rows),
              'by_role': dict(Counter(row['role'] for row in rows)), 'global_classes': len(clusters.graphs),
              'all_records': overlap(annotations),
              'multistep_records': overlap([row for row in annotations if row['steps'] > 1]),
              'one_step_records': overlap([row for row in annotations if row['steps'] == 1]),
              'definition': 'One-step two-state MC (RC); multistep unordered MC; same labels, global exact iso',
              'grouping_excludes': ['role', 'steps', 'map_numbers', 'local_center_ids'],
              'source_dump': str(dump), 'source_dump_sha256': digest(dump),
              'native_provenance_sha256': digest(provenance),
              'implementation_sha256': digest(Path(__file__)),
              'files': {path.name: digest(path) for path in (csv_path, templates)}}
    write_json(output / 'center_report.json', report)
    return report


def main():
    """Build global center annotations from the pinned FlowER inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, default=repo_path('Data/dumps/flower.dump.zip'))
    parser.add_argument('--provenance', type=Path,
                        default=repo_path('Data/tables/development/aam/flower/native_provenance.jsonl.gz'))
    parser.add_argument('--output', type=Path, default=REPO / 'Data/metadata/reconstruction/flower')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error('Workers and batch size must be positive')
    build(args.dump, args.provenance, args.output, args.workers, args.batch_size)


if __name__ == '__main__':
    main()
