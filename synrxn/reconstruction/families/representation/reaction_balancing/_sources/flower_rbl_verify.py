"""Verify complete RBL bundles and their immutable parent/base relationships."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, require

SCHEMAS = {'base': 'synrxn.flower-rbl.v2', 'hard': 'synrxn.flower-rbl-hard.v1'}
TRACE = ('original_id', 'parent_r_id', 'case_type', 'role', 'steps', 'center_cluster_id', 'center_type',
         'source_group', 'source_parent_group', 'source_lines', 'source_seed_id', 'source_member', 'source_split',
         'removed_reactant', 'removed_product')


def read_rows(path):
    with gzip.open(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def fragments(value):
    return Counter(value.split('.')) if value else Counter()


def verify(folder, kind):
    """Check payload completeness, source hashes, schema, counts and row relationships."""
    manifest = json.loads((folder / 'manifest.json').read_text())
    require(manifest['schema'] == SCHEMAS[kind], 'Unexpected RBL bundle schema')
    actual = {path.relative_to(folder).as_posix(): digest(path) for path in folder.rglob('*')
              if path.is_file() and path != folder / 'manifest.json'}
    require(actual == manifest['files'], 'RBL bundle checksum mismatch')
    source = json.loads((folder / 'source.json').read_text())
    require(source['schema'] == manifest['schema'], 'RBL source schema mismatch')
    parent_path = Path(source['source_aam'])
    parent_sha = source['source_sha256' if kind == 'base' else 'source_aam_sha256']
    require(digest(parent_path) == parent_sha, 'RBL parent AAM source changed')
    parents = {row['r_id']: row for row in read_rows(parent_path)}
    bases = None
    if kind == 'hard':
        base_path = Path(source['base_file'])
        require(digest(base_path) == source['base_sha256'], 'Hard RBL base source changed')
        require(digest(Path(source['source_dump'])) == source['source_dump_sha256'], 'Hard RBL raw source changed')
        bases = {row['r_id']: row for row in read_rows(base_path)}
    counts, roles, seen, base_ids, context = Counter(), Counter(), set(), set(), Counter()
    for row in read_rows(folder / 'rbl.csv.gz'):
        require(row['r_id'] not in seen, 'Duplicate RBL row ID')
        seen.add(row['r_id'])
        parent = parents[row['parent_r_id']]
        require(row['gt_rxn'] == row['ground_truth'], 'RBL reference alias differs')
        for key in ('original_id', 'role', 'steps', 'center_cluster_id', 'center_type',
                    'source_group', 'source_lines', 'source_member', 'source_split'):
            require(row[key] == parent[key], f'RBL parent provenance differs: {key}')
        if kind == 'base':
            require(row['gt_aam'] == parent['ground_truth'], 'Mapped RBL reference differs from parent')
        else:
            base_ids.add(row['base_r_id'])
            base = bases[row['base_r_id']]
            require(all(row[key] == base[key] for key in TRACE), 'Hard RBL core/source fields differ')
            old_left, old_right = base['rxn'].split('>>')
            left, right = row['rxn'].split('>>')
            require(right == old_right, 'Hard input product side changed')
            require(fragments(left) == fragments(old_left) + fragments(row['input_context_rxn']),
                    'Hard input does not match base plus native context')
            for current, original in zip(row['gt_rxn'].split('>>'), base['gt_rxn'].split('>>')):
                require(fragments(current) == fragments(original) + fragments(row['context_rxn']),
                        'Hard target context changed')
            expected = '>>'.join(side + ('.' + row['context_aam'] if row['context_aam'] else '')
                                 for side in base['gt_aam'].split('>>'))
            require(row['gt_aam'] == expected, 'Hard mapped target does not preserve source mapping/context')
            require((row['has_context'] == 'True') == bool(row['input_context_rxn']), 'Hard context flag differs')
            context['augmented' if row['has_context'] == 'True' else 'no_heavy_context'] += 1
        counts[row['case_type']] += 1
        roles[row['role']] += 1
    require(len(seen) == source['rows'] == manifest['rows'], 'RBL row count mismatch')
    require(dict(counts) == source['by_case_type'] and dict(roles) == source['by_role'], 'RBL split/type counts differ')
    if kind == 'hard':
        require(len(base_ids) == len(seen) and base_ids == set(bases), 'Hard/base rows do not join one-to-one')
        require(dict(context) == source['context_counts'], 'Hard context counts differ')
    return {'variant': kind, 'rows': len(seen), 'by_case_type': dict(counts), 'by_role': dict(roles),
            'csv_sha256': digest(folder / 'rbl.csv.gz'), 'verified': True}
