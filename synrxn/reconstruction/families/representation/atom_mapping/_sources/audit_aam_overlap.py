#!/usr/bin/env python
"""Audit repeated AAM inputs and conflicting ITS graphs without removing records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
from functools import lru_cache
import gzip
from importlib.metadata import version
from itertools import combinations, islice
import json
from pathlib import Path
import signal
import sqlite3
import time

from rdkit import RDLogger

from synrxn.reconstruction.paths import REPO as ROOT, repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_mapping_constraints as constraints
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, read_registry, write_json
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import csv_writer

SCHEMA = 'synrxn.aam-overlap.v2'
MODES = ('exact', 'normalized', 'undirected')
TEXT_FIELDS = ('dataset', 'r_id', 'original_id', 'role', 'steps', 'rxn', 'ground_truth',
               'canonical_rxn', 'normalized_rxn', 'exact_key', 'normalized_key', 'undirected_key',
               'input_status', 'mapping_status', 'mapping_encoding', 'explicit_mapping_status', 'target_rxn',
               'target_normalized_rxn', 'represented_rxn', 'represented_normalized_rxn', 'error')
INT_FIELDS = ('source_row', 'reverse_direction', 'spectator_pairs', 'explicit_h_atoms',
              'endpoint_match', 'normalized_endpoint_match', 'unmapped_heavy_atoms')


def quiet():
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')


def bounded_map(pool, function, values, batch_size):
    iterator = iter(values)
    while batch := list(islice(iterator, batch_size)):
        yield from (pool.map(function, batch, chunksize=16) if pool else map(function, batch))


def index_record(source):
    row = {key: '' for key in TEXT_FIELDS}
    row.update({key: 0 for key in INT_FIELDS})
    row.update(source, input_status='ok', mapping_status='not_checked')
    try:
        identity = chem.identities(row['rxn'])
        row.update({k: v for k, v in identity.items() if k != 'reverse'})
        row['reverse_direction'] = identity['reverse']
    except Exception as error:
        row.update(input_status='parse_failed', error=str(error)[:220])
    try:
        reference = chem.reference_info(row['ground_truth'])
        row.update({k: v for k, v in reference.items() if k not in {'mols', 'active'}})
        row['endpoint_match'] = row['target_rxn'] == row['canonical_rxn']
        row['normalized_endpoint_match'] = row['target_normalized_rxn'] == row['normalized_rxn']
    except Exception as error:
        row.update(mapping_status='parse_failed', error=(row['error'] + ' ' + str(error))[:220])
    return row


@lru_cache(maxsize=128)
def prepared_graphs(value):
    result = chem.reference_info(value)
    full = chem.pack(chem.its_graph(result['mols']))
    active = (full if result['target_rxn'] == result['target_normalized_rxn']
              else chem.pack(chem.its_graph(result['active'])))
    explicit, explicit_active = None, None
    if result['explicit_mapping_status'] in {'ok', 'equivalent', 'partial'} and result['explicit_h_atoms']:
        explicit = chem.pack(chem.its_graph(result['mols'], heavy=False))
        explicit_active = (explicit if result['target_rxn'] == result['target_normalized_rxn']
                           else chem.pack(chem.its_graph(result['active'], heavy=False)))
    return full, active, explicit, explicit_active


def graph_record(row):
    try:
        return (row['ordinal'], *prepared_graphs(row['ground_truth']), 'ok', '')
    except Exception as error:
        return (row['ordinal'], None, None, None, None, 'graph_failed', str(error)[:220])


def source_rows(entries, input_root):
    for entry in entries:
        with gzip.open(input_root / entry['name'] / 'aam.csv.gz', 'rt', newline='') as handle:
            for index, row in enumerate(csv.DictReader(handle), 1):
                yield {key: row.get(key, '') for key in
                       ('r_id', 'original_id', 'role', 'steps', 'rxn', 'ground_truth')} | {
                           'dataset': entry['name'], 'source_row': index}


def limited(function, timeout):
    def expired(*_):
        raise TimeoutError('Exact ITS isomorphism exceeded time limit')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def assign(classes, graph, timeout):
    return limited(lambda: classes.assign(graph), timeout)


def class_relations(graphs, timeout):
    """Certify incompatible sets; a broad class cannot hide conflicting full maps."""
    relation = {(i, i): True for i in range(len(graphs))}
    for i, j in combinations(range(len(graphs)), 2):
        try:
            relation[i, j] = limited(lambda: constraints.compare(graphs[i], graphs[j], distinct=True), timeout)
        except TimeoutError:
            relation[i, j] = None
    conflict = any(value is False for value in relation.values())
    anchor = 0 if len(graphs) == 1 else None
    if not conflict and len(graphs) > 1:
        for i in range(len(graphs)):
            if not constraints.Constraints(graphs[i]).complete:
                continue
            try:
                if all(relation[tuple(sorted((i, j)))] is True and
                       limited(lambda: constraints.compare(graphs[i], graphs[j], refine=True), timeout)
                       for j in range(len(graphs)) if i != j):
                    anchor = i
                    break
            except TimeoutError:
                continue
    return relation, conflict, anchor


def source_pair(result, a, b):
    values = [result['relations'][tuple(sorted((i, j)))]
              for i in result['raw_sources'][a] for j in result['raw_sources'][b]]
    return {'comparable_groups': any(x is not None for x in values),
            'shared_mapping_groups': any(x is True for x in values),
            'mapping_disagreement_groups': any(x is False for x in values),
            'no_shared_mapping_groups': bool(values) and all(x is False for x in values)}


def classify(rows, mode, timeout):
    classes, explicit_classes = chem.ITSClasses(), {}
    members, sources, representatives = [], defaultdict(set), {}
    explicit_sets = defaultdict(set)
    for row in rows:
        full = mode == 'exact'
        reverse = mode == 'undirected' and bool(row['reverse_direction'])
        status = 'ok' if row['mapping_status'] in {'ok', 'equivalent', 'partial'} else row['mapping_status']
        class_id, explicit_id = None, None
        matches = row['endpoint_match'] if full else row['normalized_endpoint_match']
        if status == 'ok' and not matches:
            status = 'input_target_endpoint_mismatch'
        if status == 'ok' and row['graph_status'] != 'ok':
            status = row['graph_status'] or 'graph_unavailable'
        if status == 'ok':
            try:
                graph = chem.unpack(row['heavy_full'] if full else row['heavy_normalized'], reverse=reverse)
                class_id = assign(classes, graph, timeout)
                sources[row['dataset']].add(class_id)
                representatives.setdefault(class_id, row)
            except TimeoutError:
                status = 'isomorphism_timeout'
        represented = row['represented_rxn'] if full else row['represented_normalized_rxn']
        payload = row['explicit_full'] if full else row['explicit_normalized']
        if status == 'ok' and payload:
            if reverse:
                represented = '>>'.join(reversed(represented.split('>>')))
            key = class_id, represented
            bucket = explicit_classes.setdefault(key, chem.ITSClasses())
            try:
                explicit_id = assign(bucket, chem.unpack(payload, reverse=reverse), timeout)
                explicit_sets[key].add(explicit_id)
            except TimeoutError:
                status = 'explicit_isomorphism_timeout'
        members.append({'dataset': row['dataset'], 'r_id': row['r_id'],
                        'original_id': row['original_id'], 'role': row['role'], 'steps': row['steps'],
                        'heavy_class': class_id, 'explicit_class': explicit_id,
                        'mapping_status': row['mapping_status'],
                        'explicit_representation_key': chem.sha(represented) if represented else '',
                        'status': status})
    relation, conflict, anchor = class_relations(classes.graphs, timeout)
    cross_conflict = any(relation[tuple(sorted((a, b)))] is False
                         for first, second in combinations(sources, 2) for a in sources[first] for b in sources[second])
    unresolved = len(classes.graphs) > 1 and not conflict and anchor is None
    for member in members:
        raw = member['heavy_class']
        member.update(constraint_class=raw, is_anchor=raw is not None and raw == anchor)
        for name, expected in [('compatible_constraint_classes', True), ('incompatible_constraint_classes', False)]:
            member[name] = json.dumps([i for i in range(len(classes.graphs))
                                       if raw is not None and relation[tuple(sorted((raw, i)))] is expected])
        if raw is not None and anchor is not None:
            member['heavy_class'] = 0
        if raw is not None and unresolved and member['status'] == 'ok':
            member['status'] = 'mapping_constraints_unresolved'
    if anchor is not None:
        representatives = {0: representatives[anchor]}
    h_conflict = any(class_relations(bucket.graphs, timeout)[1] for bucket in explicit_classes.values()
                     if len(bucket.graphs) > 1)
    return {'classes': 1 if anchor is not None else len(classes.graphs), 'members': members,
            'sources': {s: ({0} if anchor is not None and ids else ids) for s, ids in sources.items()},
            'raw_sources': sources, 'relations': relation, 'conflict': conflict, 'constraints_unresolved': unresolved,
            'representatives': representatives, 'cross_conflict': cross_conflict, 'hydrogen_disagreement': h_conflict}


def analyze_mode(db, mode, timeout, writers):
    key = {'exact': 'exact_key', 'normalized': 'normalized_key', 'undirected': 'undirected_key'}[mode]
    totals, pairs = Counter(), defaultdict(Counter)
    rows = db.execute(f'SELECT COUNT(*), COUNT(DISTINCT {key}) FROM records WHERE input_status="ok"').fetchone()
    totals.update(parsed_rows=rows[0], distinct_reactions=rows[1])
    groups = db.execute(f'SELECT {key}, COUNT(*) FROM records WHERE input_status="ok" '
                        f'GROUP BY {key} HAVING COUNT(*)>1 ORDER BY {key}').fetchall()
    for index, (group_id, count) in enumerate(groups, 1):
        records = [dict(r) for r in db.execute(
            f'SELECT r.*,g.* FROM records r LEFT JOIN graphs g ON r.ordinal=g.record_id '
            f'WHERE r.{key}=? ORDER BY r.ordinal', (group_id,))]
        result = classify(records, mode, timeout)
        names = sorted({r['dataset'] for r in records})
        canonical = records[0]['canonical_rxn' if mode == 'exact' else 'normalized_rxn']
        if mode == 'undirected':
            canonical = '>>'.join(sorted(canonical.split('>>')))
        totals.update(overlap_groups=1, overlap_rows=count, rows_beyond_unique=count - 1,
                      cross_source_overlap_groups=len(names) > 1,
                      within_source_repeat_groups=len(names) < count,
                      its_disagreement_groups=result['conflict'],
                      cross_source_its_disagreement_groups=result['cross_conflict'],
                      hydrogen_only_disagreement_groups=result['hydrogen_disagreement'],
                      groups_with_unverifiable_members=any(m['status'] != 'ok' for m in result['members']))
        for a, b in combinations(names, 2):
            tally = pairs[a, b]
            tally['shared_reaction_groups'] += 1
            tally.update(source_pair(result, a, b))
        group = {'mode': mode, 'group_id': group_id, 'rxn': canonical, 'records': count,
                 'datasets': json.dumps(names), 'heavy_its_classes': result['classes'],
                 'cross_source_disagreement': result['cross_conflict'],
                 'confirmed_mapping_conflict': result['conflict'],
                 'constraints_unresolved': result['constraints_unresolved'],
                 'hydrogen_disagreement': result['hydrogen_disagreement'],
                 'unverifiable_records': sum(m['status'] != 'ok' for m in result['members'])}
        writers['groups'].writerow(group)
        for member in result['members']:
            writers['members'].writerow(dict(member, mode=mode, group_id=group_id))
        if result['conflict'] or result['hydrogen_disagreement']:
            examples = [{key: row[key] for key in ('dataset', 'r_id', 'original_id', 'role', 'steps',
                                                   'rxn', 'ground_truth')} | {'heavy_class': class_id}
                        for class_id, row in sorted(result['representatives'].items())]
            writers['conflicts'].write((json.dumps(dict(group, examples=examples)) + '\n').encode())
        if index % 500 == 0:
            print(f'{mode}: compared {index:,}/{len(groups):,} repeated reaction groups', flush=True)
    ordered = sorted(pairs.items(), key=lambda x: -x[1]['shared_reaction_groups'])
    return {'totals': dict(totals),
            'pairs': [{'first': a, 'second': b, **dict(c)} for (a, b), c in ordered]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT / 'Data/metadata/reconstruction/aam/registry.json')
    parser.add_argument('--input-root', type=Path, default=repo_path('Data/tables/development/aam'))
    parser.add_argument('--output', type=Path, default=repo_path('Data/tables/development/aam_overlap'))
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--isomorphism-timeout', type=float, default=5)
    args = parser.parse_args(argv)
    if min(args.workers, args.batch_size, args.isomorphism_timeout) <= 0:
        parser.error('Worker, batch and timeout values must be positive')
    registry = read_registry(args.registry)
    entries = [e for e in registry['datasets'] if not args.datasets or e['name'] in args.datasets]
    if not entries or args.datasets and set(args.datasets) - {e['name'] for e in entries}:
        parser.error('Unknown or empty dataset selection')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a fresh audit output directory; existing audit evidence is preserved')
    quiet()
    pins = {e['name']: digest(args.input_root / e['name'] / 'aam.csv.gz') for e in entries}
    for entry in entries:
        manifest = json.loads((args.input_root / entry['name'] / 'manifest.json').read_text())
        if pins[entry['name']] != manifest['files']['aam.csv.gz']:
            raise ValueError(f'Source checksum mismatch: {entry["name"]}')
    started = time.monotonic()
    db = sqlite3.connect(args.output / 'overlap.sqlite3')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    fields = list(TEXT_FIELDS) + list(INT_FIELDS)
    columns = ','.join([f'{k} TEXT' for k in TEXT_FIELDS] + [f'{k} INTEGER' for k in INT_FIELDS])
    db.execute(f'CREATE TABLE records(ordinal INTEGER PRIMARY KEY,{columns},UNIQUE(dataset,r_id))')
    db.execute('CREATE TABLE graphs(record_id INTEGER PRIMARY KEY,heavy_full BLOB,heavy_normalized BLOB,'
               'explicit_full BLOB,explicit_normalized BLOB,graph_status TEXT,graph_error TEXT)')
    with ExitStack() as stack:
        pool = (stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=quiet))
                if args.workers > 1 else None)
        insert = f'INSERT INTO records({",".join(fields)}) VALUES({",".join("?" for _ in fields)})'
        indexed = bounded_map(pool, index_record, source_rows(entries, args.input_root), args.batch_size)
        index = 0
        for index, row in enumerate(indexed, 1):
            db.execute(insert, [row[key] for key in fields])
            if index % 5000 == 0:
                db.commit()
                print(f'Indexed {index:,} source records', flush=True)
        db.commit()
        if index != sum(e['rows'] for e in entries):
            raise ValueError('Source row count differs from selected registry')
        for entry in entries:
            actual = db.execute('SELECT COUNT(*) FROM records WHERE dataset=?', (entry['name'],)).fetchone()[0]
            if actual != entry['rows']:
                raise ValueError(f'Source row count differs: {entry["name"]}')
        for key in ('exact_key', 'normalized_key', 'undirected_key'):
            db.execute(f'CREATE INDEX records_{key} ON records({key})')
        selected = db.execute('SELECT r.ordinal,r.ground_truth FROM records r JOIN '
                              '(SELECT undirected_key FROM records WHERE input_status="ok" GROUP BY undirected_key '
                              'HAVING COUNT(*)>1) g ON r.undirected_key=g.undirected_key '
                              'WHERE r.mapping_status IN ("ok","equivalent","partial") '
                              'AND (r.endpoint_match OR r.normalized_endpoint_match)')
        prepared = bounded_map(pool, graph_record, (dict(r) for r in selected), args.batch_size)
        for count, result in enumerate(prepared, 1):
            db.execute('INSERT INTO graphs VALUES(?,?,?,?,?,?,?)', result)
            if count % 2500 == 0:
                db.commit()
                print(f'Prepared {count:,} overlapping reference ITS graphs', flush=True)
        db.commit()
    with ExitStack() as stack:
        group_fields = ['mode', 'group_id', 'rxn', 'records', 'datasets', 'heavy_its_classes',
                        'cross_source_disagreement', 'hydrogen_disagreement', 'unverifiable_records',
                        'confirmed_mapping_conflict', 'constraints_unresolved']
        member_fields = ['mode', 'group_id', 'dataset', 'r_id', 'original_id', 'role', 'steps',
                         'heavy_class', 'constraint_class', 'is_anchor', 'explicit_class',
                         'explicit_representation_key', 'mapping_status', 'status',
                         'compatible_constraint_classes', 'incompatible_constraint_classes']
        writers = {'groups': csv_writer(stack, args.output / 'groups.csv.gz', group_fields),
                   'members': csv_writer(stack, args.output / 'members.csv.gz', member_fields)}
        raw = stack.enter_context((args.output / 'conflicts.jsonl.gz').open('wb'))
        writers['conflicts'] = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0))
        modes = {mode: analyze_mode(db, mode, args.isomorphism_timeout, writers) for mode in MODES}
        qc_fields = ['dataset', 'r_id', 'original_id', 'role', 'input_status', 'mapping_status',
                     'mapping_encoding', 'unmapped_heavy_atoms', 'explicit_mapping_status',
                     'endpoint_match', 'normalized_endpoint_match', 'error']
        qc = csv_writer(stack, args.output / 'record_qc.csv.gz', qc_fields)
        qc.writerows(dict(r) for r in db.execute('SELECT ' + ','.join(qc_fields) + ' FROM records ORDER BY ordinal'))
    pairs = ['mode', 'first', 'second', 'shared_reaction_groups', 'comparable_groups',
             'shared_mapping_groups', 'mapping_disagreement_groups', 'no_shared_mapping_groups']
    with (args.output / 'pairwise.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=pairs, restval=0)
        writer.writeheader()
        writer.writerows(dict(row, mode=mode) for mode, data in modes.items() for row in data['pairs'])
    query = ('SELECT dataset,input_status,mapping_status,COUNT(*) records '
             'FROM records GROUP BY dataset,input_status,mapping_status')
    quality = [dict(r) for r in db.execute(query)]
    graphs = [dict(r) for r in db.execute('SELECT graph_status,COUNT(*) records FROM graphs GROUP BY graph_status')]
    db.close()
    if pins != {e['name']: digest(args.input_root / e['name'] / 'aam.csv.gz') for e in entries}:
        raise ValueError('Source tables changed during the audit')
    report = {'schema': SCHEMA, 'records': index, 'datasets': [e['name'] for e in entries],
              'modes': modes, 'record_quality': quality, 'graph_preparation': graphs,
              'roles_used_for_grouping': False, 'source_tables_changed': False,
              'reaction_identity': 'canonical stereo-free unmapped endpoints; preserve charge, isotopes and tautomers',
              'normalized_mode': 'remove unchanged whole-component spectators; retain full identity reactions',
              'its_comparison': ('stereo-free heavy-atom ITS; equivalent labels retain distinct atoms and mapping '
                                 'classes; exact isomorphism plus bounded constraint compatibility/refinement'),
              'hydrogen_comparison': ('separate full represented-atom ITS, only with matching '
                                      'endpoint representations and complete maps'),
              'qualification': ('Mapping disagreement is not proof that either reference is wrong. '
                                'Invalid maps and mismatched endpoints are not called ITS conflicts.'),
              'source_sha256': pins, 'registry_sha256': digest(args.registry),
              'versions': {key: version(key) for key in ['rdkit', 'networkx']},
              'implementation_sha256': {
                  str(p.relative_to(ROOT)): digest(p) for p in
                  [Path(__file__).resolve(), ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_overlap_chem.py',
                   ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_mapping_constraints.py']},
              'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_json(args.output / 'summary.json', report)
    write_json(args.output / 'manifest.json', {'schema': SCHEMA, 'files': {
        p.name: digest(p) for p in sorted(args.output.iterdir()) if p.name != 'manifest.json'}})
    print(json.dumps({k: v['totals'] for k, v in modes.items()}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
