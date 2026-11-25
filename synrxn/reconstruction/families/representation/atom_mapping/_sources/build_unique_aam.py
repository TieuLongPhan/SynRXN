#!/usr/bin/env python
"""Keep raw AAM references after deduplication and the agreed exclusions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import gzip
import hashlib
from importlib.metadata import version
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import time

from synrxn.reconstruction.paths import REPO as ROOT, source_path, repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources import audit_aam_overlap as audit
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import csv_writer
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, read_registry, write_json

SCHEMA = 'synrxn.aam-unique.v5'
INVENTORY = ['r_id', 'rxn', 'status', 'quality_flags', 'source_records', 'sources', 'heavy_its_classes',
             'hydrogen_disagreement', 'unverifiable_records', 'partial_references', 'source_flagged_records',
             'reference_ids', 'selected_reference_id', 'split_group_id', 'reference_comparable',
             'confirmed_mapping_conflict', 'constraints_unresolved']
CORE = ['r_id', 'ground_truth', 'rxn', 'original_id', 'reference_id', 'representative_source',
        'representative_r_id', 'sources', 'source_records', 'mapping_scope', 'reference_tier',
        'validation_status', 'split_group_id']
RETAINED = CORE + ['mapping_status', 'comparison_status', 'quality_flags', 'source_validation_status']
REFERENCES = RETAINED + ['heavy_class', 'source_mapping_status']
MEMBERS = ['r_id', 'reference_id', 'dataset', 'source_r_id', 'source_row', 'original_id', 'role', 'steps',
           'heavy_class', 'constraint_class', 'is_anchor', 'explicit_class', 'explicit_representation_key',
           'mapping_status', 'mapping_encoding', 'unmapped_heavy_atoms',
           'comparison_status', 'comparison_error', 'source_exclusion', 'source_reference_tier',
           'source_validation_status', 'compatible_constraint_classes', 'incompatible_constraint_classes']
EXCLUDED = ['r_id', 'rxn', 'exclusion_reason', 'source_records', 'sources', 'split_group_id',
            'heavy_its_classes', 'hydrogen_disagreement', 'unverifiable_records', 'confirmed_mapping_conflict']
HYDROGEN = ['r_id', 'dataset', 'source_r_id', 'original_id', 'role', 'steps', 'source_parent_group',
            'source_member', 'source_lines', 'explicit_class', 'explicit_representation_key', 'ground_truth']


def is_identity(rxn):
    left, right = rxn.split('>>')
    return left == right


def lines(stack, path):
    raw = stack.enter_context(path.open('wb'))
    return stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0))


def source_evidence(row):
    """Rank explicit source declarations; never infer review from a dataset's name."""
    status = row.get('validation_status', '')
    excluded = status if status == 'qc_failed' or status.startswith('reference_excluded_') else ''
    tier = row.get('reference_tier', '')
    if status == 'human_review_verified':
        rank = 0
    elif row.get('mapping_curated', '').lower() == 'true' or tier == 'upstream_user_curated':
        rank = 1
    elif tier == 'author_supplied_first_reference':
        rank = 2
    elif tier == 'upstream_generated_mapping':
        rank = 4
    else:
        rank = 3
    return rank, excluded


def prepare(row):
    try:
        info = chem.reference_info(row['ground_truth'])
        heavy = chem.pack(chem.its_graph(info['mols']))
        explicit = (chem.pack(chem.its_graph(info['mols'], heavy=False))
                    if info['explicit_mapping_status'] in {'ok', 'equivalent', 'partial'}
                    and info['explicit_h_atoms'] else None)
        return row['ordinal'], heavy, explicit, 'ok', ''
    except Exception as error:
        return row['ordinal'], None, None, 'graph_failed', str(error)[:220]


def checked_inputs(args):
    manifest_path = args.audit / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, checksum in manifest['files'].items():
        if digest(args.audit / name) != checksum:
            raise ValueError(f'Audit checksum mismatch: {name}')
    report = json.loads((args.audit / 'summary.json').read_text())
    if report['schema'] != audit.SCHEMA or report['registry_sha256'] != digest(args.registry):
        raise ValueError('Audit registry or schema mismatch; rebuild the overlap audit')
    if report['roles_used_for_grouping'] or report['source_tables_changed']:
        raise ValueError('Audit must preserve sources and ignore original split roles')
    for name, checksum in report['implementation_sha256'].items():
        if digest(source_path(name, ROOT)) != checksum:
            raise ValueError(f'Audit implementation changed: {name}')
    for name, release in report['versions'].items():
        if version(name) != release:
            raise ValueError(f'Audit dependency changed: {name}; rebuild the overlap audit')
    entries = read_registry(args.registry)['datasets']
    if [e['name'] for e in entries] != report['datasets']:
        raise ValueError('Use an audit covering every active AAM dataset')
    pins = {e['name']: digest(args.input_root / e['name'] / 'aam.csv.gz') for e in entries}
    if pins != report['source_sha256']:
        raise ValueError('Source tables differ from the completed overlap audit')
    return entries, report, digest(manifest_path)


def metadata(db, entries, input_root):
    db.execute('CREATE TABLE sources(record_id INTEGER PRIMARY KEY,record_json TEXT,evidence_rank INTEGER,'
               'source_exclusion TEXT,excluded_identity INTEGER)')
    for entry in entries:
        count = 0
        with gzip.open(input_root / entry['name'] / 'aam.csv.gz', 'rt', newline='') as handle:
            for count, row in enumerate(csv.DictReader(handle), 1):
                record = db.execute('SELECT ordinal,source_row,canonical_rxn FROM upstream.records '
                                    'WHERE dataset=? AND r_id=?',
                                    (entry['name'], row['r_id'])).fetchone()
                if record is None or record['source_row'] != count:
                    raise ValueError('Source row identity differs from audit index')
                rank, excluded = source_evidence(row)
                db.execute('INSERT INTO sources VALUES(?,?,?,?,?)',
                           (record['ordinal'], chem.encode(row), rank, excluded,
                            is_identity(record['canonical_rxn'])))
        if count != entry['rows']:
            raise ValueError(f'Source count mismatch: {entry["name"]}')
        db.commit()


def graph_cache(db, args):
    db.execute('CREATE TABLE prepared(record_id INTEGER PRIMARY KEY,heavy_full BLOB,explicit_full BLOB,'
               'graph_status TEXT,graph_error TEXT)')
    selected = db.execute('SELECT r.ordinal,r.ground_truth FROM upstream.records r '
                          'JOIN sources s ON r.ordinal=s.record_id LEFT JOIN upstream.graphs g '
                          'ON r.ordinal=g.record_id WHERE s.excluded_identity=0 '
                          'AND g.record_id IS NULL AND r.input_status="ok" '
                          'AND r.mapping_status IN ("ok","equivalent","partial") '
                          'AND r.endpoint_match=1 ORDER BY r.ordinal')
    with ExitStack() as stack:
        pool = (stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=audit.quiet))
                if args.workers > 1 else None)
        results = audit.bounded_map(pool, prepare, (dict(r) for r in selected), args.batch_size)
        count = 0
        for count, row in enumerate(results, 1):
            db.execute('INSERT INTO prepared VALUES(?,?,?,?,?)', row)
            if count % 5000 == 0:
                db.commit()
                print(f'Checked {count:,} additional reference ITS graphs', flush=True)
        db.commit()
    return count


def representative(rows):
    rows = [r for r in rows if r.get('reference_anchor')] or rows
    ranks = {'ok': 0, 'equivalent': 1, 'partial': 2}
    return min(rows, key=lambda r: (ranks.get(r['mapping_status'], 3), bool(r['source_exclusion']),
                                    r['evidence_rank'], r['dataset'], r['source_row']))


def mapping_scope(row, member):
    if member['status'] != 'ok':
        return 'unverified_heavy_atoms'
    scope = 'heavy_atom_equivalence_classes' if row['mapping_encoding'] == 'equivalence_classes' else 'heavy_atoms'
    return 'partial_' + scope if row['mapping_status'] == 'partial' else scope


def write_member(row, member, r_id, reference_id, writers, source_pins, prefix=''):
    source = json.loads(row['record_json'])
    writers[prefix + 'members'].writerow({
        'r_id': r_id, 'reference_id': reference_id, 'dataset': row['dataset'], 'source_r_id': row['r_id'],
        'source_row': row['source_row'], 'original_id': row['original_id'], 'role': row['role'],
        'steps': row['steps'], 'heavy_class': member['heavy_class'], 'explicit_class': member['explicit_class'],
        'constraint_class': member.get('constraint_class'), 'is_anchor': member.get('is_anchor', False),
        'explicit_representation_key': member['explicit_representation_key'], 'mapping_status': row['mapping_status'],
        'mapping_encoding': row['mapping_encoding'], 'unmapped_heavy_atoms': row['unmapped_heavy_atoms'],
        'compatible_constraint_classes': member.get('compatible_constraint_classes', '[]'),
        'incompatible_constraint_classes': member.get('incompatible_constraint_classes', '[]'),
        'comparison_status': member['status'], 'comparison_error': row['graph_error'] or row['error'],
        'source_exclusion': row['source_exclusion'], 'source_reference_tier': source.get('reference_tier', ''),
        'source_validation_status': source.get('validation_status', '')})
    writers[prefix + 'provenance'].write((chem.encode({
        'r_id': r_id, 'reference_id': reference_id, 'dataset': row['dataset'], 'source_r_id': row['r_id'],
        'source_row': row['source_row'], 'source_table_sha256': source_pins[row['dataset']],
        'source_record': source}) + '\n').encode())


def emit_excluded(rows, reason, writers, totals, by_source, source_pins, result=None):
    first = rows[0]
    r_id = 'aam_' + first['exact_key']
    writers['excluded'].writerow({'r_id': r_id, 'rxn': first['canonical_rxn'],
                                  'exclusion_reason': reason, 'source_records': len(rows),
                                  'sources': chem.encode(sorted({r['dataset'] for r in rows})),
                                  'split_group_id': first['undirected_key'],
                                  'heavy_its_classes': result['classes'] if result else '',
                                  'hydrogen_disagreement': result['hydrogen_disagreement'] if result else '',
                                  'confirmed_mapping_conflict': result['conflict'] if result else False,
                                  'unverifiable_records': sum(m['status'] != 'ok' for m in result['members'])
                                  if result else ''})
    for index, row in enumerate(rows):
        member = (result['members'][index] if result else {
            'heavy_class': None, 'explicit_class': None, 'explicit_representation_key': '',
            'status': 'excluded_identity_reaction'})
        write_member(row, member, r_id, '', writers, source_pins, 'excluded_')
    key = ('excluded_identity_reactions' if reason == 'identical_canonical_endpoints'
           else 'excluded_mapping_conflict_reactions')
    totals.update({key: 1, 'excluded_reactions': 1, 'excluded_source_records': len(rows)})
    for name in {r['dataset'] for r in rows}:
        by_source[name].update({key: 1, 'excluded_reactions': 1,
                                'excluded_source_records': sum(r['dataset'] == name for r in rows)})


def emit_group(rows, args, writers, totals, by_source, source_pins):
    if is_identity(rows[0]['canonical_rxn']):
        emit_excluded(rows, 'identical_canonical_endpoints', writers, totals, by_source, source_pins)
        return
    first = rows[0]
    group_id = first['exact_key']
    r_id = 'aam_' + group_id
    result = audit.classify(rows, 'exact', args.isomorphism_timeout)
    if result['conflict']:
        emit_excluded(rows, 'conflicting_heavy_atom_its', writers, totals, by_source, source_pins, result)
        return
    classes = {}
    member_by_ordinal = {}
    for row, member in zip(rows, result['members']):
        row['reference_anchor'] = member['is_anchor']
        member_by_ordinal[row['ordinal']] = member
        if member['heavy_class'] is not None:
            classes.setdefault(member['heavy_class'], []).append(row)
    unverifiable = sum(m['status'] != 'ok' for m in result['members'])
    partial = sum(r['mapping_status'] == 'partial' for r in rows)
    flagged = sum(bool(r['source_exclusion']) for r in rows)
    conditions = [('mapping_constraints_unresolved', result['constraints_unresolved']),
                  ('unverifiable_reference', unverifiable > 0),
                  ('partial_mapping', partial > 0), ('import_qc_flag', flagged > 0)]
    reasons = [name for name, applies in conditions if applies]
    if not classes and 'unverifiable_reference' not in reasons:
        reasons.append('unverifiable_reference')
    class_count = len(classes)
    # An unverified raw target is retained without calling it an ITS class.
    if not classes:
        classes[None] = rows
    selected = representative([r for variants in classes.values() for r in variants])
    selected_member = member_by_ordinal[selected['ordinal']]
    comparable = selected_member['status'] == 'ok' and not result['constraints_unresolved']
    names = sorted({r['dataset'] for r in rows})
    reference_ids = {c: r_id + (f':its{c}' if c is not None else ':raw') for c in classes}
    selected_id = reference_ids[selected_member['heavy_class']]
    inventory = {'r_id': r_id, 'rxn': first['canonical_rxn'], 'status': reasons[0] if reasons else 'comparable',
                 'quality_flags': chem.encode(reasons), 'source_records': len(rows), 'sources': chem.encode(names),
                 'heavy_its_classes': class_count, 'hydrogen_disagreement': result['hydrogen_disagreement'],
                 'unverifiable_records': unverifiable, 'partial_references': partial,
                 'source_flagged_records': flagged, 'reference_ids': chem.encode(list(reference_ids.values())),
                 'selected_reference_id': selected_id,
                 'split_group_id': first['undirected_key'], 'reference_comparable': comparable}
    inventory.update(confirmed_mapping_conflict=result['conflict'],
                     constraints_unresolved=result['constraints_unresolved'])
    writers['inventory'].writerow(inventory)
    if reasons:
        writers['diagnostics'].writerow(inventory)
    for class_id, variants in classes.items():
        best = representative(variants)
        member = member_by_ordinal[best['ordinal']]
        source = json.loads(best['record_json'])
        record = {'r_id': r_id, 'ground_truth': best['ground_truth'], 'rxn': first['canonical_rxn'],
                  'original_id': best['dataset'] + ':' + (best['original_id'] or best['r_id']),
                  'reference_id': reference_ids[class_id], 'representative_source': best['dataset'],
                  'representative_r_id': best['r_id'], 'sources': chem.encode(sorted({r['dataset'] for r in variants})),
                  'source_records': len(variants),
                  'mapping_scope': mapping_scope(best, member),
                  'reference_tier': 'merged_source_reference',
                  'validation_status': 'raw_source_reference_retained',
                  'split_group_id': first['undirected_key'], 'mapping_status': best['mapping_status'],
                  'comparison_status': member['status'], 'quality_flags': chem.encode(reasons),
                  'source_validation_status': source.get('validation_status', '')}
        writers['references'].writerow(dict(record, heavy_class=class_id,
                                            source_mapping_status=best['mapping_status']))
        if best['ordinal'] == selected['ordinal']:
            writers['retained'].writerow(record)
    for row, member in zip(rows, result['members']):
        reference_id = reference_ids.get(member['heavy_class'], '')
        write_member(row, member, r_id, reference_id, writers, source_pins)
        if result['hydrogen_disagreement']:
            source = json.loads(row['record_json'])
            record = {k: source.get(k, '') for k in HYDROGEN}
            record.update(r_id=r_id, dataset=row['dataset'], source_r_id=row['r_id'],
                          explicit_class=member['explicit_class'],
                          explicit_representation_key=member['explicit_representation_key'])
            writers['hydrogen'].writerow(record)
    totals.update(unique_reactions=1, source_records=len(rows), reference_classes=class_count,
                  reference_entries=len(classes), retained_reactions=1, diagnostic_reactions=bool(reasons),
                  reference_comparable_reactions=comparable, raw_only_reference_reactions=class_count == 0,
                  duplicate_records_merged=len(rows) - 1, mapping_conflict_reactions=result['conflict'],
                  hydrogen_conflict_reactions=result['hydrogen_disagreement'],
                  cross_source_overlap_reactions=len(names) > 1)
    for reason in reasons:
        totals['flag_' + reason] += 1
    totals['selected_mapping_status_' + selected['mapping_status']] += 1
    totals['selected_mapping_scope_' + mapping_scope(selected, selected_member)] += 1
    for name in names:
        by_source[name].update(unique_reactions=1, retained_reactions=1, diagnostic_reactions=bool(reasons),
                               reference_comparable_reactions=comparable,
                               source_records=sum(r['dataset'] == name for r in rows))


def csv_rows(path):
    with gzip.open(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def verify_pins(output, input_root, audit_root, registry):
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unknown unique AAM schema')
    files = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if files != manifest['files']:
        raise ValueError('Unique AAM payload checksum mismatch')
    summary = json.loads((output / 'summary.json').read_text())
    if digest(audit_root / 'manifest.json') != summary['audit_manifest_sha256']:
        raise ValueError('Upstream overlap audit changed')
    if digest(registry) != summary['registry_sha256']:
        raise ValueError('Active AAM registry changed')
    for name, checksum in summary['source_sha256'].items():
        if digest(input_root / name / 'aam.csv.gz') != checksum:
            raise ValueError(f'Original source changed: {name}')
    for name, checksum in summary['implementation_sha256'].items():
        if digest(source_path(name, ROOT)) != checksum:
            raise ValueError(f'Unique AAM implementation changed: {name}')
    return summary


def inventory_index(output):
    inventory, diagnostics, expected_references, counts = {}, set(), set(), Counter()
    for row in csv_rows(output / 'reactions.csv.gz'):
        if is_identity(row['rxn']):
            raise ValueError('Identical endpoints remain in the active reaction pool')
        if row['confirmed_mapping_conflict'] == 'True':
            raise ValueError('Conflicting heavy-atom ITS references remain in the active pool')
        if row['r_id'] in inventory or row['r_id'] != 'aam_' + chem.sha(row['rxn']):
            raise ValueError('Repeated or unstable reaction identity')
        inventory[row['r_id']] = row
        if json.loads(row['quality_flags']):
            diagnostics.add(row['r_id'])
        counts['reference_comparable_reactions'] += row['reference_comparable'] == 'True'
        counts['raw_only_reference_reactions'] += row['heavy_its_classes'] == '0'
        expected_references.update(json.loads(row['reference_ids']))
    return inventory, diagnostics, expected_references, counts


def verify_inventory(output):
    inventory, diagnostics, expected_references, counts = inventory_index(output)
    ids = set(inventory)
    for filename, expected in [('aam.csv.gz', ids), ('diagnostics.csv.gz', diagnostics)]:
        seen = set()
        for row in csv_rows(output / filename):
            if row['r_id'] in seen or row['r_id'] not in expected:
                raise ValueError(f'Duplicate or unexpected reaction in {filename}')
            seen.add(row['r_id'])
            entry = inventory[row['r_id']]
            if row['rxn'] != entry['rxn'] or row['quality_flags'] != entry['quality_flags']:
                raise ValueError('Retained reaction lost its input identity or annotations')
            if filename == 'aam.csv.gz':
                if (row['reference_id'] != entry['selected_reference_id']
                        or row['reference_id'] not in expected_references or not row['ground_truth'] or 'role' in row):
                    raise ValueError('Retained table contains an invalid source-reference join')
                comparable = row['comparison_status'] == 'ok' and entry['constraints_unresolved'] != 'True'
                if comparable != (entry['reference_comparable'] == 'True'):
                    raise ValueError('Retained reference comparability annotation differs from inventory')
        if seen != expected:
            raise ValueError(f'Missing unique reactions in {filename}')
    references = set()
    for row in csv_rows(output / 'references.csv.gz'):
        if row['reference_id'] in references or row['r_id'] not in ids:
            raise ValueError('Duplicate or orphan reference')
        references.add(row['reference_id'])
        counts['reference_classes'] += row['heavy_class'] != ''
    if references != expected_references:
        raise ValueError('Reference classes do not cover the inventory')
    return ids, diagnostics, references, counts


def verify_excluded(output, active_ids):
    ids, reasons = set(), Counter()
    for row in csv_rows(output / 'excluded_reactions.csv.gz'):
        if row['r_id'] != 'aam_' + chem.sha(row['rxn']) or row['r_id'] in ids or row['r_id'] in active_ids:
            raise ValueError('Invalid or repeated exclusion')
        reason = row['exclusion_reason']
        if reason == 'identical_canonical_endpoints':
            if not is_identity(row['rxn']):
                raise ValueError('Identity exclusion has different endpoints')
        elif reason == 'conflicting_heavy_atom_its':
            if (is_identity(row['rxn']) or int(row['heavy_its_classes']) < 2
                    or row['confirmed_mapping_conflict'] != 'True'):
                raise ValueError('Conflict exclusion has no distinct heavy-atom ITS classes')
        else:
            raise ValueError('Unknown exclusion rule')
        ids.add(row['r_id'])
        reasons[reason] += 1
    return ids, reasons


def source_fingerprints(output, input_root, summary):
    """Verify raw target preservation while fingerprinting every original row."""
    source_records, targets = {}, {}
    for row in csv_rows(output / 'references.csv.gz'):
        targets.setdefault((row['representative_source'], row['representative_r_id']), []).append(row)
    for name in summary['source_counts']:
        for index, row in enumerate(csv_rows(input_root / name / 'aam.csv.gz'), 1):
            source_records[name, row['r_id']] = chem.sha(chem.encode(row)), index
            for target in targets.pop((name, row['r_id']), []):
                if (target['ground_truth'] != row['ground_truth']
                        or target['source_validation_status'] != row.get('validation_status', '')):
                    raise ValueError('Raw source reference or its original QC annotation was changed')
    if targets:
        raise ValueError('Retained reference has no original source record')
    reference_rows = {r['reference_id']: r for r in csv_rows(output / 'references.csv.gz')}
    for row in csv_rows(output / 'aam.csv.gz'):
        if row != {key: reference_rows[row['reference_id']][key] for key in RETAINED}:
            raise ValueError('Selected target differs from its preserved source reference')
    return source_records


def verify_members(output, input_root, summary, ids, references, excluded):
    member_keys, counts, joins = set(), Counter(), hashlib.sha256()
    cohort_counts = Counter()
    for prefix, allowed in [('', ids), ('excluded_', excluded)]:
        for row in csv_rows(output / (prefix + 'members.csv.gz')):
            key = row['dataset'], row['source_r_id']
            if key in member_keys or row['r_id'] not in allowed:
                raise ValueError('Repeated or orphan source membership')
            if row['reference_id'] and row['reference_id'] not in references:
                raise ValueError('Orphan source reference')
            if prefix and row['reference_id']:
                raise ValueError('Excluded input has an active reference')
            member_keys.add(key)
            counts[row['dataset']] += 1
            cohort_counts[prefix] += 1
            joins.update(chem.encode([*key, row['r_id'], row['reference_id']]).encode())
    source_records = source_fingerprints(output, input_root, summary)
    provenance, witnessed = hashlib.sha256(), Counter()
    for prefix in ['', 'excluded_']:
        with gzip.open(output / (prefix + 'provenance.jsonl.gz'), 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                if row['source_record']['r_id'] != row['source_r_id']:
                    raise ValueError('Original source ID changed in provenance')
                fingerprint = source_records.pop((row['dataset'], row['source_r_id']), None)
                if fingerprint != (chem.sha(chem.encode(row['source_record'])), row['source_row']):
                    raise ValueError('Original source record changed in provenance')
                provenance.update(chem.encode([row['dataset'], row['source_r_id'],
                                              row['r_id'], row['reference_id']]).encode())
                witnessed[row['dataset']] += 1
    if (source_records or joins.digest() != provenance.digest() or counts != witnessed
            or dict(counts) != summary['source_counts']):
        raise ValueError('Source memberships and provenance do not cover every original record')
    return cohort_counts


def verify(output, input_root, audit_root, registry):
    summary = verify_pins(output, input_root, audit_root, registry)
    ids, diagnostics, references, counts = verify_inventory(output)
    excluded, exclusions = verify_excluded(output, ids)
    source_count = verify_members(output, input_root, summary, ids, references, excluded)
    totals = summary['totals']
    actual = (len(ids), len(ids), len(diagnostics), counts['reference_classes'], len(references),
              counts['reference_comparable_reactions'], counts['raw_only_reference_reactions'], source_count[''],
              len(excluded), exclusions['identical_canonical_endpoints'], exclusions['conflicting_heavy_atom_its'],
              source_count['excluded_'], sum(source_count.values()), len(ids) + len(excluded))
    count_fields = ['unique_reactions', 'retained_reactions', 'diagnostic_reactions', 'reference_classes',
                    'reference_entries', 'reference_comparable_reactions', 'raw_only_reference_reactions',
                    'source_records',
                    'excluded_reactions', 'excluded_identity_reactions', 'excluded_mapping_conflict_reactions',
                    'excluded_source_records', 'input_source_records', 'input_unique_reactions']
    expected = tuple(totals[k] for k in count_fields)
    if actual != expected:
        raise ValueError('Unique AAM summary count mismatch')
    return dict(zip(count_fields, actual), verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=repo_path('Data/tables/development/aam_overlap'))
    parser.add_argument('--input-root', type=Path, default=repo_path('Data/tables/development/aam'))
    parser.add_argument('--registry', type=Path, default=ROOT / 'Data/metadata/reconstruction/aam/registry.json')
    parser.add_argument('--output', type=Path, default=repo_path('Data/tables/development/aam_unique'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--isomorphism-timeout', type=float, default=5)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(args.output, args.input_root, args.audit, args.registry), indent=2))
        return 0
    if min(args.workers, args.batch_size, args.isomorphism_timeout) <= 0:
        parser.error('Worker, batch and timeout values must be positive')
    entries, upstream, audit_pin = checked_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a fresh output directory; existing unique AAM evidence is preserved')
    started = time.monotonic()
    audit.quiet()
    db_path = args.output / '.work.sqlite3'
    db = sqlite3.connect(db_path, uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('ATTACH DATABASE ? AS upstream', (f'file:{args.audit / "overlap.sqlite3"}?mode=ro',))
    if db.execute('SELECT COUNT(*) FROM upstream.records WHERE input_status!="ok"').fetchone()[0]:
        raise ValueError('Unparseable inputs require review before unique-reaction construction')
    metadata(db, entries, args.input_root)
    prepared = graph_cache(db, args)
    count_fields = ['unique_reactions', 'retained_reactions', 'diagnostic_reactions', 'reference_classes',
                    'reference_entries', 'reference_comparable_reactions', 'raw_only_reference_reactions',
                    'source_records',
                    'excluded_identity_reactions', 'excluded_mapping_conflict_reactions', 'excluded_reactions',
                    'excluded_source_records', 'mapping_conflict_reactions',
                    'hydrogen_conflict_reactions', 'duplicate_records_merged', 'cross_source_overlap_reactions']
    totals = Counter({key: 0 for key in count_fields})
    by_source = {e['name']: Counter() for e in entries}
    query = ('SELECT r.*,s.*,COALESCE(g.heavy_full,p.heavy_full) heavy_full,'
             'COALESCE(g.explicit_full,p.explicit_full) explicit_full,'
             'COALESCE(g.graph_status,p.graph_status) graph_status,'
             'COALESCE(g.graph_error,p.graph_error) graph_error '
             'FROM upstream.records r JOIN sources s ON r.ordinal=s.record_id '
             'LEFT JOIN upstream.graphs g ON r.ordinal=g.record_id '
             'LEFT JOIN prepared p ON r.ordinal=p.record_id ORDER BY r.exact_key,r.ordinal')
    with ExitStack() as stack:
        writers = {'inventory': csv_writer(stack, args.output / 'reactions.csv.gz', INVENTORY),
                   'retained': csv_writer(stack, args.output / 'aam.csv.gz', RETAINED),
                   'diagnostics': csv_writer(stack, args.output / 'diagnostics.csv.gz', INVENTORY),
                   'references': csv_writer(stack, args.output / 'references.csv.gz', REFERENCES),
                   'members': csv_writer(stack, args.output / 'members.csv.gz', MEMBERS),
                   'provenance': lines(stack, args.output / 'provenance.jsonl.gz'),
                   'excluded': csv_writer(stack, args.output / 'excluded_reactions.csv.gz', EXCLUDED),
                   'excluded_members': csv_writer(stack, args.output / 'excluded_members.csv.gz', MEMBERS),
                   'excluded_provenance': lines(stack, args.output / 'excluded_provenance.jsonl.gz'),
                   'hydrogen': csv_writer(stack, args.output / 'hydrogen_conflicts.csv.gz', HYDROGEN)}
        for index, (_, group) in enumerate(groupby(db.execute(query), key=lambda r: r['exact_key']), 1):
            emit_group([dict(r) for r in group], args, writers, totals, by_source, upstream['source_sha256'])
            if index % 10000 == 0:
                print(f'Grouped {index:,} unique reactions', flush=True)
    db.close()
    db_path.unlink()
    totals['input_source_records'] = totals['source_records'] + totals['excluded_source_records']
    totals['input_unique_reactions'] = totals['unique_reactions'] + totals['excluded_reactions']
    if totals['input_source_records'] != upstream['records'] or totals['input_unique_reactions'] != (
            upstream['modes']['exact']['totals']['distinct_reactions']):
        raise ValueError('Unique reaction grouping lost original records')
    excluded_conflicts = sum(r['mode'] == 'exact' and is_identity(r['rxn'])
                             and r['confirmed_mapping_conflict'] == 'True'
                             for r in csv_rows(args.audit / 'groups.csv.gz'))
    if (totals['mapping_conflict_reactions'] + totals['excluded_mapping_conflict_reactions'] + excluded_conflicts
            != upstream['modes']['exact']['totals']['its_disagreement_groups']):
        raise ValueError('Repeated-input ITS classes differ from the original audit')
    if upstream['source_sha256'] != {e['name']: digest(args.input_root / e['name'] / 'aam.csv.gz') for e in entries}:
        raise ValueError('Source tables changed during unique AAM construction')
    if digest(args.audit / 'manifest.json') != audit_pin:
        raise ValueError('Overlap audit changed during unique AAM construction')
    implementation = [Path(__file__).resolve(), ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/audit_aam_overlap.py',
                      ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_overlap_chem.py',
                      ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_mapping_constraints.py']
    summary = {'schema': SCHEMA, 'totals': dict(totals), 'by_source': {k: dict(v) for k, v in by_source.items()},
               'source_counts': {e['name']: e['rows'] for e in entries}, 'source_sha256': upstream['source_sha256'],
               'audit_manifest_sha256': audit_pin, 'registry_sha256': digest(args.registry),
               'implementation_sha256': {str(p.relative_to(ROOT)): digest(p) for p in implementation},
               'versions': upstream['versions'], 'roles_used_for_grouping': False, 'new_splits_assigned': False,
               'excluded_audited_mapping_conflict_reactions': excluded_conflicts,
               'source_tables_changed': False, 'additional_reference_graphs_checked': prepared,
               'identity': 'directed canonical stereo-free unmapped endpoints; preserve isotopes, charge, tautomers',
               'mapping_scope': 'raw heavy-atom mappings: complete, equivalence-class, partial or unverified; '
                                'explicit-H alternatives remain diagnostic',
               'pool_exclusion_policy': ['identical canonical endpoints', 'certified incompatible mapping constraints'],
               'retention_policy': ['one raw source target per unique directed input',
                                    'partial maps, local import QC and unverifiable references are retained',
                                    'diagnostic flags are annotations, not exclusion rules'],
               'representative_policy': 'prefer comparable common/refining maps; complete maps, no import QC flag, '
                                        'explicit curation declarations, '
                                        'author-supplied references, unspecified/database, generated; '
                                        'ties use source name and row, never dataset size',
               'qualification': 'Retention does not establish technical or chemical correctness or human validation. '
                                'Source agreement is not assumed independent. No conflict winner is selected.',
               'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_json(args.output / 'summary.json', summary)
    source_snapshot = {'schema': SCHEMA, 'datasets': entries, 'source_tables_root': str(args.input_root)}
    write_json(args.output / 'sources.json', source_snapshot)
    (args.output / 'NOTICE.txt').write_text(
        'Local derived AAM comparison pool. Each exact directed canonical input appears once.\n'
        'aam.csv.gz retains one original source target for every unique active input.\n'
        'Identical canonical endpoints are excluded; excluded_* files retain the ledger.\n'
        'Inputs with certified incompatible mapping constraints are excluded; source alternatives remain logged.\n'
        'Repeated labels encode mapping equivalence classes; missing labels are partial; stereo is ignored.\n'
        'Partial, equivalence-class and locally QC-flagged raw mappings remain included.\n'
        'Unverifiable raw targets remain included with explicit comparison_status and unverified mapping scope.\n'
        'diagnostics.csv.gz annotates retained inputs; it is not an exclusion or holdout list.\n'
        'references.csv.gz preserves ITS classes and explicitly unverified raw fallback references.\n'
        'Active and excluded sidecars preserve all original records,\n'
        'including explicit-H alternatives, source roles, FlowER step/center/parent metadata and source QC flags.\n'
        'hydrogen_conflicts.csv.gz traces the retained explicit-H-only disagreements to original source records.\n'
        'Explicit-H-only alternatives remain included in aam.csv.gz.\n'
        'No new split, tautomer normalization, reaction reversal, remapping or human-validation claim is made.\n'
        'split_group_id groups spectator-normalized reactions in either direction for a later split;\n'
        'FlowER shared mechanism/parent provenance must also be considered when designing that split.\n'
        'Applicable source terms, partitions and citations remain per source; see sources.json and\n'
        'the unchanged source dataset source.json, NOTICE.txt and licenses/. No new redistribution grant is made.\n'
        'Do not treat multiple sources with shared ancestry as independent mapping votes.\n')
    write_json(args.output / 'manifest.json', {'schema': SCHEMA, 'files': {
        p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != 'manifest.json'}})
    result = verify(args.output, args.input_root, args.audit, args.registry)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
