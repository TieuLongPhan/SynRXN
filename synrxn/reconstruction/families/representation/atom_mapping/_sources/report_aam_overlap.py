#!/usr/bin/env python
"""Produce review tables from a completed AAM overlap audit without changing source data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import gzip
import json
from pathlib import Path
import sqlite3

from synrxn.reconstruction.paths import REPO as ROOT, repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import csv_writer
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=repo_path('Data/tables/development/aam_overlap'))
    parser.add_argument('--output', type=Path, default=repo_path('Data/tables/development/aam_overlap_review'))
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'Data/metadata/reconstruction/aam/overlap-report.rst')
    args = parser.parse_args(argv)
    manifest = json.loads((args.audit / 'manifest.json').read_text())
    for filename, checksum in manifest['files'].items():
        if digest(args.audit / filename) != checksum:
            raise ValueError(f'Audit checksum mismatch: {filename}')
    report = json.loads((args.audit / 'summary.json').read_text())
    with gzip.open(args.audit / 'groups.csv.gz', 'rt') as handle:
        groups = {r['group_id']: r for r in csv.DictReader(handle) if r['mode'] == 'exact'}
    members = defaultdict(list)
    with gzip.open(args.audit / 'members.csv.gz', 'rt') as handle:
        for row in csv.DictReader(handle):
            if row['mode'] == 'exact':
                members[row['group_id']].append(row)
    db = sqlite3.connect(f'file:{args.audit / "overlap.sqlite3"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not (args.output / 'manifest.json').is_file():
        raise ValueError('Refusing to overwrite an unmanaged review folder')
    with ExitStack() as stack:
        fields = ['group_id', 'rxn', 'dataset', 'r_id', 'original_id', 'role', 'steps', 'heavy_class',
                  'mapping_status', 'comparison_status', 'ground_truth']
        conflicts = csv_writer(stack, args.output / 'conflicting_records.csv.gz', fields)
        for key, group in groups.items():
            if group['confirmed_mapping_conflict'] != 'True':
                continue
            for member in members[key]:
                record = db.execute('SELECT * FROM records WHERE dataset=? AND r_id=?',
                                    (member['dataset'], member['r_id'])).fetchone()
                row = {k: member.get(k, '') for k in fields}
                row.update(group_id=key, rxn=group['rxn'], comparison_status=member['status'],
                           ground_truth=record['ground_truth'])
                conflicts.writerow(row)
    ring = []
    for record in db.execute('SELECT * FROM records WHERE dataset="metambd_cyclic" ORDER BY ordinal'):
        key = record['exact_key']
        peers = members.get(key, [])
        this = next((r for r in peers if r['dataset'] == 'metambd_cyclic' and r['r_id'] == record['r_id']), {})
        other = [r for r in peers if r['dataset'] != 'metambd_cyclic']
        other_classes = {r['constraint_class'] for r in other if r['constraint_class'] != '' and r['status'] == 'ok'}
        own = this.get('heavy_class', '')
        compatible = {str(c) for c in json.loads(this.get('compatible_constraint_classes', '[]'))}
        incompatible = {str(c) for c in json.loads(this.get('incompatible_constraint_classes', '[]'))}
        comparable = this.get('status') == 'ok' and bool(other_classes & (compatible | incompatible))
        ring.append({'r_id': record['r_id'], 'original_id': record['original_id'], 'role': record['role'],
                     'group_id': key, 'rxn': record['canonical_rxn'],
                     'other_sources': json.dumps(sorted({r['dataset'] for r in other})),
                     'cross_source_overlap': bool(other), 'cross_source_comparable': comparable,
                     'cross_source_its_disagreement': comparable and bool(other_classes & incompatible),
                     'no_shared_mapping': comparable and other_classes <= incompatible,
                     'within_ring_repeat': sum(r['dataset'] == 'metambd_cyclic' for r in peers) > 1,
                     'heavy_class': own, 'comparison_status': this.get('status', 'unique_input_not_compared')})
    ring_fields = ['r_id', 'original_id', 'role', 'group_id', 'rxn', 'other_sources', 'cross_source_overlap',
                   'cross_source_comparable', 'cross_source_its_disagreement', 'no_shared_mapping',
                   'within_ring_repeat', 'heavy_class', 'comparison_status']
    with (args.output / 'ring_review.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=ring_fields)
        writer.writeheader()
        writer.writerows(ring)
    datasets = []
    for name in report['datasets']:
        counts = db.execute('SELECT COUNT(*) records,COUNT(DISTINCT exact_key) distinct_inputs,'
                            'SUM(mapping_status="ok") closed_maps,SUM(mapping_status="equivalent") class_maps,'
                            'SUM(mapping_status="partial") partial_maps,'
                            'SUM(mapping_status NOT IN ("ok","equivalent","partial")) ambiguous_maps,'
                            'SUM(endpoint_match=0) input_target_mismatches FROM records WHERE dataset=?',
                            (name,)).fetchone()
        internal = 0
        for rows in members.values():
            own = [r for r in rows if r['dataset'] == name and r['constraint_class'] != '']
            raw = {r['constraint_class'] for r in own}
            internal += any(raw & {str(c) for c in json.loads(r['incompatible_constraint_classes'])} for r in own)
        datasets.append(dict(counts, dataset=name, internal_its_disagreement_groups=internal))
    db.close()
    with (args.output / 'datasets.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(datasets[0]))
        writer.writeheader()
        writer.writerows(datasets)
    ring_stats = {'records': len(ring), 'distinct_inputs': len({r['group_id'] for r in ring}),
                  'cross_source_overlap_inputs': len({r['group_id'] for r in ring if r['cross_source_overlap']}),
                  'cross_source_comparable_inputs': len({r['group_id'] for r in ring if r['cross_source_comparable']}),
                  'cross_source_its_disagreement_inputs': len({r['group_id'] for r in ring
                                                              if r['cross_source_its_disagreement']}),
                  'no_shared_mapping_inputs': len({r['group_id'] for r in ring if r['no_shared_mapping']})}
    overview = {'records': report['records'], 'datasets': report['datasets'], 'ring': ring_stats,
                'exact': report['modes']['exact'], 'source_tables_changed': report['source_tables_changed'],
                'roles_used_for_grouping': report['roles_used_for_grouping']}
    write_json(args.output / 'overview.json', overview)
    totals = report['modes']['exact']['totals']
    text = ['AAM overlap and ITS results', '============================', '',
            f"Audited {report['records']:,} records across {len(report['datasets'])} active datasets.",
            'Train/test/validation roles were ignored for comparison. Source tables are unchanged.', '',
            f"There are {totals['distinct_reactions']:,} distinct directed canonical reaction inputs,",
            f"{totals['overlap_groups']:,} repeated-input groups and {totals['cross_source_overlap_groups']:,}",
            f"groups overlapping across sources. {totals['its_disagreement_groups']:,} groups contain",
            f"certified incompatible mappings; {totals['cross_source_its_disagreement_groups']:,} have cross-source",
            f"disagreements. {totals['groups_with_unverifiable_members']:,} repeated groups also contain",
            'unverifiable members. A disagreement is not a verdict about chemical correctness.', '',
            '.. list-table:: Exact directed overlap', '   :header-rows: 1', '',
            '   * - Source pair', '     - Shared inputs', '     - Comparable', '     - Different ITS',
            '     - No common ITS']
    for row in report['modes']['exact']['pairs']:
        text += [f"   * - {row['first']} / {row['second']}", f"     - {row['shared_reaction_groups']}",
                 f"     - {row.get('comparable_groups', 0)}", f"     - {row.get('mapping_disagreement_groups', 0)}",
                 f"     - {row.get('no_shared_mapping_groups', 0)}"]
    text += ['', 'Ring references', '----------------', '',
             f"The ring dataset retains {ring_stats['records']} first-reference records representing",
             f"{ring_stats['distinct_inputs']} distinct exact inputs. {ring_stats['cross_source_overlap_inputs']}",
             f"inputs overlap another source; {ring_stats['cross_source_comparable_inputs']} are comparable,",
             f"and {ring_stats['cross_source_its_disagreement_inputs']} have at least one ITS disagreement.",
             'See ring_review.csv for every ring record, native ID and comparison status.', '',
             'Reproduction and interpretation', '--------------------------------', '',
             'The pinned first-reference build is documented in Data/metadata/reconstruction/metamdb/processing.rst.',
             'The overlap method and partial-ITS/H/stereo treatment are in overlap-method.rst.',
             'Run synrxn.reconstruction.families.representation.atom_mapping._sources.audit_aam_overlap into a fresh folder, then synrxn.reconstruction.families.representation.atom_mapping._sources.report_aam_overlap.',
             'Exact identities ignore stereochemistry and preserve tautomers, protonation and isotopes.',
             'Other audit modes remove unchanged spectators and optionally allow reversal.',
             'No records were deleted, deduplicated, reassigned a reference, or given new splits.',
             'FlowER MC/pathway identity remains separate from endpoint/ITS equivalence.', '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(text))
    write_json(args.output / 'manifest.json', {
        'schema': 'synrxn.aam-overlap-review.v1', 'audit_manifest_sha256': digest(args.audit / 'manifest.json'),
        'report_script_sha256': digest(Path(__file__).resolve()),
        'files': {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.name != 'manifest.json'}})
    print(json.dumps(ring_stats, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
