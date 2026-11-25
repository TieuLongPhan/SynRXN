#!/usr/bin/env python
"""Record a read-only AAM cleaning checkpoint and per-source segmentation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import gzip
import json
from pathlib import Path

from synrxn.reconstruction.paths import REPO as ROOT
from synrxn.reconstruction.paths import repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, write_json

FIELDS = ['dataset', 'original_rows', 'excluded_source_rows', 'excluded_unchanged_source_rows',
          'excluded_conflicting_source_rows', 'excluded_unchanged_unique_reactions',
          'excluded_conflicting_unique_reactions', 'retained_source_rows',
          'within_source_duplicates_merged', 'retained_unique_source_reactions',
          'selected_pool_rows', 'represented_by_other_sources', 'selected_complete_individual',
          'selected_complete_classes', 'selected_partial', 'selected_unverified']


def rows(path):
    with gzip.open(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def checked_manifest(folder):
    value = json.loads((folder / 'manifest.json').read_text())
    for name, checksum in value['files'].items():
        if digest(folder / name) != checksum:
            raise ValueError(f'Input checksum mismatch: {folder / name}')
    return digest(folder / 'manifest.json')


def segmentation(pool, summary):
    memberships, source_rows, selected, scopes = defaultdict(set), Counter(), Counter(), defaultdict(Counter)
    exclusion_reasons = {r['r_id']: r['exclusion_reason'] for r in rows(pool / 'excluded_reactions.csv.gz')}
    excluded_rows = defaultdict(Counter)
    for row in rows(pool / 'excluded_members.csv.gz'):
        excluded_rows[row['dataset']][exclusion_reasons[row['r_id']]] += 1
    for row in rows(pool / 'members.csv.gz'):
        memberships[row['dataset']].add(row['r_id'])
        source_rows[row['dataset']] += 1
    selected_ids = set()
    for row in rows(pool / 'aam.csv.gz'):
        dataset = row['representative_source']
        if row['r_id'] in selected_ids or row['r_id'] not in memberships[dataset]:
            raise ValueError('Repeated selected input or representative outside source membership')
        selected_ids.add(row['r_id'])
        selected[dataset] += 1
        scopes[dataset][row['mapping_scope']] += 1
    result = []
    for dataset, original in summary['source_counts'].items():
        upstream = summary['by_source'][dataset]
        count = len(memberships[dataset])
        excluded = upstream.get('excluded_source_records', 0)
        if (count != upstream['retained_reactions'] or source_rows[dataset] != upstream['source_records']
                or original != source_rows[dataset] + excluded or sum(excluded_rows[dataset].values()) != excluded):
            raise ValueError(f'Per-source count mismatch: {dataset}')
        scope = scopes[dataset]
        result.append(dict(dataset=dataset, original_rows=original, excluded_source_rows=excluded,
                           excluded_unchanged_source_rows=excluded_rows[dataset]['identical_canonical_endpoints'],
                           excluded_conflicting_source_rows=excluded_rows[dataset]['conflicting_heavy_atom_its'],
                           excluded_unchanged_unique_reactions=upstream.get('excluded_identity_reactions', 0),
                           excluded_conflicting_unique_reactions=upstream.get('excluded_mapping_conflict_reactions', 0),
                           retained_source_rows=source_rows[dataset],
                           within_source_duplicates_merged=source_rows[dataset] - count,
                           retained_unique_source_reactions=count, selected_pool_rows=selected[dataset],
                           represented_by_other_sources=count - selected[dataset],
                           selected_complete_individual=scope['heavy_atoms'],
                           selected_complete_classes=scope['heavy_atom_equivalence_classes'],
                           selected_partial=(scope['partial_heavy_atoms']
                                             + scope['partial_heavy_atom_equivalence_classes']),
                           selected_unverified=scope['unverified_heavy_atoms']))
    if len(selected_ids) != summary['totals']['retained_reactions'] or set.union(*memberships.values()) != selected_ids:
        raise ValueError('Source memberships and selected targets do not cover the same retained inputs')
    return result


def report(checkpoint):
    journey = checkpoint['unique_input_history']
    result = ['AAM cleaning checkpoint and dataset segmentation',
              '================================================', '',
              f"Recorded {checkpoint['record_date']}. Further cleaning is deferred for later review.",
              'This checkpoint does not change source data or the selected pool.', '',
              'Current decisions', '-------------------', '',
              '* Keep raw source maps, including partial and equivalent-atom mappings.',
              '* Keep local QC flags and unverified references as annotations.',
              '* Merge duplicate canonical reactions while preserving all source records.',
              '* Exclude identical endpoints and certified conflicting heavy-atom mappings.',
              '* Ignore stereo for reaction identity and ITS comparison; retain raw stereo in provenance.',
              '* Adopt no computed completion and assign no new train/validation/test split.', '',
              'Cleaning history', '------------------', '',
              '.. list-table:: Unique directed input counts', '   :header-rows: 1', '',
              '   * - Stage', '     - Change', '     - Remaining']
    for row in journey:
        result.extend([f"   * - {row['stage']}", f"     - {row['change']:,}", f"     - {row['remaining']:,}"])
    totals = checkpoint['current_totals']
    accounting = checkpoint['source_row_accounting']
    result.extend(['', 'Source-row accounting', '-----------------------', '',
                   'The history above counts unique reaction groups. The following accounting',
                   'starts from original source rows, including repeated observations. Each',
                   'excluded reaction group can contain several original rows.', '',
                   '.. list-table:: All changes measured in source-row counts', '   :header-rows: 1', '',
                   '   * - Action', '     - Rows',
                   '   * - Original source rows', f"     - {accounting['original_rows']:,}",
                   '   * - Exclude rows in unchanged-reaction groups',
                   f"     - {accounting['excluded_unchanged_source_rows']:,}",
                   '   * - Exclude rows in conflicting-mapping groups',
                   f"     - {accounting['excluded_conflicting_source_rows']:,}",
                   '   * - Merge repeated rows within each source',
                   f"     - {accounting['within_source_duplicates_merged']:,}",
                   '   * - Merge shared reactions across sources',
                   f"     - {accounting['represented_by_other_sources']:,}",
                   '   * - Selected unique reactions', f"     - {accounting['selected_pool_rows']:,}", '',
                   f"There are still {totals['input_source_records']:,} original source records.",
                   f"The exclusions cover {totals['excluded_source_records']:,} source rows;",
                   f"{totals['source_records']:,} source rows remain, merging into",
                   f"{totals['retained_reactions']:,} selected unique reactions. All originals remain",
                   'in active or excluded provenance. The latest policy added back 5,666',
                   'previously held-out inputs: 74,004 to 79,670. The excluded payloads',
                   'and prior selected raw targets were unchanged in that policy update.', '',
                   'Dataset segmentation', '----------------------', '',
                   '.. list-table:: Source memberships and chosen representatives',
                   '   :header-rows: 1', '', '   * - Dataset', '     - Original rows',
                   '     - Retained unique source reactions', '     - Chosen pool rows'])
    for row in checkpoint['datasets']:
        result.extend([f"   * - {row['dataset']}", f"     - {row['original_rows']:,}",
                       f"     - {row['retained_unique_source_reactions']:,}", f"     - {row['selected_pool_rows']:,}"])
    result.extend(['', '.. list-table:: Per-dataset source-row reconciliation', '   :header-rows: 1', '',
                   '   * - Dataset', '     - Original rows', '     - Unchanged rows', '     - Conflicting rows',
                   '     - Repeats within source', '     - Represented through another source', '     - Selected rows'])
    for row in checkpoint['datasets']:
        result.append(f"   * - {row['dataset']}")
        keys = ['original_rows', 'excluded_unchanged_source_rows', 'excluded_conflicting_source_rows',
                'within_source_duplicates_merged', 'represented_by_other_sources', 'selected_pool_rows']
        result.extend(f"     - {row[key]:,}" for key in keys)
    result.extend(['', 'Retained source-reaction counts overlap across datasets. Chosen pool rows',
                   'count the selected raw-reference source once per global input and sum to',
                   f"{totals['retained_reactions']:,}. A source with zero chosen representatives "
                   'still has its retained',
                   'reactions and original records linked through members and provenance.', '',
                   'dataset-segments.csv includes source-row exclusions, repeated rows,',
                   'representatives supplied by another source and selected mapping scopes.', '',
                   'Reproduction and saved evidence', '--------------------------------', '',
                   'Run from the repository root into a fresh report directory::', '',
                   '   python -m synrxn.reconstruction builders report_aam_cleaning --output /tmp/aam-cleaning-checkpoint-new', '',
                   'checkpoint.json pins the source pool, baseline audit, policies and counts.',
                   'manifest.json pins this report and its CSV/JSON payloads.',
                   'Current processing is documented in unique-method.rst; the current data',
                   'builder is synrxn.reconstruction.families.representation.atom_mapping._sources.build_unique_aam. Previous versions and the completion',
                   'experiment remain under archive/datasets/data-archives/aam_review/. No data is changed by',
                   'this reporting command.', ''])
    return '\n'.join(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=repo_path('Data/tables/development/aam_unique'))
    parser.add_argument('--baseline', type=Path, default=repo_path(
        'Data/archives/aam_review/before-equivalence-partial-stereo-2026-10-27/aam_overlap'))
    parser.add_argument('--record-date', default=date.today().isoformat())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / f'Data/metadata/reconstruction/aam/cleaning-checkpoint-{args.record_date}'
    pool_pin, baseline_pin = checked_manifest(args.pool), checked_manifest(args.baseline)
    summary = json.loads((args.pool / 'summary.json').read_text())
    previous = json.loads((args.baseline / 'summary.json').read_text())
    audit = repo_path('Data/tables/development/aam_overlap')
    current = json.loads((audit / 'summary.json').read_text())
    if (summary['audit_manifest_sha256'] != digest(audit / 'manifest.json')
            or summary['source_sha256'] != previous['source_sha256']):
        raise ValueError('History must compare audits of the same pinned source records')
    before = previous['modes']['exact']['totals']['distinct_reactions']
    stereo_free = current['modes']['exact']['totals']['distinct_reactions']
    totals = summary['totals']
    unchanged, conflicts = totals['excluded_identity_reactions'], totals['excluded_mapping_conflict_reactions']
    if stereo_free - unchanged - conflicts != totals['retained_reactions']:
        raise ValueError('Cleaning history does not reconcile to the selected pool')
    history = [dict(stage='Earlier stereo-sensitive unique inputs', change=0, remaining=before),
               dict(stage='Additional merging when stereo is ignored', change=stereo_free - before,
                    remaining=stereo_free),
               dict(stage='Exclude identical endpoints', change=-unchanged, remaining=stereo_free - unchanged),
               dict(stage='Exclude conflicting heavy-atom mappings', change=-conflicts,
                    remaining=totals['retained_reactions'])]
    checkpoint = dict(schema='synrxn.aam-cleaning-checkpoint.v1', record_date=args.record_date,
                      further_cleaning='deferred_by_user', source_pool_changed=False,
                      pool_manifest_sha256=pool_pin, baseline_audit_manifest_sha256=baseline_pin,
                      implementation_sha256=digest(Path(__file__)),
                      policies={k: summary[k] for k in ['identity', 'retention_policy', 'pool_exclusion_policy']},
                      unique_input_history=history,
                      current_totals=totals, datasets=segmentation(args.pool, summary))
    accounting = {key: sum(row[key] for row in checkpoint['datasets']) for key in
                  ['original_rows', 'excluded_unchanged_source_rows', 'excluded_conflicting_source_rows',
                   'within_source_duplicates_merged', 'represented_by_other_sources', 'selected_pool_rows']}
    if accounting['original_rows'] != sum(accounting[k] for k in accounting if k != 'original_rows'):
        raise ValueError('Source-row accounting does not reconcile')
    checkpoint['source_row_accounting'] = accounting
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh checkpoint directory; previous records are preserved')
    with (output / 'dataset-segments.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(checkpoint['datasets'])
    write_json(output / 'checkpoint.json', checkpoint)
    (output / 'checkpoint.rst').write_text(report(checkpoint))
    if digest(args.pool / 'manifest.json') != pool_pin:
        raise ValueError('Source pool changed during reporting')
    write_json(output / 'manifest.json', {'files': {p.name: digest(p) for p in sorted(output.iterdir())}})
    print(json.dumps({'report': str(output), 'retained_reactions': totals['retained_reactions'],
                      'dataset_memberships': sum(r['retained_unique_source_reactions'] for r in checkpoint['datasets']),
                      'chosen_pool_rows': sum(r['selected_pool_rows'] for r in checkpoint['datasets'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
