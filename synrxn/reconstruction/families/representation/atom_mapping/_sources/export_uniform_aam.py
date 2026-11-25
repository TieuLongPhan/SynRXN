#!/usr/bin/env python
"""Export the retained AAM pool as id, data_source, rxn, ground_truth."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
from itertools import zip_longest
import json
from pathlib import Path
import re

from synrxn.reconstruction.paths import REPO as ROOT, repo_path
FIELDS = ['id', 'data_source', 'rxn', 'ground_truth']
SCHEMA = 'synrxn.aam-uniform.v1'
INPUT_FILES = ['aam.csv.gz', 'reactions.csv.gz', 'summary.json']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        yield from csv.DictReader(handle)


def checked_inputs(pool):
    manifest = json.loads((pool / 'manifest.json').read_text())
    if manifest['schema'] != 'synrxn.aam-unique.v5':
        raise ValueError('Expected the retained v5 AAM pool')
    pins = {name: digest(pool / name) for name in INPUT_FILES}
    for name, checksum in pins.items():
        if checksum != manifest['files'][name]:
            raise ValueError(f'Input checksum mismatch: {name}')
    pins['manifest.json'] = digest(pool / 'manifest.json')
    return pins, json.loads((pool / 'summary.json').read_text())


def projected_rows(pool):
    seen_ids, seen_rxns = set(), set()
    for selected, inventory in zip_longest(rows(pool / 'aam.csv.gz'), rows(pool / 'reactions.csv.gz')):
        if selected is None or inventory is None:
            raise ValueError('Selected references and reaction inventory differ in length')
        if selected['r_id'] != inventory['r_id'] or selected['rxn'] != inventory['rxn']:
            raise ValueError('Selected references and reaction inventory are not aligned')
        names = json.loads(inventory['sources'])
        if (not names or not all(isinstance(name, str) and name and ';' not in name for name in names)
                or names != sorted(set(names)) or selected['representative_source'] not in names):
            raise ValueError(f"Invalid source memberships: {selected['r_id']}")
        result = dict(id=selected['r_id'], data_source=';'.join(names),
                      rxn=selected['rxn'], ground_truth=selected['ground_truth'])
        if not all(result.values()) or re.search(r':\d+\]', result['rxn']):
            raise ValueError(f"Empty field or atom-mapped input: {result['id']}")
        if result['id'] in seen_ids or result['rxn'] in seen_rxns:
            raise ValueError(f"Repeated reaction or ID: {result['id']}")
        seen_ids.add(result['id'])
        seen_rxns.add(result['rxn'])
        yield result


def update_stats(stats, row):
    names = row['data_source'].split(';')
    stats['rows'] += 1
    stats['shared_source_reactions'] += len(names) > 1
    stats['source_memberships'].update(names)


def checked_stats(stats, summary):
    expected = {name: counts['retained_reactions'] for name, counts in summary['by_source'].items()}
    if (stats['rows'] != summary['totals']['retained_reactions']
            or dict(stats['source_memberships']) != expected):
        raise ValueError('Export does not cover every retained reaction and source membership')
    return stats


def export_manifest(pool, output, pins, stats):
    try:
        input_path = pool.relative_to(ROOT).as_posix()
    except ValueError:
        input_path = str(pool)
    return {
        'schema': SCHEMA,
        'columns': FIELDS,
        'input_pool': input_path,
        'input_sha256': pins,
        'implementation_sha256': digest(Path(__file__)),
        'files': {'aam.csv.gz': digest(output / 'aam.csv.gz')},
        'counts': stats,
        'field_rules': {
            'id': 'Unchanged retained-pool r_id; joins to full audit and provenance records.',
            'data_source': 'Sorted semicolon-separated datasets containing the reaction; '
                           'all inventory memberships, not a claim that every source mapping is verified.',
            'rxn': 'Unchanged canonical stereo-free unmapped reaction from the retained pool.',
            'ground_truth': 'Selected source atom-mapped reaction copied verbatim; '
                            'original stereo, explicit H and partial labels preserved.',
        },
        'audit_location': input_path,
        'cleaning_changed': False,
        'new_split_assigned': False,
    }


def build(pool, output):
    pins, summary = checked_inputs(pool)
    output.mkdir(parents=True, exist_ok=False)
    stats = dict(rows=0, shared_source_reactions=0, source_memberships=Counter())
    with (output / 'aam.csv.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                for row in projected_rows(pool):
                    writer.writerow(row)
                    update_stats(stats, row)
    checked_stats(stats, summary)
    manifest = export_manifest(pool, output, pins, stats)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def verify(pool, output):
    pins, summary = checked_inputs(pool)
    manifest = json.loads((output / 'manifest.json').read_text())
    with gzip.open(output / 'aam.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError('Unexpected uniform export columns')
        stats = dict(rows=0, shared_source_reactions=0, source_memberships=Counter())
        for expected, actual in zip_longest(projected_rows(pool), reader):
            if expected != actual:
                raise ValueError('Uniform row differs from the retained source pool')
            update_stats(stats, actual)
    checked_stats(stats, summary)
    if manifest != export_manifest(pool, output, pins, stats):
        raise ValueError('Uniform manifest, input pins or output checksum differ')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=repo_path('Data/tables/development/aam_unique'))
    parser.add_argument('--output', type=Path, default=repo_path('Data/tables/development/aam_uniform'))
    parser.add_argument('--verify', action='store_true', help='Verify an existing export without changing files')
    args = parser.parse_args()
    pool, output = args.input.resolve(), args.output.resolve()
    if not args.verify:
        build(pool, output)
    stats = verify(pool, output)
    print(json.dumps(dict(output=str(output / 'aam.csv.gz'), verified=True, **stats), sort_keys=True))


if __name__ == '__main__':
    main()
