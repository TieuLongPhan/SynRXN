"""Export the curated FlowER AAM table from its single reproducibility dump.

Default: verify and export the frozen records. --rebuild recomputes uncapped
RC/MC selection from the dump's raw inputs and pinned elementary selection.
Both routes require identical AAM bytes. Work is resumable and discarded on
successful completion unless --keep-work is requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import gzip
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import shutil
import tempfile

from synrxn.reconstruction.paths import REPO, repo_path

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, require, write_json
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_dump import COLUMNS, OUTPUT_SCHEMA as SCHEMA, FlowerDump

DUMP = repo_path('Data/dumps/flower.dump.zip')
OUTPUT = REPO / '.cache/flower/native-product'
IMPLEMENTATIONS = ('synrxn/reconstruction/families/representation/atom_mapping/_sources/flower_representatives.py',
                   'synrxn/reconstruction/families/representation/atom_mapping/_sources/flower_dump.py',
                   'synrxn/reconstruction/families/representation/atom_mapping/_sources/flower_mc_pipeline.py',
                   'synrxn/reconstruction/families/representation/atom_mapping/_sources/flower_mc_export.py',
                   'synrxn/reconstruction/families/representation/atom_mapping/_sources/flower_mc_store.py',
                   'synrxn/reconstruction/families/representation/atom_mapping/_sources/flower_mechanistic_center.py',
                   'synrxn/reconstruction/families/representation/atom_mapping/_sources/flower.py')


def read_csv(path):
    with gzip.open(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle))


def read_jsonlines(path):
    with gzip.open(path, 'rt') as handle:
        yield from (json.loads(line) for line in handle)


def verify(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    require(manifest['schema'] == SCHEMA, 'Unknown RC/MC bundle schema')
    actual = {path.name: digest(path) for path in output.iterdir() if path.is_file() and path.name != 'manifest.json'}
    require(actual == manifest['files'], 'Output checksum mismatch')
    rows = read_csv(output / 'aam.csv.gz')
    witnesses = list(read_jsonlines(output / 'provenance.jsonl.gz'))
    require(len(rows) == len(witnesses) == manifest['rows'], 'Missing rows or provenance')
    keys = set()
    for number, (row, witness) in enumerate(zip(rows, witnesses)):
        require(row['r_id'] == str(number), 'Invalid row identifiers')
        require(row['role'] in ('train', 'val', 'test') and int(row['steps']) >= 1, 'Invalid role or length')
        require(witness['center_type'] == ('RC' if int(row['steps']) == 1 else 'MC'), 'Incorrect center type')
        require(all(row[k] == str(witness[k]) for k in COLUMNS), 'Target/provenance changed')
        key = (row['role'], row['steps'], witness['center_id'])
        require(key not in keys, 'Repeated center representative at one length')
        keys.add(key)
    return {'rows': len(rows), 'roles': dict(Counter(row['role'] for row in rows)), 'files_verified': len(actual)}


def configuration(entry, args):
    return {'schema': SCHEMA, 'source_sha256': entry['sha256'], 'role': entry['role'],
            'dump_sha256': args.dump_source.sha256, 'maximum_steps': 0,
            'maximum_seeds': args.max_seeds, 'seed': args.seed,
            'elementary_representatives_sha256': entry['elementary_sha256'],
            'center_definition': 'RC at one step; unordered MC with induced active-atom context at multiple steps',
            'selection_key': '(role, steps, center_id)', 'additional_neighbour_shell': False,
            'implementation_sha256': {name: digest(REPO / name) for name in IMPLEMENTATIONS},
            'synkit_sha256': {name: digest(Path(find_spec(name).origin)) for name in
                              ('synkit.Graph.MTG.mtg', 'synkit.IO.chem_converter', 'synkit.IO.mol_to_graph')},
            'versions': {name: version(name) for name in ('synkit', 'rdkit', 'networkx')}}


def notices(output, args, config=None):
    output.mkdir(parents=True, exist_ok=True)
    for name in ('LICENSE-MIT.txt', 'UPSTREAM_LICENSE.json'):
        args.dump_source.copy('licenses/' + name, output / name)
    metadata = args.dump_source.metadata
    record = {'schema': 'synrxn.flower-dump-export.v1', 'dump_file': str(args.dump_source.path),
              'dump_sha256': args.dump_source.sha256, 'recipe': metadata['recipe'],
              'dump_rows': metadata['rows'], 'dump_by_role': metadata['by_role'], 'license_spdx': 'MIT',
              'runtime_versions': metadata['runtime_versions']}
    if config is not None:
        record['configuration'] = config
    write_json(output / 'source.json', record)
    (output / 'NOTICE.txt').write_text(
        'FlowER RC/MC AAM selection. MIT; copyright (c) 2025 FongMunHong.\n'
        'Native train/val/test roles and source witnesses retained. No step upper limit.\n'
        'One step uses the pinned curated RC selection; multiple steps use native SynKit MTG MCs.\n'
        'The single source dump contains original curated inputs, frozen elementary references,\n'
        'a verified export snapshot and pre-curation source history.\n')


def refresh_manifest(output):
    rows = read_csv(output / 'aam.csv.gz')
    write_json(output / 'manifest.json', {'schema': SCHEMA, 'rows': len(rows),
               'files': {path.name: digest(path) for path in output.iterdir()
                         if path.is_file() and path.name != 'manifest.json'}})


def restore_snapshot(args, output):
    for name, member in args.dump_source.metadata['reference_files'].items():
        args.dump_source.copy(member, output / name)
    notices(output, args)
    refresh_manifest(output)
    return verify(output)


def run_split(entry, args, output):
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_export import export_bundle
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_pipeline import process_sources
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_store import get_meta, open_store

    config = configuration(entry, args)
    checkpoint = args.work_root / (entry['split'] + '.sqlite3')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    folder = output / entry['split']
    with checkpoint.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = open_store(checkpoint, config)
        try:
            if not get_meta(db, 'source_complete', False):
                done = process_sources(db, args.dump_source.text(entry), entry, args,
                                       checkpoint.with_suffix('.progress.json'))
                if not done:
                    return False
            notices(folder, args, config)
            result = export_bundle(db, folder, args.dump_source.elementary(entry), args.seed, config)
            require(not result['unrepresented_centers'], 'Some center classes have no valid representative')
            require(digest(folder / 'aam.csv.gz') == entry['expected_aam_sha256'],
                    f"Raw rebuild differs from pinned data: {entry['split']}")
            verify(folder)
            return True
        finally:
            db.close()


def publish(source, output):
    """Replace only managed output after the new candidate passes every check."""
    if output.exists():
        marker = output / 'manifest.json'
        require(marker.is_file(), 'Refusing to replace an unmanaged output')
        schema = json.loads(marker.read_text()).get('schema', '')
        require(schema in (SCHEMA, 'synrxn.flower-aam.v1'), 'Unexpected output schema')
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    return verify(output)


def combine(args, entries):
    from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_export import _write_csv, _write_jsonlines

    rows, witnesses, summaries = [], [], {}
    for entry in entries:
        folder = args.output / entry['split']
        verify(folder)
        summaries[entry['role']] = json.loads((folder / 'summary.json').read_text())
        for row, witness in zip(read_csv(folder / 'aam.csv.gz'), read_jsonlines(folder / 'provenance.jsonl.gz')):
            index = len(rows)
            rows.append({**row, 'r_id': index})
            witnesses.append({**witness, 'r_id': index})
    _write_csv(args.output / 'aam.csv.gz', rows, COLUMNS)
    _write_jsonlines(args.output / 'provenance.jsonl.gz', witnesses)
    lengths = sorted({int(row['steps']) for row in rows})
    segments = [{'steps': length, 'center_type': 'RC' if length == 1 else 'MC',
                 **{role: summaries.get(role, {}).get('by_steps', {}).get(str(length), 0)
                    for role in ('train', 'val', 'test')}} for length in lengths]
    with (args.output / 'segments.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['steps', 'center_type', 'train', 'val', 'test'])
        writer.writeheader()
        writer.writerows(segments)
    combined = {'schema': SCHEMA, 'status': 'complete', 'rows': len(rows), 'maximum_steps': 0,
                'segments': segments, 'by_role': {role: summary['rows'] for role, summary in summaries.items()},
                'splits': summaries}
    write_json(args.output / 'summary.json', combined)
    for name in ('LICENSE-MIT.txt', 'UPSTREAM_LICENSE.json'):
        args.dump_source.copy('licenses/' + name, args.output / name)
    write_json(args.output / 'manifest.json', {'schema': SCHEMA, 'rows': len(rows),
               'files': {path.name: digest(path) for path in args.output.iterdir()
                         if path.is_file() and path.name != 'manifest.json'}})
    print(json.dumps({'verification': verify(args.output), 'segments': segments}, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, default=DUMP)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--rebuild', action='store_true',
                        help='Recompute the raw-source selection instead of exporting the snapshot')
    parser.add_argument('--keep-work', action='store_true')
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'],
                        default=['train', 'validation', 'test'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--recycle-batches', type=int, default=32)
    parser.add_argument('--memory-budget-gib', type=float, default=6)
    parser.add_argument('--max-new-groups', type=int, default=0)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(args.output), indent=2))
        return 0
    limits = (args.workers, args.batch_size, args.recycle_batches)
    require(min(limits) > 0 and args.memory_budget_gib > 0 and args.max_new_groups >= 0, 'Invalid resource limit')
    require(len(args.splits) == len(set(args.splits)), 'Duplicate split requested')
    require(args.rebuild or args.max_new_groups == 0, '--max-new-groups requires --rebuild')
    args.dump_source = FlowerDump(args.dump)
    metadata = args.dump_source.metadata
    args.max_seeds, args.seed = metadata['recipe']['maximum_seeds'], metadata['recipe']['seed']
    args.work_root = args.work_root or REPO / '.cache/flower' / ('rebuild-' + args.dump_source.sha256[:16])
    if not args.rebuild:
        require(set(args.splits) == {'train', 'validation', 'test'}, 'Subset export requires --rebuild')
        with tempfile.TemporaryDirectory(prefix='flower-export-') as temporary:
            staging = Path(temporary) / 'output'
            restore_snapshot(args, staging)
            print(json.dumps(publish(staging, args.output), indent=2))
        return 0
    require(not args.output.resolve().is_relative_to(args.work_root.resolve()),
            'Output cannot be inside the disposable work directory')
    require(not args.work_root.resolve().is_relative_to(args.output.resolve()),
            'Work directory cannot be inside the output directory')
    marker = args.work_root / '.flower-work.json'
    if args.work_root.exists() and any(args.work_root.iterdir()):
        require(marker.is_file(), 'Refusing to use an unmanaged work directory')
        require(json.loads(marker.read_text())['dump_sha256'] == args.dump_source.sha256,
                'Work directory belongs to a different dump')
    args.work_root.mkdir(parents=True, exist_ok=True)
    write_json(marker, {'dump_sha256': args.dump_source.sha256})
    require(all(version(name) == pinned for name, pinned in metadata['runtime_versions'].items()),
            'Raw rebuild requires the runtime versions pinned in the dump')
    indexed = {entry['split']: entry for entry in metadata['inputs']}
    staging = args.work_root / 'output'
    for split in args.splits:
        if not run_split(indexed[split], args, staging):
            print(json.dumps({'status': 'checkpointed', 'work_root': str(args.work_root)}))
            return 0
    original_output = args.output
    args.output = staging
    combine(args, [indexed[s] for s in ('train', 'validation', 'test') if s in args.splits])
    expected = metadata['files'][metadata['reference_files']['aam.csv.gz']]
    if set(args.splits) == {'train', 'validation', 'test'}:
        require(digest(staging / 'aam.csv.gz') == expected, 'Combined rebuild differs from pinned snapshot')
    for split in args.splits:
        shutil.rmtree(staging / split)
    notices(staging, args)
    refresh_manifest(staging)
    print(json.dumps(publish(staging, original_output), indent=2))
    if not args.keep_work:
        shutil.rmtree(args.work_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
