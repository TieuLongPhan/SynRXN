"""Check self-contained FlowER dump export and raw-source rebuilding."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_representatives as BUILDER
from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_mc_export as export
from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_mc_pipeline as pipeline
from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_mc_store as store
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_dump import DumpText, FlowerDump
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower import canonical_unmapped_reaction

from synrxn.reconstruction.paths import REPO as ROOT
INITIAL = '[CH3:1][C:2](=[O:3])[Cl:4].[NH3:5]'
MIDDLE = '[CH3:1][C:2]([O-:3])([Cl:4])[NH3+:5]'
FINAL = '[CH3:1][C:2](=[O:3])[NH3+:5].[Cl-:4]'
REACTIONS = [INITIAL + '>>' + MIDDLE, MIDDLE + '>>' + FINAL]


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def dump_file(tmp_path):
    """Package three tiny raw splits and independently frozen expected export bytes."""
    from importlib.metadata import version
    pipeline.initialize_worker()
    lines = [reaction + '|' + str(group) for group in range(1, 4)
             for reaction in (*REACTIONS, FINAL + '>>' + FINAL)]
    original = ('\n'.join(lines) + '\n').encode()
    entries, payloads, combined, witnesses = [], {}, [], []
    for split, role in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        raw_member, rc_member = f'inputs/{split}.txt.gz', f'elementary/{split}.csv.gz'
        payloads[raw_member] = gzip.compress(original, mtime=0)
        elementary = {'n_steps': '1', 'ground_truth': REACTIONS[0],
                      'rxn': canonical_unmapped_reaction(REACTIONS[0]),
                      'original_id': split + '-elementary', 'rc_cluster_id': 'rc-0'}
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text, fieldnames=list(elementary))
        writer.writeheader()
        writer.writerow(elementary)
        payloads[rc_member] = gzip.compress(text.getvalue().encode(), mtime=0)
        entry = {'split': split, 'role': role, 'filename': split + '.txt', 'sha256': digest(original),
                 'member': raw_member, 'elementary_member': rc_member, 'raw_records': 9,
                 'elementary_sha256': digest(payloads[rc_member])}
        db = store.open_store(tmp_path / (split + '.sqlite3'), {'role': role})
        clusters = store.DiskClusters(db)
        for group in range(1, 4):
            offset = (group - 1) * 3
            edges = [(INITIAL, MIDDLE, offset + 1), (MIDDLE, FINAL, offset + 2)]
            result = pipeline.project_group((str(group), edges, {FINAL}, entry, 4))
            pipeline.save_groups(db, clusters, [result], 42, split)
        store.set_meta(db, 'source_counts', {'raw_source_lines': 9, 'elementary_source_rows': 6})
        db.commit()
        folder = tmp_path / ('native-' + split)
        report = export.export_bundle(db, folder, [elementary], 42, {'role': role})
        db.close()
        entry.update(expected_rows=2, expected_steps=report['by_steps'],
                     expected_aam_sha256=digest((folder / 'aam.csv.gz').read_bytes()))
        entries.append(entry)
        with gzip.open(folder / 'aam.csv.gz', 'rt') as handle:
            rows = list(csv.DictReader(handle))
        with gzip.open(folder / 'provenance.jsonl.gz', 'rt') as handle:
            sources = [json.loads(line) for line in handle]
        for row, source in zip(rows, sources):
            index = len(combined)
            combined.append({**row, 'r_id': index})
            witnesses.append({**source, 'r_id': index})
    reference = tmp_path / 'reference'
    reference.mkdir()
    export._write_csv(reference / 'aam.csv.gz', combined, export.COLUMNS)
    export._write_jsonlines(reference / 'provenance.jsonl.gz', witnesses)
    segments = [{'steps': step, 'center_type': 'RC' if step == 1 else 'MC', 'train': 1, 'val': 1, 'test': 1}
                for step in (1, 2)]
    (reference / 'segments.csv').write_text('steps,center_type,train,val,test\n1,RC,1,1,1\n2,MC,1,1,1\n')
    report = {'schema': export.SCHEMA, 'status': 'complete', 'rows': 6,
              'segments': segments, 'maximum_steps': 0}
    (reference / 'summary.json').write_text(json.dumps(report))
    for file in reference.iterdir():
        payloads['reference/' + file.name] = file.read_bytes()
    for name in ('LICENSE-MIT.txt', 'UPSTREAM_LICENSE.json'):
        payloads['licenses/' + name] = b'test-only fixture evidence\n'
    manifest = {'schema': 'synrxn.flower-reproducibility-dump.v1', 'columns': list(export.COLUMNS),
                'rows': 6, 'by_role': {'train': 2, 'val': 2, 'test': 2}, 'inputs': entries,
                'runtime_versions': {name: version(name) for name in ('rdkit', 'synkit', 'networkx')},
                'recipe': {'maximum_steps': 0, 'maximum_seeds': 4, 'seed': 42},
                'files': {name: digest(value) for name, value in payloads.items()},
                'reference_files': {file.name: 'reference/' + file.name for file in reference.iterdir()}}
    archive = tmp_path / 'flower.dump.zip'
    with zipfile.ZipFile(archive, 'w') as target:
        for name, value in payloads.items():
            target.writestr(name, value)
        target.writestr('source.json', json.dumps(manifest))
    return archive, manifest, original


def command(archive, output, *extra):
    result = subprocess.run([sys.executable, '-m', 'synrxn.reconstruction.families.representation.atom_mapping._sources.flower_representatives',
                             '--dump', str(archive), '--output', str(output), *extra],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_dump_streams_original_text_without_loose_files(dump_file):
    archive, manifest, original = dump_file
    dump = FlowerDump(archive)
    assert len(dump.elementary(manifest['inputs'][0])) == 1
    text = DumpText(archive, manifest['inputs'][0]['member'])
    with text.open('rb') as handle:
        assert handle.read() == original
    with text.open() as handle:
        assert len(list(handle)) == 9


def test_snapshot_export_is_standard_and_byte_identical_on_repeat(dump_file, tmp_path):
    archive, manifest, _ = dump_file
    output = tmp_path / 'output'
    command(archive, output)
    assert BUILDER.verify(output)['roles'] == {'train': 2, 'val': 2, 'test': 2}
    assert digest((output / 'aam.csv.gz').read_bytes()) == manifest['files']['reference/aam.csv.gz']
    before = {p.name: digest(p.read_bytes()) for p in output.iterdir()}
    command(archive, output)
    assert before == {p.name: digest(p.read_bytes()) for p in output.iterdir()}


def test_fresh_raw_rebuild_resumes_without_older_outputs_and_cleans_work(dump_file, tmp_path):
    archive, manifest, _ = dump_file
    output, work = tmp_path / 'rebuilt', tmp_path / 'work'
    options = ['--rebuild', '--work-root', str(work), '--workers', '2', '--batch-size', '1',
               '--recycle-batches', '2']
    first = command(archive, output, *options, '--max-new-groups', '1')
    assert 'checkpointed' in first.stdout and not output.exists()
    command(archive, output, *options)
    assert not work.exists()
    assert digest((output / 'aam.csv.gz').read_bytes()) == manifest['files']['reference/aam.csv.gz']
    assert BUILDER.verify(output)['rows'] == 6
    with gzip.open(output / 'aam.csv.gz', 'rt') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == list(export.COLUMNS)
    assert [row['role'] for row in rows] == ['train', 'train', 'val', 'val', 'test', 'test']


def test_corrupt_dump_leaves_existing_output_unchanged(dump_file, tmp_path):
    archive, _, _ = dump_file
    output = tmp_path / 'output'
    command(archive, output)
    before = (output / 'aam.csv.gz').read_bytes()
    with zipfile.ZipFile(archive) as source:
        payloads = {name: source.read(name) for name in source.namelist()}
    payloads['inputs/train.txt.gz'] = b'corrupt'
    with zipfile.ZipFile(archive, 'w') as target:
        for name, value in payloads.items():
            target.writestr(name, value)
    with pytest.raises(ValueError, match='checksum mismatch'):
        BUILDER.main(['--dump', str(archive), '--output', str(output)])
    assert (output / 'aam.csv.gz').read_bytes() == before


def test_dump_export_does_not_import_the_chemistry_stack(dump_file, tmp_path):
    import os
    archive, _, _ = dump_file
    blocker = tmp_path / 'blocker'
    blocker.mkdir()
    (blocker / 'sitecustomize.py').write_text(
        "import sys\n"
        "class BlockChemistry:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('rdkit', 'synkit'):\n"
        "            raise ImportError('Chemistry import forbidden during snapshot export')\n"
        "sys.meta_path.insert(0, BlockChemistry())\n"
    )
    env = {**os.environ, 'PYTHONPATH': str(blocker)}
    result = subprocess.run([sys.executable, '-m', 'synrxn.reconstruction.families.representation.atom_mapping._sources.flower_representatives',
                             '--dump', str(archive), '--output', str(tmp_path / 'lightweight')],
                            env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rebuild_preserves_unmanaged_work_and_rejects_nested_output(dump_file, tmp_path):
    archive, _, _ = dump_file
    work = tmp_path / 'unmanaged'
    work.mkdir()
    important = work / 'important.txt'
    important.write_text('preserve')
    with pytest.raises(ValueError, match='unmanaged work directory'):
        BUILDER.main(['--dump', str(archive), '--rebuild', '--work-root', str(work),
                      '--output', str(tmp_path / 'output')])
    assert important.read_text() == 'preserve'
    with pytest.raises(ValueError, match='inside the disposable work directory'):
        BUILDER.main(['--dump', str(archive), '--rebuild', '--work-root', str(work),
                      '--output', str(work / 'output')])
    assert important.read_text() == 'preserve'
    with pytest.raises(ValueError, match='inside the output directory'):
        BUILDER.main(['--dump', str(archive), '--rebuild', '--work-root', str(work),
                      '--output', str(tmp_path)])
    assert important.read_text() == 'preserve'
