"""Check the one-command RBL workflow and independent bundle verification."""
import csv
import gzip
import json
import zipfile

import pytest

from synrxn.reconstruction.families.representation.reaction_balancing._sources import flower_rbl as base
from synrxn.reconstruction.families.representation.reaction_balancing._sources import build_flower_rbl as CLI
from synrxn.reconstruction.families.representation.reaction_balancing._sources import flower_rbl_hard as hard
from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl_verify import verify

@pytest.fixture
def source(tmp_path):
    folder = tmp_path / 'source' / 'flower'
    folder.mkdir(parents=True)
    row = {key: '' for key in base.TRACE_FIELDS}
    row.update(r_id='7', original_id='native', role='test', steps='1', center_type='RC',
               center_cluster_id='center-1', source_group='9', source_lines='[1]',
               source_member='inputs/test.txt.gz', source_split='test',
               ground_truth='[CH3:1][CH2:2][OH:3].[OH2:4]>>[CH3:1][O:3][CH3:2].[OH2:4]')
    with gzip.open(folder / 'aam.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (folder / 'source.json').write_text(json.dumps({'rights': {'license_expression': 'MIT'}}))
    (folder / 'licenses').mkdir()
    (folder / 'licenses' / 'LICENSE.txt').write_text('MIT fixture evidence')
    left, right = row['ground_truth'].split('>>')
    dump = tmp_path / 'flower.zip'
    with zipfile.ZipFile(dump, 'w') as archive:
        text = f'{left}.[CH3:10][OH:11]>>{right}.[CH3:10][OH:11]|9\n'
        archive.writestr('inputs/test.txt.gz', gzip.compress(text.encode(), mtime=0))
    return folder, dump


def test_one_command_builds_both_variants_and_verify_never_rebuilds(source, tmp_path, monkeypatch):
    folder, dump = source
    output = tmp_path / 'rbl'
    reports = CLI.main(['--all', '--source', str(folder), '--dump', str(dump),
                        '--output', str(output), '--workers', '1'])
    assert [r['variant'] for r in reports] == ['base', 'hard']
    assert [r['rows'] for r in reports] == [3, 3]
    before = {variant: (output / variant / 'rbl.csv.gz').read_bytes() for variant in ('flower', 'flower-hard')}

    def forbidden(*args, **kwargs):
        raise AssertionError('Read-only verification attempted to rebuild')

    monkeypatch.setattr(CLI, 'build', forbidden)
    monkeypatch.setattr(CLI, 'build_hard_variant', forbidden)
    checked = CLI.main(['--all', '--verify', '--output', str(output)])
    assert all(report['verified'] for report in checked)
    assert before == {variant: (output / variant / 'rbl.csv.gz').read_bytes()
                      for variant in ('flower', 'flower-hard')}


def test_verify_rejects_payload_tampering_and_changed_parent_source(source, tmp_path):
    folder, _ = source
    output = tmp_path / 'base'
    base.build(folder, output)
    assert verify(output, 'base')['verified']
    table = output / 'rbl.csv.gz'
    previous = table.read_bytes()
    table.write_bytes(previous + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(output, 'base')
    table.write_bytes(previous)
    parent = folder / 'aam.csv.gz'
    parent.write_bytes(parent.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='parent AAM source changed'):
        verify(output, 'base')


def test_verify_hard_rejects_changed_base_source(source, tmp_path):
    folder, dump = source
    original, output = tmp_path / 'base', tmp_path / 'hard'
    base.build(folder, original)
    hard.build(folder, original, dump, output, workers=1)
    table = original / 'rbl.csv.gz'
    table.write_bytes(table.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='base source changed'):
        verify(output, 'hard')


def test_prepare_source_dispatch_and_conflict_with_read_only_verify(source, tmp_path, monkeypatch):
    folder, _ = source
    called = []
    monkeypatch.setattr(CLI.subprocess, 'run', lambda command, **kwargs: called.append(command))
    monkeypatch.setattr(CLI, 'build', lambda *args: None)
    monkeypatch.setattr(CLI, 'verify', lambda *args: {'verified': True})
    CLI.main(['--prepare-source', '--source', str(folder), '--output', str(tmp_path / 'base')])
    assert called[0][1:] == ['-m', 'synrxn.reconstruction.families.representation.atom_mapping._sources.aam_benchmark',
                             '--datasets', 'flower', '--output', str(folder.parent)]
    with pytest.raises(SystemExit):
        CLI.main(['--prepare-source', '--verify'])
    assert len(called) == 1
