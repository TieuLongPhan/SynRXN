"""Source materialization and compatibility adapters preserve runtime behavior."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from synrxn.__main__ import main
from synrxn.reconstruction.provenance import compat
from synrxn.reconstruction import paths
from synrxn.reconstruction.paths import REPO, source_path


ORIGINAL = '''"""A source-backed, guarded example."""
from __future__ import annotations
from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
def source_info():
    return {'path': Path(__file__).relative_to(ROOT).as_posix(), 'sha256': sha(__file__)}
def read(repo):
    return (repo / 'kdd/input.json').read_text()
if __name__ == '__main__':
    print(source_info())
'''


def prepare(tmp_path, monkeypatch, *, archive):
    monkeypatch.setattr(paths, 'REPO', tmp_path)
    monkeypatch.setattr(compat, 'REPO', tmp_path)
    original = tmp_path / 'original.py'
    original.write_text(ORIGINAL)
    text = compat.materialize_source('script/example.py', original)
    module_path = tmp_path / 'synrxn/reconstruction/builders/example.py'
    module_path.parent.mkdir(parents=True)
    module_path.write_text(text)
    archived = tmp_path / 'archive/source-code/script/example.py'
    if archive:
        archived.parent.mkdir(parents=True)
        archived.write_bytes(original.read_bytes())
    return module_path, archived


def import_materialized(module_path):
    spec = importlib.util.spec_from_file_location('maintained_example', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialization_emits_direct_maintainable_code_and_valid_future_import(tmp_path, monkeypatch):
    module_path, _ = prepare(tmp_path, monkeypatch, archive=False)
    text = module_path.read_text()
    assert 'def source_info():' in text
    assert 'load_legacy_module' not in text
    assert 'exec(' not in text
    parsed = ast.parse(text)
    assert not any(isinstance(n, ast.Name) and n.id == '__file__' for n in ast.walk(parsed))
    assert '__source_file__' in text
    assert text.index('__configure_runtime(globals()') < text.index("if __name__ == '__main__':")
    module = import_materialized(module_path)
    assert module.__doc__ == 'A source-backed, guarded example.'
    assert module.__file__ == str(module_path)
    assert module.__source_file__ == str(tmp_path / 'script/example.py')
    assert str(module.ROOT) == str(tmp_path)


def test_active_import_and_operations_need_no_archived_source(tmp_path, monkeypatch):
    module_path, archived = prepare(tmp_path, monkeypatch, archive=False)
    monkeypatch.setattr(compat, 'legacy_path', lambda *a, **kw: pytest.fail('active import read source archive'))
    module = import_materialized(module_path)
    current = tmp_path / 'Data/metadata/reconstruction/input.json'
    current.parent.mkdir(parents=True)
    current.write_text('{"status": "current"}')
    assert module.read(tmp_path) == '{"status": "current"}'
    assert not archived.exists()
    assert not (tmp_path / 'kdd').exists()
    assert module.source_info() == {'path': 'script/example.py',
                                    'sha256': hashlib.sha256(module_path.read_bytes()).hexdigest()}


def test_checkout_own_source_digest_verifies_preserved_original_bytes(tmp_path, monkeypatch):
    module_path, archived = prepare(tmp_path, monkeypatch, archive=True)
    module = import_materialized(module_path)
    expected = hashlib.sha256(ORIGINAL.encode()).hexdigest()
    assert module.source_info() == {'path': 'script/example.py', 'sha256': expected}
    assert expected != hashlib.sha256(module_path.read_bytes()).hexdigest()
    archived.write_text(ORIGINAL + '\n# changed original\n')
    assert module.source_info()['sha256'] != expected


def test_missing_foreign_source_pin_cannot_use_own_source_fallback(tmp_path, monkeypatch):
    module_path, _ = prepare(tmp_path, monkeypatch, archive=False)
    module = import_materialized(module_path)
    with pytest.raises(FileNotFoundError):
        module.sha(tmp_path / 'script/another_missing_kernel.py')


def test_materialization_does_not_rewrite_original_source(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'REPO', tmp_path)
    original = tmp_path / 'source.py'
    original.write_bytes(ORIGINAL.encode())
    before = original.stat()
    first = compat.materialize_source('script/example.py', original)
    second = compat.materialize_source('script/example.py', original)
    after = original.stat()
    assert first == second
    assert original.read_bytes() == ORIGINAL.encode()
    assert (before.st_ino, before.st_mtime_ns, before.st_size) == (after.st_ino, after.st_mtime_ns, after.st_size)


def test_materialization_rejects_escaping_source_identity(tmp_path):
    original = tmp_path / 'source.py'
    original.write_text(ORIGINAL)
    with pytest.raises(ValueError, match='repository-relative'):
        compat.materialize_source('../outside.py', original)


def test_materialization_retains_original_code_license_notice(tmp_path):
    from synrxn.reconstruction.provenance.compat import materialize_source
    source = tmp_path / 'licensed.py'
    notice = '# SPDX-License-Identifier: LGPL-3.0-or-later'
    source.write_text(notice + '\nVALUE = 1\n')
    generated = materialize_source('script/licensed.py', source)
    assert notice in generated
    compile(generated, 'licensed.py', 'exec')


def test_packaged_builder_and_preserved_original_produce_identical_typed_output(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_text('reaction_smiles,expert_score,chemcensor_score,chemcensor_score_bin\n'
                      'C.O>>CO,1,2.0,1\nCC>>C,0,0.0,0\n')
    outputs = []
    for name, command in [
        ('package', [sys.executable, '-m', 'synrxn.reconstruction.families.outcomes.reaction_feasibility._feasibility']),
        ('historical', [sys.executable, '-c',
                        'import sys; from synrxn.reconstruction.provenance.compat import load_legacy_module; '
                        'load_legacy_module({"__name__": "__main__"}, "script/build_feasibility_dataset.py")']),
    ]:
        output = tmp_path / (name + '.csv.gz')
        report = tmp_path / (name + '.json')
        subprocess.run([*command, '--ursa', str(source), '--out', str(output),
                        '--report', str(report), '--skip-checksum'],
                       cwd=REPO, check=True, capture_output=True, text=True)
        assert json.loads(report.read_text())['datasets'][0]['accepted_rows'] == 2
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]

    # The top-level CLI must find package modules with a different working directory.
    output = tmp_path / 'cli.csv.gz'
    assert main(['build', '--feasibility', '--cwd', str(tmp_path), '--',
                 '--ursa', str(source), '--out', str(output),
                 '--report', str(tmp_path / 'cli.json'), '--skip-checksum']) == 0
    assert output.read_bytes() == outputs[0]


def test_report_defaults_use_run_cache_and_keep_explicit_paths(monkeypatch):
    from synrxn.reconstruction.families.outcomes.reaction_feasibility import _feasibility as feasibility
    monkeypatch.setattr(sys, 'argv', ['builder', '--ursa', 'input.csv', '--out', 'output.csv.gz'])
    assert feasibility.parse_args().report == Path('.cache/synrxn/runs/builders/feasibility_build_report.json')
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--report', 'chosen/report.json'])
    assert feasibility.parse_args().report == Path('chosen/report.json')


def test_evidence_copy_uses_frozen_source_and_mcsa_size_checks_still_fail_on_change(tmp_path):
    import shutil
    from synrxn.reconstruction.families.mechanisms.mechanistic_prediction import _mcsa_views as mcsa

    relative = 'script/prepare_mcsa_task.py'
    destination = tmp_path / 'snapshot.py'
    compat._SourceCopies(shutil).copyfile(REPO / relative, destination)
    assert destination.read_bytes() == source_path(relative).read_bytes()
    lock = {'assets': [{'path': relative, 'bytes': source_path(relative).stat().st_size,
                        'sha256': hashlib.sha256(source_path(relative).read_bytes()).hexdigest()}]}
    mcsa.verify_assets(lock)
    lock['assets'][0]['bytes'] += 1
    import pytest
    with pytest.raises(ValueError, match='Missing/changed'):
        mcsa.verify_assets(lock)


def test_native_driver_redirects_only_the_retired_adapter_command():
    calls = []
    proxy = compat._PackagedMcsaSubprocess(SimpleNamespace(
        run=lambda command, **kwargs: calls.append((command, kwargs))))
    proxy.run([sys.executable, str(REPO / 'script/prepare_mcsa_task.py'), '--check-backend'], check=True)
    proxy.run(['converter-python', '/external/scripts/rebuild_mcsa.py', '--jobs', '2'])
    assert calls == [([sys.executable, '-m', 'synrxn.reconstruction.families.mechanisms.mechanistic_prediction._mcsa', '--check-backend'], {'check': True}),
                     (['converter-python', '/external/scripts/rebuild_mcsa.py', '--jobs', '2'], {})]


def test_quantumpioneer_fetch_uses_the_packaged_range_downloader(tmp_path, monkeypatch):
    from synrxn.reconstruction.expansion import fetch as expansion_fetch
    from synrxn.reconstruction.families.physics.reaction_rates import _quantumpioneer as quantumpioneer
    import requests

    payload = b'x' * (3 * 1024 * 1024)
    digest = hashlib.md5(payload).hexdigest()
    metadata = {'metadata': {'license': {'id': 'cc-by-4.0'}}, 'files': [
        {'key': 'fixture.csv', 'size': len(payload), 'checksum': 'md5:' + digest,
         'links': {'self': 'https://example.invalid/pinned-file'}}]}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: metadata)
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: response)
    monkeypatch.setattr(quantumpioneer, 'RAW', tmp_path)
    monkeypatch.setattr(quantumpioneer, 'ASSETS', {'fixture.csv': {'bytes': len(payload), 'md5': digest}})
    calls = []

    def ranged_download(asset, destination, workers):
        calls.append((asset, workers))
        destination.write_bytes(payload)

    monkeypatch.setattr(expansion_fetch, 'ranged_download', ranged_download)
    quantumpioneer.fetch()
    assert (tmp_path / 'fixture.csv').read_bytes() == payload
    assert calls[0][0]['url'] == 'https://example.invalid/pinned-file'
    assert calls[0][1] == 10


def test_frozen_import_rewrite_is_scoped_and_uses_package_facades():
    source = '''from pathlib import Path, PurePosixPath
from kdd.coverage import rights
from kdd.reproduce import local_path
from kdd import reproduce, prepare_ncs_yield_splits as assay
from kdd.prepare_ncs_yield_splits import SOURCE_NAME
'''
    rewritten = ast.fix_missing_locations(compat._HistoricalImports().visit(ast.parse(source)))
    rendered = ast.unparse(rewritten)
    assert 'HistoricalPath as Path' in rendered
    assert 'from pathlib import PurePosixPath' in rendered
    assert 'from synrxn.reconstruction.coverage import rights' in rendered
    assert 'from synrxn.reconstruction.families.suite import local_path' in rendered
    assert 'suite as reproduce' in rendered
    assert 'assay as assay' in rendered
    assert 'from synrxn.reconstruction.families.outcomes.reaction_success._bh_hte_ood import SOURCE_NAME' in rendered
    assert 'from kdd' not in rendered


def test_frozen_callers_can_supply_ordinary_path_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'REPO', tmp_path)
    current = tmp_path / 'Data/metadata/reconstruction/input.json'
    current.parent.mkdir(parents=True)
    current.write_text('{"verified": true}')

    @compat._historical_arguments
    def read(repo, *, input_path):
        return (repo / 'kdd/input.json').read_text(), input_path.read_text()

    assert read(tmp_path, input_path=tmp_path / 'kdd/input.json') == (
        '{"verified": true}', '{"verified": true}')
    assert not (tmp_path / 'kdd').exists()


def test_frozen_kernel_text_digests_keep_the_original_string_semantics():
    from synrxn.reconstruction.families.biocatalysis import _source_tables as bio

    value = 'paper:A\npaper:B\nparent:X\nparent:Y'
    assert bio.digest(value) == hashlib.sha256(value.encode()).hexdigest()
