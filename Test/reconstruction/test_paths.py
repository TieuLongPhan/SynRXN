"""Logical source identities survive physical storage changes."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from synrxn.reconstruction import paths
from synrxn.reconstruction.families import suite


def test_aliases_use_longest_prefix_and_reverse_to_stable_keys(tmp_path):
    # Several retired physical spellings can share one canonical logical key.
    canonical = {}
    for logical, current in paths.RELOCATIONS:
        canonical.setdefault(current, logical)
    for logical, current in paths.RELOCATIONS:
        target = tmp_path / current / 'nested/input.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}\n')
        name = logical + '/nested/input.json'
        assert paths.physical_path(name, repo=tmp_path) == target
        assert paths.physical_path(tmp_path / name, repo=tmp_path) == target
        assert paths.logical_relative(target, repo=tmp_path) == canonical[current] + "/nested/input.json"
    assert paths.repo_path('kdd', 'literature', 'nested/input.json', repo=tmp_path) == (
        tmp_path / 'Data/metadata/literature/nested/input.json')


def test_native_workspaces_keep_existing_historical_locations(tmp_path):
    logical = tmp_path / 'kdd/task_registry.json'
    logical.parent.mkdir()
    logical.write_text('historical workspace')
    relocated = tmp_path / 'Data/metadata/reconstruction/task_registry.json'
    relocated.parent.mkdir(parents=True)
    relocated.write_text('current checkout')
    assert paths.physical_path('kdd/task_registry.json', repo=tmp_path) == logical
    assert paths.physical_path(logical, repo=tmp_path) == logical


def test_source_alias_prioritizes_original_kernel_bytes(tmp_path):
    original = tmp_path / 'archive/source-code/kdd/prepare_cohort.py'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'# original kernel\n')
    current = tmp_path / 'Data/metadata/reconstruction/prepare_cohort.py'
    current.parent.mkdir(parents=True)
    current.write_bytes(b'# facade\n')
    assert paths.source_path('kdd/prepare_cohort.py', repo=tmp_path) == original
    assert paths.physical_path('kdd/prepare_cohort.py', repo=tmp_path) == current


def test_resolver_rejects_traversal_and_preserves_external_files(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        paths.physical_path('kdd/../../outside', repo=tmp_path)
    external = tmp_path.parent / 'external-input.csv'
    assert paths.physical_path(external, repo=tmp_path) == external
    with pytest.raises(ValueError, match='outside'):
        paths.logical_relative(external, repo=tmp_path)


def test_scoped_frozen_paths_keep_provenance_and_use_physical_io(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'REPO', tmp_path)
    logical = paths.HistoricalPath(tmp_path / 'kdd/coverage/example.json')
    current = tmp_path / 'Data/metadata/reconstruction/coverage/example.json'
    logical.parent.mkdir(parents=True)
    logical.write_text('{"answer": 42}\n')
    assert str(logical) == str(tmp_path / 'kdd/coverage/example.json')
    assert logical.relative_to(tmp_path).as_posix() == 'kdd/coverage/example.json'
    assert os.fspath(logical) == str(current)
    assert logical.read_bytes() == current.read_bytes()
    assert logical.stat().st_size == current.stat().st_size
    assert logical.resolve(strict=True) == logical
    assert not (tmp_path / 'kdd').exists()
    assert [p.relative_to(tmp_path).as_posix() for p in logical.parent.glob('*.json')] == [
        'kdd/coverage/example.json']
    assert [p.relative_to(tmp_path).as_posix() for p in logical.parent.iterdir()] == [
        'kdd/coverage/example.json']
    renamed = logical.rename(logical.with_name('renamed.json'))
    assert renamed.read_text() == '{"answer": 42}\n'
    assert not logical.exists()
    renamed.unlink()
    assert not current.with_name('renamed.json').exists()


def test_scoped_paths_work_with_builtin_and_pandas_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'REPO', tmp_path)
    current = tmp_path / 'Data/tables/canonical/yield/example.csv'
    current.parent.mkdir(parents=True)
    current.write_text('reaction,yield\nC>>C,42\n')
    logical = paths.HistoricalPath(tmp_path / 'Data/yield/example.csv')
    with open(logical) as handle:
        assert handle.readline() == 'reaction,yield\n'
    import pandas as pd
    assert pd.read_csv(logical)['yield'].tolist() == [42]
    assert logical.relative_to(tmp_path).as_posix() == 'Data/yield/example.csv'


def test_stringifying_property_reader_and_writer_use_storage_aliases(tmp_path, monkeypatch):
    from synrxn.reconstruction.families.physics import _property as property_builder
    monkeypatch.setattr(paths, 'REPO', tmp_path)
    current = tmp_path / 'Data/tables/canonical/property/input.csv'
    current.parent.mkdir(parents=True)
    current.write_text('reaction,barrier\nC>>C,4.2\n')
    frame = property_builder.curate_reactions(
        tmp_path / 'Data/property/input.csv', data_name='fixture',
        rxn_col='reaction', target_cols='barrier',
        out_csv=tmp_path / 'Data/property/output.csv')
    assert frame['barrier'].tolist() == [4.2]
    assert current.with_name('output.csv').is_file()
    assert not (tmp_path / 'Data/property').exists()


def test_actual_frozen_rights_and_adapter_reads_relocated_dependencies():
    from synrxn.reconstruction.coverage import rights
    from synrxn.reconstruction.families.physics.activation_barrier import _solution_context as solution_context

    catalogue = json.loads(paths.physical_path(
        'Data/coverage_expansion/solution_context/catalog.json').read_text())
    assert all(rights.source_has_evidence(source, catalogue, paths.REPO)
               for source in catalogue['sources'])
    solution_context.verify_assets()
    assert solution_context.LOCK.relative_to(paths.REPO).as_posix() == (
        'kdd/coverage/adapters/solution_context_sources.json')
    assert solution_context.__file__ == str(paths.REPO / 'synrxn/reconstruction/families/physics/activation_barrier/_solution_context.py')
    assert solution_context.__source_file__ == str(paths.REPO / 'kdd/coverage/adapters/solution_context.py')
    original = paths.source_path(solution_context.__source_file__)
    assert solution_context.digest(solution_context.__source_file__) == hashlib.sha256(original.read_bytes()).hexdigest()


@pytest.mark.parametrize('name', ['saccharose_hydrolysis', 'pmg_284', 'pmg_287'])
def test_nmr_spectra_keep_native_keys_and_old_physical_read_aliases(tmp_path, name):
    native = f'Data/coverage_expansion/process/nmr_{name}_snapshots.jsonl.gz'
    former = f'Data/tables/coverage/process/nmr_{name}_snapshots.jsonl.gz'
    current = tmp_path / f'Data/tables/coverage/process/nmr_{name}_spectra.jsonl.gz'
    current.parent.mkdir(parents=True)
    current.write_bytes(b'native compressed spectra')
    assert paths.physical_path(native, tmp_path) == current
    assert paths.physical_path(former, tmp_path) == current
    assert paths.source_path(former, tmp_path) == current
    assert paths.logical_relative(current, tmp_path) == native


def test_retired_metadata_is_read_only_fallback_not_an_output_alias(tmp_path):
    archived = tmp_path / 'archive/retired-metadata/reconstruction/work_plan.json'
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b'{"historical": true}\n')
    current = tmp_path / 'Data/metadata/reconstruction/work_plan.json'
    assert paths.source_path('kdd/work_plan.json', tmp_path) == archived
    assert paths.source_path(current, tmp_path) == archived
    # Ordinary output resolution never writes into preserved evidence.
    assert paths.physical_path('kdd/work_plan.json', tmp_path) == current
    current.parent.mkdir(parents=True)
    current.write_bytes(b'{"native workspace": true}\n')
    assert paths.source_path('kdd/work_plan.json', tmp_path) == current
    assert archived.read_bytes() == b'{"historical": true}\n'


def test_ordinary_data_paths_are_never_redirected_to_the_source_archive():
    relative = 'Data/yield/buchwald_hartwig_ood.csv.gz'
    expected = paths.physical_path(relative)
    assert paths.source_path(relative) == expected
    assert suite.local_path(paths.REPO, relative) == expected
    assert expected.is_file()
