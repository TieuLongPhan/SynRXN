"""Selection, source-boundary and corruption checks for family/task commands."""
import gzip
import hashlib
import json
import subprocess
import sys

import pytest

from synrxn.reconstruction.families import _selection as tasks
from synrxn.reconstruction.families import _tables as tables
from synrxn.reconstruction.families._entries import ROOT, sync_entries
from synrxn.reconstruction.paths import REPO, physical_path


def test_family_tree_covers_the_registry_with_one_module_per_binding():
    registry = tasks.read_registry()
    summary = sync_entries(registry, check=True)
    assert summary == {'entries': 146, 'updated': [], 'obsolete': []}
    task_folders = {(path.parent.parent.name, path.parent.name)
                    for path in ROOT.glob('*/*/*.py')}
    assert task_folders == {(t['family_id'], t['id']) for t in registry['tasks']}
    assert {family for family, _ in task_folders} == {f['id'] for f in registry['families']}
    assert not list(ROOT.glob('*/tasks'))


def test_table_entry_point_selects_one_binding_and_retains_source_closure():
    module = 'synrxn.reconstruction.families.selectivity.product_distribution.catechol'
    result = subprocess.run([sys.executable, '-S', '-m', module, 'plan'],
                            cwd=REPO, capture_output=True, text=True, check=True)
    selected = json.loads(result.stdout)
    assert selected['task_ids'] == ['product_distribution']
    assert selected['counts']['task_table_bindings'] == selected['counts']['unique_tables'] == 1
    assert selected['tables'][0]['key'] == 'C06/catechol'
    assert 'C06/claisen' in selected['expansion_closure_table_keys']
    assert 'C06/single_solvent' in selected['expansion_closure_table_keys']


def test_pmg_284_build_runs_the_table_recipe_and_matches_the_frozen_release(tmp_path):
    module = 'synrxn.reconstruction.families.analysis.analytical_interpretation.nmr_pmg_284_spectra'
    subprocess.run([sys.executable, '-m', module, 'build', '--output', str(tmp_path)],
                   cwd=REPO, capture_output=True, text=True, check=True)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    comparison = json.loads((tmp_path / 'table_comparison.json').read_text())
    assert [step['id'] for step in manifest['steps']] == ['table_recipe']
    assert manifest['regenerated_selected_tables'] == manifest['regenerated_closure_tables'] == 1
    assert comparison['observed_rows'] == 999
    assert comparison['observed_sha256'] == comparison['expected_sha256']


@pytest.mark.parametrize('offline', [True, False])
def test_pmg_284_recovers_pinned_inputs_before_build(tmp_path, monkeypatch, offline):
    from synrxn.reconstruction.families.analysis.analytical_interpretation import nmr_pmg_284_spectra as pmg

    repo, cache = tmp_path / 'repo', tmp_path / 'cache'
    config = repo / pmg.CALIBRATION
    config.parent.mkdir(parents=True)
    config.write_bytes(b'calibration')
    monkeypatch.setattr(pmg, 'CALIBRATION_SHA256', hashlib.sha256(b'calibration').hexdigest())
    monkeypatch.setattr(pmg, 'SPECTRA_SHA256', hashlib.sha256(b'spectra').hexdigest())
    cached_spectra = cache / pmg.SPECTRA.relative_to('Data')
    if offline:
        cached_spectra.parent.mkdir(parents=True)
        cached_spectra.write_bytes(b'spectra')
    monkeypatch.setattr(pmg, '_download_spectra', lambda path: path.write_bytes(b'spectra'))
    result = pmg.acquire(repo, cache, offline)
    assert result == ('cached native inputs' if offline else 'upstream and local inputs')
    assert physical_path(pmg.SPECTRA, repo).read_bytes() == b'spectra'


def test_every_family_and_task_resolves_exactly_the_current_source_inventory():
    registry = tables.reproduce.read_json(REPO / 'kdd/task_registry.json')
    all_plan = tasks.scope_plan(select_all=True)
    assert all_plan['counts']['families'] == 12
    assert all_plan['counts']['tasks'] == 41
    assert all_plan['counts']['unique_tables'] == 105
    assert all_plan['counts']['overlapping_views'] == 7
    selected = set()
    for family in registry['families']:
        plan = tasks.scope_plan(families=[family['id']])
        assert set(plan['task_ids']) == set(family['task_ids'])
        selected.update((t['namespace'], t['key']) for t in plan['tables'])
    for task in registry['tasks']:
        assert tasks.scope_plan(tasks=[task['id']])['task_ids'] == [task['id']]
    assert selected == {(t['namespace'], t['key']) for t in all_plan['tables']}


def test_combined_selectors_deduplicate_shared_tables_but_preserve_contracts():
    plan = tasks.scope_plan(tasks=['reaction_success', 'yield_conversion', 'reaction_success'])
    ncs = [t for t in plan['tables'] if t['key'] == 'yield/buchwald_hartwig_ood']
    assert len(ncs) == 1
    assert len([b for b in plan['bindings'] if b['key'] == ncs[0]['key']]) == 2
    assert all('release_eligible' in b for b in plan['bindings'])


def test_invalid_or_ambiguous_selectors_fail_without_creating_outputs():
    registry = tables.reproduce.read_json(REPO / 'kdd/task_registry.json')
    for kwargs in [{}, {'tasks': ['made_up']}, {'families': ['made_up']},
                   {'select_all': True, 'tasks': ['reaction_success']}]:
        with pytest.raises(ValueError):
            tasks.select_tasks(registry, **kwargs)


def test_plan_never_hashes_or_opens_observations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Plan must not hash observations')
    monkeypatch.setattr(tables.reproduce, 'sha', forbidden)
    assert tasks.scope_plan(tasks=['reaction_feasibility'])['counts']['unique_tables'] == 2


def test_native_adapter_closure_exposes_process_side_effect_and_minot_dependency():
    process = tasks.scope_plan(tasks=['process_control'])
    assert process['coverage_adapters'] == ['process']
    assert process['coverage_output_adapter_closure'] == ['impurity', 'process']
    assert 'impurity' in process['coverage_branches']
    impurity = tasks.scope_plan(tasks=['impurity_prediction'])
    assert impurity['coverage_adapters'] == ['impurity']
    assert impurity['coverage_branches'] == ['impurity']
    knowledge = tasks.scope_plan(tasks=['literature_extraction'])
    assert knowledge['expansion_sources'] == ['C02']
    assert len(knowledge['expansion_closure_table_keys']) == 5


def small_plan(repo):
    path = repo / 'Data/tiny.csv.gz'
    path.parent.mkdir(parents=True)
    with gzip.open(path, 'wt') as handle:
        handle.write('id,value\nr0,1\nr1,2\n')
    record = {'namespace': 'canonical', 'key': 'tiny', 'path': 'Data/tiny.csv.gz',
              'expected_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'expected_bytes': path.stat().st_size, 'expected_rows': 2,
              'format': 'csv.gz', 'role': 'primary'}
    return {'selection': {'tasks': ['tiny']}, 'counts': {'unique_tables': 1},
            'canonical_reproduction': 'frozen inputs verified; native importers are not run',
            'tables': [record], 'expansion_sources': [], 'coverage_adapters': [],
            'coverage_closure_table_keys': [], 'expansion_closure_table_keys': []}


def test_canonical_only_run_reports_verification_without_regeneration(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    plan = small_plan(repo)
    monkeypatch.setattr(tables, 'dependency_records', lambda *args: [])
    monkeypatch.setattr(tables.reproduce, 'environment', lambda *args: {'python': 'fixture'})
    result = tables.run_scope(plan, tmp_path / 'run', mode='rebuild', check_rows=True, repo=repo)
    assert result['status'] == 'passed'
    assert result['verified_selected_tables'] == result['verified_only_canonical_tables'] == 1
    assert result['regenerated_selected_tables'] == result['regenerated_closure_tables'] == 0
    assert result['predictive_models_run'] == 0
    assert result['steps'] == []
    audit = json.loads((tmp_path / 'run/dataset_verification.json').read_text())
    assert audit[0]['row_count_status'] == 'verified'


def test_corrupted_data_fails_before_any_rebuild(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    plan = small_plan(repo)
    (repo / plan['tables'][0]['path']).write_bytes(b'corrupted')
    monkeypatch.setattr(tables, 'dependency_records', lambda *args: [])
    monkeypatch.setattr(tables, 'rebuild_expansion', lambda *args: pytest.fail('Must verify first'))
    result = tables.run_scope(plan, tmp_path / 'run', mode='rebuild', repo=repo)
    assert result['status'] == 'failed'
    assert 'verification failed' in result['errors'][0]
    assert result['regenerated_selected_tables'] == 0


def test_reports_never_overwrite_inputs_or_an_existing_run(tmp_path):
    repo = tmp_path / 'repo'
    plan = small_plan(repo)
    occupied = tmp_path / 'occupied'
    occupied.mkdir()
    (occupied / 'review.txt').write_text('reviewed')
    for output in [repo, repo / 'script/new', repo / 'kdd/new', repo / 'ncs/new',
                   repo / 'Data/tables/new', repo / 'Data/metadata/new',
                   repo / 'synrxn/new', repo / 'archive/research/previous/new', occupied]:
        with pytest.raises(ValueError):
            tables.run_scope(plan, output, repo=repo)
    assert (occupied / 'review.txt').read_text() == 'reviewed'


def test_run_cache_is_writable_without_allowing_native_cache_overwrites(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    plan = small_plan(repo)
    monkeypatch.setattr(tables, 'dependency_records', lambda *args: [])
    monkeypatch.setattr(tables.reproduce, 'environment', lambda *args: {'python': 'fixture'})
    with pytest.raises(ValueError):
        tables.run_scope(plan, repo / '.cache/native-source/new', repo=repo)
    output = repo / '.cache/synrxn/runs/fixture'
    result = tables.run_scope(plan, output, repo=repo)
    assert result['status'] == 'passed'
    assert (output / 'manifest.json').is_file()


def test_scope_detects_changed_physical_shim_with_unchanged_kernel(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    plan = small_plan(repo)
    relative = 'kdd/coverage/rights.py'
    shim = repo / relative
    archived = repo / 'archive/source-code' / relative
    shim.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    original_shim = '# physical compatibility shim\n'
    shim.write_text(original_shim)
    archived.write_text('# fixed historical source\n')
    kernel_hash = hashlib.sha256(archived.read_bytes()).hexdigest()
    record = {'namespace': 'metadata', 'key': relative, 'path': relative,
              'expected_sha256': kernel_hash}
    monkeypatch.setattr(tables, 'dependency_records', lambda *args: [record])
    def mutate_after_snapshot(*args):
        shim.write_text('# changed while the scope run was active\n')
        return {'python': 'fixture'}
    monkeypatch.setattr(tables.reproduce, 'environment', mutate_after_snapshot)
    result = tables.run_scope(plan, tmp_path / 'run', mode='verify', repo=repo)
    assert result['status'] == 'failed'
    assert result['physical_post_run_changes'] == [relative]
    assert relative in result['dependency_post_run_changes']
    assert (tmp_path / 'run/snapshot' / relative).read_text() == original_shim
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == kernel_hash
    assert tables.reproduce.audit_file(record, repo)['status'] == 'verified'



@pytest.mark.parametrize('arguments,task_ids,unique_tables', [
    (['plan', '--task', 'reaction_success'], ['reaction_success'], 5),
    (['families', 'plan', '--family', 'outcomes'],
     ['yield_conversion', 'reaction_success', 'reaction_feasibility'], None),
])
def test_metadata_planning_needs_no_optional_packages(arguments, task_ids, unique_tables):
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'synrxn.reconstruction', *arguments],
        cwd=REPO, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert set(plan['task_ids']) == set(task_ids)
    assert plan['counts']['tasks'] == len(task_ids)
    if unique_tables is not None:
        assert plan['counts']['unique_tables'] == unique_tables
