"""Scientific coverage is an executable join, with disagreement made explicit."""
from copy import deepcopy
import json

from synrxn.benchmarks.protocols import task_protocol_catalog
from synrxn.reconstruction.paths import physical_path


def _registry():
    return json.loads(physical_path('kdd/task_registry.json').read_text())


def test_all_scientific_tasks_have_metric_and_split_profiles():
    report = task_protocol_catalog()
    assert report['status'] == 'passed', report['issues']
    assert report['summary'] == {
        'scientific_families': 12, 'scientific_tasks': 41,
        'metric_profiles': 41, 'split_protocols': 41,
        'dummy_protocols': 41, 'generic_dummy_defaults': 35,
        'specialized_dummy_defaults': 6,
    }
    assert len({row['task_id'] for row in report['tasks']}) == 41
    assert all(row['registry_metrics'] for row in report['tasks'])
    assert all(row['dummy_protocols'] for row in report['tasks'])
    json.dumps(report, allow_nan=False)


def test_task_additions_and_family_disagreements_cannot_report_complete():
    registry = deepcopy(_registry())
    added = deepcopy(registry['tasks'][0])
    added['id'] = 'unimplemented_scientific_task'
    registry['tasks'].append(added)
    first = registry['tasks'][0]
    first['family_id'] = next(f['id'] for f in registry['families']
                              if f['id'] != first['family_id'])
    report = task_protocol_catalog(registry)
    assert report['status'] == 'failed'
    assert report['issues']['missing_metric_profiles'] == [added['id']]
    assert report['issues']['missing_split_protocols'] == [added['id']]
    assert report['issues']['missing_dummy_protocols'] == [added['id']]
    assert first['id'] in report['issues']['metric_family_mismatches']
    assert first['id'] in report['issues']['split_family_mismatches']


def test_explicit_relative_registry_uses_the_callers_directory(tmp_path, monkeypatch):
    registry = _registry()
    supplied = tmp_path / 'my-registry.json'
    supplied.write_text(json.dumps(registry))
    monkeypatch.chdir(tmp_path)
    report = task_protocol_catalog('my-registry.json')
    assert report['status'] == 'passed'
    assert report['registry']['path'] == str(supplied.resolve())


def test_binding_assignment_labels_are_not_recommended_as_dependency_ids():
    registry = deepcopy(_registry())
    task = registry['tasks'][0]
    task['bindings'] = [{
        'namespace': 'fixture', 'key': 'table',
        'task_fields': {task['id']: {'group_fields': [
            'panel_id', 'native_split', 'derived_split', 'outer_fold', 'scale',
        ]}},
    }]
    report = task_protocol_catalog(registry)
    row = next(row for row in report['tasks'] if row['task_id'] == task['id'])
    metadata = row['binding_split_metadata'][0]
    assert metadata['membership_fields'] == ['native_split', 'derived_split', 'outer_fold']
    assert metadata['candidate_dependency_fields'] == ['panel_id']
    assert metadata['context_or_transfer_fields'] == ['scale']
    assert metadata['metadata_status'] == 'not_source_qualified_by_this_audit'
    assert task['bindings'][0]['task_fields'][task['id']]['group_fields'][1] == 'native_split'
