"""Update the active study protocol from the scientific task registry."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from ..paths import REPO, historical_file, physical_path

__source_file__ = historical_file('kdd/coverage/sync_plan.py')


def synchronize(registry: dict, protocol: dict) -> dict:
    """Refresh derived inventory and budget counts, retaining evaluation policy."""
    suite = deepcopy(protocol)
    summary = registry['summary']
    for name in ('total_task_definitions', 'families_with_local_task_data',
                 'initial_roadmap_task_definitions', 'promoted_roadmap_task_definitions',
                 'release_eligible_task_definitions'):
        suite[name] = summary[name]
    suite['task_ids'] = [task['id'] for task in registry['tasks']]
    suite['future_task_ids'] = [task['id'] for task in registry['future_tasks']]
    suite['excluded_task_ids'] = [task['id'] for task in registry['excluded_tasks']]
    suite['scientific_family_count'] = len(registry['families'])
    suite['local_data_backed_task_count'] = len(registry['tasks'])
    suite['roadmap_only_family_count'] = summary['roadmap_only_families']
    suite['data_policy']['coverage_tables'] = summary['coverage_tables_covered']
    suite['scope_coverage_audit'] = '../../../doc/data_concept.rst'
    suite['reproducibility']['entrypoint'] = 'python -m synrxn.reconstruction full'
    suite['reproducibility']['full_audit_command'] = (
        'python -m synrxn.reconstruction full --profile preparation --check-rows --rebuild-coverage')
    budget = suite['compute_allocation']
    budget['local_primary_task_dataset_bindings'] = summary['primary_task_dataset_bindings']
    bindings = summary['release_eligible_primary_task_dataset_bindings']
    budget['primary_task_dataset_bindings'] = bindings
    budget['search_execution_ceiling'] = (
        bindings * budget['trainable_baseline_families_per_binding'] * budget['validation_trials_per_family'])
    budget['additional_seed_execution_ceiling'] = (
        bindings * budget['trainable_baseline_families_per_binding'] * (len(budget['stochastic_final_seeds']) - 1))
    budget['suite_execution_ceiling'] = budget['search_execution_ceiling'] + budget['additional_seed_execution_ceiling']
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=physical_path('kdd/task_registry.json'))
    parser.add_argument('--protocol', type=Path, default=physical_path('kdd/study_protocol.json'))
    args = parser.parse_args(argv)
    path = args.protocol.resolve()
    if path.is_relative_to(REPO / 'archive'):
        parser.error('Historical evidence is read-only; use the active study protocol')
    registry = json.loads(args.registry.read_text(encoding='utf8'))
    suite = synchronize(registry, json.loads(path.read_text(encoding='utf8')))
    path.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(json.dumps({'local_tasks': len(suite['task_ids']),
                      'future_tasks': len(suite['future_task_ids']),
                      'release_bindings': suite['compute_allocation']['primary_task_dataset_bindings'],
                      'provisional_execution_slots': suite['compute_allocation']['suite_execution_ceiling']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
