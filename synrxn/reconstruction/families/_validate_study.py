"""Validate the suite protocol and source-qualified HiTEA case preparation."""
from collections import Counter, defaultdict
import argparse
import csv
import gzip
import hashlib
import json
from math import prod
from pathlib import Path

from synrxn.reconstruction.paths import REPO, historical_file, physical_path, source_path

__source_file__ = historical_file('kdd/validate_plan.py')


def validate(repo=REPO):
    """Check the active suite protocol and the frozen HiTEA case inputs."""
    REPO = Path(repo).resolve()
    ROOT = physical_path("kdd", repo=REPO)

    def sha(path):
        with source_path(path, repo=REPO).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    suite = json.loads((ROOT / 'study_protocol.json').read_text())
    registry = json.loads((ROOT / 'task_registry.json').read_text())
    protocol = json.loads((ROOT / 'case_studies/hitea_protocol.json').read_text())
    audit = json.loads((ROOT / 'preparation/audit.json').read_text())
    problems = []

    def check(condition, message):
        if not condition:
            problems.append(message)
    registered_ids = {task['id'] for task in registry['tasks']}
    original = json.loads((REPO / 'synrxn/data/task_protocols.json').read_text())['tasks']
    check(set(registry['legacy_view_map']) == {task['id'] for task in original}, 'Original export view lost')
    check(set(suite['task_ids']) == registered_ids, 'Suite/registry task mismatch')
    check(len(registered_ids) == suite['local_data_backed_task_count'], 'Scientific task count mismatch')
    check(len(registry['families']) == suite['scientific_family_count'], 'Scientific family count mismatch')
    future_ids = {task['id'] for task in registry['future_tasks']}
    excluded_ids = {task['id'] for task in registry['excluded_tasks']}
    check(future_ids == set(suite['future_task_ids']), 'Roadmap task mismatch')
    check(len(registered_ids) + len(future_ids) == registry['summary']['total_task_definitions'] == suite['total_task_definitions'], 'Total task definition count mismatch')
    check(excluded_ids == set(suite['excluded_task_ids']), 'Excluded task mismatch')
    check('multistep_planning' in excluded_ids and (not (registered_ids | future_ids) & excluded_ids), 'Route-planning exclusion not respected')
    local_families = sum((bool(family['task_ids']) for family in registry['families']))
    roadmap_families = len(registry['families']) - local_families
    check(local_families == suite['families_with_local_task_data'], 'Local family readiness mismatch')
    check(roadmap_families == suite['roadmap_only_family_count'], 'Roadmap family readiness mismatch')
    check(set(suite['family_baselines']) == {family['id'] for family in registry['families']}, 'Family baseline policy mismatch')
    for name, digest in registry['source_sha256'].items():
        check(sha(REPO / name) == digest, 'Task registry source changed: ' + name)
    suite_budget = suite['compute_allocation']
    check(suite_budget['roadmap_tasks_included'] is False, 'Roadmap tasks incorrectly counted as executable bindings')
    local_binding_count = sum((task['primary_table_count'] for task in registry['tasks']))
    binding_count = sum((task.get('release_eligible_table_count', task['primary_table_count']) for task in registry['tasks']))
    check(local_binding_count == suite_budget.get('local_primary_task_dataset_bindings', local_binding_count), 'Local binding inventory mismatch')
    suite_search = binding_count * suite_budget['trainable_baseline_families_per_binding'] * suite_budget['validation_trials_per_family']
    suite_repeats = binding_count * suite_budget['trainable_baseline_families_per_binding'] * (len(suite_budget['stochastic_final_seeds']) - 1)
    check(binding_count == suite_budget['primary_task_dataset_bindings'], 'Suite binding budget mismatch')
    check(suite_search == suite_budget['search_execution_ceiling'], 'Suite search budget mismatch')
    check(suite_repeats == suite_budget['additional_seed_execution_ceiling'], 'Suite repeated-seed budget mismatch')
    check(suite_search + suite_repeats == suite_budget['suite_execution_ceiling'], 'Suite execution ceiling mismatch')
    for model in protocol['models']:
        if 'search' in model:
            check(prod((len(values) for values in model['search'].values())) == 8, 'Search budget mismatch: ' + model['id'])
    search_fits = 4 * 8 * 5 * (3 + 1)
    seed_fits = len(protocol['training']['repeat_models']) * (len(protocol['training']['stochastic_final_seeds']) - 1) * 5 * (3 + 1)
    check(search_fits == protocol['compute_allocation']['core_search_fit_ceiling'], 'Core fit budget mismatch')
    check(seed_fits == protocol['compute_allocation']['core_additional_seed_fit_ceiling'], 'Seed fit budget mismatch')
    check(search_fits + seed_fits == protocol['compute_allocation']['core_fit_ceiling_excluding_constant'], 'Total fit budget mismatch')
    source = source_path(protocol['population']['canonical_dataset'], repo=REPO)
    check(sha(REPO / 'kdd/prepare_cohort.py') == audit['preparation_code_sha256'], 'Preparation code changed since audit')
    check(sha(source) == protocol['population']['canonical_sha256'] == audit['source']['canonical_sha256'], 'Canonical source hash mismatch')
    check(sha(ROOT / 'preparation/raw/hitea.csv') == protocol['population']['upstream_sha256'], 'Raw source hash mismatch')
    check(sha(REPO / 'Data/yield/buchwald_hartwig_ood.csv.gz') == audit['bh_replication']['canonical_sha256'], 'BH source hash mismatch')
    membership = source_path(ROOT / protocol['memberships']['artifact'], repo=REPO)
    check(sha(membership) == audit['membership_sha256'], 'Membership hash mismatch')
    with gzip.open(source, 'rt', newline='', encoding='utf8') as handle:
        canonical = list(csv.DictReader(handle))
    canonical_by_id = {row['sample_id']: row for row in canonical}
    with gzip.open(membership, 'rt', encoding='utf8') as handle:
        records = [json.loads(line) for line in handle]
    observed = {row['sample_id'] for row in records}
    expected = {row['sample_id'] for row in canonical if row['reactants'] and row['products'] and row['screen_id']}
    exclusions = json.loads((ROOT / 'preparation/exclusions.json').read_text())
    excluded = {row['sample_id'] for row in exclusions}
    check(len(observed) == len(records) == protocol['population']['eligible_rows'], 'Membership count or unique IDs mismatch')
    check(observed == expected, 'Membership does not cover the eligible records')
    check(excluded == set(canonical_by_id) - observed, 'Exclusions do not partition the canonical source')
    check(len(excluded) == len(exclusions) == audit['source']['excluded_rows'], 'Exclusion count mismatch')
    relation_folds = {field: defaultdict(set) for field in protocol['memberships']['relation_closure']}
    panel_groups, panel_candidates = (defaultdict(set), defaultdict(set))
    components = set()
    cell_counts = Counter((record['cell_id'] for record in records))
    for record in records:
        row = canonical_by_id[record['sample_id']]
        check(canonical[record['canonical_row']]['sample_id'] == record['sample_id'], 'Incorrect source row locator')
        for field, values in relation_folds.items():
            if row.get(field):
                values[row[field]].add(record['outer_fold'])
        inputs = record['inputs']
        check(set(inputs) == {'reactants', 'products', 'source_context'}, 'Unexpected model input block')
        check(inputs['reactants'] == row['reactants'] and inputs['products'] == row['products'], 'Endpoint mismatch')
        check(set(inputs['source_context']) <= set(audit['source_context_fields']), 'Unpermitted source context field')
        check(record['cell_row_count'] == cell_counts[record['cell_id']], 'Recorded-cell count mismatch')
        panel_groups[record['panel_id']].add((record['component_id'], record['outer_fold']))
        panel_candidates[record['panel_id']].add(record['candidate_id'])
        components.add(record['component_id'])
    for field, values in relation_folds.items():
        check(all((len(folds) == 1 for folds in values.values())), 'Declared relation crosses folds: ' + field)
    check(all((len(groups) == 1 for groups in panel_groups.values())), 'Panel spans predictive groups or folds')
    check(len(components) == protocol['population']['dependence_components'], 'Component count mismatch')
    check(len(cell_counts) == protocol['population']['recorded_input_cells'], 'Recorded input cell count mismatch')
    check(len(panel_candidates) == protocol['population']['all_panels'], 'Panel count mismatch')
    eligible_panels = {panel for panel, values in panel_candidates.items() if len(values) >= protocol['population']['minimum_ranking_candidates']}
    check(len(eligible_panels) == protocol['population']['ranking_panels'], 'Ranking panel count mismatch')
    check(all((record['ranking_eligible'] == (record['panel_id'] in eligible_panels) for record in records)), 'Ranking eligibility flags mismatch')
    check({str(key): value for key, value in Counter((record['outer_fold'] for record in records)).items()} == audit['outer_fold_rows'], 'Fold counts mismatch')
    cards_path = ROOT / 'preparation/panel_cards.csv'
    cards = list(csv.DictReader(cards_path.open()))
    check({row['panel_id'] for row in cards} == set(panel_candidates), 'Panel cards do not cover the candidate population')
    independent = {row['panel_id'] for row in cards if row['independent_probability_sample'] == 'True'}
    targeted = {row['panel_id'] for row in cards if row['separate_targeted_sample'] == 'True'}
    check(len(independent) == 200 and len(targeted) == 60 and (not independent & targeted), 'Review sample sizes or separation mismatch')
    check(sha(cards_path) == audit['initial_panel_cards_sha256'], 'Initial panel cards changed; reviewed annotations need a separate version')
    report = {'passed': not problems, 'eligible_rows_checked': len(records), 'ranking_panels_checked': len(eligible_panels), 'panel_cards_prepared': len(cards), 'independent_review_sample': len(independent), 'scientific_families': len(registry['families']), 'scientific_task_definitions': len(registered_ids), 'families_with_local_task_data': local_families, 'roadmap_only_families': roadmap_families, 'roadmap_task_definitions': len(future_ids), 'excluded_task_ids': sorted(excluded_ids), 'total_task_definitions': len(registered_ids) + len(future_ids), 'legacy_export_variants_preserved': len(registry['legacy_view_map']), 'suite_execution_ceiling': suite_search + suite_repeats, 'optional_hitea_case_fit_ceiling': search_fits + seed_fits, 'problems': problems}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Write a validation report here; default prints without modifying inputs")
    args = parser.parse_args(argv)
    report = validate()
    if args.output is not None:
        output = args.output.resolve()
        if output.is_relative_to(REPO / "archive"):
            parser.error("Historical evidence is read-only; choose a run output outside archive/")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == '__main__':
    raise SystemExit(main())
