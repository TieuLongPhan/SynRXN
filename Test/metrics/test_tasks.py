"""Numerical fixtures for task metrics and their scientific denominator rules."""
import json
import math
from importlib import import_module
import subprocess
import sys

import numpy as np
import pytest

from synrxn.reconstruction.paths import REPO
from synrxn.metrics import (
    evaluate_task, get_task_metrics, list_task_metrics, metric_protocol_coverage,
)
from synrxn.metrics import primitives as score


def test_every_registered_scientific_task_has_a_metric_profile():
    registry = json.loads((REPO / 'Data/metadata/reconstruction/task_registry.json').read_text())
    audit = metric_protocol_coverage(registry)
    assert audit['complete']
    assert audit['implemented_task_count'] == 41
    assert audit['family_count'] == 12
    for task in registry['tasks']:
        profile = get_task_metrics(task['id'])
        assert profile.family_id == task['family_id']
        assert profile.default_variant in profile.variants
        module = import_module(f"synrxn.metrics.{profile.family_id}.{profile.task_id}")
        assert profile == module.PROFILE
        for variant in profile.variants.values():
            assert variant.requirements and variant.interpretation and variant.unit_policy
            assert variant.primary_metric and variant.primary_metric != variant.secondary_metric
            assert set(variant.metrics.values()) <= {'minimize', 'maximize', 'declared_by_maximize', 'coverage', 'curve', 'diagnostic'}
        json.dumps(profile.as_dict())


def test_audit_reports_missing_extra_and_mismatched_families():
    audit = metric_protocol_coverage({'tasks': [{'id': 'yield_conversion', 'family_id': 'wrong'}, {'id': 'absent', 'family_id': 'outcomes'}]})
    assert not audit['complete']
    assert audit['missing_task_ids'] == ['absent']
    assert audit['family_mismatches'] == ['yield_conversion']
    assert 'atom_mapping' in audit['extra_task_ids']


def test_discovery_does_not_import_numeric_or_chemical_backends():
    code = "from synrxn.metrics import list_task_metrics; import sys; assert len(list_task_metrics()) == 41; assert not any(n in sys.modules for n in ('numpy', 'pandas', 'rdkit', 'sklearn', 'synkit'))"
    subprocess.run([sys.executable, '-S', '-c', code], cwd=REPO, check=True)


def test_task_dispatch_requires_unit_and_rejects_unrecognized_context():
    with pytest.raises(TypeError, match='unit'):
        evaluate_task('yield_conversion', [1.], [1.])
    with pytest.raises(ValueError, match='unit'):
        evaluate_task('yield_conversion', [1.], [1.], unit='')
    with pytest.raises(TypeError, match='unknown metric context'):
        evaluate_task('yield_conversion', [1.], [1.], unit='%', unt='bad')
    with pytest.raises(KeyError):
        evaluate_task('yield_conversion', [1.], [1.], variant='made_up', unit='%')
    with pytest.raises(KeyError):
        list_task_metrics('made_up')


def test_native_errors_keep_missing_outputs_visible_and_constant_r2_undefined():
    result = evaluate_task('yield_conversion', [1, 2, 3], [2, None, 3], unit='yield percentage points', source_id='fixture', cohort_id='held-out')
    m = result['metrics']
    assert m['coverage'] == pytest.approx(2/3)
    assert m['error_denominator'] == 2 and m['eligible_count'] == 3
    assert m['mae'] == .5 and m['rmse'] == pytest.approx(math.sqrt(.5))
    assert result['primary_metric'] == 'mae'
    assert result['secondary_metric'] == 'rmse'
    assert result['reported_metrics'] == {'mae': m['mae'], 'rmse': m['rmse']}
    assert m['r2'] == .5 and m['spearman'] == 1
    assert m['direction_accuracy'] is None
    assert result['source_id'] == 'fixture' and result['directions']['mae'] == 'minimize'
    assert score.regression([2, 2], [2, 2], unit='kJ/mol')['r2'] is None
    assert score.regression([2, 2], [2, 2], unit='kJ/mol')['spearman'] is None
    assert score.regression([2], [None], unit='kJ/mol')['mae'] is None


def test_signed_direction_requires_explicit_signed_target_and_excludes_zero_reference():
    result = score.regression([-1, 1, 0], [-2, 0, 9], unit='native delta', signed=True)
    assert result['direction_accuracy'] == .5
    assert result['direction_denominator'] == 2
    assert evaluate_task('mutation_effects', [-1, 1], [-1, 1], unit='native delta')['metrics']['direction_accuracy'] == 1


@pytest.mark.parametrize('reference,prediction', [([], []), ([1], []), ([float('nan')], [0]), ([None], [0])])
def test_invalid_regression_reference_cohorts_raise(reference, prediction):
    with pytest.raises(ValueError):
        score.regression(reference, prediction, unit='native')


def test_classification_invalid_outputs_are_all_cohort_failures():
    result = score.classification(['a', 'b', 'a', 'b'], ['a', 'b', None, 'unknown'], labels=['a', 'b'])
    assert result['coverage'] == .5 and result['accuracy'] == .5
    assert result['macro_f1'] == pytest.approx(2/3)
    assert result['balanced_accuracy'] == .5 and result['mcc'] is None
    assert score.classification([1, 1], [1, 1])['mcc'] is None


def test_binary_metrics_have_known_probability_answers_and_explicit_polarity():
    m = score.binary([0, 0, 1, 1], [.1, .4, .35, .8], bins=2)
    assert m['auroc'] == .75
    assert m['average_precision'] == pytest.approx(5/6)
    assert m['brier_score'] == pytest.approx(.158125)
    assert m['accuracy'] == .75
    assert m['expected_calibration_error'] == pytest.approx(.0875)
    assert m['positive_label'] == 1 and m['threshold'] == .5
    assert sum(b['count'] for b in m['reliability_diagram']) == 4


def test_binary_undefined_discrimination_and_extreme_log_loss_are_visible():
    m = score.binary([1, 1], [1, 0])
    assert m['auroc'] is None and m['average_precision'] is None
    assert m['log_loss'] == math.inf
    assert score.binary([0, 1], [0, 1])['log_loss'] == 0
    assert score.binary([0, 1], [float('nan'), 2])['coverage'] == 0
    with pytest.raises(ValueError, match='binary'):
        score.binary(['no', 'yes'], [.1, .9])


def test_binary_missing_probabilities_do_not_disappear_from_accuracy_or_coverage():
    m = score.binary([0, 1, 1, 0], [.1, .9, None, 2])
    assert m['accuracy'] == .5 and m['coverage'] == .5
    assert m['probability_denominator'] == 2 and m['auroc'] == 1


def test_selective_risk_retains_equal_confidence_ties_together():
    m = score.binary([0, 1, 1], [.1, .9, .6])
    curve = m['selective_risk_coverage']
    assert len(curve) == 2
    assert curve[0]['retained_count'] == 2 and curve[0]['coverage'] == pytest.approx(2/3)
    assert curve[-1]['coverage'] == 1


def test_set_recovery_and_unknown_impurity_labels_remain_distinct():
    complete = score.set_recovery([{'a', 'b'}, set()], [{'a', 'c'}, None])
    assert complete['micro_f1'] == .5 and complete['macro_f1'] == .25
    assert complete['observed_recall'] == .5
    assert complete['exact_match'] == 0 and complete['coverage'] == .5
    assert complete['false_edit_rate'] is None and complete['false_edit_denominator'] == 0
    observed = evaluate_task('impurity_prediction', [{'a'}], [{'a', 'unreported'}])['metrics']
    assert observed['observed_recall'] == 1
    assert observed['micro_f1'] is None and observed['false_positives'] is None


def test_false_edit_rate_uses_qualified_unchanged_cases_with_valid_outputs():
    m = score.set_recovery([set(), set(), {'edit'}], [set(), {'bad-edit'}, {'edit'}])
    assert m['false_edit_rate'] == .5 and m['unchanged_reference_count'] == 2
    assert m['false_edit_denominator'] == 2


def test_retrieval_duplicates_and_unknowns_consume_rank():
    result = score.ranked_retrieval([{'a': 1, 'b': 1, 'c': 0}], [['bad', 'a', 'a', 'b']], top_k=(1, 3, 4))
    assert result['top_k'][1]['recall'] == 0
    assert result['top_k'][3]['recall'] == .5
    assert result['top_k'][4]['recall'] == 1
    assert result['top_k'][3]['reciprocal_rank'] == .5
    assert result['invalid_or_duplicate_candidates'] == 2
    ideal = 1 + 1 / math.log2(3)
    assert result['top_k'][3]['ndcg'] == pytest.approx((1 / math.log2(3)) / ideal)


def test_reported_rank_scores_use_the_declared_cutoff():
    reference, prediction = [{'answer': 1}], [['other', 'answer']]
    result = evaluate_task('reaction_retrieval', reference, prediction)
    assert result['reported_metrics'] == {'top_k.5.recall': 1, 'top_k.5.reciprocal_rank': 0.5}
    shorter = evaluate_task('reaction_retrieval', reference, prediction, top_k=(1,))
    assert shorter['metrics']['top_k'][1]['recall'] == 0
    assert shorter['reported_metrics'] == {'top_k.5.recall': None, 'top_k.5.reciprocal_rank': None}


def test_condition_ranking_reports_regret_at_one_and_three_proposals():
    result = evaluate_task(
        'condition_ranking', [{'warm': 1, 'cold': 3}], [['warm', 'cold']], unit='yield percentage points',
    )
    assert result['reported_metrics'] == {
        'budgets.1.mean_observed_regret': 2,
        'budgets.3.mean_observed_regret': 0,
    }


def test_retrieval_unjudged_queries_and_huge_grades_do_not_get_invented_scores():
    result = score.ranked_retrieval([{'a': 0}], [[]], top_k=(1,))
    assert result['top_k'][1]['ndcg'] is None and result['top_k'][1]['recall_denominator'] == 0
    assert result['coverage'] == 0
    assert score.ranked_retrieval([{'a': 2000., 'b': 1.}], [['a']], top_k=(1,))['top_k'][1]['ndcg'] == 1
    assert score.ranked_retrieval([{'a': 1}], [['unknown']], top_k=(1,))['coverage'] == 0


def test_observed_positive_rankings_keep_unjudged_ids_separate_from_invalid_submissions():
    result = score.ranked_retrieval(
        [{'positive': 1, 'unannotated': 0}],
        [['unannotated', 'unreported', 'positive', 'positive', []]],
        top_k=(1, 3, 5), observed_only=True,
    )
    assert result['top_k'][1]['recall'] == 0
    assert result['top_k'][3]['recall'] == 1
    assert result['top_k'][3]['reciprocal_rank'] == pytest.approx(1/3)
    assert result['top_k'][3]['ndcg'] is None
    assert result['top_k'][3]['ndcg_denominator'] == 0
    assert result['unjudged_candidates'] == 2
    assert result['invalid_or_duplicate_candidates'] == 2
    assert result['coverage'] == 1
    unjudged = score.ranked_retrieval([{'positive': 1}], [['unreported']], observed_only=True)
    assert unjudged['coverage'] == 1 and unjudged['top_k'][1]['recall'] == 0
    assert unjudged['invalid_or_duplicate_candidates'] == 0


@pytest.mark.parametrize('task_id', ['reaction_retrieval', 'enzyme_retrieval', 'active_sites'])
def test_positive_only_sources_default_to_observed_ranking_without_implied_negative_labels(task_id):
    result = evaluate_task(task_id, [{'known-positive': 1}], [['unreported', 'known-positive']])
    assert result['variant'] == 'observed_positive_ranking'
    assert result['metrics']['observed_only'] is True
    assert result['metrics']['top_k'][3]['recall'] == 1
    assert 'top_k.ndcg' not in result['directions']
    assert 'residue_probability' in get_task_metrics('active_sites').variants


def test_measured_oracle_replays_budgets_without_unmeasured_imputation():
    m = score.observed_optimization([{'a': -2., 'b': 3., 'c': 1.}], [['bad', 'a', 'b', 'b']], budgets=(1, 2, 3, 4), unit='native response', hit_threshold=2)
    assert m['budgets'][1]['mean_best_response'] is None
    assert m['budgets'][1]['coverage'] == 0 and m['budgets'][1]['hit_rate'] == 0
    assert m['budgets'][2]['mean_observed_regret'] == 5
    assert m['budgets'][3]['mean_observed_regret'] == 0
    assert m['budgets'][3]['source_hit_recovery'] == 1
    assert m['budgets'][4]['observed_candidate_coverage'] == pytest.approx(2/3)
    assert m['invalid_candidates'] == 1 and m['repeated_candidates'] == 1


def test_measured_oracle_minimization_direction_is_explicit():
    m = score.observed_optimization([{'a': 1., 'b': 4.}], [['b', 'a']], budgets=(1, 2), maximize=False, unit='cost', hit_threshold=2)
    assert m['budgets'][1]['mean_observed_regret'] == 3
    assert m['budgets'][2]['mean_observed_regret'] == 0
    assert m['budgets'][2]['hit_rate'] == 1


def test_scope_recovery_counts_hit_bearing_substrates_once_across_positive_attempts():
    oracle = {'a1': 1, 'a2': 1, 'b1': 1, 'b0': 0, 'c0': 0}
    groups = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'b0': 'B', 'c0': 'C'}
    result = evaluate_task(
        'new_reactivity_discovery', [oracle], [['a1', 'a2', 'unknown', 'a1', 'b0', 'b1']],
        unit='native hit label', hit_threshold=.5, budgets=(1, 2, 3, 4, 5, 6),
        candidate_groups=[groups],
    )
    metrics = result['metrics']
    assert metrics['budgets'][2]['source_hit_recovery'] == pytest.approx(2/3)
    assert metrics['budgets'][2]['scope_hit_group_recovery'] == .5
    assert metrics['budgets'][5]['scope_hit_group_recovery'] == .5
    assert metrics['budgets'][6]['scope_hit_group_recovery'] == 1
    assert metrics['budgets'][6]['scope_hit_groups_recovered_total'] == 2
    assert metrics['invalid_candidates'] == metrics['repeated_candidates'] == 1
    panel = metrics['per_panel'][0]['budgets'][2]
    assert panel['scope_hit_groups_recovered'] == 1
    assert panel['scope_hit_group_denominator'] == 2
    assert result['directions']['budgets.scope_hit_group_recovery'] == 'maximize'


def test_scope_recovery_macro_denominators_keep_no_hit_pools_explicit():
    result = score.observed_optimization(
        [{'a1': 1, 'a2': 1, 'b': 1}, {'c': 1, 'd': 1, 'e': 1}, {'f': 0}],
        [['a1', 'a2', 'unknown'], ['c', 'd', 'e'], ['f']],
        candidate_groups=[{'a1': 'A', 'a2': 'A', 'b': 'B'}, {'c': 'C', 'd': 'D', 'e': 'E'}, {'f': 'F'}],
        hit_threshold=.5, budgets=(3,), unit='native hit label',
    )
    budget = result['budgets'][3]
    assert budget['scope_hit_group_recovery'] == .75
    assert budget['scope_hit_groups_recovered_total'] == 4
    assert budget['scope_hit_group_denominator_total'] == 5
    assert budget['mean_scope_hit_groups_recovered'] == pytest.approx(4/3)
    assert budget['scope_hit_group_recovery_panel_denominator'] == 2
    assert budget['scope_hit_group_panel_denominator'] == 3
    no_hits = result['per_panel'][2]['budgets'][3]
    assert no_hits['scope_hit_groups_recovered'] == no_hits['scope_hit_group_denominator'] == 0
    assert no_hits['scope_hit_group_recovery'] is None


def test_scope_recovery_requires_groups_and_uses_declared_hit_direction():
    plain = score.observed_optimization([{'a': 1}], [['a']], unit='native', hit_threshold=.5)
    assert plain['budgets'][1]['scope_hit_group_recovery'] is None
    assert plain['budgets'][1]['scope_hit_group_denominator_total'] is None
    missing = score.observed_optimization(
        [{'a': 1}], [None], candidate_groups=[{'a': 'A'}],
        unit='native', hit_threshold=.5,
    )
    assert missing['budgets'][1]['scope_hit_group_recovery'] == 0
    minimized = score.observed_optimization(
        [{'a': 1, 'b': 4}], [['a']], candidate_groups=[{'a': 'A', 'b': 'B'}],
        maximize=False, hit_threshold=2, unit='cost',
    )
    assert minimized['budgets'][1]['scope_hit_group_recovery'] == 1
    assert minimized['per_panel'][0]['budgets'][1]['scope_hit_group_denominator'] == 1


@pytest.mark.parametrize('groups', [[], [{}], [None], [{'a': ''}], [{'a': None}], [{'a': 'A', 'extra': 'B'}], {'a': 'A'}])
def test_scope_recovery_rejects_incomplete_or_malformed_parent_mappings(groups):
    with pytest.raises(ValueError, match='candidate_groups'):
        score.observed_optimization([{'a': 1}], [['a']], candidate_groups=groups,
                                    hit_threshold=.5, unit='native')
    with pytest.raises(ValueError, match='hit_threshold'):
        score.observed_optimization([{'a': 1}], [['a']], candidate_groups=[{'a': 'A'}], unit='native')


def test_action_edit_distance_is_not_execution_success():
    m = score.sequence_recovery([['add', 'heat'], [], ['wash']], [['add', 'cool'], None, ['wash']])
    assert m['exact_match'] == pytest.approx(1/3)
    assert m['mean_edit_distance'] == pytest.approx(2/3)
    assert m['mean_normalized_edit_distance'] == .5
    assert m['coverage'] == pytest.approx(2/3)


def test_composition_keeps_missing_mass_and_native_scale():
    m = score.composition([[.7, .3], [.5, .5]], [[.7, .1], None], unit='fraction')
    assert m['coverage'] == .5
    assert m['component_mae'] == pytest.approx(.1)
    assert m['mean_l1_error'] == pytest.approx(.2)
    assert m['mean_total_mass_error'] == pytest.approx(.2)
    with pytest.raises(ValueError):
        score.composition([[1, -.1]], [[1, 0]], unit='fraction')


def test_composition_reports_named_component_errors_and_keeps_finite_negative_predictions():
    result = score.composition(
        [[.8, .1], [.4, .5]], [[.7, -.1], [.5, .6]],
        unit='native fraction', component_ids=['starting_material', 'product'],
    )
    assert result['coverage'] == 1
    assert result['component_mae_by_id']['starting_material'] == pytest.approx(.1)
    assert result['component_mae_by_id']['product'] == pytest.approx(.15)
    assert result['component_error_denominator'] == 2
    assert result['negative_prediction_component_count'] == 1
    assert result['negative_prediction_row_count'] == 1
    assert result['mean_l1_error'] == pytest.approx(.25)


def test_composition_does_not_confuse_recorded_sum_agreement_with_complete_mass_accounting():
    result = score.composition([[.9031, 0, 0]], [[.9031, 0, 0]], unit='native fraction')
    assert result['mean_total_mass_error'] == 0
    assert result['component_mae'] == 0
    assert 'unreported source mass' in result['total_error_policy']
    with pytest.raises(ValueError, match='shared component'):
        score.composition([[.7, .3], [.8]], [[.7, .3], [.8]], unit='fraction')
    with pytest.raises(ValueError, match='component_ids'):
        score.composition([[.7, .3]], [[.7, .3]], unit='fraction', component_ids=['same', 'same'])


def test_clustering_is_label_permutation_invariant_but_coverage_is_separate():
    m = score.clustering([0, 0, 1, 1, 2], [4, 4, 8, 8, None])
    assert m['adjusted_rand_index'] == m['normalized_mutual_information'] == 1
    assert m['coverage'] == .8


def test_geometry_aligns_rigid_motion_and_only_declared_atom_permutations():
    truth = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = truth @ rotation + 7
    m = score.geometry([truth], [moved], unit='angstrom')
    assert m['mean_aligned_rmsd'] == pytest.approx(0, abs=1e-12)
    swapped = moved[[1, 0, 2, 3]]
    assert score.geometry([truth], [swapped], unit='angstrom')['mean_aligned_rmsd'] > .01
    assert score.geometry([truth], [swapped], unit='angstrom', permutations=[[[1, 0, 2, 3]]])['mean_aligned_rmsd'] == pytest.approx(0, abs=1e-12)
    reflected = truth * [-1, 1, 1]
    assert score.geometry([truth], [reflected], unit='angstrom')['mean_aligned_rmsd'] > .01


def test_geometry_rejects_invalid_reference_correspondences_and_reports_missing_output():
    truth = [[0, 0, 0], [1, 0, 0]]
    with pytest.raises(ValueError, match='bijections'):
        score.geometry([truth], [truth], unit='angstrom', permutations=[[[0, 0]]])
    assert score.geometry([truth], [None], unit='angstrom')['mean_aligned_rmsd'] is None


def test_native_time_integration_and_partial_trajectory_denominators():
    m = score.trajectory([[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [1, None, 1]], times=[[0, 1, 3], [0, 1, 3]], unit='concentration', time_unit='s')
    assert m['coverage'] == .5 and m['macro_rmse'] == 1
    assert m['mean_integrated_squared_error'] == 3
    assert m['per_trajectory'][1]['coverage'] == pytest.approx(2/3)
    with pytest.raises(ValueError, match='increasing'):
        score.trajectory([[1, 2]], [[1, 2]], times=[[1, 1]], unit='native', time_unit='s')


def test_control_reports_native_effort_and_sampled_violations_separately():
    m = score.control([[0, 0, 0]], [[1, 1, 1]], times=[[0, 1, 2]], unit='mol/L', time_unit='s', actions=[[0, 2, 0]], action_unit='flow', action_bounds=(0, 1), state_bounds=(0, .5))
    assert m['mean_integrated_squared_error'] == 2
    assert m['control_records'][0]['integrated_squared_action'] == 4
    assert m['control_records'][0]['action_bound_violations'] == 1
    assert m['control_records'][0]['state_bound_violations'] == 3
    assert m['control_effort_unit'] == '(flow)^2 * s'


def test_active_sites_are_protein_macro_with_undefined_one_class_proteins_counted():
    m = score.residue_probabilities([[0, 1], [0, 0]], [[.1, .9], [.1, .2]])
    assert m['protein_macro_average_precision'] == 1
    assert m['average_precision_protein_denominator'] == 1
    assert len(m['per_protein']) == 2


def test_residue_probability_macro_ap_uses_only_complete_qualified_proteins():
    result = score.residue_probabilities(
        [[0, 1], [0, 1, 0], [0, 0]],
        [[.1, .9], [.1, .9, None], [.1, .2]],
    )
    assert result['per_protein'][1]['average_precision'] == 1
    assert result['protein_macro_average_precision'] == 1
    assert result['average_precision_protein_denominator'] == 1
    assert result['coverage'] == pytest.approx(2/3)
    assert result['eligible_residue_count'] == 7
    assert result['valid_residue_count'] == 6
    partial = score.residue_probabilities([[0, 1, 0]], [[.1, .9, None]])
    assert partial['protein_macro_average_precision'] is None
    assert partial['average_precision_protein_denominator'] == 0


def test_energy_and_force_errors_keep_separate_units_and_component_denominators():
    m = score.energy_forces({'energy': [1, 2], 'forces': [[[1, 0, 0]], [[0, 1, 0]]]}, {'energy': [2, 2], 'forces': [[[2, 0, 0]], [[0, 1, 0]]]}, energy_unit='eV', force_unit='eV/angstrom')
    assert m['energy']['mae'] == .5
    assert m['force_components']['mae'] == pytest.approx(1/6)
    assert m['force_components']['error_denominator'] == 6


def test_energy_force_reference_frames_must_align_and_partial_frames_are_explicit():
    with pytest.raises(ValueError, match='matching configurations'):
        score.energy_forces(
            {'energy': [1, 2], 'forces': [[[1, 0, 0]]]},
            {'energy': [1, 2], 'forces': [[[1, 0, 0]]]},
            energy_unit='eV', force_unit='eV/angstrom',
        )
    result = score.energy_forces(
        {'energy': [1, 2], 'forces': [[[1, 0, 0]], [[0, 1, 0]]]},
        {'energy': [1, None], 'forces': [[[float('nan'), 0, 0]], [[0, 1, 0]]]},
        energy_unit='eV', force_unit='eV/angstrom',
    )
    assert result['force_components']['coverage'] == pytest.approx(5/6)
    assert result['force_configuration_coverage']['coverage'] == .5
    assert result['energy']['coverage'] == .5
    assert result['coverage'] == 0


def test_forward_and_mapping_reference_contracts_cannot_be_swapped():
    from synrxn.benchmarks.contracts import OUTCOME_POLICY, ReferenceRecord, TaskContract
    contract = TaskContract('forward.fixture', '1', 'forward', ('reactants',), 'reported_product_multiset', OUTCOME_POLICY, ('test_review',), status='runnable')
    ref = ReferenceRecord(reference_id='r', observation_id='o', revision=1, source_record_id='s', evidence_tier='test_review', resolution='accepted', evidence_locators=('fixture',), task_eligibility={'forward': True}, accepted_outcomes=(('CO',),))
    instances = [dict(instance_id='i', reference_id='r', reference_revision=1, observation_id='o', inputs={'reactants': 'C'})]
    m = evaluate_task('forward_prediction', [ref], [dict(instance_id='i', status='ok', candidates=['CO'])], instances=instances, contract=contract)
    assert m['metrics']['top1_accuracy'] == 1
    with pytest.raises(ValueError, match='does not match'):
        evaluate_task('atom_mapping', [ref], [], instances=instances, contract=contract)


def test_selected_label_probability_defines_confidence_at_nondefault_threshold():
    m = score.binary([0, 1], [.7, .9], threshold=.8)
    # The .7 prediction selects negative with confidence .3; it cannot be
    # ranked ahead of a correct positive selected with confidence .9.
    assert m['selective_risk_coverage'][0]['confidence_threshold'] == .9
    assert m['selective_risk_coverage'][-1]['confidence_threshold'] == pytest.approx(.3)


@pytest.mark.parametrize('task_id', [task for task, p in list_task_metrics().items() if p.variants[p.default_variant].scorer not in {'reference_contract', 'mechanism'}])
def test_each_nonchemical_default_dispatches_with_its_declared_input_schema(task_id):
    profile = get_task_metrics(task_id)
    kind = profile.variants[profile.default_variant].scorer
    unit = {'unit': 'source-native'}
    samples = {
        'regression': ([1., 2.], [1., 2.], unit),
        'classification': (['a', 'b'], ['a', 'b'], {}),
        'binary': ([0, 1], [.1, .9], {}),
        'set_recovery': ([{'a'}, {'b'}], [{'a'}, {'b'}], {}),
        'ranked_retrieval': ([{'a': 1, 'b': 0}], [['a']], {}),
        'observed_optimization': ([{'a': 1, 'b': 2}], [['a', 'b']], unit),
        'sequence_recovery': ([['add', 'heat']], [['add', 'heat']], {}),
        'composition': ([[.5, .5]], [[.5, .5]], unit),
        'clustering': ([0, 0, 1, 1], [4, 4, 8, 8], {}),
        'geometry': ([[[0, 0, 0], [1, 0, 0]]], [[[0, 0, 0], [1, 0, 0]]], unit),
        'trajectory': ([[0, 1]], [[0, 1]], {**unit, 'times': [[0, 1]], 'time_unit': 's'}),
        'control': ([[0, 1]], [[0, 1]], {**unit, 'times': [[0, 1]], 'time_unit': 's', 'actions': [[0, 0]], 'action_unit': 'native-action', 'action_bounds': (0, 1)}),
        'residue_probabilities': ([[0, 1]], [[.1, .9]], {'threshold': .5, 'bins': 2}),
        'energy_forces': ({'energy': [1], 'forces': [[[1, 0, 0]]]}, {'energy': [1], 'forces': [[[1, 0, 0]]]}, {'energy_unit': 'eV', 'force_unit': 'eV/angstrom'}),
    }
    reference, prediction, context = samples[kind]
    result = evaluate_task(task_id, reference, prediction, **context)
    assert result['task_id'] == task_id and result['family_id'] == profile.family_id
    assert result['variant'] == profile.default_variant
    assert result['metrics']


def test_atom_mapping_dispatch_uses_the_existing_symmetry_aware_reference_contract():
    from synrxn.benchmarks.contracts import MAPPING_POLICY, ReferenceRecord, TaskContract
    contract = TaskContract('mapping.fixture', '1', 'mapping', ('reactants', 'products'), 'complete_correspondence', MAPPING_POLICY, ('test_review',), status='runnable')
    ref = ReferenceRecord(reference_id='r', observation_id='o', revision=1, source_record_id='s', evidence_tier='test_review', resolution='accepted', evidence_locators=('fixture',), task_eligibility={'mapping': True}, accepted_correspondences=((0, 1),), review_events=({'reviewer': 'fixture'},))
    instances = [dict(instance_id='i', reference_id='r', reference_revision=1, observation_id='o', inputs={'reactants': 'CC', 'products': 'CC'})]
    m = evaluate_task('atom_mapping', [ref], [dict(instance_id='i', status='ok', candidates=[[1, 0]])], instances=instances, contract=contract)
    assert m['metrics']['top1_accuracy'] == 1


def test_clustering_invalid_predictions_are_coverage_failures_and_invalid_references_raise():
    m = score.clustering([0, 0, 1, 1], ['x', 'x', float('nan'), None])
    assert m['coverage'] == .5
    with pytest.raises(ValueError, match='labels'):
        score.clustering([float('nan')], [0])
    with pytest.raises(ValueError, match='labels'):
        score.classification([float('nan')], [0])


def test_optimizer_direction_and_threshold_cannot_be_ambiguous_or_nonfinite():
    with pytest.raises(ValueError, match='maximize'):
        score.observed_optimization([{'a': 1}], [['a']], unit='native', maximize='minimize')
    with pytest.raises(ValueError, match='hit_threshold'):
        score.observed_optimization([{'a': 1}], [['a']], unit='native', hit_threshold=float('nan'))


def test_public_profile_edits_cannot_change_later_evaluation_or_provenance():
    first = get_task_metrics('yield_conversion')
    first.variants['native_response'].defaults['signed'] = True
    first.variants['native_response'].metrics['mae'] = 'maximize'
    listed = list_task_metrics()
    listed['yield_conversion'].variants.clear()
    result = evaluate_task('yield_conversion', [1], [1], unit='yield percentage points')
    assert result['directions']['mae'] == 'minimize'
    assert result['metrics']['direction_accuracy'] is None
    assert get_task_metrics('yield_conversion').variants['native_response'].defaults == {}
