"""Shared scorer metadata and scientific metric profile types."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MetricVariant:
    """Scorer, reporting choices and scientific requirements for one variant."""

    scorer: str
    metrics: Mapping[str, str]
    requirements: tuple[str, ...]
    unit_policy: str
    interpretation: str
    defaults: Mapping[str, Any]
    primary_metric: str | None = None
    secondary_metric: str | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TaskMetricProfile:
    """Available metric variants for one registered scientific task."""

    task_id: str
    family_id: str
    default_variant: str
    variants: Mapping[str, MetricVariant]
    qualification: str

    def as_dict(self):
        return asdict(self)


# Directions refer to the named measured quantity. Coverage is separate from
# accuracy and is not a substitute for a full-population performance score.
_METRICS = {
    'regression': {
        'mae': 'minimize',
        'rmse': 'minimize',
        'r2': 'maximize',
        'spearman': 'maximize',
        'direction_accuracy': 'maximize',
    },
    'classification': {
        'accuracy': 'maximize',
        'macro_f1': 'maximize',
        'balanced_accuracy': 'maximize',
        'mcc': 'maximize',
    },
    'binary': {
        'accuracy': 'maximize',
        'macro_f1': 'maximize',
        'balanced_accuracy': 'maximize',
        'mcc': 'maximize',
        'auroc': 'maximize',
        'average_precision': 'maximize',
        'brier_score': 'minimize',
        'log_loss': 'minimize',
        'expected_calibration_error': 'minimize',
        'selective_risk_coverage': 'curve',
    },
    'set_recovery': {
        'observed_recall': 'maximize',
        'query_macro_recall': 'maximize',
        'micro_f1': 'maximize',
        'macro_f1': 'maximize',
        'exact_match': 'maximize',
        'false_edit_rate': 'minimize',
    },
    'ranked_retrieval': {
        'top_k.recall': 'maximize',
        'top_k.ndcg': 'maximize',
        'top_k.reciprocal_rank': 'maximize',
        'top_k.hit_rate': 'maximize',
    },
    'observed_optimization': {
        'budgets.mean_best_response': 'declared_by_maximize',
        'budgets.mean_observed_regret': 'minimize',
        'budgets.hit_rate': 'maximize',
        'budgets.observed_candidate_coverage': 'coverage',
        'budgets.source_hit_recovery': 'maximize',
        'budgets.mean_scope_hit_groups_recovered': 'maximize',
        'budgets.scope_hit_group_recovery': 'maximize',
        'budgets.scope_hit_groups_recovered_total': 'maximize',
        'budgets.scope_hit_group_denominator_total': 'diagnostic',
        'budgets.scope_hit_group_recovery_panel_denominator': 'diagnostic',
        'budgets.scope_hit_group_panel_denominator': 'diagnostic',
        'budgets.mean_observed_failures': 'diagnostic',
    },
    'sequence_recovery': {
        'exact_match': 'maximize',
        'mean_edit_distance': 'minimize',
        'mean_normalized_edit_distance': 'minimize',
    },
    'composition': {
        'component_mae': 'minimize',
        'component_mae_by_id': 'minimize',
        'mean_l1_error': 'minimize',
        'mean_total_mass_error': 'minimize',
        'negative_prediction_component_count': 'diagnostic',
    },
    'clustering': {'adjusted_rand_index': 'maximize', 'normalized_mutual_information': 'maximize'},
    'geometry': {'mean_aligned_rmsd': 'minimize'},
    'trajectory': {'macro_mae': 'minimize', 'macro_rmse': 'minimize', 'mean_integrated_squared_error': 'minimize'},
    'control': {
        'macro_mae': 'minimize',
        'macro_rmse': 'minimize',
        'mean_integrated_squared_error': 'minimize',
        'control_records.integrated_squared_action': 'minimize',
        'control_records.action_bound_violations': 'minimize',
        'control_records.state_bound_violations': 'minimize',
    },
    'residue_probabilities': {'protein_macro_average_precision': 'maximize'},
    'energy_forces': {
        'energy.mae': 'minimize',
        'energy.rmse': 'minimize',
        'force_components.mae': 'minimize',
        'force_components.rmse': 'minimize',
    },
    'reference_contract': {
        'top1_accuracy': 'maximize',
        'topk_accuracy': 'maximize',
        'valid_output_coverage': 'coverage',
    },
    'mechanism': {
        'top_k.event_recovery': 'maximize',
        'top_k.trajectory_recovery': 'maximize',
        'top_k.strict_validity': 'maximize',
        'top_k.net_equivalence': 'diagnostic',
    },
}
_REQUIREMENTS = {
    'regression': (
        'finite scalar native references',
        'predictions aligned to every eligible reference',
        'explicit native unit; separate sources/targets before scoring',
    ),
    'classification': (
        'native reference labels',
        'explicit label universe recommended',
        'None or unknown prediction labels count as failures',
    ),
    'binary': (
        'native binary reference labels encoded 0/1 with 1 the declared positive class',
        'positive-class probabilities in [0,1]',
        'threshold fixed without test-label tuning',
    ),
    'set_recovery': (
        'already normalized set items, e.g. entity spans, bond edits, residue IDs',
        'complete annotations for precision/F1; observed_only=True otherwise',
    ),
    'ranked_retrieval': (
        'one candidate-ID to nonnegative graded relevance mapping per query',
        'submitted ranked candidate IDs; unknown/duplicate IDs consume rank',
    ),
    'observed_optimization': (
        'one finite measured candidate-ID to response oracle per panel/run',
        'ordered submitted candidate IDs',
        'native response unit and declared maximize/minimize direction',
        'optional complete candidate-to-audited-substrate-parent mappings plus explicit source hit threshold for '
        'group-scope recovery',
    ),
    'sequence_recovery': (
        'reference action/token lists',
        'prediction lists in the same action vocabulary',
    ),
    'composition': (
        'nonnegative reference vectors aligned to one declared component order; finite raw model outputs',
        'native unit; missing source components/mass are not inferred from component-sum disagreement',
    ),
    'clustering': (
        'held-out external class labels',
        'one predicted cluster label per eligible observation',
    ),
    'geometry': (
        'reference and predicted N by 3 arrays in the same units',
        'fixed atom correspondence or explicit accepted per-case permutations',
    ),
    'trajectory': (
        'scalar reference and predicted series per complete trajectory',
        'strictly increasing native time grid for each trajectory',
        'native target unit and native time_unit',
    ),
    'control': (
        'desired and controlled scalar state trajectories',
        'native time grids',
        'actions sampled on the same grids, native action_unit/time_unit and explicit action bounds',
        'optional explicit state bounds; separate constrained temperature channel when needed',
    ),
    'residue_probabilities': (
        'complete eligible binary residue labels grouped per protein',
        'positive-class residue probabilities',
        'explicit eligibility masks applied upstream; unannotated is not negative',
    ),
    'energy_forces': (
        'energy and Cartesian forces mappings for one declared trajectory/campaign',
        'native energy_unit and force_unit; matching configurations/atom order',
    ),
    'reference_contract': (
        'qualified ReferenceRecord objects',
        'instances and a validated TaskContract',
        'ranked prediction records with explicit missing/failure statuses',
    ),
    'mechanism': (
        'accepted strictly replay-valid mechanism annotations per case',
        'ranked supplied mechanism predictions per case',
        'fixed SynKit backend with explicit-hydrogen comparator capability',
    ),
}
_INTERPRETATIONS = {
    'regression': (
        'Errors are conditional on finite output coverage; units and targets are never pooled or transformed '
        'implicitly. Direction accuracy is only meaningful for a qualified signed target.'
    ),
    'classification': (
        'All eligible rows contribute to accuracy/F1. Source taxonomies are not merged; no hierarchical label '
        'distances are inferred.'
    ),
    'binary': (
        'Positive class is explicitly 1. Average precision is the step-weighted precision-recall summary, not '
        'trapezoidal PR-AUC. Calibration is conditional on valid probabilities; risk curves preserve confidence '
        'ties.'
    ),
    'set_recovery': (
        'Agreement with supplied normalized annotation sets, not inferred chemical equivalence. Observed-only '
        'mode does not penalize unreported labels.'
    ),
    'ranked_retrieval': (
        'Reference-relative recovery in a declared judged candidate universe, not open-world chemical validity or '
        'novelty.'
    ),
    'observed_optimization': (
        'Retrospective finite measured-oracle replay; no prospective laboratory success or unseen candidate '
        'response is inferred. Source-hit recovery counts attempts; optional audited parent-group recovery counts '
        'distinct hit-bearing substrate parents in each measured pool.'
    ),
    'sequence_recovery': 'Action-sequence reference agreement; not laboratory execution validation.',
    'composition': (
        'Native per-component errors without renormalization or clipping. Component-sum disagreement does not '
        'measure unreported source mass; negative predictions remain numeric errors and separate bound '
        'diagnostics.'
    ),
    'clustering': 'External label agreement; stability across perturbations/seeds requires separately repeated runs.',
    'geometry': (
        'Proper-rotation aligned RMSD with only supplied atom permutations. No saddle, Hessian or connectivity '
        'verification.'
    ),
    'trajectory': (
        'Complete-trajectory macro errors and native-time integrated squared error. No temporal resampling or '
        'interpolation.'
    ),
    'control': (
        'Scores supplied closed-loop traces; does not simulate a controller. Action effort and sampled bound '
        'violations are diagnostics under declared units and constraints.'
    ),
    'residue_probabilities': (
        'Protein-macro average precision requires complete outputs and qualified labels for both classes per '
        'scored protein. Partial per-protein binary metrics remain diagnostic; protein and residue coverage are '
        'separate.'
    ),
    'energy_forces': (
        'Separate errors for native energies and Cartesian force components; no energy-force gradient consistency '
        'or conservative dynamics certification.'
    ),
    'reference_contract': (
        'Existing bounded, failure-inclusive chemical reference scorer; symmetry and candidate normalization '
        'follow the supplied validated contract.'
    ),
    'mechanism': (
        'Strict replay plus supplied event and trajectory reference equivalence; representation consistency is '
        'not experimental confirmation of a physical mechanism.'
    ),
}


def direction_key(name: str) -> str:
    """Use the same direction for every cutoff of a metric."""
    return ".".join(part for part in name.split(".") if not part.isdecimal())


def metric_variant(scorer, *, primary, secondary=None, **defaults):
    """Declare a task-specific reporting choice using a shared scorer."""
    metrics = {**_METRICS[scorer], "coverage": "coverage"}
    requirements, interpretation = _REQUIREMENTS[scorer], _INTERPRETATIONS[scorer]
    if scorer == "ranked_retrieval" and defaults.get("observed_only") is True:
        metrics.pop("top_k.ndcg")
        requirements = (
            "one candidate-ID to supplied-positive relevance mapping per query; zero/absent judgments remain unknown",
            "ranked candidate IDs with an independently declared gallery when available; no inferred negative activity "
            "labels",
        )
        interpretation = (
            "Reference-relative observed-positive recall, hit rate and reciprocal rank. Unjudged candidates are "
            "unknown, consume rank and are not false positives. nDCG is suppressed; coverage is syntactic submission "
            "coverage without a complete gallery."
        )
    if not primary:
        raise ValueError("A primary reporting metric is required")
    for name in (primary, secondary):
        if name is not None and metrics.get(direction_key(name)) not in {"minimize", "maximize"}:
            raise ValueError(f"Unknown reporting metric {name!r} for {scorer}")
    if primary == secondary:
        raise ValueError("Primary and secondary metrics must be distinct")
    unit_policy = "Dimensionless reference-relative scores; label and equivalence policies remain source-specific."
    if scorer in {"regression", "composition", "observed_optimization", "geometry", "trajectory",
                  "control", "energy_forces"}:
        unit_policy = "Explicit native units required; no implicit conversion."
    return MetricVariant(
        scorer=scorer, metrics=metrics, requirements=requirements, unit_policy=unit_policy,
        interpretation=interpretation, defaults=defaults, primary_metric=primary, secondary_metric=secondary,
    )
