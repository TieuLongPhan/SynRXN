"""Metrics for reaction balancing."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_balancing',
    family_id='representation',
    default_variant='normalized_species',
    variants={
        'normalized_species': metric_variant('set_recovery', primary='exact_match', secondary='macro_f1'),
        'action_sequence': metric_variant(
            'sequence_recovery',
            primary='exact_match',
            secondary='mean_normalized_edit_distance',
        ),
    },
    qualification=(
        'Use explicit stoichiometry-bearing species tokens. Set agreement does not establish atom/charge '
        'conservation; run a qualified chemical accounting validator separately.'
    ),
)
