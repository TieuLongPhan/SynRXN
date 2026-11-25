"""Metrics for reaction network modeling."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_network_modeling',
    family_id='mechanisms',
    default_variant='directed_edges',
    variants={
        'directed_edges': metric_variant('set_recovery', primary='micro_f1', secondary='exact_match'),
        'species_trajectory': metric_variant('trajectory', primary='macro_rmse'),
    },
    qualification=(
        'Edge tokens must encode direction and stoichiometry. Edge or trajectory agreement alone does not '
        'prove dynamical equivalence or kinetic identifiability.'
    ),
)
