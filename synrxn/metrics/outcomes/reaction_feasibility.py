"""Metrics for reaction feasibility."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_feasibility',
    family_id='outcomes',
    default_variant='positive_probability',
    variants={
        'positive_probability': metric_variant('binary', primary='average_precision', secondary='mcc'),
    },
    qualification=(
        'Declare the native plausible/feasible positive class; expert labels, generated supervision and '
        'measured outcomes are separate populations.'
    ),
)
