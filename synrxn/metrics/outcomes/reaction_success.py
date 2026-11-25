"""Metrics for reaction success."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_success',
    family_id='outcomes',
    default_variant='positive_probability',
    variants={
        'positive_probability': metric_variant('binary', primary='average_precision', secondary='brier_score'),
    },
    qualification=(
        'Declare the measured success definition/threshold and 1=success polarity. Missing observations and '
        'synthetic distractors are not measured failures.'
    ),
)
