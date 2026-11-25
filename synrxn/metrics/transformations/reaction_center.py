"""Metrics for reaction center."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_center',
    family_id='transformations',
    default_variant='bond_edits',
    variants={
        'bond_edits': metric_variant('set_recovery', primary='micro_f1', secondary='exact_match'),
    },
    qualification=(
        'Native atom indices, bond conventions and label derivation must be qualified before set scoring.'
    ),
)
