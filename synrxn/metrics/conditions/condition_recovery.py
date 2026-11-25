"""Metrics for condition recovery."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='condition_recovery',
    family_id='conditions',
    default_variant='role_aware_sets',
    variants={
        'role_aware_sets': metric_variant('set_recovery', primary='micro_f1', secondary='macro_f1'),
        'numeric_target': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Include role in each normalized token and score per-role subsets separately; unknown roles are not '
        'absent agents and a recorded recipe is not uniquely correct.'
    ),
)
