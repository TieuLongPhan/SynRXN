"""Metrics for analytical interpretation."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='analytical_interpretation',
    family_id='analysis',
    default_variant='components',
    variants={
        'components': metric_variant('composition', primary='component_mae_by_id', secondary='mean_l1_error'),
        'native_target': metric_variant('regression', primary='mae'),
    },
    qualification=(
        'Native analytical areas and derived relative quantities remain distinct from independent absolute '
        'concentration references.'
    ),
)
