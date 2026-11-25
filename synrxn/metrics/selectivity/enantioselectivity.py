"""Metrics for enantioselectivity."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='enantioselectivity',
    family_id='selectivity',
    default_variant='native_response',
    variants={
        'native_response': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Signed ee, absolute ee and energy differences are distinct; direction accuracy is interpretable '
        'only with a resolved stereochemical sign convention.'
    ),
)
