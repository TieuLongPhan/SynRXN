"""Metrics for yield conversion."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='yield_conversion',
    family_id='outcomes',
    default_variant='native_response',
    variants={
        'native_response': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Yield, conversion and assay response are distinct native endpoints with separate units, '
        'measurement methods and source-specific scores.'
    ),
)
