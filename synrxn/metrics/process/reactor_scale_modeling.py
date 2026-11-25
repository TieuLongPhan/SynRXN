"""Metrics for reactor scale modeling."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reactor_scale_modeling',
    family_id='process',
    default_variant='native_endpoint',
    variants={
        'native_endpoint': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Score source endpoints separately and split by declared reactor geometry/scale; distinguish '
        'measured records from interpolated descendants.'
    ),
)
