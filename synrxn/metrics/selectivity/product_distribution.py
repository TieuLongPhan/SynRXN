"""Metrics for product distribution."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='product_distribution',
    family_id='selectivity',
    default_variant='components',
    variants={
        'components': metric_variant('composition', primary='component_mae_by_id', secondary='mean_l1_error'),
    },
    qualification=(
        'Preserve component identity, native scale and missing mass; do not silently normalize incomplete '
        'mixtures.'
    ),
)
