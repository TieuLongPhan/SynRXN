"""Metrics for process impact assessment."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='process_impact_assessment',
    family_id='process',
    default_variant='native_endpoint',
    variants={
        'native_endpoint': metric_variant('regression', primary='mae'),
    },
    qualification=(
        'Current source endpoint is a narrow native quantity; it does not establish complete LCA, cost, '
        'energy or operational hazard coverage.'
    ),
)
