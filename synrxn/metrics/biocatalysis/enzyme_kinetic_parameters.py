"""Metrics for enzyme kinetic parameters."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='enzyme_kinetic_parameters',
    family_id='biocatalysis',
    default_variant='native_parameter',
    variants={
        'native_parameter': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Score kcat, Km and Ki separately by parameter, reaction and assay. Supply native units and encode '
        'the logarithm base in the unit when the source target is a log parameter.'
    ),
)
