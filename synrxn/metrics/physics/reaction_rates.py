"""Metrics for reaction rates."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_rates',
    family_id='physics',
    default_variant='native_target',
    variants={
        'native_target': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Score raw rates and log rates separately, with each source’s units, logarithm base, temperature, '
        'phase and measured or computed evidence. Use comparable reaction orders and rate definitions.'
    ),
)
