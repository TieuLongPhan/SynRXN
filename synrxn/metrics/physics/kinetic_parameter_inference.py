"""Metrics for kinetic parameter inference."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='kinetic_parameter_inference',
    family_id='physics',
    default_variant='parameters',
    variants={
        'parameters': metric_variant('regression', primary='mae', secondary='rmse'),
        'held_out_trajectory': metric_variant('trajectory', primary='macro_rmse'),
    },
    qualification=(
        'Native parameters and held-out simulated trajectories are separate targets. Identifiability, '
        'design rank and dynamical equivalence need model-specific analysis, not parameter errors alone.'
    ),
)
