"""Metrics for activation barrier."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='activation_barrier',
    family_id='physics',
    default_variant='native_target',
    variants={
        'native_target': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Score each barrier definition and direction in its native energy unit. Keep electronic and free- '
        'energy barriers, calculation methods, phases, solvents, temperatures and standard states separate.'
    ),
)
