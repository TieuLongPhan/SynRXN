"""Metrics for reaction energetics."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_energetics',
    family_id='physics',
    default_variant='native_target',
    variants={
        'native_target': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Score signed reaction energy, enthalpy and free energy as separate quantities in native units. '
        'Retain reaction direction, phase, solvent, temperature, standard state and method.'
    ),
)
