"""Metrics for reactive potential."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reactive_potential',
    family_id='physics',
    default_variant='energy_forces',
    variants={
        'energy_forces': metric_variant('energy_forces', primary='force_components.mae', secondary='energy.mae'),
    },
    qualification=(
        'Score one declared trajectory at a time and report energy and force units separately; do not '
        'equate configuration count with reaction count.'
    ),
)
