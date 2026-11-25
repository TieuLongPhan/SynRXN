"""Metrics for ts geometry."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='ts_geometry',
    family_id='mechanisms',
    default_variant='aligned_geometry',
    variants={
        'aligned_geometry': metric_variant('geometry', primary='mean_aligned_rmsd'),
    },
    qualification=(
        'Only explicitly supplied symmetry correspondences are accepted. Independent saddle validation, '
        'charge/spin completeness and electronic-structure checks are not provided by RMSD.'
    ),
)
