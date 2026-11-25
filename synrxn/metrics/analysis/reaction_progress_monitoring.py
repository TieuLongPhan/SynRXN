"""Metrics for reaction progress monitoring."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_progress_monitoring',
    family_id='analysis',
    default_variant='native_trajectory',
    variants={
        'native_trajectory': metric_variant('trajectory', primary='macro_mae'),
        'next_step': metric_variant('regression', primary='mae', secondary='rmse'),
    },
    qualification=(
        'Score causal held-out native trajectories. Persistence baselines must be supplied/evaluated '
        'separately; end-point timing and anomaly detection need qualified labels.'
    ),
)
