"""Metrics for process control."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='process_control',
    family_id='process',
    default_variant='closed_loop_trace',
    variants={
        'closed_loop_trace': metric_variant('control', primary='mean_integrated_squared_error'),
    },
    qualification=(
        'Supply actual closed-loop plant/simulator traces. Each controlled/constrained state channel is '
        'scored separately; sampled bounds do not prove continuous-time safety.'
    ),
)
