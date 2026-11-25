"""Metrics for sequential optimization."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='sequential_optimization',
    family_id='conditions',
    default_variant='finite_oracle',
    variants={
        'finite_oracle': metric_variant(
            'observed_optimization',
            primary='budgets.5.mean_observed_regret',
            secondary='budgets.5.hit_rate',
        ),
    },
    qualification=(
        'The candidate pool and responses are fixed by the measured source. Declare initialization, oracle '
        'access and evaluation budgets. Hit rate requires a source-defined hit threshold.'
    ),
)
