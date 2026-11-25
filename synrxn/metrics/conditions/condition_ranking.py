"""Metrics for condition ranking."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='condition_ranking',
    family_id='conditions',
    default_variant='measured_panels',
    variants={
        'measured_panels': metric_variant(
            'observed_optimization',
            primary='budgets.1.mean_observed_regret',
            secondary='budgets.3.mean_observed_regret',
            budgets=(1, 3),
        ),
    },
    qualification=(
        'Only comparable measured panels define observed regret; panel completeness, utility and failures '
        'must be qualified upstream.'
    ),
)
