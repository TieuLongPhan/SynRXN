"""Metrics for reaction component design."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_component_design',
    family_id='design',
    default_variant='recorded_candidates',
    variants={
        'recorded_candidates': metric_variant(
            'ranked_retrieval',
            primary='top_k.5.hit_rate',
            secondary='top_k.5.reciprocal_rank',
        ),
        'measured_oracle': metric_variant('observed_optimization', primary='budgets.5.mean_observed_regret'),
    },
    qualification=(
        'Exact observed identities support reference recovery or measured lookup utility. Off-library '
        'performance, sequence/structure constraints and training-relative novelty need explicit '
        'independent validators.'
    ),
)
