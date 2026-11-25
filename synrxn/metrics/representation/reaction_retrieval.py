"""Metrics for reaction retrieval."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_retrieval',
    family_id='representation',
    default_variant='observed_positive_ranking',
    variants={
        'observed_positive_ranking': metric_variant(
            'ranked_retrieval',
            primary='top_k.5.recall',
            secondary='top_k.5.reciprocal_rank',
            observed_only=True,
        ),
        'judged_candidates': metric_variant('ranked_retrieval', primary='top_k.5.ndcg', secondary='top_k.5.recall'),
    },
    qualification=(
        'Expert-linked query references define observed associations; unjudged gallery chemistry is '
        'unknown. Complete-judgment metrics require an independently qualified population.'
    ),
)
