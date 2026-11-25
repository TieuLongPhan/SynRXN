"""Metrics for enzyme retrieval."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='enzyme_retrieval',
    family_id='biocatalysis',
    default_variant='observed_positive_ranking',
    variants={
        'observed_positive_ranking': metric_variant(
            'ranked_retrieval',
            primary='top_k.5.reciprocal_rank',
            secondary='top_k.5.recall',
            observed_only=True,
        ),
        'judged_candidates': metric_variant('ranked_retrieval', primary='top_k.5.ndcg', secondary='top_k.5.recall'),
    },
    qualification=(
        'Native positive associations do not establish inactivity of other candidates. Candidate universe, '
        'query direction and homology controls remain source-specific; complete-judgment metrics require '
        'qualified labels.'
    ),
)
