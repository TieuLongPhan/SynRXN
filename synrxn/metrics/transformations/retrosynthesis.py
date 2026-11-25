"""Metrics for retrosynthesis."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='retrosynthesis',
    family_id='transformations',
    default_variant='recorded_candidates',
    variants={
        'recorded_candidates': metric_variant(
            'ranked_retrieval',
            primary='top_k.1.hit_rate',
            secondary='top_k.5.hit_rate',
        ),
    },
    qualification=(
        'Use canonicalized precursor-multiset candidate IDs and recorded alternatives. This scores recorded '
        'recovery, not exhaustive acceptable chemistry or multistep route planning.'
    ),
)
