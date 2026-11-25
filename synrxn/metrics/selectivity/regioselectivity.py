"""Metrics for regioselectivity."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='regioselectivity',
    family_id='selectivity',
    default_variant='site_sets',
    variants={
        'site_sets': metric_variant('set_recovery', primary='exact_match', secondary='macro_f1'),
        'recorded_products': metric_variant(
            'ranked_retrieval',
            primary='top_k.1.hit_rate',
            secondary='top_k.5.hit_rate',
        ),
    },
    qualification=(
        'Supply symmetry-normalized native site IDs or canonical product IDs; product-only annotations do '
        'not establish site labels.'
    ),
)
