"""Metrics for reaction record curation."""

from .._profiles import TaskMetricProfile, metric_variant


PROFILE = TaskMetricProfile(
    task_id='reaction_record_curation',
    family_id='data',
    default_variant='corrections',
    variants={
        'corrections': metric_variant('set_recovery', primary='micro_f1', secondary='false_edit_rate'),
    },
    qualification=(
        'Score declared edit tuples. False-edit rate uses explicitly unchanged reference records and its '
        'valid-output denominator; naturally observed error adjudication remains source-specific.'
    ),
)
