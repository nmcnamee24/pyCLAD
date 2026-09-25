"""Adapt frame curves to pyCLAD's existing concept-metric callback."""

import numpy as np

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.video.metrics.frame_score_utils import (
    flatten_video_curves,
    window_scores_to_frame_scores,
)


class VideoFrameMetricCallback(ConceptMetricCallback):
    """Evaluate any BaseMetric on frames, reusing core matrix reporting."""

    def __init__(self, base_metric, summarized_metrics=()):
        super().__init__(base_metric, summarized_metrics)

    def after_evaluation(self, evaluated_concept, window_scores, windows, *args, **kwargs):
        """Expand temporal scores against the evaluated concept's frame labels."""
        labels = evaluated_concept.frame_labels
        scores = window_scores_to_frame_scores(
            windows, window_scores, {key: len(value) for key, value in labels.items()}
        )
        super().after_evaluation(
            evaluated_concept=evaluated_concept,
            y_true=flatten_video_curves(labels),
            y_pred=np.empty(0),
            anomaly_scores=flatten_video_curves(scores),
        )
