"""NOLA evaluation with per-video sequential-state resets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from pyclad.strategies.strategy import Strategy
from pyclad.video.data.concepts import VideoFeatureConcept
from pyclad.video.datasets.nola import NolaPreparedTestDataset
from pyclad.video.features.windows import window_scores_to_frame_scores
from pyclad.video.metrics.frame import VideoFrameMetrics, compute_video_frame_metrics
from pyclad.video.metrics.nola import AveragePrecisionDelay, compute_average_precision_delay


@dataclass(frozen=True)
class NolaBenchmarkResult:
    strategy_name: str
    dataset_name: str
    frame_metrics: VideoFrameMetrics
    average_precision_delay: AveragePrecisionDelay
    frame_scores: Dict[str, np.ndarray]
    window_scores: Dict[str, np.ndarray]


class NolaBenchmarkRunner:
    """Train on NOLA stages and evaluate each test video's ODIT state alone."""

    def __init__(self, *, frame_aggregation: str = "max"):
        if frame_aggregation not in {"mean", "max"}:
            raise ValueError("frame_aggregation must be one of: 'mean', 'max'")
        self.frame_aggregation = frame_aggregation

    def run(
        self,
        dataset: NolaPreparedTestDataset,
        strategy: Strategy,
        *,
        train_concepts: Sequence[VideoFeatureConcept] = (),
        learn_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
        predict_kwargs: Optional[Mapping[str, object]] = None,
    ) -> NolaBenchmarkResult:
        learn_kwargs = {} if learn_kwargs is None else learn_kwargs
        for concept in train_concepts:
            strategy.learn(
                concept.strategy_matrix(),
                **dict(learn_kwargs.get(concept.name, {})),
            )

        predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
        test_windows = tuple(dataset.windows("test"))
        labels = dataset.frame_labels("test")
        frame_scores: Dict[str, np.ndarray] = {}
        window_scores: Dict[str, np.ndarray] = {}
        for video_id in sorted(labels):
            windows = tuple(window for window in test_windows if window.video_id == video_id)
            indices = [window.feature_index for window in windows]
            prediction = strategy.predict(
                dataset.feature_store().take(indices),
                **predict_kwargs,
            )
            scores = np.asarray(prediction.anomaly_scores, dtype=np.float64).reshape(-1)
            if len(scores) != len(windows):
                raise ValueError(
                    f"strategy returned {len(scores)} scores for {len(windows)} NOLA rows in {video_id!r}"
                )
            window_scores[video_id] = scores
            frame_scores.update(
                window_scores_to_frame_scores(
                    windows,
                    scores,
                    {video_id: len(labels[video_id])},
                    aggregation=self.frame_aggregation,
                )
            )

        return NolaBenchmarkResult(
            strategy_name=strategy.name(),
            dataset_name=dataset.name(),
            frame_metrics=compute_video_frame_metrics(frame_scores, labels),
            average_precision_delay=compute_average_precision_delay(
                frame_scores,
                dataset.anomaly_intervals,
            ),
            frame_scores=frame_scores,
            window_scores=window_scores,
        )
