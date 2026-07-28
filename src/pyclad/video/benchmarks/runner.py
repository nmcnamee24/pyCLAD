"""Evaluate regular pyCLAD strategies on video-window feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.strategy import Strategy
from pyclad.video.data.base import VideoDataset
from pyclad.video.features.windows import window_scores_to_frame_scores
from pyclad.video.metrics.frame import VideoFrameMetrics, compute_video_frame_metrics


@dataclass(frozen=True)
class BenchmarkResult:
    strategy_name: str
    dataset_name: str
    metrics: VideoFrameMetrics
    prediction: PredictionResults
    frame_scores: Dict[str, np.ndarray]
    window_scores: np.ndarray


class VideoBenchmarkRunner:
    """Train and evaluate an existing strategy without adapting its class."""

    def __init__(self, frame_aggregation: str = "mean"):
        if frame_aggregation not in {"mean", "max"}:
            raise ValueError("frame_aggregation must be one of: 'mean', 'max'")
        self.frame_aggregation = frame_aggregation

    def run(
        self,
        dataset: VideoDataset,
        strategy: Strategy,
        *,
        train_splits: Sequence[str] = (),
        test_split: str = "test",
        learn_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        predict_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> BenchmarkResult:
        learn_kwargs = {} if learn_kwargs is None else learn_kwargs
        for split in train_splits:
            strategy.learn(
                dataset.feature_matrix(split=split),
                **dict(learn_kwargs.get(split, {})),
            )

        prediction = strategy.predict(
            dataset.feature_matrix(split=test_split),
            **({} if predict_kwargs is None else dict(predict_kwargs)),
        )
        window_scores = np.asarray(prediction.anomaly_scores, dtype=np.float64).reshape(-1)
        windows = tuple(dataset.windows(split=test_split))
        if len(window_scores) != len(windows):
            raise ValueError(f"strategy returned {len(window_scores)} scores for {len(windows)} video windows")

        frame_labels = dataset.frame_labels(split=test_split)
        frame_scores = window_scores_to_frame_scores(
            windows=windows,
            window_scores=window_scores,
            frame_counts={video_id: len(labels) for video_id, labels in frame_labels.items()},
            aggregation=self.frame_aggregation,
        )
        return BenchmarkResult(
            strategy_name=strategy.name(),
            dataset_name=dataset.name(),
            metrics=compute_video_frame_metrics(frame_scores, frame_labels),
            prediction=prediction,
            frame_scores=frame_scores,
            window_scores=window_scores,
        )
