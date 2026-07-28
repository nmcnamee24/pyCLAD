"""A lightweight adapter for array-based video anomaly scorers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.models.base import VideoAnomalyModel
from pyclad.video.prediction_results import VideoPredictionResults


class CallableVideoAnomalyModel(VideoAnomalyModel):
    """Wrap fit and scoring callables in pyCLAD's existing Model interface."""

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        *,
        fit_fn: Optional[Callable[[np.ndarray], Any]] = None,
        model_name: str = "CallableVideoAnomalyModel",
        threshold: float = 0.5,
        higher_is_anomalous: bool = True,
        strategy_schema: Optional[VideoStrategySchema] = None,
    ):
        self._score_fn = score_fn
        self._fit_fn = fit_fn
        self._model_name = model_name
        self._threshold = float(threshold)
        self._higher_is_anomalous = higher_is_anomalous
        self._strategy_schema = strategy_schema
        self._fit_calls = 0

    def fit(self, data: np.ndarray):
        matrix = _feature_matrix(data, self._strategy_schema)
        if self._fit_fn is not None:
            self._fit_fn(matrix)
        self._fit_calls += 1
        return self

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        matrix = _feature_matrix(data, self._strategy_schema)
        raw_scores = np.asarray(self._score_fn(matrix), dtype=np.float64).reshape(-1)
        if len(raw_scores) != len(matrix):
            raise ValueError(f"score_fn returned {len(raw_scores)} scores for {len(matrix)} video windows")
        scores = raw_scores if self._higher_is_anomalous else -raw_scores
        return VideoPredictionResults(
            y_pred=(scores >= self._threshold).astype(np.int64),
            anomaly_scores=scores,
            window_scores=scores.copy(),
        )

    def name(self) -> str:
        return self._model_name

    def additional_info(self) -> Dict[str, Any]:
        return {
            "threshold": self._threshold,
            "higher_is_anomalous": self._higher_is_anomalous,
            "fit_calls": self._fit_calls,
        }


class CallableWeaklySupervisedVideoModel(CallableVideoAnomalyModel):
    """Callable adapter whose fit function receives named numeric targets."""

    def __init__(
        self,
        fit_fn: Callable[[np.ndarray, Mapping[str, np.ndarray]], Any],
        score_fn: Callable[[np.ndarray], np.ndarray],
        *,
        strategy_schema: VideoStrategySchema,
        model_name: str = "CallableWeaklySupervisedVideoModel",
        threshold: float = 0.5,
        higher_is_anomalous: bool = True,
    ):
        if not strategy_schema.target_names:
            raise ValueError("weakly supervised models require at least one target column")
        super().__init__(
            score_fn=score_fn,
            fit_fn=None,
            model_name=model_name,
            threshold=threshold,
            higher_is_anomalous=higher_is_anomalous,
            strategy_schema=strategy_schema,
        )
        self._supervised_fit_fn = fit_fn

    def fit(self, data: np.ndarray):
        matrix = self._strategy_schema._matrix(data)
        self._supervised_fit_fn(
            self._strategy_schema.features(matrix),
            self._strategy_schema.targets(matrix),
        )
        self._fit_calls += 1
        return self


def _feature_matrix(
    data: np.ndarray,
    schema: Optional[VideoStrategySchema] = None,
) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"video model data must be 2D, got shape {matrix.shape}")
    return matrix if schema is None else schema.features(matrix)
