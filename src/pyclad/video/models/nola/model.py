"""Strategy-compatible nonparametric NOLA video anomaly model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.models.base import VideoAnomalyModel
from pyclad.video.models.nola.features import NolaFeatureLayout
from pyclad.video.models.nola.scoring import odit_cusum
from pyclad.video.prediction_results import VideoPredictionResults


class NolaVideoModel(VideoAnomalyModel):
    """Modern NOLA adaptation using nominal kNN memories and ODIT scoring.

    The model is nonparametric, like the original NOLA implementation. It can
    therefore use pyCLAD's Naive, Cumulative, MSTE, and ordinary replay
    strategies without modifying those strategies. Differentiable strategies
    such as EWC should use a ``TorchVideoBackbone`` such as COMMAND.
    """

    def __init__(
        self,
        *,
        layout: NolaFeatureLayout = NolaFeatureLayout(),
        strategy_schema: Optional[VideoStrategySchema] = None,
        neighbors: int = 5,
        distance_aggregation: str = "sum",
        spatial_weight: float = 1.0,
        temporal_weight: float = 1.0,
        trajectory_weight: float = 1.0,
        apply_odit: bool = True,
        drift: float = 7.0,
        threshold: float = 0.5,
    ):
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if distance_aggregation not in {"sum", "mean", "max"}:
            raise ValueError("distance_aggregation must be 'sum', 'mean', or 'max'")
        if min(spatial_weight, temporal_weight, trajectory_weight) < 0:
            raise ValueError("NOLA feature-family weights must be non-negative")
        if drift < 0:
            raise ValueError("drift must be non-negative")

        self.layout = layout
        self.strategy_schema = strategy_schema or VideoStrategySchema(feature_dim=layout.feature_dim)
        if self.strategy_schema.feature_dim < layout.feature_dim:
            raise ValueError("strategy_schema does not contain all columns required by layout")
        self.neighbors = int(neighbors)
        self.distance_aggregation = distance_aggregation
        self.spatial_weight = float(spatial_weight)
        self.temporal_weight = float(temporal_weight)
        self.trajectory_weight = float(trajectory_weight)
        self.apply_odit = bool(apply_odit)
        self.drift = float(drift)
        self.threshold = float(threshold)

        self._spatial_scaler: Optional[MinMaxScaler] = None
        self._temporal_scaler: Optional[MinMaxScaler] = None
        self._spatial_memory: Optional[NearestNeighbors] = None
        self._temporal_memory: Optional[NearestNeighbors] = None
        self._effective_neighbors = 0
        self._trajectory_baseline = 0.0
        self._fit_rows = 0
        self._fit_calls = 0

    def fit(self, data: np.ndarray):
        features = self._features(data)
        if len(features) == 0:
            raise ValueError("NOLA cannot fit an empty feature matrix")
        spatial, temporal, trajectory_error = self.layout.split(features)
        self._ensure_finite(spatial, "spatial")
        self._ensure_finite(temporal, "temporal")

        self._spatial_scaler = MinMaxScaler().fit(spatial)
        self._temporal_scaler = MinMaxScaler().fit(temporal)
        self._effective_neighbors = min(self.neighbors, len(features))
        self._spatial_memory = NearestNeighbors(n_neighbors=self._effective_neighbors).fit(
            self._spatial_scaler.transform(spatial)
        )
        self._temporal_memory = NearestNeighbors(n_neighbors=self._effective_neighbors).fit(
            self._temporal_scaler.transform(temporal)
        )

        if trajectory_error is not None:
            self._ensure_finite(trajectory_error, "trajectory_error")
            self._trajectory_baseline = float(np.median(np.maximum(trajectory_error, 0.0)))

        self._fit_rows = len(features)
        self._fit_calls += 1
        return self

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        raw_scores = self.score_samples(data)
        anomaly_scores = odit_cusum(raw_scores, drift=self.drift) if self.apply_odit else raw_scores
        return VideoPredictionResults(
            y_pred=(anomaly_scores >= self.threshold).astype(np.int64),
            anomaly_scores=anomaly_scores,
            window_scores=anomaly_scores.copy(),
        )

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        if (
            self._spatial_scaler is None
            or self._temporal_scaler is None
            or self._spatial_memory is None
            or self._temporal_memory is None
        ):
            raise RuntimeError("NOLA must be fitted before scoring")

        features = self._features(data, allow_features_only=True)
        spatial, temporal, trajectory_error = self.layout.split(features)
        self._ensure_finite(spatial, "spatial")
        self._ensure_finite(temporal, "temporal")

        spatial_distances = self._spatial_memory.kneighbors(
            self._spatial_scaler.transform(spatial),
            n_neighbors=self._effective_neighbors,
            return_distance=True,
        )[0]
        temporal_distances = self._temporal_memory.kneighbors(
            self._temporal_scaler.transform(temporal),
            n_neighbors=self._effective_neighbors,
            return_distance=True,
        )[0]

        statistic = self.spatial_weight * self._aggregate(spatial_distances) + self.temporal_weight * self._aggregate(
            temporal_distances
        )
        if trajectory_error is not None:
            self._ensure_finite(trajectory_error, "trajectory_error")
            residual = np.maximum(trajectory_error - self._trajectory_baseline, 0.0)
            statistic = statistic + self.trajectory_weight * residual
        return np.asarray(statistic, dtype=np.float64)

    def name(self) -> str:
        return "NOLA"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "neighbors": self.neighbors,
            "effective_neighbors": self._effective_neighbors,
            "distance_aggregation": self.distance_aggregation,
            "spatial_weight": self.spatial_weight,
            "temporal_weight": self.temporal_weight,
            "trajectory_weight": self.trajectory_weight,
            "apply_odit": self.apply_odit,
            "drift": self.drift,
            "threshold": self.threshold,
            "fit_rows": self._fit_rows,
            "fit_calls": self._fit_calls,
        }

    def _features(self, data: np.ndarray, *, allow_features_only: bool = False) -> np.ndarray:
        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"NOLA data must be two-dimensional, got {matrix.shape}")
        if matrix.shape[1] == self.strategy_schema.matrix_width:
            return self.strategy_schema.features(matrix)
        if (allow_features_only or not self.strategy_schema.target_names) and (
            matrix.shape[1] == self.strategy_schema.feature_dim
        ):
            return matrix
        raise ValueError(
            f"NOLA data width must be {self.strategy_schema.matrix_width}"
            + (f" or {self.strategy_schema.feature_dim}" if allow_features_only else "")
            + f", got {matrix.shape[1]}"
        )

    def _aggregate(self, distances: np.ndarray) -> np.ndarray:
        if self.distance_aggregation == "sum":
            return distances.sum(axis=1)
        if self.distance_aggregation == "mean":
            return distances.mean(axis=1)
        return distances.max(axis=1)

    @staticmethod
    def _ensure_finite(values: np.ndarray, name: str) -> None:
        if not np.isfinite(values).all():
            raise ValueError(f"NOLA {name} features must contain only finite values")
