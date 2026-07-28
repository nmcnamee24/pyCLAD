"""Schemas for carrying numeric video targets through array-only strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VideoStrategySchema:
    """Describe feature and target columns in a strategy-facing matrix.

    Core pyCLAD sees only an ordinary matrix. Video model adapters use this
    schema to prevent reserved weak-supervision columns from becoming features.
    """

    feature_dim: int
    target_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must be unique")
        if any(not name for name in self.target_names):
            raise ValueError("target_names must be non-empty strings")

    @property
    def matrix_width(self) -> int:
        return self.feature_dim + len(self.target_names)

    def pack(
        self,
        features: np.ndarray,
        targets: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (rows, {self.feature_dim}), got {features.shape}")
        if not self.target_names:
            if targets:
                raise ValueError("targets were provided but the schema has no target columns")
            return features

        targets = {} if targets is None else targets
        unknown = set(targets) - set(self.target_names)
        if unknown:
            raise ValueError(f"targets contain names absent from the schema: {sorted(unknown)}")

        columns = []
        for name in self.target_names:
            values = np.asarray(
                targets.get(name, np.full(len(features), np.nan)),
                dtype=np.float32,
            ).reshape(-1)
            if len(values) != len(features):
                raise ValueError(f"target {name!r} has {len(values)} rows; expected {len(features)}")
            columns.append(values[:, None])
        return np.concatenate([features, *columns], axis=1)

    def features(self, matrix: np.ndarray) -> np.ndarray:
        matrix = self._matrix(matrix)
        return matrix[:, : self.feature_dim]

    def targets(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        matrix = self._matrix(matrix)
        return {name: matrix[:, self.feature_dim + index] for index, name in enumerate(self.target_names)}

    def _matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.matrix_width:
            raise ValueError(f"strategy matrix must have shape (rows, {self.matrix_width}), " f"got {matrix.shape}")
        return matrix
