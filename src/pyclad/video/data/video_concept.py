"""A regular pyCLAD concept with temporal video metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from pyclad.data.concept import Concept
from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.data.sample import VideoWindow


@dataclass
class VideoConcept(Concept):
    """A :class:`Concept` whose rows are aligned with temporal windows.

    ``data`` is the same two-dimensional matrix consumed by existing pyCLAD
    strategies. ``windows`` retains video and frame metadata outside that
    matrix. A schema is needed only when weak labels or bag identifiers are
    carried as reserved numeric columns for a model such as COMMAND.
    """

    windows: Tuple[VideoWindow, ...] = ()
    strategy_schema: Optional[VideoStrategySchema] = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"VideoConcept.data must be two-dimensional, got {matrix.shape}")

        windows = tuple(self.windows)
        if len(matrix) != len(windows):
            raise ValueError(
                "VideoConcept.windows must align with data on the row dimension: " f"{len(windows)} != {len(matrix)}"
            )

        labels = self.labels
        if labels is None and windows and all(window.label is not None for window in windows):
            labels = np.asarray([window.label for window in windows], dtype=np.int64)
        elif labels is not None:
            labels = np.asarray(labels, dtype=np.int64).reshape(-1)
            if len(labels) != len(matrix):
                raise ValueError(
                    "VideoConcept.labels must align with data on the row dimension: " f"{len(labels)} != {len(matrix)}"
                )

        schema = self.strategy_schema or VideoStrategySchema(feature_dim=matrix.shape[1])
        if matrix.shape[1] != schema.matrix_width:
            raise ValueError(
                f"VideoConcept.data must have {schema.matrix_width} columns for its schema, " f"got {matrix.shape[1]}"
            )

        self.data = matrix
        self.labels = labels
        self.windows = windows
        self.strategy_schema = schema

    @classmethod
    def from_features(
        cls,
        name: str,
        features: np.ndarray,
        windows: Sequence[VideoWindow],
        *,
        labels: Optional[np.ndarray] = None,
        strategy_schema: Optional[VideoStrategySchema] = None,
        strategy_targets: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> "VideoConcept":
        """Build a concept from feature rows and optional reserved targets."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"VideoConcept features must be two-dimensional, got {values.shape}")
        schema = strategy_schema or VideoStrategySchema(feature_dim=values.shape[1])
        return cls(
            name=name,
            data=schema.pack(values, strategy_targets),
            labels=labels,
            windows=tuple(windows),
            strategy_schema=schema,
        )

    @property
    def features(self) -> np.ndarray:
        """Return model features without reserved strategy columns."""

        return self.strategy_schema.features(self.data)

    @property
    def strategy_targets(self) -> Mapping[str, np.ndarray]:
        """Return named reserved columns carried in :attr:`data`."""

        return self.strategy_schema.targets(self.data)

    def select(self, indices: Sequence[int]) -> "VideoConcept":
        selected = np.asarray(indices, dtype=np.int64)
        labels = None if self.labels is None else self.labels[selected]
        return type(self)(
            name=self.name,
            data=self.data[selected],
            labels=labels,
            windows=tuple(self.windows[index] for index in selected),
            strategy_schema=self.strategy_schema,
        )
