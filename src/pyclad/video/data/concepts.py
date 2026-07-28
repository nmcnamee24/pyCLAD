"""Adapters from video feature sidecars to unchanged pyCLAD concepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.data.sample import VideoWindow


@dataclass
class VideoFeatureConcept:
    """A feature matrix plus video metadata kept outside the strategy input."""

    name: str
    features: np.ndarray
    windows: Sequence[VideoWindow]
    labels: Optional[np.ndarray] = None
    strategy_schema: Optional[VideoStrategySchema] = None
    strategy_targets: Optional[Mapping[str, Sequence[float]]] = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        windows = tuple(self.windows)
        if len(features) != len(windows):
            raise ValueError(f"features and windows must have the same length: {len(features)} != {len(windows)}")

        labels = self.labels
        if labels is None and windows and all(window.label is not None for window in windows):
            labels = np.asarray([window.label for window in windows], dtype=np.int64)
        elif labels is not None:
            labels = np.asarray(labels, dtype=np.int64).reshape(-1)
            if len(labels) != len(features):
                raise ValueError(f"labels and features must have the same length: {len(labels)} != {len(features)}")

        self.features = features
        self.windows = windows
        self.labels = labels
        if self.strategy_schema is None:
            self.strategy_schema = VideoStrategySchema(feature_dim=features.shape[1])
        if self.strategy_schema.feature_dim != features.shape[1]:
            raise ValueError(
                f"strategy_schema expects {self.strategy_schema.feature_dim} features, " f"got {features.shape[1]}"
            )
        self.strategy_targets = {
            name: np.asarray(values, dtype=np.float32).reshape(-1)
            for name, values in ({} if self.strategy_targets is None else self.strategy_targets).items()
        }
        self.strategy_schema.pack(self.features, self.strategy_targets)

    def __len__(self) -> int:
        return len(self.features)

    def to_pyclad(self) -> Concept:
        """Return the exact Concept interface consumed by existing scenarios."""
        return Concept(name=self.name, data=self.strategy_matrix(), labels=self.labels)

    def strategy_matrix(self) -> np.ndarray:
        return self.strategy_schema.pack(self.features, self.strategy_targets)

    def select(self, indices: Sequence[int]) -> "VideoFeatureConcept":
        selected = np.asarray(indices, dtype=np.int64)
        labels = None if self.labels is None else self.labels[selected]
        targets = {name: values[selected] for name, values in self.strategy_targets.items()}
        return type(self)(
            name=self.name,
            features=self.features[selected],
            windows=tuple(self.windows[index] for index in selected),
            labels=labels,
            strategy_schema=self.strategy_schema,
            strategy_targets=targets,
        )


class VideoConceptsDataset(ConceptsDataset):
    """A regular ConceptsDataset with video metadata retained as sidecars."""

    def __init__(
        self,
        name: str,
        train_concepts: Sequence[VideoFeatureConcept],
        test_concepts: Sequence[VideoFeatureConcept],
    ):
        self._video_train_concepts: Tuple[VideoFeatureConcept, ...] = tuple(train_concepts)
        self._video_test_concepts: Tuple[VideoFeatureConcept, ...] = tuple(test_concepts)
        super().__init__(
            name=name,
            train_concepts=[concept.to_pyclad() for concept in self._video_train_concepts],
            test_concepts=[concept.to_pyclad() for concept in self._video_test_concepts],
        )

    def video_train_concepts(self) -> Tuple[VideoFeatureConcept, ...]:
        return self._video_train_concepts

    def video_test_concepts(self) -> Tuple[VideoFeatureConcept, ...]:
        return self._video_test_concepts

    def video_concept(self, name: str, split: str) -> VideoFeatureConcept:
        if split not in {"train", "test"}:
            raise ValueError("split must be one of: 'train', 'test'")
        concepts = self._video_train_concepts if split == "train" else self._video_test_concepts
        matches = [concept for concept in concepts if concept.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one {split} video concept named {name!r}, found {len(matches)}")
        return matches[0]

    def additional_info(self):
        info = super().additional_info()
        info["modality"] = "video"
        return info
