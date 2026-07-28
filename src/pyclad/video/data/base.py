"""Read-only interfaces for feature-based video anomaly datasets."""

from __future__ import annotations

import abc
from typing import Mapping, Sequence

import numpy as np

from pyclad.video.data.sample import VideoWindow
from pyclad.video.features.store import VideoFeatureStore


class VideoDataset(abc.ABC):
    """A video dataset whose strategy-facing representation is a matrix."""

    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def feature_store(self) -> VideoFeatureStore: ...

    @abc.abstractmethod
    def windows(self, split: str = "test") -> Sequence[VideoWindow]: ...

    @abc.abstractmethod
    def frame_labels(self, split: str = "test") -> Mapping[str, np.ndarray]: ...

    def feature_matrix(self, split: str = "test") -> np.ndarray:
        """Return the plain matrix passed to unchanged pyCLAD strategies."""
        indices = [window.feature_index for window in self.windows(split=split)]
        return np.asarray(self.feature_store().take(indices), dtype=np.float32)

    def feature_concept(self, name: str, split: str = "test"):
        """Build a video-owned concept that can emit a regular pyCLAD Concept."""
        from pyclad.video.data.concepts import VideoFeatureConcept

        windows = tuple(self.windows(split=split))
        return VideoFeatureConcept(
            name=name,
            features=self.feature_matrix(split=split),
            windows=windows,
        )
