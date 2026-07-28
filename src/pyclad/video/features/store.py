"""Stores for model-ready video-window feature matrices."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


class VideoFeatureStore(abc.ABC):
    @abc.abstractmethod
    def embeddings(self) -> np.ndarray: ...

    def take(self, indices: Sequence[int]) -> np.ndarray:
        return self.embeddings()[np.asarray(indices, dtype=np.int64)]

    @property
    def embedding_dim(self) -> int:
        return int(_matrix(self.embeddings()).shape[1])

    def __len__(self) -> int:
        return int(_matrix(self.embeddings()).shape[0])


class NpyVideoFeatureStore(VideoFeatureStore):
    """Lazily load a two-dimensional ``.npy`` feature matrix."""

    def __init__(self, features_path: Union[str, Path], mmap_mode: Optional[str] = None):
        self.features_path = Path(features_path)
        self.mmap_mode = mmap_mode
        self._embeddings: Optional[np.ndarray] = None

    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = _matrix(np.load(self.features_path, mmap_mode=self.mmap_mode))
        return self._embeddings


class InMemoryVideoFeatureStore(VideoFeatureStore):
    def __init__(self, embeddings: np.ndarray):
        self._embeddings = _matrix(embeddings).astype(np.float32, copy=False)

    def embeddings(self) -> np.ndarray:
        return self._embeddings


def _matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"video features must be 2D, got shape {values.shape}")
    return values
