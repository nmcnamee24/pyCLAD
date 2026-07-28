"""Video datasets backed by precomputed window embeddings."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from pyclad.video.data.base import VideoDataset
from pyclad.video.data.sample import VideoWindow
from pyclad.video.features.store import VideoFeatureStore


class PrecomputedVideoDataset(VideoDataset):
    def __init__(
        self,
        dataset_name: str,
        feature_store: VideoFeatureStore,
        windows: Sequence[VideoWindow],
        frame_labels_by_split: Mapping[str, Mapping[str, np.ndarray]],
    ):
        self._dataset_name = dataset_name
        self._feature_store = feature_store
        self._windows = tuple(windows)
        self._frame_labels_by_split = {
            split: {
                video_id: np.asarray(labels, dtype=np.int64).reshape(-1) for video_id, labels in labels_by_video.items()
            }
            for split, labels_by_video in frame_labels_by_split.items()
        }
        self._validate()

    def name(self) -> str:
        return self._dataset_name

    def feature_store(self) -> VideoFeatureStore:
        return self._feature_store

    def windows(self, split: str = "test") -> Sequence[VideoWindow]:
        return tuple(window for window in self._windows if window.split == split)

    def frame_labels(self, split: str = "test") -> Mapping[str, np.ndarray]:
        if split not in self._frame_labels_by_split:
            raise KeyError(f"No frame labels registered for split={split!r}")
        return self._frame_labels_by_split[split]

    def _validate(self) -> None:
        if not self._windows:
            raise ValueError("PrecomputedVideoDataset requires at least one VideoWindow")

        max_feature_index = max(window.feature_index for window in self._windows)
        if max_feature_index >= len(self._feature_store):
            raise ValueError(
                f"feature_index={max_feature_index} is outside feature store length " f"{len(self._feature_store)}"
            )

        for window in self._windows:
            try:
                frame_labels = self._frame_labels_by_split[window.split][window.video_id]
            except KeyError as error:
                raise ValueError(
                    f"Missing frame labels for split={window.split!r}, video_id={window.video_id!r}"
                ) from error
            if window.start_frame >= len(frame_labels):
                raise ValueError(
                    f"Window starts outside labels for video_id={window.video_id!r}: "
                    f"start_frame={window.start_frame}, frame_count={len(frame_labels)}"
                )
