"""Convert frame-level feature sequences into strategy-ready window matrices."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from pyclad.video.data.sample import VideoWindow


def window_frame_features(
    frame_features: np.ndarray,
    video_id: str,
    window_size: int,
    stride: int,
    *,
    frame_labels: Optional[Sequence[int]] = None,
    aggregation: str = "mean",
    split: str = "train",
    anomaly_class: Optional[str] = None,
    concept_id: Optional[str] = None,
    domain_id: Optional[str] = None,
    feature_index_offset: int = 0,
    drop_last: bool = False,
) -> Tuple[np.ndarray, Tuple[VideoWindow, ...]]:
    """Pool frame features and create aligned metadata sidecars."""
    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"frame_features must be 2D, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("frame_features must contain at least one frame")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be one of: 'mean', 'max'")

    labels = None
    if frame_labels is not None:
        labels = np.asarray(frame_labels, dtype=np.int64).reshape(-1)
        if len(labels) != len(values):
            raise ValueError(
                f"frame_labels and frame_features must have the same length: " f"{len(labels)} != {len(values)}"
            )

    pooled = []
    windows = []
    for start in range(0, len(values), stride):
        stop = min(start + window_size, len(values))
        if drop_last and stop - start < window_size:
            break
        feature_window = values[start:stop]
        if aggregation == "mean":
            pooled_feature = np.mean(feature_window, axis=0)
        else:
            pooled_feature = np.max(feature_window, axis=0)
        label = None if labels is None else int(np.any(labels[start:stop] == 1))
        pooled.append(pooled_feature)
        windows.append(
            VideoWindow(
                video_id=video_id,
                start_frame=start,
                end_frame=stop - 1,
                feature_index=feature_index_offset + len(pooled) - 1,
                split=split,
                label=label,
                anomaly_class=anomaly_class,
                concept_id=concept_id,
                domain_id=domain_id,
            )
        )
        if stop == len(values):
            break

    if not pooled:
        raise ValueError("windowing produced no video windows")
    return np.asarray(pooled, dtype=np.float32), tuple(windows)
