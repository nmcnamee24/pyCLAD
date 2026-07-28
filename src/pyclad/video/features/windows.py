"""Project window-level anomaly scores back onto video frames."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from pyclad.video.data.sample import VideoWindow


def window_scores_to_frame_scores(
    windows: Sequence[VideoWindow],
    window_scores: Sequence[float],
    frame_counts: Mapping[str, int],
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be one of: 'mean', 'max'")
    if len(windows) != len(window_scores):
        raise ValueError("windows and window_scores must have the same length")

    scores_by_video: Dict[str, np.ndarray] = {}
    counts_by_video: Dict[str, np.ndarray] = {}
    touched: Dict[str, np.ndarray] = {}
    for video_id, frame_count in frame_counts.items():
        if frame_count <= 0:
            raise ValueError(f"frame count must be positive for video_id={video_id!r}")
        initial = -np.inf if aggregation == "max" else 0.0
        scores_by_video[video_id] = np.full(frame_count, initial, dtype=np.float64)
        counts_by_video[video_id] = np.zeros(frame_count, dtype=np.float64)
        touched[video_id] = np.zeros(frame_count, dtype=bool)

    for window, score in zip(windows, window_scores):
        if window.video_id not in frame_counts:
            raise KeyError(f"Missing frame count for video_id={window.video_id!r}")
        frame_count = frame_counts[window.video_id]
        if window.start_frame >= frame_count:
            raise ValueError(
                f"Window starts outside video_id={window.video_id!r}: " f"start_frame={window.start_frame}"
            )
        frame_slice = slice(window.start_frame, min(window.end_frame, frame_count - 1) + 1)
        if aggregation == "max":
            scores_by_video[window.video_id][frame_slice] = np.maximum(
                scores_by_video[window.video_id][frame_slice],
                float(score),
            )
        else:
            scores_by_video[window.video_id][frame_slice] += float(score)
        counts_by_video[window.video_id][frame_slice] += 1.0
        touched[window.video_id][frame_slice] = True

    for video_id, scores in scores_by_video.items():
        covered = touched[video_id]
        if aggregation == "mean":
            scores[covered] /= counts_by_video[video_id][covered]
        scores[~covered] = 0.0
    return scores_by_video


def flatten_video_curves(curves_by_video: Mapping[str, np.ndarray]) -> np.ndarray:
    if not curves_by_video:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([np.asarray(curves_by_video[video_id]).reshape(-1) for video_id in sorted(curves_by_video)])
