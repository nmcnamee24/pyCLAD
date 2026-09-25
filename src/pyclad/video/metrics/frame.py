"""Frame-level metrics for video anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from pyclad.video.data.sample import VideoWindow
from pyclad.video.metrics.frame_average_precision import FrameAveragePrecision
from pyclad.video.metrics.frame_roc_auc import FrameRocAuc


def window_scores_to_frame_scores(
    windows: Sequence[VideoWindow],
    window_scores: Sequence[float],
    frame_counts: Mapping[str, int],
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    """Aggregate temporal-window scores onto their covered frames."""

    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be one of: 'mean', 'max'")
    if len(windows) != len(window_scores):
        raise ValueError("windows and window_scores must have the same length")

    initial = -np.inf if aggregation == "max" else 0.0
    scores = {
        video_id: np.full(frame_count, initial, dtype=np.float64) for video_id, frame_count in frame_counts.items()
    }
    counts = {video_id: np.zeros(frame_count, dtype=np.float64) for video_id, frame_count in frame_counts.items()}
    for video_id, frame_count in frame_counts.items():
        if frame_count <= 0:
            raise ValueError(f"frame count must be positive for video_id={video_id!r}")

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
            scores[window.video_id][frame_slice] = np.maximum(
                scores[window.video_id][frame_slice],
                float(score),
            )
        else:
            scores[window.video_id][frame_slice] += float(score)
        counts[window.video_id][frame_slice] += 1.0

    for video_id, values in scores.items():
        covered = counts[video_id] > 0
        if aggregation == "mean":
            values[covered] /= counts[video_id][covered]
        values[~covered] = 0.0
    return scores


def flatten_video_curves(curves_by_video: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenate video curves in stable video-id order."""

    if not curves_by_video:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([np.asarray(curves_by_video[video_id]).reshape(-1) for video_id in sorted(curves_by_video)])


@dataclass(frozen=True)
class VideoFrameMetrics:
    auc: float
    ap: float
    auc_anomalous_videos: float
    ap_anomalous_videos: float
    snr: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "AUC": self.auc,
            "AP": self.ap,
            "AUC_A": self.auc_anomalous_videos,
            "AP_A": self.ap_anomalous_videos,
            "SNR": self.snr,
        }


def compute_video_frame_metrics(
    frame_scores: Mapping[str, np.ndarray],
    frame_labels: Mapping[str, np.ndarray],
) -> VideoFrameMetrics:
    _validate_matching_videos(frame_scores, frame_labels)
    y_score = flatten_video_curves(frame_scores)
    y_true = flatten_video_curves(frame_labels).astype(np.int64)

    anomalous_video_ids = [
        video_id for video_id, labels in frame_labels.items() if np.any(np.asarray(labels).reshape(-1) == 1)
    ]
    anomalous_scores = {video_id: frame_scores[video_id] for video_id in anomalous_video_ids}
    anomalous_labels = {video_id: frame_labels[video_id] for video_id in anomalous_video_ids}

    normal_scores = y_score[y_true == 0]
    anomaly_scores = y_score[y_true == 1]
    return VideoFrameMetrics(
        auc=_binary_metric(FrameRocAuc(), y_true, y_score),
        ap=_binary_metric(FrameAveragePrecision(), y_true, y_score),
        auc_anomalous_videos=_binary_metric(
            FrameRocAuc(),
            flatten_video_curves(anomalous_labels).astype(np.int64),
            flatten_video_curves(anomalous_scores),
        ),
        ap_anomalous_videos=_binary_metric(
            FrameAveragePrecision(),
            flatten_video_curves(anomalous_labels).astype(np.int64),
            flatten_video_curves(anomalous_scores),
        ),
        snr=_snr(normal_scores, anomaly_scores),
    )


def _validate_matching_videos(
    frame_scores: Mapping[str, np.ndarray],
    frame_labels: Mapping[str, np.ndarray],
) -> None:
    if set(frame_scores) != set(frame_labels):
        raise ValueError("frame_scores and frame_labels must contain the same video ids")
    for video_id in frame_scores:
        score_shape = np.asarray(frame_scores[video_id]).reshape(-1).shape
        label_shape = np.asarray(frame_labels[video_id]).reshape(-1).shape
        if score_shape != label_shape:
            raise ValueError(f"Score and label length mismatch for video_id={video_id!r}")


def _binary_metric(metric, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return metric.compute(y_true=y_true, anomaly_scores=y_score, y_pred=None)


def _snr(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        return float("nan")
    normal_std = float(np.std(normal_scores))
    if normal_std == 0.0:
        return float("inf")
    return float((np.mean(anomaly_scores) - np.mean(normal_scores)) / normal_std)
