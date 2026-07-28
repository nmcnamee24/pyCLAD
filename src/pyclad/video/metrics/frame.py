"""Frame-level metrics for video anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pyclad.video.features.windows import flatten_video_curves


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
        auc=_binary_metric(roc_auc_score, y_true, y_score),
        ap=_binary_metric(average_precision_score, y_true, y_score),
        auc_anomalous_videos=_binary_metric(
            roc_auc_score,
            flatten_video_curves(anomalous_labels).astype(np.int64),
            flatten_video_curves(anomalous_scores),
        ),
        ap_anomalous_videos=_binary_metric(
            average_precision_score,
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
    return float(metric(y_true=y_true, y_score=y_score))


def _snr(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        return float("nan")
    normal_std = float(np.std(normal_scores))
    if normal_std == 0.0:
        return float("inf")
    return float((np.mean(anomaly_scores) - np.mean(normal_scores)) / normal_std)
