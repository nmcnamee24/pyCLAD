"""Average Precision-Delay evaluation used by NOLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AveragePrecisionDelay:
    """Threshold sweep and area under the precision-delay curve."""

    score: float
    thresholds: np.ndarray
    normalized_delays: np.ndarray
    precisions: np.ndarray


def compute_average_precision_delay(
    video_scores: Mapping[str, Sequence[float]],
    anomaly_intervals: Mapping[str, Sequence[Tuple[int, int]]],
    *,
    thresholds: Optional[Sequence[float]] = None,
    maximum_delay: Optional[int] = None,
) -> AveragePrecisionDelay:
    """Compute NOLA's APD metric across videos.

    An alarm before the first anomalous interval counts as a false positive.
    The first alarm inside an interval counts as a true positive and determines
    delay. A miss receives the maximum normalized delay of one.
    """

    if not video_scores:
        raise ValueError("video_scores must not be empty")
    score_arrays: Dict[str, np.ndarray] = {}
    for video_id, scores in video_scores.items():
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError(f"scores for {video_id!r} must be finite")
        score_arrays[video_id] = values

    unknown = set(anomaly_intervals) - set(score_arrays)
    if unknown:
        raise ValueError(f"anomaly_intervals contain unknown videos: {sorted(unknown)}")
    normalized_intervals = {
        video_id: _validate_intervals(intervals, len(score_arrays[video_id]), video_id)
        for video_id, intervals in anomaly_intervals.items()
    }

    all_scores = np.concatenate(list(score_arrays.values()))
    if thresholds is None:
        threshold_values = np.unique(
            np.concatenate(
                [
                    np.array([np.nextafter(all_scores.max(), np.inf)]),
                    all_scores,
                    np.array([np.nextafter(all_scores.min(), -np.inf)]),
                ]
            )
        )[::-1]
    else:
        threshold_values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if not len(threshold_values) or not np.isfinite(threshold_values).all():
            raise ValueError("thresholds must contain at least one finite value")

    anomaly_video_count = sum(bool(intervals) for intervals in normalized_intervals.values())
    if anomaly_video_count == 0:
        raise ValueError("at least one video must contain an anomaly interval")

    if maximum_delay is None:
        maximum_delay_value = max(
            len(scores) for video_id, scores in score_arrays.items() if normalized_intervals.get(video_id)
        )
    else:
        if maximum_delay <= 0:
            raise ValueError("maximum_delay must be positive")
        maximum_delay_value = int(maximum_delay)

    precisions = []
    delays = []
    for threshold in threshold_values:
        true_positives = 0
        false_positives = 0
        video_delays = []

        for video_id, scores in score_arrays.items():
            intervals = normalized_intervals.get(video_id, ())
            alarms = np.flatnonzero(scores > threshold)
            if not intervals:
                false_positives += int(bool(len(alarms)))
                continue

            first_start = intervals[0][0]
            false_positives += int(bool(np.any(alarms < first_start)))
            detections = [int(alarm) for alarm in alarms if any(start <= alarm < stop for start, stop in intervals)]
            if detections:
                detection = detections[0]
                containing_start = next(start for start, stop in intervals if start <= detection < stop)
                true_positives += 1
                video_delays.append(min(detection - containing_start, maximum_delay_value))
            else:
                video_delays.append(maximum_delay_value)

        denominator = true_positives + false_positives
        precisions.append(true_positives / denominator if denominator else 0.0)
        delays.append(float(np.mean(video_delays)) / maximum_delay_value)

    delays_array = np.asarray(delays, dtype=np.float64)
    precisions_array = np.asarray(precisions, dtype=np.float64)
    order = np.argsort(delays_array, kind="stable")
    sorted_delay = delays_array[order]
    sorted_precision = precisions_array[order]
    unique_delay = np.unique(sorted_delay)
    envelope = np.array(
        [sorted_precision[sorted_delay == delay].max() for delay in unique_delay],
        dtype=np.float64,
    )
    score = float(np.trapezoid(envelope, unique_delay)) if len(unique_delay) > 1 else 0.0

    return AveragePrecisionDelay(
        score=score,
        thresholds=threshold_values,
        normalized_delays=delays_array,
        precisions=precisions_array,
    )


def _validate_intervals(
    intervals: Sequence[Tuple[int, int]],
    length: int,
    video_id: str,
) -> Tuple[Tuple[int, int], ...]:
    normalized = tuple(sorted((int(start), int(stop)) for start, stop in intervals))
    for start, stop in normalized:
        if start < 0 or stop <= start or stop > length:
            raise ValueError(f"invalid anomaly interval {(start, stop)} for {video_id!r} of length {length}")
    return normalized
