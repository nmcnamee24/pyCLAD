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
    true_positives: np.ndarray
    false_positives: np.ndarray


def compute_average_precision_delay(
    video_scores: Mapping[str, Sequence[float]],
    anomaly_intervals: Mapping[str, Sequence[Tuple[int, int]]],
    *,
    thresholds: Optional[Sequence[float]] = None,
    maximum_delay: int = 9_000,
) -> AveragePrecisionDelay:
    """Compute NOLA's event-level Average Precision-Delay metric.

    Each annotated activity is evaluated independently. The first alarm inside
    its interval is a true alarm and determines the delay; subsequent alarms in
    the same interval are ignored. Contiguous alarm runs outside all annotated
    intervals are false alarms. A missed activity receives ``maximum_delay``.

    NOLA clips are five minutes long at 30 FPS, so the paper's default maximum
    tolerable delay is 9,000 frames.
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
    if not len(all_scores):
        raise ValueError("video_scores must contain at least one score")
    if thresholds is not None:
        threshold_values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if not len(threshold_values) or not np.isfinite(threshold_values).all():
            raise ValueError("thresholds must contain at least one finite value")

    activity_count = sum(len(intervals) for intervals in normalized_intervals.values())
    if activity_count == 0:
        raise ValueError("at least one video must contain an anomaly interval")
    if maximum_delay <= 0:
        raise ValueError("maximum_delay must be positive")
    maximum_delay_value = int(maximum_delay)

    if thresholds is None:
        (
            threshold_values,
            true_positives,
            false_positives,
            delay_sums,
        ) = _exact_alarm_state_sweep(
            score_arrays,
            normalized_intervals,
            maximum_delay_value,
        )
    else:
        true_positives = np.zeros(len(threshold_values), dtype=np.int64)
        false_positives = np.zeros(len(threshold_values), dtype=np.int64)
        delay_sums = np.zeros(len(threshold_values), dtype=np.float64)
        for video_id, scores in score_arrays.items():
            intervals = normalized_intervals.get(video_id, ())
            relevant_mask = np.zeros(len(scores), dtype=bool)
            for start, stop in intervals:
                relevant_mask[start:stop] = True

            for threshold_index, threshold in enumerate(threshold_values):
                alarms = scores > threshold
                for start, stop in intervals:
                    deadline = min(stop, start + maximum_delay_value, len(scores))
                    alarm_positions = np.flatnonzero(alarms[start:deadline])
                    if len(alarm_positions):
                        true_positives[threshold_index] += 1
                        delay_sums[threshold_index] += int(alarm_positions[0])
                    else:
                        delay_sums[threshold_index] += maximum_delay_value

                false_alarm_mask = alarms & ~relevant_mask
                false_positives[threshold_index] += _count_alarm_runs(false_alarm_mask)

    denominator = true_positives + false_positives
    precisions_array = np.divide(
        true_positives,
        denominator,
        out=np.zeros(len(threshold_values), dtype=np.float64),
        where=denominator != 0,
    )
    delays_array = delay_sums / activity_count / maximum_delay_value
    order = np.argsort(delays_array, kind="stable")
    sorted_delay = delays_array[order]
    sorted_precision = precisions_array[order]
    unique_delay, first_indices = np.unique(sorted_delay, return_index=True)
    envelope = np.maximum.reduceat(
        sorted_precision,
        first_indices,
    )
    score = float(np.trapezoid(envelope, unique_delay)) if len(unique_delay) > 1 else 0.0

    return AveragePrecisionDelay(
        score=score,
        thresholds=threshold_values,
        normalized_delays=delays_array,
        precisions=precisions_array,
        true_positives=true_positives,
        false_positives=false_positives,
    )


def _exact_alarm_state_sweep(
    score_arrays: Mapping[str, np.ndarray],
    anomaly_intervals: Mapping[str, Tuple[Tuple[int, int], ...]],
    maximum_delay: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every distinct alarm state in ``O(n log n)`` time."""

    event_records = []
    events_by_video: Dict[str, list[tuple[int, int, int]]] = {}
    relevant_masks = {}
    active_masks = {}
    ranked_frames = []
    for video_id, scores in score_arrays.items():
        relevant = np.zeros(len(scores), dtype=bool)
        video_events = []
        for start, stop in anomaly_intervals.get(video_id, ()):
            event_id = len(event_records)
            deadline = min(stop, start + maximum_delay, len(scores))
            event_records.append((start, deadline))
            video_events.append((event_id, start, deadline))
            relevant[start:stop] = True
        events_by_video[video_id] = video_events
        relevant_masks[video_id] = relevant
        active_masks[video_id] = np.zeros(len(scores), dtype=bool)
        ranked_frames.extend((float(score), video_id, frame_id) for frame_id, score in enumerate(scores))

    ranked_frames.sort(key=lambda item: item[0], reverse=True)
    earliest_alarms = np.full(len(event_records), -1, dtype=np.int64)
    true_positive_count = 0
    false_positive_count = 0
    delay_sum = float(len(event_records) * maximum_delay)
    thresholds = [np.nextafter(ranked_frames[0][0], np.inf)]
    true_positives = [0]
    false_positives = [0]
    delay_sums = [delay_sum]

    cursor = 0
    while cursor < len(ranked_frames):
        score = ranked_frames[cursor][0]
        group_end = cursor + 1
        while group_end < len(ranked_frames) and ranked_frames[group_end][0] == score:
            group_end += 1
        for _, video_id, frame_id in ranked_frames[cursor:group_end]:
            active = active_masks[video_id]
            if not relevant_masks[video_id][frame_id]:
                left_active = frame_id > 0 and active[frame_id - 1] and not relevant_masks[video_id][frame_id - 1]
                right_active = (
                    frame_id + 1 < len(active) and active[frame_id + 1] and not relevant_masks[video_id][frame_id + 1]
                )
                false_positive_count += 1 - int(left_active) - int(right_active)
            active[frame_id] = True

            for event_id, start, deadline in events_by_video[video_id]:
                if not start <= frame_id < deadline:
                    continue
                previous = earliest_alarms[event_id]
                if previous < 0:
                    earliest_alarms[event_id] = frame_id
                    true_positive_count += 1
                    delay_sum += frame_id - start - maximum_delay
                elif frame_id < previous:
                    earliest_alarms[event_id] = frame_id
                    delay_sum += frame_id - previous

        thresholds.append(np.nextafter(score, -np.inf))
        true_positives.append(true_positive_count)
        false_positives.append(false_positive_count)
        delay_sums.append(delay_sum)
        cursor = group_end

    return (
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(true_positives, dtype=np.int64),
        np.asarray(false_positives, dtype=np.int64),
        np.asarray(delay_sums, dtype=np.float64),
    )


def _count_alarm_runs(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not len(values):
        return 0
    starts = values & ~np.concatenate(([False], values[:-1]))
    return int(np.count_nonzero(starts))


def _validate_intervals(
    intervals: Sequence[Tuple[int, int]],
    length: int,
    video_id: str,
) -> Tuple[Tuple[int, int], ...]:
    normalized = tuple(sorted((int(start), int(stop)) for start, stop in intervals))
    for start, stop in normalized:
        if start < 0 or stop <= start:
            raise ValueError(f"invalid anomaly interval {(start, stop)} for {video_id!r} of length {length}")
    return tuple((start, min(stop, length)) for start, stop in normalized if start < length)
