"""Sequential NOLA scoring and track-cleaning utilities."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def odit_cusum(statistics: Sequence[float], drift: float = 7.0, initial: float = 0.0) -> np.ndarray:
    """Apply the non-negative ODIT/CUSUM recurrence used by NOLA."""

    values = np.asarray(statistics, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("statistics must contain only finite values")
    if drift < 0:
        raise ValueError("drift must be non-negative")
    if initial < 0:
        raise ValueError("initial must be non-negative")

    result = np.empty(len(values), dtype=np.float64)
    running = float(initial)
    for index, value in enumerate(values):
        running = max(0.0, running + float(value) - drift)
        result[index] = running
    return result


def non_maximum_suppression(
    boxes: np.ndarray,
    overlap_threshold: float = 0.7,
    scores: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return indices of boxes retained after intersection-over-union NMS."""

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (rows, 4), got {boxes.shape}")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between 0 and 1")
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    if scores is None:
        priorities = np.arange(len(boxes), dtype=np.float64)
    else:
        priorities = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(priorities) != len(boxes):
            raise ValueError("scores must have one value per box")

    x1, y1, x2, y2 = boxes.T
    widths = np.maximum(0.0, x2 - x1)
    heights = np.maximum(0.0, y2 - y1)
    areas = widths * heights
    order = priorities.argsort()[::-1]
    keep = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        intersection_x1 = np.maximum(x1[current], x1[remaining])
        intersection_y1 = np.maximum(y1[current], y1[remaining])
        intersection_x2 = np.minimum(x2[current], x2[remaining])
        intersection_y2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
            0.0, intersection_y2 - intersection_y1
        )
        union = areas[current] + areas[remaining] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[overlap <= overlap_threshold]

    return np.asarray(keep, dtype=np.int64)
