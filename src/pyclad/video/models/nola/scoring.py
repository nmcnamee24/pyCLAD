"""Sequential NOLA scoring, diagnostics, and track-cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


class DegenerateNolaScoresError(RuntimeError):
    """Raised when a completed NOLA evaluation has no ranking information."""


@dataclass(frozen=True)
class NolaScoreDiagnostics:
    rows: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    nonzero_fraction: float
    unique_values: int

    @property
    def is_degenerate(self) -> bool:
        return self.rows == 0 or self.unique_values < 2 or self.standard_deviation == 0.0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "rows": self.rows,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "nonzero_fraction": self.nonzero_fraction,
            "unique_values": self.unique_values,
            "degenerate": self.is_degenerate,
        }


def nola_score_diagnostics(scores: Sequence[float]) -> NolaScoreDiagnostics:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(values):
        return NolaScoreDiagnostics(0, float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0)
    if not np.isfinite(values).all():
        raise ValueError("NOLA scores must contain only finite values")
    return NolaScoreDiagnostics(
        rows=len(values),
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
        standard_deviation=float(values.std()),
        nonzero_fraction=float(np.count_nonzero(values) / len(values)),
        unique_values=int(len(np.unique(values))),
    )


def require_non_degenerate_nola_scores(scores: Sequence[float]) -> NolaScoreDiagnostics:
    diagnostics = nola_score_diagnostics(scores)
    if diagnostics.is_degenerate:
        raise DegenerateNolaScoresError(
            "NOLA produced degenerate anomaly scores "
            f"(rows={diagnostics.rows}, unique={diagnostics.unique_values}, "
            f"std={diagnostics.standard_deviation}, min={diagnostics.minimum}, "
            f"max={diagnostics.maximum}). Refusing to report a chance-level run."
        )
    return diagnostics


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
