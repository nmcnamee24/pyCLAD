"""Feature layout helpers for NOLA's spatial, temporal, and track signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

NOLA_OBJECT_ALIASES = {
    "bicycle": "bike",
    "motorbike": "bike",
    "motorcycle": "bike",
    "potted plant": "pottedplant",
}


def canonical_nola_object_name(name: object) -> str:
    """Normalize Darknet, COCO, and torchvision class spellings."""

    normalized = str(name).strip().lower()
    return NOLA_OBJECT_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class NolaFeatureLayout:
    """Column layout for a strategy-facing NOLA feature matrix.

    The original model uses five spatial object features (bounding box and
    class), three temporal count/time features, and an optional trajectory
    prediction error.
    """

    spatial_columns: Tuple[int, ...] = (0, 1, 2, 3, 4)
    temporal_columns: Tuple[int, ...] = (5, 6, 7)
    trajectory_error_column: Optional[int] = 8

    def __post_init__(self) -> None:
        spatial = tuple(self.spatial_columns)
        temporal = tuple(self.temporal_columns)
        object.__setattr__(self, "spatial_columns", spatial)
        object.__setattr__(self, "temporal_columns", temporal)
        all_columns = spatial + temporal
        if not spatial or not temporal:
            raise ValueError("NOLA requires at least one spatial and one temporal column")
        if any(not isinstance(column, int) or column < 0 for column in all_columns):
            raise ValueError("NOLA feature columns must be non-negative integers")
        if len(set(all_columns)) != len(all_columns):
            raise ValueError("spatial_columns and temporal_columns must not overlap")
        if self.trajectory_error_column is not None:
            if not isinstance(self.trajectory_error_column, int) or self.trajectory_error_column < 0:
                raise ValueError("trajectory_error_column must be a non-negative integer or None")
            if self.trajectory_error_column in all_columns:
                raise ValueError("trajectory_error_column must not overlap other columns")

    @property
    def feature_dim(self) -> int:
        columns = self.spatial_columns + self.temporal_columns
        if self.trajectory_error_column is not None:
            columns += (self.trajectory_error_column,)
        return max(columns) + 1

    def split(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < self.feature_dim:
            raise ValueError(f"NOLA features must have shape (rows, at least {self.feature_dim}), got {matrix.shape}")
        spatial = matrix[:, self.spatial_columns]
        temporal = matrix[:, self.temporal_columns]
        trajectory_error = None if self.trajectory_error_column is None else matrix[:, self.trajectory_error_column]
        return spatial, temporal, trajectory_error


def pack_nola_features(
    spatial: np.ndarray,
    temporal: np.ndarray,
    trajectory_error: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, NolaFeatureLayout]:
    """Pack the standard NOLA feature families into one float32 matrix."""

    spatial_array = np.asarray(spatial, dtype=np.float32)
    temporal_array = np.asarray(temporal, dtype=np.float32)
    if spatial_array.ndim != 2 or temporal_array.ndim != 2:
        raise ValueError("spatial and temporal features must be two-dimensional")
    if len(spatial_array) != len(temporal_array):
        raise ValueError("spatial and temporal features must have the same number of rows")

    arrays = [spatial_array, temporal_array]
    trajectory_column = None
    if trajectory_error is not None:
        error = np.asarray(trajectory_error, dtype=np.float32).reshape(-1)
        if len(error) != len(spatial_array):
            raise ValueError("trajectory_error must have one value per row")
        trajectory_column = spatial_array.shape[1] + temporal_array.shape[1]
        arrays.append(error[:, None])

    layout = NolaFeatureLayout(
        spatial_columns=tuple(range(spatial_array.shape[1])),
        temporal_columns=tuple(range(spatial_array.shape[1], spatial_array.shape[1] + temporal_array.shape[1])),
        trajectory_error_column=trajectory_column,
    )
    return np.concatenate(arrays, axis=1), layout


def nola_temporal_object_features(
    objects: Sequence[Mapping[str, object]],
    hour: float,
    *,
    vehicle_names: Sequence[str] = ("car", "bike", "truck", "cart"),
    confidence_threshold: float = 0.6,
) -> np.ndarray:
    """Create NOLA's vehicle-count, person-count, and time feature."""

    vehicles = set(vehicle_names)
    vehicle_count = 0
    person_count = 0
    for detected in objects:
        name = canonical_nola_object_name(detected.get("name", ""))
        confidence = float(detected.get("confidence", 0.0))
        if confidence <= confidence_threshold:
            continue
        if name == "person":
            person_count += 1
        elif name in vehicles:
            vehicle_count += 1
    return np.asarray([vehicle_count, person_count, float(hour)], dtype=np.float32)


def nola_spatial_object_features(
    boxes: np.ndarray,
    class_names: Sequence[str],
    *,
    relevant_classes: Sequence[str] = ("car", "bike", "truck", "cart"),
) -> np.ndarray:
    """Create NOLA's bounding-box and categorical object features."""

    box_array = np.asarray(boxes, dtype=np.float32)
    if box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError(f"boxes must have shape (rows, 4), got {box_array.shape}")
    if len(class_names) != len(box_array):
        raise ValueError("class_names must have one entry per box")
    class_to_index = {canonical_nola_object_name(name): index for index, name in enumerate(relevant_classes)}
    try:
        class_indices = np.asarray(
            [class_to_index[canonical_nola_object_name(name)] for name in class_names],
            dtype=np.float32,
        )
    except KeyError as error:
        raise ValueError(f"unknown NOLA object class: {error.args[0]!r}") from error
    return np.concatenate([box_array, class_indices[:, None]], axis=1)


def build_nola_trajectory_examples(
    tracks: Sequence[np.ndarray],
    *,
    sequence_length: int = 20,
    stride: int = 5,
    frame_size: Tuple[float, float] = (1280.0, 720.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized bounding-box sequences and next-box targets."""

    if sequence_length <= 0 or stride <= 0:
        raise ValueError("sequence_length and stride must be positive")
    frame_width, frame_height = map(float, frame_size)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_size dimensions must be positive")
    scale = np.asarray([frame_width, frame_height, frame_width, frame_height], dtype=np.float32)

    sequences = []
    targets = []
    for track in tracks:
        boxes = np.asarray(track, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"each track must have shape (rows, 4), got {boxes.shape}")
        normalized = boxes / scale
        for start in range(0, max(0, len(normalized) - sequence_length), stride):
            sequences.append(normalized[start : start + sequence_length])
            targets.append(normalized[start + sequence_length])

    if not sequences:
        return (
            np.empty((0, sequence_length, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
    return np.stack(sequences), np.stack(targets)
