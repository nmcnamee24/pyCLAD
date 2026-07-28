"""NOLA continual-stage and prepared-test dataset adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from pyclad.video.data import VideoFeatureConcept, VideoStrategySchema, VideoWindow
from pyclad.video.data.precomputed import PrecomputedVideoDataset
from pyclad.video.features.store import InMemoryVideoFeatureStore
from pyclad.video.models.nola.features import NolaFeatureLayout
from pyclad.video.models.nola.scoring import non_maximum_suppression

NOLA_STAGE_ORDER = ("M-Train", *(f"Train{index}" for index in range(10)))
NOLA_RELEVANT_CLASSES = ("car", "bike", "truck", "cart")


@dataclass(frozen=True)
class NolaGroundTruth:
    """NOLA test intervals using zero-based, half-open frame ranges."""

    intervals: Mapping[str, Tuple[Tuple[int, int], ...]]

    def labels(self, video_id: str, frame_count: int) -> np.ndarray:
        labels = np.zeros(frame_count, dtype=np.int64)
        for start, stop in self.intervals.get(video_id, ()):
            labels[max(0, start) : min(stop, frame_count)] = 1
        return labels


class NolaContinualDataset:
    """Build regular pyCLAD concepts from NOLA's eleven nominal stages."""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        frame_stride: int = 30,
        confidence_threshold: float = 0.6,
        stage_order: Sequence[str] = NOLA_STAGE_ORDER,
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        self.root = Path(root).expanduser().resolve()
        self.train_root = self.root / "Train"
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not self.train_root.is_dir():
            raise FileNotFoundError(f"NOLA training directory does not exist: {self.train_root}")

        self.frame_stride = int(frame_stride)
        self.confidence_threshold = float(confidence_threshold)
        self.stage_order = tuple(stage_order)
        self.frame_size = tuple(map(int, frame_size))
        self.layout = NolaFeatureLayout(
            spatial_columns=(0, 1, 2, 3, 4),
            temporal_columns=(5, 6, 7),
            trajectory_error_column=None,
        )
        self.strategy_schema = VideoStrategySchema(feature_dim=self.layout.feature_dim)

        missing = [stage for stage in self.stage_order if not (self.train_root / stage).is_dir()]
        if missing:
            raise FileNotFoundError(f"NOLA training stages are missing: {missing}")

    def training_concepts(
        self,
        *,
        stages: Optional[Sequence[str]] = None,
        max_videos_per_stage: Optional[int] = None,
        max_frames_per_video: Optional[int] = None,
    ) -> Tuple[VideoFeatureConcept, ...]:
        """Load stage concepts in a deterministic continual order."""

        if max_videos_per_stage is not None and max_videos_per_stage <= 0:
            raise ValueError("max_videos_per_stage must be positive")
        if max_frames_per_video is not None and max_frames_per_video <= 0:
            raise ValueError("max_frames_per_video must be positive")
        selected = self.stage_order if stages is None else tuple(stages)
        unknown = sorted(set(selected) - set(self.stage_order))
        if unknown:
            raise ValueError(f"unknown NOLA stages: {unknown}")

        concepts = []
        for stage in selected:
            video_directories = sorted(path for path in (self.train_root / stage).iterdir() if path.is_dir())
            if max_videos_per_stage is not None:
                video_directories = video_directories[:max_videos_per_stage]
            features = []
            windows = []
            feature_offset = 0
            for video_directory in video_directories:
                video_features, video_windows = extract_nola_video_features(
                    video_directory,
                    split="train",
                    concept_id=stage,
                    feature_offset=feature_offset,
                    frame_stride=self.frame_stride,
                    max_frames=max_frames_per_video,
                    confidence_threshold=self.confidence_threshold,
                    frame_size=self.frame_size,
                )
                if len(video_features) == 0:
                    continue
                features.append(video_features)
                windows.extend(video_windows)
                feature_offset += len(video_features)
            if not features:
                raise ValueError(f"NOLA stage {stage!r} produced no model-ready rows")
            matrix = np.concatenate(features, axis=0).astype(np.float32, copy=False)
            concepts.append(
                VideoFeatureConcept(
                    name=stage,
                    features=matrix,
                    windows=tuple(windows),
                    strategy_schema=self.strategy_schema,
                )
            )
        return tuple(concepts)


class NolaPreparedTestDataset(PrecomputedVideoDataset):
    """NOLA test videos after object detection and tracking preprocessing."""

    def __init__(
        self,
        processed_root: Union[str, Path],
        ground_truth_path: Union[str, Path],
        *,
        source_test_root: Optional[Union[str, Path]] = None,
        video_ids: Optional[Iterable[str]] = None,
        frame_stride: int = 1,
        confidence_threshold: float = 0.6,
        frame_size: Tuple[int, int] = (1280, 720),
        default_frame_count: int = 9000,
        dataset_name: str = "NOLA-Prepared-Test",
    ):
        processed_path = Path(processed_root).expanduser().resolve()
        if not processed_path.is_dir():
            raise FileNotFoundError(f"NOLA processed test directory does not exist: {processed_path}")
        source_path = None if source_test_root is None else Path(source_test_root).expanduser().resolve()
        selected_ids = None if video_ids is None else set(video_ids)
        ground_truth = load_nola_ground_truth(ground_truth_path)

        directories = sorted(path for path in processed_path.iterdir() if path.is_dir())
        if selected_ids is not None:
            directories = [path for path in directories if path.name in selected_ids]
        if not directories:
            raise ValueError("no prepared NOLA test videos were selected")

        features = []
        windows = []
        frame_labels: Dict[str, np.ndarray] = {}
        feature_offset = 0
        for video_directory in directories:
            video_id = video_directory.name
            frame_count = _nola_frame_count(
                video_directory,
                None if source_path is None else source_path / video_id / "video.mp4",
                default=default_frame_count,
            )
            labels = ground_truth.labels(video_id, frame_count)
            video_features, video_windows = extract_nola_video_features(
                video_directory,
                split="test",
                feature_offset=feature_offset,
                frame_stride=frame_stride,
                confidence_threshold=confidence_threshold,
                frame_size=frame_size,
            )
            if len(video_features) == 0:
                continue
            features.append(video_features)
            windows.extend(
                VideoWindow(
                    video_id=window.video_id,
                    split=window.split,
                    feature_index=window.feature_index,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    label=int(labels[window.start_frame]),
                    anomaly_class="anomaly" if video_id in ground_truth.intervals else None,
                    payload=window.payload,
                )
                for window in video_windows
            )
            frame_labels[video_id] = labels
            feature_offset += len(video_features)
        if not features:
            raise ValueError("prepared NOLA test videos produced no model-ready rows")

        self.layout = NolaFeatureLayout(
            spatial_columns=(0, 1, 2, 3, 4),
            temporal_columns=(5, 6, 7),
            trajectory_error_column=None,
        )
        self.ground_truth = ground_truth
        self.anomaly_intervals = {
            video_id: ground_truth.intervals.get(video_id, ())
            for video_id in frame_labels
        }
        super().__init__(
            dataset_name=dataset_name,
            feature_store=InMemoryVideoFeatureStore(np.concatenate(features, axis=0)),
            windows=tuple(windows),
            frame_labels_by_split={"test": frame_labels},
        )


def extract_nola_video_features(
    video_directory: Union[str, Path],
    *,
    split: str,
    concept_id: Optional[str] = None,
    feature_offset: int = 0,
    frame_stride: int = 30,
    max_frames: Optional[int] = None,
    confidence_threshold: float = 0.6,
    relevant_classes: Sequence[str] = NOLA_RELEVANT_CLASSES,
    frame_size: Tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, Tuple[VideoWindow, ...]]:
    """Convert one NOLA JSON/tracks directory into aligned frame features."""

    directory = Path(video_directory).expanduser().resolve()
    tracks_path = directory / "tracks.npy"
    annotation_path = directory / f"{directory.name}.json"
    if not annotation_path.exists():
        candidates = sorted(directory.glob("*.json"))
        candidates = [path for path in candidates if path.name != "metadata.json"]
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one NOLA annotation JSON in {directory}")
        annotation_path = candidates[0]
    if not tracks_path.exists():
        raise FileNotFoundError(f"NOLA tracks file does not exist: {tracks_path}")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    with annotation_path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    if not isinstance(annotations, list):
        raise ValueError(f"NOLA annotations must be a list: {annotation_path}")
    tracks = np.load(tracks_path, allow_pickle=True)
    if tracks.ndim != 2 or tracks.shape[1] < 4:
        raise ValueError(f"NOLA tracks must have shape (rows, at least 4), got {tracks.shape}")

    relevant = tuple(relevant_classes)
    class_to_index = {name: index for index, name in enumerate(relevant)}
    tracks_by_frame: Dict[int, list[tuple[str, np.ndarray]]] = {}
    for row in tracks:
        name = str(row[2])
        if name not in class_to_index:
            continue
        frame_id = int(row[0])
        box = np.asarray(row[3], dtype=np.float32).reshape(-1)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        tracks_by_frame.setdefault(frame_id, []).append((name, box))

    width, height = map(float, frame_size)
    hour = _nola_hour(directory.name)
    rows = []
    windows = []
    selected_count = 0
    for annotation in annotations:
        frame_id = int(annotation["frame_id"])
        if (frame_id - 1) % frame_stride:
            continue
        if max_frames is not None and selected_count >= max_frames:
            break

        objects = annotation.get("objects", ())
        vehicle_count = 0
        person_count = 0
        for detected in objects:
            confidence = float(detected.get("confidence", 0.0))
            if confidence <= confidence_threshold:
                continue
            name = str(detected.get("name", ""))
            if name == "person":
                person_count += 1
            elif name in class_to_index:
                vehicle_count += 1

        candidates = tracks_by_frame.get(frame_id, ())
        if candidates:
            boxes = np.stack([box for _, box in candidates])
            kept = non_maximum_suppression(boxes, overlap_threshold=0.7)
            boxes = boxes[kept]
            names = [candidates[index][0] for index in kept]
            areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
            selected_index = int(np.argmax(areas))
            spatial = np.concatenate(
                [
                    boxes[selected_index],
                    np.asarray([class_to_index[names[selected_index]]], dtype=np.float32),
                ]
            )
        else:
            spatial = _spatial_from_annotations(
                objects,
                class_to_index=class_to_index,
                confidence_threshold=confidence_threshold,
                frame_size=(width, height),
            )

        temporal = np.asarray([vehicle_count, person_count, hour], dtype=np.float32)
        rows.append(np.concatenate([spatial, temporal]).astype(np.float32, copy=False))
        windows.append(
            VideoWindow(
                video_id=directory.name,
                split=split,
                feature_index=feature_offset + selected_count,
                start_frame=frame_id - 1,
                end_frame=frame_id - 1,
                concept_id=concept_id,
                timestamp=(frame_id - 1) / 30.0,
                payload={"source_directory": str(directory), "frame_id": frame_id},
            )
        )
        selected_count += 1

    matrix = np.stack(rows) if rows else np.empty((0, 8), dtype=np.float32)
    return matrix.astype(np.float32, copy=False), tuple(windows)


def load_nola_ground_truth(path: Union[str, Path]) -> NolaGroundTruth:
    """Parse NOLA's comma-separated anomaly interval file."""

    ground_truth_path = Path(path).expanduser().resolve()
    intervals: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    with ground_truth_path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            fields = [field.strip() for field in raw_line.split(",")]
            if not fields or not fields[0]:
                continue
            values = [int(value) for value in fields[1:] if value and value != "-1"]
            if len(values) % 2:
                raise ValueError(f"NOLA ground-truth row has an unpaired boundary: {raw_line.rstrip()!r}")
            intervals[fields[0]] = tuple(
                (max(0, values[index] - 1), values[index + 1])
                for index in range(0, len(values), 2)
            )
    return NolaGroundTruth(intervals=intervals)


def _spatial_from_annotations(
    objects: Sequence[Mapping[str, object]],
    *,
    class_to_index: Mapping[str, int],
    confidence_threshold: float,
    frame_size: Tuple[float, float],
) -> np.ndarray:
    width, height = frame_size
    candidates = []
    for detected in objects:
        name = str(detected.get("name", ""))
        confidence = float(detected.get("confidence", 0.0))
        if name not in class_to_index or confidence <= confidence_threshold:
            continue
        coordinates = detected.get("relative_coordinates", {})
        if not isinstance(coordinates, Mapping):
            continue
        center_x = float(coordinates.get("center_x", 0.0)) * width
        center_y = float(coordinates.get("center_y", 0.0)) * height
        box_width = float(coordinates.get("width", 0.0)) * width
        box_height = float(coordinates.get("height", 0.0)) * height
        box = np.asarray(
            [
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            ],
            dtype=np.float32,
        )
        candidates.append((box_width * box_height, name, box))
    if not candidates:
        return np.asarray([0.0, 0.0, 0.0, 0.0, len(class_to_index)], dtype=np.float32)
    _, name, box = max(candidates, key=lambda item: item[0])
    return np.concatenate([box, np.asarray([class_to_index[name]], dtype=np.float32)])


def _nola_hour(video_id: str) -> float:
    fields = video_id.split("_")
    if len(fields) < 2:
        raise ValueError(f"NOLA video id does not encode an hour: {video_id!r}")
    return float(fields[1])


def _nola_frame_count(
    processed_directory: Path,
    source_video: Optional[Path],
    *,
    default: int,
) -> int:
    metadata_path = processed_directory / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        frame_count = int(metadata.get("source_frame_count", 0))
        if frame_count > 0:
            return frame_count
    if source_video is not None and source_video.exists():
        try:
            import cv2

            capture = cv2.VideoCapture(str(source_video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frame_count > 0:
                return frame_count
        except ImportError:
            pass
    if default <= 0:
        raise ValueError("default_frame_count must be positive")
    return int(default)
