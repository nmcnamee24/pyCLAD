"""Paper-faithful NOLA feature datasets with context and trajectory cues."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from pyclad.video.data import VideoFeatureConcept, VideoStrategySchema, VideoWindow
from pyclad.video.data.precomputed import PrecomputedVideoDataset
from pyclad.video.datasets.nola import (
    NOLA_RELEVANT_CLASSES,
    NOLA_STAGE_ORDER,
    _nola_frame_count,
    load_nola_ground_truth,
)
from pyclad.video.features.store import InMemoryVideoFeatureStore
from pyclad.video.models.nola.features import canonical_nola_object_name

DARKNET_COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

NOLA_PAPER_FEATURE_DIM = 5 + len(DARKNET_COCO_CLASSES) + 1 + 1 + 4 + 1

_COCO_TO_DARKNET_ALIASES = {
    "airplane": "aeroplane",
    "couch": "sofa",
    "dining table": "diningtable",
    "hair dryer": "hair drier",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
}


class NolaPaperContinualDataset:
    """NOLA's eleven nominal splits using the paper's contextual features."""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        processed_train_root: Optional[Union[str, Path]] = None,
        trajectory_predictor=None,
        frame_stride: int = 30,
        confidence_threshold: float = 0.6,
        stage_order: Sequence[str] = NOLA_STAGE_ORDER,
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        self.root = Path(root).expanduser().resolve()
        self.train_root = self.root / "Train"
        if not self.train_root.is_dir():
            raise FileNotFoundError(f"NOLA training directory does not exist: {self.train_root}")
        self.processed_train_root = (
            None if processed_train_root is None else Path(processed_train_root).expanduser().resolve()
        )
        if self.processed_train_root is not None and not self.processed_train_root.is_dir():
            raise FileNotFoundError(
                "NOLA processed training override directory does not exist: " f"{self.processed_train_root}"
            )
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        self.trajectory_predictor = trajectory_predictor
        self.frame_stride = int(frame_stride)
        self.confidence_threshold = float(confidence_threshold)
        self.stage_order = tuple(stage_order)
        self.frame_size = tuple(map(int, frame_size))
        self.strategy_schema = VideoStrategySchema(feature_dim=NOLA_PAPER_FEATURE_DIM)
        missing = [stage for stage in self.stage_order if not (self.train_root / stage).is_dir()]
        if missing:
            raise FileNotFoundError(f"NOLA training stages are missing: {missing}")

    @property
    def feature_dim(self) -> int:
        return NOLA_PAPER_FEATURE_DIM

    def video_directories(
        self,
        stages: Optional[Sequence[str]] = None,
        max_videos_per_stage: Optional[int] = None,
    ) -> Tuple[Path, ...]:
        selected = self.stage_order if stages is None else tuple(stages)
        unknown = sorted(set(selected) - set(self.stage_order))
        if unknown:
            raise ValueError(f"unknown NOLA stages: {unknown}")
        directories = []
        for stage in selected:
            stage_directories = sorted(path for path in (self.train_root / stage).iterdir() if path.is_dir())
            if max_videos_per_stage is not None:
                stage_directories = stage_directories[:max_videos_per_stage]
            directories.extend(self._cache_directory(stage, path) for path in stage_directories)
        return tuple(directories)

    def training_concepts(
        self,
        *,
        stages: Optional[Sequence[str]] = None,
        max_videos_per_stage: Optional[int] = None,
        max_frames_per_video: Optional[int] = None,
    ) -> Tuple[VideoFeatureConcept, ...]:
        selected = self.stage_order if stages is None else tuple(stages)
        concepts = []
        for stage in selected:
            directories = sorted(path for path in (self.train_root / stage).iterdir() if path.is_dir())
            if max_videos_per_stage is not None:
                directories = directories[:max_videos_per_stage]
            matrices = []
            windows = []
            offset = 0
            for source_directory in directories:
                directory = self._cache_directory(stage, source_directory)
                matrix, video_windows = extract_nola_paper_video_features(
                    directory,
                    split="train",
                    concept_id=stage,
                    feature_offset=offset,
                    frame_stride=self.frame_stride,
                    max_frames=max_frames_per_video,
                    confidence_threshold=self.confidence_threshold,
                    frame_size=self.frame_size,
                    trajectory_predictor=self.trajectory_predictor,
                )
                if not len(matrix):
                    continue
                matrices.append(matrix)
                windows.extend(video_windows)
                offset += len(matrix)
            if not matrices:
                raise ValueError(f"NOLA paper stage {stage!r} produced no feature rows")
            concepts.append(
                VideoFeatureConcept(
                    name=stage,
                    features=np.concatenate(matrices).astype(np.float32, copy=False),
                    windows=tuple(windows),
                    strategy_schema=self.strategy_schema,
                )
            )
        return tuple(concepts)

    def _cache_directory(self, stage: str, source_directory: Path) -> Path:
        if self.processed_train_root is None:
            return source_directory
        override = self.processed_train_root / stage / source_directory.name
        return override if override.is_dir() else source_directory


class NolaPaperPreparedTestDataset(PrecomputedVideoDataset):
    """Prepared NOLA test videos encoded with the paper feature schema."""

    def __init__(
        self,
        processed_root: Union[str, Path],
        ground_truth_path: Union[str, Path],
        *,
        source_test_root: Optional[Union[str, Path]] = None,
        video_ids: Optional[Iterable[str]] = None,
        trajectory_predictor=None,
        confidence_threshold: float = 0.6,
        frame_size: Tuple[int, int] = (1280, 720),
        default_frame_count: int = 9_000,
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
            raise ValueError("no prepared NOLA paper test videos were selected")

        matrices = []
        windows = []
        frame_labels: Dict[str, np.ndarray] = {}
        offset = 0
        for directory in directories:
            video_id = directory.name
            frame_count = min(
                9_000,
                _nola_frame_count(
                    directory,
                    None if source_path is None else source_path / video_id / "video.mp4",
                    default=default_frame_count,
                ),
            )
            labels = ground_truth.labels(video_id, frame_count)
            matrix, video_windows = extract_nola_paper_video_features(
                directory,
                split="test",
                feature_offset=offset,
                frame_stride=1,
                max_frames=frame_count,
                confidence_threshold=confidence_threshold,
                frame_size=frame_size,
                trajectory_predictor=trajectory_predictor,
            )
            kept_windows = tuple(window for window in video_windows if window.start_frame < frame_count)
            matrix = matrix[: len(kept_windows)]
            if not len(matrix):
                continue
            matrices.append(matrix)
            windows.extend(
                VideoWindow(
                    video_id=window.video_id,
                    split=window.split,
                    feature_index=offset + index,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    label=int(labels[window.start_frame]),
                    anomaly_class="anomaly" if video_id in ground_truth.intervals else None,
                    payload=window.payload,
                )
                for index, window in enumerate(kept_windows)
            )
            frame_labels[video_id] = labels
            offset += len(matrix)
        if not matrices:
            raise ValueError("prepared NOLA paper test videos produced no feature rows")

        self.ground_truth = ground_truth
        self.anomaly_intervals = {video_id: ground_truth.intervals.get(video_id, ()) for video_id in frame_labels}
        self.strategy_schema = VideoStrategySchema(feature_dim=NOLA_PAPER_FEATURE_DIM)
        super().__init__(
            dataset_name="NOLA-Paper-Prepared-Test",
            feature_store=InMemoryVideoFeatureStore(np.concatenate(matrices)),
            windows=tuple(windows),
            frame_labels_by_split={"test": frame_labels},
        )


def extract_nola_paper_video_features(
    video_directory: Union[str, Path],
    *,
    split: str,
    concept_id: Optional[str] = None,
    feature_offset: int = 0,
    frame_stride: int = 30,
    max_frames: Optional[int] = None,
    confidence_threshold: float = 0.6,
    frame_size: Tuple[int, int] = (1280, 720),
    trajectory_predictor=None,
) -> tuple[np.ndarray, Tuple[VideoWindow, ...]]:
    """Build object, context, and path-error features described in the paper."""

    directory = Path(video_directory).expanduser().resolve()
    annotation_path = directory / f"{directory.name}.json"
    tracks_path = directory / "tracks.npy"
    if not annotation_path.exists() or not tracks_path.exists():
        raise FileNotFoundError(f"NOLA paper cache is incomplete: {directory}")
    with annotation_path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    tracks = np.load(tracks_path, allow_pickle=True)

    tracks_by_frame: Dict[int, list[tuple[str, np.ndarray]]] = {}
    for row in tracks:
        frame_id = int(row[0])
        name = canonical_nola_object_name(row[2])
        box = np.asarray(row[3], dtype=np.float32).reshape(-1)
        if box.shape == (4,) and np.isfinite(box).all():
            tracks_by_frame.setdefault(frame_id, []).append((name, box))

    trajectory_errors = _trajectory_errors_by_frame(
        tracks,
        trajectory_predictor,
        frame_size=frame_size,
    )
    class_to_index = {name: index for index, name in enumerate(DARKNET_COCO_CLASSES)}
    relevant_to_index = {name: index for index, name in enumerate(NOLA_RELEVANT_CLASSES)}
    weekday, time_category = _nola_context(directory.name)

    rows = []
    windows = []
    selected = 0
    for annotation in annotations:
        frame_id = int(annotation["frame_id"])
        if (frame_id - 1) % frame_stride:
            continue
        if max_frames is not None and selected >= max_frames:
            break

        counts = np.zeros(len(DARKNET_COCO_CLASSES), dtype=np.float32)
        for detected in annotation.get("objects", ()):
            if float(detected.get("confidence", 0.0)) <= confidence_threshold:
                continue
            name = _canonical_darknet_coco_name(detected.get("name", ""))
            if name in class_to_index:
                counts[class_to_index[name]] += 1.0

        relevant = [(name, box) for name, box in tracks_by_frame.get(frame_id, ()) if name in relevant_to_index]
        if relevant:
            name, box = max(
                relevant,
                key=lambda item: max(0.0, float(item[1][2] - item[1][0])) * max(0.0, float(item[1][3] - item[1][1])),
            )
            spatial = np.concatenate([box, np.asarray([relevant_to_index[name]], dtype=np.float32)])
        else:
            spatial = np.asarray([0.0, 0.0, 0.0, 0.0, len(relevant_to_index)], dtype=np.float32)

        context = np.concatenate(
            [
                np.asarray([np.count_nonzero(counts), weekday], dtype=np.float32),
                np.eye(4, dtype=np.float32)[time_category],
            ]
        )
        trajectory_error = np.asarray([trajectory_errors.get(frame_id, 0.0)], dtype=np.float32)
        row = np.concatenate([spatial, counts, context, trajectory_error])
        if len(row) != NOLA_PAPER_FEATURE_DIM:
            raise RuntimeError(f"unexpected NOLA paper feature width: {len(row)}")
        rows.append(row)
        windows.append(
            VideoWindow(
                video_id=directory.name,
                split=split,
                feature_index=feature_offset + selected,
                start_frame=frame_id - 1,
                end_frame=frame_id - 1,
                concept_id=concept_id,
                timestamp=(frame_id - 1) / 30.0,
                payload={"source_directory": str(directory), "frame_id": frame_id},
            )
        )
        selected += 1

    matrix = np.stack(rows) if rows else np.empty((0, NOLA_PAPER_FEATURE_DIM), dtype=np.float32)
    return matrix.astype(np.float32, copy=False), tuple(windows)


def build_nola_paper_trajectory_training_data(
    video_directories: Sequence[Union[str, Path]],
    *,
    sequence_length: int = 20,
    stride: int = 5,
    minimum_track_length: int = 50,
    frame_size: Tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, np.ndarray]:
    """Collect the paper's 20-step next-box training examples."""

    sequences = []
    targets = []
    scale = np.asarray([frame_size[0], frame_size[1], frame_size[0], frame_size[1]], dtype=np.float32)
    for raw_directory in video_directories:
        tracks = np.load(Path(raw_directory) / "tracks.npy", allow_pickle=True)
        for rows in _group_track_rows(tracks).values():
            if len(rows) < minimum_track_length:
                continue
            boxes = np.stack([row[1] for row in rows]).astype(np.float32) / scale
            for start in range(0, len(boxes) - sequence_length, stride):
                sequences.append(boxes[start : start + sequence_length])
                targets.append(boxes[start + sequence_length])
    if not sequences:
        return (
            np.empty((0, sequence_length, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
    return np.stack(sequences).astype(np.float32), np.stack(targets).astype(np.float32)


def _trajectory_errors_by_frame(
    tracks: np.ndarray,
    predictor,
    *,
    sequence_length: int = 20,
    minimum_track_length: int = 50,
    frame_size: Tuple[int, int],
) -> Dict[int, float]:
    if predictor is None:
        return {}
    scale = np.asarray([frame_size[0], frame_size[1], frame_size[0], frame_size[1]], dtype=np.float32)
    sequences = []
    targets = []
    frame_ids = []
    for rows in _group_track_rows(tracks).values():
        if len(rows) < minimum_track_length:
            continue
        boxes = np.stack([row[1] for row in rows]).astype(np.float32) / scale
        for start in range(0, len(boxes) - sequence_length):
            sequences.append(boxes[start : start + sequence_length])
            targets.append(boxes[start + sequence_length])
            frame_ids.append(rows[start + sequence_length][0])
    if not sequences:
        return {}
    errors = predictor.errors(np.stack(sequences).astype(np.float32), np.stack(targets).astype(np.float32))
    by_frame: Dict[int, float] = {}
    for frame_id, error in zip(frame_ids, errors):
        by_frame[frame_id] = max(by_frame.get(frame_id, 0.0), float(error))
    return by_frame


def _group_track_rows(tracks: np.ndarray) -> Dict[str, list[tuple[int, np.ndarray]]]:
    grouped: Dict[str, list[tuple[int, np.ndarray]]] = {}
    for row in tracks:
        box = np.asarray(row[3], dtype=np.float32).reshape(-1)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        grouped.setdefault(str(row[1]), []).append((int(row[0]), box))
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return grouped


def _nola_context(video_id: str) -> tuple[float, int]:
    fields = video_id.lower().split("_")
    if len(fields) < 2:
        raise ValueError(f"NOLA video id does not encode day and hour: {video_id!r}")
    weekday = 1.0 if fields[0] in {"mon", "tue", "wed", "thu", "fri"} else 0.0
    hour = int(fields[1])
    if not 0 <= hour <= 23:
        raise ValueError(f"NOLA video id has an invalid hour: {video_id!r}")
    return weekday, min(hour // 6, 3)


def _canonical_darknet_coco_name(name: object) -> str:
    normalized = str(name).strip().lower()
    return _COCO_TO_DARKNET_ALIASES.get(normalized, normalized)
