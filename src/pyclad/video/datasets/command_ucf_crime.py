"""COMMAND's RGB/flow UCF-Crime feature archive.

The reference archive stores one 32-window RGB matrix and one 32-window flow
matrix per video.  This adapter concatenates the two streams, keeps weak
video-level labels and bag identifiers in ``VideoStrategySchema`` columns, and
retains frame ranges as ``VideoWindow`` sidecars.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.data.sample import VideoWindow
from pyclad.video.data.video_concept import VideoConcept

COMMAND_UCF_CRIME_CONCEPT_ORDER = (
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
)

# PyCLAD's recreation of the three continual tasks reported in COMMAND Section
# IV-C. The existing class-by-class adapter remains unchanged; recreation runs
# use this explicit grouping through ``paper_training_tasks``.
COMMAND_UCF_CRIME_PAPER_TASKS = (
    ("Abuse", "Arrest", "Arson", "Assault"),
    ("Burglary", "Explosion", "Fighting", "RoadAccidents"),
    ("Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"),
)


@dataclass(frozen=True)
class CommandUcfCrimeRecord:
    """One video entry from COMMAND's UCF-Crime split files."""

    relative_path: str
    weak_label: int
    anomaly_class: Optional[str]
    frame_count: Optional[int] = None
    anomaly_intervals: Tuple[Tuple[int, int], ...] = ()

    @property
    def video_id(self) -> str:
        return self.relative_path


class CommandUcfCrimeDataset:
    """Load the exact UCF-Crime feature layout published with COMMAND.

    Training concepts are balanced continual experiences: each anomaly class
    is paired with a disjoint, equally sized shard of the normal training
    videos. The COMMAND release manifest contains 810 anomalous rows and 810
    normal rows. Ten normal paths are repeated, so this represents the official
    800 unique normal and 810 unique anomalous UCF-Crime training videos while
    retaining the paper-release row contract.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        modality: str = "two",
        concept_order: Sequence[str] = COMMAND_UCF_CRIME_CONCEPT_ORDER,
        dataset_name: str = "COMMAND-UCF-Crime",
    ):
        self.root = Path(root).expanduser().resolve()
        self.modality = modality.lower()
        if self.modality not in {"two", "rgb", "flow"}:
            raise ValueError("modality must be one of: 'two', 'rgb', 'flow'")
        self._dataset_name = dataset_name
        self.concept_order = tuple(concept_order)
        if len(set(self.concept_order)) != len(self.concept_order):
            raise ValueError("concept_order must not contain duplicates")

        self._validate_layout()
        self._normal_train = self._read_train_records("train_normal.txt", weak_label=0)
        self._anomaly_train = self._read_train_records("train_anomaly.txt", weak_label=1)
        self._test_records = (
            *self._read_normal_test_records(),
            *self._read_anomaly_test_records(),
        )
        self._training_record_instance_ids = {
            id(record): f"normal:{index:04d}:{record.relative_path}" for index, record in enumerate(self._normal_train)
        }
        self._training_record_instance_ids.update(
            {
                id(record): f"anomaly:{index:04d}:{record.relative_path}"
                for index, record in enumerate(self._anomaly_train)
            }
        )
        self._feature_dim = 2048 if self.modality == "two" else 1024
        self.strategy_schema = VideoStrategySchema(
            feature_dim=self._feature_dim,
            target_names=("weak_label", "bag_id"),
        )
        # The COMMAND release manifest has 1,620 training rows but only 1,610
        # unique paths. The official UCF-Crime split itself contains 1,610
        # training videos. Bag identity belongs to the COMMAND manifest row so
        # duplicate release rows do not collapse during recreation runs.
        self._bag_ids = {
            id(record): float(index) for index, record in enumerate((*self._normal_train, *self._anomaly_train))
        }
        unknown = sorted({record.anomaly_class for record in self._anomaly_train} - set(self.concept_order))
        if unknown:
            raise ValueError(f"concept_order is missing anomaly classes: {unknown}")

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def training_records(self) -> Tuple[CommandUcfCrimeRecord, ...]:
        """Released training rows in normal-then-anomaly order.

        Records are intentionally not deduplicated. The COMMAND release
        manifest repeats ten video paths, and each row retains a distinct bag
        identifier for recreation runs. These are not additional videos in the
        official UCF-Crime benchmark split.
        """

        return (*self._normal_train, *self._anomaly_train)

    @property
    def test_records(self) -> Tuple[CommandUcfCrimeRecord, ...]:
        """Released test rows in normal-then-anomaly order."""

        return tuple(self._test_records)

    def name(self) -> str:
        return self._dataset_name

    def frame_labels(self, split: str = "test") -> Dict[str, np.ndarray]:
        if split != "test":
            raise KeyError("frame labels are available only for split='test'")
        return {record.video_id: self._frame_labels_for_record(record) for record in self._test_records}

    def training_concepts(
        self,
        *,
        concepts: Optional[Sequence[str]] = None,
        max_videos_per_class: Optional[int] = None,
    ) -> Tuple[VideoConcept, ...]:
        """Return balanced anomaly-class experiences in continual order."""

        if max_videos_per_class is not None and max_videos_per_class <= 0:
            raise ValueError("max_videos_per_class must be positive")
        selected = self.concept_order if concepts is None else tuple(concepts)
        unknown = sorted(set(selected) - set(self.concept_order))
        if unknown:
            raise ValueError(f"unknown COMMAND concepts: {unknown}")

        anomaly_by_class = {
            concept: [record for record in self._anomaly_train if record.anomaly_class == concept]
            for concept in self.concept_order
        }
        normal_offset = 0
        normal_by_class = {}
        for concept in self.concept_order:
            count = len(anomaly_by_class[concept])
            normal_by_class[concept] = self._normal_train[normal_offset : normal_offset + count]
            normal_offset += count
        if normal_offset != len(self._normal_train):
            raise ValueError(
                "normal/anomaly training counts do not balance across concepts: "
                f"used {normal_offset} of {len(self._normal_train)} normal videos"
            )

        result = []
        for concept in selected:
            anomalies = anomaly_by_class[concept]
            normals = normal_by_class[concept]
            if max_videos_per_class is not None:
                anomalies = anomalies[:max_videos_per_class]
                normals = normals[:max_videos_per_class]
            result.append(self._training_concept(concept, (*anomalies, *normals)))
        return tuple(result)

    def paper_training_tasks(
        self,
        *,
        max_videos_per_class: Optional[int] = None,
    ) -> Tuple[VideoConcept, ...]:
        """Return PyCLAD's recreation of COMMAND's three-task protocol.

        Complete videos and globally unique bag identifiers are preserved. The
        paper trainer consumes these concepts directly and never converts them
        to row-oriented pyCLAD strategy matrices.
        """

        tasks = []
        for task_index, class_names in enumerate(COMMAND_UCF_CRIME_PAPER_TASKS, start=1):
            concepts = self.training_concepts(
                concepts=class_names,
                max_videos_per_class=max_videos_per_class,
            )
            tasks.append(
                VideoConcept.from_features(
                    name=f"T{task_index}",
                    features=np.concatenate([concept.features for concept in concepts]).astype(np.float32, copy=False),
                    windows=tuple(window for concept in concepts for window in concept.windows),
                    labels=np.concatenate([concept.labels for concept in concepts]),
                    strategy_schema=self.strategy_schema,
                    strategy_targets={
                        name: np.concatenate([concept.strategy_targets[name] for concept in concepts])
                        for name in self.strategy_schema.target_names
                    },
                )
            )
        return tuple(tasks)

    def test_concept(
        self,
        *,
        name: str = "test",
        video_ids: Optional[Iterable[str]] = None,
        max_normal_videos: Optional[int] = None,
        max_anomaly_videos: Optional[int] = None,
    ) -> VideoConcept:
        """Build a feature-only test concept, optionally restricted for smoke runs."""

        selected_ids = None if video_ids is None else set(video_ids)
        normal_count = 0
        anomaly_count = 0
        records = []
        for record in self._test_records:
            if selected_ids is not None and record.video_id not in selected_ids:
                continue
            if record.weak_label == 0:
                if max_normal_videos is not None and normal_count >= max_normal_videos:
                    continue
                normal_count += 1
            else:
                if max_anomaly_videos is not None and anomaly_count >= max_anomaly_videos:
                    continue
                anomaly_count += 1
            records.append(record)
        if not records:
            raise ValueError("no COMMAND test videos were selected")

        features, windows, _ = self._load_records(records, split="test")
        return VideoConcept.from_features(
            name=name,
            features=features,
            windows=windows,
            labels=np.asarray([window.label for window in windows], dtype=np.int64),
            strategy_schema=self.strategy_schema,
        )

    def _training_concept(
        self,
        concept: str,
        records: Sequence[CommandUcfCrimeRecord],
    ) -> VideoConcept:
        features, windows, row_records = self._load_records(records, split="train")
        weak_labels = np.asarray([record.weak_label for record in row_records], dtype=np.float32)
        bag_ids = np.asarray([self._bag_ids[id(record)] for record in row_records], dtype=np.float32)
        return VideoConcept.from_features(
            name=concept,
            features=features,
            windows=windows,
            labels=weak_labels.astype(np.int64),
            strategy_schema=self.strategy_schema,
            strategy_targets={
                "weak_label": weak_labels,
                "bag_id": bag_ids,
            },
        )

    def _load_records(
        self,
        records: Sequence[CommandUcfCrimeRecord],
        *,
        split: str,
    ) -> tuple[np.ndarray, Tuple[VideoWindow, ...], Tuple[CommandUcfCrimeRecord, ...]]:
        feature_blocks = []
        windows = []
        row_records = []
        feature_index = 0
        for record_index, record in enumerate(records):
            video_features = self._load_video_features(record)
            frame_ranges = self._frame_ranges(record, len(video_features))
            frame_labels = None if record.frame_count is None else self._frame_labels_for_record(record)
            for window_index, (start_frame, end_frame) in enumerate(frame_ranges):
                label = record.weak_label
                if frame_labels is not None:
                    label = int(np.any(frame_labels[start_frame : end_frame + 1]))
                windows.append(
                    VideoWindow(
                        video_id=record.video_id,
                        split=split,
                        feature_index=feature_index,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        label=label,
                        anomaly_class=record.anomaly_class,
                        payload={
                            "relative_path": record.relative_path,
                            "window_index": window_index,
                            "weak_label": record.weak_label,
                            "record_index": record_index,
                            **(
                                {"record_instance_id": self._training_record_instance_ids[id(record)]}
                                if split == "train"
                                else {}
                            ),
                        },
                    )
                )
                row_records.append(record)
                feature_index += 1
            feature_blocks.append(video_features)
        return (
            np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False),
            tuple(windows),
            tuple(row_records),
        )

    def _load_video_features(self, record: CommandUcfCrimeRecord) -> np.ndarray:
        rgb_path = self.root / "all_rgbs" / f"{record.relative_path}.npy"
        flow_path = self.root / "all_flows" / f"{record.relative_path}.npy"
        rgb = np.load(rgb_path, allow_pickle=False)
        flow = np.load(flow_path, allow_pickle=False)
        if rgb.shape != flow.shape or rgb.ndim != 2 or rgb.shape[1] != 1024:
            raise ValueError(
                f"COMMAND expects matching (windows, 1024) RGB/flow arrays for {record.relative_path}; "
                f"got RGB {rgb.shape}, flow {flow.shape}"
            )
        if self.modality == "rgb":
            return np.asarray(rgb, dtype=np.float32)
        if self.modality == "flow":
            return np.asarray(flow, dtype=np.float32)
        return np.concatenate([rgb, flow], axis=1).astype(np.float32, copy=False)

    def _frame_ranges(
        self,
        record: CommandUcfCrimeRecord,
        window_count: int,
    ) -> Tuple[Tuple[int, int], ...]:
        if record.frame_count is None:
            return tuple((index, index) for index in range(window_count))
        edges = np.rint(np.linspace(0, record.frame_count, window_count + 1)).astype(np.int64)
        return tuple(
            (int(edges[index]), max(int(edges[index]), int(edges[index + 1]) - 1)) for index in range(window_count)
        )

    @staticmethod
    def _frame_labels_for_record(record: CommandUcfCrimeRecord) -> np.ndarray:
        if record.frame_count is None:
            raise ValueError(f"frame count is missing for {record.video_id}")
        labels = np.zeros(record.frame_count, dtype=np.int64)
        for start, stop in record.anomaly_intervals:
            labels[max(0, start) : min(stop, record.frame_count)] = 1
        return labels

    def _read_train_records(
        self,
        filename: str,
        *,
        weak_label: int,
    ) -> Tuple[CommandUcfCrimeRecord, ...]:
        records = []
        with (self.root / filename).open(encoding="utf-8") as stream:
            for raw_line in stream:
                relative_path = raw_line.strip()
                if not relative_path:
                    continue
                anomaly_class = relative_path.split("/", 1)[0] if weak_label else None
                records.append(
                    CommandUcfCrimeRecord(
                        relative_path=relative_path,
                        weak_label=weak_label,
                        anomaly_class=anomaly_class,
                    )
                )
        return tuple(records)

    def _read_normal_test_records(self) -> Tuple[CommandUcfCrimeRecord, ...]:
        records = []
        with (self.root / "test_normalv2.txt").open(encoding="utf-8") as stream:
            for raw_line in stream:
                fields = raw_line.split()
                if not fields:
                    continue
                if len(fields) != 3:
                    raise ValueError(f"invalid COMMAND normal test row: {raw_line.rstrip()!r}")
                records.append(
                    CommandUcfCrimeRecord(
                        relative_path=fields[0],
                        weak_label=0,
                        anomaly_class=None,
                        frame_count=int(fields[1]),
                    )
                )
        return tuple(records)

    def _read_anomaly_test_records(self) -> Tuple[CommandUcfCrimeRecord, ...]:
        records = []
        with (self.root / "test_anomalyv2.txt").open(encoding="utf-8") as stream:
            for raw_line in stream:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                fields = raw_line.split("|")
                if len(fields) != 3:
                    raise ValueError(f"invalid COMMAND anomaly test row: {raw_line!r}")
                boundaries = tuple(int(value) for value in ast.literal_eval(fields[2]))
                if len(boundaries) % 2:
                    raise ValueError(f"anomaly boundaries must contain start/stop pairs: {raw_line!r}")
                intervals = tuple(
                    (max(0, boundaries[index] - 1), boundaries[index + 1]) for index in range(0, len(boundaries), 2)
                )
                anomaly_class = fields[0].split("/", 1)[0]
                records.append(
                    CommandUcfCrimeRecord(
                        relative_path=fields[0],
                        weak_label=1,
                        anomaly_class=anomaly_class,
                        frame_count=int(fields[1]),
                        anomaly_intervals=intervals,
                    )
                )
        return tuple(records)

    def _validate_layout(self) -> None:
        required = (
            self.root / "all_rgbs",
            self.root / "all_flows",
            self.root / "train_normal.txt",
            self.root / "train_anomaly.txt",
            self.root / "test_normalv2.txt",
            self.root / "test_anomalyv2.txt",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"COMMAND UCF-Crime layout is incomplete; missing: {missing}")
