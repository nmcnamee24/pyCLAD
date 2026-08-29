"""Preflight auditing for COMMAND's UCF-Crime feature archive."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from pyclad.video.datasets.command_ucf_crime import (
    CommandUcfCrimeDataset,
    CommandUcfCrimeRecord,
)

OFFICIAL_TRAIN_VIDEOS = 1_610
OFFICIAL_TRAIN_NORMAL_VIDEOS = 800
OFFICIAL_TRAIN_ANOMALY_VIDEOS = 810
OFFICIAL_TEST_VIDEOS = 290
OFFICIAL_TEST_NORMAL_VIDEOS = 150
OFFICIAL_TEST_ANOMALY_VIDEOS = 140

COMMAND_RELEASE_TRAIN_RECORDS = 1_620
COMMAND_RELEASE_TRAIN_NORMAL_RECORDS = 810
COMMAND_RELEASE_TRAIN_ANOMALY_RECORDS = 810

# Compatibility names retained for the initial public package draft. They now
# describe the official UCF-Crime benchmark accurately rather than treating
# duplicate COMMAND manifest rows as additional dataset videos.
OFFICIAL_TRAIN_RECORDS = OFFICIAL_TRAIN_VIDEOS
OFFICIAL_UNIQUE_TRAIN_VIDEOS = OFFICIAL_TRAIN_VIDEOS


@dataclass(frozen=True)
class CommandUcfCrimeAudit:
    """Read-only readiness report for one local COMMAND UCF-Crime archive."""

    root: str
    modality: str
    train_normal_records: int
    train_anomaly_records: int
    train_unique_normal_videos: int
    train_unique_anomaly_videos: int
    train_unique_videos: int
    test_normal_videos: int
    test_anomaly_videos: int
    anomaly_train_by_class: Dict[str, int]
    anomaly_test_by_class: Dict[str, int]
    duplicate_training_records: Tuple[str, ...]
    train_test_overlaps: Tuple[str, ...]
    missing_feature_files: Tuple[str, ...]
    invalid_feature_arrays: Tuple[str, ...]
    checked_feature_arrays: bool
    feature_arrays_checked: int

    @property
    def train_records(self) -> int:
        return self.train_normal_records + self.train_anomaly_records

    @property
    def test_videos(self) -> int:
        return self.test_normal_videos + self.test_anomaly_videos

    @property
    def official_split_counts_match(self) -> bool:
        return (
            self.train_unique_normal_videos == OFFICIAL_TRAIN_NORMAL_VIDEOS
            and self.train_unique_anomaly_videos == OFFICIAL_TRAIN_ANOMALY_VIDEOS
            and self.train_unique_videos == OFFICIAL_TRAIN_VIDEOS
            and self.test_normal_videos == OFFICIAL_TEST_NORMAL_VIDEOS
            and self.test_anomaly_videos == OFFICIAL_TEST_ANOMALY_VIDEOS
        )

    @property
    def command_release_counts_match(self) -> bool:
        return (
            self.train_normal_records == COMMAND_RELEASE_TRAIN_NORMAL_RECORDS
            and self.train_anomaly_records == COMMAND_RELEASE_TRAIN_ANOMALY_RECORDS
            and self.train_records == COMMAND_RELEASE_TRAIN_RECORDS
            and self.official_split_counts_match
        )

    @property
    def ready(self) -> bool:
        return not (self.train_test_overlaps or self.missing_feature_files or self.invalid_feature_arrays)

    def as_dict(self) -> Dict[str, object]:
        return {
            "root": self.root,
            "modality": self.modality,
            "ready": self.ready,
            "official_split_counts_match": self.official_split_counts_match,
            "command_release_counts_match": self.command_release_counts_match,
            "provenance": {
                "dataset_split": "official UCF-Crime anomaly-detection split",
                "dataset_source": "https://www.crcv.ucf.edu/projects/real-world/",
                "continual_stream": "COMMAND paper recreation - UCF-Crime 4-4-5",
                "continual_source_doi": "10.1016/j.neucom.2026.132943",
                "manifest_relationship": (
                    "COMMAND release manifest; duplicate rows are retained for recreation "
                    "but are not additional official UCF-Crime videos"
                ),
            },
            "counts": {
                "train_records": self.train_records,
                "train_unique_videos": self.train_unique_videos,
                "train_normal_records": self.train_normal_records,
                "train_anomaly_records": self.train_anomaly_records,
                "train_unique_normal_videos": self.train_unique_normal_videos,
                "train_unique_anomaly_videos": self.train_unique_anomaly_videos,
                "test_videos": self.test_videos,
                "test_normal_videos": self.test_normal_videos,
                "test_anomaly_videos": self.test_anomaly_videos,
                "anomaly_train_by_class": dict(sorted(self.anomaly_train_by_class.items())),
                "anomaly_test_by_class": dict(sorted(self.anomaly_test_by_class.items())),
            },
            "duplicate_training_records": list(self.duplicate_training_records),
            "train_test_overlaps": list(self.train_test_overlaps),
            "features": {
                "checked_arrays": self.checked_feature_arrays,
                "arrays_checked": self.feature_arrays_checked,
                "missing_files": list(self.missing_feature_files),
                "invalid_arrays": list(self.invalid_feature_arrays),
            },
            "official_expected_counts": {
                "train_videos": OFFICIAL_TRAIN_VIDEOS,
                "train_normal_videos": OFFICIAL_TRAIN_NORMAL_VIDEOS,
                "train_anomaly_videos": OFFICIAL_TRAIN_ANOMALY_VIDEOS,
                "test_videos": OFFICIAL_TEST_VIDEOS,
                "test_normal_videos": OFFICIAL_TEST_NORMAL_VIDEOS,
                "test_anomaly_videos": OFFICIAL_TEST_ANOMALY_VIDEOS,
            },
            "command_release_expected_counts": {
                "train_records": COMMAND_RELEASE_TRAIN_RECORDS,
                "train_normal_records": COMMAND_RELEASE_TRAIN_NORMAL_RECORDS,
                "train_anomaly_records": COMMAND_RELEASE_TRAIN_ANOMALY_RECORDS,
                "train_unique_videos": OFFICIAL_TRAIN_VIDEOS,
            },
        }


def audit_command_ucf_crime(
    root: Union[str, Path],
    *,
    modality: str = "two",
    check_feature_arrays: bool = False,
) -> CommandUcfCrimeAudit:
    """Audit split composition, leakage, feature presence, and optional shapes.

    UCF-Crime's official anomaly-detection partition contains 1,610 unique
    training videos and 290 test videos. The COMMAND release manifest contains
    1,620 training rows because ten paths are repeated. Those rows are reported
    and retained for COMMAND recreation, but are not described as additional
    official dataset videos. Train/test identity overlap, missing feature
    files, and malformed arrays make the report not ready.
    """

    dataset = CommandUcfCrimeDataset(root, modality=modality)
    training = tuple(dataset.training_records)
    testing = tuple(dataset.test_records)
    training_paths = [record.relative_path for record in training]
    test_paths = [record.relative_path for record in testing]
    unique_normal_training_paths = {record.relative_path for record in training if record.weak_label == 0}
    unique_anomaly_training_paths = {record.relative_path for record in training if record.weak_label == 1}
    duplicates = tuple(sorted(path for path, count in Counter(training_paths).items() if count > 1))
    overlaps = tuple(sorted(set(training_paths) & set(test_paths)))
    unique_records = _unique_records((*training, *testing))

    streams = ("all_rgbs", "all_flows") if modality == "two" else (f"all_{modality}s",)
    missing = []
    invalid = []
    arrays_checked = 0
    feature_root = Path(dataset.root)
    for record in unique_records:
        shapes = []
        for stream in streams:
            path = feature_root / stream / f"{record.relative_path}.npy"
            if not path.is_file():
                missing.append(str(path))
                continue
            if not check_feature_arrays:
                continue
            arrays_checked += 1
            try:
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                shape = tuple(values.shape)
            except (OSError, ValueError) as error:
                invalid.append(f"{path}: {type(error).__name__}: {error}")
                continue
            if len(shape) != 2 or shape[0] <= 0 or shape[1] != 1024:
                invalid.append(f"{path}: expected (windows, 1024), got {shape}")
            shapes.append((stream, shape))
        if modality == "two" and len(shapes) == 2 and shapes[0][1] != shapes[1][1]:
            invalid.append(f"{record.relative_path}: RGB/flow shapes do not match: {shapes[0][1]} vs {shapes[1][1]}")

    anomaly_train = Counter(record.anomaly_class for record in training if record.weak_label == 1)
    anomaly_test = Counter(record.anomaly_class for record in testing if record.weak_label == 1)
    return CommandUcfCrimeAudit(
        root=str(feature_root),
        modality=modality,
        train_normal_records=sum(record.weak_label == 0 for record in training),
        train_anomaly_records=sum(record.weak_label == 1 for record in training),
        train_unique_normal_videos=len(unique_normal_training_paths),
        train_unique_anomaly_videos=len(unique_anomaly_training_paths),
        train_unique_videos=len(set(training_paths)),
        test_normal_videos=sum(record.weak_label == 0 for record in testing),
        test_anomaly_videos=sum(record.weak_label == 1 for record in testing),
        anomaly_train_by_class={str(name): count for name, count in anomaly_train.items()},
        anomaly_test_by_class={str(name): count for name, count in anomaly_test.items()},
        duplicate_training_records=duplicates,
        train_test_overlaps=overlaps,
        missing_feature_files=tuple(sorted(missing)),
        invalid_feature_arrays=tuple(sorted(invalid)),
        checked_feature_arrays=bool(check_feature_arrays),
        feature_arrays_checked=arrays_checked,
    )


def _unique_records(records: Sequence[CommandUcfCrimeRecord]) -> Tuple[CommandUcfCrimeRecord, ...]:
    by_path = {}
    for record in records:
        by_path.setdefault(record.relative_path, record)
    return tuple(by_path[path] for path in sorted(by_path))
