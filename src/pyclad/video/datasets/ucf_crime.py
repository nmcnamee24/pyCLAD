"""UCF-Crime adapters for precomputed window embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from pyclad.video.data.precomputed import PrecomputedVideoDataset
from pyclad.video.data.sample import VideoWindow
from pyclad.video.features.store import (
    InMemoryVideoFeatureStore,
    NpyVideoFeatureStore,
)

UCF_CRIME_WINDOW_MANIFEST_FIELDS = (
    "video_id",
    "split",
    "feature_index",
    "start_frame",
    "end_frame",
    "frame_labels_path",
    "label",
    "anomaly_class",
)
UCF_CRIME_I3D_FRAME_STEP = 16


class UcfCrimeSubsetDataset(PrecomputedVideoDataset):
    """Load a manifest-backed subset with one row per window embedding."""

    def __init__(
        self,
        features_path: Union[str, Path],
        manifest_csv: Union[str, Path],
        categories: Optional[Sequence[str]] = None,
        dataset_name: str = "UCF-Crime-Precomputed-Subset",
        mmap_mode: Optional[str] = None,
    ):
        windows, labels_by_split = load_ucf_crime_window_manifest(
            manifest_csv=manifest_csv,
            categories=categories,
        )
        super().__init__(
            dataset_name=dataset_name,
            feature_store=NpyVideoFeatureStore(features_path, mmap_mode=mmap_mode),
            windows=windows,
            frame_labels_by_split=labels_by_split,
        )


class UcfCrimeI3DTestDataset(PrecomputedVideoDataset):
    """Load TenCrop I3D UCF-Crime features and their frame labels."""

    def __init__(
        self,
        features_dir: Union[str, Path],
        ground_truth_json: Union[str, Path],
        categories: Optional[Sequence[str]] = None,
        dataset_name: str = "UCF-Crime-I3D-Test",
        frame_step: int = UCF_CRIME_I3D_FRAME_STEP,
    ):
        windows, labels_by_split, embeddings = load_ucf_crime_i3d_test_split(
            features_dir=features_dir,
            ground_truth_json=ground_truth_json,
            categories=categories,
            frame_step=frame_step,
        )
        super().__init__(
            dataset_name=dataset_name,
            feature_store=InMemoryVideoFeatureStore(embeddings),
            windows=windows,
            frame_labels_by_split=labels_by_split,
        )


def load_ucf_crime_window_manifest(
    manifest_csv: Union[str, Path],
    categories: Optional[Sequence[str]] = None,
) -> Tuple[Sequence[VideoWindow], Dict[str, Dict[str, np.ndarray]]]:
    manifest_path = Path(manifest_csv).expanduser().resolve()
    selected_categories = None if categories is None else set(categories)
    windows = []
    label_paths: Dict[Tuple[str, str], Path] = {}

    with manifest_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = sorted(set(UCF_CRIME_WINDOW_MANIFEST_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"UCF-Crime window manifest missing columns: {missing}")

        for row in reader:
            anomaly_class = row["anomaly_class"] or None
            if (
                selected_categories is not None
                and anomaly_class is not None
                and anomaly_class not in selected_categories
            ):
                continue

            split = row["split"] or "test"
            video_id = row["video_id"]
            label_paths[(split, video_id)] = _resolve_label_path(
                manifest_path,
                row["frame_labels_path"],
            )
            windows.append(
                VideoWindow(
                    video_id=video_id,
                    split=split,
                    feature_index=int(row["feature_index"]),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    label=int(row["label"]) if row["label"] else None,
                    anomaly_class=anomaly_class,
                )
            )

    labels_by_split: Dict[str, Dict[str, np.ndarray]] = {}
    for (split, video_id), label_path in label_paths.items():
        labels_by_split.setdefault(split, {})[video_id] = np.load(label_path).astype(np.int64).reshape(-1)
    return windows, labels_by_split


def load_ucf_crime_i3d_test_split(
    features_dir: Union[str, Path],
    ground_truth_json: Union[str, Path],
    categories: Optional[Sequence[str]] = None,
    frame_step: int = UCF_CRIME_I3D_FRAME_STEP,
) -> Tuple[Sequence[VideoWindow], Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    features_path = Path(features_dir).expanduser().resolve()
    ground_truth_path = Path(ground_truth_json).expanduser().resolve()
    if not features_path.exists():
        raise FileNotFoundError(f"features directory does not exist: {features_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"ground-truth file does not exist: {ground_truth_path}")

    selected_categories = None if categories is None else set(categories)
    ground_truth = _load_ground_truth(ground_truth_path)
    windows = []
    labels_by_video: Dict[str, np.ndarray] = {}
    embeddings_by_video = []
    feature_index = 0

    for feature_file in sorted(features_path.glob("*.npy")):
        video_id = feature_file.name
        anomaly_class = _ucf_crime_anomaly_class(video_id)
        if selected_categories is not None and anomaly_class is not None and anomaly_class not in selected_categories:
            continue
        if video_id not in ground_truth:
            raise KeyError(f"Missing ground-truth labels for {video_id}")

        video_embeddings = _load_i3d_video_embeddings(feature_file)
        frame_labels = np.asarray(ground_truth[video_id], dtype=np.int64).reshape(-1)
        if len(frame_labels) < len(video_embeddings) * frame_step:
            raise ValueError(
                f"Ground-truth labels for {video_id} are too short for "
                f"{len(video_embeddings)} windows at frame_step={frame_step}"
            )

        labels_by_video[video_id] = frame_labels
        embeddings_by_video.append(video_embeddings)
        video_label = int(np.any(frame_labels == 1))
        for window_index in range(len(video_embeddings)):
            start_frame = window_index * frame_step
            windows.append(
                VideoWindow(
                    video_id=video_id,
                    split="test",
                    feature_index=feature_index,
                    start_frame=start_frame,
                    end_frame=start_frame + frame_step - 1,
                    label=video_label,
                    anomaly_class=anomaly_class,
                )
            )
            feature_index += 1

    if not embeddings_by_video:
        raise ValueError(f"No UCF-Crime I3D feature files selected from {features_path}")

    embeddings = np.concatenate(embeddings_by_video, axis=0).astype(np.float32, copy=False)
    return windows, {"test": labels_by_video}, embeddings


def _resolve_label_path(manifest_path: Path, label_path: str) -> Path:
    resolved = Path(label_path).expanduser()
    if not resolved.is_absolute():
        resolved = manifest_path.parent / resolved
    return resolved.resolve()


def _load_ground_truth(path: Path) -> Dict[str, Sequence[float]]:
    with path.open(encoding="utf-8") as file:
        ground_truth = json.load(file)
    if not isinstance(ground_truth, dict):
        raise ValueError(f"Expected a ground-truth JSON object at {path}")
    return ground_truth


def _load_i3d_video_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path)
    if embeddings.ndim == 3:
        embeddings = np.mean(embeddings, axis=1)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D or 3D I3D embeddings at {path}, got {embeddings.shape}")
    return np.asarray(embeddings, dtype=np.float32)


def _ucf_crime_anomaly_class(video_id: str) -> Optional[str]:
    if video_id.startswith("Normal"):
        return None
    stem = Path(video_id).stem
    for suffix in ("_i3d", "_x264"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.rstrip("0123456789") or None
