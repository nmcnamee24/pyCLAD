"""Prepare NOLA test MP4s with a modern detector and lightweight tracker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class NolaDetection:
    """One object detection in absolute ``x1, y1, x2, y2`` pixels."""

    name: str
    confidence: float
    box: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("detection name must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")
        if len(self.box) != 4 or not np.isfinite(self.box).all():
            raise ValueError("detection box must contain four finite coordinates")
        x1, y1, x2, y2 = self.box
        if x2 < x1 or y2 < y1:
            raise ValueError("detection box must satisfy x2 >= x1 and y2 >= y1")


class SimpleIouTracker:
    """Deterministic class-aware IoU tracker producing NOLA track IDs."""

    def __init__(self, *, iou_threshold: float = 0.3, max_age: int = 60):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self._next_track_id = 1
        self._active: Dict[int, tuple[int, NolaDetection]] = {}

    def update(
        self,
        frame_id: int,
        detections: Sequence[NolaDetection],
    ) -> Tuple[Tuple[int, NolaDetection], ...]:
        if frame_id <= 0:
            raise ValueError("frame_id must be positive")
        self._active = {
            track_id: state
            for track_id, state in self._active.items()
            if frame_id - state[0] <= self.max_age
        }

        candidates = []
        for detection_index, detection in enumerate(detections):
            for track_id, (_, previous) in self._active.items():
                if detection.name != previous.name:
                    continue
                overlap = _box_iou(detection.box, previous.box)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, detection_index))
        candidates.sort(reverse=True)

        assigned_tracks = set()
        assigned_detections = set()
        result: Dict[int, NolaDetection] = {}
        for _, track_id, detection_index in candidates:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            result[track_id] = detections[detection_index]

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            result[track_id] = detection

        for track_id, detection in result.items():
            self._active[track_id] = (frame_id, detection)
        return tuple(sorted(result.items()))


class TorchvisionNolaDetector:
    """COCO SSDLite detector mapped to NOLA's object vocabulary.

    Torch and torchvision are imported lazily.  Instantiating this class may
    download torchvision's public pretrained weights into the normal PyTorch
    cache on first use.
    """

    DEFAULT_CLASS_MAPPING = {
        "person": "person",
        "bicycle": "bike",
        "motorcycle": "bike",
        "car": "car",
        "truck": "truck",
    }

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        class_mapping: Optional[Dict[str, str]] = None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self._torch = torch
        self._weights = weights
        self._categories = tuple(weights.meta["categories"])
        self._transform = weights.transforms()
        self._device = torch.device(device)
        self._model = ssdlite320_mobilenet_v3_large(weights=weights).to(self._device).eval()
        self.confidence_threshold = float(confidence_threshold)
        self.class_mapping = dict(self.DEFAULT_CLASS_MAPPING if class_mapping is None else class_mapping)

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[NolaDetection, ...]:
        from PIL import Image

        rgb = np.asarray(frame_bgr)[..., ::-1]
        image = Image.fromarray(rgb)
        tensor = self._transform(image).to(self._device)
        with self._torch.no_grad():
            output = self._model([tensor])[0]

        detections = []
        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            confidence = float(score)
            if confidence < self.confidence_threshold:
                continue
            source_name = self._categories[int(label)]
            if source_name not in self.class_mapping:
                continue
            detections.append(
                NolaDetection(
                    name=self.class_mapping[source_name],
                    confidence=confidence,
                    box=tuple(float(value) for value in box),
                )
            )
        return tuple(detections)


def preprocess_nola_video(
    video_path: Union[str, Path],
    output_directory: Union[str, Path],
    detector: Callable[[np.ndarray], Sequence[NolaDetection]],
    *,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    tracker: Optional[SimpleIouTracker] = None,
    overwrite: bool = False,
) -> Path:
    """Generate NOLA-compatible JSON, ``tracks.npy``, and metadata."""

    try:
        import cv2
    except ImportError as error:
        raise ImportError("OpenCV is required to preprocess NOLA videos") from error

    source = Path(video_path).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"NOLA source video does not exist: {source}")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    output_files = (
        destination / f"{destination.name}.json",
        destination / "tracks.npy",
        destination / "Names.txt",
        destination / "metadata.json",
    )
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"NOLA preprocessing outputs already exist: {existing}")
    destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open NOLA video: {source}")
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = SimpleIouTracker(max_age=max(60, frame_stride * 2)) if tracker is None else tracker

    annotations = []
    track_rows = []
    names = []
    frame_id = 0
    processed_count = 0
    while True:
        if max_frames is not None and processed_count >= max_frames:
            break
        success, frame = capture.read()
        if not success:
            break
        frame_id += 1
        if (frame_id - 1) % frame_stride:
            continue

        detections = tuple(detector(frame))
        tracked = tracker.update(frame_id, detections)
        objects = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            objects.append(
                {
                    "class_id": _nola_class_id(detection.name),
                    "name": detection.name,
                    "relative_coordinates": {
                        "center_x": ((x1 + x2) / 2.0) / width,
                        "center_y": ((y1 + y2) / 2.0) / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                    },
                    "confidence": detection.confidence,
                }
            )
        frame_name = f"{source}#frame={frame_id}"
        annotations.append(
            {
                "frame_id": frame_id,
                "filename": frame_name,
                "objects": objects,
            }
        )
        names.append(frame_name)
        for track_id, detection in tracked:
            track_rows.append(
                [
                    frame_id,
                    str(track_id),
                    detection.name,
                    tuple(int(round(value)) for value in detection.box),
                ]
            )
        processed_count += 1
    capture.release()

    annotation_path, tracks_path, names_path, metadata_path = output_files
    with annotation_path.open("w", encoding="utf-8") as stream:
        json.dump(annotations, stream, indent=2)
    track_array = (
        np.asarray(track_rows, dtype=object)
        if track_rows
        else np.empty((0, 4), dtype=object)
    )
    np.save(tracks_path, track_array)
    names_path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "source_video": str(source),
                "source_frame_count": source_frame_count,
                "decoded_frame_count": frame_id,
                "source_fps": source_fps,
                "source_width": width,
                "source_height": height,
                "frame_stride": frame_stride,
                "processed_frames": processed_count,
            },
            stream,
            indent=2,
        )
    return destination


def _box_iou(
    left: Iterable[float],
    right: Iterable[float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = map(float, left)
    right_x1, right_y1, right_x2, right_y2 = map(float, right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _nola_class_id(name: str) -> int:
    return {
        "person": 0,
        "bike": 1,
        "car": 2,
        "truck": 7,
        "cart": 80,
    }.get(name, -1)
