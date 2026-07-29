"""Prepare NOLA test MP4s with paper-compatible detection and tracking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
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
    class_id: int = -1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("detection name must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")
        if len(self.box) != 4 or not np.isfinite(self.box).all():
            raise ValueError("detection box must contain four finite coordinates")
        if self.class_id < -1:
            raise ValueError("class_id must be non-negative or -1 when unknown")
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
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, NolaDetection], ...]:
        if frame_id <= 0:
            raise ValueError("frame_id must be positive")
        self._active = {
            track_id: state for track_id, state in self._active.items() if frame_id - state[0] <= self.max_age
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

    def additional_info(self) -> dict[str, object]:
        return {
            "name": "simple-class-aware-iou",
            "iou_threshold": self.iou_threshold,
            "max_age": self.max_age,
        }


class DeepSortNolaTracker:
    """Adapter for the DeepSORT tracker used by the NOLA paper.

    ``deep-sort-realtime`` supplies the maintained PyTorch implementation. Its
    default MobileNet appearance embedder is used unless ``embedder`` is
    explicitly overridden.
    """

    def __init__(
        self,
        *,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
        nn_budget: Optional[int] = 100,
        embedder: str = "mobilenet",
        embedder_gpu: bool = False,
    ):
        if max_age <= 0:
            raise ValueError("max_age must be positive")
        if n_init <= 0:
            raise ValueError("n_init must be positive")
        if not 0.0 < max_cosine_distance <= 1.0:
            raise ValueError("max_cosine_distance must be between zero and one")
        if nn_budget is not None and nn_budget <= 0:
            raise ValueError("nn_budget must be positive or None")
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as error:
            raise ImportError("DeepSORT preprocessing requires the optional " "'deep-sort-realtime' package") from error

        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.max_cosine_distance = float(max_cosine_distance)
        self.nn_budget = None if nn_budget is None else int(nn_budget)
        self.embedder = str(embedder)
        self.embedder_gpu = bool(embedder_gpu)
        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_cosine_distance=self.max_cosine_distance,
            nn_budget=self.nn_budget,
            embedder=self.embedder,
            embedder_gpu=self.embedder_gpu,
        )

    def update(
        self,
        frame_id: int,
        detections: Sequence[NolaDetection],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, NolaDetection], ...]:
        if frame_id <= 0:
            raise ValueError("frame_id must be positive")
        if frame_bgr is None:
            raise ValueError("DeepSORT requires the source video frame")
        raw_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            raw_detections.append(
                (
                    [x1, y1, x2 - x1, y2 - y1],
                    detection.confidence,
                    detection.name,
                )
            )
        tracks = self._tracker.update_tracks(raw_detections, frame=frame_bgr)
        results = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            box = track.to_ltrb(orig=True)
            if box is None:
                box = track.to_ltrb()
            name = str(track.get_det_class())
            confidence = track.get_det_conf()
            results.append(
                (
                    int(track.track_id),
                    NolaDetection(
                        name=name,
                        confidence=1.0 if confidence is None else float(confidence),
                        box=tuple(float(value) for value in box),
                        class_id=_nola_class_id(name),
                    ),
                )
            )
        return tuple(sorted(results))

    def additional_info(self) -> dict[str, object]:
        return {
            "name": "deep-sort-realtime",
            "package_version": importlib.metadata.version("deep-sort-realtime"),
            "max_age": self.max_age,
            "n_init": self.n_init,
            "max_cosine_distance": self.max_cosine_distance,
            "nn_budget": self.nn_budget,
            "embedder": self.embedder,
            "embedder_gpu": self.embedder_gpu,
        }


class TorchvisionNolaDetector:
    """COCO SSDLite detector mapped to NOLA's object vocabulary.

    Torch and torchvision are imported lazily.  Instantiating this class may
    download torchvision's public pretrained weights into the normal PyTorch
    cache on first use.
    """

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
        self.class_mapping = (
            {category: category for category in self._categories if category and category != "N/A"}
            if class_mapping is None
            else dict(class_mapping)
        )

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
                    class_id=_nola_class_id(self.class_mapping[source_name]),
                )
            )
        return tuple(detections)

    def additional_info(self) -> dict[str, object]:
        return {
            "name": "torchvision-ssdlite320-mobilenet-v3-large",
            "weights": str(self._weights),
            "confidence_threshold": self.confidence_threshold,
            "class_count": len(self.class_mapping),
            "device": str(self._device),
        }


class DarknetNolaDetector:
    """YOLOv4-CSP detector using Darknet cfg/weights through OpenCV DNN."""

    def __init__(
        self,
        config_path: Union[str, Path],
        weights_path: Union[str, Path],
        names_path: Union[str, Path],
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (512, 512),
        device: str = "cpu",
    ):
        import cv2

        self.config_path = Path(config_path).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.names_path = Path(names_path).expanduser().resolve()
        for path in (self.config_path, self.weights_path, self.names_path):
            if not path.is_file():
                raise FileNotFoundError(f"Darknet NOLA asset does not exist: {path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between zero and one")

        self._cv2 = cv2
        self._classes = tuple(
            line.strip() for line in self.names_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        self._network = cv2.dnn.readNetFromDarknet(str(self.config_path), str(self.weights_path))
        if device == "cuda":
            self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        elif device != "cpu":
            raise ValueError("DarknetNolaDetector device must be 'cpu' or 'cuda'")
        self._model = cv2.dnn_DetectionModel(self._network)
        self._model.setInputParams(
            scale=1.0 / 255.0,
            size=tuple(map(int, input_size)),
            swapRB=True,
            crop=False,
        )
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = tuple(map(int, input_size))
        self.device = device

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[NolaDetection, ...]:
        class_ids, confidences, boxes = self._model.detect(
            frame_bgr,
            confThreshold=self.confidence_threshold,
            nmsThreshold=self.nms_threshold,
        )
        detections = []
        for class_id, confidence, box in zip(
            np.asarray(class_ids).reshape(-1),
            np.asarray(confidences).reshape(-1),
            np.asarray(boxes).reshape(-1, 4),
        ):
            zero_based_id = int(class_id)
            if not 0 <= zero_based_id < len(self._classes):
                continue
            x, y, width, height = map(float, box)
            detections.append(
                NolaDetection(
                    name=self._classes[zero_based_id],
                    confidence=float(confidence),
                    box=(x, y, x + width, y + height),
                    class_id=zero_based_id,
                )
            )
        return tuple(detections)

    def additional_info(self) -> dict[str, object]:
        return {
            "name": "darknet-yolov4-csp",
            "config": str(self.config_path),
            "weights": str(self.weights_path),
            "names": str(self.names_path),
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "input_size": list(self.input_size),
            "device": self.device,
        }


class DarknetCliNolaDetector:
    """The native Darknet batch invocation used by the NOLA authors."""

    def __init__(
        self,
        binary_path: Union[str, Path],
        data_path: Union[str, Path],
        config_path: Union[str, Path],
        weights_path: Union[str, Path],
        names_path: Union[str, Path],
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        source_commit: Optional[str] = None,
        weights_sha256: Optional[str] = None,
    ):
        self.binary_path = Path(binary_path).expanduser().resolve()
        self.data_path = Path(data_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.names_path = Path(names_path).expanduser().resolve()
        for path in (
            self.binary_path,
            self.data_path,
            self.config_path,
            self.weights_path,
            self.names_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"native Darknet NOLA asset does not exist: {path}")
        if not os.access(self.binary_path, os.X_OK):
            raise PermissionError(f"native Darknet binary is not executable: {self.binary_path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between zero and one")
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.source_commit = None if source_commit is None else str(source_commit)
        self.weights_sha256 = (
            _validate_sha256(self.weights_path, weights_sha256) if weights_sha256 is not None else None
        )

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[NolaDetection, ...]:
        raise TypeError("DarknetCliNolaDetector operates on a complete video through " "preprocess_nola_video")

    def detect_video(
        self,
        video_path: Union[str, Path],
        *,
        frame_stride: int,
        max_frames: Optional[int],
    ) -> Dict[int, Tuple[NolaDetection, ...]]:
        """Extract lossless frames, run native Darknet once, and align results."""

        import cv2

        source = Path(video_path).expanduser().resolve()
        temporary_root = os.environ.get("TMPDIR") or None
        with tempfile.TemporaryDirectory(
            prefix="pyclad-nola-darknet-",
            dir=temporary_root,
        ) as temporary:
            workspace = Path(temporary)
            frames_directory = workspace / "frames"
            frames_directory.mkdir()
            input_path = workspace / "Names.txt"
            output_path = workspace / "detections.json"
            log_path = workspace / "darknet.log"

            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open NOLA video: {source}")
            source_frame_id = 0
            selected_count = 0
            frame_records = {}
            input_lines = []
            try:
                while True:
                    if max_frames is not None and selected_count >= max_frames:
                        break
                    success, frame = capture.read()
                    if not success:
                        break
                    source_frame_id += 1
                    if (source_frame_id - 1) % frame_stride:
                        continue
                    frame_path = frames_directory / f"{source_frame_id:06d}.png"
                    if not cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1],
                    ):
                        raise OSError(f"OpenCV could not write Darknet frame: {frame_path}")
                    frame_records[frame_path.resolve()] = (
                        source_frame_id,
                        int(frame.shape[1]),
                        int(frame.shape[0]),
                    )
                    input_lines.append(str(frame_path.resolve()))
                    selected_count += 1
            finally:
                capture.release()
            if not input_lines:
                raise ValueError(f"NOLA video contains no selected frames: {source}")
            input_path.write_text(
                "".join(f"{line}\n" for line in input_lines),
                encoding="utf-8",
            )

            command = [
                str(self.binary_path),
                "detector",
                "test",
                str(self.data_path),
                str(self.config_path),
                str(self.weights_path),
                "-thresh",
                str(self.confidence_threshold),
                "-nms",
                str(self.nms_threshold),
                "-ext_output",
                "-out",
                str(output_path),
                "-dont_show",
            ]
            with input_path.open("rb") as input_stream, log_path.open("wb") as log_stream:
                completed = subprocess.run(
                    command,
                    stdin=input_stream,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                log_tail = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-4_000:]
                raise RuntimeError(f"native Darknet failed with exit code {completed.returncode}: " f"{log_tail}")
            if not output_path.is_file():
                raise RuntimeError("native Darknet completed without writing detection JSON")
            with output_path.open(encoding="utf-8") as stream:
                annotations = json.load(stream)

            detections_by_frame = {}
            for annotation in annotations:
                filename = Path(str(annotation["filename"])).expanduser().resolve()
                record = frame_records.get(filename)
                if record is None:
                    raise ValueError(f"Darknet returned an unknown frame path: {filename}")
                frame_id, width, height = record
                detections_by_frame[frame_id] = tuple(
                    _darknet_json_detection(detected, width, height) for detected in annotation.get("objects", ())
                )
            for frame_id, _, _ in frame_records.values():
                detections_by_frame.setdefault(frame_id, ())
            return detections_by_frame

    def additional_info(self) -> dict[str, object]:
        return {
            "name": "native-darknet-yolov4-csp",
            "binary": str(self.binary_path),
            "data": str(self.data_path),
            "config": str(self.config_path),
            "weights": str(self.weights_path),
            "names": str(self.names_path),
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "batch_input_format": "lossless-png",
            "source_commit": self.source_commit,
            "weights_sha256": self.weights_sha256,
        }


def preprocess_nola_video(
    video_path: Union[str, Path],
    output_directory: Union[str, Path],
    detector: Callable[[np.ndarray], Sequence[NolaDetection]],
    *,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    tracker: Optional[Union[SimpleIouTracker, DeepSortNolaTracker]] = None,
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
    batch_detections = (
        detector.detect_video(
            source,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        if hasattr(detector, "detect_video")
        else None
    )

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

        detections = (
            tuple(batch_detections.get(frame_id, ())) if batch_detections is not None else tuple(detector(frame))
        )
        tracked = tracker.update(frame_id, detections, frame_bgr=frame)
        objects = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            objects.append(
                {
                    "class_id": detection.class_id if detection.class_id >= 0 else _nola_class_id(detection.name),
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
    track_array = np.asarray(track_rows, dtype=object) if track_rows else np.empty((0, 4), dtype=object)
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
                "detector": (
                    detector.additional_info()
                    if hasattr(detector, "additional_info")
                    else {"name": type(detector).__name__}
                ),
                "tracker": (
                    tracker.additional_info()
                    if hasattr(tracker, "additional_info")
                    else {"name": type(tracker).__name__}
                ),
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
    from pyclad.video.datasets.nola_paper import DARKNET_COCO_CLASSES

    aliases = {
        "airplane": "aeroplane",
        "bike": "bicycle",
        "couch": "sofa",
        "dining table": "diningtable",
        "hair dryer": "hair drier",
        "motorcycle": "motorbike",
        "potted plant": "pottedplant",
        "tv": "tvmonitor",
    }
    normalized = str(name).strip().lower()
    normalized = aliases.get(normalized, normalized)
    try:
        return DARKNET_COCO_CLASSES.index(normalized)
    except ValueError:
        return -1


def _darknet_json_detection(
    detected: dict,
    frame_width: int,
    frame_height: int,
) -> NolaDetection:
    relative = detected["relative_coordinates"]
    center_x = float(relative["center_x"]) * frame_width
    center_y = float(relative["center_y"]) * frame_height
    width = float(relative["width"]) * frame_width
    height = float(relative["height"]) * frame_height
    name = str(detected["name"])
    return NolaDetection(
        name=name,
        confidence=float(detected["confidence"]),
        box=(
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        class_id=int(detected.get("class_id", _nola_class_id(name))),
    )


def _validate_sha256(path: Path, expected: str) -> str:
    normalized = str(expected).strip().lower()
    valid_characters = all(character in "0123456789abcdef" for character in normalized)
    if len(normalized) != 64 or not valid_characters:
        raise ValueError("weights_sha256 must contain 64 hexadecimal characters")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != normalized:
        raise ValueError(f"Darknet weights SHA-256 is {actual}, expected {normalized}")
    return actual
