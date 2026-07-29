"""Validate that one prepared NOLA test-video cache is complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def _count_decodable_frames(source_video: Path) -> int:
    try:
        import cv2
    except ImportError as error:
        raise ValueError(
            "OpenCV is required to verify a cache when the container frame count "
            "does not match the processed frame count"
        ) from error

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open NOLA source video: {source_video}")
    decoded_frames = 0
    try:
        while True:
            success, _ = capture.read()
            if not success:
                break
            decoded_frames += 1
    finally:
        capture.release()
    return decoded_frames


def validate_nola_cache(
    cache_directory: Path,
    *,
    expected_frame_stride: int = 1,
) -> dict:
    directory = cache_directory.expanduser().resolve()
    if expected_frame_stride <= 0:
        raise ValueError("expected_frame_stride must be positive")
    required = {
        "annotations": directory / f"{directory.name}.json",
        "tracks": directory / "tracks.npy",
        "names": directory / "Names.txt",
        "metadata": directory / "metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"NOLA cache is missing required files: {missing}")

    with required["metadata"].open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    reported_source_frames = int(metadata["source_frame_count"])
    processed_frames = int(metadata["processed_frames"])
    frame_stride = int(metadata["frame_stride"])
    if frame_stride != expected_frame_stride:
        raise ValueError(
            f"NOLA cache frame_stride is {frame_stride}, expected {expected_frame_stride}"
        )

    decoded_source_frames = metadata.get("decoded_frame_count")
    validation_basis = "decoded_frame_count"
    if decoded_source_frames is not None:
        decoded_source_frames = int(decoded_source_frames)
    else:
        reported_expected = math.ceil(reported_source_frames / expected_frame_stride)
        if processed_frames == reported_expected:
            decoded_source_frames = reported_source_frames
            validation_basis = "container_frame_count"
        else:
            source_value = metadata.get("source_video")
            source_video = (
                None
                if source_value is None
                else Path(str(source_value)).expanduser().resolve()
            )
            if source_video is None or not source_video.is_file():
                raise ValueError(
                    f"NOLA cache contains {processed_frames} frames, expected "
                    f"{reported_expected} from "
                    f"source_frame_count={reported_source_frames}; source_video "
                    "is unavailable for decode-to-EOF verification"
                )
            decoded_source_frames = _count_decodable_frames(source_video)
            validation_basis = "decode_to_eof"

    expected_processed = math.ceil(decoded_source_frames / expected_frame_stride)
    if processed_frames != expected_processed:
        raise ValueError(
            f"NOLA cache contains {processed_frames} frames, expected "
            f"{expected_processed} from "
            f"decoded_frame_count={decoded_source_frames} "
            f"(container source_frame_count={reported_source_frames})"
        )

    with required["annotations"].open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    if not isinstance(annotations, list) or len(annotations) != processed_frames:
        raise ValueError(
            f"NOLA annotation count is {len(annotations)}, expected {processed_frames}"
        )
    annotation_frame_ids = [int(annotation["frame_id"]) for annotation in annotations]
    expected_frame_ids = list(
        range(1, decoded_source_frames + 1, expected_frame_stride)
    )
    if annotation_frame_ids != expected_frame_ids:
        raise ValueError(
            "NOLA annotation frame IDs do not cover every expected decodable "
            f"frame at stride {expected_frame_stride}"
        )
    names_count = sum(
        1
        for line in required["names"].read_text(encoding="utf-8").splitlines()
        if line
    )
    if names_count != processed_frames:
        raise ValueError(
            f"NOLA Names.txt contains {names_count} rows, expected {processed_frames}"
        )
    tracks = np.load(required["tracks"], allow_pickle=True)
    if tracks.ndim != 2 or tracks.shape[1] != 4:
        raise ValueError(f"NOLA tracks must have shape (rows, 4), got {tracks.shape}")
    return {
        "valid": True,
        "video_id": directory.name,
        "cache_directory": str(directory),
        "source_frame_count": reported_source_frames,
        "decoded_frame_count": decoded_source_frames,
        "processed_frames": processed_frames,
        "frame_stride": frame_stride,
        "track_rows": len(tracks),
        "validation_basis": validation_basis,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_directory", type=Path)
    parser.add_argument("--expected-frame-stride", type=int, default=1)
    arguments = parser.parse_args(argv)
    result = validate_nola_cache(
        arguments.cache_directory,
        expected_frame_stride=arguments.expected_frame_stride,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
