"""Metadata for model-ready video windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class VideoWindow:
    """Sidecar metadata for one row of a video feature matrix."""

    video_id: str
    start_frame: int
    end_frame: int
    feature_index: int
    split: str = "test"
    label: Optional[int] = None
    anomaly_class: Optional[str] = None
    concept_id: Optional[str] = None
    domain_id: Optional[str] = None
    timestamp: Optional[float] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must be non-empty")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if self.feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        if self.label not in {None, 0, 1}:
            raise ValueError("label must be one of: None, 0, 1")
