"""Video feature storage, windowing, and frame projection."""

from pyclad.video.features.store import (
    InMemoryVideoFeatureStore,
    NpyVideoFeatureStore,
    VideoFeatureStore,
)
from pyclad.video.features.windowing import window_frame_features
from pyclad.video.features.windows import (
    flatten_video_curves,
    window_scores_to_frame_scores,
)

__all__ = [
    "InMemoryVideoFeatureStore",
    "NpyVideoFeatureStore",
    "VideoFeatureStore",
    "flatten_video_curves",
    "window_frame_features",
    "window_scores_to_frame_scores",
]
