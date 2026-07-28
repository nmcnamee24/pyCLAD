"""Video anomaly detection through pyCLAD's unchanged public interfaces.

The package converts video windows into ordinary two-dimensional NumPy feature
matrices before they reach a pyCLAD strategy. Existing scenarios, strategies,
models, and replay buffers therefore require no video-specific changes.
"""

from pyclad.video.benchmarks import BenchmarkResult, VideoBenchmarkRunner
from pyclad.video.data import (
    PrecomputedVideoDataset,
    VideoConceptsDataset,
    VideoDataset,
    VideoFeatureConcept,
    VideoStrategySchema,
    VideoWindow,
)
from pyclad.video.datasets import (
    UcfCrimeI3DTestDataset,
    UcfCrimeSubsetDataset,
)
from pyclad.video.features import (
    InMemoryVideoFeatureStore,
    NpyVideoFeatureStore,
    VideoFeatureStore,
    flatten_video_curves,
    window_scores_to_frame_scores,
)
from pyclad.video.metrics import VideoFrameMetrics, compute_video_frame_metrics
from pyclad.video.models import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
    VideoAnomalyModel,
)
from pyclad.video.prediction_results import VideoPredictionResults

__all__ = [
    "BenchmarkResult",
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "InMemoryVideoFeatureStore",
    "NpyVideoFeatureStore",
    "PrecomputedVideoDataset",
    "TorchVideoBackbone",
    "UcfCrimeI3DTestDataset",
    "UcfCrimeSubsetDataset",
    "VideoAnomalyModel",
    "VideoBenchmarkRunner",
    "VideoConceptsDataset",
    "VideoDataset",
    "VideoFeatureConcept",
    "VideoFeatureStore",
    "VideoFrameMetrics",
    "VideoPredictionResults",
    "VideoStrategySchema",
    "VideoWindow",
    "compute_video_frame_metrics",
    "flatten_video_curves",
    "window_scores_to_frame_scores",
]


def __getattr__(name):
    if name == "TorchVideoBackbone":
        from pyclad.video.models import TorchVideoBackbone

        return TorchVideoBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
