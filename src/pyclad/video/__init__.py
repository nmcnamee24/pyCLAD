"""Video anomaly detection through pyCLAD's unchanged public interfaces.

The package converts video windows into ordinary two-dimensional NumPy feature
matrices before they reach a pyCLAD strategy. Existing scenarios, strategies,
models, and replay buffers therefore require no video-specific changes.
"""

from pyclad.video.benchmarks import (
    BenchmarkResult,
    NolaBenchmarkResult,
    NolaBenchmarkRunner,
    VideoBenchmarkRunner,
)
from pyclad.video.data import (
    PrecomputedVideoDataset,
    VideoConceptsDataset,
    VideoDataset,
    VideoFeatureConcept,
    VideoStrategySchema,
    VideoWindow,
)
from pyclad.video.datasets import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    CommandUcfCrimeDataset,
    CommandUcfCrimeRecord,
    NOLA_RELEVANT_CLASSES,
    NOLA_STAGE_ORDER,
    NolaContinualDataset,
    NolaGroundTruth,
    NolaPreparedTestDataset,
    UcfCrimeI3DTestDataset,
    UcfCrimeSubsetDataset,
    extract_nola_video_features,
    load_nola_ground_truth,
)
from pyclad.video.features import (
    InMemoryVideoFeatureStore,
    NpyVideoFeatureStore,
    VideoFeatureStore,
    flatten_video_curves,
    window_scores_to_frame_scores,
)
from pyclad.video.metrics import (
    AveragePrecisionDelay,
    VideoFrameMetrics,
    compute_average_precision_delay,
    compute_video_frame_metrics,
)
from pyclad.video.models import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
    NolaFeatureLayout,
    NolaVideoModel,
    VideoAnomalyModel,
    build_nola_trajectory_examples,
    nola_spatial_object_features,
    nola_temporal_object_features,
    non_maximum_suppression,
    odit_cusum,
    pack_nola_features,
)
from pyclad.video.prediction_results import VideoPredictionResults
from pyclad.video.preprocessing import (
    NolaDetection,
    SimpleIouTracker,
    TorchvisionNolaDetector,
    preprocess_nola_video,
)

__all__ = [
    "AveragePrecisionDelay",
    "BenchmarkResult",
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "COMMAND_UCF_CRIME_CONCEPT_ORDER",
    "CommandUcfCrimeDataset",
    "CommandUcfCrimeRecord",
    "CommandVideoModel",
    "InMemoryVideoFeatureStore",
    "NolaFeatureLayout",
    "NolaBenchmarkResult",
    "NolaBenchmarkRunner",
    "NolaContinualDataset",
    "NolaDetection",
    "NolaGroundTruth",
    "NolaPreparedTestDataset",
    "NOLA_RELEVANT_CLASSES",
    "NOLA_STAGE_ORDER",
    "NolaTrajectoryPredictor",
    "NolaVideoModel",
    "NpyVideoFeatureStore",
    "PrecomputedVideoDataset",
    "SimpleIouTracker",
    "TorchVideoBackbone",
    "TorchvisionNolaDetector",
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
    "build_nola_trajectory_examples",
    "compute_average_precision_delay",
    "compute_video_frame_metrics",
    "flatten_video_curves",
    "extract_nola_video_features",
    "load_nola_ground_truth",
    "nola_spatial_object_features",
    "nola_temporal_object_features",
    "non_maximum_suppression",
    "odit_cusum",
    "pack_nola_features",
    "preprocess_nola_video",
    "window_scores_to_frame_scores",
]


def __getattr__(name):
    if name in {"CommandVideoModel", "NolaTrajectoryPredictor", "TorchVideoBackbone"}:
        from pyclad.video import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
